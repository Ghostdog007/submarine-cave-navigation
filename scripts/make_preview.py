"""Turn replay videos into a small animated WebP preview for the README (GitHub shows WebP inline, not .mp4).

    # two navigators side by side, until both have run 17.5 s of video
    python -m scripts.make_preview videos/10_memory_caves_clean/rings_cave1000008_memnav_cc_exit_blind.mp4 \
        videos/10_memory_caves_clean/rings_cave1000008_memnav_cc_none_blind.mp4 --duration 17.5 \
        --out docs/media/v33_rings.webp

    # one clip, a 12 s excerpt from 4 s in, 640 px wide
    python -m scripts.make_preview videos/04_game_renderer/oracle_follow_cam_ice_theme_d1.00_cave1000002.mp4 \
        --start 4 --duration 12 --width 640 --out docs/media/game_renderer.webp

Several inputs are placed side by side (--stack v: on top of each other); a clip that ends early holds its last
frame. Defaults (10 fps, 480 px per panel, quality 55) give ~1-3 MB for 15-25 s. Animated water makes every frame
different, so GIF would be ~10x larger. Encoding at --effort 6 takes several minutes per preview; --effort 2 is
much faster and ~20-30% larger.
"""
import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="one or more .mp4 replays")
    p.add_argument("--out", required=True, help="output .webp (docs/media/ for the README)")
    p.add_argument("--start", type=float, default=0.0, help="seconds into the video(s)")
    p.add_argument("--duration", type=float, required=True, help="seconds of video to keep (before --speed)")
    p.add_argument("--speed", type=float, default=1.0, help="extra speed-up on top of the replay's own")
    p.add_argument("--width", type=int, default=480, help="width of each panel in pixels")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--quality", type=int, default=55, help="0-100")
    p.add_argument("--effort", type=int, default=6, choices=range(7), help="compression effort 0-6")
    p.add_argument("--stack", choices=["h", "v"], default="h")
    args = p.parse_args()

    n = len(args.inputs)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y"]
    for f in args.inputs:
        cmd += ["-ss", str(args.start), "-i", f]
    panel = (f"setpts=PTS/{args.speed},fps={args.fps},scale={args.width}:-2:flags=lanczos,"
             "tpad=stop_mode=clone:stop_duration=600")
    graph = ";".join(f"[{i}]{panel}[p{i}]" for i in range(n))
    stacked = "".join(f"[p{i}]" for i in range(n)) + (f"{args.stack}stack=inputs={n}," if n > 1 else "")
    graph += f";{stacked}trim=0:{args.duration / args.speed},setpts=PTS-STARTPTS"
    cmd += ["-filter_complex", graph, "-an", "-c:v", "libwebp_anim", "-lossless", "0",
            "-quality", str(args.quality), "-compression_level", str(args.effort), "-loop", "0", args.out]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    print(f"{args.out}: {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

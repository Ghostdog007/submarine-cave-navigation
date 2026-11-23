"""Turn replay videos into a README preview (animated WebP) or a post-ready MP4.

    # two navigators side by side, until both have run 17.5 s of video (WebP for the README)
    python -m scripts.make_preview videos/10_memory_caves_clean/rings_cave1000008_memnav_cc_exit_blind.mp4 \
        videos/10_memory_caves_clean/rings_cave1000008_memnav_cc_none_blind.mp4 --duration 17.5 \
        --out docs/media/v33_rings.webp

    # one clip, a 12 s excerpt from 4 s in, 640 px wide
    python -m scripts.make_preview videos/04_game_renderer/oracle_follow_cam_ice_theme_d1.00_cave1000002.mp4 \
        --start 4 --duration 12 --width 640 --out docs/media/game_renderer.webp

    # four versions in a 2x2 grid with a label above each panel, as an MP4 for a social-media post
    python -m scripts.make_preview a.mp4 b.mp4 c.mp4 d.mp4 --grid 2 --labels "Round 2" "v3" "v3.2" "v3.3" \
        --width 960 --fps 30 --duration 22 --out post/versions.mp4

The output format follows the extension: .webp (GitHub shows it inline in a README, not .mp4) or .mp4 (H.264,
plays everywhere, autoplays on social media). Several inputs are placed side by side, stacked (--stack v) or in a
grid of --grid columns; a clip that ends early holds its last frame. --speed 0.5 plays at half speed (slow motion).
WebP defaults (10 fps, 480 px per panel, quality 55) give ~1-3 MB for 15-25 s; animated water makes every frame
different, so GIF would be ~10x larger. WebP encoding at --effort 6 takes several minutes; --effort 2 is faster.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def label_bar(text, width, height, path):
    """A dark bar with centred white text, saved as PNG (drawn with Pillow: no font paths needed in ffmpeg)."""
    img = Image.new("RGB", (width, height), (10, 22, 36))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=int(height * 0.55))
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (box[2] - box[0])) / 2 - box[0], (height - (box[3] - box[1])) / 2 - box[1]), text,
              font=font, fill=(240, 244, 248))
    img.save(path)


def build_command(args, label_files):
    n = len(args.inputs)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y"]
    for f in args.inputs:
        cmd += ["-ss", str(args.start), "-i", f]
    total = args.duration / args.speed + 1                     # output seconds (+1 s margin for the label inputs)
    for f in label_files:
        cmd += ["-loop", "1", "-t", f"{total:.2f}", "-i", f]
    parts = []
    for i in range(n):
        parts.append(f"[{i}]setpts=PTS/{args.speed},fps={args.fps},scale={args.width}:-2:flags=lanczos,format=rgb24,"
                     f"tpad=stop_mode=clone:stop_duration=600[v{i}]")
        if label_files:
            parts.append(f"[{n + i}]fps={args.fps},format=rgb24[l{i}];[l{i}][v{i}]vstack[p{i}]")
        else:
            parts.append(f"[v{i}]null[p{i}]")
    panels = "".join(f"[p{i}]" for i in range(n))
    if n == 1:
        stacked = f"{panels}null"
    elif args.grid:
        cols = args.grid
        pos = []
        for i in range(n):
            r, c = divmod(i, cols)
            x = "+".join(f"w{j}" for j in range(c)) or "0"
            y = "+".join(f"h{j * cols}" for j in range(r)) or "0"
            pos.append(f"{x}_{y}")
        stacked = f"{panels}xstack=inputs={n}:layout={'|'.join(pos)}:fill=0x0A1624"
    else:
        stacked = f"{panels}{args.stack}stack=inputs={n}"
    parts.append(f"{stacked},trim=0:{args.duration / args.speed},setpts=PTS-STARTPTS")
    cmd += ["-filter_complex", ";".join(parts), "-an"]
    if args.out.lower().endswith(".mp4"):
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart"]
    else:
        cmd += ["-c:v", "libwebp_anim", "-lossless", "0", "-quality", str(args.quality),
                "-compression_level", str(args.effort), "-loop", "0"]
    return cmd + [args.out]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="one or more .mp4 replays")
    p.add_argument("--out", required=True, help="output .webp (README, docs/media/) or .mp4 (posts)")
    p.add_argument("--start", type=float, default=0.0, help="seconds into the video(s)")
    p.add_argument("--duration", type=float, required=True, help="seconds of video to keep (before --speed)")
    p.add_argument("--speed", type=float, default=1.0, help="speed factor on top of the replay's own (0.5 = slow)")
    p.add_argument("--width", type=int, default=480, help="width of each panel in pixels")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--stack", choices=["h", "v"], default="h")
    p.add_argument("--grid", type=int, default=0, help="arrange the inputs in a grid with this many columns")
    p.add_argument("--labels", nargs="*", default=[], help="one label per input, shown in a bar above its panel")
    p.add_argument("--label-height", type=int, default=44)
    p.add_argument("--quality", type=int, default=55, help="WebP quality 0-100")
    p.add_argument("--effort", type=int, default=6, choices=range(7), help="WebP compression effort 0-6")
    p.add_argument("--crf", type=int, default=20, help="MP4 quality (lower = better, 18-23 typical)")
    args = p.parse_args()
    if args.labels and len(args.labels) != len(args.inputs):
        p.error("give one label per input")
    if not args.out.lower().endswith((".webp", ".mp4")):
        p.error("--out must end in .webp or .mp4")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        labels = []
        for i, text in enumerate(args.labels):
            f = str(Path(tmp) / f"label{i}.png")
            label_bar(text, args.width, args.label_height - args.label_height % 2, f)
            labels.append(f)
        subprocess.run(build_command(args, labels), check=True)
    print(f"{args.out}: {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

"""Replay videos: run one or more policies on the same cave and write an MP4 (or GIF).

Several policies are stepped in lockstep and stacked vertically, so you can
watch e.g. the gap follower and a trained agent take the same cave side by
side. Each policy gets the same cave, sensor noise and current as in
evaluate(), so the replay matches the numbers in the evaluation tables.
Frames are streamed to the encoder, so long episodes don't fill memory.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from submarine.world import descend
from submarine.env.supervisor import TIER_NAMES
from submarine.env import CaveEnv, coerce_config

from submarine.evaluation.metrics import episode_seed

OUTCOME_STYLE = {
    "success": ("GOAL", (120, 240, 140)),
    "collision": ("COLLISION", (255, 100, 100)),
    "timeout": ("TIMEOUT", (255, 190, 90)),
    "abort": ("SAFE ABORT", (190, 160, 255)),
}


class _Run:
    def __init__(self, policy, cfg, cave_seed, difficulty, base_seed, viewer_kwargs):
        from submarine.render.viewer import Viewer

        from submarine.evaluation.policies import env_config_for

        self.policy = policy
        self.env = CaveEnv(env_config_for(policy, cfg))
        self.obs, info = self.env.reset(seed=episode_seed(cave_seed, difficulty, base_seed),
                                        options={"cave_seed": cave_seed, "difficulty": difficulty})
        policy.reset(self.env)
        cave = self.env.cave
        self.viewer = Viewer(cave, headless=True, **viewer_kwargs)
        res = cave.world.resolution
        cells = descend(cave.geodesic, cave.world_to_cell(*cave.start), res)
        self.reference = [((ix + 0.5) * res, (iy + 0.5) * res) for iy, ix in cells]
        self.d0 = info["geodesic"]
        self.trail = [info["position"]]
        self.speeds = [0.0]
        self.action = np.zeros(2)
        self.ret = 0.0
        self.done = False
        self.outcome = None
        self.info = info
        self.frame = None
        self.final = False

    def step(self):
        a = np.asarray(self.policy.act(self.obs), dtype=np.float32).reshape(2)
        self.obs, r, term, trunc, info = self.env.step(a)
        self.action, self.info = np.clip(a, -1, 1), info
        self.ret += r
        self.trail.append(info["position"])
        self.speeds.append(self.env.sim.state.u)
        if term or trunc:
            self.done = True
            self.outcome = "success" if info["is_success"] else "collision" if info["collided"] else ("abort" if info.get("aborted") else "timeout")

    def render(self):
        env, info, cave = self.env, self.info, self.env.cave
        s = env.sim.state
        t = info["t"] * env.cfg.vehicle.dt
        banner, color = None, (255, 255, 255)
        if self.done:
            word, color = OUTCOME_STYLE[self.outcome]
            banner = f"{word}  {t:.1f} s"
        progress = 1.0 - info["geodesic"] / self.d0 if self.d0 > 0 else 0.0
        self.viewer.draw(
            env.sim, env._last, self.trail, [
                f"t {t:5.1f} s   speed {s.u:+.2f} m/s   yaw rate {s.r:+.2f} rad/s   return {self.ret:6.1f}",
                f"clearance {info['clearance']:4.2f} m (min {info['min_clearance']:4.2f})   "
                f"to goal {info['geodesic']:5.1f} m   path {info['path_length']:5.1f} m"
                + (f"   hull {info['hull']:5.1f}% ({info['contacts']} contacts)"
                   if env.cfg.collision.mode == "damage" else "")
                + (f"   supervisor: {TIER_NAMES[info['tier']].upper()}" if env.cfg.supervisor.enabled else "")
                + (f"   FAILED: {', '.join(info['failures'])}" if info.get("failures") else ""),
            ],
            banner, speeds=self.speeds, action=self.action, progress=progress, reference=self.reference,
            label=f"{self.policy.name}   d={cave.difficulty:.2f}  cave {cave.seed}", banner_color=color,
        )
        self.frame = self.viewer.rgb_array()
        return self.frame


def _even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge") if h % 2 or w % 2 else frame


def record_replay(policies, env_config, cave_seed: int, difficulty: float, out_path: str | Path,
                  base_seed: int = 0, scale: float | None = None, speed: float = 2.0,
                  hold_seconds: float = 2.0, style: str = "game", theme: str = "ice",
                  camera: str | None = None) -> list[dict]:
    """Run `policies` on one cave and write a video to out_path (.mp4 or .gif).

    speed: playback speed relative to real time (2.0 = a 60 s episode plays in 30 s).
    style: "game" or "plain"; theme: "ice" or "rock" (game style).
    camera: "follow" (zoomed, tracks the submarine, minimap) or "full" (whole map);
            default follow for one policy, full when comparing several.
    Returns one dict per policy with outcome, time and return.
    """
    cfg = coerce_config(env_config)
    if camera is None:
        camera = "follow" if len(policies) == 1 else "full"
    if scale is None and camera == "full":
        scale = 24.0 if len(policies) == 1 else 16.0
    viewer_kwargs = dict(scale=scale, style=style, theme=theme, camera=camera)
    runs = [_Run(p, cfg, cave_seed, float(difficulty), base_seed, viewer_kwargs) for p in policies]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = speed / cfg.vehicle.dt
    gif = out_path.suffix.lower() == ".gif"
    if gif:
        fps = min(fps, 15.0)   # keep GIFs small: drop frames instead of encoding all of them
        stride = max(1, int(round(speed / cfg.vehicle.dt / fps)))
        writer = imageio.get_writer(out_path, mode="I", duration=1000.0 / fps, loop=0)
    else:
        stride = 1
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1,
                                    ffmpeg_log_level="error")

    def panel(r):
        """Finished runs are drawn once more (with their outcome banner), then frozen."""
        if r.done and r.final:
            return r.frame
        frame = r.render()
        r.final = r.done
        return frame

    def compose():
        frame = np.concatenate([panel(r) for r in runs], axis=0)
        if gif:
            frame = frame[::2, ::2]
        return _even(frame)

    tick = 0
    try:
        writer.append_data(compose())
        while not all(r.done for r in runs):
            for r in runs:
                if not r.done:
                    r.step()
            tick += 1
            if tick % stride == 0 or any(r.done and not r.final for r in runs):
                writer.append_data(compose())
        last = compose()
        for _ in range(int(hold_seconds * fps)):
            writer.append_data(last)
    finally:
        writer.close()
        for r in runs:
            r.env.close()

    return [{"policy": r.policy.name, "outcome": r.outcome, "time": r.info["t"] * cfg.vehicle.dt,
             "return": r.ret} for r in runs]

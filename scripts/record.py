"""Record replay videos of baselines and/or trained agents.

    python -m scripts.record --policies gap --models runs/archive/ppo_mlp_mixed_s0 --difficulties 0.5 1.0
    python -m scripts.record --models runs/archive/ppo_mlp_mixed_s0 --cave-seeds 1000003 --difficulties 0.75 --format gif
    python -m scripts.record --policies gap oracle --caves 3 --speed 4
    python -m scripts.record --models runs/archive/ppo_mlp_mixed_s0 --theme rock --camera full --style plain

All listed policies run on the same cave and are stacked in one video.
One video per (difficulty, cave). Output: runs/videos/<name>_d<d>_cave<seed>.mp4
"""

from __future__ import annotations

import os  # noqa: E402

# NumPy's BLAS reserves a buffer per CPU thread (~1.5 GB commit per process on a 32-thread machine);
# cap it before NumPy is imported (also inherited by worker processes). Override by setting the variable.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
from dataclasses import replace
from pathlib import Path

from submarine.world import seeds_for
from submarine.env import EnvConfig
from submarine.evaluation import make_policy
from submarine.render.replay import record_replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="*", default=[], help="baseline names: random gap centering oracle")
    p.add_argument("--models", nargs="*", default=[], help="trained run dirs or model .zip files")
    p.add_argument("--difficulties", type=float, nargs="+", default=[0.5])
    p.add_argument("--caves", type=int, default=1, help="first N caves of the split")
    p.add_argument("--cave-seeds", type=int, nargs="*", default=None, help="explicit cave seeds instead")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--layout", type=str, default=None, help="cave layout: tunnel maze branches crack breakdown")
    p.add_argument("--max-steps", type=int, default=None, help="episode time limit in steps (0.1 s each)")
    p.add_argument("--speed", type=float, default=2.0, help="playback speed vs real time")
    p.add_argument("--scale", type=float, default=None, help="pixels per metre")
    p.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    p.add_argument("--style", choices=["game", "plain"], default="game")
    p.add_argument("--theme", choices=["ice", "rock"], default="ice")
    p.add_argument("--camera", choices=["follow", "full"], default=None,
                   help="default: follow for one policy, full map when comparing several")
    p.add_argument("--name", type=str, default="replay")
    p.add_argument("--out-dir", type=Path, default=Path("runs/videos"))
    args = p.parse_args()

    names = args.policies + args.models
    if not names:
        p.error("give at least one --policies or --models")
    cfg = EnvConfig.from_yaml(args.config)
    if args.layout:
        cfg = replace(cfg, layout=args.layout)
    if args.max_steps:
        cfg = replace(cfg, max_steps=args.max_steps)
    seeds = args.cave_seeds or seeds_for(args.split, args.caves)

    for d in args.difficulties:
        for s in seeds:
            policies = [make_policy(n) for n in names]      # fresh policies (recurrent state etc.) per video
            out = args.out_dir / f"{args.name}_d{d:.2f}_cave{s}.{args.format}"
            results = record_replay(policies, cfg, s, d, out, speed=args.speed, scale=args.scale,
                                    style=args.style, theme=args.theme, camera=args.camera)
            summary = ", ".join(f"{r['policy']}: {r['outcome']} {r['time']:.1f}s" for r in results)
            print(f"saved {out}   [{summary}]")


if __name__ == "__main__":
    main()

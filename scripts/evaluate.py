"""Evaluate baselines on held-out caves and print the section 7 metrics.

    python -m scripts.evaluate
    python -m scripts.evaluate --policies gap oracle --difficulties 0 1 --n-caves 100
    python -m scripts.evaluate --goal-cue none --current 0.3 --name no_goal_current

Writes runs/eval/<name>_summary.csv, <name>_episodes.csv and (with --plot N)
<name>_paths.png showing each policy's path on the first N caves per difficulty.
"""

from __future__ import annotations

import os  # noqa: E402

# NumPy's BLAS reserves a buffer per CPU thread (~1.5 GB commit per process on a 32-thread machine);
# cap it before NumPy is imported (also inherited by worker processes). Override by setting the variable.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from submarine.baselines import BASELINES
from submarine.env import EnvConfig
from submarine.render.replay import record_replay
from submarine.evaluation import (episodes_to_rows, evaluate, format_table, make_policy, save_csv,
                                  summarize)


def _job(args):
    name, cfg, d, n_caves, base_seed, n_traj = args
    return evaluate(make_policy(name), cfg, [d], n_caves, base_seed, n_traj)


def plot_paths(results, n, out, layout="tunnel"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from submarine.world import generate_cave, test_seeds
    from scripts.plot_caves import draw_cave

    ds = sorted({r.difficulty for r in results})
    seeds = test_seeds(n)
    policies = list(dict.fromkeys(r.policy for r in results))
    colors = dict(zip(policies, ["#888888", "#f5a623", "#7ed321", "#ff4d6d", "#4dd0ff", "#c77dff"]))
    fig, axes = plt.subplots(len(ds), n, figsize=(3.6 * n, 2.4 * len(ds)), squeeze=False)
    for i, d in enumerate(ds):
        for j, s in enumerate(seeds):
            ax = axes[i, j]
            draw_cave(ax, generate_cave(s, d, layout=layout))
            for r in results:
                if r.difficulty == d and r.cave_seed == s and r.trajectory:
                    xs, ys = zip(*r.trajectory)
                    ax.plot(xs, ys, color=colors[r.policy], lw=1.0, label=r.policy)
                    end = {"success": "", "collision": "x", "timeout": "s"}[r.outcome]
                    if end:
                        ax.plot(xs[-1], ys[-1], end, color=colors[r.policy], ms=5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(policies), fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    p.add_argument("--models", nargs="*", default=[],
                   help="trained run dirs (uses best_model.zip) or model .zip files")
    p.add_argument("--difficulties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--n-caves", type=int, default=50)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--goal-cue", choices=["bearing", "none"], default=None)
    p.add_argument("--current", type=float, default=None, help="override max current, m/s")
    p.add_argument("--layout", type=str, default=None, help="cave layout: tunnel maze branches crack breakdown")
    p.add_argument("--max-steps", type=int, default=None, help="episode time limit in steps (0.1 s each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", type=int, default=3, help="plot paths on the first N caves per difficulty (0 = off)")
    p.add_argument("--video", type=int, default=0,
                   help="also record replay videos (all policies side by side) of the first N caves per difficulty")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--name", type=str, default="baselines")
    p.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    args = p.parse_args()

    cfg = EnvConfig.from_yaml(args.config)
    if args.goal_cue:
        cfg = replace(cfg, goal_cue=args.goal_cue)
    if args.current is not None:
        cfg = replace(cfg, vehicle=replace(cfg.vehicle, max_current=args.current))
    if args.layout:
        cfg = replace(cfg, layout=args.layout)
    if args.max_steps:
        cfg = replace(cfg, max_steps=args.max_steps)

    names = list(args.policies) + list(args.models)
    jobs = [(name, cfg, d, args.n_caves, args.seed, args.plot) for name in names for d in args.difficulties]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            chunks = list(ex.map(_job, jobs))
    else:
        chunks = [_job(j) for j in jobs]
    results = [r for c in chunks for r in c]

    summary = summarize(results)
    order = list(dict.fromkeys(r.policy for r in results))
    summary.sort(key=lambda r: (order.index(r["policy"]), str(r["difficulty"])))
    print(format_table(summary))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(summary, args.out_dir / f"{args.name}_summary.csv")
    save_csv(episodes_to_rows(results), args.out_dir / f"{args.name}_episodes.csv")
    print(f"saved {args.out_dir / (args.name + '_summary.csv')} and _episodes.csv")
    if args.plot:
        plot_paths(results, args.plot, args.out_dir / f"{args.name}_paths.png", cfg.layout[0])
    if args.video:
        from submarine.world import test_seeds

        vdir = args.out_dir / f"{args.name}_videos"
        for d in args.difficulties:
            for s in test_seeds(args.video):
                out = vdir / f"d{d:.2f}_cave{s}.mp4"
                record_replay([make_policy(n) for n in names], cfg, s, d, out, base_seed=args.seed)
                print(f"saved {out}")


if __name__ == "__main__":
    main()

"""Memory evaluation: success and dead-end behaviour per cave layout.

    python -m scripts.eval_memory --models runs/archive/ppo_mlp_mixed_s0 --policies gap oracle

Writes runs/eval/<name>_episodes.csv and prints, per layout and policy: success / collision / timeout,
dead-end entries and re-entries, branches explored, backtracking, time and energy.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path


def _job(args):
    name, cfg, layout, d, n_caves = args
    from submarine.world import generate_cave, test_seeds
    from submarine.evaluation import evaluate, make_policy
    from submarine.evaluation.deadends import trajectory_stats

    cfg = replace(cfg, layout=layout)
    res = evaluate(make_policy(name), cfg, [d], n_caves, n_trajectories=n_caves)
    rows = []
    for r in res:
        cave = generate_cave(r.cave_seed, d, layout=layout)
        row = {"policy": r.policy, "layout": layout, "cave_seed": r.cave_seed, "outcome": r.outcome,
               "time": r.time, "energy_kj": r.energy_kj, "episode_return": r.episode_return}
        row.update(trajectory_stats(cave, r.trajectory))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="*", default=["gap", "oracle"])
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--layouts", nargs="+", default=["maze", "branches", "tunnel", "crack", "breakdown"])
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--n-caves", type=int, default=30)
    p.add_argument("--config", type=str, default="configs/eval_layouts.yaml")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--name", type=str, default="memory")
    args = p.parse_args()

    import csv

    import numpy as np

    from submarine.env import EnvConfig

    cfg = EnvConfig.from_yaml(args.config)
    names = args.policies + args.models
    jobs = [(n, cfg, lay, args.difficulty, args.n_caves) for lay in args.layouts for n in names]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for chunk in ex.map(_job, jobs) for r in chunk]

    out = Path("runs/eval") / f"{args.name}_episodes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    head = (f"{'layout':<10} {'policy':<16} {'success':>8} {'collide':>8} {'timeout':>8} {'DE entries':>11} "
            f"{'re-entries':>11} {'branches':>9} {'backtrack m':>12} {'t_goal':>7} {'kJ':>6}")
    print(head + "\n" + "-" * len(head))
    for lay in args.layouts:
        for n in dict.fromkeys(r["policy"] for r in rows):
            rs = [r for r in rows if r["layout"] == lay and r["policy"] == n]
            if not rs:
                continue
            m = lambda k: float(np.mean([r[k] for r in rs]))
            succ = [r for r in rs if r["outcome"] == "success"]
            tg = float(np.mean([r["time"] for r in succ])) if succ else float("nan")
            frac = lambda o: 100 * np.mean([r["outcome"] == o for r in rs])
            print(f"{lay:<10} {n[:16]:<16} {frac('success'):7.0f}% {frac('collision'):7.0f}% {frac('timeout'):7.0f}% "
                  f"{m('dead_end_entries'):11.2f} {m('dead_end_reentries'):11.2f} {m('branches_explored'):9.2f} "
                  f"{m('backtrack_m'):12.1f} {tg:7.1f} {m('energy_kj'):6.1f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

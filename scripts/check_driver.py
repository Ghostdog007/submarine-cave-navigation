"""Phase A0 check: does the driver (hand controller + built-in safety filter) drive safely in every cave type?

The driver gets a scripted teacher navigator (goal-ward exits, no memory), so success in mazes is NOT the
point; what matters is: collisions ~0, little hull damage, few stalls, and how often the filter had to step in.
The gap follower is the reference.

    python -m scripts.check_driver                                   # all layouts, 30 test caves, d 0.5
    python -m scripts.check_driver --layouts crack tunnel --n-caves 10 --damage
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

LAYOUTS = ["tunnel", "maze", "branches", "crack", "breakdown"]


def _job(args):
    name, cfg, layout, d, n_caves = args
    import time

    import numpy as np

    from submarine.evaluation import evaluate, make_policy

    pol = make_policy(name)
    t0 = time.perf_counter()
    res = evaluate(pol, replace(cfg, layout=layout), [d], n_caves)
    wall = time.perf_counter() - t0
    steps = sum(r.steps for r in res)
    rate = getattr(pol, "override_rate", float("nan"))
    rows = [{"policy": r.policy, "layout": layout, "cave_seed": r.cave_seed, "outcome": r.outcome,
             "time": r.time, "min_clearance": r.min_clearance, "contacts": r.contacts,
             "hull_damage": r.hull_damage, "energy_kj": r.energy_kj} for r in res]
    return rows, {"policy": res[0].policy, "layout": layout, "override_rate": rate,
                  "ms_per_step": 1000 * wall / max(steps, 1), "mean_speed": float(np.mean(
                      [r.path_length / max(r.time, 1e-9) for r in res]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_layouts.yaml")
    p.add_argument("--layouts", nargs="+", default=LAYOUTS)
    p.add_argument("--policies", nargs="+", default=["driver", "gap"])
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--n-caves", type=int, default=30)
    p.add_argument("--damage", action="store_true", help="hull-damage contact model instead of crash = end")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--name", default="driver_check")
    args = p.parse_args()

    import csv

    import numpy as np

    from submarine.env import EnvConfig

    cfg = EnvConfig.from_yaml(args.config)
    if args.damage:
        cfg = replace(cfg, collision=replace(cfg.collision, mode="damage"))
    jobs = [(n, cfg, lay, args.difficulty, args.n_caves) for lay in args.layouts for n in args.policies]
    with ProcessPoolExecutor(args.workers) as ex:
        out = list(ex.map(_job, jobs))
    rows = [r for chunk, _ in out for r in chunk]
    extra = {(e["policy"], e["layout"]): e for _, e in out}

    head = (f"{'layout':10s} {'policy':8s} {'success':>8s} {'collide':>8s} {'timeout':>8s} {'min clr':>8s} "
            f"{'contacts':>9s} {'hull -%':>8s} {'speed':>6s} {'filter':>7s} {'ms/step':>8s}")
    print(head)
    print("-" * len(head))
    for lay in args.layouts:
        for pol in sorted({r["policy"] for r in rows if r["layout"] == lay}):
            g = [r for r in rows if r["layout"] == lay and r["policy"] == pol]
            frac = lambda o: 100 * np.mean([r["outcome"] == o for r in g])
            e = extra[(pol, lay)]
            filt = f"{100 * e['override_rate']:6.1f}%" if np.isfinite(e["override_rate"]) else "      -"
            print(f"{lay:10s} {pol:8s} {frac('success'):7.0f}% {frac('collision'):7.0f}% {frac('timeout'):7.0f}% "
                  f"{np.mean([r['min_clearance'] for r in g]):8.2f} {np.mean([r['contacts'] for r in g]):9.1f} "
                  f"{np.mean([r['hull_damage'] for r in g]):8.1f} {e['mean_speed']:6.2f} {filt} "
                  f"{e['ms_per_step']:8.2f}")
    path = Path("runs/eval") / f"{args.name}_episodes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()

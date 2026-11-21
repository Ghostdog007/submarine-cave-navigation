"""Evaluate navigators (phase B) on held-out test caves, per cave type, over the frozen driver.

    python -m scripts.eval_navigator --models runs/nav_none runs/nav_retention runs/nav_junction
    python -m scripts.eval_navigator --policies greedy random --n-caves 10 --layouts maze

Every policy faces the same caves, sensor noise and currents (fixed per-episode seeds). Reported per cave type:
success / collision (hull breach) / timeout / abort, time to goal, energy, decisions per dive, hull damage, and the
route-memory measures from evaluation/deadends.py: dead-end branch entries, re-entries of an already explored
dead end, and metres of backtracking. CSV: runs/eval/<name>_episodes.csv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

LAYOUTS = ["tunnel", "maze", "branches", "crack", "breakdown"]


class ModelNavigator:
    """A trained navigator (PPO or RecurrentPPO), deterministic, with its recurrent state across decisions."""

    def __init__(self, path: str):
        path = Path(path)
        zip_ = path / "final_model.zip" if path.is_dir() else path
        import yaml

        cfg_file = (path if path.is_dir() else path.parent) / "config.yaml"
        memory = yaml.safe_load(open(cfg_file)).get("memory", "none") if cfg_file.exists() else "none"
        if memory == "none":
            from stable_baselines3 import PPO

            self.model, self.recurrent = PPO.load(zip_, device="cpu"), False
        else:
            from sb3_contrib import RecurrentPPO

            from submarine.memory.policies import POLICIES

            cls = POLICIES["RetentionNavPolicy" if memory == "retention" else "JunctionNavPolicy"]
            self.model = RecurrentPPO.load(zip_, device="cpu", custom_objects={"policy_class": cls})
            self.recurrent = True
        self.name = (path.name if path.is_dir() else f"{path.parent.name}/{path.stem}")
        self.memory = memory

    def reset(self):
        self.state, self.start = None, np.ones(1, dtype=bool)

    def act(self, obs):
        if not self.recurrent:
            return int(self.model.predict(obs, deterministic=True)[0])
        a, self.state = self.model.predict(obs[None], state=self.state, episode_start=self.start, deterministic=True)
        self.start = np.zeros(1, dtype=bool)
        return int(a[0])


def make_navigator(name: str):
    from submarine.navigator import GreedyNavigator, OracleNavigator, RandomNavigator

    if name == "greedy":
        return GreedyNavigator()
    if name == "oracle":
        return OracleNavigator()
    if name == "random":
        return RandomNavigator()
    return ModelNavigator(name)


def run_episode(env, nav, cave_seed: int, difficulty: float):
    from submarine.evaluation.metrics import episode_seed

    obs, info = env.reset(seed=episode_seed(cave_seed, difficulty), options={"cave_seed": cave_seed,
                                                                              "difficulty": difficulty})
    if hasattr(nav, "attach"):
        nav.attach(env)                                   # privileged reference navigators read the true map
    nav.reset()
    ret = 0.0
    while True:
        obs, r, te, tr, info = env.step(nav.act(obs))
        ret += r
        if te or tr:
            break
    outcome = ("success" if info["is_success"] else "collision" if info["collided"]
               else "abort" if info.get("aborted") else "timeout")
    return outcome, info, ret


def _job(args):
    name, cfg, layout, d, n_caves = args
    from dataclasses import replace

    from submarine.evaluation.deadends import trajectory_stats
    from submarine.navigator import NavigatorEnv
    from submarine.world import test_seeds

    env = NavigatorEnv(replace(cfg, env=replace(cfg.env, layout=layout)))
    nav = make_navigator(name)
    rows = []
    for s in test_seeds(n_caves):
        outcome, info, ret = run_episode(env, nav, s, d)
        row = {"policy": nav.name, "layout": layout, "cave_seed": s, "outcome": outcome,
               "time": info["t"] * env.dt, "energy_kj": info["energy_j"] / 1000.0, "decisions": env.n_decisions,
               "hull": info.get("hull", 100.0), "contacts": info.get("contacts", 0), "return": ret,
               "path_length": info["path_length"]}
        row.update(trajectory_stats(env.base.cave, env.base._trail))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/navigator.yaml")
    p.add_argument("--eval-config", default="configs/eval_layouts.yaml", help="test conditions (EnvConfig YAML)")
    p.add_argument("--policies", nargs="*", default=["greedy"])
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--layouts", nargs="+", default=LAYOUTS)
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--n-caves", type=int, default=30)
    p.add_argument("--driver", default=None)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--name", default="navigators")
    args = p.parse_args()

    import yaml

    from submarine.env import EnvConfig
    from submarine.navigator import NavigatorEnvConfig

    C = yaml.safe_load(open(args.config))
    env = EnvConfig.from_yaml(args.eval_config).to_dict()
    cfg = NavigatorEnvConfig.from_dict({**{k: v for k, v in C["base"].items() if k != "env"},
                                        **({"driver_actor": args.driver} if args.driver else {}), "env": env})
    names = args.policies + args.models
    jobs = [(n, cfg, lay, args.difficulty, args.n_caves) for lay in args.layouts for n in names]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for chunk in ex.map(_job, jobs) for r in chunk]

    head = (f"{'layout':10s} {'policy':16s} {'success':>8s} {'breach':>7s} {'timeout':>8s} {'t_goal':>7s} "
            f"{'kJ':>6s} {'decide':>7s} {'hull':>6s} {'DE in':>6s} {'re-ent':>7s} {'back m':>7s}")
    print(head)
    print("-" * len(head))
    for lay in args.layouts:
        for pol in dict.fromkeys(r["policy"] for r in rows if r["layout"] == lay):
            g = [r for r in rows if r["layout"] == lay and r["policy"] == pol]
            f = lambda o: 100 * np.mean([r["outcome"] == o for r in g])
            succ = [r["time"] for r in g if r["outcome"] == "success"]
            m = lambda k: np.mean([r[k] for r in g])
            print(f"{lay:10s} {pol:16s} {f('success'):7.0f}% {f('collision'):6.0f}% {f('timeout'):7.0f}% "
                  f"{(np.mean(succ) if succ else float('nan')):7.1f} {m('energy_kj'):6.1f} {m('decisions'):7.1f} "
                  f"{m('hull'):6.1f} {m('dead_end_entries'):6.2f} {m('dead_end_reentries'):7.2f} "
                  f"{m('backtrack_m'):7.1f}")
    out = Path("runs/eval") / f"{args.name}_episodes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

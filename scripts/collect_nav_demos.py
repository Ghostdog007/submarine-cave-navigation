"""Collect navigator demonstrations from the map-aware teacher (OracleNavigator), for the warm start.

    python -m scripts.collect_nav_demos --decisions 40000 --out runs/nav_demos.npz

Episodes run over the frozen driver on *training* caves, with the curriculum's stage mix (weighted by stage
budget). The executed action is the teacher's, or with probability `--noise` a random valid exit, so the data
also covers wrong turns and their recovery (entering a dead end, turning back, not going in again). The label is
always the teacher's choice. Saved per decision: observation, label, executed action, reward, episode start.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml


def _worker(args):
    base, stages, weights, n_decisions, noise, seed = args
    from scripts.train_navigator import stage_config
    from submarine.navigator import NavigatorEnv, OracleNavigator, slot_valid

    rng = np.random.default_rng(seed)
    envs = [NavigatorEnv(stage_config(base, s)) for s in stages]
    teacher = OracleNavigator()
    O, A, X, R, S = [], [], [], [], []
    ep = 0
    while len(O) < n_decisions:
        env = envs[rng.choice(len(envs), p=weights)]
        obs, _ = env.reset(seed=int(seed * 100003 + ep))
        teacher.attach(env)
        ep += 1
        start = True
        while True:
            label = teacher.act(obs)
            if rng.random() < noise:
                valid = np.flatnonzero(slot_valid(obs))
                act = int(rng.choice(valid)) if valid.size else 4
            else:
                act = label
            nobs, r, te, tr, _ = env.step(act)
            O.append(obs); A.append(label); X.append(act); R.append(r); S.append(start)
            start = False
            obs = nobs
            if te or tr:
                break
    return (np.array(O, np.float32), np.array(A, np.int64), np.array(X, np.int64), np.array(R, np.float32),
            np.array(S, bool))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/navigator.yaml")
    p.add_argument("--decisions", type=int, default=40000)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--out", default="runs/nav_demos.npz")
    args = p.parse_args()
    C = yaml.safe_load(open(args.config))
    stages = C["stages"]
    w = np.array([s["steps"] for s in stages], float)
    w /= w.sum()
    per = -(-args.decisions // args.workers)
    t0 = time.perf_counter()
    jobs = [(C["base"], stages, w, per, args.noise, 7000 + i) for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(_worker, jobs))
    data = {k: np.concatenate([p_[i] for p_ in parts]) for i, k in enumerate(["obs", "label", "action", "reward",
                                                                               "start"])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    n, eps = len(data["obs"]), int(data["start"].sum())
    print(f"{n} decisions, {eps} episodes in {time.perf_counter() - t0:.0f} s -> {args.out}; "
          f"labels {np.bincount(data['label'], minlength=5).tolist()}, "
          f"teacher overridden by noise {100 * np.mean(data['label'] != data['action']):.0f}%")


if __name__ == "__main__":
    main()

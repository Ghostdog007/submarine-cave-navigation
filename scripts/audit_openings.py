"""Audit the openings detector against the true map, on what the navigator needs from it.

At random feasible positions/headings in test caves of every layout, with an ideal sonar scan:

  route recall   the true way toward the goal (direction to the point 3 m along the shortest path) is within
                 35 degrees of one of the detected exits. Missing it would leave the navigator no right choice.
  pocket rate    share of detected exits that don't lead anywhere: free space reachable inside a +-25 degree
                 sector around the exit ends within 4.5 m (an alcove, not a way on).
  exits          mean number of exits offered (incl. the way back).

    python -m scripts.audit_openings
    python -m scripts.audit_openings --half-width 0.45 --prominence 3.5
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np

LAYOUTS = ["tunnel", "maze", "branches", "crack", "breakdown"]


def route_bearing(cave, x, y, psi, ahead=3.0):
    """Body-frame bearing of the shortest path toward the goal (None if within `ahead` of the goal)."""
    from submarine.world.geodesic import descend

    res = cave.world.resolution
    cell = (int(y / res), int(x / res))
    if not np.isfinite(cave.geodesic[cell]) or cave.geodesic[cell] < ahead + 0.5:
        return None
    path = descend(cave.geodesic, cell, res, max_length=ahead)
    iy, ix = path[-1]
    b = np.arctan2((iy + 0.5) * res - y, (ix + 0.5) * res - x) - psi
    return float((b + np.pi) % (2 * np.pi) - np.pi)


def leads_on(feasible, res, x, y, heading, reach=4.5, half_sector_deg=25.0, limit=6.0) -> bool:
    """Is there feasible space >= `reach` m away, connected through the sector around `heading`?"""
    h, w = feasible.shape
    ci, cj = int(y / res), int(x / res)
    half = np.deg2rad(half_sector_deg)
    seen = {(ci, cj)}
    q = deque([(ci, cj)])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if (a, b) in seen or not (0 <= a < h and 0 <= b < w) or not feasible[a, b]:
                continue
            dy, dx = (a - ci) * res, (b - cj) * res
            d = np.hypot(dx, dy)
            if d > limit:
                continue
            if d > 0.8:                                      # near the vehicle any direction is fine
                off = (np.arctan2(dy, dx) - heading + np.pi) % (2 * np.pi) - np.pi
                if abs(off) > half:
                    continue
            if d >= reach:
                return True
            seen.add((a, b))
            q.append((a, b))
    return False


def audit(cfg, n_caves=8, per_cave=30, difficulty=0.5, seed=0):
    from submarine.control import detect_openings
    from submarine.sim import Sonar, SonarConfig
    from submarine.world import generate_cave, test_seeds

    sonar = Sonar(SonarConfig().ideal())
    rng = np.random.default_rng(seed)
    out = {}
    for lay in LAYOUTS:
        hits = n_route = pockets = n_exits = n_pts = 0
        for s in test_seeds(n_caves):
            cave = generate_cave(s, difficulty, layout=lay)
            res = cave.world.resolution
            ii, jj = np.nonzero(cave.feasible)
            for k in rng.choice(len(ii), size=min(per_cave, len(ii)), replace=False):
                y, x = (ii[k] + 0.5) * res, (jj[k] + 0.5) * res
                psi = rng.uniform(-np.pi, np.pi)
                ops = detect_openings(sonar.true_ranges(cave.rock, res, x, y, psi), sonar.angles, 10.0, cfg)
                n_pts += 1
                n_exits += len(ops)
                rb = route_bearing(cave, x, y, psi)
                if rb is not None:
                    n_route += 1
                    hits += any(abs((o.bearing - rb + np.pi) % (2 * np.pi) - np.pi) < np.deg2rad(35) for o in ops)
                pockets += sum(not leads_on(cave.feasible, res, x, y, psi + o.bearing) for o in ops)
        out[lay] = {"recall": hits / max(n_route, 1), "pocket": pockets / max(n_exits, 1),
                    "exits": n_exits / max(n_pts, 1)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--half-width", type=float, default=None)
    p.add_argument("--open-range", type=float, default=None)
    p.add_argument("--prominence", type=float, default=None)
    p.add_argument("--n-caves", type=int, default=8)
    p.add_argument("--per-cave", type=int, default=30)
    args = p.parse_args()

    from dataclasses import replace

    from submarine.control import OpeningsConfig

    cfg = OpeningsConfig()
    for k in ("half_width", "open_range", "prominence"):
        if getattr(args, k) is not None:
            cfg = replace(cfg, **{k: getattr(args, k)})
    print(cfg)
    res = audit(cfg, args.n_caves, args.per_cave)
    print(f"{'layout':10s} {'route recall':>12s} {'pockets':>8s} {'exits':>6s}")
    for lay, r in res.items():
        print(f"{lay:10s} {100 * r['recall']:11.0f}% {100 * r['pocket']:7.0f}% {r['exits']:6.2f}")


if __name__ == "__main__":
    main()

"""Plot a grid of caves (rows = difficulty, columns = seeds) and print generator stats.

    python -m scripts.plot_caves
    python -m scripts.plot_caves --difficulties 0 0.5 1 --seeds 0 1 2 --geodesic
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from submarine.world import generate_cave


def draw_cave(ax, cave, show_geodesic: bool = False):
    w = cave.world
    extent = (0, w.length, 0, w.height)
    # 0 = rock, 1 = free but too tight for the vehicle, 2 = feasible
    img = np.where(cave.rock, 0, np.where(cave.feasible, 2, 1))
    ax.imshow(img, origin="lower", extent=extent, cmap="Blues", vmin=-0.5, vmax=2.5, interpolation="nearest")
    if show_geodesic:
        g = np.where(np.isfinite(cave.geodesic), cave.geodesic, np.nan)
        ax.contour(np.linspace(0, w.length, w.nx), np.linspace(0, w.height, w.ny), g,
                   levels=np.arange(0, np.nanmax(g), 5.0), colors="k", linewidths=0.4, alpha=0.6)
    for x0, x1, _ in cave.cracks:
        ax.axvspan(x0, x1, color="orange", alpha=0.15, lw=0)
    ax.plot(*cave.start, "o", color="limegreen", ms=5, mec="k", mew=0.5)
    ax.plot(*cave.goal, "*", color="red", ms=8, mec="k", mew=0.5)
    ax.set_xlim(0, w.length)
    ax.set_ylim(0, w.height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"d={cave.difficulty:.2f}  seed {cave.seed}  path {cave.shortest_path_length:.1f} m", fontsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--difficulties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--geodesic", action="store_true", help="overlay geodesic distance contours (every 5 m)")
    p.add_argument("--stats-seeds", type=int, default=50, help="caves per difficulty for the stats table")
    p.add_argument("--out", type=Path, default=Path("runs/figures/caves.png"))
    args = p.parse_args()

    nr, nc = len(args.difficulties), len(args.seeds)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.4 * nr), squeeze=False)
    for i, d in enumerate(args.difficulties):
        for j, s in enumerate(args.seeds):
            draw_cave(axes[i, j], generate_cave(s, d), args.geodesic)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")

    print(f"\nstats over {args.stats_seeds} seeds per difficulty")
    print(f"{'d':>5} {'ms/cave':>8} {'attempts':>9} {'retry%':>7} {'path m':>7} {'feasible%':>10} {'cracks':>7} {'pillars':>8}")
    for d in args.difficulties:
        t0 = time.perf_counter()
        caves = [generate_cave(s, d) for s in range(args.stats_seeds)]
        ms = (time.perf_counter() - t0) / len(caves) * 1e3
        att = np.array([c.attempt + 1 for c in caves])
        print(f"{d:5.2f} {ms:8.1f} {att.mean():9.2f} {100 * (att > 1).mean():7.0f} "
              f"{np.mean([c.shortest_path_length for c in caves]):7.1f} "
              f"{100 * np.mean([c.feasible.mean() for c in caves]):10.1f} "
              f"{np.mean([len(c.cracks) for c in caves]):7.2f} {np.mean([len(c.pillars) for c in caves]):8.2f}")


if __name__ == "__main__":
    main()

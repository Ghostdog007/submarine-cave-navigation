"""Catalogue of test scenarios: each cave layout next to the real cave-exploration situation it models.

    python -m scripts.plot_scenarios                    # -> docs/figures/scenario_catalogue.png
    python -m scripts.plot_scenarios --difficulty 0.8 --seeds 1000003 1000004 1000005
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from submarine.world import descend, generate_cave
from submarine.env import EnvConfig
from scripts.plot_caves import draw_cave
from submarine.sim import make_turbidity

SCENARIOS = [
    ("tunnel", "Single winding passage (reference)",
     "The baseline environment: one route, cracks, rough walls and pillars. The agents were trained here."),
    ("maze", "Labyrinth / anastomosing cave system",
     "Many junctions and side leads, most ending blind. Real cave divers lay a continuous guideline and mark "
     "explored leads, because without memory you revisit the same dead ends. Tests: memory, exploration."),
    ("branches", "Dead-end leads that point the right way",
     "Side passages longer than sonar range that curve out of sight; about half head toward the exit, so the "
     "goal bearing actively misleads. Tests: memory, not trusting the bearing blindly."),
    ("crack", "Glacier crevasse / ice fissure with restrictions",
     "Angular zigzag, 'squeezes' barely wider than the vehicle, side fissures that look like openings but are too "
     "narrow to enter. Tests: precise low-speed control in tight clearance."),
    ("breakdown", "Breakdown chamber / boulder choke",
     "Roof collapse leaves a wide chamber full of rubble; many gaps look passable, few connect. Tests: local "
     "route-finding through clutter."),
    ("siltout", "Silt-out (zero visibility)",
     "Disturbed sediment drops visibility to zero, the classic cause of diver disorientation. The camera goes "
     "blind (brown), sonar keeps working. Tests: falling back on sonar."),
]


def route(cave):
    res = cave.world.resolution
    cells = descend(cave.geodesic, cave.world_to_cell(*cave.start), res)
    return np.array([((ix + 0.5) * res, (iy + 0.5) * res) for iy, ix in cells])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[1000000, 1000001, 1000002])
    p.add_argument("--out", type=Path, default=Path("docs/figures/scenario_catalogue.png"))
    args = p.parse_args()

    silt = EnvConfig.from_yaml("configs/scenario_siltout.yaml").turbidity
    n = len(args.seeds)
    fig = plt.figure(figsize=(4.2 + 3.6 * n, 2.55 * len(SCENARIOS)))
    gs = fig.add_gridspec(len(SCENARIOS), n + 1, width_ratios=[1.25] + [1] * n, wspace=0.05, hspace=0.25,
                          top=0.95, bottom=0.01)
    for i, (layout, title, text) in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[i, 0])
        ax.axis("off")
        ax.text(0.0, 0.97, textwrap.fill(title, 34), fontsize=10.5, fontweight="bold", va="top",
                transform=ax.transAxes)
        ax.text(0.0, 0.62, textwrap.fill(text, 46), fontsize=8.0, va="top", transform=ax.transAxes, color="#333")
        for j, s in enumerate(args.seeds):
            cave = generate_cave(s, args.difficulty, layout="tunnel" if layout == "siltout" else layout)
            ax = fig.add_subplot(gs[i, j + 1])
            draw_cave(ax, cave)
            if layout == "siltout":
                t = make_turbidity(cave, silt)
                rgba = np.zeros((*t.shape, 4))
                rgba[..., :3] = np.array([126, 110, 66]) / 255
                rgba[..., 3] = np.clip(t, 0, 1) * 0.85 * ~cave.rock
                w = cave.world
                ax.imshow(rgba, origin="lower", extent=(0, w.length, 0, w.height), interpolation="bilinear")
            r = route(cave)
            ax.plot(r[:, 0], r[:, 1], "--", color="white", lw=1.0, alpha=0.9)
            ax.set_title(f"cave {s}   shortest route {cave.shortest_path_length:.0f} m", fontsize=8)
    fig.suptitle(f"Test scenarios (difficulty {args.difficulty}); dashed = shortest route, "
                 f"green = start, red = exit", fontsize=12, y=0.985)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

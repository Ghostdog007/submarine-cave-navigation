"""Dead-end behaviour from trajectories (analysis only; agents never see any of this).

Main route = shortest path from start to exit. Dead-end branches = connected regions of passable water more than
`margin` metres from the main route; in maze / branches caves these are exactly the dead ends.

Per episode:
  dead_end_entries    times the agent entered a dead-end branch
  dead_end_reentries  entries into a branch it had already explored earlier in the episode (memory failure)
  branches_explored   distinct dead-end branches visited
  backtrack_m         progress lost: metres by which the distance-to-exit went *up* (turning back, detours)
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage

from submarine.world import descend
from submarine.sim import sample_bilinear


def branch_labels(cave, margin: float = 2.0, min_area_m2: float = 3.0):
    res = cave.world.resolution
    route = np.zeros_like(cave.feasible)
    for iy, ix in descend(cave.geodesic, cave.world_to_cell(*cave.start), res):
        route[iy, ix] = True
    dist = ndimage.distance_transform_edt(~route) * res
    off = cave.feasible & (dist > margin)
    labels, n = ndimage.label(off, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1)) * res * res
    for k, s in enumerate(sizes, start=1):
        if s < min_area_m2:
            labels[labels == k] = 0
    return labels


def trajectory_stats(cave, trajectory, labels=None) -> dict:
    labels = branch_labels(cave) if labels is None else labels
    entries: Counter = Counter()
    prev = 0
    for x, y in trajectory:
        lab = int(labels[cave.world_to_cell(x, y)])
        if lab and lab != prev:
            entries[lab] += 1
        prev = lab
    g = np.array([sample_bilinear(cave.geodesic, x, y, cave.world.resolution) for x, y in trajectory])
    g = g[np.isfinite(g)]
    return {
        "dead_end_entries": int(sum(entries.values())),
        "dead_end_reentries": int(sum(max(0, c - 1) for c in entries.values())),
        "branches_explored": len(entries),
        "backtrack_m": float(np.clip(np.diff(g), 0, None).sum()) if len(g) > 1 else 0.0,
    }

"""Sampling grid fields (clearance, geodesic distance) at continuous world positions."""

from __future__ import annotations

import math

import numpy as np


def sample_bilinear(grid: np.ndarray, x, y, resolution: float):
    """Bilinear interpolation of a cell-centred grid indexed [iy, ix] at world (x, y).

    Non-finite corners (e.g. inf geodesic distance in rock) are ignored and the
    remaining weights renormalised; if no finite corner has weight, returns inf.
    Positions outside the grid are clamped to the edge. Works on scalars or arrays.
    """
    ny, nx = grid.shape
    if isinstance(x, (float, int)) and isinstance(y, (float, int)):
        return _sample_scalar(grid, float(x), float(y), resolution, nx, ny)
    gx = np.asarray(x, dtype=float) / resolution - 0.5
    gy = np.asarray(y, dtype=float) / resolution - 0.5
    x0 = np.clip(np.floor(gx).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(gy).astype(int), 0, ny - 2)
    fx = np.clip(gx - x0, 0.0, 1.0)
    fy = np.clip(gy - y0, 0.0, 1.0)

    vals = np.stack([grid[y0, x0], grid[y0, x0 + 1], grid[y0 + 1, x0], grid[y0 + 1, x0 + 1]])
    w = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
    finite = np.isfinite(vals)
    w = np.where(finite, w, 0.0)
    wsum = w.sum(axis=0)
    total = np.where(finite, vals, 0.0)
    out = np.where(wsum > 1e-12, (w * total).sum(axis=0) / np.maximum(wsum, 1e-12), np.inf)
    return float(out) if out.ndim == 0 else out


def _sample_scalar(grid, x: float, y: float, resolution: float, nx: int, ny: int) -> float:
    """Same arithmetic as the array path, in the same order (bit-identical), without NumPy overhead:
    the simulator samples single points 3x per step."""
    gx = x / resolution - 0.5
    gy = y / resolution - 0.5
    x0 = min(max(math.floor(gx), 0), nx - 2)
    y0 = min(max(math.floor(gy), 0), ny - 2)
    fx = min(max(gx - x0, 0.0), 1.0)
    fy = min(max(gy - y0, 0.0), 1.0)
    vals = (float(grid[y0, x0]), float(grid[y0, x0 + 1]), float(grid[y0 + 1, x0]), float(grid[y0 + 1, x0 + 1]))
    ws = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
    wsum = 0.0
    total = 0.0
    for w, v in zip(ws, vals):
        if math.isfinite(v):
            wsum += w
            total += w * v
        else:
            wsum += 0.0
            total += 0.0
    return total / max(wsum, 1e-12) if wsum > 1e-12 else math.inf

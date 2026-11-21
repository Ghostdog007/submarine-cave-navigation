"""Per-cave murky-water field and wall albedo (deterministic from the cave seed)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import TurbidityConfig


def make_turbidity(cave, cfg: TurbidityConfig) -> np.ndarray:
    """Turbidity grid (ny, nx) in [0, 1]: a few smooth murky patches centred in open water."""
    if not cfg.enabled or cfg.scale <= 0:
        return np.zeros(cave.rock.shape, dtype=np.float32)
    rng = np.random.default_rng([cave.seed, 7])
    res = cave.world.resolution
    ny, nx = cave.rock.shape
    ys, xs = (np.mgrid[0:ny, 0:nx] + 0.5) * res
    free_y, free_x = np.nonzero(cave.feasible)

    field = np.full((ny, nx), cfg.background, dtype=float)
    for _ in range(rng.integers(cfg.n_zones[0], cfg.n_zones[1] + 1)):
        i = rng.integers(len(free_x))
        cx, cy = (free_x[i] + 0.5) * res, (free_y[i] + 0.5) * res
        sigma = rng.uniform(*cfg.radius) / 2
        field += rng.uniform(*cfg.peak) * np.exp(-0.5 * ((xs - cx) ** 2 + (ys - cy) ** 2) / sigma**2)
    # irregular, wispy edges
    wisp = gaussian_filter(rng.standard_normal((ny, nx)), 4.0)
    field *= 1.0 + 0.6 * wisp / (wisp.std() + 1e-9)
    return np.clip(field * cfg.scale, 0.0, 1.0).astype(np.float32)


def make_albedo(cave) -> np.ndarray:
    """Wall reflectivity grid in [0.6, 1.0] (smooth variation, like patchy ice)."""
    rng = np.random.default_rng([cave.seed, 11])
    n = gaussian_filter(rng.standard_normal(cave.rock.shape), 3.0)
    n = (n - n.min()) / (np.ptp(n) + 1e-9)
    return (0.6 + 0.4 * n).astype(np.float32)

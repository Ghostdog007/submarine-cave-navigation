"""Multi-beam sonar by ray marching on the occupancy grid (context.md, section 4.1)."""

from __future__ import annotations

import numpy as np

from .config import SonarConfig


class Sonar:
    def __init__(self, cfg: SonarConfig = SonarConfig()):
        self.cfg = cfg
        n = cfg.n_beams
        if cfg.fov_deg >= 360.0:
            # evenly around the circle, beam n//2 points straight ahead
            self.angles = (np.arange(n) - n // 2) * (2 * np.pi / n)
        else:
            half = np.deg2rad(cfg.fov_deg) / 2
            self.angles = np.linspace(-half, half, n) if n > 1 else np.zeros(1)

        if cfg.beam_width_deg > 0 and cfg.rays_per_beam > 1:
            w = np.deg2rad(cfg.beam_width_deg)
            sub = np.linspace(-w / 2, w / 2, cfg.rays_per_beam)
        else:
            sub = np.zeros(1)
        self._ray_angles = self.angles[:, None] + sub[None, :]               # (beams, rays)
        n_steps = int(np.ceil(cfg.max_range / cfg.step))
        self._dists = np.arange(1, n_steps + 1) * cfg.step                     # (steps,)

    def true_ranges(self, rock: np.ndarray, resolution: float, x: float, y: float, psi: float) -> np.ndarray:
        """Noise-free range per beam in metres. Off-map counts as rock."""
        ny, nx = rock.shape
        a = psi + self._ray_angles
        px = x + np.cos(a)[..., None] * self._dists                           # (beams, rays, steps)
        py = y + np.sin(a)[..., None] * self._dists
        ix = np.floor(px / resolution).astype(int)
        iy = np.floor(py / resolution).astype(int)
        oob = (ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)
        hit = oob | rock[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]
        first = hit.argmax(axis=-1)
        # the surface lies between the last free sample and the first rock sample
        r = np.where(hit.any(axis=-1), self._dists[first] - self.cfg.step / 2, self.cfg.max_range)
        return np.minimum(r.min(axis=1), self.cfg.max_range)

    def measure(self, rock, resolution, x, y, psi, rng: np.random.Generator) -> np.ndarray:
        """Ranges with Gaussian noise, blind zone and dropout, in metres within [min_range, max_range]."""
        c = self.cfg
        r = self.true_ranges(rock, resolution, x, y, psi)
        if c.noise_std > 0 or c.noise_frac > 0:
            r = r + rng.standard_normal(r.shape) * (c.noise_std + c.noise_frac * r)
        r = np.clip(r, c.min_range, c.max_range)
        if c.dropout > 0:
            r = np.where(rng.random(r.shape) < c.dropout, c.max_range, r)
        return r

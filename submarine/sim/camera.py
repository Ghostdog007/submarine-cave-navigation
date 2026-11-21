"""Forward camera with headlight and underwater image formation (1D scanline in the 2D world).

Per pixel ray, with optical depth tau(s) = integral of c along the ray and
c = c_clear + c_turbid * turbidity:
    wall      = albedo * cone * illum(d_hit) * exp(-2 tau(d_hit))      light out to the wall and back
    backscat  = sum over s < d_hit of b(s) * ds * cone * illum(s) * exp(-2 tau(s))
    image     = gain * (wall + backscat) + noise, clipped to [0, 1]
with illum(d) = 1 / (1 + (d / light_range)^2) and b = b_clear + b_turbid * turbidity.
In clear water nearby walls are bright and distance is readable from
brightness; in murk the wall term vanishes within a metre or two while the
backscatter veil (and its speckle) grows, so the image stops being useful.
"""

from __future__ import annotations

import numpy as np

from .config import CameraConfig


class Camera:
    def __init__(self, cfg: CameraConfig = CameraConfig()):
        self.cfg = cfg
        half = np.deg2rad(cfg.fov_deg) / 2
        n = cfg.n_pixels
        # pixel i looks at angle angles[i]; i = 0 is the rightmost pixel (negative angle)
        self.angles = -half + (np.arange(n) + 0.5) * (2 * half / n)
        self._dists = np.arange(1, int(np.ceil(cfg.max_range / cfg.step)) + 1) * cfg.step
        self._illum = 1.0 / (1.0 + (self._dists / cfg.light_range) ** 2)
        self._cone = np.exp(-0.5 * (self.angles / np.deg2rad(cfg.cone_deg)) ** 2)

    def render(self, rock, turbidity, albedo, resolution, x, y, psi, rng: np.random.Generator | None = None):
        """Returns (image in [0, 1], true distance to the wall per pixel)."""
        c = self.cfg
        ny, nx = rock.shape
        a = psi + self.angles
        px = x + np.cos(a)[:, None] * self._dists
        py = y + np.sin(a)[:, None] * self._dists
        ix = np.floor(px / resolution).astype(int)
        iy = np.floor(py / resolution).astype(int)
        oob = (ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)
        ixc, iyc = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
        hit = oob | rock[iyc, ixc]
        any_hit = hit.any(axis=1)
        first = np.where(any_hit, hit.argmax(axis=1), len(self._dists))
        before = np.arange(len(self._dists))[None, :] < first[:, None]       # water samples in front of the wall

        turb = np.where(before, turbidity[iyc, ixc], 0.0)
        atten = (c.c_clear + c.c_turbid * turb) * c.step * before
        tau = np.cumsum(atten, axis=1)
        trans2 = np.exp(-2.0 * tau)
        light = self._cone[:, None] * self._illum[None, :]
        back = ((c.b_clear + c.b_turbid * turb) * c.step * light * trans2 * before).sum(axis=1)

        k = np.minimum(first, len(self._dists) - 1)
        rows = np.arange(len(k))
        d_hit = np.where(any_hit, self._dists[k] - c.step / 2, c.max_range)
        tau_hit = np.where(first > 0, tau[rows, np.maximum(k - 1, 0)], 0.0)
        alb = albedo[iyc[rows, k], ixc[rows, k]]
        wall = np.where(any_hit, alb * self._cone / (1 + (d_hit / c.light_range) ** 2) * np.exp(-2 * tau_hit), 0.0)

        img = c.gain * (wall + back)
        if rng is not None:
            img = img + rng.standard_normal(img.shape) * (c.read_noise + c.shot_noise * np.sqrt(np.maximum(img, 0)))
            img = img + rng.standard_normal(img.shape) * c.speckle * c.gain * back
        return np.clip(img, 0.0, 1.0).astype(np.float32), d_hit

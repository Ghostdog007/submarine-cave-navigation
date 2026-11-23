"""Procedural 2D cave generator (docs/context.md, section 2).

Grid convention: arrays are indexed [iy, ix] with shape (ny, nx). Cell (iy, ix)
has its centre at world coordinates ((ix + 0.5) * res, (iy + 0.5) * res).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .config import CaveConfig, WorldConfig
from .geodesic import geodesic_distance


@dataclass
class Cave:
    seed: int
    attempt: int                  # sub-seed that produced a solvable cave
    config: CaveConfig
    world: WorldConfig
    rock: np.ndarray              # bool (ny, nx), True = rock
    clearance: np.ndarray         # float32, m from cell centre to the nearest rock surface (0 in rock)
    feasible: np.ndarray          # bool, clearance >= vehicle radius
    geodesic: np.ndarray          # float32, m to the goal through feasible space (inf if unreachable)
    centerline: np.ndarray        # y_c(x) per column, m
    width: np.ndarray             # nominal width per column (before roughness and pillars), m
    start: tuple[float, float]
    start_heading: float          # rad, centerline tangent at the start
    goal: tuple[float, float]
    pillars: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, r)
    cracks: list[tuple[float, float, float]] = field(default_factory=list)   # (x_start, x_end, width) of the narrow plateau

    @property
    def difficulty(self) -> float:
        return self.config.difficulty

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        res = self.world.resolution
        ix = min(max(int(x / res), 0), self.world.nx - 1)
        iy = min(max(int(y / res), 0), self.world.ny - 1)
        return iy, ix

    @property
    def shortest_path_length(self) -> float:
        return float(self.geodesic[self.world_to_cell(*self.start)])


def generate_cave(
    seed: int,
    difficulty: float = 0.0,
    config: CaveConfig | None = None,
    world: WorldConfig | None = None,
    layout: str = "tunnel",
) -> Cave:
    """Generate a solvable cave. Deterministic in (seed, difficulty, layout, config, world).

    layout: "tunnel" (default, one winding passage) or one of cave.layouts.LAYOUTS.
    """
    cfg = config if config is not None else CaveConfig.from_difficulty(difficulty, layout=layout)
    world = world if world is not None else WorldConfig()
    for attempt in range(cfg.max_attempts):
        rng = np.random.default_rng([seed, attempt])
        cave = _build(rng, cfg, world, seed, attempt)
        if cave is not None:
            return cave
    raise RuntimeError(f"no solvable cave for seed {seed} after {cfg.max_attempts} attempts")


# ---------------------------------------------------------------------------


def _build_layout(rng, cfg: CaveConfig, world: WorldConfig, seed: int, attempt: int) -> Cave | None:
    """Non-tunnel layouts (cave/layouts.py): carve, then finish exactly like a tunnel."""
    from .layouts import build_free, heading_from_route

    res, nx, ny = world.resolution, world.nx, world.ny
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    noise = _smooth_noise(rng, (ny, nx), cfg.roughness_scale / res)
    free, boulders, stats = build_free(rng, cfg, world, noise)
    X = xs[None, :]
    free &= (X > world.end_margin) & (X < world.length - world.end_margin)
    b = int(round(world.border / res))
    free[:b, :] = free[-b:, :] = False
    free[:, :b] = free[:, -b:] = False

    clearance = np.maximum(distance_transform_edt(free) * res - 0.5 * res, 0.0).astype(np.float32)
    feasible = clearance >= world.vehicle_radius
    if "start_xy" in stats:                                   # the layout places the start (e.g. inside rings)
        start_cell = _cell_near(feasible, clearance, xs, ys, stats["start_xy"], radius=1.5)
    else:
        start_cell = _best_cell(feasible, clearance, xs, world.start_window)
    goal_cell = _best_cell(feasible, clearance, xs, world.goal_window)
    if start_cell is None or goal_cell is None:
        return None
    geodesic = geodesic_distance(feasible, goal_cell, res)
    if not np.isfinite(geodesic[start_cell]):
        return None
    cave = Cave(
        seed=seed, attempt=attempt, config=cfg, world=world,
        rock=~free, clearance=clearance, feasible=feasible, geodesic=geodesic,
        centerline=np.full(nx, np.nan), width=np.zeros(nx),
        start=(float(xs[start_cell[1]]), float(ys[start_cell[0]])),
        start_heading=(float(stats["start_heading"]) if "start_heading" in stats      # (not along the route)
                       else heading_from_route(geodesic, start_cell, res)),
        goal=(float(xs[goal_cell[1]]), float(ys[goal_cell[0]])), pillars=boulders, cracks=[],
    )
    cave.stats = stats
    return cave


def _build(rng, cfg: CaveConfig, world: WorldConfig, seed: int, attempt: int) -> Cave | None:
    if cfg.layout != "tunnel":
        return _build_layout(rng, cfg, world, seed, attempt)
    res, nx, ny = world.resolution, world.nx, world.ny
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)

    yc, slope = _centerline(rng, cfg, world, xs)
    width, cracks = _width_profile(rng, cfg, xs)
    noise = _smooth_noise(rng, (ny, nx), cfg.roughness_scale / res)

    # Free space: slope-corrected (approximately perpendicular) distance to the
    # centerline below the local half-width, perturbed by roughness. The
    # roughness is capped relative to the half-width so cracks don't close.
    d_perp = np.abs(Y - yc) / np.sqrt(1.0 + slope**2)
    amp = np.minimum(cfg.roughness, cfg.roughness_max_frac * width / 2)
    free = d_perp < width / 2 + amp * noise
    free &= (X > world.end_margin) & (X < world.length - world.end_margin)

    pillars = _place_pillars(rng, cfg, xs, yc, slope, width, amp)
    for px, py, r in pillars:
        free &= (X - px) ** 2 + (Y - py) ** 2 >= r**2

    b = int(round(world.border / res))
    free[:b, :] = free[-b:, :] = False
    free[:, :b] = free[:, -b:] = False

    clearance = np.maximum(distance_transform_edt(free) * res - 0.5 * res, 0.0).astype(np.float32)
    feasible = clearance >= world.vehicle_radius

    start_cell = _best_cell(feasible, clearance, xs, world.start_window)
    goal_cell = _best_cell(feasible, clearance, xs, world.goal_window)
    if start_cell is None or goal_cell is None:
        return None
    geodesic = geodesic_distance(feasible, goal_cell, res)
    if not np.isfinite(geodesic[start_cell]):
        return None

    def centre(cell):
        return (float(xs[cell[1]]), float(ys[cell[0]]))

    return Cave(
        seed=seed, attempt=attempt, config=cfg, world=world,
        rock=~free, clearance=clearance, feasible=feasible, geodesic=geodesic,
        centerline=yc, width=width,
        start=centre(start_cell), start_heading=float(np.arctan(slope[start_cell[1]])),
        goal=centre(goal_cell), pillars=pillars, cracks=cracks,
    )


def _centerline(rng, cfg: CaveConfig, world: WorldConfig, xs):
    """y_c(x) as a sum of low-frequency sinusoids, scaled to the wander amplitude and slope limit."""
    g = np.zeros_like(xs)
    dg = np.zeros_like(xs)
    for _ in range(cfg.n_sinusoids):
        k = 2 * np.pi / rng.uniform(*cfg.wander_wavelength)
        phase = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0.5, 1.0)
        g += a * np.sin(k * xs + phase)
        dg += a * k * np.cos(k * xs + phase)
    g -= g.mean()

    scale = cfg.wander / np.abs(g).max()
    scale = min(scale, cfg.max_slope / np.abs(dg).max())
    room = world.height / 2 - world.border - cfg.base_width / 2 - 0.5
    scale = min(scale, room / np.abs(g).max())
    return world.height / 2 + scale * g, scale * dg


def _width_profile(rng, cfg: CaveConfig, xs):
    """Width along x: base + low-frequency noise + chambers, then cracks carved in."""
    w = np.full_like(xs, cfg.base_width)
    for _ in range(2):
        k = 2 * np.pi / rng.uniform(10.0, 30.0)
        w += cfg.base_width * cfg.width_noise * rng.uniform(0.5, 1.0) / 2 * np.sin(k * xs + rng.uniform(0, 2 * np.pi))

    for _ in range(rng.integers(cfg.n_chambers[0], cfg.n_chambers[1] + 1)):
        c = rng.uniform(5.0, 45.0)
        sigma = rng.uniform(*cfg.chamber_sigma)
        w += rng.uniform(*cfg.chamber_gain) * np.exp(-0.5 * ((xs - c) / sigma) ** 2)

    cracks: list[tuple[float, float, float]] = []
    n_max = cfg.n_cracks_max
    n_cracks = int(rng.integers(max(0, n_max - 1), n_max + 1)) if n_max > 0 else 0
    lo, hi = cfg.crack_region
    for _ in range(100):
        if len(cracks) >= n_cracks:
            break
        length = rng.uniform(*cfg.crack_length)
        x0 = rng.uniform(lo + cfg.crack_taper, hi - cfg.crack_taper - length)
        x1 = x0 + length
        pad = cfg.crack_taper * 2 + cfg.crack_spacing
        if any(x0 - pad < c1 and x1 + pad > c0 for c0, c1, _ in cracks):
            continue
        cracks.append((x0, x1, cfg.crack_width))
    cracks.sort()

    for x0, x1, wc in cracks:
        t = np.maximum(np.maximum(x0 - xs, xs - x1), 0.0)       # distance outside the plateau
        m = np.where(t < cfg.crack_taper, 0.5 * (1 + np.cos(np.pi * t / cfg.crack_taper)), 0.0)
        w = np.minimum(w, w * (1 - m) + wc * m)
    return w, cracks


def _smooth_noise(rng, shape, sigma_cells: float):
    """Smooth 2D noise in [-1, 1]: Gaussian-filtered white noise, two octaves."""
    coarse = gaussian_filter(rng.standard_normal(shape), sigma_cells, mode="reflect")
    fine = gaussian_filter(rng.standard_normal(shape), sigma_cells / 3, mode="reflect")
    n = coarse / coarse.std() + 0.35 * fine / fine.std()
    return np.clip(n / (2.0 * n.std()), -1.0, 1.0)


def _place_pillars(rng, cfg: CaveConfig, xs, yc, slope, width, rough_amp):
    """Rejection-sample pillars that don't overlap and leave a passable gap on both sides.

    The gap is checked against the narrowest nominal width over the pillar's x
    extent, after subtracting the worst-case roughness intrusion, so pillars
    only land in genuinely wide sections (not next to cracks).
    """
    pillars: list[tuple[float, float, float]] = []
    res = xs[1] - xs[0]
    for _ in range(30 * cfg.n_pillars):
        if len(pillars) >= cfg.n_pillars:
            break
        px = rng.uniform(*cfg.pillar_region)
        r = rng.uniform(*cfg.pillar_radius)
        ix = min(int(px / res), len(xs) - 1)
        span = np.abs(xs - px) <= r + cfg.pillar_gap
        bound = width[span].min() / 2 - r - cfg.pillar_gap - rough_amp[span].max()
        if bound <= 0:
            continue
        offset = rng.uniform(-bound, bound)
        py = yc[ix] + offset * np.sqrt(1 + slope[ix] ** 2)   # perpendicular offset -> vertical
        if any(np.hypot(px - qx, py - qy) < r + qr + 2 * cfg.pillar_gap for qx, qy, qr in pillars):
            continue
        pillars.append((float(px), float(py), float(r)))
    return pillars


def _cell_near(feasible, clearance, xs, ys, xy, radius):
    """Feasible cell with the largest clearance within `radius` m of xy, or None."""
    X, Y = np.meshgrid(xs, ys)
    near = feasible & (np.hypot(X - xy[0], Y - xy[1]) <= radius)
    if not near.any():
        return None
    iy, ix = np.unravel_index(np.argmax(np.where(near, clearance, -1.0)), near.shape)
    return int(iy), int(ix)


def _best_cell(feasible, clearance, xs, window):
    """Feasible cell with the largest clearance inside the x window, or None."""
    cols = np.flatnonzero((xs >= window[0]) & (xs <= window[1]))
    sub = np.where(feasible[:, cols], clearance[:, cols], -1.0)
    if sub.max() < 0:
        return None
    iy, j = np.unravel_index(np.argmax(sub), sub.shape)
    return int(iy), int(cols[j])

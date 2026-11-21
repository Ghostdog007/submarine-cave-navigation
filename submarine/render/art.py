"""Procedural art for the "game" render style: cave textures, caustics, submarine sprite, lights.

Everything is generated from code (no image assets), deterministic per cave
seed. Rendering only: nothing here affects the simulation or the agent.
"""

from __future__ import annotations

import numpy as np
import pygame
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

THEMES = {
    # glacier cave: bright blue-white ice walls with fractures, deep teal water
    "ice": dict(wall_edge=(214, 242, 255), wall_deep=(34, 74, 116), crack=(36, 112, 176), glint=(255, 255, 255),
                water_edge=(30, 118, 150), water_deep=(4, 24, 46), caustic=(110, 205, 235), snow=(205, 235, 255),
                sparkle=True),
    # basalt cave: warm grey-brown rock, darker water
    "rock": dict(wall_edge=(138, 120, 100), wall_deep=(26, 23, 26), crack=(16, 14, 16), glint=(205, 188, 160),
                 water_edge=(32, 98, 122), water_deep=(5, 26, 42), caustic=(105, 190, 200), snow=(215, 222, 205),
                 sparkle=False),
}

SPRITE_PX_PER_M = 200     # resolution the submarine is drawn at before scaling down (for smooth edges)


def _lerp(a, b, t):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t


def _fbm(shape, rng, octaves=((64, 0.45), (32, 0.25), (16, 0.15), (8, 0.09), (4, 0.06))):
    """Fractal value noise at pixel resolution, zero mean, unit std."""
    h, w = shape
    out = np.zeros(shape)
    for cell, weight in octaves:
        n = rng.standard_normal((h // cell + 2, w // cell + 2))
        out += weight * zoom(n, cell, order=3)[:h, :w]
    return out / out.std()


def _to_shape(a, shape):
    h, w = shape
    a = a[:h, :w]
    return np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1])), mode="edge")


def signed_distance_px(cave, scale: float) -> np.ndarray:
    """Signed distance to the rock surface in metres (+ in water, - in rock) at screen pixels, row 0 = top."""
    res = cave.world.resolution
    rock_depth = np.maximum(distance_transform_edt(cave.rock) * res - 0.5 * res, 0.0)
    sd = np.where(cave.rock, -rock_depth, cave.clearance)
    # smooth away the grid staircase before upscaling (moves the drawn shoreline by < 0.1 m;
    # collisions still use the exact grid)
    sd = gaussian_filter(sd, 0.9)
    sd = zoom(sd, scale * res, order=1)
    h, w = int(round(cave.world.height * scale)), int(round(cave.world.length * scale))
    return _to_shape(sd[::-1], (h, w))


def build_background(cave, scale: float, theme: str):
    """Textured cave image (h, w, 3) uint8, water mask (h, w) and signed distance (h, w)."""
    t = THEMES[theme]
    rng = np.random.default_rng(cave.seed + 12345)
    sd = signed_distance_px(cave, scale)
    shape = sd.shape

    # noise textures are defined in metres, so generate them at ~20 px/m and upsample
    k = max(1, int(round(scale / 20)))
    small = (shape[0] // k + 1, shape[1] // k + 1)

    def up(a):
        return _to_shape(zoom(a, k, order=1) if k > 1 else a, shape)

    tex = up(_fbm(small, rng))
    ridge = 1.0 - np.abs(_fbm(small, rng, ((48, 0.5), (24, 0.3), (12, 0.2))))
    cracks = up(np.clip((ridge - 0.86) / 0.14, 0, 1) ** 1.5)
    fine = gaussian_filter(rng.standard_normal(shape), 0.8)
    fine /= fine.std()

    # walls: bright at the surface, darker deeper in; texture, fractures and an edge glint
    wt = np.clip(-sd / 1.8, 0, 1)[..., None]
    wall = _lerp(t["wall_edge"], t["wall_deep"], wt ** 0.7) * (1 + 0.10 * tex + 0.04 * fine)[..., None]
    wall = _lerp(wall, t["crack"], (cracks * 0.75 * (1 - 0.5 * wt[..., 0]))[..., None])
    glint = np.clip(1 + sd / 0.22, 0, 1) * (sd < 0)
    wall = _lerp(wall, t["glint"], (glint * 0.55)[..., None])
    if t["sparkle"]:
        sp = (rng.random(shape) < 0.004) & (sd < 0) & (sd > -1.2)
        wall[sp] = t["glint"]

    # water: lighter near the walls, deep in the middle, faint texture
    wtr = np.clip(sd / 2.8, 0, 1)[..., None]
    water = _lerp(t["water_edge"], t["water_deep"], wtr ** 0.8) * (1 + 0.05 * tex)[..., None]

    a = np.clip(sd * scale + 0.5, 0, 1)[..., None]        # one-pixel anti-aliased shoreline
    img = np.clip(_lerp(wall, water, a), 0, 255).astype(np.uint8)
    return img, sd > 0, sd


def caustic_frames(water: np.ndarray, scale: float, theme: str, n_frames: int = 12, ds: int = 2):
    """Looping animated light caustics (half-resolution RGB surfaces, to be scaled up and added)."""
    color = np.array(THEMES[theme]["caustic"], float)
    mask = water[::ds, ::ds].astype(float)
    mask = gaussian_filter(mask, 2.0)
    hh, ww = mask.shape
    Y, X = np.mgrid[0:hh, 0:ww] * (ds / scale)            # metres
    frames = []
    for k in range(n_frames):
        ph = 2 * np.pi * k / n_frames
        a = (np.sin(X * 3.4 + 1.5 * np.sin(Y * 2.3 + ph)) + np.sin(Y * 3.7 + 1.4 * np.sin(X * 1.9 - ph))
             + 0.7 * np.sin((X - Y) * 2.5 + ph))
        c = np.exp(-(a ** 2) / 0.05) * mask * 0.16
        rgb = np.clip(c[..., None] * color, 0, 255).astype(np.uint8)
        frames.append(pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2))))
    return frames


def vignette(w: int, h: int, strength: int = 120) -> pygame.Surface:
    y, x = np.mgrid[0:h, 0:w]
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    a = (np.clip((d - 0.75) / 0.6, 0, 1) ** 1.5 * strength).astype(np.uint8)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels_alpha(s)[:] = a.T
    return s


def light_sprites(scale: float, cone_deg: float = 32.0, length: float = 6.5, glow_r: float = 1.8,
                  color=(95, 84, 52)):
    """Headlight (cone ahead + glow around the hull), centred on the vehicle, nose along +x.

    Returns (alpha_cut, glow): alpha_cut subtracts darkness, glow is added for a warm tint.
    A short, bright version (small length, light colour) draws the silt-lit veil in murky water.
    """
    R = length + 0.5
    n = int(2 * R * scale)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / scale - R
    y = -y
    d = np.hypot(x, y)
    ang = np.abs(np.arctan2(y, x))
    half = np.deg2rad(cone_deg)
    cone = np.clip(1 - d / length, 0, 1) ** 1.3 * np.clip((half - ang) / (0.35 * half), 0, 1) * (x > 0)
    glow = np.clip(1 - d / glow_r, 0, 1) ** 2
    light = np.clip(cone + glow, 0, 1)

    cut = pygame.Surface((n, n), pygame.SRCALPHA)
    cut.fill((0, 0, 0, 0))
    pygame.surfarray.pixels_alpha(cut)[:] = (light * 255).astype(np.uint8).T
    warm = np.clip(light[..., None] * np.array(color, float), 0, 255).astype(np.uint8)
    glow_s = pygame.surfarray.make_surface(np.ascontiguousarray(warm.transpose(1, 0, 2)))
    return cut, glow_s


SILT = (126, 110, 66)            # murky water: brownish-olive suspended silt


def silt_clouds(w: int, h: int, scale: float, seed: int = 0) -> pygame.Surface:
    """Billowy cloud texture (alpha = density) that the renderer scrolls slowly inside murky patches."""
    rng = np.random.default_rng(seed)
    k = max(1, int(round(scale / 10)))            # build at ~10 px/m, then upsample
    hs, ws = h // k + 2, w // k + 2
    n = _fbm((hs, ws), rng, ((40, 0.5), (20, 0.3), (10, 0.2)))
    n = np.clip((n + 0.3) / 2.2, 0, 1) ** 1.6
    a = _to_shape(zoom(n, k, order=1), (h, w)) if k > 1 else _to_shape(n, (h, w))
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((*SILT, 0))
    pygame.surfarray.pixels_alpha(s)[:] = (a * 150).astype(np.uint8).T
    return s


def radial_light(radius_px: int, color=(255, 255, 255)):
    """Round soft light: (alpha_cut, rgb glow) of the given radius."""
    n = 2 * radius_px
    y, x = np.mgrid[0:n, 0:n] + 0.5 - radius_px
    v = np.clip(1 - np.hypot(x, y) / radius_px, 0, 1) ** 1.6
    cut = pygame.Surface((n, n), pygame.SRCALPHA)
    cut.fill((0, 0, 0, 0))
    pygame.surfarray.pixels_alpha(cut)[:] = (v * 255).astype(np.uint8).T
    rgb = np.clip(v[..., None] * np.array(color, float), 0, 255).astype(np.uint8)
    return cut, pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))


# --------------------------------------------------------------------------- submarine


def submarine_frames(n_phases: int = 4, hull=(242, 188, 58)):
    """Top-down submarine sprite (nose to the right) drawn at SPRITE_PX_PER_M, one frame per propeller phase.

    About 1.15 m long and 0.45 m wide in the world; the vehicle's collision body is the 0.4 m disc.
    """
    P = SPRITE_PX_PER_M
    W, H = int(1.5 * P), int(0.8 * P)
    cx, cy = W // 2, H // 2

    def pt(x, y):           # body metres -> sprite pixels (y = left = up on the sprite)
        return int(cx + x * P), int(cy - y * P)

    hull = np.array(hull, float)
    dark = tuple(int(c) for c in hull * 0.35)
    frames = []
    for k in range(n_phases):
        s = pygame.Surface((W, H), pygame.SRCALPHA)
        # propeller (blades sweep: length varies with phase)
        blade = 0.05 + 0.13 * abs(np.cos(np.pi * k / n_phases))
        pygame.draw.ellipse(s, (70, 70, 78), (*pt(-0.66, blade), int(0.05 * P), int(2 * blade * P)))
        pygame.draw.line(s, (40, 40, 46), pt(-0.66, 0), pt(-0.56, 0), int(0.03 * P))
        # tail fins and bow planes
        for side in (1, -1):
            pygame.draw.polygon(s, dark, [pt(-0.38, 0.12 * side), pt(-0.58, 0.27 * side), pt(-0.60, 0.20 * side),
                                          pt(-0.52, 0.08 * side)])
            pygame.draw.polygon(s, dark, [pt(0.28, 0.16 * side), pt(0.18, 0.31 * side), pt(0.10, 0.30 * side),
                                          pt(0.14, 0.16 * side)])
        # hull with layered shading (lit from the upper side of the sprite)
        for i in range(14):
            f = i / 13
            col = tuple(int(c) for c in np.clip(_lerp(hull * 0.55, np.minimum(hull * 1.25, 255), f), 0, 255))
            hl, hw = 0.56 * (1 - 0.55 * f), 0.22 * (1 - 0.75 * f)
            ox, oy = -0.02 * f, 0.07 * f
            rect = pygame.Rect(0, 0, int(2 * hl * P), int(2 * hw * P))
            rect.center = pt(ox, oy)
            pygame.draw.ellipse(s, col, rect)
        rect = pygame.Rect(0, 0, int(1.12 * P), int(0.44 * P))
        rect.center = pt(0, 0)
        pygame.draw.ellipse(s, (60, 42, 16), rect, max(2, int(0.012 * P)))
        # glass nose dome
        dome = pygame.Rect(0, 0, int(0.2 * P), int(0.26 * P))
        dome.center = pt(0.45, 0)
        pygame.draw.ellipse(s, (120, 200, 235), dome)
        pygame.draw.ellipse(s, (220, 245, 255), (dome.x + dome.w // 3, dome.y + dome.h // 5, dome.w // 3, dome.h // 4))
        pygame.draw.ellipse(s, (40, 60, 80), dome, max(2, int(0.01 * P)))
        # conning tower with hatch and periscope
        tower = pygame.Rect(0, 0, int(0.34 * P), int(0.17 * P))
        tower.center = pt(-0.06, 0)
        pygame.draw.ellipse(s, tuple(int(c) for c in hull * 0.7), tower)
        pygame.draw.ellipse(s, (60, 42, 16), tower, max(2, int(0.01 * P)))
        pygame.draw.circle(s, (90, 90, 96), pt(-0.10, 0), int(0.045 * P))
        pygame.draw.circle(s, (40, 40, 46), pt(-0.10, 0), int(0.045 * P), max(1, int(0.008 * P)))
        pygame.draw.circle(s, (50, 50, 56), pt(0.03, 0), int(0.02 * P))
        # portholes along the deck
        for x in (0.22, -0.32):
            pygame.draw.circle(s, (40, 40, 46), pt(x, 0), int(0.04 * P))
            pygame.draw.circle(s, (150, 215, 240), pt(x, 0), int(0.028 * P))
        # headlights
        for y in (0.09, -0.09):
            pygame.draw.circle(s, (255, 250, 210), pt(0.52, y), int(0.022 * P))
        frames.append(s)
    return frames


# --------------------------------------------------------------------------- particles


class Particles:
    """Marine snow drifting with the current, and bubbles from the propeller."""

    def __init__(self, cave, rng: np.random.Generator, n_snow: int = 160):
        L, H = cave.world.length, cave.world.height
        self.rng = rng
        self.snow = np.column_stack([rng.uniform(0, L, n_snow), rng.uniform(0, H, n_snow)])
        self.snow_size = rng.choice([1, 1, 1, 2], n_snow)
        self.snow_alpha = rng.integers(50, 140, n_snow)
        self.size = (L, H)
        self.bubbles = np.zeros((0, 6))       # x, y, vx, vy, age, life

    def update(self, dt, current, stern, heading, thrust):
        L, H = self.size
        self.snow += dt * (np.asarray(current) + self.rng.normal(0, 0.05, self.snow.shape))
        self.snow %= (L, H)
        rate = max(thrust, 0.0) * 3.0
        n_new = self.rng.poisson(rate)
        if n_new:
            back = -np.array([np.cos(heading), np.sin(heading)])
            v = back * (0.25 + 0.3 * thrust) + self.rng.normal(0, 0.12, (n_new, 2))
            p = np.asarray(stern) + self.rng.normal(0, 0.04, (n_new, 2))
            life = self.rng.uniform(0.8, 1.6, n_new)
            self.bubbles = np.vstack([self.bubbles, np.column_stack([p, v, np.zeros(n_new), life])])
        if len(self.bubbles):
            b = self.bubbles
            b[:, 0:2] += dt * (b[:, 2:4] + np.asarray(current))
            b[:, 2:4] *= 0.9
            b[:, 4] += dt
            self.bubbles = b[b[:, 4] < b[:, 5]]

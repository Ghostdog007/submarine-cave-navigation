"""pygame renderer for the 2D cave, the submarine and its sonar.

Used by the keyboard driver (scripts/drive.py), env.render() and replay videos
(evaluation/replay.py).

style="game" (default): top-down 2D-game look, procedurally generated (sim/art.py)
  - textured cave walls (theme "ice": glacier ice with fractures and glints, or "rock")
  - animated light caustics, drifting marine snow, propeller bubbles
  - submarine sprite with a spinning propeller
  - darker water lit by the submarine's headlight cone, glowing EXIT beacon
  - sonar fan with periodic ping rings, vignette
style="plain": flat analysis colours; shows the margin where the vehicle centre can't go.

camera="full": the whole 50 m x 30 m map on screen.
camera="follow": zoomed in on the submarine, smoothly tracking it, with a minimap.

Rendering only: the simulation is identical in every style and camera.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pygame
from scipy.ndimage import distance_transform_edt

from submarine.world import Cave

from . import art
from submarine.sim.simulator import SimStep, Simulator

HUD_H = 64

WATER_EDGE = np.array([40, 105, 160])
WATER_DEEP = np.array([10, 38, 78])
WATER_RIM = np.array([78, 128, 170])
ROCK_EDGE = np.array([96, 86, 78])
ROCK_DEEP = np.array([26, 24, 28])
BAND_TINT = np.array([10, 12, 14])

SLOW = np.array([80, 150, 255])
FAST = np.array([255, 215, 70])

# plain-style hull outline in body coordinates (metres), nose along +x
_HULL = np.array([[0.55, 0.0], [0.42, 0.17], [0.15, 0.24], [-0.38, 0.22], [-0.52, 0.10], [-0.52, -0.10],
                  [-0.38, -0.22], [0.15, -0.24], [0.42, -0.17]])
_FIN = np.array([[-0.45, 0.0], [-0.70, 0.22], [-0.70, -0.22]])
_TOWER = np.array([[0.12, 0.0], [-0.05, 0.08], [-0.20, 0.0], [-0.05, -0.08]])

PING_PERIOD = 12          # frames between sonar ping rings (game style)
MINIMAP_W = 260

_BG_CACHE: OrderedDict = OrderedDict()   # (seed, difficulty, scale, theme, geodesic) -> (img, water)


def _lerp(a, b, t):
    return a + (b - a) * t


class Viewer:
    def __init__(self, cave: Cave, scale: float | None = None, show_geodesic: bool = False, headless: bool = False,
                 style: str = "game", theme: str = "ice", darkness: float = 0.5,
                 camera: str = "full", view_size: tuple[int, int] = (1280, 720)):
        """scale: pixels per metre (default 20 for camera="full", 48 for "follow").
        headless=True draws off-screen (rgb_array rendering / video).
        darkness (game style): 0 = fully lit, 1 = black outside the lights.
        view_size: on-screen map size in pixels for camera="follow".
        """
        if style not in ("game", "plain"):
            raise ValueError("style must be 'game' or 'plain'")
        if camera not in ("full", "follow"):
            raise ValueError("camera must be 'full' or 'follow'")
        if theme not in art.THEMES:
            raise ValueError(f"theme must be one of {list(art.THEMES)}")
        scale = scale if scale is not None else (48.0 if camera == "follow" else 20.0)
        self.headless, self.scale, self.style, self.theme = headless, scale, style, theme
        self.camera, self.darkness = camera, darkness
        self.w = int(round(cave.world.length * scale))          # world canvas size
        self.h = int(round(cave.world.height * scale))
        if camera == "follow":
            self.vw, self.vh = min(view_size[0], self.w), min(view_size[1], self.h)
        else:
            self.vw, self.vh = self.w, self.h
        size = (self.vw, self.vh + HUD_H)
        if headless:
            pygame.font.init()
            self.screen = pygame.Surface(size)
        else:
            pygame.init()
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Submarine cave navigation")
        self.canvas = pygame.Surface((self.w, self.h))
        self.overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.font = pygame.font.SysFont("consolas", 14)
        self.small = pygame.font.SysFont("consolas", 12)
        self.label_font = pygame.font.SysFont("consolas", 16, bold=True)
        self.big = pygame.font.SysFont("consolas", 32, bold=True)
        self.show_geodesic = show_geodesic
        self.frame = 0
        self.cam = None

        if style == "game":
            self.sub_frames = art.submarine_frames()
            self.light_cut, self.light_glow = art.light_sprites(scale)
            # in murk: the beam only reaches ~2.4 m, and lights up the silt in front of the hull
            self.murk_cut, _ = art.light_sprites(scale, cone_deg=40, length=2.4, glow_r=1.2)
            self.murk_glows = [art.light_sprites(scale, cone_deg=40, length=2.4, glow_r=1.3,
                                                 color=tuple(np.array([175, 160, 115]) * k))[1]
                               for k in np.linspace(0.3, 1.0, 5)]
            self.clouds = None
            r = int(2.6 * scale)
            self.goal_cut, _ = art.radial_light(r)
            self.goal_glows = [art.radial_light(r, tuple(np.array([60, 255, 150]) * k))[1]
                               for k in np.linspace(0.18, 0.45, 8)]
            self.dark = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            self.vignette = art.vignette(self.vw, self.vh)
            self.prop_phase = 0.0
        self.set_cave(cave)

    # ------------------------------------------------------------ background

    def set_cave(self, cave: Cave):
        self.cave = cave
        self.cam = None
        if self.style == "game":
            key = (cave.seed, cave.difficulty, self.scale, self.theme, self.show_geodesic)
            if key in _BG_CACHE:
                img, self.water = _BG_CACHE[key]
                _BG_CACHE.move_to_end(key)
            else:
                img, self.water, _ = art.build_background(cave, self.scale, self.theme)
                if self.show_geodesic:
                    band = (np.floor(self._geodesic_px(cave) / 2.5) % 2 == 1) & self.water
                    img[band] = np.clip(img[band].astype(int) + 12, 0, 255)
                _BG_CACHE[key] = (img, self.water)
                if len(_BG_CACHE) > 6:
                    _BG_CACHE.popitem(last=False)
            self.background = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
            self.caustics = art.caustic_frames(self.water, self.scale, self.theme)   # half resolution
            self.particles = art.Particles(cave, np.random.default_rng(cave.seed))
        else:
            res = cave.world.resolution
            water_t = np.clip(cave.clearance / 2.5, 0.0, 1.0)[..., None]
            rock_t = np.clip(distance_transform_edt(cave.rock) * res / 1.5, 0.0, 1.0)[..., None]
            grain = np.random.default_rng(cave.seed).normal(0.0, 6.0, cave.rock.shape + (1,))
            img = np.where(cave.rock[..., None], _lerp(ROCK_EDGE, ROCK_DEEP, rock_t) + grain,
                           _lerp(WATER_EDGE, WATER_DEEP, water_t))
            img[~cave.rock & ~cave.feasible] = WATER_RIM
            if self.show_geodesic:
                band = cave.feasible & (np.floor(np.nan_to_num(cave.geodesic, posinf=0) / 2.5) % 2 == 1)
                img[band] += BAND_TINT
            img = np.clip(img, 0, 255).astype(np.uint8)
            # arrays are [iy, ix] with y up; pygame surfaces are [x, y] with y down
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(img[::-1].transpose(1, 0, 2)))
            self.background = pygame.transform.smoothscale(surf, (self.w, self.h))
        mh = int(MINIMAP_W * self.h / self.w)
        self.minimap = pygame.transform.smoothscale(self.background, (MINIMAP_W, mh))

    def _fog(self, sim):
        """Murky-water layers, built once per turbidity field.

        Returns the static silt haze (None in clear caves) and prepares, for the game style:
        a mask that confines the drifting silt clouds to murky patches, silt particles
        concentrated where the water is murkiest, and a minimap that shows the murk.
        """
        turb = sim.turbidity
        if getattr(self, "_fog_src", None) is turb:
            return self._fog_surf
        from scipy.ndimage import gaussian_filter, zoom

        self._fog_src, self._fog_surf, self._fog_mask, self._silt = turb, None, None, None
        if turb is None or turb.max() <= 0.02:
            return None
        k = self.scale * self.cave.world.resolution
        t = art._to_shape(zoom(gaussian_filter(turb.astype(float), 1.5), k, order=1)[::-1], (self.h, self.w))
        if self.style == "game":
            water = self.water
        else:
            water = ~art._to_shape(zoom(self.cave.rock.astype(np.uint8), k, order=0)[::-1], (self.h, self.w)).astype(bool)
        # dense, brownish silt haze: thick murk (t = 1) leaves only ~15% of the scene visible
        alpha = np.clip(255 * (1 - np.exp(-2.0 * t)), 0, 220) * water
        surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        surf.fill((*art.SILT, 0))
        pygame.surfarray.pixels_alpha(surf)[:] = alpha.astype(np.uint8).T
        self._fog_surf = surf

        if self.style == "game":
            mask = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            mask.fill((255, 255, 255, 0))
            pygame.surfarray.pixels_alpha(mask)[:] = (np.clip(t * 1.6, 0, 1) * 255 * water).astype(np.uint8).T
            self._fog_mask = mask
            if getattr(self, "clouds", None) is None:
                self.clouds = art.silt_clouds(self.w, self.h, self.scale, seed=self.cave.seed)
                self._cloud_tmp = pygame.Surface((self.vw, self.vh), pygame.SRCALPHA)
            # silt particles: sampled where turbidity is high, so murky patches look full of suspended matter
            rng = np.random.default_rng(self.cave.seed + 99)
            res = self.cave.world.resolution
            w8 = (turb.astype(float) ** 2 * self.cave.feasible).ravel()    # water only, away from the walls
            if w8.sum() > 0:
                idx = rng.choice(w8.size, size=750, p=w8 / w8.sum())
                iy, ix = np.unravel_index(idx, turb.shape)
                pos = np.column_stack([(ix + rng.random(idx.size)) * res, (iy + rng.random(idx.size)) * res])
                self._silt = (pos, rng.uniform(0, 2 * np.pi, idx.size), turb[iy, ix], rng.choice([1, 1, 2], idx.size))
            mm = self.background.copy()
            mm.blit(surf, (0, 0))
            self.minimap = pygame.transform.smoothscale(mm, self.minimap.get_size())
        return surf

    def _draw_murk_game(self, sim, view, s, origin, deg):
        """Murk on top of the lit scene: haze, drifting silt clouds, suspended particles, and the headlight
        lighting up the silt right in front of the submarine (the backscatter that blinds the camera)."""
        scr, sc = self.canvas, self.scale
        fog = self._fog(sim)
        if fog is None:
            return
        scr.blit(fog, view.topleft, area=view)
        # drifting clouds, wrapped around the texture and masked to the murky patches
        tmp, cl = self._cloud_tmp, self.clouds
        cw, ch = cl.get_size()
        ox, oy = int(self.frame * 0.9 * sc / 24), int(self.frame * 0.35 * sc / 24)
        x0, y0 = -((view.x + ox) % cw), -((view.y + oy) % ch)
        tmp.fill((0, 0, 0, 0))
        for dx in (x0, x0 + cw):
            for dy in (y0, y0 + ch):
                tmp.blit(cl, (dx, dy))
        tmp.blit(self._fog_mask, (0, 0), area=view, special_flags=pygame.BLEND_RGBA_MULT)
        scr.blit(tmp, view.topleft)
        # suspended silt particles, gently swirling
        if self._silt is not None:
            pos, phase, tval, size = self._silt
            ph = self.frame * 0.04 + phase
            xs = pos[:, 0] + 0.25 * np.sin(ph) + 0.05 * self.frame * sim.current[0]
            ys = pos[:, 1] + 0.25 * np.cos(0.7 * ph) + 0.05 * self.frame * sim.current[1]
            ov = self.overlay
            ov.fill((0, 0, 0, 0))
            dot = max(1, int(sc / 32))
            for x, y, tv, sz in zip(xs, ys, tval, size):
                p = self.to_px(x, y)
                if view.collidepoint(p) and self.water[p[1], p[0]]:
                    pygame.draw.circle(ov, (196, 178, 126, int(55 + 120 * tv)), p, int(sz) * dot)
            scr.blit(ov, view.topleft, area=view)
        # headlight veil: in murk the beam lights the silt just ahead and dies within ~2 m
        t_here = sim.turbidity_at(s.x, s.y)
        if t_here > 0.12:
            k = min(1.0, (t_here - 0.12) / 0.6)
            glow = self.murk_glows[int(round(k * (len(self.murk_glows) - 1)))]
            self._blit_centered(pygame.transform.rotate(glow, deg), origin, pygame.BLEND_RGB_ADD)

    def _camera_wedge(self, surf, sim, s, length=3.0, alpha=110):
        if sim.camera is None:
            return
        half = np.deg2rad(sim.camera.cfg.fov_deg) / 2
        o = self.to_px(s.x, s.y)
        for a in (s.psi - half, s.psi + half):
            pygame.draw.line(surf, (255, 255, 255, alpha), o,
                             self.to_px(s.x + length * np.cos(a), s.y + length * np.sin(a)), 1)

    def _camera_strip(self, image):
        """The camera's 1D image, stretched into a strip at the top centre (left of image = left of the sub)."""
        scr = self.screen
        px = max(2, min(4, (self.vw - 560) // len(image)))
        w, h = px * len(image), 26
        x0, y0 = (self.vw - w) // 2, 10
        box = pygame.Surface((w + 12, h + 24), pygame.SRCALPHA)
        box.fill((0, 0, 0, 160))
        scr.blit(box, (x0 - 6, y0 - 4))
        scr.blit(self.small.render("camera", True, (200, 220, 230)), (x0, y0 - 2))
        vals = np.clip(image[::-1], 0, 1)
        strip = (np.repeat(vals[:, None], 3, 1) * np.array([200, 235, 255])).astype(np.uint8)
        surf = pygame.surfarray.make_surface(strip[:, None, :].repeat(h, axis=1))
        scr.blit(pygame.transform.scale(surf, (w, h)), (x0, y0 + 14))

    def _geodesic_px(self, cave):
        from scipy.ndimage import zoom

        g = np.nan_to_num(np.where(np.isfinite(cave.geodesic), cave.geodesic, -1.0), nan=-1.0)
        g = zoom(g, self.scale * cave.world.resolution, order=0)[::-1]
        return art._to_shape(g, (self.h, self.w))

    # ---------------------------------------------------------------- helpers

    def to_px(self, x: float, y: float) -> tuple[int, int]:
        """World metres -> canvas pixels."""
        return int(round(x * self.scale)), int(round(self.h - y * self.scale))

    def _body_poly(self, pts, x, y, psi):
        c, s = np.cos(psi), np.sin(psi)
        wx = x + pts[:, 0] * c - pts[:, 1] * s
        wy = y + pts[:, 0] * s + pts[:, 1] * c
        return [self.to_px(a, b) for a, b in zip(wx, wy)]

    def _blit_centered(self, surf, center, flags=0):
        self.canvas.blit(surf, surf.get_rect(center=center), special_flags=flags)

    def _update_camera(self, x, y):
        """Viewport rect on the canvas (smoothed follow, clamped to the map)."""
        if self.camera == "full":
            return pygame.Rect(0, 0, self.w, self.h)
        target = np.array(self.to_px(x, y), float)
        self.cam = target if self.cam is None else self.cam + 0.25 * (target - self.cam)
        cx = np.clip(self.cam[0], self.vw / 2, self.w - self.vw / 2)
        cy = np.clip(self.cam[1], self.vh / 2, self.h - self.vh / 2)
        return pygame.Rect(int(cx - self.vw / 2), int(cy - self.vh / 2), self.vw, self.vh)

    def _bar(self, x, y, w, h, value, signed, color, label):
        scr = self.screen
        scr.blit(self.small.render(label, True, (190, 190, 190)), (x, y - 1))
        bx = x + 64
        pygame.draw.rect(scr, (45, 45, 52), (bx, y, w, h), border_radius=3)
        v = float(np.clip(value, -1 if signed else 0, 1))
        if signed:
            mid = bx + w // 2
            fill = int(abs(v) * w / 2)
            rect = (mid, y, fill, h) if v >= 0 else (mid - fill, y, fill, h)
            pygame.draw.rect(scr, color, rect, border_radius=3)
            pygame.draw.line(scr, (140, 140, 150), (mid, y - 2), (mid, y + h + 1), 1)
        else:
            pygame.draw.rect(scr, color, (bx, y, int(v * w), h), border_radius=3)

    def _trail(self, surf, sim, trail, speeds, width, alpha=255):
        if len(trail) < 2:
            return
        pts = [self.to_px(*p) for p in trail]
        if speeds is not None and len(speeds) == len(trail):
            u_max = sim.vcfg.u_max
            for i in range(1, len(pts)):
                t = float(np.clip(speeds[i] / u_max, 0.0, 1.0))
                col = tuple(int(c) for c in _lerp(SLOW, FAST, t))
                pygame.draw.line(surf, (*col, alpha), pts[i - 1], pts[i], width)
        else:
            pygame.draw.lines(surf, (240, 200, 80, alpha), False, pts, width)

    def _reference(self, surf, reference, alpha):
        if reference is not None and len(reference) > 1:
            pts = [self.to_px(*p) for p in reference]
            for i in range(0, len(pts) - 1, 4):          # dashed
                pygame.draw.line(surf, (255, 255, 255, alpha), pts[i], pts[min(i + 2, len(pts) - 1)], 2)

    def _sonar(self, surf, sim, step, fan_alpha, ray_alpha):
        s = step.state
        origin = self.to_px(s.x, s.y)
        max_r = sim.sonar.cfg.max_range
        ends = [self.to_px(s.x + r * np.cos(s.psi + a), s.y + r * np.sin(s.psi + a))
                for a, r in zip(sim.sonar.angles, step.sonar)]
        poly = ends if sim.sonar.cfg.fov_deg >= 360 else [origin, *ends]
        # cool cyan-blue, so the sonar never blends with the brownish murk
        if len(poly) >= 3:
            pygame.draw.polygon(surf, (120, 200, 255, fan_alpha), poly)
        for end, r in zip(ends, step.sonar):
            hit = r < max_r - 1e-6
            pygame.draw.line(surf, (150, 215, 255, ray_alpha) if hit else (160, 180, 200, ray_alpha // 2),
                             origin, end, 1)
        return origin, ends

    # ------------------------------------------------------------------ draw

    def draw(self, sim: Simulator, step: SimStep, trail: list[tuple[float, float]], hud: list[str],
             banner: str | None = None, *, speeds=None, action=None, progress: float | None = None,
             reference=None, label: str | None = None, banner_color=(255, 255, 255), memory=None, recall=None):
        """memory / recall (junction-memory agents): stored memories [{xy, dead_end, idx}] and the attention
        weights on them this step [{xy, dead_end, idx, w}]."""
        self.frame += 1
        s = step.state
        view = self._update_camera(s.x, s.y)
        # only touch the visible part of the (possibly large) canvas
        clip = view.inflate(8, 8).clip(pygame.Rect(0, 0, self.w, self.h))
        for surf in (self.canvas, self.overlay) + ((self.dark,) if self.style == "game" else ()):
            surf.set_clip(clip)

        if self.style == "game":
            self._draw_world_game(sim, step, trail, speeds, action, reference, view)
        else:
            self._draw_world_plain(sim, step, trail, speeds, reference)
        if memory is not None:
            self._draw_memory(s, memory, recall or [])

        self.screen.fill((10, 12, 18))
        self.screen.blit(self.canvas, (0, 0), area=view)
        if self.style == "game":
            self.screen.blit(self.vignette, (0, 0))
        if self.camera == "follow":
            self._draw_minimap(sim, step, trail, view)
        self._draw_ui(hud, action, progress, label, banner, banner_color)
        if step.camera is not None:
            self._camera_strip(step.camera)
        if not self.headless:
            pygame.display.flip()

    def _draw_world_plain(self, sim, step, trail, speeds, reference):
        s, scr, sc, ov = step.state, self.canvas, self.scale, self.overlay
        scr.blit(self.background, (0, 0))
        fog = self._fog(sim)
        if fog is not None:
            scr.blit(fog, (0, 0))
        ov.fill((0, 0, 0, 0))
        self._reference(ov, reference, 90)
        self._camera_wedge(ov, sim, s)

        gp = self.to_px(*self.cave.goal)
        pulse = 0.5 + 0.5 * np.sin(self.frame * 0.25)
        pygame.draw.circle(ov, (255, 90, 90, int(40 + 50 * pulse)), gp, int(1.5 * sc))
        _, ends = self._sonar(ov, sim, step, 38, 70)
        scr.blit(ov, (0, 0))
        pygame.draw.circle(scr, (255, 110, 110), gp, int(1.5 * sc), 2)
        pygame.draw.circle(scr, (255, 255, 255), gp, 5)
        pygame.draw.circle(scr, (220, 50, 50), gp, 3)
        pygame.draw.circle(scr, (120, 230, 120), self.to_px(*self.cave.start), 5, 2)
        for end, r in zip(ends, step.sonar):
            if r < sim.sonar.cfg.max_range - 1e-6:
                pygame.draw.circle(scr, (190, 235, 255), end, 2)
        self._trail(scr, sim, trail, speeds, 3)

        origin = self.to_px(s.x, s.y)
        pygame.draw.circle(scr, (255, 80, 80) if step.collided else (200, 220, 235), origin,
                           max(int(sim.vcfg.radius * sc), 2), 1)
        hull_col = (255, 90, 90) if step.collided else (238, 196, 72)
        pygame.draw.polygon(scr, (120, 96, 40), self._body_poly(_FIN, s.x, s.y, s.psi))
        pygame.draw.polygon(scr, hull_col, self._body_poly(_HULL, s.x, s.y, s.psi))
        pygame.draw.polygon(scr, (60, 50, 30), self._body_poly(_HULL, s.x, s.y, s.psi), 1)
        pygame.draw.polygon(scr, (90, 74, 36), self._body_poly(_TOWER, s.x, s.y, s.psi))

    def _draw_world_game(self, sim, step, trail, speeds, action, reference, view):
        s, scr, sc, ov = step.state, self.canvas, self.scale, self.overlay
        dt = sim.vcfg.dt
        deg = np.rad2deg(s.psi)
        origin = self.to_px(s.x, s.y)

        # world: textured cave + moving caustics (half-res frames, scaled up for the visible part only)
        scr.blit(self.background, view.topleft, area=view)
        cf = self.caustics[(self.frame // 2) % len(self.caustics)]
        half = pygame.Rect(view.x // 2, view.y // 2, (view.w + 1) // 2, (view.h + 1) // 2).clip(cf.get_rect())
        scr.blit(pygame.transform.smoothscale(cf.subsurface(half), (half.w * 2, half.h * 2)),
                 (half.x * 2, half.y * 2), special_flags=pygame.BLEND_RGB_ADD)

        # particles, path, wake
        thrust = float(action[0]) if action is not None else s.u / sim.vcfg.u_max
        stern = (s.x - 0.62 * np.cos(s.psi), s.y - 0.62 * np.sin(s.psi))
        self.particles.update(dt, sim.current, stern, s.psi, thrust)
        ov.fill((0, 0, 0, 0))
        snow_col = art.THEMES[self.theme]["snow"]
        dot = max(1, int(sc / 24))
        for (x, y), size, a in zip(self.particles.snow, self.particles.snow_size, self.particles.snow_alpha):
            px, py = self.to_px(x, y)
            if view.collidepoint(px, py) and self.water[py, px]:
                pygame.draw.circle(ov, (*snow_col, int(a)), (px, py), int(size) * dot)
        self._reference(ov, reference, 55)
        self._trail(ov, sim, trail, speeds, max(2, int(sc / 16)), alpha=150)
        for x, y, _, _, age, life in self.particles.bubbles:
            f = age / life
            r = max(1, int((0.03 + 0.07 * f) * sc))
            pygame.draw.circle(ov, (225, 245, 255, int(200 * (1 - f))), self.to_px(x, y), r, 1)
        scr.blit(ov, view.topleft, area=view)

        # lighting: darker water, headlight cone and glow, goal beacon
        gp = self.to_px(*self.cave.goal)
        in_murk = sim.turbidity_at(s.x, s.y) > 0.35
        if self.darkness > 0:
            self.dark.fill((2, 8, 20, int(255 * self.darkness)))
            cut = pygame.transform.rotate(self.murk_cut if in_murk else self.light_cut, deg)
            self.dark.blit(cut, cut.get_rect(center=origin), special_flags=pygame.BLEND_RGBA_SUB)
            self.dark.blit(self.goal_cut, self.goal_cut.get_rect(center=gp), special_flags=pygame.BLEND_RGBA_SUB)
            scr.blit(self.dark, view.topleft, area=view)
        if not in_murk:
            self._blit_centered(pygame.transform.rotate(self.light_glow, deg), origin, pygame.BLEND_RGB_ADD)
        pulse = 0.5 + 0.5 * np.sin(self.frame * 0.22)
        self._blit_centered(self.goal_glows[int(pulse * (len(self.goal_glows) - 1))], gp, pygame.BLEND_RGB_ADD)
        self._draw_murk_game(sim, view, s, origin, deg)

        # submarine sprite on top of the lighting (propeller spins with speed); tinted red on collision
        self.prop_phase += 0.4 + 1.6 * abs(s.u) / sim.vcfg.u_max
        sprite = self.sub_frames[int(self.prop_phase) % len(self.sub_frames)]
        img = pygame.transform.rotozoom(sprite, deg, sc / art.SPRITE_PX_PER_M)
        if step.collided:
            img = img.copy()
            img.fill((255, 110, 110, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self._blit_centered(img, origin)

        # goal ring with rotating dashes, start marker
        ov.fill((0, 0, 0, 0))
        R = int(1.5 * sc)
        lw = max(3, int(sc / 8))
        for k in range(12):
            a0 = self.frame * 0.05 + k * np.pi / 6
            p0 = (gp[0] + R * np.cos(a0), gp[1] + R * np.sin(a0))
            p1 = (gp[0] + R * np.cos(a0 + 0.3), gp[1] + R * np.sin(a0 + 0.3))
            pygame.draw.line(ov, (140, 255, 180, 230), p0, p1, lw)
        pygame.draw.circle(ov, (200, 255, 220, 255), gp, max(3, int(0.15 * sc)))
        pygame.draw.circle(ov, (140, 255, 180, 200), self.to_px(*self.cave.start), max(3, int(0.15 * sc)), 2)

        # sonar: faint fan, ping ring, glowing hits; camera field of view
        _, ends = self._sonar(ov, sim, step, 26, 30)
        self._camera_wedge(ov, sim, s)
        f = (self.frame % PING_PERIOD) / PING_PERIOD
        ring = int(f * sim.sonar.cfg.max_range * 0.9 * sc)
        if ring > 2:
            pygame.draw.circle(ov, (150, 215, 255, int(130 * (1 - f))), origin, ring, 2)
        scr.blit(ov, view.topleft, area=view)
        hit_r = max(2, int(sc / 16))
        for end, r in zip(ends, step.sonar):
            if r < sim.sonar.cfg.max_range - 1e-6:
                pygame.draw.circle(scr, (190, 235, 255), end, hit_r)
        txt = (self.label_font if sc >= 32 else self.small).render("EXIT", True, (170, 255, 200))
        scr.blit(txt, txt.get_rect(midbottom=(gp[0], gp[1] - R - 4)))

    def _draw_memory(self, s, memory, recall):
        """Junction memories at their remembered (dead-reckoned) positions, and what is being recalled now:
        cyan ring = left a junction here, red cross = dead end; lines to the sub = attention this step,
        a pulsing ring = the memory recalled most strongly ("I have been here")."""
        scr, sc = self.canvas, self.scale
        r = max(6, int(0.45 * sc))
        font = self.label_font if sc >= 32 else self.small
        origin = self.to_px(s.x, s.y)
        ov = self.overlay
        ov.fill((0, 0, 0, 0))
        for m in recall:
            if m["w"] >= 0.05:                                       # attention lines, thickness ~ weight
                col = (255, 110, 110) if m["dead_end"] else (120, 230, 255)
                pygame.draw.line(ov, (*col, int(60 + 190 * m["w"])), origin, self.to_px(*m["xy"]),
                                 max(1, int(1 + 6 * m["w"])))
        scr.blit(ov, (0, 0))
        for m in memory:
            p = self.to_px(*m["xy"])
            if m["dead_end"]:
                pygame.draw.line(scr, (255, 80, 80), (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), 3)
                pygame.draw.line(scr, (255, 80, 80), (p[0] - r, p[1] + r), (p[0] + r, p[1] - r), 3)
            else:
                pygame.draw.circle(scr, (120, 230, 255), p, r, 3)
            scr.blit(font.render(f"#{m['idx']}", True, (255, 255, 255)), (p[0] + r + 2, p[1] - r - 2))
        top = max(recall, key=lambda m: m["w"], default=None)
        if top is not None and top["w"] >= 0.3:                       # strong recall: highlight it
            p = self.to_px(*top["xy"])
            pulse = int(r * (1.6 + 0.5 * np.sin(self.frame * 0.4)))
            pygame.draw.circle(scr, (255, 235, 120), p, pulse, 3)

    def _draw_minimap(self, sim, step, trail, view):
        scr = self.screen
        mm = self.minimap
        mw, mh = mm.get_size()
        x0, y0 = self.vw - mw - 12, self.vh - mh - 12
        frame = pygame.Surface((mw + 6, mh + 6), pygame.SRCALPHA)
        frame.fill((0, 0, 0, 170))
        scr.blit(frame, (x0 - 3, y0 - 3))
        scr.blit(mm, (x0, y0))
        k = mw / self.w

        def m(p):
            return int(x0 + p[0] * k), int(y0 + p[1] * k)

        if len(trail) > 1:
            pygame.draw.lines(scr, (255, 215, 70), False, [m(self.to_px(*p)) for p in trail[::3] + [trail[-1]]], 1)
        pygame.draw.rect(scr, (255, 255, 255), pygame.Rect(m(view.topleft), (int(view.w * k), int(view.h * k))), 1)
        pygame.draw.circle(scr, (120, 255, 170), m(self.to_px(*self.cave.goal)), 4)
        s = step.state
        pygame.draw.circle(scr, (255, 90, 90) if step.collided else (255, 225, 90), m(self.to_px(s.x, s.y)), 3)

    def _draw_ui(self, hud, action, progress, label, banner, banner_color):
        scr = self.screen
        # scale bar (screen space, so it stays put while the camera moves)
        x0, y0, L = 14, self.vh - 16, int(5 * self.scale)
        pygame.draw.line(scr, (235, 235, 235), (x0, y0), (x0 + L, y0), 2)
        for x in (x0, x0 + L):
            pygame.draw.line(scr, (235, 235, 235), (x, y0 - 4), (x, y0 + 4), 2)
        scr.blit(self.small.render("5 m", True, (235, 235, 235)), (x0 + L + 6, y0 - 7))

        if label:
            txt = self.label_font.render(label, True, (255, 255, 255))
            box = pygame.Surface((txt.get_width() + 14, txt.get_height() + 8), pygame.SRCALPHA)
            box.fill((0, 0, 0, 150))
            scr.blit(box, (8, 8))
            scr.blit(txt, (15, 12))

        pygame.draw.rect(scr, (20, 22, 30), (0, self.vh, self.vw, HUD_H))
        pygame.draw.line(scr, (70, 80, 100), (0, self.vh), (self.vw, self.vh), 1)
        for i, line in enumerate(hud):
            scr.blit(self.font.render(line, True, (225, 225, 225)), (10, self.vh + 8 + 18 * i))
        bx, bw = self.vw - 64 - 170 - 12, 170
        if action is not None:
            self._bar(bx, self.vh + 8, bw, 10, action[0], True, (238, 196, 72), "thrust")
            self._bar(bx, self.vh + 26, bw, 10, action[1], True, (120, 200, 255), "yaw")
        if progress is not None:
            self._bar(bx, self.vh + 44, bw, 10, progress, False, (120, 230, 140), "progress")

        if banner:
            text = self.big.render(banner, True, banner_color)
            box = pygame.Surface((text.get_width() + 30, text.get_height() + 14), pygame.SRCALPHA)
            box.fill((0, 0, 0, 170))
            rect = box.get_rect(topright=(self.vw - 10, 8))   # top-right, clear of the label
            scr.blit(box, rect)
            scr.blit(text, text.get_rect(center=rect.center))

    def rgb_array(self) -> np.ndarray:
        """Current frame as (height, width, 3) uint8."""
        return np.ascontiguousarray(pygame.surfarray.array3d(self.screen).transpose(1, 0, 2))

    def save(self, path: str):
        pygame.image.save(self.screen, path)

    def close(self):
        if not self.headless:
            pygame.quit()

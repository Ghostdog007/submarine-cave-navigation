"""Drive the submarine with the keyboard.

    python -m scripts.drive --seed 0 --difficulty 0.5
    python -m scripts.drive --difficulty 1 --current 0.3 --fov 120 --beams 24

Default look: game style (ice theme), camera following the submarine with a minimap;
use --style plain / --theme rock / --camera full to change it.

Keys: Up/W thrust, Down/S reverse, Left/A turn left, Right/D turn right,
      R restart, N / P next / previous seed, G toggle geodesic bands, Esc quit.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pygame

from submarine.world import generate_cave
from submarine.sim import SonarConfig, Simulator, VehicleConfig
from submarine.render.viewer import Viewer

GOAL_RADIUS = 1.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difficulty", type=float, default=0.3)
    p.add_argument("--current", type=float, default=0.0, help="max current magnitude, m/s")
    p.add_argument("--fov", type=float, default=360.0)
    p.add_argument("--beams", type=int, default=32)
    p.add_argument("--ideal", action="store_true", help="noise-free sonar")
    p.add_argument("--scale", type=float, default=None, help="pixels per metre (default 48 follow, 20 full)")
    p.add_argument("--style", choices=["game", "plain"], default="game")
    p.add_argument("--theme", choices=["ice", "rock"], default="ice")
    p.add_argument("--camera", choices=["follow", "full"], default="follow")
    p.add_argument("--geodesic", action="store_true")
    p.add_argument("--frames", type=int, default=0, help="headless check: run N frames at full thrust, then exit")
    p.add_argument("--screenshot", type=str, default=None, help="save the last frame here")
    args = p.parse_args()

    vcfg = VehicleConfig(max_current=args.current)
    scfg = SonarConfig(n_beams=args.beams, fov_deg=args.fov)
    if args.ideal:
        scfg = scfg.ideal()

    seed = args.seed
    cave = generate_cave(seed, args.difficulty)
    sim = Simulator(cave, vcfg, scfg, seed=seed)
    viewer = Viewer(cave, args.scale, args.geodesic, headless=False, style=args.style, theme=args.theme,
                    camera=args.camera)
    clock = pygame.time.Clock()

    def restart(new_seed=None):
        nonlocal seed, cave, sim
        if new_seed is not None:
            seed = new_seed
            cave = generate_cave(seed, args.difficulty)
            sim = Simulator(cave, vcfg, scfg, seed=seed)
            viewer.set_cave(cave)
        st = sim.reset()
        speeds[:] = [0.0]
        return st, [(st.state.x, st.state.y)], 0, None

    speeds = [0.0]
    step, trail, t, banner = restart()
    frame = 0
    colour = (255, 255, 255)
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r:
                    step, trail, t, banner = restart()
                elif ev.key == pygame.K_n:
                    step, trail, t, banner = restart(seed + 1)
                elif ev.key == pygame.K_p:
                    step, trail, t, banner = restart(max(seed - 1, 0))
                elif ev.key == pygame.K_g:
                    viewer.show_geodesic = not viewer.show_geodesic
                    viewer.set_cave(cave)

        if args.frames:
            action = (1.0, 0.0)
        else:
            k = pygame.key.get_pressed()
            thrust = (k[pygame.K_UP] or k[pygame.K_w]) - (k[pygame.K_DOWN] or k[pygame.K_s])
            yaw = (k[pygame.K_LEFT] or k[pygame.K_a]) - (k[pygame.K_RIGHT] or k[pygame.K_d])
            action = (float(thrust), float(yaw))

        if banner is None:
            step = sim.step(action)
            t += 1
            trail.append((step.state.x, step.state.y))
            speeds.append(step.state.u)
            if step.collided:
                banner, colour = "COLLISION - press R", (255, 100, 100)
            elif sim.goal_distance() < GOAL_RADIUS:
                banner, colour = f"GOAL in {t * vcfg.dt:.1f} s - press N", (120, 240, 140)

        s = step.state
        geo = sim.geodesic_at(s.x, s.y)
        cx, cy = sim.current
        viewer.draw(sim, step, trail, [
            f"seed {seed}  d={args.difficulty:.2f}  t={t * vcfg.dt:5.1f}s   speed {s.u:+.2f} m/s   "
            f"yaw rate {s.r:+.2f} rad/s   current ({cx:+.2f}, {cy:+.2f}) m/s",
            f"clearance {step.clearance:4.2f} m   to goal: geodesic {geo:5.1f} m, straight {sim.goal_distance():5.1f} m   "
            f"min sonar {step.sonar.min():4.2f} m",
        ], banner, speeds=speeds, action=action, label=f"cave {seed}   d={args.difficulty:.2f}",
            progress=1 - geo / cave.shortest_path_length if np.isfinite(geo) else None, banner_color=colour)

        frame += 1
        if args.frames and frame >= args.frames:
            running = False
        else:
            clock.tick(round(1 / vcfg.dt))

    if args.screenshot:
        viewer.save(args.screenshot)
        print(f"saved {args.screenshot}")
    viewer.close()


if __name__ == "__main__":
    main()

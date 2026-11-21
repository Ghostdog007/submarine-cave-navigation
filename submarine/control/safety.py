"""Collision-avoidance safety filter, built into the driver (predict-then-check, a simple shield).

Every step the driver proposes [thrust, yaw]. The filter predicts the path the vehicle would take if it applied
that command for one step and then braked to a standstill (same turn), and checks that path against the walls
the sonar has seen. If the path comes too close, it substitutes the nearest thrust level whose stopping path is
safe, then gentler turns (or, if nothing is safe, the command that keeps the most clearance).

Only on-board information is used:
  * wall points come from the observed sonar scan (after sensor failures), placed in a local odometry frame
    integrated from the measured surge speed and yaw rate (no true position, no map);
  * the water current is not measured, so it is not in the prediction (covered by the margins).

Fixes over the first (post-hoc) version, found from 0.4 s-sonar crashes:
  * the whole stopping path is predicted (stopping from 1.5 m/s takes ~2.1 s), not a fixed 1.5 s;
  * a hard floor: never plan within radius + floor of a seen wall (sonar noise + range quantisation);
  * single-beam dropouts between two similar returns are bridged, and the previous scan is kept, so a
    dropout can't open a fake hole in the wall that the vehicle "fits" through;
  * the margin grows with speed x scan age (walls seen 0.4 s ago, vehicle has moved since);
  * fallback picks the thrust closest to the request among safe ones; if none is safe with the requested turn,
    gentler turns are tried (turning while moving sweeps the hull sideways); at standstill turning is free;
  * inside the floor (already closer than radius + floor to a seen wall) the vehicle may only creep: forward speed
    is capped at `creep_speed`. There the check is only "no closer than now, as seen by sonar", and a wall corner
    between two beams is invisible: phase B had one hull touch at 0.33 m/s at a crevasse bend this way.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from submarine.sim import VehicleConfig


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool = True
    horizon_s: float = 4.0            # s, max look-ahead (stopping from full speed takes ~2.1 s)
    margin: float = 0.08              # m added to the vehicle radius
    floor: float = 0.12               # m; never plan closer than radius + floor to a seen wall
    age_margin: float = 1.0           # extra margin = age_margin * |speed| * scan age
    keep_scans: int = 2               # wall points from the last N pings (fills dropout holes)
    join_frac: float = 0.35           # neighbouring returns are one wall if |dr| < join_frac * r + join_abs
    join_abs: float = 0.3             # m
    spacing: float = 0.1              # m between interpolated wall points
    thrusts: tuple[float, ...] = (1.0, 0.75, 0.5, 0.3, 0.15, 0.0, -0.25, -0.5, -1.0)   # candidate levels
    yaw_scales: tuple[float, ...] = (0.5, 0.0)    # gentler turns tried when no thrust level is safe
    creep_speed: float | None = 0.2   # m/s, max forward speed inside the floor (None: no cap)


class Odometry:
    """Local pose from measured surge speed and yaw rate, integrated like the vehicle model (no current)."""

    def __init__(self):
        self.x = self.y = self.psi = 0.0

    def step(self, u: float, r: float, dt: float):
        self.psi += r * dt
        self.x += u * np.cos(self.psi) * dt
        self.y += u * np.sin(self.psi) * dt


def scan_points(ranges: np.ndarray, angles: np.ndarray, max_range: float, x: float, y: float, psi: float,
                cfg: SafetyConfig, full_circle: bool = True) -> np.ndarray:
    """Wall points (P, 2) of one scan: the returns plus points along wall segments between neighbours."""
    r = np.asarray(ranges, dtype=float).copy()
    hit = r < max_range - 1e-6
    n = len(r)
    if n >= 3:                                            # bridge single-beam dropouts
        left, right = np.roll(r, 1), np.roll(r, -1)
        hl, hr = np.roll(hit, 1), np.roll(hit, -1)
        bridge = ~hit & hl & hr & (np.abs(left - right) < cfg.join_frac * np.maximum(left, right) + cfg.join_abs)
        if not full_circle:
            bridge[0] = bridge[-1] = False
        r[bridge] = 0.5 * (left[bridge] + right[bridge])
        hit |= bridge
    a = psi + np.asarray(angles, dtype=float)
    px, py = x + r * np.cos(a), y + r * np.sin(a)
    pts = [np.stack([px[hit], py[hit]], axis=1)]
    i = np.arange(n) if full_circle else np.arange(n - 1)
    j = (i + 1) % n
    join = hit[i] & hit[j] & (np.abs(r[i] - r[j]) < cfg.join_frac * np.maximum(r[i], r[j]) + cfg.join_abs)
    i, j = i[join], j[join]
    if i.size:
        k = np.ceil(np.hypot(px[j] - px[i], py[j] - py[i]) / cfg.spacing).astype(int) - 1   # interior points
        keep = k > 0
        i, j, k = i[keep], j[keep], k[keep]
        if i.size:
            seg = np.repeat(np.arange(i.size), k)                      # which segment each point belongs to
            m = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k) + 1  # 1..k within its segment
            t = m / (k[seg] + 1)
            pts.append(np.stack([px[i[seg]] + t * (px[j[seg]] - px[i[seg]]),
                                 py[i[seg]] + t * (py[j[seg]] - py[i[seg]])], axis=1))
    return np.concatenate(pts)


class SafetyFilter:
    def __init__(self, cfg: SafetyConfig, vehicle: VehicleConfig, angles: np.ndarray, max_range: float,
                 full_circle: bool = True):
        self.cfg, self.v = cfg, vehicle
        self.angles, self.max_range, self.full_circle = np.asarray(angles, float), max_range, full_circle
        self.n = max(2, int(round(cfg.horizon_s / vehicle.dt)))
        self.reset()

    def reset(self):
        self.odom = Odometry()
        self._scans: deque = deque(maxlen=max(1, self.cfg.keep_scans))
        self._tree = None
        self.points = np.zeros((0, 2))

    # ------------------------------------------------------------------ sensing
    def update_motion(self, u: float, r: float):
        self.odom.step(u, r, self.v.dt)

    def update_scan(self, ranges: np.ndarray):
        o = self.odom
        self._scans.append(scan_points(ranges, self.angles, self.max_range, o.x, o.y, o.psi, self.cfg,
                                       self.full_circle))
        self.points = np.concatenate(list(self._scans))
        self._tree = cKDTree(self.points) if len(self.points) else None

    # ------------------------------------------------------------------ prediction
    def clearances(self, u: float, r: float, actions: np.ndarray) -> np.ndarray:
        """Min distance to seen walls along each action's stopping path (one step of the action, then brake).
        Batched reference implementation (tests check the fast scalar path in `filter` against it)."""
        if self._tree is None:
            return np.full(len(actions), np.inf)
        v = self.v
        actions = np.asarray(actions, dtype=float)
        K = len(actions)
        x = np.full(K, self.odom.x)
        y = np.full(K, self.odom.y)
        psi = np.full(K, self.odom.psi)
        uu = np.full(K, float(u))
        rr = np.full(K, float(r))
        yaw = np.clip(actions[:, 1], -1.0, 1.0)
        thrust = actions[:, 0]
        path = np.empty((K, self.n, 2))
        done = np.zeros(K, dtype=bool)
        for k in range(self.n):
            th = np.clip(thrust, -1.0, 1.0)
            th = np.where(th < 0, th * v.reverse_gain, th)
            un = np.clip(uu + v.dt * (th * v.a_max - v.k_u * uu), -v.u_max * v.reverse_gain, v.u_max)
            rn = rr + v.dt * (yaw * v.r_max - rr) / v.tau_r
            pn = psi + rn * v.dt
            xn, yn = x + v.dt * un * np.cos(pn), y + v.dt * un * np.sin(pn)
            live = ~done                                     # stopped vehicles stay where they are
            uu, rr, psi = np.where(live, un, uu), np.where(live, rn, rr), np.where(live, pn, psi)
            x, y = np.where(live, xn, x), np.where(live, yn, y)
            path[:, k, 0], path[:, k, 1] = x, y
            done |= np.abs(uu) <= 0.02
            if done.all():
                path[:, k + 1:, 0], path[:, k + 1:, 1] = x[:, None], y[:, None]
                break
            thrust = np.where(uu > 0, -1.0, 1.0)              # brake against the direction of motion
        d, _ = self._tree.query(path.reshape(-1, 2))
        return d.reshape(K, self.n).min(axis=1)

    def _path(self, u: float, r: float, thrust: float, yaw: float):
        """Stopping path of one command (scalar maths: ~20x cheaper than the batched version for one path).
        Returns (points (k, 2), path length)."""
        v = self.v
        x, y, psi = self.odom.x, self.odom.y, self.odom.psi
        yaw = min(1.0, max(-1.0, yaw))
        th = min(1.0, max(-1.0, thrust))
        umin, umax = -v.u_max * v.reverse_gain, v.u_max
        xs, ys, length = [], [], 0.0
        for _ in range(self.n):
            t = th * v.reverse_gain if th < 0 else th
            u = min(umax, max(umin, u + v.dt * (t * v.a_max - v.k_u * u)))
            r = r + v.dt * (yaw * v.r_max - r) / v.tau_r
            psi += r * v.dt
            dx, dy = v.dt * u * math.cos(psi), v.dt * u * math.sin(psi)
            x += dx
            y += dy
            length += math.hypot(dx, dy)
            xs.append(x)
            ys.append(y)
            if abs(u) <= 0.02:
                break
            th = -1.0 if u > 0 else 1.0                          # brake against the direction of motion
        return np.column_stack([xs, ys]), length

    def clearance(self, u: float, r: float, thrust: float, yaw: float, now: float, need: float) -> float:
        """Min distance to seen walls along one command's stopping path. If even the whole path length can't
        bring it within `need` of the nearest wall (now - length >= need), returns that lower bound unqueried."""
        pts, length = self._path(u, r, thrust, yaw)
        if now - length >= need:
            return now - length
        return float(self._tree.query(pts)[0].min())

    def required(self, u: float, scan_age: float) -> float:
        """Clearance a command's stopping path must keep (see the module docstring)."""
        return self._need(u, scan_age, float(self._tree.query([self.odom.x, self.odom.y])[0]))

    def _need(self, u: float, scan_age: float, now: float) -> float:
        c, rad = self.cfg, self.v.radius
        floor = rad + c.floor
        want = max(floor, rad + c.margin + c.age_margin * abs(u) * scan_age)
        # keep the wanted clearance; already closer (a narrow passage): don't get any closer than now
        return max(floor, min(want, now)) if now >= floor else now

    def filter(self, action, u: float, r: float, scan_age: float) -> tuple[np.ndarray, bool]:
        """(safe action, overridden?)"""
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        if not self.cfg.enabled or self._tree is None:
            return action, False
        now = float(self._tree.query([self.odom.x, self.odom.y])[0])
        need = self._need(u, scan_age, now)
        thrust, yaw = float(action[0]), float(action[1])
        capped, t_max = False, 1.0
        if self.cfg.creep_speed is not None and now < self.v.radius + self.cfg.floor:
            t_max = self.creep_thrust(u)
            if thrust > t_max:
                thrust, capped = t_max, True
                action = np.array([thrust, yaw], dtype=np.float32)
        if self.clearance(u, r, thrust, yaw, now, need) >= need:        # the common case: one cheap check
            return action, capped
        best, best_clr = None, -np.inf
        for cand in self.candidates(thrust, yaw):                         # nearest safe command
            if cand[0] > t_max + 1e-9:
                continue
            c = self.clearance(u, r, cand[0], cand[1], now, need)
            if c >= need:
                return np.array(cand, dtype=np.float32), True
            if c > best_clr:
                best, best_clr = cand, c
        return np.array(best, dtype=np.float32), True                    # none safe: most clearance

    def creep_thrust(self, u: float) -> float:
        """Largest thrust whose next-step forward speed stays within creep_speed (brakes harder if faster)."""
        v = self.v
        t = ((self.cfg.creep_speed - u) / v.dt + v.k_u * u) / v.a_max
        if t < 0:
            t = t / v.reverse_gain if v.reverse_gain > 0 else -1.0    # reverse thrust is weaker
        return float(min(1.0, max(-1.0, t)))

    def candidates(self, thrust: float, yaw: float) -> list[tuple[float, float]]:
        """Fallback commands in order of preference: the requested turn with every thrust level (nearest first),
        then a gentler turn (half, none). A hard turn while still moving swings the hull through an arc into a
        side wall, which no braking alone prevents (found in phase B: a turn-back at 0.8 m/s)."""
        ts = sorted(self.cfg.thrusts, key=lambda t: abs(t - thrust))
        out = [(t, yaw) for t in ts]
        for y in self.cfg.yaw_scales:
            if abs(yaw * y - yaw) > 1e-9:
                out += [(t, yaw * y) for t in ts]
        return out

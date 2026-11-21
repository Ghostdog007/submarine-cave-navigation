"""Hand-written tracking controller: follow a commanded direction at a commanded speed through the passage.

The command is a bearing (body frame) and a cruise speed. The controller steers toward the most open
vehicle-width corridor near the commanded bearing (so it follows the passage around bends instead of driving
into the wall the bearing points at), keeps off the side walls, and sets the speed from the distance to the
first obstacle ahead, with drag feed-forward so it actually holds the commanded speed.

It is the teacher for the learned driver's imitation warm start (phase A1), the reference it must beat, and
the fallback. Its output still goes through the safety filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from submarine.baselines.reactive import corridor_front, corridor_ranges
from submarine.sim import VehicleConfig


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class ControllerConfig:
    half_width: float = 0.55          # m, corridor the vehicle must fit through
    search_deg: float = 60.0          # only directions within this of the commanded bearing are considered
    deviation_weight: float = 2.0     # m of reach traded per rad of deviation from the commanded bearing
    range_cap: float = 6.0            # m; reach beyond this doesn't make a direction better
    k_yaw: float = 1.5                # yaw command per rad of heading error
    k_damp: float = 0.3               # yaw-rate damping (per rad/s)
    k_side: float = 0.4               # side-wall repulsion
    k_speed: float = 1.5              # thrust per m/s of speed error
    stop: float = 0.6                 # m ahead: back off
    slow: float = 3.0                 # m ahead: full speed allowed beyond this
    behind_deg: float = 100.0         # commanded bearing further round than this: turn in place first


class HandController:
    def __init__(self, cfg: ControllerConfig, vehicle: VehicleConfig, angles: np.ndarray, max_range: float):
        self.cfg, self.v = cfg, vehicle
        self.angles, self.max_range = np.asarray(angles, float), max_range

    def act(self, ranges: np.ndarray, u: float, r: float, bearing: float, speed: float) -> tuple[np.ndarray, float]:
        """([thrust, yaw], chosen direction in the body frame)."""
        c, v, th = self.cfg, self.v, self.angles
        ranges = np.asarray(ranges, float)
        reach = corridor_ranges(ranges, th, c.half_width, self.max_range)
        dev = np.abs(wrap(th - bearing))
        cand = dev <= np.deg2rad(c.search_deg)
        if not cand.any():
            cand = np.ones_like(cand)
        score = np.where(cand, np.minimum(reach, c.range_cap) - c.deviation_weight * dev, -np.inf)
        target = float(th[int(np.argmax(score))])

        yaw = c.k_yaw * target - c.k_damp * r
        left = ranges[(th > np.deg2rad(30)) & (th < np.deg2rad(120))]
        right = ranges[(th < -np.deg2rad(30)) & (th > -np.deg2rad(120))]
        if left.size and right.size:
            yaw += c.k_side * (1 / max(right.min(), 0.3) - 1 / max(left.min(), 0.3))

        front = corridor_front(ranges, th, c.half_width, self.max_range)
        if abs(target) > np.deg2rad(c.behind_deg):
            v_des = 0.0                                         # turn in place toward the way back
        else:
            room = np.clip((front - c.stop) / (c.slow - c.stop), 0.0, 1.0)
            v_des = min(speed, v.u_max * room * max(0.0, np.cos(target)))
        thrust = v.k_u * v_des / v.a_max + c.k_speed * (v_des - u)
        if front < c.stop:
            thrust = min(thrust, -0.5)
        return np.array([np.clip(thrust, -1, 1), np.clip(yaw, -1, 1)], dtype=np.float32), target

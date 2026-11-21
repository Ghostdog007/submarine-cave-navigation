"""Simplified submarine dynamics (context.md, section 3).

Surge with linear drag, first-order yaw-rate lag, constant water current.
Angles are in radians, counter-clockwise from +x (the world y axis points up).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import VehicleConfig


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    psi: float          # heading, rad
    u: float = 0.0      # surge speed, m/s
    r: float = 0.0      # yaw rate, rad/s


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_vehicle(s: VehicleState, action, cfg: VehicleConfig, current=(0.0, 0.0)) -> VehicleState:
    """Advance one time step. action = [thrust, yaw_cmd], each clipped to [-1, 1]."""
    thrust = min(1.0, max(-1.0, float(action[0])))
    yaw_cmd = min(1.0, max(-1.0, float(action[1])))
    if thrust < 0:
        thrust *= cfg.reverse_gain
    dt = cfg.dt

    u = s.u + dt * (thrust * cfg.a_max - cfg.k_u * s.u)
    u = min(cfg.u_max, max(-cfg.u_max * cfg.reverse_gain, float(u)))
    r = s.r + dt * (yaw_cmd * cfg.r_max - s.r) / cfg.tau_r
    psi = wrap_angle(s.psi + r * dt)

    x = s.x + dt * (u * np.cos(psi) + current[0])
    y = s.y + dt * (u * np.sin(psi) + current[1])
    return VehicleState(float(x), float(y), float(psi), u, float(r))

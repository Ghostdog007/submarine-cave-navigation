"""Tiered safety supervisor: "Stop, think, act", between the learned policy and the thrusters.

Modelled on cave-diver lost-line / silt-out practice (stop at once, think, then move slowly along what you
know), on real AUV fault handling (escalate on persistent faults, abort as a last resort) and on Simplex /
shielding in safe RL (a learned controller plus a simple trusted safety controller, switching back when safe).

Blind = every fitted range sensor is reported failed by its health flag (sonar; and camera if fitted).

  tier 0  normal        all flagged sensors healthy               learned policy drives
  tier 1  degraded      some, not all, sensors failed             learned policy drives (trained for this)
  tier 2  STOP          blind for less than `think_s`             override: brake to a halt, hold heading
  tier 3  ACT slowly    blind for longer                          learned policy, speed capped at `blind_speed`
  tier 4  ABORT         blind for longer than `abort_s`           hold, end the episode as a safe abort
  recovery              any sensor healthy again                  straight back to tier 0 / 1

Penalty: every step the supervisor changes the policy's action costs override_penalty * |change|, so the
agent learns to slow down by itself before the supervisor has to step in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TIER_NAMES = ("normal", "degraded", "stop", "act slowly", "abort")


@dataclass(frozen=True)
class SupervisorConfig:
    enabled: bool = False
    think_s: float = 3.0             # tier 2 duration: brake and hold
    blind_speed: float = 0.2         # m/s cap in tier 3
    abort_s: float = 30.0            # blind longer than this -> safe abort
    abort_penalty: float = 5.0       # reward penalty for a safe abort (a collision costs 10)
    override_penalty: float = 0.05   # per unit of action change the supervisor makes


class SafetySupervisor:
    def __init__(self, cfg: SupervisorConfig, vehicle):
        self.cfg, self.v = cfg, vehicle
        self.blind_time = 0.0
        self.tier = 0
        # thrust command that settles at blind_speed: a_max * thrust = k_u * u
        self.cap_thrust = min(1.0, vehicle.k_u * cfg.blind_speed / vehicle.a_max)

    def update(self, healthy: list[bool], dt: float) -> int:
        """Update the tier from the current health flags; returns the tier."""
        blind = len(healthy) > 0 and not any(healthy)
        self.blind_time = self.blind_time + dt if blind else 0.0
        if not blind:
            self.tier = 1 if not all(healthy) else 0
        elif self.blind_time > self.cfg.abort_s:
            self.tier = 4
        elif self.blind_time <= self.cfg.think_s:
            self.tier = 2
        else:
            self.tier = 3
        return self.tier

    def filter(self, action: np.ndarray, u: float) -> np.ndarray:
        """The action actually sent to the thrusters, given the policy's action and current surge speed u."""
        thrust, yaw = float(action[0]), float(action[1])
        if self.tier in (2, 4):
            # brake to a halt (reverse thrust while still moving forward), hold heading
            thrust = -1.0 if u > 0.05 else (0.5 if u < -0.05 else 0.0)
            yaw = 0.0
        elif self.tier == 3:
            cap = self.cfg.blind_speed
            if u > cap:
                thrust = min(thrust, -0.5)                     # too fast: brake
            elif u < -cap:
                thrust = max(thrust, 0.5)
            else:
                thrust = float(np.clip(thrust, -self.cap_thrust, self.cap_thrust))
        return np.array([thrust, yaw], dtype=np.float32)

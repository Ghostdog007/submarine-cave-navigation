"""The driver: executes the navigator's "follow exit k at speed v" commands at 10 Hz, safely.

    sense(ranges, u, r, sonar_age)        every step, with the observed sensors
    exits                                 openings in the latest scan, as headings in the odometry frame
    follow(exit or heading, speed)        set the command (the navigator's decision)
    act() -> action, info                 controller -> built-in safety filter -> [thrust, yaw]

The followed heading lives in the driver's odometry frame and slowly adapts to the direction the controller
actually picks, so "follow exit k" keeps following that passage around bends instead of holding a fixed compass
heading into a wall. The hand controller here is phase A0; the learned driver (phases A1-A3) replaces the
controller and keeps everything else, including the filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from submarine.sim import Sonar, SonarConfig, VehicleConfig

from .controller import ControllerConfig, HandController, wrap
from .openings import Opening, OpeningsConfig, detect_openings
from .safety import SafetyConfig, SafetyFilter


@dataclass(frozen=True)
class DriverConfig:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    openings: OpeningsConfig = field(default_factory=OpeningsConfig)
    cruise_speed: float = 1.2         # m/s, default commanded speed
    track: float = 0.3                # per step: how fast the followed heading adapts to the passage

    @classmethod
    def from_dict(cls, d: dict | None) -> "DriverConfig":
        d = dict(d or {})
        sub = {"controller": ControllerConfig, "safety": SafetyConfig, "openings": OpeningsConfig}
        for k, c in sub.items():
            if isinstance(d.get(k), dict):
                v = dict(d[k])
                for key in ("thrusts", "yaw_scales"):
                    if key in v:
                        v[key] = tuple(v[key])
                d[k] = c(**v)
        return cls(**d)


@dataclass(frozen=True)
class Exit:
    heading: float        # rad, odometry frame
    opening: Opening      # as detected (body-frame bearing at the time of the scan)


class Driver:
    def __init__(self, cfg: DriverConfig, vehicle: VehicleConfig, sonar: SonarConfig):
        self.cfg, self.v = cfg, vehicle
        self.angles = Sonar(sonar).angles
        self.max_range = sonar.max_range
        self.full_circle = sonar.fov_deg >= 360.0
        self.controller = HandController(cfg.controller, vehicle, self.angles, self.max_range)
        self.safety = SafetyFilter(cfg.safety, vehicle, self.angles, self.max_range, self.full_circle)
        self.reset()

    def reset(self):
        self.safety.reset()
        self._started = False
        self.ranges = np.full(len(self.angles), self.max_range)
        self.prev_ranges = self.ranges.copy()     # the scan before the current one (motion cue between pings)
        self.a_prev = np.zeros(2, np.float32)     # last applied (filtered) action
        self.overridden = False                   # did the filter change the last action?
        self.u = self.r = 0.0
        self.age = 0.0
        self.exits: list[Exit] = []
        self.heading = 0.0                        # followed heading, odometry frame
        self.speed = self.cfg.cruise_speed
        self.n_steps = self.n_overrides = 0

    @property
    def psi(self) -> float:
        return self.safety.odom.psi

    def sense(self, ranges: np.ndarray, u: float, r: float, sonar_age: float):
        """Call once per step with the observed sonar (held between pings), speed, yaw rate and scan age."""
        if self._started:
            self.safety.update_motion(u, r)       # odometry: the motion of the step just taken
        self.u, self.r, self.age = float(u), float(r), float(sonar_age)
        if not self._started or sonar_age <= 1e-9:   # a new ping
            self.prev_ranges = self.ranges if self._started else np.asarray(ranges, float)
            self.ranges = np.asarray(ranges, float)
            self.safety.update_scan(self.ranges)
            ops = detect_openings(self.ranges, self.angles, self.max_range, self.cfg.openings, self.full_circle)
            self.exits = [Exit(float(wrap(self.psi + o.bearing)), o) for o in ops]
        if not self._started:
            self.heading = self.psi
            self._started = True

    def follow(self, target: Exit | float, speed: float | None = None):
        self.heading = float(target.heading if isinstance(target, Exit) else target)
        if speed is not None:
            self.speed = float(speed)

    def bearing(self) -> float:
        """Followed heading relative to the current body heading."""
        return float(wrap(self.heading - self.psi))

    def propose(self) -> np.ndarray:
        """The controller's command before the safety filter (also the imitation target in phase A1)."""
        a, chosen = self.controller.act(self.ranges, self.u, self.r, self.bearing(), self.speed)
        self.heading = float(wrap(self.heading + self.cfg.track * wrap(self.psi + chosen - self.heading)))
        return a

    def act(self, proposed: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Safe action. `proposed` overrides the hand controller (the learned driver passes its own)."""
        raw = self.propose() if proposed is None else np.asarray(proposed, np.float32)
        a, overridden = self.safety.filter(raw, self.u, self.r, self.age)
        self.n_steps += 1
        self.n_overrides += overridden
        self.a_prev, self.overridden = a, bool(overridden)
        return a, {"raw": raw, "overridden": overridden}

    # ------------------------------------------------------------------ learned driver input
    def features(self, ping_interval: float, extras=()) -> np.ndarray:
        """Input vector of the learned driver (phases A1-A3). Same layout in training, evaluation and phase B:
        sonar now (n) and previous scan (n) / max range, scan age / ping interval, speed / u_max,
        yaw rate / r_max, command bearing (cos, sin), commanded speed / u_max, last applied action (2),
        filter-intervened flag, then `extras` (contact, hull, sensor health, supervisor tier when enabled)."""
        v, b = self.v, self.bearing()
        age = min(self.age / ping_interval, 1.0) if ping_interval > 0 else 0.0
        return np.concatenate([
            self.ranges / self.max_range, self.prev_ranges / self.max_range,
            [age, np.clip(self.u / v.u_max, -1, 1), np.clip(self.r / v.r_max, -1, 1), np.cos(b), np.sin(b),
             self.speed / v.u_max], self.a_prev, [float(self.overridden)], np.asarray(extras, float),
        ]).astype(np.float32)

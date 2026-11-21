"""Policy interface shared by baselines and trained agents, plus an observation decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from submarine.sim import Sonar


class Policy:
    """reset(env) is called after every env.reset(); act(obs) returns [thrust, yaw_cmd].

    Sensor-based policies must only use `obs`. Privileged policies (the oracle)
    may read the true map and state from the env passed to reset().
    """

    name = "policy"
    privileged = False

    def reset(self, env) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ObsLayout:
    """Decodes the flat observation vector (see env/cave_env.py)."""

    n_beams: int
    max_range: float
    angles: np.ndarray        # beam angles in the body frame, rad
    full_circle: bool
    has_goal: bool
    u_max: float
    r_max: float = 1.0
    ping_interval: float = 0.0    # > 0: the scan age is in the observation (sonar slower than the control rate)

    @classmethod
    def from_env(cls, env) -> "ObsLayout":
        c = env.unwrapped.cfg
        if not c.use_sonar:
            raise ValueError("this baseline needs the sonar (sensors must include 'sonar')")
        ping = c.sonar.ping_interval if c.sonar.ping_interval > c.vehicle.dt + 1e-9 else 0.0
        return cls(c.sonar.n_beams, c.sonar.max_range, Sonar(c.sonar).angles,
                   c.sonar.fov_deg >= 360.0, c.goal_cue == "bearing", c.vehicle.u_max, c.vehicle.r_max, ping)

    def yaw_rate(self, obs) -> float:
        return float(self.vec(obs)[self.n_beams + 1]) * self.r_max

    def ping_age(self, obs) -> float:
        """Seconds since the sonar scan in `obs` was taken (0 when the sonar pings every step)."""
        if not self.ping_interval:
            return 0.0
        i = self.n_beams + 2 + (3 if self.has_goal else 0) + 2
        return float(self.vec(obs)[i]) * self.ping_interval

    @staticmethod
    def vec(obs) -> np.ndarray:
        """The state vector (dict observations from camera configs carry it under 'state')."""
        return obs["state"] if isinstance(obs, dict) else obs

    def ranges(self, obs) -> np.ndarray:
        return np.asarray(self.vec(obs)[: self.n_beams], dtype=float) * self.max_range

    def speed(self, obs) -> float:
        return float(self.vec(obs)[self.n_beams]) * self.u_max

    def goal(self, obs) -> tuple[float, float] | None:
        """(bearing in rad, normalised straight-line distance), or None without a goal cue."""
        if not self.has_goal:
            return None
        c, s, d = self.vec(obs)[self.n_beams + 2: self.n_beams + 5]
        return float(np.arctan2(s, c)), float(d)

    def beam_near(self, angle: float) -> int:
        """Index of the beam closest to `angle` (rad)."""
        diff = np.abs((self.angles - angle + np.pi) % (2 * np.pi) - np.pi)
        return int(np.argmin(diff))

    def neighbourhood_min(self, r: np.ndarray, k: int) -> np.ndarray:
        """Each beam's range replaced by the minimum over +-k neighbouring beams."""
        if k <= 0:
            return r
        if self.full_circle:
            return np.min([np.roll(r, s) for s in range(-k, k + 1)], axis=0)
        padded = np.pad(r, k, mode="edge")
        return np.min([padded[k + s: k + s + r.size] for s in range(-k, k + 1)], axis=0)

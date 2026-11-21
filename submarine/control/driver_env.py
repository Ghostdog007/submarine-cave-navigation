"""Training environment for the learned driver (phases A1-A3). Pure NumPy, so it runs in the torch-free workers.

One step = 0.1 s:
  1. the scripted teacher navigator (DriverPolicy) senses and, at decision points, picks an exit + speed;
  2. the hand controller's proposal for that command is computed (the imitation target, info["expert_action"]);
  3. the network's action goes through the driver's built-in safety filter to the simulator.

The network observes Driver.features(): sonar now + previous scan, scan age, speed, yaw rate, the command
(bearing to the followed heading, commanded speed), its last applied action and whether the filter stepped in,
plus contact/hull and sensor-health/supervisor tier when those are enabled. It never sees the goal or a map.

Reward (privileged quantities like the true displacement are fine here: they only shape training):
  + k_track * metres moved along the commanded heading
  - k_overspeed * (speed above the command) * dt
  - k_override * |filtered - proposed|_1            drive so the filter never has to step in
  - k_smooth * |a - a_prev|^2
  - k_energy * electrical energy (kJ)
  - k_damage * hull % lost (damage mode)            - collision / breach penalty at the end
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from submarine.env import CaveEnv, EnvConfig

from .driver import DriverConfig
from .policy import DriverPolicy


@dataclass(frozen=True)
class DriverRewardConfig:
    k_track: float = 1.0
    k_overspeed: float = 2.0
    k_override: float = 0.05
    k_smooth: float = 0.01
    k_energy: float = 1.0
    k_damage: float = 0.1
    collision: float = 10.0


@dataclass(frozen=True)
class DriverEnvConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    driver: DriverConfig = field(default_factory=DriverConfig)
    reward: DriverRewardConfig = field(default_factory=DriverRewardConfig)
    teacher_mode: str = "random"              # random | goal
    p_random: float = 0.5
    speed_range: tuple[float, float] = (0.4, 1.5)
    expert_drives: bool = False               # True: the hand controller drives (imitation data collection)
    expert_noise: float = 0.0                 # std of noise added to the expert's executed action (labels stay
                                              # clean), so the data also covers recovering from small errors

    @classmethod
    def from_dict(cls, d: dict | None) -> "DriverEnvConfig":
        d = dict(d or {})
        if isinstance(d.get("env"), dict):
            d["env"] = EnvConfig.from_dict(d["env"])
        if isinstance(d.get("driver"), dict):
            d["driver"] = DriverConfig.from_dict(d["driver"])
        if isinstance(d.get("reward"), dict):
            d["reward"] = DriverRewardConfig(**d["reward"])
        if "speed_range" in d:
            d["speed_range"] = tuple(d["speed_range"])
        return cls(**d)


N_EXTRAS = 4          # contact, hull, sonar ok, supervisor tier: always present (neutral when disabled), so the
                      # input size is the same in every curriculum stage and one network trains through all of them


def driver_extras(e: CaveEnv) -> list[float]:
    """Contact, hull fraction, sonar-healthy flag, supervisor tier / 4 (neutral values when not enabled)."""
    c = e.cfg
    contact, hull = (float(e._contact), e._hull / 100.0) if c.collision.mode == "damage" else (0.0, 1.0)
    sonar_ok = float(e._failures.sonar_ok()) if (c.failures.enabled or e._failures.active) else 1.0
    tier = e._supervisor.tier / 4.0 if c.supervisor.enabled else 0.0
    return [contact, hull, sonar_ok, tier]


class DriverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: DriverEnvConfig | dict | None = None):
        self.cfg = config if isinstance(config, DriverEnvConfig) else DriverEnvConfig.from_dict(config)
        if not self.cfg.env.use_sonar:
            raise ValueError("the driver needs the sonar")
        self.base = CaveEnv(self.cfg.env)
        self.teacher = DriverPolicy(self.cfg.driver, mode=self.cfg.teacher_mode, p_random=self.cfg.p_random,
                                    speed_range=self.cfg.speed_range)
        c = self.cfg.env
        self.ping = c.sonar.ping_interval if c.sonar.ping_interval > c.vehicle.dt + 1e-9 else 0.0
        n = 2 * c.sonar.n_beams + 9 + N_EXTRAS
        self.observation_space = gym.spaces.Box(-1.5, 1.5, (n,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    # ------------------------------------------------------------------ helpers
    def _obs(self):
        return np.clip(self.teacher.driver.features(self.ping, driver_extras(self.base)), -1.5, 1.5)

    def _world_heading(self) -> float:
        return self.teacher.driver.heading + self._psi_offset

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.base.reset(seed=seed, options=options)
        self.teacher.reset(self.base)
        self.teacher.update(obs)
        self._psi_offset = self.base.sim.state.psi          # odometry frame starts at heading 0
        self._expert = self.teacher.driver.propose()
        self._a_prev = np.zeros(2, np.float32)
        info = dict(info, expert_action=self._expert)
        return self._obs(), info

    def step(self, action):
        d, e, rc = self.teacher.driver, self.base, self.cfg.reward
        if self.cfg.expert_drives:
            proposed = self._expert
            if self.cfg.expert_noise > 0:
                proposed = np.clip(proposed + self.np_random.normal(0, self.cfg.expert_noise, 2), -1, 1).astype(
                    np.float32)
        else:
            proposed = np.clip(np.asarray(action, np.float32), -1, 1)
        heading = self._world_heading()
        v_cmd = d.speed
        s0 = e.sim.state
        a, finfo = d.act(proposed)
        obs, _, terminated, truncated, info = e.step(a)
        s1 = e.sim.state
        self.teacher.t += e.cfg.vehicle.dt

        along = (s1.x - s0.x) * np.cos(heading) + (s1.y - s0.y) * np.sin(heading)
        terms = {
            "track": rc.k_track * along,
            "overspeed": -rc.k_overspeed * max(0.0, s1.u - v_cmd) * e.cfg.vehicle.dt,
            "override": -rc.k_override * float(np.abs(a - proposed).sum()),
            "smooth": -rc.k_smooth * float(np.sum((a - self._a_prev) ** 2)),
            "energy": -rc.k_energy * info["power_w"] * e.cfg.vehicle.dt / 1000.0,
        }
        if e.cfg.collision.mode == "damage":
            lost = -info["reward_terms"].get("damage", 0.0) / max(e.cfg.collision.damage_reward, 1e-9)  # hull %
            terms["damage"] = -rc.k_damage * lost
        if info["collided"]:
            terms["terminal"] = -rc.collision
        self._a_prev = a

        if not (terminated or truncated):
            self.teacher.update(obs)
            self._expert = d.propose()
        reward = float(sum(terms.values()))
        info = dict(info, expert_action=self._expert, driver_terms=terms, filtered=finfo["overridden"],
                    proposed=proposed)
        return self._obs(), reward, terminated, truncated, info

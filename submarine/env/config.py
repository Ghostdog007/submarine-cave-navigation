"""Environment configuration, loadable from YAML (see configs/default.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from submarine.sim import CameraConfig, SonarConfig, TurbidityConfig, VehicleConfig

from .collision import MODES as COLLISION_MODES
from .collision import CollisionConfig
from .failures import SensorFailureConfig
from .navigation import NavigationConfig
from .supervisor import SupervisorConfig

GOAL_CUES = ("bearing", "none")
SEED_SPLITS = ("train", "test")
SENSORS = ("sonar", "camera")


@dataclass(frozen=True)
class RewardConfig:
    k_prog: float = 1.0           # per metre of geodesic progress
    k_time: float = 0.01          # per step
    k_smooth: float = 0.01        # times ||a - a_prev||^2
    goal: float = 20.0            # bonus on reaching the goal
    collision: float = 10.0       # penalty on collision (subtracted)
    k_prox: float = 0.0           # optional wall-proximity penalty weight (off by default)
    prox_dist: float = 0.8        # m of clearance below which the proximity penalty ramps up
    k_energy: float = 0.0         # penalty per kJ of electrical energy (EnergyConfig); 0 = off.
                                  # A typical 45 m run uses ~12 kJ, so k_energy = 1 costs ~12 reward,
                                  # against ~+65 for a success. Steady cruising costs ~0.03 per step.


@dataclass(frozen=True)
class EnergyConfig:
    """Electrical power model, used for the energy metric (it does not enter the reward).

    power = p_surge * |thrust|^1.5 + p_yaw * |yaw_cmd|^1.5 + p_hotel   [W]
    Propeller power grows faster than linearly with thrust; the hotel load
    (electronics, sonar, lights) is drawn all the time, so idling also costs.
    """

    p_surge: float = 300.0        # W at full thrust (forward or reverse)
    p_yaw: float = 120.0          # W at full yaw command
    p_hotel: float = 60.0         # W, always on
    exponent: float = 1.5


def _build(cls, d):
    """Dataclass from a dict, turning YAML lists back into tuples where the field is a tuple."""
    if isinstance(d, cls):
        return d
    d = dict(d or {})
    for f in fields(cls):
        if f.name in d and isinstance(d[f.name], list) and "tuple" in str(f.type):
            d[f.name] = tuple(d[f.name])
    return cls(**d)


def _lists(x):
    if isinstance(x, dict):
        return {k: _lists(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_lists(v) for v in x]
    return x


@dataclass(frozen=True)
class EnvConfig:
    difficulty: tuple[float, float] = (0.0, 0.0)   # each episode samples d uniformly from this range
    difficulty_step: float = 0.05                  # sampled d is rounded to this grid (so caves can be cached)
    seed_split: str = "train"                      # cave seeds from the train or held-out test range
    n_caves: int = 1000                            # size of the cave seed pool
    goal_cue: str = "bearing"                      # "bearing" or "none"
    max_steps: int = 600                           # truncation (60 s at dt = 0.1)
    goal_radius: float = 1.5                       # m
    cave_cache_size: int = 512
    sensors: tuple[str, ...] = ("sonar",)          # any of "sonar", "camera"
    layout: tuple[str, ...] = ("tunnel",)          # cave layouts to sample from (cave/layouts.py)
    reward: RewardConfig = field(default_factory=RewardConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    sonar: SonarConfig = field(default_factory=SonarConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    turbidity: TurbidityConfig = field(default_factory=TurbidityConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)   # dead reckoning + junction events
    failures: SensorFailureConfig = field(default_factory=SensorFailureConfig)   # partial observability
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)       # tiered "stop, think, act"
    collision: CollisionConfig = field(default_factory=CollisionConfig)          # terminate, or damage + hull

    def __post_init__(self):
        d = self.difficulty
        d = (float(d), float(d)) if isinstance(d, (int, float)) else (float(d[0]), float(d[1]))
        object.__setattr__(self, "difficulty", d)
        object.__setattr__(self, "sensors", tuple(self.sensors))
        lay = (self.layout,) if isinstance(self.layout, str) else tuple(self.layout)
        object.__setattr__(self, "layout", lay)
        from submarine.world.layouts import LAYOUTS

        if not lay or not set(lay) <= set(LAYOUTS):
            raise ValueError(f"layout must be one or more of {LAYOUTS}")
        if not 0.0 <= d[0] <= d[1] <= 1.0:
            raise ValueError(f"difficulty range must satisfy 0 <= lo <= hi <= 1, got {d}")
        if self.goal_cue not in GOAL_CUES:
            raise ValueError(f"goal_cue must be one of {GOAL_CUES}")
        if self.seed_split not in SEED_SPLITS:
            raise ValueError(f"seed_split must be one of {SEED_SPLITS}")
        if not self.sensors or not set(self.sensors) <= set(SENSORS):
            raise ValueError(f"sensors must be a non-empty subset of {SENSORS}")
        if self.collision.mode not in COLLISION_MODES:
            raise ValueError(f"collision.mode must be one of {COLLISION_MODES}")

    @property
    def use_camera(self) -> bool:
        return "camera" in self.sensors

    @property
    def use_sonar(self) -> bool:
        return "sonar" in self.sensors

    # ---- dict / YAML -----------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        d = dict(d or {})
        nested = {"reward": RewardConfig, "vehicle": VehicleConfig, "sonar": SonarConfig,
                  "camera": CameraConfig, "turbidity": TurbidityConfig, "energy": EnergyConfig,
                  "navigation": NavigationConfig, "failures": SensorFailureConfig, "supervisor": SupervisorConfig,
                  "collision": CollisionConfig}
        for key, sub in nested.items():
            if key in d:
                d[key] = _build(sub, d[key])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data.get("env", data))

    def to_dict(self) -> dict:
        return _lists(asdict(self))


def coerce_config(config) -> EnvConfig:
    if config is None:
        return EnvConfig()
    if isinstance(config, EnvConfig):
        return config
    if isinstance(config, (str, Path)):
        return EnvConfig.from_yaml(config)
    if isinstance(config, dict):
        return EnvConfig.from_dict(config)
    raise TypeError(f"cannot build EnvConfig from {type(config).__name__}")


__all__ = ["EnergyConfig", "EnvConfig", "RewardConfig", "SENSORS", "coerce_config"]

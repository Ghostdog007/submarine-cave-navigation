"""Training run configuration (YAML). The `env` section is an EnvConfig."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from submarine.env import EnvConfig

ALGOS = ("ppo", "sac", "recurrent_ppo")


@dataclass
class EvalSettings:
    difficulties: list[float] = field(default_factory=lambda: [0.0])
    n_caves: int = 20            # validation caves per difficulty
    freq: int = 50_000           # env steps between evaluations


@dataclass
class TrainConfig:
    name: str = "ppo_mlp_d0"
    algo: str = "ppo"
    policy: str = "MlpPolicy"
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    vec_env: str = "light"       # "light": one torch-free worker process per env (fast, ~150 MB each);
                                 # "subproc": SB3's workers (each loads PyTorch, ~2 GB commit on Windows);
                                 # "dummy": all envs stepped one after another in the main process
    seed: int = 0
    device: str = "cpu"
    frame_stack: int = 1
    algo_kwargs: dict = field(default_factory=dict)
    policy_kwargs: dict = field(default_factory=dict)
    checkpoint_freq: int = 250_000
    eval: EvalSettings = field(default_factory=EvalSettings)
    # Optional curriculum: phases [{at: fraction of total_timesteps, layout: [...], difficulty: [lo, hi],
    # k_energy: x}], each applied to the training envs when training reaches `at` (any key may be omitted).
    curriculum: list = field(default_factory=list)
    env: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self):
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}")
        if isinstance(self.eval, dict):
            self.eval = EvalSettings(**self.eval)
        if isinstance(self.env, dict):
            self.env = EnvConfig.from_dict(self.env)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**(yaml.safe_load(f) or {}))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["env"] = self.env.to_dict()
        return d

    def save(self, path: str | Path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

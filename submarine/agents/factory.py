"""Build vectorised envs and SB3 models from a TrainConfig; load trained runs."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from submarine.env import CaveEnv

from .config import TrainConfig


def _algo_class(algo: str):
    if algo == "ppo":
        return PPO
    if algo == "sac":
        return SAC
    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO
    raise ValueError(algo)


def make_train_env(cfg: TrainConfig):
    """vec_env: "light" (torch-free worker processes, default), "subproc" (SB3 workers), "dummy" (in-process)."""
    if cfg.vec_env == "light" and cfg.n_envs > 1:
        from stable_baselines3.common.vec_env import VecMonitor

        from .light_vec_env import LightSubprocVecEnv

        venv = LightSubprocVecEnv(cfg.env, cfg.n_envs)
        venv.seed(cfg.seed)                      # env i gets seed + i on the first reset, like make_vec_env
        venv = VecMonitor(venv)
    else:
        vec_cls = SubprocVecEnv if cfg.n_envs > 1 and cfg.vec_env == "subproc" else DummyVecEnv
        venv = make_vec_env(CaveEnv, n_envs=cfg.n_envs, seed=cfg.seed, vec_env_cls=vec_cls,
                            env_kwargs={"config": cfg.env})
    if cfg.frame_stack > 1:
        venv = VecFrameStack(venv, cfg.frame_stack)
    return venv


class LinearSchedule:
    """Decays linearly from `start` to 0 over training (SB3 passes progress_remaining: 1 -> 0)."""

    def __init__(self, start: float):
        self.start = start

    def __call__(self, progress_remaining: float) -> float:
        return self.start * progress_remaining


def _parse_kwargs(kwargs: dict) -> dict:
    """Values written as "lin_<x>" in YAML (e.g. learning_rate: lin_3e-4) become linear schedules."""
    out = dict(kwargs)
    for k, v in kwargs.items():
        if isinstance(v, str) and v.startswith("lin_"):
            out[k] = LinearSchedule(float(v[4:]))
    return out


def _policy_kwargs(cfg: TrainConfig) -> dict | None:
    """YAML can only name the feature extractor; map the name to the class."""
    from .extractors import EXTRACTORS

    pk = dict(cfg.policy_kwargs or {})
    fe = pk.get("features_extractor_class")
    if isinstance(fe, str):
        pk["features_extractor_class"] = EXTRACTORS[fe]
    return pk or None


def _policy(cfg: TrainConfig):
    """Policy name -> class for custom policies; SB3 resolves its own names ("MlpPolicy", ...)."""
    return cfg.policy


def make_model(cfg: TrainConfig, venv, tensorboard_log: str | None = None):
    return _algo_class(cfg.algo)(
        _policy(cfg), venv, seed=cfg.seed, device=cfg.device, verbose=0,
        tensorboard_log=tensorboard_log, policy_kwargs=_policy_kwargs(cfg),
        **_parse_kwargs(cfg.algo_kwargs),
    )


def load_run(path: str | Path, which: str = "best_model"):
    """Load a trained model from a run directory (or a .zip inside one).

    Returns (model, TrainConfig). `which` picks best_model / final_model when given a directory.
    """
    path = Path(path)
    run_dir, zip_path = (path, path / f"{which}.zip") if path.is_dir() else (path.parent, path)
    cfg = TrainConfig.from_yaml(run_dir / "config.yaml")
    model = _algo_class(cfg.algo).load(zip_path, device=cfg.device)
    return model, cfg

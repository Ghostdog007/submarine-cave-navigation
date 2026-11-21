"""Torch-free worker processes and parallel evaluation."""

import numpy as np
import pytest
from stable_baselines3 import PPO

from submarine.agents import TrainConfig, make_train_env
from submarine.agents.light_vec_env import LightSubprocVecEnv
from submarine.env import CaveEnv, EnvConfig
from submarine.evaluation import SB3Policy, evaluate
from submarine.evaluation.parallel import ParallelEvaluator

FUSED = EnvConfig.from_dict({"sensors": ["sonar", "camera"], "turbidity": {"enabled": True},
                             "sonar": {"ping_interval": 0.4}, "difficulty": [0, 1]})


def test_workers_do_not_load_torch():
    venv = LightSubprocVecEnv(FUSED, 2)
    try:
        assert all(not d["torch_loaded"] for d in venv.diagnostics())
    finally:
        venv.close()


def test_training_env_workers_stay_torch_free():
    """The full training path (VecMonitor wrapping, seeding, a few steps) must not pull torch into workers."""
    cfg = TrainConfig(name="t", n_envs=2, vec_env="light", env=EnvConfig())
    venv = make_train_env(cfg)
    try:
        venv.reset()
        venv.step(np.zeros((2, 2), dtype=np.float32))
        diags = venv.venv.diagnostics()
        assert all(not d["torch_loaded"] for d in diags), diags
    finally:
        venv.close()


def test_light_vec_env_matches_single_env():
    """Worker env i, seeded seed + i, behaves exactly like a local CaveEnv with that seed."""
    cfg = TrainConfig(name="t", n_envs=2, seed=5, vec_env="light", env=EnvConfig())
    venv = make_train_env(cfg)
    try:
        obs = venv.reset()
        local = CaveEnv()
        lobs, _ = local.reset(seed=6)
        assert np.allclose(obs[1], lobs)
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.uniform(-1, 1, (2, 2)).astype(np.float32)
            obs, rew, done, info = venv.step(a)
            lobs, lr, te, tr, _ = local.step(a[1])
            if done[1]:
                break
            assert np.allclose(obs[1], lobs) and np.isclose(rew[1], lr)
    finally:
        venv.close()


@pytest.mark.parametrize("cfg", [EnvConfig(difficulty=(0, 1)), FUSED], ids=["sonar", "fused"])
def test_parallel_evaluation_matches_sequential(cfg):
    policy = "MultiInputPolicy" if cfg.use_camera else "MlpPolicy"
    model = PPO(policy, CaveEnv(cfg), seed=0, device="cpu")       # untrained but deterministic
    seq = evaluate(SB3Policy(model, "rl"), cfg, [0.0, 1.0], n_caves=3, split="val")
    pool = ParallelEvaluator(cfg, n_workers=3)
    try:
        par = pool.evaluate(model, [0.0, 1.0], n_caves=3, split="val")
    finally:
        pool.close()
    key = lambda r: (r.difficulty, r.cave_seed)
    for a, b in zip(sorted(seq, key=key), sorted(par, key=key)):
        assert (a.cave_seed, a.outcome, a.steps) == (b.cave_seed, b.outcome, b.steps)
        assert a.episode_return == pytest.approx(b.episode_return, abs=1e-3)


def test_multi_env_workers_match_single_envs():
    """MultiEnvVecEnv (several envs per worker): env i, seeded seed + i, behaves exactly like a local env."""
    from submarine.agents.light_vec_env import MultiEnvVecEnv

    venv = MultiEnvVecEnv(EnvConfig(), 4, envs_per_worker=2)
    try:
        venv.seed(5)
        obs = venv.reset()
        local = CaveEnv()
        lobs, _ = local.reset(seed=8)                     # env 3 = worker 1, slot 1
        assert np.allclose(obs[3], lobs)
        rng = np.random.default_rng(0)
        for _ in range(30):
            a = rng.uniform(-1, 1, (4, 2)).astype(np.float32)
            obs, rew, done, info = venv.step(a)
            lobs, lr, te, tr, _ = local.step(a[3])
            if done[3]:
                break
            assert np.allclose(obs[3], lobs) and np.isclose(rew[3], lr)
        assert venv.get_attr("cfg", [1, 2])[0] == EnvConfig() and all(not d["torch_loaded"] for d in venv.diagnostics())
    finally:
        venv.close()

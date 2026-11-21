"""Phase A1-A3 plumbing: driver training env, NumPy export, train/eval input parity, torch-free workers."""

import numpy as np
import pytest

from submarine.control import DriverEnv, DriverEnvConfig
from submarine.control.learned import LearnedDriverPolicy, NumpyActor, export_sb3_actor
from submarine.env import EnvConfig

FULL = {"env": {"layout": ["tunnel", "crack"], "difficulty": [0.0, 0.5], "sonar": {"ping_interval": 0.4},
                "failures": {"enabled": True, "rate_sonar_total": 20.0}, "supervisor": {"enabled": True},
                "collision": {"mode": "damage"}, "max_steps": 300}}


def test_input_size_same_in_every_stage():
    shapes = {DriverEnv({"env": e}).observation_space.shape for e in (
        {}, {"sonar": {"ping_interval": 0.4}}, FULL["env"], {"collision": {"mode": "damage"}})}
    assert shapes == {(2 * 32 + 9 + 4,)}


def test_expert_drives_easy_tunnel_without_collision():
    env = DriverEnv({"env": {"difficulty": 0.0, "max_steps": 400}, "expert_drives": True, "teacher_mode": "goal"})
    obs, info = env.reset(seed=0, options={"cave_seed": 1000000, "difficulty": 0.0})
    total = {"track": 0.0}
    for _ in range(400):
        obs, r, te, tr, info = env.step(np.zeros(2, np.float32))
        assert env.observation_space.contains(obs)
        total["track"] += info["driver_terms"]["track"]
        if te or tr:
            break
    assert not info["collided"] and total["track"] > 10.0       # actually drove along its commands


def test_expert_noise_changes_executed_not_label():
    env = DriverEnv({"env": {"difficulty": 0.0}, "expert_drives": True, "expert_noise": 0.3})
    obs, info = env.reset(seed=1)
    label = info["expert_action"]
    _, _, _, _, info = env.step(np.zeros(2, np.float32))
    assert not np.allclose(info["proposed"], label)


def test_numpy_actor_matches_sb3(tmp_path):
    import torch as th
    from stable_baselines3 import PPO

    env = DriverEnv({"env": {"difficulty": 0.0}})
    model = PPO("MlpPolicy", env, seed=0, device="cpu", policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32]}})
    actor = export_sb3_actor(model, tmp_path / "d.npz")
    loaded = NumpyActor.load(tmp_path / "d.npz")
    obs = np.stack([env.reset(seed=i)[0] for i in range(5)])
    want = np.clip(model.predict(obs, deterministic=True)[0], -1, 1)
    assert np.allclose(actor(obs), want, atol=1e-5) and np.allclose(loaded(obs), want, atol=1e-5)


def test_eval_policy_builds_same_input_as_training_env(tmp_path):
    """LearnedDriverPolicy (decoding a CaveEnv observation) must feed the network exactly DriverEnv's input."""
    from stable_baselines3 import PPO

    cfg = DriverEnvConfig.from_dict(FULL)
    denv = DriverEnv(cfg)
    model = PPO("MlpPolicy", denv, seed=0, device="cpu", policy_kwargs={"net_arch": [16]})
    export_sb3_actor(model, tmp_path / "d.npz")
    pol = LearnedDriverPolicy(tmp_path / "d.npz", mode="random", seed=7, p_random=cfg.p_random,
                              speed_range=cfg.speed_range)
    denv.teacher.seed = 7                                   # same teacher decisions
    obs_d, _ = denv.reset(seed=3, options={"cave_seed": 5, "difficulty": 0.3,
                                           "failures": [("sonar_total", 1.0, 2.0)]})
    from submarine.env import CaveEnv

    cave_env = CaveEnv(cfg.env)
    obs_c, _ = cave_env.reset(seed=3, options={"cave_seed": 5, "difficulty": 0.3,
                                               "failures": [("sonar_total", 1.0, 2.0)]})
    pol.reset(cave_env)
    for t in range(60):
        a_pol = pol.act(obs_c)                               # policy builds the network input internally
        a_d = np.asarray(model.predict(obs_d, deterministic=True)[0])
        assert np.allclose(pol.last_input, obs_d, atol=1e-5), t
        obs_d, _, te, tr, _ = denv.step(a_d)
        obs_c, _, te2, tr2, _ = cave_env.step(a_pol)
        if te or tr:
            break


def test_workers_build_driver_env_without_torch():
    from submarine.agents.light_vec_env import LightSubprocVecEnv

    venv = LightSubprocVecEnv(("submarine.control.driver_env:DriverEnv", DriverEnvConfig.from_dict(FULL)), 2)
    try:
        venv.reset()
        venv.step(np.zeros((2, 2), np.float32))
        assert all(not d["torch_loaded"] for d in venv.diagnostics())
    finally:
        venv.close()

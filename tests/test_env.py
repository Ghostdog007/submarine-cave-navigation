from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import submarine.env as _register  # noqa: F401  (registers SubmarineCave-v0)
from submarine.world import TEST_SEED_OFFSET
from submarine.env import CaveEnv, EnvConfig

ROOT = Path(__file__).resolve().parents[1]


def test_spaces():
    e = CaveEnv()
    assert e.observation_space.shape == (39,)
    assert CaveEnv({"goal_cue": "none"}).observation_space.shape == (36,)
    assert CaveEnv({"sonar": {"n_beams": 24, "fov_deg": 120}}).observation_space.shape == (31,)
    assert e.action_space.shape == (2,)


def test_checkers():
    gym_check_env(CaveEnv({"difficulty": [0, 1]}), skip_render_check=True)
    sb3_check_env(CaveEnv({"difficulty": [0, 1]}), warn=True)


def test_registered():
    e = gym.make("SubmarineCave-v0", config={"difficulty": 0.5}, max_episode_steps=None)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (39,)


def test_obs_in_bounds_and_deterministic():
    def rollout(seed):
        e = CaveEnv({"difficulty": [0, 1]})
        obs, _ = e.reset(seed=seed)
        rng = np.random.default_rng(0)
        out = [obs]
        for _ in range(100):
            obs, r, term, trunc, _ = e.step(rng.uniform(-1, 1, 2))
            assert e.observation_space.contains(obs)
            out.append(obs)
            if term or trunc:
                obs, _ = e.reset()
        return np.array(out)

    assert np.array_equal(rollout(3), rollout(3))
    assert not np.array_equal(rollout(3), rollout(4))


def test_progress_reward_telescopes():
    """Summed progress reward = k_prog * (geodesic at start - geodesic at end)."""
    e = CaveEnv()
    _, info = e.reset(seed=0)
    d0 = info["geodesic"]
    total = 0.0
    for _ in range(40):
        _, _, term, trunc, info = e.step([0.6, 0.0])
        total += info["reward_terms"]["progress"]
        if term or trunc:
            break
    assert total == pytest.approx(d0 - info["geodesic"], abs=1e-6)
    assert total > 0                         # driving along the passage makes progress


def test_collision_terminates_with_penalty():
    e = CaveEnv()
    e.reset(seed=0, options={"cave_seed": 0, "difficulty": 0.0})
    for _ in range(600):
        _, r, term, trunc, info = e.step([1.0, 0.0])   # no steering: hits the first bend
        if term or trunc:
            break
    assert term and not trunc and info["collided"] and not info["is_success"]
    assert info["reward_terms"]["terminal"] == -10.0


def test_goal_terminates_with_bonus():
    e = CaveEnv({"goal_radius": 100.0})
    e.reset(seed=0)
    _, r, term, trunc, info = e.step([0.0, 0.0])
    assert term and info["is_success"] and info["reward_terms"]["terminal"] == 20.0
    assert r == pytest.approx(20.0 - 0.01, abs=1e-6)


def test_timeout_truncates():
    e = CaveEnv({"max_steps": 5})
    e.reset(seed=0)
    for i in range(5):
        _, _, term, trunc, info = e.step([0.0, 0.0])
        assert not term and trunc == (i == 4)
    assert info["timeout"]


def test_difficulty_sampling_and_curriculum():
    e = CaveEnv({"difficulty": [0.2, 0.6]})
    ds = set()
    for i in range(60):
        _, info = e.reset(seed=i)
        ds.add(info["difficulty"])
    assert all(0.2 <= d <= 0.6 for d in ds) and len(ds) > 3
    assert all(abs(d / 0.05 - round(d / 0.05)) < 1e-9 for d in ds)
    e.set_difficulty(0.9)
    assert e.reset(seed=0)[1]["difficulty"] == 0.9


def test_test_split_uses_held_out_seeds():
    e = CaveEnv({"seed_split": "test", "n_caves": 10})
    seeds = {e.reset(seed=i)[1]["cave_seed"] for i in range(20)}
    assert all(TEST_SEED_OFFSET <= s < TEST_SEED_OFFSET + 10 for s in seeds)


def test_goal_cue_points_at_goal():
    e = CaveEnv()
    obs, _ = e.reset(seed=0)
    s, (gx, gy) = e.sim.state, e.cave.goal
    bearing = np.arctan2(gy - s.y, gx - s.x) - s.psi
    assert obs[34] == pytest.approx(np.cos(bearing), abs=1e-5)
    assert obs[35] == pytest.approx(np.sin(bearing), abs=1e-5)


def test_yaml_matches_defaults():
    assert EnvConfig.from_yaml(ROOT / "configs" / "default.yaml") == EnvConfig()
    assert EnvConfig.from_dict(EnvConfig().to_dict()) == EnvConfig()


def test_rgb_render():
    e = CaveEnv(render_mode="rgb_array")
    e.reset(seed=0)
    e.step([1.0, 0.0])
    frame = e.render()
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
    e.close()

import numpy as np
import pytest

from submarine.baselines import BASELINES, GapFollower, ObsLayout, Oracle, RandomPolicy
from submarine.baselines.reactive import corridor_front, corridor_ranges
from submarine.env import CaveEnv
from submarine.evaluation import SB3Policy, evaluate, summarize


def test_obs_layout_decodes_env_obs():
    env = CaveEnv()
    obs, _ = env.reset(seed=0)
    L = ObsLayout.from_env(env)
    assert np.allclose(L.ranges(obs), np.clip(env._last.sonar, 0, 10), atol=1e-5)
    bearing, _ = L.goal(obs)
    s, (gx, gy) = env.sim.state, env.cave.goal
    assert np.isclose(np.cos(bearing), np.cos(np.arctan2(gy - s.y, gx - s.x) - s.psi), atol=1e-5)
    assert ObsLayout.from_env(CaveEnv({"goal_cue": "none"})).goal(obs) is None


def test_corridor_geometry():
    th = np.deg2rad([-90.0, -30.0, 0.0, 30.0, 90.0])
    r = np.array([1.0, 10.0, 10.0, 2.0, 1.0])            # side walls at 1 m, an obstacle 2 m out at +30 deg
    # obstacle at lateral offset 2 sin 30 = 1.0 m: outside a 0.6 m corridor, inside a 1.2 m one
    assert corridor_front(r, th, 0.6, 10.0) == 10.0
    assert corridor_front(r, th, 1.2, 10.0) == pytest.approx(2 * np.cos(np.deg2rad(30)))
    assert corridor_ranges(r, th, 0.6, 10.0)[3] == pytest.approx(2.0)   # looking straight at it


@pytest.mark.parametrize("d", [0.0, 1.0])
def test_oracle_solves_caves(d):
    rows = summarize(evaluate(Oracle(), None, [d], n_caves=5))
    assert rows[0]["success"] == 1.0


def test_gap_beats_random_on_easy_caves():
    gap = summarize(evaluate(GapFollower(), None, [0.0], n_caves=6))[0]
    rnd = summarize(evaluate(RandomPolicy(), None, [0.0], n_caves=6))[0]
    assert gap["success"] >= 0.8 and rnd["success"] == 0.0


def test_same_caves_and_noise_for_every_policy():
    a = evaluate(BASELINES["centering"](), None, [0.5], n_caves=3)
    b = evaluate(BASELINES["centering"](), None, [0.5], n_caves=3)
    assert [r.cave_seed for r in a] == [r.cave_seed for r in b]
    assert [r.episode_return for r in a] == [r.episode_return for r in b]


def test_sb3_adapter_runs():
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", CaveEnv(), n_steps=64, batch_size=64, device="cpu", seed=0)
    rows = summarize(evaluate(SB3Policy(model, "ppo_untrained"), None, [0.0], n_caves=2))
    assert rows[0]["policy"] == "ppo_untrained" and rows[0]["episodes"] == 2

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from submarine.baselines import GapFollower, Oracle
from submarine.env import CaveEnv, EnvConfig
from submarine.evaluation import evaluate, summarize
from submarine.sim import Camera, CameraConfig, TurbidityConfig, make_turbidity
from submarine.world import generate_cave

FUSED = {"sensors": ["sonar", "camera"], "turbidity": {"enabled": True}, "sonar": {"ping_interval": 0.4},
         "difficulty": [0, 1]}


def wall_scene(turb):
    rock = np.zeros((50, 200), bool)
    rock[:, 150:] = True                                   # wall at x = 30 m
    return rock, np.full(rock.shape, turb, np.float32), np.full(rock.shape, 0.85, np.float32)


def test_camera_brightness_falls_with_distance_in_clear_water():
    cam = Camera(CameraConfig())
    rock, turb, alb = wall_scene(0.0)
    centre = [cam.render(rock, turb, alb, 0.2, 30 - d, 5.0, 0.0)[0][32] for d in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(centre, centre[1:]))
    _, depth = cam.render(rock, turb, alb, 0.2, 27.0, 5.0, 0.0)
    assert abs(depth[32] - 3.0) < 0.15


def test_murk_hides_walls():
    cam = Camera(CameraConfig())
    contrast = {}
    for t in (0.0, 1.0):
        rock, turb, alb = wall_scene(t)
        near = cam.render(rock, turb, alb, 0.2, 27.0, 5.0, 0.0)[0][32]      # wall 3 m ahead
        rock[:] = False
        far = cam.render(rock, turb, alb, 0.2, 27.0, 5.0, 0.0)[0][32]       # no wall at all
        contrast[t] = near - far
    assert contrast[0.0] > 0.3 and contrast[1.0] < 0.02


def test_turbidity_field():
    cave = generate_cave(3, 0.5)
    t1 = make_turbidity(cave, TurbidityConfig(enabled=True))
    assert np.array_equal(t1, make_turbidity(cave, TurbidityConfig(enabled=True)))   # fixed per cave
    assert t1.min() >= 0 and t1.max() <= 1 and t1.max() > 0.5
    assert make_turbidity(cave, TurbidityConfig(enabled=False)).max() == 0
    half = make_turbidity(cave, TurbidityConfig(enabled=True, scale=0.5))
    assert half.mean() < t1.mean()


@pytest.mark.parametrize("sensors,state_dim", [(["sonar", "camera"], 40), (["camera"], 7), (["sonar"], 40)])
def test_observation_layouts(sensors, state_dim):
    e = CaveEnv({**FUSED, "sensors": sensors})
    obs, _ = e.reset(seed=0)
    if "camera" in sensors:
        assert obs["camera"].shape == (1, 64) and obs["state"].shape == (state_dim,)
    else:
        assert obs.shape == (state_dim,)
    assert e.observation_space.contains(obs)


def test_checkers_with_camera():
    gym_check_env(CaveEnv(FUSED), skip_render_check=True)
    sb3_check_env(CaveEnv(FUSED), warn=True)


def test_sonar_held_between_pings():
    e = CaveEnv(FUSED)
    e.reset(seed=0)
    scans = []
    for _ in range(8):
        e.step([0.8, 0.2])
        scans.append(e._last.sonar.copy())
    assert np.array_equal(scans[0], scans[1]) and np.array_equal(scans[1], scans[2])
    assert not np.array_equal(scans[2], scans[3])        # new ping every 4 steps


def test_default_config_unchanged():
    """Old configs and trained models: sonar only, ping every step, flat 39-value observation."""
    e = CaveEnv()
    assert e.observation_space.shape == (39,)
    assert EnvConfig.from_dict(EnvConfig(**{"sensors": ("sonar", "camera")}).to_dict()).sensors == ("sonar", "camera")


def test_baselines_and_cnn_policy_on_dict_obs(tmp_path):
    from submarine.agents import TrainConfig, load_run, make_model, make_train_env
    from submarine.evaluation import make_policy

    rows = summarize(evaluate(GapFollower(), FUSED, [0.0], n_caves=2))
    assert rows[0]["episodes"] == 2
    cfg = TrainConfig(name="t", policy="MultiInputPolicy", n_envs=1, env=EnvConfig.from_dict(FUSED),
                      algo_kwargs={"n_steps": 32, "batch_size": 32},
                      policy_kwargs={"features_extractor_class": "camera_state", "net_arch": [32]})
    model = make_model(cfg, make_train_env(cfg))
    model.learn(64)
    cfg.save(tmp_path / "config.yaml")
    model.save(tmp_path / "best_model")
    pol = make_policy(str(tmp_path))
    # evaluated under a sonar-only base config, it still gets its camera
    rows = summarize(evaluate(pol, {"difficulty": 0.0, "turbidity": {"enabled": True}}, [0.0], n_caves=1))
    assert rows[0]["episodes"] == 1

from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from submarine.agents import TrainConfig, load_run, make_model, make_train_env
from submarine.world import TEST_SEED_OFFSET, VAL_SEED_OFFSET, train_seeds, val_seeds
from submarine.world import test_seeds as held_out_seeds
from submarine.env import CaveEnv
from submarine.evaluation import SB3Policy

ROOT = Path(__file__).resolve().parents[1]


def test_seed_splits_disjoint():
    tr, va, te = set(train_seeds(1000)), set(val_seeds(1000)), set(held_out_seeds(1000))
    assert not (tr & va or tr & te or va & te)
    assert max(tr) < VAL_SEED_OFFSET <= min(va) and max(va) < TEST_SEED_OFFSET <= min(te)


def test_config_roundtrip(tmp_path):
    cfg = TrainConfig.from_yaml(ROOT / "configs" / "ppo_d0.yaml")
    assert cfg.algo == "ppo" and cfg.env.difficulty == (0.0, 0.0)
    cfg.save(tmp_path / "config.yaml")
    assert TrainConfig.from_yaml(tmp_path / "config.yaml") == cfg


@pytest.mark.parametrize("algo,policy", [("ppo", "MlpPolicy"), ("sac", "MlpPolicy"),
                                         ("recurrent_ppo", "MlpLstmPolicy")])
def test_models_build_and_save(tmp_path, algo, policy):
    kwargs = {"ppo": {"n_steps": 32, "batch_size": 32},
              "sac": {"learning_starts": 10, "buffer_size": 1000},
              "recurrent_ppo": {"n_steps": 32, "batch_size": 32}}[algo]
    cfg = TrainConfig(name="t", algo=algo, policy=policy, n_envs=1, algo_kwargs=kwargs)
    venv = make_train_env(cfg)
    model = make_model(cfg, venv)
    model.learn(64)
    cfg.save(tmp_path / "config.yaml")
    model.save(tmp_path / "best_model")
    loaded, lcfg = load_run(tmp_path)
    assert lcfg.algo == algo
    pol = SB3Policy(loaded, frame_stack=lcfg.frame_stack)
    env = CaveEnv()
    obs, _ = env.reset(seed=0)
    pol.reset(env)
    for _ in range(5):
        obs, *_ = env.step(pol.act(obs))


def test_frame_stack_matches_vecframestack():
    k = 4
    venv = VecFrameStack(DummyVecEnv([lambda: CaveEnv()]), k)
    venv.seed(0)
    stacked = [venv.reset()[0]]
    env = CaveEnv()
    obs, _ = env.reset(seed=0)
    pol = SB3Policy(model=None, frame_stack=k)
    pol.reset(env)
    mine = [pol._stacked(obs)]
    rng = np.random.default_rng(0)
    for _ in range(6):
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        stacked.append(venv.step(a[None])[0][0])
        obs, *_ = env.step(a)
        mine.append(pol._stacked(obs))
    assert np.allclose(np.array(stacked), np.array(mine))

from submarine.agents import CurriculumCallback, TrainConfig, make_model, make_train_env
from submarine.env import CaveEnv, EnvConfig


def test_env_setters():
    e = CaveEnv()
    e.set_layouts(["maze", "crack"])
    e.set_energy_penalty(0.5)
    e.set_difficulty(0.3, 0.6)
    assert e.cfg.layout == ("maze", "crack") and e.cfg.reward.k_energy == 0.5
    lays = {e.reset(seed=i)[1]["layout"] for i in range(20)}
    assert lays == {"maze", "crack"}


def test_curriculum_switches_phases_in_worker_envs():
    phases = [{"at": 0.0, "layout": ["crack"], "difficulty": [0.2, 0.2], "k_energy": 0.0},
              {"at": 0.5, "layout": ["maze"], "k_energy": 2.0}]
    cfg = TrainConfig(name="t", n_envs=2, vec_env="light", env=EnvConfig(),
                      algo_kwargs={"n_steps": 32, "batch_size": 32, "n_epochs": 1})
    venv = make_train_env(cfg)
    try:
        model = make_model(cfg, venv)
        cb = CurriculumCallback(phases, total_timesteps=256)
        model.learn(256, callback=cb)
        assert cb.current == 1
        cfgs = venv.get_attr("cfg")
        assert all(c.layout == ("maze",) and c.reward.k_energy == 2.0 for c in cfgs)
        assert all(d == (0.2, 0.2) for d in venv.get_attr("_difficulty"))
    finally:
        venv.close()

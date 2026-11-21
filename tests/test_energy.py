import numpy as np
import pytest

from submarine.env import CaveEnv


def test_energy_metric_and_power_model():
    e = CaveEnv()
    e.reset(seed=0)
    _, _, _, _, info = e.step([1.0, 0.0])
    assert info["power_w"] == pytest.approx(300 + 60)            # full thrust + hotel load
    _, _, _, _, info = e.step([0.0, -1.0])
    assert info["power_w"] == pytest.approx(120 + 60)            # full yaw + hotel load
    _, _, _, _, info = e.step([-0.25, 0.0])
    assert info["power_w"] == pytest.approx(300 * 0.25 ** 1.5 + 60)
    assert info["energy_j"] == pytest.approx(0.1 * (360 + 180 + 300 * 0.125 + 60))


def test_energy_penalty_off_by_default_and_on_when_set():
    def rollout(cfg):
        e = CaveEnv(cfg)
        e.reset(seed=3)
        rng = np.random.default_rng(0)
        out = []
        for _ in range(30):
            _, r, te, tr, info = e.step(rng.uniform(-1, 1, 2))
            out.append((r, info))
            if te or tr:
                break
        return out

    base, on = rollout(None), rollout({"reward": {"k_energy": 1.0}})
    assert all("energy" not in info["reward_terms"] for _, info in base)
    for (r0, i0), (r1, i1) in zip(base, on):
        expected = -1.0 * i1["power_w"] * 0.1 / 1000
        assert i1["reward_terms"]["energy"] == pytest.approx(expected)
        assert r1 == pytest.approx(r0 + expected)

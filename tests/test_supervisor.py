import numpy as np
import pytest

from submarine.env import CaveEnv
from submarine.env.navigation import N_NAV_FEATURES

BASE = {"difficulty": 0.0, "sonar": {"ping_interval": 0.4}, "max_steps": 600}
SUP = {**BASE, "supervisor": {"enabled": True}}


def run(cfg, schedule, steps, action=(1.0, 0.0), seed=0):
    e = CaveEnv(cfg)
    obs, info = e.reset(seed=seed, options={"cave_seed": 1000001, "difficulty": 0.0, "failures": schedule})
    log = []
    for _ in range(steps):
        obs, r, te, tr, info = e.step(action)
        log.append((obs, r, info, e.sim.state.u))
        if te or tr:
            break
    return e, log


def test_defaults_unchanged():
    assert CaveEnv().observation_space.shape == (39,)
    assert CaveEnv(BASE).observation_space.shape == (40,)


def test_sonar_failure_and_recovery():
    e, log = run({**BASE, "failures": {"enabled": False}, "supervisor": {"enabled": False}},
                 [("sonar_total", 1.0, 2.0)], 50, action=(0.2, 0.0))
    healthy_sonar = log[5][0][:32]
    failed_sonar = log[15][0][:32]
    assert not np.allclose(healthy_sonar, 1.0) and np.allclose(failed_sonar, 1.0)   # "open water everywhere"
    assert log[15][2]["failures"] == ("sonar_total",)
    assert not np.allclose(log[40][0][:32], 1.0)                                    # back after 3 s


def test_supervisor_tiers_stop_then_slow_then_recover():
    e, log = run(SUP, [("sonar_total", 3.0, 8.0)], 140)       # blind from 3 s to 11 s, policy wants full speed
    tiers = [l[2]["tier"] for l in log]
    speeds = [l[3] for l in log]
    assert tiers[20] == 0 and speeds[25] > 0.5                  # cruising before the failure
    t_fail = next(i for i, t in enumerate(tiers) if t != 0)
    assert tiers[t_fail] == 2                                    # STOP first
    assert speeds[t_fail + 28] < 0.1                             # braked to (almost) a halt within ~3 s
    assert all(t == 3 for t in tiers[t_fail + 32:t_fail + 75])   # then ACT slowly ...
    assert max(speeds[t_fail + 45:t_fail + 75]) <= 0.2 + 0.02    # ... at <= 0.2 m/s
    assert tiers[-1] == 0                                        # recovered once the sonar is back
    assert any(l[2]["overridden"] for l in log) and min(l[2]["reward_terms"]["supervisor"] for l in log) < 0


def test_abort_when_blind_too_long():
    e, log = run({**SUP, "supervisor": {"enabled": True, "abort_s": 5.0}}, [("sonar_total", 1.0, 100.0)], 200)
    last = log[-1][2]
    assert last["aborted"] and last["reward_terms"]["terminal"] == -5.0 and not last["collided"]
    assert len(log) < 80                                         # ended at ~6 s


def test_camera_failures_and_obs_layout():
    cfg = {**BASE, "sensors": ["sonar", "camera"], "turbidity": {"enabled": True},
           "failures": {"enabled": True}, "supervisor": {"enabled": True}, "navigation": {"enabled": True}}
    e, log = run(cfg, [("camera_blackout", 1.0, 2.0), ("camera_frozen", 5.0, 2.0)], 75, action=(0.3, 0.0))
    obs_black = log[15][0]
    assert np.all(obs_black["camera"] == 0) and obs_black["state"][-N_NAV_FEATURES - 2] == 0   # camera flag 0
    assert np.array_equal(log[55][0]["camera"], log[60][0]["camera"])                            # frozen frames
    assert log[60][0]["state"][-N_NAV_FEATURES - 2] == 1                                         # unflagged
    assert e.observation_space.contains(log[60][0])
    # camera alone failing is degraded (tier 1), not blind
    assert log[15][2]["tier"] == 1


def test_random_failures_occur():
    e = CaveEnv({**BASE, "failures": {"enabled": True, "rate_sonar_total": 30.0}})
    e.reset(seed=0)
    seen = set()
    for _ in range(300):
        *_, info = e.step([0.0, 0.0])
        seen |= set(info["failures"])
    assert "sonar_total" in seen

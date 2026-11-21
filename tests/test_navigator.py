"""Navigator environment: every decision is a real exit or a turn-back (no free "carry on"), and the teacher."""

import numpy as np

from submarine.navigator import N_SLOTS, NavigatorEnv, OracleNavigator, slot_valid

V = lambda *front: list(front) + [True]                   # 4 forward slots + turn back (always valid)


def test_empty_slot_resolves_to_nearest_exit_or_turn_back():
    r = NavigatorEnv.resolve
    assert r(1, V(True, True, False, False)) == 1             # valid: unchanged
    assert r(3, V(True, True, False, False)) == 1             # empty: nearest valid slot index
    assert r(2, V(True, False, False, True)) == 3             # slot 3 is 1 away, slot 0 is 2 away
    assert r(0, V(False, False, True, False)) == 2
    assert r(2, V(False, False, False, False)) == N_SLOTS - 1  # dead end: turn back
    assert r(N_SLOTS - 1, V(True, False, False, False)) == N_SLOTS - 1


def test_navigator_env_never_carries_on_through_an_empty_slot():
    env = NavigatorEnv({"env": {"layout": "maze", "difficulty": 0.0, "max_steps": 400}})
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        valid = slot_valid(obs)
        obs2, r, te, tr, info = env.step(3)                   # slot 3 is usually empty
        assert valid[info["choice"]] or info["choice"] == N_SLOTS - 1
        assert info["invalid"] == (info["requested"] != info["choice"])
        obs = obs2
        if te or tr:
            break


def test_oracle_prefers_the_way_to_the_goal():
    """The map-aware teacher beats random exits on the same caves (it is the warm start's label source)."""
    from scripts.eval_navigator import run_episode
    from submarine.navigator import RandomNavigator

    env = NavigatorEnv({"env": {"layout": "branches", "max_steps": 900}})
    wins = {"oracle": 0, "random": 0}
    for seed in (1000000, 1000001, 1000002):
        for name, nav in (("oracle", OracleNavigator()), ("random", RandomNavigator(seed))):
            outcome, info, _ = run_episode(env, nav, seed, 0.3)
            wins[name] += outcome == "success"
    assert wins["oracle"] >= 2 and wins["oracle"] > wins["random"]

import numpy as np
import pytest

from submarine.env import CaveEnv

DMG = {"difficulty": 0.0, "collision": {"mode": "damage"}, "max_steps": 600}


def drive(cfg, action, steps, seed=0):
    e = CaveEnv(cfg)
    e.reset(seed=seed, options={"cave_seed": 0, "difficulty": 0.0})
    log = []
    for _ in range(steps):
        obs, r, te, tr, info = e.step(action)
        log.append((obs, r, te, info, e.sim.state.u))
        if te or tr:
            break
    return e, log


def test_terminate_mode_unchanged():
    e, log = drive({"difficulty": 0.0}, (1.0, 0.0), 600)
    assert log[-1][2] and log[-1][3]["collided"]                       # first contact ends the episode
    assert CaveEnv().observation_space.shape == (39,)


def test_full_speed_hit_damages_but_continues():
    e, log = drive(DMG, (1.0, 0.0), 600)
    hits = [(i, l) for i, l in enumerate(log) if l[3]["contact"]]
    assert hits, "should hit the first bend"
    i, first = hits[0]
    assert first[3]["impact_speed"] > 0.5
    assert first[3]["hull"] < 95                                        # a hard hit costs real hull
    assert first[1] < -0.5                                             # damage penalty in the reward
    assert e.sim.clearance_at(*first[3]["position"]) >= 0.4 - 1e-3     # pushed back out of the wall
    assert log[i][4] < log[i - 1][4]                                   # lost speed
    assert not (log[i][2] and not log[i][3]["collided"])               # didn't end unless breached


def test_repeated_hard_hits_breach_the_hull():
    e, log = drive({**DMG, "max_steps": 1500}, (1.0, 0.0), 1500)
    last = log[-1][3]
    assert last["collided"] and last["hull"] <= 1e-6                    # ended by a breach
    assert last["reward_terms"]["terminal"] == -10.0
    assert last["contacts"] >= 2


def test_gentle_contact_is_cheap():
    """Creeping into a wall at <= safe speed costs only speed-proportional scrapes: a few % over a long time."""
    e = CaveEnv(DMG)
    e.reset(seed=0, options={"cave_seed": 0, "difficulty": 0.0})
    cap = 0.5 * 0.15 / 0.75                                           # thrust settling at 0.15 m/s
    contacts = 0
    for _ in range(1200):
        *_, info = e.step([cap, 0.0])
        if info["contact"]:
            contacts += 1
            assert info["impact_speed"] <= 0.2 + 1e-6
    assert contacts > 50                                                # leaning on the wall for a long time ...
    assert 100 - info["hull"] < 10                                      # ... costs only a few % of hull


def test_pinned_full_thrust_grinding_is_bounded():
    """Pushing into a wall at full thrust: speed is killed by the wall, so it grinds slowly, not breaching fast."""
    e, log = drive(DMG, (1.0, 0.0), 400)
    first = next(i for i, l in enumerate(log) if l[3]["contact"])
    hull_after_hit = log[first][3]["hull"]
    hull_later = log[min(first + 100, len(log) - 1)][3]["hull"]
    assert hull_after_hit - hull_later < 25                             # 10 s of grinding < 25 % extra


def test_obs_has_touch_and_hull():
    e = CaveEnv(DMG)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (41,) and obs[-2] == 0 and obs[-1] == 1.0

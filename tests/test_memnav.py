"""memory-v3: exit-memory model, its trainer, the memory scenarios, the aux labels, and the exact speed-ups."""

from dataclasses import replace

import numpy as np
import pytest
import torch as th

from submarine.memory.exitmem import VARIANTS, ExitMemoryConfig, ExitMemoryNet
from submarine.navigator import AUX_DIM, N_SLOTS, OBS, OBS_DIM, POS_SCALE, SLOT_DIM

F = OBS_DIM + AUX_DIM


def make_obs(n, x=0.0, y=0.0, exits=((1.0, 0.0),), junction=False, dead=False, prev=None, labels=None):
    """Synthetic navigator observations: exits as nav-frame unit directions (front slots), turn back always."""
    o = th.zeros(n, F)
    s = o[:, OBS["slots"]].view(n, N_SLOTS, SLOT_DIM)
    for k, (cx, cy) in enumerate(exits):
        s[:, k, 0], s[:, k, 1], s[:, k, 3], s[:, k, 4], s[:, k, 5] = 1.0, 1.0, cx, cy, 0.5
    s[:, N_SLOTS - 1, 0], s[:, N_SLOTS - 1, 3] = 1.0, -1.0       # (s is a view: o is written in place)
    o[:, OBS["x"]], o[:, OBS["y"]] = x / POS_SCALE, y / POS_SCALE
    o[:, OBS["cos_psi"]] = 1.0
    o[:, OBS["junction"]], o[:, OBS["dead_end"]] = float(junction), float(dead)
    if prev is not None:
        (px, py), (hx, hy) = prev
        o[:, OBS["prev_junction"]] = 1.0
        o[:, OBS["prev_x"]], o[:, OBS["prev_y"]] = px / POS_SCALE, py / POS_SCALE
        o[:, OBS["prev_cos"]], o[:, OBS["prev_sin"]] = hx, hy
    if labels is not None:
        o[:, OBS_DIM:] = labels
    return o


@pytest.mark.parametrize("variant", VARIANTS)
def test_model_ignores_the_aux_labels(variant):
    """The labels are training targets riding in the observation; they must never influence the policy."""
    th.manual_seed(0)
    net = ExitMemoryNet(ExitMemoryConfig(variant=variant))
    a = make_obs(3, junction=True, exits=((1, 0), (0, 1)), prev=((0, 0), (1, 0)), labels=th.zeros(AUX_DIM))
    b = a.clone()
    b[:, OBS_DIM:] = 1.0
    s0, st = net.initial_state(3), th.zeros(3)
    la, va, _, sa = net.step(a, s0, st)
    lb, vb, _, sb = net.step(b, s0, st)
    assert th.equal(la, lb) and th.equal(va, vb) and th.equal(sa, sb)


@pytest.mark.parametrize("variant", VARIANTS)
def test_unroll_equals_step_by_step_and_resets(variant):
    th.manual_seed(0)
    net = ExitMemoryNet(ExitMemoryConfig(variant=variant))
    T, N = 10, 3
    obs = th.randn(T, N, F) * 0.3
    obs[:, :, OBS["prev_junction"]] = (th.rand(T, N) > 0.5).float()
    obs[:, :, OBS["dead_end"]] = (th.rand(T, N) > 0.7).float()
    starts = th.zeros(T, N)
    starts[0] = 1.0
    starts[6, 1] = 1.0
    lg, v, aux, _ = net.unroll(obs, net.initial_state(N), starts)
    s, outs = net.initial_state(N), []
    for t in range(T):
        l1, _, _, s = net.step(obs[t], s, starts[t])
        outs.append(l1)
    assert th.allclose(lg, th.stack(outs), atol=1e-5)
    fresh, _, _, _ = net.unroll(obs[6:, 1:2], net.initial_state(1), th.cat([th.ones(1, 1), th.zeros(T - 7, 1)]))
    assert th.allclose(lg[6:, 1], fresh[:, 0], atol=1e-5)
    assert set(aux) == {"taken", "dead_known", "loop"} and aux["taken"].shape == (T, N, N_SLOTS)


def test_store_merges_revisits_and_evicts_the_oldest():
    net = ExitMemoryNet(ExitMemoryConfig(variant="graph", n_entries=3))
    s, st = net.initial_state(1), th.zeros(1)
    for k in range(6):                                    # stuck at one dead end: 6 writes, 10 cm drift each
        _, _, _, s = net.step(make_obs(1, x=10.0 + 0.1 * k, dead=True), s, st)
    e = net.entries(s)
    assert len(e) == 1 and e[0]["dead_end"] and abs(e[0]["pos"][0] - 10.5) < 1e-4
    _, _, _, s = net.step(make_obs(1, x=12.0, prev=((0, 0), (1, 0))), s, st)        # departure east from (0, 0)
    _, _, _, s = net.step(make_obs(1, x=12.0, prev=((0, 0), (0, 1))), s, st)        # same place, other way
    assert len(net.entries(s)) == 3                       # different directions: separate departures
    _, _, _, s = net.step(make_obs(1, x=30.0, dead=True), s, st)                    # full: oldest replaced
    assert sorted(round(e["pos"][0], 1) for e in net.entries(s)) == [0.0, 0.0, 30.0]


def test_rule_reads_point_at_the_right_exit():
    """Graph variant: a dead end stored 6 m east of a junction is reported on the east exit only; a departure
    north from here is reported as taken on the north exit only."""
    net = ExitMemoryNet(ExitMemoryConfig(variant="graph"))
    s, st = net.initial_state(1), th.zeros(1)
    _, _, _, s = net.step(make_obs(1, x=6.0, dead=True), s, st)
    _, _, _, s = net.step(make_obs(1, x=3.0, prev=((0, 0), (0, 1))), s, st)
    net.step(make_obs(1, x=0.0, junction=True, exits=((1, 0), (0, 1))), s, st)
    strengths = net.exit_strengths()[0]                   # (5, 2): taken, dead ahead
    assert strengths[0, 1] > 0.5 and strengths[1, 1] < 0.1        # east: dead end ahead; north: none
    assert strengths[1, 0] > 0.5 and strengths[0, 0] < 0.1        # north: taken before; east: not


def test_invalid_exits_are_masked():
    net = ExitMemoryNet(ExitMemoryConfig(variant="exit"))
    logits, _, _, _ = net.step(make_obs(2, exits=((1, 0),)), net.initial_state(2), th.ones(2))
    p = th.softmax(logits, -1)
    assert float(p[:, 1:N_SLOTS - 1].sum()) < 1e-6        # empty slots 1-3 never chosen


def test_ppo_update_and_warm_start_run(tmp_path):
    """Trainer plumbing on a fake vector env: finite losses, the gate stays a probability, warm start runs."""
    from submarine.memory.trainer import PPOConfig, RecurrentPPO, warm_start

    class FakeVec:
        num_envs = 4

        def __init__(self):
            self.rng = np.random.default_rng(0)
            self.t = 0

        def _obs(self):
            o = make_obs(4, exits=((1, 0), (0, 1)), junction=True).numpy()
            o[:, OBS_DIM:] = self.rng.random((4, AUX_DIM)) > 0.7
            return o

        def reset(self):
            return self._obs()

        def step(self, a):
            self.t += 1
            d = np.full(4, self.t % 7 == 0)
            return self._obs(), self.rng.normal(size=4), d, [{"is_success": False} for _ in range(4)]

    net = ExitMemoryNet(ExitMemoryConfig(variant="exit"))
    algo = RecurrentPPO(net, FakeVec(), PPOConfig(n_steps=8, epochs=2, minibatch_envs=2))
    algo.total_timesteps = 64
    st = algo.update(algo.collect())
    assert all(np.isfinite(v) for v in st.values()) and 0 < st["train/memory_gate"] < 1
    n = 300
    demos = {"obs": np.concatenate([FakeVec()._obs() for _ in range(n // 4)]).astype(np.float32),
             "label": np.zeros(n, np.int64), "reward": np.zeros(n, np.float32),
             "start": np.arange(n) % 25 == 0}
    res = warm_start(net, demos, epochs=2, lanes=4, T=16, log=lambda *_: None)
    assert res["agree"] > 0.5                             # label 0 is always valid: easy to clone


# ---------------------------------------------------------------- environment and scenarios

@pytest.mark.parametrize("layout", ["decoy", "loops"])
def test_memory_scenarios_generate_solvable_caves(layout):
    from submarine.world import generate_cave

    for seed in range(5):
        c = generate_cave(seed, 0.5, layout=layout)
        assert np.isfinite(c.shortest_path_length)


def test_aux_labels_ride_in_the_observation_and_mark_revisits():
    from submarine.navigator import AUX, NavigatorEnv, OracleNavigator, slot_valid

    env = NavigatorEnv({"env": {"layout": "maze", "difficulty": 0.0, "max_steps": 1200}, "aux_labels": True})
    obs, _ = env.reset(seed=3, options={"cave_seed": 3, "difficulty": 0.0})
    assert obs.shape == (OBS_DIM + AUX_DIM,) and env.observation_space.shape == obs.shape
    teacher = OracleNavigator()
    teacher.attach(env)
    seen = {"taken": 0, "dead": 0}
    rng = np.random.default_rng(0)
    for _ in range(150):
        valid = np.flatnonzero(slot_valid(obs))
        a = int(rng.choice(valid)) if rng.random() < 0.5 and valid.size else teacher.act(obs)
        obs, r, te, tr, info = env.step(a)
        lab = obs[OBS_DIM:]
        assert set(np.unique(lab)) <= {0.0, 1.0}
        seen["taken"] += int(lab[AUX["taken"]].any())
        seen["dead"] += int(lab[AUX["dead_known"]].any())
        if te or tr:
            break
    assert seen["taken"] > 0                              # wandering a maze revisits junctions


def _paired_dives(extra: dict, layout="maze", steps=120, seed=5):
    """The same dive (same cave, seeds, actions) in a default env and one with `extra` reward settings."""
    from submarine.navigator import NavigatorEnv, OracleNavigator, slot_valid

    base = {"env": {"layout": layout, "difficulty": 0.0, "max_steps": 1500}, "aux_labels": True}
    envs = [NavigatorEnv(base), NavigatorEnv({**base, **extra})]
    rs = [e.reset(seed=seed, options={"cave_seed": seed, "difficulty": 0.0}) for e in envs]
    obs = [r[0] for r in rs]
    teacher = OracleNavigator()
    teacher.attach(envs[0])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        valid = np.flatnonzero(slot_valid(obs[0]))
        a = int(rng.choice(valid)) if rng.random() < 0.6 and valid.size else teacher.act(obs[0])
        res = [e.step(a) for e in envs]
        np.testing.assert_array_equal(res[0][0][OBS_DIM:], res[1][0][OBS_DIM:])   # same dive, same labels
        out.append((obs[1], a, res[0][1], res[1][1], res[1][4]))
        obs = [r[0] for r in res]
        if res[0][2] or res[0][3]:
            break
    return out, rs[1][1]["geodesic"]


def test_record_progress_pays_only_new_best_geodesic():
    dive, g_start = _paired_dives({"progress": "record"})
    geos = [d[4]["geodesic"] for d in dive]
    diff = sum(d[3] - d[2] for d in dive)
    # dense pays (start - end), record pays (start - best): the difference is k_prog * (best - end) <= 0
    assert diff == pytest.approx(min(min(geos), g_start) - geos[-1], abs=1e-3)   # k_prog = 1
    assert diff <= 1e-6


def test_repeat_penalties_follow_the_true_labels():
    from submarine.navigator import AUX, NavigatorEnv

    dive, _ = _paired_dives({"repeat_dead_penalty": 3.0, "repeat_taken_penalty": 1.0}, steps=200)
    total = 0.0
    for obs, a, r0, r1, info in dive:
        lab = obs[OBS_DIM:]
        taken, dead = lab[AUX["taken"]], lab[AUX["dead_known"]]
        valid = obs[OBS["slots"]].reshape(N_SLOTS, SLOT_DIM)[:N_SLOTS - 1, 0] > 0.5
        k = NavigatorEnv.resolve(a, list(valid) + [True])
        untried = bool(np.any(valid & (taken[:N_SLOTS - 1] < 0.5)))
        front = k < N_SLOTS - 1                                  # turning back is never penalised
        p = 3.0 if front and dead[k] > 0.5 else (1.0 if front and taken[k] > 0.5 and untried else 0.0)
        assert r1 == pytest.approx(r0 - p, abs=1e-4)
        total += p
    assert total > 0                                             # the wandering dive did repeat itself


def test_dead_end_revisit_and_u_turn_penalties_add_up():
    """A dive that turns back often: the reward gap equals the counted revisit and U-turn-loop penalties."""
    from submarine.navigator import NavigatorEnv, slot_valid

    base = {"env": {"layout": "maze", "difficulty": 0.0, "max_steps": 1500}, "aux_labels": True}
    envs = [NavigatorEnv(base), NavigatorEnv({**base, "dead_end_revisit_penalty": 1.5, "dither_penalty": 0.3})]
    obs = [e.reset(seed=4, options={"cave_seed": 4, "difficulty": 0.0})[0] for e in envs]
    rng = np.random.default_rng(2)
    gap, n_back_at_dead_end = 0.0, 0
    for _ in range(250):
        valid = np.flatnonzero(slot_valid(obs[0]))
        a = N_SLOTS - 1 if rng.random() < 0.4 else int(rng.choice(valid))
        dead_end = not obs[0][OBS["slots"]].reshape(N_SLOTS, SLOT_DIM)[:N_SLOTS - 1, 0].any()
        before = dict(envs[1]._ep)
        res = [e.step(a) for e in envs]
        ep = res[1][4]
        d_rev, d_dith = ep["dead_end_revisits"] - before["dead_end_revisits"], ep["dithers"] - before["dithers"]
        if dead_end:
            assert d_dith == 0                                   # the forced turn at a dead end is free
            n_back_at_dead_end += 1
        assert res[1][1] == pytest.approx(res[0][1] - 1.5 * d_rev - 0.3 * d_dith, abs=1e-4)
        gap += res[0][1] - res[1][1]
        obs = [r[0] for r in res]
        if res[0][2] or res[0][3]:
            break
    assert envs[1]._ep["dithers"] > 0 and gap > 0


def test_exploration_bonus_only_without_the_goal_cue():
    with_cue, _ = _paired_dives({"explore_bonus": 0.5}, steps=30)
    assert all(d[3] == pytest.approx(d[2]) for d in with_cue)   # cue shown (default 1.0): no bonus
    blind, _ = _paired_dives({"explore_bonus": 0.5, "goal_cue_prob": 0.0}, steps=30)
    bonus = blind[-1][4]["explore"]
    assert bonus > 0 and (bonus / 0.5) == pytest.approx(round(bonus / 0.5))


def test_taken_label_waits_until_the_sub_comes_back():
    from submarine.navigator import AUX, NavigatorEnv

    env = NavigatorEnv({"env": {"layout": "maze", "difficulty": 0.0, "max_steps": 1500}, "aux_labels": True})
    obs, info = env.reset(seed=3, options={"cave_seed": 3, "difficulty": 0.0})
    rng = np.random.default_rng(1)
    for _ in range(150):
        if obs[OBS_DIM:][AUX["taken"]].any():                    # only for a departure >= 6 m of travel ago
            pl = env._info["path_length"]
            assert any(pl - d["pl"] >= env.cfg.aux_loop_path for d in env._departures)
        obs, r, te, tr, info = env.step(int(rng.integers(N_SLOTS)))
        if te or tr:
            break


def test_goal_cue_can_be_withheld():
    from submarine.navigator import NavigatorEnv

    env = NavigatorEnv({"env": {"layout": "tunnel", "difficulty": 0.0}, "goal_cue_prob": 0.0})
    obs, info = env.reset(seed=0)
    assert not info["goal_cue"] and not obs[OBS["goal"]].any()


# ---------------------------------------------------------------- exact speed-ups

def test_scan_cache_matches_uncached_controller():
    from submarine.control.controller import ControllerConfig, HandController
    from submarine.sim import Sonar, SonarConfig, VehicleConfig

    sonar = Sonar(SonarConfig())
    rng = np.random.default_rng(0)
    a = HandController(ControllerConfig(), VehicleConfig(), sonar.angles, 10.0)
    for _ in range(40):
        ranges = rng.uniform(0.5, 10.0, len(sonar.angles))
        for _ in range(4):                                # same array held for 4 steps, like a 0.4 s ping
            u, r, b = rng.uniform(-0.3, 1.4), rng.uniform(-1, 1), rng.uniform(-3, 3)
            fresh = HandController(ControllerConfig(), VehicleConfig(), sonar.angles, 10.0)
            got, want = a.act(ranges, u, r, b, 1.2), fresh.act(ranges.copy(), u, r, b, 1.2)
            assert np.array_equal(got[0], want[0]) and got[1] == want[1]
        ranges[3] = 0.7                                   # mutated in place: the cache must notice
        fresh = HandController(ControllerConfig(), VehicleConfig(), sonar.angles, 10.0)
        assert np.array_equal(a.act(ranges, 0.5, 0.0, 0.0, 1.2)[0], fresh.act(ranges.copy(), 0.5, 0.0, 0.0, 1.2)[0])


def test_switching_off_junction_flags_does_not_change_the_dive():
    """The navigator switches CaveEnv's per-step opening count off; the driven trajectory must be identical."""
    from submarine.control import DriverPolicy
    from submarine.env import CaveEnv, EnvConfig

    trails = []
    for flags in (True, False):
        cfg = EnvConfig.from_dict({"layout": "maze", "difficulty": 0.3, "max_steps": 300,
                                   "navigation": {"enabled": True, "junction_flags": flags}})
        e = CaveEnv(cfg)
        obs, _ = e.reset(seed=5, options={"cave_seed": 5, "difficulty": 0.3})
        pol = DriverPolicy()
        pol.reset(e)
        xs = []
        for _ in range(300):
            obs, r, te, tr, info = e.step(pol.act(obs))
            xs.append((e.sim.state.x, e.sim.state.y))
            if te or tr:
                break
        trails.append(np.array(xs))
    assert trails[0].shape == trails[1].shape and np.array_equal(trails[0], trails[1])

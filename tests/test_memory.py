"""Navigator memory modules (retention, junction memory) and their RecurrentPPO policies."""

import numpy as np
import pytest
import torch as th

from submarine.memory.modules import JunctionMemory, RetentionMemory
from submarine.navigator import OBS, OBS_DIM

IDX = {k: OBS[k] for k in ("x", "y", "prev_x", "prev_y", "prev_junction", "dead_end")}


def zeros(m, n):
    return th.zeros(1, n, m.hidden_size), th.zeros(1, n, m.hidden_size)


@pytest.mark.parametrize("make", [lambda: RetentionMemory(OBS_DIM), lambda: JunctionMemory(OBS_DIM, IDX)])
def test_sequence_equals_step_by_step_and_resets(make):
    th.manual_seed(0)
    m = make()
    T, N = 12, 3
    x = th.randn(T, N, OBS_DIM)
    x[:, :, IDX["prev_junction"]] = (th.rand(T, N) > 0.6).float()
    x[:, :, IDX["dead_end"]] = (th.rand(T, N) > 0.8).float()
    starts = th.zeros(T, N)
    starts[5, 1] = 1.0                                   # env 1 starts a new episode at t = 5
    seq, (h1, c1) = m(x, zeros(m, N), starts)
    st, outs = zeros(m, N), []
    for t in range(T):
        o, st = m(x[t:t + 1], st, starts[t:t + 1])
        outs.append(o[0])
    assert th.allclose(seq, th.stack(outs), atol=1e-5) and th.allclose(c1, st[1], atol=1e-5)
    # after the reset, env 1 behaves like a fresh start on the remaining inputs
    fresh, _ = m(x[5:, 1:2], zeros(m, 1), th.zeros(T - 5, 1))
    assert th.allclose(seq[5:, 1], fresh[:, 0], atol=1e-5)


def test_junction_slots_write_read_and_replace():
    th.manual_seed(0)
    m = JunctionMemory(OBS_DIM, IDX, n_slots=3)
    st = zeros(m, 1)

    def step(x_pos, prev=None, dead=False):
        nonlocal st
        x = th.zeros(1, 1, OBS_DIM)
        x[0, 0, IDX["x"]], x[0, 0, IDX["y"]] = x_pos
        if prev is not None:
            x[0, 0, IDX["prev_junction"]] = 1.0
            x[0, 0, IDX["prev_x"]], x[0, 0, IDX["prev_y"]] = prev
        x[0, 0, IDX["dead_end"]] = float(dead)
        _, st = m(x, st)

    def slots():
        K, D, M = m.n_slots, m.slot_dim, m.memo_dim
        s = st[1][0, 0, m.state_size:m.state_size + K * D].reshape(K, D)
        return s[:, M:M + 3]                              # x, y, used

    step((0.1, 0.0), prev=(0.0, 0.0))                     # left junction A at (0, 0)
    step((0.5, 0.0), dead=True)                           # dead end at (0.5, 0)
    assert sorted(slots()[:, 2].tolist()) == [0.0, 1.0, 1.0]
    step((0.0, 0.02))                                     # back near A: attention should be on A's slot
    att = m.last_attention[0]
    pos = slots()[:, :2]
    assert int(att.argmax()) == int(((pos - th.tensor([0.0, 0.0])) ** 2).sum(1).argmin())
    step((1.0, 0.0), prev=(0.9, 0.0))                     # third slot
    step((2.0, 0.0), prev=(1.9, 0.0))                     # full: replaces the OLDEST (A at (0, 0))
    xs = sorted(round(v, 2) for v in slots()[:, 0].tolist())
    assert xs == [0.5, 0.9, 1.9]


def test_gradient_reaches_memo_through_reads():
    th.manual_seed(0)
    m = JunctionMemory(OBS_DIM, IDX)
    x = th.zeros(3, 1, OBS_DIM)
    x[0, 0, IDX["prev_junction"]] = 1.0                   # write at t = 0, read at t = 2
    out, _ = m(x, zeros(m, 1))
    out[2].sum().backward()
    assert m.memo[0].weight.grad is not None and m.memo[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("policy", ["RetentionNavPolicy", "JunctionNavPolicy"])
def test_recurrent_ppo_trains_saves_and_loads(tmp_path, policy):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from submarine.memory.policies import POLICIES
    from submarine.navigator import NavigatorEnv

    cfg = {"env": {"difficulty": 0.0, "max_steps": 200}}
    venv = DummyVecEnv([lambda: NavigatorEnv(cfg)] * 2)
    kw = {"memory_kwargs": {"obs_index": IDX}} if policy == "JunctionNavPolicy" else {}
    model = RecurrentPPO(POLICIES[policy], venv, n_steps=16, batch_size=16, n_epochs=1, seed=0, device="cpu",
                         policy_kwargs=kw)
    model.learn(64)
    model.save(tmp_path / "m.zip")
    loaded = RecurrentPPO.load(tmp_path / "m.zip", device="cpu", custom_objects={"policy_class": POLICIES[policy]})
    obs = venv.reset()
    a1, s1 = model.predict(obs, deterministic=True, episode_start=np.ones(2, bool))
    a2, s2 = loaded.predict(obs, deterministic=True, episode_start=np.ones(2, bool))
    assert np.array_equal(a1, a2) and np.allclose(s1[1], s2[1])

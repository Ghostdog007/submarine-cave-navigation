"""Recurrent PPO with auxiliary memory losses, for the exit-memory navigators (memory-v3).

Why not sb3-contrib's RecurrentPPO: it has no hook for auxiliary losses, and its policies flatten the memory into
one vector the policy must decode. This trainer is small and specific:
  * rollouts of T decisions from N envs; each env's sequence is trained from the recurrent state it had at the
    start of the rollout (stored state, R2D2) with episode starts resetting inside the sequence;
  * loss = PPO clip + value + entropy + aux_w * (loop closure + per-exit taken + per-exit known dead end), aux_w
    annealed from `aux_start` to `aux_end` (Nav A3C / UNREAL style auxiliary tasks);
  * invalid exits are masked out of the policy (logits = -1e8), so every action is a real exit or a turn-back;
  * early stop of an update when the approximate KL exceeds `target_kl`;
  * warm start: behaviour cloning of the map-aware teacher over WHOLE episodes (lanes of consecutive episodes,
    windows of T with the state carried across windows), plus value and aux pretraining;
  * logs the same TensorBoard tags as SB3 (rollout/ep_rew_mean, rollout/success_rate, time/fps, train/approx_kl,
    train/explained_variance, ...) plus aux/*, so scripts.watch_training works unchanged.

Memory budget (no overflow): rollout tensors are N x T x 72 floats (32 x 64 x 72 = 0.6 MB); the recurrent state is
<= 800 floats per env; workers stay torch-free (they only run the simulator).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch as th
from torch import nn

from submarine.navigator.nav_env import AUX_DIM, OBS_DIM

from .exitmem import ExitMemoryNet, aux_targets


@dataclass
class PPOConfig:
    n_steps: int = 64             # decisions per env per rollout (= training sequence length)
    epochs: int = 4
    minibatch_envs: int = 8       # sequences per minibatch
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent: float = 0.01
    vf: float = 0.5
    aux_start: float = 0.5
    aux_end: float = 0.1
    max_grad: float = 0.5
    target_kl: float = 0.03


def masked_bce(logit, target, mask):
    loss = nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def aux_loss(aux: dict, obs: th.Tensor, valid: th.Tensor, mask_t: th.Tensor | None = None):
    """Aux BCE losses and accuracies. obs (..., >= OBS_DIM + AUX_DIM); valid (..., 5) bool; mask_t (...) step mask."""
    taken, dead, loop = aux_targets(obs)
    m = valid.to(obs.dtype)
    if mask_t is not None:
        m = m * mask_t[..., None]
    lt = masked_bce(aux["taken"], taken, m)
    ld = masked_bce(aux["dead_known"], dead, m)
    ml = th.ones_like(loop) if mask_t is None else mask_t
    ll = masked_bce(aux["loop"], loop, ml)
    with th.no_grad():
        acc = lambda lg, y, mm: float((((lg > 0).to(y.dtype) == y).to(y.dtype) * mm).sum() / mm.sum().clamp(min=1))
        pos = lambda lg, y, mm: float((((lg > 0) & (y > 0.5)).to(y.dtype) * mm).sum() /
                                      ((y > 0.5).to(y.dtype) * mm).sum().clamp(min=1))
        stats = {"aux/taken_acc": acc(aux["taken"], taken, m), "aux/dead_known_recall": pos(aux["dead_known"], dead, m),
                 "aux/taken_recall": pos(aux["taken"], taken, m), "aux/loop_acc": acc(aux["loop"], loop, ml),
                 "aux/dead_known_rate": float((dead * m).sum() / m.sum().clamp(min=1))}
    return lt + ld + ll, stats


def valid_mask(obs: th.Tensor) -> th.Tensor:
    from submarine.navigator.nav_env import N_SLOTS, OBS, SLOT_DIM

    s = obs[..., OBS["slots"]].reshape(*obs.shape[:-1], N_SLOTS, SLOT_DIM)
    v = s[..., 0] > 0.5
    return th.cat([v[..., :N_SLOTS - 1], th.ones_like(v[..., :1])], -1)


class RecurrentPPO:
    def __init__(self, net: ExitMemoryNet, venv, cfg: PPOConfig, writer=None, log=print):
        self.net, self.venv, self.cfg = net, venv, cfg
        self.opt = th.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
        self.writer, self.log = writer, log
        self.n = venv.num_envs
        self.num_timesteps = 0
        self.total_timesteps = 1
        self._obs = None
        self.ep_info: list[dict] = []
        self._ret = np.zeros(self.n)
        self._len = np.zeros(self.n, int)

    def set_env(self, venv):
        self.venv, self._obs = venv, None

    def _reset(self):
        o = self.venv.reset()
        self._obs = th.as_tensor(np.asarray(o), dtype=th.float32)
        if self._obs.shape[1] < OBS_DIM + AUX_DIM:
            raise ValueError("the training env must append AUX labels (aux_labels: true)")
        self._state = self.net.initial_state(self.n)
        self._start = th.ones(self.n)
        self._ret[:] = 0
        self._len[:] = 0

    # ------------------------------------------------------------------ rollout
    @th.no_grad()
    def collect(self):
        c, T, N = self.cfg, self.cfg.n_steps, self.n
        if self._obs is None:
            self._reset()
        state0 = self._state.clone()
        obs_b = th.zeros(T, N, self._obs.shape[1])
        act_b, logp_b, val_b, rew_b = th.zeros(T, N, dtype=th.long), th.zeros(T, N), th.zeros(T, N), th.zeros(T, N)
        start_b, done_b, trunc_b = th.zeros(T, N), th.zeros(T, N), th.zeros(T, N)
        for t in range(T):
            obs_b[t], start_b[t] = self._obs, self._start
            logits, v, _, self._state = self.net.step(self._obs, self._state, self._start)
            dist = th.distributions.Categorical(logits=logits)
            a = dist.sample()
            act_b[t], logp_b[t], val_b[t] = a, dist.log_prob(a), v
            o, r, d, infos = self.venv.step(a.numpy())
            rew_b[t] = th.as_tensor(r, dtype=th.float32)
            done_b[t] = th.as_tensor(d, dtype=th.float32)
            self._ret += r
            self._len += 1
            for i in np.flatnonzero(d):
                info = infos[i]
                trunc_b[t, i] = float(info.get("TimeLimit.truncated", False))
                self.ep_info.append({"r": float(self._ret[i]), "l": int(self._len[i]),
                                     "s": float(info.get("is_success", False)),
                                     "cue": float(info.get("goal_cue", True))})
                self._ret[i], self._len[i] = 0.0, 0
            self.ep_info = self.ep_info[-100:]
            self._obs = th.as_tensor(np.asarray(o), dtype=th.float32)
            self._start = done_b[t].clone()
        _, last_v, _, _ = self.net.step(self._obs, self._state.clone(), self._start)
        # GAE; a time-limit truncation bootstraps from the value of the last decision before it (approximation:
        # the terminal observation's value would need the recurrent state at that point)
        adv = th.zeros(T, N)
        gae = th.zeros(N)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_b[t + 1]
            boot = trunc_b[t] * val_b[t]
            nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + c.gamma * (nv * nonterm + boot) - val_b[t]
            gae = delta + c.gamma * c.lam * nonterm * gae
            adv[t] = gae
        self.num_timesteps += T * N
        return {"obs": obs_b, "act": act_b, "logp": logp_b, "val": val_b, "adv": adv, "ret": adv + val_b,
                "start": start_b, "state0": state0}

    # ------------------------------------------------------------------ update
    def update(self, b):
        c, N = self.cfg, self.n
        frac = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
        aux_w = c.aux_start + (c.aux_end - c.aux_start) * frac
        adv = b["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        valid = valid_mask(b["obs"])
        stats = {k: [] for k in ("pg", "vf", "ent", "aux", "kl", "clipfrac")}
        aux_stats: dict[str, list] = {}
        stop = False
        for _ in range(c.epochs):
            for idx in th.randperm(N).split(c.minibatch_envs):
                logits, v, aux, _ = self.net.unroll(b["obs"][:, idx], b["state0"][idx], b["start"][:, idx])
                dist = th.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b["act"][:, idx])
                ratio = th.exp(logp - b["logp"][:, idx])
                a = adv[:, idx]
                pg = -th.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                vf = ((v - b["ret"][:, idx]) ** 2).mean()
                ent = dist.entropy().mean()
                la, ast = aux_loss(aux, b["obs"][:, idx], valid[:, idx])
                loss = pg + c.vf * vf - c.ent * ent + aux_w * la
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad)
                self.opt.step()
                with th.no_grad():
                    lr_ = logp - b["logp"][:, idx]
                    kl = float(((th.exp(lr_) - 1) - lr_).mean())
                for k, val in (("pg", pg), ("vf", vf), ("ent", ent), ("aux", la), ("kl", kl),
                               ("clipfrac", ((ratio - 1).abs() > c.clip).float().mean())):
                    stats[k].append(float(val))
                for k, val in ast.items():
                    aux_stats.setdefault(k, []).append(val)
                if kl > c.target_kl:
                    stop = True
                    break
            if stop:
                break
        yv, yp = b["ret"].flatten(), b["val"].flatten()
        ev = float(1 - th.var(yv - yp) / (th.var(yv) + 1e-8))
        out = {f"train/{k}": float(np.mean(v)) for k, v in stats.items()}
        out.update({k: float(np.mean(v)) for k, v in aux_stats.items()})
        out.update({"train/explained_variance": ev, "train/aux_weight": aux_w, "train/approx_kl": out["train/kl"],
                    "train/entropy_loss": -out["train/ent"], "train/early_stop": float(stop)})
        if self.net.gate is not None:
            out["train/memory_gate"] = float(th.sigmoid(self.net.gate))
        return out

    def learn(self, steps: int, tb_step0: int = 0):
        t0, n0 = time.perf_counter(), self.num_timesteps
        while self.num_timesteps - n0 < steps:
            b = self.collect()
            st = self.update(b)
            fps = (self.num_timesteps - n0) / (time.perf_counter() - t0)
            if self.ep_info:
                st["rollout/ep_rew_mean"] = float(np.mean([e["r"] for e in self.ep_info]))
                st["rollout/ep_len_mean"] = float(np.mean([e["l"] for e in self.ep_info]))
                st["rollout/success_rate"] = float(np.mean([e["s"] for e in self.ep_info]))
                blind = [e["s"] for e in self.ep_info if e["cue"] < 0.5]
                if blind:
                    st["rollout/success_no_cue"] = float(np.mean(blind))
            st["time/fps"] = fps
            if self.writer is not None:
                for k, v in st.items():
                    self.writer.add_scalar(k, v, self.num_timesteps)
                self.writer.flush()
        return self


# ---------------------------------------------------------------------------- warm start
def warm_start(net: ExitMemoryNet, demos: dict, epochs: int = 6, lanes: int = 32, T: int = 64, lr: float = 1e-3,
               vf_coef: float = 0.05, aux_w: float = 0.5, gamma: float = 0.99, log=print, seed: int = 0):
    """Behaviour cloning over whole episodes: episodes are laid end to end in `lanes` streams; each stream is
    processed in windows of T decisions with the recurrent state carried (detached) from window to window, so
    the memory sees complete dives exactly as at run time. 10% of episodes are held out for the agreement score."""
    obs, label, rew, start = (np.asarray(demos[k]) for k in ("obs", "label", "reward", "start"))
    if obs.shape[1] < OBS_DIM + AUX_DIM:
        raise ValueError("demonstrations need the AUX labels (collect them with aux_labels: true)")
    ret = np.zeros_like(rew)
    g = 0.0
    for i in range(len(rew) - 1, -1, -1):
        g = rew[i] + gamma * g
        ret[i] = g
        if start[i]:
            g = 0.0
    bounds = np.flatnonzero(start).tolist() + [len(obs)]
    episodes = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(episodes))
    n_val = max(1, len(episodes) // 10)
    val_eps, train_eps = [episodes[i] for i in perm[:n_val]], [episodes[i] for i in perm[n_val:]]
    opt = th.optim.Adam(net.parameters(), lr=lr)

    def streams(eps, n_lanes):
        """Pack episodes into lanes: arrays (L, n_lanes, ...) padded with mask 0 at the end."""
        lanes_ = [[] for _ in range(n_lanes)]
        lens = np.zeros(n_lanes, int)
        for a, b in eps:                                          # greedy: shortest lane next
            i = int(np.argmin(lens))
            lanes_[i].append((a, b))
            lens[i] += b - a
        L = int(lens.max())
        idx = np.zeros((L, n_lanes), int)
        mask = np.zeros((L, n_lanes), np.float32)
        st = np.zeros((L, n_lanes), np.float32)
        for j, lane in enumerate(lanes_):
            t = 0
            for a, b in lane:
                idx[t:t + b - a, j] = np.arange(a, b)
                mask[t:t + b - a, j] = 1.0
                st[t, j] = 1.0
                t += b - a
            st[t:, j] = 1.0                                       # padding: keep resetting
        return idx, mask, st

    def run(eps, n_lanes, train: bool):
        idx, mask, st = streams(eps, n_lanes)
        state = net.initial_state(n_lanes)
        tot = {"nll": 0.0, "agree": 0.0, "n": 0.0, "aux": 0.0}
        for w in range(0, idx.shape[0], T):
            sl = slice(w, w + T)
            o = th.as_tensor(obs[idx[sl]])
            m = th.as_tensor(mask[sl])
            s = th.as_tensor(st[sl])
            y = th.as_tensor(label[idx[sl]])
            r = th.as_tensor(ret[idx[sl]])
            with th.set_grad_enabled(train):
                logits, v, aux, state = net.unroll(o, state, s)
                nll = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                                  reduction="none").reshape(m.shape)
                nll = (nll * m).sum() / m.sum().clamp(min=1)
                vl = (((v - r) ** 2) * m).sum() / m.sum().clamp(min=1)
                la, _ = aux_loss(aux, o, valid_mask(o), m)
                loss = nll + vf_coef * vl + aux_w * la
                if train:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
            state = state.detach()
            with th.no_grad():
                tot["agree"] += float(((logits.argmax(-1) == y).float() * m).sum())
            tot["nll"] += float(nll) * float(m.sum())
            tot["aux"] += float(la) * float(m.sum())
            tot["n"] += float(m.sum())
        return {k: v / max(tot["n"], 1) for k, v in tot.items() if k != "n"}

    for ep in range(epochs):
        rng.shuffle(train_eps)
        tr = run(train_eps, lanes, True)
        net.eval()
        va = run(val_eps, min(lanes, len(val_eps)), False)
        net.train()
        log(f"warm start epoch {ep + 1}: train NLL {tr['nll']:.3f} aux {tr['aux']:.3f} | held-out NLL "
            f"{va['nll']:.3f} aux {va['aux']:.3f} teacher agreement {100 * va['agree']:.0f}%")
    return va

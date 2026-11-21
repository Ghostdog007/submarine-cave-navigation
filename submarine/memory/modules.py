"""Route memory for the navigator, run once per decision (not at 10 Hz).

Both modules have the call signature of a one-layer nn.LSTM plus episode starts:
    out, (h, c) = module(x (T, N, F), (h, c) each (1, N, H), starts (T, N))
so sb3-contrib's RecurrentPPO stores their state in its LSTM state buffers (see policies.py). The state is
reset where `starts` is 1.

RetentionMemory: multi-scale retention (decaying linear attention, no LSTM). Per head h with decay g_h:
    S_t = g_h * S_{t-1} + k_t v_t^T          (d_k x d_v outer-product memory)
    o_t = q_t^T S_t
Heads with g = 0.5 / 0.8 / 0.95 / 0.99 per decision remember ~2 / 5 / 20 / 100 decisions: "what happened
recently" (e.g. I just hit a dead end) down to "what happened this dive".

JunctionMemory: RetentionMemory plus K position-keyed slots (episodic / spatial memory).
  write  when the previous decision was made at a junction: slot = (its dead-reckoned position, a learned memo
         of the observation, which includes the heading chosen there); when a dead end is detected: slot =
         (current position, learned memo). Free slots first, then the oldest is overwritten.
  read   attention over stored slots: logit = q . memo / sqrt(d) - alpha * |position - slot position|^2, so the
         slots of *this* place dominate ("at this junction I went left, and that was a dead end").
Writes are gated by the observation flags, not learned; the memo content is learned end to end.
"""

from __future__ import annotations

import math

import torch as th
from torch import nn


class RetentionMemory(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, d_model: int = 64, n_heads: int = 4,
                 d_k: int = 8, d_v: int = 8, gammas=(0.5, 0.8, 0.95, 0.99)):
        super().__init__()
        if len(gammas) != n_heads:
            raise ValueError("one decay per head")
        self.input_size, self.hidden_size = input_size, hidden_size
        self.num_layers = 1                                         # RecurrentPPO sizes its state buffers with it
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.state_size = n_heads * d_k * d_v
        if self.state_size > hidden_size:
            raise ValueError("retention state must fit in the hidden state")
        self.inp = nn.Sequential(nn.Linear(input_size, d_model), nn.Tanh())
        self.qkv = nn.Linear(d_model, n_heads * (2 * d_k + d_v))
        self.norm = nn.LayerNorm(n_heads * d_v)
        self.register_buffer("gamma", th.tensor(gammas, dtype=th.float32).view(1, n_heads, 1, 1))
        self.out = nn.Sequential(nn.Linear(d_model + n_heads * d_v + self.extra_out(), hidden_size), nn.Tanh())

    def extra_out(self) -> int:
        return 0

    # ------------------------------------------------------------------ one step
    def retention_step(self, z: th.Tensor, S: th.Tensor):
        """z (N, d_model), S (N, H, d_k, d_v) -> o (N, H*d_v), S'"""
        N = z.shape[0]
        q, k, v = self.qkv(z).view(N, self.n_heads, -1).split([self.d_k, self.d_k, self.d_v], dim=-1)
        k = k / math.sqrt(self.d_k)
        S = self.gamma * S + k.unsqueeze(-1) * v.unsqueeze(-2)
        o = (q.unsqueeze(-2) @ S).squeeze(-2)                      # (N, H, d_v)
        return self.norm(o.reshape(N, -1)), S

    def forward(self, x, states, starts=None):
        h, c = states
        T, N, _ = x.shape
        S = c[0, :, :self.state_size].reshape(N, self.n_heads, self.d_k, self.d_v)
        outs = []
        for t in range(T):
            if starts is not None:
                S = S * (1.0 - starts[t]).view(N, 1, 1, 1)
            z = self.inp(x[t])
            o, S = self.retention_step(z, S)
            outs.append(self.out(th.cat([z, o], dim=-1)))
        c_new = c.clone()
        c_new[0, :, :self.state_size] = S.reshape(N, -1)
        return th.stack(outs), (h, c_new)


class JunctionMemory(RetentionMemory):
    """Retention + position-keyed junction / dead-end slots. `obs_index` gives the observation positions of
    x, y (current dead-reckoned position), prev_x, prev_y, prev_junction and dead_end (see navigator.OBS)."""

    def __init__(self, input_size: int, obs_index: dict, hidden_size: int = 512, n_slots: int = 8,
                 memo_dim: int = 21, **kw):
        self.n_slots, self.memo_dim = n_slots, memo_dim
        super().__init__(input_size, hidden_size, **kw)
        self.slot_dim = memo_dim + 3                               # memo, x, y, used
        if self.state_size + n_slots * self.slot_dim > hidden_size:
            raise ValueError("retention state + slots must fit in the hidden state")
        self.idx = {k: int(v) for k, v in obs_index.items()}
        d_model = self.inp[0].out_features
        self.memo = nn.Sequential(nn.Linear(d_model, memo_dim), nn.Tanh())
        self.query = nn.Linear(d_model, memo_dim)
        self.log_alpha = nn.Parameter(th.tensor(math.log(100.0)))  # position units are metres / 25
        self.last_attention = None                                  # (N, K) at the last T == 1 call (for videos)

    def extra_out(self) -> int:
        return self.memo_dim + 1                                    # read vector + "found something here"

    def forward(self, x, states, starts=None):
        h, c = states
        T, N, _ = x.shape
        K, D, M = self.n_slots, self.slot_dim, self.memo_dim
        ss = self.state_size
        S = c[0, :, :ss].reshape(N, self.n_heads, self.d_k, self.d_v)
        slots = c[0, :, ss:ss + K * D].reshape(N, K, D)
        age = h[0, :, :K]                                           # write order (for oldest-first replacement)
        I = self.idx
        outs = []
        for t in range(T):
            xt = x[t]
            if starts is not None:
                keep = (1.0 - starts[t]).view(N, 1)
                S = S * keep.view(N, 1, 1, 1)
                slots = slots * keep.view(N, 1, 1)
                age = age * keep
            z = self.inp(xt)
            o, S = self.retention_step(z, S)
            # ---- write (gated by the flags in the observation)
            memo = self.memo(z)
            for flag, kx, ky in (("prev_junction", "prev_x", "prev_y"), ("dead_end", "x", "y")):
                w = (xt[:, I[flag]] > 0.5).float().view(N, 1)
                if w.any():
                    used = slots[:, :, M + 2]
                    # first free slot, else the oldest (largest age)
                    priority = th.where(used > 0.5, age, th.full_like(age, 1e9))
                    target = th.nn.functional.one_hot(priority.argmax(dim=1), K).float() * w   # (N, K)
                    new = th.cat([memo, xt[:, I[kx]:I[kx] + 1], xt[:, I[ky]:I[ky] + 1], th.ones(N, 1)], dim=1)
                    slots = slots * (1 - target.unsqueeze(-1)) + target.unsqueeze(-1) * new.unsqueeze(1)
                    age = (age + w) * (1 - target) + target * 0.0     # written slot is newest (age 0)
            # ---- read
            used = slots[:, :, M + 2]
            pos = th.stack([xt[:, I["x"]], xt[:, I["y"]]], dim=1)
            d2 = ((slots[:, :, M:M + 2] - pos.unsqueeze(1)) ** 2).sum(-1)
            q = self.query(z)
            logits = (slots[:, :, :M] * q.unsqueeze(1)).sum(-1) / math.sqrt(M) - self.log_alpha.exp() * d2
            logits = th.where(used > 0.5, logits, th.full_like(logits, -1e9))
            att = th.softmax(logits, dim=1) * (used.sum(1, keepdim=True) > 0).float()
            read = (att.unsqueeze(-1) * slots[:, :, :M]).sum(1)
            near = (att * th.exp(-self.log_alpha.exp() * d2)).sum(1, keepdim=True)
            if T == 1:
                self.last_attention = att.detach()
            outs.append(self.out(th.cat([z, o, read, near], dim=-1)))
        c_new, h_new = c.clone(), h.clone()
        c_new[0, :, :ss] = S.reshape(N, -1)
        c_new[0, :, ss:ss + K * D] = slots.reshape(N, -1)
        h_new[0, :, :K] = age
        return th.stack(outs), (h_new, c_new)

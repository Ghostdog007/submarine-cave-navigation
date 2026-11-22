"""Exit memory (memory-v3): a navigator whose memory answers questions about each exit it is offered.

Design (see docs/memory_v3.md for the literature behind each piece):
  * store   up to K entries, written sparsely (Neural Map): a DEPARTURE when the previous decision was made at a
            junction (its dead-reckoned position + the heading taken) and a DEAD END when one is detected
            (position). A write within `merge_radius` of an entry of the same kind (and, for departures, the
            same direction) updates that entry, so revisits never flood the store. Entry = position, heading,
            learned memo, dead-end flag, used, age.
  * read    PER EXIT (pointer-style policy). For exit k with navigation-frame direction e_k:
              local  "did I leave from here in this direction before?": attention over entries with a bias for
                     being close to here and pointing along e_k, plus the rule-based strength of that match;
              ahead  "is there a dead end down this exit?": attention over entries with a bias for lying near
                     the ray from here along e_k (0-14 m), plus the rule-based strength.
            So the memory's answer is attached to the exit it is about, and the policy scores exits.
  * gate    memory features enter the policy through a learned gate. "scalar" (v3): one number initialised nearly
            closed (GTrXL). It turned out to be a redundant scale (the scorer's first layer can rescale the same
            features), so it never moved. "event" (v3.2): one gate PER EXIT, computed from the events that make the
            past relevant: a dead end hit recently, a recent turn-back, no new ground for a while (decaying traces
            kept in the recurrent state), the model's own revisit belief, stored entries nearby, no goal cue, a
            junction, and that exit's own dead-end / taken beliefs and rule strengths. It starts at sigmoid(gate_init)
            everywhere (zero-initialised last layer) and learns WHEN memory matters.
  * bias    (belief_bias) the exit scores get g_k * (w_dead * p_dead_k + w_taken * p_taken_k), w learnable, initialised
            negative: from the first update, "I believe this exit is a dead end" lowers its score; PPO can weaken
            it. The beliefs are detached, so the aux heads stay honest predictors.
  * aux     prediction heads trained from simulator ground truth (Nav A3C loop closure, UNREAL, MERLIN): loop
            closure, per-exit "taken before", per-exit "known dead end". They read the UNGATED memory, so the
            memory learns what to hold even while the gate is nearly closed.

Variants share everything except the memory (same encoder, GRU, scorer, aux heads, budget):
  none   MLP (no recurrence, no store)          gru    GRU core, no store (POPGym's strong baseline)
  graph  GRU + store with RULE-based reads only (explicit junction graph, SPTM-style; no learned memo)
  exit   GRU + store with learned reads (the full model)

Recurrent state per env is one flat vector (GRU hidden + K entries), so it can be stored per rollout sequence
(R2D2-style stored state) and reset by episode starts. Only obs[:, :OBS_DIM] is read: the AUX labels that the
environment appends are training targets, never inputs (tested).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch as th
from torch import nn

from submarine.navigator.nav_env import AUX, AUX_DIM, N_SLOTS, OBS, OBS_DIM, POS_SCALE, SLOT_DIM

VARIANTS = ("none", "gru", "graph", "exit")
N_TRACES = 3                     # event traces kept in the state: dead end, turn back, stale (no new ground)
EVENTS = ("dead_end_recent", "turned_back_recent", "stale", "revisit_belief", "stored_nearby", "no_cue",
          "junction")            # per-decision trigger features of the event gate (+ 4 per exit, see _events)


@dataclass(frozen=True)
class ExitMemoryConfig:
    variant: str = "exit"
    d_model: int = 64            # observation encoding
    gru: int = 64                # recurrent core
    n_entries: int = 32          # store capacity (a dive has ~10-40 junction decisions)
    memo: int = 16               # learned memo per entry
    hidden: int = 64             # exit scorer / value MLP width
    merge_radius: float = 1.5    # m: a write this close to an entry of the same kind updates it
    sigma: float = 2.0           # m: "here" for the rule-based strengths
    ray_length: float = 14.0     # m: how far down an exit the "ahead" read looks for dead ends
    gate_init: float = -2.0      # sigmoid(-2) = 0.12
    gate: str = "scalar"         # "scalar" (v3) | "event" (v3.2: per exit, from trigger events; store variants)
    belief_bias: bool = False    # add w_dead * p_dead + w_taken * p_taken to the exit scores (v3.2)
    bias_init: tuple = (-2.0, -0.5)   # (w_dead, w_taken) at the start
    event_decay: float = 0.6     # per decision, for the dead-end / turn-back traces (0.6^3 = 0.2 after 3 decisions)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.gate not in ("scalar", "event"):
            raise ValueError("gate must be 'scalar' or 'event'")
        if self.gate == "event" and not self.has_store:
            raise ValueError("the event gate needs a store variant (graph, exit)")
        object.__setattr__(self, "bias_init", tuple(self.bias_init))

    @property
    def event_state(self) -> int:
        return N_TRACES if self.gate == "event" else 0

    @property
    def has_store(self) -> bool:
        return self.variant in ("graph", "exit")

    @property
    def entry_dim(self) -> int:
        return 2 + 2 + self.memo + 3            # pos, heading, memo, dead, used, age

    @property
    def state_size(self) -> int:
        return (self.gru if self.variant != "none" else 0) + (self.n_entries * self.entry_dim if self.has_store
                                                               else 0) + self.event_state


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, o))


class ExitMemoryNet(nn.Module):
    def __init__(self, cfg: ExitMemoryConfig = ExitMemoryConfig()):
        super().__init__()
        self.cfg = c = cfg
        self.enc = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.Tanh(), nn.Linear(128, c.d_model), nn.Tanh())
        G = c.gru if c.variant != "none" else 0
        self.core = nn.GRUCell(c.d_model, c.gru) if G else None
        ctx = c.d_model + G
        self.ctx_dim = ctx
        M = c.memo
        if c.variant == "exit":
            self.write_memo = nn.Sequential(nn.Linear(ctx, M), nn.Tanh())
            self.q_local = nn.Linear(ctx + 2, M)
            self.q_ahead = nn.Linear(ctx + 2, M)
            self.q_loop = nn.Linear(ctx, M)
            # biases of the attention logits (learned): distance scale, direction match, dead-end preference
            self.b_local = nn.Parameter(th.tensor([1.0, 2.0, 0.0]))
            self.b_ahead = nn.Parameter(th.tensor([1.0, 2.0]))
            self.mem_dim = (M + 2 + 1) + (M + 1 + 1)          # local read + strength, ahead read + strength
            self.loop_dim = M + 1
        elif c.variant == "graph":
            self.mem_dim, self.loop_dim = 3, 1                # taken strength, dead-ahead strength, both
        else:
            self.mem_dim, self.loop_dim = 0, 0
        self.gate = nn.Parameter(th.tensor(float(c.gate_init))) if self.mem_dim and c.gate == "scalar" else None
        if c.gate == "event":
            self.gate_net = _mlp(len(EVENTS) + 4, 32, 1)
            nn.init.zeros_(self.gate_net[2].weight)          # starts at sigmoid(gate_init) for every exit, event
            nn.init.constant_(self.gate_net[2].bias, float(c.gate_init))
        if c.belief_bias:
            self.w_belief = nn.Parameter(th.tensor([float(v) for v in c.bias_init]))
        self.scorer = _mlp(SLOT_DIM + self.mem_dim + ctx, c.hidden, 1)
        self.value = _mlp(ctx + self.mem_dim, c.hidden, 1)
        # aux heads: every variant gets them (fair comparison); store variants answer from their memory features
        aux_in = (self.mem_dim if self.mem_dim else 0) + SLOT_DIM + ctx
        self.aux_taken = _mlp(aux_in, 32, 1)
        self.aux_dead = _mlp(aux_in, 32, 1)
        self.aux_loop = _mlp(ctx + self.loop_dim, 32, 1)

    # ------------------------------------------------------------------ state
    def initial_state(self, n: int) -> th.Tensor:
        return th.zeros(n, self.cfg.state_size)

    def _split(self, state):
        c = self.cfg
        G = c.gru if c.variant != "none" else 0
        h = state[:, :G]
        KE = c.n_entries * c.entry_dim if c.has_store else 0
        E = state[:, G:G + KE].reshape(state.shape[0], c.n_entries, c.entry_dim) if c.has_store else None
        return h, E

    def _traces(self, state):
        c = self.cfg
        return state[:, state.shape[1] - c.event_state:] if c.event_state else None

    def _join(self, h, E, tr=None):
        parts = [h] if h.shape[1] else []
        if E is not None:
            parts.append(E.reshape(E.shape[0], -1))
        if tr is not None:
            parts.append(tr)
        return th.cat(parts, 1) if parts else h

    # ------------------------------------------------------------------ store writes
    def _write(self, E, mask, pos, head, memo, dead: float):
        """Write (pos, head, memo) where mask: merge into the same-kind entry within merge_radius (departures:
        also the same direction), else a free entry, else the oldest."""
        c, M = self.cfg, self.cfg.memo
        N, K = E.shape[0], c.n_entries
        P, H = E[:, :, 0:2], E[:, :, 2:4]
        D, U, A = E[:, :, 4 + M], E[:, :, 5 + M], E[:, :, 6 + M]
        d2 = ((P - pos[:, None]) ** 2).sum(-1)
        same = (U > 0.5) & ((D > 0.5) == (dead > 0.5)) & (d2 < c.merge_radius ** 2)
        if dead < 0.5:
            same = same & ((H * head[:, None]).sum(-1) > 0.7)
        d2m = th.where(same, d2, th.full_like(d2, 1e9))
        near = d2m.min(1)
        free_or_old = th.where(U > 0.5, A, th.full_like(A, 1e9)).argmax(1)
        idx = th.where(near.values < 1e8, near.indices, free_or_old)
        onehot = nn.functional.one_hot(idx, K).to(E.dtype) * mask[:, None]
        new = th.cat([pos, head, memo, th.full_like(pos[:, :1], dead), th.ones_like(pos[:, :1]),
                      th.zeros_like(pos[:, :1])], 1)
        E = E * (1 - onehot[..., None]) + onehot[..., None] * new[:, None]
        age = E[:, :, 6 + M] + (E[:, :, 5 + M] > 0.5).to(E.dtype) * mask[:, None] * (1 - onehot)
        return th.cat([E[:, :, :6 + M], age[..., None]], -1)

    # ------------------------------------------------------------------ one decision
    def step(self, obs: th.Tensor, state: th.Tensor, start: th.Tensor, ablate: bool = False):
        """obs (N, >=OBS_DIM), state (N, S), start (N,) -> (logits (N, 5) masked, value (N,), aux dict, state).
        ablate: the same decision with the memory read zeroed and no belief bias (reliance probes: how much the
        choice depends on memory). The returned state is the normal one."""
        c, M = self.cfg, self.cfg.memo
        x = obs[:, :OBS_DIM]
        state = state * (1.0 - start)[:, None]
        h, E = self._split(state)
        tr = self._traces(state)
        z = self.enc(x)
        if self.core is not None:
            h = self.core(z, h)
        ctx = th.cat([z, h], 1) if self.core is not None else z
        slots = x[:, OBS["slots"]].reshape(-1, N_SLOTS, SLOT_DIM)
        valid = slots[:, :, 0] > 0.5
        valid = th.cat([valid[:, :N_SLOTS - 1], th.ones_like(valid[:, :1])], 1)     # turn back always possible
        pos = th.stack([x[:, OBS["x"]], x[:, OBS["y"]]], 1) * POS_SCALE              # metres, nav frame

        if c.has_store:
            memo = self.write_memo(ctx) if c.variant == "exit" else x.new_zeros(x.shape[0], M)
            prev = th.stack([x[:, OBS["prev_x"]], x[:, OBS["prev_y"]]], 1) * POS_SCALE
            prev_h = th.stack([x[:, OBS["prev_cos"]], x[:, OBS["prev_sin"]]], 1)
            E = self._write(E, (x[:, OBS["prev_junction"]] > 0.5).to(x.dtype), prev, prev_h, memo, 0.0)
            here_h = th.stack([x[:, OBS["cos_psi"]], x[:, OBS["sin_psi"]]], 1)
            E = self._write(E, (x[:, OBS["dead_end"]] > 0.5).to(x.dtype), pos, here_h, memo, 1.0)
            mem, loop_feat = self._read(E, ctx, slots, pos)
        else:
            mem = x.new_zeros(x.shape[0], N_SLOTS, 0)
            loop_feat = x.new_zeros(x.shape[0], 0)

        ctx_k = ctx[:, None].expand(-1, N_SLOTS, -1)
        aux_in = th.cat([mem, slots, ctx_k], -1)                  # aux heads read the UNGATED memory
        aux = {"taken": self.aux_taken(aux_in).squeeze(-1), "dead_known": self.aux_dead(aux_in).squeeze(-1),
               "loop": self.aux_loop(th.cat([ctx, loop_feat], 1)).squeeze(-1)}
        p_dead, p_taken = th.sigmoid(aux["dead_known"]).detach(), th.sigmoid(aux["taken"]).detach()
        g = None
        if c.gate == "event":
            ev, tr = self._events(x, tr, mem, loop_feat, aux, p_dead, p_taken)
            g = th.sigmoid(self.gate_net(ev))                     # (N, 5, 1): per exit
        elif self.gate is not None:
            g = th.sigmoid(self.gate)
        gated = mem if g is None else g * mem
        if ablate:
            gated = th.zeros_like(gated)
        logits = self.scorer(th.cat([slots, gated, ctx_k], -1)).squeeze(-1)
        if c.belief_bias and not ablate:
            gk = 1.0 if c.gate != "event" else g[..., 0]
            logits = logits + gk * (self.w_belief[0] * p_dead + self.w_belief[1] * p_taken)
        logits = th.where(valid, logits, th.full_like(logits, -1e8))
        vmask = valid.to(x.dtype)[..., None]
        pooled = (gated * vmask).sum(1) / vmask.sum(1).clamp(min=1.0)
        value = self.value(th.cat([ctx, pooled], 1)).squeeze(-1)
        self.last_reads = (mem.detach(), valid.detach())          # for replays / probes (T = 1 calls)
        self.last_gate = None if c.gate != "event" else g[..., 0].detach()       # (N, 5)
        return logits, value, aux, self._join(h, E, tr)

    def _events(self, x, tr, mem, loop_feat, aux, p_dead, p_taken):
        """Trigger features of the event gate -> ((N, 5, len(EVENTS) + 4), updated traces (N, 3)).
        Traces: dead end / turn back are 1 at the decision they happen and decay by event_decay per decision;
        stale is an average of "stored entries here" (high = going over old ground)."""
        c, d = self.cfg, self.cfg.event_decay
        near = loop_feat[:, -1:].clamp(0.0, 1.0)                  # rule: how much is stored right here
        t_dead = th.maximum(x[:, OBS["dead_end"]:OBS["dead_end"] + 1], d * tr[:, 0:1])
        t_back = th.maximum(x[:, OBS["prev_back"]:OBS["prev_back"] + 1], d * tr[:, 1:2])
        t_stale = 0.7 * tr[:, 2:3] + 0.3 * near
        tr = th.cat([t_dead, t_back, t_stale], 1)
        no_cue = (x[:, OBS["goal"]].abs().sum(1, keepdim=True) == 0).to(x.dtype)
        glob = th.cat([tr, th.sigmoid(aux["loop"]).detach()[:, None], near, no_cue,
                       x[:, OBS["junction"]:OBS["junction"] + 1]], 1)
        self.last_events = glob.detach()
        st = self._strengths(mem)                                 # (N, 5, 2): taken / dead-ahead rule strengths
        per = th.cat([p_dead[..., None], p_taken[..., None], st.clamp(max=3.0) / 3.0], -1)
        return th.cat([glob[:, None].expand(-1, N_SLOTS, -1), per], -1), tr

    def _strengths(self, mem):
        if self.cfg.variant == "graph":
            return mem[..., :2]
        M = self.cfg.memo
        return th.stack([mem[..., M + 2], mem[..., -1]], -1)

    def _read(self, E, ctx, slots, pos):
        c, M = self.cfg, self.cfg.memo
        P, H, memo = E[:, :, 0:2], E[:, :, 2:4], E[:, :, 4:4 + M]
        D, U = E[:, :, 4 + M], E[:, :, 5 + M]
        used = U > 0.5
        e = slots[:, :, 3:5]                                                  # exit direction, nav frame (N,5,2)
        rel = P[:, None] - pos[:, None, None]                                 # (N,1,K,2) -> broadcast (N,5,K,2)
        d2 = (rel ** 2).sum(-1)                                               # (N,1,K)
        hsim = (e[:, :, None] * H[:, None]).sum(-1)                           # (N,5,K) direction match
        t = (rel * e[:, :, None]).sum(-1)                                     # along the exit's ray
        perp2 = (d2 - t ** 2).clamp(min=0.0)
        on_ray = (t > 0.5) & (t < c.ray_length)
        s2 = 2 * c.sigma ** 2
        dep = used & (D < 0.5)
        dend = used & (D > 0.5)
        near_w = th.exp(-d2 / s2)                                             # (N,1,K)
        taken_s = (near_w * hsim.clamp(min=0.0) * dep[:, None]).sum(-1)        # rule: left here this way
        ahead_s = (th.exp(-perp2 / s2) * on_ray * dend[:, None]).sum(-1)      # rule: known dead end down it
        loop_s = (near_w[:, 0] * used).sum(-1, keepdim=True)                  # rule: something stored here
        if c.variant == "graph":                              # replays: which entries each exit's rules used
            self.last_weights = {"local": (near_w * hsim.clamp(min=0.0) * dep[:, None]).detach(),
                                 "ahead": (th.exp(-perp2 / s2) * on_ray * dend[:, None]).detach()}
            return th.stack([taken_s, ahead_s, taken_s * ahead_s], -1), loop_s.clamp(max=3.0)
        # learned reads: content attention + the same geometric terms as learned biases
        ctx_e = th.cat([ctx[:, None].expand(-1, N_SLOTS, -1), e], -1)
        neg = th.full_like(hsim, -1e9)
        bl, ba = self.b_local, self.b_ahead
        lg = (self.q_local(ctx_e)[:, :, None] * memo[:, None]).sum(-1) / math.sqrt(M) \
            - nn.functional.softplus(bl[0]) * d2 / s2 + bl[1] * hsim + bl[2] * D[:, None]
        lg = th.where(used[:, None], lg, neg)
        att = th.softmax(lg, -1) * used.any(-1, keepdim=True)[:, None]
        r_local = th.cat([(att[..., None] * memo[:, None]).sum(2), (att * D[:, None]).sum(-1, keepdim=True),
                          (att * hsim).sum(-1, keepdim=True)], -1)
        la = (self.q_ahead(ctx_e)[:, :, None] * memo[:, None]).sum(-1) / math.sqrt(M) \
            - nn.functional.softplus(ba[0]) * perp2 / s2 + ba[1] * D[:, None]
        la = th.where(used[:, None] & on_ray, la, neg)
        att_a = th.softmax(la, -1) * (used[:, None] & on_ray).any(-1, keepdim=True)
        r_ahead = th.cat([(att_a[..., None] * memo[:, None]).sum(2), (att_a * D[:, None]).sum(-1, keepdim=True)], -1)
        mem = th.cat([r_local, taken_s[..., None], r_ahead, ahead_s[..., None]], -1)
        lq = (self.q_loop(ctx)[:, None] * memo).sum(-1) / math.sqrt(M) - d2[:, 0] / s2
        lq = th.where(used, lq, th.full_like(lq, -1e9))
        att_l = th.softmax(lq, -1) * used.any(-1, keepdim=True)
        loop_feat = th.cat([(att_l[..., None] * memo).sum(1), loop_s.clamp(max=3.0)], 1)
        self.last_weights = {"local": att.detach(), "ahead": att_a.detach()}   # replays: attention per exit, entry
        return mem, loop_feat

    # ------------------------------------------------------------------ sequences (training)
    def unroll(self, obs, state0, starts):
        """obs (T, N, F), state0 (N, S), starts (T, N) -> logits (T,N,5), values (T,N), aux {k: (T,N,...)}."""
        out_l, out_v, out_a = [], [], {"taken": [], "dead_known": [], "loop": []}
        s = state0
        for t in range(obs.shape[0]):
            lg, v, aux, s = self.step(obs[t], s, starts[t])
            out_l.append(lg)
            out_v.append(v)
            for k in out_a:
                out_a[k].append(aux[k])
        return th.stack(out_l), th.stack(out_v), {k: th.stack(v) for k, v in out_a.items()}, s

    # ------------------------------------------------------------------ describe (replays, probes)
    def entries(self, state) -> list[dict]:
        """Decoded store of env 0: [{slot, pos (m, nav frame), heading (rad), dead_end, age}]."""
        if not self.cfg.has_store:
            return []
        _, E = self._split(state[:1])
        M = self.cfg.memo
        out = []
        for k, row in enumerate(E[0].detach()):
            if row[5 + M] > 0.5:
                out.append({"slot": k, "pos": (float(row[0]), float(row[1])),
                            "heading": float(th.atan2(row[3], row[2])), "dead_end": bool(row[4 + M] > 0.5),
                            "age": float(row[6 + M])})
        return out

    def exit_strengths(self):
        """(taken, dead-ahead) rule strengths per exit at the last step(): (N, 5, 2), or None without a store."""
        if not self.cfg.has_store or getattr(self, "last_reads", None) is None:
            return None
        mem, _ = self.last_reads
        return self._strengths(mem)

    def config_dict(self) -> dict:
        return asdict(self.cfg)


def aux_targets(obs: th.Tensor):
    """(taken (.., 5), dead_known (.., 5), loop (..)) from the labels appended to the observation."""
    a = obs[..., OBS_DIM:OBS_DIM + AUX_DIM]
    return a[..., AUX["taken"]], a[..., AUX["dead_known"]], a[..., AUX["loop"]]


def save_model(net: ExitMemoryNet, path, extra: dict | None = None):
    th.save({"config": net.config_dict(), "state_dict": net.state_dict(), **(extra or {})}, path)


def load_model(path) -> ExitMemoryNet:
    ck = th.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    if "bias_init" in cfg:
        cfg["bias_init"] = tuple(cfg["bias_init"])
    net = ExitMemoryNet(ExitMemoryConfig(**cfg))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net

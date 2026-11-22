"""Replay videos of navigators (phase B), showing what the route memory registered.

Everything memory-related is read from the agent's actual state, not reconstructed:
  * junction-memory slots are decoded from the recurrent state the policy carries (positions are the
    dead-reckoned ones it stored, converted to the map); a faint line joins each to the true position at the
    time of writing, so dead-reckoning drift is visible;
  * attention weights are the ones the policy computed at that decision;
  * exit arrows and probabilities are the navigator's own options and action distribution at that decision.
Side panel: memory log (STORED / RECALLED with attention %), the decision bar, and the HUD.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

SLOT_NAMES = ["A", "B", "C", "D", "back"]


class MemoryTrace:
    """Decodes a junction-memory navigator's state into world-frame markers, and logs writes/recalls."""

    def __init__(self, nav, env):
        self.nav, self.env = nav, env
        mod = getattr(getattr(getattr(nav, "model", None), "policy", None), "lstm_actor", None)
        self.module = mod if hasattr(mod, "n_slots") else None
        self.log: list[tuple[float, str]] = []
        self.markers: list[dict] = []
        self.recall: list[dict] = []
        self.n_written = 0
        self.unexplained = 0                          # decoded slots no write event accounts for (should be 0)
        self._last_decision_true = (env.base.sim.state.x, env.base.sim.state.y)

    @property
    def active(self):
        return self.module is not None

    def _slots(self):
        """Stored slots: [{slot, age, nav (x, y) in metres, xy world}] decoded from the policy's state."""
        m, st = self.module, self.nav.state
        if st is None:
            return []
        K, D, M = m.n_slots, m.slot_dim, m.memo_dim
        c = np.asarray(st[1]).reshape(-1)[m.state_size:m.state_size + K * D].reshape(K, D)
        age = np.asarray(st[0]).reshape(-1)[:K]
        out = []
        for k in range(K):
            if c[k, M + 2] < 0.5:
                continue
            nx, ny = float(c[k, M]) * 25.0, float(c[k, M + 1]) * 25.0
            out.append({"slot": k, "age": float(age[k]), "nav": (nx, ny), "xy": self.to_world(nx, ny)})
        return out

    def to_world(self, nx: float, ny: float):
        """Navigation frame (start = origin, start heading = +x) -> map coordinates."""
        x0, y0, p0 = self.env.base._nav_origin
        return (x0 + nx * np.cos(p0) - ny * np.sin(p0), y0 + nx * np.sin(p0) + ny * np.cos(p0))

    def after_decision(self, obs_in, t: float):
        """Call right after the policy acted on `obs_in`: its state then includes the writes made from obs_in."""
        if not self.active:
            return
        from submarine.navigator import OBS

        st = self.env.base.sim.state
        prev = {(m["slot"], m["nav"]): m for m in self.markers}
        by_slot = {m["slot"]: m for m in self.markers}
        merge = 25.0 * float(getattr(self.module, "merge_radius", 0.0))      # metres
        slots = self._slots()
        kept = [dict(prev[(s["slot"], s["nav"])], age=s["age"]) for s in slots if (s["slot"], s["nav"]) in prev]
        new, updated = [], []
        for s in slots:
            if (s["slot"], s["nav"]) in prev:
                continue
            old = by_slot.get(s["slot"])
            if old is not None and np.hypot(s["nav"][0] - old["nav"][0], s["nav"][1] - old["nav"][1]) < merge:
                updated.append((s, old))                  # same place, written again: merged into its slot
            else:
                new.append(s)
        # what was written now, and where (navigation frame, metres): junction left at the previous decision's
        # position; dead end at the current position
        events = []
        if obs_in[OBS["prev_junction"]] > 0.5:
            h = np.degrees(np.arctan2(obs_in[OBS["prev_sin"]], obs_in[OBS["prev_cos"]]))
            events.append(("junction", f"junction, left heading {h:+.0f} deg",
                           (obs_in[OBS["prev_x"]] * 25.0, obs_in[OBS["prev_y"]] * 25.0), self._last_decision_true))
        if obs_in[OBS["dead_end"]] > 0.5:
            events.append(("dead end", "dead end", (obs_in[OBS["x"]] * 25.0, obs_in[OBS["y"]] * 25.0),
                           (st.x, st.y)))
        dist = lambda a, pos: (a[0] - pos[0]) ** 2 + (a[1] - pos[1]) ** 2
        for kind, text, pos, true in events:          # junction first: the module writes in that order
            if new:
                s = min(new, key=lambda s: dist(s["nav"], pos))
                new.remove(s)
                self.n_written += 1
                kept.append({**s, "idx": self.n_written, "dead_end": kind == "dead end", "true": true})
                self.log.append((t, f"STORED #{self.n_written}: {text}"))
            elif updated:
                s, old = min(updated, key=lambda so: dist(so[0]["nav"], pos))
                updated.remove((s, old))
                kept.append({**old, **s, "dead_end": kind == "dead end", "true": true})
                self.log.append((t, f"UPDATED #{old['idx']}: {text} (same place)"))
            else:                                     # rewritten at exactly the stored position
                same = [m for m in kept if dist(m["nav"], pos) < merge ** 2]
                if same:
                    m = min(same, key=lambda m: dist(m["nav"], pos))
                    self.log.append((t, f"UPDATED #{m['idx']}: {text} (same place)"))
        self.unexplained += len(new) + len(updated)   # must stay 0 (checked by the tests)
        self.markers = kept
        att = getattr(self.module, "last_attention", None)
        self.recall = []
        if att is not None and kept:
            w = np.asarray(att).reshape(-1)
            self.recall = [{**m, "w": float(w[m["slot"]])} for m in kept]
            top = max(self.recall, key=lambda m: m["w"])
            if top["w"] >= 0.3:
                kind = "dead end" if top["dead_end"] else "junction"
                self.log.append((t, f"RECALLED #{top['idx']} ({kind}, {100 * top['w']:.0f}%)"))
        self._last_decision_true = (st.x, st.y)


class ExitMemoryTrace:
    """The same display for exit-memory navigators (memory-v3): store entries decoded from the navigator's state,
    write events logged as STORED / UPDATED (merged at the same place) / REPLACED (oldest evicted), and checked
    against the observation's write flags (an entry change without a flag counts as unexplained: must stay 0)."""

    ARCH = {"none": ("MLP, NO MEMORY", (200, 200, 200),
                     "beliefs come from the current sonar view only: it cannot remember"),
            "gru": ("GRU MEMORY", (190, 150, 255),
                    "remembers in 64 hidden units, no map: markers show what its heads believe"),
            "graph": ("JUNCTION GRAPH STORE", (120, 230, 140),
                      "32-entry store, fixed geometric reads: lines show the entries each rule uses"),
            "exit": ("EXIT MEMORY", (255, 170, 80),
                     "32-entry store, learned attention reads: lines show what it attends to")}

    def __init__(self, nav, env):
        self.nav, self.env = nav, env
        m = getattr(nav, "model", None)
        self.net = m                                          # any memory-v3 navigator (beliefs, architecture)
        self.variant = m.cfg.variant
        self.model = m if m.cfg.has_store else None           # store variants: decoded entries
        self.beliefs = None                                   # {taken (5,), dead_known (5,), loop} last decision
        self.revisit = False
        self.gate_k = None                                    # event gate (v3.2): per exit at the last decision
        self.triggers: list[str] = []                         # ... and the events that were active
        self._gate_open = False
        self.log: list[tuple[float, str]] = []
        self.markers: list[dict] = []
        self.recall: list[dict] = []
        self.n_written = 0
        self.unexplained = 0
        self._ids: dict[int, int] = {}                 # store slot -> display number

    @property
    def active(self):
        return self.model is not None

    def to_world(self, nx, ny):
        x0, y0, p0 = self.env.base._nav_origin
        return (x0 + nx * np.cos(p0) - ny * np.sin(p0), y0 + nx * np.sin(p0) + ny * np.cos(p0))

    @property
    def gate(self):
        if self.gate_k is not None:                           # event gate: mean over the exits on offer
            return float(self.gate_k[self._valid].mean()) if self._valid.any() else float(self.gate_k.mean())
        g = getattr(self.net, "gate", None)
        return None if g is None else float(1 / (1 + np.exp(-float(g.detach()))))

    def _read_triggers(self):
        """Names of the gate's trigger events that were active at the last decision."""
        from submarine.memory.exitmem import EVENTS
        ev = getattr(self.net, "last_events", None)
        if ev is None:
            return []
        e = dict(zip(EVENTS, ev[0].numpy()))
        d = self.net.cfg.event_decay
        out = []
        if e["dead_end_recent"] > 0.2:
            n = int(round(np.log(max(e["dead_end_recent"], 1e-6)) / np.log(d)))
            out.append("dead end now" if n == 0 else f"dead end {n} dec. ago")
        if e["turned_back_recent"] > 0.2:
            out.append("turned back")
        if e["stale"] > 0.5:
            out.append("old ground")
        if e["revisit_belief"] > 0.5:
            out.append("revisit")
        if e["no_cue"] > 0.5:
            out.append("no cue")
        return out

    def hud(self) -> list[str]:
        name, _, what = self.ARCH[self.variant]
        g = self.gate
        gtxt = "" if g is None else f"   memory gate {g:.2f}" + (
            f" <- {', '.join(self.triggers)}" if self.gate_k is not None and self.triggers else
            (" (event gate: no trigger)" if self.gate_k is not None else ""))
        lines = [f"ARCHITECTURE: {name}" + gtxt + ("" if self.gate_k is not None else f"   ({what})")]
        parts = []
        if self.model is not None:
            n_j = sum(not m["dead_end"] for m in self.markers)
            n_d = sum(m["dead_end"] for m in self.markers)
            parts.append(f"store: {n_j} junction departures, {n_d} dead ends")
        b = self.beliefs
        if b is not None:
            parts.append(f"revisit belief {100 * b['loop']:.0f}%" + ("  << RECOGNISES THIS PLACE" if self.revisit
                                                                    else ""))
        if parts:
            lines.append("   |   ".join(parts))
        return lines

    def legend(self) -> list[str]:
        out = []
        if self.model is not None:
            out += ["ring #n   left a junction (stored)", "red X #n  dead end (stored)",
                    "line      entry recalled for this choice", "yellow pulse  strongest recall"]
        out += ["DEAD? n%  exit believed a known dead end", "BEEN n%   exit believed already taken",
                "orange ring  believes it is revisiting"]
        return out

    def decide(self, obs_in, choice: int, t: float):
        """After the navigator acted: beliefs (aux heads), recall lines (store reads), decision events."""
        from submarine.navigator import N_SLOTS, OBS, SLOT_DIM

        aux = getattr(self.nav, "last_aux", None)
        if aux is None:
            return
        self.beliefs = {"taken": aux["taken"], "dead_known": aux["dead_known"], "loop": float(aux["loop"])}
        lg = getattr(self.net, "last_gate", None)
        if lg is not None:                                    # event gate (v3.2)
            _, vmask = self.net.last_reads
            self._valid = vmask[0].numpy().astype(bool)
            self.gate_k = lg[0].numpy()
            self.triggers = self._read_triggers()
            g = self.gate
            if g > 0.7 and not self._gate_open:
                self.log.append((t, f"GATE OPENED {g:.2f} <- " + (", ".join(self.triggers) or "no trigger")))
            self._gate_open = g > 0.6
        rev = self.beliefs["loop"] > 0.5
        if rev and not self.revisit:
            self.log.append((t, f"RECOGNISED a place it has been ({100 * self.beliefs['loop']:.0f}%)"))
        self.revisit = rev
        slots = obs_in[OBS["slots"]].reshape(N_SLOTS, SLOT_DIM)
        valid = [k for k in range(N_SLOTS - 1) if slots[k, 0] > 0.5]
        if len(valid) >= 2:                                    # a real choice: report what memory steered
            dk = self.beliefs["dead_known"]
            for k in valid:
                if dk[k] > 0.5 and k != choice:
                    self.log.append((t, f"AVOIDED {SLOT_NAMES[k]}: believes dead end ({100 * dk[k]:.0f}%)"))
            if choice in valid and dk[choice] > 0.5:
                self.log.append((t, f"ENTERED {SLOT_NAMES[choice]} despite dead-end belief "
                                    f"({100 * dk[choice]:.0f}%)"))
        self.recall = []
        w = getattr(self.model, "last_weights", None) if self.model is not None else None
        if w is not None:
            wl, wa = w["local"][0, choice].numpy(), w["ahead"][0, choice].numpy()
            for m in self.markers:
                wk = float(min(1.0, max(wl[m["slot"]], wa[m["slot"]])))
                if wk > 0.05:
                    self.recall.append({"xy": m["xy"], "dead_end": m["dead_end"], "w": wk})

    def after_decision(self, obs_in, t: float):
        if not self.active:
            return
        from submarine.navigator import OBS

        old = {m["slot"]: m for m in self.markers}
        flags = int(obs_in[OBS["prev_junction"]] > 0.5) + int(obs_in[OBS["dead_end"]] > 0.5)
        changed = 0
        markers = []
        for e in self.model.entries(self.nav.state):
            k, prev = e["slot"], old.get(e["slot"])
            same = prev is not None and np.hypot(e["pos"][0] - prev["nav"][0], e["pos"][1] - prev["nav"][1]) < 1e-4 \
                and prev["dead_end"] == e["dead_end"]
            if not same:
                changed += 1
                what = "dead end" if e["dead_end"] else f"left junction, heading {np.degrees(e['heading']):+.0f} deg"
                if prev is not None and prev["dead_end"] == e["dead_end"] and \
                        np.hypot(e["pos"][0] - prev["nav"][0], e["pos"][1] - prev["nav"][1]) < \
                        self.model.cfg.merge_radius:
                    self.log.append((t, f"UPDATED #{self._ids[k]}: {what} (same place)"))
                else:
                    self.n_written += 1
                    verb = "REPLACED oldest with" if prev is not None else "STORED"
                    self._ids[k] = self.n_written
                    self.log.append((t, f"{verb} #{self.n_written}: {what}"))
            markers.append({"slot": k, "nav": e["pos"], "xy": self.to_world(*e["pos"]), "dead_end": e["dead_end"],
                            "idx": self._ids.get(k, 0)})
        self.unexplained += max(0, changed - flags)
        self.markers = markers
        s = self.model.exit_strengths()
        self.exit_memory = None if s is None else s[0].numpy()


def _probs(nav, obs):
    """The navigator's action distribution at this decision (without advancing its state)."""
    import torch as th

    m = getattr(nav, "model", None)
    if m is None:
        return None
    with th.no_grad():
        ot = th.as_tensor(obs[None], dtype=th.float32)
        if hasattr(m, "step"):                             # exit-memory navigator (memory-v3)
            logits, _, _, _ = m.step(ot, nav.state.clone(), nav.start)
            return th.softmax(logits[0], -1).numpy()
        if nav.recurrent:
            st = nav.state
            if st is None:
                n = m.policy.lstm_actor.hidden_size
                st = (np.zeros((1, 1, n), np.float32), np.zeros((1, 1, n), np.float32))
            states = (th.as_tensor(st[0]), th.as_tensor(st[1]))
            dist, _ = m.policy.get_distribution(ot, states, th.as_tensor(nav.start.astype(np.float32)))
            return dist.distribution.probs[0].numpy()     # (also recomputes last_attention: same input, same value)
        return m.policy.get_distribution(ot).distribution.probs[0].numpy()


def record_nav_replay(nav, cfg, layout: str, cave_seed: int, difficulty: float, out_path: str | Path,
                      speed: float = 4.0, theme: str = "ice") -> dict:
    """Record one dive of `nav` (a navigator from scripts.eval_navigator.make_navigator) as an .mp4."""
    import pygame

    from submarine.evaluation.metrics import episode_seed
    from submarine.navigator import N_SLOTS, OBS, SLOT_DIM, NavigatorEnv
    from submarine.render.viewer import Viewer

    env = NavigatorEnv(replace(cfg, env=replace(cfg.env, layout=layout)))
    obs, info = env.reset(seed=episode_seed(cave_seed, difficulty),
                          options={"cave_seed": cave_seed, "difficulty": difficulty})
    if hasattr(nav, "attach"):
        nav.attach(env)
    nav.reset()
    v3 = hasattr(getattr(nav, "model", None), "entries")
    trace = ExitMemoryTrace(nav, env) if v3 else MemoryTrace(nav, env)
    c = env.cfg                                               # memory-pricing reward (memory-v3.1) shown live
    rw_on = c.progress == "record" or c.explore_bonus > 0 or any(
        x > 0 for x in (c.repeat_dead_penalty, c.repeat_taken_penalty, c.dead_end_revisit_penalty, c.dither_penalty))
    viewer = Viewer(env.base.cave, scale=None, style="game", theme=theme, camera="follow",
                    hud_h=82 if rw_on else 64)                # a 4th status line for the reward
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=speed / env.dt, codec="libx264", quality=8, macro_block_size=1,
                                ffmpeg_log_level="error")
    state = {"exits": [], "probs": None, "choice": None, "t_dec": -1.0, "done": False, "outcome": ""}
    rw = {"ret": 0.0, "pen": 0.0, "n_dead": 0, "n_taken": 0, "n_rev": 0, "n_dither": 0, "explore": 0.0,
          "t_explore": -9.0}

    def reward_events(t):
        """Live from the env's episode counters: penalties are set when the decision is taken, the exploration
        bonus as each new cell is entered."""
        ep = env._ep
        k = state["choice"]
        if ep["repeat_dead"] > rw["n_dead"]:
            trace.log.append((t, f"PENALTY -{c.repeat_dead_penalty:g}: took {SLOT_NAMES[k]}, an exit that "
                                 "already led to a dead end"))
        if ep["repeat_taken"] > rw["n_taken"]:
            trace.log.append((t, f"PENALTY -{c.repeat_taken_penalty:g}: re-took {SLOT_NAMES[k]} while an exit "
                                 "here is untried"))
        if ep["dithers"] > rw["n_dither"]:
            trace.log.append((t, f"PENALTY -{c.dither_penalty:g}: turned back again (U-turn loop, no junction "
                                 "in between)"))
        if ep["dead_end_revisits"] > rw["n_rev"] and c.dead_end_revisit_penalty:
            trace.log.append((t, f"PENALTY -{c.dead_end_revisit_penalty:g}: back at a dead end it already "
                                 "reached and left"))
        if ep["explore"] > rw["explore"]:
            rw["t_explore"] = t
        rw.update(n_dead=ep["repeat_dead"], n_taken=ep["repeat_taken"], n_rev=ep["dead_end_revisits"],
                  n_dither=ep["dithers"], pen=ep["repeat_penalty"], explore=ep["explore"])

    def frame(info):
        e = env.base
        s = e.sim.state
        t = e._t * env.dt
        hud = [f"t {t:5.1f} s   speed {s.u:+.2f} m/s   decisions {env.n_decisions}   "
               f"to goal {info['geodesic']:5.1f} m   energy {info['energy_j'] / 1000:5.1f} kJ   "
               f"filter {'ON' if env.pilot.last_info.get('overridden') else 'off'}"
               if hasattr(env.pilot, "last_info") else f"t {t:5.1f} s"]
        if v3:
            hud += trace.hud()
        elif trace.active:
            n_j = sum(not m["dead_end"] for m in trace.markers)
            n_d = sum(m["dead_end"] for m in trace.markers)
            hud.append(f"junction memory: {n_j} junctions, {n_d} dead ends stored")
        if rw_on:
            reward_events(t)
            hud.append(f"reward so far {rw['ret']:+6.1f}   progress: {c.progress}   "
                       f"repeat penalties {-rw['pen']:+.1f} (dead-end exit {rw['n_dead']}x, re-taken {rw['n_taken']}x, "
                       f"dead-end revisit {rw['n_rev']}x, U-turn loop {rw['n_dither']}x)   exploration bonus {rw['explore']:+.1f}"
                       + ("" if c.explore_bonus > 0 and not (c.explore_without_cue_only and env._cue)
                          else " (off: goal cue shown)" if c.explore_bonus > 0 else ""))
        banner = state["outcome"].upper() if state["done"] else None
        viewer.draw(e.sim, e._last, e._trail, hud, banner, speeds=None, action=e._a_prev,
                    progress=1 - info["geodesic"] / max(e.cave.shortest_path_length, 1e-6),
                    label=f"{nav.name}   {layout}  d={difficulty:.2f}  cave {cave_seed}",
                    memory=trace.markers if trace.active else None, recall=trace.recall)
        cv = viewer.screen                                   # drawn after viewer.draw composed the frame

        def sp(x, y):                                        # world -> screen pixels (follow camera)
            px, py = viewer.to_px(x, y)
            return px - viewer.view.x, py - viewer.view.y
        # exits at the last decision (shown for 1.5 s), probability per exit, chosen one highlighted
        if 0 <= t - state["t_dec"] < 1.5 and state["exits"]:
            o = sp(s.x, s.y)
            font = viewer.small
            for k, (b, depth) in state["exits"]:
                h = s.psi + b
                L = min(depth, 4.0)
                p = sp(s.x + L * np.cos(h), s.y + L * np.sin(h))
                chosen = k == state["choice"]
                col = (255, 220, 80) if chosen else (180, 200, 220)
                pygame.draw.line(cv, col, o, p, 4 if chosen else 2)
                lab = SLOT_NAMES[k] + (f" {100 * state['probs'][k]:.0f}%" if state["probs"] is not None else "")
                em = getattr(trace, "exit_memory", None)          # store variants: what the store says (rules)
                if em is not None:
                    lab += ((" [store: dead end]" if em[k, 1] > 0.5 else "")
                            + (" [store: taken]" if em[k, 0] > 0.5 else ""))
                b = getattr(trace, "beliefs", None)                # every v3 variant: what its heads believe
                if b is not None:
                    acol = ExitMemoryTrace.ARCH[trace.variant][1]
                    if b["dead_known"][k] > 0.5:                   # red X on the exit, ringed in the arch colour
                        rr = 7
                        pygame.draw.line(cv, (255, 80, 80), (p[0] - rr, p[1] - rr), (p[0] + rr, p[1] + rr), 3)
                        pygame.draw.line(cv, (255, 80, 80), (p[0] - rr, p[1] + rr), (p[0] + rr, p[1] - rr), 3)
                        pygame.draw.circle(cv, acol, p, rr + 4, 2)
                        lab += f" DEAD? {100 * b['dead_known'][k]:.0f}%"
                    if b["taken"][k] > 0.5:                        # filled dot: believes already taken
                        pygame.draw.circle(cv, acol, p, 5)
                        lab += f" BEEN {100 * b['taken'][k]:.0f}%"
                    gk = getattr(trace, "gate_k", None)            # event gate: how much memory this exit uses
                    if gk is not None:
                        lab += f" mem {gk[k]:.2f}"
                cv.blit(font.render(lab, True, col), (p[0] + 4, p[1] - 8))
        # every v3 variant: architecture badge; a pulsing ring on the sub while it believes it is revisiting
        if v3:
            name, acol, _ = ExitMemoryTrace.ARCH[trace.variant]
            badge = viewer.label_font.render(name, True, (0, 0, 0))
            bw, bh = badge.get_size()
            bx, by = cv.get_width() - bw - 24, (70 if state["done"] else 12)   # under the outcome banner at the end
            pygame.draw.rect(cv, acol, (bx - 8, by - 4, bw + 16, bh + 8), border_radius=6)
            cv.blit(badge, (bx, by))
            leg = trace.legend()
            lw = max(viewer.small.size(l)[0] for l in leg) + 16
            lp = pygame.Surface((lw, 16 * len(leg) + 10), pygame.SRCALPHA)
            lp.fill((0, 0, 0, 150))
            lx = cv.get_width() - lw - 16
            cv.blit(lp, (lx, by + bh + 12))
            for i, l in enumerate(leg):
                cv.blit(viewer.small.render(l, True, acol), (lx + 8, by + bh + 17 + 16 * i))
            if trace.revisit:
                o = sp(s.x, s.y)
                pulse = int(22 + 6 * np.sin(viewer.frame * 0.4))
                pygame.draw.circle(cv, (255, 150, 60), o, pulse, 3)
                cv.blit(viewer.small.render("revisit", True, (255, 150, 60)), (o[0] + pulse + 2, o[1] - 8))
        # exploration bonus: a short "+0.5 new area" flash at the sub
        if rw_on and 0 <= t - rw["t_explore"] < 0.8:
            o = sp(s.x, s.y)
            cv.blit(viewer.label_font.render(f"+{c.explore_bonus:g} new area", True, (120, 255, 140)),
                    (o[0] + 18, o[1] + 14))
        # drift: remembered (dead-reckoned) vs true position at write time
        if trace.active:
            for m in trace.markers:
                if m.get("true") is not None:
                    pygame.draw.line(cv, (200, 200, 200), sp(*m["xy"]), sp(*m["true"]), 1)
        # memory log panel
        lines = [f"{tt:5.1f}s  {msg}" for tt, msg in trace.log[-8:]]
        if lines:
            font = viewer.small
            w = max(font.size(l)[0] for l in lines) + 16
            panel = pygame.Surface((w, 18 * len(lines) + 12), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 170))
            cv.blit(panel, (10, 48))
            for i, l in enumerate(lines):
                col = ((120, 230, 255) if "STORED" in l else (120, 255, 140) if "AVOIDED" in l
                       else (255, 150, 60) if "RECOGNISED" in l
                       else (230, 140, 255) if "GATE" in l
                       else (255, 110, 110) if ("ENTERED" in l or "PENALTY" in l)
                       else (255, 235, 120))
                cv.blit(font.render(l, True, col), (18, 54 + 18 * i))
        writer.append_data(viewer.rgb_array())

    orig_step = env.base.step

    def step_and_draw(a):
        out = orig_step(a)
        frame(out[4])
        return out

    env.base.step = step_and_draw
    while True:
        slots = obs[OBS["slots"]].reshape(N_SLOTS, SLOT_DIM)
        state["exits"] = [(k, (float(np.arctan2(slots[k, 2], slots[k, 1])), float(slots[k, 5] * 10.0) or 2.0))
                          for k in range(N_SLOTS) if slots[k, 0] > 0.5 or k == N_SLOTS - 1]
        state["probs"] = _probs(nav, obs)
        a = nav.act(obs)
        state["choice"], state["t_dec"] = a, env.base._t * env.dt
        trace.after_decision(obs, state["t_dec"])
        if v3:
            trace.decide(obs, a, state["t_dec"])
        obs, r, te, tr, info = env.step(a)
        rw["ret"] += r                                        # (per decision; the counters below are live)
        if te or tr:
            break
    state["done"] = True
    state["outcome"] = ("success" if info["is_success"] else "hull breach" if info["collided"]
                        else "timeout")
    for _ in range(int(2.0 * speed / env.dt / 4)):
        frame(info)
    writer.close()
    return {"outcome": state["outcome"], "time": info["t"] * env.dt, "decisions": env.n_decisions,
            "stored": trace.n_written, "unexplained": trace.unexplained, "log": trace.log}

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

    def __init__(self, nav, env):
        self.nav, self.env = nav, env
        m = getattr(nav, "model", None)
        self.model = m if getattr(getattr(m, "cfg", None), "has_store", False) else None
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
    trace = ExitMemoryTrace(nav, env) if hasattr(getattr(nav, "model", None), "entries") else MemoryTrace(nav, env)
    viewer = Viewer(env.base.cave, scale=None, style="game", theme=theme, camera="follow")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=speed / env.dt, codec="libx264", quality=8, macro_block_size=1,
                                ffmpeg_log_level="error")
    state = {"exits": [], "probs": None, "choice": None, "t_dec": -1.0, "done": False, "outcome": ""}

    def frame(info):
        e = env.base
        s = e.sim.state
        t = e._t * env.dt
        hud = [f"t {t:5.1f} s   speed {s.u:+.2f} m/s   decisions {env.n_decisions}   "
               f"to goal {info['geodesic']:5.1f} m   energy {info['energy_j'] / 1000:5.1f} kJ   "
               f"filter {'ON' if env.pilot.last_info.get('overridden') else 'off'}"
               if hasattr(env.pilot, "last_info") else f"t {t:5.1f} s"]
        if trace.active:
            n_j = sum(not m["dead_end"] for m in trace.markers)
            n_d = sum(m["dead_end"] for m in trace.markers)
            hud.append(f"junction memory: {n_j} junctions, {n_d} dead ends stored")
        banner = state["outcome"].upper() if state["done"] else None
        viewer.draw(e.sim, e._last, e._trail, hud, banner, speeds=None, action=e._a_prev,
                    progress=1 - info["geodesic"] / max(e.cave.shortest_path_length, 1e-6),
                    label=f"{nav.name}   {layout}  d={difficulty:.2f}  cave {cave_seed}",
                    memory=trace.markers if trace.active else None, recall=trace.recall)
        cv = viewer.canvas
        # exits at the last decision (shown for 1.5 s), probability per exit, chosen one highlighted
        if 0 <= t - state["t_dec"] < 1.5 and state["exits"]:
            o = viewer.to_px(s.x, s.y)
            font = viewer.small
            for k, (b, depth) in state["exits"]:
                h = s.psi + b
                L = min(depth, 4.0)
                p = viewer.to_px(s.x + L * np.cos(h), s.y + L * np.sin(h))
                chosen = k == state["choice"]
                col = (255, 220, 80) if chosen else (180, 200, 220)
                pygame.draw.line(cv, col, o, p, 4 if chosen else 2)
                lab = SLOT_NAMES[k] + (f" {100 * state['probs'][k]:.0f}%" if state["probs"] is not None else "")
                em = getattr(trace, "exit_memory", None)          # memory-v3: what memory says about this exit
                if em is not None:
                    lab += (" [known dead end]" if em[k, 1] > 0.5 else "") + (" [taken]" if em[k, 0] > 0.5 else "")
                cv.blit(font.render(lab, True, col), (p[0] + 4, p[1] - 8))
        # drift: remembered (dead-reckoned) vs true position at write time
        if trace.active:
            for m in trace.markers:
                if m.get("true") is not None:
                    pygame.draw.line(cv, (200, 200, 200), viewer.to_px(*m["xy"]), viewer.to_px(*m["true"]), 1)
        # memory log panel
        lines = [f"{tt:5.1f}s  {msg}" for tt, msg in trace.log[-6:]]
        if lines:
            font = viewer.small
            w = max(font.size(l)[0] for l in lines) + 16
            panel = pygame.Surface((w, 18 * len(lines) + 12), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 170))
            cv.blit(panel, (10, 80))
            for i, l in enumerate(lines):
                col = (120, 230, 255) if "STORED" in l else (255, 235, 120)
                cv.blit(font.render(l, True, col), (18, 86 + 18 * i))
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
        obs, r, te, tr, info = env.step(a)
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

"""Navigator environment (phase B): one step = one route decision; the frozen driver drives in between.

At a decision point the navigator sees the exits the openings detector reports and picks one (or turns back).
The driver (learned network or hand controller, both with the built-in safety filter) then follows that exit
at 10 Hz until the next decision point:
  * the set of exits changes (junction reached / passed, dead end ahead) for 2 consecutive scans,
  * the followed exit disappears,
  * `decide_s` seconds have passed (in a plain tunnel: a routine "carry on?" decision),
  * or the episode ends.
After turning back, no decision is due for `hold_after_turn_s` (so it actually turns round first).

Action: Discrete(5). Slots 0-3 = forward/side exits sorted by bearing (right to left), slot 4 = turn back.
Empty slots (`empty_slot`):
  * "remap" (default): an empty slot means the valid forward exit with the nearest slot index, or turn back if there
    is none (a dead end). Every decision is a real exit or a turn-back: there is no "carry on regardless" option.
  * "carry_on" (first phase-B runs): keep the current heading, cost `invalid_penalty`. Navigators learned to lean
    on it (64% of the retention navigator's choices at maze junctions) and pressed into dead ends until timeout.

Observation (61 values, see OBS): per slot [valid, cos/sin bearing (body), cos/sin heading (navigation frame),
depth/10, width/pi, narrow], goal cue (cos/sin bearing, distance) as in CaveEnv, dead-reckoned pose (x, y / 25,
cos/sin heading), the previous decision (its pose, chosen heading, whether it was at a junction / a turn-back),
junction / dead-end flags, speed, time since the last decision, episode time, sonar health, supervisor tier, hull.
The memory modules (submarine/memory) read the pose and the flags at the fixed indices in OBS.

memory-v3 additions: `goal_cue_prob` withholds the goal cue in a fraction of dives ("find the exit"), and
`aux_labels` appends ground-truth labels (AUX) for the memory's auxiliary losses; CaveEnv's own per-step junction
counting is switched off here (unused by the navigator, ~11% of a decision).

Reward: the CaveEnv reward summed over the driver steps (shortest-path progress, time, energy, damage, goal,
collision/abort), plus the invalid-slot penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import gymnasium as gym
import numpy as np

from submarine.control import DriverConfig, DriverPolicy
from submarine.control.controller import wrap
from submarine.control.driver_env import driver_extras
from submarine.env import CaveEnv, EnvConfig

N_SLOTS = 5                    # 4 forward/side exits + turn back
SLOT_DIM = 8
POS_SCALE = 25.0


def _layout():
    names, i = {}, 0

    def add(name, n=1):
        nonlocal i
        names[name] = i if n == 1 else slice(i, i + n)
        i += n

    add("slots", N_SLOTS * SLOT_DIM)
    add("goal", 3)
    add("x"); add("y"); add("cos_psi"); add("sin_psi")
    add("prev_x"); add("prev_y"); add("prev_cos"); add("prev_sin"); add("prev_junction"); add("prev_back")
    add("junction"); add("dead_end")
    add("speed"); add("since"); add("time")
    add("sonar_ok"); add("tier"); add("hull")
    return names, i


OBS, OBS_DIM = _layout()

# Ground-truth labels for the memory's auxiliary losses (memory-v3), appended AFTER the 61 navigator inputs when
# `aux_labels` is on. They ride in the observation so they survive vector-env auto-resets; models must read only
# obs[:OBS_DIM] (tested). All from the TRUE trajectory, at the moment of this observation:
#   loop       1  back within `aux_radius` of an earlier decision point, after >= `aux_loop_path` m of travel
#   taken      5  per exit slot: left an earlier junction from (here, this exit's direction)
#   dead_known 5  per exit slot: ... and that departure ran into a dead end before coming back
AUX = {"loop": 0, "taken": slice(1, 6), "dead_known": slice(6, 11)}
AUX_DIM = 11


@dataclass(frozen=True)
class NavigatorEnvConfig:
    env: EnvConfig = field(default_factory=EnvConfig)      # navigation (dead reckoning) is switched on in the env
    driver: DriverConfig = field(default_factory=DriverConfig)
    driver_actor: str | None = None           # path to a learned driver (driver.npz); None = hand controller
    cruise_speed: float = 1.2
    decide_s: float = 2.0
    min_decide_s: float = 0.4                 # never two decisions closer than this
    hold_after_turn_s: float = 3.0
    invalid_penalty: float = 0.1              # only with empty_slot = "carry_on"
    empty_slot: str = "remap"                 # "remap" | "carry_on" (see the module docstring)
    goal_cue_prob: float = 1.0                # P(an episode shows the goal cue); otherwise "find the exit" blind
    aux_labels: bool = False                  # append the AUX ground-truth labels to the observation
    aux_radius: float = 2.0                   # m, "same place"
    aux_loop_path: float = 6.0                # m travelled before a return counts as a loop closure
    aux_heading_cos: float = 0.77             # same exit direction: within ~40 deg
    # memory-pricing reward (memory-v3.1; all off by default = the original reward). See docs/memory_v3.md.
    progress: str = "dense"                   # "dense": k_prog per metre of geodesic progress (refunded when
                                              # backing out of a branch); "record": paid only for beating the
                                              # best geodesic so far: detours cost their time, the map no longer
                                              # signals a wrong branch on entry
    repeat_dead_penalty: float = 0.0          # choosing an exit that already led to a dead end (true labels)
    repeat_taken_penalty: float = 0.0         # re-taking an exit already taken from here while an untried exit
                                              # exists HERE. Both only for forward exits at a decision: turning
                                              # back and driving through explored tunnels (e.g. back to a far
                                              # junction) are never penalised; they cost only time and energy
    dead_end_revisit_penalty: float = 0.0     # arriving again at the tip of a dead end already reached and left
                                              # (>= dead_end_away m): catches re-entry by any route, including
                                              # turning back in the corridor; never fires heading away from it
    dead_end_away: float = 4.0                # m
    dither_penalty: float = 0.0               # turning back again within 2 decisions of a turn back, with no
                                              # junction reached in between (the U-turn loop out of a dead end);
                                              # the forced turn at a dead end itself is never penalised
    explore_bonus: float = 0.0                # per newly visited explore_cell x explore_cell cell
    explore_cell: float = 2.0                 # m
    explore_without_cue_only: bool = True     # the bonus only in dives without the goal cue

    @classmethod
    def from_dict(cls, d: dict | None) -> "NavigatorEnvConfig":
        d = dict(d or {})
        env = dict(d.get("env") or {})
        env.setdefault("navigation", {})
        env["navigation"] = dict(env["navigation"], enabled=True)     # the navigator needs dead reckoning
        d["env"] = EnvConfig.from_dict(env)
        if isinstance(d.get("driver"), dict):
            d["driver"] = DriverConfig.from_dict(d["driver"])
        return cls(**d)


class NavigatorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: NavigatorEnvConfig | dict | None = None):
        cfg = config if isinstance(config, NavigatorEnvConfig) else NavigatorEnvConfig.from_dict(config)
        # dead reckoning on; CaveEnv's own per-step junction counting off (the navigator has its own detector,
        # nothing reads those flags here, and they cost ~11% of a decision)
        cfg = replace(cfg, env=replace(cfg.env, navigation=replace(cfg.env.navigation, enabled=True,
                                                                   junction_flags=False)))
        self.cfg = cfg
        self.base = CaveEnv(cfg.env)
        dcfg = replace(cfg.driver, cruise_speed=cfg.cruise_speed)
        kw = dict(mode="external", decide_s=cfg.decide_s, hold_after_turn_s=cfg.hold_after_turn_s)
        if cfg.driver_actor:
            from submarine.control.learned import LearnedDriverPolicy

            self.pilot = LearnedDriverPolicy(cfg.driver_actor, dcfg, **kw)
        else:
            self.pilot = DriverPolicy(dcfg, **kw)
        self.observation_space = gym.spaces.Box(-2.0, 2.0, (OBS_DIM + (AUX_DIM if cfg.aux_labels else 0),),
                                                np.float32)
        self.action_space = gym.spaces.Discrete(N_SLOTS)
        self.dt = cfg.env.vehicle.dt

    # ------------------------------------------------------------------ helpers
    def _slots(self):
        """(slot list: Exit or None) for the 4 forward/side slots + turn back."""
        d = self.pilot.driver
        front = sorted((x for x in d.exits if not x.opening.rear), key=lambda x: x.opening.bearing)[:N_SLOTS - 1]
        rear = [x for x in d.exits if x.opening.rear]
        slots = list(front) + [None] * (N_SLOTS - 1 - len(front))
        slots.append(rear[0] if rear else None)               # None: turn back = heading + pi
        return slots

    def _obs(self):
        d, e, O = self.pilot.driver, self.base, OBS
        nav = e._nav
        o = np.zeros(OBS_DIM, np.float32)
        slots = self._slots()
        rows = []
        for k, x in enumerate(slots):
            if x is None and k < N_SLOTS - 1:
                rows.append(np.zeros(SLOT_DIM))
                continue
            if x is None:                                     # synthetic turn back: straight behind
                b, depth, width, narrow = np.pi, 0.0, 0.0, 0.0
            else:
                b, depth, width, narrow = x.opening.bearing, x.opening.depth, x.opening.width, float(x.opening.narrow)
            h = nav.psi + b                                   # heading in the (dead-reckoned) navigation frame
            rows.append([1.0, np.cos(b), np.sin(b), np.cos(h), np.sin(h), depth / 10.0, width / np.pi, narrow])
        o[O["slots"]] = np.concatenate(rows)
        obs = self._base_obs
        L = self.pilot.L
        goal = L.goal(obs)
        if goal is not None and self._cue:
            o[O["goal"]] = [np.cos(goal[0]), np.sin(goal[0]), goal[1]]
        o[O["x"]], o[O["y"]] = nav.x / POS_SCALE, nav.y / POS_SCALE
        o[O["cos_psi"]], o[O["sin_psi"]] = np.cos(nav.psi), np.sin(nav.psi)
        p = self._prev
        o[O["prev_x"]], o[O["prev_y"]] = p["x"] / POS_SCALE, p["y"] / POS_SCALE
        o[O["prev_cos"]], o[O["prev_sin"]] = np.cos(p["h"]), np.sin(p["h"])
        o[O["prev_junction"]], o[O["prev_back"]] = p["junction"], p["back"]
        n_front = sum(x is not None for x in slots[:N_SLOTS - 1])
        o[O["junction"]], o[O["dead_end"]] = float(n_front >= 2), float(n_front == 0)
        o[O["speed"]] = d.u / e.cfg.vehicle.u_max
        o[O["since"]] = min((self.pilot.t - self._t_last) / self.cfg.decide_s, 2.0)
        o[O["time"]] = e._t / e.cfg.max_steps
        _, hull, sonar_ok, tier = driver_extras(e)
        o[O["sonar_ok"]], o[O["tier"]], o[O["hull"]] = sonar_ok, tier, hull
        o = np.clip(o, -2.0, 2.0)
        bearings = [None if x is None and k < N_SLOTS - 1 else (np.pi if x is None else x.opening.bearing)
                    for k, x in enumerate(slots)]
        labels = self._aux_labels(bearings, n_front)
        return np.concatenate([o, labels]) if self.cfg.aux_labels else o

    # ------------------------------------------------------------------ ground truth for the memory's aux losses
    def _aux_labels(self, bearings, n_front: int) -> np.ndarray:
        """Labels for THIS observation from the true trajectory (see AUX), and the bookkeeping behind them:
        decision points (for loop closure) and junction departures (exit taken; dead end found before return)."""
        c, s = self.cfg, self.base.sim.state
        pl = float(self._info.get("path_length", 0.0))
        R2 = c.aux_radius ** 2
        near = lambda q: (q["x"] - s.x) ** 2 + (q["y"] - s.y) ** 2 < R2
        out = np.zeros(AUX_DIM, np.float32)
        out[AUX["loop"]] = float(any(near(q) and pl - q["pl"] >= c.aux_loop_path for q in self._points))
        for dep in self._departures:                          # returned to where a departure left from: closed
            if dep["open"] and near(dep) and pl - dep["pl"] >= c.aux_loop_path:
                dep["open"] = False
        self._update_tips(s.x, s.y, n_front)
        if n_front == 0:                                      # dead end: the departure that led straight here
            if self._departures and self._departures[-1]["open"]:     # (no other junction taken since; else the
                self._departures[-1]["dead"] = True               # dead end belongs to a later, unseen branch)
        heads = [None if b is None else s.psi + b for b in bearings]
        for k, h in enumerate(heads):
            if h is None:
                continue
            for dep in self._departures:                      # (only once it has come back: not the departure
                if near(dep) and pl - dep["pl"] >= c.aux_loop_path \
                        and np.cos(dep["h"] - h) > c.aux_heading_cos:     # it is still leaving)
                    out[1 + k] = 1.0
                    if dep["dead"]:
                        out[6 + k] = 1.0
        self._points.append({"x": s.x, "y": s.y, "pl": pl})
        self._pending = {"x": s.x, "y": s.y, "pl": pl, "heads": heads, "junction": n_front >= 2}
        self._labels = out
        return out

    def _update_tips(self, x: float, y: float, n_front: int):
        """Dead-end tips (true positions): a revisit = back at a tip after having been >= dead_end_away from it."""
        c = self.cfg
        for tip in self._tips:
            d2 = (tip["x"] - x) ** 2 + (tip["y"] - y) ** 2
            if d2 > c.dead_end_away ** 2:
                tip["away"] = True
            elif d2 < c.aux_radius ** 2 and tip["away"] and n_front == 0:
                tip["away"] = False
                self._revisits += 1
        if n_front == 0 and not any((t["x"] - x) ** 2 + (t["y"] - y) ** 2 < c.aux_radius ** 2 for t in self._tips):
            self._tips.append({"x": x, "y": y, "away": False})
        if n_front >= 2:
            self._junction_since_back = True

    def _record_departure(self, k: int):
        p = self._pending
        if p is not None and p["junction"] and k < N_SLOTS - 1 and p["heads"][k] is not None:
            self._departures.append({"x": p["x"], "y": p["y"], "pl": p["pl"], "h": p["heads"][k],
                                     "open": True, "dead": False})

    def _drive_until_decision(self):
        """Run the driver until a decision is due or the episode ends. Returns (reward, terminated, truncated)."""
        e, p, c = self.base, self.pilot, self.cfg
        total, steps = 0.0, 0
        min_steps = int(round(c.min_decide_s / self.dt))
        k_prog = e.cfg.reward.k_prog
        explore = c.explore_bonus > 0 and not (c.explore_without_cue_only and self._cue)
        while True:
            a = p.drive()                                     # sensing for this step was done by update()
            g0 = self._info.get("geodesic")
            self._base_obs, r, te, tr, info = e.step(a)
            if c.progress == "record" and g0 is not None:    # swap the dense progress term for record progress
                g1 = info["geodesic"]
                r += -k_prog * (g0 - g1) + k_prog * max(0.0, self._best_geo - g1)
                self._best_geo = min(self._best_geo, g1)
            if explore:
                s = e.sim.state
                cell = (int(np.floor(s.x / c.explore_cell)), int(np.floor(s.y / c.explore_cell)))
                if cell not in self._cells:
                    self._cells.add(cell)
                    r += c.explore_bonus
                    self._ep["explore"] += c.explore_bonus
            total += r
            steps += 1
            self._info = info
            if te or tr:
                return total, te, tr
            p.update(self._base_obs)                          # sense the new step; sets decision_due
            if steps >= min_steps and p.decision_due:
                return total, False, False

    @staticmethod
    def resolve(k: int, valid) -> int:
        """Slot actually taken for action k under empty_slot = "remap" (valid: 5 booleans, slot 4 always valid)."""
        if k == N_SLOTS - 1 or valid[k]:
            return k
        front = [j for j in range(N_SLOTS - 1) if valid[j]]
        if not front:
            return N_SLOTS - 1                                # dead end: turn back
        return min(front, key=lambda j: (abs(j - k), j))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._base_obs, self._info = self.base.reset(seed=seed, options=options)
        self.pilot.reset(self.base)
        self.pilot.update(self._base_obs)
        self._prev = {"x": 0.0, "y": 0.0, "h": 0.0, "junction": 0.0, "back": 0.0}
        self._t_last = 0.0
        self.n_decisions = 0
        self._points, self._departures, self._pending = [], [], None
        self._labels = np.zeros(AUX_DIM, np.float32)
        self._best_geo = float(self._info.get("geodesic", np.inf))
        self._cells = set()
        self._ep = {"repeat_dead": 0, "repeat_taken": 0, "repeat_penalty": 0.0, "explore": 0.0,
                    "dead_end_revisits": 0, "dithers": 0}
        self._tips, self._revisits = [], 0
        self._last_back, self._junction_since_back = -10, True
        # the cue draw uses its own generator (seeded from the episode) so it never shifts the base env's stream
        rs = np.random.default_rng(None if seed is None else seed + 7919)
        self._cue = self.cfg.goal_cue_prob >= 1.0 or rs.random() < self.cfg.goal_cue_prob
        if self.cfg.explore_bonus > 0:                        # the start cell is not a discovery
            s0 = self.base.sim.state
            self._cells.add((int(np.floor(s0.x / self.cfg.explore_cell)), int(np.floor(s0.y / self.cfg.explore_cell))))
        return self._obs(), dict(self._info, goal_cue=self._cue)

    def step(self, action):
        k = int(action)
        slots = self._slots()
        p, d, nav = self.pilot, self.pilot.driver, self.base._nav
        n_front = sum(x is not None for x in slots[:N_SLOTS - 1])
        penalty = 0.0
        requested = k
        if self.cfg.empty_slot == "remap":
            k = self.resolve(k, [x is not None for x in slots[:N_SLOTS - 1]] + [True])
        if k < N_SLOTS - 1 and slots[k] is None:              # (carry_on mode) empty slot: carry on, small penalty
            target, back = d.heading, False
            penalty = self.cfg.invalid_penalty
        elif k == N_SLOTS - 1:
            target = slots[k].heading if slots[k] is not None else float(wrap(d.psi + np.pi))
            back = True
        else:
            target, back = slots[k].heading, False
        # repeat penalties from the true labels at this decision (computed with the observation the agent saw)
        carry_on = penalty > 0
        lab, c = self._labels, self.cfg
        taken, dead = lab[AUX["taken"]], lab[AUX["dead_known"]]
        front = k < N_SLOTS - 1
        if front and c.repeat_dead_penalty and dead[k] > 0.5:
            penalty += c.repeat_dead_penalty
            self._ep["repeat_dead"] += 1
            self._ep["repeat_penalty"] += c.repeat_dead_penalty
        elif front and c.repeat_taken_penalty and taken[k] > 0.5 and any(
                slots[j] is not None and taken[j] < 0.5 for j in range(N_SLOTS - 1)):
            penalty += c.repeat_taken_penalty
            self._ep["repeat_taken"] += 1
            self._ep["repeat_penalty"] += c.repeat_taken_penalty
        if k == N_SLOTS - 1:                                  # U-turn of a U-turn (not the forced one at a dead end)
            if c.dither_penalty and n_front > 0 and self.n_decisions - self._last_back <= 2                     and not self._junction_since_back:
                penalty += c.dither_penalty
                self._ep["dithers"] += 1
                self._ep["repeat_penalty"] += c.dither_penalty
            self._last_back, self._junction_since_back = self.n_decisions, n_front >= 2
        self._record_departure(k)
        heading_nav = nav.psi + float(wrap(target - d.psi))   # chosen heading in the navigation frame
        self._prev = {"x": nav.x, "y": nav.y, "h": heading_nav, "junction": float(n_front >= 2),
                      "back": float(back)}
        p.commit(target, self.cfg.cruise_speed, turned=back)
        self._t_last = p.t
        self.n_decisions += 1
        reward, te, tr = self._drive_until_decision()
        obs = self._obs()                                     # (updates the dead-end tips: revisits of this step)
        if self._revisits:
            if c.dead_end_revisit_penalty:
                penalty += c.dead_end_revisit_penalty * self._revisits
                self._ep["repeat_penalty"] += c.dead_end_revisit_penalty * self._revisits
            self._ep["dead_end_revisits"] += self._revisits
            self._revisits = 0
        info = dict(self._info, n_decisions=self.n_decisions, invalid=requested != k or carry_on, choice=k,
                    requested=requested, goal_cue=self._cue, **self._ep)
        return obs, float(reward - penalty), te, tr, info

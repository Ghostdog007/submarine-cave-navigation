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
        if not cfg.env.navigation.enabled:
            cfg = replace(cfg, env=replace(cfg.env, navigation=replace(cfg.env.navigation, enabled=True)))
        self.cfg = cfg
        self.base = CaveEnv(cfg.env)
        dcfg = replace(cfg.driver, cruise_speed=cfg.cruise_speed)
        kw = dict(mode="external", decide_s=cfg.decide_s, hold_after_turn_s=cfg.hold_after_turn_s)
        if cfg.driver_actor:
            from submarine.control.learned import LearnedDriverPolicy

            self.pilot = LearnedDriverPolicy(cfg.driver_actor, dcfg, **kw)
        else:
            self.pilot = DriverPolicy(dcfg, **kw)
        self.observation_space = gym.spaces.Box(-2.0, 2.0, (OBS_DIM,), np.float32)
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
        if goal is not None:
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
        return np.clip(o, -2.0, 2.0)

    def _drive_until_decision(self):
        """Run the driver until a decision is due or the episode ends. Returns (reward, terminated, truncated)."""
        e, p = self.base, self.pilot
        total, steps = 0.0, 0
        min_steps = int(round(self.cfg.min_decide_s / self.dt))
        while True:
            a = p.drive()                                     # sensing for this step was done by update()
            self._base_obs, r, te, tr, info = e.step(a)
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
        return self._obs(), dict(self._info)

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
        heading_nav = nav.psi + float(wrap(target - d.psi))   # chosen heading in the navigation frame
        self._prev = {"x": nav.x, "y": nav.y, "h": heading_nav, "junction": float(n_front >= 2),
                      "back": float(back)}
        p.commit(target, self.cfg.cruise_speed, turned=back)
        self._t_last = p.t
        self.n_decisions += 1
        reward, te, tr = self._drive_until_decision()
        info = dict(self._info, n_decisions=self.n_decisions, invalid=requested != k or penalty > 0, choice=k,
                    requested=requested)
        return self._obs(), float(reward - penalty), te, tr, info

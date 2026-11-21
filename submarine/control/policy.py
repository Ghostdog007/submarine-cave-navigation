"""Driver + scripted "teacher navigator" as an ordinary Policy, so the driver runs in CaveEnv and evaluate().

The teacher picks exits at decision points, like the learned navigator will:
  * a new scan shows a different number of exits than at the last decision (junction appeared / passed),
  * the followed exit has disappeared (no exit within 45 degrees of the followed heading),
  * `decide_s` seconds since the last decision,
  * dead end (no exit ahead or to the sides): turn back, then hold that decision for `hold_after_turn_s`.
Modes:
  goal    exit closest to the goal bearing, with a preference for continuing straight on
  random  with probability p_random a random exit and a random cruise speed (diverse commands for phase A1)
  external  no decisions here: update() only sets `decision_due`; the caller decides and calls commit()

It has no memory: in a maze it can re-enter dead ends. That is the navigator's job, not the driver's; this
policy is for checking the driver (crashes, damage, stalls, filter interventions), not for route finding.
"""

from __future__ import annotations

import numpy as np

from submarine.baselines.base import ObsLayout, Policy

from .controller import wrap
from .driver import Driver, DriverConfig


class DriverPolicy(Policy):
    privileged = False

    def __init__(self, cfg: DriverConfig | None = None, mode: str = "goal", p_random: float = 0.3,
                 speed_range: tuple[float, float] = (0.5, 1.5), decide_s: float = 2.0,
                 hold_after_turn_s: float = 4.0, straight_weight: float = 0.5, confirm_scans: int = 2,
                 seed: int | None = None):
        self.cfg = cfg or DriverConfig()
        self.confirm_scans = confirm_scans
        self.mode, self.p_random, self.speed_range = mode, p_random, speed_range
        self.decide_s, self.hold_after_turn_s, self.straight_weight = decide_s, hold_after_turn_s, straight_weight
        self.seed = seed
        self.name = "driver" if mode == "goal" else f"driver-{mode}"
        self.driver = None
        self.total_steps = self.total_overrides = 0       # across episodes (for evaluation summaries)

    def reset(self, env):
        e = env.unwrapped
        if self.driver is not None:
            self.total_steps += self.driver.n_steps
            self.total_overrides += self.driver.n_overrides
        self.L = ObsLayout.from_env(env)
        if self.driver is None:
            self.driver = Driver(self.cfg, e.cfg.vehicle, e.cfg.sonar)
        self.driver.reset()
        seed = self.seed if self.seed is not None else int(e.np_random.integers(2**31))
        self.rng = np.random.default_rng(seed)
        self.dt = e.cfg.vehicle.dt
        self.t = 0.0
        self.t_decided = -np.inf
        self.hold_until = -np.inf
        self.n_exits = -1
        self._changed_scans = 0
        self.decisions = []                   # (time, n_exits, chosen heading, turned back) for inspection

    # ------------------------------------------------------------------ decisions
    def _need_decision(self, new_ping: bool) -> bool:
        d = self.driver
        if self.t < self.hold_until:
            return False
        if self.t_decided == -np.inf:
            return True
        if not new_ping:
            return False
        front = [x for x in d.exits if not x.opening.rear]
        lost = not any(abs(wrap(x.heading - d.heading)) < np.deg2rad(45) for x in front)
        changed = len(front) != self.n_exits or lost
        # a change must persist for `confirm_scans` scans, so one noisy scan can't flip the route
        self._changed_scans = self._changed_scans + 1 if changed else 0
        return self._changed_scans >= self.confirm_scans or self.t - self.t_decided >= self.decide_s

    def _decide(self, goal_bearing: float | None):
        d = self.driver
        front = [x for x in d.exits if not x.opening.rear]
        rear = [x for x in d.exits if x.opening.rear]
        speed = d.cfg.cruise_speed
        turned = False
        if not front:                                     # dead end: turn back
            target = rear[0].heading if rear else float(wrap(d.psi + np.pi))
            turned = True
        elif self.mode == "random" and self.rng.random() < self.p_random:
            target = front[int(self.rng.integers(len(front)))].heading
            speed = float(self.rng.uniform(*self.speed_range))
        else:
            goal = d.psi + goal_bearing if goal_bearing is not None else d.heading
            score = [np.cos(wrap(x.heading - goal)) + self.straight_weight * np.cos(wrap(x.heading - d.heading))
                     for x in front]
            target = front[int(np.argmax(score))].heading
        self.commit(target, speed, turned)

    def commit(self, target, speed: float, turned: bool = False):
        """Apply a decision (made here, or by the learned navigator in mode "external")."""
        d = self.driver
        front = [x for x in d.exits if not x.opening.rear]
        d.follow(target, speed)
        if turned:
            self.hold_until = self.t + self.hold_after_turn_s
        heading = target.heading if hasattr(target, "heading") else float(target)
        self.t_decided, self.n_exits, self._changed_scans = self.t, len(front), 0
        self.decisions.append((self.t, len(front), heading, turned))

    # ------------------------------------------------------------------ act
    def update(self, obs):
        """Sense and (at decision points) decide; the driver's command is then up to date for this step."""
        L, d = self.L, self.driver
        age = L.ping_age(obs)
        d.sense(L.ranges(obs), L.speed(obs), L.yaw_rate(obs), age)
        due = self._need_decision(age <= 1e-9)
        if self.mode == "external":                       # the learned navigator decides (see navigator env)
            self.decision_due = due
        elif due:
            goal = L.goal(obs)
            self._decide(goal[0] if goal is not None else None)

    def act(self, obs):
        self.update(obs)
        return self.drive()

    def drive(self):
        """One control step with the current command (after update() has sensed this step)."""
        a, self.last_info = self.driver.act(self.proposal())
        self.t += self.dt
        return a

    def proposal(self):
        """The command before the filter: the hand controller here; the learned driver overrides this."""
        return None

    @property
    def override_rate(self) -> float:
        """Fraction of steps the safety filter changed the controller's command, over all episodes so far."""
        d = self.driver
        steps = self.total_steps + (d.n_steps if d else 0)
        return (self.total_overrides + (d.n_overrides if d else 0)) / max(steps, 1)

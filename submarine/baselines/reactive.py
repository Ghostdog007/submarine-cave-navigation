"""Sensor-only baselines (docs/context.md, section 7): random, gap-following, centering.

All of them read only the observation vector, like the RL agent does.
"""

from __future__ import annotations

import numpy as np

from .base import ObsLayout, Policy


def corridor_front(r: np.ndarray, th: np.ndarray, half_width: float, max_range: float) -> float:
    """Distance to the nearest return inside a straight corridor of the given half-width ahead.

    Unlike "minimum range over a forward cone", side walls of a narrow passage
    don't count as blocking unless they actually intrude on the vehicle's path.
    """
    ahead = np.abs(th) < np.pi / 2
    x = r * np.cos(th)
    lateral = np.abs(r * np.sin(th))
    hits = x[ahead & (lateral < half_width) & (r < max_range - 1e-6)]
    return float(hits.min()) if hits.size else max_range


def corridor_ranges(r: np.ndarray, th: np.ndarray, half_width: float, max_range: float) -> np.ndarray:
    """corridor_front evaluated along every beam direction: how far the vehicle could go that way."""
    rel = th[None, :] - th[:, None]                               # (candidate, beam)
    x = r[None, :] * np.cos(rel)
    hit = (np.cos(rel) > 0) & (np.abs(r[None, :] * np.sin(rel)) < half_width) & (r[None, :] < max_range - 1e-6)
    return np.where(hit, x, max_range).min(axis=1)


def _thrust_from_front(front: float, stop: float, slow: float) -> float:
    """Full thrust when the way ahead is open, tapering to a crawl, reversing if nearly touching."""
    if front < stop:
        return -0.5
    return float(np.clip((front - stop) / (slow - stop), 0.15, 1.0))


class RandomPolicy(Policy):
    name = "random"

    def reset(self, env):
        self.rng = np.random.default_rng(int(env.unwrapped.np_random.integers(2**31)))

    def act(self, obs):
        return self.rng.uniform(-1.0, 1.0, 2).astype(np.float32)


class GapFollower(Policy):
    """Steer toward the most open sonar direction, slow down when the front is blocked.

    A direction's openness is how far a vehicle-width corridor reaches that
    way (the same test used for the speed), so the chosen heading is always
    one the vehicle actually fits through. The chosen direction maximises
    that distance, with a preference for small turns and (when the goal cue
    is available) for the goal bearing, which also stops it turning back in
    wide chambers. A small side-repulsion term keeps it off the walls.
    """

    name = "gap"

    def __init__(self, k_yaw=1.5, k_side=0.4, goal_weight=2.5, forward_weight=0.4,
                 range_cap=6.0, search_deg=100.0, half_width=0.6, stop=0.6, slow=3.0):
        self.k_yaw, self.k_side = k_yaw, k_side
        self.goal_weight, self.forward_weight = goal_weight, forward_weight
        self.range_cap, self.search = range_cap, np.deg2rad(search_deg)
        self.half_width = half_width
        self.stop, self.slow = stop, slow

    def reset(self, env):
        self.L = ObsLayout.from_env(env)

    def act(self, obs):
        L = self.L
        r = L.ranges(obs)
        th = L.angles
        r_eff = corridor_ranges(r, th, self.half_width, L.max_range)

        cand = np.abs(th) <= self.search
        score = np.minimum(r_eff, self.range_cap) - self.forward_weight * np.abs(th)
        goal = L.goal(obs)
        if goal is not None:
            score = score + self.goal_weight * np.cos(th - goal[0])
        score = np.where(cand, score, -np.inf)
        target = float(th[np.argmax(score)])

        yaw = self.k_yaw * target
        left = r[(th > np.deg2rad(30)) & (th < np.deg2rad(120))]
        right = r[(th < -np.deg2rad(30)) & (th > -np.deg2rad(120))]
        if left.size and right.size:
            yaw += self.k_side * (1 / max(right.min(), 0.3) - 1 / max(left.min(), 0.3))

        front = corridor_front(r, th, self.half_width, L.max_range)
        thrust = _thrust_from_front(front, self.stop, self.slow)
        if thrust > 0:
            thrust *= max(0.4, np.cos(target))
        return np.array([thrust, np.clip(yaw, -1, 1)], dtype=np.float32)


class Centering(Policy):
    """Keep equal distance to the left and right walls; turn toward the open side when blocked.

    Ignores the goal cue: it simply follows the passage it is in.
    """

    name = "centering"

    def __init__(self, k_center=1.5, k_open=1.5, side_deg=(45.0, 90.0), half_width=0.6, stop=0.6, slow=3.0):
        self.k_center, self.k_open = k_center, k_open
        self.side = np.deg2rad(side_deg)
        self.half_width = half_width
        self.stop, self.slow = stop, slow

    def reset(self, env):
        self.L = ObsLayout.from_env(env)
        self.left_idx = [self.L.beam_near(a) for a in self.side]
        self.right_idx = [self.L.beam_near(-a) for a in self.side]

    def act(self, obs):
        L = self.L
        r = L.ranges(obs)
        left, right = r[self.left_idx].mean(), r[self.right_idx].mean()
        center = (left - right) / max(left + right, 1e-6)      # > 0: more room on the left
        front = corridor_front(r, L.angles, self.half_width, L.max_range)
        blocked = max(0.0, 1.0 - front / self.slow)
        yaw = self.k_center * center + self.k_open * np.sign(left - right) * blocked
        thrust = _thrust_from_front(front, self.stop, self.slow)
        return np.array([thrust, np.clip(yaw, -1, 1)], dtype=np.float32)

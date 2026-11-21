"""Navigator (phase B): decision-level environment over the frozen driver, plus scripted reference navigators."""

import numpy as np

from .nav_env import N_SLOTS, OBS, OBS_DIM, SLOT_DIM, NavigatorEnv, NavigatorEnvConfig


def slot_valid(obs: np.ndarray) -> np.ndarray:
    return obs[OBS["slots"]].reshape(N_SLOTS, SLOT_DIM)[:, 0] > 0.5


class GreedyNavigator:
    """Exit closest to the straight-line goal bearing (the A0 teacher's rule); turn back at dead ends. No memory."""

    name = "nav-greedy"

    def __init__(self, straight_weight: float = 0.5):
        self.w = straight_weight

    def reset(self):
        pass

    def act(self, obs: np.ndarray) -> int:
        s = obs[OBS["slots"]].reshape(N_SLOTS, SLOT_DIM)
        front = [k for k in range(N_SLOTS - 1) if s[k, 0] > 0.5]
        if not front:
            return N_SLOTS - 1
        gc, gs = obs[OBS["goal"]][:2]
        goal = np.arctan2(gs, gc)
        score = [np.cos(np.arctan2(s[k, 2], s[k, 1]) - goal) + self.w * s[k, 1] for k in front]
        return int(front[int(np.argmax(score))])


class OracleNavigator:
    """Map-aware teacher (privileged: reads the true map, so only for demonstrations and as an upper bound).
    For each offered exit it marches along the exit's direction from the true position (stopping at the first
    point the vehicle can't occupy, so it never looks through rock) and scores the exit by the lowest
    shortest-path distance to the goal seen on the way; the turn-back slot is scored the same way, straight
    behind. It picks the best-scoring exit: dead-end branches score badly (the geodesic grows into them)."""

    name = "nav-oracle"

    def __init__(self, probe: float = 4.0, step: float = 0.25):
        self.probe, self.step = probe, step
        self.env = None

    def attach(self, env):
        self.env = env

    def reset(self):
        pass

    def scores(self, obs: np.ndarray) -> np.ndarray:
        e = self.env.base
        sim, s = e.sim, e.sim.state
        slots = obs[OBS["slots"]].reshape(N_SLOTS, SLOT_DIM)
        out = np.full(N_SLOTS, np.inf)
        for k in range(N_SLOTS):
            if not (slots[k, 0] > 0.5 or k == N_SLOTS - 1):
                continue
            a = s.psi + float(np.arctan2(slots[k, 2], slots[k, 1]))
            depth = float(slots[k, 5]) * 10.0
            L = min(self.probe, depth) if depth > 0 else self.probe
            best = np.inf
            for d in np.arange(self.step, L + 1e-9, self.step):
                g = sim.geodesic_at(s.x + d * np.cos(a), s.y + d * np.sin(a))
                if not np.isfinite(g):
                    break
                best = min(best, g)
            out[k] = best
        return out

    def act(self, obs: np.ndarray) -> int:
        sc = self.scores(obs)
        return int(np.argmin(sc)) if np.isfinite(sc).any() else N_SLOTS - 1


class RandomNavigator:
    name = "nav-random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs: np.ndarray) -> int:
        valid = np.flatnonzero(slot_valid(obs))
        return int(self.rng.choice(valid)) if valid.size else N_SLOTS - 1


__all__ = ["N_SLOTS", "OBS", "OBS_DIM", "SLOT_DIM", "GreedyNavigator", "NavigatorEnv", "NavigatorEnvConfig",
           "OracleNavigator", "RandomNavigator", "slot_valid"]

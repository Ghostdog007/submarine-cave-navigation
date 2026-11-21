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
           "RandomNavigator", "slot_valid"]

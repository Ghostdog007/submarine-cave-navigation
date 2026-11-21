"""Adapter so a trained Stable-Baselines3 model can be evaluated like a baseline."""

from __future__ import annotations

import numpy as np

from submarine.baselines.base import Policy


class SB3Policy(Policy):
    """Wraps model.predict.

    - Recurrent (RecurrentPPO) policies: carries the hidden state across steps.
    - frame_stack > 1: stacks observations exactly like VecFrameStack does
      (oldest first, zero-filled at the start of an episode).
    """

    def __init__(self, model, name: str = "rl", deterministic: bool = True, frame_stack: int = 1):
        self.model, self.name, self.deterministic = model, name, deterministic
        self.frame_stack = frame_stack

    def reset(self, env):
        self.state = None
        self.episode_start = np.ones(1, dtype=bool)
        self.stack = None

    def _stacked(self, obs):
        if self.frame_stack <= 1:
            return obs
        if self.stack is None:
            self.stack = np.zeros((self.frame_stack, obs.shape[-1]), dtype=obs.dtype)
        self.stack = np.roll(self.stack, -1, axis=0)
        self.stack[-1] = obs
        return self.stack.reshape(-1)

    def act(self, obs):
        action, self.state = self.model.predict(self._stacked(obs), state=self.state,
                                                episode_start=self.episode_start,
                                                deterministic=self.deterministic)
        self.episode_start = np.zeros(1, dtype=bool)
        return action

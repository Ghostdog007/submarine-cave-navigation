"""RecurrentPPO policies whose "LSTM" is a retention or junction memory (submarine/memory/modules.py).

sb3-contrib keeps the recurrent state in (h, c) buffers of size lstm_hidden_size and calls
_process_sequence(features, states, episode_starts, lstm). We build the normal policy, then swap the actor and
critic LSTMs for memory modules with the same call signature and pass episode starts into them, so resets inside
a training sequence are handled by the module itself.
"""

from __future__ import annotations

import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from .modules import JunctionMemory, RetentionMemory


class _MemoryPolicy(RecurrentActorCriticPolicy):
    memory_cls = None

    def __init__(self, observation_space, action_space, lr_schedule, memory_kwargs: dict | None = None,
                 **kwargs):
        self.memory_kwargs = dict(memory_kwargs or {})
        hidden = self.memory_kwargs.get("hidden_size", 256 if self.memory_cls is RetentionMemory else 512)
        self.memory_kwargs["hidden_size"] = hidden
        kwargs.update(lstm_hidden_size=hidden, n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=True)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.lstm_actor = self.memory_cls(self.features_dim, **self.memory_kwargs)
        self.lstm_critic = self.memory_cls(self.features_dim, **self.memory_kwargs)
        # the optimizer was built over the LSTMs' parameters: rebuild it over the memory modules'
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        d = super()._get_constructor_parameters()
        d["memory_kwargs"] = self.memory_kwargs
        return d

    @staticmethod
    def _process_sequence(features, lstm_states, episode_starts, lstm):
        n_seq = lstm_states[0].shape[1]
        x = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)       # (T, n_seq, F)
        starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)            # (T, n_seq)
        out, states = lstm(x, lstm_states, starts)
        return th.flatten(out.transpose(0, 1), start_dim=0, end_dim=1), states


class RetentionNavPolicy(_MemoryPolicy):
    memory_cls = RetentionMemory


class JunctionNavPolicy(_MemoryPolicy):
    memory_cls = JunctionMemory


POLICIES = {"RetentionNavPolicy": RetentionNavPolicy, "JunctionNavPolicy": JunctionNavPolicy}

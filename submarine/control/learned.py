"""The learned driver at run time: the trained actor exported to NumPy (no PyTorch needed).

Phase B runs the frozen driver inside the torch-free worker processes, and evaluation runs it like any other
policy, so the network is stored as plain weight arrays (`driver.npz`) and evaluated here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .driver import DriverConfig
from .driver_env import N_EXTRAS
from .policy import DriverPolicy


class NumpyActor:
    """Deterministic actor of an SB3 MlpPolicy: tanh hidden layers, linear output (the Gaussian's mean)."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray], activation: str = "tanh"):
        self.W, self.b = [np.asarray(w, np.float32) for w in weights], [np.asarray(b, np.float32) for b in biases]
        self.act = np.tanh if activation == "tanh" else (lambda x: np.maximum(x, 0.0))

    @classmethod
    def load(cls, path: str | Path) -> "NumpyActor":
        z = np.load(path)
        n = int(z["n_layers"])
        return cls([z[f"W{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)], str(z["activation"]))

    def save(self, path: str | Path, activation: str = "tanh"):
        np.savez(path, n_layers=len(self.W), activation=activation,
                 **{f"W{i}": w for i, w in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)})

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, np.float32)
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w.T + b
            if i < len(self.W) - 1:
                h = self.act(h)
        return np.clip(h, -1.0, 1.0)


def export_sb3_actor(model, path: str | Path):
    """Write the actor of an SB3 PPO MlpPolicy (mlp_extractor.policy_net + action_net) as driver.npz."""
    import torch.nn as nn

    pol = model.policy
    layers = [m for m in pol.mlp_extractor.policy_net if isinstance(m, nn.Linear)] + [pol.action_net]
    acts = {type(m).__name__ for m in pol.mlp_extractor.policy_net if not isinstance(m, nn.Linear)}
    activation = "tanh" if acts == {"Tanh"} else "relu"
    actor = NumpyActor([l.weight.detach().cpu().numpy() for l in layers],
                       [l.bias.detach().cpu().numpy() for l in layers], activation)
    actor.save(path, activation)
    return actor


class LearnedDriverPolicy(DriverPolicy):
    """The teacher navigator + driver, with the learned network in place of the hand controller.

    The network's input is built exactly as in DriverEnv; the extras (contact, hull, sonar health, tier) are
    decoded from the CaveEnv observation, where the env publishes them when those systems are enabled.
    """

    def __init__(self, actor_path: str | Path, cfg: DriverConfig | None = None, mode: str = "goal", **kw):
        super().__init__(cfg, mode=mode, **kw)
        self.actor = NumpyActor.load(actor_path)
        self.name = "driver-net" if mode == "goal" else f"driver-net-{mode}"

    def reset(self, env):
        super().reset(env)
        c = env.unwrapped.cfg
        self.ping = c.sonar.ping_interval if c.sonar.ping_interval > c.vehicle.dt + 1e-9 else 0.0
        self.damage = c.collision.mode == "damage"
        self.health = c.failures.enabled or c.supervisor.enabled
        self.n_health = int(c.use_sonar) + int(c.use_camera)
        self.supervised = c.supervisor.enabled
        L = self.L
        self.i_extra = L.n_beams + 2 + (3 if L.has_goal else 0) + 2 + (1 if L.ping_interval else 0)

    def _extras(self, obs) -> list[float]:
        v = ObsVec(obs)
        i = self.i_extra
        contact, hull = (v[i], v[i + 1]) if self.damage else (0.0, 1.0)
        i += 2 if self.damage else 0
        sonar_ok, tier = 1.0, 0.0
        if self.health:
            sonar_ok = v[i]                                   # sonar is the first range sensor
            tier = v[i + self.n_health] if self.supervised else 0.0
        out = [float(contact), float(hull), float(sonar_ok), float(tier)]
        assert len(out) == N_EXTRAS
        return out

    def update(self, obs):
        self._obs = obs                                       # the network reads contact/health from it
        super().update(obs)

    def proposal(self):
        # the hand controller still runs: its choice of direction keeps the followed heading on the passage
        # (exactly as in DriverEnv, where it is also the imitation target); only its action is discarded
        self.expert = self.driver.propose()
        self.last_input = np.clip(self.driver.features(self.ping, self._extras(self._obs)), -1.5, 1.5)
        return self.actor(self.last_input)


def ObsVec(obs):
    return obs["state"] if isinstance(obs, dict) else obs

"""Evaluate an SB3 model on many caves at once: episodes spread over torch-free worker processes,
actions for all of them computed in one batched model.predict call.

Each episode uses the same seed / cave / noise as the sequential evaluate(), so results match.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from submarine.world import seeds_for
from submarine.env import coerce_config

from .metrics import EpisodeResult, episode_result, episode_seed


def _set_row(batch, i, obs):
    if isinstance(batch, dict):
        for k in batch:
            batch[k][i] = obs[k]
    else:
        batch[i] = obs


class ParallelEvaluator:
    """Keeps a pool of worker envs alive between evaluations (e.g. during training)."""

    def __init__(self, env_config, n_workers: int = 8):
        from submarine.agents.light_vec_env import LightSubprocVecEnv

        self.cfg = replace(coerce_config(env_config), seed_split="test")
        self.venv = LightSubprocVecEnv(self.cfg, n_workers)
        self.n = n_workers

    def evaluate(self, model, difficulties, n_caves: int, split: str = "val", base_seed: int = 0,
                 name: str = "rl", deterministic: bool = True) -> list[EpisodeResult]:
        tasks = deque((s, float(d)) for d in difficulties for s in seeds_for(split, n_caves))
        venv, n = self.venv, self.n
        active: list[tuple[int, float] | None] = [None] * n
        ret, murky = np.zeros(n), np.zeros(n)
        episode_start = np.ones(n, dtype=bool)
        batch = None

        def start(i):
            nonlocal batch
            if not tasks:
                active[i] = None
                return
            s, d = tasks.popleft()
            obs, _ = venv.env_method("reset", seed=episode_seed(s, d, base_seed),
                                     options={"cave_seed": s, "difficulty": d}, indices=[i])[0]
            if batch is None:   # allocate the batch from the first observation
                batch = ({k: np.zeros((n, *v.shape), v.dtype) for k, v in obs.items()} if isinstance(obs, dict)
                         else np.zeros((n, *obs.shape), obs.dtype))
            _set_row(batch, i, obs)
            active[i], ret[i], murky[i], episode_start[i] = (s, d), 0.0, 0, True

        for i in range(n):
            start(i)
        results, state = [], None
        dt = self.cfg.vehicle.dt
        while any(a is not None for a in active):
            actions, state = model.predict(batch, state=state, episode_start=episode_start,
                                           deterministic=deterministic)
            episode_start[:] = False
            next_obs, rews, dones, infos = venv.step(actions)
            for i in range(n):
                if active[i] is None:
                    continue
                ret[i] += rews[i]
                murky[i] += infos[i]["turbidity"] > 0.4
                if dones[i]:
                    s, d = active[i]
                    results.append(episode_result(name, s, d, infos[i], float(ret[i]), int(murky[i]), dt))
                    start(i)
                else:
                    _set_row(batch, i, {k: v[i] for k, v in next_obs.items()} if isinstance(next_obs, dict)
                             else next_obs[i])
        results.sort(key=lambda r: (r.difficulty, r.cave_seed))
        return results

    def close(self):
        self.venv.close()

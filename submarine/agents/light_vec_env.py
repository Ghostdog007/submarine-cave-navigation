"""Multi-process vectorised env with torch-free workers (see env/vec_worker.py).

Same interface and behaviour as SB3's SubprocVecEnv (it reuses all of its
methods), but each worker builds the CaveEnv directly from a picklable
EnvConfig and never imports torch, so a worker costs ~150 MB instead of ~2 GB
of memory commit on Windows. Episode statistics come from VecMonitor in the
main process instead of a Monitor wrapper inside each worker.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from submarine.env.vec_worker import multi_worker, worker


class LightSubprocVecEnv(SubprocVecEnv):
    # Each worker steps one env on one core. Without these, NumPy's BLAS library reserves a buffer for every
    # CPU thread in every worker (~1.5 GB of memory commit per worker on a 32-thread machine).
    WORKER_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def __init__(self, env_config, n_envs: int, start_method: str = "spawn"):
        self.waiting = False
        self.closed = False
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        saved = {k: os.environ.get(k) for k in self.WORKER_ENV}
        os.environ.update(self.WORKER_ENV)           # inherited by the spawned workers
        try:
            for work_remote, remote in zip(self.work_remotes, self.remotes):
                p = ctx.Process(target=worker, args=(work_remote, remote, env_config), daemon=True)
                p.start()
                self.processes.append(p)
                work_remote.close()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, n_envs, observation_space, action_space)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        """Workers hold bare CaveEnvs. Answer locally: SB3 would otherwise send the wrapper *class*
        (e.g. VecMonitor asks about Monitor) to every worker, and unpickling it imports
        stable-baselines3 + torch there (~870 MB of memory commit per worker)."""
        return [False] * len(self._get_indices(indices))

    def diagnostics(self) -> list[dict]:
        for r in self.remotes:
            r.send(("diag", None))
        return [r.recv() for r in self.remotes]


class MultiEnvVecEnv(VecEnv):
    """Like LightSubprocVecEnv, but each torch-free worker process steps `envs_per_worker` envs in turn.

    A vector step waits for the slowest worker. With one env per worker, the occasional slow step (a sonar ping,
    a safety-filter fallback, a reset that generates a new cave: ~180 ms) of *any* of the workers sets the pace;
    with several envs per worker their step times average out. Env i is env (i % k) of worker (i // k).
    """

    WORKER_ENV = LightSubprocVecEnv.WORKER_ENV

    def __init__(self, env_config, n_envs: int, envs_per_worker: int = 2, start_method: str = "spawn"):
        if n_envs % envs_per_worker:
            raise ValueError("n_envs must be a multiple of envs_per_worker")
        self.k = envs_per_worker
        self.n_workers = n_envs // envs_per_worker
        self.waiting = self.closed = False
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_workers)])
        self.processes = []
        saved = {k: os.environ.get(k) for k in self.WORKER_ENV}
        os.environ.update(self.WORKER_ENV)
        try:
            for work_remote, remote in zip(self.work_remotes, self.remotes):
                p = ctx.Process(target=multi_worker, args=(work_remote, remote, env_config, self.k), daemon=True)
                p.start()
                self.processes.append(p)
                work_remote.close()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(n_envs, observation_space, action_space)

    # ------------------------------------------------------------------ stepping
    def step_async(self, actions):
        for w, remote in enumerate(self.remotes):
            remote.send(("step", actions[w * self.k:(w + 1) * self.k]))
        self.waiting = True

    def step_wait(self):
        from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs as _flatten_obs

        results = [r for remote in self.remotes for r in remote.recv()]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self):
        from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs as _flatten_obs

        for w, remote in enumerate(self.remotes):
            remote.send(("reset", [(self._seeds[i], self._options[i]) for i in range(w * self.k, (w + 1) * self.k)]))
        results = [r for remote in self.remotes for r in remote.recv()]
        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        self.closed = True

    # ------------------------------------------------------------------ attributes / methods
    def _by_worker(self, indices):
        groups: dict[int, list[int]] = {}
        for i in self._get_indices(indices):
            groups.setdefault(i // self.k, []).append(i % self.k)
        return groups

    def get_attr(self, attr_name, indices=None):
        groups = self._by_worker(indices)
        for w, local in groups.items():
            self.remotes[w].send(("get_attr", (attr_name, local)))
        return [x for w in groups for x in self.remotes[w].recv()]

    def set_attr(self, attr_name, value, indices=None):
        groups = self._by_worker(indices)
        for w, local in groups.items():
            self.remotes[w].send(("set_attr", (attr_name, value, local)))
        for w in groups:
            self.remotes[w].recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        groups = self._by_worker(indices)
        for w, local in groups.items():
            self.remotes[w].send(("env_method", (method_name, method_args, method_kwargs, local)))
        return [x for w in groups for x in self.remotes[w].recv()]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def get_images(self):
        return [None] * self.num_envs

    def diagnostics(self):
        for r in self.remotes:
            r.send(("diag", None))
        return [r.recv() for r in self.remotes]

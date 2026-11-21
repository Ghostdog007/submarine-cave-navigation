"""Subprocess worker for agents.light_vec_env.LightSubprocVecEnv.

Importing this module must NOT import torch / stable-baselines3: on Windows
every worker process re-imports what it needs, and the CUDA build of torch
costs ~2 GB of memory commit per process. The env itself is pure NumPy.
Speaks the same command protocol as SB3's SubprocVecEnv worker.
"""

from __future__ import annotations

import sys


def build_env(env_config):
    """An EnvConfig builds a CaveEnv; ("package.module:Class", config) builds any other pure-NumPy env
    (e.g. the driver's training env), so the worker stays torch-free whatever it steps."""
    if isinstance(env_config, tuple):
        import importlib

        target, cfg = env_config
        mod, cls = target.split(":")
        return getattr(importlib.import_module(mod), cls)(cfg)
    from submarine.env.cave_env import CaveEnv

    return CaveEnv(env_config)


def multi_worker(remote, parent_remote, env_config, n_envs: int) -> None:
    """Steps `n_envs` envs one after another per command (see agents.light_vec_env.MultiEnvVecEnv)."""
    parent_remote.close()
    envs = [build_env(env_config) for _ in range(n_envs)]
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for env, action in zip(envs, data):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    reset_info = {}
                    if done:
                        info["terminal_observation"] = obs
                        obs, reset_info = env.reset()
                    out.append((obs, reward, done, info, reset_info))
                remote.send(out)
            elif cmd == "reset":
                out = []
                for env, (seed, options) in zip(envs, data):
                    kw = {"options": options} if options else {}
                    out.append(env.reset(seed=seed, **kw))
                remote.send(out)
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "env_method":
                name, args, kwargs, local = data
                remote.send([getattr(envs[i], name)(*args, **kwargs) for i in local])
            elif cmd == "get_attr":
                name, local = data
                remote.send([getattr(envs[i], name) for i in local])
            elif cmd == "set_attr":
                name, value, local = data
                for i in local:
                    setattr(envs[i], name, value)
                remote.send(None)
            elif cmd == "diag":
                remote.send({"torch_loaded": "torch" in sys.modules, "imported_before_torch": []})
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the multi-env worker")
        except (EOFError, KeyboardInterrupt):
            break


def worker(remote, parent_remote, env_config) -> None:
    parent_remote.close()
    env = build_env(env_config)
    reset_info: dict = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = obs
                    obs, reset_info = env.reset()
                remote.send((obs, reward, done, info, reset_info))
            elif cmd == "reset":
                kw = {"options": data[1]} if data[1] else {}
                remote.send(env.reset(seed=data[0], **kw))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                remote.send(getattr(env, data[0])(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "has_attr":
                remote.send(hasattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(False)
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "diag":
                mods = list(sys.modules)
                first = next((i for i, m in enumerate(mods) if m == "torch" or m.startswith("torch.")), None)
                before = mods[max(0, first - 12):first] if first is not None else []
                remote.send({"torch_loaded": "torch" in sys.modules, "imported_before_torch": before})
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except (EOFError, KeyboardInterrupt):
            break

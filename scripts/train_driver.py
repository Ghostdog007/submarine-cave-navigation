"""Train the learned driver: phase A1 (imitation warm start) then A2/A3 (PPO curriculum, safety filter inside).

    python -m scripts.train_driver --config configs/driver.yaml
    python -m scripts.train_driver --config configs/driver.yaml --scale 0.05      # smoke test, 5% of the steps

A1  the hand controller drives (with noise on its executed actions; labels stay clean) in parallel processes;
    the network's mean action is fitted to the controller's proposals and its value head to the observed
    discounted returns, so the first PPO updates don't destroy the warm start.
A2  PPO stages from the config (tunnels -> all cave types + slow sonar + currents -> full difficulty).
A3  robustness stage(s): sensor failures, blindness supervisor, hull-damage contact model.

After every stage: <run>/<stage>.zip (SB3 model) and <run>/<stage>.npz (NumPy actor, used by evaluation and
phase B); the last stage is also written as <run>/driver.npz. Progress per stage goes to <run>/log.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

TARGET = "submarine.control.driver_env:DriverEnv"


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def stage_config(base: dict, stage: dict):
    from submarine.control import DriverEnvConfig

    d = deep_merge(base, {k: v for k, v in stage.items() if k not in ("name", "steps")})
    return DriverEnvConfig.from_dict(d)


# ---------------------------------------------------------------------------- A1: imitation data
def _collect(args):
    cfg_dict, n_steps, seed, gamma = args
    from dataclasses import replace

    from submarine.control import DriverEnv, DriverEnvConfig

    cfg = DriverEnvConfig.from_dict(cfg_dict)
    env = DriverEnv(replace(cfg, expert_drives=True))
    obs_l, act_l, ret_l = [], [], []
    ep_obs, ep_act, ep_rew = [], [], []
    obs, info = env.reset(seed=seed)
    expert = info["expert_action"]
    k = 0
    while k < n_steps:
        ep_obs.append(obs)
        ep_act.append(expert)
        obs, r, te, tr, info = env.step(np.zeros(2, np.float32))
        expert = info["expert_action"]
        ep_rew.append(r)
        k += 1
        if te or tr or k == n_steps:
            g, rets = 0.0, []
            for x in reversed(ep_rew):
                g = x + gamma * g
                rets.append(g)
            obs_l += ep_obs
            act_l += ep_act
            ret_l += rets[::-1]
            ep_obs, ep_act, ep_rew = [], [], []
            if k < n_steps:
                obs, info = env.reset()
                expert = info["expert_action"]
    return np.array(obs_l, np.float32), np.array(act_l, np.float32), np.array(ret_l, np.float32)


def imitation(model, cfg_dicts, bc: dict, gamma: float, workers: int, log):
    import torch as th

    n = int(bc["steps"])
    per = int(np.ceil(n / workers))
    jobs = [(cfg_dicts[i % len(cfg_dicts)], per, 10_000 + i, gamma) for i in range(workers)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_collect, jobs))
    obs = np.concatenate([p[0] for p in parts])
    act = np.concatenate([p[1] for p in parts])
    ret = np.concatenate([p[2] for p in parts])
    log(f"A1 collected {len(obs)} expert steps in {time.perf_counter() - t0:.0f} s "
        f"(action mean {act.mean(0).round(2)}, return mean {ret.mean():.2f})")

    pol = model.policy
    pol.set_training_mode(True)
    opt = th.optim.Adam(pol.parameters(), lr=float(bc["lr"]))
    O, A, R = (th.as_tensor(x) for x in (obs, act, ret))
    rng = np.random.default_rng(0)
    bs = int(bc["batch"])
    for epoch in range(int(bc["epochs"])):
        perm = rng.permutation(len(O))
        la = lv = 0.0
        for i in range(0, len(O), bs):
            idx = th.as_tensor(perm[i:i + bs])
            feats = pol.extract_features(O[idx], pol.features_extractor)
            lat_pi, lat_vf = pol.mlp_extractor(feats)
            mean = pol.action_net(lat_pi)
            value = pol.value_net(lat_vf).squeeze(-1)
            loss_a = ((mean - A[idx]) ** 2).mean()
            loss_v = ((value - R[idx]) ** 2).mean()
            opt.zero_grad()
            (loss_a + 0.5 * loss_v).backward()
            opt.step()
            la += float(loss_a) * len(idx)
            lv += float(loss_v) * len(idx)
        log(f"A1 epoch {epoch + 1}: action MSE {la / len(O):.4f}  value MSE {lv / len(O):.3f}")
    with th.no_grad():
        pol.log_std.fill_(float(bc["log_std"]))
    pol.set_training_mode(False)
    return float(la / len(O))


# ---------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/driver.yaml")
    p.add_argument("--scale", type=float, default=1.0, help="multiply all step counts (smoke tests)")
    p.add_argument("--name", default=None)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        C = yaml.safe_load(f)
    name = args.name or C["name"]
    run = Path("runs") / name
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(C, f, sort_keys=False)
    logf = open(run / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    from submarine.agents.light_vec_env import MultiEnvVecEnv
    from submarine.control.learned import export_sb3_actor

    th.set_num_threads(args.threads)
    base, stages = C["base"], C["stages"]
    n_envs, seed = int(C.get("n_envs", 16)), int(C.get("seed", 0))
    k = int(C.get("envs_per_worker", 1))
    ppo = dict(C.get("ppo", {}))

    def make_venv(stage):
        venv = MultiEnvVecEnv((TARGET, stage_config(base, stage)), n_envs, k)
        venv.seed(seed)
        return VecMonitor(venv)

    venv = make_venv(stages[0])
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, tensorboard_log=str(run / "tb"),
                policy_kwargs={"net_arch": {"pi": C["net_arch"], "vf": C["net_arch"]},
                               "log_std_init": float(C["bc"]["log_std"])}, **ppo)
    log(f"observation {venv.observation_space.shape}, {n_envs} envs on {n_envs // k} workers, stages: {[s['name'] for s in stages]}")

    bc = dict(C["bc"])
    bc["steps"] = max(2000, int(bc["steps"] * args.scale))
    bc_cfgs = [deep_merge(base, {k: v for k, v in s.items() if k not in ("name", "steps")})
               for s in stages if s["name"] in C.get("bc_stages", [stages[0]["name"]])]
    imitation(model, bc_cfgs, bc, float(ppo.get("gamma", 0.99)), int(C.get("bc_workers", n_envs)), log)
    model.save(run / "a1_imitation.zip")
    export_sb3_actor(model, run / "a1_imitation.npz")

    summary = {"stages": []}
    for i, stage in enumerate(stages):
        if i > 0:
            venv.close()
            venv = make_venv(stage)
            model.set_env(venv)
        steps = max(n_envs * ppo.get("n_steps", 256), int(stage["steps"] * args.scale))
        t0 = time.perf_counter()
        model.learn(steps, reset_num_timesteps=False, tb_log_name="driver")
        dt = time.perf_counter() - t0
        ep = list(model.ep_info_buffer)
        stats = {"name": stage["name"], "steps": steps, "seconds": round(dt), "steps_per_s": round(steps / dt),
                 "ep_rew_mean": float(np.mean([e["r"] for e in ep])) if ep else float("nan"),
                 "ep_len_mean": float(np.mean([e["l"] for e in ep])) if ep else float("nan")}
        summary["stages"].append(stats)
        log(f"stage {stage['name']}: {steps} steps in {dt / 60:.1f} min ({stats['steps_per_s']} steps/s), "
            f"episode reward {stats['ep_rew_mean']:.1f}, length {stats['ep_len_mean']:.0f}")
        model.save(run / f"{stage['name']}.zip")
        export_sb3_actor(model, run / f"{stage['name']}.npz")
    export_sb3_actor(model, run / "driver.npz")
    venv.close()
    with open(run / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"done: {run / 'driver.npz'}")


if __name__ == "__main__":
    main()

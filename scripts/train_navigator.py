"""Train a navigator (phase B) over the frozen driver.

    python -m scripts.train_navigator --memory none        # PPO, MLP (no memory)
    python -m scripts.train_navigator --memory retention   # RecurrentPPO, retention memory
    python -m scripts.train_navigator --memory junction    # RecurrentPPO, retention + junction memory
    ... --scale 0.02                                       # smoke test

All three use the same config (configs/navigator.yaml): same curriculum, envs, budget and seed, so differences
come from the memory alone. Steps are *decisions* (each ~1.5-2 s of driving). After every stage the model is
saved as <run>/<stage>.zip; <run>/final_model.zip is the last one. Progress goes to <run>/log.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

TARGET = "submarine.navigator.nav_env:NavigatorEnv"
MEMORY = {"none": None, "retention": "RetentionNavPolicy", "junction": "JunctionNavPolicy"}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def warm_start(model, demos_path: str, bc: dict, gamma: float, recurrent: bool, log):
    """Behaviour cloning of the map-aware teacher's choices (scripts.collect_nav_demos), plus value pretraining on
    the discounted returns of the demonstration episodes. Recurrent policies train on consecutive chunks of
    `seq_len` decisions (state reset at each chunk start), through the same forward pass PPO uses."""
    import torch as th

    d = np.load(demos_path)
    obs, label, rew, start = d["obs"], d["label"], d["reward"], d["start"]
    ret = np.zeros_like(rew)
    g = 0.0
    for i in range(len(rew) - 1, -1, -1):                    # discounted return within each episode
        g = rew[i] + gamma * g
        ret[i] = g
        if start[i]:
            g = 0.0
    L = int(bc.get("seq_len", 64)) if recurrent else 1
    n_chunks = len(obs) // L
    idx = np.arange(n_chunks * L).reshape(n_chunks, L)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_chunks)
    n_val = max(1, n_chunks // 10)
    val, train = perm[:n_val], perm[n_val:]
    pol = model.policy
    opt = th.optim.Adam(pol.parameters(), lr=float(bc.get("lr", 1e-3)))
    batch = int(bc.get("batch_seqs", 16)) if recurrent else int(bc.get("batch", 512))

    def run(chunks, grad: bool):
        rows = idx[chunks].reshape(-1)
        o = th.as_tensor(obs[rows])
        a = th.as_tensor(label[rows])
        r = th.as_tensor(ret[rows])
        with th.set_grad_enabled(grad):
            if recurrent:
                from sb3_contrib.common.recurrent.type_aliases import RNNStates

                n = len(chunks)
                hs = lambda m: (th.zeros(1, n, m.hidden_size), th.zeros(1, n, m.hidden_size))
                states = RNNStates(hs(pol.lstm_actor), hs(pol.lstm_critic))
                st = th.as_tensor(start[rows].astype(np.float32)).view(n, L)
                st[:, 0] = 1.0
                v, logp, _ = pol.evaluate_actions(o, a, states, st.reshape(-1))
            else:
                v, logp, _ = pol.evaluate_actions(o, a)
            pi_loss = -logp.mean()
            v_loss = th.nn.functional.mse_loss(v.flatten(), r)
            acc = None
            if not grad:
                dist = pol.get_distribution(o, states.pi, st.reshape(-1))[0] if recurrent else pol.get_distribution(o)
                acc = float((dist.distribution.probs.argmax(1) == a).float().mean())
        return pi_loss, v_loss, acc

    for epoch in range(int(bc.get("epochs", 8))):
        rng.shuffle(train)
        lp = lv = 0.0
        nb = 0
        for b in range(0, len(train), batch):
            pi_loss, v_loss, _ = run(train[b:b + batch], True)
            loss = pi_loss + float(bc.get("vf_coef", 0.05)) * v_loss
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
            opt.step()
            lp += float(pi_loss)
            lv += float(v_loss)
            nb += 1
        vp, vv, acc = run(val, False)
        log(f"warm start epoch {epoch + 1}: train NLL {lp / nb:.3f} value MSE {lv / nb:.1f} | held-out NLL "
            f"{float(vp):.3f} value MSE {float(vv):.1f} teacher agreement {100 * acc:.0f}%")


def stage_config(base: dict, stage: dict):
    from submarine.navigator import NavigatorEnvConfig

    return NavigatorEnvConfig.from_dict(deep_merge(base, {k: v for k, v in stage.items() if k not in ("name", "steps")}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/navigator.yaml")
    p.add_argument("--memory", choices=list(MEMORY), required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--name", default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=None, help="override worker processes (envs stay the same)")
    p.add_argument("--driver", default=None, help="override base.driver_actor (path to a driver .npz)")
    p.add_argument("--demos", default=None, help="override bc.demos (teacher demonstrations for the warm start)")
    p.add_argument("--no-warm-start", action="store_true")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        C = yaml.safe_load(f)
    name = args.name or f"nav_{args.memory}"
    run = Path("runs") / name
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(C, memory=args.memory), f, sort_keys=False)
    logf = open(run / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch as th
    from stable_baselines3.common.vec_env import VecMonitor

    from submarine.agents.light_vec_env import MultiEnvVecEnv
    from submarine.navigator import OBS

    th.set_num_threads(args.threads)
    base, stages = C["base"], C["stages"]
    if args.driver:
        base = dict(base, driver_actor=args.driver)
    if not Path(base["driver_actor"]).exists():
        raise FileNotFoundError(f"driver {base['driver_actor']} not found (train it first: scripts.train_driver)")
    log(f"driver: {base['driver_actor']}")
    n_envs, seed = int(C["n_envs"]), int(C.get("seed", 0))
    k = int(C.get("envs_per_worker", 1))
    if args.workers:
        k = n_envs // args.workers
    algo = dict(C["ppo"])

    def make_venv(stage):
        venv = MultiEnvVecEnv((TARGET, stage_config(base, stage)), n_envs, k)
        venv.seed(seed)
        return VecMonitor(venv)

    venv = make_venv(stages[0])
    policy = MEMORY[args.memory]
    if policy is None:
        from stable_baselines3 import PPO

        model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, tensorboard_log=str(run / "tb"),
                    policy_kwargs={"net_arch": C["net_arch"]}, **algo)
    else:
        from sb3_contrib import RecurrentPPO

        from submarine.memory.policies import POLICIES

        mem = dict(C.get("memory", {}).get(args.memory, {}))
        if args.memory == "junction":
            mem["obs_index"] = {k_: OBS[k_] for k_ in ("x", "y", "prev_x", "prev_y", "prev_junction", "dead_end")}
        model = RecurrentPPO(POLICIES[policy], venv, seed=seed, device="cpu", verbose=0,
                             tensorboard_log=str(run / "tb"),
                             policy_kwargs={"net_arch": C["net_arch"], "memory_kwargs": mem}, **algo)
    n_params = sum(p_.numel() for p_ in model.policy.parameters())
    log(f"memory={args.memory}: {n_params} parameters, {n_envs} envs on {n_envs // k} workers, "
        f"stages {[s['name'] for s in stages]}")
    demos = args.demos or C.get("bc", {}).get("demos")
    if demos and not args.no_warm_start:
        if not Path(demos).exists():
            raise FileNotFoundError(f"demonstrations {demos} not found (run scripts.collect_nav_demos first)")
        warm_start(model, demos, dict(C.get("bc", {})), float(algo.get("gamma", 0.99)), policy is not None, log)
        model.save(run / "b0_warm_start.zip")

    summary = {"memory": args.memory, "parameters": n_params, "stages": []}
    for i, stage in enumerate(stages):
        if i > 0:
            venv.close()
            venv = make_venv(stage)
            model.set_env(venv)
        steps = max(n_envs * algo["n_steps"], int(stage["steps"] * args.scale))
        t0 = time.perf_counter()
        model.learn(steps, reset_num_timesteps=False, tb_log_name="nav")
        dt = time.perf_counter() - t0
        ep = list(model.ep_info_buffer)
        st = {"name": stage["name"], "decisions": steps, "seconds": round(dt), "decisions_per_s": round(steps / dt),
              "ep_rew_mean": float(np.mean([e["r"] for e in ep])) if ep else float("nan"),
              "ep_len_mean": float(np.mean([e["l"] for e in ep])) if ep else float("nan")}
        summary["stages"].append(st)
        log(f"stage {stage['name']}: {steps} decisions in {dt / 60:.1f} min ({st['decisions_per_s']}/s), "
            f"episode reward {st['ep_rew_mean']:.1f}, decisions/episode {st['ep_len_mean']:.0f}")
        model.save(run / f"{stage['name']}.zip")
    model.save(run / "final_model.zip")
    venv.close()
    with open(run / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()

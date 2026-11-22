"""Train an exit-memory navigator (memory-v3) over the frozen learned driver.

    python -m scripts.collect_nav_demos --config configs/memnav.yaml --out runs/memnav_demos.npz   # once (~10 min)
    python -m scripts.train_memnav --variant exit            # learned exit memory (the full model)
    python -m scripts.train_memnav --variant graph           # explicit junction graph (rule-based reads)
    python -m scripts.train_memnav --variant gru             # GRU, no store
    python -m scripts.train_memnav --variant none            # MLP
    ... --scale 0.02 --workers 2                             # smoke test

All variants use configs/memnav.yaml: same curriculum (with the memory scenarios), budget, seed, warm start and
auxiliary losses; only the memory differs. Outputs in runs/memnav_<variant>/: b0_warm_start.pt, <stage>.pt,
final_model.pt, log.txt, tb/ (SB3-compatible tags: python -m scripts.watch_training runs/memnav_exit ...).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from scripts.train_navigator import TARGET, stage_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/memnav.yaml")
    p.add_argument("--variant", choices=["none", "gru", "graph", "exit"], required=True)
    p.add_argument("--scale", type=float, default=1.0, help="multiply every stage budget (smoke tests)")
    p.add_argument("--name", default=None)
    p.add_argument("--workers", type=int, default=None, help="worker processes (envs stay the same)")
    p.add_argument("--threads", type=int, default=2, help="torch threads in the learner")
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--demos", default=None, help="override bc.demos")
    p.add_argument("--init", default=None, help="start from these weights (e.g. a saved b0_warm_start.pt) and "
                                               "skip the warm start")
    args = p.parse_args()

    C = yaml.safe_load(open(args.config, encoding="utf-8"))
    name = args.name or f"memnav_{args.variant}"
    run = Path("runs") / name
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(C, memory=args.variant), f, sort_keys=False)     # "memory": the watcher's cue
    logf = open(run / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch as th
    from torch.utils.tensorboard import SummaryWriter

    from submarine.agents.light_vec_env import MultiEnvVecEnv
    from submarine.memory.exitmem import ExitMemoryConfig, ExitMemoryNet, save_model
    from submarine.memory.trainer import PPOConfig, RecurrentPPO, warm_start

    th.set_num_threads(args.threads)
    th.manual_seed(int(C.get("seed", 0)))
    base, stages = C["base"], C["stages"]
    if not Path(base["driver_actor"]).exists():
        raise FileNotFoundError(f"driver {base['driver_actor']} not found (copy runs/driver/driver.npz)")
    n_envs = int(C["n_envs"])
    k = n_envs // args.workers if args.workers else int(C.get("envs_per_worker", 4))
    if n_envs % k:                                            # fail now, not after a 10-minute warm start
        raise ValueError(f"n_envs ({n_envs}) must be a multiple of envs per worker ({k})")
    net = ExitMemoryNet(ExitMemoryConfig(variant=args.variant, **C.get("model", {})))
    n_params = sum(p_.numel() for p_ in net.parameters())
    log(f"variant={args.variant}: {n_params} parameters, state {net.cfg.state_size} floats/env, {n_envs} envs on "
        f"{n_envs // k} workers, stages {[s['name'] for s in stages]}")

    bc = dict(C.get("bc", {}))
    if args.demos:
        bc["demos"] = args.demos
    if args.init:
        ck = th.load(args.init, map_location="cpu", weights_only=False)
        if ck["config"]["variant"] != args.variant:
            raise ValueError(f"{args.init} is a {ck['config']['variant']} model, not {args.variant}")
        net.load_state_dict(ck["state_dict"])
        log(f"initialised from {args.init} (warm start skipped)")
    elif bc.get("demos") and not args.no_warm_start:
        if not Path(bc["demos"]).exists():
            raise FileNotFoundError(f"{bc['demos']} not found: python -m scripts.collect_nav_demos "
                                    f"--config {args.config} --out {bc['demos']}")
        demos = dict(np.load(bc["demos"]))
        if args.scale < 1.0:                                  # smoke test: a slice of the demonstrations
            n = max(2000, int(len(demos["obs"]) * args.scale))
            demos = {k_: v[:n] for k_, v in demos.items()}
        warm_start(net, demos, epochs=int(bc.get("epochs", 6)), lanes=int(bc.get("lanes", 32)),
                   T=int(bc.get("seq_len", 64)), lr=float(bc.get("lr", 1e-3)), vf_coef=float(bc.get("vf_coef", 0.05)),
                   aux_w=float(bc.get("aux_weight", 0.5)), log=log)
        save_model(net, run / "b0_warm_start.pt")

    ppo_cfg = PPOConfig(**C.get("ppo", {}))
    writer = None

    def make_venv(stage):
        venv = MultiEnvVecEnv((TARGET, stage_config(base, stage)), n_envs, k)
        venv.seed(int(C.get("seed", 0)))
        return venv

    total = sum(max(n_envs * ppo_cfg.n_steps, int(s["steps"] * args.scale)) for s in stages)
    algo, venv = None, None
    summary = {"variant": args.variant, "parameters": n_params, "stages": []}
    for i, stage in enumerate(stages):
        if venv is not None:
            venv.close()
        venv = make_venv(stage)
        if writer is not None:
            writer.close()                                    # one event file per stage (the watcher's convention)
        writer = SummaryWriter(str(run / "tb" / "memnav_0"), filename_suffix=f".{i}")
        if algo is None:
            algo = RecurrentPPO(net, venv, ppo_cfg, writer, log)
            algo.total_timesteps = total
        else:
            algo.set_env(venv)
            algo.writer = writer
        steps = max(n_envs * ppo_cfg.n_steps, int(stage["steps"] * args.scale))
        t0 = time.perf_counter()
        algo.learn(steps)
        dt = time.perf_counter() - t0
        ep = algo.ep_info
        st = {"name": stage["name"], "decisions": steps, "seconds": round(dt),
              "ep_rew_mean": float(np.mean([e["r"] for e in ep])) if ep else float("nan"),
              "ep_len_mean": float(np.mean([e["l"] for e in ep])) if ep else float("nan"),
              "success": float(np.mean([e["s"] for e in ep])) if ep else float("nan")}
        summary["stages"].append(st)
        log(f"stage {stage['name']}: {steps} decisions in {dt / 60:.1f} min ({steps / dt:.0f}/s), "
            f"episode reward {st['ep_rew_mean']:.1f}, decisions/episode {st['ep_len_mean']:.0f}, "
            f"success {100 * st['success']:.0f}%")
        save_model(net, run / f"{stage['name']}.pt")
    save_model(net, run / "final_model.pt")
    venv.close()
    writer.close()
    with open(run / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()

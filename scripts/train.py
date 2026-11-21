"""Train an agent from a YAML config.

    python -m scripts.train --config configs/ppo_d0.yaml
    python -m scripts.train --config configs/ppo_d0.yaml --seed 1 --timesteps 200000 --name quick_test

Outputs go to runs/<name>/: config.yaml, tb/ (TensorBoard), validation.csv,
best_model.zip (best on validation caves), final_model.zip, checkpoints/.
Watch training with:  tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# NumPy's BLAS reserves a buffer per CPU thread (~1.5 GB commit on a 32-thread machine); the learner needs
# few threads, so cap it before NumPy is imported (inside main). Override by setting the variable yourself.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

# NOTE: no torch / stable-baselines3 imports at module level. On Windows every
# worker process re-imports this module, and importing torch there would cost
# ~2 GB of memory commit per worker. Heavy imports live inside main().


def main():
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    from submarine.agents import CurriculumCallback, TrainConfig, ValidationCallback, make_model, make_train_env

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--name", type=str, default=None, help="run name (default: from config, plus _s<seed>)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--no-video", action="store_true", help="skip the replay videos at the end")
    p.add_argument("--vec-env", choices=["light", "subproc", "dummy"], default=None,
                   help="light = torch-free worker processes (default); dummy = all envs in the main process")
    p.add_argument("--threads", type=int, default=4,
                   help="PyTorch threads for the learner (keep low when running several trainings at once)")
    args = p.parse_args()

    import torch

    torch.set_num_threads(args.threads)

    cfg = TrainConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps
    if args.device is not None:
        cfg.device = args.device
    if args.vec_env is not None:
        cfg.vec_env = args.vec_env
    cfg.name = args.name or f"{cfg.name}_s{cfg.seed}"

    run_dir = args.runs_dir / cfg.name
    if (run_dir / "final_model.zip").exists():
        raise SystemExit(f"{run_dir} already has a finished run; pick another --name")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    venv = make_train_env(cfg)
    model = make_model(cfg, venv, tensorboard_log=str(run_dir / "tb"))
    n_params = sum(p.numel() for p in model.policy.parameters())
    space = venv.observation_space
    obs_desc = (", ".join(f"{k} {tuple(s.shape)}" for k, s in space.spaces.items()) if hasattr(space, "spaces")
                else f"{space.shape[0]}")
    print(f"run {cfg.name}: {cfg.algo} {cfg.policy}, {n_params:,} params, {cfg.n_envs} envs ({cfg.vec_env}), "
          f"{cfg.total_timesteps:,} steps, obs {obs_desc}, device {model.device}")

    cbs = [ValidationCallback(cfg.env, cfg.eval.difficulties, cfg.eval.n_caves, cfg.eval.freq, run_dir,
                              cfg.frame_stack),
           CheckpointCallback(max(cfg.checkpoint_freq // cfg.n_envs, 1), str(run_dir / "checkpoints"), "model")]
    if cfg.curriculum:
        cbs.insert(0, CurriculumCallback(cfg.curriculum, cfg.total_timesteps))
    callbacks = CallbackList(cbs)
    t0 = time.perf_counter()
    model.learn(cfg.total_timesteps, callback=callbacks, tb_log_name="train")
    model.save(run_dir / "final_model")
    venv.close()
    dt = time.perf_counter() - t0
    print(f"done in {dt / 60:.1f} min ({cfg.total_timesteps / dt:,.0f} steps/s). saved {run_dir}")
    if not args.no_video:
        record_run_videos(run_dir, cfg)


def record_run_videos(run_dir: Path, cfg: TrainConfig, n_difficulties: int = 3):
    """Replay the best model next to the gap follower on validation caves (easy / middle / hard)."""
    from submarine.world import val_seeds
    from submarine.evaluation import make_policy
    from submarine.render.replay import record_replay

    ds = sorted(cfg.eval.difficulties)
    picks = sorted({ds[0], ds[len(ds) // 2], ds[-1]})[:n_difficulties]
    for d in picks:
        out = run_dir / "videos" / f"best_vs_gap_d{d:.2f}.mp4"
        try:
            res = record_replay([make_policy(str(run_dir)), make_policy("gap")], cfg.env, val_seeds(1)[0], d, out)
            print(f"saved {out}   [" + ", ".join(f"{r['policy']}: {r['outcome']}" for r in res) + "]")
        except Exception as e:   # a video failure must never lose a finished training run
            print(f"video for d={d} failed: {e}")


if __name__ == "__main__":
    main()

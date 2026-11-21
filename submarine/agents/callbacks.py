"""Periodic evaluation on validation caves during training."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from submarine.evaluation import SB3Policy, evaluate, summarize


class CurriculumCallback(BaseCallback):
    """Switches the training envs' layouts / difficulty range / energy penalty at fractions of training.

    phases: [{"at": 0.0, "layout": [...], "difficulty": [lo, hi], "k_energy": x}, ...]; logged under curriculum/.
    """

    def __init__(self, phases: list[dict], total_timesteps: int):
        super().__init__()
        self.phases = sorted(phases, key=lambda p: p.get("at", 0.0))
        self.total = total_timesteps
        self.current = -1

    def _apply(self, i: int):
        p, env = self.phases[i], self.training_env
        if "layout" in p:
            env.env_method("set_layouts", list(p["layout"]))
        if "difficulty" in p:
            lo, hi = p["difficulty"]
            env.env_method("set_difficulty", float(lo), float(hi))
        if "k_energy" in p:
            env.env_method("set_energy_penalty", float(p["k_energy"]))
        self.current = i
        self.logger.record("curriculum/phase", i)
        print(f"[{self.num_timesteps:>9,} steps] curriculum phase {i}: "
              + ", ".join(f"{k}={v}" for k, v in p.items() if k != "at"), flush=True)

    def _on_training_start(self):
        self._apply(0)

    def _on_step(self) -> bool:
        nxt = self.current + 1
        if nxt < len(self.phases) and self.num_timesteps >= self.phases[nxt].get("at", 0.0) * self.total:
            self._apply(nxt)
        return True


class ValidationCallback(BaseCallback):
    """Every `freq` env steps, run the deterministic policy on validation caves.

    Logs success / collision / timeout / efficiency / clearance to TensorBoard
    under val/, appends them to validation.csv, and saves best_model.zip by
    (success rate, mean return).
    """

    def __init__(self, env_config, difficulties, n_caves: int, freq: int, run_dir: Path, frame_stack: int = 1,
                 n_workers: int = 8):
        """n_workers > 1: run validation episodes in parallel worker processes (much faster);
        0 or 1, or frame stacking: one episode after another in this process."""
        super().__init__()
        self.env_config, self.difficulties, self.n_caves = env_config, list(difficulties), n_caves
        self.freq, self.run_dir, self.frame_stack = freq, Path(run_dir), frame_stack
        self.n_workers = n_workers if frame_stack <= 1 else 0
        self.pool = None
        self.best = (-1.0, -float("inf"))
        self._last = 0
        self.csv_path = self.run_dir / "validation.csv"

    def _on_training_start(self):
        if self.n_workers > 1:
            from submarine.evaluation.parallel import ParallelEvaluator

            self.pool = ParallelEvaluator(self.env_config, self.n_workers)
        self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.freq:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.num_timesteps != self._last:
            self._evaluate()
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def _evaluate(self):
        self._last = self.num_timesteps
        t0 = time.perf_counter()
        if self.pool is not None:
            episodes = self.pool.evaluate(self.model, self.difficulties, self.n_caves, split="val")
        else:
            policy = SB3Policy(self.model, "rl", deterministic=True, frame_stack=self.frame_stack)
            episodes = evaluate(policy, self.env_config, self.difficulties, self.n_caves, split="val")
        rows = summarize(episodes)
        overall = next(r for r in rows if r["difficulty"] == "all")

        for key in ("success", "collision", "timeout", "efficiency", "time_to_goal",
                    "min_clearance", "mean_clearance", "return"):
            self.logger.record(f"val/{key}", overall[key])
        if len(self.difficulties) > 1:
            for r in rows:
                if r["difficulty"] != "all":
                    self.logger.record(f"val_success/d{r['difficulty']:.2f}", r["success"])
        self.logger.dump(self.num_timesteps)

        new_file = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["timesteps", *rows[0].keys()])
            if new_file:
                w.writeheader()
            for r in rows:
                w.writerow({"timesteps": self.num_timesteps, **r})

        score = (overall["success"], overall["return"])
        tag = ""
        if score > self.best:
            self.best = score
            self.model.save(self.run_dir / "best_model")
            tag = "  * new best"
        print(f"[{self.num_timesteps:>9,} steps] val success {100 * overall['success']:5.1f}%  "
              f"collision {100 * overall['collision']:5.1f}%  timeout {100 * overall['timeout']:5.1f}%  "
              f"return {overall['return']:6.1f}  ({time.perf_counter() - t0:.1f}s){tag}", flush=True)

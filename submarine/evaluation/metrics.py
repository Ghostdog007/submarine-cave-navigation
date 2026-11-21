"""Run policies on held-out caves and compute the metrics in context.md, section 7."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from submarine.world import seeds_for
from submarine.env import CaveEnv, EnvConfig, coerce_config


@dataclass
class EpisodeResult:
    policy: str
    difficulty: float
    cave_seed: int
    outcome: str                 # success | collision | timeout
    steps: int
    time: float                  # s
    path_length: float           # m
    shortest_path: float         # m, geodesic start -> goal
    efficiency: float            # path_length / shortest_path (successes only, else nan)
    min_clearance: float         # m
    mean_clearance: float        # m
    episode_return: float
    murk_fraction: float = 0.0   # share of steps spent in murky water (turbidity > 0.4)
    end_turbidity: float = 0.0   # turbidity where the episode ended (where collisions happen)
    energy_kj: float = 0.0       # electrical energy used over the episode (metric only, not in the reward)
    energy_per_m: float = float("nan")   # J per metre of route covered (successes only)
    contacts: int = 0            # wall contacts survived (collision.mode "damage")
    hull_damage: float = 0.0     # % hull lost
    trajectory: list = field(default_factory=list, repr=False)


def episode_seed(cave_seed: int, difficulty: float, base_seed: int = 0) -> int:
    """Same (cave, difficulty) -> same sensor noise and current for every policy."""
    return int(base_seed * 1_000_003 + cave_seed * 101 + round(difficulty * 100))


def run_episode(env: CaveEnv, policy, cave_seed: int, difficulty: float,
                base_seed: int = 0, keep_trajectory: bool = False) -> EpisodeResult:
    obs, info = env.reset(seed=episode_seed(cave_seed, difficulty, base_seed),
                          options={"cave_seed": cave_seed, "difficulty": difficulty})
    policy.reset(env)
    traj = [info["position"]] if keep_trajectory else []
    ret, done, murky = 0.0, False, 0
    while not done:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        ret += r
        murky += info["turbidity"] > 0.4
        done = term or trunc
        if keep_trajectory:
            traj.append(info["position"])
    return episode_result(policy.name, cave_seed, difficulty, info, ret, murky, env.cfg.vehicle.dt, traj)


def episode_result(name, cave_seed, difficulty, info, ret, murky, dt, traj=()) -> EpisodeResult:
    """EpisodeResult from the final step's info dict."""
    outcome = "success" if info["is_success"] else "collision" if info["collided"] else ("abort" if info.get("aborted") else "timeout")
    return EpisodeResult(
        policy=name, difficulty=difficulty, cave_seed=cave_seed, outcome=outcome,
        steps=info["t"], time=info["t"] * dt, path_length=info["path_length"],
        shortest_path=info["shortest_path"],
        # the episode ends inside the goal radius, so compare with the geodesic distance actually covered
        efficiency=info["path_length"] / (info["shortest_path"] - info["geodesic"])
        if outcome == "success" else float("nan"),
        min_clearance=info["min_clearance"], mean_clearance=info["mean_clearance"],
        episode_return=ret, murk_fraction=murky / max(info["t"], 1), end_turbidity=info["turbidity"],
        energy_kj=info.get("energy_j", 0.0) / 1000,
        contacts=int(info.get("contacts", 0)), hull_damage=100.0 - float(info.get("hull", 100.0)),
        # per metre of shortest route actually covered, so detours and dawdling both show up as cost
        energy_per_m=info.get("energy_j", 0.0) / max(info["shortest_path"] - info["geodesic"], 1e-6)
        if outcome == "success" else float("nan"),
        trajectory=list(traj),
    )


def evaluate(policy, config: EnvConfig | dict | str | None = None,
             difficulties=(0.0, 0.25, 0.5, 0.75, 1.0), n_caves: int = 50,
             base_seed: int = 0, n_trajectories: int = 0, split: str = "test") -> list[EpisodeResult]:
    """Run `policy` on the first n_caves caves of `split` ("test" or "val") at each difficulty."""
    from .policies import env_config_for

    cfg = env_config_for(policy, replace(coerce_config(config), seed_split="test"))
    env = CaveEnv(cfg)
    results = []
    for d in difficulties:
        for i, s in enumerate(seeds_for(split, n_caves)):
            results.append(run_episode(env, policy, s, float(d), base_seed, keep_trajectory=i < n_trajectories))
    env.close()
    return results


def summarize(results: list[EpisodeResult]) -> list[dict]:
    """One row per (policy, difficulty), plus an 'all' row per policy."""
    rows = []
    groups: dict[tuple, list[EpisodeResult]] = {}
    for r in results:
        groups.setdefault((r.policy, r.difficulty), []).append(r)
        groups.setdefault((r.policy, "all"), []).append(r)

    def nanmean(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

    for (pol, d), rs in groups.items():
        succ = [r for r in rs if r.outcome == "success"]
        rows.append({
            "policy": pol, "difficulty": d, "episodes": len(rs),
            "success": np.mean([r.outcome == "success" for r in rs]),
            "collision": np.mean([r.outcome == "collision" for r in rs]),
            "timeout": np.mean([r.outcome == "timeout" for r in rs]),
            "efficiency": nanmean([r.efficiency for r in succ]),
            "time_to_goal": nanmean([r.time for r in succ]),
            "min_clearance": float(np.mean([r.min_clearance for r in rs])),
            "mean_clearance": float(np.mean([r.mean_clearance for r in rs])),
            "return": float(np.mean([r.episode_return for r in rs])),
            # of the collisions, the share that happened in murky water (turbidity > 0.4)
            "collisions_in_murk": nanmean([r.end_turbidity > 0.4 for r in rs if r.outcome == "collision"]),
            # energy of successful runs (failures end early, so their energy isn't comparable)
            "energy_kj": nanmean([r.energy_kj for r in succ]),
            "energy_per_m": nanmean([r.energy_per_m for r in succ]),
            "abort": np.mean([r.outcome == "abort" for r in rs]),
            "contacts": float(np.mean([r.contacts for r in rs])),
            "hull_damage": float(np.mean([r.hull_damage for r in rs])),
        })
    return rows


def format_table(rows: list[dict]) -> str:
    head = f"{'policy':<10} {'d':>5} {'n':>4} {'success':>8} {'collide':>8} {'timeout':>8} " \
           f"{'effic.':>7} {'t_goal':>7} {'min_clr':>8} {'mean_clr':>9} {'return':>7} {'kJ':>6} {'J/m':>6}"
    lines = [head, "-" * len(head)]
    for r in rows:
        d = r["difficulty"] if isinstance(r["difficulty"], str) else f"{r['difficulty']:.2f}"
        lines.append(
            f"{r['policy']:<10} {d:>5} {r['episodes']:>4} {100 * r['success']:7.0f}% {100 * r['collision']:7.0f}% "
            f"{100 * r['timeout']:7.0f}% {r['efficiency']:7.2f} {r['time_to_goal']:7.1f} "
            f"{r['min_clearance']:8.2f} {r['mean_clearance']:9.2f} {r['return']:7.1f} "
            f"{r.get('energy_kj', float('nan')):6.1f} {r.get('energy_per_m', float('nan')):6.0f}")
    return "\n".join(lines)


def save_csv(rows: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def episodes_to_rows(results: list[EpisodeResult]) -> list[dict]:
    return [{k: v for k, v in asdict(r).items() if k != "trajectory"} for r in results]

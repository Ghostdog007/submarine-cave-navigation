"""Comparison replays: every navigator on the SAME complex test cave per cave type.

    python -m scripts.compare_videos --csv runs/eval/a_episodes.csv runs/eval/b_episodes.csv \
        --navigators greedy oracle runs/nav_none_v2 runs/nav_retention_v2 runs/nav_junction_v2 \
        --out videos/08_round2 [--extra maze:1000000] [--workers 5]

Cave choice per type (from the evaluation CSVs): a test cave the oracle solved, where the evaluated navigators
did worst (fewest successes, then most dead-end entries, then longest time): the situations that separate them.
Every navigator is recorded on that cave (same seed, difficulty, sensor noise and currents). Memory replays must
show exactly what the agent stored (unexplained = 0) or the video is flagged. Writes <out>/README.md with the
chosen caves, why, and each navigator's outcome.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def pick_caves(csv_files, layouts):
    rows = [r for f in csv_files for r in csv.DictReader(open(f))]
    by = defaultdict(list)
    for r in rows:
        by[(r["layout"], int(r["cave_seed"]))].append(r)
    picks = {}
    for lay in layouts:
        best, key = None, None
        for (l, seed), rs in by.items():
            if l != lay:
                continue
            oracle_ok = any(r["policy"] == "nav-oracle" and r["outcome"] == "success" for r in rs)
            others = [r for r in rs if r["policy"] != "nav-oracle"]
            if not others or (any(r["policy"] == "nav-oracle" for r in rs) and not oracle_ok):
                continue
            succ = sum(r["outcome"] == "success" for r in others)
            de = sum(float(r.get("dead_end_entries", 0) or 0) for r in others)
            t = sum(float(r["time"]) for r in others)
            k = (succ, -de, -t)
            if key is None or k < key:
                best, key = seed, k
        if best is not None:
            picks[lay] = (best, f"{key[0]} of {len(by[(lay, best)]) - 1} evaluated navigators succeeded; "
                                f"{-key[1]:.0f} dead-end entries in total")
    return picks


def _record(args):
    nav_name, config, layout, seed, difficulty, out_dir, speed, reward_stage, *rest = args
    no_cue = bool(rest and rest[0])
    import json

    short = Path(nav_name).name if Path(nav_name).exists() else nav_name
    out = Path(out_dir) / f"{layout}_cave{seed}_{short}{'_blind' if no_cue else ''}.mp4"
    side = out.with_suffix(".json")                          # outcome sidecar: lets an interrupted run resume
    if out.exists() and side.exists():
        return json.loads(side.read_text(encoding="utf-8"))
    import yaml

    from scripts.eval_navigator import make_navigator
    from submarine.env import EnvConfig
    from submarine.navigator import NavigatorEnvConfig
    from submarine.render.nav_replay import record_nav_replay

    C = yaml.safe_load(open(config))
    env = EnvConfig.from_yaml("configs/eval_layouts.yaml").to_dict()
    base = {k: v for k, v in C["base"].items() if k != "env"}
    if reward_stage:                                         # that stage's reward (shown live), same caves and cue
        st = next(s_ for s_ in C["stages"] if s_["name"] == reward_stage)
        base.update({k: v for k, v in st.items() if k not in ("name", "steps", "env", "goal_cue_prob")})
        env = {**env, "reward": {**env.get("reward", {}), **st.get("env", {}).get("reward", {})}}
    if no_cue:                                               # blind dive: memory (and the exploration bonus) only
        base["goal_cue_prob"] = 0.0
    cfg = NavigatorEnvConfig.from_dict({**base, "env": env})
    nav = make_navigator(nav_name)
    r = record_nav_replay(nav, cfg, layout, seed, difficulty, out, speed=speed)
    res = {"layout": layout, "seed": seed, "navigator": short, "file": out.name, "outcome": r["outcome"],
           "time": round(r["time"], 1), "decisions": r["decisions"], "stored": r["stored"],
           "unexplained": r["unexplained"]}
    side.write_text(json.dumps(res), encoding="utf-8")       # written last: only finished videos are skipped
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--navigators", nargs="+", required=True)
    p.add_argument("--layouts", nargs="+", default=["tunnel", "maze", "branches", "crack", "breakdown"])
    p.add_argument("--extra", nargs="*", default=[], help="additional layout:seed pairs (e.g. maze:1000000)")
    p.add_argument("--config", default="configs/navigator.yaml")
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.add_argument("--speed", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--reward-stage", default=None, help="use (and display) this training stage's reward, e.g. "
                                                        "m3_full with configs/memnav_reward.yaml")
    p.add_argument("--no-cue", action="store_true", help="blind dives: no goal bearing (files end in _blind)")
    args = p.parse_args()

    picks = pick_caves(args.csv, args.layouts)
    for e in args.extra:
        lay, seed = e.split(":")
        key = f"{lay} (extra)" if f"{lay} (extra)" not in picks else f"{lay} (extra {seed})"   # two extras of a type
        picks[key] = (int(seed), "requested")
    jobs = [(n, args.config, lay.split(" ")[0], seed, args.difficulty, args.out, args.speed, args.reward_stage, args.no_cue)
            for lay, (seed, _) in picks.items() for n in args.navigators]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:     # (max_tasks_per_child hung the pool on Python 3.12)
        res = list(ex.map(_record, jobs))
    lines = ["# Comparison replays", "",
             "Every navigator on the same test cave per cave type (same seed, difficulty d = "
             f"{args.difficulty}, sensor noise and currents). Caves chosen from the evaluation as the hardest "
             "one the oracle solves. Memory displays are checked against the agent's stored state "
             "(unexplained must be 0).", "",
             "Markers (memory-v3 navigators): the badge names the architecture. Store variants show stored junction "
             "departures (ring #n) and dead ends (red X #n), lines to the entries recalled for the current choice and "
             "a yellow pulse on the strongest. Every v3 variant shows its belief heads: DEAD? n% on an exit it "
             "believes leads to a known dead end, BEEN n% on one it believes it already took, an orange ring while "
             "it believes it is revisiting a place; the log lists AVOIDED / ENTERED-despite decisions. With "
             "--reward-stage, a fourth status line shows the dive's reward, repeat PENALTY events and the "
             "exploration bonus (+n new area).", "",
             "| Cave type | Cave | Why this cave | Navigator | Outcome | Time (s) | Decisions | Memories stored | "
             "Display check | File |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lay, (seed, why) in picks.items():
        for r in [r for r in res if r["layout"] == lay.split(" ")[0] and r["seed"] == seed]:
            check = "ok" if r["unexplained"] == 0 else f"MISMATCH ({r['unexplained']})"
            lines.append(f"| {lay} | {seed} | {why} | {r['navigator']} | {r['outcome']} | {r['time']} | "
                         f"{r['decisions']} | {r['stored']} | {check} | `{r['file']}` |")
    (Path(args.out) / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = [r for r in res if r["unexplained"]]
    print("\n".join(lines))
    if bad:
        raise SystemExit(f"{len(bad)} replays with a memory-display mismatch")


if __name__ == "__main__":
    main()

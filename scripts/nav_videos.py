"""Record navigator replays (phase B) with the memory display, under the evaluation's test conditions.

    python -m scripts.nav_videos --models runs/nav_none runs/nav_junction --layout maze --caves 1000000
    python -m scripts.nav_videos --models greedy runs/nav_retention --layout maze --caves 1000000 1000003

Writes videos/07_navigators/<layout>_cave<seed>_<model>.mp4 and prints what each memory registered; a junction-
memory replay fails loudly if any decoded memory slot is not accounted for by a write event (the display must show
exactly what the agent stored).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="run folders, or greedy / random")
    p.add_argument("--layout", default="maze")
    p.add_argument("--caves", nargs="+", type=int, default=[1000000])
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--config", default="configs/navigator.yaml")
    p.add_argument("--eval-config", default="configs/eval_layouts.yaml")
    p.add_argument("--out", default="videos/07_navigators")
    p.add_argument("--speed", type=float, default=4.0)
    args = p.parse_args()

    from scripts.eval_navigator import make_navigator
    from submarine.env import EnvConfig
    from submarine.navigator import NavigatorEnvConfig
    from submarine.render.nav_replay import record_nav_replay

    C = yaml.safe_load(open(args.config))
    env = EnvConfig.from_yaml(args.eval_config).to_dict()
    cfg = NavigatorEnvConfig.from_dict({**{k: v for k, v in C["base"].items() if k != "env"}, "env": env})
    cfg = replace(cfg, env=replace(cfg.env, layout=args.layout))
    for seed in args.caves:
        for m in args.models:
            nav = make_navigator(m)
            name = Path(m).name if Path(m).exists() else m
            out = Path(args.out) / f"{args.layout}_cave{seed}_{name}.mp4"
            r = record_nav_replay(nav, cfg, args.layout, seed, args.difficulty, out, speed=args.speed)
            print(f"{out}: {r['outcome']} in {r['time']:.1f} s, {r['decisions']} decisions, "
                  f"{r['stored']} memories stored, unexplained slots {r['unexplained']}")
            for t, msg in r["log"]:
                print(f"    {t:6.1f} s  {msg}")
            if r["unexplained"]:
                raise SystemExit(f"memory display mismatch in {out}: {r['unexplained']} slots without a write event")


if __name__ == "__main__":
    main()

"""Live training monitor: run in a second terminal while scripts.train_driver / scripts.train_navigator run.

    python -m scripts.watch_training                      # every active run under runs/
    python -m scripts.watch_training runs/driver          # specific run(s)
    python -m scripts.watch_training runs/nav_none runs/nav_retention runs/nav_junction --every 30
    python -m scripts.watch_training --once               # print once and exit

Read-only: it reads <run>/config.yaml, <run>/log.txt and the TensorBoard event files the trainers write
(one event file per curriculum stage), and never touches the training processes. Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

BARS = "▁▂▃▄▅▆▇█"
TAGS = {"reward": "rollout/ep_rew_mean", "success": "rollout/success_rate", "length": "rollout/ep_len_mean",
        "fps": "time/fps", "expl_var": "train/explained_variance", "kl": "train/approx_kl", "std": "train/std"}
_acc: dict[str, object] = {}


def spark(values, width=40):
    if not values:
        return ""
    if len(values) > width:                                   # average into `width` buckets
        n = len(values)
        values = [sum(values[i * n // width:(i + 1) * n // width]) / max(1, (i + 1) * n // width - i * n // width)
                  for i in range(width)]
    lo, hi = min(values), max(values)
    return "".join(BARS[int((v - lo) / (hi - lo + 1e-12) * (len(BARS) - 1))] for v in values)


def scalars(path: Path) -> dict[str, list]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    key = str(path)
    if key not in _acc:
        _acc[key] = EventAccumulator(key, size_guidance={"scalars": 0})
    acc = _acc[key]
    acc.Reload()                                              # incremental: only reads what is new
    have = acc.Tags()["scalars"]
    return {name: [(e.step, e.value) for e in acc.Scalars(tag)] for name, tag in TAGS.items() if tag in have}


def fmt_time(s: float) -> str:
    if s != s or s < 0 or s == float("inf"):
        return "?"
    h, m = divmod(int(s) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def run_report(run: Path) -> list[str]:
    out = [f"== {run.name} " + "=" * max(0, 70 - len(run.name))]
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) if (run / "config.yaml").exists() else {}
    stages = cfg.get("stages", [])
    unit = "decisions" if "memory" in cfg else "steps"
    log = (run / "log.txt").read_text(encoding="utf-8").splitlines() if (run / "log.txt").exists() else []
    tb_dirs = sorted((run / "tb").glob("*"), key=lambda p: p.stat().st_mtime) if (run / "tb").exists() else []
    files = sorted(tb_dirs[-1].glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime) if tb_dirs else []
    done = any(l.split("] ", 1)[-1].startswith("done") for l in log)

    if done:
        out.append("  finished:")
        out += [f"    {l}" for l in log if "] stage " in l or "] done" in l]
        return out
    if not files:
        out.append("  no TensorBoard data yet (imitation phase or start-up)")
    else:
        i = len(files) - 1
        cur = scalars(files[-1])
        prev_end = scalars(files[-2])["reward"][-1][0] if len(files) > 1 and scalars(files[-2]).get("reward") else 0
        rew = cur.get("reward", [])
        step = rew[-1][0] if rew else prev_end
        name = stages[i]["name"] if i < len(stages) else f"stage {i}"
        budget = int(stages[i]["steps"]) if i < len(stages) else 0
        in_stage = step - prev_end
        frac = min(1.0, in_stage / budget) if budget else 0.0
        fps = cur["fps"][-1][1] if cur.get("fps") else float("nan")
        later = sum(int(s["steps"]) for s in stages[i + 1:])
        stage_done = done or (i < len(stages) and f"stage {name}:" in "\n".join(log))
        bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
        out.append(f"  stage {i + 1}/{len(stages)} {name:<12} [{bar}] {100 * frac:5.1f}%  "
                   f"{in_stage:,}/{budget:,} {unit}")
        if not stage_done:
            left = (budget - in_stage) / fps if fps == fps and fps > 0 else float("nan")
            out.append(f"  speed {fps:,.0f} {unit}/s   stage ETA {fmt_time(left)}   "
                       f"all stages ETA {fmt_time(left + later / fps if fps == fps and fps > 0 else float('nan'))}"
                       f"   (finish ~{time.strftime('%H:%M', time.localtime(time.time() + left + later / fps)) if fps == fps and fps > 0 else '?'})")

        def last(k, f="{:.3f}"):
            return f.format(cur[k][-1][1]) if cur.get(k) else "-"

        out.append(f"  episode reward {last('reward', '{:.1f}')}   success {last('success', '{:.0%}')}   "
                   f"length {last('length', '{:.0f}')}   expl.var {last('expl_var', '{:.2f}')}   "
                   f"kl {last('kl', '{:.4f}')}   std {last('std', '{:.2f}')}")
        if rew:
            vals = [v for _, v in rew]
            out.append(f"  reward this stage  {spark(vals)}  ({min(vals):.1f} .. {max(vals):.1f})")
        if cur.get("success"):
            vals = [v for _, v in cur["success"]]
            out.append(f"  success this stage {spark(vals)}  ({min(vals):.0%} .. {max(vals):.0%})")
        age = time.time() - files[-1].stat().st_mtime
        if not done and age > 300:
            out.append(f"  !! no new data for {fmt_time(age)}: the trainer may have stopped or be switching stage")
    out.append("  log:")
    out += [f"    {l}" for l in log[-4:]] or ["    (empty)"]
    return out


def find_runs(hours: float) -> list[Path]:
    root = Path("runs")
    found = []
    for r in root.iterdir() if root.exists() else []:
        if r.name in ("archive", "eval") or not r.is_dir() or not (r / "log.txt").exists():
            continue
        if time.time() - (r / "log.txt").stat().st_mtime < hours * 3600 or \
                any(time.time() - f.stat().st_mtime < hours * 3600 for f in (r / "tb").rglob("events.*")):
            found.append(r)
    return sorted(found)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run folders (default: runs active in the last --hours)")
    p.add_argument("--every", type=float, default=15.0, help="refresh interval in seconds")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    os.system("")                                             # enables ANSI escapes in the Windows console
    if not args.once:                                         # alternate screen, hidden cursor (restored on exit)
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    try:
        loop(args)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.once:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()


def loop(args):
    while True:
        runs = [Path(r) for r in args.runs] or find_runs(args.hours)
        lines = [f"training monitor  {time.strftime('%H:%M:%S')}   (refresh {args.every:.0f} s, Ctrl+C to quit)", ""]
        for r in runs:
            try:
                lines += run_report(r) + [""]
            except Exception as e:                            # a file mid-write: show it, retry next refresh
                lines += [f"== {r.name}: could not read yet ({type(e).__name__}: {e})", ""]
        if not runs:
            lines.append("no active runs under runs/")
        text = "\n".join(lines)
        if args.once:
            print(text)
            return
        # redraw in place: cursor home, overwrite each line (clearing its tail), clear anything left below
        sys.stdout.write("\x1b[H" + "\n".join(l + "\x1b[K" for l in text.splitlines()) + "\x1b[J")
        sys.stdout.flush()
        time.sleep(args.every)


if __name__ == "__main__":
    main()

"""Live training dashboard: rewrites runs/live_status.html (auto-refreshing) every N seconds.

    python -m scripts.live_status                        # runs matching cur_*, refresh every 30 s
    python -m scripts.live_status --runs <run_name> ... --every 60

Open runs/live_status.html in a browser; it reloads itself. Stops on its own once every run has finished
(after one last update), or with Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import re
import time
from datetime import datetime
from pathlib import Path

RUNS = Path("runs")
MEMORY = {
          "MlpPolicy": "none (MLP)", "MultiInputPolicy": "none (CNN + MLP)"}


def speed_and_step(run: Path):
    """(latest step, recent steps/s, seconds since last update) from TensorBoard, or Nones."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        files = glob.glob(str(run / "tb" / "*" / "events.*"))
        if not files:
            return None, None, None
        ev = EventAccumulator(max(files, key=lambda f: Path(f).stat().st_mtime)).Reload().Scalars("time/fps")
        if len(ev) < 2:
            return (ev[-1].step if ev else None), None, None
        a, b = ev[max(0, len(ev) - 10)], ev[-1]
        rate = (b.step - a.step) / max(b.wall_time - a.wall_time, 1e-9)
        return b.step, rate, time.time() - b.wall_time
    except Exception:
        return None, None, None


def run_info(run: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(open(run / "config.yaml", encoding="utf-8"))
    total = int(cfg.get("total_timesteps", 0))
    val = []
    if (run / "validation.csv").exists():
        val = [r for r in csv.DictReader(open(run / "validation.csv", encoding="utf-8")) if r["difficulty"] == "all"]
    log = RUNS / f"{run.name.removesuffix('_s0')}_train.log"
    phase = "–"
    if log.exists():
        phases = re.findall(r"curriculum phase (\d+)", log.read_text(encoding="utf-8", errors="ignore"))
        if phases:
            phase = f"{int(phases[-1]) + 1} / {len(cfg.get('curriculum', [])) or '?'}"
    step, rate, age = speed_and_step(run)
    done = (run / "final_model.zip").exists()
    if val and (step is None or int(val[-1]["timesteps"]) > step):
        step = int(val[-1]["timesteps"])
    status = "done" if done else ("running" if age is not None and age < 300 else "stalled?" if age else "starting")
    best = max(val, key=lambda r: float(r["success"])) if val else None
    eta = (total - step) / rate if (rate and step is not None and not done) else None
    return {"name": run.name, "memory": MEMORY.get(cfg.get("policy"), cfg.get("policy")), "status": status,
            "step": step or 0, "total": total, "rate": rate, "eta": eta, "phase": phase, "val": val, "best": best}


def fmt_eta(s):
    if s is None:
        return "–"
    finish = datetime.fromtimestamp(time.time() + s).strftime("%H:%M")
    return f"{s / 3600:.1f} h (≈{finish})" if s > 3600 else f"{s / 60:.0f} min (≈{finish})"


def render(infos: list[dict], every: int) -> str:
    rows = []
    for i in infos:
        last = i["val"][-1] if i["val"] else None
        pct = 100 * i["step"] / i["total"] if i["total"] else 0
        cls = {"done": "done", "running": "run"}.get(i["status"], "warn")
        rate = f"{i['rate']:.0f}" if i["rate"] else "–"
        eta = fmt_eta(i["eta"]) if i["status"] != "done" else "–"
        cells = [
            f"<td><b>{html.escape(i['name'])}</b><br><small>{html.escape(str(i['memory']))}</small></td>",
            f"<td class='{cls}'>{i['status']}</td>",
            f"<td><div class='bar'><div style='width:{pct:.0f}%'></div></div>"
            f"{i['step'] / 1e6:.2f}M / {i['total'] / 1e6:.0f}M ({pct:.0f}%)</td>",
            f"<td>{rate}</td>", f"<td>{eta}</td>", f"<td>{i['phase']}</td>",
        ]
        if last:
            cells += [
                f"<td class='num'>{100 * float(last['success']):.0f}%</td>",
                f"<td class='num'><b>{100 * float(i['best']['success']):.1f}%</b>"
                f"<br><small>at {int(i['best']['timesteps']) / 1e6:.1f}M</small></td>",
                f"<td class='num'>{100 * float(last['collision']):.0f}%</td>",
                f"<td class='num'>{100 * float(last['timeout']):.0f}%</td>",
            ]
        else:
            cells += ["<td>–</td>"] * 4
        rows.append("<tr>" + "".join(cells) + "</tr>")

    # success history: one column per validation step shared by the runs
    steps = sorted({int(r["timesteps"]) // 100_000 * 100_000 for i in infos for r in i["val"]})
    steps = [s for s in steps if s % 400_000 == 0 or s == steps[-1]] if steps else []
    hist_head = "".join(f"<th>{s / 1e6:.1f}M</th>" for s in steps)
    hist_rows = []
    for i in infos:
        by = {int(r["timesteps"]) // 100_000 * 100_000: float(r["success"]) for r in i["val"]}
        best = max(by.values()) if by else None
        cells = []
        for s in steps:
            v = by.get(s)
            if v is None:
                cells.append("<td></td>")
            else:
                star = " ★" if best is not None and v == best else ""
                cells.append(f"<td class='num' style='background:rgba(46,160,67,{v:.2f})'>{100 * v:.0f}%{star}</td>")
        hist_rows.append(f"<tr><td><b>{html.escape(i['name'])}</b></td>{''.join(cells)}</tr>")

    now = datetime.now().strftime("%H:%M:%S")
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="{every}">
<title>Training status</title><style>
:root {{ --bg:#fff; --fg:#1b1f24; --mut:#667; --line:#d8dee4; --bar:#2da44e; --head:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --mut:#8b949e; --line:#30363d;
  --bar:#3fb950; --head:#161b22; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,Segoe UI,sans-serif; margin:24px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .mut {{ color:var(--mut); }}
table {{ border-collapse:collapse; margin:14px 0 26px; min-width:760px; }}
th, td {{ border:1px solid var(--line); padding:6px 10px; text-align:left; vertical-align:middle; }}
th {{ background:var(--head); font-weight:600; }} td.num {{ text-align:right; }}
.bar {{ width:150px; height:8px; background:var(--line); border-radius:4px; margin-bottom:3px; }}
.bar div {{ height:8px; background:var(--bar); border-radius:4px; }}
.run {{ color:#1f6feb; font-weight:600; }} .done {{ color:var(--bar); font-weight:600; }} .warn {{ color:#d29922; }}
small {{ color:var(--mut); }}
</style></head><body>
<h1>Training status</h1>
<div class="mut">updated {now} · refreshes every {every} s · validation = 30 caves, mixed cave types, d = 0.5
(each cave ≈ 3 points; ±9 points is noise)</div>
<table><tr><th>Run</th><th>Status</th><th>Progress</th><th>Steps/s</th><th>Time left</th><th>Curriculum stage</th>
<th>Latest success</th><th>Best success</th><th>Collisions</th><th>Timeouts</th></tr>{''.join(rows)}</table>
<h2 style="font-size:16px">Validation success over training</h2>
<table><tr><th>Run</th>{hist_head}</tr>{''.join(hist_rows)}</table>
<div class="mut">Curriculum: 1 tunnels / crevasses / boulder chokes → 2 + branches → 3 + mazes →
4 + extra maze & branch practice, energy penalty on. Validation always uses the final mix, including mazes.</div>
</body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=None, help="run folder names (default: cur_*)")
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--out", type=Path, default=RUNS / "live_status.html")
    args = p.parse_args()
    while True:
        names = args.runs or sorted(p.name for p in RUNS.glob("cur_*") if p.is_dir())
        infos = [run_info(RUNS / n) for n in names if (RUNS / n / "config.yaml").exists()]
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(render(infos, args.every), encoding="utf-8")
        tmp.replace(args.out)
        if infos and all(i["status"] == "done" for i in infos):
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()

# Workflow: from an idea to a video in the README

Every experiment in this repo follows the same loop. This guide walks through it with the real commands, using
**v3.3** (exit memory trained on the memory caves, against a memoryless control) as the running example. Swap in
your own config and run names.

```
 1 config ──► 2 smoke test ──► 3 teacher demos ──► 4 train (model + control) ──► 5 monitor
                                                                                     │
 10 document + tag ◄── 9 README preview ◄── 8 record videos ◄── 7 pick caves ◄── 6 evaluate
```

| Step | Script | Writes to | Typical time (CPU) |
|---|---|---|---|
| 1. Config | (edit YAML) | `configs/<name>.yaml` | – |
| 2. Smoke test | `pytest`, `train_memnav --scale 0.01` | `runs/<name>_smoke/` | ~15 min |
| 3. Teacher demos | `collect_nav_demos` | `runs/<demos>.npz` | ~10 min |
| 4. Train | `train_memnav` | `runs/<prefix>_<variant>/` | 1.5–3 h |
| 5. Monitor | `watch_training`, TensorBoard | (read-only) | – |
| 6. Evaluate | `eval_navigator` | `runs/eval/<name>_episodes.csv` | ~10 min per model |
| 7. Pick showcase caves | `pick_showcase` | stdout | seconds |
| 8. Record videos | `compare_videos` | `videos/NN_<topic>/` (.mp4 + README.md) | 1–4 h |
| 9. README preview | `make_preview` | `docs/media/<name>.webp` | 3–10 min each |
| 10. Document + tag | git | README, docs, tag | – |

## Conventions

- **Seeds decide the split.** Training caves come from training seeds; evaluation uses test caves (seed ≥ 1,000,000)
  that no model ever trains on. Every policy in an evaluation faces the *same* caves, sensor noise and currents, so
  results can be compared cave by cave, and a replay of cave *s* reproduces the evaluation's outcome on cave *s*.
- **One change at a time, with a control.** A model is always trained next to a twin that differs only in what is
  being tested (here: `exit` memory vs `none`), with the same config, seed, budget and caves.
- **Run folders** (`runs/<run>/`, git-ignored): `config.yaml` (the exact settings used), `log.txt` (one line per
  stage), `tb/` (TensorBoard), `<stage>.pt` after each curriculum stage, `final_model.pt`, `summary.json`.
- **Evaluation output:** a table on stdout (redirect it to `runs/<name>.out`) and one CSV row per dive in
  `runs/eval/<name>_episodes.csv`, with outcome, time, energy, decisions, dead-end entries / re-entries, etc.
- **Videos** (`videos/NN_<topic>/`, git-ignored) are numbered by development stage, each folder with a generated
  `README.md` that lists every clip and its outcome. Only small WebP previews (`docs/media/`) are committed.

## 1. Write a config

Copy the closest existing config and change one thing. A navigator config has `base:` (environment and reward),
`stages:` (the curriculum: cave mix, budget, goal-cue probability), `ppo:`, `bc:` (warm start from teacher
demonstrations) and `run_prefix:`, which names the runs:

```yaml
# configs/memnav_caves_clean.yaml (excerpt)
run_prefix: memnav_cc          # -> runs/memnav_cc_exit, runs/memnav_cc_none
bc:
  demos: runs/memnav_c_demos.npz
```

Write *why* at the top of the file: what changed from the previous version and what result would confirm it.

## 2. Smoke test

```powershell
python -m pytest -q                                          # full suite, ~12 min (155 tests)
python -m scripts.collect_nav_demos --config configs/memnav_caves_clean.yaml --decisions 3000 --workers 4 --out runs/smoke_demos.npz
python -m scripts.train_memnav --config configs/memnav_caves_clean.yaml --variant exit --scale 0.01 --workers 2 --name smoke --demos runs/smoke_demos.npz
```

`--scale 0.01` runs every stage at 1% of its budget: it catches config and shape errors in minutes instead of
hours. Delete `runs/smoke*` afterwards.

## 3. Collect teacher demonstrations

The map-aware oracle drives the training caves (with some random wrong turns, so the data also shows recovering
from them). The navigator is warm-started by imitating it before RL.

```powershell
python -m scripts.collect_nav_demos --config configs/memnav_caves_clean.yaml --workers 8 --out runs/memnav_c_demos.npz
```

Re-collect whenever the cave mix or the observation / labels change.

## 4. Train the model and its control

Run both at once in separate processes:

```powershell
foreach ($v in "exit","none") {
  Start-Process .venv\Scripts\python.exe "-m scripts.train_memnav --config configs/memnav_caves_clean.yaml --variant $v"
}
```

Each run writes `runs/memnav_cc_<variant>/` and ends with `final_model.pt`. On a 16 GB machine two navigator
trainings at once are fine. Leave other heavy jobs (video recording in particular) for later.

## 5. Monitor

```powershell
python -m scripts.watch_training runs/memnav_cc_exit runs/memnav_cc_none     # live table: stage, success, reward
tensorboard --logdir runs/memnav_cc_exit/tb                                  # curves, incl. memory/* probes
```

`log.txt` gets one line per finished stage, e.g.
`stage m2_memory: 100000 decisions in 54.1 min (31/s), episode reward 33.4, decisions/episode 53, success 68%`.

## 6. Evaluate

`eval_navigator` runs any mix of baselines (`greedy`, `oracle`, `random`, `wall-left`, `wall-right`) and trained
models (run folders) on 30 test caves per cave type:

```powershell
$E = "--config configs/memnav_reward.yaml --models runs/memnav_cc_exit runs/memnav_cc_none --workers 6"

# memory caves, with and without the goal cue
python -m scripts.eval_navigator $E.Split() --layouts rings hub --name v33_mem > runs/v33_mem.out
python -m scripts.eval_navigator $E.Split() --layouts rings hub --goal-cue-prob 0 --name v33_memblind > runs/v33_memblind.out

# the older cave types: no regression?
python -m scripts.eval_navigator $E.Split() --layouts tunnel maze branches crack breakdown decoy loops --name v33_all > runs/v33_all.out

# baselines once (their numbers don't change between model versions)
python -m scripts.eval_navigator --config configs/memnav_reward.yaml --layouts rings hub --policies greedy oracle wall-left wall-right --name memcaves_cue
```

Useful extras: `--wipe-memory` (the memory-wipe ablation: is the memory actually used?), `--n-caves`,
`--difficulty`. All v3.x evaluations use `configs/memnav_reward.yaml` for the test conditions, so numbers stay
comparable across versions.

**Is the difference real?** Compare model and control cave by cave: count caves where only one of them succeeded
and run a sign test.

```python
import csv
from scipy.stats import binomtest
ok = lambda f, pol: {(r["layout"], r["cave_seed"]): r["outcome"] == "success"
                     for r in csv.DictReader(open(f)) if r["policy"] == pol}
a = ok("runs/eval/v33_memblind_episodes.csv", "memnav_cc_exit")
b = ok("runs/eval/v33_memblind_episodes.csv", "memnav_cc_none")
wins, losses = sum(a[k] and not b[k] for k in a), sum(b[k] and not a[k] for k in a)
print(wins, losses, binomtest(wins, wins + losses).pvalue)
```

(For v3.3, pooled over cue and blind dives: 22 wins to 7, p = 0.008.)

## 7. Pick the caves to film

Two kinds of cave make good videos:
- **A stress test:** `compare_videos` picks it automatically from the CSVs: per cave type, a cave the oracle solved
  where the evaluated models did worst.
- **A showcase:** a cave where the new model succeeds and its control fails. `pick_showcase` finds them:

```powershell
python -m scripts.pick_showcase --csv runs/eval/v33_memblind_episodes.csv --winner memnav_cc_exit --losers memnav_cc_none `
    --reference-csv runs/eval/memcaves_cue_episodes.csv --layouts rings hub
# rings:1000008  memnav_cc_exit 61.1 s, 6 such caves
# hub:1000012  memnav_cc_exit 73.7 s, 6 such caves
```

It prints `layout:seed` pairs for `--extra` in the next step, the fastest winning dive first (the shortest clear
video).

## 8. Record the comparison videos

Every navigator is recorded on the same caves, side by side in one folder:

```powershell
python -m scripts.compare_videos --config configs/memnav_caves_clean.yaml --reward-stage m3_full --no-cue `
    --csv runs/eval/memcaves_cue_episodes.csv runs/eval/v33_mem_episodes.csv `
    --navigators runs/memnav_cc_exit runs/memnav_cc_none wall-left oracle `
    --layouts rings hub --extra rings:1000008 hub:1000012 `
    --out videos/10_memory_caves_clean --workers 1
```

- `--navigators`: run folders and/or `greedy`, `oracle`, `random`, `wall-left`, `wall-right`.
- `--no-cue`: blind dives (file names end in `_blind`). `--reward-stage m3_full`: show that stage's reward live.
- **Memory:** each worker holds a renderer and an encoder. Use `--workers 1` (or 2) on 16 GB: three workers ran out
  of memory here and crashed mid-video with `Broken pipe`.
- **Resumable:** each finished video gets a `.json` outcome next to it; re-running the same command skips those.
- **Checked:** memory overlays must match what the agent stored (`Display check: ok` in the folder's README).
- Output: one `<layout>_cave<seed>_<navigator>[_blind].mp4` per navigator and cave (1280×802, 40 fps), plus a
  generated `README.md` table with each outcome and time.

Other recorders: `scripts.record` (end-to-end SB3 agents, e.g. the PPO driver, with `--camera follow`, `--format gif`)
and `scripts.nav_videos` (a few navigators on chosen caves).

## 9. Make a README preview

GitHub plays animated WebP inline (not `.mp4`), so the README shows compressed excerpts:

```powershell
python -m scripts.make_preview videos/10_memory_caves_clean/rings_cave1000008_memnav_cc_exit_blind.mp4 `
    videos/10_memory_caves_clean/rings_cave1000008_memnav_cc_none_blind.mp4 --duration 17.5 --out docs/media/v33_rings.webp
```

Several inputs go side by side (`--stack v` stacks them); a clip that ends early holds its last frame. Defaults
(10 fps, 480 px per panel, quality 55) give 1–3 MB for 15–25 s. `--start` / `--duration` cut an excerpt and
`--speed 2` doubles the speed of a long clip. Encoding takes a few minutes; `--effort 2` is faster but larger. Keep
each preview under ~3 MB: every committed preview stays in the git history forever.

Then embed it in the README:

```markdown
![v3.3 memory vs its memoryless twin on a rings cave](docs/media/v33_rings.webp)
```

## 10. Document and tag

1. **Results:** add the tables to the lab notebook ([memory_v3.md](memory_v3.md) for memory work,
   [progress.md](progress.md) for everything else): what changed, the numbers, and what they mean, including what
   didn't work.
2. **README:** a row in *The journey*, the headline numbers, the preview in *Watch it*.
3. **Videos:** list the clips worth keeping in [videos/README.md](../videos/README.md); delete the rest.
4. **Commit, tag the milestone, push:**

```powershell
git add -A; git commit -m "v3.3 results: memory beats its memoryless twin on rings / hub (p = 0.008)"
git tag -a v3.3 -m "memory v3.3: trained on memory caves; first significant memory win"
git push --follow-tags
```

`git checkout v3.3` later brings back exactly the code that produced those numbers.

# submarine-cave-navigation

**Reinforcement learning for an autonomous submarine in underwater caves: sonar, vision, route memory and a built-in safety filter.**

A 2D simulator and training framework for an autonomous submarine that has to find its way through narrow,
winding underwater ice caves using only on-board sensing: a multi-beam **sonar** and a headlight-lit **camera**
that degrades in **murky water**. Agents are trained with PPO (Stable-Baselines3) and compared against classical
controllers and a map-aware oracle on held-out caves.

> **Status (Nov 2026): two-level architecture: built, trained and evaluated; memory fixes next.** The end-to-end agents (one network
> driving at 10 Hz *and* remembering junctions) reached their limits: memory cut dead-end re-entries ~11×, but
> crashes prevented any gain in success, and a safety filter bolted on after training made the agents stall.
> Their code is removed, their runs are in `runs/archive/` and their results in [progress.md](progress.md).
>
> | Phase | State |
> |---|---|
> | A0: hand-written driver + built-in safety filter + openings detector | ✅ 0 crashes in 150 test dives (gap follower: 23–90%) |
> | A1–A3: learned driver (imitation, then PPO curriculum with slow sonar, currents, sensor failures, hull damage) | ✅ 0 crashes in 150 test dives; as good as the hand driver |
> | B: navigators (no memory / retention / retention + junction memory) | ✅ trained: 1 crash in 750 test dives; no memory 82% success (greedy heuristic 83%); **memory navigators worse (69–72%)** |
> | B fixes: masked exits, deduplicated memory writes, warm start, filter floor patch | ✅ built; round 2 training on `main` (diagnosis in [progress.md](progress.md#stage-9-two-level-architecture-learned-driver-with-a-built-in-safety-filter-memory-navigator)) |
> | **M (this branch, `memory-v3`): exit memory, trained by auxiliary losses on scenarios that need memory** | **v3: negative result.** The memory learned its facts (85–90% recall), but the policy barely used them (gate ~0.17); every variant 69–73% on the standard caves, below greedy (82%). [docs/memory_v3.md](docs/memory_v3.md#results-of-the-first-v3-run-22-nov-2026-negative-with-known-causes) |
> | M v3.1: reward that prices forgetting | trained and tested: **no gain for memory** (exit 70.7%, no memory 72.0%); it made both explore untried exits more |
> | M v3.2: event-triggered per-exit memory gate + belief bias | trained and tested: **memory is used selectively** (gate 0.83 at known dead ends vs 0.40 at fresh junctions) and known dead ends are avoided most of any model (61–70%), but success did not rise (66.7% standard, 48–55% rings / hub) |
> | M memory caves (`rings`, `hub`) + wall-follower baselines | built: the old caves turned out **solvable without memory**; on the new ones every reactive rule fails (greedy 5%, wall following 37–42%, oracle 100%) and v3 exit memory re-enters dead ends ~25% less than no memory. 148/148 tests pass |
> | **M v3.3: v3.2 trained on the memory caves (+ a memoryless control)** | **First significant memory win.** On unseen rings / hub caves: 71.7% vs 58.3% for the control with the cue, 68.3% vs 56.7% blind, ~20–30% fewer dead-end re-entries; paired over the same caves 22 wins to 7 (sign test p = 0.008). Also the best v3 model on the old caves (74.7% standard). [Results](#v33-results-23-nov-2026) |
> | Training speed-up | memory reads as batched matmuls: same outputs, learner ~2× faster. The GPU was measured *slower* for this tiny step-by-step model |

![game-style renderer](docs/figures/game_renderer_drive.png)

- **Design document:** [context.md](context.md), with the goals, world, vehicle, sensors, reward and research questions.
- **Development log and results:** [progress.md](progress.md), covering what changed at each stage and how well each agent did.
- **Scenarios and capabilities:** [scenarios.md](scenarios.md), the same progression told through one example dive:
  what each stage's agent was meant to handle and what it actually does.
- **Replay videos by stage:** [`videos/`](videos).

## Highlights

**Clear water, single-passage caves** (success on unseen test caves):

| | All difficulties | Hardest caves (d = 1) |
|---|---|---|
| Gap-following controller | 80% | 34% |
| PPO trained on easy caves only | 31% | 2% |
| **PPO trained on mixed difficulty** | **94%** | **82%** |
| Oracle with the true map (upper bound) | 100% | 100% |

**Murky water, slow sonar** (sensor experiment):

| | All difficulties | d = 1 |
|---|---|---|
| Gap follower | 72% | 26% |
| PPO, camera only | 84% | 64% |
| PPO, sonar only | 88% | 64% |
| **PPO, sonar + camera** | **93%** | **86%** |

**Harder cave types** (scenarios modelled on real cave-exploration hazards, d = 0.5, 30 test caves each,
end-to-end agents, now archived):

| | Maze | Branches | Tunnel | Crevasse | Boulder choke | Maze dead-end re-entries |
|---|---|---|---|---|---|---|
| No memory (MLP) | 13% | 83% | 20% | 53% | 53% | 0.80 |
| Retention memory | 13% | 57% | 23% | 0% | 37% | **0.07** |
| Retention + junction memory | 10% | 60% | 20% | 13% | 17% | 0.27 |

Memory clearly changes *route* behaviour, but the agents crash too often for it to pay off. The two-level
design is meant to fix that.

## Architecture (two-level, in progress)

```
            +---------------- NAVIGATOR (learned, decides rarely) -------------+
  sonar --> | openings detector -> candidates [opening 1 .. k, "turn back"]   |
  goal  --> | retention memory (what happened, multi-scale decay)             |
  pose  --> | junction memory (where: slots keyed by dead-reckoned position)  |--> "take opening k"
            | acts only at decision points: junction, dead end, or every ~2 s |    (+ planned: speed mode)
            +------------------------------------------------------------------+
                                          | target heading + speed
            +---------------- DRIVER (always on, part of the agent) ----------+
            | heading/speed tracking controller -> SAFETY FILTER -> thrust/yaw |--> 10 Hz control
            | blindness supervisor (stop / think / act slowly / abort)         |
            +------------------------------------------------------------------+
```

- **The safety filter is part of the agent** and is on in training and testing alike. It predicts the
  stopping path of each candidate command against the walls the sonar currently sees, and only lets safe
  commands through. The navigator never outputs raw thrust, so it can't learn wall-ramming habits, and the
  filter never surprises it.
- **Memory handles what it's good at: choosing exits.** A junction-memory slot means exactly "at junction #2
  I took opening B; it was a dead end".
- **Cheaper training:** the navigator makes ~30–75 decisions per episode instead of ~1,500 network calls, and
  the 10 Hz control runs as plain code inside the parallel workers. That should cut wall-clock time per run
  ~5–10× (to be measured).

| Package | Role |
|---|---|
| `submarine/control/` | driver: openings detector, hand controller, integrated safety filter, learned driver |
| `submarine/navigator/` | decision-level environment (+ AUX ground-truth labels, goal-cue withholding), greedy / random / oracle navigators |
| `submarine/memory/` | round 1-2: retention + junction memory (SB3 RecurrentPPO); **memory-v3: `exitmem.py` (exit memory, 4 variants), `trainer.py` (recurrent PPO + aux losses + warm start)** |
| `submarine/world/layouts.py` | cave types, **+ `decoy` and `loops` (memory-v3), `rings` and `hub` (memory caves, v3.2)** |

## Scenarios the new agent should handle

| Scenario | Real-world analogue | What the agent must do | Which part handles it |
|---|---|---|---|
| Winding tunnel, d up to 1 | normal cave passage | drive fast without touching walls | driver + safety filter |
| Crevasse with 1.4 m squeezes | glacier crack, restriction | slow down and thread the squeeze, never ram | safety filter (speed-dependent margin) |
| Maze with dead ends | multi-passage cave system | back out of a dead end and try an *untried* exit, not return to the start | junction memory |
| Misleading side branches | side passages pointing toward the goal | prefer the branch that keeps progressing, remember failed ones | navigator + retention |
| Boulder choke | breakdown chamber | find the gap among debris | openings detector + driver |
| Murk / silt-out | silted water, camera useless | rely on sonar, keep moving safely | driver (sonar-based filter) |
| Sonar or camera failure | equipment fault | stop, wait, creep slowly, abort if blind too long | blindness supervisor |
| Gentle wall contact | hull scrape | tolerate light contact (hull damage), avoid hard impacts | hull-damage model + filter |
| Energy budget | battery limit | avoid wasteful speed and dithering | energy penalty in the reward |
| Return through known passage (planned) | exit dive along the way in | go faster where it already passed safely | place memory + speed mode, filter as the hard limit |
| **Decoy dead ends** (`decoy`, memory-v3) | side passages that head straight for the exit and end blind | enter a decoy at most once, even though the goal bearing keeps recommending it | exit memory ("known dead end down this exit") |
| **Loops** (`loops`, memory-v3) | cave systems with interconnected passages | don't circle: notice "I've been here" and take an exit not yet tried | exit memory (loop closure, "taken before") |
| **No goal cue** (memory-v3, 30% of training dives) | exploring for an unknown exit | systematic exploration without re-exploring | exit memory |
| **Rings** (`rings`, v3.2) | starting inside a looped passage around a pillar of rock | notice "I've gone round once" and take the one exit not yet tried; wall following and the goal bearing both circle forever | exit memory (loop closure, "taken before") |
| **Hub** (`hub`, v3.2) | a chamber where several passages meet, most of them blind | after each failed passage, come back and pick a *different* one, although the chamber looks the same every time | exit memory ("taken before", "known dead end") |

## Memory v3: exit memory (this branch)

### The story so far (22–23 Nov 2026)

| Step | What we tried | What we found |
|---|---|---|
| **v3** | Exit memory: a store of junctions and dead ends, read once per exit, trained by auxiliary "what do you remember?" losses. Four variants (none / gru / graph / exit) | The memory **knew** its facts (85–90% recall) but the policy **ignored** them: its gate stayed ~0.17, all variants 69–73%, below a greedy rule (82%) |
| **v3.1** | A reward that prices forgetting: progress paid only for new ground, small penalties for re-entering dead ends, re-taking exits and U-turn loops, an exploration bonus, the goal cue hidden in 70% of dives | Both models explored more (untried exits 2–4× more often), but memory still did not help: **no memory did best** |
| **Diagnosis** | Wall-follower baselines; blind tests | **The caves never needed memory.** "Keep a hand on the wall" solves every old cave; a memoryless policy learned such a reactive rule and re-entered dead ends as rarely as the oracle (0.26 vs 0.24 per dive) |
| **Memory caves** | `rings` (start inside two rings: both walls are islands) and `hub` (one chamber, dead-end spokes pointing at the exit) | Every reactive rule collapses (greedy 5%, wall following 37–42%, oracle 100%). **First real memory signal:** v3 exit memory re-enters dead ends ~25% less than its memoryless twin |
| **v3.2** | Event-triggered per-exit memory gate + belief bias + reliance probes | **Breakthroughs below.** Memory is used *when it matters* and known dead ends are avoided most of any model, but success did not rise: its training caves still did not reward memory, so RL pruned it |
| **Speed** | Profiling | The memory model was per-op-overhead bound, not simulation-bound. Batched matmul reads: same outputs, learner ~2× faster. GPU measured slower for this tiny model |
| **v3.3** | v3.2 trained on rings and hub, with v3.2's exact training settings, and a memoryless control trained identically | **Memory finally wins:** +13 points over its memoryless twin on rings / hub with the cue, +12 blind, fewer dead-end re-entries, p = 0.008. The architecture was fine; it needed caves where remembering pays |

**v3.2 breakthroughs:**
1. **The memory gate finally moves, and selectively.** The old single gate never moved: it was a redundant scale
   with no learning signal. The event gate learned to open at a known dead end (0.83–0.97) and stay half-closed at
   a fresh junction (0.40–0.82): memory switches on *after* the events that make the past relevant.
2. **Knowing now turns into acting.** With the belief bias, a known dead-end exit is avoided 61–70% of the time
   (v3: 37–48%), and v3.2 has the fewest dead-end re-entries on rings / hub with the cue (1.22 per dive).
3. **Memory use can be measured.** Re-scoring decisions with the memory removed showed that at a known dead end
   memory changed the choice 39% of the time after the warm start, against ~9% at fresh junctions.
4. **And it showed the real bottleneck:** during RL that 39% fell to 7%, while RL strengthened the simple belief
   rule (w_dead −1.73 → −1.82). On caves that do not reward memory, RL keeps only its cheapest useful part.
   Hence v3.3.

### Exit memory: the design

Round 1 showed the navigators **recalled the right memory and ignored it**. The causes were a free "carry on"
action, no training signal for what the memory should hold, and duplicate writes. Memory v3 redesigns the
navigator's memory around one question per exit: *what do I know about this exit?* Every piece comes from
published navigation-memory work. The [design document](docs/memory_v3.md) gives the reasoning and sources.

```
 observation (61) --> encoder --> GRU ------------------------------------ context c
                                    |                                         |
 STORE (32 entries, sparse writes)  |   write: DEPARTURE when leaving a junction (pos, heading taken, memo)
   position | heading | memo(16)    |          DEAD END when one is detected   (pos, memo)
   dead-end | used | age            |          same place again -> UPDATE that entry (no flooding)
                                    v
 for each exit k (direction e_k):  LOCAL read  "did I leave from HERE along e_k?"    (attention + rule)
                                   AHEAD read  "is there a dead end DOWN e_k (0-14 m)?" (attention + rule)
                                    |
                    memory gate g (starts ~0.12, opens as memory pays off)
                                    v
 exit scorer: [exit features, g * memory_k, c] -> score_k  ->  softmax over VALID exits (+ turn back)
 aux heads (ungated memory): loop closure, per-exit "taken before", per-exit "known dead end"
```

| Piece | What it fixes | From |
|---|---|---|
| Per-exit reads, pointer-style policy | "recalled but not used": memory's answer sits on the exit it concerns | Scene Memory Transformer, SPTM |
| Auxiliary losses from simulator ground truth | no signal for what to store; memory trained densely, every decision | Nav A3C (loop closure), UNREAL, MERLIN |
| Sparse, merged writes keyed by position | duplicates evicting useful memories | Neural Map |
| Memory gate initialised nearly closed | noisy early memory teaching the policy to ignore it | GTrXL |
| Scenarios where the present misleads (`decoy`, `loops`, no goal cue) | memory not needed to score well | Memory Maze (isolate memory) |
| Warm start over whole episodes (state carried) | random-start policies collapsing to a default | our driver and round-2 results |
| Stored-state recurrent PPO (custom, 4 variants) | SB3 has no aux-loss hook | R2D2 / recurrent PPO practice |

Four navigators are trained identically: `none` (MLP), `gru` (POPGym's strong baseline), `graph`
(rule-based reads over the same store, an explicit junction graph), and `exit` (learned reads). They differ
**only** in their memory. The test reports success per cave type plus two memory-specific rates:
- **avoid:** a known dead-end exit is offered; is it avoided?
- **untried:** at a revisited junction, is an untried exit taken?

There is also a memory-wipe ablation.

**Training speed (meant to be exact; equivalence tests written, to run first):** the navigator turns off the base env's per-step opening count, which only
the old end-to-end agents use. The hand controller's scan-only terms are cached per sonar ping, and hot scalar
`np.clip` calls are replaced. Together that is about 25–30% less CPU per decision. Workers stay PyTorch-free,
and the learner's memory is a few MB: rollouts are 32 × 64 × 72 floats, and the recurrent state is at most
800 floats per env.

### Running it (v3)

```powershell
# 0. tests (new: tests/test_memnav.py), then a 2-minute smoke run
python -m pytest tests/test_memnav.py -q
python -m scripts.collect_nav_demos --config configs/memnav.yaml --decisions 3000 --workers 4 --out runs/memnav_demos_smoke.npz
python -m scripts.train_memnav --variant exit --scale 0.01 --workers 2 --name memnav_smoke --demos runs/memnav_demos_smoke.npz

# 1. teacher demonstrations (~10 min), 2. four variants in parallel (~2-2.5 h), watch live
python -m scripts.collect_nav_demos --config configs/memnav.yaml
foreach ($v in "none","gru","graph","exit") { Start-Process .venv\Scripts\python.exe "-m scripts.train_memnav --variant $v" }
python -m scripts.watch_training runs/memnav_none runs/memnav_gru runs/memnav_graph runs/memnav_exit

# 3. evaluation: all cave types, memory scenarios, blind dives, memory-wipe ablation
python -m scripts.eval_navigator --config configs/memnav.yaml --policies greedy oracle --models runs/memnav_none runs/memnav_gru runs/memnav_graph runs/memnav_exit
python -m scripts.eval_navigator --config configs/memnav.yaml --layouts maze decoy loops --goal-cue-prob 0 --wipe-memory --models runs/memnav_gru runs/memnav_graph runs/memnav_exit
python -m scripts.nav_videos --config configs/memnav.yaml --models runs/memnav_exit --layout decoy --caves 1000000
```

**v3.1: a reward that prices forgetting** (`configs/memnav_reward.yaml`). Under the dense reward, a wasted dead-end
detour cost only about 5 reward, refunded on the way out, so memory had nothing to gain. From stage 2 on:
- progress is paid only for beating the best distance to the goal so far;
- small repeat penalties: −1.5 for an exit that already led to a dead end, −0.3 for re-taking an exit while one at
  the same junction is untried. Forward choices only: backtracking to a far junction is never penalised;
- −1.5 for arriving again at a dead end already reached and left, and −0.3 for turning back again within 2
  decisions (the U-turn loop out of a dead end). v3 models re-entered 0.55 dead ends per dive, as many as greedy;
- +0.5 per new 2 m cell in dives without the goal cue;
- a bigger exit bonus (40), time costs twice as much, and the goal cue drops to 30%.

Also: fixed labels and the gate starting at 0.5. Details in
[docs/memory_v3.md](docs/memory_v3.md#v31-a-reward-that-prices-forgetting-configsmemnav_rewardyaml).

```powershell
python -m scripts.collect_nav_demos --config configs/memnav_reward.yaml          # demos with the fixed labels
python -m scripts.train_memnav --config configs/memnav_reward.yaml --variant none    # -> runs/memnav_r_none
python -m scripts.train_memnav --config configs/memnav_reward.yaml --variant exit    # -> runs/memnav_r_exit
python -m scripts.compare_videos --config configs/memnav_reward.yaml --reward-stage m3_full --no-cue ...   # reward shown live
```

**v3.1 results (22 Nov): the reward made both models explore, not remember.** Same 7 cave types × 30 test caves as
v3, goal cue shown:

| | 5 standard types | All 7 | Dead-end re-entries / dive | Untried exit at a revisited junction |
|---|---|---|---|---|
| v3 no memory | 73.3% | 77.1% | 0.55 | 8% |
| v3 exit memory | 72.7% | 75.7% | 0.55 | 6% |
| v3.1 no memory | 72.0% | 76.7% | 0.57 | **24%** |
| v3.1 exit memory | 70.7% | 71.9% | 0.63 | **16%** |
| greedy / oracle | 82.0% / 96.0% | 80.5% / 97.1% | 0.58 / 0.24 | |

The re-taken penalty worked (both models now try untried exits 2–4× more often), but dead-end avoidance did not
improve. Blind (no goal cue, maze / decoy / loops), the **memoryless** v3.1 model was the best of all: 76.7%
success and 0.26 dead-end re-entries per dive, almost the oracle's 0.24.

### Why memory kept losing: the caves did not need it

A memoryless policy that re-enters dead ends as rarely as the oracle must have learned a **reactive exploration
rule**. The classic one is wall following: keep one hand on the wall and you reach the exit of any cave whose
start and exit lie on the same wall, without remembering anything. Every layout above starts and ends on the
outer wall. So the architectures were never tested on what memory is for, and the policies learned to ignore it.

**New baselines:** `wall-left` / `wall-right` (`WallFollowerNavigator`: the forward exit furthest to that side,
turn back only at dead ends; no memory, no goal cue).

**New memory caves** (`submarine/world/layouts.py`; tested: both walls at a rings start are islands):
- `rings`: the dive starts **inside** two concentric rings of passage. The inner ring's only way out is a gap on
  the far side from the exit; the exit leaves the outer ring. Either hand on the wall circles a pillar of rock
  forever, and the goal bearing keeps saying "carry on round". Getting out needs loop closure.
- `hub`: one chamber with 3–5 spokes. Most are dead ends (8–11 m, beyond sonar range) **pointing at the exit**;
  the true route leaves at ~90°, and often a dead end sits behind it so neither hand finds it first. Every failed
  spoke returns the sub to the same chamber, which looks the same: only remembering the tried spokes moves on.

| 30 test caves each | rings | hub | Both | Dead-end re-entries / dive |
|---|---|---|---|---|
| Oracle (reads the map) | 100% | 100% | **100%** | 0.02 |
| v3.1 no memory | 60% | 63% | 61.7% | 1.40 |
| v3 exit memory | 57% | 60% | 58.3% | **1.32** |
| v3 no memory | 40% | 60% | 50.0% | 1.65 |
| Wall-left / wall-right | 43% / 37% | 40% / 37% | 41.7% / 36.7% | 2.10 / 1.70 |
| v3.1 exit memory | 33% | 33% | 33.3% | 1.50 |
| Greedy (goal bearing) | 0% | 10% | 5.0% | 3.17 |

Goal cue shown; blind gives the same picture (v3 exit 56.7% vs its memoryless twin 53.3%, re-entries 1.22 vs
1.68). Every reactive rule collapses (greedy 82% → 5%), every model is 40–65 points below the oracle, and **v3
exit memory re-enters dead ends ~25% less than v3 no memory, with or without the cue**: the first consistent
sign of memory in behaviour. None of these models trained on rings or hub.

### v3.2: memory gated by events (`configs/memnav_event.yaml`)

The v3/v3.1 gate was one learned number. It never moved (0.17, then 0.53 flat), because the exit scorer can
rescale the same features itself, so the gate got no learning signal. And nothing tied the beliefs the memory
learned (dead-end recall ~0.9) to the choice. v3.2 (`submarine/memory/exitmem.py`, `gate: event`,
`belief_bias: true`):
- **Event-triggered gate, one per exit**, from the events that make the past relevant: a dead end hit recently,
  a recent turn-back, no new ground for a while (decaying traces in the recurrent state), the model's own revisit
  belief, entries stored nearby, no goal cue, being at a junction, and that exit's own beliefs. It starts at 0.5
  everywhere and learns *when* memory matters.
- **Belief bias:** score_k += g_k · (w_dead · p_dead_k + w_taken · p_taken_k), w learnable from (−2, −0.5), so
  "I believe this exit is a dead end" lowers its score from the first update.
- **Reliance probes** (logged as `memory/*`): every 4th decision is re-scored with the memory read zeroed, which
  shows how much it changes the choice, split by situation: known dead end ahead / revisit / fresh junction.
- Replays show the gate and its triggers ("memory gate 0.88 ← dead end 1 dec. ago, no cue"), a `mem 0.xx` per
  exit, and GATE OPENED log lines.

What training showed:

| | End of warm start | End of training |
|---|---|---|
| Gate at a known dead end / a fresh junction | 0.97 / 0.82 | **0.83 / 0.40** |
| Choice changes without memory, at a known dead end | 39% | 7% |
| Belief bias w_dead / w_taken | −1.73 / −0.35 | **−1.82 / −0.42** |
| Dead-end recall | 0.94 | 0.65 |
| Success by stage (v3.1 exit for comparison) | | 77% / 71% / 51% (68% / 68% / 55%) |

- **The gate learned when to use memory**: twice as open at known dead ends as at fresh junctions.
- **RL then pruned the rich memory** (39% → 7% of choices at known dead ends depend on it) and kept only the
  simple belief rule, which it strengthened. On caves that do not reward memory, the cheapest policy wins.
- Stage 2 (71%) was the best of any run; full difficulty with currents (stage 3) drags every memory model down.

Test results (30 test caves per type; "avoid" = a known dead-end exit is offered next to another, and it is
avoided):

| | Standard 5 / all 7 (cue) | Blind maze, decoy, loops | rings + hub (cue / blind) | Avoid (old caves / rings+hub) | Re-entries (rings+hub, cue) |
|---|---|---|---|---|---|
| v3 exit memory | 72.7% / 75.7% | 68.9% | 58.3% / 56.7% | 48% / 45% | 1.32 |
| v3.1 no memory | 72.0% / 76.7% | **76.7%** | **61.7% / 60.0%** | 59% / 60% | 1.40 |
| **v3.2 event gate** | 66.7% / 71.0% | 75.6% | 48.3% / 55.0% | **61% / 70%** | **1.22** |

- **It acts on its memory more than any model so far:** when a known dead-end exit is offered, v3.2 avoids it
  61–70% of the time (v3 exit 37–48%), and it has the fewest dead-end re-entries on rings / hub with the cue.
- **That has not turned into more successes.** It is the best memory model blind on the old memory layouts
  (75.6%, level with the memoryless v3.1), but weaker with the cue (66.7% on the standard types, 48% on rings /
  hub with the cue). Avoiding the dead ends it knows is not enough when it still wanders elsewhere.
- So the belief bias works as designed. What is missing is training where remembering pays, which is v3.3.

### v3.3: train on the memory caves (`configs/memnav_caves_clean.yaml`)

The v3.2 model, reward and training settings, with rings and hub as a third of stages 2–3 (hub also in stage 1)
and teacher demos that include them: the cave mix is the only change from v3.2. Two runs: `exit` (event gate + belief bias) and a memoryless control `none` on the same caves
(`train_memnav` drops the gate / bias settings for variants without a store). Evaluation uses different cave seeds
from training, so rings / hub stay unseen caves. **The question:** with caves where forgetting costs, does RL keep
the memory instead of pruning it, and does exit memory beat its memoryless twin on rings / hub?

**What the v3.3 memory model should be able to do** (and what to look for in the replays):
- **rings:** go round the inner ring at most once, recognise the place ("RECOGNISED a place it has been"), and at
  the gap take the exit it has not tried, instead of circling. The memoryless control and the wall follower are
  expected to circle.
- **hub:** after a dead-end spoke, come back to the chamber and choose a *different* spoke ("AVOIDED … believes
  dead end"), instead of re-entering the one it just left. Greedy is expected to re-enter the same spoke forever.
- **its gate:** open after a dead end or on a revisit ("memory gate 0.9 ← dead end 1 dec. ago"), low at fresh
  junctions.
- **in numbers:** clearly fewer dead-end re-entries and higher success than the v3.3 control on rings / hub (with
  and without the goal cue), the reliance probes staying high at known dead ends through training instead of
  falling to ~7%, and no loss on the old cave types. The oracle (100%) is the ceiling; "clearly better than its
  memoryless twin" is the bar.

#### v3.3 results (23 Nov 2026)

Training (success in the last stage window): exit 66 / 68 / 51% over stages 1–3, control 79 / 66 / 50%. The exit
model trains at ~30 decisions/s against ~55/s for the control (3.1 h vs 1.4 h in all, warm start included).

**Rings and hub** (30 unseen caves each):

| | With cue: rings / hub / **both** | Blind: **both** | Dead-end re-entries per dive (cue / blind) | Known dead end avoided (cue / blind) |
|---|---|---|---|---|
| **v3.3 exit memory** (`memnav_cc_exit`) | 67% / 77% / **71.7%** | **68.3%** | **1.10 / 1.02** | **73% / 69%** |
| v3.3 control, no memory (`memnav_cc_none`) | 53% / 63% / 58.3% | 56.7% | 1.35 / 1.45 | 61% / 62% |
| v3.2 event gate (not trained on these caves) | 43% / 53% / 48.3% | 55.0% | 1.22 / 1.62 | 70% / 70% |
| v3 exit memory | 57% / 60% / 58.3% | 56.7% | 1.32 / 1.22 | 45% / 37% |
| Wall-left / greedy / oracle (cue) | 41.7% / 5.0% / 100% | | | |

- **+13 points with the cue, +12 blind** over the twin that differs only in memory, and the best learned model on
  rings / hub so far (~10 points above every earlier version).
- **Paired on the same caves** (cue and blind, 120 pairs): exit succeeds where the control fails 22 times, the
  reverse 7 times; two-sided sign test **p = 0.008**.
- Dead-end re-entries **−19% with the cue, −30% blind**.

**The old caves** (with cue): no loss, the best v3 model so far.

| | 5 standard types | All 7 | Dead-end re-entries |
|---|---|---|---|
| **v3.3 exit memory** | **74.7%** | **77.6%** | 0.63 |
| v3.3 control | 70.0% | 72.9% | 0.69 |
| v3 exit / v3.1 no memory | 72.7% / 72.0% | 75.7% / 76.7% | 0.55 / 0.57 |
| Greedy | 82.0% | 80.5% | 0.58 |

**Weak spot:** blind on the old maze / decoy / loops caves, 68.9% against the control's 72.2% (small, not
significant). Greedy still leads on the old caves, where following the bearing is enough.

**What it shows:** v3.2's event gate and belief bias were the right architecture; what memory lacked was training
caves where forgetting costs. (A first v3.3 attempt with faster training settings reached only 36.7% on rings / hub;
those settings, not the caves, were the problem, and were dropped.)

**Comparison videos** (`videos/10_memory_caves_clean`; rings and hub, blind dives), six navigators side by side:

| In the video | Run | Memory | Trained on rings / hub |
|---|---|---|---|
| New memory model | `memnav_cc_exit` (v3.3) | yes (event gate + belief bias) | yes |
| New model without memory | `memnav_cc_none` (v3.3 control) | no | yes |
| Event-gate model | `memnav_e_exit` (v3.2) | yes (event gate + belief bias) | no |
| Old memory model | `memnav_exit` (v3) | yes | no |
| Wall follower | `nav-wall-left` (a fixed rule, not trained) | no | |
| Oracle | `nav-oracle` (a fixed rule that reads the true map) | sees the map | |

**What the videos show** (recorded 23 Nov; blind, d = 0.5; ✓ = exit reached, ✗ = 150 s timeout). The first two
caves are the hardest ones the oracle solves (a stress test); the last two were picked to show the v3.3 memory model:

| Cave | v3.3 memory | v3.3 control | v3.2 | v3 | Wall-left | Oracle |
|---|---|---|---|---|---|---|
| rings 1000007 (hardest; 0 of 11 navigators solved it in evaluation) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ 57 s |
| hub 1000013 (hardest; 1 of 11) | ✗ | ✗ | ✗ | ✗ | ✓ 58 s | ✓ 56 s |
| **rings 1000008** | **✓ 61 s** | ✗ | ✗ | ✗ | ✗ | ✓ 61 s |
| **hub 1000000** | **✓ 82 s** | ✗ | ✓ 84 s | ✗ | ✗ | ✓ 54 s |

Watch `rings_cave1000008_*` (only the v3.3 memory model and the oracle get out, equally fast) and
`hub_cave1000000_*` (the memoryless control, v3 and the wall follower all time out). The full per-video table is
in [videos/10_memory_caves_clean/README.md](videos/10_memory_caves_clean/README.md); the `.mp4` files (1.1 GB) are
git-ignored like all replays and are regenerated with the command below.

```powershell
python -m scripts.collect_nav_demos --config configs/memnav_caves_clean.yaml --workers 8 --out runs/memnav_c_demos.npz
python -m scripts.train_memnav --config configs/memnav_caves_clean.yaml --variant exit    # -> runs/memnav_cc_exit
python -m scripts.train_memnav --config configs/memnav_caves_clean.yaml --variant none    # -> runs/memnav_cc_none
python -m scripts.eval_navigator --config configs/memnav_reward.yaml --layouts rings hub --policies greedy oracle wall-left wall-right --models runs/memnav_cc_none runs/memnav_cc_exit
python -m scripts.compare_videos --config configs/memnav_caves_clean.yaml --reward-stage m3_full --no-cue --layouts rings hub --navigators runs/memnav_cc_exit runs/memnav_cc_none runs/memnav_e_exit runs/memnav_exit wall-left oracle --csv runs/eval/memcaves_cue_episodes.csv --out videos/10_memory_caves_clean
```

## Features

- **Procedural caves** from an integer seed, with a difficulty knob d ∈ [0, 1]: width, cracks down to 1.7 m,
  rough walls, pillars, winding paths. Every cave is checked to be solvable. Train / validation / test seeds are disjoint.
- **Cave layouts modelled on real exploration hazards:** single tunnel, maze with dead ends, misleading side
  branches, glacier crevasse with squeezes, boulder choke; plus a silt-out scenario.
- **Energy model:** power from thrust, turning and on-board load; an evaluation metric and an optional reward penalty.
- **Vehicle:** thrust with drag, a lagged turn-rate command, optional water current; 10 Hz control.
- **Sensors:**
  - 32-beam 360° sonar: noise, dropout, configurable ping rate, field of view and beam width.
  - Forward camera: 64-pixel scanline with an underwater image model (attenuation, backscatter, speckle).
  - Murky-water patches that blind the camera but not the sonar.
- **Gymnasium environment** with reward shaping based on shortest-path distance through the cave
  (rewards progress *through the passage*, not toward a goal behind a wall).
- **Baselines:** random, follow-the-gap, wall centering, and a cost-aware planner on the true map (oracle).
- **Robustness:** random sensor failures (flagged and silent), tiered blindness supervisor, hull-damage contact
  model, dead reckoning with drift, and a junction / dead-end detector.
- **Training:** PPO, SAC, RecurrentPPO; a 1D-CNN + MLP extractor for camera + state observations; curriculum over
  difficulty, layouts and energy penalty; YAML configs; validation-based checkpointing; live validation dashboard;
  TensorBoard; memory-light parallel workers (~58 MB each).
- **Evaluation:** success / collision / timeout, path efficiency, wall clearance, time to goal, collisions in murk;
  every policy gets the same caves, noise and current.
- **Visualization:**
  - Game-style renderer (ice or rock theme, follow camera with minimap, live camera strip).
  - Plain analysis view.
  - MP4 / GIF replays with several policies side by side.
  - Keyboard driving.

## Setup

Requires Windows / Linux, Python 3.12. A CUDA GPU is optional; small networks train fine on the CPU.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt         # pulls the CUDA 13.0 build of PyTorch
pytest                                  # full suite, ~1 min
```

If PowerShell refuses to run the activation script:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

## Quick start

```powershell
# drive the submarine yourself (arrows/WASD, R restart, N/P next/previous cave, G distance bands)
python -m scripts.drive --difficulty 0.5
python -m scripts.drive --difficulty 1 --theme rock --camera full

# look at generated caves
python -m scripts.plot_caves --geodesic                     # -> runs/figures/caves.png
python -m scripts.plot_scenarios                            # -> docs/figures/scenario_catalogue.png

# test any agent on a scenario (layouts: tunnel maze branches crack breakdown)
python -m scripts.evaluate --policies gap oracle --models runs/archive/ppo_mlp_mixed_s0 --layout maze --max-steps 1500
python -m scripts.evaluate --config configs/scenario_siltout.yaml --policies gap --models runs/archive/ppo_fused_murky_s0

# evaluate baselines on held-out caves (tables + CSV + path plots)
python -m scripts.evaluate                                   # all baselines, 50 caves x 5 difficulties
python -m scripts.evaluate --config configs/eval_murky.yaml --policies gap oracle --models runs/archive/ppo_fused_murky_s0

# train an agent (outputs in runs/<name>_s<seed>/), watch with: tensorboard --logdir runs
python -m scripts.train --config configs/ppo_mixed.yaml
python -m scripts.train --config configs/ppo_fused_murky.yaml --seed 1

# replay videos (several policies side by side on the same cave)
python -m scripts.record --policies gap --models runs/archive/ppo_mlp_mixed_s0 --difficulties 0.75 1.0 --caves 2
python -m scripts.record --models runs/archive/ppo_mlp_mixed_s0 --difficulties 1.0 --camera follow --format gif
```

## Configurations (`configs/`)

| File | Purpose |
|---|---|
| `default.yaml` | All environment settings with their defaults |
| `ppo_d0.yaml` | PPO on easy caves only (stage 2) |
| `ppo_mixed.yaml` | PPO on mixed difficulty (stage 3) |
| `ppo_fused_murky.yaml`, `ppo_sonar_murky.yaml`, `ppo_camera_murky.yaml` | Sensor experiment in murky water (stage 5) |
| `eval_murky.yaml` | Evaluation conditions for the sensor experiment |
| `scenario_siltout.yaml` | Silt-out scenario (one huge dense murk cloud) |
| `eval_layouts.yaml` | Evaluation conditions for the cave-layout scenarios (d 0.5, slow sonar, energy penalty) |

A training config has run settings (`algo`, `policy`, `total_timesteps`, `n_envs`, `vec_env`, `device`,
`algo_kwargs`, `policy_kwargs`, `eval`) and an `env:` section with any environment setting.
`learning_rate: lin_3e-4` means linear decay to 0. `features_extractor_class: camera_state` selects the CNN.
Environment options added later: `layout`, `max_steps`, `sensors`, `turbidity`, `energy`, `reward.k_energy`,
`navigation`, `failures`, `supervisor`, `collision`.

## Project structure

```
submarine/
  world/       procedural cave generator, scenario layouts, geodesic distance, seed splits
  sim/         vehicle dynamics, sonar, camera, turbidity, simulator
  env/         Gymnasium environment, config (energy, failures, supervisor, hull damage, navigation sensing),
               torch-free worker for parallel envs
  control/     NEW: driver (heading/speed controller + integrated safety filter)
  navigator/   NEW: openings detector + decision-level environment
  memory/      NEW: retention + junction memory for the navigator
  agents/      training config, model/env factory, CNN extractor, callbacks, parallel envs
  baselines/   random, gap follower, wall centering, oracle planner
  evaluation/  metrics, dead-end analysis, parallel evaluation, SB3 policy adapter
  render/      game-style viewer, art, replay videos
scripts/       drive, plot_caves, plot_scenarios, evaluate, eval_memory, train, record, live_status
configs/       YAML configs
tests/         pytest suite
docs/figures/  figures used in the docs
videos/        replay videos organized by development stage (git-ignored, regenerate with scripts.record)
runs/          training runs and evaluations (git-ignored); runs/archive/ holds the end-to-end era
```

## The environment in one table

| | |
|---|---|
| World | 50 m × 30 m, 0.2 m grid, top-down 2D |
| Action | `[thrust, yaw command]` ∈ [−1, 1]², 10 Hz |
| Vehicle | max 1.5 m/s (reverse 0.75), turn rate up to 1 rad/s with 0.3 s lag, disc radius 0.4 m |
| Observation (sonar) | 32 ranges + speed + yaw rate + goal bearing/distance + previous action (39 values) |
| Observation (camera) | `{"camera": 1×64 image, "state": vector}` |
| Reward | +1 per metre of shortest-path progress, −0.01 per step, −0.01‖Δa‖², +20 goal, −10 collision |
| Episode end | goal within 1.5 m, collision, or 60 s |

Full details in [context.md](context.md) and [progress.md](progress.md).

## Notes

- **Windows memory:** SB3's `SubprocVecEnv` loads PyTorch in every worker (~2 GB commit each with the CUDA build).
  The default `vec_env: light` uses torch-free workers (~58 MB each). The scripts also cap NumPy's BLAS threads,
  which would otherwise reserve ~1.5 GB per process on a 32-thread CPU. Use `--vec-env dummy` to keep everything
  in one process.
- **Several trainings at once:** use `--threads 4` (PyTorch threads per learner) so they don't oversubscribe the CPU.
- **GPU:** not worth it for these small networks. Updates get faster, but acting one step at a time is slower on
  the GPU, so end-to-end training was 2.7× slower (see the A/B check in progress.md). Train on CPU.
- **Reproducibility:** caves, sensor noise and currents are seeded. Evaluation uses fixed per-episode seeds, so all
  policies face identical conditions and replays reproduce evaluation numbers exactly.

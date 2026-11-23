# Memory v3: an exit memory the navigator has to use

The full lab notebook of the memory work (v3 → v3.3, 22–23 Nov 2026), merged into `main`; milestones are tagged
`v3`, `v3.1`, `v3.2` and `v3.3`. The design and plan come first, written before training; results follow in order.
**Outcome:** v3.3 is the first memory model to clearly beat its memoryless twin
(71.7% vs 58.3% on unseen rings / hub caves, p = 0.008), see [v3.3 results](#v33-results-23-nov-2026).

## Why a third design

Round 1 (`main`, stage 9 phase B in [progress.md](progress.md)) trained three navigators: none, retention,
and retention + junction memory. The memory versions were *worse* (69–72% success vs 82%). The replays showed
why:

1. **A free "carry on" action** (empty exit slot = keep heading, 0.1 penalty). Retention put 64% of its
   probability on it at maze junctions and pressed into dead ends until timeout.
2. **Recall without action.** Junction memory recalled the stored dead end with 100% attention for 16 s and
   still pressed on: nothing had taught the policy what a recalled memo means.
3. **Duplicate writes.** A dead end was re-written at every decision there. It filled all 8 slots and evicted
   every junction.

Round 2 (`main`) fixes 1 and 3 with exit remapping and merged writes, and helps 2 with a teacher warm start.
What round 2 does not add is a **reason to need memory** or a **direct training signal for it**. In most caves
the goal bearing plus the visible exits are enough, so a network can do well while ignoring its memory. Memory v3
adds both.

## What the literature says, and what we take from it

| Work | How it made agents use memory | What we take |
|---|---|---|
| **UNREAL** (Jaderberg et al. 2016, [arXiv:1611.05397](https://arxiv.org/abs/1611.05397)) | Self-supervised auxiliary tasks on the shared CNN+LSTM (pixel control, reward prediction with rare events rebalanced). Labyrinth: 87% vs 54% of human score, ~10× faster. | Dense auxiliary targets; oversample the rare events (here: revisits, known dead ends). |
| **Nav A3C** (Mirowski et al. 2017, [arXiv:1611.03673](https://arxiv.org/abs/1611.03673)) | Stacked LSTM plus **loop-closure classification** ("been here before?", from SLAM) and depth prediction as auxiliary losses. Learning sped up dramatically. | **Loop-closure head**, labels free from the simulator. |
| **Neural Map** (Parisotto & Salakhutdinov 2017, [arXiv:1702.08360](https://arxiv.org/abs/1702.08360)) | Memory structured like space, written **only at the agent's position** (sparse writes). Beat LSTM and memory networks, and generalised to unseen mazes. | Position-keyed, event-gated, **merged** writes. |
| **MERLIN** (Wayne et al. 2018, [arXiv:1803.10760](https://arxiv.org/abs/1803.10760)) | Memory written every step; its content is trained by a **predictive** objective; the policy only reads it (no policy gradient into memory). | The memory's content is trained by prediction (our aux heads read the *ungated* memory). |
| **SPTM** (Savinov et al. 2018, [arXiv:1803.00653](https://arxiv.org/abs/1803.00653)) | Non-parametric **graph of places** + learned retrieval; ~3× baseline success. Weakness: needs an exploration trajectory to build. | The **`graph` variant**: an explicit junction graph built online from the openings detector. It is the baseline the learned memory must beat. |
| **Scene Memory Transformer** (Fang et al. 2019, [arXiv:1903.03878](https://arxiv.org/abs/1903.03878)) | Store every observation (with pose) and attend over all of it, instead of compressing history into a recurrent state. | A decision-level store (a dive is ~30–100 decisions, so everything fits), with **per-exit attention**. |
| **GTrXL** (Parisotto et al. 2020, [arXiv:1910.06764](https://arxiv.org/abs/1910.06764)) | Gated residuals + pre-layer-norm make attention memory trainable under RL. It beat LSTMs on memory levels and matched them on reactive ones. | A **memory gate initialised nearly closed**, so the policy starts reactive and opens the gate as memory pays off. |
| **Grid cells** (Banino et al. 2018, [DeepMind](https://deepmind.google/blog/grid-cells/)) | Path-integration training produces grid-like codes that enable shortcuts. | Not needed yet (dead-reckoned pose is an input); relevant once drift matters. |
| **POPGym / Memory Gym / Memory Maze** ([2303.01859](https://arxiv.org/abs/2303.01859), [2309.17207](https://arxiv.org/abs/2309.17207), [2210.13383](https://arxiv.org/abs/2210.13383)) | PPO + GRU is the strongest simple baseline; isolate memory from other skills; **probe** what the memory holds. | The **`gru` variant**; memory scenarios; memory-specific metrics and the wipe ablation. |
| **Recurrent PPO practice** ([Pleines et al. 2022](https://arxiv.org/abs/2205.11104), [stale states](https://kandouss.github.io/blog/ppo_stale_states)) | Train each sequence from its stored start state; refreshing stale states matters on memory-heavy tasks. | Stored-state sequences (= the rollout, 64 decisions); the warm start runs over **whole episodes** with the state carried. |

## The design (`submarine/memory/exitmem.py`)

**Store:** 32 entries of (position, heading, 16-d learned memo, dead-end flag, used, age).
- **Writes are event-gated:** a departure is written when the previous decision was taken at a junction (from the
  observation's previous-decision fields); a dead end is written when one is detected.
- **Same place:** a write within 1.5 m of an entry of the same kind (departures also need the same direction)
  updates that entry. Otherwise it goes to a free entry, else the oldest.

**Per-exit reads:** for each exit k with direction e_k:
- **Local:** attention over entries with logit = q(c, e_k)·memo/√16 − a·|p − pos|²/2σ² + b·(e_k·heading) +
  d·dead. It answers "did I leave here along e_k before, and what happened?". A rule-based strength (distance
  kernel × direction match over departures) is added.
- **Ahead:** attention over entries lying near the ray from here along e_k (0–14 m), with a perpendicular-distance
  bias. It answers "is there a known dead end down this exit?". A rule-based strength is added.

**Policy:** score_k = MLP([exit features, gate · memory_k, context]). The softmax runs over valid exits only,
plus turn back, so there is no free action. Value = MLP([context, mean of gated memory]).

**Auxiliary heads (ungated memory):** loop closure; per-exit "taken before"; per-exit "known dead end". Targets
come from the true trajectory (`NavigatorEnv._aux_labels`) and ride at the end of the observation, which survives
vector-env auto-resets. Models read only the first 61 values; a test checks this.

**Variants (identical except memory):**

| Variant | Recurrence | Store | Reads |
|---|---|---|---|
| none | – | – | – |
| gru | GRU 64 | – | – |
| graph | GRU 64 | yes | rule-based only (explicit junction graph) |
| exit | GRU 64 | yes | learned attention + rule strengths |

Parameters 32k / 71k / 72k / 87k. Recurrent state 0 / 64 / 800 / 800 floats per env.

## Scenarios where only memory helps (`submarine/world/layouts.py`)

- **`decoy`:** the true route detours along the far edge of the cave. From the detour, 2–3 dead-end passages
  (7–11 m, ending out of easy sonar view) leave 35–65° toward the goal side. A decoy is kept only if the goal
  bearing is closer to its direction than to the true route's, so a goal-follower enters every one. Only memory
  prevents entering it again.
- **`loops`:** a maze plus 3–5 extra passages that close cycles. Without memory an agent can circle.
- **No goal cue:** 30% of training dives (stages m2–m3) withhold the goal bearing ("find the exit"). Systematic
  exploration then needs memory.

## Training regimen (`configs/memnav.yaml`, `scripts/train_memnav.py`)

| Phase | What | Budget |
|---|---|---|
| Demos | Map-aware teacher (`OracleNavigator`). 35% random exits **at junctions** (wrong branches → dead ends → returns), 10% elsewhere | 60k decisions, ~10 min |
| Warm start | Behaviour cloning over whole episodes (8 lanes, windows of 64, state carried) + value + aux losses, 10 epochs (~1,100 updates; the smoke run showed 32 lanes gave too few updates) | ~10–20 min per variant |
| m1_core | tunnel, branches, maze, decoy; d 0–0.5; goal cue always | 60k decisions |
| m2_memory | maze ×2, decoy ×2, loops ×2, branches, breakdown, crack, tunnel; d 0–0.75; 30% blind | 100k |
| m3_full | same mix; d 0–1; currents up to 0.2 m/s; 30% blind | 80k |

PPO: rollouts of 64 decisions × 32 envs, 4 epochs, 8 sequences per minibatch, lr 3e-4, clip 0.2, entropy
0.01, early stop at KL 0.03. Auxiliary weight annealed from 0.5 to 0.1.

**Compute:** four variants in parallel, about 5–6 worker processes each, should take ~2–2.5 h at ~25–35
decisions/s per run (to be measured).

**Memory:** the learner holds about 1 MB of rollout data, and the workers stay PyTorch-free.

## Evaluation

`scripts/eval_navigator.py` (with `--config configs/memnav.yaml`) reports per cave type:
- success / breach / timeout, dead-end entries and re-entries, backtracking;
- **avoid:** % of choices avoiding an exit known (true history) to lead to an explored dead end;
- **untried:** % of choices at revisited junctions that take an exit not taken before.

Options:
- `--goal-cue-prob 0`: blind dives;
- `--wipe-memory`: each model again with its memory erased at every decision. The gap measures how much the
  memory is actually used.

Replays (`scripts/nav_videos.py`) show the decoded store on the map (STORED / UPDATED / REPLACED, checked
against the write flags) and the memory's verdict on each exit arrow ("[known dead end]", "[taken]").

## What would count as success

- `exit` (and `graph`) beat `none` and `gru` on **avoid** and **untried** in maze, decoy and loops.
  Wiping the memory removes that advantage.
- Decoy dead ends entered at most once, with re-entries near 0.
- No loss on easy caves (tunnel, crevasse).
- If `graph` matches `exit`, the explicit graph is enough at this scale. Learned memory would then need to earn
  its place by robustness, for example under dead-reckoning drift. That is still a useful result.

## Speed changes in this branch (meant to be exact: same trajectories; tests written, to run first)

| Change | Saving per decision |
|---|---|
| Navigator switches off CaveEnv's per-step opening count (`navigation.junction_flags`), used only by the end-to-end agents | ~11% |
| Hand controller's scan-only terms (corridor reach, front distance, side minima) cached per sonar ping | ~5–7% |
| Scalar `np.clip` in hot paths replaced by `min`/`max` | ~5% |

## Results of the first v3 run (22 Nov 2026): negative, with known causes

Four variants trained in parallel: 240k decisions each, ~2.5–3 h, 5 workers each. Then a test on 30 held-out caves
per cave type (d = 0.5). Raw tables, per-episode CSVs and training logs are in [`results/`](results/).

| Success | Tunnel | Maze | Branches | Crevasse | Boulder choke | Decoy | Loops | 5 standard types | All 7 |
|---|---|---|---|---|---|---|---|---|---|
| Oracle (true map) | 100% | 100% | 100% | 100% | 80% | 100% | 100% | 96.0% | 97.1% |
| Greedy heuristic | 100% | 70% | 97% | 90% | 53% | 63% | 90% | 82.0% | 80.5% |
| none (v3) | 100% | 53% | 77% | 93% | 43% | 100% | 73% | 73.3% | 77.1% |
| gru | 100% | 37% | 80% | 93% | 50% | 100% | 70% | 72.0% | 75.7% |
| graph | 100% | 33% | 67% | 93% | 50% | 100% | 63% | 68.7% | 72.4% |
| exit | 100% | 47% | 77% | 93% | 47% | 100% | 67% | 72.7% | 75.7% |
| *Round 2 retention (`main`)* | *100%* | *83%* | *97%* | *93%* | *70%* | – | – | *88.7%* | – |

Paired by cave (all 7 types): exit vs none 6 / 9 caves only-one-solved (p = 0.61); exit vs gru 8 / 8; exit vs graph
13 / 6 (p = 0.17). **No variant beats another significantly, and every v3 variant is below greedy and round 2.**

**Memory-wipe ablation** (memory erased at every decision):

| Success, normal → wiped | Maze | Loops | Boulder choke |
|---|---|---|---|
| exit | 47 → 43% | 67 → 63% | 47 → 37% |
| graph | 33 → 50% | 63 → 60% | 50 → 43% |
| gru | 37 → 47% | 70 → 70% | 50 → 53% |

Dead-end re-entries for exit go up when its memory is wiped (loops 0.30 → 0.63, boulder choke 1.63 → 1.97). Exit memory
is the only variant whose memory consistently helps itself, but only a little. Graph memory *hurts* in mazes.

**What the training logs showed** ([`results/train_memory_metrics.txt`](results/train_memory_metrics.txt)):
- **The store variants learned the memory facts.** Recall of "known dead end" was 83–91% and of "taken before"
  83–88%, against ~45% / ~50% for none and gru.
- **The policy hardly used those facts.** The memory gate stayed at 0.16–0.17 for exit (init 0.12) and 0.21–0.25 for
  graph.

**Why it failed (diagnosed, not yet fixed):**
1. **Weaker training overall.** The v3 memoryless navigator (73.3%) is far below round 2's (83.3%, SB3 PPO). Several
   things changed at once: the curriculum (decoys, loops, 30% blind dives), my custom PPO trainer, auxiliary losses on
   a network without memory, and the budget. A controlled run of `none` with round 2's exact setup on this trainer
   would separate them.
2. **The decoy scenario is solvable without memory.** Every variant scores 100%, because the layout is too regular:
   the route always runs along the far edge and the decoys always point at the goal. The policy learned "don't take
   the goal-pointing exit". Decoys must be randomised, so that the goal-pointing exit is sometimes the route.
3. **Two auxiliary labels are flawed:**
   - *"taken before"* is already true right after leaving a junction (still within 2 m). It should require ≥ 6 m of
     travel away, as loop closure does.
   - *"known dead end"* is credited to the latest open departure, which in rubble chambers often isn't the passage
     that led there. Even the oracle "avoids" it only 11% of the time in boulder chokes.

   The memory was trained to reproduce these labels, and the *avoid* / *untried* metrics rest on them, so those
   metrics are not reported as results. The graph variant reads the store through fixed rules and is hurt most.
4. **The memory gate initialised at 0.12 stayed nearly shut.** GTrXL-style gating suits deep residual stacks; here it
   just muted a small, informative signal.

**Proposed v3.1** (not started):
- fix both labels;
- randomise decoys;
- start the gate open (~0.5), or feed the memory facts straight into each exit's score;
- add the controlled trainer check.

Until then, **round 2's retention navigator on `main` remains the best model** (88.7%).

Comparison replays (the same caves as `main`'s `videos/08_round2`, plus the hardest decoy and loops caves):
`videos/09_memory_v3/` (see its README).

The replays make the problem visible. The old exit model's own belief head rates an exit "DEAD? 87%", and its
strongest recall points at the stored dead end, yet it takes that exit anyway.

## v3.1: a reward that prices forgetting (`configs/memnav_reward.yaml`)

**Why memory went unused.** Under the dense progress reward, memory had almost nothing to gain:
- *Detours are refunded.* The sub earns +1 per metre of shortest-path progress. Entering a dead end loses that
  reward and backing out pays it all back, so a wasted 8 m detour costs only its time and energy: about 5 reward,
  against about +57 for a success.
- *The map flags wrong turns.* Progress is computed from the true map, so the reward reports a wrong branch the
  moment it is entered; no memory is needed to learn that.
- *The goal bearing is usually there.* With it shown in 70% of dives, a memoryless policy collects nearly all the
  reward.

The belief heads learned the facts (83–91% recall), but acting on them paid too little to open the gate.

**Changes.** Stage 1 keeps the dense reward so the basics are learned fast. From stage 2 on:

| Change | Setting | Effect |
|---|---|---|
| Record progress | `progress: record` | paid only for beating the best distance to the goal so far: no "wrong branch" signal on entry, no refund on the way out |
| Repeat a dead end | `repeat_dead_penalty: 1.5` | choosing an exit that already led to a dead end (a small nudge on top of the detour's own cost) |
| Re-take an exit | `repeat_taken_penalty: 0.3` | only while an exit at the same junction is untried |
| Revisit a dead end | `dead_end_revisit_penalty: 1.5` | arriving again at the tip of a dead end already reached and left (≥ 4 m). Catches re-entry by any route, including turning back in its corridor, which has no junction for the exit penalty to fire at. The v3 models re-entered 0.55 dead ends per dive, the same as greedy (0.58); the oracle re-entered 0.24 |
| U-turn loop | `dither_penalty: 0.3` | turning back again within 2 decisions with no junction in between: the flip-flop out of a dead end (the goal bearing behind the sub pulls it back in). The forced turn at a dead end is free |
| Exploration bonus | `explore_bonus: 0.5` | per new 2 m cell, only in dives without the goal cue |
| Finishing | goal 20 → 40, `k_time` 0.01 → 0.02 | a timeout is clearly worse than finishing |
| Goal cue | 0.7 → 0.3 | memory, not the bearing, has to carry most dives |

**Backtracking to a far junction is never penalised.** The penalties apply only to *forward* exit choices at a
decision, using the simulator's true labels. Turning back and driving through explored tunnels cost only their
time and energy. With record progress, nothing is clawed back on the way. The exploration bonus then pays for
reaching new ground at the far end.

**Other fixes:**
- *Labels:* a departure counts as "taken" only once the sub has travelled 6 m and come back. A dead end is credited
  only to the departure that led straight to it, with no other junction taken in between.
- *Gate:* starts at 0.5 (`gate_init: 0`).
- *Demonstrations:* re-collected (`runs/memnav_r_demos.npz`), because the old ones carry the old labels.

Decoys are not yet randomised.

**Test.** A controlled pair under the same reward: `none` (the control) and `exit`, in `runs/memnav_r_*`. If the
reward was the bottleneck:
- `exit` should pull ahead of `none` on loops, mazes and breakdown caves;
- its gate should open;
- dead-end repeats should fall.

Tests: `tests/test_memnav.py`:
- record progress pays exactly start minus best;
- penalties match the true labels and never apply to turning back;
- the exploration bonus is off when the cue is shown;
- "taken" waits for a return.

**Replays** show the reward live:
- a fourth status line with the dive's reward, repeat penalties and exploration bonus;
- red `PENALTY` log lines saying why;
- a "+0.5 new area" flash for each bonus.

```
python -m scripts.compare_videos --config configs/memnav_reward.yaml --reward-stage m3_full [--no-cue] ...
```

## v3.1 results (22 Nov 2026): the reward made both models explore, not remember

 Same 7 cave types × 30 test caves as
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

## Why memory kept losing: the caves did not need it

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

## v3.2: memory gated by events (`configs/memnav_event.yaml`)

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

## v3.3: train on the memory caves (`configs/memnav_caves_clean.yaml`)

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

### v3.3 results (23 Nov 2026)

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
in [videos/10_memory_caves_clean/README.md](../videos/10_memory_caves_clean/README.md); the `.mp4` files (1.1 GB) are
git-ignored like all replays and are regenerated with the command below.

```powershell
python -m scripts.collect_nav_demos --config configs/memnav_caves_clean.yaml --workers 8 --out runs/memnav_c_demos.npz
python -m scripts.train_memnav --config configs/memnav_caves_clean.yaml --variant exit    # -> runs/memnav_cc_exit
python -m scripts.train_memnav --config configs/memnav_caves_clean.yaml --variant none    # -> runs/memnav_cc_none
python -m scripts.eval_navigator --config configs/memnav_reward.yaml --layouts rings hub --policies greedy oracle wall-left wall-right --models runs/memnav_cc_none runs/memnav_cc_exit
python -m scripts.compare_videos --config configs/memnav_caves_clean.yaml --reward-stage m3_full --no-cue --layouts rings hub --navigators runs/memnav_cc_exit runs/memnav_cc_none runs/memnav_e_exit runs/memnav_exit wall-left oracle --csv runs/eval/memcaves_cue_episodes.csv --out videos/10_memory_caves_clean
```

## Every version on the memory caves, and harder caves (23 Nov 2026)

**Round-2 navigators on rings / hub** (never evaluated there before; SB3 models, trained always with the goal cue;
same 30 + 30 test caves, d = 0.5, `configs/memnav_reward.yaml`; runs `r2_mem`, `r2_memblind`):

| Cue: rings / hub / both; blind: both | Cue | Blind | Re-entries (cue / blind) | Avoid (cue / blind) |
|---|---|---|---|---|
| Round 2 none | 17% / 13% / 15.0% | 50.0% | 3.10 / 1.52 | 37% / 66% |
| Round 2 retention | 37% / 10% / 23.3% | 51.7% | 2.62 / 1.40 | 38% / 60% |
| Round 2 retention + junction | 3% / 13% / 8.3% | 38.3% | 3.37 / 1.55 | 33% / 65% |
| v3.3 exit (for comparison) | 67% / 77% / 71.7% | 68.3% | 1.10 / 1.02 | 73% / 69% |

With the cue the round-2 navigators follow the bearing into the dead ends like greedy (re-entries 3.17); blind
they do much better but stay below v3.3. Paired at d = 0.5 (cue + blind): v3.3 exit vs round-2 retention **50 caves
to 11, p = 5e-7**.

**Harder memory caves** (d = 0.75 and 1.0; runs `hard_d*_{cue,blind}`; 0 collisions in all 1,440 dives):

| | d 0.75 cue | d 0.75 blind | d 1.0 cue | d 1.0 blind |
|---|---|---|---|---|
| v3.3 exit | **56.7%** (1.37) | 51.7% (1.38) | **58.3%** (1.42) | **50.0%** (1.50) |
| v3.3 none | 50.0% (1.55) | **55.0%** (1.62) | 48.3% (1.62) | 48.3% (1.78) |
| Round-2 retention | 20.0% (3.17) | 51.7% (1.70) | 20.0% (3.00) | 38.3% (1.95) |
| Wall-left | 35.0% | 35.0% | 40.0% | 40.0% |
| Greedy | 5.0% | 10.0% | 10.0% | 5.0% |
| Oracle | 100% | 100% | 100% | 100% |

(Success; dead-end re-entries per dive in brackets.)

- Exit memory keeps the fewest re-entries at every difficulty and beats round-2 retention clearly (77 to 25,
  p = 2e-7 over the harder runs).
- Its lead over the memoryless twin shrinks: 41 caves to 32 over the harder runs (p = 0.35, not significant);
  pooled over d = 0.5, 0.75 and 1.0: 63 to 39, p = 0.02. v3.3 trained on rings / hub up to d = 1.0 (stage m3), so
  this is not just unseen difficulty: in narrower passages driving and exit detection dominate, and memory is not
  yet the deciding factor.

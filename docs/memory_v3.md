# Memory v3: an exit memory the navigator has to use

Branch `memory-v3`. Status: **built.** Checked so far: the new cave layouts were generated and inspected,
the oracle teacher was evaluated, and all four model variants build and run forward. `tests/test_memnav.py` is
written but **not yet run**, and nothing is trained. The training plan is at the end.

## Why a third design

Round 1 (`main`, stage 9 phase B in [progress.md](../progress.md)) trained three navigators: none, retention,
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

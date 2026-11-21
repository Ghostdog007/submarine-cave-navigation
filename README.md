# submarine-cave-navigation

**Reinforcement learning for an autonomous submarine in underwater caves: sonar, vision, route memory and a built-in safety filter.**

A 2D simulator and training framework for an autonomous submarine that has to find its way through narrow,
winding underwater ice caves using only on-board sensing: a multi-beam **sonar** and a headlight-lit **camera**
that degrades in **murky water**. Agents are trained with PPO (Stable-Baselines3) and compared against classical
controllers and a map-aware oracle on held-out caves.

> **Status (Nov 2026): two-level architecture, training in progress.** The end-to-end agents (one network
> driving at 10 Hz *and* remembering junctions) reached their limits: memory cut dead-end re-entries ~11×, but
> crashes prevented any gain in success, and a safety filter bolted on after training made the agents stall.
> Their code is removed, their runs are in `runs/archive/` and their results in [progress.md](progress.md).
>
> | Phase | State |
> |---|---|
> | A0: hand-written driver + built-in safety filter + openings detector | ✅ done: **0 crashes in 150 test dives** (gap follower: 23–90%) |
> | A1: learned driver, imitation warm start | ✅ done |
> | A2–A3: learned driver, PPO curriculum (cave types, slow sonar, currents, sensor failures, hull damage) | 🔄 training |
> | B: navigators (no memory / retention / retention + junction memory) | built and tested, training next |
> | Evaluation, memory replays | next |

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
| `submarine/control/` | driver: controller + integrated safety filter (to build) |
| `submarine/navigator/` | openings detector + decision-level environment (to build) |
| `submarine/memory/` | retention + junction memory for the navigator (to build) |

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

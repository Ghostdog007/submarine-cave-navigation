# Context: 2D Submarine Cave Navigation (Computer Vision + RL)

Status: design document. No code written yet. All numbers are proposed defaults and can be changed.

## 1. Goal

Train an RL agent to pilot a small submarine through narrow, winding underwater passages (ice caverns and glacier cracks) using only on-board sensing. The 2D version is a fast prototype for testing the learning problem (partial observability, tight clearance, noisy sensing) before moving to 3D.

Research questions the 2D version should answer:
1. Can an RL policy navigate unseen caves using only range sensing?
2. How does performance change with passage width, wall roughness, sensor noise and current?
3. Does memory (recurrent policy) help versus frame-stacking?
4. How much does the goal cue help, and what happens without it?

## 2. World

- Top-down 2D map, 50 m x 30 m, stored as an occupancy grid with 0.2 m cells (250 x 150).
- The map border is solid rock.
- The submarine starts at the left end and must reach the exit at the right end.
- Every cave is generated from an integer seed (reproducible). Seeds are split into train and held-out test sets.

### Cave generation (procedural)

- Centerline: a smooth, winding curve y_c(x) made from a sum of random low-frequency sinusoids, with a slope limit.
- Width profile along x: base width plus
  - cracks: short tapered narrow sections (3 to 6 m long),
  - chambers: Gaussian bulges that widen the passage,
  - low-frequency width noise.
- Wall roughness: a smooth 2D noise field added to the free-space test, so walls are not straight.
- Pillars: random discs (radius 0.4 to 0.9 m) placed only in wide sections.
- Free space = cells where the (slope-corrected) distance to the centerline is less than half the local width, after noise. Everything else is rock.
- Solvability check: the vehicle-feasible free space (distance to the wall at least the vehicle radius) must connect start to goal. Otherwise regenerate with a new sub-seed.

### Difficulty knob d in [0, 1]

| Parameter | d = 0 (easy) | d = 1 (hard) |
|---|---|---|
| Base passage width | about 7 m | about 3 m |
| Number of cracks | 0 | up to 3 |
| Crack width | about 5 m | about 1.7 m (vehicle diameter 0.8 m) |
| Wall roughness amplitude | 0 | about 0.8 m |
| Centerline wander | about 2 m | about 8 m |
| Pillars | few | many |

Planned extension: a second layout family for glacier cracks (long, narrow, nearly vertical slots with occasional side openings).

## 3. Vehicle

State: position (x, y), heading psi, surge speed u, yaw rate r.

Dynamics (time step dt = 0.1 s):
- Surge: u += dt * (thrust_cmd * a_max - k_u * u), with speed capped at about 1.5 m/s. Reverse thrust is allowed but weaker.
- Yaw: first-order lag, r += dt * ((yaw_cmd * r_max - r) / tau_r), then psi += r * dt.
- World velocity = u * [cos psi, sin psi] + current.
- Optional constant water current per episode, magnitude up to about 0.3 m/s in a random direction (0 by default). It causes sideways drift the policy must compensate.
- Body is a disc of radius 0.4 m. A collision happens when clearance (distance to nearest rock, bilinear on the distance transform) is less than the radius.

This is intentionally simplified: inertia, drag and a lag in turning, but no full hydrodynamics.

## 4. Sensors

### 4.1 Multi-beam sonar (primary sensor)

- 32 beams, evenly spaced over 360 degrees in the body frame (FOV and beam count configurable, for example a 120 degree forward-looking sonar with 24 beams).
- Maximum range 10 m. Output per beam is the range to the first rock cell, or max range if nothing is hit.
- Implemented by ray marching on the occupancy grid at 0.1 m steps.
- Imperfections (configurable):
  - Gaussian range noise: about 0.05 m plus 2 percent of range.
  - Dropout: each beam independently returns max range with probability about 2 percent.
  - Optional minimum range blind zone, and optional beam-width smoothing (a beam returns the minimum over a small angular spread).
- Observation uses range / max_range, in [0, 1].

### 4.2 Proprioception

- Normalized surge speed u / u_max.
- Normalized yaw rate r / r_max.
- Previous action (thrust, yaw command).

### 4.3 Goal cue (switchable)

- Mode "bearing": cos and sin of the bearing to the goal in the body frame, plus normalized straight-line distance.
- Mode "none": no goal information. The agent must infer "go forward through the passage" from the sonar alone, which is a harder and more realistic setting.
- Note: the straight-line bearing can point into rock in winding caves, so the sonar must still be used.

### 4.4 Planned later: low-resolution camera model (for the CV study)

- A 1D or small 2D depth-like image with visibility that falls off with turbidity: range-dependent attenuation, backscatter noise and a limited light cone.
- Compare sonar only, camera only and fused inputs as turbidity and darkness increase.

## 5. Observation and action spaces

- Observation (vector, float32): 32 sonar values + 2 proprioception + 3 goal cue + 2 previous action = 39 dimensions.
- Action (continuous, in [-1, 1]^2): [thrust, yaw rate command].

## 6. Reward and termination

Per-step reward:

r = k_prog * (d_prev - d_now) - k_time - k_smooth * ||a - a_prev||^2 + terminal terms

- d is the geodesic (shortest-path-through-feasible-space) distance to the goal, precomputed per cave. This progress term is potential-based shaping: it rewards actual progress through the passage, not straight-line progress.
- Defaults: k_prog = 1.0 per meter, k_time = 0.01, k_smooth = 0.01.
- Goal reached: +20.
- Collision: -10, episode terminated.
- Optional proximity penalty near walls (default 0, since it conflicts with unavoidable tight cracks).
- Timeout: truncated at 600 steps (60 s).

Goal reached = within 1.5 m of the goal point.

## 7. Evaluation

Metrics, always reported on held-out caves:
- Success rate, collision rate, timeout rate.
- Path efficiency: path length divided by shortest-path length.
- Minimum and mean wall clearance.
- Time to goal.

Report each metric per difficulty level and per sensor or noise setting.

Baselines:
- Random policy (sanity floor).
- Gap-following controller: steer toward the most open sonar direction, slow down when the front is blocked.
- Wall-following or centering controller.
- Oracle planner with the true map (upper bound, not learnable from sensors).

RL algorithms: PPO and SAC. Compare an MLP with frame-stacking against a recurrent (GRU or LSTM) policy.

## 8. Experiments planned

1. Reward and observation sanity checks on easy caves (d = 0).
2. Curriculum from wide to narrow passages versus training on a fixed hard difficulty.
3. Goal cue on versus off.
4. Sonar noise, dropout and beam-count ablations.
5. Water current robustness.
6. Generalization: train on N caves, test on unseen caves and unseen difficulty.
7. Memory ablation (MLP + frame-stack versus GRU).

## 9. Known risks

- Sparse or misleading reward in narrow cracks: addressed by geodesic shaping and curriculum.
- Reward hacking (hovering safely): time penalty and progress-based reward.
- Overfitting to a small set of caves: use many procedural seeds and a held-out test set.
- Claims about real vehicles: the sim is simplified, so describe results as simulated navigation only.

## 10. Path to 3D

- Same cave concept extended to 3D (3D noise meshed with marching cubes) in PyBullet or MuJoCo, with a 6-DoF vehicle.
- Sonar becomes ray-cast multibeam or depth rendering; camera becomes rendered RGB with underwater degradation.
- Optional higher-fidelity simulators later: Stonefish or HoloOcean.

## 11. Open decisions

- Default FOV: full 360 degrees or a forward-looking sonar?
- Default goal cue: on or off?
- Should the first experiments include currents?
- Include the glacier-crack layout from the start, or add it after tunnels work?
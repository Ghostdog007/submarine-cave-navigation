# Replay videos by development stage

All replays use held-out test caves (seeds ≥ 1,000,000) with the same seeds as the evaluation tables, so what
you see matches the reported numbers. File names end in `_d<difficulty>_cave<seed>`.
Regenerate any of them with `python -m scripts.record` (see the README). The `.mp4` files are git-ignored.
This folder will be curated to one clip (two at most) per stage once the retention-memory results are in.

## 01_classical_baselines
- `gap_vs_oracle_d1.00_cave1000004.mp4`: original **plain** analysis view. On a hardest-level cave the
  follow-the-gap controller collides at 46.8 s; the oracle (knows the map) reaches the exit at 50.7 s.

## 02_ppo_trained_on_easy_caves
- `ppo_easy_only_on_easy_cave_d0.00_cave1000001.mp4`: the agent trained only on easy caves, on an easy cave: fast and clean (30.2 s).
- `gap_vs_ppo_easy_only_d0.75_cave1000004.mp4`: the same agent on a harder cave crashes at 23.6 s, while the
  gap follower gets through. It learned to chase the goal bearing, not to avoid obstacles.

## 03_ppo_mixed_difficulty
- `ppo_mixed_follow_cam_success_d1.00_cave1000004.mp4`: **showcase.** Follow camera, agent clears a hardest-level cave in 42.2 s.
- `gap_vs_ppo_mixed_d1.00_cave1000004.mp4`: same cave, gap follower collides (46.8 s), agent succeeds.
- `gap_vs_ppo_mixed_d0.75_*.mp4`: both succeed; the agent is 4–6 s faster.
- `ppo_mixed_follow_cam_failure_d1.00_cave1000002.mp4` and `gap_vs_ppo_mixed_d1.00_cave1000002.mp4`: a failure
  case, a collision in a tight crack at 14.3 s.

## 04_game_renderer
- `oracle_follow_cam_ice_theme_d1.00_cave1000002.mp4`: the game-style renderer: ice theme, follow camera,
  minimap, headlight, caustics, bubbles, sonar pings.

## 05_camera_and_murky_water
- `sonar_vs_camera_vs_fused_d1.00_cave1000009.mp4`: **key result of the sensor experiment.** Same hard, murky
  cave: the camera-only agent crashes in thick silt (19.6 s), the sonar-only agent misses the narrow exit, detours
  around a pillar and crashes (24.2 s), the sonar + camera agent reaches the exit (34.5 s). The detour is analysed
  in progress.md (stage 5).
- `fused_follow_cam_d1.00_cave1000009.mp4`: the winning sonar + camera run seen from the follow camera, with the
  live camera strip at the top.
- `gap_in_murky_water_d0.50_cave1000000.mp4`: what murk looks like: brown silt haze, drifting clouds, suspended
  particles, the headlight dying in a bright veil of silt, the camera's field of view and live camera strip (a
  flat grey veil inside murk, structure in clear water). Sonar pings every 0.4 s.

## 06_test_scenarios
Gap follower, best memoryless PPO (mixed difficulty) and oracle side by side on the new scenarios (d = 0.5, cave 1000000).

| File | Gap follower | PPO (no memory) | Oracle |
|---|---|---|---|
| `maze_…` | collision 16.2 s | collision 15.7 s | exit 44.0 s |
| `branches_…` | exit 33.8 s | exit 31.9 s | exit 37.6 s |
| `crack_…` | timeout 150 s | exit 32.5 s | exit 42.7 s |
| `breakdown_…` | exit 36.5 s | exit 32.6 s | exit 37.6 s |
| `siltout_d0.50_cave1000001.mp4` (sonar-only PPO) | exit 34.5 s | exit 31.8 s | exit 37.1 s |

- `maze_follow_cam_d0.50_cave1000000.mp4`: follow camera on the memoryless PPO in the maze: it crashes into a
  90° corner at 15.7 s, the weakness the step-2 training (mixed layouts + memory) addresses.

## 10_memory_caves_clean
v3.3 memory model vs its memoryless control, v3.2, v3, a wall follower and the oracle on the memory caves (rings,
hub; blind dives). Showcases: `rings_cave1000008_*` (only the v3.3 memory model and the oracle reach the exit) and
`hub_cave1000000_*` (the memory model gets out in 82 s; the control times out). Full table in
[10_memory_caves_clean/README.md](10_memory_caves_clean/README.md).

## Coming
- Stage 7: retention memory vs memoryless on maze, branches and the other scenarios (follow camera + side by side).
- Stage 8: sensor-failure robustness.

# Replay videos by development stage

All replays use held-out test caves (seeds ≥ 1,000,000), with the same seeds as the evaluation tables, so what you
see matches the reported numbers. The `.mp4` files are git-ignored. This folder was curated on 23 Nov 2026 to one or
two clips per stage (about 400 MB). The `README.md` in folders 08–10 still lists every video recorded at the time,
with its outcome; clips not listed below were deleted and can be re-recorded from the tagged commit with
`scripts.record` or `scripts.compare_videos`.

| Folder | Clip | What happens |
|---|---|---|
| 01_classical_baselines | `gap_vs_oracle_d1.00_cave1000004` | on a hardest-level cave the gap follower collides at 46.8 s; the oracle (knows the map) exits at 50.7 s |
| 02_ppo_trained_on_easy_caves | `gap_vs_ppo_easy_only_d0.75_cave1000004` | trained only on easy caves, PPO crashes at 23.6 s on a harder one, while the gap follower gets through: it learned to chase the goal bearing, not to avoid walls |
| 03_ppo_mixed_difficulty | `ppo_mixed_follow_cam_success_d1.00_cave1000004` | **showcase:** follow camera, PPO clears a hardest-level cave in 42.2 s |
| | `gap_vs_ppo_mixed_d1.00_cave1000004` | same cave: the gap follower collides (46.8 s), PPO succeeds |
| 04_game_renderer | `oracle_follow_cam_ice_theme_d1.00_cave1000002` | the game-style renderer: ice theme, follow camera, minimap, headlight, caustics, bubbles, sonar pings |
| 05_camera_and_murky_water | `sonar_vs_camera_vs_fused_d1.00_cave1000009` | **key sensor result:** in thick silt the camera-only agent crashes (19.6 s), sonar-only misses the narrow exit and crashes (24.2 s), sonar + camera reaches the exit (34.5 s) |
| 06_test_scenarios | `maze_d0.50_cave1000000` | gap follower and memoryless PPO both crash in the maze (16.2 s, 15.7 s); the oracle exits at 44.0 s |
| | `crack_d0.50_cave1000000` | glacier crack: the gap follower times out, PPO exits at 32.5 s |
| 08_round2 | `maze_cave1000000_nav_retention_v2`, `maze_cave1000000_nav_none_v2` | two-level navigators: retention memory is the only learned navigator to solve this maze (143.8 s); no memory times out |
| 09_memory_v3 | `decoy_cave1000008_memnav_exit`, `decoy_cave1000008_greedy` | a decoy passage points straight at the exit: greedy is trapped in it, the v3 exit-memory navigator exits at 57.7 s. Overlays show stored junctions, dead ends, recalls and the belief heads |
| 10_memory_caves_clean | `rings_cave1000008_memnav_cc_exit_blind`, `rings_cave1000008_memnav_cc_none_blind` | ⭐ **v3.3:** blind, the memory model escapes the rings in 61.1 s (as fast as the oracle); its memoryless twin times out |
| | `hub_cave1000000_memnav_cc_exit_blind`, `hub_cave1000000_memnav_cc_none_blind` | ⭐ **v3.3:** the memory model leaves the hub in 82.2 s; its memoryless twin times out |

Stage 07 (round-1 navigators) was removed entirely: round 2 replays the same maze cave with the fixed navigators.

## 11_versions_hub and post/

- `11_versions_hub/`: hub cave 1000000 (blind) with round-2 retention (times out), v3 exit memory (times out: it
  enters a passage "despite dead-end belief 89%") and v3.2 (exits at 83.6 s); v3.3 is in `10_memory_caves_clean`.
- `post/`: post-ready MP4s made with `scripts.make_preview` (H.264, label bars):
  `1_memory_vs_no_memory_rings.mp4` (v3.3 vs its memoryless twin, 17.5 s), `2_versions_hub.mp4` (round 2 → v3 →
  v3.2 → v3.3 in a 2×2 grid, 22.5 s), `3_what_it_thinks_hub.mp4` (v3.3 alone, 2× real time, 14 s).

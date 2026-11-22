# Comparison replays

Every navigator on the same test cave per cave type (same seed, difficulty d = 0.5, sensor noise and currents). Caves chosen from the evaluation as the hardest one the oracle solves. Memory displays are checked against the agent's stored state (unexplained must be 0).

Markers (memory-v3 navigators): the badge names the architecture. Store variants show stored junction departures (ring #n) and dead ends (red X #n), lines to the entries recalled for the current choice and a yellow pulse on the strongest. Every v3 variant shows its belief heads: DEAD? n% on an exit it believes leads to a known dead end, BEEN n% on one it believes it already took, an orange ring while it believes it is revisiting a place; the log lists AVOIDED / ENTERED-despite decisions. With --reward-stage, a fourth status line shows the dive's reward, repeat PENALTY events and the exploration bonus (+n new area).

| Cave type | Cave | Why this cave | Navigator | Outcome | Time (s) | Decisions | Memories stored | Display check | File |
|---|---|---|---|---|---|---|---|---|---|
| decoy | 1000008 | 4 of 5 evaluated navigators succeeded; 5 dead-end entries in total | greedy | timeout | 150.0 | 80 | 0 | ok | `decoy_cave1000008_greedy.mp4` |
| decoy | 1000008 | 4 of 5 evaluated navigators succeeded; 5 dead-end entries in total | oracle | success | 57.4 | 34 | 0 | ok | `decoy_cave1000008_oracle.mp4` |
| decoy | 1000008 | 4 of 5 evaluated navigators succeeded; 5 dead-end entries in total | memnav_none | success | 57.7 | 34 | 0 | ok | `decoy_cave1000008_memnav_none.mp4` |
| decoy | 1000008 | 4 of 5 evaluated navigators succeeded; 5 dead-end entries in total | memnav_gru | success | 57.7 | 34 | 0 | ok | `decoy_cave1000008_memnav_gru.mp4` |
| decoy | 1000008 | 4 of 5 evaluated navigators succeeded; 5 dead-end entries in total | memnav_graph | success | 57.7 | 34 | 10 | ok | `decoy_cave1000008_memnav_graph.mp4` |
| decoy | 1000008 | 4 of 5 evaluated navigators succeeded; 5 dead-end entries in total | memnav_exit | success | 57.7 | 34 | 10 | ok | `decoy_cave1000008_memnav_exit.mp4` |
| loops | 1000023 | 0 of 5 evaluated navigators succeeded; 14 dead-end entries in total | greedy | timeout | 150.0 | 92 | 0 | ok | `loops_cave1000023_greedy.mp4` |
| loops | 1000023 | 0 of 5 evaluated navigators succeeded; 14 dead-end entries in total | oracle | success | 63.0 | 41 | 0 | ok | `loops_cave1000023_oracle.mp4` |
| loops | 1000023 | 0 of 5 evaluated navigators succeeded; 14 dead-end entries in total | memnav_none | timeout | 150.0 | 94 | 0 | ok | `loops_cave1000023_memnav_none.mp4` |
| loops | 1000023 | 0 of 5 evaluated navigators succeeded; 14 dead-end entries in total | memnav_gru | timeout | 150.0 | 90 | 0 | ok | `loops_cave1000023_memnav_gru.mp4` |
| loops | 1000023 | 0 of 5 evaluated navigators succeeded; 14 dead-end entries in total | memnav_graph | timeout | 150.0 | 92 | 15 | ok | `loops_cave1000023_memnav_graph.mp4` |
| loops | 1000023 | 0 of 5 evaluated navigators succeeded; 14 dead-end entries in total | memnav_exit | timeout | 150.0 | 91 | 19 | ok | `loops_cave1000023_memnav_exit.mp4` |
| tunnel (extra) | 1000009 | requested | greedy | success | 67.5 | 45 | 0 | ok | `tunnel_cave1000009_greedy.mp4` |
| tunnel (extra) | 1000009 | requested | oracle | success | 58.3 | 34 | 0 | ok | `tunnel_cave1000009_oracle.mp4` |
| tunnel (extra) | 1000009 | requested | memnav_none | success | 56.9 | 35 | 0 | ok | `tunnel_cave1000009_memnav_none.mp4` |
| tunnel (extra) | 1000009 | requested | memnav_gru | success | 49.8 | 30 | 0 | ok | `tunnel_cave1000009_memnav_gru.mp4` |
| tunnel (extra) | 1000009 | requested | memnav_graph | success | 49.8 | 30 | 14 | ok | `tunnel_cave1000009_memnav_graph.mp4` |
| tunnel (extra) | 1000009 | requested | memnav_exit | success | 67.5 | 45 | 18 | ok | `tunnel_cave1000009_memnav_exit.mp4` |
| maze (extra) | 1000029 | requested | greedy | timeout | 150.0 | 88 | 0 | ok | `maze_cave1000029_greedy.mp4` |
| maze (extra) | 1000029 | requested | oracle | success | 63.8 | 38 | 0 | ok | `maze_cave1000029_oracle.mp4` |
| maze (extra) | 1000029 | requested | memnav_none | timeout | 150.0 | 83 | 0 | ok | `maze_cave1000029_memnav_none.mp4` |
| maze (extra) | 1000029 | requested | memnav_gru | timeout | 150.0 | 75 | 0 | ok | `maze_cave1000029_memnav_gru.mp4` |
| maze (extra) | 1000029 | requested | memnav_graph | timeout | 150.0 | 71 | 8 | ok | `maze_cave1000029_memnav_graph.mp4` |
| maze (extra) | 1000029 | requested | memnav_exit | timeout | 150.0 | 81 | 17 | ok | `maze_cave1000029_memnav_exit.mp4` |
| branches (extra) | 1000028 | requested | greedy | timeout | 150.0 | 85 | 0 | ok | `branches_cave1000028_greedy.mp4` |
| branches (extra) | 1000028 | requested | oracle | success | 46.9 | 27 | 0 | ok | `branches_cave1000028_oracle.mp4` |
| branches (extra) | 1000028 | requested | memnav_none | success | 46.9 | 27 | 0 | ok | `branches_cave1000028_memnav_none.mp4` |
| branches (extra) | 1000028 | requested | memnav_gru | success | 46.9 | 27 | 0 | ok | `branches_cave1000028_memnav_gru.mp4` |
| branches (extra) | 1000028 | requested | memnav_graph | success | 46.9 | 27 | 9 | ok | `branches_cave1000028_memnav_graph.mp4` |
| branches (extra) | 1000028 | requested | memnav_exit | success | 46.9 | 27 | 9 | ok | `branches_cave1000028_memnav_exit.mp4` |
| crack (extra) | 1000014 | requested | greedy | timeout | 150.0 | 73 | 0 | ok | `crack_cave1000014_greedy.mp4` |
| crack (extra) | 1000014 | requested | oracle | success | 87.5 | 41 | 0 | ok | `crack_cave1000014_oracle.mp4` |
| crack (extra) | 1000014 | requested | memnav_none | success | 135.4 | 61 | 0 | ok | `crack_cave1000014_memnav_none.mp4` |
| crack (extra) | 1000014 | requested | memnav_gru | success | 136.8 | 62 | 0 | ok | `crack_cave1000014_memnav_gru.mp4` |
| crack (extra) | 1000014 | requested | memnav_graph | timeout | 150.0 | 69 | 5 | ok | `crack_cave1000014_memnav_graph.mp4` |
| crack (extra) | 1000014 | requested | memnav_exit | timeout | 150.0 | 69 | 6 | ok | `crack_cave1000014_memnav_exit.mp4` |
| breakdown (extra) | 1000027 | requested | greedy | timeout | 150.0 | 111 | 0 | ok | `breakdown_cave1000027_greedy.mp4` |
| breakdown (extra) | 1000027 | requested | oracle | success | 66.1 | 41 | 0 | ok | `breakdown_cave1000027_oracle.mp4` |
| breakdown (extra) | 1000027 | requested | memnav_none | timeout | 150.0 | 106 | 0 | ok | `breakdown_cave1000027_memnav_none.mp4` |
| breakdown (extra) | 1000027 | requested | memnav_gru | timeout | 150.0 | 100 | 0 | ok | `breakdown_cave1000027_memnav_gru.mp4` |
| breakdown (extra) | 1000027 | requested | memnav_graph | success | 61.3 | 42 | 19 | ok | `breakdown_cave1000027_memnav_graph.mp4` |
| breakdown (extra) | 1000027 | requested | memnav_exit | timeout | 150.0 | 106 | 37 | ok | `breakdown_cave1000027_memnav_exit.mp4` |
| maze (extra 1000000) | 1000000 | requested | greedy | success | 77.6 | 47 | 0 | ok | `maze_cave1000000_greedy.mp4` |
| maze (extra 1000000) | 1000000 | requested | oracle | success | 59.0 | 37 | 0 | ok | `maze_cave1000000_oracle.mp4` |
| maze (extra 1000000) | 1000000 | requested | memnav_none | timeout | 150.0 | 73 | 0 | ok | `maze_cave1000000_memnav_none.mp4` |
| maze (extra 1000000) | 1000000 | requested | memnav_gru | timeout | 150.0 | 69 | 0 | ok | `maze_cave1000000_memnav_gru.mp4` |
| maze (extra 1000000) | 1000000 | requested | memnav_graph | timeout | 150.0 | 70 | 5 | ok | `maze_cave1000000_memnav_graph.mp4` |
| maze (extra 1000000) | 1000000 | requested | memnav_exit | timeout | 150.0 | 73 | 9 | ok | `maze_cave1000000_memnav_exit.mp4` |

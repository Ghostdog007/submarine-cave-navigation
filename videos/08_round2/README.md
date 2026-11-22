# Comparison replays

Every navigator on the same test cave per cave type (same seed, difficulty d = 0.5, sensor noise and currents). Caves chosen from the evaluation as the hardest one the oracle solves. Memory displays are checked against the agent's stored state (unexplained must be 0).

| Cave type | Cave | Why this cave | Navigator | Outcome | Time (s) | Decisions | Memories stored | Display check | File |
|---|---|---|---|---|---|---|---|---|---|
| tunnel | 1000009 | 5 of 5 evaluated navigators succeeded; 17 dead-end entries in total | greedy | success | 67.5 | 45 | 0 | ok | `tunnel_cave1000009_greedy.mp4` |
| tunnel | 1000009 | 5 of 5 evaluated navigators succeeded; 17 dead-end entries in total | oracle | success | 58.3 | 34 | 0 | ok | `tunnel_cave1000009_oracle.mp4` |
| tunnel | 1000009 | 5 of 5 evaluated navigators succeeded; 17 dead-end entries in total | nav_none_v2 | success | 56.9 | 35 | 0 | ok | `tunnel_cave1000009_nav_none_v2.mp4` |
| tunnel | 1000009 | 5 of 5 evaluated navigators succeeded; 17 dead-end entries in total | nav_retention_v2 | success | 67.1 | 43 | 0 | ok | `tunnel_cave1000009_nav_retention_v2.mp4` |
| tunnel | 1000009 | 5 of 5 evaluated navigators succeeded; 17 dead-end entries in total | nav_junction_v2 | success | 51.4 | 32 | 14 | ok | `tunnel_cave1000009_nav_junction_v2.mp4` |
| maze | 1000029 | 0 of 5 evaluated navigators succeeded; 19 dead-end entries in total | greedy | timeout | 150.0 | 88 | 0 | ok | `maze_cave1000029_greedy.mp4` |
| maze | 1000029 | 0 of 5 evaluated navigators succeeded; 19 dead-end entries in total | oracle | success | 63.8 | 38 | 0 | ok | `maze_cave1000029_oracle.mp4` |
| maze | 1000029 | 0 of 5 evaluated navigators succeeded; 19 dead-end entries in total | nav_none_v2 | timeout | 150.0 | 85 | 0 | ok | `maze_cave1000029_nav_none_v2.mp4` |
| maze | 1000029 | 0 of 5 evaluated navigators succeeded; 19 dead-end entries in total | nav_retention_v2 | timeout | 150.0 | 82 | 0 | ok | `maze_cave1000029_nav_retention_v2.mp4` |
| maze | 1000029 | 0 of 5 evaluated navigators succeeded; 19 dead-end entries in total | nav_junction_v2 | timeout | 150.0 | 81 | 11 | ok | `maze_cave1000029_nav_junction_v2.mp4` |
| branches | 1000028 | 0 of 5 evaluated navigators succeeded; 31 dead-end entries in total | greedy | timeout | 150.0 | 85 | 0 | ok | `branches_cave1000028_greedy.mp4` |
| branches | 1000028 | 0 of 5 evaluated navigators succeeded; 31 dead-end entries in total | oracle | success | 46.9 | 27 | 0 | ok | `branches_cave1000028_oracle.mp4` |
| branches | 1000028 | 0 of 5 evaluated navigators succeeded; 31 dead-end entries in total | nav_none_v2 | timeout | 150.0 | 85 | 0 | ok | `branches_cave1000028_nav_none_v2.mp4` |
| branches | 1000028 | 0 of 5 evaluated navigators succeeded; 31 dead-end entries in total | nav_retention_v2 | timeout | 150.0 | 88 | 0 | ok | `branches_cave1000028_nav_retention_v2.mp4` |
| branches | 1000028 | 0 of 5 evaluated navigators succeeded; 31 dead-end entries in total | nav_junction_v2 | timeout | 150.0 | 78 | 9 | ok | `branches_cave1000028_nav_junction_v2.mp4` |
| crack | 1000014 | 0 of 5 evaluated navigators succeeded; 0 dead-end entries in total | greedy | timeout | 150.0 | 73 | 0 | ok | `crack_cave1000014_greedy.mp4` |
| crack | 1000014 | 0 of 5 evaluated navigators succeeded; 0 dead-end entries in total | oracle | success | 87.5 | 41 | 0 | ok | `crack_cave1000014_oracle.mp4` |
| crack | 1000014 | 0 of 5 evaluated navigators succeeded; 0 dead-end entries in total | nav_none_v2 | timeout | 150.0 | 66 | 0 | ok | `crack_cave1000014_nav_none_v2.mp4` |
| crack | 1000014 | 0 of 5 evaluated navigators succeeded; 0 dead-end entries in total | nav_retention_v2 | timeout | 150.0 | 66 | 0 | ok | `crack_cave1000014_nav_retention_v2.mp4` |
| crack | 1000014 | 0 of 5 evaluated navigators succeeded; 0 dead-end entries in total | nav_junction_v2 | timeout | 150.0 | 68 | 5 | ok | `crack_cave1000014_nav_junction_v2.mp4` |
| breakdown | 1000027 | 0 of 5 evaluated navigators succeeded; 30 dead-end entries in total | greedy | timeout | 150.0 | 111 | 0 | ok | `breakdown_cave1000027_greedy.mp4` |
| breakdown | 1000027 | 0 of 5 evaluated navigators succeeded; 30 dead-end entries in total | oracle | success | 66.1 | 41 | 0 | ok | `breakdown_cave1000027_oracle.mp4` |
| breakdown | 1000027 | 0 of 5 evaluated navigators succeeded; 30 dead-end entries in total | nav_none_v2 | timeout | 150.0 | 101 | 0 | ok | `breakdown_cave1000027_nav_none_v2.mp4` |
| breakdown | 1000027 | 0 of 5 evaluated navigators succeeded; 30 dead-end entries in total | nav_retention_v2 | timeout | 150.0 | 100 | 0 | ok | `breakdown_cave1000027_nav_retention_v2.mp4` |
| breakdown | 1000027 | 0 of 5 evaluated navigators succeeded; 30 dead-end entries in total | nav_junction_v2 | timeout | 150.0 | 88 | 12 | ok | `breakdown_cave1000027_nav_junction_v2.mp4` |
| maze (extra) | 1000000 | requested | greedy | success | 77.6 | 47 | 0 | ok | `maze_cave1000000_greedy.mp4` |
| maze (extra) | 1000000 | requested | oracle | success | 59.0 | 37 | 0 | ok | `maze_cave1000000_oracle.mp4` |
| maze (extra) | 1000000 | requested | nav_none_v2 | timeout | 150.0 | 76 | 0 | ok | `maze_cave1000000_nav_none_v2.mp4` |
| maze (extra) | 1000000 | requested | nav_retention_v2 | success | 143.8 | 78 | 0 | ok | `maze_cave1000000_nav_retention_v2.mp4` |
| maze (extra) | 1000000 | requested | nav_junction_v2 | timeout | 150.0 | 83 | 11 | ok | `maze_cave1000000_nav_junction_v2.mp4` |

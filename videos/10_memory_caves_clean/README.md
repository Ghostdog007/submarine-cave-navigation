# Comparison replays

Every navigator on the same test cave per cave type (same seed, difficulty d = 0.5, sensor noise and currents). Caves chosen from the evaluation as the hardest one the oracle solves. Memory displays are checked against the agent's stored state (unexplained must be 0).

Markers (memory-v3 navigators): the badge names the architecture. Store variants show stored junction departures (ring #n) and dead ends (red X #n), lines to the entries recalled for the current choice and a yellow pulse on the strongest. Every v3 variant shows its belief heads: DEAD? n% on an exit it believes leads to a known dead end, BEEN n% on one it believes it already took, an orange ring while it believes it is revisiting a place; the log lists AVOIDED / ENTERED-despite decisions. With --reward-stage, a fourth status line shows the dive's reward, repeat PENALTY events and the exploration bonus (+n new area).

| Cave type | Cave | Why this cave | Navigator | Outcome | Time (s) | Decisions | Memories stored | Display check | File |
|---|---|---|---|---|---|---|---|---|---|
| rings | 1000007 | 0 of 11 evaluated navigators succeeded; 54 dead-end entries in total | memnav_cc_exit | timeout | 150.0 | 80 | 7 | ok | `rings_cave1000007_memnav_cc_exit_blind.mp4` |
| rings | 1000007 | 0 of 11 evaluated navigators succeeded; 54 dead-end entries in total | memnav_cc_none | timeout | 150.0 | 76 | 0 | ok | `rings_cave1000007_memnav_cc_none_blind.mp4` |
| rings | 1000007 | 0 of 11 evaluated navigators succeeded; 54 dead-end entries in total | memnav_e_exit | timeout | 150.0 | 80 | 7 | ok | `rings_cave1000007_memnav_e_exit_blind.mp4` |
| rings | 1000007 | 0 of 11 evaluated navigators succeeded; 54 dead-end entries in total | memnav_exit | timeout | 150.0 | 80 | 7 | ok | `rings_cave1000007_memnav_exit_blind.mp4` |
| rings | 1000007 | 0 of 11 evaluated navigators succeeded; 54 dead-end entries in total | wall-left | timeout | 150.0 | 82 | 0 | ok | `rings_cave1000007_wall-left_blind.mp4` |
| rings | 1000007 | 0 of 11 evaluated navigators succeeded; 54 dead-end entries in total | oracle | success | 56.7 | 30 | 0 | ok | `rings_cave1000007_oracle_blind.mp4` |
| hub | 1000013 | 1 of 11 evaluated navigators succeeded; 58 dead-end entries in total | memnav_cc_exit | timeout | 150.0 | 81 | 16 | ok | `hub_cave1000013_memnav_cc_exit_blind.mp4` |
| hub | 1000013 | 1 of 11 evaluated navigators succeeded; 58 dead-end entries in total | memnav_cc_none | timeout | 150.0 | 79 | 0 | ok | `hub_cave1000013_memnav_cc_none_blind.mp4` |
| hub | 1000013 | 1 of 11 evaluated navigators succeeded; 58 dead-end entries in total | memnav_e_exit | timeout | 150.0 | 82 | 15 | ok | `hub_cave1000013_memnav_e_exit_blind.mp4` |
| hub | 1000013 | 1 of 11 evaluated navigators succeeded; 58 dead-end entries in total | memnav_exit | timeout | 150.0 | 79 | 13 | ok | `hub_cave1000013_memnav_exit_blind.mp4` |
| hub | 1000013 | 1 of 11 evaluated navigators succeeded; 58 dead-end entries in total | wall-left | success | 57.5 | 31 | 0 | ok | `hub_cave1000013_wall-left_blind.mp4` |
| hub | 1000013 | 1 of 11 evaluated navigators succeeded; 58 dead-end entries in total | oracle | success | 56.3 | 29 | 0 | ok | `hub_cave1000013_oracle_blind.mp4` |
| rings (extra) | 1000008 | requested | memnav_cc_exit | success | 61.1 | 33 | 7 | ok | `rings_cave1000008_memnav_cc_exit_blind.mp4` |
| rings (extra) | 1000008 | requested | memnav_cc_none | timeout | 150.0 | 74 | 0 | ok | `rings_cave1000008_memnav_cc_none_blind.mp4` |
| rings (extra) | 1000008 | requested | memnav_e_exit | timeout | 150.0 | 75 | 14 | ok | `rings_cave1000008_memnav_e_exit_blind.mp4` |
| rings (extra) | 1000008 | requested | memnav_exit | timeout | 150.0 | 75 | 14 | ok | `rings_cave1000008_memnav_exit_blind.mp4` |
| rings (extra) | 1000008 | requested | wall-left | timeout | 150.0 | 82 | 0 | ok | `rings_cave1000008_wall-left_blind.mp4` |
| rings (extra) | 1000008 | requested | oracle | success | 61.1 | 33 | 0 | ok | `rings_cave1000008_oracle_blind.mp4` |
| hub (extra) | 1000000 | requested | memnav_cc_exit | success | 82.2 | 42 | 10 | ok | `hub_cave1000000_memnav_cc_exit_blind.mp4` |
| hub (extra) | 1000000 | requested | memnav_cc_none | timeout | 150.0 | 78 | 0 | ok | `hub_cave1000000_memnav_cc_none_blind.mp4` |
| hub (extra) | 1000000 | requested | memnav_e_exit | success | 83.6 | 46 | 10 | ok | `hub_cave1000000_memnav_e_exit_blind.mp4` |
| hub (extra) | 1000000 | requested | memnav_exit | timeout | 150.0 | 74 | 8 | ok | `hub_cave1000000_memnav_exit_blind.mp4` |
| hub (extra) | 1000000 | requested | wall-left | timeout | 150.0 | 85 | 0 | ok | `hub_cave1000000_wall-left_blind.mp4` |
| hub (extra) | 1000000 | requested | oracle | success | 53.5 | 30 | 0 | ok | `hub_cave1000000_oracle_blind.mp4` |

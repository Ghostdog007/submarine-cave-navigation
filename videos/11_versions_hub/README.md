# Comparison replays

Every navigator on the same test cave per cave type (same seed, difficulty d = 0.5, sensor noise and currents). Caves chosen from the evaluation as the hardest one the oracle solves. Memory displays are checked against the agent's stored state (unexplained must be 0).

Markers (memory-v3 navigators): the badge names the architecture. Store variants show stored junction departures (ring #n) and dead ends (red X #n), lines to the entries recalled for the current choice and a yellow pulse on the strongest. Every v3 variant shows its belief heads: DEAD? n% on an exit it believes leads to a known dead end, BEEN n% on one it believes it already took, an orange ring while it believes it is revisiting a place; the log lists AVOIDED / ENTERED-despite decisions. With --reward-stage, a fourth status line shows the dive's reward, repeat PENALTY events and the exploration bonus (+n new area).

| Cave type | Cave | Why this cave | Navigator | Outcome | Time (s) | Decisions | Memories stored | Display check | File |
|---|---|---|---|---|---|---|---|---|---|
| hub (extra) | 1000000 | requested | nav_retention_v2 | timeout | 150.0 | 81 | 0 | ok | `hub_cave1000000_nav_retention_v2_blind.mp4` |
| hub (extra) | 1000000 | requested | memnav_exit | timeout | 150.0 | 74 | 8 | ok | `hub_cave1000000_memnav_exit_blind.mp4` |
| hub (extra) | 1000000 | requested | memnav_e_exit | success | 83.6 | 46 | 10 | ok | `hub_cave1000000_memnav_e_exit_blind.mp4` |

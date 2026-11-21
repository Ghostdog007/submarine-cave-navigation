# Scenarios and capabilities, stage by stage

How the environment, the agent's components, its training and the test scenarios changed over the project,
what each stage was meant to make possible, and what was actually measured. Everything is anchored on one
running example, **"the dive"**, so the progression is concrete. Detailed numbers are in
[progress.md](progress.md); videos in [videos/](videos).

Legend: **measured** = shown on held-out test caves; *hoped / planned* = not yet demonstrated.

---

## The running example: one dive, six challenges

An autonomous submarine has to get from the entrance of a glacier cave to an exit ~50 m away. It has no map,
only its own sensors.

| Segment | What happens | Real-world counterpart | Capability needed |
|---|---|---|---|
| **A. Entrance tunnel** | a winding passage with rough walls and a pillar | everyday cave passage | basic obstacle avoidance |
| **B. Ice crevasse** | angular zigzag with a 1.4 m squeeze (vehicle is 0.8 m wide), side fissures too narrow to enter | glacier fissure, "restriction" | precise low-speed control |
| **C. Silted chamber** | a wide chamber with a pillar; the exit is a narrow crack; thick silt drifts across it | silt-out, the classic cause of disorientation | find the exit; use sonar when the camera is blind |
| **D. Junction with a dead end** | two passages; the one pointing toward the exit is a 15 m dead end that curves out of sonar range | blind lead that "looks right" | explore, recognise the dead end, **remember it**, take the other way, don't go back to the start |
| **E. Camera failure** | the camera stops working for 10 s in the middle of a passage | equipment failure | continue on sonar |
| **F. Total blackout** | both sensors drop out for 5 s in a narrow passage with a weak current | power or bus fault | carry on from memory or stop safely, then resume when the sensors return |

The table further down records, for each stage's agent, how far it gets through this dive.

---

## Stage 1: the world and classical controllers

**What changed**
- *Environment:* procedural caves (one winding tunnel per seed) with a difficulty knob (width, cracks down to 1.7 m, roughness, pillars).
- *Vehicle:* thrust with drag, lagged turning, optional current.
- *Sensors:* 32-beam 360° sonar with noise and dropouts.
- *Controllers:* random, follow-the-gap, wall-centering, and an oracle that knows the map.

**Hoped:** a fair benchmark: solvable caves, strong classical baselines, a perfect upper bound.

**Measured:** oracle 100% (every cave solvable); gap follower 80% overall but **34% on the hardest caves**.

**In the dive:** the gap follower handles **A**, often fails **B** (squeezes and kinks), has no camera for **C**, and has no memory for **D**.

## Stage 2: first learning agent, trained on easy caves only

**What changed:** PPO with a small MLP, trained on wide, almost straight caves.

**Hoped:** learn to drive from sonar.

**Measured:** 94% on easy caves but **31% overall**. It learned to chase the goal bearing at full speed and never learned to avoid obstacles.

**Lesson:** an agent only learns what its training caves force it to learn.

**In the dive:** crashes in **A** as soon as the passage bends sharply.

## Stage 3: training on the full range of difficulty

**What changed:** each training episode draws a random difficulty; a decaying learning rate; checkpoints chosen on a separate validation set.

**Hoped:** real obstacle avoidance across all difficulties.

**Measured:** **94% overall, 82% on the hardest tunnels** (gap follower 34%), and faster than the oracle.

**In the dive:** **A** reliably, **B** often; still no camera and no memory.

## Stage 4: seeing it

**What changed:** replay videos and a game-style renderer (follow camera, minimap, murk, headlight, live camera strip). Rendering only; the simulation is unchanged.

**Why it matters:** the failure analyses below came from watching replays next to the numbers.

## Stage 5: vision and murky water

**What changed**
- *Environment:* random murky patches (turbidity up to 1.0).
- *Sonar:* now realistic: a new scan only every 0.4 s, held in between.
- *Camera:* new forward camera, 90°, 64 pixels, lit by the headlight; murk dims and veils it.
- *Model:* a small 1D CNN on the camera, joined with the sonar/state network.

**Hoped:** sharper perception from the camera in clear water, with a fall-back to sonar in murk.

**Measured:**
- Sonar + camera **93%** (86% at d = 1) vs sonar only 88% vs camera only 84%.
- The camera-only agent's crashes cluster inside murk (75% of its collisions, 22% of its time).
- **A worked example of segment C** (test cave 1000009): the sonar-only agent couldn't single out the narrow exit among many open directions, followed the goal bearing into a closed pocket, circled the pillar and crashed. The sonar + camera agent saw the exit as a clear gap and got out.
- **But:** with its camera completely dead, the sonar + camera agent drops to **0%**. It uses the camera whenever possible and **never learned a sonar fall-back**.

**In the dive:** **A–C** (often), fails **E** and **F**.

## Stage 6: test scenarios from real cave exploration, and energy

**What changed**
- *New cave layouts:* maze (junctions and dead ends), misleading side branches, glacier crevasse with squeezes, boulder choke, and a silt-out scenario.
- *Energy:* a power model (thrust, turning, always-on electronics) and an optional energy penalty in the reward.

**Hoped:** expose where the agents break.

**Measured** (best memoryless agent):
- tunnel 95%, crevasse 70%, boulder choke 60%;
- **maze 5%**, and 95% of those failures were collisions at 90° corners, not getting lost.

**Lesson:** before memory can matter, the agent has to be able to *drive* these caves.

## Stage 7: retention memory (decaying attention, no LSTM)

**What changed:** a memory of the past: four attention "heads" whose memories fade with half-lives of 0.3 s, 1.4 s, 7 s and 70 s. Trained on a mix of all layouts with the energy penalty.

**Hoped:** remember recent passages and avoid going back into dead ends (segment D).

**Measured** (checkpoint at 2.6M steps):
- In mazes it **re-entered explored dead ends about 8× less** than the memoryless agent (0.07 vs 0.57 per run) and backtracked half as much: the first sign that memory helps.
- **But both agents were under-trained at basic driving** (crashing 40–90% of the time, even in plain tunnels), because five cave types shared 3M steps.

**Limit identified:** retention remembers *what happened*, not *where it happened*.

## Stage 7b: junction memory, position sense and a curriculum *(training in progress)*

**What changed**
- *Position sense:* **path integration**: the agent estimates its own position from its speed and turn rate, with realistic drift (2–6% of distance), without any external positioning.
- *Event detector:* recognises **junctions** (two or more openings) and **dead ends** from sonar.
- *Junction memory:* 16 lasting memory slots, written only at those events, each holding a summary of the moment, the estimated position and the event type. They're read by attention that combines "looks similar" with "is nearby".
- *Training:* a **curriculum**: learn to drive first (tunnel, crevasse, boulder choke), then add branches, then mazes; the energy penalty switches on only once it can drive. 6M steps.
- *Comparison:* memoryless vs retention vs retention + junction memory, all with the position sense.

**Hoped:** exactly segment **D**, the behaviour this project set out to get:

> The submarine reaches a junction with two openings and stores *"junction at about (18, 4) m, I left heading 40°"*. It follows that passage, finds it closes after 15 m, and stores *"dead end at about (27, 12) m"*. It turns back. Approaching the junction again, its estimated position and the similar sonar picture recall both memories: "I've been here, I went that way, it was a dead end". It takes the **other** opening, and it doesn't head back toward the entrance, which is also a remembered place.

**How we'll know:** dead-end re-entries and backtracking from trajectories (should approach the oracle's zero), maze and branch success rates, and replays with the recalled memories drawn live.

## Stage 8: partial observability and sensor failures *(planned)*

**What will change**
- *Sensor failures during training:* camera blackout or frozen image; sonar dropout or a dead sector; both at once; lasting 1–10 s, occasionally a whole run.
- *Health flags:* each sensor reports "OK / failed" (later version: no flags, so failures must be noticed from the data itself).
- *Weak water currents,* so that stopping isn't free.
- **Tiered safety supervisor (built and tested):** blind → **stop** (brake to a halt within ~3 s) → **act slowly**
  (≤ 0.2 m/s, from memory) → **safe abort** after 30 s; control returns the moment a sensor recovers. Modelled on
  the cave divers' "stop, breathe, think, act".
- **Hull damage instead of instant failure (built and tested):** bumping a wall no longer ends the dive. Damage
  grows with impact speed (gentle touches cost almost nothing; a 1 m/s hit costs ~23% of the hull), and a breach
  at 0% ends it. A touch sensor lets the agent feel its way along walls when blind, like a diver following the
  line by touch. Going slow is safe but costs energy and time, so the agent has to choose its speed.
- *A curriculum* from rare, short failures to frequent, long ones.
- *The model:* the best memory variant from stage 7b.

**Hoped**, with nothing hand-coded, all from the reward balance:

| Situation | Hoped behaviour |
|---|---|
| **E. Camera fails** | continue on sonar (fixing the stage-5 weakness) |
| **Sonar fails** | continue on the camera where the water is clear |
| **F. Both fail briefly** | continue on the remembered passage plus its own speed and heading |
| **Both fail for long** | slow down, hold position against the current, wait |
| **Blind because of silt** | back out of the silt cloud to regain the camera, then continue |

**How we'll know:** a failure test suite (whole-run failures, blackouts at the worst moment), measuring success, collisions during blackouts, speed while blind and recovery time. Plus held-out conditions never seen in training (longer blackouts, stronger currents, one cave type left out entirely) to measure real generalisation.

---

## How far each agent gets through the dive

| Agent | A tunnel | B crevasse | C silted chamber | D junction + dead end | E camera fails | F blackout |
|---|---|---|---|---|---|---|
| Gap follower (stage 1) | ✓ | ✗ often | ~ (sonar only) | ✗ no memory | n/a | ✗ |
| PPO, easy caves only (2) | ✗ | ✗ | ✗ | ✗ | n/a | ✗ |
| PPO, mixed difficulty (3) | ✓ | ~ | ~ | ✗ | n/a | ✗ |
| PPO, sonar + camera (5) | ✓ | ~ | **✓ measured** | ✗ | **✗ measured (0%)** | ✗ |
| Retention (7) | ~ under-trained | ~ | – | **~ fewer re-entries, measured** | – | – |
| Retention + junction memory (7b) | *hoped ✓* | *hoped ✓* | – | ***hoped ✓*** | – | – |
| + failure training (8) | *hoped ✓* | *hoped ✓* | *hoped ✓* | *hoped ✓* | ***hoped ✓*** | ***hoped: continue or stop safely*** |
| Oracle (knows the map) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

✓ handles it · ~ sometimes · ✗ fails · – not tested yet · n/a sensor not fitted. *Italics* = goal, not yet measured.
This table is updated as each stage's results come in.

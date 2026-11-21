"""Collision handling: end the episode on contact (default) or bump, take damage and carry on.

mode "terminate"  (default; all earlier results use it) any contact ends the episode, reward -10.
mode "damage"     contact is resolved physically and costs hull integrity:
  - the vehicle is pushed back out of the wall along the wall normal (gradient of the clearance map);
  - speed is lost according to how head-on the hit was: u *= 1 - cos(angle into wall) * (1 - restitution),
    so a head-on hit keeps `restitution` of its speed and a glancing scrape keeps almost all of it;
  - damage (% hull) = damage_coeff * max(0, v_in - safe_speed)^2 + scrape_coeff * speed, where v_in is the
    speed component INTO the wall and `speed` the vehicle's speed at contact (abrasion grows with sliding
    speed): a resting touch costs almost nothing, bumps at or below safe_speed only a small scrape
    (fenders / bumpers), sliding along a wall at speed (wall-surfing) adds up quickly;
  - hull starts at 100 %; at 0 % the hull is breached and the episode ends (reported as a collision).
Reward in damage mode: -damage_reward * damage (per %), -breach_penalty on a breach. Slow is safe but the
always-on hotel power and the time penalty make it inefficient, so the agent has to pick its speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from submarine.sim import VehicleState, sample_bilinear

MODES = ("terminate", "damage")


@dataclass(frozen=True)
class CollisionConfig:
    mode: str = "terminate"
    safe_speed: float = 0.2          # m/s into the wall without real damage
    damage_coeff: float = 35.0       # % hull per (m/s above safe)^2: a 1.5 m/s head-on hit costs ~60 %
    scrape_coeff: float = 1.0        # % per contact step per m/s of speed: sliding 1 s along a wall at 1 m/s ~ 10 %
    restitution: float = 0.2         # share of speed kept after a head-on hit
    damage_reward: float = 0.1       # reward per % of hull lost (100 % = 10, like the old collision penalty)
    breach_penalty: float = 10.0


def wall_normal(clearance: np.ndarray, x: float, y: float, res: float) -> np.ndarray:
    """Unit vector pointing away from the nearest wall (gradient of the clearance field)."""
    h = res
    gx = sample_bilinear(clearance, x + h, y, res) - sample_bilinear(clearance, x - h, y, res)
    gy = sample_bilinear(clearance, x, y + h, res) - sample_bilinear(clearance, x, y - h, res)
    n = np.array([gx, gy], dtype=float)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else np.zeros(2)


def resolve_contact(sim, cfg: CollisionConfig, prev_xy) -> dict | None:
    """If the vehicle penetrates a wall, push it out, reduce its speed and return the impact details."""
    s = sim.state
    radius = sim.vcfg.radius
    res = sim.cave.world.resolution
    clr = sim.clearance_at(s.x, s.y)
    if clr >= radius:
        return None
    n = wall_normal(sim.cave.clearance, s.x, s.y, res)
    heading = np.array([np.cos(s.psi), np.sin(s.psi)])
    if not np.any(n):                                     # degenerate: step back to where it came from
        x, y = prev_xy
        n = -heading
    else:
        x, y = s.x, s.y
        for _ in range(8):                                # push out along the normal until clear
            depth = radius - sim.clearance_at(x, y)
            if depth <= 0:
                break
            x, y = x + n[0] * (depth + 0.01), y + n[1] * (depth + 0.01)
            n2 = wall_normal(sim.cave.clearance, x, y, res)
            n = n2 if np.any(n2) else n
    v = s.u * heading + np.asarray(sim.current)
    v_in = max(0.0, -float(v @ n))                        # speed component into the wall
    head_on = max(0.0, -float(heading @ n))
    u = s.u * (1.0 - head_on * (1.0 - cfg.restitution))
    sim.state = VehicleState(float(x), float(y), s.psi, float(u), s.r)
    damage = cfg.damage_coeff * max(0.0, v_in - cfg.safe_speed) ** 2 + cfg.scrape_coeff * float(np.linalg.norm(v))
    return {"impact_speed": v_in, "damage": damage}

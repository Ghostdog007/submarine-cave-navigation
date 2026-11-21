"""Oracle planner with the true map (upper bound, not learnable from sensors).

Plans once per episode: a cost-weighted distance-to-goal field where cells
near walls are expensive, so the planned route stays near the middle of the
passage instead of hugging the feasibility boundary. Each step it walks
downhill on that field to a "carrot" point a short distance ahead and steers
toward it (pure pursuit).
"""

from __future__ import annotations

import numpy as np

from submarine.world import descend, geodesic_distance
from submarine.sim import wrap_angle

from .base import Policy


class Oracle(Policy):
    name = "oracle"
    privileged = True

    def __init__(self, lookahead=1.6, wall_cost=6.0, wall_scale=0.35, k_yaw=2.0, k_damp=0.3):
        self.lookahead, self.wall_cost, self.wall_scale = lookahead, wall_cost, wall_scale
        self.k_yaw, self.k_damp = k_yaw, k_damp

    def reset(self, env):
        self.env = env.unwrapped
        cave = self.env.cave
        self.cave = cave
        margin = np.maximum(cave.clearance - cave.world.vehicle_radius, 0.0)
        cost = 1.0 + self.wall_cost * np.exp(-margin / self.wall_scale)
        self.field = geodesic_distance(cave.feasible, cave.world_to_cell(*cave.goal), cave.world.resolution, cost)

    def _carrot(self, x, y):
        cave, f = self.cave, self.field
        iy, ix = cave.world_to_cell(x, y)
        if not np.isfinite(f[iy, ix]):     # centre slipped into the margin: start from the best nearby cell
            y0, y1, x0, x1 = max(iy - 2, 0), iy + 3, max(ix - 2, 0), ix + 3
            sub = f[y0:y1, x0:x1]
            if not np.isfinite(sub).any():
                return cave.goal
            dy, dx = np.unravel_index(np.argmin(sub), sub.shape)
            iy, ix = y0 + dy, x0 + dx
        res = cave.world.resolution
        iy, ix = descend(f, (iy, ix), res, self.lookahead)[-1]
        return (ix + 0.5) * res, (iy + 0.5) * res

    def act(self, obs):
        s = self.env.sim.state
        cx, cy = self._carrot(s.x, s.y)
        err = wrap_angle(np.arctan2(cy - s.y, cx - s.x) - s.psi)
        yaw = self.k_yaw * err - self.k_damp * s.r
        thrust = np.clip(1.0 - abs(err) / (np.pi / 3), 0.0, 1.0)
        clearance = self.env.sim.clearance_at(s.x, s.y)
        thrust *= np.clip((clearance - 0.3) / 0.6, 0.35, 1.0)   # ease off in tight spots
        return np.array([thrust, np.clip(yaw, -1, 1)], dtype=np.float32)

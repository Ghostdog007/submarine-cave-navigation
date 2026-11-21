"""Couples a cave, the vehicle and the sensors. The Gymnasium env wraps this."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from submarine.world import Cave

from .camera import Camera
from .config import CameraConfig, SonarConfig, VehicleConfig
from .geometry import sample_bilinear
from .sonar import Sonar
from .vehicle import VehicleState, step_vehicle


@dataclass(frozen=True)
class SimStep:
    state: VehicleState
    sonar: np.ndarray                # latest sonar scan, m (held between pings)
    clearance: float                 # m from vehicle centre to nearest rock surface
    collided: bool
    sonar_age: float = 0.0           # s since that scan was taken
    camera: np.ndarray | None = None  # camera image in [0, 1], if a camera is fitted


class Simulator:
    def __init__(self, cave: Cave, vehicle: VehicleConfig = VehicleConfig(),
                 sonar: SonarConfig = SonarConfig(), seed: int | None = None,
                 camera: CameraConfig | None = None, turbidity: np.ndarray | None = None,
                 albedo: np.ndarray | None = None):
        self.cave = cave
        self.vcfg = vehicle
        self.sonar = Sonar(sonar)
        self.camera = Camera(camera) if camera is not None else None
        self.turbidity = turbidity if turbidity is not None else np.zeros(cave.rock.shape, np.float32)
        self.albedo = albedo if albedo is not None else np.full(cave.rock.shape, 0.85, np.float32)
        self.rng = np.random.default_rng(seed)
        self.ping_every = max(1, int(round(sonar.ping_interval / vehicle.dt)))
        self.state: VehicleState | None = None
        self.current = np.zeros(2)

    def reset(self, current: tuple[float, float] | None = None) -> SimStep:
        """Place the vehicle at the cave start, at rest, facing along the passage.

        current=None samples a constant current with magnitude in [0, max_current]
        and a random direction; pass (cx, cy) to fix it.
        """
        if current is None:
            mag = self.rng.uniform(0.0, self.vcfg.max_current)
            ang = self.rng.uniform(0.0, 2 * np.pi)
            self.current = mag * np.array([np.cos(ang), np.sin(ang)])
        else:
            self.current = np.asarray(current, dtype=float)
        x, y = self.cave.start
        self.state = VehicleState(x, y, self.cave.start_heading)
        self._tick = 0
        self._scan = None
        return self._observe()

    def step(self, action) -> SimStep:
        self.state = step_vehicle(self.state, action, self.vcfg, self.current)
        self._tick += 1
        return self._observe()

    def _observe(self) -> SimStep:
        s = self.state
        res = self.cave.world.resolution
        clr = self.clearance_at(s.x, s.y)
        if self._scan is None or self._tick % self.ping_every == 0:
            self._scan = self.sonar.measure(self.cave.rock, res, s.x, s.y, s.psi, self.rng)
            self._scan_tick = self._tick
        cam = None
        if self.camera is not None:
            cam, _ = self.camera.render(self.cave.rock, self.turbidity, self.albedo, res, s.x, s.y, s.psi, self.rng)
        age = (self._tick - self._scan_tick) * self.vcfg.dt
        return SimStep(s, self._scan, clr, clr < self.vcfg.radius, age, cam)

    def refresh(self) -> SimStep:
        """Re-read the sensors at the current state without advancing time (after a contact push-out)."""
        return self._observe()

    def clearance_at(self, x: float, y: float) -> float:
        return sample_bilinear(self.cave.clearance, x, y, self.cave.world.resolution)

    def geodesic_at(self, x: float, y: float) -> float:
        return sample_bilinear(self.cave.geodesic, x, y, self.cave.world.resolution)

    def turbidity_at(self, x: float, y: float) -> float:
        return sample_bilinear(self.turbidity, x, y, self.cave.world.resolution)

    def goal_distance(self) -> float:
        """Straight-line distance from the vehicle to the goal."""
        return float(np.hypot(self.cave.goal[0] - self.state.x, self.cave.goal[1] - self.state.y))

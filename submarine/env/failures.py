"""Sensor failures (partial observability): what the agent sees when its sensors break.

Failure types
  sonar  total   all beams report max range ("open water everywhere": dangerous if trusted). Flagged.
  sonar  sector  a contiguous sector of beams (default 90 deg) reports max range. NOT flagged: the agent
                 has to notice the inconsistency itself, as with a real transducer fault.
  camera blackout  black image. Flagged.
  camera frozen    the last image keeps repeating. NOT flagged (looks plausible, as with a stuck driver).
  both            sonar total + camera blackout at once (power or bus fault). Flagged.

Onsets are random (rates per minute of simulated time), durations uniform in `duration`; with
probability `whole_episode` a sensor fails for the entire episode. Evaluation can force a schedule
(reset option "failures": [(kind, start_s, duration_s), ...]) so every policy faces identical faults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("sonar_total", "sonar_sector", "camera_blackout", "camera_frozen", "both")


@dataclass(frozen=True)
class SensorFailureConfig:
    enabled: bool = False
    rate_sonar_total: float = 0.3        # onsets per minute
    rate_sonar_sector: float = 0.3
    rate_camera_blackout: float = 0.4
    rate_camera_frozen: float = 0.3
    rate_both: float = 0.2
    duration: tuple[float, float] = (1.0, 10.0)   # s
    whole_episode: float = 0.03          # probability per sensor of failing for the entire episode
    sector_deg: float = 90.0


class FailureProcess:
    """Tracks which failures are active; applied to the sensor readings before the agent sees them."""

    def __init__(self, cfg: SensorFailureConfig, rng: np.random.Generator, dt: float, has_camera: bool,
                 schedule=None):
        self.cfg, self.rng, self.dt, self.has_camera = cfg, rng, dt, has_camera
        self.t = 0.0
        self.active: dict[str, float] = {}          # kind -> end time (inf = whole episode)
        self.sector_start = 0
        self.frozen_img = None
        self.schedule = sorted(schedule or [], key=lambda f: f[1])
        self.random = schedule is None and cfg.enabled
        if self.random:
            if rng.random() < cfg.whole_episode:
                self.active["sonar_total"] = np.inf
            if has_camera and rng.random() < cfg.whole_episode:
                self.active["camera_blackout"] = np.inf

    def _start(self, kind: str, duration: float):
        self.active[kind] = max(self.active.get(kind, 0.0), self.t + duration)
        if kind == "sonar_sector":
            self.sector_start = int(self.rng.integers(0, 10_000))
        if kind == "camera_frozen":
            self.frozen_img = None

    def step(self):
        """Advance one time step: start scheduled / random failures, end expired ones."""
        self.t += self.dt
        while self.schedule and self.schedule[0][1] <= self.t + 1e-9:
            kind, _, dur = self.schedule.pop(0)
            self._start(kind, dur)
        if self.random:
            c = self.cfg
            rates = {"sonar_total": c.rate_sonar_total, "sonar_sector": c.rate_sonar_sector, "both": c.rate_both}
            if self.has_camera:
                rates.update(camera_blackout=c.rate_camera_blackout, camera_frozen=c.rate_camera_frozen)
            for kind, rate in rates.items():
                if kind not in self.active and self.rng.random() < rate / 60.0 * self.dt:
                    self._start(kind, self.rng.uniform(*c.duration))
        self.active = {k: end for k, end in self.active.items() if end > self.t}

    # ------------------------------------------------------------------ effects

    def sonar_ok(self) -> bool:
        """Health flag: False for flagged (total) sonar failures."""
        return not ({"sonar_total", "both"} & self.active.keys())

    def camera_ok(self) -> bool:
        return not ({"camera_blackout", "both"} & self.active.keys())

    def apply_sonar(self, ranges: np.ndarray, max_range: float) -> np.ndarray:
        if not self.sonar_ok():
            return np.full_like(ranges, max_range)
        if "sonar_sector" in self.active:
            n = len(ranges)
            k = max(1, int(round(n * self.cfg.sector_deg / 360.0)))
            idx = (self.sector_start + np.arange(k)) % n
            ranges = ranges.copy()
            ranges[idx] = max_range
        return ranges

    def apply_camera(self, img: np.ndarray) -> np.ndarray:
        if not self.camera_ok():
            return np.zeros_like(img)
        if "camera_frozen" in self.active:
            if self.frozen_img is None:
                self.frozen_img = img.copy()
            return self.frozen_img
        return img

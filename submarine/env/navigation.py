"""On-board navigation features for spatial / episodic memory (no external localisation).

PathIntegrator: dead reckoning from the vehicle's own measured surge speed and yaw rate, starting at (0, 0)
with heading 0 at the cave entrance. Realistic errors: a per-episode speed-scale error, a per-episode gyro bias,
white noise on both, and the water current is not measured at all, so the estimate drifts like a real vehicle's.

JunctionDetector: from the sonar scan only, counts the separate openings (contiguous runs of beams that see
far, wide enough for the vehicle), ignoring the opening behind (the way it came).
  junction  = 2 or more openings
  dead end  = no opening ahead or to the sides
Write events (the moments a memory slot is stored):
  departure = the vehicle stops being at a junction (it has just chosen an exit; heading and position are stored)
  dead end  = a dead end is first detected
A write is skipped if the last one was less than `min_write_spacing` metres away (no duplicate slots).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_NAV_FEATURES = 7          # x, y, cos(heading), sin(heading), junction, dead end, write event


@dataclass(frozen=True)
class NavigationConfig:
    enabled: bool = False
    pos_scale: float = 25.0           # m; position features are x / pos_scale, clipped to [-1, 1] * 2
    speed_scale_err: float = 0.03     # per-episode multiplicative error, std
    speed_noise: float = 0.02         # m/s white noise, std
    gyro_bias: float = 0.002          # rad/s per-episode bias, std (~7 deg of heading drift per minute)
    gyro_noise: float = 0.01          # rad/s white noise, std
    open_range: float = 4.0           # m; a beam "sees an opening" beyond this range
    min_gap_beams: int = 2            # an opening must span at least this many beams
    rear_deg: float = 135.0           # openings centred beyond +-rear_deg (behind) are ignored
    min_write_spacing: float = 2.0    # m
    junction_flags: bool = True       # count openings every step for the junction / dead-end flags (end-to-end
                                      # agents); the navigator has its own detector and switches this off (~11%)


class PathIntegrator:
    def __init__(self, cfg: NavigationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.scale = 1.0 + rng.normal(0, cfg.speed_scale_err)
        self.bias = rng.normal(0, cfg.gyro_bias)
        self.x = self.y = self.psi = 0.0

    def step(self, u: float, r: float, dt: float):
        c = self.cfg
        u_meas = u * self.scale + self.rng.normal(0, c.speed_noise)
        r_meas = r + self.bias + self.rng.normal(0, c.gyro_noise)
        self.psi += r_meas * dt
        self.x += u_meas * np.cos(self.psi) * dt
        self.y += u_meas * np.sin(self.psi) * dt


def count_openings(ranges: np.ndarray, angles: np.ndarray, cfg: NavigationConfig) -> list[float]:
    """Centre angles (rad, body frame) of the openings in a 360-degree scan, excluding the one behind."""
    open_ = ranges > cfg.open_range
    n = len(ranges)
    if open_.all():
        return [0.0]                                    # open water all round: treat as one wide opening
    if not open_.any():
        return []
    start = int(np.argmin(open_))                       # rotate so the scan starts on a closed beam
    o = np.roll(open_, -start)
    a = np.roll(angles, -start)
    gaps, run = [], []
    for k in range(n):
        if o[k]:
            run.append(a[k])
        elif run:
            gaps.append(run)
            run = []
    if run:
        gaps.append(run)
    centres = []
    for g in gaps:
        if len(g) < cfg.min_gap_beams:
            continue                                    # single long beams look like sonar dropouts
        c = float(np.angle(np.mean(np.exp(1j * np.array(g)))))
        if abs(np.degrees(c)) <= cfg.rear_deg:
            centres.append(c)
    return centres


class JunctionDetector:
    def __init__(self, cfg: NavigationConfig):
        self.cfg = cfg
        self.at_junction = False
        self.at_dead_end = False
        self.last_write = None

    def update(self, ranges, angles, pos) -> tuple[bool, bool, bool]:
        """Returns (junction, dead_end, write_event) for the current scan and estimated position."""
        openings = count_openings(ranges, angles, self.cfg)
        junction = len(openings) >= 2
        dead_end = len(openings) == 0
        event = (self.at_junction and not junction) or (dead_end and not self.at_dead_end)
        if event and self.last_write is not None and \
                np.hypot(pos[0] - self.last_write[0], pos[1] - self.last_write[1]) < self.cfg.min_write_spacing:
            event = False
        if event:
            self.last_write = (pos[0], pos[1])
        self.at_junction, self.at_dead_end = junction, dead_end
        return junction, dead_end, event

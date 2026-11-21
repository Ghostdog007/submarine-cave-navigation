"""Openings detector: turns one sonar scan into the short list of exits the navigator chooses from.

A direction's *reach* is how far a vehicle-width corridor extends that way (the gap follower's test, so an
exit is one the vehicle actually fits through, not a long beam between two pillars). Exits are the peaks of
the reach profile around the vehicle:

  * a peak must reach at least `open_range` metres;
  * two peaks are separate exits only if the reach dips by at least `prominence` metres between them, so a
    wide open chamber gives one exit per way out of it (where reach is deepest), not one huge "opening";
  * an exit centred further round than `rear_deg` is the way back (`rear`), kept as the "turn back" option.

The first version used contiguous open runs; in a chamber the whole circle was one run whose deepest direction
could point backwards, so open water was reported as a dead end. Peaks fix that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from submarine.baselines.reactive import corridor_ranges


@dataclass(frozen=True)
class OpeningsConfig:
    half_width: float = 0.45          # m; vehicle radius 0.4 + 5 cm: corridor that must be free (audited:
                                      # 0.55 missed 21% of the true routes in crevasses)
    open_range: float = 3.0           # m; an exit must reach at least this far
    prominence: float = 2.5           # m; dip in reach needed between two peaks for them to be separate exits
    min_beams: int = 1                # a 2 m passage is only one beam direction wide (11.25 deg beams) once the
                                      # vehicle width counts; dropouts are rejected by the corridor test itself
    narrow_range: float = 1.5         # m; if no exit reaches open_range, the deepest forward direction reaching
                                      # this far is offered as a *narrow* exit (zig-zag crevasses: the corridor
                                      # rarely runs 3 m straight, and 21 of 34 crevasse "dead ends" were false)
    rear_deg: float = 135.0           # exits centred beyond +-rear_deg are the way back
    narrow_deg: float = 45.0          # narrow exits only within +-narrow_deg: wider cones see the side walls of a
                                      # real dead end as "narrow exits" (audited: 90 deg hid 30 of 32 maze dead ends)
    max_openings: int = 4             # forward/side exits kept (deepest first), plus at most one rear


@dataclass(frozen=True)
class Opening:
    bearing: float        # rad, body frame (0 = straight ahead, + = left): direction of deepest reach
    width: float          # rad, angular span where reach stays within `prominence` of the peak
    depth: float          # m, corridor reach along `bearing` (capped at the sonar range)
    rear: bool            # True: this is the way back
    narrow: bool = False  # True: only reaches narrow_range (offered because nothing ahead reaches open_range)


def _peaks(reach: np.ndarray, circular: bool, open_range: float, prominence: float) -> list[tuple[int, np.ndarray]]:
    """(peak index, member indices) of prominent reach peaks.

    Directions are visited deepest first. Each one grows a region both ways while the reach stays at or above
    max(open_range, its reach - prominence). If the region runs into a direction already claimed by a deeper
    exit, there is no dip of `prominence` between them: it is part of that exit (claimed, not reported).
    """
    n = len(reach)
    taken = np.zeros(n, dtype=bool)
    out = []
    for k in np.argsort(-reach, kind="stable"):
        if taken[k] or reach[k] < open_range:
            continue
        level = max(open_range, reach[k] - prominence)
        members, joins_deeper = [k], False
        for step in (-1, 1):
            i = k
            for _ in range(n - 1):
                j = i + step
                if circular:
                    j %= n
                elif not 0 <= j < n:
                    break
                if j == k or reach[j] < level:
                    break
                if taken[j]:
                    joins_deeper = True
                    break
                members.append(j)
                i = j
        m = np.unique(members)
        taken[m] = True
        if not joins_deeper:
            out.append((int(k), m))
    return out


def detect_openings(ranges: np.ndarray, angles: np.ndarray, max_range: float,
                    cfg: OpeningsConfig = OpeningsConfig(), full_circle: bool = True) -> list[Opening]:
    """Exits in one scan, sorted by bearing (right to left); the rear exit (if any) last."""
    angles = np.asarray(angles, float)
    reach = corridor_ranges(np.asarray(ranges, float), angles, cfg.half_width, max_range)
    step = float(np.mean(np.abs(np.diff(angles)))) if len(angles) > 1 else 0.0
    out = []
    for k, members in _peaks(reach, full_circle, cfg.open_range, cfg.prominence):
        if len(members) < cfg.min_beams:
            continue
        deepest = members[reach[members] >= reach[k] - 1e-6]     # plateau (e.g. all at max range): its middle
        bearing = float(np.angle(np.mean(np.exp(1j * angles[deepest]))))
        out.append(Opening(bearing, len(members) * step, float(reach[k]),
                           abs(np.degrees(bearing)) > cfg.rear_deg))
    if not any(not o.rear for o in out):                         # nothing open ahead: a squeeze or a dead end?
        fwd = np.abs(angles) <= np.deg2rad(cfg.narrow_deg)        # ahead only, never back the way it came
        if fwd.any() and reach[fwd].max() >= cfg.narrow_range:
            k = np.flatnonzero(fwd)[int(np.argmax(reach[fwd]))]
            out.append(Opening(float(angles[k]), step, float(reach[k]), False, narrow=True))
    front = sorted((o for o in out if not o.rear), key=lambda o: -o.depth)[:cfg.max_openings]
    rear = sorted((o for o in out if o.rear), key=lambda o: -o.depth)[:1]
    return sorted(front, key=lambda o: o.bearing) + rear

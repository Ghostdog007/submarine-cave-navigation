"""Configuration for the 2D world and the procedural cave generator.

`WorldConfig` holds fixed map / vehicle geometry. `CaveConfig` holds the
generator parameters; `CaveConfig.from_difficulty(d)` maps the difficulty knob
d in [0, 1] onto them (see context.md, section 2).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass(frozen=True)
class WorldConfig:
    length: float = 50.0          # m, x extent
    height: float = 30.0          # m, y extent
    resolution: float = 0.2       # m per cell
    border: float = 0.4           # m of solid rock around the map edge
    end_margin: float = 0.6       # m of rock between the map edge and the cave's end walls
    vehicle_radius: float = 0.4   # m
    start_window: tuple[float, float] = (1.5, 3.5)    # x range where the start point is chosen
    goal_window: tuple[float, float] = (46.5, 48.5)   # x range where the goal point is chosen

    @property
    def nx(self) -> int:
        return int(round(self.length / self.resolution))

    @property
    def ny(self) -> int:
        return int(round(self.height / self.resolution))


@dataclass(frozen=True)
class CaveConfig:
    difficulty: float = 0.0
    layout: str = "tunnel"        # tunnel | maze | branches | crack | breakdown (see cave/layouts.py)

    # Passage width profile
    base_width: float = 7.0
    width_noise: float = 0.15                         # low-frequency width noise, fraction of base width
    n_chambers: tuple[int, int] = (1, 3)              # inclusive range
    chamber_gain: tuple[float, float] = (1.5, 4.0)    # m added to the width at the bulge peak
    chamber_sigma: tuple[float, float] = (2.5, 5.0)   # m, Gaussian std along x

    # Cracks: short tapered narrow sections
    n_cracks_max: int = 0
    crack_width: float = 5.0
    crack_length: tuple[float, float] = (3.0, 6.0)    # m, length of the narrow plateau
    crack_taper: float = 1.5                          # m, cosine taper on each side
    crack_region: tuple[float, float] = (8.0, 42.0)   # x range cracks may occupy
    crack_spacing: float = 3.0                        # m, minimum gap between cracks

    # Wall roughness (smooth 2D noise added to the free-space test)
    roughness: float = 0.0                # m, amplitude
    roughness_scale: float = 1.2          # m, feature size of the noise
    roughness_max_frac: float = 0.35      # cap amplitude at this fraction of the local half-width

    # Centerline
    wander: float = 2.0                   # m, peak offset of the centerline from mid-height
    n_sinusoids: int = 3
    wander_wavelength: tuple[float, float] = (20.0, 60.0)
    max_slope: float = 1.5                # |dy_c/dx| limit (about 56 degrees)

    # Pillars (only in wide sections)
    n_pillars: int = 1                    # target count; fewer are placed if they don't fit
    pillar_radius: tuple[float, float] = (0.4, 0.9)
    pillar_gap: float = 1.2               # m free on each side of a pillar after roughness (vehicle 0.8 m + margin)
    pillar_region: tuple[float, float] = (6.0, 44.0)

    max_attempts: int = 50                # regeneration attempts before giving up

    @classmethod
    def from_difficulty(cls, d: float, **overrides) -> "CaveConfig":
        if not 0.0 <= d <= 1.0:
            raise ValueError(f"difficulty must be in [0, 1], got {d}")
        cfg = cls(
            difficulty=d,
            base_width=_lerp(7.0, 3.0, d),
            n_cracks_max=int(round(3 * d)),
            crack_width=_lerp(5.0, 1.7, d),
            roughness=_lerp(0.0, 0.8, d),
            wander=_lerp(2.0, 8.0, d),
            n_pillars=int(round(_lerp(1, 10, d))),
        )
        return replace(cfg, **overrides)

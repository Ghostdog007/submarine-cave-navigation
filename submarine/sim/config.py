"""Vehicle and sonar configuration (context.md, sections 3 and 4.1)."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class VehicleConfig:
    dt: float = 0.1               # s
    radius: float = 0.4           # m, body is a disc
    a_max: float = 0.75           # m/s^2 surge acceleration at full thrust
    k_u: float = 0.5              # 1/s linear drag; steady speed at full thrust = a_max / k_u = 1.5 m/s
    u_max: float = 1.5            # m/s hard cap
    reverse_gain: float = 0.5     # reverse thrust and reverse speed cap are this fraction of forward
    r_max: float = 1.0            # rad/s commanded yaw rate at |yaw_cmd| = 1
    tau_r: float = 0.3            # s, yaw-rate lag
    max_current: float = 0.0      # m/s, per-episode current magnitude drawn from [0, max_current]


@dataclass(frozen=True)
class SonarConfig:
    n_beams: int = 32
    fov_deg: float = 360.0
    max_range: float = 10.0       # m
    step: float = 0.1             # m, ray-march step
    noise_std: float = 0.05       # m, constant part of the range noise std
    noise_frac: float = 0.02      # range-proportional part of the noise std
    dropout: float = 0.02         # probability a beam returns max range
    min_range: float = 0.0        # m, blind zone: closer returns are reported as min_range
    beam_width_deg: float = 0.0   # > 0: each beam returns the minimum over this angular spread
    rays_per_beam: int = 3        # sub-rays used when beam_width_deg > 0
    ping_interval: float = 0.1    # s between pings; between pings the last scan is held (stale)

    def ideal(self) -> "SonarConfig":
        """Same geometry, no noise or dropout."""
        return replace(self, noise_std=0.0, noise_frac=0.0, dropout=0.0)


@dataclass(frozen=True)
class CameraConfig:
    """Forward-looking camera lit by the submarine's headlight (context.md, section 4.4).

    In the 2D world the camera produces a 1D image (one scanline): each pixel
    is the brightness returned along its ray. Brightness falls off with
    distance (headlight spreading + water attenuation), murky water attenuates
    much more strongly and adds a backscatter "veil" with speckle, so walls
    fade out in turbid zones. Sonar is not affected by turbidity.
    """

    n_pixels: int = 64
    fov_deg: float = 90.0
    max_range: float = 10.0       # m, ray-march limit
    step: float = 0.1             # m
    c_clear: float = 0.05         # 1/m attenuation of clear water
    c_turbid: float = 1.2         # 1/m extra attenuation at turbidity 1
    b_clear: float = 0.01         # 1/m backscatter coefficient of clear water
    b_turbid: float = 0.9         # 1/m extra backscatter at turbidity 1
    light_range: float = 2.5      # m, headlight spreading: illumination = 1 / (1 + (d / light_range)^2)
    cone_deg: float = 55.0        # headlight beam width (Gaussian falloff toward the image edges)
    gain: float = 1.6             # exposure
    read_noise: float = 0.01      # sensor noise floor
    shot_noise: float = 0.02      # noise growing with sqrt(brightness)
    speckle: float = 0.3          # relative noise on the backscatter veil


@dataclass(frozen=True)
class TurbidityConfig:
    """Murky patches of water: a smooth field in [0, 1] per cave (0 = clear)."""

    enabled: bool = False
    n_zones: tuple[int, int] = (2, 4)           # inclusive range of murky patches per cave
    radius: tuple[float, float] = (3.0, 7.0)    # m, patch size (about 2 standard deviations)
    peak: tuple[float, float] = (0.6, 1.0)      # turbidity at a patch centre
    background: float = 0.05                    # everywhere else
    scale: float = 1.0                          # multiplies the whole field (for murk sweeps)

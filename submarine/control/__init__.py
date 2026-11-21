"""Driver: tracking controller + built-in safety filter + openings detector (two-level architecture).

Phase A0 is the hand-written controller; the learned driver (phases A1-A3) replaces only the controller.
"""

from .controller import ControllerConfig, HandController
from .driver import Driver, DriverConfig, Exit
from .driver_env import N_EXTRAS, DriverEnv, DriverEnvConfig, DriverRewardConfig
from .openings import Opening, OpeningsConfig, detect_openings
from .policy import DriverPolicy
from .safety import Odometry, SafetyConfig, SafetyFilter, scan_points

__all__ = ["N_EXTRAS", "ControllerConfig", "Driver", "DriverConfig", "DriverEnv", "DriverEnvConfig",
           "DriverRewardConfig", "DriverPolicy", "Exit", "HandController", "Odometry",
           "Opening", "OpeningsConfig", "SafetyConfig", "SafetyFilter", "detect_openings", "scan_points"]

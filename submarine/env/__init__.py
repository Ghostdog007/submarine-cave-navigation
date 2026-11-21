from gymnasium.envs.registration import register, registry

from .cave_env import CaveEnv
from .config import EnvConfig, RewardConfig, coerce_config

if "SubmarineCave-v0" not in registry:
    register(id="SubmarineCave-v0", entry_point="submarine.env.cave_env:CaveEnv")

__all__ = ["CaveEnv", "EnvConfig", "RewardConfig", "coerce_config"]

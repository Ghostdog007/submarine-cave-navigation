"""Build a policy from a name: a baseline ("gap", "oracle", ...) or a trained run dir / model .zip."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from submarine.baselines import BASELINES

from .sb3_policy import SB3Policy


def make_policy(name: str):
    if name in BASELINES:
        return BASELINES[name]()
    if name in ("driver", "driver-random"):            # phase A0 driver with a scripted teacher navigator
        from submarine.control import DriverPolicy

        return DriverPolicy(mode="goal" if name == "driver" else "random")
    if name.startswith("driver-net:"):                 # learned driver (exported actor) + teacher navigator
        from submarine.control.learned import LearnedDriverPolicy

        return LearnedDriverPolicy(name.split(":", 1)[1])
    from submarine.agents import load_run

    path = Path(name)
    model, tcfg = load_run(path)
    label = path.name if path.is_dir() else f"{path.parent.name}/{path.stem}"
    pol = SB3Policy(model, label, deterministic=True, frame_stack=tcfg.frame_stack)
    # what the vehicle itself carries: its sensors and its on-board navigation (dead reckoning + junction
    # events). Caves, murk, currents, failures and reward stay those of the evaluation config.
    pol.sensor_setup = {"sensors": tcfg.env.sensors, "camera": tcfg.env.camera, "sonar": tcfg.env.sonar,
                        "navigation": tcfg.env.navigation}
    return pol


def env_config_for(policy, base):
    """The evaluation config, with the policy's own sensor setup if it was trained with one.

    Caves, difficulty, murk, current and reward stay those of `base`, so all
    policies face the same conditions; only what the policy observes changes.
    """
    setup = getattr(policy, "sensor_setup", None)
    return replace(base, **setup) if setup else base

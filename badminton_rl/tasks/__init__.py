"""Gym registrations for the badminton curriculum stages."""

import gymnasium as gym

from .. import training_hooks
from . import agents

training_hooks.install()

_STAGES = {
    "Grasp": "BadmintonGraspEnvCfg",
    "Hit": "BadmintonHitEnvCfg",
    "Target": "BadmintonTargetEnvCfg",
    "Mobile": "BadmintonMobileEnvCfg",
    # the same curriculum from scratch on the wheeled robot
    "MobileGrasp": "BadmintonMobileGraspEnvCfg",
    "MobileHit": "BadmintonMobileHitEnvCfg",
    "MobileTarget": "BadmintonMobileTargetEnvCfg",
    "MobileCoverage": "BadmintonMobileCoverageEnvCfg",
    "MobileRally": "BadmintonMobileRallyEnvCfg",
}

for _stage, _cfg in _STAGES.items():
    gym.register(
        id=f"Badminton-{_stage}-KukaAllegro",
        entry_point="badminton_rl.tasks.badminton_env:BadmintonEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.badminton_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Badminton{_stage}PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )

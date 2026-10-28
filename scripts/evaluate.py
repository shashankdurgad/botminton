"""Evaluate a badminton checkpoint over many episodes with the deterministic (mean) policy.

The curriculum is frozen at the requested difficulty. Every finished episode is counted, and the
per-stage success metrics are reported over all of them.

Usage:
    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 OMNI_KIT_ACCEPT_EULA=YES \
        ../IsaacLab/.venv/bin/python scripts/evaluate.py --task Badminton-Target-KukaAllegro \
        --checkpoint logs/.../model_5000.pt --episodes 2000 --difficulty 1.0
"""

import argparse
import ast
import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--difficulty", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--stochastic", action="store_true", help="sample actions like training rollouts do")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="env cfg overrides (Python literals)")
parser.add_argument("--zero_base_actions", action="store_true", help="mobile: zero the policy's base actions")
add_launcher_args(parser)
args = parser.parse_args()

METRICS = ("success_rate", "held_rate", "hit_rate", "cleared_net_rate")


def main():
    import gymnasium as gym
    import torch

    import badminton_rl.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )
    from isaaclab_tasks.utils import resolve_task_config

    sys.argv = sys.argv[:1]  # resolve_task_config parses Hydra overrides from sys.argv
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.curriculum_init_difficulty = args.difficulty
    env_cfg.curriculum_step = 0.0  # frozen
    for kv in args.set:
        key, value = kv.split("=", 1)
        assert hasattr(env_cfg, key), key
        setattr(env_cfg, key, ast.literal_eval(value))
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped

        totals = dict.fromkeys(METRICS, 0.0)
        finished = 0
        obs = env.get_observations()
        while finished < args.episodes:
            with torch.inference_mode():
                act = runner.alg.actor(obs, stochastic_output=True) if args.stochastic else policy(obs)
                if args.zero_base_actions:
                    act[:, u.num_dofs :] = 0.0
                obs, _, dones, extras = env.step(act)
            n = int(dones.sum().item())
            if n == 0:
                continue
            # the env logs these metrics over exactly the episodes that reset in this step
            log = u.extras.get("log", {})
            for m in METRICS:
                totals[m] += float(log[f"Metrics/{m}"]) * n
            finished += n

        mode = "stochastic" if args.stochastic else "deterministic"
        print(f"EVAL task={args.task} difficulty={u.difficulty:.2f} episodes={finished} {mode}")
        print(f"EVAL checkpoint={args.checkpoint} set={args.set} zero_base_actions={args.zero_base_actions}")
        for m in METRICS:
            print(f"EVAL {m:18s} {totals[m] / finished:.3f}")
        env.close()


if __name__ == "__main__":
    main()

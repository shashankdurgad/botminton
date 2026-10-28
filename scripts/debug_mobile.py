"""Debug the floating-root mobile base: run a checkpoint with the base commanded still and trace the root.

Prints, for a few envs, the commanded base pose vs the simulated root pose and the grip error over time.
"""

import argparse
import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Badminton-Mobile-KukaAllegro")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--num_envs", type=int, default=16)
add_launcher_args(parser)
args = parser.parse_args()


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

    sys.argv = sys.argv[:1]
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.curriculum_step = 0.0
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped
        env.reset()
        obs = env.get_observations()
        mobile = getattr(u, "mobile", False)
        for k in range(args.steps):
            with torch.inference_mode():
                act = policy(obs)
                if mobile:
                    act[:, -3:] = 0.0
                obs, _, dones, _ = env.step(act)
            if k % 12 == 0:
                o = u.scene.env_origins
                root = u.robot.data.root_pos_w.torch - o
                rq = u.robot.data.root_quat_w.torch
                rv = u.robot.data.root_lin_vel_w.torch
                link0 = u.robot.data.body_pos_w.torch[:, 0] - o
                msg = (f"step {k:3d} t={u.episode_length_buf[0].item() * u.step_dt:4.2f} "
                       f"root0={root[0].cpu().numpy().round(4)} q0={rq[0].cpu().numpy().round(3)} "
                       f"v0={rv[0].cpu().numpy().round(3)} link0={link0[0].cpu().numpy().round(4)} "
                       f"grip_err mean={u.grip_error.mean().item():.3f} max={u.grip_error.max().item():.3f} "
                       f"root_drift max={(root[:, :2] - (u.base_pos if mobile else 0)).norm(dim=-1).max().item():.4f} "
                       f"z min/max={root[:, 2].min().item():.3f}/{root[:, 2].max().item():.3f}")
                print("DBG", msg, flush=True)
        env.close()


if __name__ == "__main__":
    main()

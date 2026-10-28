"""Debug the mobile base grip: drive the base (scripted) with the arm and fingers frozen, and trace the grip.

Prints, for a few envs, the commanded base pose vs the simulated root pose and the grip error over time.
"""

import argparse
import ast
import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Badminton-Mobile-KukaAllegro")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--difficulty", type=float, default=1.0)
parser.add_argument("--freeze_arm", action="store_true")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
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
    env_cfg.curriculum_init_difficulty = args.difficulty
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
        env.reset()
        obs = env.get_observations()
        mobile = getattr(u, "mobile", False)
        drops, episodes = [], 0
        drift_max = 0.0
        prev_vel = u.base_vel.clone()
        for k in range(args.steps):
            snap = dict(vel=u.base_vel.clone(), acc=(u.base_vel - prev_vel) / u.step_dt, pos=u.base_pos.clone(),
                        t=u.episode_length_buf.float() * u.step_dt, hit=u.hit.clone())
            prev_vel = u.base_vel.clone()
            with torch.inference_mode():
                act = policy(obs)
                if args.freeze_arm:
                    act[:, : u.num_dofs] = 0.0
                if mobile:
                    act[:, u.num_dofs :] = 0.0
                obs, _, dones, _ = env.step(act)
            episodes += int(dones.sum().item())
            root = u.robot.data.root_pos_w.torch - u.scene.env_origins
            live = ~dones.bool()
            if live.any():
                drift = torch.cat(((root[:, :2] - u.base_pos).norm(dim=-1, keepdim=True),
                                   (root[:, 2:] - u.cfg.base_height).abs()), dim=-1)[live]
                drift_max = max(drift_max, drift.max().item())
            for i in u._dropped.nonzero(as_tuple=True)[0].tolist():
                drops.append((snap["t"][i].item(), snap["vel"][i, :2].norm().item(), snap["acc"][i, :2].norm().item(),
                              *snap["pos"][i].tolist(), bool(snap["hit"][i])))
            if k % 120 == 0:
                o = u.scene.env_origins
                root = u.robot.data.root_pos_w.torch - o
                rq = u.robot.data.root_quat_w.torch
                rv = u.robot.data.root_lin_vel_w.torch
                link0 = u.robot.data.body_pos_w.torch[:, 0] - o
                msg = (f"step {k:3d} t={u.episode_length_buf[0].item() * u.step_dt:4.2f} "
                       f"root0={root[0].cpu().numpy().round(4)} q0={rq[0].cpu().numpy().round(3)} "
                       f"v0={rv[0].cpu().numpy().round(3)} link0={link0[0].cpu().numpy().round(4)} "
                       f"grip_err mean={u.grip_error.mean().item():.3f} max={u.grip_error.max().item():.3f} "
                       f"base_speed max={u.base_vel[:, :2].norm(dim=-1).max().item():.2f} held={(u.grip_error < 0.05).float().mean().item():.2f} "
                       f"")
                print("DBG", msg, flush=True)
        import numpy as np
        d = np.array(drops) if drops else np.zeros((0, 6))
        print(f"DROPS {len(d)} of {episodes} finished episodes, max root drift from commanded {drift_max:.4f} m")
        if len(d):
            for name, col in (("t", 0), ("speed", 1), ("accel", 2), ("base_x", 3), ("base_y", 4), ("after_hit", 5)):
                print(f"DROPS {name:9s} pcts10/50/90 = {np.percentile(d[:, col], [10, 50, 90]).round(2)}")
            print(f"DROPS at x bound (>5.1) {np.mean(d[:, 3] > 5.1):.2f}, at |y| bound (>3.5) {np.mean(np.abs(d[:, 4]) > 3.5):.2f}")
        env.close()


if __name__ == "__main__":
    main()

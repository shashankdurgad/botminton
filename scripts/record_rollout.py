"""Record a policy rollout (robot link, racket and shuttle poses per control step) for offline rendering.

Runs one env with the deterministic policy and the curriculum frozen, and saves an ``.npz`` that
``scripts/build_showcase.py`` turns into a time-sampled USD animation.

Usage:
    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 OMNI_KIT_ACCEPT_EULA=YES \
        ../IsaacLab/.venv/bin/python scripts/record_rollout.py --task Badminton-Target-KukaAllegro \
        --checkpoint logs/.../model_10599.pt --episodes 30 --out rollout.npz
"""

import argparse
import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--difficulty", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", required=True)
add_launcher_args(parser)
args = parser.parse_args()


def main():
    import gymnasium as gym
    import numpy as np
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
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.curriculum_init_difficulty = args.difficulty
    env_cfg.curriculum_step = 0.0  # frozen
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        from pxr import UsdPhysics  # only after Kit has loaded its USD libraries

        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped
        origin = u.scene.env_origins[0]

        # map articulation bodies to their prims, relative to the robot root
        stage = u.sim.stage
        robot_root = "/World/envs/env_0/Robot"
        body_names = list(u.robot.body_names)
        body_paths = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(robot_root + "/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if prim.GetName() in body_names:
                    body_paths[prim.GetName()] = path[len(robot_root) :]
        missing = [b for b in body_names if b not in body_paths]
        assert not missing, f"no prim for bodies {missing}"

        frames = {k: [] for k in ("body_pos", "body_quat", "racket_pos", "racket_quat", "shuttle_pos", "shuttle_vel")}
        episode_of_frame, outcomes = [], []
        episode = 0

        def snapshot():
            frames["body_pos"].append((u.robot.data.body_pos_w.torch[0] - origin).cpu().numpy())
            frames["body_quat"].append(u.robot.data.body_quat_w.torch[0].cpu().numpy())
            frames["racket_pos"].append((u.racket.data.root_pos_w.torch[0] - origin).cpu().numpy())
            frames["racket_quat"].append(u.racket.data.root_quat_w.torch[0].cpu().numpy())
            frames["shuttle_pos"].append((u.shuttle.data.root_pos_w.torch[0] - origin).cpu().numpy())
            frames["shuttle_vel"].append(u.shuttle.data.root_lin_vel_w.torch[0].cpu().numpy())
            episode_of_frame.append(episode)

        env.reset()
        obs = env.get_observations()
        targets = [u.land_target[0].cpu().numpy().copy()]
        snapshot()
        while episode < args.episodes:
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            if dones[0]:
                # the env already reset; its log holds the finished episode's metrics
                log = u.extras["log"]
                outcomes.append(
                    {k: float(log[f"Metrics/{k}"]) for k in ("success_rate", "hit_rate", "cleared_net_rate")}
                )
                episode += 1
                if episode == args.episodes:
                    break
                targets.append(u.land_target[0].cpu().numpy().copy())
            snapshot()

        np.savez_compressed(
            args.out,
            dt=u.step_dt,
            body_names=np.array(body_names),
            body_paths=np.array([body_paths[b] for b in body_names]),
            episode=np.array(episode_of_frame),
            land_target=np.array(targets),
            success=np.array([o["success_rate"] for o in outcomes]),
            hit=np.array([o["hit_rate"] for o in outcomes]),
            cleared_net=np.array([o["cleared_net_rate"] for o in outcomes]),
            robot_usd=u.cfg.scene.robot.spawn.usd_path,
            robot_pos=np.array(u.cfg.scene.robot.init_state.pos),
            **{k: np.array(v) for k, v in frames.items()},
        )
        s = np.array([o["success_rate"] for o in outcomes])
        print(f"RECORD frames={len(episode_of_frame)} episodes={len(outcomes)} in_target={s.mean():.2f} -> {args.out}")
        env.close()


if __name__ == "__main__":
    main()

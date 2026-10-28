"""Sanity probes for the badminton environments.

palm     - print the palm frame and the fingertip / knuckle positions in it (to set the grasp offset)
shuttle  - drop and launch shuttles to check the drag terminal velocity and the launch solver
grip     - run the grasp stage with zero actions and report the racket grip error over time

Run (aarch64 needs the libgomp preload):
    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 .venv/bin/python ../badminton_rl/scripts/probe.py palm --headless
"""

import argparse
import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["palm", "shuttle", "grip", "sweep"])
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--close", default="(index|middle|ring)_joint_(1|2|3)|thumb_joint_.*", help="joints driven to flexion in grip mode")
parser.add_argument("--trace", action="store_true")
parser.add_argument("--finger_gain", type=float, default=1.0, help="scale finger stiffness/damping/effort")
parser.add_argument("--friction", type=float, default=None)
parser.add_argument("--grace", type=float, default=None, help="override the racket pin period [s]")
add_launcher_args(parser)
args = parser.parse_args()


def main():
    import gymnasium as gym
    import torch

    import badminton_rl.tasks  # noqa: F401
    from isaaclab.utils.math import quat_apply, subtract_frame_transforms
    from isaaclab_tasks.utils import resolve_task_config

    if args.mode == "sweep":
        xs, zs = (0.0, 0.02, 0.04, 0.06), (0.03, 0.04, 0.05)
        candidates = [(x, 0.02, z) for x in xs for z in zs]
        args.num_envs = len(candidates)
    task = "Badminton-Hit-KukaAllegro" if args.mode == "shuttle" else "Badminton-Grasp-KukaAllegro"
    sys.argv = sys.argv[:1]  # resolve_task_config parses Hydra overrides from sys.argv
    cfg, _ = resolve_task_config(task, "")
    cfg.scene.num_envs = args.num_envs
    if args.mode == "palm":
        cfg.grace_period_s = 100.0
    if args.finger_gain != 1.0:
        act = cfg.scene.robot.actuators["kuka_allegro_actuators"]
        for attr in ("stiffness", "damping", "joint_effort_limit"):
            d = getattr(act, attr)
            for key in d:
                if "index" in key or "thumb" in key or "middle" in key or "ring" in key:
                    d[key] = d[key] * args.finger_gain
    if args.friction is not None:
        cfg.sim.physics_material.static_friction = args.friction
        cfg.sim.physics_material.dynamic_friction = args.friction
    if args.mode == "shuttle":
        cfg.drop_distance = 10.0  # zero actions drop the racket; keep the episode alive for the flight
        cfg.episode_length_s = 10.0
    if args.mode == "sweep":
        cfg.drop_distance = 10.0  # never terminate, measure the slip instead
        cfg.grace_period_s = 2.5
        cfg.episode_length_s = 6.0
    if args.grace is not None:
        cfg.grace_period_s = args.grace
        cfg.episode_length_s = args.grace + 2.0
    args.device = cfg.sim.device

    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg)
        u = env.unwrapped
        env.reset()
        zero = torch.zeros((u.num_envs, cfg.action_space), device=u.device)

        if args.mode == "palm":
            for _ in range(5):
                env.step(zero)
            r = u.robot
            print("joint names:", r.joint_names)
            print("body names:", r.body_names)
            palm_p = r.data.body_pos_w.torch[0, u.palm_id]
            palm_q = r.data.body_quat_w.torch[0, u.palm_id]
            print(f"palm pos (env) {(palm_p - u.scene.env_origins[0]).tolist()}  quat xyzw {palm_q.tolist()}")
            for axis, v in (("x", (1.0, 0, 0)), ("y", (0, 1.0, 0)), ("z", (0, 0, 1.0))):
                w = quat_apply(palm_q.unsqueeze(0), torch.tensor([v], device=u.device))[0]
                print(f"  palm local {axis} -> world {[round(x, 3) for x in w.tolist()]}")
            for i, name in enumerate(r.body_names):
                if any(k in name for k in ("tip", "link_0", "link_3", "palm")):
                    p, _ = subtract_frame_transforms(
                        palm_p.unsqueeze(0), palm_q.unsqueeze(0), r.data.body_pos_w.torch[0, i].unsqueeze(0)
                    )
                    print(f"  {name:20s} in palm frame: {[round(x, 4) for x in p[0].tolist()]}")
            print("racket grip error [m]:", u.grip_error[0].item(), " nominal head (env):", u.nominal_head_pos.tolist())

        elif args.mode == "shuttle":
            # free fall from 30 m (with drag) to measure the terminal velocity
            ids = torch.arange(u.num_envs, device=u.device)
            start = torch.tensor([-6.0, 0.0, 30.0], device=u.device).repeat(u.num_envs, 1) + u.scene.env_origins
            u._write_shuttle(ids, start, torch.zeros((u.num_envs, 3), device=u.device))
            u.shuttle_active[:] = True
            u.episode_length_buf[:] = 0
            for _ in range(int(3.0 / u.step_dt)):
                u.step(zero)  # bypass the gym wrapper's resets bookkeeping
            print(f"fall speed after 3 s: {u.shuttle_vel[:, 2].tolist()} (expect ~ -{cfg.shuttle_terminal_velocity} x drag noise)")
            # launches: min distance from the intercept point
            env.reset()
            best = torch.full((u.num_envs,), 1e9, device=u.device)
            centre = u.nominal_head_pos
            for _ in range(int(1.6 / u.step_dt)):
                env.step(zero)
                best = torch.minimum(best, (u.shuttle_pos - centre).norm(dim=-1))
            print(f"closest approach to the nominal intercept (difficulty 0, +-5 cm box): {best.tolist()}")

        elif args.mode == "sweep":
            u.grasp_pos[:] = torch.tensor(candidates, device=u.device)
            env.reset()
            close = zero.clone()
            finger_ids, _ = u.robot.find_joints(args.close)
            close[:, finger_ids] = 1.0
            fid, fnames = u.robot.find_joints("(index|middle|ring|thumb)_joint_.*")
            release = int(cfg.grace_period_s / u.step_dt)
            worst = torch.zeros(u.num_envs, device=u.device)
            for k in range(release + int(2.0 / u.step_dt)):
                u.step(close)
                if k == release - 2:
                    settled = u.joint_pos[:, fid].clone()
                    pinned_err = u.grip_error.clone()
                if k > release:
                    worst = torch.maximum(worst, u.grip_error)
            for i, c in enumerate(candidates):
                q = {n: round(v, 2) for n, v in zip(fnames, settled[i].tolist())}
                print(f"SWEEP offset={c} pinned_err={pinned_err[i].item():.4f} max_slip_2s={worst[i].item():.4f} final_err={u.grip_error[i].item():.4f}")
                print(f"   q={q}")
        else:
            # close every finger joint towards its upper limit (flexion) and hold
            close = zero.clone()
            finger_ids, _ = u.robot.find_joints(args.close)
            close[:, finger_ids] = 1.0
            fid, fnames = u.robot.find_joints("(index|middle|ring|thumb)_joint_.*")
            print("finger joint upper limits:", dict(zip(fnames, [round(x, 2) for x in u.dof_upper[0, fid].tolist()])))
            for k in range(int(cfg.episode_length_s / u.step_dt) - 2):
                env.step(close)
                if k == int(cfg.grace_period_s / u.step_dt) - 2:
                    q = u.joint_pos[0, fid]
                    print("SETTLED_FINGER_Q", {n: round(v, 3) for n, v in zip(fnames, q.tolist())})
                if args.trace and 40 <= k <= 80 and k % 4 == 0:
                    print(
                        f"  k={k} err={u.grip_error[0].item():.4f} rel={[round(x, 3) for x in u.racket_rel_pos[0].tolist()]}"
                        f" q={[round(x, 2) for x in u.joint_pos[0, fid].tolist()]}"
                    )
                if k % 12 == 0:
                    print(f"t={k * u.step_dt:4.2f}s grip error mean {u.grip_error.mean().item():.4f} max {u.grip_error.max().item():.4f}")

        env.close()


if __name__ == "__main__":
    main()

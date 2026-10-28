"""Two robots rallying: run the trained wheeled policy on both sides of the net and record the rallies.

Each robot plays in its own court (env), and courts are paired: env ``i`` (robot A) with env ``i + pairs`` (robot B).
Neither robot touches the shuttle while it crosses the net, so when a return crosses the net plane its state is
mirrored through the net (a half turn about the vertical axis at the court centre) into the partner's court, where it
continues its flight under the same drag. The receiver's intercept point is predicted from that flight, as the
launcher computed it in training. The robot without the shuttle sees a shadow of it, mirrored into its own court each
step (so its opponent's shot comes towards it from the far side, like a launcher shot), and its racket head is
commanded home, which drives its base back to the centre of its court.

A rally starts with a launcher serve to robot A and ends when the shuttle lands or leaves the court, a racket drops,
or a shot runs out of time. Pair 0 is recorded for ``scripts/render_showcase.py`` (robot B's poses mirrored onto the
far side); all pairs feed the rally-length statistics.

Usage:
    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 OMNI_KIT_ACCEPT_EULA=YES \
        ../IsaacLab/.venv/bin/python scripts/record_rally.py --checkpoint ~/robotics/wheeled_coverage.pt \
        --out rally.npz
"""

import argparse
import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Badminton-MobileCoverage-KukaAllegro")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--pairs", type=int, default=64, help="paired courts simulated in parallel")
parser.add_argument("--rallies", type=int, default=12, help="rallies recorded from pair 0")
parser.add_argument("--stat_rallies", type=int, default=1000, help="rallies over all pairs for the statistics")
parser.add_argument("--serve_difficulty", type=float, default=0.3, help="curriculum difficulty of the launcher serve")
parser.add_argument("--max_shots", type=int, default=40, help="a rally stops after this many returns")
parser.add_argument("--shot_timeout", type=float, default=5.0, help="seconds a shot may stay on one side")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--out", required=True)
parser.add_argument("--bank_out", default=None, help="also save the returns that land in court as a return bank")
parser.add_argument("--bank_size", type=int, default=0, help="keep running until the bank has this many returns")
parser.add_argument("--debug", type=int, default=0, help="print pair 0 state for this many steps")
add_launcher_args(parser)
args = parser.parse_args()

END_REASONS = ("miss", "net", "out", "drop", "timeout", "max")


def main():
    import gymnasium as gym
    import numpy as np
    import torch

    import badminton_rl.tasks  # noqa: F401
    from badminton_rl.tasks.badminton_env import GRAVITY, predict_landing
    from badminton_rl.tasks.badminton_env_cfg import NET_HEIGHT, NET_X, SHUTTLE_MASS, SHUTTLE_RADIUS
    from isaaclab.utils.math import sample_uniform
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
    P = args.pairs
    env_cfg.scene.num_envs = 2 * P
    env_cfg.seed = args.seed
    env_cfg.curriculum_init_difficulty = args.serve_difficulty
    env_cfg.curriculum_step = 0.0  # frozen
    args.device = env_cfg.sim.device
    court_back = NET_X + 6.7  # baseline of the robot's half [m]
    singles_y = 2.59

    with launch_simulation(env_cfg, args):
        from pxr import UsdPhysics  # only after Kit has loaded its USD libraries

        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped
        cfg, dev, n = u.cfg, u.device, u.num_envs
        origins = u.scene.env_origins
        partner = (torch.arange(n, device=dev) + P) % n
        grace_steps = int(np.ceil(cfg.grace_period_s / u.step_dt))

        # -- the env plays one shot per episode: take over termination and the base's idle behaviour --
        active = torch.zeros(n, dtype=torch.bool, device=dev)  # this robot's court has the shuttle in play
        orig_get_dones = u._get_dones

        def get_dones():
            orig_get_dones()  # still detects hits, landings, drops and out-of-court shuttles
            none = torch.zeros(n, dtype=torch.bool, device=dev)
            return none, none.clone()

        u._get_dones = get_dones

        def mirror(p, is_point=True):
            """Half turn about the vertical axis through the net centre: one court's frame to the other's."""
            out = p.clone()
            out[:, 0] = (2 * NET_X - p[:, 0]) if is_point else -p[:, 0]
            out[:, 1] = -p[:, 1]
            return out

        z_home = float(u.nominal_head_pos[2])

        def predict_intercept(p, v, c):
            """Where the receiver should meet the shuttle: the last point of its flight inside the intercept area that
            is still above the arm's home height (less a margin), which leaves the most time to get there; else the
            in-area point nearest that height."""
            g = torch.tensor([0.0, 0.0, -GRAVITY], device=dev)
            z_min = z_home - 0.1
            best = torch.zeros_like(p)
            found = torch.zeros(len(p), dtype=torch.bool, device=dev)
            fallback = p.clone()
            fallback_err = torch.full((len(p),), float("inf"), device=dev)
            c = c.unsqueeze(-1)
            for _ in range(300):
                v = v + (g - c * v.norm(dim=-1, keepdim=True) * v) * 0.01
                p = p + v * 0.01
                in_area = (
                    (p[:, 0] > cfg.intercept_x_range[0]) & (p[:, 0] < cfg.intercept_x_range[1])
                    & (p[:, 1].abs() < cfg.intercept_y_range[1]) & (p[:, 2] > SHUTTLE_RADIUS)
                )
                take = in_area & (p[:, 2] >= z_min)
                best[take] = p[take]
                found |= take
                err = (p[:, 2] - z_min).abs()
                closer = in_area & (err < fallback_err)
                fallback[closer] = p[closer]
                fallback_err[closer] = err[closer]
            best[~found] = fallback[~found]
            best[:, 0] = best[:, 0].clamp(*cfg.intercept_x_range)
            best[:, 1] = best[:, 1].clamp(*cfg.intercept_y_range)
            return best

        def set_waiting(ids):
            # racket head commanded to its home point, which drives the base back to the centre of the court; the
            # court's shuttle becomes a shadow of the one in play (see shadow_shuttles)
            u.shuttle_active[ids] = False  # no drag on the shadow and no hit detection
            u.hit[ids] = False
            u.cmd[ids] = u.nominal_head_pos
            active[ids] = False
            aimed[ids] = False

        def aim(ids):
            """Give waiting robots their intercept, landing target and shot clock as soon as the opponent hits (from
            the shadow's flight), as the launcher gave them at the start of a training episode."""
            src = partner[ids]
            pos, vel = mirror(u.shuttle_pos[src]), mirror(u.shuttle_vel[src], is_point=False)
            u.cmd[ids] = predict_intercept(pos, vel, u.drag_k[src] / SHUTTLE_MASS)
            m = len(ids)
            u.land_target[ids, 0] = sample_uniform(*cfg.target_x_range, (m,), device=dev)
            u.land_target[ids, 1] = sample_uniform(*cfg.target_y_range, (m,), device=dev)
            u.episode_length_buf[ids] = grace_steps  # shot clock as at the end of the grip grace period
            aimed[ids] = True
            # return bank: the shot just off the opponent's racket, as the receiver sees it; kept if it lands in court
            bank_rows[ids] = bank_count[0] + torch.arange(m, device=dev)
            bank_count[0] += m
            bank["pos"].append(pos.cpu())
            bank["vel"].append(vel.cpu())
            bank["drag_k"].append(u.drag_k[src].cpu())
            bank["cmd"].append(u.cmd[ids].cpu())

        def shadow_shuttles():
            """Show each waiting robot the shuttle in play, mirrored into its court: it sees its opponent's shot come
            towards it from the far side, as the launcher's shots did in training."""
            w = (~active).nonzero(as_tuple=False).squeeze(-1)
            if len(w) > 0:
                src = partner[w]
                pos, vel = mirror(u.prev_shuttle_pos[src]), mirror(u.prev_shuttle_vel[src], is_point=False)
                u._write_shuttle(w, pos + origins[w], vel)

        def start_rallies(pair_ids):
            ids = torch.cat((pair_ids, pair_ids + P))
            u._reset_idx(ids)  # both robots home, a launcher serve in each court
            u.scene.write_data_to_sim()
            u.sim.forward()
            set_waiting(pair_ids + P)  # robot B waits, robot A receives the serve
            active[pair_ids] = True
            shots[pair_ids] = 0
            shot_steps[ids] = 0
            shadow_shuttles()

        def transfer(src):
            """Hand the shuttle that just crossed the net in court ``src`` to the partner court."""
            dst = partner[src]
            pos, vel = mirror(u.shuttle_pos[src]), mirror(u.shuttle_vel[src], is_point=False)
            c = u.drag_k[src] / SHUTTLE_MASS
            land, _ = predict_landing(pos, vel, c)
            in_court = (land[:, 0] > NET_X) & (land[:, 0] < court_back) & (land[:, 1].abs() < singles_y)
            ok = in_court & aimed[dst]
            bank_ok.extend(bank_rows[dst][ok].tolist())
            u._write_shuttle(dst, pos + origins[dst], vel)
            u.shuttle_active[dst] = True
            u.drag_k[dst] = u.drag_k[src]
            u.hit[dst] = False
            u.cleared_net[dst] = False
            u.landed_in_target[dst] = False
            late = dst[~aimed[dst]]  # aimed when the opponent hit, unless that went unnoticed
            if len(late) > 0:
                aim(late)
            set_waiting(src)
            active[dst] = True
            shot_steps[dst] = 0
            return in_court

        # -- recording (pair 0) --
        stage = u.sim.stage
        robot_root = "/World/envs/env_0/Robot"
        body_names = list(u.robot.body_names)
        body_paths = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(robot_root + "/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if prim.GetName() in body_names:
                    body_paths[prim.GetName()] = path[len(robot_root) :]
        assert all(b in body_paths for b in body_names), "no prim for some bodies"
        keys = ("body_pos", "body_quat", "racket_pos", "racket_quat", "body_pos_b", "body_quat_b", "racket_pos_b",
                "racket_quat_b", "shuttle_pos", "shuttle_vel")
        frames = {k: [] for k in keys}
        episode_of_frame = []
        a, b = 0, P

        def snapshot(rally):
            for suffix, e in (("", a), ("_b", b)):
                frames["body_pos" + suffix].append((u.robot.data.body_pos_w.torch[e] - origins[e]).cpu().numpy())
                frames["body_quat" + suffix].append(u.robot.data.body_quat_w.torch[e].cpu().numpy())
                frames["racket_pos" + suffix].append((u.racket.data.root_pos_w.torch[e] - origins[e]).cpu().numpy())
                frames["racket_quat" + suffix].append(u.racket.data.root_quat_w.torch[e].cpu().numpy())
            # the shuttle as the env last wrote or stepped it (fresh after a transfer), in robot A's court frame
            e = a if bool(active[a]) else b
            pos, vel = u.prev_shuttle_pos[e : e + 1], u.prev_shuttle_vel[e : e + 1]
            if e == b:
                pos, vel = mirror(pos), mirror(vel, is_point=False)
            frames["shuttle_pos"].append(pos[0].cpu().numpy())
            frames["shuttle_vel"].append(vel[0].cpu().numpy())
            episode_of_frame.append(rally)

        # -- rally loop --
        bank = {k: [] for k in ("pos", "vel", "drag_k", "cmd")}
        bank_rows = torch.zeros(n, dtype=torch.long, device=dev)  # each receiver's pending bank row
        bank_ok = []
        bank_count = [0]
        aimed = torch.zeros(n, dtype=torch.bool, device=dev)
        shots = torch.zeros(P, dtype=torch.long, device=dev)
        shot_steps = torch.zeros(n, dtype=torch.long, device=dev)
        lengths, reasons, recorded = [], [], []
        in_court_total = transfers_total = 0
        timeout_steps = int(args.shot_timeout / u.step_dt)
        rally0 = 0

        with torch.inference_mode():
            env.reset()
            start_rallies(torch.arange(P, device=dev))
            obs = env.get_observations()
            last_x = u.prev_shuttle_pos[:, 0].clone()
            snapshot(rally0)
            while rally0 < args.rallies or len(lengths) < args.stat_rallies or len(bank_ok) < args.bank_size:
                obs, _, _, _ = env.step(policy(obs))
                shot_steps += 1
                if args.debug and u.common_step_counter < args.debug and bool(active[b] | aimed[b]):
                    r = lambda t: [round(v, 2) for v in t.tolist()]  # noqa: E731
                    print(f"DBG t={u.common_step_counter} B buf={int(u.episode_length_buf[b])} hit={bool(u.hit[b])} "
                          f"shuttle={r(u.shuttle_pos[b])} vel={r(u.shuttle_vel[b])} head={r(u.head_pos[b])} "
                          f"cmd={r(u.cmd[b])} base={r(u.base_pos[b])} grip={float(u.grip_error[b]):.3f}", flush=True)
                x = u.shuttle_pos[:, 0]
                crossed_plane = active & u.hit & (last_x >= NET_X) & (x < NET_X)
                # over the net tape; a shot passing under the (hanging) net is a fault
                crossed = crossed_plane & (u.shuttle_pos[:, 2] > NET_HEIGHT + SHUTTLE_RADIUS)
                under_net = crossed_plane & ~crossed
                hitter_fault = active & u.hit & ~crossed
                landed = active & u._landed
                out = active & u._out & ~crossed
                timed_out = active & (shot_steps > timeout_steps)
                ended = landed | out | under_net | timed_out | u._dropped
                pair_end = ended[:P] | ended[P:]
                pair_end |= shots >= args.max_shots

                # the opponent just hit: the waiting robot reads the shot
                read = ~active & ~aimed & u.hit[partner] & active[partner]
                if read.any():
                    aim(read.nonzero(as_tuple=False).squeeze(-1))

                # returns that crossed the net in a rally that goes on
                go = crossed & ~torch.cat((pair_end, pair_end))
                if go.any():
                    src = go.nonzero(as_tuple=False).squeeze(-1)
                    in_court = transfer(src)
                    shots[src % P] += 1
                    in_court_total += int(in_court.sum())
                    transfers_total += len(src)

                if pair_end.any():
                    done = pair_end.nonzero(as_tuple=False).squeeze(-1)
                    for p in done.tolist():
                        both = [p, p + P]
                        if bool(u._dropped[both].any()):
                            why = "drop"
                        elif int(shots[p]) >= args.max_shots:
                            why = "max"
                        elif bool(under_net[both].any()):
                            why = "net"
                        elif bool(timed_out[both].any()):
                            why = "timeout"
                        elif bool((landed | out)[both].any()):
                            # a shot that fell on the hitter's own side or went wide is the hitter's fault
                            why = "net" if bool(hitter_fault[both].any() and landed[both].any()) else (
                                "out" if bool(hitter_fault[both].any()) else "miss")
                        else:
                            why = "miss"
                        lengths.append(int(shots[p]))
                        reasons.append(why)
                        if p == 0:
                            recorded.append((int(shots[p]), why))
                            rally0 += 1
                    start_rallies(done)

                if go.any() or pair_end.any() or read.any():
                    obs = env.get_observations()  # the receivers' new intercepts and shot clocks
                last_x = u.prev_shuttle_pos[:, 0].clone()
                shadow_shuttles()
                if rally0 < args.rallies:
                    snapshot(rally0)

        def mirror_np(p):
            out = p.copy()
            out[..., 0] = 2 * NET_X - p[..., 0]
            out[..., 1] = -p[..., 1]
            return out

        def mirror_quat_np(q):
            # half turn about z applied to xyzw quaternions: (x, y, z, w) -> (-y, x, w, -z)
            return np.stack((-q[..., 1], q[..., 0], q[..., 3], -q[..., 2]), axis=-1)

        rec = {k: np.array(v) for k, v in frames.items()}
        for k in ("body_pos_b", "racket_pos_b"):
            rec[k] = mirror_np(rec[k])
        for k in ("body_quat_b", "racket_quat_b"):
            rec[k] = mirror_quat_np(rec[k])
        rec_len = np.array([r[0] for r in recorded][: args.rallies])
        np.savez_compressed(
            args.out,
            dt=u.step_dt,
            body_names=np.array(body_names),
            body_paths=np.array([body_paths[name] for name in body_names]),
            episode=np.array(episode_of_frame),
            land_target=np.zeros((len(rec_len), 2)),
            success=rec_len >= 2,
            hit=rec_len >= 1,
            cleared_net=rec_len >= 1,
            rally_length=rec_len,
            rally_end=np.array([r[1] for r in recorded][: args.rallies]),
            stat_lengths=np.array(lengths),
            stat_reasons=np.array(reasons),
            robot_usd=u.cfg.scene.robot.spawn.usd_path,
            robot_pos=np.array(u.cfg.scene.robot.init_state.pos),
            **rec,
        )
        if args.bank_out:
            flat = {k: torch.cat(v).numpy() for k, v in bank.items()}
            keep = np.array(sorted(set(bank_ok)), dtype=np.int64)
            np.savez_compressed(args.bank_out, **{k: v[keep] for k, v in flat.items()})
            print(f"RALLY return bank: {len(keep)} in-court returns (of {len(flat['pos'])} read) -> {args.bank_out}")
        L = np.array(lengths)
        print(f"RALLY home racket-head height {z_home:.2f} m")
        print(f"RALLY rallies={len(L)} mean_returns={L.mean():.2f} median={np.median(L):.0f} max={L.max()}")
        for k in (1, 2, 3, 5, 10, 20):
            print(f"RALLY P(returns >= {k}) = {(L >= k).mean():.3f}")
        print("RALLY end reasons: " + ", ".join(f"{r}={reasons.count(r) / len(reasons):.3f}" for r in END_REASONS))
        print(f"RALLY returns landing in the opponent's singles court: {in_court_total / max(transfers_total, 1):.3f}")
        print(f"RALLY pair 0 rallies (returns, end): {recorded[: args.rallies]}")
        print(f"RALLY frames={len(episode_of_frame)} -> {args.out}")
        env.close()


if __name__ == "__main__":
    main()

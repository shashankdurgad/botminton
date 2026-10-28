"""Direct-workflow badminton environment: a Kuka iiwa arm with an Allegro hand holds a racket and returns a shuttle.

One class serves all three curriculum stages (``cfg.stage``):

* ``grasp``  - close the fingers on the racket during a short grace period, then keep it while swinging the
  racket head to commanded points.
* ``hit``    - a shuttle is launched towards the robot; reward contact and a strong, rising return.
* ``target`` - as ``hit``, plus clearing the net and landing near a randomized target on the far court.
* ``mobile`` - as ``target``, with the arm on a velocity-controlled omnidirectional base and shots spread
  over the half court.

With ``cfg.mobile_base`` the arm rides the base in every stage (the from-scratch wheeled curriculum), and the
observation and action layout is the mobile stage's throughout.

The observation layout is identical across stages so a policy can warm-start from the previous stage; later
stages only append entries (and actions) at the end.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    sample_uniform,
    scale_transform,
    subtract_frame_transforms,
)

from .badminton_env_cfg import NET_HEIGHT, NET_X, SHUTTLE_MASS, SHUTTLE_RADIUS

if TYPE_CHECKING:
    from .badminton_env_cfg import BadmintonEnvCfg

GRAVITY = 9.81


class BadmintonEnv(DirectRLEnv):
    """Kuka-Allegro badminton, see the module docstring."""

    cfg: BadmintonEnvCfg

    def __init__(self, cfg: BadmintonEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.racket = self.scene["racket"]
        self.shuttle = self.scene["shuttle"]
        dev = self.device
        n = self.num_envs

        # -- joints (the mobile base appends three base-velocity actions) --
        self.mobile = cfg.mobile_base
        self.num_dofs = self.robot.num_joints
        if self.num_dofs != cfg.action_space - (3 if self.mobile else 0):
            raise ValueError(f"Expected {cfg.action_space} actions on the Kuka-Allegro, found {self.num_dofs} joints.")
        arm_ids, _ = self.robot.find_joints("iiwa7_joint_.*")
        self.action_scale = torch.full((self.num_dofs,), cfg.finger_action_scale, device=dev)
        self.action_scale[arm_ids] = cfg.arm_action_scale
        limits = self.robot.data.joint_limits.torch.to(dev)
        self.dof_lower = limits[..., 0]
        self.dof_upper = limits[..., 1]
        self.targets = torch.zeros((n, self.num_dofs), device=dev)
        self.actions = torch.zeros((n, cfg.action_space), device=dev)
        self.prev_actions = torch.zeros_like(self.actions)

        closing_ids, _ = self.robot.find_joints(cfg.closing_joint_expr)
        self.closing_ids = torch.tensor(closing_ids, dtype=torch.long, device=dev)

        palm_ids, _ = self.robot.find_bodies(cfg.palm_body_name)
        self.palm_id = palm_ids[0]

        # -- grip geometry --
        self.grasp_pos = torch.tensor(cfg.grasp_offset_pos, device=dev).repeat(n, 1)
        self.grasp_quat = torch.tensor(cfg.grasp_offset_quat, device=dev).repeat(n, 1)
        self.head_offset = torch.tensor(cfg.racket_head_offset, device=dev).repeat(n, 1)

        # -- shuttle --
        drag_k = SHUTTLE_MASS * GRAVITY / cfg.shuttle_terminal_velocity**2
        self.drag_k_nominal = drag_k
        self.drag_k = torch.full((n,), drag_k, device=dev)
        self.shuttle_active = torch.zeros(n, dtype=torch.bool, device=dev)

        # -- task state --
        self.cmd = torch.zeros((n, 3), device=dev)  # racket-head goal: swing command (grasp) or intercept point
        self.land_target = torch.zeros((n, 2), device=dev)  # far-court landing target (target stage)
        self.hit = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cleared_net = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_shuttle_pos = torch.zeros((n, 3), device=dev)
        self.prev_shuttle_vel = torch.zeros((n, 3), device=dev)
        self.held_all_episode = torch.ones(n, dtype=torch.bool, device=dev)
        self.landed_in_target = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._landed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._out = torch.zeros(n, dtype=torch.bool, device=dev)
        self.difficulty = float(cfg.curriculum_init_difficulty)
        self._window_successes = 0.0
        self._window_episodes = 0

        # -- mobile base: planar pose and world-frame velocity (vx, vy, yaw rate) --
        self.base_pos = torch.zeros((n, 2), device=dev)
        self.base_yaw = torch.zeros(n, device=dev)
        self.base_vel = torch.zeros((n, 3), device=dev)
        self.base_cmd = torch.zeros((n, 3), device=dev)
        self.base_cmd_filtered = torch.zeros((n, 3), device=dev)

        # -- return bank: recorded rally returns (pos, vel, drag_k, intercept), launched after the grace period --
        self.bank = None
        if cfg.return_bank_path and cfg.return_bank_frac > 0.0:
            import numpy as np

            data = np.load(os.path.expanduser(cfg.return_bank_path))
            self.bank = {k: torch.as_tensor(data[k], dtype=torch.float32, device=dev) for k in ("pos", "vel", "drag_k", "cmd")}
            print(f"[badminton] return bank: {len(self.bank['pos'])} returns from {cfg.return_bank_path}")
        self.bank_feed = torch.zeros(n, dtype=torch.bool, device=dev)  # this episode's shot comes from the bank
        self.bank_pending = torch.zeros(n, dtype=torch.bool, device=dev)  # ... and is not launched yet
        self.bank_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.bank_launch_step = torch.zeros(n, dtype=torch.long, device=dev)
        self._grace_steps = int(-(-cfg.grace_period_s // self.step_dt))

        self._write_targets = self.robot.actuators.target_command.set_position_index
        self._all_ids = torch.arange(n, device=dev)
        self._compute_intermediate_values()

        # racket-head position with the arm at its default pose (the scene was updated at the
        # initial state during construction): the centre of the intercept and swing-command boxes
        grip_pos, grip_quat = combine_frame_transforms(self.palm_pos, self.palm_quat, self.grasp_pos, self.grasp_quat)
        self.nominal_head_pos = (grip_pos + quat_apply(grip_quat, self.head_offset)).mean(dim=0)
        # the base starts at the env origin, so this is also the head's home offset in the base frame
        self.home_head_offset = self.nominal_head_pos[:2].clone()

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions = actions.clamp(-1.0, 1.0)
        joint_actions = self.actions[:, : self.num_dofs]
        self.targets = torch.clamp(self.targets + self.action_scale * joint_actions, self.dof_lower, self.dof_upper)
        if self.mobile:
            cfg = self.cfg
            base_actions = self.actions[:, self.num_dofs :]
            if cfg.base_scripted:
                # scripted drive to the intercept, the policy's base actions only correct it
                lin, wz = self._scripted_base_cmd()
                lin = lin + base_actions[:, :2] * cfg.base_residual_lin_vel
                wz = wz + base_actions[:, 2] * cfg.base_residual_ang_vel
            else:
                lin, wz = base_actions[:, :2] * cfg.base_max_lin_vel, base_actions[:, 2] * cfg.base_max_ang_vel
            # the speed limit is on the planar speed, not per axis
            lin = lin * (cfg.base_max_lin_vel / lin.norm(dim=-1, keepdim=True).clamp(min=cfg.base_max_lin_vel))
            wz = wz.clamp(-cfg.base_max_ang_vel, cfg.base_max_ang_vel)
            self.base_cmd = torch.cat((lin, wz.unsqueeze(-1)), dim=-1)
        if self.bank is not None:
            self._launch_bank_feeds()

    def _scripted_base_cmd(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Drive so the arm's home head position ends up under the intercept point, as fast as the limits allow.

        The speed follows the braking curve v = sqrt(2 a d), on the distance left after the command filter's lag,
        so the base arrives without overshoot; a proportional cap settles it at the goal. It holds still during the
        grasp grace period and after the hit.

        Returns:
            The planar velocity command [m/s], shape (num_envs, 2), and the yaw-rate command [rad/s], shape (num_envs,).
        """
        cfg = self.cfg
        err = self.cmd[:, :2] - self._home_head_xy()
        # hold still while the fingers close on the pinned racket: the pin zeroes its velocity, so a moving base
        # would leave it behind at release
        in_grace = self.episode_length_buf.float() * self.step_dt < cfg.grace_period_s
        err[self.hit | in_grace] = 0.0
        d = err.norm(dim=-1, keepdim=True)
        lag = self.base_vel[:, :2].norm(dim=-1, keepdim=True) * cfg.base_cmd_tau
        speed = torch.sqrt(2.0 * cfg.scripted_brake_acc * (d - lag).clamp(min=0.0))
        speed = torch.minimum(speed, cfg.scripted_gain * d)
        lin = err / d.clamp(min=1e-6) * speed
        wz = -cfg.scripted_yaw_gain * self.base_yaw  # keep facing the net
        return lin, wz

    def _apply_action(self) -> None:
        self._write_targets(value=self.targets)
        if self.mobile:
            self._drive_base(self.physics_dt)

        # pin the racket to the palm during the grace period so the fingers can close on it
        in_grace = self.episode_length_buf.float() * self.step_dt < self.cfg.grace_period_s
        grace_ids = in_grace.nonzero(as_tuple=False).squeeze(-1)
        if len(grace_ids) > 0:
            palm_pos = self.robot.data.body_pos_w.torch[grace_ids, self.palm_id]
            palm_quat = self.robot.data.body_quat_w.torch[grace_ids, self.palm_id]
            pos, quat = combine_frame_transforms(palm_pos, palm_quat, self.grasp_pos[grace_ids], self.grasp_quat[grace_ids])
            self.racket.write_root_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=grace_ids)
            self.racket.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((len(grace_ids), 6), device=self.device), env_ids=grace_ids
            )

        # quadratic aerodynamic drag on the shuttle: F = -k |v| v
        vel = self.shuttle.data.root_lin_vel_w.torch
        force = -(self.drag_k * self.shuttle_active.float()).unsqueeze(-1) * vel.norm(dim=-1, keepdim=True) * vel
        self.shuttle.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=force.unsqueeze(1), is_global=True
        )

    def _drive_base(self, dt: float) -> None:
        """Track the commanded base velocity under acceleration limits and move the arm's root with it."""
        cfg = self.cfg
        self.base_cmd_filtered += (self.base_cmd - self.base_cmd_filtered) * min(dt / cfg.base_cmd_tau, 1.0)
        dv = self.base_cmd_filtered[:, :2] - self.base_vel[:, :2]
        max_dv = cfg.base_max_lin_acc * dt
        dv = dv * (max_dv / dv.norm(dim=-1, keepdim=True).clamp(min=max_dv))
        self.base_vel[:, :2] += dv
        dw = (self.base_cmd_filtered[:, 2] - self.base_vel[:, 2]).clamp(-cfg.base_max_ang_acc * dt, cfg.base_max_ang_acc * dt)
        self.base_vel[:, 2] += dw
        self.base_pos += self.base_vel[:, :2] * dt
        self.base_yaw = torch.remainder(self.base_yaw + self.base_vel[:, 2] * dt + torch.pi, 2 * torch.pi) - torch.pi
        # court limits act like walls: clamp the position and stop the motion into them
        for axis, (lo, hi) in enumerate((cfg.base_bounds_x, cfg.base_bounds_y)):
            outside = (self.base_pos[:, axis] < lo) | (self.base_pos[:, axis] > hi)
            self.base_pos[:, axis] = self.base_pos[:, axis].clamp(lo, hi)
            self.base_vel[outside, axis] = 0.0
        self._write_base(self._all_ids, pose=self.cfg.base_teleport)

    def _home_head_xy(self) -> torch.Tensor:
        """Where the racket head would be, in the env frame, with the arm at its default pose on the current base."""
        c, s = torch.cos(self.base_yaw), torch.sin(self.base_yaw)
        hx, hy = self.home_head_offset
        return self.base_pos + torch.stack((c * hx - s * hy, s * hx + c * hy), dim=-1)

    def _write_base(self, env_ids: torch.Tensor, pose: bool = True) -> None:
        """Write the base velocity and, with ``pose``, teleport the root to the integrated base pose.

        Without the pose write the solver moves the arm, so the finger contacts carry the (separate) racket body
        along; teleporting every substep makes it slip in the hand.
        """
        n = len(env_ids)
        pos = torch.cat(
            (self.base_pos[env_ids], torch.full((n, 1), self.cfg.base_height, device=self.device)), dim=-1
        ) + self.scene.env_origins[env_ids]
        half = 0.5 * self.base_yaw[env_ids]
        zeros = torch.zeros_like(half)
        quat = torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), dim=-1)  # xyzw, yaw about +z
        vel = torch.stack(
            (self.base_vel[env_ids, 0], self.base_vel[env_ids, 1], zeros, zeros, zeros, self.base_vel[env_ids, 2]), dim=-1
        )
        if pose:
            self.robot.write_root_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)

    def _compute_intermediate_values(self) -> None:
        origins = self.scene.env_origins
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch
        self.palm_pos = self.robot.data.body_pos_w.torch[:, self.palm_id] - origins
        self.palm_quat = self.robot.data.body_quat_w.torch[:, self.palm_id]
        self.racket_pos = self.racket.data.root_pos_w.torch - origins
        self.racket_quat = self.racket.data.root_quat_w.torch
        self.racket_rel_pos, self.racket_rel_quat = subtract_frame_transforms(
            self.palm_pos, self.palm_quat, self.racket_pos, self.racket_quat
        )
        self.head_pos = self.racket_pos + quat_apply(self.racket_quat, self.head_offset)
        racket_lin_vel = self.racket.data.root_lin_vel_w.torch
        racket_ang_vel = self.racket.data.root_ang_vel_w.torch
        # velocity of the string-bed centre (root velocity is at the centre of mass, the lever arm is small)
        self.head_vel = racket_lin_vel + torch.cross(
            racket_ang_vel, quat_apply(self.racket_quat, self.head_offset), dim=-1
        )
        self.shuttle_pos = self.shuttle.data.root_pos_w.torch - origins
        self.shuttle_vel = self.shuttle.data.root_lin_vel_w.torch
        self.grip_error = (self.racket_rel_pos - self.grasp_pos).norm(dim=-1)

    def _to_base_frame(self, p: torch.Tensor, is_point: bool = True) -> torch.Tensor:
        """Express env-frame points (or, with ``is_point`` False, velocities) in the mobile base's frame."""
        c, s = torch.cos(self.base_yaw), torch.sin(self.base_yaw)
        xy = p[:, :2] - (self.base_pos if is_point else self.base_vel[:, :2])
        out = torch.stack((c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]), dim=-1)
        return torch.cat((out, p[:, 2:]), dim=-1)

    def _get_observations(self) -> dict:
        t = self.episode_length_buf.float().unsqueeze(-1) * self.step_dt / self.max_episode_length_s
        palm, head, shuttle, cmd = self.palm_pos, self.head_pos, self.shuttle_pos, self.cmd
        head_vel, shuttle_vel, land_target = self.head_vel, self.shuttle_vel, self.land_target
        if self.mobile and self.cfg.base_frame_obs:
            # the arm sees the world from its base: with the base still at the origin (the stationary stages)
            # these equal the env-frame values, so the stationary skill carries over wherever the base parks
            palm, head, shuttle, cmd = (self._to_base_frame(p) for p in (palm, head, shuttle, cmd))
            head_vel, shuttle_vel = (self._to_base_frame(v, is_point=False) for v in (head_vel, shuttle_vel))
            land_target = self._to_base_frame(land_target)
            if self.cfg.clamp_cmd_obs:
                # the arm learned intercepts inside its reach box; one metres away (the base still on its way)
                # makes it lunge and fling the racket, so it sees the nearest reachable point until the base arrives
                half = torch.tensor(self.cfg.intercept_half_extent_max, device=self.device)
                cmd = self.nominal_head_pos + (cmd - self.nominal_head_pos).clamp(-half, half)
        obs = torch.cat(
            (
                scale_transform(self.joint_pos, self.dof_lower, self.dof_upper),
                0.1 * self.joint_vel,
                palm,
                self.racket_rel_pos,
                self.racket_rel_quat,
                head,
                0.2 * head_vel,
                shuttle,
                0.2 * shuttle_vel,
                shuttle - head,
                cmd,
                self.hit.float().unsqueeze(-1),
                t,
                self.actions[:, : self.num_dofs],
            ),
            dim=-1,
        )
        if self.cfg.stage in ("target", "mobile"):
            # appended last so a hit-stage checkpoint can be widened with zero input weights
            obs = torch.cat((obs, land_target), dim=-1)
        elif self.mobile:
            # the wheeled stages keep the slot (unused before the target stage) so the layout never changes
            obs = torch.cat((obs, torch.zeros_like(land_target)), dim=-1)
        if self.mobile:
            # intercept relative to the arm's home head position, in the base frame: where to drive
            c, s_ = torch.cos(self.base_yaw), torch.sin(self.base_yaw)
            dx, dy = (self.cmd[:, :2] - self._home_head_xy()).unbind(-1)
            obs = torch.cat(
                (
                    obs,
                    self.actions[:, self.num_dofs :],
                    self.base_pos,
                    c.unsqueeze(-1),
                    s_.unsqueeze(-1),
                    0.3 * self.base_vel,
                    torch.stack((c * dx + s_ * dy, -s_ * dx + c * dy), dim=-1),
                ),
                dim=-1,
            )
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        after_grace = self.episode_length_buf.float() * self.step_dt >= self.cfg.grace_period_s
        self._dropped = after_grace & (self.grip_error > self.cfg.drop_distance)
        self.held_all_episode &= ~self._dropped

        if self.cfg.stage == "grasp":
            self._landed = torch.zeros_like(self._dropped)
            self._out = torch.zeros_like(self._dropped)
        else:
            self._update_shuttle_events()
            self._landed = self.shuttle_active & (self.shuttle_pos[:, 2] < SHUTTLE_RADIUS + 0.01)
            sp = self.shuttle_pos
            self._out = (sp[:, 0] > self.cfg.shuttle_out_x_max) | (sp[:, 0] < -9.5) | (sp[:, 1].abs() > 4.5)

        terminated = self._dropped | self._landed | self._out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        ended = (terminated | time_out).float().sum().clamp(min=1.0)
        log = self.extras.setdefault("log", {})
        log["Termination/dropped"] = self._dropped.float().sum() / ended
        log["Termination/landed"] = self._landed.float().sum() / ended
        log["Termination/out_of_bounds"] = self._out.float().sum() / ended
        log["Termination/time_out"] = (time_out & ~terminated).float().sum() / ended
        return terminated, time_out

    def _update_shuttle_events(self) -> None:
        """Detect the racket hit and the net crossing from the shuttle trajectory."""
        vel, prev_vel = self.shuttle_vel, self.prev_shuttle_vel
        near_head = (self.shuttle_pos - self.head_pos).norm(dim=-1) < 0.35
        # the incoming shuttle travels towards +x; a return reverses it
        self._new_hit = (~self.hit) & self.shuttle_active & near_head & (prev_vel[:, 0] > 0.0) & (vel[:, 0] < -0.5)
        self.hit |= self._new_hit

        crossed = self.hit & (self.prev_shuttle_pos[:, 0] >= NET_X) & (self.shuttle_pos[:, 0] < NET_X)
        self.cleared_net |= crossed & (self.shuttle_pos[:, 2] > NET_HEIGHT + SHUTTLE_RADIUS)

        self.prev_shuttle_pos[:] = self.shuttle_pos
        self.prev_shuttle_vel[:] = vel

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        after_grace = self.episode_length_buf.float() * self.step_dt >= cfg.grace_period_s
        rew = torch.zeros(self.num_envs, device=self.device)
        log = self.extras.setdefault("log", {})

        # grip: keep the handle where it was placed
        hold = torch.exp(-self.grip_error / 0.03) * after_grace.float()
        rew += cfg.rew_hold * hold
        rew += cfg.rew_drop * self._dropped.float()

        if cfg.stage == "grasp":
            self._resample_swing_command()
            track = torch.exp(-(self.head_pos - self.cmd).norm(dim=-1) / 0.15) * after_grace.float()
            rew += cfg.rew_swing_track * track
            log["Rewards/swing_track"] = track.mean()
        else:
            d = (self.shuttle_pos - self.head_pos).norm(dim=-1)
            approach = torch.exp(-d / 0.3) * (~self.hit & (self.shuttle_pos[:, 0] < self.head_pos[:, 0])).float()
            rew += cfg.rew_approach * approach
            # while the shuttle is still incoming, keep the grasp-stage skill: racket head towards the intercept
            # point. It must stop once the shuttle is past the racket, or blocking and holding still outscores a hit.
            incoming = self.shuttle_pos[:, 0] < self.head_pos[:, 0]
            track = torch.exp(-(self.head_pos - self.cmd).norm(dim=-1) / 0.15)
            track *= (after_grace & incoming & ~self.hit).float()
            rew += cfg.rew_intercept_track * track
            log["Rewards/intercept_track"] = track.mean()
            out_speed = (-self.shuttle_vel[:, 0]).clamp(0.0, 12.0) / 12.0
            up_speed = self.shuttle_vel[:, 2].clamp(0.0, 8.0) / 8.0
            hit_rew = cfg.rew_hit + cfg.rew_hit_out_speed * out_speed + cfg.rew_hit_up_speed * up_speed
            rew += hit_rew * self._new_hit.float()
            rew += cfg.rew_miss * (self._landed & ~self.hit).float()
            log["Rewards/approach"] = approach.mean()

            if cfg.stage in ("target", "mobile"):
                # dense aiming signal at contact: the actual landing comes 1-2 s later, too late to credit the swing
                hit_ids = self._new_hit.nonzero(as_tuple=True)[0]
                if len(hit_ids) > 0:
                    pred_xy, pred_clears = predict_landing(
                        self.shuttle_pos[hit_ids], self.shuttle_vel[hit_ids], self.drag_k[hit_ids] / SHUTTLE_MASS
                    )
                    pred_err = (pred_xy - self.land_target[hit_ids]).norm(dim=-1)
                    rew[hit_ids] += cfg.rew_aim * torch.exp(-(pred_err / cfg.target_sigma) ** 2) * pred_clears.float()
                    log["Metrics/pred_land_error"] = pred_err.mean()
                landed_after_hit = self._landed & self.hit & self.cleared_net
                land_err = (self.shuttle_pos[:, :2] - self.land_target).norm(dim=-1)
                rew += cfg.rew_clear_net * (self._landed & self.cleared_net).float()
                rew += cfg.rew_land_target * torch.exp(-(land_err / cfg.target_sigma) ** 2) * landed_after_hit.float()
                self.landed_in_target |= landed_after_hit & (land_err < cfg.target_success_radius)

        if self.mobile:
            # base placed so the arm's home head position sits under the intercept point, while it is incoming
            incoming = (self.shuttle_pos[:, 0] < self.head_pos[:, 0]) & ~self.hit
            d = (self._home_head_xy() - self.cmd[:, :2]).norm(dim=-1)
            base_track = 0.5 * (torch.exp(-d / 2.0) + torch.exp(-d / 0.3)) * incoming.float()
            rew += cfg.rew_base_track * base_track
            rew += cfg.rew_base_effort * self.base_vel.pow(2).sum(dim=-1)
            log["Rewards/base_track"] = base_track.mean()
            log["Metrics/base_speed"] = self.base_vel[:, :2].norm(dim=-1).mean()

        rew += cfg.rew_action_rate * (self.actions - self.prev_actions).pow(2).sum(dim=-1)
        rew += cfg.rew_joint_vel * self.joint_vel.pow(2).sum(dim=-1)
        log["Rewards/hold"] = hold.mean()
        return rew

    # ------------------------------------------------------------------
    # resets
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self._all_ids
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._log_episode_metrics(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        if self.mobile:
            # base back at its home spot (the env origin), at rest
            self.base_pos[env_ids] = 0.0
            self.base_yaw[env_ids] = 0.0
            self.base_vel[env_ids] = 0.0
            self.base_cmd[env_ids] = 0.0
            self.base_cmd_filtered[env_ids] = 0.0
            self._write_base(env_ids)

        # robot: default arm pose (keeps the racket spawn consistent), small finger noise
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        finger_noise = 0.05 * sample_uniform(-1.0, 1.0, joint_pos.shape, device=self.device)
        finger_noise[:, self.action_scale == self.cfg.arm_action_scale] = 0.0
        joint_pos = torch.clamp(joint_pos + finger_noise, self.dof_lower[env_ids], self.dof_upper[env_ids])
        joint_vel = torch.zeros_like(joint_pos)
        # the hand starts open with its finger targets at the flexion limit, so it closes on the racket by default
        targets = joint_pos.clone()
        targets[:, self.closing_ids] = self.dof_upper[env_ids][:, self.closing_ids]
        self.targets[env_ids] = targets
        self._write_targets(value=targets, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        # racket: placed from the current palm pose; the grace-period pin corrects it after the first substep
        palm_pos = self.robot.data.body_pos_w.torch[env_ids, self.palm_id]
        palm_quat = self.robot.data.body_quat_w.torch[env_ids, self.palm_id]
        pos, quat = combine_frame_transforms(palm_pos, palm_quat, self.grasp_pos[env_ids], self.grasp_quat[env_ids])
        self.racket.write_root_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
        self.racket.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=self.device), env_ids=env_ids)

        # task state
        self.hit[env_ids] = False
        self.cleared_net[env_ids] = False
        self.held_all_episode[env_ids] = True
        self.landed_in_target[env_ids] = False
        scale = 1.0 + self.cfg.drag_randomization * sample_uniform(-1.0, 1.0, (n,), device=self.device)
        self.drag_k[env_ids] = self.drag_k_nominal * scale

        if self.cfg.stage == "grasp":
            self.shuttle_active[env_ids] = False
            park = torch.tensor([-6.0, 0.0, SHUTTLE_RADIUS], device=self.device).repeat(n, 1) + origins
            self._write_shuttle(env_ids, park, torch.zeros((n, 3), device=self.device))
            self._sample_swing_command(env_ids)
        else:
            # the intercept point takes the swing command's slot: "head to cmd" now means "meet the shuttle"
            self.cmd[env_ids] = self._launch_shuttle(env_ids)
            if self.bank is not None:
                self._assign_bank_feeds(env_ids)
            if self.cfg.stage in ("target", "mobile"):
                self.land_target[env_ids, 0] = sample_uniform(*self.cfg.target_x_range, (n,), device=self.device)
                self.land_target[env_ids, 1] = sample_uniform(*self.cfg.target_y_range, (n,), device=self.device)

    def _write_shuttle(self, env_ids: torch.Tensor, pos_w: torch.Tensor, vel_w: torch.Tensor) -> None:
        n = len(env_ids)
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 3] = 1.0
        self.shuttle.write_root_pose_to_sim_index(root_pose=torch.cat((pos_w, quat), dim=-1), env_ids=env_ids)
        velocity = torch.cat((vel_w, torch.zeros((n, 3), device=self.device)), dim=-1)
        self.shuttle.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        self.prev_shuttle_pos[env_ids] = pos_w - self.scene.env_origins[env_ids]
        self.prev_shuttle_vel[env_ids] = vel_w

    def _launch_shuttle(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Launch the shuttle from the far side so that, with drag, it passes an intercept point near the racket.

        Returns:
            The intercept points in the env frame [m], shape (len(env_ids), 3).
        """
        cfg, n, dev = self.cfg, len(env_ids), self.device
        p0 = torch.stack(
            (
                sample_uniform(*cfg.launch_x_range, (n,), device=dev),
                sample_uniform(*cfg.launch_y_range, (n,), device=dev),
                sample_uniform(*cfg.launch_z_range, (n,), device=dev),
            ),
            dim=-1,
        )
        centre = self.nominal_head_pos
        half = torch.tensor(cfg.intercept_half_extent_max, device=dev)
        if cfg.stage != "mobile":
            # the mobile stage starts from the full box; earlier stages grow it with the curriculum
            half = (half * self.difficulty).clamp(min=cfg.intercept_half_extent_min)
        target = centre + half * sample_uniform(-1.0, 1.0, (n, 3), device=dev)
        flight_time = sample_uniform(*cfg.flight_time_range, (n,), device=dev)
        if self.mobile:
            # plus a court-wide offset that grows with the curriculum
            target[:, :2] = self._coverage_offset(target[:, :2])
            reach = (target[:, :2] - centre[:2]).norm(dim=-1)
            flight_time += cfg.flight_time_per_m * reach
        v0 = solve_launch_velocity(p0, target, flight_time, self.drag_k[env_ids] / SHUTTLE_MASS)
        self.shuttle_active[env_ids] = True
        self._write_shuttle(env_ids, p0 + self.scene.env_origins[env_ids], v0)
        return target

    def _assign_bank_feeds(self, env_ids: torch.Tensor) -> None:
        """Replace some launcher shots by bank returns. During the grip grace period the shuttle flies away from the
        robot, as its own previous return would in a rally (another bank shot mirrored into the hitter's frame, from
        0.15 s after contact so it clears the racket). After a random wait, in which the robot holds its ready
        position, the opponent's return appears just off the far racket. A parked shuttle instead is unlike anything
        in training and costs the grip. The intercept is given when the return appears, as a rally receiver reads it
        when the opponent hits."""
        pick = torch.rand(len(env_ids), device=self.device) < self.cfg.return_bank_frac
        self.bank_feed[env_ids] = pick
        self.bank_pending[env_ids] = pick
        ids = env_ids[pick]
        if len(ids) == 0:
            return
        idx = torch.randint(len(self.bank["pos"]), (len(ids),), device=self.device)
        self.bank_idx[ids] = idx
        delay = sample_uniform(*self.cfg.return_bank_delay_range, (len(ids),), device=self.device)
        self.bank_launch_step[ids] = self._grace_steps + (delay / self.step_dt).round().long()
        self.cmd[ids] = self.nominal_head_pos  # home until the opponent hits, as a waiting robot in a rally
        self.shuttle_active[ids] = False  # no drag, landing or hit detection on the outgoing decoy
        j = torch.randint(len(self.bank["pos"]), (len(ids),), device=self.device)
        pos, vel = self.bank["pos"][j].clone(), self.bank["vel"][j].clone()
        pos[:, 0] = 2 * NET_X - pos[:, 0]
        pos[:, 1] = -pos[:, 1]
        vel[:, :2] = -vel[:, :2]
        c = (self.bank["drag_k"][j] / SHUTTLE_MASS).unsqueeze(-1)
        g = torch.tensor([0.0, 0.0, -GRAVITY], device=self.device)
        for _ in range(15):
            vel = vel + (g - c * vel.norm(dim=-1, keepdim=True) * vel) * 0.01
            pos = pos + vel * 0.01
        self._write_shuttle(ids, pos + self.scene.env_origins[ids], vel)

    def _launch_bank_feeds(self) -> None:
        due = self.bank_pending & (self.episode_length_buf >= self.bank_launch_step)
        ids = due.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        idx = self.bank_idx[ids]
        self.cmd[ids] = self.bank["cmd"][idx]
        self.drag_k[ids] = self.bank["drag_k"][idx]
        self.shuttle_active[ids] = True
        self._write_shuttle(ids, self.bank["pos"][idx] + self.scene.env_origins[ids], self.bank["vel"][idx])
        self.bank_pending[ids] = False

    def _sample_swing_command(self, env_ids: torch.Tensor) -> None:
        centre = self.nominal_head_pos
        half = torch.tensor(self.cfg.swing_cmd_half_extent, device=self.device) * max(self.difficulty, 0.2)
        self.cmd[env_ids] = centre + half * sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        if self.mobile:
            # the base drives between the commands, so the grip learns to ride its accelerations
            self.cmd[env_ids, :2] = self._coverage_offset(self.cmd[env_ids, :2])

    def _coverage_offset(self, xy: torch.Tensor) -> torch.Tensor:
        """Shift points by a random court-coverage offset that grows with the difficulty, kept on the court."""
        cfg, n, dev = self.cfg, len(xy), self.device
        scale = self.difficulty * cfg.coverage_scale
        x = xy[:, 0] + scale * sample_uniform(*cfg.coverage_offset_x, (n,), device=dev)
        y = xy[:, 1] + scale * sample_uniform(*cfg.coverage_offset_y, (n,), device=dev)
        return torch.stack((x.clamp(*cfg.intercept_x_range), y.clamp(*cfg.intercept_y_range)), dim=-1)

    def _resample_swing_command(self) -> None:
        every = max(int(round(self.cfg.swing_cmd_resample_s / self.step_dt)), 1)
        due = (self.episode_length_buf % every == 0) & (self.episode_length_buf > 0)
        ids = due.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self._sample_swing_command(ids)

    def _log_episode_metrics(self, env_ids: torch.Tensor) -> None:
        """Log per-stage success over the finishing episodes and update the curriculum difficulty."""
        if self.common_step_counter == 0:
            return
        cfg = self.cfg
        if cfg.stage == "grasp":
            success = self.held_all_episode[env_ids]
        elif cfg.stage == "hit":
            success = self.hit[env_ids] & self.held_all_episode[env_ids]
        else:
            success = self.landed_in_target[env_ids]
        rate = success.float().mean().item()
        # target stage: the feed curriculum advances on clearing the net (landing in the target is the reward's
        # job); gating it on target accuracy would hold the feeds at their easiest setting
        promote = self.cleared_net[env_ids] if cfg.stage in ("target", "mobile") else success
        log = self.extras.setdefault("log", {})
        if self.bank is not None:
            # the court-coverage curriculum moves on the launcher shots only
            bank = self.bank_feed[env_ids]
            promote = promote[~bank]
            if bank.any():
                log["Metrics/bank_hit_rate"] = self.hit[env_ids][bank].float().mean().item()
                log["Metrics/bank_cleared_rate"] = self.cleared_net[env_ids][bank].float().mean().item()
                log["Metrics/bank_held_rate"] = self.held_all_episode[env_ids][bank].float().mean().item()
        # difficulty moves once per window of finished episodes, not on every (partial) reset
        self._window_successes += promote.float().sum().item()
        self._window_episodes += len(promote)
        if self._window_episodes >= cfg.curriculum_window:
            window_rate = self._window_successes / self._window_episodes
            if window_rate > cfg.curriculum_promote:
                self.difficulty = min(1.0, self.difficulty + cfg.curriculum_step)
            elif window_rate < cfg.curriculum_demote:
                self.difficulty = max(0.0, self.difficulty - cfg.curriculum_step)
            self._window_successes, self._window_episodes = 0.0, 0
        log["Metrics/success_rate"] = rate
        log["Metrics/held_rate"] = self.held_all_episode[env_ids].float().mean().item()
        log["Metrics/hit_rate"] = self.hit[env_ids].float().mean().item()
        log["Metrics/cleared_net_rate"] = self.cleared_net[env_ids].float().mean().item()
        log["Curriculum/difficulty"] = self.difficulty


def solve_launch_velocity(
    p0: torch.Tensor, target: torch.Tensor, flight_time: torch.Tensor, drag_per_mass: torch.Tensor, iters: int = 5
) -> torch.Tensor:
    """Shooting method: launch velocity that reaches ``target`` after ``flight_time`` under gravity and quadratic drag.

    Args:
        p0: Launch positions [m], shape (N, 3).
        target: Intercept positions [m], shape (N, 3).
        flight_time: Flight times [s], shape (N,).
        drag_per_mass: Drag coefficient divided by mass [1/m], shape (N,).
        iters: Shooting iterations.

    Returns:
        Launch velocities [m/s], shape (N, 3).
    """
    g = torch.tensor([0.0, 0.0, -GRAVITY], device=p0.device)
    T = flight_time.unsqueeze(-1)
    c = drag_per_mass.unsqueeze(-1)
    v0 = (target - p0) / T - 0.5 * g * T  # drag-free solution as the initial guess
    steps = 40
    dt = T / steps
    for _ in range(iters):
        p, v = p0.clone(), v0.clone()
        for _ in range(steps):
            v = v + (g - c * v.norm(dim=-1, keepdim=True) * v) * dt
            p = p + v * dt
        v0 = v0 + (target - p) / T
    return v0


def predict_landing(
    pos: torch.Tensor, vel: torch.Tensor, drag_per_mass: torch.Tensor, dt: float = 0.01, max_steps: int = 400
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate the shuttle flight under gravity and quadratic drag until it reaches the floor.

    Args:
        pos: Shuttle positions in the env frame [m], shape (N, 3).
        vel: Shuttle velocities [m/s], shape (N, 3).
        drag_per_mass: Drag coefficient divided by mass [1/m], shape (N,).
        dt: Integration step [s].
        max_steps: Integration horizon in steps.

    Returns:
        Landing points (x, y) [m], shape (N, 2), and whether the flight clears the net, shape (N,).
    """
    g = torch.tensor([0.0, 0.0, -GRAVITY], device=pos.device)
    c = drag_per_mass.unsqueeze(-1)
    p, v = pos.clone(), vel.clone()
    landing = p[:, :2].clone()
    landed = torch.zeros(len(p), dtype=torch.bool, device=p.device)
    clears = torch.zeros_like(landed)
    for _ in range(max_steps):
        prev = p
        v = v + (g - c * v.norm(dim=-1, keepdim=True) * v) * dt
        p = p + v * dt
        flying = ~landed
        crossed = flying & (prev[:, 0] >= NET_X) & (p[:, 0] < NET_X)
        clears |= crossed & (p[:, 2] > NET_HEIGHT + SHUTTLE_RADIUS)
        down = flying & (p[:, 2] < SHUTTLE_RADIUS)
        landing[down] = p[down, :2]
        landed |= down
        if landed.all():
            break
    return landing, clears

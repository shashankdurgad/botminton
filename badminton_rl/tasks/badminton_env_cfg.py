"""Configuration for the Kuka-Allegro badminton environments (grasp, hit and target stages)."""

from __future__ import annotations

import copy
import os
from typing import Literal

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from isaaclab_tasks.utils import PresetCfg

from .spawners import spawn_floating_root_usd

RACKET_USD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "racket.usda")
"""Procedural racket: grip centre at the origin, head centre at +0.475 m along local Z."""

PEDESTAL_HEIGHT = 0.6
"""Height of the arm's base above the floor [m]."""

NET_X = -2.0
"""Net plane along the env-frame x axis [m]; the robot stands at x = 0 and faces -x."""

NET_HEIGHT = 1.55
"""Height of the net top [m] (regulation)."""

SHUTTLE_RADIUS = 0.03
"""Collision radius of the shuttle proxy sphere [m]."""

SHUTTLE_MASS = 0.005
"""Shuttle mass [kg] (regulation 4.74-5.50 g)."""

FINGER_GAIN = 5.0
"""Scale on the Allegro finger stiffness, damping and torque limit.

At the real 0.7 N*m limit the hand cannot hold a 90 g, 0.6 m racket by friction even at rest
(``scripts/probe.py sweep``); at 5x (3.5 N*m) with friction 2.0 the calibrated grip holds statically.
"""


def _strengthen_fingers(robot: ArticulationCfg, gain: float) -> ArticulationCfg:
    robot = copy.deepcopy(robot)  # the actuator dicts are shared with the stock KUKA_ALLEGRO_CFG
    act = robot.actuators["kuka_allegro_actuators"]
    for attr in ("stiffness", "damping", "joint_effort_limit"):
        values = getattr(act, attr)
        for key in values:
            if not key.startswith("iiwa7"):
                values[key] = values[key] * gain
    return robot


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets. PhysX (Isaac Sim) by default, Newton/MJWarp as an alternative."""

    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.5,
        gpu_max_rigid_patch_count=4 * 5 * 2**15,
        gpu_found_lost_pairs_capacity=2**26,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            impratio=1.0,
            cone="pyramidal",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=4000000),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=1,
        debug_mode=False,
    )
    default = isaacsim_physx


@configclass
class BadmintonSceneCfg(InteractiveSceneCfg):
    """One half court per env: pedestal-mounted Kuka-Allegro, racket, shuttle and net."""

    num_envs = 4096
    env_spacing = 10.0
    replicate_physics = True

    ground = AssetBaseCfg(prim_path="/World/ground", collision_group=-1, spawn=sim_utils.GroundPlaneCfg())

    robot: ArticulationCfg = _strengthen_fingers(replace(
        KUKA_ALLEGRO_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=replace(
            KUKA_ALLEGRO_CFG.init_state,
            pos=(0.0, 0.0, PEDESTAL_HEIGHT),
            # open hand so the racket handle can be placed in the palm without interpenetration
            joint_pos={
                "iiwa7_joint_(1|2|7)": 0.0,
                "iiwa7_joint_3": 0.7854,
                "iiwa7_joint_4": 1.5708,
                "iiwa7_joint_(5|6)": -1.5708,
                "(index|middle|ring)_joint_(0|1|2|3)": 0.0,
                "thumb_joint_0": 0.6,
                "thumb_joint_(1|2|3)": 0.0,
            },
        ),
    ), FINGER_GAIN)

    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, PEDESTAL_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.32)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, PEDESTAL_HEIGHT / 2)),
    )

    net = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Net",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 6.1, 0.76),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9), opacity=0.4),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(NET_X, 0.0, NET_HEIGHT - 0.38)),
    )

    racket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Racket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=RACKET_USD,
            rigid_props=PhysxRigidBodyCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.6, 0.0, 1.2)),
    )

    shuttle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shuttle",
        spawn=sim_utils.SphereCfg(
            radius=SHUTTLE_RADIUS,
            rigid_props=PhysxRigidBodyCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                linear_damping=0.0,
                angular_damping=0.5,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassCfg(mass=SHUTTLE_MASS),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.3, dynamic_friction=0.3, restitution=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.95)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.0, 0.0, SHUTTLE_RADIUS)),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )


@configclass
class BadmintonEnvCfg(DirectRLEnvCfg):
    """Shared settings; the stage subclasses below switch the task and reward terms."""

    stage: Literal["grasp", "hit", "target", "mobile"] = "grasp"
    mobile_base = False
    """Mount the arm on the omnidirectional base (the mobile stage, and the from-scratch wheeled curriculum)."""

    # env: 240 Hz physics, 120 Hz control
    decimation = 2
    episode_length_s = 4.0
    action_space = 23
    observation_space = 99
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
        # high friction so the Allegro fingers can hold the handle
        physics_material=RigidBodyMaterialBaseCfg(static_friction=2.0, dynamic_friction=2.0),
        physics=PhysicsCfg(),
        # court-side view of env 0 (record play videos with --num_envs 1 so env 0 sits at the world origin)
        default_visualizer_cfg=VisualizerCfg(eye=(1.0, 2.9, 2.1), lookat=(-1.3, 0.0, 1.4)),
    )

    scene: BadmintonSceneCfg = BadmintonSceneCfg()

    # -- grip --
    palm_body_name = "palm_link"
    grasp_offset_pos: tuple[float, float, float] = (0.02, 0.02, 0.04)
    """Racket grip-centre position in the palm frame [m] (from ``scripts/probe.py palm``).

    The palm frame has the fingers along +x, index-to-ring along +y and the palm face towards +z;
    the handle lies across the palm, 4 cm above the palm frame, so the fingers wrap over it.
    """
    grasp_offset_quat: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, 0.7071068)
    """Racket orientation in the palm frame (x, y, z, w): +90 deg about x, so the racket head points to the
    index side (-y) and the string bed lies parallel to the palm."""
    racket_head_offset: tuple[float, float, float] = (0.0, 0.0, 0.475)
    """String-bed centre in the racket frame [m]."""
    closing_joint_expr = "(index|middle|ring)_joint_(1|2|3)|thumb_joint_.*"
    """Finger joints whose targets reset to their flexion limit, so a zero action closes the hand."""
    grace_period_s = 0.6
    """The racket is pinned to the palm for this long after reset while the fingers close."""
    drop_distance = 0.10
    """Grip-centre deviation from the grasp offset that counts as a drop [m]."""

    # -- actions: relative joint position targets per control step [rad] --
    arm_action_scale = 0.04
    finger_action_scale = 0.05

    # -- shuttle --
    shuttle_terminal_velocity = 6.7
    """Terminal velocity used to set the quadratic drag coefficient [m/s]."""
    drag_randomization = 0.2
    """Uniform +/- fraction applied to the drag coefficient per episode."""
    launch_x_range = (-4.8, -3.6)
    launch_y_range = (-1.0, 1.0)
    launch_z_range = (1.0, 1.6)
    flight_time_range = (1.0, 1.4)
    """Shuttle flight time from the launcher to the intercept point [s]."""
    intercept_half_extent_max = (0.25, 0.35, 0.3)
    """Half extent of the intercept box around the nominal racket-head position at full difficulty [m]."""
    intercept_half_extent_min = 0.05
    """Half extent of the intercept box at zero difficulty [m]."""

    # -- target zone (target stage) --
    target_x_range = (-6.5, -3.0)
    target_y_range = (-2.0, 2.0)
    target_sigma = 0.8
    """Width of the landing reward around the target [m]."""
    target_success_radius = 1.0
    shuttle_out_x_max = 2.0
    """The shuttle is out of play beyond this x on the robot's side [m]."""

    # -- grasp-stage swing command --
    swing_cmd_half_extent = (0.3, 0.4, 0.3)
    swing_cmd_resample_s = 1.0

    # -- curriculum --
    curriculum_init_difficulty = 0.0
    curriculum_step = 0.02
    curriculum_window = 2000
    """Finished episodes aggregated before each difficulty update."""
    curriculum_promote = 0.7
    curriculum_demote = 0.3

    # -- return bank (self-play feeds) --
    return_bank_path: str | None = None
    """npz of real returns recorded by ``scripts/record_rally.py --bank_out``, in the receiver's court frame."""
    return_bank_frac = 0.0
    """Fraction of episodes fed a bank return instead of a launcher shot."""
    return_bank_delay_range = (0.0, 1.2)
    """Wait after the grace period before the bank return comes [s]: in a rally the receiver holds a ready position
    while its opponent plays the previous shot."""

    # -- reward scales --
    rew_hold = 1.0
    rew_swing_track = 2.0
    rew_approach = 1.0
    rew_intercept_track = 0.3
    rew_hit = 50.0
    rew_hit_out_speed = 5.0
    rew_hit_up_speed = 3.0
    rew_clear_net = 10.0
    rew_land_target = 20.0
    rew_aim = 0.0
    """Paid at contact for the predicted landing point near the target (target stage)."""
    rew_miss = -2.0
    rew_drop = -20.0
    rew_action_rate = -0.005
    rew_joint_vel = -1.0e-4


@configclass
class BadmintonGraspEnvCfg(BadmintonEnvCfg):
    """Stage 1: close the hand on the racket and keep it while swinging the head to commanded points."""

    stage = "grasp"
    episode_length_s = 4.0


@configclass
class BadmintonHitEnvCfg(BadmintonEnvCfg):
    """Stage 2: return a launched shuttle."""

    stage = "hit"
    episode_length_s = 4.0
    rew_hold = 0.1


@configclass
class BadmintonTargetEnvCfg(BadmintonEnvCfg):
    """Stage 3: return the shuttle over the net into a randomized target zone."""

    stage = "target"
    episode_length_s = 4.5
    observation_space = 101  # + landing target (x, y)
    rew_hold = 0.1
    # hit-stage policies block the shuttle back but rarely clear the net: pay for pace, lift and the net
    # pace bonuses reward the hardest hit, which overshoots short targets once the net is cleared reliably;
    # aiming at contact (predicted landing) now carries most of the shot-quality signal
    rew_hit_out_speed = 5.0
    rew_hit_up_speed = 5.0
    rew_clear_net = 40.0
    rew_land_target = 40.0
    rew_aim = 60.0


@configclass
class BadmintonMobileSceneCfg(BadmintonSceneCfg):
    """The arm rides an omnidirectional base: its root is freed and driven by the env every physics step."""

    robot: ArticulationCfg = replace(
        BadmintonSceneCfg().robot,
        spawn=replace(BadmintonSceneCfg().robot.spawn, func=spawn_floating_root_usd),
    )
    pedestal = None  # the base is drawn by the renderer; it has no collision in training either


@configclass
class BadmintonMobileEnvCfg(BadmintonTargetEnvCfg):
    """Stage 4: the target task on a wheeled omnidirectional base that must cover the half court.

    The base is a velocity-controlled holonomic platform (like a mecanum KMR base): the last three
    actions command world-frame (vx, vy, yaw rate), tracked under speed and acceleration limits. The
    curriculum difficulty now spreads the intercept points from the arm's reach to the whole half court.
    """

    stage = "mobile"
    mobile_base = True
    episode_length_s = 5.5
    action_space = 26  # 23 joints + base (vx, vy, wz)
    observation_space = 113  # target obs + base actions (3) + base pose/velocity (7) + intercept in base frame (2)
    scene: BadmintonMobileSceneCfg = BadmintonMobileSceneCfg()

    # -- base --
    base_height = PEDESTAL_HEIGHT
    base_max_lin_vel = 3.0
    base_max_ang_vel = 2.0
    base_max_lin_acc = 6.0
    base_max_ang_acc = 6.0
    base_cmd_tau = 0.2
    """Time constant of the low-pass filter on the base velocity command [s] (the base's velocity controller):
    policy noise resampled every control step would otherwise shake the arm at the full acceleration limit."""
    base_teleport = True
    """Write the root pose every substep (True), or only its velocity and let the solver move it (False). Velocity-only
    tracked better but a 2-node run hung in the GPU simulation step with it; grip drops were the same either way."""
    base_frame_obs = True
    """Observe positions and velocities in the base frame, so a parked base looks like the stationary stages."""
    clamp_cmd_obs = True
    """With base-frame observations, show the arm the intercept clamped into its stationary reach box."""
    base_scripted = True
    """Drive the base with a scripted intercept controller and use the base actions as a residual on it."""
    base_residual_lin_vel = 0.5
    """Planar speed of a full residual base action [m/s] (scripted mode)."""
    base_residual_ang_vel = 0.5
    scripted_brake_acc = 4.8
    """Deceleration the scripted controller plans its braking with [m/s^2], kept under base_max_lin_acc."""
    scripted_gain = 4.0
    """Proportional cap near the goal [1/s]."""
    scripted_yaw_gain = 2.0
    base_bounds_x = (NET_X + 0.6, 5.2)
    """Base centre limits [m]: stays off the net and within the hall run-off behind the court."""
    base_bounds_y = (-3.6, 3.6)

    # -- court coverage curriculum (difficulty 0: current reach, 1: whole half court) --
    coverage_offset_x = (-0.6, 3.6)
    """Intercept offset from the arm's home position at full coverage, along x [m]."""
    coverage_offset_y = (-2.7, 2.7)
    coverage_scale = 1.0
    """Fraction of the coverage offsets reached at difficulty 1. The earlier wheeled stages keep it small, since
    their difficulty also grows the arm's own intercept box."""
    intercept_x_range = (NET_X + 0.7, 4.4)
    intercept_y_range = (-2.9, 2.9)
    flight_time_per_m = 0.45
    """Extra flight time per metre of intercept offset [s/m], so far shots stay reachable."""
    shuttle_out_x_max = 6.5
    curriculum_init_difficulty = 0.0
    curriculum_promote = 0.5
    """Lower than the target stage's: moving to the shot costs some net clearance, and the run plateaued near 0.6."""

    # -- rewards --
    rew_base_track = 0.2
    """Dense: base positioned so the arm's home head position lies under the intercept (incoming only).
    Half of it uses a court-wide kernel so it still pulls from metres away, half a tight one for the final placement.
    Small with the scripted base, which earns it regardless of the arm: it only adds variance to the returns."""
    rew_base_effort = -0.002


# -- the same curriculum trained from scratch on the wheeled robot --
# Every wheeled stage shares the mobile observation (113) and action (26) layout, so each warm-starts the next
# without widening. The base is scripted from the start (the policy's base actions are a residual on it), and the
# stages spread the shots over part of the court as they grow the arm's intercept box.


@configclass
class BadmintonMobileGraspEnvCfg(BadmintonMobileEnvCfg):
    """Stage 1 on wheels: grasp and swing to commanded points while the base drives between them."""

    stage = "grasp"
    episode_length_s = 4.0
    rew_hold = 1.0
    coverage_scale = 0.3


@configclass
class BadmintonMobileHitEnvCfg(BadmintonMobileEnvCfg):
    """Stage 2 on wheels: return a launched shuttle, driving to it."""

    stage = "hit"
    rew_hit_out_speed = 5.0
    rew_hit_up_speed = 3.0
    rew_clear_net = 10.0
    rew_land_target = 20.0
    rew_aim = 0.0
    curriculum_promote = 0.7
    coverage_scale = 0.3


@configclass
class BadmintonMobileTargetEnvCfg(BadmintonMobileEnvCfg):
    """Stage 3 on wheels: the target task with the intercept box still growing; the mobile stage then spreads the
    shots over the whole half court."""

    stage = "target"
    curriculum_promote = 0.7
    coverage_scale = 0.3


@configclass
class BadmintonMobileCoverageEnvCfg(BadmintonMobileEnvCfg):
    """Stage 4 on wheels: the mobile stage after the wheeled target stage. Its difficulty 0.3 matches the coverage
    the wheeled stages already reached, so it starts there."""

    curriculum_init_difficulty = 0.3


@configclass
class BadmintonMobileRallyEnvCfg(BadmintonMobileCoverageEnvCfg):
    """Self-play stage: half the feeds are real returns from rallies of the current policy (fast, flat shots that
    arrive ~0.4 s after the opponent's hit), half launcher shots so the coverage skill stays."""

    return_bank_path = "~/robotics/return_bank.npz"
    return_bank_frac = 0.5
    curriculum_init_difficulty = 0.7

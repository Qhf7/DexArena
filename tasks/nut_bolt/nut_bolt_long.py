"""Pick and place a bolt and nut, then tighten the nut by half a turn.

``NutBoltLong-Sharpa-v1`` is a three-stage mobile-base task:

1. place the bolt upright, with its hexagonal head on the designated table region;
2. place the nut over the bolt on the invisible support plate;
3. hand the seated nut to a prebuilt pure-revolute mechanism and rotate it
   counterclockwise by ``pi`` radians when viewed from above.

Only sparse reward is exposed.  The three milestones are latched in order and
each contributes one point.  A dynamically simulated bolt is used during the
first stage.  Once it has genuinely been set down (upright, released and
settled), its pose is latched and enforced on every simulation substep.  This
gives the later threading stages a fixed bolt without relying on a runtime
dynamic-to-static conversion, which PhysX GPU simulation does not support.
The free placement nut and the revolute nut both exist before GPU initialization;
episode logic only parks one representation and activates the other, so the scene
topology stays fixed for batched training.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np
import sapien
import sapien.physx as physx
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.robots.panda_gripper_left_mobile_base import PandaGripperLeftMobileBase
from tasks.robots.panda_gripper_right_mobile_base import PandaGripperRightMobileBase
from tasks.robots.panda_dual_gripper_mobile_base import PandaDualGripperMobileBase
from tasks.robots.panda_dual_sharpa_hand_mobile_base import (
    PandaDualSharpaHandMobileBase,
)
from tasks.robots.panda_dual_wuji_hand_mobile_base import (
    PandaDualWujiHandMobileBase,
)
from tasks.robots.panda_sharpa_hand_left_mobile_base import (
    PandaSharpaHandLeftMobileBase,
)
from tasks.robots.panda_sharpa_hand_right_mobile_base import (
    PandaSharpaHandRightMobileBase,
)
from tasks.robots.panda_wuji_hand_left_mobile_base import (
    PandaWujiHandLeftMobileBase,
)
from tasks.robots.panda_wuji_hand_right_mobile_base import (
    PandaWujiHandRightMobileBase,
)

from mani_skill.agents.registration import REGISTERED_AGENTS
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_quaternion,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
)
from tasks.config.hands import register_task
from tasks.scene_builders.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.config.object_materials import object_material
from tasks.robots.panda_xhand_hand_left_mobile_base import PandaXHandHandLeftMobileBase  # noqa: F401
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.robots.panda_dual_xhand_hand_mobile_base import PandaDualXHandHandMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.config.sim_profiles import make_sim_config


ENV_ID = "NutBoltLong-Sharpa-v1"
# Standard task defaults use the ANGLED Sharpa mobile-base family throughout:
# one arm for each stage env and both arms for the composed long-horizon env.
# Other supported robots are explicit compatibility overrides only.
SCREWING_DEFAULT_ROBOT_UID = "panda_sharpa_hand_right_angled_mobile_base"
SCREWING_DEFAULT_DUAL_ROBOT_UID = "panda_dual_sharpa_hand_angled_mobile_base"

# Each stage task uses the arm assigned to it in the composed dual-arm task.
# The ANGLED single-arm agents keep the full-size trolley and the matching
# ANGLED dual mount at y = +/-0.22.
#
#   screwing_bolt_pick_place.py        LEFT
#   screwing_nut_pick_place.py         RIGHT
#   screwing_turn.py                   RIGHT
SCREWING_DEFAULT_ROBOT_UID_LEFT = (
    "panda_sharpa_hand_left_angled_mobile_base"
)
SCREWING_DEFAULT_ROBOT_UID_RIGHT = (
    "panda_sharpa_hand_right_angled_mobile_base"
)
SCREWING_SUPPORTED_CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")
SCREWING_DEFAULT_CONTROL_MODE = "pd_joint_pos"
SCREWING_DEFAULT_ARM_INIT_QPOS = [-0.5, -0.2, 0.0, -2.6, 0.0, 2.4, np.pi / 4]

TASK_PHASES = ("full", "bolt_pick_place", "nut_pick_place", "turn")
# These are task-layout placements, not robot/arm parameters. Reuse the
# workspace-calibrated trolley clearances of the two reference task families.
ROBOT_BASE_POSE = sapien.Pose(p=[-0.76, 0.0, 0.0])
DUAL_ROBOT_BASE_POSE = sapien.Pose(p=[-0.87, 0.0, 0.0])
WORK_TABLE_X_OFFSET = 0.30
BOLT_HEX_HEAD_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "task_object_meshes"
    / "screwing"
    / "bolt_hex_head.obj"
)
BOLT_PLATE_COLLISION_BIT = 28
BOLT_THREADED_NUT_COLLISION_BIT = 27
NUT_THREAD_FRICTION_TORQUE = 0.012
NUT_THREAD_DRIVE_DAMPING = 1.0
# SAPIEN's ground is a HALF-SPACE, not a floor: a body parked at z<0 sits metres
# INSIDE static geometry and the solver fights that penetration on every step.
# Measured at z=-5.0: 6 contact manifolds at |J|=inf, min separation -4.088 m,
# and because `nut_revolute` is fixed-root the only DoF that can absorb it --
# the continuous spin joint -- reached qvel -4.7e16 rad/s. That value reached
# env_state and polluted 88.9% of the frames of an exported dataset. Park ABOVE
# the scene instead. Nothing falls from up there, by two separate mechanisms
# that were already in place: the free `nut` rigid body is held by
# `set_locked_motion_axes([True]*6)` (measured 0.0000 mm drift over 30 s, and
# the latch runs in _before_control_step so the body never sees a single free
# substep), and the `threaded_nut` link hangs off a FIXED root link whose spin
# joint carries no gravity torque (CoM on the axis; measured qpos=qvel=0 over
# 30 s).
THREADED_NUT_PARK_Z = 5.0
THREADED_NUT_URDF_PATH = BOLT_HEX_HEAD_PATH.parent / "threaded_nut_revolute.urdf"

GRIPPER_AGENT_TYPES = (PandaGripperLeftMobileBase, PandaGripperRightMobileBase, PandaDualGripperMobileBase)


def quaternion_about_z(theta: torch.Tensor) -> torch.Tensor:
    half = theta / 2
    zeros = torch.zeros_like(theta)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def yaw_about_z(q: torch.Tensor) -> torch.Tensor:
    """Signed twist angle about local z, ignoring swing."""
    return 2.0 * torch.atan2(q[..., 3], q[..., 0])


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + np.pi, 2 * np.pi) - np.pi


def sample_screwing_disc_xy(center, radius: float, batch_size: int) -> torch.Tensor:
    """Sample xy points uniformly by area from a disc."""
    angle = torch.rand(batch_size) * (2 * np.pi)
    radial = radius * torch.sqrt(torch.rand(batch_size))
    return torch.stack(
        [center[0] + radial * torch.cos(angle), center[1] + radial * torch.sin(angle)],
        dim=-1,
    )


def local_axis(q: torch.Tensor, axis) -> torch.Tensor:
    direction = torch.tensor(axis, dtype=q.dtype, device=q.device).expand(
        q.shape[0], -1
    )
    return quaternion_apply(q, direction)


def local_point(pose: Pose, point) -> torch.Tensor:
    offset = torch.tensor(point, dtype=pose.p.dtype, device=pose.p.device).expand(
        pose.p.shape[0], -1
    )
    return pose.p + quaternion_apply(pose.q, offset)


@register_task(
    "NutBoltLong",
    side="both",
    max_episode_steps=2400,
    obs_mode="state",
    control_mode=SCREWING_DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class NutBoltLongEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        "panda_gripper_left_mobile_base",
        "panda_gripper_right_mobile_base",
        "panda_sharpa_hand_left_mobile_base",
        "panda_sharpa_hand_left_angled_mobile_base",
        "panda_sharpa_hand_right_mobile_base",
        "panda_sharpa_hand_right_angled_mobile_base",
        "panda_wuji_hand_left_angled_mobile_base",
        "panda_wuji_hand_right_angled_mobile_base",
        "panda_dual_gripper_mobile_base",
        "panda_dual_sharpa_hand_mobile_base",
        "panda_dual_sharpa_hand_angled_mobile_base",
        "panda_dual_wuji_hand_angled_mobile_base",
            "panda_xhand_hand_left_angled_mobile_base",
        "panda_xhand_hand_right_angled_mobile_base",
        "panda_dual_xhand_hand_angled_mobile_base",
]
    SUPPORTED_CONTROL_MODES = SCREWING_SUPPORTED_CONTROL_MODES
    SUPPORTED_REWARD_MODES = ("sparse", "none")
    agent: Union[
        PandaGripperLeftMobileBase,
        PandaGripperRightMobileBase,
        PandaSharpaHandLeftMobileBase,
        PandaSharpaHandRightMobileBase,
        PandaWujiHandLeftMobileBase,
        PandaWujiHandRightMobileBase,
        PandaDualGripperMobileBase,
        PandaDualSharpaHandMobileBase,
        PandaDualWujiHandMobileBase,
    ]

    # Subtasks disable actors that their stage never touches.
    has_bolt = True
    has_platform = True
    has_nut = True
    show_bolt_place_region = True

    # -- geometry --------------------------------------------------------- #
    bolt_radius = 0.015
    bolt_height = 0.08
    bolt_tip_height = 0.015
    bolt_thread_smooth_top = 0.02
    bolt_thread_pitch = 0.005
    bolt_thread_ridge_radius = 0.0015
    bolt_thread_segs_per_turn = 16
    hex_head_circumradius = 0.024
    hex_head_height = 0.012

    nut_half_height = 0.008
    nut_circumradius = 0.022
    nut_wall_thickness = 0.006

    platform_start_z = 0.060
    platform_half_height = 0.003
    platform_radius = 0.024

    # -- layout: tabletop z=0; robot root uses single/dual pose above ----- #
    bolt_spawn_xy = (-0.25, -0.22)
    bolt_spawn_radius = 0.05
    # Angle between the bolt axis and world +z.  The bolt therefore starts
    # fallen on the table, rather than already standing in the goal pose.
    bolt_spawn_tilt_range = (np.deg2rad(75.0), np.deg2rad(90.0))
    # Keep the goal and loose nut on their original y lanes, but bring both
    # 14 cm toward the robot (the robot approaches from world -x).
    bolt_place_center_xy = (-0.12, 0.0)
    bolt_place_radius = 0.10
    nut_spawn_xy = (-0.08, -0.22)

    # -- success thresholds ---------------------------------------------- #
    # Placement is judged from the real hex-head support face, not from the
    # root/centre point. All six bottom vertices must lie on the tabletop and
    # inside the visible goal disc for three consecutive control frames.
    bolt_upright_cos_threshold = float(np.cos(np.deg2rad(5.0)))
    bolt_bottom_min_z = -5.0e-4
    bolt_bottom_max_z = 1.0e-3
    bolt_place_hold_steps = 3
    bolt_static_threshold = 0.04
    bolt_angular_static_threshold = 0.4

    nut_xy_align_threshold = 0.005
    nut_tilt_cos_threshold = float(np.cos(np.deg2rad(12.0)))
    nut_seat_z_window = 0.003
    nut_xy_align_exit = 0.018
    nut_tilt_cos_exit = 0.84
    nut_seat_z_window_exit = 0.014
    nut_aligned_exit_steps = 5
    nut_engage_hold_steps = 3
    nut_static_threshold = 0.04
    nut_angular_static_threshold = 0.4

    screwing_success_rotation = np.pi
    # Kept as a compatibility observation channel. A pure revolute has no
    # prismatic DOF, so threaded_depth is identically zero.
    screw_pitch = bolt_thread_pitch
    success_hold_steps = 5

    def __init__(
        self,
        *args,
        robot_uids=SCREWING_DEFAULT_DUAL_ROBOT_UID,
        robot_init_qpos_noise=0.02,
        object_init_xy_noise=0.015,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if "sim_config" in kwargs and kwargs["sim_config"] is not None:
            raise ValueError(
                "Screwing freezes the Plant configuration in "
                "tasks/config/sim_profiles.py; runtime sim_config overrides are not allowed"
            )
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.object_init_xy_noise = float(object_init_xy_noise)
        if not BOLT_HEX_HEAD_PATH.exists():
            raise FileNotFoundError(f"Bolt head mesh not found: {BOLT_HEX_HEAD_PATH}")
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        # Locked fleet profile: 400 Hz / 20 Hz, solver 24/8, 1 mm contact
        # offset, enhanced determinism and the project fallback material. The
        # task owns only spacing and buffer capacities here.
        return make_sim_config(
            spacing=3,
            max_rigid_contact_count=2**22,
            max_rigid_patch_count=2**21,
        )

    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(
            eye=[-0.75, -0.60, 0.60], target=[-0.02, 0.0, 0.10]
        )
        wrist_pose = sapien_utils.look_at(
            eye=[0.03, 0.05, 0.0], target=[0.0, 0.0, 0.0]
        )
        wrist_mount = getattr(
            self.agent.robot, "links_map", {}
        ).get("wrist", self.agent.tcp)
        return [
            CameraConfig("base_camera", base_pose, 256, 256, np.pi / 2, 0.01, 100),
            CameraConfig(
                "task_wrist_camera",
                wrist_pose,
                256,
                256,
                np.pi / 2,
                0.01,
                100,
                mount=wrist_mount,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[-0.80, -0.85, 0.70], target=[-0.02, 0.0, 0.08]
        )
        return CameraConfig("render_camera", pose, 1280, 720, 1.15, 0.01, 100)

    @property
    def _robot_base_pose(self):
        uid = self.robot_uids if isinstance(self.robot_uids, str) else ""
        entry = REGISTERED_AGENTS.get(uid)
        scale = (
            getattr(entry.agent_cls, "mobile_base_horizontal_scale", 1.0)
            if entry
            else 1.0
        )
        return (
            ROBOT_BASE_POSE
            if scale < 1.0
            else DUAL_ROBOT_BASE_POSE
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, self._robot_base_pose)

    # ------------------------------------------------------------------ #
    # scene
    # ------------------------------------------------------------------ #
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._set_work_table_pose()
        if self.show_guidance_markers and self.show_bolt_place_region:
            self._load_bolt_place_region_marker()
        if self.has_bolt:
            self._load_bolt()
        if self.has_platform:
            self._load_platform()
        if self.has_nut:
            self._load_nut()
            self._load_threaded_nut()

    def _load_bolt_place_region_marker(self):
        """Render the bolt placement-success disc without adding collision."""
        marker_material = sapien.render.RenderMaterial(
            base_color=[0.05, 0.80, 0.10, 1.0], roughness=0.8
        )
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                p=[
                    self.bolt_place_center_xy[0],
                    self.bolt_place_center_xy[1],
                    0.0006,
                ],
                q=[0.7071068, 0.0, -0.7071068, 0.0],
            ),
            radius=self.bolt_place_radius,
            half_length=0.0005,
            material=marker_material,
        )
        builder.initial_pose = sapien.Pose()
        self.bolt_place_region_marker = builder.build_static(
            name="bolt_placement_region"
        )
        self.remove_from_state_dict_registry(self.bolt_place_region_marker)

    def _set_work_table_pose(self):
        if self.gpu_sim_enabled and not self.scene._gpu_sim_initialized:
            return
        self.table_scene.table.set_pose(
            sapien.Pose(
                p=[-0.12 + WORK_TABLE_X_OFFSET, 0.0, -self.table_scene.table_height],
                q=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            )
        )

    def _load_bolt(self):
        # SAPIEN cylinder primitives extend along local x.
        vertical_q = [0.7071068, 0.0, 0.7071068, 0.0]
        vertical_pose = sapien.Pose(q=vertical_q)
        render_material = sapien.render.RenderMaterial(
            base_color=[0.55, 0.55, 0.60, 1.0], roughness=0.4, metallic=0.8
        )
        # Steel task objects consume the frozen silicone-pad/metal lookup;
        # friction is never tuned inside this task.
        physical_material = object_material("metal")
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            pose=vertical_pose,
            radius=self.bolt_radius,
            half_length=self.bolt_height / 2,
            material=physical_material,
            density=1200.0,
        )
        builder.add_cylinder_visual(
            pose=vertical_pose,
            radius=self.bolt_radius,
            half_length=self.bolt_height / 2,
            material=render_material,
        )

        # A short stepped taper is much cheaper and more stable than hundreds
        # of sub-millimetre collision cylinders.
        num_tip_segments = 20
        segment_half_length = self.bolt_tip_height / num_tip_segments / 2
        for i in range(num_tip_segments):
            z_center = self.bolt_height + (
                i + 0.5
            ) * self.bolt_tip_height / num_tip_segments
            local_z = z_center - self.bolt_height / 2
            radius = self.bolt_radius * (0.8 - 0.55 * i / (num_tip_segments - 1))
            pose = sapien.Pose([0.0, 0.0, local_z], vertical_q)
            builder.add_cylinder_collision(
                pose=pose,
                radius=radius,
                half_length=segment_half_length,
                material=physical_material,
                density=1200.0,
            )
            builder.add_cylinder_visual(
                pose=pose,
                radius=radius,
                half_length=segment_half_length,
                material=render_material,
            )

        # Keep the decorative helix inside the shaft's collision envelope so
        # the visible outer radius never protrudes beyond the contact surface;
        # the shaft collision stays smooth for stable nut placement/rotation.
        radius = self.bolt_radius - self.bolt_thread_ridge_radius
        z0 = -self.bolt_height / 2
        z1 = self.bolt_height / 2 - self.bolt_thread_smooth_top
        length = z1 - z0
        num_segments = int(
            round(length / self.bolt_thread_pitch * self.bolt_thread_segs_per_turn)
        )
        segment_half = 0.55 * np.sqrt(
            (2 * np.pi * radius / self.bolt_thread_segs_per_turn) ** 2
            + (self.bolt_thread_pitch / self.bolt_thread_segs_per_turn) ** 2
        )
        for i in range(num_segments):
            z_center = z0 + (i + 0.5) / num_segments * length
            theta = 2 * np.pi * (z_center - z0) / self.bolt_thread_pitch
            center = [
                radius * np.cos(theta),
                radius * np.sin(theta),
                z_center,
            ]
            tangent = np.array(
                [
                    -radius * np.sin(theta),
                    radius * np.cos(theta),
                    self.bolt_thread_pitch / (2 * np.pi),
                ]
            )
            tangent /= np.linalg.norm(tangent)
            w = 1.0 + tangent[0]
            if w < 1e-6:
                q = [0.0, 0.0, 0.0, 1.0]
            else:
                q = np.array([w, 0.0, -tangent[2], tangent[1]])
                q = (q / np.linalg.norm(q)).tolist()
            builder.add_capsule_visual(
                pose=sapien.Pose(center, q),
                radius=self.bolt_thread_ridge_radius,
                half_length=segment_half,
                material=render_material,
            )

        head_pose = sapien.Pose(
            [0.0, 0.0, -self.bolt_height / 2 + self.hex_head_height / 2]
        )
        builder.add_visual_from_file(
            filename=str(BOLT_HEX_HEAD_PATH),
            pose=head_pose,
            material=render_material,
        )
        builder.add_convex_collision_from_file(
            filename=str(BOLT_HEX_HEAD_PATH),
            pose=head_pose,
            material=physical_material,
            density=1200.0,
        )
        builder.initial_pose = sapien.Pose(
            [*self.bolt_spawn_xy, self.bolt_height / 2]
        )
        self.bolt = builder.build(name="bolt")
        self.bolt.set_mass(0.25)
        self._set_task_object_energy_fence(self.bolt)
        self.bolt.set_collision_group_bit(
            group=2, bit_idx=BOLT_PLATE_COLLISION_BIT, bit=1
        )
        # The revolute joint is the thread constraint. Letting its child also
        # contact the smooth bolt cylinder would over-constrain the same radial
        # DOFs; the free placement nut still collides with the bolt normally.
        self.bolt.set_collision_group_bit(
            group=2, bit_idx=BOLT_THREADED_NUT_COLLISION_BIT, bit=1
        )

    def _load_platform(self):
        """Build the collision-only support that the nut seats on.

        APPROXIMATION -- this disc stands in for the thread itself. On a real
        bolt the thread carries the nut: it stops where the thread starts and
        travels down as you turn. Here the thread is modelled as PURE ROTATION
        with no axial travel (see `_set_threaded_nut_state`), so something has
        to hold the FREE nut at the thread-start plane long enough for the
        seating predicate to qualify it (`_nut_seating_state` measures the
        nut's lowest vertex against `platform_start_z`).

        Measured shape: r = 24 mm, 6 mm thick, top face at
        `platform_start_z` = 60 mm, centred on the bolt axis, mass 10.8 g,
        and NO render body at all (collision-only, `add_cylinder_collision`
        only). It ignores the bolt via the group-2 bit
        `BOLT_PLATE_COLLISION_BIT`, so the shaft passes through it -- but it
        does NOT ignore the robot (measured: plate and hand share no ignore
        bit). So once the bolt is locked there is an invisible 48 mm disc
        60 mm above the table that a teleoperator can bump into without seeing
        it. Until the bolt is locked it is parked at z = -0.20; that is below
        SAPIEN's ground half-space, which is harmless ONLY because the plate is
        kinematic (PhysX generates no kinematic-vs-static contacts; measured 0
        contacts). A dynamic body parked there is the F1 bug -- see
        THREADED_NUT_PARK_Z.
        """
        vertical_q = [0.7071068, 0.0, 0.7071068, 0.0]
        material = object_material("metal")
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            pose=sapien.Pose(q=vertical_q),
            radius=self.platform_radius,
            half_length=self.platform_half_height,
            material=material,
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -0.20])
        self.platform = builder.build_kinematic(name="invisible_bolt_plate")
        self._set_task_object_energy_fence(self.platform)
        self.platform.set_collision_group_bit(
            group=2, bit_idx=BOLT_PLATE_COLLISION_BIT, bit=1
        )

    def _load_nut(self):
        render_material = sapien.render.RenderMaterial(
            base_color=[0.90, 0.50, 0.10, 1.0], roughness=0.6, metallic=0.3
        )
        physical_material = object_material("metal")
        apothem = self.nut_circumradius * np.cos(np.pi / 6)
        segment_half_length = self.nut_circumradius * 0.55
        segment_half_thickness = self.nut_wall_thickness / 2
        builder = self.scene.create_actor_builder()
        local_vertices = []
        corner_signs = np.array(
            [
                [sx, sy, sz]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        )
        for i in range(6):
            angle = i * np.pi / 3
            center = np.array(
                [apothem * np.cos(angle), apothem * np.sin(angle), 0.0]
            )
            pose = sapien.Pose(p=center)
            pose.set_rpy([0.0, 0.0, angle + np.pi / 2])
            half_size = [
                segment_half_length,
                segment_half_thickness,
                self.nut_half_height,
            ]
            box_corners = corner_signs * np.asarray(half_size)
            yaw = angle + np.pi / 2
            rotation = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            local_vertices.append(box_corners @ rotation.T + center)
            builder.add_box_collision(
                pose=pose,
                half_size=half_size,
                material=physical_material,
                density=7870.0,
            )
            builder.add_box_visual(
                pose=pose, half_size=half_size, material=render_material
            )
        builder.initial_pose = sapien.Pose(
            [*self.nut_spawn_xy, self.nut_half_height]
        )
        self.nut = builder.build(name="nut")
        self.nut.set_mass(0.10)
        self._set_task_object_energy_fence(self.nut)

        # Exact vertices of the six collision boxes are used to measure the
        # lowest point even when the nut is tilted.  This keeps pick-and-place
        # success independent of a horizontal-orientation requirement.
        self._nut_local_vertices = torch.tensor(
            np.concatenate(local_vertices, axis=0),
            dtype=torch.float32,
            device=self.device,
        )

    def _generate_threaded_nut_urdf(self) -> Path:
        """Generate the fixed-root, one-DOF nut mechanism used after seating.

        The six boxes exactly match the free nut's visual/collision geometry.
        Generating a URDF is intentional: on this SAPIEN build, URDF-loaded
        articulation joints are the validated fully constrained path, whereas
        programmatically assembled task articulations have shown missing-axis
        constraints. The file is derived and may remain gitignored.
        """
        apothem = self.nut_circumradius * np.cos(np.pi / 6)
        half_length = self.nut_circumradius * 0.55
        half_thickness = self.nut_wall_thickness / 2
        boxes = []
        for i in range(6):
            angle = i * np.pi / 3
            x = apothem * np.cos(angle)
            y = apothem * np.sin(angle)
            yaw = angle + np.pi / 2
            size = (
                2 * half_length,
                2 * half_thickness,
                2 * self.nut_half_height,
            )
            boxes.append(
                f'    <visual><origin xyz="{x:.9g} {y:.9g} 0" '
                f'rpy="0 0 {yaw:.9g}"/><geometry><box size="{size[0]:.9g} '
                f'{size[1]:.9g} {size[2]:.9g}"/></geometry>'
                '<material name="nut_orange"/></visual>\n'
                f'    <collision><origin xyz="{x:.9g} {y:.9g} 0" '
                f'rpy="0 0 {yaw:.9g}"/><geometry><box size="{size[0]:.9g} '
                f'{size[1]:.9g} {size[2]:.9g}"/></geometry></collision>'
            )
        geometry = "\n".join(boxes)
        text = f"""<?xml version="1.0"?>
<robot name="threaded_nut_revolute">
  <material name="nut_orange"><color rgba="0.90 0.50 0.10 1.0"/></material>
  <link name="nut_mount">
    <inertial><mass value="0.001"/><inertia ixx="1e-8" iyy="1e-8" izz="1e-8" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="threaded_nut">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="2.5092579e-5" iyy="2.5092579e-5" izz="4.5504552e-5" ixy="0" ixz="0" iyz="0"/>
    </inertial>
{geometry}
  </link>
  <joint name="nut_turn_joint" type="continuous">
    <parent link="nut_mount"/>
    <child link="threaded_nut"/>
    <axis xyz="0 0 1"/>
    <dynamics damping="0" friction="0"/>
  </joint>
</robot>
"""
        if (
            not THREADED_NUT_URDF_PATH.exists()
            or THREADED_NUT_URDF_PATH.read_text() != text
        ):
            THREADED_NUT_URDF_PATH.write_text(text)
        return THREADED_NUT_URDF_PATH

    def _load_threaded_nut(self):
        """Prebuild the pure-revolute representation; never change topology later."""
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        builders = loader.parse(str(self._generate_threaded_nut_urdf()))[
            "articulation_builders"
        ]
        if len(builders) != 1:
            raise RuntimeError("Expected one threaded-nut articulation")
        builder = builders[0]
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, THREADED_NUT_PARK_Z])
        self.nut_revolute = builder.build(name="nut_revolute")
        self.threaded_nut = self.nut_revolute.links_map["threaded_nut"]
        self.nut_turn_joint = self.nut_revolute.active_joints_map["nut_turn_joint"]
        for joint in self.nut_turn_joint._objs:
            joint.set_drive_properties(
                0.0,
                NUT_THREAD_DRIVE_DAMPING,
                force_limit=NUT_THREAD_FRICTION_TORQUE,
                mode="force",
            )
            joint.set_drive_velocity_target(0.0)

        material = object_material("metal")
        for link in self.nut_revolute.get_links():
            for component in link._objs:
                component.set_max_depenetration_velocity(0.5)
                component.set_max_linear_velocity(3.0)
                for shape in component.get_collision_shapes():
                    shape.set_physical_material(material)
        self.threaded_nut.set_collision_group_bit(
            group=2, bit_idx=BOLT_THREADED_NUT_COLLISION_BIT, bit=1
        )
        # Once engaged, the articulation joint itself carries the nut axially;
        # retaining exact bottom contact with the placement plate injects large
        # solver torques into the otherwise free revolute axis.
        self.threaded_nut.set_collision_group_bit(
            group=2, bit_idx=BOLT_PLATE_COLLISION_BIT, bit=1
        )

    @staticmethod
    def _set_task_object_energy_fence(actor):
        """Apply solver-energy ceilings to task objects only, never robots."""
        for obj in actor._objs:
            component = obj.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            if component is not None:
                component.set_max_depenetration_velocity(0.5)
                component.set_max_linear_velocity(3.0)

    # ------------------------------------------------------------------ #
    # common robot and reset helpers
    # ------------------------------------------------------------------ #
    def _initial_robot_qpos(self, batch_size: int) -> torch.Tensor:
        single_arm_qpos = torch.tensor(
            SCREWING_DEFAULT_ARM_INIT_QPOS,
            dtype=torch.float32,
            device=self.device,
        )
        arm_joint_names = list(self.agent.arm_joint_names)
        if len(arm_joint_names) % len(single_arm_qpos) != 0:
            raise RuntimeError(
                f"Expected a multiple of {len(single_arm_qpos)} arm joints, "
                f"found {len(arm_joint_names)}"
            )
        num_arms = len(arm_joint_names) // len(single_arm_qpos)
        arm_qpos = single_arm_qpos.repeat(num_arms).expand(batch_size, -1).clone()
        arm_qpos += (
            torch.randn((batch_size, len(arm_joint_names)), device=self.device)
            * self.robot_init_qpos_noise
        )

        # Dual keyframes are grouped by arm, whereas articulation joints are
        # side-interleaved. Consume dual reset poses through the agent's
        # name-based helper; keep the standard positional path for one arm.
        if hasattr(self.agent, "get_reset_pos"):
            robot_qpos = (
                self.agent.get_reset_pos(device=self.device)
                .to(torch.float32)
                .expand(batch_size, -1)
                .clone()
            )
        else:
            robot_qpos = torch.tensor(
                self.agent.keyframes["rest"].qpos,
                dtype=torch.float32,
                device=self.device,
            ).expand(batch_size, -1).clone()
        active_joint_names = [
            joint.name for joint in self.agent.robot.get_active_joints()
        ]
        if robot_qpos.shape[-1] != len(active_joint_names):
            raise RuntimeError(
                "The robot rest keyframe and articulation joint counts differ: "
                f"{robot_qpos.shape[-1]} versus {len(active_joint_names)}"
            )
        arm_indices = [active_joint_names.index(name) for name in arm_joint_names]
        robot_qpos[:, arm_indices] = arm_qpos
        return robot_qpos

    def _reset_episode_buffers(self, env_idx: torch.Tensor, task_phase="full"):
        with torch.device(self.device):
            if not hasattr(self, "_success_hold_count"):
                shape = (self.num_envs,)
                self._success_hold_count = torch.zeros(
                    shape, dtype=torch.int32, device=self.device
                )
                self._success_last_eval_step = torch.zeros_like(
                    self._success_hold_count
                )
                self._bolt_place_hold_count = torch.zeros_like(
                    self._success_hold_count
                )
                self._bolt_place_last_eval_step = torch.zeros_like(
                    self._success_last_eval_step
                )
                self._stage_bolt_placed = torch.zeros(
                    shape, dtype=torch.bool, device=self.device
                )
                self._stage_nut_placed = torch.zeros_like(self._stage_bolt_placed)
                self._stage_turned = torch.zeros_like(self._stage_bolt_placed)
                self._nut_engaged = torch.zeros_like(self._stage_bolt_placed)
                self._nut_engage_hold_count = torch.zeros(
                    shape, dtype=torch.int32, device=self.device
                )
                self._nut_engage_last_eval_step = torch.zeros_like(
                    self._nut_engage_hold_count
                )
                self._task_phase = torch.zeros(
                    shape, dtype=torch.int8, device=self.device
                )
                self._bolt_locked = torch.zeros_like(self._stage_bolt_placed)
                self._bolt_lock_pose = torch.zeros(
                    (self.num_envs, 7), dtype=torch.float32, device=self.device
                )
                self._bolt_lock_pose[:, 3] = 1.0
                self._cum_rotation = torch.zeros(
                    shape, dtype=torch.float32, device=self.device
                )
                self._threaded_depth = torch.zeros_like(self._cum_rotation)
                self._nut_yaw_prev = torch.zeros_like(self._cum_rotation)
                self._nut_aligned = torch.zeros_like(self._stage_bolt_placed)
                self._nut_aligned_exit_streak = torch.zeros_like(
                    self._cum_rotation
                )
            self._task_phase[env_idx] = TASK_PHASES.index(task_phase)
            for flag in (
                self._stage_bolt_placed,
                self._stage_nut_placed,
                self._stage_turned,
                self._bolt_locked,
                self._nut_aligned,
            ):
                flag[env_idx] = False
            self._nut_engaged[env_idx] = False
            self._success_hold_count[env_idx] = 0
            self._success_last_eval_step[env_idx] = 0
            self._bolt_place_hold_count[env_idx] = 0
            self._bolt_place_last_eval_step[env_idx] = 0
            self._cum_rotation[env_idx] = 0.0
            self._threaded_depth[env_idx] = 0.0
            self._nut_engage_hold_count[env_idx] = 0
            self._nut_engage_last_eval_step[env_idx] = 0
            self._nut_yaw_prev[env_idx] = 0.0
            self._nut_aligned_exit_streak[env_idx] = 0.0

    def _initialize_workspace(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._set_work_table_pose()
            self.agent.reset(self._initial_robot_qpos(len(env_idx)))
            self.agent.robot.set_root_pose(self._robot_base_pose)

    def _sample_placed_bolt_xy_yaw(self, b: int):
        """Sample the complete output distribution accepted by the goal disc.

        The admissible centre radius depends on hex-head yaw (roughly
        7.6--7.92 cm), so a smaller fixed disc would omit valid long-task
        outputs. Rejection uses the same six support vertices as success.
        """
        centre = torch.tensor(
            self.bolt_place_center_xy, dtype=torch.float32, device=self.device
        )
        xy = sample_screwing_disc_xy(
            self.bolt_place_center_xy, self.bolt_place_radius, b
        )
        yaw = torch.rand(b) * 2 * np.pi - np.pi
        head_angles = torch.arange(
            6, dtype=torch.float32, device=self.device
        ) * (np.pi / 3)

        for _ in range(100):
            vertex_angles = yaw[:, None] + head_angles[None, :]
            vertices_xy = xy[:, None, :] + self.hex_head_circumradius * torch.stack(
                [torch.cos(vertex_angles), torch.sin(vertex_angles)], dim=-1
            )
            valid = (
                torch.linalg.norm(
                    vertices_xy - centre[None, None, :], dim=-1
                ).amax(dim=-1)
                <= self.bolt_place_radius
            )
            if bool(valid.all()):
                return xy, yaw
            invalid = ~valid
            count = int(invalid.sum().item())
            xy[invalid] = sample_screwing_disc_xy(
                self.bolt_place_center_xy, self.bolt_place_radius, count
            )
            yaw[invalid] = torch.rand(count) * 2 * np.pi - np.pi
        raise RuntimeError("Unable to sample a valid placed-bolt pose")

    def _place_bolt(self, env_idx: torch.Tensor, b: int, already_placed=False):
        with torch.device(self.device):
            xyz = torch.zeros((b, 3))
            if already_placed:
                xyz[:, :2], yaw = self._sample_placed_bolt_xy_yaw(b)
                xyz[:, 2] = self.bolt_height / 2 + 1e-3
                quat = quaternion_about_z(yaw)
            else:
                # Sample the bolt in a disc and lay it down at a large random
                # tilt.  Random world yaw chooses its direction on the table;
                # random local spin changes which hex-head face touches it.
                xyz[:, :2] = sample_screwing_disc_xy(
                    self.bolt_spawn_xy, self.bolt_spawn_radius, b
                )
                tilt_min, tilt_max = self.bolt_spawn_tilt_range
                tilt = tilt_min + torch.rand(b) * (tilt_max - tilt_min)
                tilt_q = axis_angle_to_quaternion(
                    torch.stack(
                        [torch.zeros_like(tilt), tilt, torch.zeros_like(tilt)],
                        dim=-1,
                    )
                )
                yaw_q = quaternion_about_z(
                    torch.rand(b) * 2 * np.pi - np.pi
                )
                spin_q = quaternion_about_z(
                    torch.rand(b) * 2 * np.pi - np.pi
                )
                quat = quaternion_multiply(
                    quaternion_multiply(yaw_q, tilt_q), spin_q
                )

                # Compute the exact support height of the convex hex head for
                # this orientation.  Its lower face is at local z=-H/2 and
                # upper face at -H/2+head_height.  A small clearance prevents
                # reset-time penetration; gravity settles the bolt at once.
                head_angles = torch.arange(6, dtype=torch.float32) * (np.pi / 3)
                head_ring = torch.stack(
                    [
                        self.hex_head_circumradius * torch.cos(head_angles),
                        self.hex_head_circumradius * torch.sin(head_angles),
                    ],
                    dim=-1,
                )
                head_z = torch.tensor(
                    [
                        -self.bolt_height / 2,
                        -self.bolt_height / 2 + self.hex_head_height,
                    ],
                    dtype=torch.float32,
                )
                head_vertices = torch.cat(
                    [
                        torch.cat(
                            [head_ring, z.expand(6, 1)], dim=-1
                        )
                        for z in head_z
                    ],
                    dim=0,
                )
                rotated_vertices = quaternion_apply(
                    quat[:, None, :], head_vertices[None, :, :]
                )
                xyz[:, 2] = -rotated_vertices[..., 2].amin(dim=-1) + 1e-3
            self.bolt.set_pose(Pose.create_from_pq(p=xyz, q=quat))
            self.bolt.set_linear_velocity(torch.zeros((b, 3)))
            self.bolt.set_angular_velocity(torch.zeros((b, 3)))
            if already_placed:
                xyz[:, 2] = self.bolt_height / 2
                self._bolt_lock_pose[env_idx] = torch.cat([xyz, quat], dim=-1)
                self._bolt_locked[env_idx] = True
                self._stage_bolt_placed[env_idx] = True
                self._bolt_place_hold_count[env_idx] = self.bolt_place_hold_steps

    def _place_nut_on_table(self, env_idx: torch.Tensor, b: int):
        with torch.device(self.device):
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.tensor(self.nut_spawn_xy, dtype=torch.float32)
                + (torch.rand((b, 2)) * 2 - 1) * self.object_init_xy_noise
            )
            xyz[:, 2] = self.nut_half_height + 1e-3
            yaw = torch.rand(b) * 2 * np.pi - np.pi
            self._set_nut_pose(xyz, quaternion_about_z(yaw))
            self._park_threaded_nut(env_idx, b)

    def _place_nut_on_platform(self, env_idx: torch.Tensor, b: int):
        """Start from a valid output distribution of the nut-placement stage."""
        with torch.device(self.device):
            lock_pose = self._bolt_lock_pose[env_idx]
            bolt_yaw = yaw_about_z(lock_pose[:, 3:])
            yaw = bolt_yaw + torch.rand(b) * 2 * np.pi - np.pi
            quat = quaternion_about_z(yaw)
            self._set_threaded_nut_state(env_idx, lock_pose, quat)
            self._stage_nut_placed[env_idx] = True
            self._nut_aligned[env_idx] = True

    def _set_nut_pose(self, xyz: torch.Tensor, quat: torch.Tensor):
        self.nut.set_pose(Pose.create_from_pq(p=xyz, q=quat))
        zeros = torch.zeros(
            (xyz.shape[0], 3), dtype=xyz.dtype, device=xyz.device
        )
        self.nut.set_linear_velocity(zeros)
        self.nut.set_angular_velocity(zeros)

    def _park_threaded_nut(self, env_idx: torch.Tensor, b: int):
        p = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
        p[:, 2] = THREADED_NUT_PARK_Z
        q = torch.zeros((b, 4), dtype=torch.float32, device=self.device)
        q[:, 0] = 1.0
        zeros = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
        joint_zeros = torch.zeros((b, 1), dtype=torch.float32, device=self.device)
        self.nut_revolute.set_state(
            torch.cat([p, q, zeros, zeros, joint_zeros, joint_zeros], dim=-1),
            env_idx=env_idx,
        )
        self._nut_engaged[env_idx] = False
        self._refresh_hold_state()
        self._sync_nut_visibility()

    def _park_free_nut(self, env_idx: torch.Tensor, b: int):
        p = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
        p[:, 2] = THREADED_NUT_PARK_Z
        q = torch.zeros((b, 4), dtype=torch.float32, device=self.device)
        q[:, 0] = 1.0
        zeros = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
        self.nut.set_state(
            torch.cat([p, q, zeros, zeros], dim=-1),
            env_idx=env_idx,
        )

    def _set_threaded_nut_state(
        self,
        env_idx: torch.Tensor,
        lock_pose: torch.Tensor,
        nut_quat: torch.Tensor,
    ):
        """Activate the fixed-root revolute nut at the locked bolt axis."""
        b = len(env_idx)
        root_p = lock_pose[:, :3].clone()
        root_p[:, 2] = self.platform_start_z + self.nut_half_height
        root_q = lock_pose[:, 3:].clone()
        relative_q = quaternion_multiply(quaternion_invert(root_q), nut_quat)
        joint_qpos = wrap_to_pi(yaw_about_z(relative_q))[:, None]
        zeros = torch.zeros((b, 3), dtype=root_p.dtype, device=self.device)
        joint_zeros = torch.zeros_like(joint_qpos)
        self.nut_revolute.set_state(
            torch.cat(
                [
                    root_p,
                    root_q,
                    zeros,
                    zeros,
                    joint_qpos,
                    joint_zeros,
                ],
                dim=-1,
            ),
            env_idx=env_idx,
        )
        self._park_free_nut(env_idx, b)
        self._nut_engaged[env_idx] = True
        self._nut_yaw_prev[env_idx] = joint_qpos[:, 0]
        self._sync_nut_visibility()

    def _sync_nut_swap_gpu(self):
        """Publish a one-time actor/articulation swap before forming observations."""
        if not self.gpu_sim_enabled:
            return
        px = self.scene.px
        px.gpu_apply_rigid_dynamic_data()
        px.gpu_apply_articulation_root_pose()
        px.gpu_apply_articulation_root_velocity()
        px.gpu_apply_articulation_qpos()
        px.gpu_apply_articulation_qvel()
        px.gpu_update_articulation_kinematics()
        self.scene._gpu_fetch_all()

    def _engage_newly_placed_nuts(self, mask: torch.Tensor):
        """Swap seated free nuts into the prebuilt revolute representation."""
        if not bool(mask.any()):
            return
        env_idx = torch.nonzero(mask).flatten()
        free_q = self.nut.pose.q[env_idx].clone()
        self._set_threaded_nut_state(
            env_idx,
            self._bolt_lock_pose[env_idx],
            free_q,
        )
        self._refresh_hold_state()
        self._sync_nut_swap_gpu()

    def _nut_pose(self) -> Pose:
        free_pose = self.nut.pose
        threaded_pose = self.threaded_nut.pose
        return Pose.create_from_pq(
            p=torch.where(
                self._nut_engaged[:, None], threaded_pose.p, free_pose.p
            ),
            q=torch.where(
                self._nut_engaged[:, None], threaded_pose.q, free_pose.q
            ),
        )

    def _nut_linear_velocity(self) -> torch.Tensor:
        return torch.where(
            self._nut_engaged[:, None],
            self.threaded_nut.get_linear_velocity(),
            self.nut.get_linear_velocity(),
        )

    def _nut_angular_velocity(self) -> torch.Tensor:
        return torch.where(
            self._nut_engaged[:, None],
            self.threaded_nut.get_angular_velocity(),
            self.nut.get_angular_velocity(),
        )

    def _nut_is_grasping(self) -> torch.Tensor:
        return torch.where(
            self._nut_engaged,
            self._is_grasping(self.threaded_nut),
            self._is_grasping(self.nut),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            task_phase = options.get("task_phase", "full")
            if task_phase not in TASK_PHASES:
                raise ValueError(f"Unknown task_phase {task_phase!r}")
            self._reset_episode_buffers(env_idx, task_phase)
            self._initialize_workspace(env_idx)

            already_placed = task_phase in {"nut_pick_place", "turn"}
            if self.has_bolt:
                self._place_bolt(env_idx, b, already_placed=already_placed)
            if self.has_nut:
                if task_phase == "turn":
                    self._place_nut_on_platform(env_idx, b)
                else:
                    self._place_nut_on_table(env_idx, b)
            if self.has_platform:
                self._update_platform_pose(env_idx)
            self._refresh_hold_state()

    # ------------------------------------------------------------------ #
    # dynamic bolt lock and threading state
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Held-body state.
    #
    # WHY THIS IS NOT A PER-SUBSTEP TELEPORT ANY MORE.
    #
    # The previous implementation re-wrote pose + linear + angular velocity of
    # the bolt, the parked free nut and the platform from
    # _before_simulation_step -- i.e. on EVERY PhysX substep. At 400 Hz sim and
    # 20 Hz control that is 20 forced writes per control step per body; measured
    # on NutBoltLong-Sharpa-v1 it was 700 writes over 5 control steps. Whenever a finger
    # touched the locked bolt, each write re-created the penetration the solver
    # had just resolved, so the depenetration impulse was re-applied every
    # substep and, because the bolt could not move, all of that energy went
    # into the robot. The same mechanism was measured on MealPreparation at
    # 31.8 rad/s peak robot qvel against a 2.2 rad/s actuator limit.
    #
    # A kinematic body holds its pose exactly, with no constraint violation to
    # resolve, so the hold no longer injects energy. Poses are written ONCE,
    # when the mask actually flips.
    # ------------------------------------------------------------------ #
    def _hold_bodies(self):
        """(actor, per-env mask) pairs that must be frozen while held."""
        out = []
        if self.has_bolt:
            out.append((self.bolt, self._bolt_locked))
        if self.has_nut:
            # The engaged nut lives in the prebuilt revolute twin; the free-nut
            # actor is parked out of play and must simply stop falling.
            out.append((self.nut, self._nut_engaged))
        return out

    def _hold_pose(self, actor, env_index: int):
        """Pose to write once, at the moment a body becomes held."""
        if actor is self.bolt:
            return sapien.Pose(
                p=self._bolt_lock_pose[env_index, :3].detach().cpu().numpy(),
                q=self._bolt_lock_pose[env_index, 3:].detach().cpu().numpy(),
            )
        current = actor._objs[env_index].get_pose()
        return sapien.Pose(p=[current.p[0], current.p[1], THREADED_NUT_PARK_Z])

    @staticmethod
    def _set_body_visibility(body, i: int, value: float):
        import sapien.render as srender

        obj = body._objs[int(i)]
        # An articulation link exposes `.entity`; a plain actor already is one.
        ent = getattr(obj, "entity", obj)
        rb = ent.find_component_by_type(srender.RenderBodyComponent)
        if rb is None:
            return
        try:
            rb.visibility = float(value)
        except AttributeError:
            rb.set_visibility(float(value))

    def _sync_nut_visibility(self):
        """Keep the parked nut representation out of the render.

        Exactly one of the two nut representations is live at a time; the other
        is parked at +5 m. Parked is physically inert (see THREADED_NUT_PARK_Z),
        but a wrist camera pointed upward would still film it and offline
        re-renders replay wrist poses we did not choose. Edge-triggered on the
        engagement mask, so in steady state this costs one tensor compare per
        control step and writes nothing.
        """
        if not hasattr(self, "_nut_engaged") or getattr(self, "nut", None) is None:
            return
        want = self._nut_engaged.to(torch.int8)
        cached = getattr(self, "_nut_vis_state", None)
        if cached is None or cached.shape != want.shape:
            cached = torch.full_like(want, -1)
        changed = (cached != want).nonzero().flatten().tolist()
        if not changed:
            return
        for i in changed:
            engaged = bool(want[i])
            self._set_body_visibility(self.threaded_nut, i, 1.0 if engaged else 0.0)
            self._set_body_visibility(self.nut, i, 0.0 if engaged else 1.0)
        self._nut_vis_state = want.clone()

    def _refresh_hold_state(self, write_pose: bool = True):
        """Sync the kinematic hold with the lock masks. Call ONLY on a flip.

        ``write_pose=False`` flips the flag but leaves pose and velocity alone.
        That is what a STATE RESTORE needs: the restored state is the truth, and
        stamping the hold pose over it silently rewrites the episode. With
        ``write_pose=True`` (reset, and a live latch) the hold pose is
        authoritative and the body is parked on it.
        """
        if not hasattr(self, "_bolt_locked"):
            return
        if self.gpu_sim_enabled:
            held = bool(self._bolt_locked.any()) or bool(self._nut_engaged.any())
            if not held:
                return
            raise RuntimeError(
                "Screwing holds the placed bolt and the parked free nut with the "
                "PhysX kinematic flag, which GPU PhysX does not support changing "
                "after scene load (set_locked_motion_axes is accepted but ignored "
                "there too -- measured 19 mm of drift on a 'locked' bolt). Run the "
                "task on CPU (num_envs=1), which is the collection path it is used "
                "for. The previous per-substep teleport did work on GPU, but it "
                "re-applied a depenetration impulse on every substep the hand "
                "touched a held body, so those GPU rollouts were not physical."
            )
        for actor, mask in self._hold_bodies():
            if actor is None:
                continue
            for i, raw in enumerate(actor._objs):
                component = raw.find_component_by_type(
                    physx.PhysxRigidDynamicComponent
                )
                if component is None:
                    continue
                want = bool(mask[i])
                held = all(bool(x) for x in component.get_locked_motion_axes())
                if held == want:
                    continue
                if want and write_pose:
                    component.set_pose(self._hold_pose(actor, i))
                    component.set_linear_velocity([0.0, 0.0, 0.0])
                    component.set_angular_velocity([0.0, 0.0, 0.0])
                # Lock the six DoF rather than setting `kinematic`. Both hold the
                # body exactly (measured 0.0000 mm drift over 300 steps), but
                # flipping `kinematic` ZEROES whatever velocity the body is
                # carrying at that instant, so a state restore that lands on a
                # held body lost its recorded velocity and
                # `replay_recording --mode state` failed on it. Locked axes keep
                # the body dynamic, so its state still round-trips.
                component.set_locked_motion_axes([want] * 6)

    def _before_control_step(self):
        """Self-heal the hold once per CONTROL step (20 Hz), never per substep.

        The mask can flip anywhere -- reset, a subtask's own _initialize_episode
        override, a state restore -- so rather than chase every call site this
        reconciles the kinematic flag with the mask. It is a no-op whenever the
        flag already matches, so a held body's pose is written exactly once, on
        the flip; there is no periodic pose write.
        """
        super()._before_control_step()
        self._refresh_hold_state()
        # Same self-heal argument as the hold: the mask can flip in a subtask's
        # own _initialize_episode or in a state restore, so reconcile the render
        # with it here rather than chasing every call site.
        self._sync_nut_visibility()

    def _lock_newly_placed_bolts(self, mask: torch.Tensor):
        """Latch a qualified placement: snap the pose once, then hold it.

        APPROXIMATION -- a real bolt standing free on a table would tip over
        the moment you thread a nut onto it; a real cell holds it in a jig. We
        model no jig, so a placed bolt is held by locking its six DoF. This is
        a LATCH, not a re-pin: measured 5 writes on the flip step (plate pose,
        bolt pose, both velocities, locked axes) and **0 writes in the 60 steps
        after**, with 0.0000 mm of drift over 30 s.

        The snap (z -> bolt_height/2, orientation -> pure yaw) is bounded by
        the gate that must pass first (`_bolt_placement_state` +
        `bolt_place_hold_steps`): tilt <= 5 deg, all six head-bottom vertices
        within [-0.5, +1.0] mm of the tabletop, whole head inside the 0.10 m
        goal disc, settled and released for 3 consecutive control steps. So it
        removes at most 5 deg of tilt and about a millimetre of height.
        """
        if not bool(mask.any()):
            return
        pose = self.bolt.pose
        yaw = yaw_about_z(pose.q)
        snapped_p = pose.p.clone()
        snapped_p[:, 2] = self.bolt_height / 2
        snapped_q = quaternion_about_z(yaw)
        snapped = torch.cat([snapped_p, snapped_q], dim=-1)
        self._bolt_lock_pose = torch.where(
            mask[:, None], snapped, self._bolt_lock_pose
        )
        self._bolt_locked |= mask
        self._update_platform_pose()
        self._refresh_hold_state()

    def _update_platform_pose(self, env_idx: torch.Tensor = None):
        if not self.has_platform:
            return
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        b = len(env_idx)
        current = self.platform.pose
        current_p = current.p[env_idx]
        target_p = self._bolt_lock_pose[env_idx, :3].clone()
        # Pure revolute: neither the nut nor its support has axial travel.
        target_p[:, 2] = self.platform_start_z - self.platform_half_height
        hidden_p = current_p.clone()
        hidden_p[:, 2] = -0.20
        platform_p = torch.where(
            self._bolt_locked[env_idx, None], target_p, hidden_p
        )
        platform_q = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            dtype=platform_p.dtype,
            device=self.device,
        ).expand(b, -1)
        self.platform.set_pose(Pose.create_from_pq(p=platform_p, q=platform_q))

    # ------------------------------------------------------------------ #
    # state tests
    # ------------------------------------------------------------------ #
    def _is_grasping(self, actor) -> torch.Tensor:
        if isinstance(self.agent, GRIPPER_AGENT_TYPES):
            return self.agent.is_grasping(actor).bool()
        contact = [
            torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(link, actor), dim=-1
            )
            for link in self.agent.tip_links
        ]
        return (torch.stack(contact, dim=-1) > 1e-3).any(dim=-1)

    def _bolt_placement_state(self):
        pose = self.bolt.pose
        bolt_axis = local_axis(pose.q, [0.0, 0.0, 1.0])
        head_bottom = local_point(pose, [0.0, 0.0, -self.bolt_height / 2])
        head_angles = torch.arange(
            6, dtype=pose.p.dtype, device=self.device
        ) * (np.pi / 3)
        head_bottom_local = torch.stack(
            [
                self.hex_head_circumradius * torch.cos(head_angles),
                self.hex_head_circumradius * torch.sin(head_angles),
                torch.full_like(head_angles, -self.bolt_height / 2),
            ],
            dim=-1,
        )
        head_bottom_world = pose.p[:, None, :] + quaternion_apply(
            pose.q[:, None, :],
            head_bottom_local[None, :, :].expand(self.num_envs, -1, -1),
        )
        goal_xy = torch.tensor(
            self.bolt_place_center_xy, dtype=pose.p.dtype, device=self.device
        )
        distance = torch.linalg.norm(
            head_bottom[:, :2] - goal_xy,
            dim=-1,
        )
        upright = bolt_axis[:, 2] > self.bolt_upright_cos_threshold
        bottom_z = head_bottom_world[..., 2]
        on_table = (
            bottom_z.amin(dim=-1) >= self.bolt_bottom_min_z
        ) & (
            bottom_z.amax(dim=-1) <= self.bolt_bottom_max_z
        )
        whole_head_in_region = (
            torch.linalg.norm(
                head_bottom_world[..., :2] - goal_xy[None, None, :],
                dim=-1,
            ).amax(dim=-1)
            <= self.bolt_place_radius
        )
        settled = (
            torch.linalg.norm(self.bolt.get_linear_velocity(), dim=-1)
            < self.bolt_static_threshold
        ) & (
            torch.linalg.norm(self.bolt.get_angular_velocity(), dim=-1)
            < self.bolt_angular_static_threshold
        )
        released = ~self._is_grasping(self.bolt)
        candidate = (
            whole_head_in_region
            & upright
            & on_table
            & settled
            & released
        )
        return candidate, distance, upright, on_table, settled, released

    def _qualify_bolt_placement(self, candidate: torch.Tensor) -> torch.Tensor:
        """Debounce the real-support-face placement before snapping the bolt."""
        next_count = torch.where(
            candidate,
            torch.clamp(
                self._bolt_place_hold_count + 1,
                max=self.bolt_place_hold_steps,
            ),
            torch.zeros_like(self._bolt_place_hold_count),
        )
        is_new_control_step = (
            self._elapsed_steps != self._bolt_place_last_eval_step
        )
        self._bolt_place_hold_count = torch.where(
            is_new_control_step, next_count, self._bolt_place_hold_count
        )
        self._bolt_place_last_eval_step = self._elapsed_steps.clone()
        return candidate & (
            self._bolt_place_hold_count >= self.bolt_place_hold_steps
        )

    def _nut_orientation(self):
        q = self._nut_pose().q
        tilt_cos = 1 - 2 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
        return tilt_cos

    def _nut_lowest_z(self, pose: Pose) -> torch.Tensor:
        vertices = self._nut_local_vertices.expand(self.num_envs, -1, -1)
        world_vertices = pose.p[:, None, :] + quaternion_apply(
            pose.q[:, None, :], vertices
        )
        return world_vertices[..., 2].amin(dim=-1)

    def _nut_seating_state(self):
        nut_pose = self._nut_pose()
        bolt_xy = torch.where(
            self._bolt_locked[:, None],
            self._bolt_lock_pose[:, :2],
            self.bolt.pose.p[:, :2],
        )
        horizontal = torch.linalg.norm(nut_pose.p[:, :2] - bolt_xy, dim=-1)
        nut_bottom = self._nut_lowest_z(nut_pose)
        platform_top = self.platform_start_z
        z_gap = torch.abs(nut_bottom - platform_top)
        tilt_cos = self._nut_orientation()
        # Pick-and-place only asks that the nut's actual bottom touches the
        # plate and that its centre lies on the bolt axis.  It does not require
        # the nut to remain horizontal.
        placement_geometric = (
            (horizontal < self.nut_xy_align_threshold)
            & (nut_bottom < self.bolt_height)
            & (z_gap < self.nut_seat_z_window)
        )
        # Turning still needs the nut axis aligned with the bolt, otherwise a
        # tilted nut beside the shaft could earn rotation reward.
        threading_geometric = placement_geometric & (
            tilt_cos > self.nut_tilt_cos_threshold
        )
        outside_exit = (
            (horizontal > self.nut_xy_align_exit)
            | (tilt_cos < self.nut_tilt_cos_exit)
            | (z_gap > self.nut_seat_z_window_exit)
        )
        self._nut_aligned_exit_streak = torch.where(
            outside_exit,
            self._nut_aligned_exit_streak + 1.0,
            torch.zeros_like(self._nut_aligned_exit_streak),
        )
        exit_now = self._nut_aligned_exit_streak >= self.nut_aligned_exit_steps
        self._nut_aligned = (self._nut_aligned | threading_geometric) & (~exit_now)

        settled = (
            torch.linalg.norm(self._nut_linear_velocity(), dim=-1)
            < self.nut_static_threshold
        ) & (
            torch.linalg.norm(self._nut_angular_velocity(), dim=-1)
            < self.nut_angular_static_threshold
        )
        released = ~self._nut_is_grasping()
        placed = placement_geometric & settled & released & self._bolt_locked
        return (
            placed,
            placement_geometric,
            threading_geometric,
            self._nut_aligned.clone(),
            horizontal,
            z_gap,
            tilt_cos,
            settled,
            released,
        )

    def _nut_engagement_ready(
        self, geometrically_seated: torch.Tensor
    ) -> torch.Tensor:
        """Debounce the geometry-only handoff into the prebuilt revolute nut.

        Dexterous-hand contact is intentional during placement and initial
        turning, so fingertip contact and angular velocity must not block the
        handoff. Requiring several distinct control steps of valid seating
        prevents a nut merely passing through the capture volume from snapping
        onto the bolt.
        """
        candidate = self._bolt_locked & geometrically_seated
        next_count = torch.where(
            candidate,
            torch.clamp(
                self._nut_engage_hold_count + 1,
                max=self.nut_engage_hold_steps,
            ),
            torch.zeros_like(self._nut_engage_hold_count),
        )
        is_new_control_step = (
            self._elapsed_steps != self._nut_engage_last_eval_step
        )
        self._nut_engage_hold_count = torch.where(
            is_new_control_step, next_count, self._nut_engage_hold_count
        )
        self._nut_engage_last_eval_step = self._elapsed_steps.clone()
        return candidate & (
            self._nut_engage_hold_count >= self.nut_engage_hold_steps
        )

    def _update_screwing_state(
        self, just_placed: torch.Tensor, geometrically_seated: torch.Tensor
    ):
        nut_pose = self._nut_pose()
        relative_q = quaternion_multiply(
            quaternion_invert(self._bolt_lock_pose[:, 3:]), nut_pose.q
        )
        yaw = yaw_about_z(relative_q)
        delta = wrap_to_pi(yaw - self._nut_yaw_prev)
        # Positive yaw is counterclockwise when viewed from world +z (above the
        # table). Tightening for this task therefore accumulates +delta.
        counterclockwise_delta = delta
        # Count twist only while the nut is currently seated.  The separate
        # ``_nut_aligned`` hysteresis is useful as an observation diagnostic,
        # but using it here would allow a few off-plate frames to add rotation.
        eligible = (
            self._stage_nut_placed
            & self._nut_engaged
            & geometrically_seated
            & (~just_placed)
        )
        next_rotation = torch.clamp(
            self._cum_rotation + counterclockwise_delta,
            min=0.0,
            max=2 * np.pi,
        )
        self._cum_rotation = torch.where(
            eligible, next_rotation, self._cum_rotation
        )
        rotation_delta = torch.where(
            eligible,
            counterclockwise_delta,
            torch.zeros_like(counterclockwise_delta),
        )
        self._nut_yaw_prev = yaw
        # Compatibility observation channel: a pure revolute has no axial DOF.
        self._threaded_depth.zero_()
        turned = self._cum_rotation >= (self.screwing_success_rotation - 1e-4)
        return turned, rotation_delta

    def _latch_success(self, success_condition: torch.Tensor) -> torch.Tensor:
        next_hold = torch.where(
            success_condition,
            torch.clamp(
                self._success_hold_count + 1, max=self.success_hold_steps
            ),
            torch.zeros_like(self._success_hold_count),
        )
        is_new_control_step = self._elapsed_steps != self._success_last_eval_step
        self._success_hold_count = torch.where(
            is_new_control_step, next_hold, self._success_hold_count
        )
        self._success_last_eval_step = self._elapsed_steps.clone()
        return self._success_hold_count >= self.success_hold_steps

    # ------------------------------------------------------------------ #
    # long-task evaluation and reward
    # ------------------------------------------------------------------ #
    def evaluate(self):
        with torch.device(self.device):
            (
                bolt_place_candidate,
                bolt_place_distance,
                bolt_upright,
                bolt_on_table,
                bolt_settled,
                bolt_released,
            ) = self._bolt_placement_state()
            bolt_placed = self._qualify_bolt_placement(bolt_place_candidate)
            newly_bolt_placed = (~self._stage_bolt_placed) & bolt_placed
            self._stage_bolt_placed |= bolt_placed
            self._lock_newly_placed_bolts(newly_bolt_placed)

            (
                nut_placed,
                nut_geometric,
                nut_threading_geometric,
                nut_aligned,
                nut_horizontal,
                nut_z_gap,
                nut_tilt_cos,
                nut_settled,
                nut_released,
            ) = self._nut_seating_state()
            nut_ready = self._nut_engagement_ready(nut_threading_geometric)
            newly_nut_placed = (
                (~self._stage_nut_placed)
                & self._stage_bolt_placed
                & nut_ready
            )
            self._stage_nut_placed |= self._stage_bolt_placed & nut_ready
            self._engage_newly_placed_nuts(newly_nut_placed)

            nut_turned, rotation_delta = self._update_screwing_state(
                newly_nut_placed, nut_threading_geometric
            )
            turn_condition = (
                self._stage_nut_placed
                & self._nut_engaged
                & nut_turned
                & nut_threading_geometric
            )
            turn_success = self._latch_success(turn_condition)
            self._stage_turned |= turn_success
            full_success_condition = turn_success

            success_condition = torch.where(
                self._task_phase == TASK_PHASES.index("bolt_pick_place"),
                bolt_placed,
                torch.where(
                    self._task_phase == TASK_PHASES.index("nut_pick_place"),
                    nut_ready,
                    torch.where(
                        self._task_phase == TASK_PHASES.index("turn"),
                        turn_success,
                        full_success_condition,
                    ),
                ),
            )
            return dict(
                bolt_placed=bolt_placed,
                bolt_place_candidate=bolt_place_candidate,
                bolt_place_hold_count=self._bolt_place_hold_count.clone(),
                bolt_place_distance=bolt_place_distance,
                bolt_upright=bolt_upright,
                bolt_on_table=bolt_on_table,
                bolt_settled=bolt_settled,
                bolt_released=bolt_released,
                bolt_locked=self._bolt_locked.clone(),
                nut_placed=nut_placed,
                nut_ready=nut_ready,
                nut_engage_hold_count=self._nut_engage_hold_count.clone(),
                nut_geometric=nut_geometric,
                nut_threading_geometric=nut_threading_geometric,
                nut_aligned=nut_aligned,
                nut_horizontal=nut_horizontal,
                nut_z_gap=nut_z_gap,
                nut_tilt_cos=nut_tilt_cos,
                nut_settled=nut_settled,
                nut_released=nut_released,
                nut_turned=nut_turned,
                cum_rotation=self._cum_rotation.clone(),
                rotation_delta=rotation_delta,
                threaded_depth=self._threaded_depth.clone(),
                stage_bolt_placed=self._stage_bolt_placed.clone(),
                stage_nut_placed=self._stage_nut_placed.clone(),
                stage_turned=self._stage_turned.clone(),
                full_success_condition=full_success_condition,
                success_condition=success_condition,
                success=success_condition,
            )

    def _milestones(self, info: dict) -> torch.Tensor:
        return torch.stack(
            [
                info["stage_bolt_placed"],
                info["stage_nut_placed"],
                info["stage_turned"],
            ],
            dim=-1,
        )

    def _get_obs_extra(self, info: dict):
        nut_pose = self._nut_pose()
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bolt_pos=self.bolt.pose.p,
            nut_pos=nut_pose.p,
            bolt_place_pos=torch.tensor(
                self.bolt_place_center_xy, device=self.device
            ).expand(self.num_envs, -1),
            nut_horizontal=info["nut_horizontal"][:, None],
            cum_rotation=info["cum_rotation"][:, None],
            threaded_depth=info["threaded_depth"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                bolt_pose=self.bolt.pose.raw_pose,
                nut_pose=nut_pose.raw_pose,
                tcp_to_bolt_pos=self.bolt.pose.p - self.agent.tcp.pose.p,
                tcp_to_nut_pos=nut_pose.p - self.agent.tcp.pose.p,
                nut_to_bolt_pos=self.bolt.pose.p - nut_pose.p,
                nut_z_gap=info["nut_z_gap"][:, None],
                nut_tilt_cos=info["nut_tilt_cos"][:, None],
                stages=self._milestones(info).float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self._milestones(info).to(torch.float32).sum(dim=-1)

    # ------------------------------------------------------------------ #
    # env-state (offline re-render / RL-from-demo-states)
    # ------------------------------------------------------------------ #
    def get_state_dict(self):
        """Persist every env-side latch and screw-progress channel."""
        state = super().get_state_dict()
        state["screwing_task_state"] = torch.cat(
            [
                self._success_hold_count[:, None].float(),
                self._success_last_eval_step[:, None].float(),
                self._stage_bolt_placed[:, None].float(),
                self._stage_nut_placed[:, None].float(),
                self._stage_turned[:, None].float(),
                self._task_phase[:, None].float(),
                self._bolt_locked[:, None].float(),
                self._bolt_lock_pose.float(),
                self._cum_rotation[:, None],
                self._threaded_depth[:, None],
                self._nut_yaw_prev[:, None],
                self._nut_aligned[:, None].float(),
                self._nut_aligned_exit_streak[:, None],
                self._nut_engaged[:, None].float(),
                self._nut_engage_hold_count[:, None].float(),
                self._nut_engage_last_eval_step[:, None].float(),
                self._bolt_place_hold_count[:, None].float(),
                self._bolt_place_last_eval_step[:, None].float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("screwing_task_state", None)
        if env_idx is None:
            restore_idx = torch.arange(self.num_envs, device=self.device)
            super().set_state_dict(state)
        else:
            restore_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            ).flatten()
            super().set_state_dict(state, restore_idx)
        if isinstance(task_state, dict):
            # LeRobot unflattens a top-level custom tensor as
            # {"value": tensor}; raw state snapshots keep it directly.
            task_state = task_state.get("value")
        if task_state is None:
            return
        task_state = common.to_tensor(task_state, device=self.device).reshape(
            len(restore_idx), -1
        )
        if task_state.shape[-1] not in (19, 20, 22, 24):
            raise ValueError(
                "screwing_task_state must have 19, 20, 22, or 24 fields"
            )
        legacy_task_state = task_state.shape[-1] == 19
        self._success_hold_count[restore_idx] = task_state[:, 0].to(torch.int32)
        self._success_last_eval_step[restore_idx] = task_state[:, 1].to(torch.int32)
        self._stage_bolt_placed[restore_idx] = task_state[:, 2].bool()
        self._stage_nut_placed[restore_idx] = task_state[:, 3].bool()
        self._stage_turned[restore_idx] = task_state[:, 4].bool()
        self._task_phase[restore_idx] = task_state[:, 5].to(torch.int8)
        self._bolt_locked[restore_idx] = task_state[:, 6].bool()
        self._bolt_lock_pose[restore_idx] = task_state[:, 7:14].to(torch.float32)
        self._cum_rotation[restore_idx] = task_state[:, 14].to(torch.float32)
        if not legacy_task_state:
            self._threaded_depth[restore_idx] = task_state[:, 15].to(torch.float32)
        else:
            self._threaded_depth[restore_idx] = 0.0
        self._nut_yaw_prev[restore_idx] = task_state[:, 16].to(torch.float32)
        self._nut_aligned[restore_idx] = task_state[:, 17].bool()
        self._nut_aligned_exit_streak[restore_idx] = task_state[:, 18].to(torch.float32)
        if legacy_task_state:
            self._nut_engaged[restore_idx] = False
        else:
            self._nut_engaged[restore_idx] = task_state[:, 19].bool()
        if task_state.shape[-1] in (22, 24):
            self._nut_engage_hold_count[restore_idx] = task_state[:, 20].to(torch.int32)
            self._nut_engage_last_eval_step[restore_idx] = task_state[:, 21].to(torch.int32)
        else:
            self._nut_engage_hold_count[restore_idx] = 0
            self._nut_engage_last_eval_step[restore_idx] = 0
        if task_state.shape[-1] == 24:
            self._bolt_place_hold_count[restore_idx] = task_state[:, 22].to(
                torch.int32
            )
            self._bolt_place_last_eval_step[restore_idx] = task_state[:, 23].to(
                torch.int32
            )
        else:
            self._bolt_place_hold_count[restore_idx] = 0
            self._bolt_place_last_eval_step[restore_idx] = 0
        if env_idx is not None:
            # reset_to_env_states starts a fresh episode clock at zero.
            # Rebase debounce timestamps so reset-time evaluation is idempotent.
            fresh_reset = self._elapsed_steps[restore_idx] == 0
            elapsed = self._elapsed_steps[restore_idx]
            self._success_last_eval_step[restore_idx] = torch.where(
                fresh_reset, elapsed, self._success_last_eval_step[restore_idx]
            )
            self._nut_engage_last_eval_step[restore_idx] = torch.where(
                fresh_reset, elapsed,
                self._nut_engage_last_eval_step[restore_idx],
            )
            self._bolt_place_last_eval_step[restore_idx] = torch.where(
                fresh_reset, elapsed,
                self._bolt_place_last_eval_step[restore_idx],
            )
        if legacy_task_state and self.has_nut:
            legacy_engage = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            legacy_engage[restore_idx] = (
                self._stage_nut_placed[restore_idx]
                & self._bolt_locked[restore_idx]
            )
            self._engage_newly_placed_nuts(legacy_engage)
        if self.has_platform:
            self._update_platform_pose(restore_idx)
        # Flag only. Writing the hold pose here would overwrite the state that
        # was just restored -- it cost a zeroed velocity on every held body and
        # broke `replay_recording --mode state` on recorded episodes.
        self._refresh_hold_state(write_pose=False)
        # Render visibility is EDGE-triggered off `_nut_engaged`: `_nut_vis_state`
        # caches what was last applied, so a restore that lands on a different
        # engagement than the live scene would leave the wrong body visible and
        # never re-apply (the cache reports "no change"). That is invisible to
        # physics and to `evaluate()`, and shows up only in the offline RGB
        # re-render -- which is this project's actual image source, since
        # collection records no cameras. Invalidate, then re-apply.
        self._nut_vis_state = None
        self._sync_nut_visibility()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=SCREWING_DEFAULT_DUAL_ROBOT_UID)
    parser.add_argument("--control_mode", default=SCREWING_DEFAULT_CONTROL_MODE)
    parser.add_argument("--task_phase", default="full", choices=TASK_PHASES)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/screwing")
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        reward_mode="sparse",
        robot_uids=args.robot_uids,
        control_mode=args.control_mode,
        render_mode="rgb_array",
    )
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_trajectory=False,
        save_video_trigger=lambda episode_idx: True,
        max_steps_per_video=args.steps,
        video_fps=25,
    )
    env.reset(seed=args.seed, options={"task_phase": args.task_phase})
    for _ in range(args.steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        env.render()
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()

"""Open a drawer and plug the charger it holds into a socket.

This is a long-horizon mobile-base task with latched-stage / sparse-reward
machinery, built around drawer manipulation and charger insertion.
The full sequence is:

  1. pull the **middle** drawer of PartNet-Mobility cabinet 45841 open,
  2. lift the charger out and place it in the designated area on the table,
  3. pick it up again and press its two prongs into the socket.

The socket and charger have no assets -- they are built out of boxes here, in
the style of ``nut_bolt_threading.py``.  The insertion is vertical: the
charger's prongs point down into the socket.

Only a sparse reward is provided.  The three milestones below are *latched* in
order, and success additionally requires the final inserted state to hold for
five consecutive control steps, as in every other task in the repo.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np
import sapien
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
    quaternion_apply,
    quaternion_invert,
)
from tasks.config.hands import register_task
from tasks.scene_builders.table import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.config.object_materials import object_friction, object_material
from tasks.robots.panda_xhand_hand_left_mobile_base import PandaXHandHandLeftMobileBase  # noqa: F401
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.robots.panda_dual_xhand_hand_mobile_base import PandaDualXHandHandMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.config.sim_profiles import make_sim_config


ENV_ID = "DrawerPlugLong-Sharpa-v1"
# Standard task defaults use the ANGLED Sharpa mobile-base family throughout:
# one arm for each stage env and both arms for the composed long-horizon env.
# Other supported robots are explicit compatibility overrides only.
DRAWER_PLUG_DEFAULT_ROBOT_UID = "panda_sharpa_hand_right_angled_mobile_base"
DRAWER_PLUG_DEFAULT_DUAL_ROBOT_UID = "panda_dual_sharpa_hand_angled_mobile_base"

# Each single-stage task uses the arm assigned to that stage in the composed
# dual-arm task. Both ANGLED single-arm agents keep the full-size trolley and
# the corresponding ANGLED dual mount at y = +/-0.22.
#
#   drawer_plug_open.py                LEFT
#   drawer_plug_pick_place.py          RIGHT
#   drawer_plug_insert.py              RIGHT
DRAWER_PLUG_DEFAULT_ROBOT_UID_LEFT = (
    "panda_sharpa_hand_left_angled_mobile_base"
)
DRAWER_PLUG_DEFAULT_ROBOT_UID_RIGHT = (
    "panda_sharpa_hand_right_angled_mobile_base"
)
DRAWER_PLUG_SELF_COLLISION_BIT = 29
# Width of ``drawer_plug_task_state``: 6 latch/phase scalars, the 2 sampled
# drawer-layout values, and the 9 charger pose finite-difference fields the
# settle gate runs on.
DRAWER_PLUG_TASK_STATE_DIM = 18
DRAWER_PLUG_SUPPORTED_CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")
DRAWER_PLUG_DEFAULT_CONTROL_MODE = "pd_joint_pos"
DRAWER_PLUG_DEFAULT_ARM_INIT_QPOS = [-0.5, -0.2, 0.0, -2.6, 0.0, 2.4, np.pi / 4]

DRAWER_PLUG_ASSET_DIR = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "task_object_meshes"
    / "drawer_plug"
)
DRAWER_URDF_PATH = DRAWER_PLUG_ASSET_DIR / "drawer" / "mobility.urdf"
# Model 45841 stacks three identical drawers; ``link_2``/``joint_2`` is the
# middle one, and its prismatic axis points along -x in the cabinet's root
# frame, i.e. the drawer slides straight out towards the robot.
DRAWER_LINK_NAME = "link_2"
DRAWER_JOINT_NAME = "joint_2"
DRAWER_FRAME_LINK_NAME = "link_0"

# Drawer slide mechanism, CALIBRATED against a measured breakaway force.
#
# SAPIEN's joint `friction` is a COEFFICIENT, not a force in newtons: the
# resistance it produces scales with the joint's internal constraint load. It
# must therefore be set by measuring the force at which the joint actually
# starts to slide, not by writing the real-world newton value into the field.
#
# Measurement method matters. set_qf() is consumed by ONE physics substep, so
# applying it once per env.step() only drives the joint for 1 of 20 substeps
# and under-reports the force badly. The numbers below re-apply the force from
# _before_simulation_step, i.e. genuinely continuously, over 20 control steps
# with "moves >10 mm" as the breakaway criterion:
#
#     coefficient 0.20 ->  4 N    (lighter than any real slide)
#     coefficient 0.35 ->  8 N    <-- real ball-bearing slide is 5-10 N
#     coefficient 0.50 -> 14 N    (friction-slide territory, 10-25 N)
#     coefficient 1.00 -> >14 N
DRAWER_SLIDE_FRICTION = 0.35       # coefficient; measured breakaway 8 N
DRAWER_SLIDE_DAMPING = 2.0         # N*s/m, residual viscous term
# Structural ceiling: a drawer front tears out long before this. Replaces the
# FLT_MAX default, which modelled no limit at all.
DRAWER_JOINT_FORCE_LIMIT = 200.0   # N

# Socket shell: moulded ABS/PC wall socket. The body is KINEMATIC so its mass
# never enters the dynamics, but the density is stated anyway so no task object
# in this family carries the PhysX default of 1000 kg/m3.
SOCKET_DENSITY = 1150.0            # kg/m3, ABS/PC socket housing

# Slightly different polished metals keep the charger readable against the
# socket while giving both parts a stronger metallic appearance.
SOCKET_METAL_COLOR = [0.62, 0.66, 0.72, 1.0]
CHARGER_METAL_COLOR = [0.72, 0.75, 0.80, 1.0]
PRONG_METAL_COLOR = [1.0, 1.0, 1.0, 1.0]
SOCKET_HOLE_COLOR = [0.008, 0.008, 0.010, 1.0]

TASK_PHASES = ("full", "drawer", "charger_pick_place", "charger_insert")

# Task-layer trolley placement, aligned with CabinetPour. Compact and full-size
# trolley footprints have different front extents, so sharing one root x would
# not preserve the same clearance to the work table.
ROBOT_BASE_POSE = sapien.Pose(p=[-0.76, 0.0, 0.0])
DUAL_ROBOT_BASE_POSE = sapien.Pose(p=[-0.87, 0.0, 0.0])
WORK_TABLE_X_OFFSET = 0.30


def yaw_quaternion(yaw: float):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def sample_drawer_plug_disc_xy(center, radius: float, batch_size: int) -> torch.Tensor:
    """Sample ``batch_size`` points uniformly by area from an xy disc."""
    angle = torch.rand(batch_size) * (2 * np.pi)
    radial = radius * torch.sqrt(torch.rand(batch_size))
    return torch.stack(
        [center[0] + radial * torch.cos(angle), center[1] + radial * torch.sin(angle)],
        dim=-1,
    )


def local_axis(q: torch.Tensor, axis) -> torch.Tensor:
    """The world direction of a body-local unit ``axis``, for batched quats."""
    direction = torch.tensor(axis, dtype=q.dtype, device=q.device).expand(
        q.shape[0], -1
    )
    return quaternion_apply(q, direction)


def local_point(pose: Pose, point) -> torch.Tensor:
    """A body-local ``point`` expressed in world coordinates."""
    offset = torch.tensor(point, dtype=pose.p.dtype, device=pose.p.device).expand(
        pose.p.shape[0], -1
    )
    return pose.p + quaternion_apply(pose.q, offset)


def add_slab(
    builder,
    half_size,
    pose: sapien.Pose,
    material=None,
    render_material=None,
    density: float = 1000.0,
):
    """Add one box as both a collision shape and a matching visual."""
    half_size = list(np.asarray(half_size, dtype=np.float64))
    builder.add_box_collision(
        pose=pose, half_size=half_size, material=material, density=density
    )
    if render_material is not None:
        builder.add_box_visual(
            pose=pose, half_size=half_size, material=render_material
        )


def link_piece_bounds(link, relative_to: sapien.Pose = None) -> np.ndarray:
    """AABBs, shape (num_pieces, 2, 3), of a link's collision pieces.

    The middle drawer's floor uses a box collision from the task-local URDF, and
    box shapes carry ``half_size`` rather than a vertex list. Bounds are
    world-frame by default; pass ``relative_to`` to get them in that frame
    instead.  Only ever called on an unmerged articulation, since ``Link`` takes
    its pose from the sim structs while the collision shapes come from the
    underlying SAPIEN component.
    """
    import trimesh

    component = link._objs[0].entity.find_component_by_type(
        sapien.physx.PhysxArticulationLinkComponent
    )
    link_pose = sapien.Pose(
        p=link.pose.p[0].cpu().numpy(), q=link.pose.q[0].cpu().numpy()
    )
    transform = link_pose.to_transformation_matrix()
    if relative_to is not None:
        transform = relative_to.inv().to_transformation_matrix() @ transform
    bounds = []
    for shape in component.get_collision_shapes():
        if isinstance(shape, sapien.physx.PhysxCollisionShapeBox):
            half = np.asarray(shape.half_size)
            mesh = trimesh.creation.box(extents=2 * half)
        else:
            mesh = trimesh.Trimesh(
                vertices=shape.vertices * shape.scale, faces=shape.triangles
            )
        mesh.apply_transform(transform @ shape.local_pose.to_transformation_matrix())
        bounds.append(mesh.bounds)
    return np.array(bounds)


@register_task(
    "DrawerPlugLong",
    side="both",
    max_episode_steps=3400,
    obs_mode="state",
    control_mode=DRAWER_PLUG_DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class DrawerPlugLongEnv(BaseEnv):
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
    SUPPORTED_CONTROL_MODES = DRAWER_PLUG_SUPPORTED_CONTROL_MODES
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

    # Which rigid pieces the scene holds.  Single-stage subtasks derived from
    # this env switch off whatever their stage never touches.
    has_drawer = True
    has_socket = True
    has_charger = True
    # Purely visual task cue. Stage envs that do not ask the operator to place
    # the charger disable it below; the full task and PickPlace keep it visible.
    show_charger_place_region = True

    # -- layout (table top z=0; robot x follows its single/dual platform) ---- #
    # Enlarged from 0.4 so the drawer and handle are easier for dexterous hands
    # to manipulate.
    drawer_scale = 0.5
    drawer_xy = (0.18, 0.18)
    drawer_yaw = 0.0
    # Keep the socket in the table's right-front work area, but within the
    # comfortable reach shared by the single- and dual-arm layouts.
    # operator request: the socket at (-0.05, -0.48) was at the very
    # edge of the right arm's comfortable range (0.860 m from the arm root).
    # Moved 0.10 m towards the operator's left (+y) and 0.075 m towards the arm
    # (-x), which brings it to 0.762 m.
    socket_center_xy = (-0.125, -0.380)
    socket_yaw = np.pi / 2
    # The charger drop-off had to move with it. At its old centre the spawn disc
    # (0.75 * radius, plus the charger's own 48.8 mm footprint half-diagonal)
    # reached 41 mm INTO the socket's 140 mm footprint, so some seeds would have
    # spawned the charger sitting on the socket. The free table strip is bounded
    # above by the open drawer, whose overhang starts at y = -0.121, and to the
    # near side by the table edge at x = -0.425; this centre keeps 32 mm to the
    # socket and 30 mm to the table edge with the radius unchanged, at 0.699 m
    # from the arm root (it was 0.650 m).
    charger_place_center_xy = (-0.29, -0.61)
    charger_place_radius = 0.14

    # -- socket: an a x a x b slab with two rectangular blind holes --------- #
    socket_side = 0.14                 # a
    socket_thickness = 0.036           # b  (a > b)
    socket_floor_thickness = 0.008     # leaves 0.028 m of hole depth
    # Half extents of one slot, (x, y).  The deliberately thin prongs leave
    # enough clearance for a dexterous hand without making insertion trivial.
    socket_slot_half = (0.0045, 0.018)
    socket_slot_offset_x = 0.022       # slot centres at x = +/- this

    # -- charger: a solid body with two thin prongs below ------------------- #
    charger_half = (0.040, 0.035, 0.028)
    charger_prong_half = (0.0022, 0.014, 0.012)
    charger_density = 600.0
    # Drawer-local -x points towards the handle/open side.  Make this value
    # more negative to place the charger farther out, or closer to zero to move
    # it back towards the drawer centre.
    # -0.10 -> -0.11. At -0.10 the four-layer gate measured the
    # charger's nearest collision vertex at 0.904 m from the arm root, 4 mm
    # outside the fleet reach band [0.54, 0.90]; 10 mm towards the handle
    # brings it to ~0.894 m with the rest of the layout untouched.
    charger_outer_offset_x = -0.11
    # Side-resting orientation in the drawer.  It maps charger-local -z (the
    # prong direction) to drawer-local +x (towards the drawer interior), and
    # charger-local +y (the broad prong edge) to world +z, making the blades
    # vertical.
    charger_initial_quat = (0.5, 0.5, -0.5, -0.5)

    # -- success criteria --------------------------------------------------- #
    drawer_success_travel = 0.18
    charger_lateral_tol = 0.009
    charger_depth_threshold = 0.018     # of the 0.028 m the slots offer
    charger_axis_tol = 0.95
    charger_yaw_tol = 0.95
    # Prong-in-slot test.  The slot (9.0 x 36.0 mm) is wider than the prong
    # (4.4 x 28.0 mm), so a FULLY seated prong may legitimately sit 2.3 mm
    # off-centre in x and 4.0 mm in y -- 4.61 mm measured as a radius.  An
    # isotropic 3 mm circle therefore rejected part of the physically-inserted
    # set: measured 3.60 mm slot error with the charger bottomed out at 24 mm,
    # upright and motionless, reported as a failure.  The gate is the slot's own
    # rectangle instead, so "the prong is inside the slot" is exactly what it
    # asks.  The margin only covers solver penetration.
    charger_slot_fit_margin = 0.0005    # m
    charger_max_depth_margin = 0.003
    # Settle gate.  Compared against a POSE finite difference, never against
    # PhysX's reported velocity: a body squeezed by position-driven fingers
    # reports 0.09 m/s / 3.75 rad/s while its pose moves 0.0002 m/s / 0.0009
    # rad/s, which made success impossible for as long as the operator still
    # held the charger.  See ``_update_charger_motion``.
    charger_static_threshold = 0.05
    charger_angular_static_threshold = 0.5
    charger_table_height_tol = 0.015
    out_of_drawer_margin = 0.01
    success_hold_steps = 5

    def __init__(
        self,
        *args,
        robot_uids=DRAWER_PLUG_DEFAULT_DUAL_ROBOT_UID,
        robot_init_qpos_noise=0.02,
        drawer_init_xy_noise=0.01,
        object_init_xy_noise=0.01,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if "sim_config" in kwargs and kwargs["sim_config"] is not None:
            raise ValueError(
                "DrawerPlug freezes the Plant configuration in "
                "tasks/config/sim_profiles.py; runtime sim_config overrides are not allowed"
            )
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.drawer_init_xy_noise = float(drawer_init_xy_noise)
        self.object_init_xy_noise = float(object_init_xy_noise)
        if not DRAWER_URDF_PATH.exists():
            raise FileNotFoundError(f"Drawer URDF not found: {DRAWER_URDF_PATH}")
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # configuration
    # ------------------------------------------------------------------ #
    @property
    def _default_sim_config(self):
        # Locked fleet profile: 400 Hz / 20 Hz, solver 24/8, 1 mm contact
        # offset, enhanced determinism and the project fallback material.  The
        # task owns only spacing and buffer capacities here.
        return make_sim_config(
            spacing=3,
            max_rigid_contact_count=2**22,
            max_rigid_patch_count=2**21,
        )

    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(eye=[-0.80, -0.95, 0.62], target=[-0.16, -0.40, 0.15])
        wrist_pose = sapien_utils.look_at(eye=[0.03, 0.05, 0.0], target=[0.0, 0.0, 0.0])
        wrist_mount = getattr(self.agent.robot, "links_map", {}).get("wrist", self.agent.tcp)
        return [
            CameraConfig("base_camera", base_pose, 256, 256, np.pi / 2, 0.01, 100),
            CameraConfig(
                "task_wrist_camera", wrist_pose, 256, 256, np.pi / 2, 0.01, 100,
                mount=wrist_mount,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-0.60, -1.05, 0.80], target=[-0.18, -0.38, 0.10])
        return CameraConfig("render_camera", pose, 1280, 720, 1.15, 0.01, 100)

    @property
    def _robot_base_pose(self):
        """Use CabinetPour's clearance-calibrated pose for each trolley size."""
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

    # -- derived geometry constants ---------------------------------------- #
    @property
    def charger_prong_tip_z(self) -> float:
        """Charger-local z of the prong tips (the deepest point of the part)."""
        return -(self.charger_half[2] + 2 * self.charger_prong_half[2])

    @property
    def socket_mouth_z(self) -> float:
        """Socket-local z of the slot mouths, i.e. the slab's top face."""
        return self.socket_thickness / 2

    @property
    def socket_hole_depth(self) -> float:
        return self.socket_thickness - self.socket_floor_thickness

    # ------------------------------------------------------------------ #
    # scene
    # ------------------------------------------------------------------ #
    def _load_scene(self, options: dict):
        self._load_table()
        if self.show_guidance_markers and self.show_charger_place_region:
            self._load_charger_place_region_marker()
        if self.has_drawer:
            self._load_drawer()
        if self.has_socket:
            self._load_socket()
        if self.has_charger:
            self._load_charger()

    def _load_table(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._set_work_table_pose()

    def _load_charger_place_region_marker(self):
        """Render the charger placement-success disc without adding collision."""
        marker_material = sapien.render.RenderMaterial(
            base_color=[0.05, 0.80, 0.10, 1.0], roughness=0.8
        )
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                p=[
                    self.charger_place_center_xy[0],
                    self.charger_place_center_xy[1],
                    0.0006,
                ],
                q=[0.7071068, 0.0, -0.7071068, 0.0],
            ),
            radius=self.charger_place_radius,
            half_length=0.0005,
            material=marker_material,
        )
        builder.initial_pose = sapien.Pose()
        self.charger_place_region_marker = builder.build_static(
            name="charger_placement_region"
        )
        self.remove_from_state_dict_registry(self.charger_place_region_marker)

    def _set_work_table_pose(self):
        """Place the work table immediately in front of the robot.

        The GPU backend has no rigid-body buffers to write to until the scene is
        set up, which happens after ``_load_scene``; skipping the load-time call
        there costs nothing, since every reset re-applies the pose anyway.
        """
        if self.gpu_sim_enabled and not self.scene._gpu_sim_initialized:
            return
        self.table_scene.table.set_pose(
            sapien.Pose(
                p=[-0.12 + WORK_TABLE_X_OFFSET, 0.0, -self.table_scene.table_height],
                q=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            )
        )

    # -- drawer cabinet ----------------------------------------------------- #
    def _load_drawer(self):
        self._drawer_pose = sapien.Pose(
            p=[self.drawer_xy[0], self.drawer_xy[1], 0.0],
            q=yaw_quaternion(self.drawer_yaw),
        )
        drawer_builder = self._create_drawer_builder()
        self._drawers: list[Articulation] = []
        for scene_idx in range(self.num_envs):
            drawer_builder.set_scene_idxs([scene_idx])
            drawer_builder.initial_pose = self._drawer_pose
            drawer = drawer_builder.build(name=f"drawer-{scene_idx}")
            self._configure_drawer(drawer)
            self.remove_from_state_dict_registry(drawer)
            self._drawers.append(drawer)
        self.drawer = Articulation.merge(self._drawers, name="drawer", merge_links=True)
        self.add_to_state_dict_registry(self.drawer)
        self.drawer_link = self.drawer.links_map[DRAWER_LINK_NAME]

    def _create_drawer_builder(self):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = self.drawer_scale
        loader.load_multiple_collisions_from_file = True
        loader.set_density(200.0)
        # Wooden task fixture: use the project material table instead of a
        # per-task friction number.  Re-measure the physical drawer's slide
        # resistance before real-hardware deployment; that mechanism value is
        # kept separately in _configure_drawer below.
        drawer_mu = object_friction("wood")
        loader.set_material(
            static_friction=drawer_mu,
            dynamic_friction=drawer_mu,
            restitution=0.0,
        )

        builders = loader.parse(str(DRAWER_URDF_PATH))["articulation_builders"]
        if len(builders) != 1:
            raise RuntimeError(f"Expected one drawer articulation, found {len(builders)}")
        return builders[0]

    def _configure_drawer(self, drawer: Articulation):
        """Free the middle drawer; hold the other two shut."""
        joints = {joint.name: joint for joint in drawer.active_joints}
        for name, joint in joints.items():
            if name == DRAWER_JOINT_NAME:
                joint.set_friction(DRAWER_SLIDE_FRICTION)
                joint.set_drive_properties(
                    stiffness=0.0,
                    damping=DRAWER_SLIDE_DAMPING,
                    force_limit=DRAWER_JOINT_FORCE_LIMIT,
                )
            else:
                # A stiff drive with the default target of 0 keeps the other two
                # drawers closed, so only the middle one is operable. The slide
                # hardware is identical, so it carries the same friction.
                joint.set_friction(DRAWER_SLIDE_FRICTION)
                joint.set_drive_properties(
                    stiffness=1000.0,
                    damping=50.0,
                    force_limit=DRAWER_JOINT_FORCE_LIMIT,
                )

        # PartNet collision meshes overlap slightly; a shared group-2 bit stops
        # the cabinet's own links fighting each other.
        for link in drawer.links:
            link.set_collision_group_bit(
                group=2, bit_idx=DRAWER_PLUG_SELF_COLLISION_BIT, bit=1
            )
            # Task-object energy fence.  This is deliberately not applied to
            # any robot link (Plant remains frozen).
            component = link._objs[0]
            component.set_max_depenetration_velocity(0.5)
            component.set_max_linear_velocity(3.0)

    # -- socket ------------------------------------------------------------- #
    def _load_socket(self):
        """An a x a x b slab, flat on the table, with two rectangular blind holes.

        Local origin sits at the slab's centre; the holes open along +z.  The
        slab is decomposed into six boxes: a full-footprint floor plate that
        makes the holes blind, two strips either side of the slots in y, two
        outer bars in x, and the bridge between the two slots.
        """
        half_a = self.socket_side / 2
        half_b = self.socket_thickness / 2
        slot_hx, slot_hy = self.socket_slot_half
        slot_dx = self.socket_slot_offset_x

        floor_half_z = self.socket_floor_thickness / 2
        floor_center_z = -half_b + floor_half_z
        upper_half_z = self.socket_hole_depth / 2
        upper_center_z = half_b - upper_half_z

        outer_half_x = (half_a - (slot_dx + slot_hx)) / 2
        outer_center_x = half_a - outer_half_x
        strip_half_y = (half_a - slot_hy) / 2
        strip_center_y = half_a - strip_half_y

        pieces = [
            # floor plate: closes the bottom of both slots
            ((half_a, half_a, floor_half_z), (0.0, 0.0, floor_center_z)),
            # the two strips in front of and behind the slots
            ((half_a, strip_half_y, upper_half_z), (0.0, strip_center_y, upper_center_z)),
            ((half_a, strip_half_y, upper_half_z), (0.0, -strip_center_y, upper_center_z)),
            # the outer bars, left and right of the slots
            ((outer_half_x, slot_hy, upper_half_z), (outer_center_x, 0.0, upper_center_z)),
            ((outer_half_x, slot_hy, upper_half_z), (-outer_center_x, 0.0, upper_center_z)),
            # the bridge between the two slots
            ((slot_dx - slot_hx, slot_hy, upper_half_z), (0.0, 0.0, upper_center_z)),
        ]

        builder = self.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(
            base_color=SOCKET_METAL_COLOR, roughness=0.10, metallic=1.0
        )
        hole_material = sapien.render.RenderMaterial(
            base_color=SOCKET_HOLE_COLOR, roughness=0.72, metallic=0.05
        )
        # A real outlet body is hard plastic; the lookup derives the engine
        # value from the frozen silicone-pad contact calibration.
        physical = object_material("hard_plastic")
        for half_size, center in pieces:
            add_slab(
                builder,
                half_size,
                sapien.Pose(p=list(center)),
                material=physical,
                render_material=material,
                density=SOCKET_DENSITY,
            )
        # Dark, visual-only liners sit just below the mouths.  Keeping them out
        # of the collision model preserves the full insertion depth while
        # making the slots read as black from the task camera's oblique view.
        hole_half_z = 0.0002
        hole_center_z = half_b - 2 * hole_half_z
        for x in (-slot_dx, slot_dx):
            builder.add_box_visual(
                pose=sapien.Pose(p=[x, 0.0, hole_center_z]),
                half_size=[slot_hx * 0.96, slot_hy * 0.96, hole_half_z],
                material=hole_material,
            )
        builder.initial_pose = sapien.Pose(
            p=[*self.socket_center_xy, half_b],
            q=yaw_quaternion(self.socket_yaw),
        )
        # Kinematic rather than static: contact never moves it, which is what the
        # task needs, but unlike a static body it can still be re-placed each
        # episode -- GPU sim refuses to move static actors once the scene is
        # loaded. Fixed task actors use the same build-time scene-index pattern.
        self.socket = builder.build_kinematic(name="socket")

    # -- charger ------------------------------------------------------------ #
    def _load_charger(self):
        """A solid cuboid body with two thin prongs below.

        Local origin is the centre of the body; the prongs run along -z and the
        charger plugs down into the socket.  The body is intentionally solid:
        there is no auxiliary socket or hollow rear section in this task.
        """
        body_hx, body_hy, body_hz = self.charger_half
        prong_hx, prong_hy, prong_hz = self.charger_prong_half
        prong_center_z = -body_hz - prong_hz

        builder = self.scene.create_actor_builder()
        body_material = sapien.render.RenderMaterial(
            base_color=CHARGER_METAL_COLOR, roughness=0.09, metallic=1.0
        )
        prong_material = sapien.render.RenderMaterial(
            base_color=PRONG_METAL_COLOR, roughness=0.06, metallic=1.0
        )
        body_physical = object_material("hard_plastic")
        prong_physical = object_material("metal")
        add_slab(
            builder,
            (body_hx, body_hy, body_hz),
            sapien.Pose(),
            material=body_physical,
            render_material=body_material,
            density=self.charger_density,
        )
        for x in (-self.socket_slot_offset_x, self.socket_slot_offset_x):
            add_slab(
                builder,
                (prong_hx, prong_hy, prong_hz),
                sapien.Pose(p=[x, 0.0, prong_center_z]),
                material=prong_physical,
                render_material=prong_material,
                density=self.charger_density,
            )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.6])
        self.charger = builder.build(name="charger")
        for obj in self.charger._objs:
            component = obj.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            if component is not None:
                component.set_max_depenetration_velocity(0.5)
                component.set_max_linear_velocity(3.0)

    # ------------------------------------------------------------------ #
    # cached geometry
    # ------------------------------------------------------------------ #
    def _after_reconfigure(self, options: dict):
        if self.has_drawer:
            self._cache_drawer_geometry()

    def _cache_drawer_geometry(self):
        """Measure the cabinet frame and the middle drawer, at zero travel."""
        frame_link = self._drawers[0].links_map[DRAWER_FRAME_LINK_NAME]
        frame_world = link_piece_bounds(frame_link)
        frame_local = link_piece_bounds(frame_link, relative_to=self._drawer_pose)
        self.cabinet_z = float(-frame_world[:, 0, 2].min() + 1e-3)
        self.frame_front_local_x = float(frame_local[:, 0, 0].min())

        drawer_link = self._drawers[0].links_map[DRAWER_LINK_NAME]
        drawer_local = link_piece_bounds(drawer_link, relative_to=self._drawer_pose)
        size = drawer_local[:, 1] - drawer_local[:, 0]

        # The floor is the widest thin horizontal piece; its top face is what
        # anything inside the drawer rests on.
        horizontal = (size[:, 2] < 0.02) & (size[:, 0] > 0.2) & (size[:, 1] > 0.2)
        if not horizontal.any():
            raise RuntimeError("Could not identify the middle drawer's floor")
        floor = drawer_local[horizontal]
        self.drawer_floor_z = float(floor[:, 1, 2].max()) + self.cabinet_z

        # The side and back walls: tall-ish pieces that bound the usable volume.
        walls = drawer_local[size[:, 2] > 0.05]
        if len(walls) == 0:
            raise RuntimeError("Could not identify the middle drawer's walls")
        self.drawer_wall_top_z = float(walls[:, 1, 2].max()) + self.cabinet_z
        self.drawer_interior_x = (
            float(floor[:, 0, 0].min()),
            float(floor[:, 1, 0].max()),
        )
        self.drawer_interior_y = (
            float(floor[:, 0, 1].min()),
            float(floor[:, 1, 1].max()),
        )

        # The handle is the assembly furthest towards the drawer front (-x): a
        # grab bar plus an end bracket at either side, so all three pieces
        # reaching that front plane are pooled rather than picking one bracket.
        front_x = drawer_local[:, 0, 0].min()
        handle_pieces = drawer_local[drawer_local[:, 0, 0] < front_x + 1e-3]
        handle = np.stack(
            [handle_pieces[:, 0].min(axis=0), handle_pieces[:, 1].max(axis=0)]
        ).mean(axis=0)
        self._drawer_handle_xyz_nominal = np.array(
            [
                self.drawer_xy[0] + handle[0],
                self.drawer_xy[1] + handle[1],
                handle[2] + self.cabinet_z,
            ]
        )
        self.drawer_handle_xyz = tuple(self._drawer_handle_xyz_nominal)

        joint_names = [joint.name for joint in self.drawer.active_joints]
        if DRAWER_JOINT_NAME not in joint_names:
            raise RuntimeError(f"Merged drawer lost {DRAWER_JOINT_NAME!r}")
        self.drawer_joint_index = joint_names.index(DRAWER_JOINT_NAME)
        self.drawer_open_limit = float(
            self.drawer.get_qlimits()[0, self.drawer_joint_index, 1].item()
        )
        if self.drawer_success_travel > self.drawer_open_limit:
            raise RuntimeError(
                f"drawer_success_travel {self.drawer_success_travel} exceeds the "
                f"joint limit {self.drawer_open_limit}"
            )

    # ------------------------------------------------------------------ #
    # drawer-frame helpers
    # ------------------------------------------------------------------ #
    @property
    def drawer_travel(self) -> torch.Tensor:
        """How far the middle drawer is pulled out, shape (num_envs,)."""
        return self.drawer.get_qpos()[:, self.drawer_joint_index]

    def _drawer_local_to_world_xy(
        self, local_xy: torch.Tensor, travel, env_idx: torch.Tensor = None
    ) -> torch.Tensor:
        """Drawer-frame xy (at zero travel) -> world xy, shape (b, 2).

        The drawer slides along the cabinet's local -x, so travel simply shifts
        the local x before the cabinet yaw is applied.
        """
        drawer_xy = (
            self._drawer_episode_xy
            if env_idx is None
            else self._drawer_episode_xy[env_idx]
        )
        shifted = local_xy.clone()
        shifted[:, 0] = shifted[:, 0] - travel
        c, s = np.cos(self.drawer_yaw), np.sin(self.drawer_yaw)
        world = torch.stack(
            [c * shifted[:, 0] - s * shifted[:, 1], s * shifted[:, 0] + c * shifted[:, 1]],
            dim=-1,
        )
        return world + drawer_xy

    def _world_to_drawer_local_xy(self, world_xy: torch.Tensor, travel) -> torch.Tensor:
        """World xy -> drawer-frame xy at zero travel, shape (b, 2)."""
        delta = world_xy - self._drawer_episode_xy
        c, s = np.cos(self.drawer_yaw), np.sin(self.drawer_yaw)
        local = torch.stack([c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]], dim=-1)
        local[:, 0] = local[:, 0] + travel
        return local

    def _out_of_drawer(self, position: torch.Tensor) -> torch.Tensor:
        """Whether a world point has left the middle drawer's open box.

        The drawer is open at the top, so a part is out either once it has been
        lifted clear of the walls or once it sits outside the footprint -- the
        drawer sliding open must not count on its own.
        """
        margin = self.out_of_drawer_margin
        above = position[:, 2] > self.drawer_wall_top_z + margin
        local = self._world_to_drawer_local_xy(position[:, :2], self.drawer_travel)
        outside = (
            (local[:, 0] < self.drawer_interior_x[0] - margin)
            | (local[:, 0] > self.drawer_interior_x[1] + margin)
            | (local[:, 1] < self.drawer_interior_y[0] - margin)
            | (local[:, 1] > self.drawer_interior_y[1] + margin)
        )
        return above | outside

    def _update_charger_motion(self):
        """Refresh the per-control-step pose delta the settle gate reads.

        PhysX's reported rigid-body velocity is NOT usable here.  Measured on
        this env: a charger held motionless in the closed hand reports
        0.092 m/s and 3.748 rad/s, while its pose actually moves 0.0002 m/s and
        0.0009 rad/s; untouched on the table and in free fall the same reading
        is correct.  The error comes from the contact solver, not from the
        object, so it appears for any body squeezed by position-driven fingers.
        Both bogus numbers sit above the settle thresholds, which made stage 3
        unsatisfiable while the operator was still holding the plug.

        A finite difference of the pose measures the same physical quantity
        without the contact term.  It is updated once per control step; the
        buffers are episode state and travel in ``get_state_dict``.
        """
        with torch.device(self.device):
            pose = self.charger.pose
            p = pose.p
            q = pose.q
            # No `_elapsed_steps == 0` case: `_initialize_episode` seeds the
            # buffers, and a zero clock also means "restored demo state", where
            # the restored values are the correct ones to keep.
            is_new_step = self._elapsed_steps != self._charger_motion_last_step
            dt = self.control_timestep
            lin = torch.linalg.norm(p - self._charger_prev_p, dim=-1) / dt
            # Geodesic angle between the two orientations; the abs() makes the
            # double cover q and -q read as the same rotation.
            cos_half = (q * self._charger_prev_q).sum(dim=-1).abs().clamp(max=1.0)
            ang = 2.0 * torch.acos(cos_half) / dt
            keep = ~is_new_step
            self._charger_speed = torch.where(keep, self._charger_speed, lin)
            self._charger_ang_speed = torch.where(keep, self._charger_ang_speed, ang)
            update = (~keep)[:, None]
            self._charger_prev_p = torch.where(update, p, self._charger_prev_p)
            self._charger_prev_q = torch.where(update, q, self._charger_prev_q)
            self._charger_motion_last_step = torch.where(
                ~keep, self._elapsed_steps, self._charger_motion_last_step
            )

    def _charger_settled(self) -> torch.Tensor:
        self._update_charger_motion()
        return (self._charger_speed < self.charger_static_threshold) & (
            self._charger_ang_speed < self.charger_angular_static_threshold
        )

    def _charger_bottom_z(self, pose: Pose) -> torch.Tensor:
        """World z of the charger's conservative oriented bounding-box bottom."""
        local_lo = torch.tensor(
            [-self.charger_half[0], -self.charger_half[1], self.charger_prong_tip_z],
            dtype=pose.p.dtype,
            device=self.device,
        )
        local_hi = torch.tensor(
            [self.charger_half[0], self.charger_half[1], self.charger_half[2]],
            dtype=pose.p.dtype,
            device=self.device,
        )
        local_center = (local_lo + local_hi) / 2
        local_half = (local_hi - local_lo) / 2
        world_center = pose.p + quaternion_apply(
            pose.q, local_center.expand(pose.p.shape[0], -1)
        )
        world_axes = torch.stack(
            [
                local_axis(pose.q, [1.0, 0.0, 0.0]),
                local_axis(pose.q, [0.0, 1.0, 0.0]),
                local_axis(pose.q, [0.0, 0.0, 1.0]),
            ],
            dim=1,
        )
        vertical_half = (world_axes[..., 2].abs() * local_half).sum(dim=-1)
        return world_center[:, 2] - vertical_half

    def _charger_table_state(self):
        """Whether the charger is settled on the designated table region."""
        pose = self.charger.pose
        radial = torch.linalg.norm(
            pose.p[:, :2]
            - torch.tensor(self.charger_place_center_xy, device=self.device),
            dim=-1,
        )
        bottom_z = self._charger_bottom_z(pose)
        on_table = bottom_z.abs() < self.charger_table_height_tol
        if self.has_drawer:
            charger_out = self._out_of_drawer(pose.p)
        else:
            charger_out = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        settled = self._charger_settled()
        placed = (
            charger_out
            & (radial < self.charger_place_radius)
            & on_table
            & settled
        )
        return placed, charger_out, bottom_z, radial, settled

    # ------------------------------------------------------------------ #
    # insertion test
    # ------------------------------------------------------------------ #
    @staticmethod
    def _seated(
        tip_position: torch.Tensor,
        insert_axis: torch.Tensor,
        mouth_position: torch.Tensor,
        mouth_axis: torch.Tensor,
        lateral_tol: float,
        depth_threshold: float,
        axis_tol: float,
    ):
        """Whether a part's tip has entered a hole, and by how much.

        ``insert_axis`` points from the part's body towards its tip;
        ``mouth_axis`` points out of the hole.  A seated part therefore has the
        two anti-parallel, its tip below the mouth plane, and little lateral
        offset from the hole's centre line.
        """
        delta = tip_position - mouth_position
        depth = -(delta * mouth_axis).sum(dim=-1)
        lateral = torch.linalg.norm(delta + depth[:, None] * mouth_axis, dim=-1)
        alignment = -(insert_axis * mouth_axis).sum(dim=-1)
        seated = (
            (depth > depth_threshold)
            & (lateral < lateral_tol)
            & (alignment > axis_tol)
        )
        return seated, depth, lateral, alignment

    def _charger_socket_state(self):
        charger_pose = self.charger.pose
        tip = local_point(charger_pose, [0.0, 0.0, self.charger_prong_tip_z])
        insert_axis = local_axis(charger_pose.q, [0.0, 0.0, -1.0])
        socket_pose = self.socket.pose
        mouth = local_point(socket_pose, [0.0, 0.0, self.socket_mouth_z])
        mouth_axis = local_axis(socket_pose.q, [0.0, 0.0, 1.0])
        seated, depth, lateral, alignment = self._seated(
            tip,
            insert_axis,
            mouth,
            mouth_axis,
            self.charger_lateral_tol,
            self.charger_depth_threshold,
            self.charger_axis_tol,
        )

        # Match each physical prong to a physical slot. Checking only the
        # charger's centre would also accept a 90-degree yaw with both prongs
        # pressing into the solid bridge between the slots.
        prong_offsets = torch.tensor(
            [
                [-self.socket_slot_offset_x, 0.0, self.charger_prong_tip_z],
                [self.socket_slot_offset_x, 0.0, self.charger_prong_tip_z],
            ],
            dtype=charger_pose.p.dtype,
            device=self.device,
        ).expand(self.num_envs, -1, -1)
        prong_tips = charger_pose.p[:, None, :] + quaternion_apply(
            charger_pose.q[:, None, :], prong_offsets
        )
        socket_inv_q = quaternion_invert(socket_pose.q)
        prong_tips_local = quaternion_apply(
            socket_inv_q[:, None, :], prong_tips - socket_pose.p[:, None, :]
        )
        slot_centers_xy = torch.tensor(
            [
                [-self.socket_slot_offset_x, 0.0],
                [self.socket_slot_offset_x, 0.0],
            ],
            dtype=charger_pose.p.dtype,
            device=self.device,
        ).expand(self.num_envs, -1, -1)
        direct_offset = (prong_tips_local[..., :2] - slot_centers_xy).abs()
        swapped_offset = (
            prong_tips_local[..., :2] - slot_centers_xy.flip(1)
        ).abs()
        direct_error = torch.linalg.norm(direct_offset, dim=-1).amax(dim=-1)
        swapped_error = torch.linalg.norm(swapped_offset, dim=-1).amax(dim=-1)
        # Reported for observations and diagnostics only: the scalar distance
        # from each prong tip to the slot centre it is matched with.
        slot_center_error = torch.minimum(direct_error, swapped_error)
        # The gate itself is the slot's own rectangle, per axis.  A circle
        # cannot express this: the slot is 2.3 mm wider than the prong in x and
        # 4.0 mm in y, so any single radius either rejects legitimate seated
        # poses or accepts prongs standing on the bridge between the slots.
        slot_fit = torch.tensor(
            [
                self.socket_slot_half[0] - self.charger_prong_half[0],
                self.socket_slot_half[1] - self.charger_prong_half[1],
            ],
            dtype=charger_pose.p.dtype,
            device=self.device,
        ) + self.charger_slot_fit_margin
        prongs_in_slots = (
            (direct_offset < slot_fit).all(dim=-1).all(dim=-1)
            | (swapped_offset < slot_fit).all(dim=-1).all(dim=-1)
        )

        charger_blade_axis = local_axis(charger_pose.q, [0.0, 1.0, 0.0])
        socket_slot_axis = local_axis(socket_pose.q, [0.0, 1.0, 0.0])
        yaw_alignment = (
            charger_blade_axis * socket_slot_axis
        ).sum(dim=-1).abs()
        prong_depth = self.socket_mouth_z - prong_tips_local[..., 2]
        within_depth = (
            prong_depth.amin(dim=-1) > self.charger_depth_threshold
        ) & (
            prong_depth.amax(dim=-1)
            < self.socket_hole_depth + self.charger_max_depth_margin
        )
        geometrically_inserted = (
            seated
            & within_depth
            & prongs_in_slots
            & (yaw_alignment > self.charger_yaw_tol)
        )
        settled = self._charger_settled()
        return (
            geometrically_inserted & settled,
            depth,
            lateral,
            alignment,
            slot_center_error,
            yaw_alignment,
            settled,
        )

    # ------------------------------------------------------------------ #
    # episode initialisation
    # ------------------------------------------------------------------ #
    def _initial_robot_qpos(self, batch_size: int) -> torch.Tensor:
        single_arm_qpos = torch.tensor(
            DRAWER_PLUG_DEFAULT_ARM_INIT_QPOS,
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

        # Dual agents' keyframe arrays are grouped per arm while their active
        # articulation joints are side-interleaved.  Consume their reset pose
        # through the name-based helper; single-arm agents retain the standard
        # positional keyframe path.
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
        active_joint_names = [joint.name for joint in self.agent.robot.get_active_joints()]
        if robot_qpos.shape[-1] != len(active_joint_names):
            raise RuntimeError(
                "The robot's 'rest' keyframe has "
                f"{robot_qpos.shape[-1]} joints, but the articulation has "
                f"{len(active_joint_names)} active joints"
            )
        arm_indices = [active_joint_names.index(name) for name in arm_joint_names]
        robot_qpos[:, arm_indices] = arm_qpos
        return robot_qpos

    def _reset_episode_buffers(self, env_idx: torch.Tensor, task_phase: str = "full"):
        """Allocate (once) and clear the stage latches and the hold counter."""
        with torch.device(self.device):
            if not hasattr(self, "_success_hold_count"):
                shape = (self.num_envs,)
                self._success_hold_count = torch.zeros(shape, dtype=torch.int32, device=self.device)
                self._success_last_eval_step = torch.zeros_like(self._success_hold_count)
                self._stage_drawer_open = torch.zeros(shape, dtype=torch.bool, device=self.device)
                self._stage_charger_placed = torch.zeros_like(self._stage_drawer_open)
                self._stage_charger_inserted = torch.zeros_like(self._stage_drawer_open)
                self._task_phase = torch.zeros(shape, dtype=torch.int8, device=self.device)
                self._drawer_episode_xy = torch.zeros(
                    (self.num_envs, 2), dtype=torch.float32, device=self.device
                )
                # Charger pose finite-difference buffers (see
                # ``_update_charger_motion``).  The first evaluate() of an
                # episode re-seeds them from the live pose, so the initial
                # values here only have to be well-formed.
                self._charger_prev_p = torch.zeros(
                    (self.num_envs, 3), dtype=torch.float32, device=self.device
                )
                self._charger_prev_q = torch.zeros(
                    (self.num_envs, 4), dtype=torch.float32, device=self.device
                )
                self._charger_prev_q[:, 0] = 1.0
                self._charger_motion_last_step = torch.zeros_like(
                    self._success_hold_count
                )
                self._charger_speed = torch.zeros(
                    (self.num_envs,), dtype=torch.float32, device=self.device
                )
                self._charger_ang_speed = torch.zeros_like(self._charger_speed)
            self._task_phase[env_idx] = TASK_PHASES.index(task_phase)
            for flag in (
                self._stage_drawer_open,
                self._stage_charger_placed,
                self._stage_charger_inserted,
            ):
                flag[env_idx] = False
            self._success_hold_count[env_idx] = 0
            self._success_last_eval_step[env_idx] = 0
            self._charger_motion_last_step[env_idx] = 0
            self._charger_speed[env_idx] = 0.0
            self._charger_ang_speed[env_idx] = 0.0

    def _initialize_workspace(self, env_idx: torch.Tensor):
        """Reset the work table and the complete mobile-base robot."""
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            # TableSceneBuilder restores its own default pose on every reset.
            self._set_work_table_pose()

            self.agent.reset(self._initial_robot_qpos(len(env_idx)))
            self.agent.robot.set_root_pose(self._robot_base_pose)

    def _initialize_drawer(self, env_idx: torch.Tensor, b: int, travel: float):
        """Randomise the cabinet pose and park the middle drawer at ``travel``."""
        with torch.device(self.device):
            xy_offset = (torch.rand((b, 2)) * 2 - 1) * self.drawer_init_xy_noise
            drawer_p = torch.zeros((b, 3))
            drawer_p[:, :2] = torch.tensor(self.drawer_xy, dtype=torch.float32) + xy_offset
            drawer_p[:, 2] = self.cabinet_z
            self._drawer_episode_xy[env_idx] = drawer_p[:, :2]
            drawer_q = torch.tensor(
                yaw_quaternion(self.drawer_yaw), dtype=torch.float32
            ).expand(b, -1)
            self.drawer.set_pose(Pose.create_from_pq(p=drawer_p, q=drawer_q))

            if self.num_envs == 1:
                handle = self._drawer_handle_xyz_nominal.copy()
                handle[:2] += xy_offset[0].cpu().numpy()
                handle[0] -= travel
                self.drawer_handle_xyz = tuple(handle)

            qpos = torch.zeros_like(self.drawer.get_qpos()[env_idx])
            qpos[:, self.drawer_joint_index] = travel
            self.drawer.set_qpos(qpos)
            self.drawer.set_qvel(torch.zeros_like(qpos))

    def _place_charger_in_drawer(self, env_idx: torch.Tensor, b: int, travel: float):
        with torch.device(self.device):
            # Start from the measured interior centre, then apply the explicit
            # outward offset so the placement remains meaningful after scaling.
            center_local_xy = (
                (self.drawer_interior_x[0] + self.drawer_interior_x[1]) / 2
                + self.charger_outer_offset_x,
                (self.drawer_interior_y[0] + self.drawer_interior_y[1]) / 2,
            )
            local = torch.tensor(center_local_xy, dtype=torch.float32).expand(b, -1)
            local = local + (torch.rand((b, 2)) * 2 - 1) * self.object_init_xy_noise
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = self._drawer_local_to_world_xy(local, travel, env_idx)
            # In the side-resting orientation local y is vertical, and the
            # solid charger body (not its inward-facing prongs) touches the floor.
            xyz[:, 2] = self.drawer_floor_z + self.charger_half[1] + 1e-3
            quat = torch.tensor(
                self.charger_initial_quat, dtype=torch.float32
            ).expand(b, -1)
            self._place_charger(b, xyz, quat)

    def _place_charger_on_table(self, b: int):
        """Start from a valid outcome of the charger pick-and-place stage."""
        with torch.device(self.device):
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = sample_drawer_plug_disc_xy(
                self.charger_place_center_xy,
                0.75 * self.charger_place_radius,
                b,
            )
            # ``charger_initial_quat`` puts local +y upward, so the charger's
            # solid body rests on its -y side and the prongs point horizontally.
            xyz[:, 2] = self.charger_half[1] + 1e-3
            quat = torch.tensor(
                self.charger_initial_quat, dtype=torch.float32
            ).expand(b, -1)
            self._place_charger(b, xyz, quat)

    def _place_charger(self, b: int, xyz: torch.Tensor, quat: torch.Tensor):
        with torch.device(self.device):
            self.charger.set_pose(Pose.create_from_pq(p=xyz, q=quat))
            self.charger.set_linear_velocity(torch.zeros((b, 3)))
            self.charger.set_angular_velocity(torch.zeros((b, 3)))

    def _place_socket(self, b: int):
        with torch.device(self.device):
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                (torch.rand((b, 2)) * 2 - 1) * self.object_init_xy_noise
                + torch.tensor(self.socket_center_xy, dtype=torch.float32)
            )
            xyz[:, 2] = self.socket_thickness / 2
            self.socket.set_pose(
                Pose.create_from_pq(
                    p=xyz,
                    q=torch.tensor(
                        yaw_quaternion(self.socket_yaw), dtype=torch.float32
                    ).expand(b, -1),
                )
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            task_phase = options.get("task_phase", "full")
            if task_phase not in TASK_PHASES:
                raise ValueError(f"Unknown task_phase {task_phase!r}")
            self._reset_episode_buffers(env_idx, task_phase)
            self._initialize_workspace(env_idx)

            # Stages after the pull start from an already-open drawer, with a
            # little more travel than success asks for.
            travel = (
                self.drawer_success_travel + 0.04
                if task_phase in {"charger_pick_place", "charger_insert"}
                else 0.0
            )
            if self.has_drawer:
                self._initialize_drawer(env_idx, b, travel)
            if self.has_socket:
                self._place_socket(b)
            if self.has_charger:
                if task_phase == "charger_insert":
                    self._place_charger_on_table(b)
                else:
                    self._place_charger_in_drawer(env_idx, b, travel)

            if task_phase in {"charger_pick_place", "charger_insert"}:
                self._stage_drawer_open[env_idx] = True
            if task_phase == "charger_insert":
                self._stage_charger_placed[env_idx] = True

            # Seed the charger finite-difference buffers from the pose this
            # episode actually starts at. `_update_charger_motion` used to do
            # the seeding itself by testing `_elapsed_steps == 0` -- but that
            # test is ALSO true on the demo-state restore path, because
            # `reset(options={"reset_to_env_states": ...})` starts a fresh
            # episode clock at zero. There it wiped the charger speeds and the
            # success hold count that had just been restored, so a mid-demo
            # frame came back as "charger stationary, no progress". Seeding here
            # lets that test go away.
            if self.has_charger:
                charger_pose = self.charger.pose
                self._charger_prev_p[env_idx] = charger_pose.p[env_idx]
                self._charger_prev_q[env_idx] = charger_pose.q[env_idx]

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    def _latch_success(self, success_condition: torch.Tensor) -> torch.Tensor:
        """Count consecutive control steps the condition holds for, and report
        whether it has now held for ``success_hold_steps`` of them."""
        next_hold = torch.where(
            success_condition,
            torch.clamp(self._success_hold_count + 1, max=self.success_hold_steps),
            torch.zeros_like(self._success_hold_count),
        )
        # `_reset_episode_buffers` already zeroes the count for a fresh episode,
        # so there is no `_elapsed_steps == 0` clamp here: a zero clock also
        # means "restored demo state", and clamping there threw away the hold
        # the restore had just brought back.
        is_new_control_step = self._elapsed_steps != self._success_last_eval_step
        self._success_hold_count = torch.where(
            is_new_control_step, next_hold, self._success_hold_count
        )
        self._success_last_eval_step = self._elapsed_steps.clone()
        return self._success_hold_count >= self.success_hold_steps

    def evaluate(self):
        with torch.device(self.device):
            travel = self.drawer_travel
            drawer_open = travel > self.drawer_success_travel

            (
                charger_placed,
                charger_out,
                charger_bottom_z,
                charger_place_distance,
                charger_settled,
            ) = self._charger_table_state()
            (
                charger_inserted,
                charger_depth,
                charger_lateral,
                charger_alignment,
                charger_slot_error,
                charger_yaw_alignment,
                _,
            ) = self._charger_socket_state()

            # Latch the stages in order so the sequence cannot be short-cut.
            self._stage_drawer_open |= drawer_open
            charger_pick_place_success = charger_placed
            self._stage_charger_placed |= (
                self._stage_drawer_open & charger_pick_place_success
            )
            self._stage_charger_inserted |= (
                self._stage_charger_placed & charger_inserted
            )

            full_success_condition = self._stage_charger_placed & charger_inserted
            # Each phase's recording begins from that phase's initial state, so
            # evaluate the phase rather than requiring the full sequence.
            success_condition = torch.where(
                self._task_phase == TASK_PHASES.index("drawer"),
                drawer_open,
                torch.where(
                    self._task_phase == TASK_PHASES.index("charger_pick_place"),
                    charger_pick_place_success,
                    torch.where(
                        self._task_phase == TASK_PHASES.index("charger_insert"),
                        charger_inserted,
                        full_success_condition,
                    ),
                ),
            )

            return dict(
                drawer_travel=travel,
                drawer_open=drawer_open,
                charger_out=charger_out,
                charger_placed=charger_placed,
                charger_bottom_z=charger_bottom_z,
                charger_place_distance=charger_place_distance,
                charger_settled=charger_settled,
                charger_inserted=charger_inserted,
                charger_seated=charger_inserted,
                charger_depth=charger_depth,
                charger_lateral=charger_lateral,
                charger_alignment=charger_alignment,
                charger_slot_error=charger_slot_error,
                charger_yaw_alignment=charger_yaw_alignment,
                stage_drawer_open=self._stage_drawer_open.clone(),
                stage_charger_placed=self._stage_charger_placed.clone(),
                stage_charger_inserted=self._stage_charger_inserted.clone(),
                full_success_condition=full_success_condition,
                success_condition=success_condition,
                success=self._latch_success(success_condition),
            )

    def _milestones(self, info: dict) -> torch.Tensor:
        return torch.stack(
            [
                info["stage_drawer_open"],
                info["stage_charger_placed"],
                info["stage_charger_inserted"],
            ],
            dim=-1,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            drawer_qpos=info["drawer_travel"][:, None],
            charger_pos=self.charger.pose.p,
            socket_pos=self.socket.pose.p,
            charger_place_pos=torch.tensor(
                self.charger_place_center_xy, device=self.device
            ).expand(self.num_envs, -1),
            charger_depth=info["charger_depth"][:, None],
            charger_place_distance=info["charger_place_distance"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                charger_pose=self.charger.pose.raw_pose,
                socket_pose=self.socket.pose.raw_pose,
                drawer_pose=self.drawer_link.pose.raw_pose,
                drawer_xy=self._drawer_episode_xy,
                tcp_to_charger_pos=self.charger.pose.p - self.agent.tcp.pose.p,
                charger_to_place_pos=torch.tensor(
                    self.charger_place_center_xy, device=self.device
                ).expand(self.num_envs, -1)
                - self.charger.pose.p[:, :2],
                charger_to_socket_pos=self.socket.pose.p - self.charger.pose.p,
                charger_lateral=info["charger_lateral"][:, None],
                charger_slot_error=info["charger_slot_error"][:, None],
                charger_yaw_alignment=info["charger_yaw_alignment"][:, None],
                stages=self._milestones(info).float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Cumulative progress score: once a sequential milestone is latched, its
        # point remains in every later step's reward.
        return self._milestones(info).to(torch.float32).sum(dim=-1)

    # ------------------------------------------------------------------ #
    # env-state (offline re-render / RL-from-demo-states)
    # ------------------------------------------------------------------ #
    def get_state_dict(self):
        """Include every env-side latch used by evaluation and replay.

        Actor/articulation poses are already handled by BaseEnv, but stage
        progress, success debounce, phase selection and sampled drawer layout
        are task state and must survive reset-to-env-state round trips.
        """
        state = super().get_state_dict()
        state["drawer_plug_task_state"] = torch.cat(
            [
                self._success_hold_count[:, None].float(),
                self._success_last_eval_step[:, None].float(),
                self._stage_drawer_open[:, None].float(),
                self._stage_charger_placed[:, None].float(),
                self._stage_charger_inserted[:, None].float(),
                self._task_phase[:, None].float(),
                self._drawer_episode_xy.float(),
                self._charger_prev_p.float(),
                self._charger_prev_q.float(),
                self._charger_motion_last_step[:, None].float(),
                self._charger_speed[:, None].float(),
                self._charger_ang_speed[:, None].float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        """Restore task state.

        ``env_idx`` is NOT optional in practice: ManiSkill's
        ``reset(options={"reset_to_env_states": ...})`` -- the PPO
        reverse-curriculum path and the offline RGB re-render path -- calls
        ``set_state_dict(env_states, env_idx)``. Without the parameter the whole
        family raised ``TypeError`` there while frame-0 action replay, which
        calls this with one argument, kept passing.
        """
        state = dict(state)
        task_state = state.pop("drawer_plug_task_state", None)
        if env_idx is None:
            restore_idx = torch.arange(self.num_envs, device=self.device)
            super().set_state_dict(state)
        else:
            restore_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            ).flatten()
            super().set_state_dict(state, restore_idx)
        if isinstance(task_state, dict):
            # A flattened LeRobot environment state restores a top-level tensor
            # as {"value": tensor}; raw_states.pkl keeps the tensor directly.
            task_state = task_state.get("value")
        if task_state is None:
            return
        task_state = common.to_tensor(task_state, device=self.device).reshape(
            len(restore_idx), DRAWER_PLUG_TASK_STATE_DIM
        )
        self._success_hold_count[restore_idx] = task_state[:, 0].to(torch.int32)
        self._success_last_eval_step[restore_idx] = task_state[:, 1].to(torch.int32)
        self._stage_drawer_open[restore_idx] = task_state[:, 2].bool()
        self._stage_charger_placed[restore_idx] = task_state[:, 3].bool()
        self._stage_charger_inserted[restore_idx] = task_state[:, 4].bool()
        self._task_phase[restore_idx] = task_state[:, 5].to(torch.int8)
        self._drawer_episode_xy[restore_idx] = task_state[:, 6:8].to(torch.float32)
        self._charger_prev_p[restore_idx] = task_state[:, 8:11].to(torch.float32)
        self._charger_prev_q[restore_idx] = task_state[:, 11:15].to(torch.float32)
        self._charger_motion_last_step[restore_idx] = task_state[:, 15].to(torch.int32)
        self._charger_speed[restore_idx] = task_state[:, 16].to(torch.float32)
        self._charger_ang_speed[restore_idx] = task_state[:, 17].to(torch.float32)
        if env_idx is not None:
            # A demo-state restore starts a fresh episode clock at zero, and the
            # reset-time get_info() would then read as a NEW control step and
            # re-derive the counts from a single frame. Rebase the step
            # registers so that first evaluate leaves the restored values alone.
            elapsed = self._elapsed_steps[restore_idx]
            self._success_last_eval_step[restore_idx] = elapsed.to(torch.int32)
            self._charger_motion_last_step[restore_idx] = elapsed.to(torch.int32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument(
        "--robot_uids",
        default=DRAWER_PLUG_DEFAULT_DUAL_ROBOT_UID,
        help="Robot to instantiate in DrawerPlugLong-Sharpa-v1",
    )
    parser.add_argument("--control_mode", default=DRAWER_PLUG_DEFAULT_CONTROL_MODE)
    parser.add_argument("--task_phase", default="full", choices=TASK_PHASES)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/drawer_plug")
    args = parser.parse_args()

    env_kwargs = dict(
        num_envs=1,
        obs_mode="state",
        reward_mode="sparse",
        control_mode=args.control_mode,
        render_mode="rgb_array",
    )
    env_kwargs["robot_uids"] = args.robot_uids
    env = gym.make(args.env_id, **env_kwargs)
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_trajectory=False,
        save_video_trigger=lambda episode_idx: True,
        max_steps_per_video=args.steps,
        video_fps=25,
    )
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Robot DoF", env.unwrapped.agent.robot.dof)

    env.reset(seed=args.seed, options={"task_phase": args.task_phase})
    for step in range(args.steps):
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
        if step % 25 == 0:
            print(
                f"step {step}: drawer {float(info['drawer_travel']):.3f} m  "
                f"placed {bool(info['charger_placed'])} "
                f"charger depth {float(info['charger_depth']):+.3f} "
                f"reward {float(reward):.1f}"
            )
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()

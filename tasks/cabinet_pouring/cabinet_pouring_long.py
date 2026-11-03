"""Open a cabinet, take out the bottle it holds, set it on the table, pour it out.

This combines `drawer_open.py`'s PartNet-Mobility cabinet with `pour.py`'s
rigid-particle "liquid".  The full sequence the robot has to perform is:

  1. swing the cabinet door open,
  2. reach into the lower compartment and grasp the bottle standing in there,
  3. carry the bottle out and set it down upright on the table,
  4. unscrew, remove, and park its cap,
  5. pick it up again and tip it so the particles land in the bowl.

The cabinet is oriented so its opening faces the robot (-x) and its door swings
towards +y; everything the robot has to put down afterwards lives on the free
-y half of the table, so the open door never crosses the working area.

Only a sparse reward is provided.  The five stages above are *latched* in order,
so a policy cannot skip the placement step by pouring straight out of the
cabinet; success additionally requires the pour itself to hold for ten
consecutive control steps, as in `pour.py` and `ball_grasping.py`.
"""

import argparse
import json
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
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
)
from tasks.config.hands import register_task
from tasks.scene_builders.table import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.robots.panda_xhand_hand_left_mobile_base import PandaXHandHandLeftMobileBase  # noqa: F401
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.robots.panda_dual_xhand_hand_mobile_base import PandaDualXHandHandMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.config.sim_profiles import make_sim_config
from tasks.config.object_materials import object_friction, object_material


ENV_ID = "CabinetPouringLong-Sharpa-v1"
# Standard task defaults use the ANGLED Sharpa mobile-base family throughout:
# one arm for each stage env and both arms for the composed long-horizon env.
# Straight/gripper robots remain in SUPPORTED_ROBOTS only as explicit compatibility
# overrides; no standard CabinetPour registration defaults to them.
DEFAULT_ROBOT_UID = "panda_sharpa_hand_right_angled_mobile_base"
DEFAULT_DUAL_ROBOT_UID = "panda_dual_sharpa_hand_angled_mobile_base"

# ---- per-subtask arm assignment --------------------------------
# Each single-stage sub-task is solved by ONE arm, fixed by the long-horizon
# task's division of labour (project decision). Each sub-task module imports
# only the side it uses:
#
#   cabinet_pour_door_open.py          LEFT
#   cabinet_pour_bottle_pick_place.py  RIGHT
#   cabinet_pour_cap_turn.py           RIGHT  (on the angled bracket; see that
#                                              module -- it was the dual rig, and
#                                              the physics note there still holds)
#   cabinet_pour_cap_pick_place.py     RIGHT
#   cabinet_pour_pour.py               RIGHT
#
# Both uids are the ANGLED Sharpa hand on the FULL-size trolley with the shoulder
# at its matching dual mount (y = +/-0.22). The standard long task uses the dual
# ANGLED robot, so each stage arm is kinematically identical, joint for joint, to
# its corresponding arm in the composed environment.
DEFAULT_ROBOT_UID_LEFT = "panda_sharpa_hand_left_angled_mobile_base"
DEFAULT_ROBOT_UID_RIGHT = "panda_sharpa_hand_right_angled_mobile_base"

# Per-task object assets live under task_object_meshes/<task>/ (repo convention:
# each task keeps its own asset folder so ownership is obvious during development,
# even when tasks share a source object like the PartNet 3615 bottle).
TASK_ASSET_DIR = (
    Path(__file__).resolve().parents[1] / "assets" / "task_object_meshes" / "cabinet_pour"
)
ROBOCASA_TARGET_DIR = TASK_ASSET_DIR / "bowl"
ROBOCASA_TARGET_MESH_SCALE = 0.12

CABINET_URDF_PATH = TASK_ASSET_DIR / "cabinet" / "mobility.urdf"
CABINET_SELF_COLLISION_BIT = 29

# Collision categories for the bottle and rigid-particle liquid proxy.
# SAPIEN's default robot/world shapes use contact type/affinity [1, 1].  Water
# particles deliberately live on their own category 8: rigid spheres hitting a
# PD-driven finger behave like ball bearings, not water, and a fingertip/cap
# overlap can otherwise pump them out of the mouth.  Category 8 still gives the
# particles physical self-collision and contact with the hollow bottle wall,
# target bowl, attached cap, and an invisible table-coincident support surface.
# The solid bottle exterior remains visible only to the default robot/world
# category.  No hand, arm, table, or global solver parameter is changed.
DEFAULT_CONTACT_CATEGORY = 1
BOTTLE_INTERIOR_CONTACT_CATEGORY = 2
BOTTLE_EXTERIOR_CONTACT_CATEGORY = 4
PARTICLE_CONTACT_CATEGORY = 8

# ---- detachable screw: pure revolute while attached, then a free-cap twin ----
# The attached cap has exactly one DOF. Its axial position is a hard kinematic
# consequence of the revolute joint, so pulling cannot raise it before the
# half-turn detach threshold. Rotation resistance is intentionally kept in one
# task-mechanism constant because VR hand feel needs empirical tuning.  The
# deployed mechanism uses the same resistance before and after seal break.
SCREW_THREAD_FRICTION = 0.40       # N*m running thread torque.
SCREW_BREAKAWAY_FACTOR = 1.0       # sealed first-crack: also 0.40 N*m.
SCREW_SEAL_SLIP = 0.05             # rad
SCREW_SPIN_FREE_FRICTION = 0.01    # N*m once unscrewed: a loose cap DRAGS on
                                   # the dead thread, it does not freewheel
# A seated cap cannot be tightened indefinitely. Keep a small real-world
# seating/backlash allowance, then let the revolute's hard lower limit carry
# clockwise load without a spring drive or per-step state projection.
SCREW_TIGHTEN_BACKLASH = np.deg2rad(2.0)
# The detached twin keeps collision enabled from scene construction onward.
# Parking it above the workspace (with gravity disabled) avoids both ground
# contact and the unreliable "turn collision back on while already overlapping
# the hand" path. Mirror reads this same task-declared parking coordinate.
CAP_TWIN_PARK_Z = 5.0
# Long-task bottle representations are all created at scene-build time. An
# inactive representation is parked far above/outside the workspace instead of
# toggling collision filters on a body already overlapping the hand.
BOTTLE_DYNAMIC_PARK_P = (6.0, 0.0, 6.0)
BOTTLE_FIXED_PARK_P = (8.0, 0.0, 6.0)
BOTTLE_OPEN_PARK_P = (10.0, 0.0, 6.0)
BOTTLE_REP_DYNAMIC = 0
BOTTLE_REP_FIXED = 1
BOTTLE_REP_OPEN = 2
# The articulated cap, detached twin, and free-cap subtask actor must be the
# same physical object across the representation switch.
CAP_MASS = 0.005
CAP_INERTIA = (1.081e-06, 1.081e-06, 1.648e-06)
CAP_CMASS = (0.0, 0.0, 0.0)

# Optional pre-built stage lock used by the long-task experiment.  These are
# TASK-MECHANISM values, not robot/hand/environment parameters.  A fixed world
# anchor and six one-DOF URDF joints exist from scene construction onward.
# Locking changes those joints from broad finite limits to narrow limits at the
# measured resting pose.  There is deliberately NO spring drive: a high-k drive
# stores contact error as energy and kicks both the bottle and the touching hand
# when the placement stage changes.
BOTTLE_BASE_LINEAR_LIMIT = 5.0
BOTTLE_BASE_ANGULAR_LIMIT = 100.0
BOTTLE_LOCK_LINEAR_EPS = 2.5e-5          # 0.025 mm solver allowance
BOTTLE_LOCK_ANGULAR_EPS = np.deg2rad(0.02)

# A placement transition is a physical set-down, not a centre-height test.  The
# complete collision support ring must lie on the z=0 tabletop, the bottle must
# be nearly vertical and quiet, and all conditions must persist at control rate.
BOTTLE_BOTTOM_VERTEX_BAND = 2.5e-4       # bottom-ring vertices within 0.25 mm
BOTTLE_BOTTOM_MIN_Z = -5.0e-4            # at most 0.5 mm numerical penetration
BOTTLE_BOTTOM_MAX_Z = 7.5e-4             # no point of the base >0.75 mm above table
BOTTLE_PLACE_TILT_COS = float(np.cos(np.deg2rad(1.0)))
BOTTLE_PLACE_LINEAR_SPEED = 0.015         # m/s
BOTTLE_PLACE_ANGULAR_SPEED = 0.08         # rad/s
BOTTLE_PLACE_HOLD_STEPS = 5               # 250 ms at frozen 20 Hz control
DOOR_OPEN_HOLD_STEPS = 5                 # 250 ms
CAP_TURN_SWITCH_HOLD_STEPS = 3           # 150 ms pause after reaching pi
CAP_TURN_SWITCH_MAX_QVEL = 0.5            # rad/s before attached -> free swap
CAP_BOTTOM_MIN_Z = -5.0e-4               # at most 0.5 mm table penetration
CAP_BOTTOM_MAX_Z = 1.5e-3                # no cap point >1.5 mm above table
CAP_PLACE_HOLD_STEPS = 5                 # 250 ms, including hand clearance
# Pairwise *net* contact force can cancel to zero while a dexterous hand still
# cages the cap. Final placement therefore also requires every physical hand
# link origin to be this far from the cap centre. The cap radius is 25.8 mm;
# 65 mm is a deliberate, observable withdrawal rather than a force heuristic.
CAP_HAND_CLEARANCE = 0.065                # m
POUR_HOLD_STEPS = 10                     # 500 ms with no airborne particles
# Cap visual markers: the knurling is rotationally symmetric — without a
# marker the operator cannot SEE screw progress. Red top stripe + side tick.
# Markers are INLAID (never proud of the collision surface): a visual-only
# box poking past the solid hull buries itself in the table when the cap
# rests on that side (measured -0.88 mm; the "arc sunk into the tabletop").
CAP_MARKER_BOXES = [
    dict(half_size=[0.0128, 0.0020, 0.0012],
         p=[0.0129, 0.0, 0.0115], color=[0.85, 0.08, 0.08, 1.0]),
    dict(half_size=[0.0015, 0.0030, 0.0080],
         p=[0.0242, 0.0, 0.0000], color=[0.85, 0.08, 0.08, 1.0]),
]

WORK_TABLE_X_OFFSET = 0.30
PARTICLE_SUPPORT_HALF_THICKNESS = 0.001

BOTTLE_ASSET_DIR = TASK_ASSET_DIR / "bottle_4403"
# Exact task-object configuration loaded by the 11:09 reproduction process.
# Restored from the 10:52 bytecode snapshot after the rollback was interrupted,
# so the source remains coherent while the regression is audited.
BOTTLE_COLLISION_GLOB = "body_col_profiled/*.obj"
BOTTLE_HAND_COLLISION_GLOB = "body_col_hand/*.obj"
BOTTLE_WALL_THICKNESS = 0.01
BOTTLE_COLLISION_DENSITY = 275.929057
PARTICLE_PACK_CENTER_RADIUS = 0.009
SOURCE_INSIDE_CENTER_RADIUS = 0.017
BOTTLE_MAX_LINEAR_VELOCITY = 0.8
PARTICLE_MAX_LINEAR_VELOCITY = 2.5
BOTTLE_ROOT_MASS = 0.127633214
# Task layer: explicit mass statement for the physical-parameter audit.
# These are the values in force, not new numbers:
# the bowl and the rigid "liquid" spheres sit at rho ~1000 by design.
TASK_OBJECT_MASS = {
    "target_container": 0.44099050760269165,  # kg, ceramic bowl (exact value in force)
    "source_container_open": BOTTLE_ROOT_MASS,
    "source_cap_twin": CAP_MASS,
    "particle": 0.0002680826000869274,  # kg each, 16 water-density spheres (exact)
}
BOTTLE_ROOT_CMASS = (-0.0005, -0.0005, -0.012)
BOTTLE_ROOT_INERTIA = (0.000137, 0.000136, 3.77e-05)

# Legacy constants imported by ``drawer_plug.py``. CabinetPour itself supports
# only the mobile-base agents below and dispatches grasp checks by concrete
# agent type.
GRIPPER_ROBOT_UIDS = {"panda"}
GRIPPER_AGENT_TYPES = (PandaGripperLeftMobileBase, PandaGripperRightMobileBase, PandaDualGripperMobileBase)

SUPPORTED_CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")
DEFAULT_CONTROL_MODE = "pd_joint_pos"

DEFAULT_ARM_INIT_QPOS = [-0.5, -0.2, 0.0, -2.6, 0.0, 2.4, np.pi / 4]
DEFAULT_GRIPPER_OPEN_QPOS = [0.04, 0.04]
# These two are keyed on TROLLEY FOOTPRINT, not on arm count -- see
# CabinetPouringLongEnv._robot_base_pose. (Renamed from ROBOT_BASE_POSE /
# DUAL_ROBOT_BASE_POSE, which had stopped describing what selects them: the sharpa/wuji
# SINGLE-arm mobile bases moved onto the dual robot's full-size trolley, so "dual" is no
# longer the same question as "full-size".)
#
# layout fix: -1.0 -> -0.8 (single-arm). At -1.0 the closed door
# handle sat 1.06m from the base (beyond Panda reach across all seeds;
# full-waypoint audit: door 36/39, pick 30/43, pour 59/71 solvable). At -0.8
# every door and pick waypoint solves and the COMPACT trolley
# (0.75-scale) clears the work table by +7.0cm (geometric AABB check — contact
# forces are BLIND here: fixed-root links vs a static table generate no
# contact pairs, so never use them to test interpenetration).
#
# The FULL-size trolley is 10.2cm longer: at -0.8 it overlapped the
# table by 3.1cm, so it gets its own pose (-0.87 -> +3.9cm clearance).
#
# Compact x = -0.76: workspace calibration (kinematic radial profile of
# the teleop IK, fixed palm-down orientation) put the reachability cliff
# along the shoulder->bottle direction at r ~ 0.79-0.80 m while the bottle
# sat at r = 0.870 m. The 15 cm compaction is split over three levers so no
# single one distorts the scene: trolley +4 cm here (table clearance
# 7 cm -> 3 cm), cabinet forward 8 cm, bottle 3 cm toward the shelf front
# (see cabinet_xy / source_local_offset). Result: bottle r = 0.725 m,
# ~7 cm inside the cliff. The full-size pose keeps -0.87 (only 3.9 cm of
# clearance left; it inherits the 11 cm cabinet+bottle share).
COMPACT_TROLLEY_BASE_POSE = sapien.Pose(p=[-0.76, 0.0, 0.0])
FULL_TROLLEY_BASE_POSE = sapien.Pose(p=[-0.87, 0.0, 0.0])
# An agent's `mobile_base_horizontal_scale` at or above this is a full-size trolley.
FULL_TROLLEY_SCALE = 1.0


def yaw_about_z(q: torch.Tensor) -> torch.Tensor:
    """Signed twist angle about local Z, ignoring any swing component."""
    return 2.0 * torch.atan2(q[..., 3], q[..., 0])


def quaternion_about_z(theta: torch.Tensor) -> torch.Tensor:
    half = theta / 2
    zeros = torch.zeros_like(theta)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + np.pi, 2 * np.pi) - np.pi


def sample_disc_xy(center, radius: float, batch_size: int) -> torch.Tensor:
    """``batch_size`` xy points spread uniformly over a disc, shape (b, 2).

    Used by the single-stage subtasks in ``cabinet_pour_*.py`` to start the
    bottle anywhere the placement stage of this task accepts it.
    """
    angle = torch.rand(batch_size) * (2 * np.pi)
    # sqrt keeps the samples uniform per unit area rather than per unit radius.
    radial = radius * torch.sqrt(torch.rand(batch_size))
    return torch.stack(
        [center[0] + radial * torch.cos(angle), center[1] + radial * torch.sin(angle)],
        dim=-1,
    )


def link_piece_bounds(link, relative_to: sapien.Pose = None) -> np.ndarray:
    """AABBs, shape (num_pieces, 2, 3), of a link's convex pieces, in piece order.

    World-frame by default; pass ``relative_to`` (a pose) to get bounds expressed
    in that frame instead -- used to identify the handle in the cabinet's own
    frame, where a yaw of the whole cabinet leaves the piece order and local axes
    unchanged.  ``Link`` exposes no collision-mesh accessor, so this reaches
    through to the underlying SAPIEN component, but takes the link's pose from
    the sim structs: the GPU backend never writes the component's CPU-side copy,
    which is still in the raw URDF frame there.  Only ever called on an unmerged
    articulation.
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
        mesh = trimesh.Trimesh(vertices=shape.vertices * shape.scale, faces=shape.triangles)
        mesh.apply_transform(transform @ shape.local_pose.to_transformation_matrix())
        bounds.append(mesh.bounds)
    return np.array(bounds)


@register_task(
    "CabinetPouringLong",
    side="both",
    max_episode_steps=3400,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
    cap_material_friction=0.8,
    cap_running_torque=0.40,
    cap_breakaway_torque=0.40,
    cap_joint_armature=5e-5,
    bottle_stage_twins=True,
)
class CabinetPouringLongEnv(BaseEnv):
    cap_twin_park_z = CAP_TWIN_PARK_Z
    # Expose the representation parking poses to render-only replicas.  The
    # VR pose protocol intentionally streams poses, not task-specific stage
    # flags, so the mirror uses these authoritative task constants to infer
    # which prebuilt A/B/C representation is active.
    bottle_dynamic_park_p = BOTTLE_DYNAMIC_PARK_P
    bottle_fixed_park_p = BOTTLE_FIXED_PARK_P
    bottle_open_park_p = BOTTLE_OPEN_PARK_P
    SUPPORTED_ROBOTS = [
        "panda_gripper_left_mobile_base",
        "panda_gripper_right_mobile_base",
        "panda_sharpa_hand_right_mobile_base",
        "panda_sharpa_hand_right_angled_mobile_base",
        "panda_wuji_hand_right_angled_mobile_base",
        # Left-arm counterparts. Standard stages select the angled variant through
        # DEFAULT_ROBOT_UID_LEFT; straight remains an explicit compatibility option.
        "panda_sharpa_hand_left_mobile_base",
        "panda_sharpa_hand_left_angled_mobile_base",
        "panda_wuji_hand_left_angled_mobile_base",
        "panda_dual_gripper_mobile_base",
        "panda_dual_sharpa_hand_mobile_base",
        "panda_dual_sharpa_hand_angled_mobile_base",
        "panda_dual_wuji_hand_angled_mobile_base",
            "panda_xhand_hand_left_angled_mobile_base",
        "panda_xhand_hand_right_angled_mobile_base",
        "panda_dual_xhand_hand_angled_mobile_base",
]
    SUPPORTED_CONTROL_MODES = SUPPORTED_CONTROL_MODES
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

    # -- assets ------------------------------------------------------------ #
    source_scale = 1.3
    target_category, target_index, target_scale = "bowl", 1, 1.5

    # Frozen liquid proxy: 16 rigid particles, with 12/16 required in the bowl.
    num_particles = 16
    particle_radius = 0.004

    # Which rigid pieces the scene holds.  The single-stage subtasks in
    # ``cabinet_pour_*.py`` switch off whatever their stage never touches; the
    # particles live inside the bottle and follow ``has_bottle``.
    has_cabinet = True
    has_bottle = True
    has_cap = True
    has_target = True

    placement_region = "sequence"
    # The long task fixes the bottle root after the bottle-placement milestone
    # so cap manipulation cannot drag or tip it. Subtask subclasses set this
    # explicitly because they are independently staged environments: most turn it
    # off, but cap_turn keeps it ON -- that lock is what lets ONE arm unscrew the
    # cap (see _capture_bottle_locks and cabinet_pour_cap_turn.py).
    lock_bottle_during_cap_stages = True
    # The screw coupling is only worth enforcing while the cap can still turn.
    enforce_cap_screw = True
    # Envs whose episode STARTS with the cap already unscrewed (cap_pick_place)
    # set this True: the cap is then a plain free actor resting on the neck.
    cap_free_start = False

    @property
    def cap_articulated(self) -> bool:
        return bool(self.has_cap and self.enforce_cap_screw
                    and not self.cap_free_start)

    @property
    def cap_screw_dynamic(self) -> bool:
        """Whether the attached cap has a physical screw coordinate."""
        return bool(self.cap_articulated and not self.cap_contact_static_fixture)

    # -- layout (table top at z = 0, single-arm robot root at x = -0.76) ---- #
    cabinet_scale = 0.5
    # Cabinet 8 cm toward the robot (0.22 -> 0.14) as part of the 15 cm
    # workspace compaction (see COMPACT_TROLLEY_BASE_POSE): front face -0.08 -> -0.16 on
    # a 1.21 m deep table, still 26 cm from the near edge. Overridable per-env
    # via the cabinet_xy kwarg.
    cabinet_xy = (0.14, 0.12)
    cabinet_yaw = np.deg2rad(15.0)

    # Bottle 3 cm toward the shelf front (-0.18 -> -0.21, cabinet frame):
    # the last of the three compaction levers. Keeps 9.5 cm of finger room
    # between bottle center and the compartment mouth.
    source_local_offset = (-0.21, -0.005)

    place_center_xy = (-0.22, -0.20)
    place_radius = 0.15
    # Keep the detached cap beside the fixed bottle and inside the right-arm
    # palm-down workspace.  The old x=0.12 target was ~0.99 m from the right
    # shoulder on the full-size trolley, beyond the calibrated ~0.79--0.80 m
    # reach cliff.  x=-0.10 is ~0.77 m from that shoulder and leaves 12 cm
    # centre-to-centre clearance from the bottle target for hand access.
    cap_place_center_xy = (-0.10, -0.20)
    cap_place_radius = 0.10
    target_center_xy = (-0.2, -0.5)

    # -- success criteria --------------------------------------------------- #
    door_success_angle = np.deg2rad(80.0)
    pour_ratio_threshold = 0.75
    max_spilled = 2
    upright_cos_threshold = 0.9
    source_static_threshold = 0.05
    cap_static_threshold = 0.05
    cap_angular_static_threshold = 0.5
    success_hold_steps = 5
    interior_radius_ratio = 0.85

    def __init__(
        self,
        *args,
        robot_uids=DEFAULT_DUAL_ROBOT_UID,
        robot_init_qpos_noise=0.02,
        cabinet_init_xy_noise=0.01,
        object_init_xy_noise=0.01,
        num_particles=None,
        cabinet_xy=None,
        source_local_offset=None,
        cap_material_friction=None,
        cap_running_torque=None,
        cap_breakaway_torque=None,
        cap_joint_armature=None,
        cap_contact_static_fixture=False,
        cap_contact_fixed_root=False,
        prebuilt_bottle_lock=False,
        bottle_stage_twins=False,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.cabinet_init_xy_noise = float(cabinet_init_xy_noise)
        self.object_init_xy_noise = float(object_init_xy_noise)
        # Task-object-only A/B knob. ``None`` preserves the calibrated
        # hard-plastic engine coefficient (0.8), which combines by average
        # with the frozen Sharpa pad coefficient 1.2 for an effective pair
        # friction of 1.0. A value of 1.2 therefore produces effective 1.2
        # without touching the bottle, robot, hand, arm, table, or solver.
        if cap_material_friction is None:
            self.cap_material_friction = None
        else:
            cap_material_friction = float(cap_material_friction)
            if not np.isfinite(cap_material_friction) or cap_material_friction < 0.0:
                raise ValueError("cap_material_friction must be finite and non-negative")
            self.cap_material_friction = cap_material_friction
        # Task-mechanism-only screw torque knobs. Keep their defaults exactly
        # equal to the deployed constants; explicit kwargs exist so teleop A/B
        # runs can vary the cap without touching the hand/arm/environment plant.
        if cap_running_torque is None:
            cap_running_torque = SCREW_THREAD_FRICTION
        cap_running_torque = float(cap_running_torque)
        if not np.isfinite(cap_running_torque) or cap_running_torque < 0.0:
            raise ValueError("cap_running_torque must be finite and non-negative")
        if cap_breakaway_torque is None:
            cap_breakaway_torque = (
                cap_running_torque * SCREW_BREAKAWAY_FACTOR
            )
        cap_breakaway_torque = float(cap_breakaway_torque)
        if not np.isfinite(cap_breakaway_torque) or cap_breakaway_torque < 0.0:
            raise ValueError(
                "cap_breakaway_torque must be finite and non-negative"
            )
        if cap_breakaway_torque < cap_running_torque:
            raise ValueError(
                "cap_breakaway_torque must be >= cap_running_torque"
            )
        self.cap_running_torque = cap_running_torque
        self.cap_breakaway_torque = cap_breakaway_torque
        # Diagnostic-only reflected inertia on the cap revolute. Zero keeps
        # the deployed mechanism unchanged; a positive value is applied once
        # when the articulation is built, never by per-step state projection.
        if cap_joint_armature is None:
            cap_joint_armature = 0.0
        cap_joint_armature = float(cap_joint_armature)
        if not np.isfinite(cap_joint_armature) or cap_joint_armature < 0.0:
            raise ValueError("cap_joint_armature must be finite and non-negative")
        self.cap_joint_armature = cap_joint_armature
        # Diagnostic-only task-object fixture. This changes the generated cap
        # joint from revolute to fixed and fixes the bottle root at build time,
        # so contact telemetry can isolate finger/cap behaviour from all screw
        # drive, joint-limit and bottle-root projection effects. Normal task
        # environments keep the deployed mechanism because the default is False.
        self.cap_contact_static_fixture = bool(cap_contact_static_fixture)
        self.cap_contact_fixed_root = bool(cap_contact_fixed_root)
        self.bottle_stage_twins = bool(bottle_stage_twins)
        if self.cap_contact_static_fixture and self.cap_contact_fixed_root:
            raise ValueError(
                "cap_contact_static_fixture and cap_contact_fixed_root are "
                "mutually exclusive diagnostic mechanisms"
            )
        if (
            self.cap_contact_static_fixture or self.cap_contact_fixed_root
        ) and prebuilt_bottle_lock:
            raise ValueError(
                "cap contact fixtures and prebuilt_bottle_lock are "
                "mutually exclusive diagnostic mechanisms"
            )
        # Experimental task-layer mechanism. False preserves the currently
        # deployed floating-root + projection implementation byte-for-byte.
        self.prebuilt_bottle_lock = bool(prebuilt_bottle_lock)
        if num_particles is not None:
            self.num_particles = int(num_particles)
        # Layout overrides (defaults come from the workspace-calibrated class
        # attributes; see the layout block above).
        if cabinet_xy is not None:
            self.cabinet_xy = (float(cabinet_xy[0]), float(cabinet_xy[1]))
        if source_local_offset is not None:
            self.source_local_offset = (
                float(source_local_offset[0]),
                float(source_local_offset[1]),
            )
        if self.bottle_stage_twins and (
            self.cap_contact_static_fixture
            or self.cap_contact_fixed_root
            or prebuilt_bottle_lock
        ):
            raise ValueError(
                "bottle_stage_twins is mutually exclusive with the legacy "
                "cap/root fixtures and prebuilt_bottle_lock"
            )
        if self.bottle_stage_twins and not self.cap_articulated:
            raise ValueError(
                "bottle_stage_twins requires an articulated bottle cap"
            )
        if not CABINET_URDF_PATH.exists():
            raise FileNotFoundError(f"Cabinet URDF not found: {CABINET_URDF_PATH}")
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
        # Unified environment-layer profile (tasks/config/sim_profiles.py):
        # 200Hz, solver 24/8, contact_offset 1mm, determinism on. Only spacing
        # and the GPU buffer capacities are task-specific (the particle swarm
        # needs the larger contact/patch buffers). NOTE: this task was authored
        # against the 15/1 + 2cm defaults; the scripted demos were re-validated
        # under the profile at integration time.
        return make_sim_config(
            spacing=3,
            max_rigid_contact_count=2**22,
            max_rigid_patch_count=2**21,
        )

    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(eye=[-0.75, -0.6, 0.6], target=[-0.05, -0.05, 0.2])
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
        pose = sapien_utils.look_at(
            eye=[-1.55, -1.75, 1.30], target=[-0.25, 0.0, 0.10]
        )
        return CameraConfig("render_camera", pose, 1280, 720, 1.15, 0.01, 100)

    @property
    def _robot_base_pose(self):
        """Per-trolley base pose: the full-size trolley is 10.2 cm longer than the
        compact one and needs the further-back spot. Used at load AND at every episode
        reset (the reset path re-sets the root pose).

        Selected by the agent's declared ``mobile_base_horizontal_scale``, NOT by
        ``"dual" in uid`` as before: now the sharpa/wuji single-arm mobile
        bases ride the dual robot's full-size trolley (arm at y = +-0.22), so arm count
        no longer predicts footprint. Under the old test they silently interpenetrated
        the work table by ~7 cm -- silently because fixed-root links against a static
        table generate no contact pairs, so nothing raises. Putting them on the full-size
        pose is also what makes a single-arm subtask compose into the long-horizon dual
        task: same trolley pose AND same shoulder position means the same reach envelope
        over the scene.

        A missing attribute defaults to FULL (1.0), the safe side: under-estimating the
        footprint interpenetrates silently, over-estimating only costs reach. Every
        registered agent declares 1.0 (the full-size trolley).
        """
        uid = self.robot_uids if isinstance(self.robot_uids, str) else ""
        entry = REGISTERED_AGENTS.get(uid)
        scale = getattr(entry.agent_cls, "mobile_base_horizontal_scale", 1.0) if entry else 1.0
        return (
            COMPACT_TROLLEY_BASE_POSE
            if scale < FULL_TROLLEY_SCALE
            else FULL_TROLLEY_BASE_POSE
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, self._robot_base_pose)

    @property
    def source_start_xy(self):
        """World xy the bottle starts at, i.e. ``source_local_offset`` rotated by the
        cabinet yaw and shifted onto the cabinet."""
        c, s = np.cos(self.cabinet_yaw), np.sin(self.cabinet_yaw)
        ox, oy = self.source_local_offset
        return (self.cabinet_xy[0] + c * ox - s * oy,
                self.cabinet_xy[1] + s * ox + c * oy)

    # ------------------------------------------------------------------ #
    # scene
    # ------------------------------------------------------------------ #
    def _load_scene(self, options: dict):
        self._load_table()
        if self.has_bottle:
            self._load_particle_support()
        if self.placement_region is not None:
            self._load_placement_region_marker()
        if self.has_cabinet:
            self._load_cabinet()
        if self.has_bottle:
            self._load_bottle_and_cap()
        if self.has_target:
            self._load_target()
        if self.has_bottle:
            self._load_particles()

    def _load_table(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._set_work_table_pose()

    def _load_particle_support(self):
        """Build a collision-only tabletop used exclusively by particles.

        Particles cannot share the default table category without also hitting
        every default robot link. This thin static box exactly overlays the
        real tabletop, has no visual, and exposes only particle affinity. It
        therefore preserves visible spill/rest behavior while keeping the
        rigid liquid proxy dynamically invisible to the hand and arm.
        """
        builder = self.scene.create_actor_builder()
        half_x = float(self.table_scene.table_length) / 2
        half_y = float(self.table_scene.table_width) / 2
        builder.add_box_collision(
            half_size=[half_x, half_y, PARTICLE_SUPPORT_HALF_THICKNESS],
            material=sapien.physx.PhysxMaterial(0.0, 0.0, 0.0),
        )
        builder.initial_pose = sapien.Pose(
            p=[
                -0.12 + WORK_TABLE_X_OFFSET,
                0.0,
                -PARTICLE_SUPPORT_HALF_THICKNESS,
            ]
        )
        self.particle_support = builder.build_static(name="particle_table_support")
        for raw in self.particle_support._objs:
            component = self._raw_collision_component(raw)
            for shape in component.get_collision_shapes():
                groups = list(shape.get_collision_groups())
                groups[0] = 0
                groups[1] = PARTICLE_CONTACT_CATEGORY
                shape.set_collision_groups(groups)

    def _load_placement_region_marker(self):
        """Build collision-free markers for the configured placement region(s)."""
        # Bound before the opt-out so _sync_placement_region_markers() stays a
        # no-op loop instead of an AttributeError when the cues are disabled.
        self.placement_region_markers = {}
        self.placement_region_marker = None
        if not self.show_guidance_markers:
            return
        regions = {
            "bottle": (self.place_center_xy, self.place_radius),
            "cap": (self.cap_place_center_xy, self.cap_place_radius),
        }
        region_names = (
            ("bottle", "cap")
            if self.placement_region == "sequence"
            else (self.placement_region,)
        )
        if any(name not in regions for name in region_names):
            raise ValueError(
                f"Unknown cabinet-pour placement region: {self.placement_region!r}"
            )

        marker_material = sapien.render.RenderMaterial(
            base_color=[0.05, 0.80, 0.10, 1.0], roughness=0.8
        )
        for region_name in region_names:
            center_xy, radius = regions[region_name]
            builder = self.scene.create_actor_builder()
            # SAPIEN cylinders point along local X; Ry(-90 deg) makes this a
            # thin horizontal disc whose lower face clears tabletop z=0.
            builder.add_cylinder_visual(
                pose=sapien.Pose(
                    p=[0.0, 0.0, 0.0006],
                    q=[0.7071068, 0.0, -0.7071068, 0.0],
                ),
                radius=radius,
                half_length=0.0005,
                material=marker_material,
            )
            builder.initial_pose = sapien.Pose(p=[center_xy[0], center_xy[1], 0.0])
            marker = builder.build_kinematic(name=f"{region_name}_placement_region")
            self.remove_from_state_dict_registry(marker)
            self.placement_region_markers[region_name] = marker

        if len(region_names) == 1:
            self.placement_region_marker = self.placement_region_markers[
                region_names[0]
            ]

    def _sync_placement_region_markers(self, env_idx: torch.Tensor = None):
        """Show the full task's current goal independently in every sub-env."""
        if self.placement_region != "sequence":
            return
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        batch_size = len(env_idx)
        identity_q = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        ).expand(batch_size, -1)
        visibility = {
            # Stage 2 becomes available after the door has been opened once.
            # _stage_door is a latch, so closing the door later neither hides
            # this target nor invalidates bottle placement.
            "bottle": (
                self._stage_door[env_idx]
                & ~self._stage_bottle_placed[env_idx]
            ),
            "cap": (
                self._stage_cap_turned[env_idx]
                & ~self._stage_cap_placed[env_idx]
            ),
        }
        desired_visibility = torch.stack(
            [visibility["bottle"], visibility["cap"]], dim=-1
        ).to(torch.int8)
        if not bool(
            (self._placement_marker_visibility[env_idx] != desired_visibility)
            .any()
            .item()
        ):
            return
        regions = {
            "bottle": self.place_center_xy,
            "cap": self.cap_place_center_xy,
        }
        for region_name, marker in self.placement_region_markers.items():
            center_xy = regions[region_name]
            xyz = torch.zeros(
                (batch_size, 3), dtype=torch.float32, device=self.device
            )
            xyz[:, 0] = center_xy[0]
            xyz[:, 1] = center_xy[1]
            # Kinematic and collision-free: parking an inactive marker below
            # the table changes neither task physics nor observations.
            xyz[:, 2] = torch.where(
                visibility[region_name],
                torch.zeros(batch_size, device=self.device),
                torch.full((batch_size,), -10.0, device=self.device),
            )
            marker.set_pose(Pose.create_from_pq(p=xyz, q=identity_q))
        self._placement_marker_visibility[env_idx] = desired_visibility
        # evaluate() runs after the physics step. Push a stage-transition pose
        # immediately so the very next rendered frame shows the new target.
        if self.gpu_sim_enabled and self.scene._gpu_sim_initialized:
            self.scene.px.gpu_apply_rigid_dynamic_data()
            self.scene.px.gpu_fetch_rigid_dynamic_data()

    def _load_cabinet(self):
        self._cabinet_pose = sapien.Pose(
            p=[self.cabinet_xy[0], self.cabinet_xy[1], 0.0],
            q=[np.cos(self.cabinet_yaw / 2), 0.0, 0.0, np.sin(self.cabinet_yaw / 2)],
        )
        cabinet_builder = self._create_cabinet_builder()
        self._cabinets: list[Articulation] = []
        for scene_idx in range(self.num_envs):
            cabinet_builder.set_scene_idxs([scene_idx])
            cabinet_builder.initial_pose = self._cabinet_pose
            cabinet = cabinet_builder.build(name=f"cabinet-{scene_idx}")
            self._configure_cabinet(cabinet)
            self.remove_from_state_dict_registry(cabinet)
            self._cabinets.append(cabinet)
        self.cabinet = Articulation.merge(self._cabinets, name="cabinet", merge_links=True)
        self.add_to_state_dict_registry(self.cabinet)
        self.door_link = self.cabinet.links_map["link_0"]

    @staticmethod
    def _set_task_object_motion_limits(obj, max_linear_velocity: float):
        """Cap solver-created velocity only on a CabinetPour task object."""
        for raw in obj._objs:
            comp = raw
            if not hasattr(comp, "set_max_depenetration_velocity"):
                comp = raw.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                )
            if comp is None:
                continue
            comp.set_max_depenetration_velocity(0.5)
            comp.set_max_linear_velocity(max_linear_velocity)

    @staticmethod
    def _set_source_mass_properties(source):
        """Apply the filled-bottle aggregate mass to a free source actor."""
        for raw in source._objs:
            comp = raw.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            if comp is None:
                continue
            comp.set_mass(BOTTLE_ROOT_MASS)
            comp.set_inertia(np.asarray(BOTTLE_ROOT_INERTIA, dtype=np.float32))
            comp.set_cmass_local_pose(sapien.Pose(p=BOTTLE_ROOT_CMASS))

    @staticmethod
    def _set_free_cap_mass_properties(cap):
        """Match a free cap/twin exactly to the articulated cap inertial."""
        for raw in cap._objs:
            comp = raw.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            if comp is None:
                continue
            comp.set_mass(CAP_MASS)
            comp.set_inertia(np.asarray(CAP_INERTIA, dtype=np.float32))
            comp.set_cmass_local_pose(sapien.Pose(p=CAP_CMASS))


    @staticmethod
    def _raw_collision_component(raw):
        if hasattr(raw, "get_collision_shapes"):
            return raw
        for component_type in (
            sapien.physx.PhysxRigidDynamicComponent,
            sapien.physx.PhysxRigidStaticComponent,
            sapien.physx.PhysxArticulationLinkComponent,
        ):
            component = raw.find_component_by_type(component_type)
            if component is not None:
                return component
        raise TypeError(f"Task object {raw!r} has no PhysX collision component")

    @classmethod
    def _configure_bottle_collision_layers(cls, source):
        """Keep robot fingers off the hollow particle-wall decomposition.

        The first shapes are the 513 inward-thickened hollow cells and belong
        to category 2. The final five shapes are full-section solid exterior
        pieces with contact type 4 and affinity 1. Particles are [8, 8], so the
        hollow cells advertise affinity 8 while the solid exterior continues
        to contact only default robot/world shapes.
        """
        inner_count = len(list(BOTTLE_ASSET_DIR.glob(BOTTLE_COLLISION_GLOB)))
        outer_count = len(list(BOTTLE_ASSET_DIR.glob(BOTTLE_HAND_COLLISION_GLOB)))
        if inner_count != 513 or outer_count != 5:
            raise RuntimeError(
                "Bottle collision assets incomplete: "
                f"inner={inner_count} (expected 513), outer={outer_count} (expected 5)"
            )
        for raw in source._objs:
            component = cls._raw_collision_component(raw)
            shapes = list(component.get_collision_shapes())
            expected = inner_count + outer_count
            if len(shapes) != expected:
                raise RuntimeError(
                    f"Bottle loaded {len(shapes)} shapes; expected {expected}"
                )

            # SAPIEN preserves builder/URDF collision order. Guard that
            # assumption geometrically: every final outer piece spans the full
            # bottle cross-section in at least two local dimensions.
            for shape in shapes[-outer_count:]:
                vertices = np.asarray(shape.get_vertices(), dtype=np.float64)
                spans = np.sort(np.ptp(vertices, axis=0))
                if len(spans) != 3 or spans[-2] < 0.03:
                    raise RuntimeError(
                        "Bottle hand-contact shapes were not loaded last/full-section"
                    )

            for shape in shapes[:inner_count]:
                groups = list(shape.get_collision_groups())
                groups[0] = BOTTLE_INTERIOR_CONTACT_CATEGORY
                groups[1] = PARTICLE_CONTACT_CATEGORY
                shape.set_collision_groups(groups)
            for shape in shapes[-outer_count:]:
                groups = list(shape.get_collision_groups())
                groups[0] = BOTTLE_EXTERIOR_CONTACT_CATEGORY
                groups[1] = DEFAULT_CONTACT_CATEGORY
                shape.set_collision_groups(groups)

    def _load_bottle_and_cap(self):
        with (BOTTLE_ASSET_DIR / "meta.json").open() as f:
            bottle_meta = json.load(f)
        bottle_scale = float(self.source_scale)
        self.cap_mount_z = float(bottle_meta["rot_pivot"][1]) * bottle_scale
        self.cap_success_rotation = np.pi

        bottle_mesh_pose = sapien.Pose(
            q=[np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
        )
        # Hard-plastic class from the object friction table (mu_obj = 0.8 ->
        # effective pad contact mu = 1.0, the real silicone-on-PET pair). The
        # original 3.0/3.0 was ~2x above the real pair value.
        bottle_material = object_material("hard_plastic")
        cap_material = bottle_material
        if self.cap_material_friction is not None:
            cap_material = sapien.physx.PhysxMaterial(
                self.cap_material_friction,
                self.cap_material_friction,
                0.0,
            )
        source_initial_xyz = np.array([*self.source_start_xy, 0.6])
        if not self.has_cap:
            self.source = self._build_mesh_asset(
                asset_dir=BOTTLE_ASSET_DIR,
                name="source_container",
                visual_glob="body_vis/*.obj",
                collision_glob=BOTTLE_COLLISION_GLOB,
                extra_collision_globs=(BOTTLE_HAND_COLLISION_GLOB,),
                scale=self.source_scale,
                density=BOTTLE_COLLISION_DENSITY,
                material=bottle_material,
                mesh_pose=bottle_mesh_pose,
                initial_pose=sapien.Pose(p=source_initial_xyz),
            )
            self._configure_bottle_collision_layers(self.source)
            self._set_source_mass_properties(self.source)
            self._set_task_object_motion_limits(
                self.source, BOTTLE_MAX_LINEAR_VELOCITY
            )
            return
        if self.cap_articulated:
            self._build_bottle_screw_articulation(
                source_initial_xyz,
                bottle_material,
                cap_material,
                bottle_mesh_pose,
            )
        else:
            # Free-cap variant (cap starts DETACHED, e.g. cap_pick_place): the
            # bottle and cap are plain actors; the extended ring shell rests on
            # the neck by contact + gravity alone (no drives — the SAPIEN D6
            # drive's lateral axes exert no force on CPU, measured).
            self.source = self._build_mesh_asset(
                asset_dir=BOTTLE_ASSET_DIR,
                name="source_container",
                visual_glob="body_vis/*.obj",
                collision_glob=BOTTLE_COLLISION_GLOB,
                extra_collision_globs=(BOTTLE_HAND_COLLISION_GLOB,),
                scale=self.source_scale,
                density=BOTTLE_COLLISION_DENSITY,
                material=bottle_material,
                mesh_pose=bottle_mesh_pose,
                initial_pose=sapien.Pose(p=source_initial_xyz),
            )
            self._configure_bottle_collision_layers(self.source)
            self._set_source_mass_properties(self.source)
            self._set_task_object_motion_limits(
                self.source, BOTTLE_MAX_LINEAR_VELOCITY
            )
            self.cap = self._build_mesh_asset(
                asset_dir=BOTTLE_ASSET_DIR,
                name="source_cap",
                visual_glob="lid_vis/*.obj",
                # 4403's lid is WIDER than sat on the jar mouth: a loose lid
                # simply RESTS on the rim, so the free cap is the same solid
                # convex as the articulated one (no ring/skirt needed).
                collision_glob="lid_col_solid/*.obj",
                scale=self.source_scale,
                density=97.0,
                material=cap_material,
                mesh_pose=bottle_mesh_pose,
                initial_pose=sapien.Pose(
                    p=source_initial_xyz + np.array([0.0, 0.0, self.cap_mount_z])
                ),
                marker_boxes=CAP_MARKER_BOXES,
            )
            self._set_free_cap_mass_properties(self.cap)
            for obj in self.cap._objs:
                comp = obj.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                )
                if comp is not None:
                    comp.set_max_depenetration_velocity(0.5)
                    comp.set_max_linear_velocity(3.0)

        self._cap_twist = torch.zeros(self.num_envs, device=self.device)
        self._cap_detached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._cap_free = torch.zeros_like(self._cap_detached)
        self._cap_seal_broken = torch.zeros_like(self._cap_detached)

    def _build_bottle_screw_articulation(
        self,
        source_initial_xyz,
        bottle_material,
        cap_material,
        bottle_mesh_pose,
    ):
        """Bottle + attached cap as one generated-URDF articulation:
        root(bottle) -[revolute z at the neck seat]-> cap.

        Why an articulation, and why via URDF: the D6-drive screw died on a
        measured SAPIEN CPU defect (only the first drive per body pair exerts
        anything, and only its X channel; lateral drives AND lateral limits
        are inert), and builder-assembled link joints show the same
        one-axis-only syndrome. URDF-LOADED articulations are the one joint
        path this whole repo already trusts (robots, cabinet, angled mounts),
        and their joints are fully constrained. The revolute carries the
        task-level thread resistance and hard-locks every translation, so the
        attached cap cannot be pulled up. At a half turn, the prebuilt free-cap
        twin takes over at the same pose. No joint is created at runtime, and
        the mechanism remains compatible with batched GPU simulation.
        """
        urdf_path = self._generate_bottle_screw_urdf()
        loader = self.scene.create_urdf_loader()
        loader.name = "bottle_screw"
        loader.fix_root_link = bool(
            self.prebuilt_bottle_lock
            or self.cap_contact_static_fixture
            or self.cap_contact_fixed_root
        )
        loader.load_multiple_collisions_from_file = True
        builders = loader.parse(str(urdf_path))["articulation_builders"]
        if len(builders) != 1:
            raise RuntimeError("Expected one bottle_screw articulation")
        builder = builders[0]
        if self.prebuilt_bottle_lock:
            # Keep the fixed anchor away from all task geometry during scene
            # setup. The six base-joint coordinates are relative to this pose.
            self._bottle_base_anchor_xyz = np.asarray(
                source_initial_xyz, dtype=np.float32
            )
            builder.initial_pose = sapien.Pose(p=self._bottle_base_anchor_xyz)
        else:
            builder.initial_pose = sapien.Pose(p=source_initial_xyz)
        self.bottle_screw = builder.build()
        links = {l.name: l for l in self.bottle_screw.get_links()}
        self.source = links["source_container"]
        self._configure_bottle_collision_layers(self.source)
        self.cap = links["source_cap"]
        joints = {j.name: j for j in self.bottle_screw.get_active_joints()}
        if self.cap_contact_static_fixture:
            self._cap_rev = None
            self._cap_screw_qindex = None
        else:
            self._cap_rev = joints["cap_screw_joint"]
            self._cap_screw_qindex = int(self._cap_rev.active_index[0].item())
            if self.cap_joint_armature > 0.0:
                for raw in self._cap_rev._objs:
                    if not hasattr(raw, "set_armature"):
                        raise RuntimeError(
                            "cap_joint_armature requested, but the active "
                            "physics backend does not expose set_armature"
                        )
                    raw.set_armature([self.cap_joint_armature])
        if self.prebuilt_bottle_lock:
            base_names = (
                "bottle_base_tx", "bottle_base_ty", "bottle_base_tz",
                "bottle_base_rx", "bottle_base_ry", "bottle_base_rz",
            )
            self._bottle_base_joints = [joints[name] for name in base_names]
            self._bottle_base_free_limits = []
            for joint in self._bottle_base_joints:
                self._bottle_base_free_limits.append(
                    [np.asarray(raw.get_limits(), dtype=np.float32).copy()
                     for raw in joint._objs]
                )
                for raw in joint._objs:
                    raw.set_drive_properties(
                        0.0, 0.0, force_limit=0.0, mode="force"
                    )
        # Contact materials (the loader default would be the 0.3 fallback).
        # Bottle remains frozen at hard-plastic 0.8; only cap representations
        # receive the optional task-object A/B coefficient.
        for raw in self.source._objs:
            for sh in self._raw_collision_component(raw).get_collision_shapes():
                sh.set_physical_material(bottle_material)
        for raw in self.cap._objs:
            for sh in self._raw_collision_component(raw).get_collision_shapes():
                sh.set_physical_material(cap_material)
        # The attached cap seals the particle proxy. The free twin deliberately
        # stays on default [1, 1], so once removed it cannot inject hard-body
        # impulses into the liquid through a penetrating fingertip contact.
        for sh in self.cap._objs[0].get_collision_shapes():
            groups = list(sh.get_collision_groups())
            groups[1] |= PARTICLE_CONTACT_CATEGORY
            sh.set_collision_groups(groups)
        # Solver-energy ceilings on EVERY link (same rationale as the
        # particles): a hand pressing into the numerical wall must not buy the
        # bottle an unlimited depenetration velocity — that wall-speed spike
        # is exactly what blasts the particles out of a quietly held bottle.
        for lk in self.bottle_screw.get_links():
            comp = lk._objs[0]
            comp.set_max_depenetration_velocity(0.5)
            comp.set_max_linear_velocity(
                BOTTLE_MAX_LINEAR_VELOCITY
                if lk.name == "source_container"
                else 3.0
            )
        # Keep bottle/cap contact masked explicitly. The revolute is the whole
        # attached-thread model, and speculative neck contacts must not add a
        # second, solver-sensitive axial or rotational constraint.
        for lk in (self.source, self.cap):
            lk.set_collision_group_bit(group=2, bit_idx=30, bit=1)
        # Detach twin: an ordinary free actor with the same look and solid
        # collision, parked above the scene with gravity off. At the detach
        # moment the articulated ("ghost") cap hides and the twin takes over
        # at its pose — the one-object-to-two-objects switch PhysX cannot do
        # with joints. Ghost<->twin collisions are masked statically. Twin
        # collision remains enabled for its entire lifetime; only pose and
        # gravity change at the swap, avoiding PhysX filter-cache ambiguity.
        self.cap_twin = self._build_mesh_asset(
            asset_dir=BOTTLE_ASSET_DIR,
            name="source_cap_twin",
            visual_glob="lid_vis/*.obj",
            collision_glob="lid_col_solid/*.obj",
            scale=self.source_scale,
            density=97.0,
            material=cap_material,
            mesh_pose=bottle_mesh_pose,
            initial_pose=sapien.Pose(p=[0.0, 0.0, CAP_TWIN_PARK_Z]),
            marker_boxes=CAP_MARKER_BOXES,
        )
        self._set_free_cap_mass_properties(self.cap_twin)
        for obj in self.cap_twin._objs:
            comp = obj.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            comp.set_disable_gravity(True)
            comp.set_max_depenetration_velocity(0.5)
            comp.set_max_linear_velocity(3.0)
        self.cap.set_collision_group_bit(group=2, bit_idx=31, bit=1)
        self.cap_twin.set_collision_group_bit(group=2, bit_idx=31, bit=1)
        # Unlike the attached joint cap, the loose twin must collide with the
        # bottle neck/rim. It is materialized in the same non-penetrating
        # +2 mm state as the legacy CapPick&Place start, while the hand is still
        # holding it, so no bottle-family ignore bit is needed here.
        if self.bottle_stage_twins:
            self._build_bottle_stage_twins(
                source_initial_xyz,
                bottle_material,
                cap_material,
                bottle_mesh_pose,
            )

    def _build_bottle_stage_twins(
        self,
        source_initial_xyz,
        bottle_material,
        cap_material,
        bottle_mesh_pose,
    ):
        """Prebuild the long task's fixed screw bottle (B) and open bottle (C).

        The original articulation remains representation A. B has a fixed root
        from construction time, matching the merged cap-stage subtask. C is an
        ordinary free bottle actor without a cap. Inactive
        representations stay parked with gravity disabled where applicable;
        collisions are never enabled while a body overlaps the hand.
        """
        urdf_path = self._generate_bottle_screw_urdf()
        loader = self.scene.create_urdf_loader()
        loader.name = "bottle_screw_fixed"
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        builders = loader.parse(str(urdf_path))["articulation_builders"]
        if len(builders) != 1:
            raise RuntimeError("Expected one fixed bottle_screw articulation")
        builder = builders[0]
        builder.initial_pose = sapien.Pose(p=BOTTLE_FIXED_PARK_P)
        self.bottle_screw_fixed = builder.build()
        links = {link.name: link for link in self.bottle_screw_fixed.get_links()}
        self.source_fixed = links["source_container"]
        self.cap_fixed = links["source_cap"]
        self._configure_bottle_collision_layers(self.source_fixed)

        joints = {
            joint.name: joint
            for joint in self.bottle_screw_fixed.get_active_joints()
        }
        self._cap_rev_fixed = joints["cap_screw_joint"]
        self._cap_screw_qindex_fixed = int(
            self._cap_rev_fixed.active_index[0].item()
        )
        if self.cap_joint_armature > 0.0:
            for raw in self._cap_rev_fixed._objs:
                if not hasattr(raw, "set_armature"):
                    raise RuntimeError(
                        "cap_joint_armature requested, but the fixed screw "
                        "backend does not expose set_armature"
                    )
                raw.set_armature([self.cap_joint_armature])

        for raw in self.source_fixed._objs:
            for shape in self._raw_collision_component(raw).get_collision_shapes():
                shape.set_physical_material(bottle_material)
        for raw in self.cap_fixed._objs:
            for shape in self._raw_collision_component(raw).get_collision_shapes():
                shape.set_physical_material(cap_material)
        for shape in self.cap_fixed._objs[0].get_collision_shapes():
            groups = list(shape.get_collision_groups())
            groups[1] |= PARTICLE_CONTACT_CATEGORY
            shape.set_collision_groups(groups)
        for link in self.bottle_screw_fixed.get_links():
            component = link._objs[0]
            component.set_max_depenetration_velocity(0.5)
            component.set_max_linear_velocity(
                BOTTLE_MAX_LINEAR_VELOCITY
                if link.name == "source_container"
                else 3.0
            )
        for link in (self.source_fixed, self.cap_fixed):
            link.set_collision_group_bit(group=2, bit_idx=30, bit=1)
        self.cap_fixed.set_collision_group_bit(group=2, bit_idx=31, bit=1)

        self.source_open = self._build_mesh_asset(
            asset_dir=BOTTLE_ASSET_DIR,
            name="source_container_open",
            visual_glob="body_vis/*.obj",
            collision_glob=BOTTLE_COLLISION_GLOB,
            extra_collision_globs=(BOTTLE_HAND_COLLISION_GLOB,),
            scale=self.source_scale,
            density=BOTTLE_COLLISION_DENSITY,
            material=bottle_material,
            mesh_pose=bottle_mesh_pose,
            initial_pose=sapien.Pose(p=BOTTLE_OPEN_PARK_P),
        )
        self._configure_bottle_collision_layers(self.source_open)
        self._set_source_mass_properties(self.source_open)
        self._set_task_object_motion_limits(
            self.source_open, BOTTLE_MAX_LINEAR_VELOCITY
        )
        for i, raw in enumerate(self.source_open._objs):
            component = raw.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            component.set_disable_gravity(True)
            self._set_body_visible(self.source_fixed, i, False)
            self._set_body_visible(self.cap_fixed, i, False)
            self._set_body_visible(self.source_open, i, False)

    def _generate_bottle_screw_urdf(self) -> Path:
        """Write bottle_screw.urdf next to the bottle meshes (regenerated on
        change; gitignored — derived from the meshes + meta.json).

        The articulated cap's collision is ONE SOLID convex cylinder
        (lid_col_solid) sized to the visual cap: cap<->bottle contact is
        masked in this design, so the thin ring shell (needed only by the
        free-actor cap to rest on the neck) would be pure liability — a
        fingertip punches through a 0.8 mm wall and gets WEDGED inside the
        ring (measured in VR: the operator could lift the bottle by a thumb
        stuck through the cap). A solid convex has no wall to pierce and no
        cavity to jam in."""
        def mesh_entries(subdir, tag):
            sc = float(self.source_scale)
            out = []
            for f in sorted((BOTTLE_ASSET_DIR / subdir).glob("*.obj")):
                rel = f"{subdir}/{f.name}"
                out.append(
                    f'    <{tag}><origin rpy="1.5708 0 0" xyz="0 0 0"/>'
                    f'<geometry><mesh filename="{rel}" scale="{sc} {sc} {sc}"/>'
                    f'</geometry></{tag}>'
                )
            return "\n".join(out)

        markers = "\n".join(
            f'    <visual><origin xyz="{mb["p"][0]} {mb["p"][1]} {mb["p"][2]}"/>'
            f'<geometry><box size="{2*mb["half_size"][0]} {2*mb["half_size"][1]} '
            f'{2*mb["half_size"][2]}"/></geometry>'
            f'<material name="marker_red"><color rgba="0.85 0.08 0.08 1"/></material></visual>'
            for mb in CAP_MARKER_BOXES
        )
        if self.prebuilt_bottle_lock:
            # The mesh/link origin sits about 71 mm above the bottle bottom.
            # Put all three angular axes at the bottom centre so even a force
            # beyond the finite drive limit pivots at the table fixture rather
            # than around the middle of the bottle.  Match the exact Rx(1.5708)
            # used by every bottle collision entry above.
            angle = 1.5708
            z_min = np.inf
            for mesh_path in sorted(
                (BOTTLE_ASSET_DIR / BOTTLE_COLLISION_GLOB.split("/")[0]).glob("*.obj")
            ):
                with mesh_path.open() as mesh_file:
                    for line in mesh_file:
                        if not line.startswith("v "):
                            continue
                        xyz = line.split()
                        y, z = float(xyz[2]), float(xyz[3])
                        local_z = (
                            np.sin(angle) * y + np.cos(angle) * z
                        ) * float(self.source_scale)
                        z_min = min(z_min, local_z)
            if not np.isfinite(z_min):
                raise RuntimeError("Bottle collision meshes contain no vertices")
            self._bottle_lock_pivot_z = float(z_min)
            source_origin_above_pivot = -self._bottle_lock_pivot_z
            base_links = """
  <link name="bottle_world_anchor">
    <inertial><mass value="0.0001"/>
      <inertia ixx="1e-08" iyy="1e-08" izz="1e-08" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="bottle_base_tx_link"><inertial><mass value="0.0001"/><inertia ixx="1e-08" iyy="1e-08" izz="1e-08" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="bottle_base_ty_link"><inertial><mass value="0.0001"/><inertia ixx="1e-08" iyy="1e-08" izz="1e-08" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="bottle_base_tz_link"><inertial><mass value="0.0001"/><inertia ixx="1e-08" iyy="1e-08" izz="1e-08" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="bottle_base_rx_link"><inertial><mass value="0.0001"/><inertia ixx="1e-08" iyy="1e-08" izz="1e-08" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="bottle_base_ry_link"><inertial><mass value="0.0001"/><inertia ixx="1e-08" iyy="1e-08" izz="1e-08" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="bottle_base_rz_link"><inertial><mass value="0.0001"/><inertia ixx="1e-08" iyy="1e-08" izz="1e-08" ixy="0" ixz="0" iyz="0"/></inertial></link>"""
            base_joints = f"""
  <joint name="bottle_base_tx" type="prismatic"><parent link="bottle_world_anchor"/><child link="bottle_base_tx_link"/><axis xyz="1 0 0"/><limit lower="{-BOTTLE_BASE_LINEAR_LIMIT}" upper="{BOTTLE_BASE_LINEAR_LIMIT}" effort="100" velocity="10"/></joint>
  <joint name="bottle_base_ty" type="prismatic"><parent link="bottle_base_tx_link"/><child link="bottle_base_ty_link"/><axis xyz="0 1 0"/><limit lower="{-BOTTLE_BASE_LINEAR_LIMIT}" upper="{BOTTLE_BASE_LINEAR_LIMIT}" effort="100" velocity="10"/></joint>
  <joint name="bottle_base_tz" type="prismatic"><parent link="bottle_base_ty_link"/><child link="bottle_base_tz_link"/><axis xyz="0 0 1"/><limit lower="{-BOTTLE_BASE_LINEAR_LIMIT}" upper="{BOTTLE_BASE_LINEAR_LIMIT}" effort="100" velocity="10"/></joint>
  <joint name="bottle_base_rx" type="revolute"><parent link="bottle_base_tz_link"/><child link="bottle_base_rx_link"/><axis xyz="1 0 0"/><limit lower="{-BOTTLE_BASE_ANGULAR_LIMIT}" upper="{BOTTLE_BASE_ANGULAR_LIMIT}" effort="10" velocity="30"/></joint>
  <joint name="bottle_base_ry" type="revolute"><parent link="bottle_base_rx_link"/><child link="bottle_base_ry_link"/><axis xyz="0 1 0"/><limit lower="{-BOTTLE_BASE_ANGULAR_LIMIT}" upper="{BOTTLE_BASE_ANGULAR_LIMIT}" effort="10" velocity="30"/></joint>
  <joint name="bottle_base_rz" type="revolute"><parent link="bottle_base_ry_link"/><child link="bottle_base_rz_link"/><axis xyz="0 0 1"/><limit lower="{-BOTTLE_BASE_ANGULAR_LIMIT}" upper="{BOTTLE_BASE_ANGULAR_LIMIT}" effort="10" velocity="30"/></joint>
  <joint name="bottle_base_to_source" type="fixed"><origin xyz="0 0 {source_origin_above_pivot}"/><parent link="bottle_base_rz_link"/><child link="source_container"/></joint>"""
        else:
            base_links = ""
            base_joints = ""
        if self.cap_contact_static_fixture:
            cap_joint = f"""
  <joint name="cap_screw_joint" type="fixed">
    <origin xyz="0 0 {self.cap_mount_z}" rpy="0 0 0"/>
    <parent link="source_container"/>
    <child link="source_cap"/>
  </joint>"""
        else:
            cap_joint = f"""
  <joint name="cap_screw_joint" type="revolute">
    <origin xyz="0 0 {self.cap_mount_z}" rpy="0 0 0"/>
    <parent link="source_container"/>
    <child link="source_cap"/>
    <axis xyz="0 0 1"/>
    <limit lower="{-SCREW_TIGHTEN_BACKLASH}" upper="12.5664" effort="5" velocity="30"/>
    <dynamics friction="{SCREW_THREAD_FRICTION * SCREW_BREAKAWAY_FACTOR}" damping="0.0"/>
  </joint>"""
        text = f"""<?xml version="1.0"?>
<robot name="bottle_screw">
{base_links}
  <link name="source_container">
    <inertial>
      <origin xyz="{BOTTLE_ROOT_CMASS[0]} {BOTTLE_ROOT_CMASS[1]} {BOTTLE_ROOT_CMASS[2]}"/>
      <mass value="{BOTTLE_ROOT_MASS}"/>
      <inertia ixx="{BOTTLE_ROOT_INERTIA[0]}" iyy="{BOTTLE_ROOT_INERTIA[1]}" izz="{BOTTLE_ROOT_INERTIA[2]}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
{mesh_entries("body_vis", "visual")}
{mesh_entries(BOTTLE_COLLISION_GLOB.split("/")[0], "collision")}
{mesh_entries(BOTTLE_HAND_COLLISION_GLOB.split("/")[0], "collision")}
  </link>
  <link name="source_cap">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="{CAP_MASS}"/>
      <inertia ixx="{CAP_INERTIA[0]}" iyy="{CAP_INERTIA[1]}" izz="{CAP_INERTIA[2]}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
{mesh_entries("lid_vis", "visual")}
{markers}
{mesh_entries("lid_col_solid", "collision")}
  </link>
{base_joints}
{cap_joint}
</robot>
"""
        if self.cap_contact_static_fixture:
            out = BOTTLE_ASSET_DIR / "bottle_screw_static_fixture.urdf"
        else:
            out = BOTTLE_ASSET_DIR / (
                "bottle_screw_prebuilt_lock.urdf"
                if self.prebuilt_bottle_lock else "bottle_screw.urdf"
            )
        if not out.exists() or out.read_text() != text:
            out.write_text(text)
        return out

    def _load_target(self):
        self.target = self._build_mesh_asset(
            asset_dir=ROBOCASA_TARGET_DIR,
            name="target_container",
            visual_glob="visual/model_normalized_0.obj",
            collision_glob="collision/collision_mesh_collision_*.obj",
            scale=ROBOCASA_TARGET_MESH_SCALE * self.target_scale,
            # Explicit hard-plastic class (was None -> silent 0.3 fallback).
            # The bowl is kinematic, so this only governs particle-bowl contact.
            material=object_material("hard_plastic"),
            initial_pose=sapien.Pose(p=[*self.target_center_xy, 0.5]),
            body_type="kinematic",
        )
        # Keep normal robot/bowl contact and add particle affinity without
        # putting particles back into the default robot collision category.
        for raw in self.target._objs:
            comp = self._raw_collision_component(raw)
            for shape in comp.get_collision_shapes():
                groups = list(shape.get_collision_groups())
                groups[1] |= PARTICLE_CONTACT_CATEGORY
                shape.set_collision_groups(groups)

    def _load_particles(self):
        self.particles = [
            actors.build_sphere(
                self.scene,
                radius=self.particle_radius,
                color=[0.15, 0.45, 0.9, 1.0],
                name=f"particle_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, 0.6 + 0.05 * i]),
            )
            for i in range(self.num_particles)
        ]
        # The particles stand in for water, and water is frictionless for our
        # purposes: mu = 0 explicitly (was the silent 0.3 fallback). Under the
        # measured `average` combine rule this yields particle-particle mu = 0
        # and particle-container mu = mu_container / 2 — the slipperiest contact
        # the engine can express (see docs/friction_calibration.md).
        water_mat = sapien.physx.PhysxMaterial(0.0, 0.0, 0.0)
        for particle in self.particles:
            for obj in particle._objs:
                comp = obj.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                )
                for shape in comp.get_collision_shapes():
                    shape.set_physical_material(water_mat)
                    groups = list(shape.get_collision_groups())
                    groups[0] = PARTICLE_CONTACT_CATEGORY
                    groups[1] = PARTICLE_CONTACT_CATEGORY
                    shape.set_collision_groups(groups)
                # Two physical ceilings against solver-injected energy. The
                # SAPIEN defaults are unbounded (1e32 / 1e16): a particle
                # caught mid-wall gets expelled at whatever depenetration
                # velocity closes the overlap in one substep, and mu=0 means
                # the pile has zero shear damping, so injected impulses keep
                # sloshing (normal restitution 0 is the only sink). Water in a
                # handheld bottle never exceeds a few m/s, so cap both.
                comp.set_max_depenetration_velocity(0.5)
                comp.set_max_linear_velocity(PARTICLE_MAX_LINEAR_VELOCITY)

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

    def _build_mesh_asset(
        self,
        asset_dir,
        name,
        visual_glob,
        collision_glob,
        extra_collision_globs=(),
        scale=1.0,
        density=1000.0,
        material=None,
        scene_idxs=None,
        add_visual=True,
        add_collision=True,
        mesh_pose=None,
        initial_pose=None,
        body_type="dynamic",
        marker_boxes=None,
    ):
        """Build one rigid actor from visual and convex-collision OBJ files.

        ``marker_boxes``: optional list of dicts(half_size, p, color) — plain
        colored box visuals in the ACTOR frame (no collision). Used to break
        rotational symmetry visually, e.g. the cap's twist marker.
        """
        asset_dir = Path(asset_dir)
        visual_files = sorted(asset_dir.glob(visual_glob))
        collision_files = sorted(asset_dir.glob(collision_glob))
        if add_visual and not visual_files:
            raise FileNotFoundError(f"No visual meshes match {asset_dir / visual_glob}")
        if add_collision and not collision_files:
            raise FileNotFoundError(
                f"No collision meshes match {asset_dir / collision_glob}"
            )
        for extra_glob in extra_collision_globs:
            extra_files = sorted(asset_dir.glob(extra_glob))
            if add_collision and not extra_files:
                raise FileNotFoundError(
                    f"No collision meshes match {asset_dir / extra_glob}"
                )
            collision_files.extend(extra_files)

        builder = self.scene.create_actor_builder()
        mesh_pose = mesh_pose or sapien.Pose()

        if add_visual:
            for filename in visual_files:
                builder.add_visual_from_file(
                    filename=str(filename),
                    scale=[scale] * 3,
                    pose=mesh_pose,
                )
            for mb in (marker_boxes or []):
                mat = sapien.render.RenderMaterial(base_color=list(mb["color"]))
                builder.add_box_visual(
                    pose=sapien.Pose(p=list(mb["p"])),
                    half_size=list(mb["half_size"]),
                    material=mat,
                )
        if add_collision:
            for filename in collision_files:
                builder.add_convex_collision_from_file(
                    filename=str(filename),
                    scale=[scale] * 3,
                    pose=mesh_pose,
                    density=density,
                    material=material,
                )

        if scene_idxs is not None:
            builder.set_scene_idxs(scene_idxs)
        builder.initial_pose = initial_pose or sapien.Pose(p=[0, 0, 0])
        if body_type == "dynamic":
            return builder.build(name=name)
        if body_type == "kinematic":
            return builder.build_kinematic(name=name)
        if body_type == "static":
            return builder.build_static(name=name)
        raise ValueError(f"unknown body_type {body_type!r}")

    def _create_cabinet_builder(self):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = self.cabinet_scale
        loader.load_multiple_collisions_from_file = True
        loader.set_density(200.0)
        # Wood class from the object friction table (mu_obj = 0.6); the cabinet
        # is a wooden PartNet unit. Was 1.0/1.0 with no source.
        _mu_wood = object_friction("wood")
        loader.set_material(
            static_friction=_mu_wood, dynamic_friction=_mu_wood, restitution=0.0
        )

        builders = loader.parse(str(CABINET_URDF_PATH))["articulation_builders"]
        if len(builders) != 1:
            raise RuntimeError(f"Expected one cabinet articulation, found {len(builders)}")
        return builders[0]

    def _configure_cabinet(self, cabinet: Articulation):
        if len(cabinet.active_joints) != 1:
            raise RuntimeError("Cabinet 46966 must have exactly one active joint")
        door_joint = cabinet.active_joints[0]
        if door_joint.name != "joint_0":
            raise RuntimeError(f"Expected door joint 'joint_0', found {door_joint.name!r}")

        door_joint.set_friction(0.05)
        door_joint.set_drive_properties(stiffness=0.0, damping=0.1)

        for link in cabinet.links:
            link.set_collision_group_bit(group=2, bit_idx=CABINET_SELF_COLLISION_BIT, bit=1)

    # ------------------------------------------------------------------ #
    # cached geometry
    # ------------------------------------------------------------------ #
    def _after_reconfigure(self, options: dict):
        if self.has_cabinet:
            self._cache_cabinet_geometry()
        if self.has_bottle:
            self._cache_bottle_geometry()
        if self.has_target:
            self._cache_container_geometry("target", self.target)

    def _cache_cabinet_geometry(self):
        body_link = self._cabinets[0].links_map["link_1"]
        body_world = link_piece_bounds(body_link)
        body_local = link_piece_bounds(body_link, relative_to=self._cabinet_pose)
        self.cabinet_z = float(-body_world[:, 0, 2].min() + 1e-3)
        self.cabinet_front_x = float(body_world[:, 0, 0].min())
        self.cabinet_front_local_x = float(body_local[:, 0, 0].min())
        # Full-width thin horizontal panels are the compartment floor and the
        # shelf above it; the bottle stands on the lower of the two.
        size = body_world[:, 1] - body_world[:, 0]
        panels = body_world[(size[:, 2] < 0.03) & (size[:, 0] > 0.2) & (size[:, 1] > 0.2)]
        if len(panels) < 2:
            raise RuntimeError("Could not identify the cabinet's shelves")
        self.shelf_z = float(panels[:, 1, 2].min()) + self.cabinet_z
        self.shelf_ceiling_z = float(panels[:, 1, 2].max()) + self.cabinet_z

        link0 = self._cabinets[0].links_map["link_0"]
        door_world = link_piece_bounds(link0)
        door_local = link_piece_bounds(link0, relative_to=self._cabinet_pose)
        handle_idx = int(np.argmin(door_local[:, 0, 0]))
        self.door_handle_bounds = door_world[handle_idx] + np.array([0.0, 0.0, self.cabinet_z])
        self._door_handle_xyz_nominal = self.door_handle_bounds.mean(axis=0)
        self.door_handle_xyz = tuple(self._door_handle_xyz_nominal)
        panel_local = np.delete(door_local, handle_idx, axis=0)
        hinge_local = np.array([panel_local[:, :, 0].mean(), panel_local[:, 1, 1].max()])
        c, s = np.cos(self.cabinet_yaw), np.sin(self.cabinet_yaw)
        self._door_hinge_xy_nominal = np.array(
            [
                self.cabinet_xy[0] + c * hinge_local[0] - s * hinge_local[1],
                self.cabinet_xy[1] + s * hinge_local[0] + c * hinge_local[1],
            ]
        )
        self.door_hinge_xy = tuple(self._door_hinge_xy_nominal)

        door_limits = self.cabinet.get_qlimits()[:, 0, :]
        self.door_closed_qpos = door_limits[:, 0]
        self.door_open_qpos = door_limits[:, 1]
        if bool(torch.any(self.door_open_qpos < self.door_success_angle).item()):
            raise RuntimeError("Success angle exceeds the cabinet door's joint limit")

    @staticmethod
    def _collision_vertices_local(obj) -> np.ndarray:
        """All collision vertices in an Actor/Link local frame."""
        try:
            return np.asarray(
                obj.get_first_collision_mesh(to_world_frame=False).vertices
            )
        except AttributeError:
            from scipy.spatial.transform import Rotation as _R
            pts = []
            for sh in obj._objs[0].get_collision_shapes():
                v = np.asarray(sh.get_vertices()) * np.asarray(sh.get_scale())
                lp = sh.get_local_pose()
                q = lp.q
                Rm = _R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                pts.append(v @ Rm.T + lp.p)
            return np.vstack(pts)

    @classmethod
    def _collision_bounds_local(cls, obj) -> np.ndarray:
        """Local-frame collision AABB for an Actor or articulation Link."""
        pts = cls._collision_vertices_local(obj)
        return np.stack([pts.min(axis=0), pts.max(axis=0)])

    def _cache_container_geometry(self, name: str, actor):
        """Cache the named container's local bounds and its interior radius."""
        bounds = self._collision_bounds_local(actor)
        half_xy = float(np.min((bounds[1, :2] - bounds[0, :2]) / 2))
        setattr(self, f"{name}_z_lo", float(bounds[0, 2]))
        setattr(self, f"{name}_z_hi", float(bounds[1, 2]))
        setattr(self, f"{name}_height", float(bounds[1, 2] - bounds[0, 2]))
        setattr(self, f"{name}_inner_radius", half_xy * self.interior_radius_ratio)

    def _cache_bottle_geometry(self):
        self._cache_container_geometry("source", self.source)
        # The generic container helper derives a circular interior from the
        # OUTER AABB (21.96 mm here), which is invalid for this profiled thick
        # wall. Use a conservative circular proxy for the wider rounded-square
        # lower body where particles settle. 17 mm contains all physically
        # valid corner positions but is still 4.96 mm tighter than the old AABB.
        self.source_inner_radius = SOURCE_INSIDE_CENTER_RADIUS
        source_vertices = self._collision_vertices_local(self.source)
        bottom_z = float(source_vertices[:, 2].min())
        bottom_ring = source_vertices[
            source_vertices[:, 2] <= bottom_z + BOTTLE_BOTTOM_VERTEX_BAND
        ]
        if len(bottom_ring) < 8:
            raise RuntimeError(
                f"Bottle collision has only {len(bottom_ring)} bottom support vertices"
            )
        # This is the actual collision support ring, not an AABB proxy.  Every
        # one of these points is checked against the tabletop at placement.
        self._source_bottom_support_local = common.to_tensor(
            bottom_ring.astype(np.float32), device=self.device
        )
        self._source_bottom_support_radius = float(
            np.linalg.norm(bottom_ring[:, :2], axis=-1).max()
        )
        if self.has_cap:
            # Keep the complete local collision vertex set: cap placement uses
            # its orientation-aware world minimum, not a local-AABB z fallback.
            cap_vertices = self._collision_vertices_local(self.cap)
            self._cap_collision_vertices_local = common.to_tensor(
                cap_vertices.astype(np.float32), device=self.device
            )
            # Compatibility/debug channel used only by reset placement.
            self.cap_z_lo = float(cap_vertices[:, 2].min())

        self._particle_local_xyz = common.to_tensor(self._pack_particles(), device=self.device)

    def _pack_particles(self) -> np.ndarray:
        """Particle positions (N, 3) in the bottle's local frame that fit inside it."""
        # The bottle has a non-circular lower body, so an AABB-derived radius is
        # not an interior clearance. Use the profiled shell's validated
        # conservative centre radius: the outer ring at 8.8 mm plus a 4 mm
        # particle stays at least 2.5 mm clear of the 10 mm inward wall.
        r_max = PARTICLE_PACK_CENTER_RADIUS
        spacing = 2.2 * self.particle_radius
        ring_xy = [(0.0, 0.0)]
        radius = spacing
        while radius <= r_max:
            count = max(1, int(2 * np.pi * radius / spacing))
            for k in range(count):
                angle = 2 * np.pi * k / count
                ring_xy.append((radius * np.cos(angle), radius * np.sin(angle)))
            radius += spacing
        z0 = (
            self.source_z_lo
            + BOTTLE_WALL_THICKNESS
            + 3 * self.particle_radius
        )
        positions, layer = [], 0
        while len(positions) < self.num_particles:
            z = z0 + layer * spacing
            if z > self.source_z_hi - self.particle_radius:
                raise RuntimeError(
                    f"{self.num_particles} particles of radius {self.particle_radius} "
                    f"do not fit in bottle {BOTTLE_ASSET_DIR.name}"
                )
            for x, y in ring_xy:
                positions.append((x, y, z))
                if len(positions) == self.num_particles:
                    break
            layer += 1
        return np.array(positions, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # episode initialisation
    # ------------------------------------------------------------------ #
    def _initial_robot_qpos(self, batch_size: int) -> torch.Tensor:
        single_arm_qpos = torch.tensor(
            DEFAULT_ARM_INIT_QPOS, dtype=torch.float32, device=self.device
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

        # Dual-arm agents expose get_reset_pos(), which maps rest values to
        # joints BY NAME. Their raw keyframe arrays are laid out per-arm
        # ([arm, hand, arm, hand]) while the articulation's active-joint order
        # is side-INTERLEAVED — consuming the keyframe positionally landed arm
        # angles (pi, -2pi/3) on hand joints and wedged fingers past their
        # limits (measured 142 N of rest-pose self-contact on the dual
        # sharpa). Single-arm keyframes ARE in active order; fall through.
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
                self._bottle_place_hold_count = torch.zeros_like(
                    self._success_hold_count
                )
                self._bottle_place_last_eval_step = torch.zeros_like(
                    self._success_last_eval_step
                )
                self._door_open_hold_count = torch.zeros_like(
                    self._success_hold_count
                )
                self._door_open_last_eval_step = torch.zeros_like(
                    self._success_last_eval_step
                )
                self._cap_turn_hold_count = torch.zeros_like(
                    self._success_hold_count
                )
                self._cap_turn_last_eval_step = torch.zeros_like(
                    self._success_last_eval_step
                )
                self._cap_place_hold_count = torch.zeros_like(
                    self._success_hold_count
                )
                self._cap_place_last_eval_step = torch.zeros_like(
                    self._success_last_eval_step
                )
                self._pour_hold_count = torch.zeros_like(
                    self._success_hold_count
                )
                self._pour_last_eval_step = torch.zeros_like(
                    self._success_last_eval_step
                )
                self._stage_door = torch.zeros(shape, dtype=torch.bool, device=self.device)
                self._stage_bottle_out = torch.zeros_like(self._stage_door)
                self._stage_bottle_placed = torch.zeros_like(self._stage_door)
                self._stage_cap_turned = torch.zeros_like(self._stage_door)
                self._stage_cap_placed = torch.zeros_like(self._stage_door)
                self._stage_poured = torch.zeros_like(self._stage_door)
                self._task_phase = torch.zeros(shape, dtype=torch.int8, device=self.device)
                self._active_bottle_rep = torch.full(
                    shape,
                    BOTTLE_REP_DYNAMIC,
                    dtype=torch.int8,
                    device=self.device,
                )
                self._cabinet_episode_xy = torch.zeros(
                    (self.num_envs, 2), dtype=torch.float32, device=self.device
                )
                self._bottle_lock_pose = torch.zeros(
                    (self.num_envs, 7), dtype=torch.float32, device=self.device
                )
                self._bottle_lock_pose[:, 3] = 1.0
                self._cap_twist_origin = torch.zeros(
                    shape, dtype=torch.float32, device=self.device
                )
                self._cap_lift_origin = torch.zeros_like(self._cap_twist_origin)
                if self.prebuilt_bottle_lock:
                    self._bottle_lock_qpos = torch.zeros(
                        (self.num_envs, 6), dtype=torch.float32,
                        device=self.device,
                    )
                    self._bottle_lock_applied = torch.zeros(
                        shape, dtype=torch.bool, device=self.device
                    )
                self._placement_marker_visibility = torch.full(
                    (self.num_envs, 2),
                    -1,
                    dtype=torch.int8,
                    device=self.device,
                )
            phase_code = {"full": 0, "door": 1, "pick_place": 2, "pour": 3}[task_phase]
            self._task_phase[env_idx] = phase_code
            self._active_bottle_rep[env_idx] = BOTTLE_REP_DYNAMIC
            for flag in (
                self._stage_door,
                self._stage_bottle_out,
                self._stage_bottle_placed,
                self._stage_cap_turned,
                self._stage_cap_placed,
                self._stage_poured,
            ):
                flag[env_idx] = False
            self._success_hold_count[env_idx] = 0
            self._success_last_eval_step[env_idx] = 0
            self._bottle_place_hold_count[env_idx] = 0
            self._bottle_place_last_eval_step[env_idx] = 0
            self._door_open_hold_count[env_idx] = 0
            self._door_open_last_eval_step[env_idx] = 0
            self._cap_turn_hold_count[env_idx] = 0
            self._cap_turn_last_eval_step[env_idx] = 0
            self._cap_place_hold_count[env_idx] = 0
            self._cap_place_last_eval_step[env_idx] = 0
            self._pour_hold_count[env_idx] = 0
            self._pour_last_eval_step[env_idx] = 0
            self._cap_twist_origin[env_idx] = 0.0
            self._cap_lift_origin[env_idx] = 0.0
            self._bottle_lock_pose[env_idx] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            )
            if self.prebuilt_bottle_lock:
                self._bottle_lock_qpos[env_idx] = 0.0
                self._set_prebuilt_bottle_lock_enabled(env_idx, False)
            self._placement_marker_visibility[env_idx] = -1

    def _initialize_workspace(self, env_idx: torch.Tensor):
        """Reset the work table and the complete robot."""
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            # TableSceneBuilder restores its own default pose on every reset.
            self._set_work_table_pose()

            self.agent.reset(self._initial_robot_qpos(len(env_idx)))
            self.agent.robot.set_root_pose(self._robot_base_pose)

    def _initialize_cabinet(
        self, env_idx: torch.Tensor, b: int, door_open: bool = False
    ) -> torch.Tensor:
        """Randomise the cabinet pose; returns the sampled xy offset, shape (b, 2).

        The door starts shut unless ``door_open``, which parks it at its joint
        limit -- the state the later stages of the task begin from.
        """
        with torch.device(self.device):
            cabinet_xy_offset = (
                (torch.rand((b, 2)) * 2 - 1) * self.cabinet_init_xy_noise
            )
            cabinet_p = torch.zeros((b, 3))
            cabinet_p[:, :2] = torch.tensor(
                self.cabinet_xy, dtype=torch.float32
            ) + cabinet_xy_offset
            cabinet_p[:, 2] = self.cabinet_z
            self._cabinet_episode_xy[env_idx] = cabinet_p[:, :2]
            cabinet_q = torch.tensor(
                [np.cos(self.cabinet_yaw / 2), 0.0, 0.0, np.sin(self.cabinet_yaw / 2)],
                dtype=torch.float32,
            ).expand(b, -1)
            self.cabinet.set_pose(Pose.create_from_pq(p=cabinet_p, q=cabinet_q))

            if self.num_envs == 1:
                offset = cabinet_xy_offset[0].cpu().numpy()
                handle = self._door_handle_xyz_nominal.copy()
                handle[:2] += offset
                self.door_handle_xyz = tuple(handle)
                self.door_hinge_xy = tuple(self._door_hinge_xy_nominal + offset)
            door_qpos = self.cabinet.get_qlimits()[env_idx, :, 0].clone()
            if door_open:
                door_qpos[:, 0] = self.cabinet.get_qlimits()[env_idx, 0, 1]
            self.cabinet.set_qpos(door_qpos)
            self.cabinet.set_qvel(torch.zeros_like(door_qpos))
            return cabinet_xy_offset

    def _shelf_bottle_xyz(self, b: int, cabinet_xy_offset: torch.Tensor) -> torch.Tensor:
        """Bottle spawn inside the cabinet, standing on the upper shelf."""
        with torch.device(self.device):
            source_xyz = torch.zeros((b, 3))
            source_xyz[:, :2] = (
                (torch.rand((b, 2)) * 2 - 1) * self.object_init_xy_noise
                + torch.tensor(self.source_start_xy, dtype=torch.float32)
                + cabinet_xy_offset
            )
            # The bottle stands on the upper shelf (top of the upper panel).
            source_xyz[:, 2] = self.shelf_ceiling_z - self.source_z_lo + 1e-3
            return source_xyz

    def _table_bottle_xyz(self, b: int, xy: torch.Tensor = None) -> torch.Tensor:
        """Bottle spawn standing on the tabletop, in the placement area by default."""
        with torch.device(self.device):
            source_xyz = torch.zeros((b, 3))
            if xy is None:
                source_xyz[:, :2] = (
                    (torch.rand((b, 2)) * 2 - 1) * self.object_init_xy_noise
                    + torch.tensor(self.place_center_xy, dtype=torch.float32)
                )
            else:
                source_xyz[:, :2] = xy
            source_xyz[:, 2] = -self.source_z_lo + 1e-3
            return source_xyz

    def _placed_bottle_xyz(self, b: int) -> torch.Tensor:
        """Bottle spawn drawn from anywhere the placement stage accepts it, i.e.
        the state the stages after the set-down inherit."""
        with torch.device(self.device):
            return self._table_bottle_xyz(
                b,
                sample_disc_xy(
                    self.place_center_xy,
                    self.place_radius - self._source_bottom_support_radius,
                    b,
                ),
            )

    def _place_bottle(self, b: int, source_xyz: torch.Tensor) -> torch.Tensor:
        """Stand the bottle upright and still at ``source_xyz``; returns its quat."""
        with torch.device(self.device):
            source_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(b, -1)
            if self.cap_articulated:
                if self.prebuilt_bottle_lock:
                    qpos = torch.zeros((b, self.bottle_screw.max_dof))
                    pivot_world = source_xyz.clone()
                    pivot_world[:, 2] += self._bottle_lock_pivot_z
                    qpos[:, :3] = pivot_world - torch.as_tensor(
                        self._bottle_base_anchor_xyz,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.bottle_screw.set_qpos(qpos)
                    self.bottle_screw.set_qvel(torch.zeros_like(qpos))
                else:
                    self.bottle_screw.set_root_pose(
                        Pose.create_from_pq(p=source_xyz, q=source_q)
                    )
                    zero = torch.zeros((b, self.bottle_screw.max_dof))
                    self.bottle_screw.set_qvel(zero)
            else:
                self.source.set_pose(Pose.create_from_pq(p=source_xyz, q=source_q))
                self.source.set_linear_velocity(torch.zeros((b, 3)))
                self.source.set_angular_velocity(torch.zeros((b, 3)))
            return source_q

    @staticmethod
    def _set_body_visible(obj, i: int, visible: bool):
        import sapien.render as srender

        raw = obj._objs[int(i)]
        entity = getattr(raw, "entity", raw)
        body = entity.find_component_by_type(srender.RenderBodyComponent)
        if body is None:
            return
        try:
            body.visibility = 1.0 if visible else 0.0
        except AttributeError:
            body.set_visibility(1.0 if visible else 0.0)

    @staticmethod
    def _set_articulation_gravity(articulation, i: int, disabled: bool):
        for link in articulation.get_links():
            raw = link._objs[int(i)]
            if hasattr(raw, "set_disable_gravity"):
                raw.set_disable_gravity(disabled)

    @staticmethod
    def _set_actor_gravity(actor, i: int, disabled: bool):
        component = actor._objs[int(i)].find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        if component is not None:
            component.set_disable_gravity(disabled)

    def _source_pose(self) -> Pose:
        """Pose of whichever bottle representation is logically active."""
        if not self.bottle_stage_twins:
            return self.source.pose
        raw_pose = self.source.pose.raw_pose
        raw_pose = torch.where(
            (self._active_bottle_rep == BOTTLE_REP_FIXED)[:, None],
            self.source_fixed.pose.raw_pose,
            raw_pose,
        )
        raw_pose = torch.where(
            (self._active_bottle_rep == BOTTLE_REP_OPEN)[:, None],
            self.source_open.pose.raw_pose,
            raw_pose,
        )
        return Pose(raw_pose)

    def _source_linvel(self):
        if not self.bottle_stage_twins:
            return self.source.get_linear_velocity()
        velocity = self.source.get_linear_velocity()
        velocity = torch.where(
            (self._active_bottle_rep == BOTTLE_REP_FIXED)[:, None],
            self.source_fixed.get_linear_velocity(),
            velocity,
        )
        return torch.where(
            (self._active_bottle_rep == BOTTLE_REP_OPEN)[:, None],
            self.source_open.get_linear_velocity(),
            velocity,
        )

    def _source_angvel(self):
        if not self.bottle_stage_twins:
            return self.source.get_angular_velocity()
        velocity = self.source.get_angular_velocity()
        velocity = torch.where(
            (self._active_bottle_rep == BOTTLE_REP_FIXED)[:, None],
            self.source_fixed.get_angular_velocity(),
            velocity,
        )
        return torch.where(
            (self._active_bottle_rep == BOTTLE_REP_OPEN)[:, None],
            self.source_open.get_angular_velocity(),
            velocity,
        )

    def _attached_cap_p(self):
        if not self.bottle_stage_twins:
            return self.cap.pose.p
        return torch.where(
            (self._active_bottle_rep == BOTTLE_REP_FIXED)[:, None],
            self.cap_fixed.pose.p,
            self.cap.pose.p,
        )

    def _attached_cap_q(self):
        if not self.bottle_stage_twins:
            return self.cap.pose.q
        return torch.where(
            (self._active_bottle_rep == BOTTLE_REP_FIXED)[:, None],
            self.cap_fixed.pose.q,
            self.cap.pose.q,
        )

    def _attached_cap_linvel(self):
        if not self.bottle_stage_twins:
            return self.cap.get_linear_velocity()
        return torch.where(
            (self._active_bottle_rep == BOTTLE_REP_FIXED)[:, None],
            self.cap_fixed.get_linear_velocity(),
            self.cap.get_linear_velocity(),
        )

    def _attached_cap_angvel(self):
        if not self.bottle_stage_twins:
            return self.cap.get_angular_velocity()
        return torch.where(
            (self._active_bottle_rep == BOTTLE_REP_FIXED)[:, None],
            self.cap_fixed.get_angular_velocity(),
            self.cap.get_angular_velocity(),
        )

    def _active_screw_values(self):
        qpos = self.bottle_screw.get_qpos()[:, self._cap_screw_qindex]
        qvel = self.bottle_screw.get_qvel()[:, self._cap_screw_qindex]
        if self.bottle_stage_twins:
            use_fixed = self._active_bottle_rep == BOTTLE_REP_FIXED
            qpos = torch.where(
                use_fixed,
                self.bottle_screw_fixed.get_qpos()[
                    :, self._cap_screw_qindex_fixed
                ],
                qpos,
            )
            qvel = torch.where(
                use_fixed,
                self.bottle_screw_fixed.get_qvel()[
                    :, self._cap_screw_qindex_fixed
                ],
                qvel,
            )
        return qpos, qvel

    def _flush_bottle_representation_state(self):
        if not self.gpu_sim_enabled:
            return
        px = self.scene.px
        px.gpu_apply_rigid_dynamic_data()
        px.gpu_apply_articulation_root_pose()
        px.gpu_apply_articulation_root_velocity()
        px.gpu_apply_articulation_qpos()
        px.gpu_apply_articulation_qvel()
        self.scene._gpu_fetch_all()

    def _sync_bottle_representation_runtime(self, env_idx: torch.Tensor):
        if not self.bottle_stage_twins:
            return
        for i in env_idx.tolist():
            rep = int(self._active_bottle_rep[i])
            cap_attached = not bool(self._cap_free[i])
            self._set_body_visible(
                self.source, i, rep == BOTTLE_REP_DYNAMIC
            )
            self._set_body_visible(
                self.cap, i, rep == BOTTLE_REP_DYNAMIC and cap_attached
            )
            self._set_body_visible(
                self.source_fixed, i, rep == BOTTLE_REP_FIXED
            )
            self._set_body_visible(
                self.cap_fixed, i, rep == BOTTLE_REP_FIXED and cap_attached
            )
            self._set_body_visible(
                self.source_open, i, rep == BOTTLE_REP_OPEN
            )
            self._set_articulation_gravity(
                self.bottle_screw, i, rep != BOTTLE_REP_DYNAMIC
            )
            self._set_actor_gravity(
                self.source_open, i, rep != BOTTLE_REP_OPEN
            )

    def _reset_bottle_stage_twins(self, env_idx: torch.Tensor):
        """Restore A active and park B/C for selected reset environments."""
        if not self.bottle_stage_twins:
            return
        b = len(env_idx)
        identity_q = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).expand(b, -1)
        fixed_p = torch.tensor(
            BOTTLE_FIXED_PARK_P, device=self.device
        ).expand(b, -1)
        self.bottle_screw_fixed.set_root_pose(
            Pose.create_from_pq(p=fixed_p, q=identity_q)
        )
        fixed_qpos = torch.zeros(
            (b, self.bottle_screw_fixed.max_dof), device=self.device
        )
        self.bottle_screw_fixed.set_qpos(fixed_qpos)
        self.bottle_screw_fixed.set_qvel(torch.zeros_like(fixed_qpos))

        open_p = torch.tensor(
            BOTTLE_OPEN_PARK_P, device=self.device
        ).expand(b, -1)
        zeros = torch.zeros((b, 6), device=self.device)
        self.source_open.set_state(
            torch.cat([open_p, identity_q, zeros], dim=-1),
            env_idx=env_idx,
        )
        self._active_bottle_rep[env_idx] = BOTTLE_REP_DYNAMIC
        self._sync_bottle_representation_runtime(env_idx)

    def _activate_open_bottle_at_reset(
        self,
        env_idx: torch.Tensor,
        source_xyz: torch.Tensor,
        source_q: torch.Tensor,
    ):
        """Initialize a pour-only reset directly in representation C."""
        if not self.bottle_stage_twins:
            return
        b = len(env_idx)
        zeros = torch.zeros((b, 6), device=self.device)
        self.source_open.set_state(
            torch.cat([source_xyz, source_q, zeros], dim=-1),
            env_idx=env_idx,
        )
        park_p = torch.tensor(
            BOTTLE_DYNAMIC_PARK_P, device=self.device
        ).expand(b, -1)
        self.bottle_screw.set_root_pose(
            Pose.create_from_pq(p=park_p, q=source_q)
        )
        self.bottle_screw.set_qvel(
            torch.zeros((b, self.bottle_screw.max_dof), device=self.device)
        )
        self._active_bottle_rep[env_idx] = BOTTLE_REP_OPEN
        self._sync_bottle_representation_runtime(env_idx)

    def _activate_fixed_bottle_at_reset(self, env_idx: torch.Tensor):
        """Initialize a B-only stage through the same A -> B transition path."""
        if not self.bottle_stage_twins or len(env_idx) == 0:
            return
        switch_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        switch_mask[env_idx] = True
        self._swap_to_fixed_bottle(switch_mask)

    def _swap_to_fixed_bottle(self, switch_mask: torch.Tensor):
        """A -> B after a qualified set-down and release."""
        if not self.bottle_stage_twins or not bool(switch_mask.any()):
            return
        switch_mask = switch_mask.bool()
        source_pose = self.source.pose.raw_pose.clone()
        source_qpos = self.bottle_screw.get_qpos().clone()

        fixed_pose = self.bottle_screw_fixed.root_pose.raw_pose.clone()
        fixed_pose = torch.where(
            switch_mask[:, None], source_pose, fixed_pose
        )
        self.bottle_screw_fixed.set_root_pose(Pose(fixed_pose))
        fixed_qpos = self.bottle_screw_fixed.get_qpos().clone()
        fixed_qpos[:, self._cap_screw_qindex_fixed] = torch.where(
            switch_mask,
            source_qpos[:, self._cap_screw_qindex],
            fixed_qpos[:, self._cap_screw_qindex_fixed],
        )
        fixed_qvel = self.bottle_screw_fixed.get_qvel().clone()
        fixed_qvel = torch.where(
            switch_mask[:, None], torch.zeros_like(fixed_qvel), fixed_qvel
        )
        self.bottle_screw_fixed.set_qpos(fixed_qpos)
        self.bottle_screw_fixed.set_qvel(fixed_qvel)
        fixed_root_zero = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.bottle_screw_fixed.set_root_linear_velocity(fixed_root_zero)
        self.bottle_screw_fixed.set_root_angular_velocity(fixed_root_zero)

        dynamic_pose = self.bottle_screw.root_pose.raw_pose.clone()
        dynamic_pose[switch_mask, :3] = torch.tensor(
            BOTTLE_DYNAMIC_PARK_P, device=self.device
        )
        self.bottle_screw.set_root_pose(Pose(dynamic_pose))
        dynamic_qvel = self.bottle_screw.get_qvel().clone()
        dynamic_qvel = torch.where(
            switch_mask[:, None], torch.zeros_like(dynamic_qvel), dynamic_qvel
        )
        self.bottle_screw.set_qvel(dynamic_qvel)

        self._active_bottle_rep[switch_mask] = BOTTLE_REP_FIXED
        env_idx = torch.nonzero(switch_mask).flatten()
        self._sync_bottle_representation_runtime(env_idx)
        self._capture_cap_twist_origins(env_idx)
        self._flush_bottle_representation_state()

    def _swap_to_open_bottle(self, switch_mask: torch.Tensor):
        """B -> C only after the detached cap is placed and released."""
        if not self.bottle_stage_twins or not bool(switch_mask.any()):
            return
        switch_mask = switch_mask.bool()
        fixed_source_pose = self.source_fixed.pose.raw_pose.clone()
        open_state = self.source_open.get_state().clone()
        open_state[:, :7] = torch.where(
            switch_mask[:, None], fixed_source_pose, open_state[:, :7]
        )
        open_state[:, 7:] = torch.where(
            switch_mask[:, None],
            torch.zeros_like(open_state[:, 7:]),
            open_state[:, 7:],
        )
        self.source_open.set_state(open_state)

        fixed_pose = self.bottle_screw_fixed.root_pose.raw_pose.clone()
        fixed_pose[switch_mask, :3] = torch.tensor(
            BOTTLE_FIXED_PARK_P, device=self.device
        )
        self.bottle_screw_fixed.set_root_pose(Pose(fixed_pose))
        fixed_qvel = self.bottle_screw_fixed.get_qvel().clone()
        fixed_qvel = torch.where(
            switch_mask[:, None], torch.zeros_like(fixed_qvel), fixed_qvel
        )
        self.bottle_screw_fixed.set_qvel(fixed_qvel)

        self._active_bottle_rep[switch_mask] = BOTTLE_REP_OPEN
        env_idx = torch.nonzero(switch_mask).flatten()
        self._sync_bottle_representation_runtime(env_idx)
        self._flush_bottle_representation_state()

    def _set_cap_screw_state(self, env_idx, twist: float, detached: bool):
        """Seed the articulated screw: joint qpos + per-env thread friction."""
        with torch.device(self.device):
            if self.cap_contact_static_fixture:
                self._cap_twist[env_idx] = 0.0
                self._cap_detached[env_idx] = False
                self._cap_free[env_idx] = False
                self._cap_seal_broken[env_idx] = False
                idx = env_idx.tolist() if torch.is_tensor(env_idx) else list(env_idx)
                for i in idx:
                    self._park_twin(int(i))
                return
            self._cap_twist[env_idx] = twist
            self._cap_detached[env_idx] = detached
            self._cap_seal_broken[env_idx] = bool(detached) or (
                float(twist) > SCREW_SEAL_SLIP
            )
            # Articulation setters write only to the scene reset mask on GPU,
            # so their input must contain exactly the reset sub-batch rather
            # than all environments.
            b = len(env_idx)
            if self.prebuilt_bottle_lock:
                qpos = self.bottle_screw.get_qpos()[env_idx].clone()
            else:
                qpos = torch.zeros(
                    (b, self.bottle_screw.max_dof), dtype=torch.float32
                )
            qpos[:, self._cap_screw_qindex] = float(twist)
            self.bottle_screw.set_qpos(qpos)
            self.bottle_screw.set_qvel(torch.zeros_like(qpos))
            idx = env_idx.tolist() if torch.is_tensor(env_idx) else list(env_idx)
            for i in idx:
                self._apply_cap_friction(int(i))
                if hasattr(self, "cap_twin"):
                    self._park_twin(int(i))

    def _ghost_render_body(self, i: int):
        import sapien.render as srender

        cap = self.cap
        if self.bottle_stage_twins and int(
            self._active_bottle_rep[int(i)]
        ) != BOTTLE_REP_DYNAMIC:
            cap = self.cap_fixed
        return cap._objs[int(i)].entity.find_component_by_type(
            srender.RenderBodyComponent
        )

    @staticmethod
    def _set_render_visibility(body, i: int, value: float):
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

    def _set_ghost_active(self, i: int, active: bool):
        """Show/hide the articulated cap and enable/disable its collisions.
        NOTE the collision re-enable relies on AABB separation re-running the
        pair filter — true across resets (the hand re-poses far away)."""
        rb = self._ghost_render_body(i)
        if rb is not None:
            try:
                rb.visibility = 1.0 if active else 0.0
            except AttributeError:
                rb.set_visibility(1.0 if active else 0.0)
        # The detached twin is the exact complement of the ghost: while the
        # ghost is shown the twin sits parked at +5 m, and a parked body must
        # not render. It is physically harmless up there (gravity off, zero
        # contacts), but a wrist camera pointed upward would film it, and
        # offline re-renders replay wrist poses we did not choose. Driving the
        # twin from here rather than from _park_twin/_swap_in_twin keeps it
        # correct on STATE RESTORE too: set_state_dict re-applies
        # _set_ghost_active, not the swap helpers.
        twin = getattr(self, "cap_twin", None)
        if twin is not None:
            self._set_render_visibility(twin, i, 0.0 if active else 1.0)
        cap = self.cap
        if self.bottle_stage_twins and int(
            self._active_bottle_rep[int(i)]
        ) != BOTTLE_REP_DYNAMIC:
            cap = self.cap_fixed
        for sh in cap._objs[int(i)].get_collision_shapes():
            g = list(sh.get_collision_groups())
            g[0] = 1 if active else 0
            g[1] = (
                DEFAULT_CONTACT_CATEGORY | PARTICLE_CONTACT_CATEGORY
                if active
                else 0
            )
            sh.set_collision_groups(g)

    def _swap_in_twin(self, i: int):
        """Replace the paused joint cap with the loose cap while it is held.

        Position/orientation are canonical (half-turn yaw and +2 mm above the
        seated mount), but the attached cap velocity is preserved. The switch
        is qualified only at low screw speed, so this keeps kinematic continuity
        without injecting a large release impulse.
        """
        with torch.device(self.device):
            idx = torch.tensor([i], device=self.device)
            source_pose = self._source_pose()
            cap_linvel = self._attached_cap_linvel()[idx].clone()
            cap_angvel = self._attached_cap_angvel()[idx].clone()
            cap_offset = torch.tensor(
                [[0.0, 0.0, self.cap_mount_z + 0.002]],
                dtype=torch.float32,
                device=self.device,
            )
            cap_p = source_pose.p[idx] + quaternion_apply(
                source_pose.q[idx], cap_offset
            )
            cap_q = quaternion_multiply(
                source_pose.q[idx],
                quaternion_about_z(
                    torch.full(
                        (1,),
                        float(self.cap_success_rotation),
                        dtype=torch.float32,
                        device=self.device,
                    )
                ),
            )
            state = torch.cat(
                [cap_p, cap_q, cap_linvel, cap_angvel],
                dim=-1,
            )
            self.cap_twin.set_state(state, env_idx=idx)
        comp = self.cap_twin._objs[int(i)].find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        comp.set_disable_gravity(False)
        self._set_ghost_active(i, False)
        self._cap_free[i] = True


    def _park_twin(self, i: int):
        with torch.device(self.device):
            idx = torch.tensor([i], device=self.device)
            self.cap_twin.set_state(
                torch.tensor(
                    [[0.0, 0.0, CAP_TWIN_PARK_Z, 1.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                env_idx=idx,
            )
        comp = self.cap_twin._objs[int(i)].find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        comp.set_disable_gravity(True)
        self._set_ghost_active(i, True)
        self._cap_free[i] = False

    def _cap_p(self):
        """Active cap position: twin once detached, ghost link before."""
        if not self.cap_articulated:
            return self.cap.pose.p
        return torch.where(
            self._cap_free.unsqueeze(-1),
            self.cap_twin.pose.p,
            self._attached_cap_p(),
        )

    def _cap_q(self):
        if not self.cap_articulated:
            return self.cap.pose.q
        return torch.where(
            self._cap_free.unsqueeze(-1),
            self.cap_twin.pose.q,
            self._attached_cap_q(),
        )

    def _cap_linvel(self):
        if not self.cap_articulated:
            return self.cap.get_linear_velocity()
        return torch.where(
            self._cap_free.unsqueeze(-1),
            self.cap_twin.get_linear_velocity(),
            self._attached_cap_linvel(),
        )

    def _cap_angvel(self):
        if not self.cap_articulated:
            return self.cap.get_angular_velocity()
        return torch.where(
            self._cap_free.unsqueeze(-1),
            self.cap_twin.get_angular_velocity(),
            self._attached_cap_angvel(),
        )

    def _apply_cap_friction(self, i: int):
        """Per-env Coulomb thread friction on the revolute — implemented as a
        velocity-target-0 drive with the friction torque as its force ceiling
        (drive damping saturates the limit within ~0.04 rad/s, i.e. Coulomb).
        MEASURED: the articulation joint's own `friction` attribute reads back
        but exerts nothing on this revolute (0.10 N*m spun a 0.18-"friction"
        joint to its limit), while articulation DRIVES are the code path every
        robot in this repo already trusts. The friction value is the ONLY
        thing the screw state machine touches."""
        if not self.cap_screw_dynamic:
            return
        if bool(self._cap_detached[i]):
            f = SCREW_SPIN_FREE_FRICTION
        elif bool(self._cap_seal_broken[i]):
            f = self.cap_running_torque
        else:
            f = self.cap_breakaway_torque
        joint = self._cap_rev
        if self.bottle_stage_twins and int(
            self._active_bottle_rep[int(i)]
        ) != BOTTLE_REP_DYNAMIC:
            joint = self._cap_rev_fixed
        joint._objs[int(i)].set_drive_properties(
            0.0, 5.0, force_limit=f, mode="force"
        )

    def _place_cap(
        self,
        env_idx: torch.Tensor,
        b: int,
        cap_xyz: torch.Tensor,
        cap_q: torch.Tensor,
        source_q: torch.Tensor,
        twist: float = 0.0,
        detached: bool = False,
        free: bool = None,
    ):
        """Seed the cap. For the attached articulation the revolute joint
        defines its pose; environments that start with an already removed cap
        use the free-actor branch and place that actor directly."""
        with torch.device(self.device):
            if self.cap_articulated:
                self._set_cap_screw_state(env_idx, twist, detached)
                return
            if free is None:
                free = detached
            self.cap.set_pose(Pose.create_from_pq(p=cap_xyz, q=cap_q))
            self.cap.set_linear_velocity(torch.zeros((b, 3)))
            self.cap.set_angular_velocity(torch.zeros((b, 3)))
            self._cap_twist[env_idx] = twist
            self._cap_detached[env_idx] = detached
            self._cap_free[env_idx] = free
            self._cap_seal_broken[env_idx] = True

    def _place_cap_on_bottle(
        self,
        env_idx: torch.Tensor,
        b: int,
        source_xyz: torch.Tensor,
        source_q: torch.Tensor,
        twist: float = 0.0,
        detached: bool = False,
    ):
        """Seat the cap at the fixed axial pose used by the attached thread."""
        with torch.device(self.device):
            cap_offset = torch.zeros((b, 3))
            lift = 0.0
            if detached and not self.cap_articulated:
                # A LOOSE free-actor cap sits DOWN on the dead thread (ring
                # wrapping the neck), not at the fully-unscrewed height where
                # the wrap has already disengaged and it just topples off.
                lift = 0.002
            cap_offset[:, 2] = self.cap_mount_z + lift
            cap_xyz = source_xyz + quaternion_apply(source_q, cap_offset)
            cap_q = quaternion_multiply(
                source_q, quaternion_about_z(torch.full((b,), float(twist)))
            )
            self._place_cap(
                env_idx, b, cap_xyz, cap_q, source_q,
                twist=twist, detached=detached, free=False,
            )

    def _place_target(self, b: int):
        """Randomise the bowl's spot on the table."""
        with torch.device(self.device):
            target_xyz = torch.zeros((b, 3))
            target_xyz[:, :2] = (
                (torch.rand((b, 2)) * 2 - 1) * self.object_init_xy_noise
                + torch.tensor(self.target_center_xy, dtype=torch.float32)
            )
            target_xyz[:, 2] = -self.target_z_lo + 1e-3
            self.target.set_pose(
                Pose.create_from_pq(p=target_xyz, q=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(b, -1))
            )

    def _place_particles(self, b: int, source_xyz: torch.Tensor, source_q: torch.Tensor):
        """Fill the bottle with the rigid-particle "liquid"."""
        with torch.device(self.device):
            local = self._particle_local_xyz[None].repeat(b, 1, 1)
            local = local + (torch.rand_like(local) * 2 - 1) * (0.2 * self.particle_radius)
            world = quaternion_apply(source_q[:, None, :], local) + source_xyz[:, None, :]
            zeros = torch.zeros((b, 3))
            for i, particle in enumerate(self.particles):
                particle.set_pose(Pose.create_from_pq(p=world[:, i, :]))
                particle.set_linear_velocity(zeros)
                particle.set_angular_velocity(zeros)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            task_phase = options.get("task_phase", "full")
            if task_phase not in {"full", "door", "pick_place", "pour"}:
                raise ValueError(f"Unknown task_phase {task_phase!r}")
            self._reset_episode_buffers(env_idx, task_phase)
            self._initialize_workspace(env_idx)
            cabinet_xy_offset = self._initialize_cabinet(
                env_idx, b, door_open=task_phase in {"pick_place", "pour"}
            )

            if task_phase == "pour":
                source_xyz = self._table_bottle_xyz(b)
            else:
                source_xyz = self._shelf_bottle_xyz(b, cabinet_xy_offset)
            source_q = self._place_bottle(b, source_xyz)

            if task_phase == "pour":
                # Preserve the existing meaning of the pour-only phase: the
                # unscrewing stage is already complete and the cap is parked.
                cap_xyz = source_xyz.clone()
                cap_xyz[:, 0] = self.cap_place_center_xy[0]
                cap_xyz[:, 1] = self.cap_place_center_xy[1]
                cap_xyz[:, 2] = -self.cap_z_lo + 1e-3
                self._place_cap(
                    env_idx,
                    b,
                    cap_xyz,
                    source_q,
                    source_q,
                    twist=self.cap_success_rotation,
                    detached=True,
                )
            else:
                self._place_cap_on_bottle(env_idx, b, source_xyz, source_q)

            self._reset_bottle_stage_twins(env_idx)
            if task_phase == "pour":
                self._activate_open_bottle_at_reset(
                    env_idx, source_xyz, source_q
                )

            self._place_target(b)
            self._place_particles(b, source_xyz, source_q)

            if task_phase in {"pick_place", "pour"}:
                self._stage_door[env_idx] = True
            if task_phase == "pour":
                self._stage_bottle_out[env_idx] = True
                self._stage_bottle_placed[env_idx] = True
                self._stage_cap_turned[env_idx] = True
                self._stage_cap_placed[env_idx] = True
            self._sync_placement_region_markers(env_idx)

    def _before_control_step(self):
        """Read the physical screw angle while the cap remains joint-attached.

        Axial/lateral rigidity is entirely the revolute joint; this hook never
        projects a pose or writes an axial target. Reaching pi alone does not
        detach: the shared state machine waits for three low-speed control
        steps before performing the in-hand representation switch.
        """
        super()._before_control_step()
        if not self.cap_screw_dynamic:
            return

        theta, _ = self._active_screw_values()
        progress = theta - self._cap_twist_origin
        eligible = self._stage_bottle_placed
        if self.bottle_stage_twins:
            eligible &= self._active_bottle_rep == BOTTLE_REP_FIXED
        attached = ~self._cap_detached

        newly_broken = (
            eligible
            & attached
            & ~self._cap_seal_broken
            # Positive joint rotation is counter-clockwise viewed from above,
            # the only direction that can break the seal / unscrew the cap.
            & (progress > SCREW_SEAL_SLIP)
        )
        if bool(newly_broken.any()):
            self._cap_seal_broken |= newly_broken
        if bool(newly_broken.any()):
            for i in torch.nonzero(newly_broken).flatten().tolist():
                self._apply_cap_friction(int(i))
        # Before bottle placement, handling impulses cannot advance the later
        # unscrewing stage. After placement, progress is relative to the
        # accepted set-down instead of the articulation absolute coordinate.
        self._cap_twist = self._derive_cap_twist()
        self._start_cap_substep_trace()

    def _derive_cap_twist(self) -> torch.Tensor:
        """Twist since the accepted set-down, from state alone.

        `_cap_twist` is DERIVED -- joint angle minus origin, zeroed until the
        bottle is placed -- but it used to be computed only in
        `_before_control_step`, which a restore does not run. A restored frame
        therefore reported the previous episode's twist until something stepped:
        harmless for a sequential replay, wrong for every single-frame consumer,
        and the twist is the success criterion for CapTurn (measured
        `cap_twist` disagreed on 24 of 55 restored frames, and the
        observation carries it). Deriving it in one place, called from both,
        removes the stale case without adding a field that could drift.
        """
        theta, _ = self._active_screw_values()
        progress = theta - self._cap_twist_origin
        eligible = self._stage_bottle_placed
        if self.bottle_stage_twins:
            eligible = eligible & (self._active_bottle_rep == BOTTLE_REP_FIXED)
        return torch.where(eligible, progress, torch.zeros_like(progress))

    def _read_cap_screw_scalar(self):
        qpos, qvel = self._active_screw_values()
        return float(qpos[0]), float(qvel[0])

    def _start_cap_substep_trace(self):
        self._cap_substep_trace = None
        if (
            not getattr(self, "_cap_substep_trace_enabled", False)
            or self.gpu_sim_enabled
            or not self.cap_screw_dynamic
            or bool(self._cap_free[0])
        ):
            return
        qpos, qvel = self._read_cap_screw_scalar()
        self._cap_substep_trace = dict(
            q0=qpos, qprev=qpos, vprev=qvel, travel=0.0,
            peak_step=0.0, velocities=[], sign_flips=0,
        )

    def _record_cap_substep_trace(self):
        stats = getattr(self, "_cap_substep_trace", None)
        if stats is None:
            return
        qpos, qvel = self._read_cap_screw_scalar()
        dq = qpos - stats["qprev"]
        stats["travel"] += abs(dq)
        stats["peak_step"] = max(stats["peak_step"], abs(dq))
        if (
            stats["velocities"]
            and qvel * stats["vprev"] < 0.0
            and abs(qvel) > 0.25
            and abs(stats["vprev"]) > 0.25
        ):
            stats["sign_flips"] += 1
        stats["qprev"] = qpos
        stats["vprev"] = qvel
        stats["velocities"].append(qvel)

    def _after_control_step(self):
        super()._after_control_step()
        stats = getattr(self, "_cap_substep_trace", None)
        if stats is None:
            return
        velocities = stats["velocities"]
        if not velocities:
            return
        vmin, vmax = min(velocities), max(velocities)
        peak = max(abs(vmin), abs(vmax))
        net = stats["qprev"] - stats["q0"]
        travel = stats["travel"]
        canceled = max(0.0, travel - abs(net))
        interesting = peak >= 1.0 or stats["sign_flips"] or canceled > np.deg2rad(0.25)
        if not interesting:
            return
        high5 = sum(abs(v) >= 5.0 for v in velocities)
        print(
            f"[cap-substep] net={np.degrees(net):+6.2f}deg "
            f"travel={np.degrees(travel):6.2f}deg cancel={np.degrees(canceled):5.2f}deg "
            f"peak_dq={np.degrees(stats['peak_step']):4.2f}deg "
            f"qvel=[{vmin:+6.2f},{vmax:+6.2f}]rad/s peak={peak:5.2f} "
            f"flips={stats['sign_flips']} |v|>=5:{high5}/{len(velocities)}",
            flush=True,
        )
        if peak >= 5.0 or stats["sign_flips"]:
            series = ",".join(f"{v:+.1f}" for v in velocities)
            print(f"  qvel@400Hz=[{series}]", flush=True)
    def _bottle_lock_mask(self) -> torch.Tensor:
        """Envs whose bottle root must stay fixed during cap manipulation."""
        if self.bottle_stage_twins:
            return self._active_bottle_rep == BOTTLE_REP_FIXED
        if not self.lock_bottle_during_cap_stages:
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        return self._stage_bottle_placed & ~self._stage_cap_placed

    def _capture_new_bottle_locks(self, newly_placed: torch.Tensor):
        """Remember the accepted tabletop pose at the placement transition."""
        if not bool(newly_placed.any()):
            return
        env_idx = torch.nonzero(newly_placed).flatten()
        if self.bottle_stage_twins:
            self._swap_to_fixed_bottle(newly_placed)
            return
        self._capture_cap_twist_origins(env_idx)
        if not self.lock_bottle_during_cap_stages:
            return
        if self.prebuilt_bottle_lock:
            current = self.bottle_screw.get_qpos()[:, :6]
            self._bottle_lock_qpos = torch.where(
                newly_placed[:, None], current.clone(), self._bottle_lock_qpos
            )
            return
        pose = (
            self.bottle_screw.root_pose
            if self.cap_articulated
            else self.source.pose
        )
        self._bottle_lock_pose = torch.where(
            newly_placed[:, None], pose.raw_pose.clone(), self._bottle_lock_pose
        )

    def _capture_bottle_locks(self, env_idx: torch.Tensor, p: torch.Tensor,
                              q: torch.Tensor):
        """Pin the bottle at a spawn pose known at reset time.

        The long task learns the lock target from the placement TRANSITION
        (_capture_new_bottle_locks, driven from evaluate()). A sub-task that starts
        already-placed never crosses that transition, so it has to declare the target
        itself -- otherwise _bottle_lock_pose is still the identity pose written by
        _reset_episode_buffers and _enforce_bottle_lock yanks the bottle to the world
        origin on the first substep. Takes the pose as arguments rather than reading
        the actor back, so it is correct regardless of when the backend flushes the
        set_pose from _place_bottle.
        """
        with torch.device(self.device):
            self._capture_cap_twist_origins(env_idx)
            if not self.lock_bottle_during_cap_stages:
                return
            if self.prebuilt_bottle_lock:
                current = self.bottle_screw.get_qpos()[env_idx, :6].clone()
                self._bottle_lock_qpos[env_idx] = current
                return
            self._bottle_lock_pose[env_idx] = torch.cat([p, q], dim=-1).to(
                self._bottle_lock_pose.dtype
            )

    def _capture_cap_twist_origins(self, env_idx: torch.Tensor):
        """Make cap progress relative to the accepted bottle set-down."""
        if not self.cap_screw_dynamic or len(env_idx) == 0:
            return
        env_idx = torch.as_tensor(
            env_idx, dtype=torch.long, device=self.device
        ).flatten()
        theta, _ = self._active_screw_values()
        self._cap_twist_origin[env_idx] = theta[env_idx]
        # Retained only as a zero-valued state-dict compatibility slot.
        self._cap_lift_origin[env_idx] = 0.0
        self._cap_twist[env_idx] = 0.0
        self._cap_seal_broken[env_idx] = False
        for i in env_idx.tolist():
            self._apply_cap_friction(int(i))

    def _set_prebuilt_bottle_lock_enabled(
        self, env_idx: torch.Tensor, enabled: bool
    ):
        """Narrow/restore pre-built base-joint limits for selected envs.

        Limits are solver constraints and store no spring energy.  The target
        is captured only after a quiet, flat set-down, so narrowing them does
        not correct pose or velocity and therefore injects no transition kick.
        """
        if not self.prebuilt_bottle_lock or not hasattr(
            self, "_bottle_base_joints"
        ):
            return
        env_idx = torch.as_tensor(
            env_idx, dtype=torch.long, device=self.device
        ).flatten()
        if len(env_idx) == 0:
            return
        for axis, joint in enumerate(self._bottle_base_joints):
            eps = (
                BOTTLE_LOCK_LINEAR_EPS
                if axis < 3
                else BOTTLE_LOCK_ANGULAR_EPS
            )
            for i in env_idx.tolist():
                raw = joint._objs[i]
                if enabled:
                    target = float(self._bottle_lock_qpos[i, axis])
                    limits = np.asarray(
                        [[target - eps, target + eps]], dtype=np.float32
                    )
                else:
                    limits = self._bottle_base_free_limits[axis][i]
                raw.set_limits(limits)
                # Drives remain disabled in both states.  This is intentional:
                # hard solver limits cannot wind up and rebound into the hand.
                raw.set_drive_properties(
                    0.0, 0.0, force_limit=0.0, mode="force"
                )
        if hasattr(self, "_bottle_lock_applied"):
            self._bottle_lock_applied[env_idx] = bool(enabled)

    def _sync_prebuilt_bottle_lock(self):
        desired = self._bottle_lock_mask()
        changed = desired ^ self._bottle_lock_applied
        if not bool(changed.any()):
            return
        enable_idx = torch.nonzero(changed & desired).flatten()
        disable_idx = torch.nonzero(changed & ~desired).flatten()
        self._set_prebuilt_bottle_lock_enabled(enable_idx, True)
        self._set_prebuilt_bottle_lock_enabled(disable_idx, False)

    def _before_simulation_step(self):
        """Hold only the bottle root fixed; leave cap joint motion untouched."""
        super()._before_simulation_step()
        self._enforce_bottle_lock()

    def _after_simulation_step(self):
        """Remove the root drift generated by the just-finished PhysX substep."""
        super()._after_simulation_step()
        self._record_cap_substep_trace()
        self._enforce_bottle_lock()

    def _enforce_bottle_lock(self):
        """Restore locked bottle roots without modifying the cap joint state."""
        if (
            self.cap_contact_static_fixture
            or self.cap_contact_fixed_root
            or self.bottle_stage_twins
        ):
            return
        if self.prebuilt_bottle_lock:
            self._sync_prebuilt_bottle_lock()
            return
        if not hasattr(self, "_bottle_lock_pose"):
            return
        lock_mask = self._bottle_lock_mask()
        if not bool(lock_mask.any()):
            return
        raise RuntimeError(
            "This lock branch used to hold the bottle root by re-writing its pose "
            "and velocity from BOTH _before_simulation_step and "
            "_after_simulation_step -- 40 forced writes per control step at "
            "400 Hz sim / 20 Hz control. Every write re-created the penetration "
            "the solver had just resolved, so a hand touching the bottle got its "
            "depenetration impulse re-applied each substep with nowhere for the "
            "energy to go except the robot (31.8 rad/s peak robot qvel against a "
            "2.2 rad/s actuator limit, measured on MealPreparation). It has been "
            "removed. No registered CabinetPour environment reached it -- all "
            "eight run bottle_stage_twins=True or prebuilt_bottle_lock=True, both "
            "of which hold the bottle with a prebuilt fixture instead of a "
            "teleport. Use one of those (prebuilt_bottle_lock=True is the direct "
            "replacement) rather than re-enabling the per-substep write."
        )

    def _particle_positions(self) -> torch.Tensor:
        """World particle positions, shape (num_envs, num_particles, 3)."""
        return torch.stack([p.pose.p for p in self.particles], dim=1)

    def _inside(self, pose: Pose, points: torch.Tensor, name: str) -> torch.Tensor:
        """Which of ``points`` (B, N, 3) lie in the named container's interior."""
        local = quaternion_apply(
            quaternion_invert(pose.q)[:, None, :], points - pose.p[:, None, :]
        )
        radial = torch.linalg.norm(local[..., :2], dim=-1)
        return (
            (radial < getattr(self, f"{name}_inner_radius"))
            & (local[..., 2] > getattr(self, f"{name}_z_lo"))
            & (local[..., 2] < getattr(self, f"{name}_z_hi"))
        )

    def _pour_counts(self, source_pose: Pose):
        """Particle counts in target/source, spilled low, and still airborne."""
        particles = self._particle_positions()
        in_source = self._inside(source_pose, particles, "source")
        in_target = self._inside(self.target.pose, particles, "target")
        outside = ~in_source & ~in_target
        spilled = outside & (
            particles[..., 2] < 3 * self.particle_radius
        )
        airborne = outside & ~spilled
        return (
            in_target.sum(dim=-1),
            in_source.sum(dim=-1),
            spilled.sum(dim=-1),
            airborne.sum(dim=-1),
        )

    def _pour_success_state(
        self,
        num_in_target: torch.Tensor,
        num_spilled: torch.Tensor,
        num_airborne: torch.Tensor,
    ):
        """Shared 12/16, <=2 spill, zero-airborne, 0.5-second predicate."""
        pour_ratio = num_in_target.float() / self.num_particles
        candidate = (
            (pour_ratio >= self.pour_ratio_threshold)
            & (num_spilled <= self.max_spilled)
            & (num_airborne == 0)
            & self._stage_cap_placed
        )
        poured = self._qualify_control_steps(
            candidate,
            "_pour_hold_count",
            "_pour_last_eval_step",
            POUR_HOLD_STEPS,
        )
        return pour_ratio, candidate, poured

    def _source_posture(self, source_pose: Pose):
        """(cosine of the bottle's tilt, upright flag, at-rest flag)."""
        up = quaternion_apply(
            source_pose.q, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        )
        tilt_cos = up[:, 2]
        source_vel = torch.linalg.norm(self._source_linvel(), dim=-1)
        source_angular_vel = torch.linalg.norm(
            self._source_angvel(), dim=-1
        )
        return (
            tilt_cos,
            tilt_cos > self.upright_cos_threshold,
            (source_vel < BOTTLE_PLACE_LINEAR_SPEED)
            & (source_angular_vel < BOTTLE_PLACE_ANGULAR_SPEED),
        )

    def _bottle_out_of_cabinet(self, source_pose: Pose) -> torch.Tensor:
        """Clear of the cabinet's front face.  Evaluated in cabinet local
        coordinates so this remains correct when xy/yaw/scale move."""
        source_delta = source_pose.p[:, :2] - self._cabinet_episode_xy
        c, s = np.cos(self.cabinet_yaw), np.sin(self.cabinet_yaw)
        source_local_x = c * source_delta[:, 0] + s * source_delta[:, 1]
        return source_local_x < self.cabinet_front_local_x - 0.02

    def _bottle_placed(self, source_pose: Pose, source_upright, settled) -> torch.Tensor:
        """Instantaneous physical set-down candidate.

        Every vertex of the real collision support ring must lie on the table;
        centre-height and AABB tests are intentionally insufficient because a
        tilted bottle can satisfy both while one side hangs in the air.
        """
        place_offset = source_pose.p[:, :2] - torch.tensor(
            self.place_center_xy, device=self.device
        )
        support = self._source_bottom_support_local
        support_world = quaternion_apply(
            source_pose.q[:, None, :],
            support[None, :, :].expand(self.num_envs, -1, -1),
        ) + source_pose.p[:, None, :]
        bottom_z = support_world[..., 2]
        entire_bottom_on_table = (
            bottom_z.amin(dim=-1) >= BOTTLE_BOTTOM_MIN_Z
        ) & (
            bottom_z.amax(dim=-1) <= BOTTLE_BOTTOM_MAX_Z
        )
        up = quaternion_apply(
            source_pose.q,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(
                self.num_envs, -1
            ),
        )
        flat = up[:, 2] >= BOTTLE_PLACE_TILT_COS
        whole_base_in_region = (
            torch.linalg.norm(place_offset, dim=-1)
            + self._source_bottom_support_radius
            <= self.place_radius
        )
        return entire_bottom_on_table & whole_base_in_region & flat & settled

    def _qualify_control_steps(
        self,
        candidate: torch.Tensor,
        count_attr: str,
        last_step_attr: str,
        hold_steps: int,
    ) -> torch.Tensor:
        """Qualify one shared stage predicate at the frozen 20 Hz control rate."""
        count = getattr(self, count_attr)
        last_step = getattr(self, last_step_attr)
        next_count = torch.where(
            candidate,
            torch.clamp(count + 1, max=hold_steps),
            torch.zeros_like(count),
        )
        is_new_control_step = self._elapsed_steps != last_step
        count = torch.where(is_new_control_step, next_count, count)
        setattr(self, count_attr, count)
        setattr(self, last_step_attr, self._elapsed_steps.clone())
        return candidate & (count >= hold_steps)

    def _qualify_bottle_placement(self, candidate: torch.Tensor) -> torch.Tensor:
        """Require a flat, quiet set-down for consecutive control frames."""
        return self._qualify_control_steps(
            candidate,
            "_bottle_place_hold_count",
            "_bottle_place_last_eval_step",
            BOTTLE_PLACE_HOLD_STEPS,
        )

    def _bottle_placement_transition_state(self, source_pose=None):
        """Return the shared A -> B transition state used by every entry point.

        Keeping the release gate, posture test, placement predicate, and
        debounce in one helper prevents the standalone pick/place task from
        drifting away from the long-horizon task's representation switch.
        """
        if source_pose is None:
            source_pose = self._source_pose()
        tilt_cos, source_upright, settled = self._source_posture(source_pose)
        bottle_released = ~self._is_grasping(self.source)
        bottle_place_candidate = self._bottle_placed(
            source_pose, source_upright, settled
        )
        if self.bottle_stage_twins:
            # Never replace the dynamic A representation while it is still
            # grasped; the fixed B representation becomes active only after a
            # qualified, released tabletop set-down.
            bottle_place_candidate &= bottle_released
        bottle_placed = self._qualify_bottle_placement(bottle_place_candidate)
        return (
            source_pose,
            tilt_cos,
            source_upright,
            settled,
            bottle_released,
            bottle_place_candidate,
            bottle_placed,
        )

    def _door_status(self):
        """Door angle, instantaneous 80-degree predicate, and qualified stage."""
        door_angle = self.cabinet.get_qpos()[:, 0]
        door_open_now = door_angle >= self.door_success_angle
        door_open = self._qualify_control_steps(
            door_open_now,
            "_door_open_hold_count",
            "_door_open_last_eval_step",
            DOOR_OPEN_HOLD_STEPS,
        )
        return door_angle, door_open_now, door_open

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

    def _cap_is_grasping(self) -> torch.Tensor:
        if not self.cap_articulated:
            return self._is_grasping(self.cap)
        attached_grasp = self._is_grasping(self.cap)
        if self.bottle_stage_twins:
            attached_grasp = torch.where(
                self._active_bottle_rep == BOTTLE_REP_FIXED,
                self._is_grasping(self.cap_fixed),
                attached_grasp,
            )
        return torch.where(
            self._cap_free,
            self._is_grasping(self.cap_twin),
            attached_grasp,
        )

    def _hand_contact_links(self):
        """Cache physical hand links used by contact and clearance gates."""
        if not hasattr(self, "_task_hand_contact_links"):
            hand_tokens = (
                "hand_C_MC",
                "thumb",
                "index",
                "middle",
                "ring",
                "pinky",
            )
            candidates = [
                link
                for link in self.agent.robot.get_links()
                if any(token in link.name for token in hand_tokens)
            ]
            collision_links = []
            for link in candidates:
                try:
                    has_collision = any(
                        bool(
                            self._raw_collision_component(
                                raw
                            ).get_collision_shapes()
                        )
                        for raw in link._objs
                    )
                except (AttributeError, TypeError):
                    has_collision = False
                if has_collision:
                    collision_links.append(link)
            self._task_hand_contact_links = (
                collision_links or candidates or list(self.agent.tip_links)
            )
        return self._task_hand_contact_links

    def _has_hand_contact(self, actor) -> torch.Tensor:
        """Whether any collision-bearing hand link contacts the actor.

        Sharpa's tip_links are collision-less retargeting frames, so they
        cannot define release. Include the palm and every physical finger
        segment (including NONPAD MP/PP links seen in contact telemetry).
        """
        contact = [
            torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(link, actor), dim=-1
            )
            for link in self._hand_contact_links()
        ]
        return (torch.stack(contact, dim=-1) > 1e-3).any(dim=-1)

    def _cap_hand_clearance(self) -> torch.Tensor:
        """Minimum cap-centre distance to any physical hand-link origin."""
        cap_p = self._cap_p()
        distances = [
            torch.linalg.norm(link.pose.p - cap_p, dim=-1)
            for link in self._hand_contact_links()
        ]
        return torch.stack(distances, dim=-1).amin(dim=-1)

    def _cap_hand_clear(self) -> torch.Tensor:
        """Force-free and geometrically withdrawn from the loose cap."""
        return (
            ~self._cap_has_hand_contact()
            & (self._cap_hand_clearance() >= CAP_HAND_CLEARANCE)
        )

    def _cap_has_hand_contact(self) -> torch.Tensor:
        if not self.cap_articulated:
            return self._has_hand_contact(self.cap)
        attached_contact = self._has_hand_contact(self.cap)
        if self.bottle_stage_twins:
            attached_contact = torch.where(
                self._active_bottle_rep == BOTTLE_REP_FIXED,
                self._has_hand_contact(self.cap_fixed),
                attached_contact,
            )
        return torch.where(
            self._cap_free,
            self._has_hand_contact(self.cap_twin),
            attached_contact,
        )

    def _cap_lowest_z(self) -> torch.Tensor:
        pose = Pose.create_from_pq(p=self._cap_p(), q=self._cap_q())
        vertices = self._cap_collision_vertices_local.expand(
            self.num_envs, -1, -1
        )
        world_vertices = pose.p[:, None, :] + quaternion_apply(
            pose.q[:, None, :], vertices
        )
        return world_vertices[..., 2].amin(dim=-1)

    def _cap_status(self):
        """Shared continuous unscrew -> free-cap -> tabletop placement state."""
        cap_p = self._cap_p()
        cap_place_offset = cap_p[:, :2] - torch.tensor(
            self.cap_place_center_xy, device=self.device
        )
        cap_lowest_z = self._cap_lowest_z()
        cap_on_table = (
            (cap_lowest_z >= CAP_BOTTOM_MIN_Z)
            & (cap_lowest_z <= CAP_BOTTOM_MAX_Z)
            & (
                torch.linalg.norm(cap_place_offset, dim=-1)
                < self.cap_place_radius
            )
        )
        cap_settled = (
            torch.linalg.norm(self._cap_linvel(), dim=-1)
            < self.cap_static_threshold
        ) & (
            torch.linalg.norm(self._cap_angvel(), dim=-1)
            < self.cap_angular_static_threshold
        )

        # Reaching pi is an internal representation boundary, not task success.
        # Keep holding the cap, pause the screw coordinate for 150 ms, and swap
        # the joint link for the +2 mm free twin in the same episode.
        attached = ~self._cap_detached
        if self.cap_screw_dynamic:
            _, screw_qvel = self._active_screw_values()
        else:
            screw_qvel = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
        switch_candidate = (
            self._stage_bottle_placed
            & attached
            & (self._cap_twist >= self.cap_success_rotation)
            & (torch.abs(screw_qvel) <= CAP_TURN_SWITCH_MAX_QVEL)
        )
        switch_ready = self._qualify_control_steps(
            switch_candidate,
            "_cap_turn_hold_count",
            "_cap_turn_last_eval_step",
            CAP_TURN_SWITCH_HOLD_STEPS,
        )
        newly_detached = attached & switch_ready
        if bool(newly_detached.any()):
            self._cap_detached |= newly_detached
            for i in torch.nonzero(newly_detached).flatten().tolist():
                self._swap_in_twin(int(i))
                self._apply_cap_friction(int(i))
            if self.gpu_sim_enabled:
                self.scene.px.gpu_apply_rigid_dynamic_data()
                self.scene._gpu_fetch_all()

        cap_turned = self._stage_bottle_placed & self._cap_detached
        if self.cap_articulated:
            cap_turned &= self._cap_free

        # Final success belongs to placement, not merely detachment. Requiring
        # both zero contact and 65 mm geometric clearance prevents a caged cap
        # from succeeding when pairwise net forces happen to cancel.
        cap_place_candidate = (
            (self._stage_cap_turned | cap_turned)
            & self._cap_detached
            & cap_on_table
            & cap_settled
            & self._cap_hand_clear()
        )
        cap_placed = self._qualify_control_steps(
            cap_place_candidate,
            "_cap_place_hold_count",
            "_cap_place_last_eval_step",
            CAP_PLACE_HOLD_STEPS,
        )
        return cap_turned, cap_settled, cap_placed

    def _latch_success(self, success_condition: torch.Tensor) -> torch.Tensor:
        """Count consecutive control steps the condition holds for, and report
        whether it has now held for ``success_hold_steps`` of them."""
        next_hold = torch.where(
            success_condition,
            torch.clamp(self._success_hold_count + 1, max=self.success_hold_steps),
            torch.zeros_like(self._success_hold_count),
        )
        is_new_control_step = self._elapsed_steps != self._success_last_eval_step
        self._success_hold_count = torch.where(
            is_new_control_step, next_hold, self._success_hold_count
        )
        self._success_last_eval_step = self._elapsed_steps.clone()
        return self._success_hold_count >= self.success_hold_steps

    def evaluate(self):
        with torch.device(self.device):
            source_pose = self._source_pose()
            (
                num_in_target,
                num_in_source,
                num_spilled,
                num_airborne,
            ) = self._pour_counts(source_pose)
            (
                source_pose,
                tilt_cos,
                source_upright,
                settled,
                bottle_released,
                bottle_place_candidate,
                bottle_placed,
            ) = self._bottle_placement_transition_state(source_pose)

            door_angle, door_open_now, door_open = self._door_status()

            bottle_out = self._bottle_out_of_cabinet(source_pose)
            cap_turned, cap_settled, cap_placed = self._cap_status()

            # Every stage latches the same qualified predicate exposed by its
            # standalone subtask; no long-task-only threshold is allowed here.
            self._stage_door |= door_open
            door_opened_once = self._stage_door
            self._stage_bottle_out |= door_opened_once & bottle_out
            newly_bottle_placed = (
                ~self._stage_bottle_placed
                & self._stage_bottle_out
                & bottle_placed
            )
            self._capture_new_bottle_locks(newly_bottle_placed)
            self._stage_bottle_placed |= self._stage_bottle_out & bottle_placed
            self._stage_cap_turned |= cap_turned
            newly_cap_placed = ~self._stage_cap_placed & cap_placed
            self._stage_cap_placed |= cap_placed
            self._swap_to_open_bottle(newly_cap_placed)
            self._sync_placement_region_markers()

            pour_ratio, pour_candidate, poured = self._pour_success_state(
                num_in_target, num_spilled, num_airborne
            )
            self._stage_poured |= poured
            full_success_condition = poured
            # Each independent recording begins from its phase's initial state,
            # so evaluate that phase instead of requiring the full sequence.
            success_condition = torch.where(
                self._task_phase == 1,
                door_open,
                torch.where(
                    self._task_phase == 2,
                    self._stage_bottle_placed,
                    torch.where(
                        self._task_phase == 3,
                        poured,
                        full_success_condition,
                    ),
                ),
            )

            return dict(
                door_angle=door_angle,
                door_angle_deg=torch.rad2deg(door_angle),
                door_open_now=door_open_now,
                door_open=door_open,
                door_open_hold_count=self._door_open_hold_count.clone(),
                num_in_target=num_in_target,
                num_in_source=num_in_source,
                num_spilled=num_spilled,
                num_airborne=num_airborne,
                pour_ratio=pour_ratio,
                pour_candidate=pour_candidate,
                pour_hold_count=self._pour_hold_count.clone(),
                tilt_cos=tilt_cos,
                source_upright=source_upright,
                settled=settled,
                bottle_place_candidate=bottle_place_candidate,
                bottle_released=bottle_released,
                bottle_place_hold_count=self._bottle_place_hold_count.clone(),
                stage_door_open=self._stage_door.clone(),
                stage_bottle_out=self._stage_bottle_out.clone(),
                stage_bottle_placed=self._stage_bottle_placed.clone(),
                bottle_locked=self._bottle_lock_mask(),
                bottle_rep=self._active_bottle_rep.clone(),
                stage_cap_turned=self._stage_cap_turned.clone(),
                stage_cap_placed=self._stage_cap_placed.clone(),
                stage_poured=self._stage_poured.clone(),
                cap_twist=self._cap_twist.clone(),
                cap_detached=self._cap_detached.clone(),
                cap_turned=cap_turned,
                cap_turn_hold_count=self._cap_turn_hold_count.clone(),
                cap_placed=cap_placed,
                cap_place_hold_count=self._cap_place_hold_count.clone(),
                cap_settled=cap_settled,
                poured=poured,
                full_success_condition=full_success_condition,
                success_condition=success_condition,
                success=success_condition,
            )

    # ------------------------------------------------------------------ #
    # env-state (offline re-render / RL-from-demo-states)
    # ------------------------------------------------------------------ #
    def get_state_dict(self):
        """Persist physical state plus every task-side sequential latch."""
        state = super().get_state_dict()
        task_state = torch.cat(
            [
                self._success_hold_count[:, None].float(),
                self._success_last_eval_step[:, None].float(),
                self._bottle_place_hold_count[:, None].float(),
                self._bottle_place_last_eval_step[:, None].float(),
                self._stage_door[:, None].float(),
                self._stage_bottle_out[:, None].float(),
                self._stage_bottle_placed[:, None].float(),
                self._stage_cap_turned[:, None].float(),
                self._stage_cap_placed[:, None].float(),
                self._stage_poured[:, None].float(),
                self._task_phase[:, None].float(),
                self._cabinet_episode_xy.float(),
                self._bottle_lock_pose.float(),
            ],
            dim=-1,
        )
        if self.prebuilt_bottle_lock:
            task_state = torch.cat(
                [task_state, self._bottle_lock_qpos.float()], dim=-1
            )
        task_state = torch.cat(
            [
                task_state,
                self._door_open_hold_count[:, None].float(),
                self._door_open_last_eval_step[:, None].float(),
                self._cap_turn_hold_count[:, None].float(),
                self._cap_turn_last_eval_step[:, None].float(),
                self._cap_place_hold_count[:, None].float(),
                self._cap_place_last_eval_step[:, None].float(),
                self._pour_hold_count[:, None].float(),
                self._pour_last_eval_step[:, None].float(),
            ],
            dim=-1,
        )
        # Serialised UNCONDITIONALLY, not just for twin builds. It used to be
        # written only when `bottle_stage_twins` was on, so DoorOpen / Pour /
        # CapPickPlace / PrebuiltLock recorded no representation at all and a
        # restored mid-episode frame came back on the wrong one. `_source_pose`
        # short-circuits without twins, but the field is read in several other
        # places, and one float column is far cheaper than reasoning about which.
        task_state = torch.cat(
            [task_state, self._active_bottle_rep[:, None].float()],
            dim=-1,
        )
        state["cabinet_pour_task_state"] = task_state
        if self.has_cap:
            state["cap_screw"] = torch.stack(
                [
                    self._cap_detached.float(),
                    self._cap_seal_broken.float(),
                    self._cap_free.float(),
                    self._cap_twist_origin,
                    self._cap_lift_origin,
                ],
                dim=-1,
            )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("cabinet_pour_task_state", None)
        cap_screw = state.pop("cap_screw", None)
        if env_idx is None:
            restore_idx = torch.arange(self.num_envs, device=self.device)
            super().set_state_dict(state)
        else:
            restore_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            ).flatten()
            super().set_state_dict(state, restore_idx)

        if isinstance(task_state, dict):
            # Flattened LeRobot environment state restores a top-level tensor
            # as {"value": tensor}; raw_states.pkl keeps the tensor directly.
            task_state = task_state.get("value")
        if task_state is not None:
            task_state = common.to_tensor(
                task_state, device=self.device
            ).reshape(len(restore_idx), -1)
            legacy_fields = 26 if self.prebuilt_bottle_lock else 20
            current_fields = legacy_fields + 8
            representation_fields = current_fields + 1
            # `representation_fields` is what this build now writes for EVERY
            # env; the two narrower widths stay accepted so recordings made
            # earlier still restore.
            valid_fields = [legacy_fields, current_fields, representation_fields]
            if task_state.shape[-1] not in valid_fields:
                expected = " or ".join(str(v) for v in valid_fields)
                raise ValueError(
                    "cabinet_pour_task_state must have "
                    f"{expected} fields, got {task_state.shape[-1]}"
                )
            self._success_hold_count[restore_idx] = task_state[:, 0].to(torch.int32)
            self._success_last_eval_step[restore_idx] = task_state[:, 1].to(torch.int32)
            self._bottle_place_hold_count[restore_idx] = task_state[:, 2].to(torch.int32)
            self._bottle_place_last_eval_step[restore_idx] = task_state[:, 3].to(torch.int32)
            self._stage_door[restore_idx] = task_state[:, 4].bool()
            self._stage_bottle_out[restore_idx] = task_state[:, 5].bool()
            self._stage_bottle_placed[restore_idx] = task_state[:, 6].bool()
            self._stage_cap_turned[restore_idx] = task_state[:, 7].bool()
            self._stage_cap_placed[restore_idx] = task_state[:, 8].bool()
            self._stage_poured[restore_idx] = task_state[:, 9].bool()
            self._task_phase[restore_idx] = task_state[:, 10].to(torch.int8)
            self._cabinet_episode_xy[restore_idx] = task_state[:, 11:13].to(torch.float32)
            self._bottle_lock_pose[restore_idx] = task_state[:, 13:20].to(torch.float32)
            if self.prebuilt_bottle_lock:
                self._bottle_lock_qpos[restore_idx] = task_state[:, 20:26].to(torch.float32)
            if task_state.shape[-1] >= current_fields:
                offset = legacy_fields
                self._door_open_hold_count[restore_idx] = task_state[:, offset].to(
                    torch.int32
                )
                self._door_open_last_eval_step[restore_idx] = task_state[:, offset + 1].to(
                    torch.int32
                )
                self._cap_turn_hold_count[restore_idx] = task_state[:, offset + 2].to(
                    torch.int32
                )
                self._cap_turn_last_eval_step[restore_idx] = task_state[:, offset + 3].to(
                    torch.int32
                )
                self._cap_place_hold_count[restore_idx] = task_state[:, offset + 4].to(
                    torch.int32
                )
                self._cap_place_last_eval_step[restore_idx] = task_state[:, offset + 5].to(
                    torch.int32
                )
                self._pour_hold_count[restore_idx] = task_state[:, offset + 6].to(
                    torch.int32
                )
                self._pour_last_eval_step[restore_idx] = task_state[:, offset + 7].to(
                    torch.int32
                )
            else:
                for count in (
                    self._door_open_hold_count,
                    self._door_open_last_eval_step,
                    self._cap_turn_hold_count,
                    self._cap_turn_last_eval_step,
                    self._cap_place_hold_count,
                    self._cap_place_last_eval_step,
                    self._pour_hold_count,
                    self._pour_last_eval_step,
                ):
                    count[restore_idx] = 0
            if task_state.shape[-1] == representation_fields:
                self._active_bottle_rep[restore_idx] = task_state[
                    :, current_fields
                ].to(torch.int8)
            elif self.bottle_stage_twins:
                # Legacy recordings predate the proxy field and contain only
                # representation A in their physical state.
                self._active_bottle_rep[restore_idx] = BOTTLE_REP_DYNAMIC
            if env_idx is not None:
                # reset_to_env_states starts a fresh episode clock at zero.
                # Rebase debounce timestamps so the reset-time get_info call
                # neither clears nor advances the saved consecutive counts.
                fresh_reset = self._elapsed_steps[restore_idx] == 0
                elapsed = self._elapsed_steps[restore_idx]
                self._success_last_eval_step[restore_idx] = torch.where(
                    fresh_reset, elapsed, self._success_last_eval_step[restore_idx]
                )
                self._bottle_place_last_eval_step[restore_idx] = torch.where(
                    fresh_reset, elapsed,
                    self._bottle_place_last_eval_step[restore_idx],
                )
                for last_step in (
                    self._door_open_last_eval_step,
                    self._cap_turn_last_eval_step,
                    self._cap_place_last_eval_step,
                    self._pour_last_eval_step,
                ):
                    last_step[restore_idx] = torch.where(
                        fresh_reset, elapsed, last_step[restore_idx]
                    )

        if isinstance(cap_screw, dict):
            cap_screw = cap_screw.get("value")
        if self.has_cap and cap_screw is not None:
            cap_screw = common.to_tensor(
                cap_screw, device=self.device
            ).reshape(len(restore_idx), -1)
            self._cap_detached[restore_idx] = cap_screw[:, 0].bool()
            self._cap_seal_broken[restore_idx] = cap_screw[:, 1].bool()
            self._cap_free[restore_idx] = cap_screw[:, 2].bool()
            if cap_screw.shape[-1] >= 5:
                self._cap_twist_origin[restore_idx] = cap_screw[:, 3]
                # Field 4 was the old prismatic origin. Pure-revolute builds
                # keep the five-field layout but deliberately ignore it.
                self._cap_lift_origin[restore_idx] = 0.0
            else:
                self._cap_twist_origin[restore_idx] = 0.0
                self._cap_lift_origin[restore_idx] = 0.0
            if self.cap_screw_dynamic:
                # The same derivation `_before_control_step` uses. This used to
                # be `theta - origin` written out again here, WITHOUT the
                # eligibility mask that zeroes the twist until the bottle is
                # placed, so a restored frame reported a twist the stepping code
                # would have reported as zero.
                self._cap_twist[restore_idx] = self._derive_cap_twist()[restore_idx]
                for i in restore_idx.tolist():
                    self._apply_cap_friction(i)
                    if hasattr(self, "cap_twin"):
                        if bool(self._cap_free[i]):
                            self._set_ghost_active(i, False)
                            comp = self.cap_twin._objs[i].find_component_by_type(
                                sapien.physx.PhysxRigidDynamicComponent
                            )
                            comp.set_disable_gravity(False)
                        else:
                            self._park_twin(i)
            elif self.cap_contact_static_fixture:
                self._cap_twist[restore_idx] = 0.0
                self._cap_twist_origin[restore_idx] = 0.0
                self._cap_detached[restore_idx] = False
                self._cap_free[restore_idx] = False
                self._cap_seal_broken[restore_idx] = False
                for i in restore_idx.tolist():
                    self._park_twin(i)
            else:
                self._cap_twist[restore_idx] = float(self.cap_success_rotation)

        if task_state is not None:
            # Goal-marker actors are intentionally excluded from the physical
            # state registry, so regenerate their poses from restored latches.
            # Prebuilt lock limits are likewise derived from the stage mask.
            # The standard projected lock needs no immediate pose write because
            # super().set_state_dict restored the recorded body state already.
            self._placement_marker_visibility[restore_idx] = -1
            self._sync_placement_region_markers(restore_idx)
            if self.prebuilt_bottle_lock:
                self._sync_prebuilt_bottle_lock()
            if self.bottle_stage_twins:
                self._sync_bottle_representation_runtime(restore_idx)

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            door_qpos=info["door_angle"][:, None],
            target_pos=self.target.pose.p,
            cap_pos=self._cap_p(),
            cap_twist=info["cap_twist"][:, None],
            cap_detached=info["cap_detached"][:, None].float(),
            pour_ratio=info["pour_ratio"][:, None],
            tilt_cos=info["tilt_cos"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                source_pose=self._source_pose().raw_pose,
                cap_pose=torch.cat([self._cap_p(), self._cap_q()], dim=-1),
                target_pose=self.target.pose.raw_pose,
                door_pose=self.door_link.pose.raw_pose,
                tcp_to_source_pos=self._source_pose().p - self.agent.tcp.pose.p,
                tcp_to_cap_pos=self._cap_p() - self.agent.tcp.pose.p,
                source_to_target_pos=self.target.pose.p - self._source_pose().p,
                bottle_rep=info["bottle_rep"][:, None].float(),
                num_spilled=info["num_spilled"][:, None].float(),
                stages=torch.stack(
                    [
                        info["stage_door_open"],
                        info["stage_bottle_placed"],
                        info["stage_cap_turned"],
                        info["stage_cap_placed"],
                        info["stage_poured"],
                    ],
                    dim=-1,
                ).float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Cumulative progress score: once a sequential milestone is latched,
        # its point remains in every later step's reward.
        milestones = torch.stack(
            [
                info["stage_door_open"],
                info["stage_bottle_placed"],
                info["stage_cap_turned"],
                info["stage_cap_placed"],
                info["stage_poured"],
            ],
            dim=-1,
        )
        return milestones.to(torch.float32).sum(dim=-1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=DEFAULT_DUAL_ROBOT_UID)
    parser.add_argument("--control_mode", default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/cabinet_pour")
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
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Robot DoF", env.unwrapped.agent.robot.dof)

    env.reset(seed=args.seed, options={"task_phase": "full"})
    for step in range(args.steps):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
        if step % 25 == 0:
            print(
                f"step {step}: door {float(info['door_angle_deg']):.1f} deg  "
                f"src {int(info['num_in_source'])} tgt {int(info['num_in_target'])} "
                f"spill {int(info['num_spilled'])}"
            )
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()

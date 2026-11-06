"""Package opening long-horizon task and shared subtask base environment.

This is the self-contained, top-level version of the package-opening task.  It
owns scene construction, the three-stage state machine, observations, rewards,
camera configuration, and the reusable single-subtask base class in one file.
The older ``tasks/package_opening/`` directory is intentionally left untouched.

Sequence:
  1. (the scissors stage was REMOVED -- see TASK_PLAN)
  2. open both package lids beyond 90 degrees;
  3. grasp the apple, lift it completely above the package rim, then release it
     stably on the plate.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# ``--list`` is intentionally handled before importing Gymnasium/SAPIEN and
# robot definitions. Those imports can take many seconds and may initialize a
# renderer even though listing tasks does not need a simulator.
if __name__ == "__main__" and "--list" in sys.argv:
    print("PackageOpeningLong stage tasks (2)")
    print("\n1. open_package")
    print("   Open both lid flaps past 90 degrees with both hands.")
    print("   Success: both flaps have rotated more than 90 degrees from the closed pose.")
    print("\n2. extract_and_place_apple")
    print("   Starting from the opened package with the apple on the box floor, take the apple out")
    print("   and put it on the plate.")
    print("   Success: the extraction condition and the stable in-plate release condition both hold.")
    raise SystemExit(0)

import gymnasium as gym
import xml.etree.ElementTree as ET

import numpy as np
import sapien
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.config.layout_randomization import (
    layout_rng,
    ARM_INIT_QPOS_NOISE_STD,
    arm_qpos_noise,
    attached_point,
    episode_rng,
    offset_pose,
    uniform_offset,
)
from tasks.config.object_materials import object_material
from tasks.config.workspace_rig import (
    apply_arm_init_qpos,
    armless_base_root_pose,
    platform_root_pose,
    set_work_table_pose,
)
from tasks.config.mass_properties import set_mass as set_realizable_mass
from tasks.robots.panda_dual_sharpa_hand_mobile_base import (
    PandaDualSharpaHandAngledMobileBase,
    PandaDualSharpaHandMobileBase,
)
from tasks.robots.panda_sharpa_hand_right import (
    PandaSharpaHandRight,
    PandaSharpaHandRightAngled,
)
from tasks.robots.panda_sharpa_hand_right_mobile_base import (
    PandaSharpaHandRightAngledMobileBase,
    PandaSharpaHandRightMobileBase,
)

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from tasks.config.hands import register_task
from tasks.scene_builders.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.record import RecordEpisode
from tasks.config.sim_profiles import CONTROL_FREQ, SIM_FREQ_DEFAULT, make_sim_config
from tasks.robots.panda_wuji_hand_right_mobile_base import PandaWujiHandRightMobileBase  # noqa: F401
from tasks.robots.panda_dual_wuji_hand_mobile_base import PandaDualWujiHandMobileBase  # noqa: F401
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.robots.panda_dual_xhand_hand_mobile_base import PandaDualXHandHandMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag


ENV_ID = "PackageOpeningLong-Sharpa-v1"
LONG_TASK_ROBOT_UID = "panda_dual_sharpa_hand_angled_mobile_base"
SUBTASK_ROBOT_UID = "panda_sharpa_hand_right_angled_mobile_base"
DEFAULT_ROBOT_UID = LONG_TASK_ROBOT_UID
STRAIGHT_ROBOT_UID = "panda_sharpa_hand_right"
ANGLED_ROBOT_UID = "panda_sharpa_hand_right_angled"
STRAIGHT_MOBILE_ROBOT_UID = "panda_sharpa_hand_right_mobile_base"
DUAL_STRAIGHT_MOBILE_ROBOT_UID = "panda_dual_sharpa_hand_mobile_base"
WUJI_MOBILE_ROBOT_UID = "panda_wuji_hand_right_angled_mobile_base"
DUAL_WUJI_MOBILE_ROBOT_UID = "panda_dual_wuji_hand_angled_mobile_base"
XHAND_MOBILE_ROBOT_UID = "panda_xhand_hand_right_angled_mobile_base"
DUAL_XHAND_MOBILE_ROBOT_UID = "panda_dual_xhand_hand_angled_mobile_base"
DEFAULT_CONTROL_MODE = "pd_joint_pos"
PACKAGE_SUPPORTED_CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")
# default arm joint noise is the fleet value from tasks/config/layout_randomization.py
# (it was frozen at 0.0 -- the family had no per-episode randomization at all).
FROZEN_ROBOT_INIT_QPOS_NOISE = ARM_INIT_QPOS_NOISE_STD

# Per-episode layout randomization, uniform around the nominal layout,
# drawn from the episode RNG. The apple rides inside the box; the plate moves on its own.
PACKAGE_BOX_RANDOM_XY = (0.030, 0.030)      # m, half-range
# The box is a FIXED-ROOT articulation. The recording contract restores raw states bit for
# bit, and PhysX's fixed-root pose write is not idempotent at the float32 ulp for a
# non-identity quaternion (measured: 1-ulp drift in q after set_state_dict), so
# fixed-root bodies get POSITION-only randomization, like the CabinetPour cabinet and the
# DrawerPlug drawer. Free bodies (apple, plate) may turn.
PACKAGE_BOX_RANDOM_YAW_DEG = 0.0
PACKAGE_PLATE_RANDOM_XY = (0.020, 0.020)    # m, half-range

TASK_ASSET_DIR = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "task_object_meshes"
    / "package_opening"
)
PACKAGE_URDF_PATH = TASK_ASSET_DIR / "package" / "mobility.urdf"
ROBOCASA_OBJECT_DIR = TASK_ASSET_DIR / "robocasa" / "objects" / "objaverse"

# The two trolley-less builds in SUPPORTED_ROBOTS have panda_link0 as their
# root link, so they are placed where a trolley-mounted right arm root lands
# instead of at ManiSkill's stock -0.615. Before this they got a DIFFERENT
# table pose and a DIFFERENT object layout from the trolley builds, which meant
# a mirror and a teleop process started with different --robot-uids showed the
# operator two scenes 0.30 m apart with nothing raising.
FIXED_BASE_ROOT_POSE = armless_base_root_pose("right")
FULL_TROLLEY_BASE_POSE = platform_root_pose()
WORK_TABLE_X_OFFSET = 0.30
MOBILE_LAYOUT_OFFSET_XY = (-0.255, -0.22)
MOBILE_ROBOT_UIDS = {
    SUBTASK_ROBOT_UID,
    STRAIGHT_MOBILE_ROBOT_UID,
    LONG_TASK_ROBOT_UID,
    DUAL_STRAIGHT_MOBILE_ROBOT_UID,
    WUJI_MOBILE_ROBOT_UID,
    DUAL_WUJI_MOBILE_ROBOT_UID,
    XHAND_MOBILE_ROBOT_UID,
    DUAL_XHAND_MOBILE_ROBOT_UID,
    "panda_gripper_right_mobile_base",
    "panda_dual_gripper_mobile_base",
}

# the scissors stage was removed (user decision). Closing or even
# picking up a pair of scissors needs a torque about the grasp axis that a
# rigid fingertip pad cannot carry (the contact-patch ceiling that also retired
# the spoon and the brush). The task is now: open the package with both hands,
# take the item out and place it where asked.
TASK_PLAN = (
    ("open_package", "Open both package lids beyond 90 degrees"),
    ("extract_and_place_apple", "Extract the apple and release it stably on the plate"),
)
STAGE_NAMES = tuple(item[0] for item in TASK_PLAN) + ("complete",)
NUM_TASK_STAGES = len(TASK_PLAN)

TASK_SUCCESS_CONDITIONS = (
    "Both lids have rotated more than 90 degrees from the closed pose.",
    "The apple was grasped and lifted out (nearest palm TCP < 0.09 m, lift > 0.03 m, bottom 0.005 m "
    "above the box wall), then placed on the plate and released (planar distance < 0.05 m, height "
    "difference < 0.02 m, speed < 0.05 m/s, every palm TCP > 0.10 m away).",
)


def print_task_list():
    """Print the executable stages and their success criteria without making an env."""
    print(f"{ENV_ID} stage tasks ({NUM_TASK_STAGES})")
    for index, ((name, description), condition) in enumerate(
        zip(TASK_PLAN, TASK_SUCCESS_CONDITIONS), start=1
    ):
        print(f"\n{index}. {name}")
        print(f"   Task: {description}")
        print(f"   Success: {condition}")


@dataclass
class PackageSceneConfig:
    """Single source of truth for scene geometry and initial joint states."""

    initial_box_open: bool = False
    box_scale: float = 0.25
    box_xy: tuple[float, float] = (0.08, 0.08)
    closed_box: tuple[float, float] = (-2.2237906456, -2.0294688542)
    # OPEN = closed + 110 deg, i.e. each flap folded back past vertical.
    #
    # This used to be the joints' UPPER LIMITS (2.4127, 2.5307), which are 266
    # deg from closed -- the flap swings all the way round and lies flat on the
    # box again. Measured flap-tip height above the table: closed 167 mm,
    # vertical (delta 79 deg) 269 mm, folded back (delta 110 deg) 265 mm, and
    # at the upper limit 160 mm, i.e. SHUT. That is why the extract-apple
    # subtask, which starts from `fully_open_box`, opened with the package
    # closed and the apple sealed inside.
    #
    # 110 deg is past the 90 deg the open predicate requires and still leaves
    # the flap standing clear of the opening.
    fully_open_box: tuple[float, float] = (-0.3063906456, -0.1120688542)
    plate_pos: tuple[float, float, float] = (0.08, 0.54, 0.012)
    apple_half_height: float = 0.03683
    package_length: float = 0.2116

    @property
    def box_pos(self):
        # +0.02 raw seats the mesh on the table; the extra +0.004 raw (1 mm at the
        # 0.25 scale) lifts every bottom face off the table plane so nothing
        # z-fights with the tabletop. floor_z / seam_z / apple_pos
        # all derive from this, so the task geometry moves with it.
        return (self.box_xy[0], self.box_xy[1], (0.391345 + 0.02 + 0.004) * self.box_scale)

    @property
    def box_floor_z(self):
        return self.box_pos[2] - 0.165803 * self.box_scale

    @property
    def seam_z(self):
        return self.box_pos[2] + 0.243588 * self.box_scale

    @property
    def apple_pos(self):
        return (
            self.box_xy[0], self.box_xy[1],
            self.box_floor_z + self.apple_half_height + 0.002,
        )

    @property
    def plate_top_z(self):
        return self.plate_pos[2] + 0.00555

    @property
    def initial_box_qpos(self):
        return self.fully_open_box if self.initial_box_open else self.closed_box


@dataclass(frozen=True)
class TaskThresholds:
    open_angle: float = np.deg2rad(90.0)
    reach_apple: float = 0.09
    lift_apple: float = 0.03
    apple_clearance: float = 0.005
    plate_xy: float = 0.05
    plate_height: float = 0.02
    stable_speed: float = 0.05
    release_apple: float = 0.10


@dataclass
class SceneHandles:
    table: TableSceneBuilder
    box: Optional[Articulation] = None
    plate: Optional[Actor] = None
    apple: Optional[Actor] = None


def _flap_tab_pose(urdf_path: Path, link_name: str, scale: float):
    """Where to mould the pull tab on one flap, in the SCALED link frame.

    The flap's link frame sits on its hinge, so the free edge is simply the
    collision geometry's far corner in the plane normal to the hinge axis (the
    URDF gives both flaps a z hinge). The tab is centred on that edge, halfway
    along it, and sticks straight out.
    """
    root = ET.parse(urdf_path).getroot()
    link = root.find(f"./link[@name='{link_name}']")
    chunks = []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            continue
        origin = collision.find("origin")
        offset = np.zeros(3)
        if origin is not None:
            offset = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        verts = []
        with open(urdf_path.parent / mesh.get("filename")) as handle:
            for line in handle:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.split()[1:4]])
        chunks.append(np.asarray(verts, dtype=np.float64) + offset)
    points = np.vstack(chunks) * float(scale)
    radial = points[:, :2]
    far = radial[np.argmax(np.linalg.norm(radial, axis=1))]
    direction = far / max(float(np.linalg.norm(far)), 1e-9)
    hinge_mid = float((points[:, 2].min() + points[:, 2].max()) / 2.0)
    half = np.asarray(FLAP_TAB_SIZE, dtype=np.float64) / 2.0
    centre = np.array([far[0] + direction[0] * half[1],
                       far[1] + direction[1] * half[1],
                       hinge_mid])
    yaw = float(np.arctan2(direction[1], direction[0]))
    quat = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
    # Tab local axes: x = out from the edge, y = along the edge, z = hinge.
    return centre, np.array([half[1], half[0], half[2]]), quat


def _add_flap_tabs(builder, urdf_path: Path, scale: float) -> None:
    """Give both package flaps a finger-lift tab (see FLAP_TAB_SIZE)."""
    links = {lb.name: lb for lb in builder.link_builders}
    missing = set(FLAP_TAB_LINKS) - links.keys()
    if missing:
        raise RuntimeError(
            "package flap links missing, cannot add pull tabs: "
            + ", ".join(sorted(missing))
        )
    for link_name in FLAP_TAB_LINKS:
        lb = links[link_name]
        material = lb.collision_records[0].material if lb.collision_records else None
        centre, half, quat = _flap_tab_pose(urdf_path, link_name, scale)
        tab_pose = sapien.Pose(p=centre, q=quat)
        lb.add_box_collision(pose=tab_pose, half_size=half, material=material,
                             density=1000.0)
        lb.add_box_visual(pose=tab_pose, half_size=half,
                          material=sapien.render.RenderMaterial(
                              base_color=[0.85, 0.72, 0.55, 1.0]))


def _load_urdf(scene, path, scene_idx, name, scale, pose, fixed=True,
               decomposition="none", flap_tabs=False):
    """Load a PartNet URDF.

    ``decomposition="coacd"`` is required for any part with a HOLE through it.
    SAPIEN's default takes one convex hull per collision mesh, and the hull of
    a closed loop is a solid disc (measured on the retired scissors asset:
    the hull of each handle mesh was 3.0-3.6x the volume of the handle
    itself, filling the finger holes). COACD decomposes each mesh into
    true convex parts and caches the result next to the mesh (*.coacd.ply), so
    the cost is paid once.

    The package carton does NOT need it: its panels are flat, each already
    loads as its own thin hull, and the measured collision bounds match the
    visual bounds exactly.
    """
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fixed
    loader.scale = scale
    loader.load_multiple_collisions_from_file = True
    loader.multiple_collisions_decomposition = decomposition
    if decomposition == "coacd":
        loader.multiple_collisions_decomposition_params = dict(COACD_PARAMS)
    builder = loader.parse(str(path))["articulation_builders"][0]
    if flap_tabs:
        _add_flap_tabs(builder, Path(path), scale)
    builder.set_scene_idxs([scene_idx])
    builder.initial_pose = pose
    return builder.build(name=name, fix_root_link=fixed)


def _load_robocasa_actor(scene, relative_path, scene_idx, name, pose, kinematic,
                         single_convex_hull: bool = False):
    path = ROBOCASA_OBJECT_DIR / relative_path / "model.xml"
    loader = scene.create_mjcf_loader()
    loader.scale = np.ones(3, dtype=np.float32)
    loader.visual_groups = [1]
    builder = loader.parse(path, package_dir=path.parent)["actor_builders"][0]
    if single_convex_hull:
        # (user: "fingers sometimes end up INSIDE the apple and get
        # stuck"): the RoboCasa apple ships as 32 convex pieces. A fingertip that
        # slips through a seam between pieces is then pushed in conflicting
        # directions by neighbouring pieces and never gets out. An apple is
        # convex to within its stem dimple, so one hull of the visual mesh is
        # both a better contact model and a well-defined depenetration
        # direction. Mass is set explicitly afterwards, so density is moot.
        visual = builder.visual_records[0]
        builder.collision_records = []
        builder.add_convex_collision_from_file(
            filename=visual.filename, scale=visual.scale, pose=visual.pose,
        )
    builder.set_scene_idxs([scene_idx])
    builder.initial_pose = pose
    return builder.build_kinematic(name) if kinematic else builder.build_dynamic(name)


def _set_task_object_energy_fence(task_object):
    """Apply the project task-object velocity ceilings, never to the robot."""
    items = (
        task_object.get_links()
        if hasattr(task_object, "get_links")
        else (task_object,)
    )
    for item in items:
        for raw in item._objs:
            component = raw
            if not hasattr(component, "set_max_depenetration_velocity"):
                component = raw.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                )
            if component is None:
                continue
            component.set_max_depenetration_velocity(0.5)
            component.set_max_linear_velocity(3.0)


# --------------------------------------------------------------------------
# Explicit physical parameters.
#
# Neither loader below carries a material or a mass: the URDF box
# fell through to the sim_profiles fallback contact material (0.30) and to the
# PhysX default density (1000 kg/m3), and the RoboCasa apple/plate inherited
# raw MJCF values (mu 0.95, rho 100). Every task object therefore states its
# material class, its dominant contact pair and its real-world mass here.
#
# Real-world references:
#   small cardboard package  0.15-0.30 kg (3 L hollow carton)
#   cardboard flap           ~0.02 kg
#   apple                    130-250 g (kept identical to tasks/juice_take_apple_to_blender)
#   ceramic plate            300-500 g
# --------------------------------------------------------------------------
PACKAGE_BODY_MASS = 0.25       # kg
PACKAGE_FLAP_MASS = 0.02       # kg, per flap
APPLE_MASS = 0.15              # kg, matches the juice family
APPLE_SINGLE_CONVEX_HULL = True   # one hull of the visual mesh instead of RoboCasa's 32 pieces (see _load_robocasa_actor)
PLATE_MASS = 0.35              # kg

# Cardboard flap crease, modelled as a FRICTION hinge, not a return spring.
#
# It used to be a position drive whose target was the CLOSED angle, with
# stiffness 0.15 and force limit 0.10 N*m. A position drive that far from its
# target is saturated, so the flap felt a constant 0.10 N*m pulling it shut AT
# EVERY ANGLE -- fold it back 180 deg and it still crawled closed. Real board
# does the opposite: it takes a set and stays where you bend it.
#
# Coulomb friction reproduces that with no target at all. The flap is 0.02 kg
# with its centre ~0.055 m from the hinge, so gravity can apply at most
# 0.02 * 9.81 * 0.055 = 0.011 N*m about the hinge; 0.05 N*m of friction holds
# it at any angle with 4.5x margin, while a hand pushing on the panel applies
# far more than that and moves it freely.
FLAP_DRIVE_STIFFNESS = 0.0     # N*m/rad -- no return spring, see above
FLAP_DRIVE_DAMPING = 0.05      # N*m*s/rad
FLAP_DRIVE_FORCE_LIMIT = 0.10  # N*m
# ⚠️ `joint.set_friction` takes a SETTING, not newton-metres. Measured breakaway
# torque on THIS hinge (qf re-applied every substep, since set_qf survives one):
#       setting 0.05 -> 0.009 N*m    0.30 -> 0.021 N*m    1.00 -> 0.076 N*m
# and measured 30 s drift of a flap left where it was put:
#       setting        vertical     folded back (the S2 entry pose)
#       0.05           -14.6 deg    +121 deg (flops right over)
#       0.30            -0.0 deg      +4.6 deg
#       1.00            -0.0 deg       0.0 deg
# 0.05 was calibrated against a 5 s check and looked fine; over a real episode
# it does not hold at all. 1.00 holds both poses dead still and its 0.076 N*m
# is still inside the 0.01-0.10 N*m a real crease resists -- 0.7 N of push at
# the flap edge.
FLAP_JOINT_FRICTION = 1.00     # setting, = 0.076 N*m breakaway

# Pull tab moulded onto each flap's free edge. A bare cardboard flap is a
# zero-thickness panel: a dexterous hand has nothing to pinch, which is why
# opening the package needed the fingers to slide underneath a panel lying flat
# on the box. The tab is a real feature of this kind of carton (the die-cut
# finger lift) and gives the thumb/index pair something to hook.
FLAP_TAB_SIZE = (0.030, 0.012, 0.006)   # m, full size along edge/out/thick
FLAP_TAB_LINKS = ("link_0", "link_1")

COACD_PARAMS = {"threshold": 0.10, "max_convex_hull": 24}



def _rigid_components(task_object):
    """Yield the PhysX component of every body in an Actor or Articulation."""
    items = (
        task_object.get_links()
        if hasattr(task_object, "get_links")
        else (task_object,)
    )
    for item in items:
        for raw in item._objs:
            component = raw
            if not hasattr(component, "get_collision_shapes"):
                component = raw.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                ) or raw.find_component_by_type(
                    sapien.physx.PhysxRigidStaticComponent
                )
            if component is not None:
                yield item, component


def _apply_object_material(task_object, material) -> int:
    """Assign an explicit contact material to every collision shape."""
    count = 0
    for _, component in _rigid_components(task_object):
        for shape in component.get_collision_shapes():
            shape.set_physical_material(material)
            count += 1
    if count == 0:
        raise RuntimeError(
            f"no collision shape found while materialising {task_object}"
        )
    return count


def _set_component_mass(component, mass: float) -> None:
    """Set an explicit mass with a realizable inertia (see tasks/config/mass_properties)."""
    set_realizable_mass(component, mass)


def _set_body_masses(task_object, mass_by_link: dict) -> None:
    """Apply explicit per-link masses; a link named in the map must exist."""
    seen = set()
    for item, component in _rigid_components(task_object):
        short = str(item.name).split("_")[-1]
        if short in mass_by_link:
            _set_component_mass(component, mass_by_link[short])
            seen.add(short)
    missing = set(mass_by_link) - seen
    if missing:
        raise RuntimeError(f"mass target links not found: {sorted(missing)}")


def build_package_scene(env, config: PackageSceneConfig) -> SceneHandles:
    """Build the table and only the task objects enabled by the environment.

    The complete environment enables every object.  A single-subtask class can
    disable an object with its ``LOAD_*`` flags.  An object disabled here is
    absent from rendering, observations, collision detection, and simulation;
    this is stronger and less error-prone than merely hiding its visual body.
    """
    table = TableSceneBuilder(env=env)
    table.build()
    boxes, plates, apples = [], [], []
    for scene_idx in range(env.num_envs):
        if env.LOAD_PACKAGE:
            boxes.append(_load_urdf(
                env.scene, PACKAGE_URDF_PATH, scene_idx, f"package_{scene_idx}",
                config.box_scale, sapien.Pose(p=config.box_pos), fixed=True,
                flap_tabs=True,
            ))
        if env.LOAD_PLATE:
            plates.append(_load_robocasa_actor(
                env.scene, "plate/plate_1", scene_idx, f"plate_{scene_idx}",
                sapien.Pose(p=config.plate_pos), kinematic=True,
            ))
        if env.LOAD_APPLE:
            apples.append(_load_robocasa_actor(
                env.scene, "apple/apple_0", scene_idx, f"apple_{scene_idx}",
                sapien.Pose(p=config.apple_pos), kinematic=False,
                single_convex_hull=APPLE_SINGLE_CONVEX_HULL,
            ))

    for task_object in [*boxes, *apples]:
        _set_task_object_energy_fence(task_object)

    # Explicit contact materials. One PhysX scalar per surface cannot make
    # every contact pair correct (see tasks/config/object_materials.py), so the
    # dominant pair is chosen per object:
    #
    #   apple           -> "pad": it is picked up, so grip fidelity decides
    #       whether the task succeeds.
    #   package, plate  -> "table": NOT because they slide (both roots are
    #       fixed, so the tabletop pair never actually loads). The reason is
    #       that the pad solve for cardboard/ceramic clamps the object to
    #       mu = 0, which would make the apple sit INSIDE the package on a
    #       frictionless wall (apple-package pair 0.00). Solving for the table
    #       gives cardboard mu = 0.70, so apple-package becomes 0.35 against a
    #       real ~0.50. The cost is a fingertip-flap pair of 0.95 vs a real
    #       0.55, i.e. the flaps are grippier than paper -- an "easier than
    #       real" gap, which is preferable to a "physically broken" one.
    box_material = object_material("cardboard", dominant="table")
    apple_material = object_material("produce", dominant="pad")
    plate_material = object_material("ceramic", dominant="table")
    for box in boxes:
        _apply_object_material(box, box_material)
        _set_body_masses(box, {"0": PACKAGE_FLAP_MASS,
                               "1": PACKAGE_FLAP_MASS,
                               "2": PACKAGE_BODY_MASS})
    for plate in plates:
        _apply_object_material(plate, plate_material)
        for _, component in _rigid_components(plate):
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                _set_component_mass(component, PLATE_MASS)
    for apple in apples:
        _apply_object_material(apple, apple_material)
        for _, component in _rigid_components(apple):
            _set_component_mass(component, APPLE_MASS)

    for box in boxes:
        for joint_idx, joint in enumerate(box.get_active_joints()[:2]):
            joint.set_friction(FLAP_JOINT_FRICTION)
            joint.set_drive_properties(
                stiffness=FLAP_DRIVE_STIFFNESS,
                damping=FLAP_DRIVE_DAMPING,
                force_limit=FLAP_DRIVE_FORCE_LIMIT,
            )
            # No drive target: the hinge is friction-held (see
            # FLAP_DRIVE_STIFFNESS). Leaving a target here with stiffness 0 is
            # harmless, but writing none makes the model unambiguous.
    handles = SceneHandles(
        table=table,
        box=Articulation.merge(boxes, "package") if boxes else None,
        plate=Actor.merge(plates, "plate") if plates else None,
        apple=Actor.merge(apples, "apple") if apples else None,
    )
    for actor in (handles.box, handles.plate, handles.apple):
        if actor is not None:
            env.add_to_state_dict_registry(actor)
    return handles


class PackageTaskEvaluator:
    """Two-stage monotonic state machine shared by full and subtask envs.

    Multi-step achievements are latched explicitly.  A stage therefore means
    "this happened earlier in the episode AND the requested final state holds",
    rather than requiring mutually exclusive grasp/release poses in one frame.
    """

    # [stage, apple_picked, apple_rest_steps, apple_prev_pos xyz].
    # `apple_prev_pos` was added. It is the reference of a pose finite
    # difference -- the same "did it move this step" gate DrawerPlug and Juice
    # use -- so a restore that does not bring it back differences the apple
    # against a pose from a different episode, and the settle counter reads a
    # phantom jump. It lives on the evaluator object rather than the env, which
    # is why the state-restore gate only saw it once it learned to walk
    # task-owned helper objects.
    TASK_STATE_LEGACY_WIDTHS = (3, 6)
    TASK_STATE_WIDTH = 8
    PLACE_SETTLE_STEPS = 40       # 2 s at the 20 Hz control rate
    PLACE_STILL_STEP = 0.002      # m per control step that still counts as "not moving"

    def __init__(self, config, device, num_envs, thresholds=TaskThresholds()):
        self.config = config
        self.device = device
        self.num_envs = num_envs
        self.thresholds = thresholds
        self.stage = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.apple_picked = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.apple_rest_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.apple_prev_pos = torch.zeros((num_envs, 3), device=device)
        # The finite difference is computed ONCE per control step and its verdict
        # is held, so `evaluate()` is idempotent and -- the reason this exists --
        # so the recorded state is sufficient to reproduce it. Updating
        # `apple_prev_pos` on every call overwrote the reference with the current
        # position, so a restored frame differenced the apple against itself and
        # always read "still" (measured: `apple_stable` disagreed on
        # 2 of 15 restored frames in every PackageOpening episode).
        self.apple_still = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.apple_motion_last_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )

    def reset(self, env_idx):
        self.stage[env_idx] = 0
        self.apple_picked[env_idx] = False
        self.apple_rest_steps[env_idx] = 0
        self.apple_prev_pos[env_idx] = 0
        self.apple_still[env_idx] = False
        self.apple_motion_last_step[env_idx] = -1

    def _active(self, active):
        if active is None:
            return torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        return active

    @staticmethod
    def _tcp_distance(env, object_position):
        """Nearest manipulation TCP distance for one- or two-arm agents.

        On dual agents ``agent.tcp`` aliases the right palm.  Reading both
        explicit palm links makes grasp success symmetric, while a release
        threshold on the minimum distance requires every palm to be clear.
        """
        # measure from the FINGERTIP-WEIGHTED point of each hand
        # (0.7 * mean fingertip + 0.3 * palm, the dual agents' own tcp
        # definition), not from the bare palm-link origin. The palm origin sits
        # ~0.10-0.12 m behind the fingers, so an apple held in the fingers read
        # 0.09-0.12 m "away" and the grasp gate (< 0.09) never latched -- the
        # operator could never trigger the place stage.
        agent = env.agent
        tcp_positions = []
        for side in ("left", "right"):
            tips = getattr(agent, f"{side}_tip_links", None)
            palm = getattr(agent, f"{side}_palm_link", None)
            if tips and palm is not None:
                tcp_positions.append(0.7 * torch.stack([l.pose.p for l in tips], 0).mean(0) + 0.3 * palm.pose.p)
        if not tcp_positions:
            tips = getattr(agent, "tip_links", None)
            palm = getattr(agent, "palm_link", None)
            if tips and palm is not None:
                tcp_positions = [0.7 * torch.stack([l.pose.p for l in tips], 0).mean(0) + 0.3 * palm.pose.p]
            else:
                tcp_positions = [agent.tcp.pose.p]
        distances = torch.linalg.norm(
            torch.stack(tcp_positions, dim=1) - object_position[:, None, :],
            dim=2,
        )
        return torch.min(distances, dim=1).values

    def evaluate_box(self, env):
        """Evaluate the shared open-package predicate."""
        t, cfg = self.thresholds, self.config
        box_qpos = env.box.get_qpos()[:, :2]
        closed_ref = torch.tensor(
            cfg.closed_box, dtype=box_qpos.dtype, device=self.device
        )
        raw_delta = torch.abs(box_qpos - closed_ref)
        box_delta = torch.where(
            raw_delta < np.deg2rad(0.1), torch.zeros_like(raw_delta), raw_delta
        )
        return dict(
            box_qpos=box_qpos,
            box_delta=box_delta,
            box_opened=torch.all(box_delta > t.open_angle, dim=1),
        )

    def evaluate_apple(self, env, active=None):
        """Update and evaluate the combined extract/place stage."""
        active = self._active(active)
        t, cfg = self.thresholds, self.config
        apple_p = env.apple.pose.p
        tcp_apple = self._tcp_distance(env, apple_p)
        apple_lifted = apple_p[:, 2] > cfg.apple_pos[2] + t.lift_apple
        apple_grasped = (tcp_apple < t.reach_apple) & apple_lifted
        apple_bottom = apple_p[:, 2] - cfg.apple_half_height
        apple_rim_clearance = apple_bottom - cfg.seam_z
        apple_extracted = (
            (apple_rim_clearance > t.apple_clearance)
            & (tcp_apple < t.reach_apple)
        )
        picked_now = apple_grasped & apple_extracted
        self.apple_picked |= active & picked_now

        plate_xy = torch.linalg.norm(
            apple_p[:, :2] - env.plate.pose.p[:, :2], dim=1
        )
        correct_height = (
            torch.abs(apple_bottom - cfg.plate_top_z) < t.plate_height
        )
        # (user): "placed" = the apple sits on the plate, released,
        # and stays put for PLACE_SETTLE_STEPS (2 s at 20 Hz). Stillness is a
        # pose finite difference, not the engine velocity (PhysX reports
        # corrupt velocities for bodies that were just squeezed).
        step = env._elapsed_steps.to(self.apple_motion_last_step.dtype)
        is_new_step = step != self.apple_motion_last_step
        moved = torch.linalg.norm(apple_p - self.apple_prev_pos, dim=1)
        self.apple_still = torch.where(
            is_new_step, moved < self.PLACE_STILL_STEP, self.apple_still
        )
        self.apple_prev_pos = torch.where(
            is_new_step[:, None], apple_p.detach(), self.apple_prev_pos
        )
        still = self.apple_still
        stable = still
        apple_released = tcp_apple > t.release_apple
        resting_now = (
            (plate_xy < t.plate_xy)
            & correct_height
            & still
            & apple_released
        )
        self.apple_rest_steps = torch.where(
            is_new_step,
            torch.where(
                active & resting_now,
                self.apple_rest_steps + 1,
                torch.zeros_like(self.apple_rest_steps),
            ),
            self.apple_rest_steps,
        )
        self.apple_motion_last_step = torch.where(
            is_new_step, step, self.apple_motion_last_step
        )
        apple_in_plate = self.apple_rest_steps >= self.PLACE_SETTLE_STEPS
        completed = self.apple_picked & apple_in_plate
        return dict(
            apple_task_completed=completed,
            tcp_apple_dist=tcp_apple,
            apple_lifted=apple_lifted,
            apple_grasped=apple_grasped,
            apple_bottom_z=apple_bottom,
            apple_rim_clearance=apple_rim_clearance,
            apple_extracted=apple_extracted,
            apple_picked_now=picked_now,
            apple_pick_completed=self.apple_picked,
            plate_xy_dist=plate_xy,
            apple_height_correct=correct_height,
            apple_stable=stable,
            apple_rest_steps=self.apple_rest_steps,
            apple_released=apple_released,
            apple_in_plate=apple_in_plate,
        )

    def evaluate(self, env):
        """Evaluate the current stage and advance by at most one stage."""
        current_stage = self.stage.clone()
        box = self.evaluate_box(env)
        apple = self.evaluate_apple(env, current_stage == 1)

        advance = (
            ((current_stage == 0) & box["box_opened"])
            | ((current_stage == 1) & apple["apple_task_completed"])
        )
        self.stage = torch.where(advance, current_stage + 1, current_stage)
        return dict(
            success=self.stage >= NUM_TASK_STAGES,
            stage=self.stage,
            **box,
            **apple,
        )

    def get_state_tensor(self):
        """Flatten evaluator latches for env-state recording/replay."""
        return torch.cat(
            [
                self.stage[:, None].float(),
                self.apple_picked[:, None].float(),
                self.apple_rest_steps[:, None].float(),
                self.apple_prev_pos.float(),
                self.apple_still[:, None].float(),
                self.apple_motion_last_step[:, None].float(),
            ],
            dim=-1,
        )

    def set_state_tensor(self, task_state, env_idx):
        task_state = torch.as_tensor(
            task_state, dtype=torch.float32, device=self.device
        ).reshape(len(env_idx), -1)
        if task_state.shape[-1] not in (
            *self.TASK_STATE_LEGACY_WIDTHS, self.TASK_STATE_WIDTH
        ):
            widths = ", ".join(str(w) for w in
                               (*self.TASK_STATE_LEGACY_WIDTHS, self.TASK_STATE_WIDTH))
            raise ValueError(
                "package_opening_task_state must have one of "
                f"{widths} fields, got {task_state.shape[-1]}"
            )
        self.stage[env_idx] = task_state[:, 0].long()
        self.apple_picked[env_idx] = task_state[:, 1].bool()
        self.apple_rest_steps[env_idx] = task_state[:, 2].long()
        if task_state.shape[-1] >= 6:
            self.apple_prev_pos[env_idx] = task_state[:, 3:6].float()
        if task_state.shape[-1] >= self.TASK_STATE_WIDTH:
            self.apple_still[env_idx] = task_state[:, 6].bool()
            self.apple_motion_last_step[env_idx] = task_state[:, 7].long()


@register_task(
    "PackageOpeningLong",
    side="both",
    max_episode_steps=800,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="normalized_dense",
    render_mode="rgb_array",
)
class PackageOpeningLongEnv(BaseEnv):
    """Complete two-stage package-opening environment: open the package, take the item out."""

    SUPPORTED_ROBOTS = [
        LONG_TASK_ROBOT_UID,
        DUAL_STRAIGHT_MOBILE_ROBOT_UID,
        SUBTASK_ROBOT_UID,
        STRAIGHT_MOBILE_ROBOT_UID,
        ANGLED_ROBOT_UID,
        STRAIGHT_ROBOT_UID,
            WUJI_MOBILE_ROBOT_UID,
        DUAL_WUJI_MOBILE_ROBOT_UID,
        XHAND_MOBILE_ROBOT_UID,
        DUAL_XHAND_MOBILE_ROBOT_UID,
]
    SUPPORTED_CONTROL_MODES = PACKAGE_SUPPORTED_CONTROL_MODES
    agent: (
        PandaDualSharpaHandAngledMobileBase
        | PandaDualSharpaHandMobileBase
        | PandaSharpaHandRightAngledMobileBase
        | PandaSharpaHandRightMobileBase
        | PandaSharpaHandRightAngled
        | PandaSharpaHandRight
    )

    # The long-horizon environment always loads the complete scene.  Subtask
    # classes override these flags to remove objects irrelevant to training.
    LOAD_PACKAGE = True
    LOAD_APPLE = True
    LOAD_PLATE = True

    def __init__(
        self,
        *args,
        robot_uids=DEFAULT_ROBOT_UID,
        initial_box_open=False,
        robot_init_qpos_noise=FROZEN_ROBOT_INIT_QPOS_NOISE,
        **kwargs,
    ):

        # Keep the task on the fleet-wide Plant even when older teleop entry
        # points redundantly inject {"control_freq": 20}.  Any actual override
        # remains an error.
        requested_sim_config = kwargs.pop("sim_config", None)
        if requested_sim_config is not None:
            if not isinstance(requested_sim_config, dict):
                raise ValueError(
                    "PackageOpening freezes the Plant configuration in "
                    "tasks/config/sim_profiles.py; SimConfig overrides are not allowed"
                )
            allowed = {
                "sim_freq": SIM_FREQ_DEFAULT,
                "control_freq": CONTROL_FREQ,
            }
            unsupported = set(requested_sim_config) - set(allowed)
            mismatched = {
                key: value
                for key, value in requested_sim_config.items()
                if key in allowed and value != allowed[key]
            }
            if unsupported or mismatched:
                raise ValueError(
                    "PackageOpening freezes the Plant configuration in "
                    "tasks/config/sim_profiles.py; only redundant "
                    f"{allowed} values are accepted (unsupported={sorted(unsupported)}, "
                    f"mismatched={mismatched})"
                )

        # The selected robot/controller owns control_mode.  The registration
        # default remains pd_joint_pos for training and nullspace teleop, while
        # controller-based teleop may select the robot's pd_ee_delta_pose path.
        # PackageOpening never rewrites arm or hand controller parameters.
        # no longer frozen at one value. The dual-arm teleop entry point
        # passes 0.0 on purpose (deterministic arms for the operator); training uses the
        # default. The value is part of the recorded env kwargs either way.
        if float(robot_init_qpos_noise) < 0.0:
            raise ValueError("robot_init_qpos_noise must be non-negative")
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)

        self.scene_config = PackageSceneConfig(initial_box_open=initial_box_open)
        # Unconditional: the layout belongs to the shared rig, not to the robot
        # build (see FIXED_BASE_ROOT_POSE above).
        dx, dy = MOBILE_LAYOUT_OFFSET_XY
        cfg = self.scene_config
        cfg.box_xy = (cfg.box_xy[0] + dx, cfg.box_xy[1] + dy)
        cfg.plate_pos = (
            cfg.plate_pos[0] + dx,
            cfg.plate_pos[1] + dy,
            cfg.plate_pos[2],
        )
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # Environment/solver/contact settings come only from the locked fleet
        # profile.  PackageOpening owns task geometry and mechanism values, not
        # Plant parameters.
        return make_sim_config(spacing=3)

    @property
    def _default_sensor_configs(self):
        target = [self.scene_config.box_xy[0], self.scene_config.box_xy[1], 0.10]
        pose = sapien_utils.look_at(
            [target[0] - 0.90, target[1] - 1.03, 0.72], target
        )
        return [CameraConfig("base_camera", pose, 640, 480, 1.05, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        target = [self.scene_config.box_xy[0], self.scene_config.box_xy[1], 0.10]
        pose = sapien_utils.look_at(
            [target[0] - 0.90, target[1] - 1.03, 0.72], target
        )
        return CameraConfig("render_camera", pose, 1920, 1080, 1.0, 0.01, 100)

    @property
    def _robot_base_pose(self):
        """Same rig for every build; only the ROOT LINK differs.

        A trolley build's root is the trolley (arms hang off it at y = +-0.220);
        a trolley-less build's root IS panda_link0. Both resolve to arm roots at
        (-0.870, -0.220, 0). This branch picks the representation, never a
        different rig -- see tasks/config/workspace_rig.py.
        """
        if self.robot_uids in MOBILE_ROBOT_UIDS:
            return FULL_TROLLEY_BASE_POSE
        return FIXED_BASE_ROOT_POSE

    def _load_agent(self, options):
        super()._load_agent(options, self._robot_base_pose)

    def _load_scene(self, options):
        handles = build_package_scene(self, self.scene_config)
        self.table_scene = handles.table
        self._set_work_table_pose()
        if handles.box is not None:
            self.box = handles.box
        if handles.plate is not None:
            self.plate = handles.plate
        if handles.apple is not None:
            self.apple = handles.apple

    def _after_reconfigure(self, options):
        self.task_evaluator = PackageTaskEvaluator(
            self.scene_config, self.device, self.num_envs
        )

    def _set_work_table_pose(self):
        set_work_table_pose(self)

    def _initialize_agent(self, env_idx):
        batch_size = len(env_idx)
        if hasattr(self.agent, "get_reset_pos"):
            qpos = self.agent.get_reset_pos(device=self.device)
        else:
            qpos = torch.as_tensor(
                self.agent.keyframes["rest"].qpos, device=self.device
            )
        # Shared teleop start pose: the operator's hands map onto whatever
        # posture the arms reset into, so every task must start from the same
        # one (tasks/config/workspace_rig.py).
        qpos = apply_arm_init_qpos(self.agent, qpos.expand(batch_size, -1).clone())
        for k, i in enumerate(env_idx.tolist()):
            index, noise = arm_qpos_noise(
                episode_rng(self, i), self.agent, 1, self.robot_init_qpos_noise, device=self.device
            )
            qpos[k, index] += noise[0]
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(self._robot_base_pose)

    def _set_actor_pose(self, actor, position, batch_size, quaternion=None):
        p = torch.tensor(position, dtype=torch.float32, device=self.device).repeat(
            batch_size, 1
        )
        q = None
        if quaternion is not None:
            q = torch.tensor(
                quaternion, dtype=torch.float32, device=self.device
            ).repeat(batch_size, 1)
        actor.set_pose(Pose.create_from_pq(p=p, q=q))

    def _randomize_layout(self, env_idx):
        """Per-episode layout from the episode RNG: the box (with the apple that
        sits inside it) and the plate, uniformly around the nominal layout.

        Draw order per sub-environment: box (dx, dy, yaw), then plate (dx, dy).
        The evaluator reads live poses for the plate and the apple and only the
        table-relative heights from the config, so it needs nothing else."""
        cfg = self.scene_config
        box_p, box_q, apple_p, plate_p = [], [], [], []
        for i in env_idx.tolist():
            rng = layout_rng(self, i)
            dx, dy, yaw = uniform_offset(rng, PACKAGE_BOX_RANDOM_XY, PACKAGE_BOX_RANDOM_YAW_DEG)
            p, q = offset_pose(cfg.box_pos, (1.0, 0.0, 0.0, 0.0), dx, dy, yaw)
            box_p.append(p)
            box_q.append(q)
            apple = attached_point(p, yaw, (cfg.apple_pos[0] - cfg.box_pos[0], cfg.apple_pos[1] - cfg.box_pos[1]))
            apple[2] = cfg.apple_pos[2]
            apple_p.append(apple)
            pdx, pdy, _ = uniform_offset(rng, PACKAGE_PLATE_RANDOM_XY)
            plate_p.append(np.array([cfg.plate_pos[0] + pdx, cfg.plate_pos[1] + pdy, cfg.plate_pos[2]]))
        as_t = lambda rows: torch.tensor(np.asarray(rows), dtype=torch.float32, device=self.device)  # noqa: E731
        if self.LOAD_PACKAGE:
            self.box.set_pose(Pose.create_from_pq(p=as_t(box_p), q=as_t(box_q)))
        if self.LOAD_APPLE:
            self.apple.set_pose(Pose.create_from_pq(p=as_t(apple_p)))
        if self.LOAD_PLATE:
            self.plate.set_pose(Pose.create_from_pq(p=as_t(plate_p)))

    def _initialize_episode(self, env_idx, options):
        batch_size = len(env_idx)
        self.table_scene.initialize(env_idx)
        cfg = self.scene_config
        self._set_work_table_pose()
        self._randomize_layout(env_idx)   # first episode-RNG draw: layout (see layout_rng)
        self._initialize_agent(env_idx)
        box_qpos = torch.zeros((batch_size, self.box.max_dof), device=self.device)
        box_qpos[:, :2] = torch.tensor(cfg.initial_box_qpos, device=self.device)
        self.box.set_qpos(box_qpos)
        self.box.set_qvel(torch.zeros_like(box_qpos))
        self.apple.set_linear_velocity(
            torch.zeros((batch_size, 3), device=self.device)
        )
        self.apple.set_angular_velocity(
            torch.zeros((batch_size, 3), device=self.device)
        )
        self.task_evaluator.reset(env_idx)

    def evaluate(self):
        return self.task_evaluator.evaluate(self)

    def _get_obs_extra(self, info):
        return dict(
            stage=info["stage"][:, None],
            box_qpos=info["box_qpos"],
            box_pose=self.box.pose.raw_pose,
            apple_pose=self.apple.pose.raw_pose,
            plate_pose=self.plate.pose.raw_pose,
            apple_pick_completed=info["apple_pick_completed"][:, None],
        )

    def get_state_dict(self):
        """Persist all stage/history latches required by state replay."""
        state = super().get_state_dict()
        state["package_opening_task_state"] = (
            self.task_evaluator.get_state_tensor()
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("package_opening_task_state", None)
        if env_idx is None:
            restore_idx = torch.arange(self.num_envs, device=self.device)
            super().set_state_dict(state)
        else:
            restore_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            ).flatten()
            super().set_state_dict(state, restore_idx)
        if isinstance(task_state, dict):
            task_state = task_state.get("value")
        if task_state is not None:
            self.task_evaluator.set_state_tensor(task_state, restore_idx)
        if env_idx is not None:
            # A demo-state restore starts a fresh episode clock at zero while the
            # register says "last differenced at step N", so the reset-time
            # get_info() would read as a new control step and difference the
            # apple against the pose it was just restored to. Rebase, as
            # CabinetPour, Screwing, DrawerPlug and MealPreparation do.
            self.task_evaluator.apple_motion_last_step[restore_idx] = (
                self._elapsed_steps[restore_idx].to(torch.long)
            )

    def compute_dense_reward(self, obs, action, info):
        return info["stage"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / NUM_TASK_STAGES


class PackageOpeningSubtaskEnv(PackageOpeningLongEnv):
    """Shared base for the two package_opening subtask envs."""

    TARGET_STAGE = 0

    def __init__(self, *args, robot_uids=SUBTASK_ROBOT_UID, **kwargs):
        kwargs.pop("initial_box_open", None)
        super().__init__(
            *args,
            robot_uids=robot_uids,
            initial_box_open=self.TARGET_STAGE >= 1,
            **kwargs,
        )

    def _set_actor_pose(self, actor, position, batch_size, quaternion=None):
        p = torch.tensor(position, dtype=torch.float32, device=self.device).repeat(
            batch_size, 1
        )
        q = None
        if quaternion is not None:
            q = torch.tensor(
                quaternion, dtype=torch.float32, device=self.device
            ).repeat(batch_size, 1)
        actor.set_pose(Pose.create_from_pq(p=p, q=q))

    def _initialize_episode(self, env_idx, options):
        cfg, batch_size = self.scene_config, len(env_idx)
        self.table_scene.initialize(env_idx)
        self._set_work_table_pose()
        self._randomize_layout(env_idx)   # first episode-RNG draw: layout (see layout_rng)
        self._initialize_agent(env_idx)

        if self.LOAD_PACKAGE:
            box_qpos = torch.zeros((batch_size, self.box.max_dof), device=self.device)
            box_qpos[:, :2] = torch.tensor(
                cfg.initial_box_qpos, device=self.device
            )
            self.box.set_qpos(box_qpos)
            self.box.set_qvel(torch.zeros_like(box_qpos))

        if self.LOAD_APPLE:
            self.apple.set_linear_velocity(
                torch.zeros((batch_size, 3), device=self.device)
            )
            self.apple.set_angular_velocity(
                torch.zeros((batch_size, 3), device=self.device)
            )

        self.task_evaluator.reset(env_idx)
        self.task_evaluator.stage[env_idx] = self.TARGET_STAGE

    @staticmethod
    def _near_score(distance, scale):
        return 1 / (1 + distance / scale)

    def _subtask_progress(self, info):
        cfg, t = self.scene_config, self.task_evaluator.thresholds
        if self.TARGET_STAGE == 0:
            return torch.mean(
                torch.clamp(info["box_delta"] / t.open_angle, 0, 1),
                dim=1,
            )

        if self.TARGET_STAGE == 1:
            apple_p = self.apple.pose.p
            near = self._near_score(info["tcp_apple_dist"], t.reach_apple)
            lift = torch.clamp(
                (apple_p[:, 2] - cfg.apple_pos[2]) / t.lift_apple,
                0,
                1,
            )
            initial_clearance = (
                cfg.apple_pos[2] - cfg.apple_half_height - cfg.seam_z
            )
            clearance = torch.clamp(
                (info["apple_rim_clearance"] - initial_clearance)
                / (t.apple_clearance - initial_clearance),
                0,
                1,
            )
            pick_progress = (near + lift + clearance) / 3
            height_error = torch.abs(
                info["apple_bottom_z"] - cfg.plate_top_z
            )
            place_progress = (
                self._near_score(info["plate_xy_dist"], t.plate_xy)
                + 1 - torch.clamp(height_error / t.plate_height, 0, 1)
                + info["apple_stable"].float()
                + info["apple_released"].float()
            ) / 4
            return torch.where(
                info["apple_pick_completed"],
                0.5 + 0.5 * place_progress,
                0.5 * pick_progress,
            )

        raise ValueError(f"Unsupported target stage: {self.TARGET_STAGE}")

    def _evaluate_subtask(self):
        """Use the exact same stage predicates as the long-horizon evaluator."""
        active = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.TARGET_STAGE == 0:
            info = self.task_evaluator.evaluate_box(self)
            success = info["box_opened"]
        elif self.TARGET_STAGE == 1:
            info = self.task_evaluator.evaluate_apple(self, active)
            success = info["apple_task_completed"]
        else:
            raise ValueError(f"Unsupported target stage: {self.TARGET_STAGE}")

        stage = torch.full(
            (self.num_envs,),
            self.TARGET_STAGE,
            dtype=torch.long,
            device=self.device,
        ) + success.long()
        info.update(success=success, stage=stage)
        return info

    def evaluate(self):
        info = self._evaluate_subtask()
        info["target_stage"] = torch.full_like(info["stage"], self.TARGET_STAGE)
        info["subtask_progress"] = self._subtask_progress(info)
        return info

    def _get_obs_extra(self, info):
        obs = {
            "stage": info["stage"][:, None],
            "target_stage": info["target_stage"][:, None],
            "subtask_progress": info["subtask_progress"][:, None],
        }
        if self.LOAD_PACKAGE:
            obs["box_qpos"] = self.box.get_qpos()[:, :2]
            obs["box_pose"] = self.box.pose.raw_pose
        if self.LOAD_APPLE:
            obs["apple_pose"] = self.apple.pose.raw_pose
        if self.LOAD_PLATE:
            obs["plate_pose"] = self.plate.pose.raw_pose
        if self.TARGET_STAGE == 1:
            obs["apple_pick_completed"] = info[
                "apple_pick_completed"
            ][:, None]
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].float()

    def compute_dense_reward(self, obs, action, info):
        return info["subtask_progress"] + info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 2


def run_env_cli(env_id=ENV_ID, output_dir="videos/package_opening"):
    """Common executable used by the full task and three stage files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list", action="store_true",
        help="List the stage tasks and their success criteria, then exit.",
    )
    parser.add_argument("--env_id", default=env_id)
    parser.add_argument("--robot_uids", default=None)
    parser.add_argument("--control_mode", default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--reward_mode", default="normalized_dense")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output_dir", default=output_dir)
    args = parser.parse_args()

    if args.robot_uids is None:
        args.robot_uids = (
            LONG_TASK_ROBOT_UID if args.env_id == ENV_ID else SUBTASK_ROBOT_UID
        )

    if args.list:
        print_task_list()
        return

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        reward_mode=args.reward_mode,
        robot_uids=args.robot_uids,
        control_mode=args.control_mode,
        render_mode="rgb_array" if args.record else None,
        render_backend="gpu" if args.record else "none",
    )
    if args.record:
        env = RecordEpisode(
            env,
            output_dir=args.output_dir,
            save_trajectory=False,
            save_video_trigger=lambda episode_idx: True,
            max_steps_per_video=args.steps,
            video_fps=25,
        )
    _, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    print("Environment", args.env_id)
    print("Robot", env.unwrapped.agent.uid)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    for step in range(args.steps):
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if args.record:
            env.render()
        if step % 25 == 0:
            print(
                f"step={step:04d} stage={int(info['stage'][0])}/{NUM_TASK_STAGES} "
                f"reward={float(reward[0]):.3f} success={bool(info['success'][0])}"
            )
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()


if __name__ == "__main__":
    run_env_cli()

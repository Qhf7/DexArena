"""
Final Blender023 lid-placement task for demo collection, RL, and IL.

Registered environments
-----------------------
PandaGripperPlaceBlenderLid-v1
SharpaPlaceBlenderLid-v1
WujiPlaceBlenderLid-v1

Initial condition
-----------------
- Blender023 is fixed on the table and faces the robot.
- Its separate lid is a dynamic rigid body resting on the table.
- The target pose aligns the lid insertion cylinder with the MJCF `reg_int`
  pitcher axis and places the lid flange directly on the pitcher rim.
- The robot starts from a fixed rest configuration, optionally with arm noise.
- Optional lid x/y/yaw noise can be enabled after fixed-pose validation.

Action
------
The environment does not transform the controller action. Demonstrations save
the exact action sent to `env.step()`.

Default controller: pd_joint_pos.

Panda:
    absolute arm-joint position targets followed by the gripper command.
    The exact action dimension and bounds are defined by `env.action_space`.

Dexterous-hand robots:
    absolute Panda arm-joint targets followed by the selected hand's native
    joint-position targets. The environment never crops, reorders, or remaps
    the controller action.

Task state added by `_get_obs_extra()`
--------------------------------------
Raw task state:
    tcp_pose                    (7,)
    all_tip_positions           (3 * num_tips,)
    lid_pose                    (7,)
    lid_linear_velocity         (3,)
    lid_angular_velocity        (3,)
    lid_grasp_point             (3,)
    seat_target_position        (3,)
    lid_to_seat_position        (3,)

Derived state:
    reach_distance              (1,)
    lid_xy_error                (1,)
    lid_z_error                 (1,)
    lid_tilt_error              (1,)
    lid_lift_height             (1,)
    lid_linear_speed            (1,)
    lid_angular_speed           (1,)
    min_tip_distance            (1,)
    tcp_to_lid_distance         (1,)
    lid_grasped                 (1,)
    lid_seated                  (1,)
    lid_stable                  (1,)
    end_effector_clear          (1,)
    picked_once                 (1,)
    transported_once            (1,)
    seated_once                 (1,)
    released_once               (1,)
    success_progress            (1,)
    task_success                (1,)

For Panda, `lid_grasped` uses the agent contact-based grasp test. For the
supported dexterous hands it is a geometric estimate requiring at least two fingertips
near the lid knob. Robot joint positions and velocities remain part of the
standard ManiSkill robot proprioception.

Success
-------
Success is latched when all conditions hold for 15 consecutive control steps:

1. the lid was lifted at least 3 cm at some earlier time;
2. current lid-to-seat XY error <= 5 mm;
3. current lid root-height error <= 3 mm;
4. lid tilt <= 5 degrees;
5. lid linear speed <= 0.02 m/s and angular speed <= 0.25 rad/s;

No scripted policy is included. Use teleoperation / VR for demonstrations.
"""

from __future__ import annotations

import base64
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, TASK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from tasks.config.hands import register_task
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from tasks.scene_builders.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import (
    Array,
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)

import copy

from tasks.config.layout_randomization import (
    layout_rng,
    ARM_INIT_QPOS_NOISE_STD,
    episode_rng,
    quat_mul_wxyz,
    rotate_xy,
    uniform_offset,
    yaw_quat_wxyz,
)
from tasks.config.object_materials import object_material
from tasks.config.workspace_rig import (
    WORK_TABLE_X_OFFSET,
    apply_arm_init_qpos,
    platform_root_pose,
    set_work_table_pose,
)
from tasks.config.mass_properties import set_mass as set_realizable_mass
from tasks.config.sim_profiles import make_sim_config

from tasks.robots.panda_gripper_right_mobile_base import PandaGripperRightMobileBase
from tasks.robots.panda_sharpa_hand_right_mobile_base import PandaSharpaHandRightAngledMobileBase
from tasks.robots.panda_wuji_hand_right_mobile_base import PandaWujiHandRightMobileBase
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag

JUICE_ASSET_DIR = TASK_DIR / "assets" / "task_object_meshes" / "juice"
DEFAULT_BLENDER_DIR = JUICE_ASSET_DIR / "Blender023"
DEFAULT_LID_DIR = DEFAULT_BLENDER_DIR / "Blender023_Lid"

HAND_ROBOT_UIDS = {
    "panda_sharpa_hand_right_angled_mobile_base",
    "panda_wuji_hand_right_angled_mobile_base",
    "panda_xhand_hand_right_angled_mobile_base",
}
GRIPPER_ROBOT_UID = "panda_gripper_right_mobile_base"

CONTROL_FREQUENCY = 20
TABLE_TOP_Z = 0.0
TABLE_CLEARANCE = 0.0015
# Blender assembly scale. Dropped to 2/3 at the operator's request: at 1.0 the
# unit stood 405 mm tall with its pitcher rim 435 mm above the table, so BOTH
# dropping the apple in and lifting the lid off put the arm at its limit stop.
# 2/3 gives a 270 mm personal blender with the rim at ~290 mm.
#
# Everything that defines the seat -- pitcher opening, lid plug and flange
# heights, the drop target -- is multiplied by this scale where it is used, so
# the lid/jar fit is preserved exactly. What does NOT scale are the ACCEPTANCE
# tolerances (LID_XY_TOL / LID_Z_TOL / LID_TILT_TOL): against a 2/3 lid they
# are relatively tighter, which is the direction the operator asked for.
GLOBAL_SCALE = 2.0 / 3.0

# The lid alone is loaded 2% larger than the rest of the Blender023 assembly.
#
# Why: the pitcher rim radius is between 83 and 84 mm (bisected: an 83.0 mm lid
# rim drops 11.0 mm through the opening, 84.0 mm holds at 1.9 mm). The lid's own
# VISIBLE rim is 83.38 mm, i.e. a hair under the opening, so it cannot carry
# itself. The asset used to paper over this with a constant r=88 mm collision
# barrel spanning the whole dome -- up to 50 mm of collision with nothing drawn
# there, which is invisible to a camera and therefore poison for any
# vision-driven policy.
#
# The collision now hugs the visual mesh (see the note in
# Blender023_Lid/model.xml) and the lid is scaled up instead, so the surface the
# operator can SEE is the surface that carries the load:
#     rim  83.21 * 1.02 = 84.87 mm  > 84.0 needed to sit on the pitcher rim
#     plug 80.41 * 1.02 = 82.02 mm  < 83.0 allowed to enter the opening
# Every lid-local constant below is multiplied by this on top of the assembly
# scale, so the seat geometry follows automatically.
LID_UPSCALE = 1.02
BLENDER_YAW = -np.pi / 2.0
# Blender + lid position, pulled in twice after operator feedback.
# 0.08 -> -0.32 in x brought the task into reach at all; then -0.46 x / -0.22 y
# after the operator reported that closing the lid put the right arm at nearly
# full extension, which is where the hand jams on withdrawal. Measured from the
# right arm root: blender 0.762 -> 0.622 m, and the whole juice scene now spans
# 0.590-0.818 m against a working band of 0.54-0.90. y also moved from -0.30 to
# -0.22, i.e. 0.08 closer to the fridge; no blender-vs-fridge contact.
BLENDER_ROOT_XY = np.array([-0.46 + WORK_TABLE_X_OFFSET, -0.22], dtype=np.float64)
# Robot platform x. LOCKED to the shared rig (tasks/config/workspace_rig.py):
# table centre +0.180 -> trolley/arm root -0.870, the CabinetPour/Screwing
# anchor. The blender x above rides the same +WORK_TABLE_X_OFFSET translation,
# so the 0.622 m reach measured here is unchanged. See ROBOT_ROOT_P in
# tasks/juice_preparation/juice_preparation_apple_to_blender.py for the full history.
ROBOT_ROOT_POSE = platform_root_pose()

# Measured from the transformed Blender023 meshes at scale=1.
BLENDER_LOCAL_LOWER = np.array(
    [-0.117230, -0.097575, -0.239918],
    dtype=np.float64,
)
BLENDER_LOCAL_UPPER = np.array(
    [0.117230, 0.097575, 0.192817],
    dtype=np.float64,
)
LID_LOCAL_LOWER = np.array(
    [-0.083000, -0.083238, -0.026052],
    dtype=np.float64,
)
LID_LOCAL_UPPER = np.array(
    [0.083016, 0.083056, 0.041538],
    dtype=np.float64,
)

# Exact Blender023 seating geometry read from the RoboCasa MJCF.
#
# `reg_int` is removed from simulation because it renders as a blue preview
# cylinder, but its local center identifies the pitcher opening axis.
PITCHER_OPENING_CENTER_LOCAL_XY = np.array(
    [-0.01954737, 0.00488804],
    dtype=np.float64,
)

# Highest point of the pitcher_main geometry in Blender023 local coordinates.
PITCHER_RIM_TOP_LOCAL_Z = 0.192817

# The lower lid cylinder is the insertion section that enters the pitcher.
LID_INSERT_CENTER_LOCAL_XY = np.array(
    [-0.000426113427849633, -0.00009082793274997527],
    dtype=np.float64,
)
# Bottom of the lowest collision cylinder (the plug that enters the pitcher).
# Derived from the rebuilt, visual-hugging collision stack in
# Blender023_Lid/model.xml: pos z -0.016800, half length 0.009300.
LID_INSERT_BOTTOM_LOCAL_Z = (
    -0.016800
    - 0.009300
)

# Bottom of the LOAD-BEARING RIM, i.e. the lowest cylinder wide enough to catch
# the pitcher rim. In the rebuilt stack that is the short r=83.21 mm band at
# pos z -0.006250, half length 0.001250. It used to be the bottom of the old
# constant r=88 mm barrel (-0.005900); that barrel no longer exists.
LID_FLANGE_BOTTOM_LOCAL_Z = (
    -0.006250
    - 0.001250
)

# Measured offset between the computed rim-contact height and where the lid
# ACTUALLY comes to rest, folded into the target so the seat predicate is
# centred on the resting pose instead of spending its 3 mm tolerance on a
# constant bias. Re-measured after the plug was pulled in to 73.5 mm: with the
# plug no longer taking part in contact the lid settles 1.9 mm lower on the rim.
# Drop the lid from 5 mm above the target and it now rests within 0.2 mm.
LID_SEAT_CLEARANCE = -0.0019

_FALLBACK_TEXTURE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF"
    "gAI/0+3HAAAAAElFTkSuQmCC"
)


@dataclass(frozen=True)
class BlenderGeometry:
    """World placement and scaled local bounds for Blender023."""

    blender_root: np.ndarray
    blender_quat: np.ndarray
    seat_position: np.ndarray
    lid_table_position: np.ndarray
    blender_lower: np.ndarray
    blender_upper: np.ndarray
    lid_lower: np.ndarray
    lid_upper: np.ndarray


def yaw_quat(yaw: float) -> np.ndarray:
    """Return a scalar-first quaternion [w, x, y, z]."""
    half = 0.5 * float(yaw)
    return np.array(
        [np.cos(half), 0.0, 0.0, np.sin(half)],
        dtype=np.float64,
    )


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    w, x, y, z = quat / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def transform_points(
    points: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    rotation = quat_to_matrix(quaternion)
    return points @ rotation.T + np.asarray(position, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-episode layout randomization shared by the whole Juice family
#
# Uniform half-ranges around the nominal layout, every draw from the episode
# RNG. Draw ORDER is fixed -- fridge (dx, dy), blender (dx, dy, yaw), lid
# (dx, dy, yaw) -- and every environment of the family draws all three even
# when it loads fewer objects, so the long task and its subtasks see the same
# layout for the same seed. The apple rides in the fridge, the seat / lid
# table point ride on the blender (see `shifted_blender_geometry`).
# ---------------------------------------------------------------------------
JUICE_FRIDGE_RANDOM_XY = (0.015, 0.015)
JUICE_BLENDER_RANDOM_XY = (0.020, 0.020)
JUICE_BLENDER_RANDOM_YAW_DEG = 10.0
JUICE_LID_RANDOM_XY = (0.030, 0.030)
JUICE_LID_RANDOM_YAW_DEG = 180.0


def sample_juice_layout(rng, lid_half_xy=JUICE_LID_RANDOM_XY, lid_yaw_half_deg=JUICE_LID_RANDOM_YAW_DEG):
    fdx, fdy, _ = uniform_offset(rng, JUICE_FRIDGE_RANDOM_XY)
    bdx, bdy, byaw = uniform_offset(rng, JUICE_BLENDER_RANDOM_XY, JUICE_BLENDER_RANDOM_YAW_DEG)
    ldx, ldy, lyaw = uniform_offset(rng, lid_half_xy, lid_yaw_half_deg)
    return {"fridge": (fdx, fdy), "blender": (bdx, bdy, byaw), "lid": (ldx, ldy, lyaw)}


def shifted_blender_geometry(nominal, dx, dy, yaw):
    """The nominal blender geometry moved by (dx, dy) and turned by `yaw` about the
    blender root's vertical axis: root, quaternion, seat and lid table point follow."""
    g = copy.deepcopy(nominal)
    root_xy = np.asarray(nominal.blender_root[:2], dtype=np.float64)
    for name in ("seat_position", "lid_table_position"):
        point = np.asarray(getattr(nominal, name), dtype=np.float64)
        ox, oy = rotate_xy(float(point[0] - root_xy[0]), float(point[1] - root_xy[1]), yaw)
        moved = point.copy()
        moved[0] = root_xy[0] + ox + dx
        moved[1] = root_xy[1] + oy + dy
        getattr(g, name)[...] = moved
    g.blender_root[0] = nominal.blender_root[0] + dx
    g.blender_root[1] = nominal.blender_root[1] + dy
    g.blender_quat[...] = quat_mul_wxyz(yaw_quat_wxyz(yaw), np.asarray(nominal.blender_quat, dtype=np.float64))
    return g


def compute_blender_geometry(
    scale: float = GLOBAL_SCALE,
    lid_table_offset_xy: np.ndarray | None = None,
) -> BlenderGeometry:
    """Compute Blender023 poses from the actual MJCF seating geometry."""
    scale = float(scale)
    # The lid is loaded LID_UPSCALE larger than the rest of the assembly, so
    # every lid-local measurement has to use that scale, not the assembly one.
    lid_scale = scale * LID_UPSCALE
    blender_lower = BLENDER_LOCAL_LOWER * scale
    blender_upper = BLENDER_LOCAL_UPPER * scale
    lid_lower = LID_LOCAL_LOWER * lid_scale
    lid_upper = LID_LOCAL_UPPER * lid_scale

    root_z = TABLE_TOP_Z + TABLE_CLEARANCE - blender_lower[2]
    blender_root = np.array(
        [BLENDER_ROOT_XY[0], BLENDER_ROOT_XY[1], root_z],
        dtype=np.float64,
    )
    blender_quat = yaw_quat(BLENDER_YAW)

    rotation = quat_to_matrix(blender_quat)

    # Align the lid insertion-cylinder axis with the pitcher opening axis.
    pitcher_axis_world = (
        blender_root
        + rotation
        @ np.array(
            [
                PITCHER_OPENING_CENTER_LOCAL_XY[0] * scale,
                PITCHER_OPENING_CENTER_LOCAL_XY[1] * scale,
                0.0,
            ],
            dtype=np.float64,
        )
    )
    lid_insert_offset_world = (
        rotation
        @ np.array(
            [
                LID_INSERT_CENTER_LOCAL_XY[0] * lid_scale,
                LID_INSERT_CENTER_LOCAL_XY[1] * lid_scale,
                0.0,
            ],
            dtype=np.float64,
        )
    )

    seat_position = pitcher_axis_world - lid_insert_offset_world
    seat_position[2] = (
        blender_root[2]
        + PITCHER_RIM_TOP_LOCAL_Z * scale
        - LID_FLANGE_BOTTOM_LOCAL_Z * lid_scale
        + LID_SEAT_CLEARANCE
    )

    if lid_table_offset_xy is None:
        lid_table_offset_xy = np.array(
            [-0.06, -0.22],
            dtype=np.float64,
        )
    lid_table_position = np.array(
        [
            blender_root[0] + float(lid_table_offset_xy[0]),
            blender_root[1] + float(lid_table_offset_xy[1]),
            TABLE_TOP_Z
            + TABLE_CLEARANCE
            - LID_INSERT_BOTTOM_LOCAL_Z * lid_scale,
        ],
        dtype=np.float64,
    )

    return BlenderGeometry(
        blender_root=blender_root,
        blender_quat=blender_quat,
        seat_position=seat_position,
        lid_table_position=lid_table_position,
        blender_lower=blender_lower,
        blender_upper=blender_upper,
        lid_lower=lid_lower,
        lid_upper=lid_upper,
    )


# -----------------------------------------------------------------------------
# MJCF repair
# -----------------------------------------------------------------------------


def _resolve_reference(
    model_dir: Path,
    compiler_dir: str | None,
    filename: str,
) -> Path:
    path = Path(filename)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)

    if compiler_dir:
        base = Path(compiler_dir)
        candidates.append(
            (base / path)
            if base.is_absolute()
            else (model_dir / base / path)
        )

    candidates.append(model_dir / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = list(model_dir.rglob(path.name))
    return matches[0].resolve() if matches else candidates[-1].resolve()


def _parse_rgba(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    return values if len(values) == 4 else None


def _is_preview_geom(
    geom: ET.Element,
    materials: dict[str, dict[str, str]],
) -> bool:
    # RoboCasa region geoms are placement-sampling markers. Blender023's
    # `reg_int` becomes a visible blue cylinder in SAPIEN, so all regions go.
    if geom.get("class") == "region":
        return True

    descriptor = " ".join(
        filter(
            None,
            (
                geom.get("name"),
                geom.get("mesh"),
                geom.get("material"),
                geom.get("class"),
            ),
        )
    ).lower()

    if any(
        token in descriptor
        for token in (
            "liquid",
            "water",
            "fill",
            "content",
            "inside",
            "interior",
            "inner",
            "fluid",
            "juice",
        )
    ):
        return True

    def is_fill_color(rgba: np.ndarray | None) -> bool:
        if rgba is None:
            return False
        r, g, b, a = rgba.tolist()
        return (
            (b > r + 0.08 and b > g + 0.08 and a < 0.98)
            or (b > 0.45 and r < 0.30 and g < 0.45)
        )

    if is_fill_color(_parse_rgba(geom.get("rgba"))):
        return True

    material_name = geom.get("material")
    return bool(
        material_name
        and material_name in materials
        and is_fill_color(
            _parse_rgba(materials[material_name].get("rgba"))
        )
    )


def _remove_preview_geoms(root: ET.Element) -> list[str]:
    materials = {
        material.get("name"): dict(material.attrib)
        for material in root.findall(".//asset/material")
        if material.get("name")
    }
    removed: list[str] = []

    def visit(parent: ET.Element) -> None:
        for child in list(parent):
            if child.tag == "geom" and _is_preview_geom(
                child,
                materials,
            ):
                removed.append(
                    child.get("name")
                    or child.get("mesh")
                    or child.get("material")
                    or "<unnamed_geom>"
                )
                parent.remove(child)
            else:
                visit(child)

    visit(root)
    return removed


def prepare_mjcf(
    model_dir: Path,
    cache_tag: str,
) -> Path:
    """Create a temporary, SAPIEN-safe copy of a RoboCasa MJCF."""
    model_dir = model_dir.resolve()
    source_xml = model_dir / "model.xml"
    if not source_xml.exists():
        raise FileNotFoundError(source_xml)

    cache_dir = (
        Path(tempfile.gettempdir())
        / "dexarena_blender023"
        / cache_tag
    )
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    fallback_texture = cache_dir / "fallback_texture.png"
    fallback_texture.write_bytes(_FALLBACK_TEXTURE)

    tree = ET.parse(source_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    mesh_dir = (
        compiler.get("meshdir")
        if compiler is not None
        else None
    )
    texture_dir = (
        compiler.get("texturedir")
        if compiler is not None
        else None
    )

    missing_meshes: list[str] = []
    fallback_textures: list[str] = []

    for mesh in root.findall(".//asset/mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        resolved = _resolve_reference(
            model_dir,
            mesh_dir,
            filename,
        )
        if resolved.exists():
            mesh.set("file", str(resolved))
        else:
            missing_meshes.append(filename)

    for texture in root.findall(".//asset/texture"):
        filename = texture.get("file")
        if not filename:
            continue
        resolved = _resolve_reference(
            model_dir,
            texture_dir,
            filename,
        )
        if not resolved.exists():
            resolved = fallback_texture
            fallback_textures.append(filename)
        texture.set("file", str(resolved))

    if missing_meshes:
        raise FileNotFoundError(
            "Missing MJCF meshes:\n"
            + "\n".join(
                f"  - {name}" for name in missing_meshes
            )
        )

    removed = _remove_preview_geoms(root)
    if removed:
        print(
            f"[asset cleanup] {model_dir.name}: "
            f"removed={removed}"
        )
    if fallback_textures:
        print(
            f"[texture fallback] {model_dir.name}: "
            f"missing={fallback_textures}"
        )

    tree.write(
        cache_dir / "model.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
    return cache_dir


# -----------------------------------------------------------------------------
# SAPIEN builders
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Blender023 explicit physical parameters.
#
# The MJCF asset declares geom-level `density="1.0"` and `density="0.001"`
# (model.xml:29,161,162), which OVERRIDE the loader's density kwarg. The whole
# appliance therefore massed 4.1 g, with a 2.4 mg blade and a 1 mg speed knob:
# contact impulses flung the links and the solver was ill-conditioned. Masses
# are now stated per link and total ~2.0 kg, inside the 1.5-3.5 kg range of a
# real countertop blender.
# ---------------------------------------------------------------------------
BLENDER_LID_MASS = 0.12   # kg, real blender lid 50-150 g

BLENDER_LINK_MASSES = {
    "object": 1.20,          # motor base / housing
    "pitcher_main": 0.45,    # jar body
    "pitcher": 0.15,         # jar sub-shell
    "blade": 0.02,           # steel blade set; 2.4 cm3 of mesh -> rho ~7800
    "knob_speed": 0.012,     # moulded knob; 9.6 cm3 of mesh -> rho ~1250
    "power_button": 0.008,   # hollow button cap
    "integrated_lid": 0.12,  # lid, when fused into the appliance
}
BLENDER_DEFAULT_LINK_MASS = 0.05   # any link the asset adds later

# Appliance push button. A real power button takes 2-6 N; 1300 N/m over the
# asset's 3 mm travel gives 3.9 N at full press. The previous 80 N/m delivered
# 0.24 N -- 8-25x too light -- and two of the four blender environments left
# the drive at stiffness 0, i.e. with no return spring at all.
BLENDER_BUTTON_STIFFNESS = 1300.0   # N/m
BLENDER_BUTTON_DAMPING = 2.0        # N*s/m
BLENDER_BUTTON_FORCE_LIMIT = 8.0    # N, must exceed the 3.9 N spring
BLENDER_BUTTON_FRICTION = 0.02      # N

# Speed knob: detented control, 0.02-0.15 N*m. Was force_limit FLT_MAX.
BLENDER_KNOB_FRICTION = 0.05        # N*m
BLENDER_KNOB_DAMPING = 0.05         # N*m*s/rad
BLENDER_KNOB_FORCE_LIMIT = 2.0      # N*m


def _set_component_mass(component, mass: float) -> None:
    """Set an explicit mass with a realizable inertia (see tasks/config/mass_properties)."""
    set_realizable_mass(component, mass)


def configure_blender_bodies(blender) -> None:
    """Give every blender link an explicit mass and every joint an explicit drive.

    Nothing here may fall through to an asset default: the MJCF densities and
    the FLT_MAX force limits are exactly what this replaces.
    """
    for link in blender.get_links():
        for component in link._objs:
            if not (hasattr(component, "mass") and hasattr(component, "inertia")):
                continue
            name = str(getattr(component, "name", ""))
            if name.endswith("dummy_root_0") or name.endswith("_base"):
                continue
            mass = BLENDER_DEFAULT_LINK_MASS
            for key, value in BLENDER_LINK_MASSES.items():
                if name.endswith(key):
                    mass = value
                    break
            _set_component_mass(component, mass)

    for raw in blender._objs:
        for joint in raw.get_active_joints():
            # Match on joint TYPE, not on the name: the articulation prefix of
            # the press-button environment contains the word "button", so a
            # substring test also captured the rotary speed knob.
            is_button = "prismatic" in str(joint.type).lower()
            if is_button:
                joint.set_drive_properties(
                    stiffness=BLENDER_BUTTON_STIFFNESS,
                    damping=BLENDER_BUTTON_DAMPING,
                    force_limit=BLENDER_BUTTON_FORCE_LIMIT,
                    mode="force",
                )
                joint.set_friction(BLENDER_BUTTON_FRICTION)
            else:
                joint.set_drive_properties(
                    stiffness=0.0,
                    damping=BLENDER_KNOB_DAMPING,
                    force_limit=BLENDER_KNOB_FORCE_LIMIT,
                    mode="force",
                )
                joint.set_friction(BLENDER_KNOB_FRICTION)


# Links whose VISUAL body has no collision behind it, or far too little of it,
# so a hand passing over the blender goes straight through what it can see.
# Measured (visual extent minus collision extent, per axis):
#   integrated_lid   NO collision at all, for a 111 x 111 x 45 mm visible lid
#   object           visual stands 18.7 mm proud of collision in z
#   knob_speed       visual stands 6.9 / 12.7 mm proud in x / y
# Only `integrated_lid` is repaired here. The lid is a solid disc, so a convex
# hull of its own visual mesh is exactly right. The base and the pitcher are NOT
# hulled: the base carries the recess the jar seats into and the pitcher must
# stay hollow for the apple, and a convex hull would fill both.
BLENDER_HULL_FROM_VISUAL_LINKS = ("integrated_lid",)


def _repair_blender_visual_only_links(builder, scale: float) -> int:
    """Give `BLENDER_HULL_FROM_VISUAL_LINKS` a convex collision from their mesh."""
    repaired = 0
    for link_builder in builder.link_builders:
        name = str(getattr(link_builder, "name", ""))
        if not name.endswith(BLENDER_HULL_FROM_VISUAL_LINKS):
            continue
        if getattr(link_builder, "collision_records", None):
            continue
        for record in getattr(link_builder, "visual_records", []):
            filename = getattr(record, "filename", "")
            if not filename:
                continue
            link_builder.add_convex_collision_from_file(
                filename=filename,
                pose=record.pose,
                scale=record.scale,
            )
            repaired += 1
    return repaired


def build_blender023(
    scene,
    patched_dir: Path,
    geometry: BlenderGeometry,
    scale: float,
    num_envs: int,
    name: str,
):
    """Build Blender023 as an articulation, with actor fallback."""
    template = MJCFObject(
        scene=scene,
        name=f"{name}_template",
        mjcf_path=str(patched_dir),
        scale=float(scale),
        density=1000,
        friction=(0.95, 0.45, 0.05),
    )
    pose = sapien.Pose(
        p=geometry.blender_root,
        q=geometry.blender_quat,
    )
    scene_indices = list(range(num_envs))

    if hasattr(template, "articulation_builder"):
        builder = template.articulation_builder
        _repair_blender_visual_only_links(builder, float(scale))
        builder.set_scene_idxs(scene_indices)
        builder.initial_pose = pose
        builder.disable_self_collisions = True
        if hasattr(builder, "set_name"):
            builder.set_name(name)
        blender = builder.build()
        template.articulation = blender
        configure_task_object_physics(blender, "hard_plastic")
        configure_blender_bodies(blender)
        return blender, template, True

    if hasattr(template, "actor_builder"):
        builder = template.actor_builder
        builder.set_scene_idxs(scene_indices)
        builder.initial_pose = pose
        blender = builder.build_static(name=name)
        template.actor = blender
        configure_task_object_physics(blender, "hard_plastic")
        return blender, template, False

    raise RuntimeError(
        "Blender023 produced neither an articulation nor actor builder"
    )


def build_blender_lid(
    scene,
    patched_dir: Path,
    scale: float,
    num_envs: int,
    position: np.ndarray,
    quaternion: np.ndarray,
    name: str,
    dynamic: bool,
    mass: float = BLENDER_LID_MASS,
):
    """Build the separate Blender023 lid as one actor."""
    template = MJCFObject(
        scene=scene,
        name=f"{name}_template",
        mjcf_path=str(patched_dir),
        scale=float(scale),
        density=180,
        friction=(0.90, 0.55, 0.03),
    )
    if not hasattr(template, "actor_builder"):
        raise RuntimeError(
            "Blender023_Lid must parse as a single actor"
        )

    builder = template.actor_builder
    builder.set_scene_idxs(list(range(num_envs)))
    builder.initial_pose = sapien.Pose(
        p=position,
        q=quaternion,
    )

    if dynamic:
        lid = builder.build_dynamic(name=name)
        if hasattr(lid, "set_mass"):
            lid.set_mass(float(mass))
        if hasattr(lid, "set_linear_damping"):
            lid.set_linear_damping(0.18)
        if hasattr(lid, "set_angular_damping"):
            lid.set_angular_damping(0.22)
    else:
        lid = builder.build_static(name=name)

    configure_task_object_physics(lid, "hard_plastic")
    template.actor = lid
    return lid, template


def configure_task_object_physics(
    task_object, material_class: str, restitution: float = 0.0
) -> None:
    """Apply the project material lookup and task-object energy fence.

    This deliberately touches task objects only. Robot, arm, hand, table, and
    global solver parameters remain owned by their respective frozen layers.
    """
    material = object_material(material_class, restitution=restitution)
    items = (
        task_object.get_links()
        if hasattr(task_object, "get_links")
        else (task_object,)
    )
    for item in items:
        for raw in item._objs:
            component = raw
            if not hasattr(component, "get_collision_shapes"):
                component = None
                for component_type in (
                    sapien.physx.PhysxRigidDynamicComponent,
                    sapien.physx.PhysxRigidStaticComponent,
                    sapien.physx.PhysxArticulationLinkComponent,
                ):
                    component = raw.find_component_by_type(component_type)
                    if component is not None:
                        break
            if component is None:
                continue
            for shape in component.get_collision_shapes():
                shape.set_physical_material(material)
            if hasattr(component, "set_max_depenetration_velocity"):
                component.set_max_depenetration_velocity(0.5)
                component.set_max_linear_velocity(3.0)


# -----------------------------------------------------------------------------
# Robot helpers
# -----------------------------------------------------------------------------


def reset_robot_agent(
    agent,
    robot_uids: str,
    batch_size: int,
    device,
    robot_init_qpos_noise: float,
    episode_rng,
    root_pose: sapien.Pose = ROBOT_ROOT_POSE,
) -> None:
    """Reset Panda gripper or one of the three Panda dexterous-hand robots."""
    if hasattr(agent, "get_reset_pos"):
        qpos = (
            agent.get_reset_pos(device=device)
            .to(torch.float32)
            .expand(batch_size, -1)
            .clone()
        )
        # Shared teleop start pose (tasks/config/workspace_rig.py).
        qpos = apply_arm_init_qpos(agent, qpos)
        active_names = [
            joint.name for joint in agent.robot.get_active_joints()
        ]
        arm_indices = [
            active_names.index(name)
            for name in getattr(agent, "arm_joint_names", [])
            if name in active_names
        ]
        if arm_indices and robot_init_qpos_noise > 0:
            index = torch.tensor(
                arm_indices, dtype=torch.long, device=device
            )
            noise = common.to_tensor(
                episode_rng.normal(
                    0.0,
                    robot_init_qpos_noise,
                    (batch_size, len(arm_indices)),
                ),
                device=device,
            )
            qpos[:, index] += noise
        agent.reset(qpos)

    elif robot_uids == GRIPPER_ROBOT_UID:
        qpos = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                -2.0 * np.pi / 3.0,
                0.0,
                2.0 * np.pi / 3.0,
                np.pi / 4.0,
                0.04,
                0.04,
            ],
            dtype=torch.float32,
            device=device,
        ).expand(batch_size, -1).clone()
        if robot_init_qpos_noise > 0:
            qpos[:, :7] += (
                torch.randn(
                    batch_size,
                    7,
                    device=device,
                )
                * robot_init_qpos_noise
            )
        agent.reset(qpos)

    elif robot_uids in HAND_ROBOT_UIDS:
        qpos = torch.tensor(
            agent.keyframes["rest"].qpos,
            dtype=torch.float32,
            device=device,
        ).expand(batch_size, -1).clone()
        # Shared teleop start pose (tasks/config/workspace_rig.py).
        qpos = apply_arm_init_qpos(agent, qpos)

        active_names = [
            joint.name
            for joint in agent.robot.get_active_joints()
        ]
        arm_indices = [
            active_names.index(name)
            for name in getattr(agent, "arm_joint_names", [])
            if name in active_names
        ]

        if arm_indices and robot_init_qpos_noise > 0:
            index = torch.tensor(
                arm_indices,
                dtype=torch.long,
                device=device,
            )
            noise = common.to_tensor(
                episode_rng.normal(
                    0.0,
                    robot_init_qpos_noise,
                    (batch_size, len(arm_indices)),
                ),
                device=device,
            )
            qpos[:, index] += noise
        agent.reset(qpos)

    else:
        raise NotImplementedError(robot_uids)

    agent.robot.set_root_pose(root_pose)


def all_tip_positions(agent, robot_uids: str) -> torch.Tensor:
    """Return shape (num_envs, num_tips, 3)."""
    if robot_uids in HAND_ROBOT_UIDS and hasattr(
        agent,
        "tip_links",
    ):
        return torch.stack(
            [link.pose.p for link in agent.tip_links],
            dim=1,
        )

    if hasattr(agent, "finger1pad_link") and hasattr(
        agent,
        "finger2pad_link",
    ):
        return torch.stack(
            [
                agent.finger1pad_link.pose.p,
                agent.finger2pad_link.pose.p,
            ],
            dim=1,
        )

    return agent.tcp.pose.p[:, None, :]


def arm_dof(agent) -> int:
    return (
        len(getattr(agent, "arm_joint_names", []))
        or min(7, agent.robot.dof)
    )


# -----------------------------------------------------------------------------
# MJCF geometry inspection
# -----------------------------------------------------------------------------


def _parse_vector(
    text: str | None,
    size: int,
    default: tuple[float, ...],
) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if size == 3 and len(values) == 1:
        values = np.repeat(values[0], 3)
    if len(values) != size:
        raise ValueError(
            f"Expected {size} values, got {text!r}"
        )
    return values


def _euler_xyz_matrix(euler: np.ndarray) -> np.ndarray:
    rx, ry, rz = euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def _element_transform(
    element: ET.Element,
    angles_are_degrees: bool,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = _parse_vector(
        element.get("pos"),
        3,
        (0.0, 0.0, 0.0),
    )

    if element.get("quat") is not None:
        rotation = quat_to_matrix(
            _parse_vector(
                element.get("quat"),
                4,
                (1.0, 0.0, 0.0, 0.0),
            )
        )
    elif element.get("euler") is not None:
        euler = _parse_vector(
            element.get("euler"),
            3,
            (0.0, 0.0, 0.0),
        )
        if angles_are_degrees:
            euler = np.deg2rad(euler)
        rotation = _euler_xyz_matrix(euler)
    elif element.get("axisangle") is not None:
        axisangle = _parse_vector(
            element.get("axisangle"),
            4,
            (0.0, 0.0, 1.0, 0.0),
        )
        axis = axisangle[:3]
        angle = float(axisangle[3])
        if angles_are_degrees:
            angle = np.deg2rad(angle)
        norm = np.linalg.norm(axis)
        rotation = (
            np.eye(3, dtype=np.float64)
            if norm < 1e-12
            else trimesh.transformations.rotation_matrix(
                angle,
                axis / norm,
            )[:3, :3]
        )
    else:
        rotation = np.eye(3, dtype=np.float64)

    transform[:3, :3] = rotation
    return transform


def _mesh_vertices(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise RuntimeError(f"No mesh geometry in {path}")
        loaded = trimesh.util.concatenate(meshes)

    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(
            f"Unsupported geometry type {type(loaded)} in {path}"
        )
    return np.asarray(loaded.vertices, dtype=np.float64)


def _primitive_vertices(geom: ET.Element) -> np.ndarray:
    geom_type = geom.get("type", "sphere")
    size = _parse_vector(
        geom.get("size"),
        3,
        (0.01, 0.01, 0.01),
    )

    if geom_type == "box":
        x, y, z = size
    elif geom_type in {"cylinder", "capsule"}:
        x = y = size[0]
        z = size[1] if len(size) > 1 else size[0]
    else:
        x = y = z = size[0]

    return np.array(
        [
            [sx * x, sy * y, sz * z]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )


def mjcf_body_vertices(
    model_xml: Path,
    body_name: str,
    global_scale: float,
) -> np.ndarray:
    """Return direct geom vertices of one MJCF body in model-local coordinates."""
    model_xml = Path(model_xml).resolve()
    root = ET.parse(model_xml).getroot()
    compiler = root.find("compiler")
    mesh_dir = (
        compiler.get("meshdir")
        if compiler is not None
        else None
    )
    angle_mode = (
        compiler.get("angle", "degree")
        if compiler is not None
        else "degree"
    )
    angles_are_degrees = angle_mode.lower() != "radian"

    mesh_assets: dict[str, tuple[Path, np.ndarray]] = {}
    model_dir = model_xml.parent

    for mesh in root.findall(".//asset/mesh"):
        name = mesh.get("name")
        filename = mesh.get("file")
        if not name or not filename:
            continue
        path = _resolve_reference(
            model_dir,
            mesh_dir,
            filename,
        )
        mesh_scale = _parse_vector(
            mesh.get("scale"),
            3,
            (1.0, 1.0, 1.0),
        )
        mesh_assets[name] = (
            path,
            mesh_scale * float(global_scale),
        )

    collected: list[np.ndarray] = []

    def collect_geom(
        geom: ET.Element,
        body_transform: np.ndarray,
    ) -> None:
        geom_transform = (
            body_transform
            @ _element_transform(
                geom,
                angles_are_degrees,
            )
        )
        mesh_name = geom.get("mesh")

        if mesh_name and mesh_name in mesh_assets:
            mesh_path, mesh_scale = mesh_assets[mesh_name]
            vertices = _mesh_vertices(mesh_path)
            vertices = vertices * mesh_scale.reshape(1, 3)
        else:
            vertices = _primitive_vertices(geom)
            vertices *= float(global_scale)

        homogeneous = np.concatenate(
            [
                vertices,
                np.ones(
                    (len(vertices), 1),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )
        collected.append(
            (geom_transform @ homogeneous.T).T[:, :3]
        )

    def walk(
        body: ET.Element,
        parent_transform: np.ndarray,
    ) -> bool:
        body_transform = (
            parent_transform
            @ _element_transform(
                body,
                angles_are_degrees,
            )
        )
        if body.get("name") == body_name:
            for geom in body.findall("geom"):
                collect_geom(geom, body_transform)
            return True

        for child in body.findall("body"):
            if walk(child, body_transform):
                return True
        return False

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No worldbody in {model_xml}")

    found = False
    for body in worldbody.findall("body"):
        if walk(body, np.eye(4, dtype=np.float64)):
            found = True
            break

    if not found or not collected:
        raise RuntimeError(
            f"No direct geometry found for MJCF body {body_name!r}"
        )

    return np.concatenate(collected, axis=0)



ENV_IDS = (
    "JuicePreparationPlaceLid-Sharpa-v1",
    "JuicePreparationPlaceLid-Wuji-v1",
    "JuicePreparationPlaceLid-XHand-v1",
    "JuicePreparationPlaceLid-Gripper-v1",
)
MAX_EPISODE_STEPS = 400

# Real blender lid: 50-150 g. 0.18 kg was above that range.
LID_MASS = 0.12
LID_GRASP_LOCAL_Z = 0.028
GRASP_TIP_DISTANCE = 0.075

PICK_HEIGHT = 0.030
TRANSPORT_XY_TOL = 0.060
TRANSPORT_Z_MARGIN = 0.035

LID_XY_TOL = 0.005
# Seat tolerance widened at the operator's request. The plug and the
# pitcher mouth are plain cylinders with no keying feature, and VR teleop gives
# no haptic feedback, so 5 deg was not something a human could hold reliably.
#
# Tilt and z have to move together: a lid resting on one edge lifts its centre,
# and that rise is what the z check sees, not a gap. Measured (released 10 mm
# above the seat, no lateral offset, settled 150 steps):
#      0 deg -> z +0.04 mm      8 deg -> z +3.30 mm
#      5 deg -> z +0.04 mm     10 deg -> z +4.72 mm
#                              15 deg -> z +6.75 mm
# so 8 mm of z is what 15 deg of tilt costs. Beyond that the tilt check is what
# rejects: at 20 deg the lid slides rather than perches, and it is refused on
# tilt regardless of z.
LID_Z_TOL = 0.008
LID_TILT_TOL = np.deg2rad(15.0)
LID_LINEAR_SPEED_TOL = 0.020
LID_ANGULAR_SPEED_TOL = 0.25
RELEASE_TCP_CLEARANCE = 0.10
RELEASE_TIP_CLEARANCE = 0.045
SUCCESS_SETTLE_STEPS = 15

SUCCESS_REWARD = 12.0
LIVING_PENALTY = 0.01
ARM_QVEL_PENALTY = 2e-4


# =============================================================================
# Shared lid predicate
#
# Stage 3 of the long Juice task and this subtask read the same seat test, so
# the tolerance triple and the velocity-gate decision below live in one place.
# =============================================================================


def lid_seat_quantities(lid_pose, seat_position, *, device):
    """Seat error and the seated/stable pair for a lid pose.

    Returns ``(position, to_seat, xy_error, z_error, tilt_error, seated,
    stable)``. ``to_seat`` points from the lid to the seat.
    """
    position = lid_pose.p
    seat_target = torch.tensor(
        np.asarray(seat_position, dtype=np.float32), device=device
    ).view(1, 3)
    to_seat = seat_target - position
    xy_error = torch.linalg.norm(to_seat[:, :2], dim=1)
    z_error = torch.abs(to_seat[:, 2])
    rotation = lid_pose.to_transformation_matrix()[..., :3, :3]
    tilt_error = torch.acos(torch.clamp(rotation[..., 2, 2], -1.0, 1.0))
    seated = (
        (xy_error <= LID_XY_TOL)
        & (z_error <= LID_Z_TOL)
        & (tilt_error <= LID_TILT_TOL)
    )
    # Motion gate. The reported rigid-body velocity is NOT usable for this
    # lid. A lid resting tilted on the pitcher rim is perfectly still --
    # sampling at substep resolution shows z moving 0.0002 mm over 22
    # substeps -- yet PhysX keeps reporting a constant 0.1998 m/s and
    # 1.28 rad/s for it. Writing zeros into the velocity buffer by hand is
    # undone one substep later, so it is regenerated, not stale. Only a
    # fully bottomed-out flat placement reads zero, which means a velocity
    # gate silently rejects every tilted seat and stalls the stage with the
    # lid visibly on. The apple shows the same error resting in this pitcher
    # (0.194 m/s and 5.13 rad/s reported for a pose frozen to 0.00000), which
    # is why its settle gate now runs on a pose finite difference -- see
    # apple_motion_update in tasks/juice_preparation/juice_preparation_apple_to_blender.py. The same
    # treatment would work here; the dwell below is what stands in for it.
    #
    # The 15-step dwell that consumes this predicate is the stability test
    # instead, and a stricter one: holding the 5 mm xy window for 0.75 s
    # bounds real motion below about 7 mm/s, well inside the 20 mm/s
    # this gate used to ask for. Speeds stay in ``info`` for debugging.
    stable = torch.ones_like(seated)
    return position, to_seat, xy_error, z_error, tilt_error, seated, stable


@register_task(
    "JuicePreparationPlaceLid",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
)
class JuicePreparationPlaceLidEnv(BaseEnv):
    """Pick the loose lid, seat it on Blender023, release, and retreat."""

    SUPPORTED_ROBOTS = [
        GRIPPER_ROBOT_UID,
            "panda_sharpa_hand_right_angled_mobile_base",
        "panda_wuji_hand_right_angled_mobile_base",
            "panda_xhand_hand_right_angled_mobile_base",
]
    agent: Any

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_sharpa_hand_right_angled_mobile_base",
        blender_dir: str = str(DEFAULT_BLENDER_DIR),
        lid_dir: str = str(DEFAULT_LID_DIR),
        global_scale: float = GLOBAL_SCALE,
        lid_mass: float = LID_MASS,
        robot_init_qpos_noise: float = ARM_INIT_QPOS_NOISE_STD,
        lid_x_noise: float = JUICE_LID_RANDOM_XY[0],
        lid_y_noise: float = JUICE_LID_RANDOM_XY[1],
        lid_yaw_noise_deg: float = JUICE_LID_RANDOM_YAW_DEG,
        terminate_on_success: bool = False,
        render_width: int = 960,
        render_height: int = 720,
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if "sim_config" in kwargs and kwargs["sim_config"] is not None:
            raise ValueError(
                "Juice freezes the Plant configuration in "
                "tasks/config/sim_profiles.py; runtime sim_config overrides "
                "are not allowed"
            )
        if robot_uids not in self.SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unsupported robot {robot_uids!r}; expected one of "
                f"{tuple(self.SUPPORTED_ROBOTS)}"
            )

        self.blender_dir = Path(blender_dir)
        self.lid_dir = Path(lid_dir)
        self.global_scale = float(global_scale)
        self.lid_mass = float(lid_mass)
        self.robot_init_qpos_noise = float(
            robot_init_qpos_noise
        )
        self.lid_x_noise = float(lid_x_noise)
        self.lid_y_noise = float(lid_y_noise)
        self.lid_yaw_noise = np.deg2rad(
            float(lid_yaw_noise_deg)
        )
        self.terminate_on_success = bool(
            terminate_on_success
        )
        self.render_width = int(render_width)
        self.render_height = int(render_height)

        self.geometry = compute_blender_geometry(
            self.global_scale
        )
        self._geometry_nominal = copy.deepcopy(self.geometry)
        self.patched_blender_dir = prepare_mjcf(
            self.blender_dir,
            "place_lid_blender023",
        )
        self.patched_lid_dir = prepare_mjcf(
            self.lid_dir,
            "place_lid_blender023_lid",
        )

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self) -> SimConfig:
        return make_sim_config(
            spacing=3,
            found_lost_pairs_capacity=2**25,
            max_rigid_contact_count=2**20,
            max_rigid_patch_count=2**19,
        )

    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(
            eye=[0.54 + WORK_TABLE_X_OFFSET, -0.76, 0.55],
            target=[0.02 + WORK_TABLE_X_OFFSET, -0.30, 0.20],
        )
        external_pose = sapien_utils.look_at(
            eye=[-0.46 + WORK_TABLE_X_OFFSET, 0.20, 0.64],
            target=[0.03 + WORK_TABLE_X_OFFSET, -0.30, 0.18],
        )
        return [
            CameraConfig(
                "base_camera",
                base_pose,
                320,
                240,
                np.deg2rad(58.0),
                0.01,
                100.0,
            ),
            CameraConfig(
                "external_camera",
                external_pose,
                320,
                240,
                np.deg2rad(58.0),
                0.01,
                100.0,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.62 + WORK_TABLE_X_OFFSET, -0.95, 0.62],
            target=[0.02 + WORK_TABLE_X_OFFSET, -0.30, 0.20],
        )
        return CameraConfig(
            "render_camera",
            pose,
            self.render_width,
            self.render_height,
            np.deg2rad(58.0),
            0.01,
            100.0,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, ROBOT_ROOT_POSE)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        set_work_table_pose(self)

        (
            self.blender,
            self.blender_template,
            self.blender_is_articulation,
        ) = build_blender023(
            scene=self.scene,
            patched_dir=self.patched_blender_dir,
            geometry=self.geometry,
            scale=self.global_scale,
            num_envs=self.num_envs,
            name="robocasa_blender023_place_lid",
        )

        (
            self.lid,
            self.lid_template,
        ) = build_blender_lid(
            scene=self.scene,
            patched_dir=self.patched_lid_dir,
            scale=self.global_scale * LID_UPSCALE,
            num_envs=self.num_envs,
            position=self.geometry.lid_table_position,
            quaternion=np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            ),
            name="robocasa_blender023_lid_dynamic",
            dynamic=True,
            mass=self.lid_mass,
        )

        self.picked_once = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.transported_once = torch.zeros_like(
            self.picked_once
        )
        self.seated_once = torch.zeros_like(
            self.picked_once
        )
        self.released_once = torch.zeros_like(
            self.picked_once
        )
        self.success_latched = torch.zeros_like(
            self.picked_once
        )
        self.success_counter = torch.zeros(
            self.num_envs,
            dtype=torch.int32,
            device=self.device,
        )

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)
            # TableSceneBuilder.initialize restores its own stock pose.
            set_work_table_pose(self)
            # Per-episode layout (FIRST episode-RNG draw): the blender moves and the seat / lid table
            # point move with it; the lid gets its own offset below.
            layouts = [
                sample_juice_layout(
                    layout_rng(self, int(i)),
                    lid_half_xy=(self.lid_x_noise, self.lid_y_noise),
                    lid_yaw_half_deg=float(np.rad2deg(self.lid_yaw_noise)),
                )
                for i in env_idx.tolist()
            ]
            bdx, bdy, byaw = layouts[0]["blender"]
            self.geometry = shifted_blender_geometry(self._geometry_nominal, bdx, bdy, byaw)
            reset_robot_agent(
                agent=self.agent,
                robot_uids=self.robot_uids,
                batch_size=batch_size,
                device=self.device,
                robot_init_qpos_noise=(
                    self.robot_init_qpos_noise
                ),
                episode_rng=self._episode_rng,
            )

            blender_p = torch.tensor(
                np.asarray(
                    [self.geometry.blender_root],
                    dtype=np.float32,
                ),
                device=self.device,
            ).expand(batch_size, -1)
            blender_q = torch.tensor(
                np.asarray(
                    [self.geometry.blender_quat],
                    dtype=np.float32,
                ),
                device=self.device,
            ).expand(batch_size, -1)

            if hasattr(self.blender, "set_pose"):
                self.blender.set_pose(
                    Pose.create_from_pq(
                        blender_p,
                        blender_q,
                    )
                )
            if self.blender_is_articulation:
                self.blender.set_qpos(
                    torch.zeros_like(
                        self.blender.get_qpos()
                    )
                )
                self.blender.set_qvel(
                    torch.zeros_like(
                        self.blender.get_qvel()
                    )
                )

            lid_p = torch.tensor(
                np.asarray(
                    [self.geometry.lid_table_position],
                    dtype=np.float32,
                ),
                device=self.device,
            ).expand(batch_size, -1).clone()

            # Lid offset and yaw from the same per-episode draw as the blender (above).
            yaw = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            for k, lay in enumerate(layouts):
                ldx, ldy, lyaw = lay["lid"]
                lid_p[k, 0] += ldx
                lid_p[k, 1] += ldy
                yaw[k] = lyaw
            lid_q = torch.zeros(
                batch_size,
                4,
                dtype=torch.float32,
                device=self.device,
            )
            lid_q[:, 0] = torch.cos(0.5 * yaw)
            lid_q[:, 3] = torch.sin(0.5 * yaw)

            zero_velocity = torch.zeros(
                batch_size,
                3,
                dtype=torch.float32,
                device=self.device,
            )
            self.lid.set_linear_velocity(zero_velocity)
            self.lid.set_angular_velocity(zero_velocity)
            self.lid.set_pose(
                Pose.create_from_pq(lid_p, lid_q)
            )
            self.lid.set_linear_velocity(zero_velocity)
            self.lid.set_angular_velocity(zero_velocity)

            self.picked_once[env_idx] = False
            self.transported_once[env_idx] = False
            self.seated_once[env_idx] = False
            self.released_once[env_idx] = False
            self.success_latched[env_idx] = False
            self.success_counter[env_idx] = 0

    # ------------------------------------------------------------------
    # Task quantities
    # ------------------------------------------------------------------

    def _lid_grasp_point(self) -> torch.Tensor:
        rotation = self.lid.pose.to_transformation_matrix()[
            ..., :3, :3
        ]
        offset = torch.tensor(
            [
                0.0,
                0.0,
                LID_GRASP_LOCAL_Z * self.global_scale,
            ],
            dtype=rotation.dtype,
            device=self.device,
        )
        return self.lid.pose.p + torch.matmul(
            rotation,
            offset.view(1, 3, 1),
        ).squeeze(-1)

    def _lid_grasped(
        self,
        tip_distances: torch.Tensor,
    ) -> torch.Tensor:
        if self.robot_uids == GRIPPER_ROBOT_UID and hasattr(
            self.agent,
            "is_grasping",
        ):
            try:
                result = self.agent.is_grasping(self.lid)
                return torch.as_tensor(
                    result,
                    dtype=torch.bool,
                    device=self.device,
                ).reshape(self.num_envs)
            except Exception:
                pass

        return (
            tip_distances <= GRASP_TIP_DISTANCE
        ).sum(dim=1) >= 2

    def _task_quantities(self) -> dict[str, torch.Tensor]:
        lid_pose = self.lid.pose
        lid_position = lid_pose.p
        linear_velocity = self.lid.linear_velocity
        angular_velocity = self.lid.angular_velocity
        linear_speed = torch.linalg.norm(
            linear_velocity,
            dim=1,
        )
        angular_speed = torch.linalg.norm(
            angular_velocity,
            dim=1,
        )

        (
            _,
            lid_to_seat,
            xy_error,
            z_error,
            tilt_error,
            seated,
            stable,
        ) = lid_seat_quantities(
            lid_pose, self.geometry.seat_position, device=self.device
        )
        seat_target = lid_position + lid_to_seat

        lid_bottom_z = (
            lid_position[:, 2]
            + float(self.geometry.lid_lower[2])
        )
        lift_height = lid_bottom_z - TABLE_TOP_Z

        grasp_point = self._lid_grasp_point()
        tips = all_tip_positions(
            self.agent,
            self.robot_uids,
        )
        tip_distances = torch.linalg.norm(
            tips - grasp_point[:, None, :],
            dim=2,
        )
        min_tip_distance = tip_distances.min(
            dim=1
        ).values
        tcp_distance = torch.linalg.norm(
            self.agent.tcp.pose.p - grasp_point,
            dim=1,
        )
        reach_distance = (
            tcp_distance
            if self.robot_uids == GRIPPER_ROBOT_UID
            else min_tip_distance
        )
        grasped = self._lid_grasped(tip_distances)

        end_effector_clear = (
            (~grasped)
            & (
                tcp_distance
                >= RELEASE_TCP_CLEARANCE
            )
            & (
                min_tip_distance
                >= RELEASE_TIP_CLEARANCE
            )
        )

        preseat_target = seat_target.clone()
        preseat_target[:, 2] += 0.08
        preseat_distance = torch.linalg.norm(
            preseat_target - lid_position,
            dim=1,
        )

        arm_velocity = self.agent.robot.get_qvel()[
            ..., :arm_dof(self.agent)
        ]

        return {
            "lid_pose": lid_pose.raw_pose,
            "lid_position": lid_position,
            "lid_linear_velocity": linear_velocity,
            "lid_angular_velocity": angular_velocity,
            "lid_linear_speed": linear_speed,
            "lid_angular_speed": angular_speed,
            "lid_grasp_point": grasp_point,
            "all_tip_positions": tips,
            "min_tip_distance": min_tip_distance,
            "tcp_to_lid_distance": tcp_distance,
            "reach_distance": reach_distance,
            "lid_grasped": grasped,
            "seat_target_position": seat_target,
            "lid_to_seat_position": lid_to_seat,
            "lid_xy_error": xy_error,
            "lid_z_error": z_error,
            "lid_tilt_error": tilt_error,
            "lid_upright_cos": torch.cos(tilt_error),
            "lid_lift_height": lift_height,
            "lid_seated": seated,
            "lid_stable": stable,
            "end_effector_clear": end_effector_clear,
            "preseat_distance": preseat_distance,
            "arm_velocity": arm_velocity,
        }

    def evaluate(self):
        q = self._task_quantities()

        self.picked_once |= q["lid_lift_height"] >= PICK_HEIGHT
        self.transported_once |= (
            self.picked_once
            & (
                q["lid_xy_error"]
                <= TRANSPORT_XY_TOL
            )
            & (
                q["lid_position"][:, 2]
                >= q["seat_target_position"][:, 2]
                + TRANSPORT_Z_MARGIN
            )
        )
        self.seated_once |= (
            q["lid_seated"] & q["lid_stable"]
        )
        self.released_once |= (
            self.seated_once
            & q["end_effector_clear"]
        )

        # Success is the physical end state itself: the lid is fully aligned,
        # seated at the measured flange/rim height, upright, and settled.  A
        # retreat is useful for demonstrations but is not part of "lid closed".
        final_condition = (
            self.picked_once & q["lid_seated"] & q["lid_stable"]
        )
        self.success_counter = torch.where(
            final_condition,
            self.success_counter + 1,
            torch.zeros_like(self.success_counter),
        )
        self.success_latched |= (
            self.success_counter
            >= SUCCESS_SETTLE_STEPS
        )

        return {
            "success": self.success_latched,
            "task_success": self.success_latched,
            "is_obj_placed": self.success_latched,
            "obj_to_goal_dist": torch.linalg.norm(
                q["lid_to_seat_position"],
                dim=1,
            ),
            "picked_once": self.picked_once,
            "transported_once": self.transported_once,
            "seated_once": self.seated_once,
            "released_once": self.released_once,
            "success_counter": self.success_counter,
            "success_progress": torch.clamp(
                self.success_counter.float()
                / SUCCESS_SETTLE_STEPS,
                0.0,
                1.0,
            ),
            **q,
        }

    def get_state_dict(self):
        state = super().get_state_dict()
        state["juice_lid_task_state"] = torch.stack(
            [
                self.picked_once.float(),
                self.transported_once.float(),
                self.seated_once.float(),
                self.released_once.float(),
                self.success_latched.float(),
                self.success_counter.float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("juice_lid_task_state", None)
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
        if task_state is None:
            return
        task_state = common.to_tensor(
            task_state, device=self.device
        ).reshape(len(restore_idx), -1)
        if task_state.shape[-1] != 6:
            raise ValueError("juice_lid_task_state must have 6 fields")
        self.picked_once[restore_idx] = task_state[:, 0].bool()
        self.transported_once[restore_idx] = task_state[:, 1].bool()
        self.seated_once[restore_idx] = task_state[:, 2].bool()
        self.released_once[restore_idx] = task_state[:, 3].bool()
        self.success_latched[restore_idx] = task_state[:, 4].bool()
        self.success_counter[restore_idx] = task_state[:, 5].to(torch.int32)

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "all_tip_positions": info[
                "all_tip_positions"
            ].reshape(self.num_envs, -1),
            "lid_pose": info["lid_pose"],
            "lid_linear_velocity": info[
                "lid_linear_velocity"
            ],
            "lid_angular_velocity": info[
                "lid_angular_velocity"
            ],
            "lid_grasp_point": info[
                "lid_grasp_point"
            ],
            "seat_target_position": info[
                "seat_target_position"
            ],
            "lid_to_seat_position": info[
                "lid_to_seat_position"
            ],
            "picked_once": info[
                "picked_once"
            ].float().unsqueeze(1),
            "transported_once": info[
                "transported_once"
            ].float().unsqueeze(1),
            "seated_once": info[
                "seated_once"
            ].float().unsqueeze(1),
            "released_once": info[
                "released_once"
            ].float().unsqueeze(1),
            "task_success": info[
                "task_success"
            ].float().unsqueeze(1),
        }

        if self.obs_mode_struct.use_state:
            obs.update(
                reach_distance=info[
                    "reach_distance"
                ].unsqueeze(1),
                lid_xy_error=info[
                    "lid_xy_error"
                ].unsqueeze(1),
                lid_z_error=info[
                    "lid_z_error"
                ].unsqueeze(1),
                lid_tilt_error=info[
                    "lid_tilt_error"
                ].unsqueeze(1),
                lid_lift_height=info[
                    "lid_lift_height"
                ].unsqueeze(1),
                lid_linear_speed=info[
                    "lid_linear_speed"
                ].unsqueeze(1),
                lid_angular_speed=info[
                    "lid_angular_speed"
                ].unsqueeze(1),
                min_tip_distance=info[
                    "min_tip_distance"
                ].unsqueeze(1),
                tcp_to_lid_distance=info[
                    "tcp_to_lid_distance"
                ].unsqueeze(1),
                lid_grasped=info[
                    "lid_grasped"
                ].float().unsqueeze(1),
                lid_seated=info[
                    "lid_seated"
                ].float().unsqueeze(1),
                lid_stable=info[
                    "lid_stable"
                ].float().unsqueeze(1),
                end_effector_clear=info[
                    "end_effector_clear"
                ].float().unsqueeze(1),
                success_progress=info[
                    "success_progress"
                ].unsqueeze(1),
            )
        return obs

    def compute_sparse_reward(
        self,
        obs: Any,
        action: Array,
        info: dict,
    ) -> torch.Tensor:
        return info["task_success"].float()

    def compute_dense_reward(
        self,
        obs: Any,
        action: Array,
        info: dict,
    ) -> torch.Tensor:
        reach = 1.0 - torch.tanh(
            7.0 * info["reach_distance"]
        )
        grasp = info["lid_grasped"].float()
        lift = torch.clamp(
            info["lid_lift_height"] / 0.08,
            0.0,
            1.0,
        )
        transport = 1.0 - torch.tanh(
            6.0 * info["preseat_distance"]
        )
        align_xy = 1.0 - torch.tanh(
            25.0 * info["lid_xy_error"]
        )
        align_z = 1.0 - torch.tanh(
            30.0 * info["lid_z_error"]
        )
        upright = torch.clamp(
            (
                info["lid_upright_cos"]
                + 1.0
            )
            * 0.5,
            0.0,
            1.0,
        )
        place = align_xy * align_z * upright
        stable = (
            1.0
            - torch.tanh(
                12.0 * info["lid_linear_speed"]
            )
        ) * (
            1.0
            - torch.tanh(
                2.0 * info["lid_angular_speed"]
            )
        )
        release = (
            info["lid_seated"]
            & info["lid_stable"]
            & info["end_effector_clear"]
        ).float()

        reward = reach
        reward += 1.5 * grasp
        reward += (
            1.5
            * info["picked_once"].float()
            * lift
        )
        reward += (
            2.0
            * info["picked_once"].float()
            * transport
        )
        reward += (
            3.0
            * info["transported_once"].float()
            * place
        )
        reward += (
            1.5
            * info["lid_seated"].float()
            * stable
        )
        reward += 2.5 * release
        reward -= ARM_QVEL_PENALTY * torch.mean(
            info["arm_velocity"].square(),
            dim=1,
        )
        reward -= LIVING_PENALTY

        reward = torch.where(
            info["task_success"],
            torch.full_like(
                reward,
                SUCCESS_REWARD,
            ),
            reward,
        )
        info["reward_terms"] = {
            "reach": reach,
            "grasp": grasp,
            "lift": lift,
            "transport": transport,
            "place": place,
            "stable": stable,
            "release": release,
        }
        return reward

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: Array,
        info: dict,
    ) -> torch.Tensor:
        return self.compute_dense_reward(
            obs,
            action,
            info,
        ) / SUCCESS_REWARD

    def _should_terminate(
        self,
        info: dict,
    ) -> torch.Tensor:
        if self.terminate_on_success:
            return info["task_success"]
        return torch.zeros_like(info["task_success"])

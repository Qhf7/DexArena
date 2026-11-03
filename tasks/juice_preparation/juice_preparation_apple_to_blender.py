"""
Pick a RoboCasa apple from an open refrigerator and place it into Blender023.

Environment IDs
---------------
Gripper/Sharpa/JuicePreparationAppleToBlender-Wuji-v1

Assets
------
tasks/assets/task_object_meshes/juice/tabletop_mini_fridge_v1/tabletop_mini_fridge.urdf
RoboCasa apple_0 and Blender023 directories

The apple uses:
- RoboCasa textured visual OBJ
- all precomputed convex collision OBJ pieces
- target width 7 cm
- target mass 0.15 kg

The refrigerator and blender:
- refrigerator starts open at exactly 75 degrees
- Blender023 is loaded without its separate lid
- remains a normal revolute articulation
- contains no orange
- contains no scripted robot motion
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from tasks.config.hands import register_task
from tasks.scene_builders.table import TableSceneBuilder
import copy

from tasks.config.layout_randomization import ARM_INIT_QPOS_NOISE_STD, episode_rng, layout_rng, quat_mul_wxyz
from tasks.config.workspace_rig import (
    WORK_TABLE_X_OFFSET,
    platform_root_pose,
    set_work_table_pose,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import (
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.config.object_materials import object_material
from tasks.config.workspace_rig import apply_arm_init_qpos
from tasks.config.sim_profiles import make_sim_config
from tasks.robots.panda_gripper_right_mobile_base import PandaGripperRightMobileBase
from tasks.robots.panda_sharpa_hand_right_mobile_base import PandaSharpaHandRightAngledMobileBase
from tasks.robots.panda_wuji_hand_right_mobile_base import PandaWujiHandRightMobileBase
from tasks.juice_preparation.juice_preparation_place_lid import (
    sample_juice_layout,
    shifted_blender_geometry,
    DEFAULT_BLENDER_DIR,
    JUICE_ASSET_DIR,
    GLOBAL_SCALE,
    build_blender023,
    compute_blender_geometry,
    configure_task_object_physics,
    prepare_mjcf,
)
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag

# =============================================================================
# Paths and geometry
# =============================================================================

DEFAULT_APPLE_DIR = JUICE_ASSET_DIR / "robocasa_apple_0"
TARGET_APPLE_WIDTH = 0.070
TARGET_APPLE_MASS = 0.150
APPLE_RESTITUTION = 0.01

FRIDGE_ASSET_DIR = JUICE_ASSET_DIR / "tabletop_mini_fridge_v1"
FRIDGE_URDF = (
    FRIDGE_ASSET_DIR / "tabletop_mini_fridge.urdf"
)

DOOR_JOINT_NAME = "door_hinge_joint"
DOOR_LINK_NAME = "fridge_door"
HANDLE_LINK_NAME = "door_handle"

# Custom refrigerator dimensions, matching tabletop_mini_fridge_v1.
FRIDGE_WIDTH = 0.36
FRIDGE_DEPTH = 0.34
FRIDGE_HEIGHT = 0.52
FRIDGE_WALL = 0.025
FRIDGE_FLOOR_TOP_Z = FRIDGE_WALL

#
# Was -1.145. The operator reported not being able to reach the fridge at all,
# and the measurement agrees: at -1.145 the CLOSEST thing in the whole juice
# scene was the door handle at 0.973 m from the arm base, while every family
# that works sits at 0.54-0.78 m (CabinetPour 0.637, package 0.588, meal 0.623,
# office 0.540) and CabinetPour's farthest GRASPED object is the bottle at
# 0.901 m. The juice blender was at 1.16-1.27 m, i.e. beyond anything that has
# ever been reachable.
#
# Moving the platform to CabinetPour's -0.870 (measured, robot re-placed and
# stepped): door handle 0.973 -> 0.705 m, blender button 1.159 -> 0.884 m,
# farthest body 1.274 -> 1.008 m, and ZERO robot-vs-scene contacts introduced.
# The scene layout (table, fridge, blender) is untouched -- only the platform
# moved, so every task threshold, which is expressed relative to the objects,
# is unaffected.
#
# the whole juice scene now rides the SHARED absolute rig
# (tasks/config/workspace_rig.py) instead of the stock table centre: table
# +0.180, trolley -0.870. That is a rigid +0.30 m translation of table +
# trolley + fridge + blender + cameras, so every distance quoted above is
# unchanged; what changes is that the operator's body now stands in the same
# place against the table as in CabinetPour/Screwing and every other family.
ROBOT_ROOT_P = np.asarray(
    platform_root_pose().p,
    dtype=np.float64,
)
# Scene moved 0.42 m towards the operator (x 0.08 -> -0.34) so the task lands in
# the arm's working band. Measured after the move, with the rig on -1.170:
# door handle 0.997 -> 0.686 m, blender button 1.184 -> 0.865 m, farthest body
# 1.299 -> 0.989 m, and zero robot-vs-scene contacts. Reference CabinetPour
# spans 0.637 (nearest) to 0.901 (bottle).
# moved again after the operator reported the fridge was "too hard
# to open" and asked for it to sit roughly where CabinetPour's cabinet does.
# The complaint was not that the door was far -- it was too NEAR: the handle sat
# 0.590 m from the left arm root, below CabinetPour's own 0.637 m nearest and
# 0.117 m inside its door (0.707 m), so the arm had to fold in on itself to
# reach the handle. +0.11 in x and -0.06 in y puts the handle at 0.705 m, i.e.
# within 2 mm of CabinetPour's door, and as a side effect brings the apple back
# inside the 0.901 m working ceiling (0.917 -> 0.882 from the right arm root).
# Fridge-vs-blender clearance stays at 144 mm.
FRIDGE_ROOT_P = np.asarray(
    [-0.34 + WORK_TABLE_X_OFFSET + 0.11, 0.24 - 0.06, 0.0],
    dtype=np.float64,
)

# Apple location in the refrigerator's local frame.
# x: front (-) -> back (+)
# y: left (-) -> right (+)
#
# Moved to the DOOR at the operator's request (was +0.020, i.e. past the middle
# and towards the back wall). The interior spans x +-0.145 after the 0.025 wall,
# and the apple is 0.070 across, so its centre can go to -0.110 at the very
# front; -0.100 leaves 10 mm of clearance so the door can still close on an
# untouched apple.
APPLE_LOCAL_X = -0.100
APPLE_LOCAL_Y = -0.035
APPLE_FLOOR_CLEARANCE = 0.0015
APPLE_INITIAL_YAW_DEG = 18.0

CONTROL_FREQUENCY = 20
SUCCESS_SETTLE_STEPS = 12
LINEAR_STATIC_THRESHOLD = 0.018
ANGULAR_STATIC_THRESHOLD = 0.20
BLENDER_Y_OFFSET = 0.0
BLENDER_INSIDE_RADIUS = 0.065
BLENDER_BOTTOM_CLEARANCE = 0.035
APPLE_LIFT_HEIGHT = 0.06

ROBOT_UID_GRIPPER = "panda_gripper_right_mobile_base"
ROBOT_UID_SHARPA = "panda_sharpa_hand_right_angled_mobile_base"
ROBOT_UID_WUJI = "panda_wuji_hand_right_angled_mobile_base"
ROBOT_UID_XHAND = "panda_xhand_hand_right_angled_mobile_base"
ROBOT_UIDS = (ROBOT_UID_GRIPPER, ROBOT_UID_SHARPA, ROBOT_UID_WUJI, ROBOT_UID_XHAND)

# Door hinge, CALIBRATED against a measured breakaway torque.
#
# SAPIEN's joint `friction` is a COEFFICIENT, not N*m. Measured on this asset
# with the torque re-applied every physics substep and the door clear of the
# tabletop ("turns >5 deg in 20 control steps" = breakaway):
#     coefficient 0.00 -> 0.2 N*m   (0.8 N at the handle)
#     coefficient 0.01 -> 0.5 N*m   (2.0 N)
#     coefficient 0.02 -> 0.8 N*m   (3.2 N)   <-- real hinge is 0.3-1.5 N*m
#     coefficient 0.05 -> 2.0 N*m   (8.0 N)
#     coefficient 0.20 -> >8 N*m    (>32 N)   <-- the previous value
#
# NOT modelled: the magnetic gasket break-away, which on a real unit is the
# dominant term (8-25 N*m at the handle). Opening this door is therefore
# lighter than the real appliance by design, not by accident.
FRIDGE_DOOR_HINGE_FRICTION = 0.02   # coefficient; measured breakaway 0.8 N*m
FRIDGE_DOOR_HANDLE_MASS = 0.15      # kg, real handle 30-200 g


# =============================================================================
# Shared apple predicate
#
# Stage 1 of the long Juice task and this subtask read the same three volume
# tests, so "apple out of the fridge and settled in the pitcher" cannot mean
# two different things in the two envs.
# =============================================================================


def apple_inside_fridge(apple_position, apple_collision_min_z, *, device, fridge_root_p=None):
    """Conservative interior test, from the apple's centre and its own bottom.

    `fridge_root_p` is the fridge's LIVE root position (it is randomized per
    episode); the nominal FRIDGE_ROOT_P is only the fallback."""
    root = (torch.as_tensor(np.asarray(FRIDGE_ROOT_P, dtype=np.float32), device=device).view(1, 3)
            if fridge_root_p is None else torch.as_tensor(fridge_root_p, dtype=torch.float32, device=device).view(-1, 3))
    local = apple_position - root
    apple_bottom_world_z = apple_position[:, 2] + apple_collision_min_z
    fridge_floor_world_z = root[:, 2] + FRIDGE_FLOOR_TOP_Z
    return (
        (local[:, 0] > -0.125)
        & (local[:, 0] < 0.115)
        & (local[:, 1] > -0.125)
        & (local[:, 1] < 0.125)
        & (apple_bottom_world_z > fridge_floor_world_z - 0.006)
        & (apple_position[:, 2] < root[:, 2] + FRIDGE_HEIGHT - 0.055)
    )


def apple_inside_blender(apple_position, blender_geometry, *, device):
    """Cylinder bounded below by the pitcher floor and above by its rim."""
    opening = torch.tensor(
        blender_geometry.seat_position, dtype=torch.float32, device=device
    ).view(1, 3)
    xy = torch.linalg.norm(apple_position[:, :2] - opening[:, :2], dim=1)
    bottom = float(
        blender_geometry.blender_root[2]
        + blender_geometry.blender_lower[2]
        + BLENDER_BOTTOM_CLEARANCE
    )
    return (
        (xy <= BLENDER_INSIDE_RADIUS)
        & (apple_position[:, 2] >= bottom)
        & (apple_position[:, 2] <= float(opening[0, 2]) + 0.015)
    ), opening, xy


APPLE_MOTION_STATE_FIELDS = 10


def new_apple_motion_state(num_envs: int, device) -> dict:
    """Zeroed buffers for :func:`apple_motion_update`."""
    return {
        "prev_p": torch.zeros((num_envs, 3), dtype=torch.float32, device=device),
        "prev_q": torch.zeros((num_envs, 4), dtype=torch.float32, device=device),
        "speed": torch.zeros((num_envs,), dtype=torch.float32, device=device),
        "ang_speed": torch.zeros((num_envs,), dtype=torch.float32, device=device),
        "last_step": torch.zeros((num_envs,), dtype=torch.int32, device=device),
    }


def apple_motion_update(pose, motion: dict, elapsed_steps, *, control_timestep):
    """Refresh the apple's real speeds from a pose finite difference.

    PhysX's reported rigid-body velocity is NOT usable for an apple resting in
    the pitcher. Measured on this env with the apple parked on the pitcher
    floor and its pose frozen -- 0.00004 m/s and 0.0 rad/s by finite difference
    over 60 control steps -- the reported reading stays at 0.015-0.05 m/s and
    **4.4-5.0 rad/s**. The angular number alone is 22x the 0.20 rad/s gate, so
    the settle test could never pass while the apple sat in the pitcher: the
    subtask never reported success and stage 1 of the long task never advanced.

    This is the contact-solver error already documented for the charger in
    ``tasks/drawer_plug/drawer_plug_long.py`` and for the lid in ``tasks/juice_place_blender_lid``;
    the concave COACD pitcher shell triggers it the same way a closed hand does.
    A finite difference of the pose measures the same physical quantity without
    the contact term, so the 18 mm/s and 0.20 rad/s thresholds keep their
    physical meaning instead of being relaxed into meaninglessness.

    Updated once per control step -- ``evaluate`` may be called more than once
    for the same step, and differencing a pose against itself would read as a
    perfectly settled object. The buffers are episode state and travel in
    ``get_state_dict``.
    """
    p = pose.p
    q = pose.q
    dt = control_timestep
    # "Buffers not seeded yet" is marked by the ZERO quaternion that
    # `new_apple_motion_state` writes -- a zero quat is not a rotation, so no
    # restore can ever produce one. It used to be marked by `elapsed_steps == 0`
    # instead, but that is ALSO true on the demo-state restore path:
    # `reset(options={"reset_to_env_states": ...})` starts a fresh episode clock
    # at zero, so the first evaluate threw away the pose history and the speeds
    # the restore had just brought back, and the apple read as perfectly settled.
    unseeded = torch.linalg.norm(motion["prev_q"], dim=-1) < 1e-6
    is_new_step = (elapsed_steps != motion["last_step"]) & ~unseeded
    keep = ~(is_new_step | unseeded)

    lin = torch.linalg.norm(p - motion["prev_p"], dim=-1) / dt
    # Geodesic angle between the two orientations; abs() makes the double
    # cover q and -q read as the same rotation.
    cos_half = (q * motion["prev_q"]).sum(dim=-1).abs().clamp(max=1.0)
    ang = 2.0 * torch.acos(cos_half) / dt

    update = (~keep)[:, None]
    return {
        "prev_p": torch.where(update, p, motion["prev_p"]),
        "prev_q": torch.where(update, q, motion["prev_q"]),
        "speed": torch.where(
            unseeded,
            torch.zeros_like(lin),
            torch.where(keep, motion["speed"], lin),
        ),
        "ang_speed": torch.where(
            unseeded,
            torch.zeros_like(ang),
            torch.where(keep, motion["ang_speed"], ang),
        ),
        "last_step": torch.where(
            ~keep, elapsed_steps.to(motion["last_step"].dtype), motion["last_step"]
        ),
    }


def apple_settled(motion: dict) -> torch.Tensor:
    """Settle gate, on the measured speeds from :func:`apple_motion_update`."""
    return (motion["speed"] <= LINEAR_STATIC_THRESHOLD) & (
        motion["ang_speed"] <= ANGULAR_STATIC_THRESHOLD
    )


def apple_route_latches(
    apple_position,
    in_fridge,
    left_fridge_latched,
    lifted_latched,
    *,
    active=None,
):
    """Latch that the apple actually LEFT the fridge and was lifted clear.

    Both are latches, not instantaneous tests: an apple dropped straight into
    the pitcher from a reach through the door would otherwise pass on geometry
    alone. ``active`` is the long task's stage mask, so the route is only
    recorded while stage 1 is the live stage.
    """
    lifted_now = apple_position[:, 2] >= (
        FRIDGE_ROOT_P[2] + FRIDGE_FLOOR_TOP_Z + APPLE_LIFT_HEIGHT
    )
    left_now = ~in_fridge
    if active is not None:
        left_now = left_now & active
        lifted_now = lifted_now & active
    return left_fridge_latched | left_now, lifted_latched | lifted_now


# =============================================================================
# Self-contained RoboCasa apple loader
# =============================================================================

def _numeric_collision_index(path: Path) -> int:
    match = re.search(r"_collision_(\d+)\.obj$", path.name)
    if match is None:
        return 10**9
    return int(match.group(1))


def _load_single_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise RuntimeError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)

    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(
            f"Unsupported geometry type {type(loaded)} for {path}"
        )

    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise RuntimeError(f"Empty mesh: {path}")

    return loaded


def _parse_scale_text(text: str | None) -> np.ndarray:
    if not text:
        return np.ones(3, dtype=np.float64)

    values = np.asarray(
        [float(value) for value in text.split()],
        dtype=np.float64,
    )
    if len(values) == 1:
        return np.repeat(values[0], 3)
    if len(values) != 3:
        raise ValueError(f"Invalid mesh scale: {text!r}")
    return values


def _mesh_scales_from_model_xml(
    model_xml: Path,
) -> dict[str, np.ndarray]:
    """
    Read mesh scales from RoboCasa's MJCF when available.

    Keys include both the full file string and basename, because different
    RoboCasa generations use different meshdir conventions.
    """
    if not model_xml.exists():
        return {}

    root = ET.parse(model_xml).getroot()
    scales: dict[str, np.ndarray] = {}

    for mesh in root.findall(".//asset/mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        scale = _parse_scale_text(mesh.get("scale"))
        scales[filename.replace("\\", "/")] = scale
        scales[Path(filename).name] = scale

    return scales


def _scale_for_path(
    path: Path,
    apple_dir: Path,
    scale_map: dict[str, np.ndarray],
) -> np.ndarray:
    try:
        relative = str(path.relative_to(apple_dir)).replace("\\", "/")
    except ValueError:
        relative = str(path).replace("\\", "/")

    # Values in scale_map are NumPy arrays. Do not use Python ``or`` here:
    # evaluating the truth value of a multi-element array raises
    # "ValueError: The truth value of an array is ambiguous".
    if relative in scale_map:
        return np.asarray(
            scale_map[relative],
            dtype=np.float64,
        ).copy()

    if path.name in scale_map:
        return np.asarray(
            scale_map[path.name],
            dtype=np.float64,
        ).copy()

    return np.ones(3, dtype=np.float64)


def _patch_mtl_texture_paths(
    source_visual_dir: Path,
) -> Path:
    """
    Copy the visual folder under /tmp and patch stale MTL texture paths.

    Loading the visual OBJ directly avoids the earlier MJCF absolute-path issue,
    but this also makes the OBJ/MTL path robust if a particular asset contains
    an author-machine texture path.
    """
    cache_dir = (
        Path(tempfile.gettempdir())
        / "robodex_robocasa_apple_visual"
        / source_visual_dir.parent.name
    )

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    shutil.copytree(source_visual_dir, cache_dir)

    image_files = sorted(
        [
            path
            for path in cache_dir.iterdir()
            if path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
            }
        ],
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    fallback_image = image_files[0] if image_files else None

    for mtl_path in cache_dir.glob("*.mtl"):
        output_lines = []

        for raw_line in mtl_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            stripped = raw_line.strip()
            parts = stripped.split(maxsplit=1)

            if (
                len(parts) == 2
                and parts[0].lower()
                in {
                    "map_kd",
                    "map_ka",
                    "map_ks",
                    "map_bump",
                    "bump",
                    "norm",
                }
            ):
                texture_text = parts[1].strip()
                texture_path = Path(texture_text)

                if texture_path.is_absolute():
                    candidate = texture_path
                else:
                    candidate = mtl_path.parent / texture_path

                if not candidate.exists() and fallback_image is not None:
                    print(
                        "[visual texture patch] "
                        f"{mtl_path.name}: {texture_text} "
                        f"-> {fallback_image.name}"
                    )
                    raw_line = (
                        f"{parts[0]} {fallback_image.name}"
                    )

            output_lines.append(raw_line)

        mtl_path.write_text(
            "\n".join(output_lines) + "\n",
            encoding="utf-8",
        )

    visual_candidates = sorted(
        cache_dir.glob("model_normalized*.obj")
    )
    if not visual_candidates:
        visual_candidates = sorted(cache_dir.glob("*.obj"))
    if not visual_candidates:
        raise FileNotFoundError(
            f"No visual OBJ found under {source_visual_dir}"
        )

    return visual_candidates[0]


@dataclass
class AppleAssetInfo:
    apple_dir: Path
    visual_path: Path
    collision_paths: list[Path]

    source_visual_scale: np.ndarray
    source_collision_scales: list[np.ndarray]

    uniform_scale: float
    final_visual_scale: np.ndarray
    final_collision_scales: list[np.ndarray]

    visual_bounds: np.ndarray
    collision_bounds: np.ndarray

    combined_collision_volume: float
    density: float
    convex_count: int
    watertight_count: int


def inspect_robocasa_apple_asset(
    apple_dir: Path,
    target_width: float,
    target_mass: float,
) -> AppleAssetInfo:
    apple_dir = apple_dir.resolve()
    visual_dir = apple_dir / "visual"
    collision_dir = apple_dir / "collision"
    model_xml = apple_dir / "model.xml"

    if not visual_dir.exists():
        raise FileNotFoundError(visual_dir)
    if not collision_dir.exists():
        raise FileNotFoundError(collision_dir)

    source_visual_candidates = sorted(
        visual_dir.glob("model_normalized*.obj")
    )
    if not source_visual_candidates:
        source_visual_candidates = sorted(visual_dir.glob("*.obj"))
    if not source_visual_candidates:
        raise FileNotFoundError(
            f"No visual OBJ under {visual_dir}"
        )

    source_visual_path = source_visual_candidates[0]
    patched_visual_path = _patch_mtl_texture_paths(visual_dir)

    collision_paths = sorted(
        collision_dir.glob(
            "model_normalized_collision_*.obj"
        ),
        key=_numeric_collision_index,
    )
    if not collision_paths:
        collision_paths = sorted(
            collision_dir.glob("*.obj")
        )
    if not collision_paths:
        raise FileNotFoundError(
            f"No collision OBJ under {collision_dir}"
        )

    scale_map = _mesh_scales_from_model_xml(model_xml)
    source_visual_scale = _scale_for_path(
        source_visual_path,
        apple_dir,
        scale_map,
    )
    source_collision_scales = [
        _scale_for_path(path, apple_dir, scale_map)
        for path in collision_paths
    ]

    visual_mesh = _load_single_mesh(source_visual_path).copy()
    visual_mesh.apply_scale(source_visual_scale)

    raw_width = float(max(
        visual_mesh.extents[0],
        visual_mesh.extents[1],
    ))
    if raw_width <= 1e-10:
        raise RuntimeError("Apple visual mesh has zero width")

    uniform_scale = float(target_width / raw_width)
    final_visual_scale = source_visual_scale * uniform_scale
    visual_mesh.apply_scale(uniform_scale)
    visual_bounds = visual_mesh.bounds.copy()

    collision_meshes = []
    convex_count = 0
    watertight_count = 0
    combined_collision_volume = 0.0
    final_collision_scales = []

    print("\n[collision pieces]")
    for index, (
        collision_path,
        source_scale,
    ) in enumerate(
        zip(collision_paths, source_collision_scales)
    ):
        mesh = _load_single_mesh(collision_path).copy()
        final_scale = source_scale * uniform_scale
        final_collision_scales.append(final_scale)

        mesh.apply_scale(final_scale)

        is_convex = bool(mesh.is_convex)
        is_watertight = bool(mesh.is_watertight)

        convex_count += int(is_convex)
        watertight_count += int(is_watertight)

        if is_watertight:
            volume = abs(float(mesh.volume))
        else:
            volume = abs(float(mesh.convex_hull.volume))

        combined_collision_volume += volume
        collision_meshes.append(mesh)

        print(
            f"  {index:02d} {collision_path.name:42s} "
            f"V={len(mesh.vertices):4d} "
            f"F={len(mesh.faces):4d} "
            f"convex={int(is_convex)} "
            f"watertight={int(is_watertight)} "
            f"volume={volume:.9f}"
        )

    combined_collision = trimesh.util.concatenate(
        collision_meshes
    )
    collision_bounds = combined_collision.bounds.copy()

    if combined_collision_volume <= 1e-12:
        raise RuntimeError(
            "Combined collision-piece volume is zero"
        )

    density = float(
        target_mass / combined_collision_volume
    )

    print("\n[apple asset summary]")
    print("apple_dir:", apple_dir)
    print("visual:", source_visual_path)
    print("patched_visual:", patched_visual_path)
    print("collision pieces:", len(collision_paths))
    print(
        "convex pieces:",
        f"{convex_count}/{len(collision_paths)}",
    )
    print(
        "watertight pieces:",
        f"{watertight_count}/{len(collision_paths)}",
    )
    print(
        "target width:",
        f"{target_width:.5f} m",
    )
    print(
        "uniform scale:",
        f"{uniform_scale:.8f}",
    )
    print(
        "visual bounds:",
        np.round(visual_bounds, 6).tolist(),
    )
    print(
        "collision bounds:",
        np.round(collision_bounds, 6).tolist(),
    )
    print(
        "visual extents:",
        np.round(
            visual_bounds[1] - visual_bounds[0],
            6,
        ).tolist(),
    )
    print(
        "collision extents:",
        np.round(
            collision_bounds[1] - collision_bounds[0],
            6,
        ).tolist(),
    )
    print(
        "collision/visual extent ratio:",
        np.round(
            (collision_bounds[1] - collision_bounds[0])
            / np.maximum(
                visual_bounds[1] - visual_bounds[0],
                1e-10,
            ),
            5,
        ).tolist(),
    )
    print(
        "combined collision volume:",
        f"{combined_collision_volume:.9f} m^3",
    )
    print(
        "target mass:",
        f"{target_mass:.4f} kg",
    )
    print(
        "calculated density:",
        f"{density:.2f} kg/m^3",
    )

    return AppleAssetInfo(
        apple_dir=apple_dir,
        visual_path=patched_visual_path,
        collision_paths=collision_paths,
        source_visual_scale=source_visual_scale,
        source_collision_scales=source_collision_scales,
        uniform_scale=uniform_scale,
        final_visual_scale=final_visual_scale,
        final_collision_scales=final_collision_scales,
        visual_bounds=visual_bounds,
        collision_bounds=collision_bounds,
        combined_collision_volume=combined_collision_volume,
        density=density,
        convex_count=convex_count,
        watertight_count=watertight_count,
    )


# =============================================================================
# Reusable SAPIEN builder
# =============================================================================


def build_robocasa_apple_actor(
    scene,
    asset_info: AppleAssetInfo,
    *,
    initial_pose: sapien.Pose,
    name: str = "robocasa_apple",
):
    """
    Build one dynamic apple actor from RoboCasa's original visual and all
    precomputed convex collision pieces.
    """
    builder = scene.create_actor_builder()

    physical_material = object_material(
        "produce", restitution=APPLE_RESTITUTION
    )

    builder.add_visual_from_file(
        filename=str(asset_info.visual_path),
        scale=asset_info.final_visual_scale.tolist(),
    )

    for collision_path, collision_scale in zip(
        asset_info.collision_paths,
        asset_info.final_collision_scales,
    ):
        builder.add_convex_collision_from_file(
            filename=str(collision_path),
            scale=collision_scale.tolist(),
            material=physical_material,
            density=asset_info.density,
        )

    builder.initial_pose = initial_pose
    actor = builder.build(name=name)
    configure_task_object_physics(
        actor, "produce", restitution=APPLE_RESTITUTION
    )

    return actor


# =============================================================================
# Reusable fridge loader
# =============================================================================


def load_custom_tabletop_fridge(
    scene,
    *,
    pose: sapien.Pose,
    scene_idx: int = 0,
    name: str = "custom_tabletop_mini_fridge",
):
    if not FRIDGE_URDF.exists():
        raise FileNotFoundError(
            "Custom refrigerator URDF was not found:\n"
            f"  {FRIDGE_URDF}\n"
            "Expected the asset directory at:\n"
            f"  {FRIDGE_ASSET_DIR}"
        )

    loader = scene.create_urdf_loader()
    loader.name = name
    loader.fix_root_link = True
    loader.scale = 1.0
    loader.load_multiple_collisions_from_file = True

    parsed = loader.parse(str(FRIDGE_URDF))
    builders = parsed["articulation_builders"]
    if len(builders) != 1:
        raise RuntimeError(
            "Expected exactly one refrigerator articulation, "
            f"got {len(builders)}"
        )

    builder = builders[0]
    builder.disable_self_collisions = True
    builder.set_scene_idxs([scene_idx])
    builder.initial_pose = pose

    fridge = builder.build(name=name)
    configure_task_object_physics(fridge, "hard_plastic")

    door_joint = fridge.active_joints_map[
        DOOR_JOINT_NAME
    ]
    door_joint.set_drive_properties(
        stiffness=0.0,
        damping=0.85,
        force_limit=85.0,
        mode="force",
    )
    if hasattr(door_joint, "set_friction"):
        door_joint.set_friction(FRIDGE_DOOR_HINGE_FRICTION)

    return fridge


# =============================================================================
# Environment
# =============================================================================


def blender_shift_from_pose(nominal, position, quaternion) -> tuple[float, float, float]:
    """The ``(dx, dy, yaw)`` that ``shifted_blender_geometry`` was called with.

    Inverts the composition that produced the pose: the shifted quaternion is
    ``yaw_quat_wxyz(yaw) * nominal_quat``, so the leftover rotation is a pure
    yaw and ``yaw = 2 * atan2(qz, qw)``.
    """
    p = np.asarray(position, dtype=np.float64).reshape(3)
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    nominal_root = np.asarray(nominal.blender_root, dtype=np.float64)
    nominal_quat = np.asarray(nominal.blender_quat, dtype=np.float64)
    inverse_nominal = nominal_quat * np.array([1.0, -1.0, -1.0, -1.0])
    inverse_nominal = inverse_nominal / max(float(np.dot(nominal_quat, nominal_quat)), 1e-12)
    yaw_quat = quat_mul_wxyz(q, inverse_nominal)
    yaw = 2.0 * float(np.arctan2(yaw_quat[3], yaw_quat[0]))
    return float(p[0] - nominal_root[0]), float(p[1] - nominal_root[1]), yaw


@register_task(
    "JuicePreparationAppleToBlender",
    side="right",
    max_episode_steps=700,
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
)
class JuicePreparationAppleToBlenderEnv(BaseEnv):
    SUPPORTED_ROBOTS = list(ROBOT_UIDS)
    agent: Any

    def __init__(
        self,
        *args,
        robot_uids: str = ROBOT_UID_SHARPA,
        apple_dir: str = str(DEFAULT_APPLE_DIR),
        target_apple_width: float = TARGET_APPLE_WIDTH,
        target_apple_mass: float = TARGET_APPLE_MASS,
        initial_door_angle_deg: float = 75.0,
        blender_dir: str = str(DEFAULT_BLENDER_DIR),
        robot_init_qpos_noise: float = ARM_INIT_QPOS_NOISE_STD,
        num_envs: int = 1,
        **kwargs,
    ):
        if "sim_config" in kwargs and kwargs["sim_config"] is not None:
            raise ValueError(
                "Juice freezes the Plant configuration in "
                "tasks/config/sim_profiles.py; runtime sim_config overrides "
                "are not allowed"
            )
        if num_envs != 1:
            raise ValueError(
                "This visual validation environment currently "
                "supports num_envs=1 only."
            )

        # Every env in the family builds its blender-side geometry at this
        # scale. It lives here, on the shared base, because the long task used
        # to reach for it with `hasattr(self, "global_scale") else 1.0` and
        # silently got 1.0: its loose blender lid was built 1.5x the size the
        # subtask uses (reported by the operator).
        self.global_scale = float(GLOBAL_SCALE)
        self.apple_dir = Path(apple_dir)
        self.target_apple_width = float(
            target_apple_width
        )
        self.target_apple_mass = float(
            target_apple_mass
        )
        self.initial_door_angle_deg = float(
            initial_door_angle_deg
        )
        self.robot_init_qpos_noise = float(
            robot_init_qpos_noise
        )
        if robot_uids not in self.SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unsupported robot {robot_uids!r}; expected one of "
                f"{tuple(self.SUPPORTED_ROBOTS)}"
            )
        self.blender_dir = Path(blender_dir)
        self.blender_geometry = compute_blender_geometry(scale=GLOBAL_SCALE)
        self.blender_geometry.blender_root[1] += BLENDER_Y_OFFSET
        self.blender_geometry.seat_position[1] += BLENDER_Y_OFFSET
        self.blender_geometry.lid_table_position[1] += BLENDER_Y_OFFSET
        self._blender_geometry_nominal = copy.deepcopy(self.blender_geometry)
        self._juice_layout = None
        self.patched_blender_dir = prepare_mjcf(
            self.blender_dir, "take_apple_to_blender_open_pitcher"
        )

        self.apple_asset_info: AppleAssetInfo = (
            inspect_robocasa_apple_asset(
                self.apple_dir,
                target_width=self.target_apple_width,
                target_mass=self.target_apple_mass,
            )
        )

        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return make_sim_config(
            spacing=3,
            max_rigid_contact_count=2**19,
            max_rigid_patch_count=2**18,
        )

    @property
    def _default_sensor_configs(self):
        # Nearly frontal view.  The camera is moved much farther along the
        # refrigerator's front normal (-X) and only slightly toward -Y.
        # The small lateral offset keeps the Panda away from the centre line,
        # while still looking directly into the cabinet instead of at its side.
        pose = sapien_utils.look_at(
            eye=[-1.10 + WORK_TABLE_X_OFFSET, -0.55, 0.62],
            target=[0.04 + WORK_TABLE_X_OFFSET, 0.0, 0.22],
        )
        return [
            CameraConfig(
                "base_camera",
                pose,
                512,
                384,
                np.deg2rad(50.0),
                0.01,
                100.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # Frontal recorded-video view, backed away enough to include the
        # complete refrigerator and its open door.  The viewing ray passes
        # beside the Panda base rather than through the robot.
        pose = sapien_utils.look_at(
            eye=[-1.22 + WORK_TABLE_X_OFFSET, -0.62, 0.68],
            target=[0.04 + WORK_TABLE_X_OFFSET, 0.0, 0.22],
        )
        return CameraConfig(
            "render_camera",
            pose,
            1600,
            900,
            np.deg2rad(52.0),
            0.01,
            100.0,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=ROBOT_ROOT_P),
        )

    def _apple_world_pose(self, fridge_root_p=None) -> sapien.Pose:
        root = np.asarray(FRIDGE_ROOT_P if fridge_root_p is None else fridge_root_p, dtype=np.float64)
        collision_min_z = float(
            self.apple_asset_info.collision_bounds[
                0, 2
            ]
        )
        apple_z = (
            root[2]
            + FRIDGE_FLOOR_TOP_Z
            + APPLE_FLOOR_CLEARANCE
            - collision_min_z
        )

        yaw = np.deg2rad(APPLE_INITIAL_YAW_DEG)
        quaternion = [
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]

        return sapien.Pose(
            p=[
                root[0] + APPLE_LOCAL_X,
                root[1] + APPLE_LOCAL_Y,
                apple_z,
            ],
            q=quaternion,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=(
                self.robot_init_qpos_noise
            ),
        )
        self.table_scene.build()
        set_work_table_pose(self)

        self.fridge = load_custom_tabletop_fridge(
            self.scene,
            pose=sapien.Pose(p=FRIDGE_ROOT_P),
        )
        self.door_joint = self.fridge.active_joints_map[
            DOOR_JOINT_NAME
        ]
        self.door_lower = float(
            self.fridge.get_qlimits()[0, 0, 0].item()
        )
        # The long task clips its stage thresholds against this the same way
        # the two fridge subtasks do (see door_success_qpos).
        self.door_upper = float(
            self.fridge.get_qlimits()[0, 0, 1].item()
        )
        self.door_link = self.fridge.links_map[
            DOOR_LINK_NAME
        ]
        self.handle_link = self.fridge.links_map[
            HANDLE_LINK_NAME
        ]

        self.apple = build_robocasa_apple_actor(
            self.scene,
            self.apple_asset_info,
            initial_pose=self._apple_world_pose(),
            name="robocasa_apple_inside_fridge",
        )
        (
            self.blender,
            self.blender_template,
            self.blender_is_articulation,
        ) = build_blender023(
            self.scene,
            patched_dir=self.patched_blender_dir,
            geometry=self.blender_geometry,
            scale=GLOBAL_SCALE,
            num_envs=self.num_envs,
            name="robocasa_blender023_open",
        )

    def _after_reconfigure(self, options: dict):
        self.stable_counter = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self.success_latched = torch.zeros(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )
        self.apple_left_fridge_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.apple_lifted_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.apple_motion = new_apple_motion_state(self.num_envs, self.device)

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            # TableSceneBuilder.initialize restores its own stock pose.
            set_work_table_pose(self)
            # Per-episode layout (FIRST episode-RNG draw, shared order for the family):
            # fridge offset (the apple rides inside), blender offset + yaw (seat and
            # lid table point ride on it). The long task reads the lid part.
            self._juice_layout = sample_juice_layout(layout_rng(self, int(env_idx[0])))
            fdx, fdy = self._juice_layout["fridge"]
            fridge_root = np.asarray(FRIDGE_ROOT_P, dtype=np.float64) + np.array([fdx, fdy, 0.0])
            bdx, bdy, byaw = self._juice_layout["blender"]
            self.blender_geometry = shifted_blender_geometry(self._blender_geometry_nominal, bdx, bdy, byaw)


            batch_size = len(env_idx)
            if hasattr(self.agent, "get_reset_pos"):
                robot_qpos = self.agent.get_reset_pos(device=self.device)
            else:
                robot_qpos = torch.as_tensor(
                    self.agent.keyframes["rest"].qpos,
                    dtype=torch.float32,
                    device=self.device,
                )
            robot_qpos = (
                robot_qpos.to(torch.float32)
                .expand(batch_size, -1)
                .clone()
            )
            # Shared teleop start pose (tasks/config/workspace_rig.py).
            robot_qpos = apply_arm_init_qpos(self.agent, robot_qpos)
            active_names = [
                joint.name for joint in self.agent.robot.get_active_joints()
            ]
            arm_indices = [
                active_names.index(name)
                for name in getattr(self.agent, "arm_joint_names", [])
                if name in active_names
            ]
            if arm_indices and self.robot_init_qpos_noise > 0:
                index = torch.tensor(
                    arm_indices, dtype=torch.long, device=self.device
                )
                noise = common.to_tensor(
                    self._episode_rng.normal(
                        0.0,
                        self.robot_init_qpos_noise,
                        (batch_size, len(arm_indices)),
                    ),
                    device=self.device,
                )
                robot_qpos[:, index] += noise
            self.agent.reset(robot_qpos)
            self.agent.robot.set_root_pose(sapien.Pose(p=ROBOT_ROOT_P))

            # Reset refrigerator root and hold the door initially open.
            self.fridge.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor(
                        [fridge_root],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    q=torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0]],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
            )

            fridge_qpos = self.fridge.get_qpos()
            fridge_qpos[:, 0] = np.deg2rad(
                np.clip(
                    self.initial_door_angle_deg,
                    0.0,
                    110.0,
                )
            )
            self.fridge.set_qpos(fridge_qpos)
            self.fridge.set_qvel(
                torch.zeros_like(
                    self.fridge.get_qvel()
                )
            )

            apple_pose = self._apple_world_pose(fridge_root)
            apple_p = torch.tensor(
                [apple_pose.p],
                dtype=torch.float32,
                device=self.device,
            )
            apple_q = torch.tensor(
                [apple_pose.q],
                dtype=torch.float32,
                device=self.device,
            )
            zero_velocity = torch.zeros(
                (1, 3),
                dtype=torch.float32,
                device=self.device,
            )

            self.apple.set_linear_velocity(
                zero_velocity
            )
            self.apple.set_angular_velocity(
                zero_velocity
            )
            self.apple.set_pose(
                Pose.create_from_pq(
                    p=apple_p,
                    q=apple_q,
                )
            )
            self.apple.set_linear_velocity(
                zero_velocity
            )
            self.apple.set_angular_velocity(
                zero_velocity
            )

            blender_p = torch.tensor(
                [self.blender_geometry.blender_root], dtype=torch.float32, device=self.device
            )
            blender_q = torch.tensor(
                [self.blender_geometry.blender_quat], dtype=torch.float32, device=self.device
            )
            self.blender.set_pose(Pose.create_from_pq(blender_p, blender_q))
            if self.blender_is_articulation:
                self.blender.set_qpos(torch.zeros_like(self.blender.get_qpos()))
                self.blender.set_qvel(torch.zeros_like(self.blender.get_qvel()))

            self.stable_counter[env_idx] = 0
            self.success_latched[env_idx] = False
            self.apple_left_fridge_latched[env_idx] = False
            self.apple_lifted_latched[env_idx] = False
            fresh = new_apple_motion_state(self.num_envs, self.device)
            for key, value in fresh.items():
                self.apple_motion[key][env_idx] = value[env_idx]

    def evaluate(self):
        apple_position = self.apple.pose.p
        apple_linear_velocity = (
            self.apple.linear_velocity
        )
        apple_angular_velocity = (
            self.apple.angular_velocity
        )

        linear_speed = torch.linalg.norm(
            apple_linear_velocity,
            dim=1,
        )
        angular_speed = torch.linalg.norm(
            apple_angular_velocity,
            dim=1,
        )
        collision_min_z = float(self.apple_asset_info.collision_bounds[0, 2])

        in_fridge = apple_inside_fridge(
            apple_position, collision_min_z, device=self.device, fridge_root_p=self.fridge.pose.p
        )
        # Diagnostic only; the interior test above owns the predicate.
        apple_local_position = apple_position - self.fridge.pose.p
        in_blender, opening_center, blender_xy_distance = apple_inside_blender(
            apple_position, self.blender_geometry, device=self.device
        )
        (
            self.apple_left_fridge_latched,
            self.apple_lifted_latched,
        ) = apple_route_latches(
            apple_position,
            in_fridge,
            self.apple_left_fridge_latched,
            self.apple_lifted_latched,
        )
        self.apple_motion = apple_motion_update(
            self.apple.pose,
            self.apple_motion,
            self._elapsed_steps,
            control_timestep=self.control_timestep,
        )
        stable_now = (
            in_blender
            & apple_settled(self.apple_motion)
            & self.apple_left_fridge_latched
            & self.apple_lifted_latched
        )

        self.stable_counter = torch.where(
            stable_now,
            self.stable_counter + 1,
            torch.zeros_like(
                self.stable_counter
            ),
        )
        success_now = (
            self.stable_counter
            >= SUCCESS_SETTLE_STEPS
        )
        self.success_latched |= success_now

        door_qpos = self.fridge.get_qpos()[:, 0]
        door_qvel = self.fridge.get_qvel()[:, 0]

        return {
            "success": self.success_latched,
            "task_success": self.success_latched,
            "apple_inside_fridge": in_fridge,
            "apple_inside_blender": in_blender,
            "apple_left_fridge_latched": self.apple_left_fridge_latched,
            "apple_lifted_latched": self.apple_lifted_latched,
            "blender_opening_center": opening_center.expand(self.num_envs, -1),
            "blender_xy_distance": blender_xy_distance,
            "apple_position": apple_position,
            "apple_local_position": (
                apple_local_position
            ),
            "apple_pose": (
                self.apple.pose.raw_pose
            ),
            "apple_linear_velocity": (
                apple_linear_velocity
            ),
            "apple_angular_velocity": (
                apple_angular_velocity
            ),
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "pose_linear_speed": self.apple_motion["speed"],
            "pose_angular_speed": self.apple_motion["ang_speed"],
            "stable_now": stable_now,
            "stable_counter": (
                self.stable_counter
            ),
            "stable_progress": torch.clamp(
                self.stable_counter.float()
                / float(SUCCESS_SETTLE_STEPS),
                0.0,
                1.0,
            ),
            "door_qpos": door_qpos,
            "door_angle_deg": torch.rad2deg(
                door_qpos
            ),
            "door_qvel": door_qvel,
            "handle_position": (
                self.handle_link.pose.p
            ),
        }

    def get_state_dict(self):
        state = super().get_state_dict()
        state["juice_apple_task_state"] = torch.cat(
            [
                torch.stack(
                    [
                        self.stable_counter.float(),
                        self.success_latched.float(),
                        self.apple_left_fridge_latched.float(),
                        self.apple_lifted_latched.float(),
                    ],
                    dim=-1,
                ),
                # The settle gate reads a pose finite difference, so its
                # history has to survive a restore or the first restored step
                # differences against a stale pose.
                self.apple_motion["prev_p"].float(),
                self.apple_motion["prev_q"].float(),
                self.apple_motion["speed"][:, None].float(),
                self.apple_motion["ang_speed"][:, None].float(),
                self.apple_motion["last_step"][:, None].float(),
            ],
            dim=-1,
        )
        return state

    def _refresh_blender_geometry(self) -> None:
        nominal = getattr(self, "_blender_geometry_nominal", None)
        blender = getattr(self, "blender", None)
        if nominal is None or blender is None:
            return
        pose = blender.pose
        dx, dy, yaw = blender_shift_from_pose(
            nominal,
            pose.p[0].detach().cpu().numpy(),
            pose.q[0].detach().cpu().numpy(),
        )
        self.blender_geometry = shifted_blender_geometry(nominal, dx, dy, yaw)

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("juice_apple_task_state", None)
        if env_idx is None:
            restore_idx = torch.arange(self.num_envs, device=self.device)
            super().set_state_dict(state)
        else:
            restore_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            ).flatten()
            super().set_state_dict(state, restore_idx)
        # The blender actor rides in the state dict and restores to 0.0 mm; the cached
        # geometry (root, quaternion, seat, lid table point) that the predicates read does
        # not, so rebuild it from the blender the state just put back (measured:
        # 19.83 mm of root / 20.62 mm of opening centre on a restored frame without this).
        self._refresh_blender_geometry()
        if isinstance(task_state, dict):
            task_state = task_state.get("value")
        if task_state is None:
            return
        task_state = common.to_tensor(
            task_state, device=self.device
        ).reshape(len(restore_idx), -1)
        expected = 4 + APPLE_MOTION_STATE_FIELDS
        if task_state.shape[-1] != expected:
            raise ValueError(
                f"juice_apple_task_state must have {expected} fields"
            )
        self.stable_counter[restore_idx] = task_state[:, 0].long()
        self.success_latched[restore_idx] = task_state[:, 1].bool()
        self.apple_left_fridge_latched[restore_idx] = task_state[:, 2].bool()
        self.apple_lifted_latched[restore_idx] = task_state[:, 3].bool()
        self.apple_motion["prev_p"][restore_idx] = task_state[:, 4:7].float()
        self.apple_motion["prev_q"][restore_idx] = task_state[:, 7:11].float()
        self.apple_motion["speed"][restore_idx] = task_state[:, 11].float()
        self.apple_motion["ang_speed"][restore_idx] = task_state[:, 12].float()
        self.apple_motion["last_step"][restore_idx] = task_state[:, 13].to(
            torch.int32
        )
        if env_idx is not None:
            # A demo-state restore starts a FRESH episode clock at zero while the
            # register we just restored says "last differenced at step N". The
            # reset-time get_info() would then read as a new control step and
            # difference the restored pose against itself, reporting a settled
            # apple. Rebase onto the new clock.
            self.apple_motion["last_step"][restore_idx] = self._elapsed_steps[
                restore_idx
            ].to(torch.int32)

    def _get_obs_extra(self, info: dict):
        return {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "apple_pose": info["apple_pose"],
            "apple_linear_velocity": info[
                "apple_linear_velocity"
            ],
            "apple_angular_velocity": info[
                "apple_angular_velocity"
            ],
            "apple_inside_fridge": info[
                "apple_inside_fridge"
            ].float().unsqueeze(1),
            "apple_inside_blender": info["apple_inside_blender"].float().unsqueeze(1),
            "blender_opening_center": info["blender_opening_center"],
            "blender_xy_distance": info["blender_xy_distance"].unsqueeze(1),
            "door_qpos": info[
                "door_qpos"
            ].unsqueeze(1),
            "handle_position": info[
                "handle_position"
            ],
        }

    def compute_sparse_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        return info["task_success"].float()

    def compute_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        tcp_position = self.agent.tcp.pose.p
        reach_distance = torch.linalg.norm(tcp_position - info["apple_position"], dim=1)
        reaching = 1.0 - torch.tanh(5.0 * reach_distance)
        removed = info["apple_left_fridge_latched"].float()
        lifted = info["apple_lifted_latched"].float()
        placement = torch.exp(-20.0 * info["blender_xy_distance"])
        stable_reward = torch.exp(
            -18.0 * info["linear_speed"]
        ) * torch.exp(
            -3.0 * info["angular_speed"]
        )
        reward = (
            reaching + removed + lifted + 2.0 * placement
            + info["apple_inside_blender"].float()
            + 0.5 * stable_reward + 1.5 * info["stable_progress"]
        )

        return torch.where(
            info["task_success"],
            torch.full_like(reward, 10.0),
            reward,
        )

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        return (
            self.compute_dense_reward(
                obs,
                action,
                info,
            )
            / 10.0
        )


# =============================================================================
# Runner
# =============================================================================


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(np.asarray(value).mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        default=(
            "JuicePreparationAppleToBlender-Sharpa-v1"
        ),
    )
    parser.add_argument(
        "--apple-dir",
        default=str(DEFAULT_APPLE_DIR),
    )
    parser.add_argument(
        "--target-apple-width",
        type=float,
        default=TARGET_APPLE_WIDTH,
    )
    parser.add_argument(
        "--target-apple-mass",
        type=float,
        default=TARGET_APPLE_MASS,
    )
    parser.add_argument(
        "--door-angle-deg",
        type=float,
        default=75.0,
    )
    parser.add_argument(
        "--control-mode",
        default="pd_joint_pos",
    )
    parser.add_argument(
        "--blender-dir",
        default=str(DEFAULT_BLENDER_DIR),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
    )
    parser.add_argument(
        "--video-dir",
        default=(
            "tasks/videos_juice_take_apple_to_blender"
        ),
    )
    parser.add_argument(
        "--debug-print",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(
        "### CODE_VERSION_MARKER: "
        "take_robocasa_apple_to_open_blender_v1 ###"
    )

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        control_mode=args.control_mode,
        reward_mode="dense",
        render_mode="rgb_array",
        apple_dir=args.apple_dir,
        target_apple_width=(
            args.target_apple_width
        ),
        target_apple_mass=(
            args.target_apple_mass
        ),
        blender_dir=args.blender_dir,
        initial_door_angle_deg=(
            args.door_angle_deg
        ),
    )

    if args.record_video:
        env = RecordEpisode(
            env,
            output_dir=args.video_dir,
            save_trajectory=False,
            save_video_trigger=lambda _: True,
            max_steps_per_video=args.steps,
            video_fps=CONTROL_FREQUENCY,
        )

    observation, info = env.reset(seed=args.seed)

    print("\n[environment]")
    print("Environment:", args.env_id)
    print("Door angle:", args.door_angle_deg)
    print("Apple directory:", args.apple_dir)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    for step in range(args.steps):
        action = np.zeros(
            env.action_space.shape,
            dtype=np.float32,
        )
        # Keep the Panda gripper open.
        action[-1] = 1.0

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)
        env.render()

        succeeded = bool(
            info["task_success"].item()
        )

        if args.debug_print and (
            step % 5 == 0 or succeeded
        ):
            apple_world = (
                info["apple_position"][0]
                .detach()
                .cpu()
                .numpy()
            )
            apple_local = (
                info["apple_local_position"][0]
                .detach()
                .cpu()
                .numpy()
            )

            print(
                f"step={step:04d} "
                f"reward={_scalar(reward):.3f} "
                f"door={_scalar(info['door_angle_deg']):.2f}deg "
                f"apple_world={np.round(apple_world, 5).tolist()} "
                f"apple_local={np.round(apple_local, 5).tolist()} "
                f"lin={_scalar(info['linear_speed']):.5f} "
                f"ang={_scalar(info['angular_speed']):.5f} "
                f"in_fridge={int(bool(info['apple_inside_fridge'].item()))} "
                f"in_blender={int(bool(info['apple_inside_blender'].item()))} "
                f"stable={int(bool(info['stable_now'].item()))} "
                f"settle={int(info['stable_counter'].item())}/"
                f"{SUCCESS_SETTLE_STEPS} "
                f"success={int(succeeded)}"
            )

        done = (
            bool(terminated.item())
            or bool(truncated.item())
        )
        # Keep recording after the apple has settled so the output video
        # clearly shows that it remains stable inside the open refrigerator.
        if done:
            break

    env.close()

    if args.record_video:
        print("\nVideo directory:", args.video_dir)


if __name__ == "__main__":
    main()

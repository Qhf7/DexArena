"""
Single-arm ManiSkill task: open the Juice refrigerator with the same left-arm
mount used by the composed dual-arm task.

Environment
-----------
{Gripper,Sharpa,Wuji}OpenFridge-v1

Scene
-----
- Right Panda arm on a mobile base with a selectable gripper/Sharpa/Wuji end effector.
- PartNet-Mobility refrigerator 10797.
- PartNet-Mobility refrigerator only; no fruit assets are loaded.
- Stable translation-only scripted policy for pd_ee_delta_pose.

Recommended command
-------------------
python tasks/juice_preparation/juice_preparation_open_fridge.py \
  --env-id JuicePreparationOpenFridge-Sharpa-v1 \
  --control-mode pd_joint_pos \
  --reward-mode dense \
  --scripted \
  --record-video \
  --debug-print

Asset path
----------
The vendored refrigerator is loaded from
``tasks/assets/task_object_meshes/juice/tabletop_mini_fridge_v1``.

The URDF must remain beside its ``textured_objs`` directory.

Notes
-----
- This first version deliberately supports num_envs=1.
- Refrigerator scaling, yaw, bottom height, hinge position, and handle position
  are estimated automatically from the URDF and OBJ meshes.
- The refrigerator door is opened by following a circular handle trajectory,
  rather than pulling in a straight line.
- This isolated version has no RoboCasa dependency and is intended only for
  validating refrigerator placement, grasping, and scripted opening.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch
import trimesh
from transforms3d.euler import euler2mat

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from tasks.config.hands import register_task
from tasks.scene_builders.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import (
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.config.layout_randomization import ARM_INIT_QPOS_NOISE_STD, arm_qpos_noise, episode_rng, layout_rng
from tasks.juice_preparation.juice_preparation_place_lid import sample_juice_layout
from tasks.config.sim_profiles import make_sim_config
from tasks.config.workspace_rig import apply_arm_init_qpos
from tasks.juice_preparation.juice_preparation_place_lid import configure_task_object_physics
from tasks.juice_preparation.juice_preparation_apple_to_blender import FRIDGE_ROOT_P, ROBOT_ROOT_P
from tasks.config.workspace_rig import (
    WORK_TABLE_X_OFFSET,
    set_work_table_pose,
)

# The door stages use the same LEFT-arm mounts as the composed dual-arm task.
from tasks.robots.panda_gripper_left_mobile_base import PandaGripperLeftMobileBase
from tasks.robots.panda_sharpa_hand_left_mobile_base import (
    PandaSharpaHandLeftAngledMobileBase,
)
from tasks.robots.panda_wuji_hand_left_mobile_base import PandaWujiHandLeftMobileBase
from tasks.robots.panda_xhand_hand_left_mobile_base import PandaXHandHandLeftMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag


# =============================================================================
# Constants
# =============================================================================

CONTROL_FREQUENCY = 20

# Same refrigerator asset used by take_apple_to_blender_v1.py.
JUICE_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "task_object_meshes" / "juice"
CUSTOM_FRIDGE_ASSET_DIR = JUICE_ASSET_DIR / "tabletop_mini_fridge_v1"
CUSTOM_FRIDGE_URDF = CUSTOM_FRIDGE_ASSET_DIR / "tabletop_mini_fridge.urdf"
DEFAULT_FRIDGE_URDF_CANDIDATES = (CUSTOM_FRIDGE_URDF,)
CUSTOM_FRIDGE_ROOT_POS = FRIDGE_ROOT_P.copy()
CUSTOM_DOOR_JOINT_NAME = "door_hinge_joint"
CUSTOM_DOOR_LINK_NAME = "fridge_door"
CUSTOM_HANDLE_LINK_NAME = "door_handle"

ROBOT_ROOT_POS = ROBOT_ROOT_P.copy()

ROBOT_UID_GRIPPER = "panda_gripper_left_mobile_base"
ROBOT_UID_SHARPA = "panda_sharpa_hand_left_angled_mobile_base"
ROBOT_UID_WUJI = "panda_wuji_hand_left_angled_mobile_base"
ROBOT_UID_XHAND = "panda_xhand_hand_left_angled_mobile_base"
ROBOT_UIDS = (
    ROBOT_UID_GRIPPER,
    ROBOT_UID_SHARPA,
    ROBOT_UID_WUJI,
    ROBOT_UID_XHAND,
)

# Keep the refrigerator inside Panda's practical workspace.
DEFAULT_FRIDGE_TARGET_HEIGHT = 0.44
DEFAULT_HANDLE_TARGET_X = 0.075
DEFAULT_HANDLE_TARGET_Y = 0.0

# Door target and success thresholds.
TARGET_OPEN_ANGLE_DEG = 75.0
SUCCESS_OPEN_ANGLE_DEG = 75.0
SUCCESS_CLOSED_ANGLE_DEG = 1.0
DOOR_STATIC_THRESHOLD = 0.12
ARM_STATIC_THRESHOLD = 0.30
SUCCESS_SETTLE_STEPS = 6
CONTACT_FORCE_THRESHOLD = 1e-4


SUCCESS_REWARD = 10.0

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
# Shared door predicate
#
# Both fridge stages of the long Juice task -- open at stage 0, close at stage
# 2 -- and both fridge subtasks read exactly these functions, so the long task
# cannot silently drift away from the subtask it is supposed to contain.
# =============================================================================


def door_success_qpos(door_lower: float, door_upper: float, angle_deg: float) -> float:
    """Hinge qpos that counts as ``angle_deg`` of travel, clipped to the limit."""
    return float(
        np.clip(door_lower + np.deg2rad(angle_deg), door_lower, door_upper)
    )


def door_settle_update(
    door_qpos: torch.Tensor,
    door_qvel: torch.Tensor,
    settle_counter: torch.Tensor,
    *,
    threshold_qpos: float,
    opening: bool,
    active: torch.Tensor = None,
):
    """Advance the settle debounce for one fridge stage.

    ``opening`` selects the travel test: the door must have reached
    ``threshold_qpos`` from below when opening, from above when closing. The
    hinge must additionally be quiet -- contact and arm speed stay diagnostics.

    ``active`` is the long task's stage mask. Envs outside the stage keep their
    counter untouched instead of being zeroed by a stage block that is not
    theirs; passing ``None`` (the subtasks) makes every env active.

    Returns ``(reached, settled_now, settle_counter)``.
    """
    reached = (
        door_qpos >= threshold_qpos if opening else door_qpos <= threshold_qpos
    )
    settled_now = reached & (torch.abs(door_qvel) <= DOOR_STATIC_THRESHOLD)
    if active is not None:
        settled_now = settled_now & active
    ticked = torch.where(
        settled_now, settle_counter + 1, torch.zeros_like(settle_counter)
    )
    if active is not None:
        ticked = torch.where(active, ticked, settle_counter)
    return reached, settled_now, ticked


# =============================================================================
# General math helpers
# =============================================================================


def _parse_floats(text: str | None, count: int, default: tuple[float, ...]):
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(x) for x in text.split()], dtype=np.float64)
    if len(values) != count:
        raise ValueError(f"Expected {count} values, received {text!r}")
    return values


def _origin_matrix(element: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if element is None:
        return matrix
    xyz = _parse_floats(element.get("xyz"), 3, (0.0, 0.0, 0.0))
    rpy = _parse_floats(element.get("rpy"), 3, (0.0, 0.0, 0.0))
    matrix[:3, :3] = euler2mat(*rpy)
    matrix[:3, 3] = xyz
    return matrix


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _normalize(vector: np.ndarray, fallback=None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        if fallback is None:
            raise ValueError("Cannot normalize a near-zero vector")
        return _normalize(np.asarray(fallback, dtype=np.float64))
    return vector / norm


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array(
        [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)],
        dtype=np.float64,
    )


def _quat_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_to_rotvec_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = _normalize_quaternion_wxyz(quaternion)
    if quaternion[0] < 0:
        quaternion = -quaternion

    w = float(np.clip(quaternion[0], -1.0, 1.0))
    xyz = quaternion[1:]
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * math.atan2(xyz_norm, w)
    return xyz / xyz_norm * angle


def _clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > max_norm and norm > 1e-12:
        return vector * (max_norm / norm)
    return vector


def _rotate_about_axis(
    vector: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rodrigues rotation."""
    vector = np.asarray(vector, dtype=np.float64)
    axis = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    return (
        vector * c
        + np.cross(axis, vector) * s
        + axis * np.dot(axis, vector) * (1.0 - c)
    )


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tensor_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(np.asarray(value).mean())


# =============================================================================
# URDF / mesh geometry extraction
# =============================================================================


def _load_mesh_vertices(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, process=False, force=None)

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise RuntimeError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"Unsupported mesh type {type(loaded)} for {path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"Invalid or empty mesh vertices in {path}")
    return vertices


def _find_link(root: ET.Element, name: str) -> ET.Element:
    for link in root.findall("link"):
        if link.get("name") == name:
            return link
    raise KeyError(f"URDF link {name!r} was not found")


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    for joint in root.findall("joint"):
        if joint.get("name") == name:
            return joint
    raise KeyError(f"URDF joint {name!r} was not found")


def _joint_parent_name(joint: ET.Element) -> str:
    parent = joint.find("parent")
    if parent is None or parent.get("link") is None:
        raise RuntimeError("Joint has no parent link")
    return str(parent.get("link"))


def _joint_child_name(joint: ET.Element) -> str:
    child = joint.find("child")
    if child is None or child.get("link") is None:
        raise RuntimeError("Joint has no child link")
    return str(child.get("link"))


def _mesh_path_from_visual(urdf_dir: Path, visual: ET.Element) -> Path:
    mesh = visual.find("./geometry/mesh")
    if mesh is None or mesh.get("filename") is None:
        raise RuntimeError("Only mesh-based refrigerator visuals are supported")
    path = Path(str(mesh.get("filename")))
    if not path.is_absolute():
        path = urdf_dir / path
    return path.resolve()


def _visual_vertices_in_link_frame(
    urdf_dir: Path,
    visual: ET.Element,
) -> tuple[np.ndarray, Path]:
    mesh_path = _mesh_path_from_visual(urdf_dir, visual)
    vertices = _load_mesh_vertices(mesh_path)
    visual_transform = _origin_matrix(visual.find("origin"))
    return _transform_points(visual_transform, vertices), mesh_path


@dataclass
class FridgeGeometry:
    scale: float

    bounds_root: np.ndarray
    body_bounds_root: np.ndarray

    hinge_root: np.ndarray
    hinge_axis_root: np.ndarray
    handle_root_closed: np.ndarray
    handle_local_door: np.ndarray
    front_outward_root: np.ndarray

    root_yaw: float
    root_position: np.ndarray
    root_quaternion: np.ndarray

    world_bounds_closed: np.ndarray
    shelf_center_world: np.ndarray
    shelf_half_size: np.ndarray

    door_joint_name: str
    door_link_name: str
    body_link_name: str


def extract_fridge_geometry(
    urdf_path: Path,
    target_height: float,
    handle_target_x: float,
    handle_target_y: float,
) -> FridgeGeometry:
    """
    Extract closed-state geometry for refrigerator 10797.

    The implementation remains generic enough to tolerate small changes to the
    same two-link URDF. The uploaded 10797 model contains:
        base -> fixed -> link_0 -> revolute joint_1 -> link_1
    """
    urdf_path = urdf_path.resolve()
    root = ET.parse(urdf_path).getroot()
    urdf_dir = urdf_path.parent

    door_joint = None
    fixed_joint = None
    for joint in root.findall("joint"):
        if joint.get("type") in {"revolute", "continuous"}:
            door_joint = joint
        elif joint.get("type") == "fixed":
            child = _joint_child_name(joint)
            if child == "link_0":
                fixed_joint = joint

    if door_joint is None:
        raise RuntimeError("No revolute refrigerator-door joint was found")
    if fixed_joint is None:
        raise RuntimeError("No fixed base-to-body joint was found")

    body_link_name = _joint_child_name(fixed_joint)
    door_link_name = _joint_child_name(door_joint)
    door_joint_name = str(door_joint.get("name"))

    if _joint_parent_name(door_joint) != body_link_name:
        raise RuntimeError(
            "Expected the door joint parent to be the refrigerator body"
        )

    body_link = _find_link(root, body_link_name)
    door_link = _find_link(root, door_link_name)

    base_to_body = _origin_matrix(fixed_joint.find("origin"))
    body_to_door_closed = _origin_matrix(door_joint.find("origin"))

    axis_element = door_joint.find("axis")
    hinge_axis_joint = _parse_floats(
        None if axis_element is None else axis_element.get("xyz"),
        3,
        (0.0, 0.0, 1.0),
    )

    body_vertices_root = []
    all_vertices_root = []
    handle_vertices_link = None
    handle_vertices_root = None

    for visual in body_link.findall("visual"):
        vertices_link, _ = _visual_vertices_in_link_frame(urdf_dir, visual)
        vertices_root = _transform_points(base_to_body, vertices_link)
        body_vertices_root.append(vertices_root)
        all_vertices_root.append(vertices_root)

    for visual in door_link.findall("visual"):
        vertices_link, mesh_path = _visual_vertices_in_link_frame(
            urdf_dir,
            visual,
        )
        vertices_body = _transform_points(body_to_door_closed, vertices_link)
        vertices_root = _transform_points(base_to_body, vertices_body)
        all_vertices_root.append(vertices_root)

        visual_name = str(visual.get("name", "")).lower()
        mesh_name = mesh_path.name.lower()
        if "handle" in visual_name or mesh_name == "new-1.obj":
            handle_vertices_link = vertices_link
            handle_vertices_root = vertices_root

    if not body_vertices_root:
        raise RuntimeError("No visual meshes were found on the refrigerator body")
    if handle_vertices_link is None or handle_vertices_root is None:
        raise RuntimeError(
            "Could not identify the handle visual. Expected name containing "
            "'handle' or mesh filename 'new-1.obj'."
        )

    body_vertices_root = np.concatenate(body_vertices_root, axis=0)
    all_vertices_root = np.concatenate(all_vertices_root, axis=0)

    unscaled_bounds = np.stack(
        [all_vertices_root.min(axis=0), all_vertices_root.max(axis=0)],
        axis=0,
    )
    unscaled_height = float(unscaled_bounds[1, 2] - unscaled_bounds[0, 2])
    if unscaled_height <= 1e-8:
        raise RuntimeError("Refrigerator mesh has invalid height")

    scale = float(target_height / unscaled_height)

    body_vertices_root *= scale
    all_vertices_root *= scale
    handle_vertices_link *= scale
    handle_vertices_root *= scale

    bounds_root = np.stack(
        [all_vertices_root.min(axis=0), all_vertices_root.max(axis=0)],
        axis=0,
    )
    body_bounds_root = np.stack(
        [body_vertices_root.min(axis=0), body_vertices_root.max(axis=0)],
        axis=0,
    )

    hinge_root = (
        base_to_body[:3, 3]
        + base_to_body[:3, :3] @ body_to_door_closed[:3, 3]
    ) * scale
    hinge_axis_root = _normalize(
        base_to_body[:3, :3]
        @ body_to_door_closed[:3, :3]
        @ hinge_axis_joint
    )

    handle_root_closed = handle_vertices_root.mean(axis=0)
    handle_local_door = handle_vertices_link.mean(axis=0)

    body_center = 0.5 * (body_bounds_root[0] + body_bounds_root[1])
    front_outward_root = handle_root_closed - body_center
    front_outward_root[2] = 0.0
    front_outward_root = _normalize(front_outward_root, fallback=[-1, 0, 0])

    # Rotate the model so the closed door faces the Panda at world -X.
    target_front = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    raw_root_yaw = (
        math.atan2(target_front[1], target_front[0])
        - math.atan2(front_outward_root[1], front_outward_root[0])
    )

    # PartNet handle geometry is slightly asymmetric, so estimating the front
    # direction from the handle center can produce a few degrees of unwanted
    # yaw. Snap the final refrigerator yaw to the nearest 90 degrees so its
    # edges remain parallel to the table edges.
    root_yaw = (
        round(raw_root_yaw / (0.5 * math.pi))
        * (0.5 * math.pi)
    )
    root_rotation = euler2mat(0.0, 0.0, root_yaw)

    corners = np.array(
        [
            [x, y, z]
            for x in (bounds_root[0, 0], bounds_root[1, 0])
            for y in (bounds_root[0, 1], bounds_root[1, 1])
            for z in (bounds_root[0, 2], bounds_root[1, 2])
        ],
        dtype=np.float64,
    )
    rotated_corners = corners @ root_rotation.T
    rotated_handle = root_rotation @ handle_root_closed

    root_position = np.zeros(3, dtype=np.float64)
    root_position[0] = handle_target_x - rotated_handle[0]
    root_position[1] = handle_target_y - rotated_handle[1]
    root_position[2] = -rotated_corners[:, 2].min() + 0.002

    world_corners = rotated_corners + root_position
    world_bounds_closed = np.stack(
        [world_corners.min(axis=0), world_corners.max(axis=0)],
        axis=0,
    )

    # A stable, visible inner shelf. Front is world -X; inside is +X.
    body_depth = float(
        world_bounds_closed[1, 0] - world_bounds_closed[0, 0]
    )
    body_width = float(
        world_bounds_closed[1, 1] - world_bounds_closed[0, 1]
    )
    fridge_height = float(
        world_bounds_closed[1, 2] - world_bounds_closed[0, 2]
    )

    shelf_center_world = np.array(
        [
            world_bounds_closed[0, 0] + 0.56 * body_depth,
            0.5
            * (world_bounds_closed[0, 1] + world_bounds_closed[1, 1]),
            world_bounds_closed[0, 2] + 0.48 * fridge_height,
        ],
        dtype=np.float64,
    )
    shelf_half_size = np.array(
        [
            max(0.055, 0.28 * body_depth),
            max(0.090, 0.34 * body_width),
            0.008,
        ],
        dtype=np.float64,
    )

    return FridgeGeometry(
        scale=scale,
        bounds_root=bounds_root,
        body_bounds_root=body_bounds_root,
        hinge_root=hinge_root,
        hinge_axis_root=hinge_axis_root,
        handle_root_closed=handle_root_closed,
        handle_local_door=handle_local_door,
        front_outward_root=front_outward_root,
        root_yaw=root_yaw,
        root_position=root_position,
        root_quaternion=_yaw_quaternion(root_yaw),
        world_bounds_closed=world_bounds_closed,
        shelf_center_world=shelf_center_world,
        shelf_half_size=shelf_half_size,
        door_joint_name=door_joint_name,
        door_link_name=door_link_name,
        body_link_name=body_link_name,
    )


def resolve_fridge_urdf(explicit_path: str | None) -> Path:
    candidates = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    environment_path = os.environ.get("FRIDGE_URDF")
    if environment_path:
        candidates.append(Path(environment_path))

    candidates.extend(DEFAULT_FRIDGE_URDF_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            textured_objs = candidate.parent / "textured_objs"
            if not textured_objs.exists():
                raise FileNotFoundError(
                    f"Found refrigerator URDF at {candidate}, but its "
                    f"textured_objs directory is missing: {textured_objs}"
                )
            return candidate.resolve()

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate refrigerator 10797 mobility.urdf.\n"
        "Searched:\n"
        f"{searched}\n"
        "Pass --fridge-urdf or export FRIDGE_URDF."
    )


# =============================================================================
# Environment
# =============================================================================


@register_task(
    "JuicePreparationOpenFridge",
    side="left",
    max_episode_steps=650,
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
)
class JuicePreparationOpenFridgeEnv(BaseEnv):
    """Open the Juice refrigerator with the assigned left-arm robot."""

    SUPPORTED_ROBOTS = list(ROBOT_UIDS)
    agent: Any

    def __init__(
        self,
        *args,
        robot_uids: str = ROBOT_UID_SHARPA,
        fridge_urdf: str | None = None,
        fridge_target_height: float = DEFAULT_FRIDGE_TARGET_HEIGHT,
        handle_target_x: float = DEFAULT_HANDLE_TARGET_X,
        handle_target_y: float = DEFAULT_HANDLE_TARGET_Y,
        robot_init_qpos_noise: float = ARM_INIT_QPOS_NOISE_STD,
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
        if fridge_urdf is not None:
            requested = Path(fridge_urdf).resolve()
            if requested != CUSTOM_FRIDGE_URDF.resolve():
                raise ValueError(
                    "Juice uses its vendored custom refrigerator; arbitrary "
                    "fridge_urdf overrides are not implemented"
                )
        legacy_placement = (
            float(fridge_target_height),
            float(handle_target_x),
            float(handle_target_y),
        )
        expected_placement = (
            DEFAULT_FRIDGE_TARGET_HEIGHT,
            DEFAULT_HANDLE_TARGET_X,
            DEFAULT_HANDLE_TARGET_Y,
        )
        if legacy_placement != expected_placement:
            raise ValueError(
                "fridge_target_height/handle_target_x/handle_target_y are "
                "legacy PartNet parameters and cannot reposition the custom "
                "Juice refrigerator; use the shared task layout constants"
            )
        if num_envs != 1:
            raise ValueError(
                "The refrigerator tasks currently support num_envs=1 only."
            )
        if robot_uids not in ROBOT_UIDS:
            raise ValueError(f"Unsupported robot {robot_uids!r}; expected one of {ROBOT_UIDS}")

        # ``fridge_urdf`` and the old PartNet placement arguments remain in the
        # signature for checkpoint/config compatibility, but this task now
        # intentionally uses the custom fridge from take_apple_to_blender_v1.
        self.fridge_urdf_path = CUSTOM_FRIDGE_URDF
        if not self.fridge_urdf_path.is_file():
            raise FileNotFoundError(f"Custom refrigerator URDF not found: {self.fridge_urdf_path}")
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)

        if reconfiguration_freq is None:
            reconfiguration_freq = 0

        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return make_sim_config(
            spacing=3,
            max_rigid_contact_count=2**20,
            max_rigid_patch_count=2**19,
        )

    @property
    def _default_sensor_configs(self):
        # Refrigerator front points approximately toward world -X. Place the
        # camera on that side and offset it in Y, so the image shows the door,
        # the handle, and the Panda from an oblique front view.
        pose = sapien_utils.look_at(
            eye=[-1.02 + WORK_TABLE_X_OFFSET, -0.18, 0.44],
            target=[0.055 + WORK_TABLE_X_OFFSET, 0.215, 0.145],
        )
        return [
            CameraConfig(
                "base_camera",
                pose,
                320,
                240,
                np.deg2rad(50.0),
                0.01,
                100.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[-1.16 + WORK_TABLE_X_OFFSET, -0.23, 0.50],
            target=[0.045 + WORK_TABLE_X_OFFSET, 0.220, 0.180],
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
            sapien.Pose(p=ROBOT_ROOT_POS.tolist()),
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        set_work_table_pose(self)

        loader = self.scene.create_urdf_loader()
        loader.name = "custom_tabletop_mini_fridge"
        loader.fix_root_link = True
        loader.scale = 1.0
        loader.load_multiple_collisions_from_file = True

        parsed = loader.parse(str(self.fridge_urdf_path))
        builders = parsed["articulation_builders"]
        if len(builders) != 1:
            raise RuntimeError(
                f"Expected one refrigerator articulation, got {len(builders)}"
            )

        builder = builders[0]
        builder.disable_self_collisions = True
        builder.set_scene_idxs([0])
        builder.initial_pose = sapien.Pose(p=CUSTOM_FRIDGE_ROOT_POS.tolist())

        self.fridge = builder.build(name="custom_tabletop_mini_fridge")
        configure_task_object_physics(self.fridge, "hard_plastic")
        # Minimal exact geometry for the vendored custom URDF. The legacy
        # PartNet extractor below is retained only for old scripted-policy
        # compatibility and does not match this asset's link naming.
        self.fridge_geometry = SimpleNamespace(
            root_position=CUSTOM_FRIDGE_ROOT_POS.copy(),
            root_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            hinge_root=np.array([-0.173, 0.180, 0.0]),
            hinge_axis_root=np.array([0.0, 0.0, -1.0]),
            handle_root_closed=np.array([-0.258, -0.115, 0.285]),
            front_outward_root=np.array([-1.0, 0.0, 0.0]),
        )

        self.door_joint = self.fridge.active_joints_map[CUSTOM_DOOR_JOINT_NAME]
        self.door_link = self.fridge.links_map[CUSTOM_DOOR_LINK_NAME]
        self.handle_link = self.fridge.links_map[CUSTOM_HANDLE_LINK_NAME]

        self.door_joint.set_drive_properties(
            stiffness=0.0,
            damping=0.85,
            force_limit=85.0,
            mode="force",
        )
        if hasattr(self.door_joint, "set_friction"):
            self.door_joint.set_friction(FRIDGE_DOOR_HINGE_FRICTION)

        qlimits = (
            self.fridge.get_qlimits()
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 2)
        )
        if qlimits.shape[0] != 1:
            raise RuntimeError(
                f"Expected one refrigerator joint, got limits={qlimits}"
            )

        self.door_lower = float(qlimits[0, 0])
        self.door_upper = float(qlimits[0, 1])
        self.target_open_qpos = door_success_qpos(
            self.door_lower, self.door_upper, TARGET_OPEN_ANGLE_DEG
        )
        self.success_open_qpos = door_success_qpos(
            self.door_lower, self.door_upper, SUCCESS_OPEN_ANGLE_DEG
        )

        print(f"[fridge asset] urdf={self.fridge_urdf_path} root_p={CUSTOM_FRIDGE_ROOT_POS.tolist()}")

    def _after_reconfigure(self, options: dict):
        self.success_settle_counter = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self.success_latched = torch.zeros(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )

    def _set_robot_root_pose(self):
        root_pose = sapien.Pose(p=ROBOT_ROOT_POS.tolist())
        if hasattr(self.agent.robot, "set_root_pose"):
            self.agent.robot.set_root_pose(root_pose)
        else:
            self.agent.robot.set_pose(root_pose)

    def _reset_agent(self, batch_size: int):
        if hasattr(self.agent, "get_reset_pos"):
            initial = self.agent.get_reset_pos(device=self.device)
        else:
            initial = torch.as_tensor(
                self.agent.keyframes["rest"].qpos,
                dtype=torch.float32,
                device=self.device,
            )
        qpos = initial.expand(batch_size, -1).clone()
        # Shared teleop start pose (tasks/config/workspace_rig.py): every task
        # must reset the arms into the same posture or the operator's hands
        # map onto a different attitude from the first frame.
        qpos = apply_arm_init_qpos(self.agent, qpos)

        for k in range(batch_size):
            index, noise = arm_qpos_noise(
                episode_rng(self, k), self.agent, 1, self.robot_init_qpos_noise, device=self.device)
            qpos[k, index] += noise[0]

        self.agent.reset(qpos)
        self._set_robot_root_pose()

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            # TableSceneBuilder.initialize restores its own stock pose.
            set_work_table_pose(self)
            # Per-episode layout (first episode-RNG draw, family-wide order): fridge offset.
            fdx, fdy = sample_juice_layout(layout_rng(self, int(env_idx[0])))["fridge"]
            self._reset_agent(len(env_idx))

            root_p = torch.tensor(
                np.asarray(CUSTOM_FRIDGE_ROOT_POS, dtype=np.float64) + np.array([fdx, fdy, 0.0]),
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3)
            root_q = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            ).view(1, 4)

            self.fridge.set_pose(Pose.create_from_pq(p=root_p, q=root_q))

            qpos = self.fridge.get_qpos()
            qpos[:, 0] = self.door_lower
            self.fridge.set_qpos(qpos)
            self.fridge.set_qvel(torch.zeros_like(self.fridge.get_qvel()))

            self.success_settle_counter[env_idx] = 0
            self.success_latched[env_idx] = False
            self.hand_contact_latched = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

    # -------------------------------------------------------------------------
    # Geometry exposed to reward and policy
    # -------------------------------------------------------------------------

    def _fridge_root_pose_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        pose = self.fridge.pose
        position = _to_numpy(pose.p)[0].astype(np.float64)
        quaternion = _to_numpy(pose.q)[0].astype(np.float64)
        return position, quaternion

    def closed_handle_world_from_geometry(self) -> np.ndarray:
        root_position = self.fridge_geometry.root_position
        root_rotation = _quat_to_matrix_wxyz(
            self.fridge_geometry.root_quaternion
        )
        return (
            root_position
            + root_rotation @ self.fridge_geometry.handle_root_closed
        )

    def door_frame_at_angle(
        self,
        angle: float,
    ) -> dict[str, np.ndarray]:
        """
        Return hinge, handle, front normal, approach, and grasp-closing axes
        for a requested door angle.
        """
        root_position, root_quaternion = self._fridge_root_pose_numpy()
        root_rotation = _quat_to_matrix_wxyz(root_quaternion)
        geometry = self.fridge_geometry

        hinge = root_position + root_rotation @ geometry.hinge_root
        axis = _normalize(root_rotation @ geometry.hinge_axis_root)

        handle_closed = (
            root_position
            + root_rotation @ geometry.handle_root_closed
        )
        radius_closed = handle_closed - hinge
        handle = hinge + _rotate_about_axis(
            radius_closed,
            axis,
            float(angle - self.door_lower),
        )

        front_closed = _normalize(
            root_rotation @ geometry.front_outward_root
        )
        front_outward = _normalize(
            _rotate_about_axis(
                front_closed,
                axis,
                float(angle - self.door_lower),
            )
        )

        # front_outward points from the fridge toward the robot.
        # Panda's approaching vector points from the robot toward the object.
        approaching = -front_outward
        closing = _normalize(
            np.cross(axis, approaching),
            fallback=[0.0, 1.0, 0.0],
        )

        return {
            "hinge": hinge,
            "axis": axis,
            "handle": handle,
            "front_outward": front_outward,
            "approaching": approaching,
            "closing": closing,
        }

    def grasp_pose_at_angle(
        self,
        angle: float,
        approach_offset: float = 0.0,
        vertical_offset: float = 0.0,
    ) -> sapien.Pose:
        frame = self.door_frame_at_angle(angle)
        center = (
            frame["handle"]
            - frame["approaching"] * float(approach_offset)
            + frame["axis"] * float(vertical_offset)
        )
        return Panda.build_grasp_pose(
            frame["approaching"],
            frame["closing"],
            center,
        )

    @property
    def current_door_qpos(self) -> float:
        return float(self.fridge.get_qpos()[0, 0].item())

    def current_handle_position_tensor(self) -> torch.Tensor:
        # The custom asset has a dedicated fixed handle link whose origin is
        # the centre of the vertical grasp bar.
        return self.handle_link.pose.p

    def _task_tcp_position(self) -> torch.Tensor:
        value = getattr(self.agent, "tcp_pos", None)
        if value is not None:
            return value
        return self.agent.tcp.pose.p

    def _arm_qvel_norm(self) -> torch.Tensor:
        active_names = [j.name for j in self.agent.robot.get_active_joints()]
        indices = [active_names.index(n) for n in self.agent.arm_joint_names]
        return torch.linalg.norm(self.agent.robot.get_qvel()[:, indices], dim=1)

    def _hand_contacts_door(self) -> torch.Tensor:
        links = list(getattr(self.agent, "tip_links", ()))
        if not links:
            links = [
                link for name in ("finger1_link", "finger2_link")
                if (link := getattr(self.agent, name, None)) is not None
            ]
        palm = getattr(self.agent, "palm_link", None)
        if palm is None:
            palm = getattr(self.agent, "tcp", None)
        if palm is not None:
            links.append(palm)
        if not links:
            raise RuntimeError("Selected robot does not expose hand contact links")
        touching = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for link in links:
            force = self.scene.get_pairwise_contact_forces(link, self.door_link)
            touching |= torch.linalg.norm(force, dim=1) > CONTACT_FORCE_THRESHOLD
        return touching

    # -------------------------------------------------------------------------
    # Task interface
    # -------------------------------------------------------------------------

    def evaluate(self):
        door_qpos = self.fridge.get_qpos()[:, 0]
        door_qvel = self.fridge.get_qvel()[:, 0]
        opening_angle = door_qpos - self.door_lower

        handle_position = self.current_handle_position_tensor()
        tcp_position = self._task_tcp_position()
        tcp_to_handle = torch.linalg.norm(
            tcp_position - handle_position,
            dim=1,
        )

        arm_qvel_norm = self._arm_qvel_norm()
        self.hand_contact_latched |= self._hand_contacts_door()

        target_denominator = max(
            self.target_open_qpos - self.door_lower,
            1e-6,
        )
        target_progress = torch.clamp(
            opening_angle / target_denominator,
            0.0,
            1.0,
        )

        arm_static = arm_qvel_norm <= ARM_STATIC_THRESHOLD

        # Pairwise contact remains diagnostic only: physical door travel and
        # a settled terminal state are the success contract.
        door_open_enough, settled_now, self.success_settle_counter = (
            door_settle_update(
                door_qpos,
                door_qvel,
                self.success_settle_counter,
                threshold_qpos=self.success_open_qpos,
                opening=True,
            )
        )
        door_static = torch.abs(door_qvel) <= DOOR_STATIC_THRESHOLD
        success_now = (
            self.success_settle_counter >= SUCCESS_SETTLE_STEPS
        )
        self.success_latched |= success_now

        return {
            "success": self.success_latched,
            "task_success": self.success_latched,
            "door_qpos": door_qpos,
            "door_qvel": door_qvel,
            "opening_angle_rad": opening_angle,
            "opening_angle_deg": torch.rad2deg(opening_angle),
            "target_progress": target_progress,
            "door_open_enough": door_open_enough,
            "door_static": door_static,
            "arm_qvel_norm": arm_qvel_norm,
            "arm_static": arm_static,
            "hand_contact_latched": self.hand_contact_latched,
            "settle_counter": self.success_settle_counter,
            "settle_progress": torch.clamp(
                self.success_settle_counter.float()
                / float(SUCCESS_SETTLE_STEPS),
                0.0,
                1.0,
            ),
            "handle_position": handle_position,
            "tcp_to_handle": tcp_to_handle,
        }

    def get_state_dict(self):
        state = super().get_state_dict()
        state["juice_door_task_state"] = torch.stack(
            [
                self.success_settle_counter.float(),
                self.success_latched.float(),
                self.hand_contact_latched.float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("juice_door_task_state", None)
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
        if task_state.shape[-1] != 3:
            raise ValueError("juice_door_task_state must have 3 fields")
        self.success_settle_counter[restore_idx] = task_state[:, 0].long()
        self.success_latched[restore_idx] = task_state[:, 1].bool()
        self.hand_contact_latched[restore_idx] = task_state[:, 2].bool()

    def _get_obs_extra(self, info: dict):
        return {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "door_qpos": info["door_qpos"].unsqueeze(1),
            "door_qvel": info["door_qvel"].unsqueeze(1),
            "opening_angle_rad": info["opening_angle_rad"].unsqueeze(1),
            "target_progress": info["target_progress"].unsqueeze(1),
            "handle_position": info["handle_position"],
            "tcp_to_handle": info["tcp_to_handle"].unsqueeze(1),
            "settle_progress": info["settle_progress"].unsqueeze(1),
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
        reaching = 1.0 - torch.tanh(5.0 * info["tcp_to_handle"])
        opening = 4.5 * info["target_progress"]

        near_handle_bonus = (
            info["tcp_to_handle"] < 0.055
        ).float() * 0.5
        open_bonus = info["door_open_enough"].float() * 1.0
        settle_bonus = info["settle_progress"] * 1.5


        reward = (
            reaching
            + opening
            + near_handle_bonus
            + 0.5 * info["hand_contact_latched"].float()
            + open_bonus
            + settle_bonus
        )

        return torch.where(
            info["task_success"],
            torch.full_like(reward, SUCCESS_REWARD),
            reward,
        )

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        return (
            self.compute_dense_reward(obs, action, info)
            / SUCCESS_REWARD
        )


# =============================================================================
# Scripted policy
# =============================================================================


class OpenFridgeScriptedPolicy:
    """
    Stable translation-only scripted policy for ``pd_ee_delta_pose``.

    The previous version continuously controlled a full target quaternion and
    required a small rotation error before leaving phase 0. The TCP reached the
    staging position, but the orientation controller kept alternating around an
    unreachable / inconvenient target, so the arm stayed in phase 0 and shook.

    This version follows the same control style as the working chip-insertion
    demo:
      - action[:3] = clip((target_xyz - tcp_xyz) * gain, -1, 1)
      - action[3:6] = 0
      - the Panda starts in a stable top-down configuration
      - phase transitions depend on position and a timeout, never orientation

    The gripper approaches the vertical refrigerator handle from above, closes
    around it, and then follows the handle's hinge-centered XY arc.
    """

    MOVE_GAIN = 8.0
    FINE_GAIN = 6.0
    PULL_GAIN = 5.5

    STAGING_HEIGHT = 0.145
    PREGRASP_HEIGHT = 0.055
    GRASP_HEIGHT = 0.008

    # Keep the TCP slightly outside the door surface to avoid scraping the
    # gripper palm against the door panel.
    OUTSIDE_OFFSET = 0.022

    STAGING_TOL = 0.025
    PREGRASP_TOL = 0.018
    GRASP_TOL = 0.013

    CLOSE_STEPS = 30
    PULL_PLAN_STEPS = 180
    PULL_MAX_STEPS = 260
    RELEASE_STEPS = 24

    def __init__(self, environment: JuicePreparationOpenFridgeEnv):
        self.environment = environment
        self.phase = 0
        self.phase_steps = 0
        self.total_steps = 0
        self.target_position = np.zeros(3, dtype=np.float64)
        self.target_distance = float("inf")

    def _tcp_position(self) -> np.ndarray:
        return _to_numpy(self.environment.agent.tcp.pose.p)[0].astype(
            np.float64
        )

    def _set_target(self, target: np.ndarray):
        self.target_position = np.asarray(
            target,
            dtype=np.float64,
        ).copy()
        self.target_distance = float(
            np.linalg.norm(self.target_position - self._tcp_position())
        )

    def _xyz_action(
        self,
        target: np.ndarray,
        gripper: float,
        gain: float,
        max_abs: float = 1.0,
    ) -> np.ndarray:
        self._set_target(target)

        action = np.zeros(7, dtype=np.float32)
        delta = (
            self.target_position - self._tcp_position()
        ) * float(gain)

        # This is intentionally the same simple per-axis clipping structure as
        # the chip-insertion scripted policy. Rotation actions stay exactly zero.
        action[:3] = np.clip(
            delta,
            -float(max_abs),
            float(max_abs),
        ).astype(np.float32)
        action[3:6] = 0.0
        action[6] = float(gripper)
        return action

    def _handle_tcp_target(
        self,
        door_angle: float,
        height_offset: float = 0.0,
        outside_offset: float | None = None,
    ) -> np.ndarray:
        frame = self.environment.door_frame_at_angle(door_angle)
        if outside_offset is None:
            outside_offset = self.OUTSIDE_OFFSET

        return (
            frame["handle"]
            + frame["front_outward"] * float(outside_offset)
            + np.array([0.0, 0.0, float(height_offset)])
        )

    def _advance(self):
        self.phase += 1
        self.phase_steps = 0

    def _advance_when_reached_or_timeout(
        self,
        tolerance: float,
        minimum_steps: int,
        timeout_steps: int,
    ):
        if (
            self.phase_steps >= minimum_steps
            and self.target_distance <= tolerance
        ) or self.phase_steps >= timeout_steps:
            self._advance()

    def act(self) -> np.ndarray:
        env = self.environment
        current_qpos = env.current_door_qpos

        self.total_steps += 1
        self.phase_steps += 1

        # ------------------------------------------------------------------
        # Phase 0: move above the handle.
        # ------------------------------------------------------------------
        if self.phase == 0:
            target = self._handle_tcp_target(
                env.door_lower,
                height_offset=self.STAGING_HEIGHT,
            )
            action = self._xyz_action(
                target,
                gripper=1.0,
                gain=self.MOVE_GAIN,
                max_abs=0.85,
            )
            self._advance_when_reached_or_timeout(
                tolerance=self.STAGING_TOL,
                minimum_steps=12,
                timeout_steps=90,
            )
            return action

        # ------------------------------------------------------------------
        # Phase 1: descend to a safe pre-grasp point.
        # ------------------------------------------------------------------
        if self.phase == 1:
            target = self._handle_tcp_target(
                env.door_lower,
                height_offset=self.PREGRASP_HEIGHT,
            )
            action = self._xyz_action(
                target,
                gripper=1.0,
                gain=self.FINE_GAIN,
                max_abs=0.65,
            )
            self._advance_when_reached_or_timeout(
                tolerance=self.PREGRASP_TOL,
                minimum_steps=10,
                timeout_steps=70,
            )
            return action

        # ------------------------------------------------------------------
        # Phase 2: descend slowly around the handle.
        # ------------------------------------------------------------------
        if self.phase == 2:
            target = self._handle_tcp_target(
                env.door_lower,
                height_offset=self.GRASP_HEIGHT,
            )
            action = self._xyz_action(
                target,
                gripper=1.0,
                gain=self.FINE_GAIN,
                max_abs=0.38,
            )
            self._advance_when_reached_or_timeout(
                tolerance=self.GRASP_TOL,
                minimum_steps=12,
                timeout_steps=65,
            )
            return action

        # ------------------------------------------------------------------
        # Phase 3: close and hold.
        # ------------------------------------------------------------------
        if self.phase == 3:
            target = self._handle_tcp_target(
                env.door_lower,
                height_offset=self.GRASP_HEIGHT,
            )
            action = self._xyz_action(
                target,
                gripper=-1.0,
                gain=self.FINE_GAIN,
                max_abs=0.30,
            )
            if self.phase_steps >= self.CLOSE_STEPS:
                self._advance()
            return action

        # ------------------------------------------------------------------
        # Phase 4: pull the door along a hinge-centered circular trajectory.
        # ------------------------------------------------------------------
        if self.phase == 4:
            planned_fraction = min(
                1.0,
                self.phase_steps / float(self.PULL_PLAN_STEPS),
            )
            planned_qpos = (
                env.door_lower
                + planned_fraction
                * (env.target_open_qpos - env.door_lower)
            )

            # Do not command the TCP too far ahead of the physically measured
            # door. This keeps contact forces modest if the handle lags.
            desired_qpos = min(
                planned_qpos,
                current_qpos + np.deg2rad(8.0),
            )

            target = self._handle_tcp_target(
                desired_qpos,
                height_offset=self.GRASP_HEIGHT,
            )
            action = self._xyz_action(
                target,
                gripper=-1.0,
                gain=self.PULL_GAIN,
                max_abs=0.55,
            )

            reached_open_angle = (
                current_qpos
                >= env.target_open_qpos - np.deg2rad(4.0)
            )
            if (
                reached_open_angle
                and self.phase_steps >= 70
            ) or self.phase_steps >= self.PULL_MAX_STEPS:
                self._advance()
            return action

        # ------------------------------------------------------------------
        # Phase 5: release while staying close to the current handle pose.
        # ------------------------------------------------------------------
        if self.phase == 5:
            target = self._handle_tcp_target(
                current_qpos,
                height_offset=self.GRASP_HEIGHT,
            )
            action = self._xyz_action(
                target,
                gripper=1.0,
                gain=self.FINE_GAIN,
                max_abs=0.30,
            )
            if self.phase_steps >= self.RELEASE_STEPS:
                self._advance()
            return action

        # ------------------------------------------------------------------
        # Phase 6: retreat upward and outward from the open door.
        # ------------------------------------------------------------------
        current_frame = env.door_frame_at_angle(current_qpos)
        retreat_target = (
            current_frame["handle"]
            + current_frame["front_outward"] * 0.175
            + np.array([0.0, 0.0, 0.105])
        )

        if self.phase == 6:
            action = self._xyz_action(
                retreat_target,
                gripper=1.0,
                gain=self.MOVE_GAIN,
                max_abs=0.75,
            )
            self._advance_when_reached_or_timeout(
                tolerance=0.030,
                minimum_steps=12,
                timeout_steps=90,
            )
            return action

        # Phase 7: hold at the safe retreat point.
        return self._xyz_action(
            retreat_target,
            gripper=1.0,
            gain=self.FINE_GAIN,
            max_abs=0.35,
        )


# =============================================================================
# Standalone runner
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A right-arm mobile-base robot opens refrigerator 10797."
    )
    parser.add_argument(
        "--env-id",
        "--env_id",
        dest="env_id",
        default="JuicePreparationOpenFridge-Sharpa-v1",
    )
    parser.add_argument(
        "--control-mode",
        "--control_mode",
        dest="control_mode",
        default="pd_joint_pos",
        choices=[
            "pd_ee_delta_pose",
            "pd_ee_delta_pos",
            "pd_joint_pos",
            "pd_joint_delta_pos",
        ],
    )
    parser.add_argument(
        "--reward-mode",
        "--reward_mode",
        dest="reward_mode",
        default="dense",
        choices=["dense", "normalized_dense", "sparse"],
    )
    parser.add_argument(
        "--obs-mode",
        "--obs_mode",
        dest="obs_mode",
        default="state",
    )
    parser.add_argument("--fridge-urdf", default=None)
    parser.add_argument(
        "--fridge-target-height",
        type=float,
        default=DEFAULT_FRIDGE_TARGET_HEIGHT,
    )
    parser.add_argument(
        "--handle-target-x",
        type=float,
        default=DEFAULT_HANDLE_TARGET_X,
    )
    parser.add_argument(
        "--handle-target-y",
        type=float,
        default=DEFAULT_HANDLE_TARGET_Y,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--video-dir",
        default="tasks/videos_juice_open_fridge",
    )
    parser.add_argument("--debug-print", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("### CODE_VERSION_MARKER: panda_open_fridge_10797_v2_stable_xyz_scripted ###")

    if args.scripted:
        raise ValueError(
            "The legacy policy only supports the stock Panda gripper action "
            "layout; use teleoperation or a learned policy for these variants."
        )

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        render_mode="rgb_array",
        fridge_urdf=args.fridge_urdf,
        fridge_target_height=args.fridge_target_height,
        handle_target_x=args.handle_target_x,
        handle_target_y=args.handle_target_y,
    )

    if args.record_video:
        env = RecordEpisode(
            env,
            output_dir=args.video_dir,
            save_trajectory=True,
            trajectory_name="trajectory",
            save_video_trigger=lambda episode_index: True,
            max_steps_per_video=args.steps,
            video_fps=CONTROL_FREQUENCY,
        )

    print("Environment:", args.env_id)
    print("Control mode:", args.control_mode)
    print("Reward mode:", args.reward_mode)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    observation, info = env.reset(seed=args.seed)

    policy = None
    if args.scripted:
        policy = OpenFridgeScriptedPolicy(env.unwrapped)

    for step in range(args.steps):
        if policy is not None:
            action = policy.act()
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            if action.shape[-1] >= 1:
                action[-1] = 1.0

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        succeeded = bool(info["task_success"].item())

        if args.debug_print and (
            step % 5 == 0
            or succeeded
        ):
            phase = -1 if policy is None else policy.phase
            policy_target_distance = (
                float("nan")
                if policy is None
                else policy.target_distance
            )
            print(
                f"step={step:04d} "
                f"phase={phase} "
                f"target_err={policy_target_distance:.4f} "
                f"reward={_tensor_scalar(reward):.3f} "
                f"door={_tensor_scalar(info['opening_angle_deg']):.1f}deg "
                f"progress={_tensor_scalar(info['target_progress']):.3f} "
                f"door_qvel={_tensor_scalar(info['door_qvel']):.4f} "
                f"tcp_handle={_tensor_scalar(info['tcp_to_handle']):.4f} "
                f"arm_qvel={_tensor_scalar(info['arm_qvel_norm']):.4f} "
                f"settle={_tensor_scalar(info['settle_progress']):.2f} "
                f"success={int(succeeded)}"
            )

        done = bool(terminated.item()) or bool(truncated.item())
        if succeeded or done:
            break

    env.close()

    if args.record_video:
        print(f"Video/trajectory directory: {args.video_dir}")


if __name__ == "__main__":
    main()

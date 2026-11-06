"""Full-key PartNet keyboard task for ManiSkill/RoboDex.

Every movable key in model 13106 remains an active joint. Each key has a
spring drive whose target is its released limit, so contact can depress any
key and the key returns after contact is removed.

Task success requires typing P, A, I, R, Enter in order. Every key must complete
a press -> hold -> release cycle before the next key is enabled.
All-key qpos, qvel and press fractions remain exposed in state observations.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2mat

# This file lives at <repo>/tasks/office_desk/office_desk_type_keys.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry.trimesh_utils import get_component_meshes, merge_meshes
from tasks.config.hands import register_task
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose, vectorize_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from tasks.config.object_materials import object_material
from tasks.config.workspace_rig import (
    WORK_TABLE_X_OFFSET,
    apply_arm_init_qpos,
    platform_root_pose,
    set_work_table_pose,
)

# ABS/PBT enclosure and keycap plastic.
OFFICE_PLASTIC_DENSITY = 1050.0
from tasks.robots.panda_xhand_hand_left_mobile_base import PandaXHandHandLeftMobileBase  # noqa: F401
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.config.sim_profiles import make_sim_config
from tasks.scene_builders import TableSceneBuilder

from tasks.robots.panda_gripper_left_mobile_base import PandaGripperLeftMobileBase
from tasks.robots.panda_sharpa_hand_left_mobile_base import (
    PandaSharpaHandLeftAngledMobileBase,
    PandaSharpaHandLeftMobileBase,
)
from tasks.robots.panda_sharpa_hand_right_mobile_base import (
    PandaSharpaHandRightAngledMobileBase,
    PandaSharpaHandRightMobileBase,
)
from tasks.robots.panda_wuji_hand_right_mobile_base import (
    PandaWujiHandRightMobileBase,
)
from tasks.robots.panda_wuji_hand_left_mobile_base import (
    PandaWujiHandLeftMobileBase,
)


# -----------------------------------------------------------------------------
# Assets and model mapping
# -----------------------------------------------------------------------------

HAND_ROBOT_UIDS = {
    "panda_sharpa_hand_left_mobile_base",
    "panda_sharpa_hand_left_angled_mobile_base",
    "panda_sharpa_hand_right_mobile_base",
    "panda_sharpa_hand_right_angled_mobile_base",
    "panda_wuji_hand_right_angled_mobile_base",
    "panda_wuji_hand_left_angled_mobile_base",
    "panda_xhand_hand_right_angled_mobile_base",
    "panda_xhand_hand_left_angled_mobile_base",
}
GRIPPER_ROBOT_UID = "panda_gripper_left_mobile_base"
DEFAULT_ROBOT_UID = "panda_sharpa_hand_left_angled_mobile_base"

KEYBOARD_ASSET_DIR = Path(
    os.environ.get(
        "KEYBOARD_ASSET_DIR",
        str(REPO_ROOT / "tasks" / "assets" / "task_object_meshes" / "office"),
    )
)

# Key mapping, corrected. Every entry used to be exactly ONE key to
# the right of the cap that carries its letter, so an operator pressing the
# caps printed P/A/I/R/return pressed five links the task was not watching and
# stage 1 never latched:
#
#     was link_40 -> printed O      now link_39 = P
#     was link_25 -> printed S      now link_24 = A
#     was link_42 -> printed U      now link_41 = I
#     was link_46 -> printed E      now link_45 = R
#     was link_36 -> printed \      now link_35 = return
#
# Derived by reconstructing the board from the caps' own geometry rather than
# by reading a texture: grouping the 77 caps by their row coordinate reproduces
# the US Apple Magic Keyboard exactly (14/14/14/13/12/8/2 keys per row), which
# pins every cap to a legend. Cross-checked against a top-down render: the five
# OLD links read O, S, U, E and \ on screen, 5/5 agreeing with the
# reconstruction. `mobility_multi_key_P_A_I_R_ENTER.urdf` in the asset folder
# was generated from the old ids and is NOT an independent source.
#
# Active-joint list indices differ by one because joint_0 is fixed.
TARGET_KEY_SEQUENCE = (
    ("P", "joint_39", "link_39"),
    ("A", "joint_24", "link_24"),
    ("I", "joint_41", "link_41"),
    ("R", "joint_45", "link_45"),
    ("ENTER", "joint_35", "link_35"),
)
TARGET_KEY_LABELS = tuple(item[0] for item in TARGET_KEY_SEQUENCE)
TARGET_SEQUENCE_LENGTH = len(TARGET_KEY_SEQUENCE)

KEYBOARD_MODELS = {
    "13106": {
        "scale": 0.22,
        "yaw_offset": 0.0,
    },
}
KEYBOARD_MODEL_IDS = tuple(KEYBOARD_MODELS.keys())


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------

CONTROL_FREQ = 20
SUPPORTED_CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")
DEFAULT_CONTROL_MODE = "pd_joint_pos"

# Press hysteresis and 0.2 s hold.
PRESS_ENTER_RATIO = 0.60
PRESS_HOLD_RATIO = 0.50
PRESS_HOLD_STEPS = 4  # 4 / 20 Hz = 0.20 s

# Key must spring back near its released stop before final success.
# 0.30 also covers the model's worst no-contact static compression (Space is
# about 0.226); the five sequence keys themselves settle around 0.034.
RELEASE_RATIO = 0.30
RELEASE_STEPS = 2
KEY_SETTLED_QVEL = 0.020

# The following region is retained as diagnostic observation geometry only.
# It is not part of success: after a valid release the task immediately
# advances to the next key.
FINAL_MIN_HEIGHT = 0.030
FINAL_MAX_HEIGHT = 0.080
FINAL_LONG_MARGIN = 0.010
FINAL_SHORT_MARGIN = 0.025
# Kept as a compatibility export for older combined-task imports.
FINAL_SETTLE_STEPS = 1

# The designated pressing tip must remain over the target-key footprint while
# accumulating hold steps. Margins make this robust to glove/retargeting noise.
PRESS_LONG_MARGIN = 0.008
PRESS_SHORT_MARGIN = 0.012

# Spring parameters applied to every active key joint.
# Real key actuation force is 0.40-0.65 N (Cherry MX 0.45-0.60 N; membrane
# 0.45-0.60 N). This asset's key travel is 1.1 mm, so the spring rate that
# reproduces a real key at full travel is 0.50 N / 1.1 mm = 455 N/m. The
# previous 150 N/m delivered only 0.165 N, i.e. a key 3x lighter than any real
# keyboard -- and 3.6x lighter than the mouse micro-switch in the same family,
# which HAD been calibrated against a real device (Omron D2FC, 0.6 N).
KEY_STIFFNESS = 455.0
KEY_DAMPING = 3.0
KEY_FORCE_LIMIT = 1.2
KEY_JOINT_FRICTION = 0.01

SUCCESS_REWARD = 10.0
LIVING_PENALTY = 0.01

# LOCKED to the shared rig (tasks/config/workspace_rig.py). Two drifts fixed
# here: the trolley stood at -0.615 + WORLD_X_OFFSET = -1.145, i.e. 25 mm
# CLOSER to the desk than the office long task's own -1.170 (the objects sat at
# identical world x in both, so the subtask handed the operator a different
# reach envelope than the long task it is a subset of); and the whole family
# was anchored on the STOCK table centre while CabinetPour/Screwing use the
# +0.30 shifted one. The scene now translates with the table and the trolley.
WORLD_X_OFFSET = -0.53 + WORK_TABLE_X_OFFSET
ROBOT_ROOT_POSE = platform_root_pose()
OFFICE_GROUP_X_OFFSET = 0.20
# SINGLE SOURCE OF TRUTH for the keyboard's world x; office.py imports it.
#
# The laptop base is 218 mm deep and reached back to x = -0.223 while the
# keyboard's back edge sat at -0.152: measured 71 mm of real collision-box
# overlap (reported by the operator). Moving the keyboard the full
# 86 mm needed to clear it would leave its nearest TARGET key 0.544 m from the
# left arm root against a 0.54 m fold-in floor, so the shift is 69 mm and the
# laptop gives the other 17 mm back. Result: 15.6 mm of clearance, and key A at
# 0.560 m.
KEYBOARD_LAPTOP_CLEARANCE_OFFSET = -0.069
KEYBOARD_X = (
    -0.19 + WORLD_X_OFFSET + OFFICE_GROUP_X_OFFSET + KEYBOARD_LAPTOP_CLEARANCE_OFFSET
)
KEYBOARD_SPAWN_CENTER = (KEYBOARD_X, 0.0)
DEFAULT_VIDEO_DIR = "tasks/videos_office_keyboard_pair_enter_sharpa_mobile"


# -----------------------------------------------------------------------------
# URDF and geometry helpers
# -----------------------------------------------------------------------------


def _parse_vector(text: str | None, default) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def _origin_matrix(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    xyz = _parse_vector(
        None if origin is None else origin.attrib.get("xyz"),
        (0.0, 0.0, 0.0),
    )
    rpy = _parse_vector(
        None if origin is None else origin.attrib.get("rpy"),
        (0.0, 0.0, 0.0),
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler2mat(*rpy, axes="sxyz")
    transform[:3, 3] = xyz
    return transform


def _read_obj_vertices(obj_path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) >= 4:
                vertices.append(
                    [float(fields[1]), float(fields[2]), float(fields[3])]
                )
    if not vertices:
        raise RuntimeError(f"No OBJ vertices found in {obj_path}")
    return np.asarray(vertices, dtype=np.float64)


def _normalize_np(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise RuntimeError(f"Cannot normalize near-zero vector {vector}")
    return vector / norm


def load_target_key_geometry(
    source_urdf: Path,
    target_link_name: str,
    target_joint_name: str,
    scale: float,
    press_sign: float,
) -> dict[str, np.ndarray | float]:
    """Extract a stable local frame and dimensions for the target key.

    Returns a top-center point, an outward/up normal, long/short in-plane axes,
    and half extents. PCA is used only to find the key's long direction;
    success checks use absolute projected distances, so PCA sign is
    irrelevant.
    """

    root = ET.parse(source_urdf).getroot()
    link = root.find(f"./link[@name='{target_link_name}']")
    joint = root.find(f"./joint[@name='{target_joint_name}']")
    if link is None or joint is None:
        raise RuntimeError(
            f"Missing {target_link_name} or {target_joint_name} in {source_urdf}"
        )

    visual = link.find("visual")
    if visual is None:
        raise RuntimeError(f"{target_link_name} has no visual geometry")
    mesh = visual.find("./geometry/mesh")
    if mesh is None:
        raise RuntimeError(f"{target_link_name} visual is not a mesh")

    obj_path = (source_urdf.parent / mesh.attrib["filename"]).resolve()
    vertices = _read_obj_vertices(obj_path)
    link_from_mesh = _origin_matrix(visual)
    vertices_h = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)],
        axis=1,
    )
    vertices_link = (link_from_mesh @ vertices_h.T).T[:, :3]

    axis_element = joint.find("axis")
    joint_axis = _parse_vector(
        None if axis_element is None else axis_element.attrib.get("xyz"),
        (1.0, 0.0, 0.0),
    )
    joint_axis = _normalize_np(joint_axis)
    press_axis = joint_axis * (1.0 if press_sign >= 0.0 else -1.0)
    up_axis = -press_axis

    # PCA on the plane perpendicular to the press axis.
    centered = vertices_link - vertices_link.mean(axis=0, keepdims=True)
    planar = centered - np.outer(centered @ up_axis, up_axis)
    covariance = planar.T @ planar / max(len(planar), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    long_axis = long_axis - up_axis * float(long_axis @ up_axis)
    long_axis = _normalize_np(long_axis)
    short_axis = _normalize_np(np.cross(up_axis, long_axis))

    long_coord = vertices_link @ long_axis
    short_coord = vertices_link @ short_axis
    up_coord = vertices_link @ up_axis

    long_min, long_max = float(long_coord.min()), float(long_coord.max())
    short_min, short_max = float(short_coord.min()), float(short_coord.max())
    top_coord = float(up_coord.max())

    top_center = (
        long_axis * (0.5 * (long_min + long_max))
        + short_axis * (0.5 * (short_min + short_max))
        + up_axis * top_coord
    )

    scale = float(scale)
    return {
        "top_center": (top_center * scale).astype(np.float32),
        "up_axis": up_axis.astype(np.float32),
        "joint_axis": joint_axis.astype(np.float32),
        "press_axis": press_axis.astype(np.float32),
        "long_axis": long_axis.astype(np.float32),
        "short_axis": short_axis.astype(np.float32),
        "half_long": float(0.5 * (long_max - long_min) * scale),
        "half_short": float(0.5 * (short_max - short_min) * scale),
    }


def quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` by wxyz quaternions ``q``."""

    q_xyz = q[..., 1:]
    q_w = q[..., :1]
    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return v + 2.0 * (q_w * uv + uuv)


def _link_collision_mesh(link, to_world_frame: bool = True):
    meshes = []
    for obj in link._objs:
        merged = merge_meshes(get_component_meshes(obj))
        if merged is not None:
            meshes.append(merged)
    if not meshes:
        return None
    if to_world_frame:
        pose = link.pose
        for index, mesh in enumerate(meshes):
            if len(pose) > 1:
                mesh.apply_transform(pose[index].sp.to_transformation_matrix())
            else:
                mesh.apply_transform(pose.sp.to_transformation_matrix())
    return merge_meshes(meshes)


def _as_column(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(-1) if value.ndim == 1 else value


def _configure_task_articulation(articulation) -> None:
    """Apply the project object material and task-object energy fences."""
    material = object_material("hard_plastic")
    for link in articulation.get_links():
        for component in link._objs:
            if hasattr(component, "set_max_depenetration_velocity"):
                component.set_max_depenetration_velocity(0.5)
                component.set_max_linear_velocity(3.0)
            if hasattr(component, "get_collision_shapes"):
                for shape in component.get_collision_shapes():
                    shape.set_physical_material(material)


# -----------------------------------------------------------------------------
# Environment registration
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Which link actually presses a key or a button.
#
# The press predicates ask whether the finger is OVER the keycap, by projecting
# ONE link's ORIGIN onto the cap's top face. Using the `*_fingertip` link for
# that carries a systematic bias, because that origin is not where the finger
# touches: measured on the sharpa hand, the pad's contact point is 17.7 mm away
# from the fingertip origin in the cap plane, against a half-accept-box of only
# 20.4 / 21.4 mm. Press a key with the side of the pad and the origin lands
# outside the box while the keycap visibly goes down -- the press was real and
# the predicate refused it.
#
# The elastomer PAD link is the surface that does the pressing, and its own
# origin is 8.4 mm from that contact point, i.e. less than half the bias. Every
# press predicate in this family resolves through here so the long task and its
# subtasks cannot drift apart.
#
# Builds without pads (wuji fingertips, the gripper's tcp) have no such link
# and keep the link they had; nothing changes for them.
#
# NOTE: this is the SUCCESS-predicate link only. `*_all_tip_positions` still
# reports fingertip links, because that is an observation the training schema
# already carries.
# ---------------------------------------------------------------------------
# Which cap to press: a marker, not a letter.
#
# TARGET_KEY_SEQUENCE now points at the caps that really carry P/A/I/R/return
# (see the mapping note there). The marker stays anyway, for two reasons: 77
# blank-looking caps at VR resolution are hard to pick out even when you know
# the letter, and the marker doubles as progress feedback -- it moves to the
# next cap the moment a stage latches, so the operator can see the chain
# advancing. It is visual only -- no collision body -- so it cannot touch the
# physics, and obs_mode=state recordings do not contain it.
KEY_MARKER_RADIUS = 0.006     # m
KEY_MARKER_HEIGHT = 0.010     # m above the cap top
KEY_MARKER_COLOR = (0.10, 0.85, 0.25, 1.0)
KEY_MARKER_PARKED_Z = -5.0    # where it hides once the typing stages are done



# =============================================================================
# Key-press cycle, shared with the long task
# =============================================================================
#
# `office.py` ran its own copy of this six-variable state machine, gated by
# `at_key`, with the same shape and the same imported ratios. Numbers could not
# drift; structure could, and structure is where this family's bugs have been.
# One implementation now drives both, so the standalone typing stage is a strict
# subset of the long task's typing stage by construction.
def key_press_cycle_update(
    press_fraction,
    key_qvel,
    tip_over_press_region,
    state: dict,
    *,
    active=None,
):
    """Advance one key's press -> hold -> release cycle.

    `state` carries the six latches/counters by name:
      press_started, press_achieved, hold_completed,
      press_hold_counter, released_after_hold, release_counter
    and is updated in place. `active` is the long task's stage gate; a
    standalone stage passes None.

    Hysteresis: enter at PRESS_ENTER_RATIO; once entered the key may sag to
    PRESS_HOLD_RATIO, and dropping below that before the hold completes
    restarts the attempt.
    """
    import torch as _torch

    if active is None:
        active = _torch.ones_like(state["press_started"])
    pressed_enter = press_fraction >= PRESS_ENTER_RATIO
    pressed_hold = press_fraction >= PRESS_HOLD_RATIO
    released_now = press_fraction <= RELEASE_RATIO
    key_settled = _torch.abs(key_qvel) <= KEY_SETTLED_QVEL

    valid_enter = active & pressed_enter & tip_over_press_region
    state["press_started"] = state["press_started"] | valid_enter
    restart = active & (~state["hold_completed"]) & (~pressed_hold)
    state["press_started"] = _torch.where(
        restart, _torch.zeros_like(state["press_started"]), state["press_started"]
    )
    state["press_achieved"] = state["press_achieved"] | valid_enter

    valid_hold = (
        active
        & state["press_started"]
        & pressed_hold
        & tip_over_press_region
        & (~state["hold_completed"])
    )
    state["press_hold_counter"] = _torch.where(
        state["hold_completed"],
        state["press_hold_counter"],
        _torch.where(
            valid_hold,
            state["press_hold_counter"] + 1,
            _torch.zeros_like(state["press_hold_counter"]),
        ),
    )
    state["hold_completed"] = state["hold_completed"] | (
        state["press_hold_counter"] >= PRESS_HOLD_STEPS
    )

    valid_release = active & state["hold_completed"] & released_now & key_settled
    state["release_counter"] = _torch.where(
        state["released_after_hold"],
        state["release_counter"],
        _torch.where(
            valid_release,
            state["release_counter"] + 1,
            _torch.zeros_like(state["release_counter"]),
        ),
    )
    state["released_after_hold"] = state["released_after_hold"] | (
        state["release_counter"] >= RELEASE_STEPS
    )
    return state, dict(
        pressed_enter=pressed_enter,
        pressed_hold=pressed_hold,
        released_now=released_now,
        key_settled=key_settled,
    )


def build_key_target_marker(scene, name="key_target_marker"):
    """A visual-only ball that marks the cap the operator must press next."""
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(
        radius=KEY_MARKER_RADIUS,
        material=sapien.render.RenderMaterial(base_color=list(KEY_MARKER_COLOR)),
    )
    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, KEY_MARKER_PARKED_Z])
    return builder.build_kinematic(name=name)


def move_key_target_marker(marker, key_link, park=False):
    """Float the marker over `key_link`'s cap, or park it out of sight."""
    if marker is None:
        return
    if park or key_link is None:
        marker.set_pose(sapien.Pose(p=[0.0, 0.0, KEY_MARKER_PARKED_Z]))
        return
    # Place it over the CAP, computed from the link's own collision geometry.
    # Neither the wrapper pose nor the raw entity pose works: this asset gives
    # every key link the same frame origin (the keyboard root) and carries the
    # per-key offset inside the mesh, so all five keys would report one point.
    raw = key_link._objs[0] if hasattr(key_link, "_objs") else key_link
    pose = raw.entity.pose
    rot = np.asarray(pose.to_transformation_matrix(), dtype=float)[:3, :3]
    origin = np.asarray(pose.p, dtype=float)
    low = high = None
    for shape in raw.get_collision_shapes():
        verts = getattr(shape, "vertices", None)
        if verts is None or len(verts) == 0:
            continue
        verts = np.asarray(verts, dtype=float) * np.asarray(
            getattr(shape, "scale", [1.0, 1.0, 1.0]), dtype=float
        )
        local = shape.get_local_pose()
        rs = np.asarray(local.to_transformation_matrix(), dtype=float)[:3, :3]
        world = (verts @ rs.T + np.asarray(local.p, dtype=float)) @ rot.T + origin
        low = world.min(0) if low is None else np.minimum(low, world.min(0))
        high = world.max(0) if high is None else np.maximum(high, world.max(0))
    if low is None:
        marker.set_pose(sapien.Pose(p=[0.0, 0.0, KEY_MARKER_PARKED_Z]))
        return
    centre = (low + high) / 2.0
    marker.set_pose(
        sapien.Pose(p=[float(centre[0]), float(centre[1]),
                       float(high[2]) + KEY_MARKER_HEIGHT])
    )


PRESS_PAD_SUFFIXES = (("_fingertip", "_elastomer"),)


def press_contact_link(agent, tip_link):
    """The pad link that contacts the key, or `tip_link` if the build has none."""
    name = str(tip_link.name)
    for suffix, pad_suffix in PRESS_PAD_SUFFIXES:
        if not name.endswith(suffix):
            continue
        pad_name = name[: -len(suffix)] + pad_suffix
        for link in agent.robot.get_links():
            if str(link.name) == pad_name:
                return link
    return tip_link


@register_task(
    "OfficeDeskTypeKeys",
    side="left",
    max_episode_steps=700,
    obs_mode="state",
    control_mode="pd_joint_pos",
    reward_mode="sparse",
    render_mode="rgb_array",
    model_id="",
)
class OfficeDeskTypeKeysEnv(BaseEnv):
    """Full-key physical keyboard with a P-A-I-R-Enter success sequence."""

    SUPPORTED_ROBOTS = [
        "panda_gripper_left_mobile_base",
        "panda_sharpa_hand_left_mobile_base",
        "panda_sharpa_hand_left_angled_mobile_base",
        "panda_sharpa_hand_right_mobile_base",
        "panda_sharpa_hand_right_angled_mobile_base",
        "panda_wuji_hand_right_angled_mobile_base",
        "panda_wuji_hand_left_angled_mobile_base",
            "panda_xhand_hand_right_angled_mobile_base",
        "panda_xhand_hand_left_angled_mobile_base",
]
    SUPPORTED_CONTROL_MODES = SUPPORTED_CONTROL_MODES
    SUPPORTED_REWARD_MODES = ("sparse", "none")
    agent: Union[
        PandaGripperLeftMobileBase,
        PandaSharpaHandLeftMobileBase,
        PandaSharpaHandLeftAngledMobileBase,
        PandaSharpaHandRightMobileBase,
        PandaSharpaHandRightAngledMobileBase,
        PandaWujiHandRightMobileBase,
        PandaWujiHandLeftMobileBase,
    ]

    def __init__(
        self,
        *args,
        robot_uids: str = DEFAULT_ROBOT_UID,
        robot_init_qpos_noise: float = 0.01,
        model_id: str = "",
        target_finger_id: int = 1,
        keyboard_random_x: float = 0.020,
        keyboard_random_y: float = 0.020,
        keyboard_random_yaw: float = np.deg2rad(5.0),
        keyboard_clearance: float = 0.002,
        render_width: int = 720,
        render_height: int = 720,
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        runtime_sim_config = kwargs.pop("sim_config", None)
        if runtime_sim_config not in (None, {"control_freq": CONTROL_FREQ}):
            raise ValueError(
                "DexKeyboardType freezes the Plant configuration in tasks/config/"
                "sim_profiles.py; only the teleop compatibility declaration "
                "{'control_freq': 20} is accepted and ignored"
            )

        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.target_finger_id = int(target_finger_id)
        self.keyboard_random_x = float(keyboard_random_x)
        self.keyboard_random_y = float(keyboard_random_y)
        self.keyboard_random_yaw = float(keyboard_random_yaw)
        self.keyboard_clearance = float(keyboard_clearance)
        self.render_width = int(render_width)
        self.render_height = int(render_height)

        if self.target_finger_id < 0:
            raise ValueError("target_finger_id must be non-negative")
        if self.keyboard_random_x < 0 or self.keyboard_random_y < 0:
            raise ValueError("keyboard position randomization must be non-negative")
        if self.keyboard_random_yaw < 0:
            raise ValueError("keyboard_random_yaw must be non-negative")
        if self.keyboard_clearance < 0:
            raise ValueError("keyboard_clearance must be non-negative")
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("render_width and render_height must be positive")
        if model_id and model_id not in KEYBOARD_MODELS:
            raise ValueError(
                f"Unknown keyboard model_id {model_id!r}; "
                f"expected one of {sorted(KEYBOARD_MODELS)}"
            )

        self.available_model_ids = np.asarray(
            [model_id] if model_id else list(KEYBOARD_MODEL_IDS)
        )
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
        return make_sim_config(
            spacing=3,
            max_rigid_contact_count=2**20,
            max_rigid_patch_count=2**19,
        )

    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(
            eye=[0.46 + WORLD_X_OFFSET, -0.46, 0.50],
            target=[0.04 + WORLD_X_OFFSET, 0.0, 0.055],
        )
        external_pose = sapien_utils.look_at(
            eye=[-0.42 + WORLD_X_OFFSET, 0.38, 0.70],
            target=[0.02 + WORLD_X_OFFSET, 0.0, 0.055],
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
            eye=[0.56 + WORLD_X_OFFSET, 0.68, 0.56],
            target=[0.02 + WORLD_X_OFFSET, 0.0, 0.10],
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
        # Visual-only cue for which cap is live (see build_key_target_marker).
        self.key_target_marker = (
            build_key_target_marker(self.scene)
            if self.show_guidance_markers else None
        )
        # Visual only, and kept OUT of the recorded env state: the marker moves
        # to whichever key is the current target, so leaving it in the state
        # dict both pads the vector and leaks the active stage to a
        # state-based policy. drawer_plug's placement disc already did this.
        if self.key_target_marker is not None:
            self.remove_from_state_dict_registry(self.key_target_marker)

        model_ids = self._batched_episode_rng.choice(
            self.available_model_ids,
            replace=True,
        )
        self._model_ids_per_env = list(model_ids)
        self._keyboards = []
        self._target_links = []
        self._target_joint_indices = []
        self._all_key_names = []

        all_released = []
        all_pressed = []
        local_top = []
        local_up = []
        local_joint_axis = []
        local_press = []
        local_long = []
        local_short = []
        half_long = []
        half_short = []

        for scene_index, model_id in enumerate(model_ids):
            cfg = KEYBOARD_MODELS[str(model_id)]
            source_urdf = (
                KEYBOARD_ASSET_DIR
                / str(model_id)
                / "mobility.urdf"
            )
            if not source_urdf.is_file():
                raise FileNotFoundError(
                    f"Keyboard URDF not found: {source_urdf}"
                )

            loader = self.scene.create_urdf_loader()
            loader.name = f"keyboard-{model_id}-{scene_index}"
            loader.fix_root_link = True
            loader.scale = float(cfg["scale"])
            loader.load_multiple_collisions_from_file = True
            # Explicit material density instead of the PhysX default (1000).
            # ABS/PBT keycap and shell plastic is 1040-1070 kg/m3; stating it
            # here keeps every link mass a chosen value rather than a default.
            loader.set_density(OFFICE_PLASTIC_DENSITY)

            builders = loader.parse(
                str(source_urdf)
            )["articulation_builders"]
            if len(builders) != 1:
                raise RuntimeError(
                    f"Expected one articulation for {model_id}, "
                    f"got {len(builders)}"
                )

            builder = builders[0]
            # `builder.disable_self_collisions = True` AFTER loader.parse() is a
            # SILENT NO-OP: the flag is only read when the link builders are
            # CREATED. Verified on this asset -- the key links came out with
            # collision_groups[2] == 0, i.e. key-vs-key collision was still ON.
            # Same mechanism and same fix as office_mouse.py.
            #
            # Measured before fixing: pressing one key moved every other key by
            # 0.00000 mm, so unlike the mouse buttons (which were welded solid)
            # the keycaps never actually touched. This is therefore a latent
            # hazard being closed, not an active defect being repaired -- an
            # asset with tighter key spacing, or a larger contact offset, would
            # have welded exactly the way the mouse did.
            builder.disable_self_collisions = True
            for lb in builder.link_builders:
                lb.collision_groups[2] |= 1 << 29
            builder.set_scene_idxs([scene_index])
            builder.initial_pose = sapien.Pose()
            keyboard = builder.build(
                name=f"keyboard-{model_id}-{scene_index}"
            )
            self.remove_from_state_dict_registry(keyboard)
            _configure_task_articulation(keyboard)

            joints = keyboard.get_active_joints()
            joint_names = [joint.name for joint in joints]
            target_names = [item[1] for item in TARGET_KEY_SEQUENCE]
            missing_targets = [
                name for name in target_names if name not in joint_names
            ]
            if missing_targets:
                raise RuntimeError(
                    f"Target keys {missing_targets} not found; "
                    f"active joints={joint_names}"
                )
            target_indices = [joint_names.index(name) for name in target_names]

            qlimits = (
                keyboard.get_qlimits()
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 2)
            )
            lower = qlimits[:, 0]
            upper = qlimits[:, 1]

            # The unpressed configuration of PartNet keyboards is qpos=0.
            # Therefore the endpoint closest to zero is treated as released;
            # the opposite endpoint is the fully pressed state.
            release_is_lower = np.abs(lower) <= np.abs(upper)
            released = np.where(
                release_is_lower,
                lower,
                upper,
            ).astype(np.float32)
            pressed = np.where(
                release_is_lower,
                upper,
                lower,
            ).astype(np.float32)

            spans = pressed - released
            if np.any(np.abs(spans) <= 1e-8):
                bad = np.flatnonzero(
                    np.abs(spans) <= 1e-8
                ).tolist()
                raise RuntimeError(
                    f"Zero-stroke active keys at indices {bad}"
                )

            for key_index, joint in enumerate(joints):
                joint.set_drive_properties(
                    stiffness=KEY_STIFFNESS,
                    damping=KEY_DAMPING,
                    force_limit=KEY_FORCE_LIMIT,
                    mode="force",
                )
                joint.set_drive_target(
                    float(released[key_index])
                )
                if hasattr(joint, "set_friction"):
                    joint.set_friction(
                        KEY_JOINT_FRICTION
                    )

            geometries = [
                load_target_key_geometry(
                    source_urdf=source_urdf,
                    target_link_name=link_name,
                    target_joint_name=joint_name,
                    scale=cfg["scale"],
                    press_sign=float(np.sign(spans[target_index])),
                )
                for (_, joint_name, link_name), target_index in zip(
                    TARGET_KEY_SEQUENCE,
                    target_indices,
                )
            ]

            self._keyboards.append(keyboard)
            self._target_links.append(
                [keyboard.links_map[item[2]] for item in TARGET_KEY_SEQUENCE]
            )
            self._target_joint_indices.append(target_indices)
            self._all_key_names.append(joint_names)
            all_released.append(released)
            all_pressed.append(pressed)

            local_top.append(np.stack([g["top_center"] for g in geometries]))
            local_up.append(np.stack([g["up_axis"] for g in geometries]))
            local_joint_axis.append(
                np.stack([g["joint_axis"] for g in geometries])
            )
            local_press.append(np.stack([g["press_axis"] for g in geometries]))
            local_long.append(np.stack([g["long_axis"] for g in geometries]))
            local_short.append(np.stack([g["short_axis"] for g in geometries]))
            half_long.append(np.asarray([g["half_long"] for g in geometries]))
            half_short.append(np.asarray([g["half_short"] for g in geometries]))

            print(
                f"[keyboard {scene_index}] model={model_id}, "
                f"active_keys={len(joints)}, "
                f"sequence={'-'.join(TARGET_KEY_LABELS)}, "
                f"indices={target_indices}"
            )

        key_counts = {
            len(values)
            for values in all_released
        }
        if len(key_counts) != 1:
            raise RuntimeError(
                f"All vector environments must have the same "
                f"key count, got {sorted(key_counts)}"
            )

        self.keyboard = Articulation.merge(
            self._keyboards, name="keyboard", merge_links=True
        )
        self.add_to_state_dict_registry(self.keyboard)
        self._all_key_released = common.to_tensor(
            np.stack(all_released),
            device=self.device,
        )
        self._all_key_pressed = common.to_tensor(
            np.stack(all_pressed),
            device=self.device,
        )
        self._target_joint_indices_tensor = torch.as_tensor(
            self._target_joint_indices,
            dtype=torch.long,
            device=self.device,
        )
        env_ids = torch.arange(self.num_envs, device=self.device)[:, None]
        self._target_qpos_released = self._all_key_released[
            env_ids,
            self._target_joint_indices_tensor,
        ]
        self._target_qpos_pressed = self._all_key_pressed[
            env_ids,
            self._target_joint_indices_tensor,
        ]

        def tensor_stack(values):
            return common.to_tensor(
                np.stack(values),
                device=self.device,
            )

        self._target_local_top = tensor_stack(
            local_top
        )
        self._target_local_up = tensor_stack(
            local_up
        )
        self._target_local_joint_axis = tensor_stack(
            local_joint_axis
        )
        self._target_local_press = tensor_stack(
            local_press
        )
        self._target_local_long = tensor_stack(
            local_long
        )
        self._target_local_short = tensor_stack(
            local_short
        )
        self._target_half_long = common.to_tensor(
            np.asarray(
                half_long,
                dtype=np.float32,
            ),
            device=self.device,
        )
        self._target_half_short = common.to_tensor(
            np.asarray(
                half_short,
                dtype=np.float32,
            ),
            device=self.device,
        )

        self.press_started = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.press_achieved = torch.zeros_like(
            self.press_started
        )
        self.hold_completed = torch.zeros_like(
            self.press_started
        )
        self.released_after_hold = torch.zeros_like(
            self.press_started
        )
        self.success_latched = torch.zeros_like(
            self.press_started
        )
        self.sequence_index = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.sequence_completed = torch.zeros(
            (self.num_envs, TARGET_SEQUENCE_LENGTH),
            dtype=torch.bool,
            device=self.device,
        )
        self.press_hold_counter = torch.zeros(
            self.num_envs,
            dtype=torch.int32,
            device=self.device,
        )
        self.release_counter = torch.zeros_like(
            self.press_hold_counter
        )

    def _after_reconfigure(self, options: dict):
        keyboard_zs = []

        for env_id, keyboard in enumerate(
            self._keyboards
        ):
            qpos = keyboard.get_qpos().clone()
            qvel = keyboard.get_qvel().clone()
            qpos[:] = self._all_key_released[env_id]
            qvel.zero_()
            keyboard.set_qpos(qpos)
            keyboard.set_qvel(qvel)
            keyboard.set_pose(sapien.Pose())

            minimum_z = None
            for link in keyboard.get_links():
                mesh = _link_collision_mesh(
                    link,
                    to_world_frame=True,
                )
                if mesh is None or len(
                    mesh.vertices
                ) == 0:
                    continue
                candidate = float(
                    mesh.bounding_box.bounds[0, 2]
                )
                minimum_z = (
                    candidate
                    if minimum_z is None
                    else min(minimum_z, candidate)
                )

            keyboard_zs.append(
                (
                    -minimum_z
                    if minimum_z is not None
                    else 0.0
                )
                + self.keyboard_clearance
            )

        self.keyboard_zs = common.to_tensor(
            np.asarray(
                keyboard_zs,
                dtype=np.float32,
            ),
            device=self.device,
        )

    def _reset_agent(self, env_idx: torch.Tensor) -> None:
        batch_size = len(env_idx)
        if (
            self.robot_uids != GRIPPER_ROBOT_UID
            and self.robot_uids not in HAND_ROBOT_UIDS
        ):
            raise NotImplementedError(self.robot_uids)
        if hasattr(self.agent, "get_reset_pos"):
            initial = self.agent.get_reset_pos(device=self.device)
        else:
            initial = torch.as_tensor(
                self.agent.keyframes["rest"].qpos,
                dtype=torch.float32,
                device=self.device,
            )
        qpos = initial.to(torch.float32).expand(batch_size, -1).clone()
        # Shared teleop start pose (tasks/config/workspace_rig.py): every task
        # must reset the arms into the same posture or the operator's hands
        # map onto a different attitude from the first frame.
        qpos = apply_arm_init_qpos(self.agent, qpos)
        active_names = [
            joint.name for joint in self.agent.robot.get_active_joints()
        ]
        arm_names = list(getattr(self.agent, "arm_joint_names", ()))
        arm_indices = [active_names.index(name) for name in arm_names]
        if self.robot_init_qpos_noise > 0 and arm_indices:
            qpos[:, arm_indices] += common.to_tensor(
                self._episode_rng.normal(
                    0.0,
                    self.robot_init_qpos_noise,
                    (batch_size, len(arm_indices)),
                ),
                device=self.device,
            )
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(ROBOT_ROOT_POSE)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)
            # TableSceneBuilder.initialize restores its own stock pose.
            set_work_table_pose(self)

            xyz = torch.zeros(
                (batch_size, 3),
                dtype=torch.float32,
                device=self.device,
            )
            xyz[:, 0] = (
                KEYBOARD_SPAWN_CENTER[0]
                + (torch.rand(batch_size, device=self.device) * 2.0 - 1.0)
                * self.keyboard_random_x
            )
            xyz[:, 1] = (
                KEYBOARD_SPAWN_CENTER[1]
                + (torch.rand(batch_size, device=self.device) * 2.0 - 1.0)
                * self.keyboard_random_y
            )
            xyz[:, 2] = self.keyboard_zs[env_idx]

            yaw = (
                torch.rand(batch_size, device=self.device) * 2.0 - 1.0
            ) * self.keyboard_random_yaw
            for local_index, env_id in enumerate(env_idx.tolist()):
                yaw[local_index] += float(
                    KEYBOARD_MODELS[self._model_ids_per_env[env_id]]["yaw_offset"]
                )
            quat = torch.zeros(
                (batch_size, 4),
                dtype=torch.float32,
                device=self.device,
            )
            quat[:, 0] = torch.cos(0.5 * yaw)
            quat[:, 3] = torch.sin(0.5 * yaw)

            for local_index, env_id in enumerate(env_idx.tolist()):
                keyboard = self._keyboards[env_id]
                keyboard.set_pose(
                    Pose.create_from_pq(
                        p=xyz[local_index],
                        q=quat[local_index],
                    )
                )
                qpos = keyboard.get_qpos().clone()
                qvel = keyboard.get_qvel().clone()
                qpos[:] = self._all_key_released[env_id]
                qvel.zero_()
                keyboard.set_qpos(qpos)
                keyboard.set_qvel(qvel)

                for key_index, joint in enumerate(
                    keyboard.get_active_joints()
                ):
                    joint.set_drive_target(
                        float(
                            self._all_key_released[
                                env_id,
                                key_index,
                            ].item()
                        )
                    )

            self.press_started[env_idx] = False
            self.press_achieved[env_idx] = False
            self.hold_completed[env_idx] = False
            self.released_after_hold[env_idx] = False
            self.success_latched[env_idx] = False
            self.sequence_index[env_idx] = 0
            self.sequence_completed[env_idx] = False
            self.press_hold_counter[env_idx] = 0
            self.release_counter[env_idx] = 0

            self.keyboard_spawn_xy = getattr(
                self,
                "keyboard_spawn_xy",
                torch.zeros(
                    (self.num_envs, 2),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            self.keyboard_spawn_yaw = getattr(
                self,
                "keyboard_spawn_yaw",
                torch.zeros(
                    (self.num_envs, 1),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            self.keyboard_spawn_xy[env_idx] = xyz[:, :2]
            self.keyboard_spawn_yaw[env_idx, 0] = yaw

            self._reset_agent(env_idx)

    # ------------------------------------------------------------------
    # State quantities
    # ------------------------------------------------------------------

    def _all_key_qpos(self) -> torch.Tensor:
        return torch.stack(
            [
                keyboard.get_qpos()[0]
                for keyboard in self._keyboards
            ],
            dim=0,
        ).to(self.device)

    def _all_key_qvel(self) -> torch.Tensor:
        return torch.stack(
            [
                keyboard.get_qvel()[0]
                for keyboard in self._keyboards
            ],
            dim=0,
        ).to(self.device)

    def _all_key_press_fraction(self) -> torch.Tensor:
        span = (
            self._all_key_pressed
            - self._all_key_released
        )
        fraction = (
            self._all_key_qpos()
            - self._all_key_released
        ) / span
        return torch.clamp(
            fraction,
            0.0,
            1.0,
        )

    def _current_target_slot(self) -> torch.Tensor:
        return torch.clamp(
            self.sequence_index,
            0,
            TARGET_SEQUENCE_LENGTH - 1,
        )

    def _select_target_tensor(self, values: torch.Tensor) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        return values[env_ids, self._current_target_slot()]

    def _space_key_qpos(self) -> torch.Tensor:
        all_qpos = self._all_key_qpos()
        joint_indices = self._select_target_tensor(
            self._target_joint_indices_tensor
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        return all_qpos[env_ids, joint_indices]

    def _space_key_qvel(self) -> torch.Tensor:
        all_qvel = self._all_key_qvel()
        joint_indices = self._select_target_tensor(
            self._target_joint_indices_tensor
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        return all_qvel[env_ids, joint_indices]

    def _space_press_fraction(self) -> torch.Tensor:
        all_fraction = self._all_key_press_fraction()
        joint_indices = self._select_target_tensor(
            self._target_joint_indices_tensor
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        return all_fraction[env_ids, joint_indices]

    def _target_key_frame(self) -> dict[str, torch.Tensor]:
        slots = self._current_target_slot().tolist()
        links = [
            self._target_links[env_id][slot]
            for env_id, slot in enumerate(slots)
        ]
        link_p = torch.stack([link.pose.p[0] for link in links]).to(self.device)
        link_q = torch.stack([link.pose.q[0] for link in links]).to(self.device)
        local_top = self._select_target_tensor(self._target_local_top)
        local_up = self._select_target_tensor(self._target_local_up)
        local_joint_axis = self._select_target_tensor(
            self._target_local_joint_axis
        )
        local_press = self._select_target_tensor(self._target_local_press)
        local_long = self._select_target_tensor(self._target_local_long)
        local_short = self._select_target_tensor(self._target_local_short)

        top = link_p + quat_rotate_wxyz(link_q, local_top)
        up = quat_rotate_wxyz(link_q, local_up)
        joint_axis = quat_rotate_wxyz(
            link_q,
            local_joint_axis,
        )
        press = quat_rotate_wxyz(
            link_q,
            local_press,
        )
        long_axis = quat_rotate_wxyz(link_q, local_long)
        short_axis = quat_rotate_wxyz(link_q, local_short)

        qpos_offset = (
            self._space_key_qpos()
            - self._select_target_tensor(self._target_qpos_released)
        )
        released_top = (
            top
            - joint_axis * qpos_offset.unsqueeze(-1)
        )
        return {
            "top": top,
            "released_top": released_top,
            "up": up,
            "joint_axis": joint_axis,
            "press": press,
            "long": long_axis,
            "short": short_axis,
        }

    def _update_key_target_marker(self) -> None:
        """Float the marker over whichever cap must be pressed next."""
        marker = getattr(self, "key_target_marker", None)
        if marker is None or not self._target_links:
            return
        # sequence_index, not _current_target_slot(): the latter clamps, which
        # would leave the marker sitting on the last cap after the run is done.
        slot = int(self.sequence_index.reshape(-1)[0])
        links = self._target_links[0]
        if 0 <= slot < len(links):
            move_key_target_marker(marker, links[slot])
        else:
            move_key_target_marker(marker, None, park=True)

    def _get_press_tip_link(self):
        if self.robot_uids in HAND_ROBOT_UIDS:
            if not hasattr(self.agent, "tip_links") or len(self.agent.tip_links) == 0:
                raise RuntimeError(f"{self.robot_uids} has no tip_links")
            if self.target_finger_id >= len(self.agent.tip_links):
                raise ValueError(
                    f"target_finger_id={self.target_finger_id} exceeds "
                    f"{self.robot_uids} tip count {len(self.agent.tip_links)}"
                )
            return press_contact_link(
                self.agent, self.agent.tip_links[self.target_finger_id]
            )
        return self.agent.tcp

    def _get_press_tip_position(self) -> torch.Tensor:
        return self._get_press_tip_link().pose.p

    def _get_all_tip_positions(self) -> torch.Tensor:
        if self.robot_uids in HAND_ROBOT_UIDS:
            return torch.stack(
                [vectorize_pose(link.pose)[..., :3] for link in self.agent.tip_links],
                dim=1,
            )
        return self.agent.tcp.pose.p[:, None, :]

    def _arm_qvel_norm(self) -> torch.Tensor:
        qvel = self.agent.robot.get_qvel()
        arm_dof = len(getattr(self.agent, "arm_joint_names", []))
        if arm_dof <= 0:
            arm_dof = min(7, qvel.shape[-1])
        return torch.linalg.norm(qvel[..., :arm_dof], dim=1)

    def _keyboard_root_pose(self) -> torch.Tensor:
        return torch.stack(
            [keyboard.pose.raw_pose[0] for keyboard in self._keyboards],
            dim=0,
        ).to(self.device)

    def _target_link_pose(self) -> torch.Tensor:
        slots = self._current_target_slot().tolist()
        return torch.stack(
            [
                self._target_links[env_id][slot].pose.raw_pose[0]
                for env_id, slot in enumerate(slots)
            ],
            dim=0,
        ).to(self.device)

    def _task_quantities(self) -> dict[str, torch.Tensor]:
        frame = self._target_key_frame()
        tip = self._get_press_tip_position()
        relative_current = tip - frame["top"]
        relative_released = tip - frame["released_top"]

        tip_long = torch.sum(relative_current * frame["long"], dim=1)
        tip_short = torch.sum(relative_current * frame["short"], dim=1)
        height_above_key = torch.sum(relative_current * frame["up"], dim=1)
        height_above_released = torch.sum(
            relative_released * frame["up"],
            dim=1,
        )

        target_half_long = self._select_target_tensor(self._target_half_long)
        target_half_short = self._select_target_tensor(self._target_half_short)
        long_limit_press = target_half_long + PRESS_LONG_MARGIN
        short_limit_press = target_half_short + PRESS_SHORT_MARGIN
        tip_over_press_region = (
            (torch.abs(tip_long) <= long_limit_press)
            & (torch.abs(tip_short) <= short_limit_press)
        )

        long_limit_final = target_half_long + FINAL_LONG_MARGIN
        short_limit_final = target_half_short + FINAL_SHORT_MARGIN
        inside_final_region = (
            (torch.abs(tip_long) <= long_limit_final)
            & (torch.abs(tip_short) <= short_limit_final)
            & (height_above_released >= FINAL_MIN_HEIGHT)
            & (height_above_released <= FINAL_MAX_HEIGHT)
        )

        final_center = frame["released_top"] + frame["up"] * (
            0.5 * (FINAL_MIN_HEIGHT + FINAL_MAX_HEIGHT)
        )
        final_center_dist = torch.linalg.norm(tip - final_center, dim=1)
        tip_to_key_dist = torch.linalg.norm(tip - frame["top"], dim=1)

        return {
            **frame,
            "press_tip_pos": tip,
            "press_tip_to_key": frame["top"] - tip,
            "tip_long": tip_long,
            "tip_short": tip_short,
            "height_above_key": height_above_key,
            "height_above_released": height_above_released,
            "tip_over_press_region": tip_over_press_region,
            "inside_final_region": inside_final_region,
            "final_center": final_center,
            "final_center_dist": final_center_dist,
            "tip_to_key_dist": tip_to_key_dist,
            "arm_qvel_norm": self._arm_qvel_norm(),
        }

    # ------------------------------------------------------------------
    # Success and observations
    # ------------------------------------------------------------------

    def evaluate(self):
        q = self._task_quantities()
        key_qpos = self._space_key_qpos()
        key_qvel = self._space_key_qvel()
        press_fraction = self._space_press_fraction()

        pressed_enter = press_fraction >= PRESS_ENTER_RATIO
        pressed_hold = press_fraction >= PRESS_HOLD_RATIO
        released_now = press_fraction <= RELEASE_RATIO
        key_settled = torch.abs(key_qvel) <= KEY_SETTLED_QVEL

        # One shared implementation with the long task's typing stages.
        _state, _ = key_press_cycle_update(
            press_fraction, key_qvel, q["tip_over_press_region"],
            {
                "press_started": self.press_started,
                "press_achieved": self.press_achieved,
                "hold_completed": self.hold_completed,
                "press_hold_counter": self.press_hold_counter,
                "released_after_hold": self.released_after_hold,
                "release_counter": self.release_counter,
            },
        )
        self.press_started = _state["press_started"]
        self.press_achieved = _state["press_achieved"]
        self.hold_completed = _state["hold_completed"]
        self.press_hold_counter = _state["press_hold_counter"]
        self.released_after_hold = _state["released_after_hold"]
        self.release_counter = _state["release_counter"]

        # A key cycle ends as soon as the key has validly sprung back for the
        # required release steps. No hover-above-old-key or arm-static phase is
        # required; the fingertip may move directly toward the next key.
        cycle_success = self.released_after_hold

        hold_progress = torch.clamp(
            self.press_hold_counter.float() / float(PRESS_HOLD_STEPS),
            0.0,
            1.0,
        )
        release_progress = torch.clamp(
            self.release_counter.float() / float(RELEASE_STEPS),
            0.0,
            1.0,
        )
        # Compatibility field for older observations/recorders. It now mirrors
        # release completion rather than a separate final-hover phase.
        final_settle_progress = release_progress
        release_counter_snapshot = self.release_counter.clone()

        # Advance exactly one key after a complete press/hold/release cycle.
        # Reset per-key state so the next target must complete its own cycle.
        advance_mask = cycle_success & (~self.success_latched)
        env_ids = torch.arange(self.num_envs, device=self.device)
        completed_slots = self._current_target_slot()
        self.sequence_completed[
            env_ids[advance_mask],
            completed_slots[advance_mask],
        ] = True
        self.sequence_index = torch.where(
            advance_mask,
            self.sequence_index + 1,
            self.sequence_index,
        )
        sequence_success = self.sequence_index >= TARGET_SEQUENCE_LENGTH
        self.success_latched |= sequence_success
        self._update_key_target_marker()

        reset_cycle = advance_mask
        self.press_started = torch.where(
            reset_cycle, torch.zeros_like(self.press_started), self.press_started
        )
        self.press_achieved = torch.where(
            reset_cycle, torch.zeros_like(self.press_achieved), self.press_achieved
        )
        self.hold_completed = torch.where(
            reset_cycle, torch.zeros_like(self.hold_completed), self.hold_completed
        )
        self.released_after_hold = torch.where(
            reset_cycle,
            torch.zeros_like(self.released_after_hold),
            self.released_after_hold,
        )
        self.press_hold_counter = torch.where(
            reset_cycle,
            torch.zeros_like(self.press_hold_counter),
            self.press_hold_counter,
        )
        self.release_counter = torch.where(
            reset_cycle,
            torch.zeros_like(self.release_counter),
            self.release_counter,
        )

        task_phase = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        task_phase = torch.where(
            self.press_achieved & (~self.hold_completed),
            torch.ones_like(task_phase),
            task_phase,
        )
        task_phase = torch.where(
            self.hold_completed & (~self.released_after_hold),
            torch.full_like(task_phase, 2),
            task_phase,
        )
        task_phase = torch.where(
            self.success_latched,
            torch.full_like(task_phase, 4),
            task_phase,
        )
        task_phase_one_hot = torch.nn.functional.one_hot(
            task_phase,
            num_classes=5,
        ).float()

        task_success = self.success_latched

        return {
            # Cross-tasks/teleop success conventions.
            "success": task_success,
            "task_success": task_success,
            "is_obj_placed": task_success,
            "cycle_success": cycle_success,
            "sequence_success": sequence_success,
            "sequence_index": self.sequence_index,
            "sequence_progress": torch.clamp(
                self.sequence_index.float() / float(TARGET_SEQUENCE_LENGTH),
                0.0,
                1.0,
            ),
            "sequence_completed": self.sequence_completed,
            "current_target_slot": self._current_target_slot(),
            "current_target_joint_index": self._select_target_tensor(
                self._target_joint_indices_tensor
            ),
            # Sequential task state.
            "task_phase": task_phase,
            "task_phase_one_hot": task_phase_one_hot,
            "press_started": self.press_started,
            "press_achieved": self.press_achieved,
            "hold_completed": self.hold_completed,
            "released_after_hold": self.released_after_hold,
            "press_hold_counter": self.press_hold_counter,
            "release_counter": self.release_counter,
            # Legacy alias: there is no separate final-settle phase anymore.
            "final_settle_counter": release_counter_snapshot,
            "hold_progress": hold_progress,
            "release_progress": release_progress,
            "final_settle_progress": final_settle_progress,
            # Key state.
            "all_key_qpos": self._all_key_qpos(),
            "all_key_qvel": self._all_key_qvel(),
            "all_key_press_fraction": (
                self._all_key_press_fraction()
            ),
            "target_key_qpos": key_qpos,
            "target_key_qvel": key_qvel,
            "target_press_fraction": press_fraction,
            # Legacy aliases retained for existing recorders.
            "space_key_qpos": key_qpos,
            "space_key_qvel": key_qvel,
            "space_press_fraction": press_fraction,
            "pressed_now": pressed_enter,
            "pressed_hold": pressed_hold,
            "released_now": released_now,
            "key_settled": key_settled,
            # Tip/goal geometry.
            "press_tip_pos": q["press_tip_pos"],
            "key_top_pos": q["top"],
            "key_released_top_pos": q["released_top"],
            "key_up_axis": q["up"],
            "key_long_axis": q["long"],
            "key_short_axis": q["short"],
            "key_half_extents": torch.stack(
                [
                    self._select_target_tensor(self._target_half_long),
                    self._select_target_tensor(self._target_half_short),
                ],
                dim=1,
            ),
            "press_tip_to_key": q["press_tip_to_key"],
            "tip_long": q["tip_long"],
            "tip_short": q["tip_short"],
            "height_above_key": q["height_above_key"],
            "height_above_released": q["height_above_released"],
            "tip_over_press_region": q["tip_over_press_region"],
            "inside_final_region": q["inside_final_region"],
            "final_region_center": q["final_center"],
            "final_region_distance": q["final_center_dist"],
            "arm_qvel_norm": q["arm_qvel_norm"],
            # Teleop recorder progress field.
            "obj_to_goal_dist": q["tip_to_key_dist"],
            # Poses and randomization parameters.
            "keyboard_root_pose": self._keyboard_root_pose(),
            "space_key_link_pose": self._target_link_pose(),
            "keyboard_spawn_xy": self.keyboard_spawn_xy,
            "keyboard_spawn_yaw": self.keyboard_spawn_yaw,
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "press_tip_pos": info["press_tip_pos"],
            "keyboard_root_pose": info["keyboard_root_pose"],
            "space_key_link_pose": info["space_key_link_pose"],
            "key_top_pos": info["key_top_pos"],
            "key_released_top_pos": info["key_released_top_pos"],
            "press_tip_to_key": info["press_tip_to_key"],
            "final_region_center": info["final_region_center"],
            "task_phase_one_hot": info["task_phase_one_hot"],
            "press_achieved": _as_column(info["press_achieved"].float()),
            "hold_completed": _as_column(info["hold_completed"].float()),
            "released_after_hold": _as_column(
                info["released_after_hold"].float()
            ),
            "task_success": _as_column(info["task_success"].float()),
            "sequence_index": _as_column(info["sequence_index"].float()),
            "sequence_progress": _as_column(info["sequence_progress"]),
            "sequence_completed": info["sequence_completed"].float(),
        }

        use_state = getattr(
            getattr(self, "obs_mode_struct", None),
            "use_state",
            "state" in str(self.obs_mode),
        )
        if use_state:
            obs.update(
                all_tip_positions=self._get_all_tip_positions().reshape(
                    self.num_envs,
                    -1,
                ),
                target_finger_id=torch.full(
                    (self.num_envs, 1),
                    float(self.target_finger_id),
                    dtype=torch.float32,
                    device=self.device,
                ),
                all_key_qpos=info["all_key_qpos"],
                all_key_qvel=info["all_key_qvel"],
                all_key_press_fraction=info[
                    "all_key_press_fraction"
                ],
                space_key_qpos=_as_column(info["space_key_qpos"]),
                space_key_qvel=_as_column(info["space_key_qvel"]),
                space_press_fraction=_as_column(
                    info["space_press_fraction"]
                ),
                target_key_qpos=_as_column(info["target_key_qpos"]),
                target_key_qvel=_as_column(info["target_key_qvel"]),
                target_press_fraction=_as_column(
                    info["target_press_fraction"]
                ),
                current_target_slot=_as_column(
                    info["current_target_slot"].float()
                ),
                current_target_joint_index=_as_column(
                    info["current_target_joint_index"].float()
                ),
                tip_long=_as_column(info["tip_long"]),
                tip_short=_as_column(info["tip_short"]),
                height_above_key=_as_column(info["height_above_key"]),
                height_above_released=_as_column(
                    info["height_above_released"]
                ),
                key_up_axis=info["key_up_axis"],
                key_long_axis=info["key_long_axis"],
                key_short_axis=info["key_short_axis"],
                key_half_extents=info["key_half_extents"],
                pressed_now=_as_column(info["pressed_now"].float()),
                tip_over_press_region=_as_column(
                    info["tip_over_press_region"].float()
                ),
                inside_final_region=_as_column(
                    info["inside_final_region"].float()
                ),
                hold_progress=_as_column(info["hold_progress"]),
                release_progress=_as_column(info["release_progress"]),
                final_settle_progress=_as_column(
                    info["final_settle_progress"]
                ),
                arm_qvel_norm=_as_column(info["arm_qvel_norm"]),
                keyboard_spawn_xy=info["keyboard_spawn_xy"],
                keyboard_spawn_yaw=info["keyboard_spawn_yaw"],
            )

        for key, value in list(obs.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim == 1:
                obs[key] = value.unsqueeze(-1)
            elif value.ndim > 2:
                obs[key] = value.reshape(
                    value.shape[0],
                    -1,
                )
        return obs

    def get_state_dict(self):
        """Persist physical state plus the sequence and per-key latches."""
        state = super().get_state_dict()
        state["office_keyboard_task_state"] = torch.cat(
            [
                self.press_started[:, None].float(),
                self.press_achieved[:, None].float(),
                self.hold_completed[:, None].float(),
                self.released_after_hold[:, None].float(),
                self.success_latched[:, None].float(),
                self.sequence_index[:, None].float(),
                self.sequence_completed.float(),
                self.press_hold_counter[:, None].float(),
                self.release_counter[:, None].float(),
                self.keyboard_spawn_xy.float(),
                self.keyboard_spawn_yaw.float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("office_keyboard_task_state", None)
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
        task_state = common.to_tensor(task_state, device=self.device).reshape(
            len(restore_idx), -1
        )
        if task_state.shape[-1] != 16:
            raise ValueError(
                f"office_keyboard_task_state must have 16 fields, "
                f"got {task_state.shape[-1]}"
            )
        self.press_started[restore_idx] = task_state[:, 0].bool()
        self.press_achieved[restore_idx] = task_state[:, 1].bool()
        self.hold_completed[restore_idx] = task_state[:, 2].bool()
        self.released_after_hold[restore_idx] = task_state[:, 3].bool()
        self.success_latched[restore_idx] = task_state[:, 4].bool()
        self.sequence_index[restore_idx] = task_state[:, 5].to(torch.long)
        self.sequence_completed[restore_idx] = task_state[:, 6:11].bool()
        self.press_hold_counter[restore_idx] = task_state[:, 11].to(torch.int32)
        self.release_counter[restore_idx] = task_state[:, 12].to(torch.int32)
        self.keyboard_spawn_xy[restore_idx] = task_state[:, 13:15].float()
        self.keyboard_spawn_yaw[restore_idx] = task_state[:, 15:16].float()

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def compute_sparse_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:
        del obs, action
        return info["task_success"].float()

    def compute_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:
        del obs, action

        # Reach the stable pre-press point above the current sequence key.
        prepress_target = info["key_released_top_pos"] + info["key_up_axis"] * 0.04
        prepress_dist = torch.linalg.norm(
            info["press_tip_pos"] - prepress_target,
            dim=1,
        )
        r_reach = 1.0 - torch.tanh(8.0 * prepress_dist)

        # Smooth in-plane alignment to the current key's footprint.
        long_scale = torch.clamp(
            self._select_target_tensor(self._target_half_long)
            + PRESS_LONG_MARGIN,
            min=1e-4,
        )
        short_scale = torch.clamp(
            self._select_target_tensor(self._target_half_short)
            + PRESS_SHORT_MARGIN,
            min=1e-4,
        )
        normalized_plane_error = torch.sqrt(
            (info["tip_long"] / long_scale) ** 2
            + (info["tip_short"] / short_scale) ** 2
        )
        r_align = torch.exp(-2.0 * normalized_plane_error**2)

        before_hold = (~info["hold_completed"]).float()
        r_press = (
            info["space_press_fraction"]
            * info["tip_over_press_region"].float()
            * before_hold
        )
        r_hold = info["hold_progress"]

        after_hold = info["hold_completed"].float()
        r_release = (
            (1.0 - info["space_press_fraction"])
            * after_hold
            * info["key_settled"].float()
        )

        r_key_complete = info["cycle_success"].float()
        r_sequence = info["sequence_progress"]

        # Each completed release advances the sequence immediately. There is
        # deliberately no reward for hovering above the previous key.
        reward = (
            1.0 * r_reach
            + 0.8 * r_reach * r_align
            + 2.0 * r_press
            + 2.0 * r_hold
            + 1.5 * r_release
            + 2.5 * r_key_complete
            + 3.0 * r_sequence
        )
        reward -= LIVING_PENALTY

        reward = torch.where(
            info["task_success"],
            torch.full_like(reward, SUCCESS_REWARD),
            reward,
        )
        info["reward_terms"] = {
            "r_reach": r_reach,
            "r_align": r_align,
            "r_press": r_press,
            "r_hold": r_hold,
            "r_release": r_release,
            "r_key_complete": r_key_complete,
            "r_sequence": r_sequence,
            "prepress_dist": prepress_dist,
            "normalized_plane_error": normalized_plane_error,
            "press_fraction": info["space_press_fraction"],
            "hold_progress": info["hold_progress"],
            "release_progress": info["release_progress"],
            "dense_reward_no_success_bonus": reward.clone(),
        }
        return reward

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info) / SUCCESS_REWARD


# -----------------------------------------------------------------------------
# Standalone test runner
# -----------------------------------------------------------------------------


def tensor_mean(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(np.asarray(value).mean())


if __name__ == "__main__":
    print(
        "### CODE_VERSION_MARKER: office_keyboard "
        "(P-A-I-R-Enter + mobile base) ###"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        "--env_id",
        dest="env_id",
        default="OfficeDeskTypeKeys-Sharpa-v1",
    )
    parser.add_argument(
        "--control-mode",
        "--control_mode",
        dest="control_mode",
        default="pd_joint_pos",
    )
    parser.add_argument(
        "--reward-mode",
        "--reward_mode",
        dest="reward_mode",
        choices=("sparse", "none"),
        default="sparse",
    )
    parser.add_argument(
        "--debug-print",
        "--debug_print",
        dest="debug_print",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--target-finger-id", type=int, default=1)
    parser.add_argument("--keyboard-random-x", type=float, default=0.020)
    parser.add_argument("--keyboard-random-y", type=float, default=0.020)
    parser.add_argument(
        "--keyboard-random-yaw-deg",
        type=float,
        default=5.0,
    )
    args = parser.parse_args()

    control_mode = args.control_mode

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        reward_mode=args.reward_mode,
        control_mode=control_mode,
        render_mode="rgb_array",
        target_finger_id=args.target_finger_id,
        keyboard_random_x=args.keyboard_random_x,
        keyboard_random_y=args.keyboard_random_y,
        keyboard_random_yaw=np.deg2rad(args.keyboard_random_yaw_deg),
    )

    max_steps = args.max_steps
    if max_steps is None:
        max_steps = getattr(env.unwrapped, "max_episode_steps", None)
    if max_steps is None and getattr(env, "spec", None) is not None:
        max_steps = env.spec.max_episode_steps
    max_steps = int(max_steps or 600)

    env = RecordEpisode(
        env,
        output_dir=args.video_dir,
        save_trajectory=True,
        trajectory_name="trajectory",
        save_video_trigger=lambda episode_index: True,
        max_steps_per_video=max_steps,
        video_fps=10,
    )
    print(f"Recording trajectory/video to {args.video_dir}")
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Robot DoF", env.unwrapped.agent.robot.dof)

    obs, _ = env.reset(seed=args.seed)
    if control_mode == "pd_joint_pos":
        hold_action = env.unwrapped.agent.robot.get_qpos().detach().clone()
    elif "delta" in control_mode:
        hold_action = torch.zeros(
            (1, int(env.action_space.shape[-1])),
            device=env.unwrapped.device,
        )
    else:
        raise RuntimeError(
            "Stationary test supports pd_joint_pos or a delta controller."
        )

    if hold_action.shape[-1] != env.action_space.shape[-1]:
        raise RuntimeError(
            "Action dimension mismatch: "
            f"hold={hold_action.shape[-1]}, "
            f"action={env.action_space.shape[-1]}"
        )

    done = False
    step = 0
    while not done and step < max_steps:
        obs, reward, terminated, truncated, info = env.step(hold_action)

        if args.debug_print and step % 2 == 0:
            print(
                f"step={step:04d} reward={tensor_mean(reward):.3f} "
                f"key={TARGET_KEY_LABELS[int(info['current_target_slot'][0].item())]} "
                f"sequence={int(info['sequence_index'][0].item())}/"
                f"{TARGET_SEQUENCE_LENGTH} "
                f"qpos={tensor_mean(info['target_key_qpos']):.7f} "
                f"frac={tensor_mean(info['target_press_fraction']):.3f} "
                f"press={tensor_mean(info['pressed_now']):.0f} "
                f"hold={tensor_mean(info['hold_progress']):.2f} "
                f"hold_done={tensor_mean(info['hold_completed']):.0f} "
                f"released={tensor_mean(info['released_after_hold']):.0f} "
                f"release_progress={tensor_mean(info['release_progress']):.2f} "
                f"key_complete={tensor_mean(info['cycle_success']):.0f} "
                f"success={tensor_mean(info['task_success']):.0f} "
                "mode=static"
            )

        done = bool(terminated) or bool(truncated)
        env.render()
        step += 1

    env.close()

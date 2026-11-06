"""Bimanual long-horizon office task on dual-Panda mobile bases.

Required order and hand assignment: the right hand opens the laptop, the left
hand types P-A-I-R-Enter, and the right hand clicks the mouse's left button.
The laptop, full 77-key keyboard, and mouse share one desk. Each stage is
latched and can advance only in this order. The laptop uses the same stable
simplified collision shell as office_laptop.py, keys use complete
press/hold/release cycles, and the mouse rests directly on the desk.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np
import sapien
import sapien.render
import torch
import xml.etree.ElementTree as ET

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from tasks.config.hands import register_task
from mani_skill.utils.wrappers.record import RecordEpisode
from tasks.robots.panda_dual_xhand_hand_mobile_base import PandaDualXHandHandMobileBase  # noqa: F401
from tasks.config.layout_randomization import arm_qpos_noise, episode_rng, uniform_offset, yaw_quat_wxyz, layout_rng
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.config.sim_profiles import make_sim_config
from tasks.config.workspace_rig import (
    WORK_TABLE_X_OFFSET,
    apply_arm_init_qpos,
    platform_root_pose,
    set_work_table_pose,
)
from tasks.scene_builders import TableSceneBuilder

# Import the dual-arm agents so their UIDs are registered before gym.make.
from tasks.robots.panda_dual_gripper_mobile_base import PandaDualGripperMobileBase
from tasks.robots.panda_dual_sharpa_hand_mobile_base import (
    PandaDualSharpaHandAngledMobileBase,
)
from tasks.robots.panda_dual_wuji_hand_mobile_base import PandaDualWujiHandMobileBase

from tasks.office_desk.office_desk_open_laptop import (
    LAPTOP_ASSET_DIR,
    lid_open_hold_update,
    LAPTOP_INWARD_OFFSET,
    LAPTOP_X,
    LAPTOP_MODELS,
    MODEL_ID as LAPTOP_MODEL_ID,
    _link_collision_mesh as _laptop_link_collision_mesh,
    TARGET_HOLD_STEPS as LAPTOP_TARGET_HOLD_STEPS,
    _configure_task_articulation,
    apply_laptop_masses,
    _screen_edge_from_local_mesh,
    _use_simplified_laptop_collisions,
)
from tasks.office_desk.office_desk_type_keys import (
    OFFICE_PLASTIC_DENSITY,
    KEYBOARD_LAPTOP_CLEARANCE_OFFSET,
    KEYBOARD_X,
    KEYBOARD_ASSET_DIR,
    KEYBOARD_MODELS,
    KEY_STIFFNESS,
    KEY_DAMPING,
    KEY_FORCE_LIMIT,
    KEY_JOINT_FRICTION,
    PRESS_ENTER_RATIO as KEY_PRESS_ENTER_RATIO,
    PRESS_HOLD_RATIO as KEY_PRESS_HOLD_RATIO,
    PRESS_HOLD_STEPS as KEY_PRESS_HOLD_STEPS,
    RELEASE_RATIO as KEY_RELEASE_RATIO,
    RELEASE_STEPS as KEY_RELEASE_STEPS,
    PRESS_LONG_MARGIN as KEY_PRESS_LONG_MARGIN,
    PRESS_SHORT_MARGIN as KEY_PRESS_SHORT_MARGIN,
    KEY_SETTLED_QVEL,
    TARGET_KEY_SEQUENCE,
    key_press_cycle_update,
    build_key_target_marker,
    load_target_key_geometry,
    move_key_target_marker,
    press_contact_link,
    quat_rotate_wxyz,
)
from tasks.office_desk.office_desk_click_mouse import (
    MOUSE_ASSET_DIR,
    mouse_click_cycle_update,
    MOUSE_X,
    MOUSE_Y,
    MOUSE_MODEL,
    PRESS_ENTER_RATIO as MOUSE_PRESS_ENTER_RATIO,
    PRESS_LONG_MARGIN as MOUSE_PRESS_LONG_MARGIN,
    PRESS_SHORT_MARGIN as MOUSE_PRESS_SHORT_MARGIN,
    RIGHT_BUTTON_MAX_RATIO,
    SUCCESS_PRESS_STEPS as MOUSE_SUCCESS_PRESS_STEPS,
    load_button_geometry,
)

CONTROL_FREQUENCY = 20
MAX_EPISODE_STEPS = 1400
SUPPORTED_CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")
DEFAULT_CONTROL_MODE = "pd_joint_pos"
# LOCKED to the shared rig (tasks/config/workspace_rig.py). The desk scene was
# authored against the STOCK table centre (-0.120) with the trolley at -1.170;
# CabinetPour/Screwing put the table at +0.180 and the trolley at -0.870. The
# whole office scene is therefore translated by +WORK_TABLE_X_OFFSET together
# with the table and the trolley -- a rigid translation, so every reach
# distance, threshold and camera framing is unchanged, but the operator now
# stands in the SAME place against the table as in every other family.
WORLD_X_OFFSET = -0.53 + WORK_TABLE_X_OFFSET
ROBOT_ROOT_POSE = platform_root_pose()
OFFICE_GROUP_X_OFFSET = 0.20

ROBOT_UID_GRIPPER = "panda_dual_gripper_mobile_base"
ROBOT_UID_SHARPA = "panda_dual_sharpa_hand_angled_mobile_base"
ROBOT_UID_WUJI = "panda_dual_wuji_hand_angled_mobile_base"
ROBOT_UID_XHAND = "panda_dual_xhand_hand_angled_mobile_base"
ROBOT_UIDS = (
    ROBOT_UID_GRIPPER,
    ROBOT_UID_SHARPA,
    ROBOT_UID_WUJI,
    ROBOT_UID_XHAND,
)

# Object x/y now come from the three sub-task modules, which own them: see
# office_laptop.LAPTOP_X, office_keyboard.KEYBOARD_X and office_mouse.MOUSE_X.
# They used to be duplicated here and kept in step by hand, which silently
# desynced the moment the long task's layout was adjusted.
#
# The keyboard width (Q..P, ~35 cm) lies along world Y. In this scene,
# decreasing X moves objects towards the operator, while +Y is the right-hand
# side in the task camera. The robot's right arm is mounted at -Y, so the mouse
# at negative Y remains in its natural workspace. Keyboard in front, laptop
# behind it.

TARGET_KEYS = tuple(item[1] for item in TARGET_KEY_SEQUENCE)
TARGET_KEY_LINKS = tuple(item[2] for item in TARGET_KEY_SEQUENCE)
TARGET_KEY_LABELS = tuple(item[0] for item in TARGET_KEY_SEQUENCE)

STAGE_OPEN_LAPTOP = 0
STAGE_TYPE_FIRST = 1
STAGE_CLICK_MOUSE = STAGE_TYPE_FIRST + len(TARGET_KEYS)
STAGE_DONE = STAGE_CLICK_MOUSE + 1
STAGE_NAMES = (
    "open_laptop",
    *(f"type_{label.lower()}" for label in TARGET_KEY_LABELS),
    "click_mouse",
    "done",
)


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a vector around an axis using Rodrigues' formula."""
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    c, s = math.cos(angle), math.sin(angle)
    return vector * c + np.cross(axis, vector) * s + axis * float(np.dot(axis, vector)) * (1.0 - c)


def _origin_matrix(origin_el) -> np.ndarray:
    matrix = np.eye(4)
    if origin_el is None:
        return matrix
    xyz = np.array([float(v) for v in origin_el.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in origin_el.get("rpy", "0 0 0").split()])
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    matrix[:3, :3] = rz @ ry @ rx
    matrix[:3, 3] = xyz
    return matrix


def extract_laptop_hinge_geometry(urdf_path: Path, scale: float) -> dict[str, np.ndarray | float]:
    """Hinge axis/pivot + a pushable screen point, all in the laptop root frame.

    See module docstring: cross-validated against office_laptop.py's own live
    screen_edge_local at scale=0.3 (agreed within 0.4 mm).
    """
    root = ET.parse(urdf_path).getroot()
    joint0 = root.find("./joint[@name='joint_0']")
    joint1 = root.find("./joint[@name='joint_1']")
    base_to_link0 = _origin_matrix(joint0.find("origin"))
    link0_to_link1 = _origin_matrix(joint1.find("origin"))
    axis1 = np.array([float(v) for v in joint1.find("axis").get("xyz").split()])
    limit = joint1.find("limit")
    qpos_lower = float(limit.get("lower"))

    hinge_axis_base = base_to_link0[:3, :3] @ axis1
    hinge_pivot_base = base_to_link0[:3, :3] @ link0_to_link1[:3, 3] + base_to_link0[:3, 3]

    link1 = root.find("./link[@name='link_1']")
    verts = []
    for visual in link1.findall("visual"):
        transform = _origin_matrix(visual.find("origin"))
        mesh_path = urdf_path.parent / visual.find("geometry/mesh").get("filename")
        import trimesh

        mesh = trimesh.load(mesh_path, process=False, force="mesh")
        v = np.asarray(mesh.vertices, dtype=np.float64)
        vh = np.concatenate([v, np.ones((len(v), 1))], axis=1)
        verts.append((transform @ vh.T).T[:, :3])
    verts = np.concatenate(verts, axis=0)
    lower, upper = verts.min(axis=0), verts.max(axis=0)
    extent = upper - lower
    thin_axis = int(np.argmin(extent))
    width_axis = int(np.argmax(extent))
    height_axis = [i for i in range(3) if i not in (thin_axis, width_axis)][0]

    edge_local = 0.5 * (lower + upper)
    edge_local[height_axis] = upper[height_axis]  # free (far-from-hinge) edge
    edge_h = np.append(edge_local, 1.0)
    edge_at_qpos0_base = (base_to_link0 @ link0_to_link1 @ edge_h)[:3]

    return {
        "hinge_axis_base": hinge_axis_base * scale / max(float(np.linalg.norm(hinge_axis_base)), 1e-12),
        "hinge_pivot_base": hinge_pivot_base * scale,
        "edge_at_qpos0_base": edge_at_qpos0_base * scale,
        "qpos_lower": qpos_lower,
    }


def screen_push_point_at_qpos(geometry: dict, qpos: float) -> np.ndarray:
    axis = geometry["hinge_axis_base"]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    pivot = geometry["hinge_pivot_base"]
    radius0 = geometry["edge_at_qpos0_base"] - pivot
    return pivot + _rotate_about_axis(radius0, axis, qpos)


def _obj_name(x) -> str:
    n = getattr(x, "name", None)
    if n is None and hasattr(x, "get_name"):
        n = x.get_name()
    return "" if n is None else str(n)


def _scalar(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().reshape(-1)[0].cpu())
    return float(np.asarray(x).reshape(-1)[0])


def _boolean(x) -> bool:
    if torch.is_tensor(x):
        return bool(x.detach().reshape(-1)[0].cpu())
    return bool(np.asarray(x).reshape(-1)[0])


# Per-episode layout half-ranges for the long task (m, m, deg) -- the subtasks' defaults:
# office_laptop laptop_xy_noise 0.02 / laptop_yaw_noise_deg 5, office_keyboard keyboard_random_x/y 0.02 /
# keyboard_random_yaw 5 deg, office_mouse mouse_xy_noise 0.02 / mouse_yaw_noise_deg 5.
# In the long task all three are FIXED-ROOT articulations; the recording contract restores
# raw states bit for bit and PhysX's fixed-root pose write drifts by one float32 ulp for a
# non-identity quaternion, so they get POSITION-only randomization here.
LONG_LAPTOP_RANDOM = (0.020, 0.020, 0.0)
# Keyboard x is capped at 10 mm here: in the long task the ENTER key sits 0.557 m from
# the nearest arm root against the 0.54 m reach floor,
# and every mm of keyboard x moves the key by a mm; 20 mm broke it.
LONG_KEYBOARD_RANDOM = (0.010, 0.020, 0.0)
LONG_MOUSE_RANDOM = (0.020, 0.020, 0.0)


@register_task(
    "OfficeDeskLong",
    side="both",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode="pd_joint_pos",
    reward_mode="sparse",
    render_mode="rgb_array",
)
class OfficeDeskLongEnv(BaseEnv):
    """Right opens, left types, then right clicks in one bimanual scene."""

    SUPPORTED_ROBOTS = list(ROBOT_UIDS)
    agent: Any
    SUPPORTED_CONTROL_MODES = SUPPORTED_CONTROL_MODES
    SUPPORTED_REWARD_MODES = ("sparse", "none")

    def __init__(
        self,
        *args,
        robot_uids: str = ROBOT_UID_SHARPA,
        robot_init_qpos_noise: float = 0.01,
        target_finger_id: int = 1,
        render_width: int = 720,
        render_height: int = 720,
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        runtime_sim_config = kwargs.pop("sim_config", None)
        if runtime_sim_config not in (None, {"control_freq": CONTROL_FREQUENCY}):
            raise ValueError(
                "OfficeDeskSequence freezes the Plant configuration in tasks/config/"
                "sim_profiles.py; only the teleop compatibility declaration "
                "{'control_freq': 20} is accepted and ignored"
            )
        if num_envs != 1:
            raise ValueError("OfficeDeskLongEnv only supports num_envs=1")
        if robot_uids not in ROBOT_UIDS:
            raise ValueError(
                f"Unsupported office robot {robot_uids!r}; expected one of "
                f"{sorted(ROBOT_UIDS)}"
            )
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.target_finger_id = int(target_finger_id)
        self.render_width = int(render_width)
        self.render_height = int(render_height)

        if self.target_finger_id < 0:
            raise ValueError("target_finger_id must be non-negative")
        if self.robot_init_qpos_noise < 0:
            raise ValueError("robot_init_qpos_noise must be non-negative")
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("render_width and render_height must be positive")

        if reconfiguration_freq is None:
            reconfiguration_freq = 1
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
            max_rigid_contact_count=2**21,
            max_rigid_patch_count=2**20,
        )

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                "base_camera",
                sapien_utils.look_at(
                    eye=[0.55 + WORLD_X_OFFSET, -0.55, 0.62],
                    target=[0.10 + WORLD_X_OFFSET, 0.10, 0.10],
                ),
                320, 240, np.deg2rad(62.0), 0.01, 100.0,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            sapien_utils.look_at(
                eye=[0.65 + WORLD_X_OFFSET, -0.60, 0.68],
                target=[0.10 + WORLD_X_OFFSET, 0.10, 0.10],
            ),
            self.render_width, self.render_height, np.deg2rad(58.0), 0.01, 100.0,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, ROBOT_ROOT_POSE)

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        set_work_table_pose(self)
        # Visual-only cue for which cap is live (see office_keyboard.py).
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
        self._load_laptop()
        self._load_keyboard()
        self._load_mouse()

    def _load_laptop(self):
        cfg = LAPTOP_MODELS[LAPTOP_MODEL_ID]
        urdf_path = LAPTOP_ASSET_DIR / LAPTOP_MODEL_ID / "mobility.urdf"
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Laptop URDF not found: {urdf_path}")

        loader = self.scene.create_urdf_loader()
        loader.name = "laptop-0"
        loader.fix_root_link = True
        loader.scale = float(cfg["scale"])
        loader.load_multiple_collisions_from_file = True
        loader.multiple_collisions_decomposition = "none"
        builders = loader.parse(str(urdf_path))["articulation_builders"]
        if not builders:
            raise RuntimeError(f"No articulation parsed from {urdf_path}")
        builder = builders[0]
        _use_simplified_laptop_collisions(builder, float(cfg["scale"]), urdf_path)
        builder.set_scene_idxs([0])
        builder.initial_pose = sapien.Pose(p=[LAPTOP_X, 0.0, 0.0])
        self.laptop = builder.build(name="laptop-0")
        self.remove_from_state_dict_registry(self.laptop)
        _configure_task_articulation(self.laptop)
        apply_laptop_masses(self.laptop)
        self.add_to_state_dict_registry(self.laptop)

        active_joints = self.laptop.get_active_joints()
        hinge_index = int(cfg["hinge_joint_index"])
        active_joints[hinge_index].set_drive_properties(
            stiffness=0.0,
            damping=float(cfg["hinge_damping"]),
            force_limit=float(cfg["hinge_force_limit"]),
            mode="force",
        )
        active_joints[hinge_index].set_friction(float(cfg["hinge_friction"]))
        links = self.laptop.get_links()
        self.laptop_hinge_joint = active_joints[hinge_index]
        self.laptop_screen_link = links[-1]
        self.laptop_cfg = cfg
        self.laptop_urdf_path = urdf_path

    def _load_keyboard(self):
        cfg = KEYBOARD_MODELS["13106"]
        urdf_path = KEYBOARD_ASSET_DIR / "13106" / "mobility.urdf"
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Keyboard URDF not found: {urdf_path}")

        loader = self.scene.create_urdf_loader()
        loader.name = "keyboard-0"
        loader.fix_root_link = True
        loader.scale = float(cfg["scale"])
        loader.load_multiple_collisions_from_file = True
        # Explicit ABS/PBT density instead of the PhysX default (1000).
        loader.set_density(OFFICE_PLASTIC_DENSITY)
        builders = loader.parse(str(urdf_path))["articulation_builders"]
        if len(builders) != 1:
            raise RuntimeError(f"Expected one articulation for keyboard, got {len(builders)}")
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
        builder.set_scene_idxs([0])
        builder.initial_pose = sapien.Pose(p=[KEYBOARD_X, 0.0, 0.0])
        self.keyboard = builder.build(name="keyboard-0")
        self.remove_from_state_dict_registry(self.keyboard)
        _configure_task_articulation(self.keyboard)
        self.add_to_state_dict_registry(self.keyboard)

        joints = self.keyboard.get_active_joints()
        joint_names = [j.name for j in joints]
        qlimits = self.keyboard.get_qlimits().detach().cpu().numpy().reshape(-1, 2)
        # The released endpoint is the limit closest to zero; this remains
        # correct if a key's URDF reverses its joint-axis sign.
        release_is_lower = np.abs(qlimits[:, 0]) <= np.abs(qlimits[:, 1])
        released = np.where(release_is_lower, qlimits[:, 0], qlimits[:, 1]).astype(np.float32)
        pressed = np.where(release_is_lower, qlimits[:, 1], qlimits[:, 0]).astype(np.float32)
        spans = pressed - released
        if np.any(np.abs(spans) <= 1e-8):
            raise RuntimeError("Zero-stroke active keys found on keyboard")

        for key_index, joint in enumerate(joints):
            joint.set_drive_properties(
                stiffness=KEY_STIFFNESS, damping=KEY_DAMPING, force_limit=KEY_FORCE_LIMIT, mode="force",
            )
            if hasattr(joint, "set_friction"):
                joint.set_friction(KEY_JOINT_FRICTION)

        self.keyboard_urdf_path = urdf_path
        self.keyboard_cfg = cfg
        self.keyboard_joint_names = joint_names
        self.keyboard_released = released
        self.keyboard_pressed = pressed
        self.target_key_links = [self.keyboard.links_map[name] for name in TARGET_KEY_LINKS]
        self.target_key_indices = [joint_names.index(jn) for jn in TARGET_KEYS]

    def _load_mouse(self):
        cfg = MOUSE_MODEL
        urdf_path = MOUSE_ASSET_DIR / cfg["model_id"] / "mobility.urdf"
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Mouse URDF not found: {urdf_path}")

        loader = self.scene.create_urdf_loader()
        loader.name = "mouse-0"
        loader.fix_root_link = True
        loader.scale = float(cfg["scale"])
        loader.load_multiple_collisions_from_file = True
        # Explicit ABS/PBT density instead of the PhysX default (1000).
        loader.set_density(OFFICE_PLASTIC_DENSITY)
        loader.multiple_collisions_decomposition = "coacd"
        builders = loader.parse(str(urdf_path))["articulation_builders"]
        if not builders:
            raise RuntimeError(f"No articulation parsed from {urdf_path}")
        builder = builders[0]
        builder.disable_self_collisions = True
        for lb in builder.link_builders:
            lb.collision_groups[2] |= 1 << 29
        builder.set_scene_idxs([0])
        yaw = float(cfg.get("yaw_offset", 0.0))
        mouse_q = [float(np.cos(0.5 * yaw)), 0.0, 0.0, float(np.sin(0.5 * yaw))]
        builder.initial_pose = sapien.Pose(p=[MOUSE_X, MOUSE_Y, 0.0], q=mouse_q)
        self.mouse = builder.build(name="mouse-0")
        self.remove_from_state_dict_registry(self.mouse)
        _configure_task_articulation(self.mouse)
        self.add_to_state_dict_registry(self.mouse)

        active_joints = self.mouse.get_active_joints()
        left_index = int(cfg["left_joint_index"])
        right_index = int(cfg["right_joint_index"])
        qlimits = self.mouse.get_qlimits().detach().cpu().numpy().reshape(-1, 2)
        press_sign = float(np.sign(cfg["source_pressed"] - cfg["source_released"]))
        if press_sign >= 0:
            left_release, left_press = qlimits[left_index]
            right_release, right_press = qlimits[right_index]
        else:
            left_press, left_release = qlimits[left_index]
            right_press, right_release = qlimits[right_index]

        for index, release in ((left_index, left_release), (right_index, right_release)):
            joint = active_joints[index]
            joint.set_drive_properties(
                stiffness=float(cfg["button_stiffness"]), damping=float(cfg["button_damping"]),
                force_limit=float(cfg["button_force_limit"]), mode="force",
            )
            joint.set_friction(float(cfg["button_friction"]))

        # Scroll wheel. office_mouse.py configures it too; the long task keeps
        # its own copy of the mouse wiring, so the same explicit values have to
        # be stated here or the wheel falls back to zero damping and FLT_MAX.
        for index, joint in enumerate(active_joints):
            if index in (left_index, right_index):
                continue
            joint.set_drive_properties(
                stiffness=0.0, damping=float(cfg["wheel_damping"]),
                force_limit=float(cfg["wheel_force_limit"]), mode="force",
            )
            joint.set_friction(float(cfg["wheel_friction"]))

        self.mouse_urdf_path = urdf_path
        self.mouse_cfg = cfg
        self.mouse_left_link = self.mouse.links_map[cfg["left_link_name"]]
        self.mouse_left_index = left_index
        self.mouse_right_index = right_index
        self.mouse_left_released = float(left_release)
        self.mouse_left_pressed = float(left_press)
        self.mouse_right_released = float(right_release)
        self.mouse_right_pressed = float(right_press)

    # ------------------------------------------------------------------
    # One-time (per reconfigure) geometry: rest heights + press-target frames
    # ------------------------------------------------------------------

    def _after_reconfigure(self, options: dict):
        self._settle_laptop_height()
        self._settle_keyboard_height()
        self._settle_mouse_height()

        self.laptop_geometry = extract_laptop_hinge_geometry(self.laptop_urdf_path, float(self.laptop_cfg["scale"]))
        self.laptop_qpos_closed = float(self.laptop_cfg["qpos_closed"])
        self.laptop_qpos_open = float(self.laptop_cfg["qpos_open"])
        # Match the standalone task's float32 tensor thresholds exactly. A
        # float64 pi/2 rejects a float32 hinge value that renders as 90.0 deg.
        self.laptop_target_angle = np.float32(
            np.deg2rad(float(self.laptop_cfg["target_angle_deg"]))
        )
        self.laptop_maintain_angle = np.float32(
            np.deg2rad(float(self.laptop_cfg["maintain_angle_deg"]))
        )
        self.laptop_target_qpos = self.laptop_qpos_closed + self.laptop_target_angle

        press_sign = float(np.sign(self.keyboard_pressed[0] - self.keyboard_released[0]))
        self.target_key_geometry = [
            load_target_key_geometry(
                source_urdf=self.keyboard_urdf_path,
                target_link_name=f"link_{jn.split('_')[1]}",
                target_joint_name=jn,
                scale=float(self.keyboard_cfg["scale"]),
                press_sign=press_sign,
            )
            for jn in TARGET_KEYS
        ]
        for geom in self.target_key_geometry:
            for key in ("top_center", "up_axis", "long_axis", "short_axis", "joint_axis", "press_axis"):
                geom[key] = common.to_tensor(geom[key], device=self.device).float()

        mouse_press_sign = float(np.sign(self.mouse_cfg["source_pressed"] - self.mouse_cfg["source_released"]))
        geom = load_button_geometry(
            source_urdf=self.mouse_urdf_path,
            link_name=self.mouse_cfg["left_link_name"],
            joint_name=self.mouse_cfg["left_joint_name"],
            scale=float(self.mouse_cfg["scale"]),
            press_sign=mouse_press_sign,
        )
        for key in ("top_center", "up_axis", "long_axis", "short_axis", "press_axis"):
            if key in geom:
                geom[key] = common.to_tensor(geom[key], device=self.device).float()
        self.mouse_button_geometry = geom

    def _settle_laptop_height(self):
        qpos = self.laptop.get_qpos()
        qpos[:, 0] = float(self.laptop_cfg["qpos_closed"])
        self.laptop.set_qpos(qpos)
        self.laptop.set_qvel(torch.zeros_like(self.laptop.get_qvel()))
        min_z = None
        for link in self.laptop.get_links():
            mesh = _laptop_link_collision_mesh(link, to_world_frame=True)
            if mesh is None or len(mesh.vertices) == 0:
                continue
            value = float(mesh.bounding_box.bounds[0, 2])
            min_z = value if min_z is None else min(min_z, value)
        dz = (-min_z if min_z is not None else 0.0) + 0.004
        self.laptop.set_pose(sapien.Pose(p=[LAPTOP_X, 0.0, dz]))
        # Start slightly open (matches office_laptop.py's start_angle range) so
        # the push motion has an initial gap to make contact through.
        start_angle = np.deg2rad(0.5 * (self.laptop_cfg["start_angle_min_deg"] + self.laptop_cfg["start_angle_max_deg"]))
        qpos[:, 0] = float(self.laptop_cfg["qpos_closed"]) + start_angle
        self.laptop.set_qpos(qpos)
        self.laptop.set_qvel(torch.zeros_like(self.laptop.get_qvel()))
        self.laptop_rest_z = dz

    def _settle_keyboard_height(self):
        qpos = self.keyboard.get_qpos()
        qpos[:] = common.to_tensor(self.keyboard_released, device=self.device)
        self.keyboard.set_qpos(qpos)
        self.keyboard.set_qvel(torch.zeros_like(self.keyboard.get_qvel()))
        min_z = None
        for link in self.keyboard.get_links():
            mesh = _laptop_link_collision_mesh(link, to_world_frame=True)
            if mesh is None or len(mesh.vertices) == 0:
                continue
            value = float(mesh.bounding_box.bounds[0, 2])
            min_z = value if min_z is None else min(min_z, value)
        dz = (-min_z if min_z is not None else 0.0) + 0.002
        self.keyboard.set_pose(sapien.Pose(p=[KEYBOARD_X, 0.0, dz]))
        self.keyboard.set_qpos(qpos)
        self.keyboard.set_qvel(torch.zeros_like(self.keyboard.get_qvel()))
        self.keyboard_rest_z = dz

    def _settle_mouse_height(self):
        # The mouse sits directly on the shared office desk.
        qpos = self.mouse.get_qpos()
        qpos[:, self.mouse_left_index] = self.mouse_left_released
        qpos[:, self.mouse_right_index] = self.mouse_right_released
        self.mouse.set_qpos(qpos)
        self.mouse.set_qvel(torch.zeros_like(self.mouse.get_qvel()))
        min_z = None
        for link in self.mouse.get_links():
            mesh = _laptop_link_collision_mesh(link, to_world_frame=True)
            if mesh is None or len(mesh.vertices) == 0:
                continue
            value = float(mesh.bounding_box.bounds[0, 2])
            min_z = value if min_z is None else min(min_z, value)
        current_z = float(self.mouse.pose.p[0, 2])
        target_bottom_z = 0.001
        correction = target_bottom_z - (min_z if min_z is not None else current_z)
        new_z = current_z + correction
        yaw = float(self.mouse_cfg.get("yaw_offset", 0.0))
        mouse_q = [float(np.cos(0.5 * yaw)), 0.0, 0.0, float(np.sin(0.5 * yaw))]
        self.mouse.set_pose(sapien.Pose(p=[MOUSE_X, MOUSE_Y, new_z], q=mouse_q))
        self.mouse.set_qpos(qpos)
        self.mouse.set_qvel(torch.zeros_like(self.mouse.get_qvel()))
        self.mouse_rest_z = new_z

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            # TableSceneBuilder.initialize restores its own stock pose.
            set_work_table_pose(self)

            # Per-episode layout, from the episode RNG, with the SAME
            # half-ranges the three subtasks use by default (laptop_xy_noise, keyboard_random_*,
            # mouse_xy_noise: 20 mm / 5 deg). Draw order: laptop, keyboard, mouse.
            rng = layout_rng(self, int(env_idx[0]))
            ldx, ldy, lyaw = uniform_offset(rng, LONG_LAPTOP_RANDOM[:2], LONG_LAPTOP_RANDOM[2])
            kdx, kdy, kyaw = uniform_offset(rng, LONG_KEYBOARD_RANDOM[:2], LONG_KEYBOARD_RANDOM[2])
            mdx, mdy, myaw = uniform_offset(rng, LONG_MOUSE_RANDOM[:2], LONG_MOUSE_RANDOM[2])

            start_angle = np.deg2rad(
                0.5 * (self.laptop_cfg["start_angle_min_deg"] + self.laptop_cfg["start_angle_max_deg"])
            )
            laptop_qpos = self.laptop.get_qpos()
            laptop_qpos[:, 0] = self.laptop_qpos_closed + start_angle
            self.laptop.set_qpos(laptop_qpos)
            self.laptop.set_qvel(torch.zeros_like(self.laptop.get_qvel()))
            self.laptop.set_pose(sapien.Pose(p=[LAPTOP_X + ldx, ldy, self.laptop_rest_z], q=yaw_quat_wxyz(lyaw).tolist()))

            keyboard_qpos = self.keyboard.get_qpos()
            keyboard_qpos[:] = common.to_tensor(self.keyboard_released, device=self.device)
            self.keyboard.set_qpos(keyboard_qpos)
            self.keyboard.set_qvel(torch.zeros_like(self.keyboard.get_qvel()))
            self.keyboard.set_pose(sapien.Pose(p=[KEYBOARD_X + kdx, kdy, self.keyboard_rest_z], q=yaw_quat_wxyz(kyaw).tolist()))
            for key_index, joint in enumerate(self.keyboard.get_active_joints()):
                joint.set_drive_target(
                    torch.full((self.num_envs,), float(self.keyboard_released[key_index]), device=self.device)
                )

            mouse_qpos = self.mouse.get_qpos()
            mouse_qpos[:, self.mouse_left_index] = self.mouse_left_released
            mouse_qpos[:, self.mouse_right_index] = self.mouse_right_released
            self.mouse.set_qpos(mouse_qpos)
            self.mouse.set_qvel(torch.zeros_like(self.mouse.get_qvel()))
            yaw = float(self.mouse_cfg.get("yaw_offset", 0.0)) + myaw
            mouse_q = [float(np.cos(0.5 * yaw)), 0.0, 0.0, float(np.sin(0.5 * yaw))]
            self.mouse.set_pose(sapien.Pose(p=[MOUSE_X + mdx, MOUSE_Y + mdy, self.mouse_rest_z], q=mouse_q))
            self.mouse.get_active_joints()[self.mouse_left_index].set_drive_target(
                torch.full((self.num_envs,), self.mouse_left_released, device=self.device)
            )
            self.mouse.get_active_joints()[self.mouse_right_index].set_drive_target(
                torch.full((self.num_envs,), self.mouse_right_released, device=self.device)
            )

            b = len(env_idx)
            self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.laptop_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.laptop_reached_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.laptop_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._reset_key_cycle()
            self.mouse_press_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.success_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._reset_agent(env_idx)

    def _reset_agent(self, env_idx: torch.Tensor):
        batch_size = len(env_idx)
        if hasattr(self.agent, "get_reset_pos"):
            initial = self.agent.get_reset_pos(device=self.device)
        else:
            initial = torch.as_tensor(
                self.agent.keyframes["rest"].qpos,
                dtype=torch.float32,
                device=self.device,
            )
        qpos = initial.expand(batch_size, -1).clone()
        # Shared teleop start pose (tasks/config/workspace_rig.py).
        qpos = apply_arm_init_qpos(self.agent, qpos)
        # the noise used to be added BEFORE apply_arm_init_qpos, which
        # overwrites the arm joints -- so the long task never had any (measured:
        # identical arm qpos across seeds). Draw it from the episode RNG afterwards.
        for k, i in enumerate(env_idx.tolist()):
            index, noise = arm_qpos_noise(
                episode_rng(self, i), self.agent, 1, self.robot_init_qpos_noise, device=self.device)
            qpos[k, index] += noise[0]
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(ROBOT_ROOT_POSE)

    def _reset_key_cycle(self):
        z = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        zb = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.key_press_started = zb.clone()
        self.key_hold_completed = zb.clone()
        self.key_released_after_hold = zb.clone()
        self.key_press_hold_counter = z.clone()
        self.key_release_counter = z.clone()

    # ------------------------------------------------------------------
    # Per-stage quantities
    # ------------------------------------------------------------------

    def _laptop_opening_angle(self) -> torch.Tensor:
        qpos = self.laptop.get_qpos()[:, 0]
        return torch.abs(qpos - self.laptop_qpos_closed)

    def _arm_qvel_norm(self) -> torch.Tensor:
        active_names = [joint.name for joint in self.agent.robot.get_active_joints()]
        arm_indices = [active_names.index(name) for name in self.agent.arm_joint_names]
        return torch.linalg.norm(self.agent.robot.get_qvel()[:, arm_indices], dim=1)

    def _side_tip_link(self, side: str):
        tip_links = getattr(self.agent, f"{side}_tip_links", None)
        if not tip_links:
            raise RuntimeError(f"{self.robot_uids} has no {side}_tip_links")
        if self.target_finger_id >= len(tip_links):
            raise ValueError(
                f"target_finger_id={self.target_finger_id} exceeds "
                f"{side}-hand tip count {len(tip_links)}"
            )
        # The PAD that presses, not the fingertip frame -- see
        # tasks/office_desk/office_desk_type_keys.py::press_contact_link.
        return press_contact_link(self.agent, tip_links[self.target_finger_id])

    def _side_tip_position(self, side: str) -> torch.Tensor:
        return self._side_tip_link(side).pose.p

    def _key_quantities(self, key_slot: int) -> dict[str, torch.Tensor]:
        geom = self.target_key_geometry[key_slot]
        link = self.target_key_links[key_slot]
        joint_index = self.target_key_indices[key_slot]

        link_p = link.pose.p
        link_q = link.pose.q
        top = link_p + quat_rotate_wxyz(link_q, geom["top_center"].unsqueeze(0))
        up = quat_rotate_wxyz(link_q, geom["up_axis"].unsqueeze(0))
        long_axis = quat_rotate_wxyz(link_q, geom["long_axis"].unsqueeze(0))
        short_axis = quat_rotate_wxyz(link_q, geom["short_axis"].unsqueeze(0))
        joint_axis = quat_rotate_wxyz(link_q, geom["joint_axis"].unsqueeze(0))

        key_qpos = self.keyboard.get_qpos()[:, joint_index]
        key_qvel = self.keyboard.get_qvel()[:, joint_index]
        released = float(self.keyboard_released[joint_index])
        pressed = float(self.keyboard_pressed[joint_index])
        press_fraction = (key_qpos - released) / (pressed - released)
        released_top = top - joint_axis * (key_qpos - released).unsqueeze(-1)

        # Typing is exclusively assigned to the left hand.
        tip = self._side_tip_position("left")
        relative = tip - top
        relative_released = tip - released_top
        tip_long = torch.sum(relative * long_axis, dim=1)
        tip_short = torch.sum(relative * short_axis, dim=1)
        height_released = torch.sum(relative_released * up, dim=1)

        tip_over_press_region = (
            (torch.abs(tip_long) <= float(geom["half_long"]) + KEY_PRESS_LONG_MARGIN)
            & (torch.abs(tip_short) <= float(geom["half_short"]) + KEY_PRESS_SHORT_MARGIN)
        )
        return {
            "top": top,
            "released_top": released_top,
            "up": up,
            "tip": tip,
            "press_fraction": press_fraction,
            "key_qvel": key_qvel,
            "tip_over_press_region": tip_over_press_region,
            "height_released": height_released,
        }

    def _mouse_quantities(self) -> dict[str, torch.Tensor]:
        geom = self.mouse_button_geometry
        link = self.mouse_left_link
        link_p = link.pose.p
        link_q = link.pose.q
        top = link_p + quat_rotate_wxyz(link_q, geom["top_center"].unsqueeze(0))
        up = quat_rotate_wxyz(link_q, geom["up_axis"].unsqueeze(0))
        press_axis = quat_rotate_wxyz(link_q, geom["press_axis"].unsqueeze(0))
        long_axis = quat_rotate_wxyz(link_q, geom["long_axis"].unsqueeze(0))
        short_axis = quat_rotate_wxyz(link_q, geom["short_axis"].unsqueeze(0))

        left_qpos = self.mouse.get_qpos()[:, self.mouse_left_index]
        right_qpos = self.mouse.get_qpos()[:, self.mouse_right_index]
        left_fraction = (left_qpos - self.mouse_left_released) / (self.mouse_left_pressed - self.mouse_left_released)
        right_fraction = (right_qpos - self.mouse_right_released) / (self.mouse_right_pressed - self.mouse_right_released)
        stroke = abs(self.mouse_left_pressed - self.mouse_left_released)
        released_top = top - press_axis * (left_fraction * stroke).unsqueeze(-1)

        # Clicking is exclusively assigned to the right hand.
        tip = self._side_tip_position("right")
        relative = tip - top
        tip_long = torch.sum(relative * long_axis, dim=1)
        tip_short = torch.sum(relative * short_axis, dim=1)
        tip_over_button = (
            (torch.abs(tip_long) <= float(geom["half_long"]) + MOUSE_PRESS_LONG_MARGIN)
            & (torch.abs(tip_short) <= float(geom["half_short"]) + MOUSE_PRESS_SHORT_MARGIN)
        )
        return {
            "top": top,
            "released_top": released_top,
            "up": up,
            "tip": tip,
            "left_fraction": left_fraction,
            "right_fraction": right_fraction,
            "tip_over_button": tip_over_button,
        }

    # ------------------------------------------------------------------
    # Staged success state machine
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        stage_at_start = self.stage.clone()
        stage = stage_at_start.clone()
        key_progress = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        mouse_progress = torch.zeros_like(key_progress)
        # Live gate diagnostics. The stage machine is a chain -- a click on the
        # mouse is IGNORED unless all five key stages have latched -- and until
        # these were published the operator had no way to see WHICH gate was
        # open. They are reported, never used to decide anything.
        key_slot_diag = torch.full((self.num_envs,), -1.0, device=self.device)
        key_depth_diag = torch.zeros_like(key_progress)
        key_over_diag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mouse_depth_diag = torch.zeros_like(key_progress)
        mouse_right_diag = torch.zeros_like(key_progress)
        mouse_over_diag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- Stage 0: open the laptop -------------------------------------
        opening_angle = self._laptop_opening_angle()
        at_laptop = stage_at_start == STAGE_OPEN_LAPTOP
        # Same helper the standalone laptop stage runs on, so the two cannot
        # drift apart in structure the way they could when each had its own copy.
        (
            self.laptop_reached_once,
            self.laptop_hold_counter,
            laptop_cycle_success,
        ) = lid_open_hold_update(
            opening_angle,
            self.laptop_reached_once,
            self.laptop_hold_counter,
            target_angle=self.laptop_target_angle,
            maintain_angle=self.laptop_maintain_angle,
            hold_steps=LAPTOP_TARGET_HOLD_STEPS,
            active=at_laptop,
        )
        self.laptop_done |= laptop_cycle_success
        advance_from_laptop = at_laptop & laptop_cycle_success
        stage = torch.where(advance_from_laptop, stage + 1, stage)

        # --- Keyboard stages: type P, A, I, R, Enter in order -------------
        key_cycle_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for slot in range(len(TARGET_KEYS)):
            at_key = stage_at_start == (STAGE_TYPE_FIRST + slot)
            if not bool(at_key.any()):
                continue
            q = self._key_quantities(slot)
            pressed_enter = q["press_fraction"] >= KEY_PRESS_ENTER_RATIO
            pressed_hold = q["press_fraction"] >= KEY_PRESS_HOLD_RATIO
            released_now = q["press_fraction"] <= KEY_RELEASE_RATIO
            key_settled = torch.abs(q["key_qvel"]) <= KEY_SETTLED_QVEL

            # Same helper the standalone typing stage runs on.
            _ks, _ = key_press_cycle_update(
                q["press_fraction"], q["key_qvel"], q["tip_over_press_region"],
                {
                    # The long task has no `press_achieved` latch of its own; it
                    # is only a diagnostic in the standalone stage.
                    "press_started": self.key_press_started,
                    "press_achieved": torch.zeros_like(self.key_press_started),
                    "hold_completed": self.key_hold_completed,
                    "press_hold_counter": self.key_press_hold_counter,
                    "released_after_hold": self.key_released_after_hold,
                    "release_counter": self.key_release_counter,
                },
                active=at_key,
            )
            self.key_press_started = _ks["press_started"]
            self.key_hold_completed = _ks["hold_completed"]
            self.key_press_hold_counter = _ks["press_hold_counter"]
            self.key_release_counter = _ks["release_counter"]
            self.key_released_after_hold = _ks["released_after_hold"]
            key_cycle_success |= at_key & self.key_released_after_hold
            press_progress = torch.clamp(q["press_fraction"] / KEY_PRESS_HOLD_RATIO, 0.0, 1.0)
            hold_progress = torch.clamp(
                self.key_press_hold_counter.float() / KEY_PRESS_HOLD_STEPS, 0.0, 1.0
            )
            release_progress = torch.clamp(
                self.key_release_counter.float() / KEY_RELEASE_STEPS, 0.0, 1.0
            )
            cycle_progress = torch.where(
                self.key_hold_completed,
                0.75 + 0.25 * release_progress,
                torch.where(self.key_press_started, 0.5 + 0.25 * hold_progress, 0.5 * press_progress),
            )
            key_progress = torch.where(at_key, cycle_progress, key_progress)
            key_slot_diag = torch.where(at_key, torch.full_like(key_slot_diag, float(slot)), key_slot_diag)
            key_depth_diag = torch.where(at_key, q["press_fraction"], key_depth_diag)
            key_over_diag |= at_key & q["tip_over_press_region"]

        advance_from_key = key_cycle_success
        stage = torch.where(advance_from_key, stage + 1, stage)
        reset_key_cycle_mask = advance_from_key
        if bool(reset_key_cycle_mask.any()):
            self.key_press_started = torch.where(reset_key_cycle_mask, torch.zeros_like(self.key_press_started), self.key_press_started)
            self.key_hold_completed = torch.where(reset_key_cycle_mask, torch.zeros_like(self.key_hold_completed), self.key_hold_completed)
            self.key_released_after_hold = torch.where(reset_key_cycle_mask, torch.zeros_like(self.key_released_after_hold), self.key_released_after_hold)
            self.key_press_hold_counter = torch.where(reset_key_cycle_mask, torch.zeros_like(self.key_press_hold_counter), self.key_press_hold_counter)
            self.key_release_counter = torch.where(reset_key_cycle_mask, torch.zeros_like(self.key_release_counter), self.key_release_counter)

        # --- Final interaction: click only the mouse's left button --------
        at_mouse = stage_at_start == STAGE_CLICK_MOUSE
        mouse_cycle_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if bool(at_mouse.any()):
            qm = self._mouse_quantities()
            # Same helper the standalone click stage runs on.
            self.mouse_press_counter, mouse_cycle_success = mouse_click_cycle_update(
                qm["left_fraction"], qm["right_fraction"], qm["tip_over_button"],
                self.mouse_press_counter, active=at_mouse,
            )
            mouse_progress = torch.where(
                at_mouse,
                torch.clamp(self.mouse_press_counter.float() / MOUSE_SUCCESS_PRESS_STEPS, 0.0, 1.0),
                mouse_progress,
            )
            mouse_depth_diag = torch.where(at_mouse, qm["left_fraction"], mouse_depth_diag)
            mouse_right_diag = torch.where(at_mouse, qm["right_fraction"], mouse_right_diag)
            mouse_over_diag |= at_mouse & qm["tip_over_button"]
        stage = torch.where(mouse_cycle_success, stage + 1, stage)

        self.stage = stage
        self._update_key_target_marker(stage)
        self.success_latched |= stage >= STAGE_DONE
        task_success = self.success_latched

        return {
            "success": task_success,
            "task_success": task_success,
            "stage": stage,
            "laptop_opening_angle_deg": torch.rad2deg(opening_angle),
            "laptop_done": self.laptop_done,
            "key_hold_completed": self.key_hold_completed,
            "key_released_after_hold": self.key_released_after_hold,
            "key_progress": key_progress,
            "mouse_progress": mouse_progress,
            # --- live gate diagnostics (see evaluate) ----------------------
            "key_slot": key_slot_diag,
            "key_press_fraction": key_depth_diag,
            "key_tip_over_region": key_over_diag,
            "mouse_left_fraction": mouse_depth_diag,
            "mouse_right_fraction": mouse_right_diag,
            "mouse_tip_over_button": mouse_over_diag,
        }

    def _update_key_target_marker(self, stage: torch.Tensor) -> None:
        """Float the marker over whichever cap the operator must press next."""
        marker = getattr(self, "key_target_marker", None)
        if marker is None:
            return
        slot = int(stage.reshape(-1)[0]) - STAGE_TYPE_FIRST
        if 0 <= slot < len(self.target_key_links):
            move_key_target_marker(marker, self.target_key_links[slot])
        else:
            move_key_target_marker(marker, None, park=True)

    def _get_obs_extra(self, info: dict):
        keyboard_qpos = self.keyboard.get_qpos()
        keyboard_qvel = self.keyboard.get_qvel()
        keyboard_released = common.to_tensor(
            self.keyboard_released, device=self.device
        ).unsqueeze(0)
        keyboard_pressed = common.to_tensor(
            self.keyboard_pressed, device=self.device
        ).unsqueeze(0)
        keyboard_press_fraction = (
            (keyboard_qpos - keyboard_released)
            / (keyboard_pressed - keyboard_released)
        )
        current_key_slot = torch.clamp(
            info["stage"] - STAGE_TYPE_FIRST,
            min=0,
            max=len(TARGET_KEYS) - 1,
        )
        target_indices = torch.as_tensor(
            self.target_key_indices, dtype=torch.long, device=self.device
        )
        current_key_joint = target_indices[current_key_slot]
        current_key_fraction = torch.gather(
            keyboard_press_fraction, 1, current_key_joint[:, None]
        )
        mouse = self._mouse_quantities()
        return {
            "stage": info["stage"].unsqueeze(1).float(),
            "stage_one_hot": torch.nn.functional.one_hot(
                info["stage"].clamp(max=STAGE_DONE), num_classes=STAGE_DONE + 1
            ).float(),
            "laptop_opening_angle_deg": info["laptop_opening_angle_deg"].unsqueeze(1),
            "laptop_hold_progress": torch.clamp(
                self.laptop_hold_counter[:, None].float()
                / LAPTOP_TARGET_HOLD_STEPS,
                0.0,
                1.0,
            ),
            "keyboard_qpos": keyboard_qpos,
            "keyboard_qvel": keyboard_qvel,
            "keyboard_press_fraction": keyboard_press_fraction,
            "current_key_slot": current_key_slot[:, None].float(),
            "current_key_press_fraction": current_key_fraction,
            "mouse_press_fraction": torch.stack(
                [mouse["left_fraction"], mouse["right_fraction"]], dim=1
            ),
            # Randomized object poses must be observable.
            "laptop_pose": self.laptop.pose.raw_pose,
            "keyboard_pose": self.keyboard.pose.raw_pose,
            "mouse_pose": self.mouse.pose.raw_pose,
            "left_task_tip_position": self._side_tip_position("left"),
            "right_task_tip_position": self._side_tip_position("right"),
            "left_tcp_pose": self.agent.left_tcp_pose.raw_pose,
            "right_tcp_pose": self.agent.right_tcp_pose.raw_pose,
        }

    def get_state_dict(self):
        """Persist physical state plus every long-horizon stage latch."""
        state = super().get_state_dict()
        state["office_task_state"] = torch.cat(
            [
                self.stage[:, None].float(),
                self.laptop_hold_counter[:, None].float(),
                self.laptop_reached_once[:, None].float(),
                self.laptop_done[:, None].float(),
                self.key_press_started[:, None].float(),
                self.key_hold_completed[:, None].float(),
                self.key_released_after_hold[:, None].float(),
                self.key_press_hold_counter[:, None].float(),
                self.key_release_counter[:, None].float(),
                self.mouse_press_counter[:, None].float(),
                self.success_latched[:, None].float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("office_task_state", None)
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
        if task_state.shape[-1] != 11:
            raise ValueError(
                f"office_task_state must have 11 fields, "
                f"got {task_state.shape[-1]}"
            )
        self.stage[restore_idx] = task_state[:, 0].to(torch.long)
        self.laptop_hold_counter[restore_idx] = task_state[:, 1].to(torch.long)
        self.laptop_reached_once[restore_idx] = task_state[:, 2].bool()
        self.laptop_done[restore_idx] = task_state[:, 3].bool()
        self.key_press_started[restore_idx] = task_state[:, 4].bool()
        self.key_hold_completed[restore_idx] = task_state[:, 5].bool()
        self.key_released_after_hold[restore_idx] = task_state[:, 6].bool()
        self.key_press_hold_counter[restore_idx] = task_state[:, 7].to(torch.long)
        self.key_release_counter[restore_idx] = task_state[:, 8].to(torch.long)
        self.mouse_press_counter[restore_idx] = task_state[:, 9].to(torch.long)
        self.success_latched[restore_idx] = task_state[:, 10].bool()

    def compute_sparse_reward(self, obs: Any, action, info: dict):
        return info["task_success"].float()

    def compute_dense_reward(self, obs: Any, action, info: dict):
        stage_progress = info["stage"].float()
        laptop_progress = torch.clamp(info["laptop_opening_angle_deg"] / np.rad2deg(self.laptop_target_angle), 0.0, 1.0)
        at_laptop = info["stage"] == STAGE_OPEN_LAPTOP
        at_keyboard = (info["stage"] >= STAGE_TYPE_FIRST) & (info["stage"] < STAGE_CLICK_MOUSE)
        current_progress = torch.where(
            at_laptop,
            laptop_progress,
            torch.where(at_keyboard, info["key_progress"], info["mouse_progress"]),
        )
        reward = stage_progress + 0.8 * current_progress
        return torch.where(info["task_success"], reward + 10.0, reward)

    def compute_normalized_dense_reward(self, obs: Any, action, info: dict):
        return self.compute_dense_reward(obs, action, info) / (STAGE_DONE + 10.0)


def hold_action(env, control_mode: str):
    base = env.unwrapped
    if control_mode == "pd_joint_pos":
        qpos = base.agent.robot.get_qpos().detach().clone()
        active_names = [joint.name for joint in base.agent.robot.get_active_joints()]
        action_joint_names = (
            list(base.agent.left_arm_joint_names)
            + list(base.agent.right_arm_joint_names)
        )
        if hasattr(base.agent, "left_hand_joint_names"):
            action_joint_names += list(base.agent.left_hand_joint_names)
            action_joint_names += list(base.agent.right_hand_joint_names)
        else:
            # Each Franka gripper controller has two mimic qpos joints but one
            # scalar action target.
            action_joint_names += [base.agent.left_gripper_joint_names[0]]
            action_joint_names += [base.agent.right_gripper_joint_names[0]]
        action = qpos[:, [active_names.index(name) for name in action_joint_names]]
        if action.shape[-1] != int(env.action_space.shape[-1]):
            raise RuntimeError(
                f"Cannot build dual-arm hold action: got {action.shape[-1]} "
                f"values for action space {env.action_space}"
            )
        return action
    return torch.zeros((1, env.action_space.shape[-1]), dtype=torch.float32, device=base.device)


def run(args):
    render_mode = "rgb_array" if args.record_video else args.render_mode
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=render_mode,
        sim_backend=args.sim_backend,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    if args.record_video:
        os.makedirs(args.output_dir, exist_ok=True)
        env = RecordEpisode(
            env, output_dir=args.output_dir, save_video=True, save_trajectory=args.save_trajectory,
            trajectory_name="office", video_fps=20, max_steps_per_video=args.max_steps,
        )

    try:
        _, info = env.reset(seed=args.seed)
        base = env.unwrapped
        action = None if args.random_actions else hold_action(env, args.control_mode)

        print("### CODE_VERSION_MARKER: office ###")
        print("Robot DoF:", base.agent.robot.dof)
        print("Action space:", env.action_space)
        print("Laptop rest z:", base.laptop_rest_z, "Keyboard rest z:", base.keyboard_rest_z, "Mouse rest z:", base.mouse_rest_z)

        for step in range(args.max_steps):
            if args.random_actions:
                action = torch.as_tensor(env.action_space.sample(), dtype=torch.float32, device=base.device)
                if action.ndim == 1:
                    action = action.unsqueeze(0)
            _, reward, terminated, truncated, info = env.step(action)
            if args.render_mode == "human" and not args.record_video:
                env.render()
            if args.debug_print and step % args.debug_interval == 0:
                print(
                    f"step={step:04d} stage={STAGE_NAMES[int(info['stage'])]:12s} "
                    f"reward={_scalar(reward):.3f} laptop_deg={_scalar(info['laptop_opening_angle_deg']):.1f} "
                    f"success={int(_boolean(info['success']))}"
                )
            if _boolean(terminated) or _boolean(truncated):
                break
        print(f"steps={step + 1} stage={STAGE_NAMES[int(info['stage'])]} success={int(_boolean(info['success']))}")
        if args.record_video:
            print("Output directory:", args.output_dir)
    finally:
        env.close()


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Bimanual office task: right opens laptop -> left types "
            "P-A-I-R-Enter -> right clicks mouse"
        )
    )
    p.add_argument(
        "--env-id",
        default="OfficeDeskLong-Sharpa-v1",
        choices=(
            "OfficeDeskLong-Gripper-v1",
            "OfficeDeskLong-Sharpa-v1",
            "OfficeDeskLong-Wuji-v1",
            "OfficeDeskLong-XHand-v1",
        ),
    )
    p.add_argument("--control-mode", default=DEFAULT_CONTROL_MODE, choices=SUPPORTED_CONTROL_MODES)
    p.add_argument("--obs-mode", default="state")
    p.add_argument("--reward-mode", default="sparse", choices=("sparse", "none"))
    p.add_argument("--sim-backend", default="auto", choices=("auto", "cpu", "gpu"))
    p.add_argument("--render-mode", default="rgb_array", choices=("rgb_array", "human"))
    p.add_argument("--random-actions", action="store_true")
    p.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--save-trajectory", action="store_true")
    p.add_argument("--output-dir", default="tasks/videos_office")
    p.add_argument("--render-width", type=int, default=960)
    p.add_argument("--render-height", type=int, default=720)
    p.add_argument("--debug-print", action="store_true")
    p.add_argument("--debug-interval", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

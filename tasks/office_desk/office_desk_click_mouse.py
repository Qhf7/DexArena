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

sys.path.append("./")

from tasks.robots.panda_gripper_right_mobile_base import PandaGripperRightMobileBase
from tasks.robots.panda_sharpa_hand_right_mobile_base import (
    PandaSharpaHandRightAngledMobileBase,
    PandaSharpaHandRightMobileBase,
)
from tasks.robots.panda_wuji_hand_right_mobile_base import (
    PandaWujiHandRightMobileBase,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry.trimesh_utils import (
    get_component_meshes,
    merge_meshes,
)
from tasks.config.hands import register_task
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose, vectorize_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from tasks.config.object_materials import object_material
from tasks.office_desk.office_desk_type_keys import press_contact_link
from tasks.config.workspace_rig import (
    WORK_TABLE_X_OFFSET,
    apply_arm_init_qpos,
    platform_root_pose,
    set_work_table_pose,
)

# ABS/PBT enclosure and keycap plastic.
OFFICE_PLASTIC_DENSITY = 1050.0
from tasks.config.sim_profiles import make_sim_config
from tasks.scene_builders import TableSceneBuilder
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag


# =============================================================================
# Configuration
# =============================================================================

# The MOUSE_ASSET_DIR environment variable can override the vendored asset root.
_OFFICE_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "task_object_meshes" / "office"
MOUSE_ASSET_DIR = Path(
    os.environ.get(
        "MOUSE_ASSET_DIR",
        str(_OFFICE_ASSET_DIR),
    )
)

MOUSE_MODEL = {
    "model_id": "101408",
    "scale": 0.08,
    "left_joint_index": 0,
    "right_joint_index": 1,
    "left_joint_name": "joint_0",
    "left_link_name": "link_0",
    "source_released": 0.0,
    "source_pressed": 0.01,
    "yaw_offset": np.pi,
    # Return-spring sized to a real micro switch (Omron D2F/D2FC class:
    # 0.44-0.74 N operating force over ~1 mm travel -> 600 N/m gives 0.6 N
    # at full stroke). The old 120 N/m peaked at 0.12 N, which could not
    # overcome the button's own weight + any joint friction, so the button
    # stalled mid-stroke instead of rebounding (success needs frac<0.20).
    "button_stiffness": 600.0,
    "button_damping": 2.0,
    "button_force_limit": 1.0,
    # Real switch plungers have negligible sliding friction. Without this,
    # PhysX's default joint frictionCoefficient (0.05 N) alone eats nearly
    # half of the return-spring force budget.
    "button_friction": 0.0,
    # Scroll wheel. The asset ships it as a free revolute joint with zero
    # damping and an FLT_MAX force limit, so it rang freely; a real wheel has
    # detent friction of 0.02-0.15 N*m and is never a force source.
    "wheel_friction": 0.05,
    "wheel_damping": 0.01,
    "wheel_force_limit": 0.5,
}

CONTROL_FREQUENCY = 20
SUCCESS_REWARD = 10.0
SUPPORTED_CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")
DEFAULT_CONTROL_MODE = "pd_joint_pos"

PRESS_ENTER_RATIO = 0.60
PRESS_HOLD_RATIO = 0.45
PRESS_HOLD_STEPS = 2

# Simplified success (per user request): a valid left-button press
# — deep enough (>= PRESS_ENTER_RATIO), target fingertip over the button, right
# button NOT mis-pressed — sustained this many consecutive control steps. The
# old release + hover phases are no longer required (their constants below are
# kept only for reward shaping / observations).
SUCCESS_PRESS_STEPS = 3

RELEASE_RATIO = 0.20
RELEASE_STEPS = 2
RIGHT_BUTTON_MAX_RATIO = 0.35

# Legacy geometry exports used by office.py. The standalone
# mouse task no longer builds this riser; keeping the values here preserves
# imports while the combined scene is migrated independently.
RISER_HEIGHT = 0.05
RISER_HALF_XY = (0.11, 0.09)
TABLE_TOP_THICKNESS = 0.008
TABLE_LEG_HALF = 0.008
TABLE_LEG_INSET = 0.012
TABLE_LEG_COLLISION = False

PRESS_LONG_MARGIN = 0.004
PRESS_SHORT_MARGIN = 0.004

FINAL_MIN_HEIGHT = 0.025
FINAL_MAX_HEIGHT = 0.075
FINAL_LONG_MARGIN = 0.008
FINAL_SHORT_MARGIN = 0.008
FINAL_SETTLE_STEPS = 3

HAND_ROBOT_UIDS = {
    "panda_sharpa_hand_right_mobile_base",
    "panda_sharpa_hand_right_angled_mobile_base",
    "panda_wuji_hand_right_angled_mobile_base",
    "panda_xhand_hand_right_angled_mobile_base",
}

GRIPPER_ROBOT_UID = "panda_gripper_right_mobile_base"
# LOCKED to the shared rig (tasks/config/workspace_rig.py). Two drifts fixed
# here: the trolley stood at -0.615 + WORLD_X_OFFSET = -1.145, i.e. 25 mm
# CLOSER to the desk than the office long task's own -1.170 (the objects sat at
# identical world x in both, so the subtask handed the operator a different
# reach envelope than the long task it is a subset of); and the whole family
# was anchored on the STOCK table centre while CabinetPour/Screwing use the
# +0.30 shifted one. The scene now translates with the table and the trolley.
WORLD_X_OFFSET = -0.53 + WORK_TABLE_X_OFFSET
OFFICE_GROUP_X_OFFSET = 0.20
ROBOT_ROOT_POSE = platform_root_pose()
DEFAULT_ROBOT_UID = "panda_sharpa_hand_right_angled_mobile_base"
# SINGLE SOURCE OF TRUTH for the mouse's world xy; office.py imports these.
MOUSE_X = -0.07 + WORLD_X_OFFSET + OFFICE_GROUP_X_OFFSET
MOUSE_Y = -0.32
MOUSE_SPAWN_CENTER = np.array([MOUSE_X, MOUSE_Y], dtype=np.float32)


# =============================================================================
# Geometry
# =============================================================================

def _parse_vector(
    text: str | None,
    default,
) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(
        [float(value) for value in text.split()],
        dtype=np.float64,
    )


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

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = euler2mat(
        *rpy,
        axes="sxyz",
    )
    matrix[:3, 3] = xyz
    return matrix


def _read_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for line in file:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) >= 4:
                vertices.append(
                    [
                        float(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                    ]
                )

    if not vertices:
        raise RuntimeError(
            f"No OBJ vertices found in {path}"
        )
    return np.asarray(vertices, dtype=np.float64)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise RuntimeError(
            f"Cannot normalize vector {vector}"
        )
    return vector / norm



# =============================================================================
# Left-click predicate, shared with the long task
# =============================================================================
#
# `office.py` carried its own copy: the same conjunction with the same imported
# ratios, differing only in that this one freezes the counter once success has
# latched. One implementation now serves both, so structure cannot drift the way
# it could when each side had its own.
def mouse_click_cycle_update(
    left_fraction,
    right_fraction,
    tip_over_button,
    press_counter,
    *,
    latched=None,
    active=None,
):
    """A valid left click sustained for SUCCESS_PRESS_STEPS control steps.

    Valid = pressed past PRESS_ENTER_RATIO, the target fingertip over the left
    button, and the right button not mis-pressed. `active` is the long task's
    stage gate; `latched` freezes the counter once success is in hand.

    Returns (press_counter, success).
    """
    import torch as _torch

    if active is None:
        active = _torch.ones_like(tip_over_button)
    pressing_now = (
        active
        & tip_over_button
        & (left_fraction >= PRESS_ENTER_RATIO)
        & (right_fraction < RIGHT_BUTTON_MAX_RATIO)
    )
    stepped = _torch.where(
        pressing_now, press_counter + 1, _torch.zeros_like(press_counter)
    )
    if latched is not None:
        stepped = _torch.where(latched, press_counter, stepped)
    return stepped, active & (stepped >= SUCCESS_PRESS_STEPS)


def load_button_geometry(
    source_urdf: Path,
    link_name: str,
    joint_name: str,
    scale: float,
    press_sign: float,
) -> dict[str, np.ndarray | float]:
    root = ET.parse(source_urdf).getroot()
    link = root.find(f"./link[@name='{link_name}']")
    joint = root.find(f"./joint[@name='{joint_name}']")

    if link is None or joint is None:
        raise RuntimeError(
            f"Missing {link_name}/{joint_name} in {source_urdf}"
        )

    visual = link.find("visual")
    mesh = (
        None
        if visual is None
        else visual.find("./geometry/mesh")
    )
    if visual is None or mesh is None:
        raise RuntimeError(
            f"{link_name} has no mesh visual"
        )

    mesh_path = (
        source_urdf.parent / mesh.attrib["filename"]
    ).resolve()
    vertices = _read_obj_vertices(mesh_path)

    vertices_h = np.concatenate(
        [
            vertices,
            np.ones(
                (len(vertices), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )
    vertices_link = (
        _origin_matrix(visual) @ vertices_h.T
    ).T[:, :3]

    axis_element = joint.find("axis")
    joint_axis = _normalize(
        _parse_vector(
            (
                None
                if axis_element is None
                else axis_element.attrib.get("xyz")
            ),
            (0.0, 0.0, -1.0),
        )
    )

    press_axis = joint_axis * (
        1.0 if press_sign >= 0 else -1.0
    )
    up_axis = -press_axis

    centered = (
        vertices_link
        - vertices_link.mean(axis=0, keepdims=True)
    )
    planar = (
        centered
        - np.outer(centered @ up_axis, up_axis)
    )
    covariance = (
        planar.T @ planar / max(len(planar), 1)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    long_axis = eigenvectors[
        :,
        int(np.argmax(eigenvalues)),
    ]
    long_axis -= up_axis * float(
        long_axis @ up_axis
    )
    long_axis = _normalize(long_axis)
    short_axis = _normalize(
        np.cross(up_axis, long_axis)
    )

    long_coord = vertices_link @ long_axis
    short_coord = vertices_link @ short_axis
    up_coord = vertices_link @ up_axis

    long_min = float(long_coord.min())
    long_max = float(long_coord.max())
    short_min = float(short_coord.min())
    short_max = float(short_coord.max())
    top_coord = float(up_coord.max())

    top_center = (
        long_axis * 0.5 * (long_min + long_max)
        + short_axis * 0.5 * (short_min + short_max)
        + up_axis * top_coord
    )

    return {
        "top_center": (
            top_center * float(scale)
        ).astype(np.float32),
        "press_axis": press_axis.astype(np.float32),
        "up_axis": up_axis.astype(np.float32),
        "long_axis": long_axis.astype(np.float32),
        "short_axis": short_axis.astype(np.float32),
        "half_long": float(
            0.5
            * (long_max - long_min)
            * float(scale)
        ),
        "half_short": float(
            0.5
            * (short_max - short_min)
            * float(scale)
        ),
    }


def quat_rotate_wxyz(
    quaternion: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    xyz = quaternion[..., 1:]
    w = quaternion[..., :1]
    uv = torch.cross(xyz, vector, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (w * uv + uuv)


def _link_collision_mesh(
    link,
    to_world_frame: bool = True,
):
    meshes = []
    for obj in link._objs:
        mesh = merge_meshes(
            get_component_meshes(obj)
        )
        if mesh is not None:
            meshes.append(mesh)

    if not meshes:
        return None

    if to_world_frame:
        pose = link.pose
        for index, mesh in enumerate(meshes):
            matrix = (
                pose[index].sp.to_transformation_matrix()
                if len(pose) > 1
                else pose.sp.to_transformation_matrix()
            )
            mesh.apply_transform(matrix)

    return merge_meshes(meshes)

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


# =============================================================================
# Environment
# =============================================================================

@register_task(
    "OfficeDeskClickMouse",
    side="right",
    max_episode_steps=500,
    obs_mode="state",
    control_mode="pd_joint_pos",
    reward_mode="sparse",
    render_mode="rgb_array",
)
class OfficeDeskClickMouseEnv(BaseEnv):
    """Hold a valid left-button click for three consecutive control frames."""

    SUPPORTED_ROBOTS = [
        "panda_gripper_right_mobile_base",
        "panda_sharpa_hand_right_mobile_base",
        "panda_sharpa_hand_right_angled_mobile_base",
        "panda_wuji_hand_right_angled_mobile_base",
            "panda_xhand_hand_right_angled_mobile_base",
]
    SUPPORTED_CONTROL_MODES = SUPPORTED_CONTROL_MODES
    SUPPORTED_REWARD_MODES = ("sparse", "none")

    agent: Union[
        PandaGripperRightMobileBase,
        PandaSharpaHandRightMobileBase,
        PandaSharpaHandRightAngledMobileBase,
        PandaWujiHandRightMobileBase,
    ]

    def __init__(
        self,
        *args,
        robot_uids: str = DEFAULT_ROBOT_UID,
        robot_init_qpos_noise: float = 0.02,
        target_finger_id: int = 1,
        mouse_xy_noise: float = 0.02,
        mouse_yaw_noise_deg: float = 5.0,
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        runtime_sim_config = kwargs.pop("sim_config", None)
        if runtime_sim_config not in (None, {"control_freq": CONTROL_FREQUENCY}):
            raise ValueError(
                "ClickMouse freezes the Plant configuration in tasks/config/"
                "sim_profiles.py; only the teleop compatibility declaration "
                "{'control_freq': 20} is accepted and ignored"
            )
        self.robot_init_qpos_noise = float(
            robot_init_qpos_noise
        )
        self.target_finger_id = int(
            target_finger_id
        )
        self.mouse_xy_noise = float(mouse_xy_noise)
        self.mouse_yaw_noise = np.deg2rad(
            float(mouse_yaw_noise_deg)
        )

        if self.target_finger_id < 0:
            raise ValueError("target_finger_id must be non-negative")
        if self.robot_init_qpos_noise < 0:
            raise ValueError("robot_init_qpos_noise must be non-negative")
        if self.mouse_xy_noise < 0:
            raise ValueError("mouse_xy_noise must be non-negative")
        if self.mouse_yaw_noise < 0:
            raise ValueError("mouse_yaw_noise_deg must be non-negative")
        if reconfiguration_freq is None:
            reconfiguration_freq = (
                1 if num_envs == 1 else 0
            )

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
        pose = sapien_utils.look_at(
            eye=[0.30 + WORLD_X_OFFSET, -0.30, 0.48],
            target=[float(MOUSE_SPAWN_CENTER[0]), float(MOUSE_SPAWN_CENTER[1]), 0.08],
        )
        return [
            CameraConfig(
                "base_camera",
                pose,
                320,
                240,
                np.deg2rad(58),
                0.01,
                100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.48 + WORLD_X_OFFSET, 0.58, 0.45],
            target=[float(MOUSE_SPAWN_CENTER[0]), float(MOUSE_SPAWN_CENTER[1]), 0.08],
        )
        return CameraConfig(
            "render_camera",
            pose,
            1024,
            768,
            1.0,
            0.01,
            100,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            ROBOT_ROOT_POSE,
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

        cfg = MOUSE_MODEL
        urdf_path = (
            MOUSE_ASSET_DIR
            / cfg["model_id"]
            / "mobility.urdf"
        )
        if not urdf_path.is_file():
            raise FileNotFoundError(
                f"Mouse URDF not found: {urdf_path}"
            )

        press_sign = np.sign(
            cfg["source_pressed"]
            - cfg["source_released"]
        )
        geometry = load_button_geometry(
            source_urdf=urdf_path,
            link_name=cfg["left_link_name"],
            joint_name=cfg["left_joint_name"],
            scale=cfg["scale"],
            press_sign=press_sign,
        )

        self._mice = []
        self._left_links = []
        self._left_indices = []
        self._right_indices = []

        left_released = []
        left_pressed = []
        right_released = []
        right_pressed = []

        for env_id in range(self.num_envs):
            loader = self.scene.create_urdf_loader()
            loader.name = f"mouse-{env_id}"
            loader.fix_root_link = True
            loader.scale = float(cfg["scale"])
            loader.load_multiple_collisions_from_file = True
            # Explicit material density instead of the PhysX default (1000).
            # ABS/PBT keycap and shell plastic is 1040-1070 kg/m3; stating it
            # here keeps every link mass a chosen value rather than a default.
            loader.set_density(OFFICE_PLASTIC_DENSITY)
            # The PartNet mouse shell is CONCAVE (button cutout + hollow
            # underside); one convex hull per mesh piece caps the cutout with
            # an INVISIBLE canopy over the buttons (measured: hull top sits
            # 40-140 mm above the visual surface at scale 0.1), so a finger
            # reaching for a button was repelled in mid-air ("bounced off")
            # and the button was physically unpressable. COACD-decompose each
            # piece into true convex parts instead; the result is cached next
            # to the mesh (*.coacd.ply), so the cost is one-time.
            loader.multiple_collisions_decomposition = "coacd"

            builders = loader.parse(
                str(urdf_path)
            )["articulation_builders"]
            if not builders:
                raise RuntimeError(
                    f"No articulation parsed from {urdf_path}"
                )

            builder = builders[0]
            # Internal link collisions OFF: the buttons/wheel are sibling links
            # packed with sub-mm clearances — well inside the scene's 0.8 mm
            # contact_offset — so with sibling collision on they are WELDED in
            # place (measured: the left button would not move even under a
            # 100 N drive push). Finger-vs-mouse collisions are unaffected
            # (cross-articulation). NOTE: `builder.disable_self_collisions =
            # True` AFTER loader.parse() is a silent no-op (the flag is only
            # read when link builders are CREATED), so set the group-2 bit on
            # each link builder directly — same mechanism the loader itself
            # uses.
            builder.disable_self_collisions = True
            for lb in builder.link_builders:
                lb.collision_groups[2] |= 1 << 29
            builder.set_scene_idxs([env_id])
            builder.initial_pose = sapien.Pose()

            mouse = builder.build(
                name=f"mouse-{env_id}"
            )
            self.remove_from_state_dict_registry(mouse)
            _configure_task_articulation(mouse)

            active_joints = mouse.get_active_joints()
            left_index = int(
                cfg["left_joint_index"]
            )
            right_index = int(
                cfg["right_joint_index"]
            )
            if max(left_index, right_index) >= len(
                active_joints
            ):
                raise RuntimeError(
                    "Configured mouse button index is invalid"
                )

            qlimits = (
                mouse.get_qlimits()
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 2)
            )

            if press_sign >= 0:
                left_release, left_press = qlimits[
                    left_index
                ]
                right_release, right_press = qlimits[
                    right_index
                ]
            else:
                left_press, left_release = qlimits[
                    left_index
                ]
                right_press, right_release = qlimits[
                    right_index
                ]

            for index, release in (
                (left_index, left_release),
                (right_index, right_release),
            ):
                joint = active_joints[index]
                joint.set_drive_properties(
                    stiffness=float(
                        cfg["button_stiffness"]
                    ),
                    damping=float(
                        cfg["button_damping"]
                    ),
                    force_limit=float(
                        cfg["button_force_limit"]
                    ),
                    mode="force",
                )
                joint.set_drive_target(
                    float(release)
                )
                joint.set_friction(
                    float(cfg["button_friction"])
                )

            # Every remaining active joint is the scroll wheel. Nothing else in
            # this family may fall through to an engine default.
            for index, joint in enumerate(active_joints):
                if index in (left_index, right_index):
                    continue
                joint.set_drive_properties(
                    stiffness=0.0,
                    damping=float(cfg["wheel_damping"]),
                    force_limit=float(cfg["wheel_force_limit"]),
                    mode="force",
                )
                joint.set_friction(float(cfg["wheel_friction"]))

            self._mice.append(mouse)
            self._left_links.append(
                mouse.links_map[
                    cfg["left_link_name"]
                ]
            )
            self._left_indices.append(left_index)
            self._right_indices.append(right_index)

            left_released.append(
                float(left_release)
            )
            left_pressed.append(float(left_press))
            right_released.append(
                float(right_release)
            )
            right_pressed.append(
                float(right_press)
            )

            print(
                f"[mouse {env_id}] "
                f"left stroke="
                f"{abs(left_press-left_release):.7f}, "
                f"right stroke="
                f"{abs(right_press-right_release):.7f}"
            )

        self.mouse = Articulation.merge(
            self._mice, name="mouse", merge_links=True
        )
        self.add_to_state_dict_registry(self.mouse)

        self.left_released = common.to_tensor(
            np.asarray(
                left_released,
                dtype=np.float32,
            ),
            device=self.device,
        )
        self.left_pressed = common.to_tensor(
            np.asarray(
                left_pressed,
                dtype=np.float32,
            ),
            device=self.device,
        )
        self.right_released = common.to_tensor(
            np.asarray(
                right_released,
                dtype=np.float32,
            ),
            device=self.device,
        )
        self.right_pressed = common.to_tensor(
            np.asarray(
                right_pressed,
                dtype=np.float32,
            ),
            device=self.device,
        )

        def repeated(value):
            array = np.asarray(
                value,
                dtype=np.float32,
            )[None]
            return common.to_tensor(
                np.repeat(
                    array,
                    self.num_envs,
                    axis=0,
                ),
                device=self.device,
            )

        self.local_top = repeated(
            geometry["top_center"]
        )
        self.local_press = repeated(
            geometry["press_axis"]
        )
        self.local_up = repeated(
            geometry["up_axis"]
        )
        self.local_long = repeated(
            geometry["long_axis"]
        )
        self.local_short = repeated(
            geometry["short_axis"]
        )
        self.half_long = torch.full(
            (self.num_envs,),
            float(geometry["half_long"]),
            device=self.device,
        )
        self.half_short = torch.full(
            (self.num_envs,),
            float(geometry["half_short"]),
            device=self.device,
        )

    def _after_reconfigure(self, options: dict):
        mouse_z = []

        for env_id, mouse in enumerate(self._mice):
            qpos = mouse.get_qpos()
            qpos[:, self._left_indices[env_id]] = (
                self.left_released[env_id]
            )
            qpos[:, self._right_indices[env_id]] = (
                self.right_released[env_id]
            )
            mouse.set_qpos(qpos)
            mouse.set_qvel(
                torch.zeros_like(mouse.get_qvel())
            )
            mouse.set_pose(sapien.Pose())

            min_z = None
            for link in mouse.get_links():
                mesh = _link_collision_mesh(
                    link,
                    to_world_frame=True,
                )
                if mesh is None or len(mesh.vertices) == 0:
                    continue
                value = float(
                    mesh.bounding_box.bounds[0, 2]
                )
                min_z = (
                    value
                    if min_z is None
                    else min(min_z, value)
                )

            mouse_z.append(
                (-min_z if min_z is not None else 0.0)
                + 0.001
            )

        self.mouse_z = common.to_tensor(
            mouse_z,
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

            xyz = torch.zeros(
                (batch_size, 3),
                device=self.device,
            )
            xyz[:, :2] = (
                torch.rand(
                    (batch_size, 2),
                    device=self.device,
                )
                * 2.0
                - 1.0
            ) * self.mouse_xy_noise
            xyz[:, :2] += torch.as_tensor(
                MOUSE_SPAWN_CENTER,
                device=self.device,
            )
            xyz[:, 2] = self.mouse_z[env_idx]

            yaw = (
                torch.rand(
                    (batch_size,),
                    device=self.device,
                )
                * 2.0
                - 1.0
            ) * self.mouse_yaw_noise
            yaw += float(
                MOUSE_MODEL["yaw_offset"]
            )

            quaternion = torch.zeros(
                (batch_size, 4),
                device=self.device,
            )
            quaternion[:, 0] = torch.cos(
                yaw / 2.0
            )
            quaternion[:, 3] = torch.sin(
                yaw / 2.0
            )

            self._ensure_buffers()
            self.press_started[env_idx] = False
            self.hold_completed[env_idx] = False
            self.released_after_hold[env_idx] = False
            self.success_latched[env_idx] = False

            self.hold_counter[env_idx] = 0
            self.release_counter[env_idx] = 0
            self.final_counter[env_idx] = 0
            self.press_success_counter[env_idx] = 0

            self.spawn_xy[env_idx] = xyz[:, :2]
            self.spawn_yaw[env_idx] = yaw

            for local_i, env_id in enumerate(
                env_idx.tolist()
            ):
                mouse = self._mice[env_id]
                mouse.set_pose(
                    Pose.create_from_pq(
                        p=xyz[local_i],
                        q=quaternion[local_i],
                    )
                )

                qpos = mouse.get_qpos()
                qpos[:, self._left_indices[env_id]] = (
                    self.left_released[env_id]
                )
                qpos[:, self._right_indices[env_id]] = (
                    self.right_released[env_id]
                )
                mouse.set_qpos(qpos)
                mouse.set_qvel(
                    torch.zeros_like(
                        mouse.get_qvel()
                    )
                )

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

    def _ensure_buffers(self):
        if hasattr(self, "press_started"):
            return

        self.press_started = torch.zeros(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device,
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

        self.hold_counter = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self.release_counter = torch.zeros_like(
            self.hold_counter
        )
        self.final_counter = torch.zeros_like(
            self.hold_counter
        )
        self.press_success_counter = torch.zeros_like(
            self.hold_counter
        )

        self.spawn_xy = torch.zeros(
            (self.num_envs, 2),
            device=self.device,
        )
        self.spawn_yaw = torch.zeros(
            (self.num_envs,),
            device=self.device,
        )

    # -------------------------------------------------------------------------
    # Task state
    # -------------------------------------------------------------------------

    def _joint_value(
        self,
        indices: list[int],
        velocity: bool = False,
    ) -> torch.Tensor:
        values = []
        for env_id, mouse in enumerate(self._mice):
            source = (
                mouse.get_qvel()
                if velocity
                else mouse.get_qpos()
            )
            values.append(
                source[0, indices[env_id]]
            )
        return torch.stack(values).to(self.device)

    def _left_qpos(self):
        return self._joint_value(
            self._left_indices
        )

    def _left_qvel(self):
        return self._joint_value(
            self._left_indices,
            velocity=True,
        )

    def _right_qpos(self):
        return self._joint_value(
            self._right_indices
        )

    def _right_qvel(self):
        return self._joint_value(
            self._right_indices,
            velocity=True,
        )

    @staticmethod
    def _fraction(
        qpos: torch.Tensor,
        released: torch.Tensor,
        pressed: torch.Tensor,
    ) -> torch.Tensor:
        return torch.clamp(
            (qpos - released) / (
                pressed - released
            ),
            0.0,
            1.0,
        )

    def _left_fraction(self):
        return self._fraction(
            self._left_qpos(),
            self.left_released,
            self.left_pressed,
        )

    def _right_fraction(self):
        return self._fraction(
            self._right_qpos(),
            self.right_released,
            self.right_pressed,
        )

    def _mouse_root_pose(self):
        return torch.stack(
            [
                vectorize_pose(
                    mouse.get_links()[0].pose
                )[0]
                for mouse in self._mice
            ]
        ).to(self.device)

    def _left_link_pose(self):
        return torch.stack(
            [
                vectorize_pose(link.pose)[0]
                for link in self._left_links
            ]
        ).to(self.device)

    def _task_tip_position(self):
        if self.robot_uids in HAND_ROBOT_UIDS:
            if self.target_finger_id >= len(self.agent.tip_links):
                raise ValueError(
                    f"target_finger_id={self.target_finger_id} exceeds "
                    f"{self.robot_uids} tip count {len(self.agent.tip_links)}"
                )
            # The PAD that presses, not the fingertip frame -- see
            # tasks/office_desk/office_desk_type_keys.py::press_contact_link.
            return press_contact_link(
                self.agent, self.agent.tip_links[self.target_finger_id]
            ).pose.p
        return self.agent.tcp.pose.p

    def _all_tip_positions(self):
        if self.robot_uids not in HAND_ROBOT_UIDS:
            return None
        return torch.stack(
            [link.pose.p for link in self.agent.tip_links],
            dim=1,
        )

    def _button_quantities(self):
        link_p = torch.stack(
            [link.pose.p[0] for link in self._left_links]
        ).to(self.device)
        link_q = torch.stack(
            [link.pose.q[0] for link in self._left_links]
        ).to(self.device)

        top = link_p + quat_rotate_wxyz(
            link_q,
            self.local_top,
        )
        press_axis = quat_rotate_wxyz(
            link_q,
            self.local_press,
        )
        up_axis = quat_rotate_wxyz(
            link_q,
            self.local_up,
        )
        long_axis = quat_rotate_wxyz(
            link_q,
            self.local_long,
        )
        short_axis = quat_rotate_wxyz(
            link_q,
            self.local_short,
        )

        left_fraction = self._left_fraction()
        stroke = torch.abs(
            self.left_pressed - self.left_released
        )
        released_top = (
            top
            - press_axis
            * (left_fraction * stroke).unsqueeze(-1)
        )

        tip = self._task_tip_position()
        relative = tip - top
        relative_released = tip - released_top

        tip_long = torch.sum(
            relative * long_axis,
            dim=1,
        )
        tip_short = torch.sum(
            relative * short_axis,
            dim=1,
        )
        height_released = torch.sum(
            relative_released * up_axis,
            dim=1,
        )

        tip_over_button = (
            (
                torch.abs(tip_long)
                <= self.half_long
                + PRESS_LONG_MARGIN
            )
            & (
                torch.abs(tip_short)
                <= self.half_short
                + PRESS_SHORT_MARGIN
            )
        )

        inside_final_region = (
            (
                torch.abs(tip_long)
                <= self.half_long
                + FINAL_LONG_MARGIN
            )
            & (
                torch.abs(tip_short)
                <= self.half_short
                + FINAL_SHORT_MARGIN
            )
            & (
                height_released
                >= FINAL_MIN_HEIGHT
            )
            & (
                height_released
                <= FINAL_MAX_HEIGHT
            )
        )

        final_center = (
            released_top
            + up_axis
            * 0.5
            * (
                FINAL_MIN_HEIGHT
                + FINAL_MAX_HEIGHT
            )
        )

        return {
            "top": top,
            "released_top": released_top,
            "press_axis": press_axis,
            "up_axis": up_axis,
            "long_axis": long_axis,
            "short_axis": short_axis,
            "tip": tip,
            "tip_long": tip_long,
            "tip_short": tip_short,
            "height_released": height_released,
            "tip_over_button": tip_over_button,
            "inside_final_region": (
                inside_final_region
            ),
            "tip_to_button_dist": torch.linalg.norm(
                tip - top,
                dim=1,
            ),
            "final_center": final_center,
            "final_dist": torch.linalg.norm(
                tip - final_center,
                dim=1,
            ),
        }

    def _phase_one_hot(
        self,
        success: torch.Tensor,
    ) -> torch.Tensor:
        phase = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        phase[self.press_started] = 1
        phase[self.hold_completed] = 2
        phase[self.released_after_hold] = 3
        phase[success] = 4
        return torch.nn.functional.one_hot(
            phase,
            num_classes=5,
        ).float()

    # -------------------------------------------------------------------------
    # Evaluation and observations
    # -------------------------------------------------------------------------

    def evaluate(self):
        q = self._button_quantities()

        left_fraction = self._left_fraction()
        right_fraction = self._right_fraction()

        pressed_enter = (
            left_fraction >= PRESS_ENTER_RATIO
        )
        pressed_hold = (
            left_fraction >= PRESS_HOLD_RATIO
        )
        left_released_now = (
            left_fraction <= RELEASE_RATIO
        )
        right_released_now = (
            right_fraction <= RELEASE_RATIO
        )
        wrong_button = (
            right_fraction >= RIGHT_BUTTON_MAX_RATIO
        )

        valid_enter = (
            pressed_enter
            & q["tip_over_button"]
            & (~wrong_button)
        )
        self.press_started |= valid_enter

        reset_attempt = (
            (~self.hold_completed)
            & (
                (~pressed_hold)
                | (~q["tip_over_button"])
                | wrong_button
            )
        )
        self.press_started = torch.where(
            reset_attempt,
            torch.zeros_like(self.press_started),
            self.press_started,
        )

        valid_hold = (
            self.press_started
            & pressed_hold
            & q["tip_over_button"]
            & (~wrong_button)
            & (~self.hold_completed)
        )
        self.hold_counter = torch.where(
            self.hold_completed,
            self.hold_counter,
            torch.where(
                valid_hold,
                self.hold_counter + 1,
                torch.zeros_like(
                    self.hold_counter
                ),
            ),
        )
        self.hold_completed |= (
            self.hold_counter
            >= PRESS_HOLD_STEPS
        )

        valid_release = (
            self.hold_completed
            & left_released_now
            & right_released_now
        )
        self.release_counter = torch.where(
            self.released_after_hold,
            self.release_counter,
            torch.where(
                valid_release,
                self.release_counter + 1,
                torch.zeros_like(
                    self.release_counter
                ),
            ),
        )
        self.released_after_hold |= (
            self.release_counter >= RELEASE_STEPS
        )

        valid_final = (
            self.released_after_hold
            & q["inside_final_region"]
        )
        self.final_counter = torch.where(
            self.success_latched,
            self.final_counter,
            torch.where(
                valid_final,
                self.final_counter + 1,
                torch.zeros_like(
                    self.final_counter
                ),
            ),
        )

        # Simplified success: a valid left-button press sustained
        # for SUCCESS_PRESS_STEPS consecutive control steps. "Valid" = pressed
        # past PRESS_ENTER_RATIO, target fingertip over the left button, and the
        # right button not mis-pressed. The hold/release/hover phases above are
        # still computed for reward shaping + observations but no longer gate
        # success.
        # One shared implementation with the long task's click stage.
        pressing_now = pressed_enter & q["tip_over_button"] & (~wrong_button)
        self.press_success_counter, _clicked = mouse_click_cycle_update(
            left_fraction, right_fraction, q["tip_over_button"],
            self.press_success_counter, latched=self.success_latched,
        )
        self.success_latched |= _clicked
        success = self.success_latched

        hold_progress = torch.clamp(
            self.hold_counter.float()
            / PRESS_HOLD_STEPS,
            0.0,
            1.0,
        )
        release_progress = torch.clamp(
            self.release_counter.float()
            / RELEASE_STEPS,
            0.0,
            1.0,
        )
        final_progress = torch.clamp(
            self.final_counter.float()
            / FINAL_SETTLE_STEPS,
            0.0,
            1.0,
        )
        press_success_progress = torch.clamp(
            self.press_success_counter.float()
            / SUCCESS_PRESS_STEPS,
            0.0,
            1.0,
        )

        info = {
            "left_qpos": self._left_qpos(),
            "left_qvel": self._left_qvel(),
            "right_qpos": self._right_qpos(),
            "right_qvel": self._right_qvel(),
            "left_press_fraction": left_fraction,
            "right_press_fraction": right_fraction,
            "press_progress": left_fraction,
            "pressed_now": pressed_hold,
            "wrong_button_pressed": wrong_button,
            "press_started": self.press_started,
            "hold_progress": hold_progress,
            "hold_completed": self.hold_completed,
            "left_released_now": left_released_now,
            "right_released_now": right_released_now,
            "release_progress": release_progress,
            "released_after_hold": (
                self.released_after_hold
            ),
            "inside_final_region": q[
                "inside_final_region"
            ],
            "final_progress": final_progress,
            "press_success_progress": (
                press_success_progress
            ),
            "button_top_pos": q["top"],
            "button_released_top_pos": q[
                "released_top"
            ],
            "button_press_axis": q["press_axis"],
            "button_up_axis": q["up_axis"],
            "button_long_axis": q["long_axis"],
            "button_short_axis": q["short_axis"],
            "button_half_extents": torch.stack(
                [self.half_long, self.half_short],
                dim=1,
            ),
            "task_tip_pos": q["tip"],
            "tip_to_button": q["top"] - q["tip"],
            "tip_long": q["tip_long"],
            "tip_short": q["tip_short"],
            "height_above_released": q[
                "height_released"
            ],
            "tip_over_button": q["tip_over_button"],
            "tip_to_button_dist": q[
                "tip_to_button_dist"
            ],
            "final_region_center": q["final_center"],
            "final_region_dist": q["final_dist"],
            # Teleop recorder progress field (same convention as keyboard_type).
            "obj_to_goal_dist": q["tip_to_button_dist"],
            "success": success,
            "task_success": success,
            "is_obj_placed": success,
        }

        all_tips = self._all_tip_positions()
        if all_tips is not None:
            info["all_tip_positions"] = all_tips

        return info

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "mouse_root_pose": self._mouse_root_pose(),
            "left_button_link_pose": (
                self._left_link_pose()
            ),
            "button_top_pos": info["button_top_pos"],
            "button_released_top_pos": info[
                "button_released_top_pos"
            ],
            "button_press_axis": info[
                "button_press_axis"
            ],
            "button_up_axis": info["button_up_axis"],
            "button_long_axis": info[
                "button_long_axis"
            ],
            "button_short_axis": info[
                "button_short_axis"
            ],
            "button_half_extents": info[
                "button_half_extents"
            ],
            "task_tip_pos": info["task_tip_pos"],
            "tip_to_button": info["tip_to_button"],
            "left_qpos": info["left_qpos"],
            "left_qvel": info["left_qvel"],
            "right_qpos": info["right_qpos"],
            "right_qvel": info["right_qvel"],
            "left_press_fraction": info[
                "left_press_fraction"
            ],
            "right_press_fraction": info[
                "right_press_fraction"
            ],
            "tip_long": info["tip_long"],
            "tip_short": info["tip_short"],
            "height_above_released": info[
                "height_above_released"
            ],
            "tip_over_button": info[
                "tip_over_button"
            ].float(),
            "hold_progress": info["hold_progress"],
            "hold_completed": info[
                "hold_completed"
            ].float(),
            "release_progress": info[
                "release_progress"
            ],
            "released_after_hold": info[
                "released_after_hold"
            ].float(),
            "inside_final_region": info[
                "inside_final_region"
            ].float(),
            "final_region_center": info[
                "final_region_center"
            ],
            "final_region_dist": info[
                "final_region_dist"
            ],
            "final_progress": info["final_progress"],
            "task_phase_one_hot": self._phase_one_hot(
                info["success"]
            ),
            "mouse_spawn_xy": self.spawn_xy,
            "mouse_spawn_yaw": self.spawn_yaw,
        }

        if "all_tip_positions" in info:
            obs["all_tip_positions"] = info[
                "all_tip_positions"
            ].reshape(self.num_envs, -1)
            obs["target_finger_id"] = torch.full(
                (self.num_envs, 1),
                float(self.target_finger_id),
                device=self.device,
            )

        # ManiSkill state flattening requires each leaf to be (B, D).
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
        """Persist physical state plus click/release diagnostic latches."""
        state = super().get_state_dict()
        state["office_mouse_task_state"] = torch.cat(
            [
                self.press_started[:, None].float(),
                self.hold_completed[:, None].float(),
                self.released_after_hold[:, None].float(),
                self.success_latched[:, None].float(),
                self.hold_counter[:, None].float(),
                self.release_counter[:, None].float(),
                self.final_counter[:, None].float(),
                self.press_success_counter[:, None].float(),
                self.spawn_xy.float(),
                self.spawn_yaw[:, None].float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("office_mouse_task_state", None)
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
                f"office_mouse_task_state must have 11 fields, "
                f"got {task_state.shape[-1]}"
            )
        self.press_started[restore_idx] = task_state[:, 0].bool()
        self.hold_completed[restore_idx] = task_state[:, 1].bool()
        self.released_after_hold[restore_idx] = task_state[:, 2].bool()
        self.success_latched[restore_idx] = task_state[:, 3].bool()
        self.hold_counter[restore_idx] = task_state[:, 4].to(torch.long)
        self.release_counter[restore_idx] = task_state[:, 5].to(torch.long)
        self.final_counter[restore_idx] = task_state[:, 6].to(torch.long)
        self.press_success_counter[restore_idx] = task_state[:, 7].to(torch.long)
        self.spawn_xy[restore_idx] = task_state[:, 8:10].float()
        self.spawn_yaw[restore_idx] = task_state[:, 10].float()

    # -------------------------------------------------------------------------
    # Rewards
    # -------------------------------------------------------------------------

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
        before_hold = (
            ~info["hold_completed"]
        ).float()
        after_hold = info[
            "hold_completed"
        ].float()
        after_release = info[
            "released_after_hold"
        ].float()

        reaching = (
            1.0
            - torch.tanh(
                10.0
                * info["tip_to_button_dist"]
            )
        ) * before_hold

        over_button = (
            info["tip_over_button"].float()
            * before_hold
            * 0.5
        )

        pressing = (
            info["left_press_fraction"]
            * info["tip_over_button"].float()
            * before_hold
            * 3.0
        )

        hold = (
            info["hold_progress"]
            * before_hold
            * 2.0
        )

        release = (
            torch.clamp(
                1.0
                - info["left_press_fraction"]
                / RELEASE_RATIO,
                0.0,
                1.0,
            )
            * after_hold
            * (
                ~info["released_after_hold"]
            ).float()
            * 2.0
        )

        final_reach = (
            1.0
            - torch.tanh(
                12.0
                * info["final_region_dist"]
            )
        ) * after_release * 2.5

        final_hold = (
            info["final_progress"]
            * after_release
            * 1.5
        )

        wrong_button_penalty = (
            info["right_press_fraction"] * 1.5
        )

        reward = (
            reaching
            + over_button
            + pressing
            + hold
            + release
            + final_reach
            + final_hold
            - wrong_button_penalty
        )

        info["reward_terms"] = {
            "reaching_reward": reaching,
            "over_button_bonus": over_button,
            "press_reward": pressing,
            "hold_reward": hold,
            "release_reward": release,
            "final_reach_reward": final_reach,
            "final_hold_reward": final_hold,
            "wrong_button_penalty": (
                wrong_button_penalty
            ),
            "dense_reward_no_success_bonus": (
                reward.clone()
            ),
        }

        return torch.where(
            info["task_success"],
            torch.full_like(
                reward,
                SUCCESS_REWARD,
            ),
            reward,
        )

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        return self.compute_dense_reward(
            obs,
            action,
            info,
        ) / SUCCESS_REWARD


# =============================================================================
# Standalone check
# =============================================================================

def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(np.asarray(value).mean())


def main():
    print(
        "### CODE_VERSION_MARKER: "
        "mouse_click_v1_sharpa_mobile_static ###"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        "--env_id",
        dest="env_id",
        default="OfficeDeskClickMouse-Sharpa-v1",
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
        default="sparse",
        choices=["sparse", "none"],
    )
    parser.add_argument(
        "--target-finger-id",
        type=int,
        default=1,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--video-dir",
        default="tasks/videos_office_mouse_click_sharpa_mobile_static",
    )
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--debug-print", action="store_true")
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        render_mode="rgb_array",
        target_finger_id=args.target_finger_id,
    )

    if not args.no_record:
        env = RecordEpisode(
            env,
            output_dir=args.video_dir,
            save_trajectory=True,
            trajectory_name="trajectory",
            save_video_trigger=lambda episode_idx: True,
            max_steps_per_video=args.steps,
            video_fps=20,
        )

    print("Environment:", args.env_id)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Robot DoF:", env.unwrapped.agent.robot.dof)

    obs, info = env.reset(seed=args.seed)

    if args.control_mode == "pd_joint_pos":
        hold_action = (
            env.unwrapped.agent.robot.get_qpos()
            .detach()
            .clone()
        )
    elif "delta" in args.control_mode:
        hold_action = torch.zeros(
            (
                1,
                int(env.action_space.shape[-1]),
            ),
            device=env.unwrapped.device,
        )
    else:
        raise RuntimeError(
            "Stationary test supports pd_joint_pos or a delta control mode."
        )

    if (
        hold_action.shape[-1]
        != env.action_space.shape[-1]
    ):
        raise RuntimeError(
            "Action dimension mismatch: "
            f"action={env.action_space.shape[-1]}, "
            f"hold={hold_action.shape[-1]}"
        )

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = (
            env.step(hold_action)
        )
        env.render()

        if args.debug_print and (
            step % 5 == 0
            or bool(
                info["task_success"][0].item()
            )
        ):
            print(
                f"step={step:04d} "
                f"reward={_scalar(reward):.3f} "
                f"left={_scalar(info['left_press_fraction']):.3f} "
                f"right={_scalar(info['right_press_fraction']):.3f} "
                f"hold={_scalar(info['hold_progress']):.2f} "
                f"released={_scalar(info['released_after_hold']):.0f} "
                f"height={_scalar(info['height_above_released']):.4f} "
                f"final={_scalar(info['inside_final_region']):.0f} "
                f"success={_scalar(info['task_success']):.0f}"
            )

        done = bool(terminated[0].item()) or bool(
            truncated[0].item()
        )
        if done:
            break

    env.close()
    print(f"Saved to: {args.video_dir}")


if __name__ == "__main__":
    main()

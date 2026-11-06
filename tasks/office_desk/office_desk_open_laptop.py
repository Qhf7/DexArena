from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np
import sapien
import torch

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
from tasks.config.workspace_rig import (
    WORK_TABLE_X_OFFSET,
    apply_arm_init_qpos,
    platform_root_pose,
    set_work_table_pose,
)
from tasks.config.mass_properties import set_mass as set_realizable_mass
from tasks.config.sim_profiles import make_sim_config
from tasks.scene_builders import TableSceneBuilder
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag


# =============================================================================
# Configuration
# =============================================================================

_OFFICE_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "task_object_meshes" / "office"
LAPTOP_ASSET_DIR = Path(
    os.environ.get(
        "LAPTOP_ASSET_DIR",
        str(_OFFICE_ASSET_DIR),
    )
)

LAPTOP_MODELS = {
    "10211": {
        "scale": 0.3,
        "hinge_joint_index": 0,
        "qpos_closed": -1.834690109696439,
        "qpos_open": 0.47123889803846897,
        "target_angle_deg": 90.0,
        "maintain_angle_deg": 85.0,
        "start_angle_min_deg": 40.0,
        "start_angle_max_deg": 50.0,
        "hinge_damping": 0.20,
        "hinge_force_limit": 2.0,
        "hinge_friction": 1.1,
    }
}

MODEL_ID = "10211"
CONTROL_FREQUENCY = 20
SUPPORTED_CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")
DEFAULT_CONTROL_MODE = "pd_joint_pos"
TARGET_HOLD_STEPS = 4

# =============================================================================
# Stage predicate, shared with the long task
# =============================================================================
#
# `office.py` used to carry its own copy of this, written with different local
# names but the same shape. The thresholds were already shared by import, so the
# numbers could not drift -- but nothing stopped the STRUCTURE from drifting,
# and that is where this family's real bugs have lived. One implementation now
# serves both, so the subtask is a strict subset of the long-task stage by
# construction rather than by inspection.
def lid_open_hold_update(
    opening_angle,
    reached_once,
    hold_counter,
    *,
    target_angle: float,
    maintain_angle: float,
    hold_steps: int,
    active=None,
):
    """Advance the "lid reached the target angle and stayed there" state.

    `active` is the long task's stage gate: the counter only runs while this
    stage is the current one. A standalone stage passes None (always active).

    Returns (reached_once, hold_counter, success).
    """
    import torch as _torch

    if active is None:
        active = _torch.ones_like(reached_once)
    reached_once = reached_once | (active & (opening_angle >= target_angle))
    holding = opening_angle >= maintain_angle
    hold_counter = _torch.where(
        active & reached_once & holding,
        hold_counter + 1,
        _torch.zeros_like(hold_counter),
    )
    return reached_once, hold_counter, active & (hold_counter >= hold_steps)

NEAR_EDGE_THRESHOLD = 0.07
SUCCESS_REWARD = 10.0

# Model 10211 is made from thin, open surface meshes. Convexifying those
# surfaces fills the empty space between the keyboard deck and the lid,
# producing an invisible wedge that blocks the hinge near 70 degrees, so each
# shell link gets ONE box instead.
#
# Those boxes used to be hand-authored constants, and the LID's had drifted off
# its own mesh: measured on the built scene, the lid's collision box sat 36 mm
# forward and 64 mm above the panel you can see (scale 0.3). The operator's
# hand therefore passed straight through the visible screen and hit nothing --
# and, in the other direction, bumped an invisible slab above it.
#
# They are now DERIVED from the asset's own collision meshes at load time
# (`laptop_shell_boxes`), and the derivation asserts that every mesh vertex
# lands inside the box it produced, so a box can never drift off its mesh
# again. The shell links are named here rather than discovered so that a model
# swap fails loudly instead of silently leaving a link uncovered.
LAPTOP_SHELL_LINKS = ("link_0", "link_1")


def _load_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64)


def _matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        w = math.sqrt(1.0 + trace) / 2.0
        x = (rot[2, 1] - rot[1, 2]) / (4.0 * w)
        y = (rot[0, 2] - rot[2, 0]) / (4.0 * w)
        z = (rot[1, 0] - rot[0, 1]) / (4.0 * w)
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            x = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) / 2.0
            w = (rot[2, 1] - rot[1, 2]) / (4.0 * x)
            y = (rot[0, 1] + rot[1, 0]) / (4.0 * x)
            z = (rot[0, 2] + rot[2, 0]) / (4.0 * x)
        elif idx == 1:
            y = math.sqrt(1.0 - rot[0, 0] + rot[1, 1] - rot[2, 2]) / 2.0
            w = (rot[0, 2] - rot[2, 0]) / (4.0 * y)
            x = (rot[0, 1] + rot[1, 0]) / (4.0 * y)
            z = (rot[1, 2] + rot[2, 1]) / (4.0 * y)
        else:
            z = math.sqrt(1.0 - rot[0, 0] - rot[1, 1] + rot[2, 2]) / 2.0
            w = (rot[1, 0] - rot[0, 1]) / (4.0 * z)
            x = (rot[0, 2] + rot[2, 0]) / (4.0 * z)
            y = (rot[1, 2] + rot[2, 1]) / (4.0 * z)
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _tightest_shell_box(points: np.ndarray):
    """Smallest of {axis-aligned box, PCA box} around `points`, in that frame.

    The keyboard deck is axis-aligned in its link frame, so its AABB is the
    tighter of the two; the lid is tilted in its own, so the PCA box wins by
    4x in volume. Picking by volume means neither link needs a special case.
    """
    aabb_center = (points.min(0) + points.max(0)) / 2.0
    aabb_half = (points.max(0) - points.min(0)) / 2.0
    aabb = (aabb_center, aabb_half, np.array([1.0, 0.0, 0.0, 0.0]))

    centred = points - points.mean(0)
    rot = np.linalg.svd(centred, full_matrices=False)[2].T
    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1
    local = centred @ rot
    lo, hi = local.min(0), local.max(0)
    obb = (
        points.mean(0) + rot @ ((hi + lo) / 2.0),
        (hi - lo) / 2.0,
        _matrix_to_quat_wxyz(rot),
    )
    return min(aabb, obb, key=lambda box: float(np.prod(box[1])))


def _display_plane_points(root, link_name: str, source: Path):
    """Vertices of the visible SCREEN panel of a laptop link, or None.

    Model 10211 names its lid visuals `shaft-*`, `screen-*` and
    `screen_frame-*`; only the second is the panel the operator looks at.
    """
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        return None
    chunks = []
    for visual in link.findall("visual"):
        name = visual.get("name", "")
        if not name.startswith("screen") or name.startswith("screen_frame"):
            continue
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        origin = visual.find("origin")
        offset = np.zeros(3)
        if origin is not None:
            offset = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        chunks.append(_load_obj_vertices(source.parent / mesh.get("filename")) + offset)
    return np.vstack(chunks) if chunks else None


def _clip_box_front_to_display(center, half, quat, display):
    """Pull the shell box's FRONT face back onto the display plane.

    The lid's own shell is ~13 mm thick and the display plane sits recessed
    inside it, so a box that merely contains the shell puts 6.3 mm of solid
    (scale 0.3) in front of the pixels: the hand stops in mid-air short of the
    screen it can see. The bezel that legitimately stands proud of the display
    is a thin rim, not a slab, and a single box cannot represent it -- so the
    box gives up the rim and gains a front face exactly where the display is.

    "Front" is the thin-axis side the display faces, which is the side AWAY
    from the hinge shaft (the rest of the shell is behind the panel).
    """
    rot = _quat_to_matrix(quat)
    thin = int(np.argmin(half))
    axis = rot[:, thin]
    local_half = (display - center) @ axis
    lo, hi = float(local_half.min()), float(local_half.max())
    # The display sits nearer one face; that face is the one to pull in.
    if abs(hi - half[thin]) < abs(-half[thin] - lo):
        new_hi, new_lo = hi, -half[thin]
    else:
        new_hi, new_lo = half[thin], lo
    shift = (new_hi + new_lo) / 2.0
    out_half = half.copy()
    out_half[thin] = (new_hi - new_lo) / 2.0
    return center + axis * shift, out_half


def _quat_to_matrix(quat):
    w, x, y, z = [float(v) for v in quat]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@lru_cache(maxsize=4)
def laptop_shell_boxes(urdf_path: str) -> dict:
    """One shell box per laptop link, in that link's own UNSCALED frame.

    Returns ``{link_name: (center, half_size, quat_wxyz)}``. Raises if a shell
    link has no collision mesh, or if the box it produced fails to contain the
    mesh it was derived from.
    """
    source = Path(urdf_path)
    root = ET.parse(source).getroot()
    boxes = {}
    for link_name in LAPTOP_SHELL_LINKS:
        link = root.find(f"./link[@name='{link_name}']")
        if link is None:
            raise RuntimeError(f"{source}: laptop shell link {link_name!r} is missing")
        chunks = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            origin = collision.find("origin")
            offset = np.zeros(3)
            if origin is not None:
                offset = np.array(
                    [float(v) for v in origin.get("xyz", "0 0 0").split()]
                )
            chunks.append(_load_obj_vertices(source.parent / mesh.get("filename")) + offset)
        if not chunks:
            raise RuntimeError(
                f"{source}: laptop shell link {link_name!r} has no collision mesh"
            )
        points = np.vstack(chunks)
        center, half, quat = _tightest_shell_box(points)
        display = _display_plane_points(root, link_name, source)
        if display is not None:
            center, half = _clip_box_front_to_display(center, half, quat, display)
        rot = np.linalg.inv(
            np.array(
                [
                    [1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
                     2 * (quat[1] * quat[2] - quat[3] * quat[0]),
                     2 * (quat[1] * quat[3] + quat[2] * quat[0])],
                    [2 * (quat[1] * quat[2] + quat[3] * quat[0]),
                     1 - 2 * (quat[1] ** 2 + quat[3] ** 2),
                     2 * (quat[2] * quat[3] - quat[1] * quat[0])],
                    [2 * (quat[1] * quat[3] - quat[2] * quat[0]),
                     2 * (quat[2] * quat[3] + quat[1] * quat[0]),
                     1 - 2 * (quat[1] ** 2 + quat[2] ** 2)],
                ]
            )
        )
        outside = float(
            np.maximum(np.abs((points - center) @ rot.T) - half, 0.0).max()
        )
        # The lid box is deliberately clipped to the display plane, so its own
        # bezel rim now sits outside it; only the UNclipped links must contain
        # every vertex.
        if display is None and outside > 1e-6:
            raise RuntimeError(
                f"{source}: shell box for {link_name!r} leaves mesh vertices "
                f"{outside * 1000:.3f} mm outside it"
            )
        boxes[link_name] = (
            center.astype(np.float32),
            half.astype(np.float32),
            quat.astype(np.float32),
        )
    return boxes

HAND_ROBOT_UIDS = {
    "panda_sharpa_hand_right_mobile_base",
    "panda_sharpa_hand_right_angled_mobile_base",
    "panda_wuji_hand_right_angled_mobile_base",
    "panda_xhand_hand_right_angled_mobile_base",
}

GRIPPER_ROBOT_UID = "panda_gripper_right_mobile_base"
DEFAULT_ROBOT_UID = "panda_sharpa_hand_right_angled_mobile_base"
# LOCKED to the shared rig (tasks/config/workspace_rig.py). Two drifts fixed
# here: the trolley stood at -0.615 + WORLD_X_OFFSET = -1.145, i.e. 25 mm
# CLOSER to the desk than the office long task's own -1.170 (the objects sat at
# identical world x in both, so the subtask handed the operator a different
# reach envelope than the long task it is a subset of); and the whole family
# was anchored on the STOCK table centre while CabinetPour/Screwing use the
# +0.30 shifted one. The scene now translates with the table and the trolley.
WORLD_X_OFFSET = -0.53 + WORK_TABLE_X_OFFSET
OFFICE_GROUP_X_OFFSET = 0.20
# SINGLE SOURCE OF TRUTH for the laptop's world x. office.py imports this
# rather than repeating the arithmetic: the two used to be independent copies
# joined only by a "must track" comment, and exactly that comment
# was not honoured -- the long task moved and the subtask did not.
#
# Pulled 0.10 m towards the arm on operator request, then 0.017 m of that given
# back so the keyboard could clear the laptop base without pushing its nearest
# target key inside the arm's fold-in limit (see office_keyboard.py).
LAPTOP_INWARD_OFFSET = -0.033
LAPTOP_X = WORLD_X_OFFSET + OFFICE_GROUP_X_OFFSET + LAPTOP_INWARD_OFFSET
ROBOT_ROOT_POSE = platform_root_pose()
LAPTOP_SPAWN_CENTER = np.array([LAPTOP_X, 0.0], dtype=np.float32)


# =============================================================================
# Geometry
# =============================================================================

def _link_collision_mesh(link, to_world_frame: bool = True):
    meshes = []
    for obj in link._objs:
        mesh = merge_meshes(get_component_meshes(obj))
        if mesh is not None:
            meshes.append(mesh)

    if not meshes:
        return None

    if to_world_frame:
        pose = link.pose
        for i, mesh in enumerate(meshes):
            matrix = (
                pose[i].sp.to_transformation_matrix()
                if len(pose) > 1
                else pose.sp.to_transformation_matrix()
            )
            mesh.apply_transform(matrix)

    return merge_meshes(meshes)


def quat_rotate_wxyz(
    quaternion: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    xyz = quaternion[..., 1:]
    w = quaternion[..., :1]
    uv = torch.cross(xyz, vector, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (w * uv + uuv)


def _screen_edge_from_local_mesh(
    screen_link,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the free edge centre from the screen collision-mesh AABB."""
    mesh = _link_collision_mesh(
        screen_link,
        to_world_frame=False,
    )
    if mesh is None or len(mesh.vertices) == 0:
        zeros = np.zeros(3, dtype=np.float32)
        return zeros, zeros

    bounds = np.asarray(
        mesh.bounding_box.bounds,
        dtype=np.float64,
    )
    lower, upper = bounds
    extent = upper - lower

    thin_axis = int(np.argmin(extent))
    width_axis = int(np.argmax(extent))
    height_axis = next(
        i
        for i in range(3)
        if i not in {thin_axis, width_axis}
    )

    edge = 0.5 * (lower + upper)
    edge[height_axis] = (
        upper[height_axis]
        if abs(upper[height_axis]) >= abs(lower[height_axis])
        else lower[height_axis]
    )
    return edge.astype(np.float32), extent.astype(np.float32)


def _use_simplified_laptop_collisions(builder, scale: float, urdf_path) -> None:
    """Replace model-10211 surface hulls with shell boxes taken from its meshes.

    ``urdf_path`` is required: the boxes are derived from the asset rather than
    from constants, so that the collision shell can never sit somewhere other
    than the geometry the operator can see (see LAPTOP_SHELL_LINKS above).
    """
    links = {
        link_builder.name: link_builder
        for link_builder in builder.link_builders
    }
    missing = set(LAPTOP_SHELL_LINKS) - links.keys()
    if missing:
        raise RuntimeError(
            "Laptop collision replacement could not find links: "
            + ", ".join(sorted(missing))
        )

    boxes = laptop_shell_boxes(str(urdf_path))
    for link_name in LAPTOP_SHELL_LINKS:
        link_builder = links[link_name]
        material = (
            link_builder.collision_records[0].material
            if link_builder.collision_records
            else None
        )
        center, half, quat = boxes[link_name]
        link_builder.collision_records.clear()
        link_builder.add_box_collision(
            pose=sapien.Pose(p=center * scale, q=quat),
            half_size=half * scale,
            material=material,
            density=1000.0,
        )


# Explicit shell masses. A 14" laptop weighs 1.3-1.8 kg in total, split
# roughly 2:1 between the keyboard base (battery + board) and the screen lid.
# The simplified collision boxes previously took the PhysX default density, so
# both halves came out at ~0.93 kg -- a screen lid twice as heavy as any real
# panel, which is also what the hinge has to hold.
LAPTOP_BASE_MASS = 1.00    # kg
LAPTOP_SCREEN_MASS = 0.50  # kg


def _set_component_mass(component, mass: float) -> None:
    """Set an explicit mass with a realizable inertia (see tasks/config/mass_properties)."""
    set_realizable_mass(component, mass)


def apply_laptop_masses(articulation) -> None:
    """Give the two laptop shells their explicit masses.

    Kept SEPARATE from _configure_task_articulation: office.py reuses that
    configurator for the keyboard and the mouse as well, so folding laptop
    masses into it silently made every keycap 0.5 kg.
    """
    screen_names = set()
    for raw_art in articulation._objs:
        for joint in raw_art.get_active_joints():
            if joint.child_link is not None:
                screen_names.add(str(joint.child_link.name))
    for link in articulation.get_links():
        for component in link._objs:
            if hasattr(component, "mass") and hasattr(component, "inertia"):
                is_screen = str(getattr(component, "name", "")) in screen_names
                _set_component_mass(
                    component,
                    LAPTOP_SCREEN_MASS if is_screen else LAPTOP_BASE_MASS,
                )


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
    "OfficeDeskOpenLaptop",
    side="right",
    max_episode_steps=600,
    obs_mode="state",
    control_mode="pd_joint_pos",
    reward_mode="sparse",
    render_mode="rgb_array",
)
class OfficeDeskOpenLaptopEnv(BaseEnv):
    """Open the laptop lid and keep it above the target angle for 0.2 s."""

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
        laptop_xy_noise: float = 0.02,
        laptop_yaw_noise_deg: float = 5.0,
        laptop_scale: float | None = None,
        start_angle_min_deg: float | None = None,
        start_angle_max_deg: float | None = None,
        arm_init_qpos=None,
        collision_decomposition: str = "simplified",
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        runtime_sim_config = kwargs.pop("sim_config", None)
        if runtime_sim_config not in (None, {"control_freq": CONTROL_FREQUENCY}):
            raise ValueError(
                "OpenLaptop freezes the Plant configuration in "
                "tasks/config/sim_profiles.py; only the teleop compatibility declaration "
                "{'control_freq': 20} is accepted and ignored"
            )
        cfg = dict(LAPTOP_MODELS[MODEL_ID])
        if laptop_scale is not None:
            cfg["scale"] = float(laptop_scale)
        if start_angle_min_deg is not None:
            cfg["start_angle_min_deg"] = float(start_angle_min_deg)
        if start_angle_max_deg is not None:
            cfg["start_angle_max_deg"] = float(start_angle_max_deg)
        if cfg["start_angle_max_deg"] < cfg["start_angle_min_deg"]:
            raise ValueError("start_angle_max_deg < start_angle_min_deg")
        self._cfg = cfg
        if collision_decomposition not in {"simplified", "none", "coacd"}:
            raise ValueError(
                "collision_decomposition must be 'simplified', 'none', or 'coacd'"
            )
        self.collision_decomposition = collision_decomposition

        if arm_init_qpos not in (None, "keyframe"):
            raise ValueError(
                "OpenLaptop uses the robot's locked rest keyframe; "
                "task-level arm_init_qpos overrides are not allowed"
            )

        self.robot_init_qpos_noise = float(
            robot_init_qpos_noise
        )
        self.target_finger_id = int(target_finger_id)
        if self.target_finger_id < 0:
            raise ValueError("target_finger_id must be non-negative")
        self.laptop_xy_noise = float(laptop_xy_noise)
        if self.laptop_xy_noise < 0:
            raise ValueError("laptop_xy_noise must be non-negative")
        self.laptop_yaw_noise = np.deg2rad(
            float(laptop_yaw_noise_deg)
        )
        if self.laptop_yaw_noise < 0:
            raise ValueError("laptop_yaw_noise_deg must be non-negative")
        if float(cfg["scale"]) <= 0:
            raise ValueError("laptop_scale must be positive")

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
    def _default_sim_config(self):
        return make_sim_config(
            spacing=3,
            max_rigid_contact_count=2**20,
            max_rigid_patch_count=2**19,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.35 + WORLD_X_OFFSET, -0.35, 0.55],
            target=[float(LAPTOP_SPAWN_CENTER[0]), 0.0, 0.16],
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
            eye=[0.55 + WORLD_X_OFFSET, 0.65, 0.55],
            target=[float(LAPTOP_SPAWN_CENTER[0]), 0.0, 0.18],
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
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        set_work_table_pose(self)

        cfg = self._cfg
        urdf_path = (
            LAPTOP_ASSET_DIR
            / MODEL_ID
            / "mobility.urdf"
        )
        if not urdf_path.is_file():
            raise FileNotFoundError(
                f"Laptop URDF not found: {urdf_path}"
            )

        self._laptops = []
        self._screen_links = []

        for env_id in range(self.num_envs):
            loader = self.scene.create_urdf_loader()
            loader.name = f"laptop-{env_id}"
            loader.fix_root_link = True
            loader.scale = float(cfg["scale"])
            loader.load_multiple_collisions_from_file = True
            # Model 10211 contains 82 collision meshes. Most cluster copies do
            # not ship all corresponding *.coacd.ply caches, and generating
            # them here serially can stall environment creation for hours.
            # Parse quickly with ordinary convex loading, then replace those
            # records with the stable boxes below. COACD remains opt-in after
            # its caches have been precomputed.
            loader.multiple_collisions_decomposition = (
                "none"
                if self.collision_decomposition == "simplified"
                else self.collision_decomposition
            )

            builders = loader.parse(
                str(urdf_path)
            )["articulation_builders"]
            if not builders:
                raise RuntimeError(
                    f"No articulation parsed from {urdf_path}"
                )

            builder = builders[0]
            if self.collision_decomposition == "simplified":
                _use_simplified_laptop_collisions(
                    builder,
                    float(cfg["scale"]),
                    urdf_path,
                )
            builder.set_scene_idxs([env_id])
            builder.initial_pose = sapien.Pose()

            laptop = builder.build(
                name=f"laptop-{env_id}"
            )
            self.remove_from_state_dict_registry(laptop)
            _configure_task_articulation(laptop)
            apply_laptop_masses(laptop)

            active_joints = laptop.get_active_joints()
            hinge_index = int(cfg["hinge_joint_index"])
            if hinge_index >= len(active_joints):
                raise RuntimeError(
                    f"Invalid hinge index {hinge_index}"
                )

            hinge = active_joints[hinge_index]
            hinge.set_drive_properties(
                stiffness=0.0,
                damping=float(cfg["hinge_damping"]),
                force_limit=float(cfg["hinge_force_limit"]),
                mode="force",
            )
            hinge_friction = float(cfg.get("hinge_friction", 0.0))
            if hinge_friction > 0.0 and hasattr(hinge, "set_friction"):
                hinge.set_friction(hinge_friction)

            links = laptop.get_links()
            if len(links) < 2:
                raise RuntimeError(
                    "Laptop must have a base and a screen link"
                )

            self._laptops.append(laptop)
            self._screen_links.append(links[-1])
        self.laptop = Articulation.merge(self._laptops, name="laptop", merge_links=True)
        self.add_to_state_dict_registry(self.laptop)


    def _after_reconfigure(self, options: dict):
        cfg = self._cfg

        self.qpos_closed = torch.full(
            (self.num_envs,),
            float(cfg["qpos_closed"]),
            device=self.device,
        )
        self.qpos_open = torch.full(
            (self.num_envs,),
            float(cfg["qpos_open"]),
            device=self.device,
        )
        self.target_angle = torch.full(
            (self.num_envs,),
            np.deg2rad(float(cfg["target_angle_deg"])),
            device=self.device,
        )
        self.maintain_angle = torch.full(
            (self.num_envs,),
            np.deg2rad(float(cfg["maintain_angle_deg"])),
            device=self.device,
        )

        laptop_z = []
        screen_edge_local = []
        screen_extent_local = []

        for env_id, laptop in enumerate(self._laptops):
            qpos = laptop.get_qpos()
            qpos[:, 0] = self.qpos_closed[env_id]
            laptop.set_qpos(qpos)
            laptop.set_qvel(
                torch.zeros_like(laptop.get_qvel())
            )
            laptop.set_pose(sapien.Pose())

            min_z = None
            for link in laptop.get_links():
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

            laptop_z.append(
                (-min_z if min_z is not None else 0.0)
                + 0.004
            )

            edge, extent = _screen_edge_from_local_mesh(
                self._screen_links[env_id]
            )
            screen_edge_local.append(edge)
            screen_extent_local.append(extent)

        self.laptop_z = common.to_tensor(
            laptop_z,
            device=self.device,
        )
        self.screen_edge_local = common.to_tensor(
            np.stack(screen_edge_local),
            device=self.device,
        )
        self.screen_extent_local = common.to_tensor(
            np.stack(screen_extent_local),
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
            xyz[:, :2] = common.to_tensor(
                LAPTOP_SPAWN_CENTER,
                device=self.device,
            ) + (
                torch.rand(
                    (batch_size, 2),
                    device=self.device,
                )
                * 2.0
                - 1.0
            ) * self.laptop_xy_noise
            xyz[:, 2] = self.laptop_z[env_idx]

            yaw = (
                torch.rand(
                    (batch_size,),
                    device=self.device,
                )
                * 2.0
                - 1.0
            ) * self.laptop_yaw_noise

            quaternion = torch.zeros(
                (batch_size, 4),
                device=self.device,
            )
            quaternion[:, 0] = torch.cos(yaw / 2.0)
            quaternion[:, 3] = torch.sin(yaw / 2.0)

            self._ensure_buffers()
            self.reached_once[env_idx] = False
            self.hold_counter[env_idx] = 0
            self.success_latched[env_idx] = False
            self.spawn_xy[env_idx] = xyz[:, :2]
            self.spawn_yaw[env_idx] = yaw

            cfg = self._cfg
            open_direction = np.sign(
                float(cfg["qpos_open"])
                - float(cfg["qpos_closed"])
            )

            for local_i, env_id in enumerate(
                env_idx.tolist()
            ):
                laptop = self._laptops[env_id]
                laptop.set_pose(
                    Pose.create_from_pq(
                        p=xyz[local_i],
                        q=quaternion[local_i],
                    )
                )

                start_angle = np.deg2rad(
                    self._episode_rng.uniform(
                        cfg["start_angle_min_deg"],
                        cfg["start_angle_max_deg"],
                    )
                )
                start_qpos = (
                    float(cfg["qpos_closed"])
                    + open_direction * start_angle
                )

                qpos = laptop.get_qpos()
                qpos[:, 0] = start_qpos
                laptop.set_qpos(qpos)
                laptop.set_qvel(
                    torch.zeros_like(laptop.get_qvel())
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
        if hasattr(self, "reached_once"):
            return

        self.reached_once = torch.zeros(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )
        self.hold_counter = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self.success_latched = torch.zeros_like(
            self.reached_once
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

    def _hinge_qpos(self) -> torch.Tensor:
        return torch.stack(
            [
                laptop.get_qpos()[0, 0]
                for laptop in self._laptops
            ]
        ).to(self.device)

    def _hinge_qvel(self) -> torch.Tensor:
        return torch.stack(
            [
                laptop.get_qvel()[0, 0]
                for laptop in self._laptops
            ]
        ).to(self.device)

    def _opening_angle(self) -> torch.Tensor:
        return torch.abs(
            self._hinge_qpos() - self.qpos_closed
        )

    def _open_fraction(self) -> torch.Tensor:
        span = self.qpos_open - self.qpos_closed
        return torch.clamp(
            (
                self._hinge_qpos()
                - self.qpos_closed
            )
            / span,
            0.0,
            1.0,
        )

    def _screen_pose(self) -> torch.Tensor:
        return torch.stack(
            [
                vectorize_pose(link.pose)[0]
                for link in self._screen_links
            ]
        ).to(self.device)

    def _screen_edge_position(self) -> torch.Tensor:
        position = torch.stack(
            [link.pose.p[0] for link in self._screen_links]
        ).to(self.device)
        quaternion = torch.stack(
            [link.pose.q[0] for link in self._screen_links]
        ).to(self.device)
        return position + quat_rotate_wxyz(
            quaternion,
            self.screen_edge_local,
        )

    def _laptop_root_pose(self) -> torch.Tensor:
        return torch.stack(
            [
                vectorize_pose(
                    laptop.get_links()[0].pose
                )[0]
                for laptop in self._laptops
            ]
        ).to(self.device)

    def _task_tip_position(self) -> torch.Tensor:
        if self.robot_uids in HAND_ROBOT_UIDS:
            if self.target_finger_id >= len(self.agent.tip_links):
                raise ValueError(
                    f"target_finger_id={self.target_finger_id} exceeds "
                    f"{self.robot_uids} tip count {len(self.agent.tip_links)}"
                )
            return self.agent.tip_links[self.target_finger_id].pose.p
        return self.agent.tcp.pose.p

    def _all_tip_positions(self):
        if self.robot_uids not in HAND_ROBOT_UIDS:
            return None
        return torch.stack(
            [link.pose.p for link in self.agent.tip_links],
            dim=1,
        )

    def _phase_one_hot(
        self,
        success: torch.Tensor,
    ) -> torch.Tensor:
        phase = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        phase[self.reached_once] = 1
        phase[success] = 2
        return torch.nn.functional.one_hot(
            phase,
            num_classes=3,
        ).float()

    # -------------------------------------------------------------------------
    # Evaluation and observations
    # -------------------------------------------------------------------------

    def evaluate(self):
        hinge_qpos = self._hinge_qpos()
        hinge_qvel = self._hinge_qvel()
        opening_angle = self._opening_angle()

        # One shared implementation with the long task's stage 0.
        self.reached_once, self.hold_counter, reached_hold = lid_open_hold_update(
            opening_angle,
            self.reached_once,
            self.hold_counter,
            target_angle=self.target_angle,
            maintain_angle=self.maintain_angle,
            hold_steps=TARGET_HOLD_STEPS,
        )
        # Reported for observations / dense reward; the gate itself is the
        # helper's return value above.
        target_reached_now = opening_angle >= self.target_angle
        maintained = opening_angle >= self.maintain_angle
        hold_progress = torch.clamp(
            self.hold_counter.float() / TARGET_HOLD_STEPS, 0.0, 1.0
        )
        self.success_latched |= reached_hold
        success = self.success_latched

        screen_edge = self._screen_edge_position()
        task_tip = self._task_tip_position()
        tip_to_edge = screen_edge - task_tip
        tip_distance = torch.linalg.norm(
            tip_to_edge,
            dim=1,
        )

        info = {
            "hinge_qpos": hinge_qpos,
            "hinge_qvel": hinge_qvel,
            "open_fraction": self._open_fraction(),
            "opening_angle_rad": opening_angle,
            "opening_angle_deg": torch.rad2deg(
                opening_angle
            ),
            "target_angle_rad": self.target_angle,
            "target_angle_deg": torch.rad2deg(
                self.target_angle
            ),
            "maintain_angle_rad": self.maintain_angle,
            "maintain_angle_deg": torch.rad2deg(
                self.maintain_angle
            ),
            "target_progress": torch.clamp(
                opening_angle / self.target_angle,
                0.0,
                1.0,
            ),
            "target_reached_now": target_reached_now,
            "target_reached_once": self.reached_once,
            "maintained": maintained,
            "hold_progress": hold_progress,
            "screen_edge_pos": screen_edge,
            "task_tip_pos": task_tip,
            "tip_to_screen_edge": tip_to_edge,
            "tip_to_screen_edge_dist": tip_distance,
            "near_screen_edge": (
                tip_distance <= NEAR_EDGE_THRESHOLD
            ),
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
            "laptop_root_pose": self._laptop_root_pose(),
            "screen_pose": self._screen_pose(),
            "screen_edge_pos": info["screen_edge_pos"],
            "task_tip_pos": info["task_tip_pos"],
            "tip_to_screen_edge": info[
                "tip_to_screen_edge"
            ],
            "hinge_qpos": info["hinge_qpos"].unsqueeze(-1),
            "hinge_qvel": info["hinge_qvel"].unsqueeze(-1),
            "open_fraction": info[
                "open_fraction"
            ].unsqueeze(-1),
            "opening_angle_rad": info[
                "opening_angle_rad"
            ].unsqueeze(-1),
            "target_angle_rad": info[
                "target_angle_rad"
            ].unsqueeze(-1),
            "maintain_angle_rad": info[
                "maintain_angle_rad"
            ].unsqueeze(-1),
            "target_progress": info[
                "target_progress"
            ].unsqueeze(-1),
            "target_reached_once": info[
                "target_reached_once"
            ].float().unsqueeze(-1),
            "maintained": info[
                "maintained"
            ].float().unsqueeze(-1),
            "hold_progress": info[
                "hold_progress"
            ].unsqueeze(-1),
            "task_phase_one_hot": self._phase_one_hot(
                info["success"]
            ),
            "screen_extent_local": self.screen_extent_local,
            "laptop_spawn_xy": self.spawn_xy,
            "laptop_spawn_yaw": self.spawn_yaw.unsqueeze(-1),
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
            if isinstance(value, torch.Tensor):
                if value.ndim == 1:
                    obs[key] = value.unsqueeze(-1)
                elif value.ndim > 2:
                    obs[key] = value.reshape(
                        value.shape[0],
                        -1,
                    )

        return obs

    def get_state_dict(self):
        """Persist physical state plus every sequential-success latch."""
        state = super().get_state_dict()
        state["office_laptop_task_state"] = torch.cat(
            [
                self.reached_once[:, None].float(),
                self.hold_counter[:, None].float(),
                self.success_latched[:, None].float(),
                self.spawn_xy.float(),
                self.spawn_yaw[:, None].float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("office_laptop_task_state", None)
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
        if task_state.shape[-1] != 6:
            raise ValueError(
                f"office_laptop_task_state must have 6 fields, "
                f"got {task_state.shape[-1]}"
            )
        self.reached_once[restore_idx] = task_state[:, 0].bool()
        self.hold_counter[restore_idx] = task_state[:, 1].to(torch.long)
        self.success_latched[restore_idx] = task_state[:, 2].bool()
        self.spawn_xy[restore_idx] = task_state[:, 3:5].float()
        self.spawn_yaw[restore_idx] = task_state[:, 5].float()

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
        reaching = (
            1.0
            - torch.tanh(
                6.0 * info["tip_to_screen_edge_dist"]
            )
        ) * (~info["target_reached_once"]).float()

        near_edge = (
            info["near_screen_edge"]
            & (~info["target_reached_once"])
        ).float() * 0.5

        opening = 6.0 * info["target_progress"]
        target_bonus = (
            info["target_reached_now"].float() * 0.5
        )
        hold = (
            info["hold_progress"]
            * info["target_reached_once"].float()
            * 2.0
        )

        opening_direction = torch.sign(
            self.qpos_open - self.qpos_closed
        )
        opening_velocity = (
            info["hinge_qvel"] * opening_direction
        )
        closing_penalty = 0.25 * torch.clamp(
            -opening_velocity,
            min=0.0,
            max=1.0,
        )

        reward = (
            reaching
            + near_edge
            + opening
            + target_bonus
            + hold
            - closing_penalty
        )

        info["reward_terms"] = {
            "reaching_reward": reaching,
            "near_edge_bonus": near_edge,
            "opening_reward": opening,
            "target_bonus": target_bonus,
            "hold_reward": hold,
            "closing_penalty": closing_penalty,
            "dense_reward_no_success_bonus": reward.clone(),
        }

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
        "laptop_open_v1_sharpa_mobile_static ###"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        "--env_id",
        dest="env_id",
        default="OfficeDeskOpenLaptop-Sharpa-v1",
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
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--video-dir",
        default="tasks/videos_office_laptop_open_sharpa_mobile_static",
    )
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--debug-print", action="store_true")
    parser.add_argument("--laptop-scale", type=float, default=None)
    parser.add_argument("--start-angle-min-deg", type=float, default=None)
    parser.add_argument("--start-angle-max-deg", type=float, default=None)
    parser.add_argument(
        "--arm-init-qpos", choices=("keyframe", "teleop"), default="keyframe"
    )
    parser.add_argument(
        "--collision-decomposition",
        choices=("simplified", "none", "coacd"),
        default="simplified",
        help=(
            "Collision mode. 'simplified' uses stable shell-sized boxes; "
            "'none' may create an invisible convex wedge, while 'coacd' is "
            "slow unless all model-10211 caches have been generated."
        ),
    )
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        render_mode="rgb_array",
        target_finger_id=args.target_finger_id,
        laptop_scale=args.laptop_scale,
        start_angle_min_deg=args.start_angle_min_deg,
        start_angle_max_deg=args.start_angle_max_deg,
        arm_init_qpos=args.arm_init_qpos,
        collision_decomposition=args.collision_decomposition,
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
    print("Loaded task file:", Path(__file__).resolve())
    print(
        "Laptop config:",
        f"start={env.unwrapped._cfg['start_angle_min_deg']:.1f}-"
        f"{env.unwrapped._cfg['start_angle_max_deg']:.1f} deg,",
        f"target={env.unwrapped._cfg['target_angle_deg']:.1f} deg,",
        f"maintain={env.unwrapped._cfg['maintain_angle_deg']:.1f} deg,",
        f"hinge_friction={env.unwrapped._cfg['hinge_friction']:.2f},",
        f"collision_decomposition={env.unwrapped.collision_decomposition}",
    )
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Robot DoF:", env.unwrapped.agent.robot.dof)

    obs, info = env.reset(seed=args.seed)
    print(f"Reset opening angle: {_scalar(info['opening_angle_deg']):.2f} deg")
    hold_action = env.unwrapped.agent.robot.get_qpos().detach().clone()
    if hold_action.shape[-1] != env.action_space.shape[-1]:
        # Stock Panda exposes two mimic finger qpos values but one action.
        if hold_action.shape[-1] == 9 and env.action_space.shape[-1] == 8:
            hold_action = torch.cat(
                [hold_action[:, :7], hold_action[:, 7:8]], dim=-1
            )
        else:
            raise RuntimeError(
                "pd_joint_pos action dimension does not match "
                f"robot qpos: action={env.action_space.shape[-1]}, "
                f"qpos={hold_action.shape[-1]}"
            )

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(
            hold_action
        )
        env.render()

        if args.debug_print and (
            step % 5 == 0
            or bool(info["task_success"][0].item())
        ):
            print(
                f"step={step:04d} "
                f"reward={_scalar(reward):.3f} "
                f"angle={_scalar(info['opening_angle_deg']):.1f}deg "
                f"target={_scalar(info['target_angle_deg']):.1f}deg "
                f"maintain={_scalar(info['maintain_angle_deg']):.1f}deg "
                f"hold={_scalar(info['hold_progress']):.2f} "
                "mode=static "
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

"""
Final Blender023 power-button task for demo collection, RL, and IL.

This file depends on `tasks/juice_preparation/juice_preparation_place_lid.py`, which contains the
shared Blender023 asset loader used by both tasks.

Registered environments
-----------------------
JuicePreparationPressButton-Gripper-v1
JuicePreparationPressButton-Sharpa-v1
JuicePreparationPressButton-Wuji-v1

Initial condition
-----------------
- Blender023 is fixed on the table and faces the robot.
- The lid visual mesh is merged directly into the Blender023 MJCF as a
  fixed child body. It is part of the blender model rather than a separate
  actor, so it cannot drift, float, or become misaligned.
- The physical power button remains an independent 3 mm prismatic joint
  with a spring-damper drive returning it to the released qpos.
- The selected robot starts from its fixed rest configuration.

Action
------
Default controller: pd_joint_pos.

The environment never crops, reorders, normalizes, or remaps the controller
action. Demonstrations therefore save the exact action passed to `env.step()`.

Panda:
    absolute arm-joint position targets followed by the gripper command.
    The exact action dimension and limits are defined by `env.action_space`.

Dexterous-hand robots:
    absolute Panda arm-joint targets followed by the selected hand's native
    joint-position targets.

Task state added by `_get_obs_extra()`
--------------------------------------
Raw state:
    tcp_pose                    (7,)
    all_tip_positions           (3 * num_tips,)
    pressing_point              (3,)
    button_link_pose            (7,)
    button_surface_position     (3,)
    button_press_direction      (3,)
    button_qpos                 (1,)
    button_qvel                 (1,)
    button_rest_qpos            (1,)
    button_pressed_qpos         (1,)

Derived state:
    pressing_point_to_button    (3,)
    pressing_point_distance     (1,)
    button_lateral_error        (1,)
    button_axial_gap            (1,)
    button_press_progress       (1,)
    button_hold_progress        (1,)
    near_button                 (1,)
    button_pressed              (1,)
    task_success                (1,)

Robot qpos and qvel remain part of the standard ManiSkill proprioception.

Success
-------
Success is latched when, for four consecutive control steps:

1. button_press_progress >= 0.80;
2. the selected pressing point remains within 3 cm laterally of the real
   button surface;
3. its axial distance is inside the contact band [-1.5 cm, 5 cm].

At 20 Hz, four steps correspond to 0.2 seconds. Releasing the button is not
required. After the operator moves away, the spring drive returns it to qpos=0.

The file contains only the environment and reusable task logic needed for
teleoperation / VR demonstration collection.
"""

from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, TASK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from tasks.config.hands import register_task
from tasks.scene_builders.table import TableSceneBuilder
import copy

from tasks.config.layout_randomization import ARM_INIT_QPOS_NOISE_STD, episode_rng, layout_rng
from tasks.config.workspace_rig import (
    WORK_TABLE_X_OFFSET,
    set_work_table_pose,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import (
    Array,
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)

from tasks.config.sim_profiles import make_sim_config
from tasks.robots.panda_gripper_right_mobile_base import PandaGripperRightMobileBase
from tasks.robots.panda_sharpa_hand_right_mobile_base import PandaSharpaHandRightAngledMobileBase
from tasks.robots.panda_wuji_hand_right_mobile_base import PandaWujiHandRightMobileBase

from tasks.juice_preparation.juice_preparation_place_lid import (
    CONTROL_FREQUENCY,
    DEFAULT_BLENDER_DIR,
    DEFAULT_LID_DIR,
    GLOBAL_SCALE,
    ROBOT_ROOT_POSE,
    all_tip_positions,
    arm_dof,
    build_blender023,
    BLENDER_BUTTON_DAMPING,
    BLENDER_BUTTON_FORCE_LIMIT,
    BLENDER_BUTTON_FRICTION,
    BLENDER_BUTTON_STIFFNESS,
    compute_blender_geometry,
    sample_juice_layout,
    shifted_blender_geometry,
    mjcf_body_vertices,
    prepare_mjcf,
    quat_to_matrix,
    reset_robot_agent,
    transform_points,
)
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag


ENV_IDS = (
    "JuicePreparationPressButton-Sharpa-v1",
    "JuicePreparationPressButton-Wuji-v1",
    "JuicePreparationPressButton-XHand-v1",
    "JuicePreparationPressButton-Gripper-v1",
)
# 250 was below the longest recorded human episode (366 steps; p95 147).
# Randomized layouts add reach; the budget only binds training / evaluation.
MAX_EPISODE_STEPS = 400

BUTTON_PRESS_RATIO = 0.80
BUTTON_HOLD_STEPS = 4
# Single source of truth lives with the blender builder so that every
# environment building a Blender023 gets the same button, including the two
# that previously left the drive at stiffness 0.
BUTTON_STIFFNESS = BLENDER_BUTTON_STIFFNESS
BUTTON_DAMPING = BLENDER_BUTTON_DAMPING
BUTTON_FORCE_LIMIT = BLENDER_BUTTON_FORCE_LIMIT
BUTTON_FRICTION = BLENDER_BUTTON_FRICTION

BUTTON_LATERAL_TOL = 0.030
BUTTON_AXIAL_CONTACT_MIN = -0.015
BUTTON_AXIAL_CONTACT_MAX = 0.050

SUCCESS_REWARD = 6.0
LIVING_PENALTY = 0.005
ARM_QVEL_PENALTY = 2e-4

BUTTON_KEYWORDS = (
    "button",
    "push",
    "switch",
    "start",
    "pulse",
    "power",
)

BUTTON_CLOSEUP_EYE = [-0.40 + WORK_TABLE_X_OFFSET, -0.335, 0.295]
BUTTON_CLOSEUP_TARGET = [-0.005 + WORK_TABLE_X_OFFSET, -0.300, 0.225]
BUTTON_CLOSEUP_FOV_DEG = 36.0



# =============================================================================
# Shared button predicate
#
# Stage 4 of the long Juice task and this subtask read the same switch travel,
# so "pressed" is one number in one place. Proximity stays a diagnostic: it
# must not veto a genuinely depressed power button.
# =============================================================================


def button_press_update(
    button_qpos: torch.Tensor,
    hold_counter: torch.Tensor,
    *,
    rest_qpos: float,
    pressed_qpos: float,
    active: torch.Tensor = None,
):
    """Normalised switch travel plus its hold debounce.

    ``active`` is the long task's stage mask; envs outside the stage keep their
    counter untouched. ``None`` (the subtask) makes every env active.

    Returns ``(press_progress, pressed, hold_counter)``.
    """
    press_progress = torch.clamp(
        (button_qpos - rest_qpos) / (pressed_qpos - rest_qpos), 0.0, 1.0
    )
    pressed = press_progress >= BUTTON_PRESS_RATIO
    if active is not None:
        pressed = pressed & active
    ticked = torch.where(pressed, hold_counter + 1, torch.zeros_like(hold_counter))
    if active is not None:
        ticked = torch.where(active, ticked, hold_counter)
    return press_progress, pressed, ticked


def _find_body(
    root: ET.Element,
    body_name: str,
) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == body_name:
            return body
    raise KeyError(f"MJCF body {body_name!r} was not found")


def _integrate_lid_visual_into_blender(
    blender_dir: Path,
    lid_dir: Path,
    geometry,
    global_scale: float,
) -> None:
    """Merge the seated lid visual into Blender023's fixed model structure.

    The lid becomes a fixed child of the Blender023 `object` body. Only visual
    geometry is copied because this task never interacts with the lid.
    The existing power_button body and prismatic joint remain unchanged.
    """
    blender_xml = Path(blender_dir) / "model.xml"
    lid_xml = Path(lid_dir) / "model.xml"

    blender_tree = ET.parse(blender_xml)
    blender_root = blender_tree.getroot()
    lid_root = ET.parse(lid_xml).getroot()

    blender_asset = blender_root.find("asset")
    lid_asset = lid_root.find("asset")
    if blender_asset is None or lid_asset is None:
        raise RuntimeError(
            "Blender or lid MJCF has no <asset> section"
        )

    blender_object = _find_body(
        blender_root,
        "object",
    )
    lid_object = _find_body(
        lid_root,
        "object",
    )

    for child in list(
        blender_object.findall("body")
    ):
        if child.get("name") == "integrated_lid":
            blender_object.remove(child)

    prefix = "integrated_lid_"
    texture_names: dict[str, str] = {}
    material_names: dict[str, str] = {}
    mesh_names: dict[str, str] = {}

    for texture in lid_asset.findall("texture"):
        old_name = texture.get("name")
        if not old_name:
            continue
        new_name = prefix + old_name
        texture_names[old_name] = new_name
        copied = copy.deepcopy(texture)
        copied.set("name", new_name)
        blender_asset.append(copied)

    for material in lid_asset.findall("material"):
        old_name = material.get("name")
        if not old_name:
            continue
        new_name = prefix + old_name
        material_names[old_name] = new_name
        copied = copy.deepcopy(material)
        copied.set("name", new_name)
        texture_name = copied.get("texture")
        if texture_name in texture_names:
            copied.set(
                "texture",
                texture_names[texture_name],
            )
        blender_asset.append(copied)

    for mesh in lid_asset.findall("mesh"):
        old_name = mesh.get("name")
        if not old_name:
            continue
        new_name = prefix + old_name
        mesh_names[old_name] = new_name
        copied = copy.deepcopy(mesh)
        copied.set("name", new_name)
        blender_asset.append(copied)

    blender_rotation = quat_to_matrix(
        geometry.blender_quat
    )
    local_position = (
        blender_rotation.T
        @ (
            geometry.seat_position
            - geometry.blender_root
        )
    ) / float(global_scale)

    lid_body = ET.SubElement(
        blender_object,
        "body",
        {
            "name": "integrated_lid",
            "pos": " ".join(
                f"{value:.9f}"
                for value in local_position
            ),
            "quat": "1 0 0 0",
        },
    )

    visual_count = 0
    for geom in lid_object.findall("geom"):
        mesh_name = geom.get("mesh")
        geom_class = geom.get("class", "")
        if not mesh_name and geom_class != "visual":
            continue

        copied = copy.deepcopy(geom)

        if mesh_name in mesh_names:
            copied.set(
                "mesh",
                mesh_names[mesh_name],
            )

        material_name = copied.get("material")
        if material_name in material_names:
            copied.set(
                "material",
                material_names[material_name],
            )

        copied.set("contype", "0")
        copied.set("conaffinity", "0")
        copied.set("group", "1")
        copied.set("mass", "0")
        copied.set(
            "name",
            f"integrated_lid_visual_{visual_count}",
        )
        lid_body.append(copied)
        visual_count += 1

    if visual_count == 0:
        raise RuntimeError(
            "No lid visual geometry was found to integrate"
        )

    blender_tree.write(
        blender_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(
        "[integrated lid] "
        f"local_position="
        f"{np.round(local_position, 6).tolist()} "
        f"visual_geoms={visual_count}"
    )


def _joint_child_link(joint):
    """Return a SAPIEN joint's child link across API variants."""
    for attribute in ("child_link", "get_child_link"):
        value = getattr(joint, attribute, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value is not None:
            return value
    return None


def _joint_type_name(joint) -> str:
    for attribute in ("type", "joint_type", "get_type"):
        value = getattr(joint, attribute, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value is not None:
            return str(value)
    return "unknown"


@register_task(
    "JuicePreparationPressButton",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
)
class JuicePreparationPressButtonEnv(BaseEnv):
    """Press and hold Blender023's spring-loaded power button."""

    SUPPORTED_ROBOTS = [
        "panda_gripper_right_mobile_base",
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
        robot_init_qpos_noise: float = ARM_INIT_QPOS_NOISE_STD,
        target_finger_id: int = 1,
        button_joint_name: str = "",
        button_link_name: str = "",
        button_press_sign: int = 0,
        button_local_offset_x: float = 0.0,
        button_local_offset_y: float = 0.0,
        button_local_offset_z: float = 0.0,
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
        if button_press_sign not in (-1, 0, 1):
            raise ValueError(
                "button_press_sign must be -1, 0, or 1"
            )

        self.blender_dir = Path(blender_dir)
        self.lid_dir = Path(lid_dir)
        self.global_scale = float(global_scale)
        self.robot_init_qpos_noise = float(
            robot_init_qpos_noise
        )
        self.target_finger_id = int(target_finger_id)
        self.requested_button_joint_name = str(
            button_joint_name
        )
        self.requested_button_link_name = str(
            button_link_name
        )
        self.button_press_sign = int(button_press_sign)
        self.button_local_offset = np.array(
            [
                button_local_offset_x,
                button_local_offset_y,
                button_local_offset_z,
            ],
            dtype=np.float64,
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
            "press_button_blender023",
        )
        self.patched_lid_dir = prepare_mjcf(
            self.lid_dir,
            "press_button_blender023_lid",
        )
        _integrate_lid_visual_into_blender(
            blender_dir=self.patched_blender_dir,
            lid_dir=self.patched_lid_dir,
            geometry=self.geometry,
            global_scale=self.global_scale,
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
        task_pose = sapien_utils.look_at(
            eye=[0.46 + WORK_TABLE_X_OFFSET, -0.66, 0.43],
            target=[0.07 + WORK_TABLE_X_OFFSET, -0.30, 0.18],
        )
        closeup_pose = sapien_utils.look_at(
            eye=BUTTON_CLOSEUP_EYE,
            target=BUTTON_CLOSEUP_TARGET,
        )
        return [
            CameraConfig(
                "base_camera",
                task_pose,
                320,
                240,
                np.deg2rad(58.0),
                0.01,
                100.0,
            ),
            CameraConfig(
                "button_camera",
                closeup_pose,
                320,
                240,
                np.deg2rad(BUTTON_CLOSEUP_FOV_DEG),
                0.01,
                10.0,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        eye = [0.52, -0.76, 0.48]
        target = [0.07, -0.30, 0.18]
        fov = 54.0

        pose = sapien_utils.look_at(
            eye=eye,
            target=target,
        )
        return CameraConfig(
            "render_camera",
            pose,
            self.render_width,
            self.render_height,
            np.deg2rad(fov),
            0.01,
            100.0,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, ROBOT_ROOT_POSE)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=(
                self.robot_init_qpos_noise
            ),
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
            name="robocasa_blender023_press_button",
        )
        if not self.blender_is_articulation:
            raise RuntimeError(
                "Blender023 must be an articulation "
                "for the power-button task"
            )

        self._select_button_joint()
        self._configure_button_joint()

        # The seated lid is already part of the Blender023 articulation's
        # fixed visual model. No separate lid actor is created in this task.

        self.button_body_vertices_local = (
            mjcf_body_vertices(
                self.patched_blender_dir / "model.xml",
                self.button_link.name,
                self.global_scale,
            )
        )

        self.button_hold_counter = torch.zeros(
            self.num_envs,
            dtype=torch.int32,
            device=self.device,
        )
        self.success_latched = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

    def _print_articulation_structure(self) -> None:
        joints = list(self.blender.get_active_joints())
        limits = (
            self.blender.get_qlimits()[0]
            .detach()
            .cpu()
            .numpy()
        )

        print("\n[Blender023 active joints]")
        for index, joint in enumerate(joints):
            child = _joint_child_link(joint)
            child_name = getattr(
                child,
                "name",
                "<unknown>",
            )
            print(
                f"  index={index:02d} "
                f"joint={joint.name!r} "
                f"type={_joint_type_name(joint)!r} "
                f"child={child_name!r} "
                f"limits="
                f"{np.round(limits[index], 6).tolist()}"
            )

        print("\n[Blender023 links]")
        for index, link in enumerate(
            self.blender.get_links()
        ):
            print(
                f"  index={index:02d} "
                f"link={link.name!r}"
            )

    def _select_button_joint(self) -> None:
        joints = list(self.blender.get_active_joints())
        if not joints:
            raise RuntimeError(
                "Blender023 has no active joints"
            )

        limits = (
            self.blender.get_qlimits()[0]
            .detach()
            .cpu()
            .numpy()
        )
        candidates: list[
            tuple[float, int, str, str]
        ] = []

        for index, joint in enumerate(joints):
            child = _joint_child_link(joint)
            child_name = getattr(child, "name", "")
            descriptor = (
                f"{joint.name} {child_name}".lower()
            )
            travel = float(
                abs(
                    limits[index, 1]
                    - limits[index, 0]
                )
            )

            if self.requested_button_joint_name:
                score = (
                    1e6
                    if joint.name
                    == self.requested_button_joint_name
                    else -1e6
                )
            else:
                score = sum(
                    100.0 - rank
                    for rank, keyword in enumerate(
                        BUTTON_KEYWORDS
                    )
                    if keyword in descriptor
                )
                if (
                    "knob" in descriptor
                    or "dial" in descriptor
                ):
                    score -= 80.0
                if 1e-5 < travel <= 0.03:
                    score += 25.0
                elif travel > 0.20:
                    score -= 20.0

            candidates.append(
                (
                    score,
                    index,
                    joint.name,
                    child_name,
                )
            )

        candidates.sort(reverse=True)
        score, index, _, _ = candidates[0]

        if (
            self.requested_button_joint_name
            and score < 0
        ):
            self._print_articulation_structure()
            raise KeyError(
                f"Button joint "
                f"{self.requested_button_joint_name!r} "
                "was not found"
            )
        if (
            not self.requested_button_joint_name
            and score <= 0
        ):
            self._print_articulation_structure()
            raise RuntimeError(
                "Could not auto-detect the push-button "
                "joint. Pass --button-joint-name."
            )

        self.button_joint_index = int(index)
        self.button_joint = joints[
            self.button_joint_index
        ]

        child = _joint_child_link(
            self.button_joint
        )
        if self.requested_button_link_name:
            links = {
                link.name: link
                for link in self.blender.get_links()
            }
            if (
                self.requested_button_link_name
                not in links
            ):
                self._print_articulation_structure()
                raise KeyError(
                    f"Button link "
                    f"{self.requested_button_link_name!r} "
                    "was not found"
                )
            child = links[
                self.requested_button_link_name
            ]

        if child is None:
            raise RuntimeError(
                "Selected button joint has no child link"
            )

        self.button_link = child
        print(
            "\n[selected button] "
            f"joint_index={self.button_joint_index} "
            f"joint={self.button_joint.name!r} "
            f"link={self.button_link.name!r}"
        )

    def _configure_button_joint(self) -> None:
        limits = (
            self.blender.get_qlimits()[
                0,
                self.button_joint_index,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        lower, upper = (
            float(limits[0]),
            float(limits[1]),
        )
        rest = float(
            np.clip(0.0, lower, upper)
        )

        if self.button_press_sign > 0:
            pressed = upper
        elif self.button_press_sign < 0:
            pressed = lower
        else:
            pressed = (
                lower
                if abs(lower - rest)
                >= abs(upper - rest)
                else upper
            )

        if abs(pressed - rest) < 1e-7:
            raise RuntimeError(
                "Selected button joint has zero travel"
            )

        self.button_rest_qpos = rest
        self.button_pressed_qpos = pressed
        self.button_joint_travel = abs(
            pressed - rest
        )

        self.button_joint.set_drive_properties(
            stiffness=BUTTON_STIFFNESS,
            damping=BUTTON_DAMPING,
            force_limit=BUTTON_FORCE_LIMIT,
            mode="force",
        )
        if hasattr(
            self.button_joint,
            "set_drive_target",
        ):
            self.button_joint.set_drive_target(rest)
        if hasattr(
            self.button_joint,
            "set_friction",
        ):
            self.button_joint.set_friction(
                BUTTON_FRICTION
            )

        print(
            "[button mechanics] "
            f"limits={[lower, upper]} "
            f"rest={rest:.6f} "
            f"pressed={pressed:.6f} "
            f"travel={self.button_joint_travel:.6f}"
        )

    def _set_button_qpos(
        self,
        value: torch.Tensor | float,
    ) -> None:
        qpos = self.blender.get_qpos().clone()
        qpos[:, self.button_joint_index] = (
            torch.as_tensor(
                value,
                dtype=qpos.dtype,
                device=qpos.device,
            )
        )
        self.blender.set_qpos(qpos)

    def _calibrate_button_geometry(self) -> None:
        """Measure joint motion and locate the real outer button surface."""
        qpos = self.blender.get_qpos().clone()

        qpos[:, self.button_joint_index] = (
            self.button_rest_qpos
        )
        self.blender.set_qpos(qpos)
        rest_link_position = (
            self.button_link.pose.p.clone()
        )
        rest_link_rotation = (
            self.button_link.pose
            .to_transformation_matrix()[
                ..., :3, :3
            ]
            .clone()
        )

        qpos[:, self.button_joint_index] = (
            self.button_pressed_qpos
        )
        self.blender.set_qpos(qpos)
        pressed_link_position = (
            self.button_link.pose.p.clone()
        )

        delta = (
            pressed_link_position
            - rest_link_position
        )
        world_travel = torch.linalg.norm(
            delta,
            dim=1,
            keepdim=True,
        )
        if bool(
            (world_travel < 1e-6).any().item()
        ):
            raise RuntimeError(
                "Selected joint does not translate its "
                "child link"
            )

        self.button_press_direction = (
            delta / world_travel
        )
        self.button_world_travel = (
            world_travel.squeeze(1)
        )

        # The link origin is at the blender root in this asset and is not the
        # visible button center. Measure the power_button body's outermost face
        # in the direction opposite the physical press motion.
        vertices_world = transform_points(
            self.button_body_vertices_local,
            self.geometry.blender_root,
            self.geometry.blender_quat,
        )
        outward = (
            -self.button_press_direction[0]
            .detach()
            .cpu()
            .numpy()
        )
        projection = vertices_world @ outward
        projection_span = float(
            projection.max() - projection.min()
        )
        face_band = max(
            0.0005,
            0.08 * projection_span,
        )
        face_vertices = vertices_world[
            projection
            >= projection.max() - face_band
        ]
        face_center = face_vertices.mean(axis=0)

        face_center_tensor = torch.tensor(
            np.asarray(
                [face_center],
                dtype=np.float32,
            ),
            device=self.device,
        ).expand(self.num_envs, -1).clone()

        local_offset = torch.tensor(
            self.button_local_offset,
            dtype=face_center_tensor.dtype,
            device=self.device,
        )
        face_center_tensor += torch.matmul(
            rest_link_rotation,
            local_offset.view(1, 3, 1),
        ).squeeze(-1)

        self.button_released_surface_position = (
            face_center_tensor
        )
        self.button_released_link_position = (
            rest_link_position
        )

        qpos[:, self.button_joint_index] = (
            self.button_rest_qpos
        )
        self.blender.set_qpos(qpos)
        self.blender.set_qvel(
            torch.zeros_like(
                self.blender.get_qvel()
            )
        )

        print(
            "[button calibration] "
            f"surface="
            f"{face_center_tensor[0].detach().cpu().numpy().round(5).tolist()} "
            f"press_direction="
            f"{self.button_press_direction[0].detach().cpu().numpy().round(5).tolist()} "
            f"world_travel="
            f"{float(self.button_world_travel[0]):.6f}"
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
            # Per-episode layout (FIRST episode-RNG draw): the blender and its button move together.
            bdx, bdy, byaw = sample_juice_layout(layout_rng(self, int(env_idx[0])))["blender"]
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
                root_pose=ROBOT_ROOT_POSE,
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
            self.blender.set_pose(
                Pose.create_from_pq(
                    blender_p,
                    blender_q,
                )
            )
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
            self._set_button_qpos(
                self.button_rest_qpos
            )
            if hasattr(
                self.button_joint,
                "set_drive_target",
            ):
                self.button_joint.set_drive_target(
                    self.button_rest_qpos
                )

            self._calibrate_button_geometry()
            self.button_hold_counter[env_idx] = 0
            self.success_latched[env_idx] = False

    # ------------------------------------------------------------------
    # Task quantities
    # ------------------------------------------------------------------

    def _pressing_point(
        self,
        tips: torch.Tensor,
    ) -> torch.Tensor:
        if self.robot_uids == "panda_gripper_right_mobile_base":
            return self.agent.tcp.pose.p

        index = int(
            np.clip(
                self.target_finger_id,
                0,
                tips.shape[1] - 1,
            )
        )
        return tips[:, index, :]

    def _task_quantities(self) -> dict[str, torch.Tensor]:
        button_qpos = self.blender.get_qpos()[
            :,
            self.button_joint_index,
        ]
        button_qvel = self.blender.get_qvel()[
            :,
            self.button_joint_index,
        ]

        press_progress, _, _ = button_press_update(
            button_qpos,
            self.button_hold_counter,
            rest_qpos=self.button_rest_qpos,
            pressed_qpos=self.button_pressed_qpos,
        )

        button_surface = (
            self.button_released_surface_position
            + self.button_press_direction
            * (
                press_progress
                * self.button_world_travel
            ).unsqueeze(1)
        )

        tips = all_tip_positions(
            self.agent,
            self.robot_uids,
        )
        pressing_point = self._pressing_point(tips)
        point_to_button = (
            button_surface - pressing_point
        )
        pressing_point_distance = (
            torch.linalg.norm(
                point_to_button,
                dim=1,
            )
        )

        outward = -self.button_press_direction
        point_relative = (
            pressing_point - button_surface
        )
        axial_gap = torch.sum(
            point_relative * outward,
            dim=1,
        )
        lateral_vector = (
            point_relative
            - axial_gap.unsqueeze(1) * outward
        )
        lateral_error = torch.linalg.norm(
            lateral_vector,
            dim=1,
        )

        near_button = (
            (lateral_error <= BUTTON_LATERAL_TOL)
            & (
                axial_gap
                >= BUTTON_AXIAL_CONTACT_MIN
            )
            & (
                axial_gap
                <= BUTTON_AXIAL_CONTACT_MAX
            )
        )
        button_pressed = press_progress >= BUTTON_PRESS_RATIO

        arm_velocity = self.agent.robot.get_qvel()[
            ..., :arm_dof(self.agent)
        ]

        return {
            "button_link_pose": (
                self.button_link.pose.raw_pose
            ),
            "button_surface_position": (
                button_surface
            ),
            "button_press_direction": (
                self.button_press_direction
            ),
            "button_qpos": button_qpos,
            "button_qvel": button_qvel,
            "button_press_progress": press_progress,
            "all_tip_positions": tips,
            "pressing_point": pressing_point,
            "pressing_point_to_button": (
                point_to_button
            ),
            "pressing_point_distance": (
                pressing_point_distance
            ),
            "button_lateral_error": lateral_error,
            "button_axial_gap": axial_gap,
            "near_button": near_button,
            "button_pressed": button_pressed,
            "arm_velocity": arm_velocity,
        }

    def evaluate(self):
        q = self._task_quantities()

        self.button_hold_counter = torch.where(
            q["button_pressed"],
            self.button_hold_counter + 1,
            torch.zeros_like(self.button_hold_counter),
        )
        self.success_latched |= (
            self.button_hold_counter
            >= BUTTON_HOLD_STEPS
        )

        hold_progress = torch.clamp(
            self.button_hold_counter.float()
            / BUTTON_HOLD_STEPS,
            0.0,
            1.0,
        )

        return {
            "success": self.success_latched,
            "task_success": self.success_latched,
            "is_obj_placed": self.success_latched,
            "obj_to_goal_dist": q[
                "pressing_point_distance"
            ],
            "button_hold_counter": (
                self.button_hold_counter
            ),
            "button_hold_progress": hold_progress,
            **q,
        }

    def get_state_dict(self):
        state = super().get_state_dict()
        state["juice_button_task_state"] = torch.stack(
            [
                self.button_hold_counter.float(),
                self.success_latched.float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("juice_button_task_state", None)
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
        if task_state.shape[-1] != 2:
            raise ValueError("juice_button_task_state must have 2 fields")
        # The buffer is int32; `.long()` here made set_state_dict raise
        # "Index put requires the source and destination dtypes match", so this
        # subtask could not round-trip its own state at all.
        self.button_hold_counter[restore_idx] = task_state[:, 0].to(
            self.button_hold_counter.dtype
        )
        self.success_latched[restore_idx] = task_state[:, 1].bool()

    def _get_obs_extra(self, info: dict):
        rest_qpos = torch.full(
            (self.num_envs, 1),
            self.button_rest_qpos,
            dtype=torch.float32,
            device=self.device,
        )
        pressed_qpos = torch.full(
            (self.num_envs, 1),
            self.button_pressed_qpos,
            dtype=torch.float32,
            device=self.device,
        )

        return {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "all_tip_positions": info[
                "all_tip_positions"
            ].reshape(self.num_envs, -1),
            "pressing_point": info[
                "pressing_point"
            ],
            "button_link_pose": info[
                "button_link_pose"
            ],
            "button_surface_position": info[
                "button_surface_position"
            ],
            "button_press_direction": info[
                "button_press_direction"
            ],
            "button_qpos": info[
                "button_qpos"
            ].unsqueeze(1),
            "button_qvel": info[
                "button_qvel"
            ].unsqueeze(1),
            "button_rest_qpos": rest_qpos,
            "button_pressed_qpos": pressed_qpos,
            "pressing_point_to_button": info[
                "pressing_point_to_button"
            ],
            "pressing_point_distance": info[
                "pressing_point_distance"
            ].unsqueeze(1),
            "button_lateral_error": info[
                "button_lateral_error"
            ].unsqueeze(1),
            "button_axial_gap": info[
                "button_axial_gap"
            ].unsqueeze(1),
            "button_press_progress": info[
                "button_press_progress"
            ].unsqueeze(1),
            "button_hold_progress": info[
                "button_hold_progress"
            ].unsqueeze(1),
            "near_button": info[
                "near_button"
            ].float().unsqueeze(1),
            "button_pressed": info[
                "button_pressed"
            ].float().unsqueeze(1),
            "task_success": info[
                "task_success"
            ].float().unsqueeze(1),
        }

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
        lateral_reward = 1.0 - torch.tanh(
            25.0
            * info["button_lateral_error"]
        )
        axial_reward = 1.0 - torch.tanh(
            12.0
            * torch.clamp(
                info["button_axial_gap"],
                min=0.0,
            )
        )
        reach_reward = (
            lateral_reward * axial_reward
        )
        press_reward = info[
            "button_press_progress"
        ]
        hold_reward = info[
            "button_hold_progress"
        ]

        reward = (
            1.5 * reach_reward
            + 3.0 * press_reward
            + 1.5 * hold_reward
            - LIVING_PENALTY
        )
        reward -= (
            ARM_QVEL_PENALTY
            * torch.mean(
                info["arm_velocity"].square(),
                dim=1,
            )
        )

        reward = torch.where(
            info["task_success"],
            torch.full_like(
                reward,
                SUCCESS_REWARD,
            ),
            reward,
        )
        info["reward_terms"] = {
            "reach": reach_reward,
            "press": press_reward,
            "hold": hold_reward,
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
        return torch.zeros_like(
            info["task_success"]
        )

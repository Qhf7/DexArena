"""Long-horizon dual-arm juice-preparation task.

Required sequence:
1. left hand opens the refrigerator to 75 degrees;
2. right hand removes the apple and places it in the open Blender023 pitcher;
3. left hand closes the refrigerator;
4. right hand places the loose lid completely on the blender;
5. right hand presses the blender power button.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mani_skill.utils import common
from tasks.config.hands import register_task
from mani_skill.utils.structs import Pose

from tasks.juice_preparation.juice_preparation_place_lid import (
    DEFAULT_LID_DIR,
    LID_MASS,
    LID_UPSCALE,
    PICK_HEIGHT,
    SUCCESS_SETTLE_STEPS as LID_SETTLE_STEPS,
    TABLE_TOP_Z,
    build_blender_lid,
    lid_seat_quantities,
    prepare_mjcf,
)
from tasks.juice_preparation.juice_preparation_press_button import (
    BUTTON_DAMPING,
    BUTTON_FORCE_LIMIT,
    BUTTON_HOLD_STEPS,
    BUTTON_FRICTION,
    BUTTON_KEYWORDS,
    BUTTON_STIFFNESS,
    _joint_child_link,
    button_press_update,
)
from tasks.juice_preparation.juice_preparation_open_fridge import (
    SUCCESS_CLOSED_ANGLE_DEG,
    SUCCESS_OPEN_ANGLE_DEG,
    SUCCESS_SETTLE_STEPS as DOOR_SETTLE_STEPS,
    door_settle_update,
    door_success_qpos,
)
from tasks.juice_preparation.juice_preparation_apple_to_blender import (
    FRIDGE_FLOOR_TOP_Z,
    FRIDGE_ROOT_P,
    ROBOT_ROOT_P,
    SUCCESS_SETTLE_STEPS as APPLE_SETTLE_STEPS,
    JuicePreparationAppleToBlenderEnv,
    apple_inside_blender,
    apple_inside_fridge,
    apple_motion_update,
    apple_route_latches,
    apple_settled,
)
from tasks.robots.panda_dual_gripper_mobile_base import PandaDualGripperMobileBase
from tasks.robots.panda_dual_sharpa_hand_mobile_base import PandaDualSharpaHandAngledMobileBase
from tasks.robots.panda_dual_wuji_hand_mobile_base import PandaDualWujiHandMobileBase
from tasks.robots.panda_dual_xhand_hand_mobile_base import PandaDualXHandHandMobileBase  # noqa: F401

ROBOT_UID_GRIPPER = "panda_dual_gripper_mobile_base"
ROBOT_UID_SHARPA = "panda_dual_sharpa_hand_angled_mobile_base"
ROBOT_UID_WUJI = "panda_dual_wuji_hand_angled_mobile_base"
ROBOT_UID_XHAND = "panda_dual_xhand_hand_angled_mobile_base"
ROBOT_UIDS = (ROBOT_UID_GRIPPER, ROBOT_UID_SHARPA, ROBOT_UID_WUJI, ROBOT_UID_XHAND)

STAGE_OPEN_FRIDGE = 0
STAGE_MOVE_APPLE = 1
STAGE_CLOSE_FRIDGE = 2
STAGE_PLACE_LID = 3
STAGE_PRESS_BUTTON = 4
STAGE_DONE = 5
STAGE_NAMES = ("open_fridge", "move_apple", "close_fridge", "place_lid", "press_button", "done")

CONTACT_FORCE_THRESHOLD = 1e-4
MAX_EPISODE_STEPS = 1800


@register_task(
    "JuicePreparationLong",
    side="both",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
)
class JuicePreparationLongEnv(JuicePreparationAppleToBlenderEnv):
    """Five-stage bimanual juice task with strict order and hand assignment."""

    SUPPORTED_ROBOTS = list(ROBOT_UIDS)
    agent: Any

    def __init__(
        self,
        *args,
        robot_uids: str = ROBOT_UID_SHARPA,
        lid_dir: str = str(DEFAULT_LID_DIR),
        lid_mass: float = LID_MASS,
        terminate_on_success: bool = True,
        **kwargs,
    ):
        self.lid_dir = Path(lid_dir)
        self.lid_mass = float(lid_mass)
        self.terminate_on_success = bool(terminate_on_success)
        self.patched_lid_dir = prepare_mjcf(self.lid_dir, "prepare_juice_blender_lid")
        self.target_finger_id = 1
        super().__init__(*args, robot_uids=robot_uids, initial_door_angle_deg=0.0, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.lid, self.lid_template = build_blender_lid(
            scene=self.scene,
            patched_dir=self.patched_lid_dir,
            scale=self.global_scale * LID_UPSCALE,
            num_envs=self.num_envs,
            position=self.blender_geometry.lid_table_position,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            name="prepare_juice_loose_blender_lid",
            dynamic=True,
            mass=self.lid_mass,
        )
        self._select_button_joint()
        self._configure_button_joint()
        # Same expression the two fridge subtasks use, clipping included, so
        # the stage thresholds here are theirs and not a parallel definition.
        self.success_open_qpos = door_success_qpos(
            self.door_lower, self.door_upper, SUCCESS_OPEN_ANGLE_DEG
        )
        self.success_closed_qpos = door_success_qpos(
            self.door_lower, self.door_upper, SUCCESS_CLOSED_ANGLE_DEG
        )

    def _select_button_joint(self):
        limits = self.blender.get_qlimits()[0].detach().cpu().numpy()
        candidates = []
        for index, joint in enumerate(self.blender.get_active_joints()):
            child = _joint_child_link(joint)
            descriptor = f"{joint.name} {getattr(child, 'name', '')}".lower()
            travel = abs(float(limits[index, 1] - limits[index, 0]))
            score = sum(100.0 - rank for rank, word in enumerate(BUTTON_KEYWORDS) if word in descriptor)
            if 1e-5 < travel <= 0.03:
                score += 25.0
            candidates.append((score, index, joint, child))
        score, index, joint, child = max(candidates, key=lambda item: item[0])
        if score <= 0 or child is None:
            raise RuntimeError("Could not identify Blender023 power button")
        self.button_joint_index = index
        self.button_joint = joint
        self.button_link = child

    def _configure_button_joint(self):
        lower, upper = self.blender.get_qlimits()[0, self.button_joint_index].detach().cpu().numpy()
        self.button_rest_qpos = float(lower if abs(lower) <= abs(upper) else upper)
        self.button_pressed_qpos = float(upper if self.button_rest_qpos == lower else lower)
        self.button_joint.set_drive_properties(
            stiffness=BUTTON_STIFFNESS, damping=BUTTON_DAMPING,
            force_limit=BUTTON_FORCE_LIMIT, mode="force",
        )
        if hasattr(self.button_joint, "set_friction"):
            self.button_joint.set_friction(BUTTON_FRICTION)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            # The lid table point already rides on the (randomized) blender; add the
            # family's lid offset + yaw from the same per-episode draw (JuicePreparationAppleToBlenderEnv).
            ldx, ldy, lyaw = self._juice_layout["lid"]
            lid_p = torch.tensor([self.blender_geometry.lid_table_position], dtype=torch.float32, device=self.device)
            lid_p[:, 0] += ldx
            lid_p[:, 1] += ldy
            lid_q = torch.tensor([[float(np.cos(0.5 * lyaw)), 0.0, 0.0, float(np.sin(0.5 * lyaw))]], dtype=torch.float32, device=self.device)
            zero3 = torch.zeros((self.num_envs, 3), device=self.device)
            self.lid.set_pose(Pose.create_from_pq(lid_p, lid_q))
            self.lid.set_linear_velocity(zero3)
            self.lid.set_angular_velocity(zero3)
            qpos = self.blender.get_qpos()
            qpos[:, self.button_joint_index] = self.button_rest_qpos
            self.blender.set_qpos(qpos)
            self.button_joint.set_drive_target(self.button_rest_qpos)
            self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.stage_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.left_open_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.left_close_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.right_apple_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.right_lid_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.right_lid_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.success_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _side_links(self, side: str):
        links = list(getattr(self.agent, f"{side}_tip_links", ()))
        links_map = getattr(self.agent.robot, "links_map", {})
        for name in getattr(self.agent, "grip_link_names", ()):
            if name.startswith(f"{side}_") and name in links_map:
                links.append(links_map[name])
        palm = getattr(self.agent, f"{side}_palm_link", None)
        if palm is not None:
            links.append(palm)
        unique_links = []
        seen_ids = set()
        for link in links:
            if id(link) not in seen_ids:
                unique_links.append(link)
                seen_ids.add(id(link))
        return unique_links

    def _side_contacts(self, side: str, target) -> torch.Tensor:
        result = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for link in self._side_links(side):
            force = self.scene.get_pairwise_contact_forces(link, target)
            result |= torch.linalg.norm(force, dim=1) > CONTACT_FORCE_THRESHOLD
        return result

    def _tick_stage_counter(self, at_stage: torch.Tensor, stage_ok: torch.Tensor) -> torch.Tensor:
        """Advance the shared settle counter for envs at THIS stage only.

        ``evaluate`` runs all five stage blocks every control step. Writing the
        counter unconditionally let each later block zero the value the active
        stage had just incremented, so no stage could ever reach its settle
        threshold (6/12/6/15/4). Envs that are not at ``at_stage`` keep their
        counter untouched; envs at the stage increment while ``stage_ok`` holds
        and reset to zero the moment it breaks (unchanged debounce semantics).
        """
        ticked = torch.where(
            stage_ok, self.stage_counter + 1, torch.zeros_like(self.stage_counter)
        )
        return torch.where(at_stage, ticked, self.stage_counter)

    def _lid_quantities(self):
        """Stage 3 quantities, read from the lid subtask's shared predicate."""
        position, error, _, _, _, seated, stable = lid_seat_quantities(
            self.lid.pose, self.blender_geometry.seat_position, device=self.device
        )
        return position, error, seated, stable

    def _apple_quantities(self):
        """Stage 1 quantities, read from the apple subtask's shared predicate."""
        position = self.apple.pose.p
        collision_min_z = float(self.apple_asset_info.collision_bounds[0, 2])
        in_fridge = apple_inside_fridge(
            position, collision_min_z, device=self.device, fridge_root_p=self.fridge.pose.p
        )
        in_blender, _, _ = apple_inside_blender(
            position, self.blender_geometry, device=self.device
        )
        self.apple_motion = apple_motion_update(
            self.apple.pose,
            self.apple_motion,
            self._elapsed_steps,
            control_timestep=self.control_timestep,
        )
        return position, in_fridge, in_blender, apple_settled(self.apple_motion)

    def evaluate(self):
        stage = self.stage.clone()
        door_qpos = self.fridge.get_qpos()[:, 0]
        door_qvel = self.fridge.get_qvel()[:, 0]
        apple_p, apple_in_fridge, apple_in_blender, apple_stable = self._apple_quantities()
        lid_p, lid_error, lid_seated, lid_stable = self._lid_quantities()
        left_contact_now = self._side_contacts("left", self.door_link)

        at_open = stage == STAGE_OPEN_FRIDGE
        self.left_open_contact |= at_open & left_contact_now
        _, _, self.stage_counter = door_settle_update(
            door_qpos,
            door_qvel,
            self.stage_counter,
            threshold_qpos=self.success_open_qpos,
            opening=True,
            active=at_open,
        )
        advance = at_open & (self.stage_counter >= DOOR_SETTLE_STEPS)
        stage = torch.where(advance, stage + 1, stage)
        self.stage_counter = torch.where(advance, torch.zeros_like(self.stage_counter), self.stage_counter)

        at_apple = stage == STAGE_MOVE_APPLE
        self.right_apple_contact |= at_apple & self._side_contacts("right", self.apple)
        (
            self.apple_left_fridge_latched,
            self.apple_lifted_latched,
        ) = apple_route_latches(
            apple_p,
            apple_in_fridge,
            self.apple_left_fridge_latched,
            self.apple_lifted_latched,
            active=at_apple,
        )
        apple_ok = (
            at_apple
            & self.apple_left_fridge_latched
            & self.apple_lifted_latched
            & apple_in_blender
            & apple_stable
        )
        self.stage_counter = self._tick_stage_counter(at_apple, apple_ok)
        advance = at_apple & (self.stage_counter >= APPLE_SETTLE_STEPS)
        stage = torch.where(advance, stage + 1, stage)
        self.stage_counter = torch.where(advance, torch.zeros_like(self.stage_counter), self.stage_counter)

        at_close = stage == STAGE_CLOSE_FRIDGE
        self.left_close_contact |= at_close & left_contact_now
        _, _, self.stage_counter = door_settle_update(
            door_qpos,
            door_qvel,
            self.stage_counter,
            threshold_qpos=self.success_closed_qpos,
            opening=False,
            active=at_close,
        )
        advance = at_close & (self.stage_counter >= DOOR_SETTLE_STEPS)
        stage = torch.where(advance, stage + 1, stage)
        self.stage_counter = torch.where(advance, torch.zeros_like(self.stage_counter), self.stage_counter)

        at_lid = stage == STAGE_PLACE_LID
        self.right_lid_contact |= at_lid & self._side_contacts("right", self.lid)
        lid_bottom_z = lid_p[:, 2] + float(self.blender_geometry.lid_lower[2])
        self.right_lid_lifted |= (
            at_lid & (lid_bottom_z - TABLE_TOP_Z >= PICK_HEIGHT)
        )
        lid_ok = (
            at_lid & self.right_lid_lifted & lid_seated & lid_stable
        )
        self.stage_counter = self._tick_stage_counter(at_lid, lid_ok)
        advance = at_lid & (self.stage_counter >= LID_SETTLE_STEPS)
        stage = torch.where(advance, stage + 1, stage)
        self.stage_counter = torch.where(advance, torch.zeros_like(self.stage_counter), self.stage_counter)

        button_qpos = self.blender.get_qpos()[:, self.button_joint_index]
        at_button = stage == STAGE_PRESS_BUTTON
        button_progress, _, self.stage_counter = button_press_update(
            button_qpos,
            self.stage_counter,
            rest_qpos=self.button_rest_qpos,
            pressed_qpos=self.button_pressed_qpos,
            active=at_button,
        )
        advance = at_button & (self.stage_counter >= BUTTON_HOLD_STEPS)
        stage = torch.where(advance, stage + 1, stage)
        self.stage_counter = torch.where(
            advance, torch.zeros_like(self.stage_counter), self.stage_counter
        )

        self.stage = stage
        self.success_latched |= stage >= STAGE_DONE
        return {
            "success": self.success_latched,
            "task_success": self.success_latched,
            "stage": stage,
            "door_angle_deg": torch.rad2deg(door_qpos),
            "apple_position": apple_p,
            "apple_inside_blender": apple_in_blender,
            "lid_position": lid_p,
            "lid_to_seat": lid_error,
            "lid_seated": lid_seated,
            "button_press_progress": button_progress,
            "stage_progress": self.stage_counter.float(),
        }

    def get_state_dict(self):
        state = super().get_state_dict()
        state["prepare_juice_task_state"] = torch.stack(
            [
                self.stage.float(),
                self.stage_counter.float(),
                self.left_open_contact.float(),
                self.left_close_contact.float(),
                self.right_apple_contact.float(),
                self.right_lid_contact.float(),
                self.right_lid_lifted.float(),
                self.success_latched.float(),
            ],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("prepare_juice_task_state", None)
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
        if task_state.shape[-1] != 8:
            raise ValueError("prepare_juice_task_state must have 8 fields")
        self.stage[restore_idx] = task_state[:, 0].long()
        self.stage_counter[restore_idx] = task_state[:, 1].long()
        self.left_open_contact[restore_idx] = task_state[:, 2].bool()
        self.left_close_contact[restore_idx] = task_state[:, 3].bool()
        self.right_apple_contact[restore_idx] = task_state[:, 4].bool()
        self.right_lid_contact[restore_idx] = task_state[:, 5].bool()
        self.right_lid_lifted[restore_idx] = task_state[:, 6].bool()
        self.success_latched[restore_idx] = task_state[:, 7].bool()

    def _get_obs_extra(self, info: dict):
        return {
            "stage": info["stage"].float().unsqueeze(1),
            "door_angle_deg": info["door_angle_deg"].unsqueeze(1),
            "apple_position": info["apple_position"],
            "lid_position": info["lid_position"],
            "button_press_progress": info["button_press_progress"].unsqueeze(1),
            # Randomized fixtures must be observable: fridge and blender root poses,
            # plus the lid seat that rides on the blender.
            "fridge_pose": self.fridge.pose.raw_pose,
            "blender_pose": self.blender.pose.raw_pose,
            "seat_target_position": torch.as_tensor(np.asarray(self.blender_geometry.seat_position, dtype=np.float32), device=self.device).view(1, 3).expand(self.num_envs, -1),
            "left_tcp_pose": self.agent.left_tcp_pose.raw_pose,
            "right_tcp_pose": self.agent.right_tcp_pose.raw_pose,
        }

    def compute_sparse_reward(self, obs: Any, action, info: dict):
        return info["task_success"].float()

    def compute_dense_reward(self, obs: Any, action, info: dict):
        reward = info["stage"].float() + 0.1 * torch.clamp(info["stage_progress"], 0.0, 10.0)
        return torch.where(info["task_success"], reward + 10.0, reward)

    def compute_normalized_dense_reward(self, obs: Any, action, info: dict):
        return self.compute_dense_reward(obs, action, info) / 15.0

    def _should_terminate(self, info: dict):
        return info["task_success"] if self.terminate_on_success else torch.zeros_like(info["task_success"])

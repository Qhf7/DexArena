"""Close the juice-task refrigerator with the long task's left arm.

This task reuses the geometry and articulation setup from the open-door stage.
The door starts open; success is the same physical closed-and-settled condition
used by the composed Juice task. Contact and arm speed remain diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.config.hands import register_task

from tasks.juice_preparation.juice_preparation_open_fridge import (
    ARM_STATIC_THRESHOLD,
    CONTACT_FORCE_THRESHOLD,
    DOOR_STATIC_THRESHOLD,
    door_settle_update,
    door_success_qpos,
    ROBOT_UID_GRIPPER,
    ROBOT_UID_SHARPA,
    ROBOT_UID_WUJI,
    ROBOT_UID_XHAND,
    SUCCESS_CLOSED_ANGLE_DEG,
    SUCCESS_REWARD,
    SUCCESS_SETTLE_STEPS,
    JuicePreparationOpenFridgeEnv,
)

INITIAL_OPEN_ANGLE_DEG = 75.0


@register_task(
    "JuicePreparationCloseFridge",
    side="left",
    max_episode_steps=650,
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
)
class JuicePreparationCloseFridgeEnv(JuicePreparationOpenFridgeEnv):
    """Close a door that starts at 75 degrees using the assigned left arm."""

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        initial_qpos = float(np.clip(
            self.door_lower + np.deg2rad(INITIAL_OPEN_ANGLE_DEG),
            self.door_lower,
            self.door_upper,
        ))
        qpos = self.fridge.get_qpos()
        qpos[:, 0] = initial_qpos
        self.fridge.set_qpos(qpos)
        self.fridge.set_qvel(torch.zeros_like(self.fridge.get_qvel()))
        self.initial_open_qpos = initial_qpos
        self.success_closed_qpos = door_success_qpos(
            self.door_lower, self.door_upper, SUCCESS_CLOSED_ANGLE_DEG
        )
        self.hand_contact_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def evaluate(self) -> dict[str, Any]:
        door_qpos = self.fridge.get_qpos()[:, 0]
        door_qvel = self.fridge.get_qvel()[:, 0]
        opening_angle = door_qpos - self.door_lower
        handle_position = self.current_handle_position_tensor()
        tcp_to_handle = torch.linalg.norm(
            self._task_tcp_position() - handle_position, dim=1
        )
        self.hand_contact_latched |= self._hand_contacts_door()

        denominator = max(self.initial_open_qpos - self.door_lower, 1e-6)
        close_progress = torch.clamp(
            (self.initial_open_qpos - door_qpos) / denominator, 0.0, 1.0
        )
        arm_qvel_norm = self._arm_qvel_norm()
        arm_static = arm_qvel_norm <= ARM_STATIC_THRESHOLD
        door_closed_enough, settled_now, self.success_settle_counter = (
            door_settle_update(
                door_qpos,
                door_qvel,
                self.success_settle_counter,
                threshold_qpos=self.success_closed_qpos,
                opening=False,
            )
        )
        door_static = torch.abs(door_qvel) <= DOOR_STATIC_THRESHOLD
        self.success_latched |= self.success_settle_counter >= SUCCESS_SETTLE_STEPS
        return {
            "success": self.success_latched,
            "task_success": self.success_latched,
            "door_qpos": door_qpos,
            "door_qvel": door_qvel,
            "opening_angle_rad": opening_angle,
            "opening_angle_deg": torch.rad2deg(opening_angle),
            "target_progress": close_progress,
            "door_closed_enough": door_closed_enough,
            "door_static": door_static,
            "arm_qvel_norm": arm_qvel_norm,
            "arm_static": arm_static,
            "hand_contact_latched": self.hand_contact_latched,
            "settle_counter": self.success_settle_counter,
            "settle_progress": torch.clamp(
                self.success_settle_counter.float() / SUCCESS_SETTLE_STEPS, 0.0, 1.0
            ),
            "handle_position": handle_position,
            "tcp_to_handle": tcp_to_handle,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reaching = 1.0 - torch.tanh(5.0 * info["tcp_to_handle"])
        reward = (
            reaching
            + 4.5 * info["target_progress"]
            + 0.5 * info["hand_contact_latched"].float()
            + info["door_closed_enough"].float()
            + 1.5 * info["settle_progress"]
        )
        return torch.where(
            info["task_success"], torch.full_like(reward, SUCCESS_REWARD), reward
        )


def main():
    parser = argparse.ArgumentParser(description="Close a refrigerator with the left hand")
    parser.add_argument("--env-id", default="JuicePreparationCloseFridge-Sharpa-v1")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--fridge-urdf", default=None)
    parser.add_argument("--steps", type=int, default=650)
    args = parser.parse_args()
    env = gym.make(
        args.env_id, num_envs=1, obs_mode="state", reward_mode="dense",
        control_mode=args.control_mode, render_mode="rgb_array",
        fridge_urdf=args.fridge_urdf,
    )
    try:
        env.reset(seed=0)
        for _ in range(args.steps):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            if bool(terminated.item()) or bool(truncated.item()):
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()

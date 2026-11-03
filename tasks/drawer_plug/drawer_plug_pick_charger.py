"""Stage 2 of ``DrawerPlugLong-Sharpa-v1``: place the charger on the table."""

import argparse
from typing import Any

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import gymnasium as gym
import torch

from tasks.config.hands import register_task
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.drawer_plug.drawer_plug_long import (
    DRAWER_PLUG_DEFAULT_CONTROL_MODE,
    DRAWER_PLUG_DEFAULT_ROBOT_UID_RIGHT,
    DrawerPlugLongEnv,
)


ENV_ID = "DrawerPlugPickCharger-Sharpa-v1"
MAX_EPISODE_STEPS = 900


@register_task(
    "DrawerPlugPickCharger",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DRAWER_PLUG_DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class DrawerPlugPickChargerEnv(DrawerPlugLongEnv):
    """Open drawer plus charger; move the charger to its table region."""

    has_socket = False
    show_charger_place_region = True

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx)
            self._initialize_workspace(env_idx)
            travel = self.drawer_success_travel + 0.04
            self._initialize_drawer(env_idx, b, travel=travel)
            self._place_charger_in_drawer(env_idx, b, travel)
            self._stage_drawer_open[env_idx] = True

    def evaluate(self):
        with torch.device(self.device):
            travel = self.drawer_travel
            drawer_open = travel > self.drawer_success_travel
            (
                charger_placed,
                charger_out,
                charger_bottom_z,
                charger_place_distance,
                charger_settled,
            ) = self._charger_table_state()
            success_condition = charger_placed
            self._stage_charger_placed |= (
                self._stage_drawer_open & success_condition
            )
            return dict(
                drawer_travel=travel,
                drawer_open=drawer_open,
                charger_out=charger_out,
                charger_placed=charger_placed,
                charger_bottom_z=charger_bottom_z,
                charger_place_distance=charger_place_distance,
                charger_settled=charger_settled,
                stage_charger_placed=self._stage_charger_placed.clone(),
                success_condition=success_condition,
                success=self._latch_success(success_condition),
            )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            drawer_qpos=info["drawer_travel"][:, None],
            charger_pos=self.charger.pose.p,
            charger_place_distance=info["charger_place_distance"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                charger_pose=self.charger.pose.raw_pose,
                drawer_pose=self.drawer_link.pose.raw_pose,
                drawer_xy=self._drawer_episode_xy,
                tcp_to_charger_pos=self.charger.pose.p - self.agent.tcp.pose.p,
                charger_to_place_pos=torch.tensor(
                    self.charger_place_center_xy, device=self.device
                ).expand(self.num_envs, -1)
                - self.charger.pose.p[:, :2],
                stages=info["stage_charger_placed"][:, None].float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_charger_placed"].to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=DRAWER_PLUG_DEFAULT_ROBOT_UID_RIGHT)
    parser.add_argument(
        "--control_mode", default=DRAWER_PLUG_DEFAULT_CONTROL_MODE
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/drawer_plug_pick_place")
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        reward_mode="sparse",
        robot_uids=args.robot_uids,
        control_mode=args.control_mode,
        render_mode="rgb_array",
    )
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_trajectory=False,
        save_video_trigger=lambda episode_idx: True,
        max_steps_per_video=args.steps,
        video_fps=25,
    )
    env.reset(seed=args.seed)
    for _ in range(args.steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        env.render()
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()

"""Stage 1 of ``DrawerPlugLong-Sharpa-v1``: pull the middle drawer open."""

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
    DRAWER_PLUG_DEFAULT_ROBOT_UID_LEFT,
    DrawerPlugLongEnv,
)


ENV_ID = "DrawerPlugOpenDrawer-Sharpa-v1"
MAX_EPISODE_STEPS = 600


@register_task(
    "DrawerPlugOpenDrawer",
    side="left",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DRAWER_PLUG_DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class DrawerPlugOpenDrawerEnv(DrawerPlugLongEnv):
    """Drawer cabinet only; open its middle drawer past the target travel."""

    has_socket = False
    has_charger = False
    show_charger_place_region = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx)
            self._initialize_workspace(env_idx)
            self._initialize_drawer(env_idx, b, travel=0.0)

    def evaluate(self):
        with torch.device(self.device):
            travel = self.drawer_travel
            drawer_open = travel > self.drawer_success_travel
            self._stage_drawer_open |= drawer_open
            return dict(
                drawer_travel=travel,
                drawer_open=drawer_open,
                stage_drawer_open=self._stage_drawer_open.clone(),
                success_condition=drawer_open,
                success=self._latch_success(drawer_open),
            )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            drawer_qpos=info["drawer_travel"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                drawer_pose=self.drawer_link.pose.raw_pose,
                drawer_xy=self._drawer_episode_xy,
                tcp_to_drawer_pos=self.drawer_link.pose.p - self.agent.tcp.pose.p,
                stages=info["stage_drawer_open"][:, None].float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_drawer_open"].to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=DRAWER_PLUG_DEFAULT_ROBOT_UID_LEFT)
    parser.add_argument(
        "--control_mode", default=DRAWER_PLUG_DEFAULT_CONTROL_MODE
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/drawer_plug_open")
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

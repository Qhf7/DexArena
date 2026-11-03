"""Stage 3 of ``DrawerPlugLong-Sharpa-v1``: insert the charger into the socket."""

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


ENV_ID = "DrawerPlugInsertPlug-Sharpa-v1"
MAX_EPISODE_STEPS = 900


@register_task(
    "DrawerPlugInsertPlug",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DRAWER_PLUG_DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class DrawerPlugInsertPlugEnv(DrawerPlugLongEnv):
    """Charger and socket only; insert both prongs into their slots."""

    has_drawer = False
    show_charger_place_region = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx)
            self._initialize_workspace(env_idx)
            self._place_socket(b)
            self._place_charger_on_table(b)
            self._stage_drawer_open[env_idx] = True
            self._stage_charger_placed[env_idx] = True

    def evaluate(self):
        with torch.device(self.device):
            (
                charger_inserted,
                charger_depth,
                charger_lateral,
                charger_alignment,
                charger_slot_error,
                charger_yaw_alignment,
                charger_settled,
            ) = self._charger_socket_state()
            self._stage_charger_inserted |= (
                self._stage_charger_placed & charger_inserted
            )
            return dict(
                charger_inserted=charger_inserted,
                charger_seated=charger_inserted,
                charger_depth=charger_depth,
                charger_lateral=charger_lateral,
                charger_alignment=charger_alignment,
                charger_slot_error=charger_slot_error,
                charger_yaw_alignment=charger_yaw_alignment,
                charger_settled=charger_settled,
                stage_charger_inserted=self._stage_charger_inserted.clone(),
                success_condition=charger_inserted,
                success=self._latch_success(charger_inserted),
            )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            charger_pos=self.charger.pose.p,
            socket_pos=self.socket.pose.p,
            charger_depth=info["charger_depth"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                charger_pose=self.charger.pose.raw_pose,
                socket_pose=self.socket.pose.raw_pose,
                tcp_to_charger_pos=self.charger.pose.p - self.agent.tcp.pose.p,
                charger_to_socket_pos=self.socket.pose.p - self.charger.pose.p,
                charger_lateral=info["charger_lateral"][:, None],
                charger_slot_error=info["charger_slot_error"][:, None],
                charger_yaw_alignment=info["charger_yaw_alignment"][:, None],
                stages=info["stage_charger_inserted"][:, None].float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_charger_inserted"].to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=DRAWER_PLUG_DEFAULT_ROBOT_UID_RIGHT)
    parser.add_argument(
        "--control_mode", default=DRAWER_PLUG_DEFAULT_CONTROL_MODE
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/drawer_plug_insert")
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

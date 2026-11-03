"""Stage 2 of ``NutBoltLong-Sharpa-v1``: place the nut on the fixed bolt plate."""

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

from tasks.nut_bolt.nut_bolt_long import (
    SCREWING_DEFAULT_CONTROL_MODE,
    SCREWING_DEFAULT_ROBOT_UID_RIGHT,
    NutBoltLongEnv,
)


ENV_ID = "NutBoltSeatNut-Sharpa-v1"
MAX_EPISODE_STEPS = 800


@register_task(
    "NutBoltSeatNut",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=SCREWING_DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class NutBoltSeatNutEnv(NutBoltLongEnv):
    """Fixed bolt plus nut; seat and release the nut on the invisible plate."""

    show_bolt_place_region = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx, "nut_pick_place")
            self._initialize_workspace(env_idx)
            self._place_bolt(env_idx, b, already_placed=True)
            self._place_nut_on_table(env_idx, b)
            self._update_platform_pose(env_idx)
            # Reconcile the kinematic hold with the lock masks this reset just
            # set (see NutBoltLongEnv._refresh_hold_state).
            self._refresh_hold_state()

    def evaluate(self):
        with torch.device(self.device):
            (
                nut_placed,
                nut_geometric,
                nut_threading_geometric,
                nut_aligned,
                nut_horizontal,
                nut_z_gap,
                nut_tilt_cos,
                nut_settled,
                nut_released,
            ) = self._nut_seating_state()
            nut_ready = self._nut_engagement_ready(nut_threading_geometric)
            newly_nut_placed = (
                (~self._stage_nut_placed)
                & self._stage_bolt_placed
                & nut_ready
            )
            self._stage_nut_placed |= self._stage_bolt_placed & nut_ready
            self._engage_newly_placed_nuts(newly_nut_placed)
            return dict(
                bolt_locked=self._bolt_locked.clone(),
                nut_placed=nut_placed,
                nut_ready=nut_ready,
                nut_engage_hold_count=self._nut_engage_hold_count.clone(),
                nut_geometric=nut_geometric,
                nut_threading_geometric=nut_threading_geometric,
                nut_aligned=nut_aligned,
                nut_horizontal=nut_horizontal,
                nut_z_gap=nut_z_gap,
                nut_tilt_cos=nut_tilt_cos,
                nut_settled=nut_settled,
                nut_released=nut_released,
                stage_nut_placed=self._stage_nut_placed.clone(),
                success_condition=nut_ready,
                success=nut_ready,
            )

    def _get_obs_extra(self, info: dict):
        nut_pose = self._nut_pose()
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bolt_pos=self.bolt.pose.p,
            nut_pos=nut_pose.p,
            nut_horizontal=info["nut_horizontal"][:, None],
            nut_z_gap=info["nut_z_gap"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                bolt_pose=self.bolt.pose.raw_pose,
                nut_pose=nut_pose.raw_pose,
                tcp_to_nut_pos=nut_pose.p - self.agent.tcp.pose.p,
                nut_to_bolt_pos=self.bolt.pose.p - nut_pose.p,
                nut_tilt_cos=info["nut_tilt_cos"][:, None],
                stages=info["stage_nut_placed"][:, None].float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_nut_placed"].to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=SCREWING_DEFAULT_ROBOT_UID_RIGHT)
    parser.add_argument("--control_mode", default=SCREWING_DEFAULT_CONTROL_MODE)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/screwing_nut_pick_place")
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

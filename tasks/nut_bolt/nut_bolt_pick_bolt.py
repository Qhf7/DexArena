"""Stage 1 of ``NutBoltLong-Sharpa-v1``: place the bolt upright in its table region."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any

import gymnasium as gym
import torch

from tasks.config.hands import register_task
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.nut_bolt.nut_bolt_long import (
    SCREWING_DEFAULT_CONTROL_MODE,
    SCREWING_DEFAULT_ROBOT_UID_LEFT,
    NutBoltLongEnv,
)

ENV_ID = "NutBoltPickBolt-Sharpa-v1"
MAX_EPISODE_STEPS = 700


@register_task(
    "NutBoltPickBolt",
    side="left",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=SCREWING_DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class NutBoltPickBoltEnv(NutBoltLongEnv):
    """Bolt only; set its hexagonal head down in the designated disc."""

    has_platform = False
    has_nut = False
    show_bolt_place_region = True

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx, "bolt_pick_place")
            self._initialize_workspace(env_idx)
            self._place_bolt(env_idx, b, already_placed=False)
            # Reconcile the kinematic hold with the lock masks this reset just
            # set (see NutBoltLongEnv._refresh_hold_state).
            self._refresh_hold_state()

    def evaluate(self):
        with torch.device(self.device):
            (
                bolt_place_candidate,
                bolt_place_distance,
                bolt_upright,
                bolt_on_table,
                bolt_settled,
                bolt_released,
            ) = self._bolt_placement_state()
            bolt_placed = self._qualify_bolt_placement(bolt_place_candidate)
            newly_placed = (~self._stage_bolt_placed) & bolt_placed
            self._stage_bolt_placed |= bolt_placed
            self._lock_newly_placed_bolts(newly_placed)
            return dict(
                bolt_placed=bolt_placed,
                bolt_place_candidate=bolt_place_candidate,
                bolt_place_hold_count=self._bolt_place_hold_count.clone(),
                bolt_place_distance=bolt_place_distance,
                bolt_upright=bolt_upright,
                bolt_on_table=bolt_on_table,
                bolt_settled=bolt_settled,
                bolt_released=bolt_released,
                bolt_locked=self._bolt_locked.clone(),
                stage_bolt_placed=self._stage_bolt_placed.clone(),
                success_condition=bolt_placed,
                success=bolt_placed,
            )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bolt_pos=self.bolt.pose.p,
            bolt_place_distance=info["bolt_place_distance"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                bolt_pose=self.bolt.pose.raw_pose,
                tcp_to_bolt_pos=self.bolt.pose.p - self.agent.tcp.pose.p,
                bolt_to_place_pos=torch.tensor(
                    self.bolt_place_center_xy, device=self.device
                ).expand(self.num_envs, -1)
                - self.bolt.pose.p[:, :2],
                stages=info["stage_bolt_placed"][:, None].float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_bolt_placed"].to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=SCREWING_DEFAULT_ROBOT_UID_LEFT)
    parser.add_argument("--control_mode", default=SCREWING_DEFAULT_CONTROL_MODE)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/screwing_bolt_pick_place")
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

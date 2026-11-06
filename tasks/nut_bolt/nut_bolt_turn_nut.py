"""Stage 3 of ``NutBoltLong-Sharpa-v1``: turn the seated nut counterclockwise by pi."""

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


ENV_ID = "NutBoltTurnNut-Sharpa-v1"
MAX_EPISODE_STEPS = 900


@register_task(
    "NutBoltTurnNut",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=SCREWING_DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class NutBoltTurnNutEnv(NutBoltLongEnv):
    """Nut starts seated; accumulate a net counterclockwise pi turn from above."""

    show_bolt_place_region = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx, "turn")
            self._initialize_workspace(env_idx)
            self._place_bolt(env_idx, b, already_placed=True)
            self._place_nut_on_platform(env_idx, b)
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
            just_placed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            nut_turned, rotation_delta = self._update_screwing_state(
                just_placed, nut_threading_geometric
            )
            success_condition = (
                nut_turned & nut_threading_geometric & self._nut_engaged
            )
            turn_success = self._latch_success(success_condition)
            self._stage_turned |= turn_success
            return dict(
                nut_placed=nut_placed,
                nut_geometric=nut_geometric,
                nut_threading_geometric=nut_threading_geometric,
                nut_aligned=nut_aligned,
                nut_horizontal=nut_horizontal,
                nut_z_gap=nut_z_gap,
                nut_tilt_cos=nut_tilt_cos,
                nut_settled=nut_settled,
                nut_released=nut_released,
                nut_turned=nut_turned,
                cum_rotation=self._cum_rotation.clone(),
                rotation_delta=rotation_delta,
                threaded_depth=self._threaded_depth.clone(),
                stage_turned=self._stage_turned.clone(),
                success_condition=turn_success,
                success=turn_success,
            )

    def _get_obs_extra(self, info: dict):
        nut_pose = self._nut_pose()
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bolt_pos=self.bolt.pose.p,
            nut_pos=nut_pose.p,
            cum_rotation=info["cum_rotation"][:, None],
            threaded_depth=info["threaded_depth"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                bolt_pose=self.bolt.pose.raw_pose,
                nut_pose=nut_pose.raw_pose,
                tcp_to_nut_pos=nut_pose.p - self.agent.tcp.pose.p,
                nut_to_bolt_pos=self.bolt.pose.p - nut_pose.p,
                nut_horizontal=info["nut_horizontal"][:, None],
                nut_z_gap=info["nut_z_gap"][:, None],
                nut_tilt_cos=info["nut_tilt_cos"][:, None],
                stages=info["stage_turned"][:, None].float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_turned"].to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=SCREWING_DEFAULT_ROBOT_UID_RIGHT)
    parser.add_argument("--control_mode", default=SCREWING_DEFAULT_CONTROL_MODE)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/screwing_turn")
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

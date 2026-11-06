"""Stage 5 of ``CabinetPouringLong-Sharpa-v1``: tip the open bottle out into the bowl.

The filled bottle and the bowl are on the work table; there is no cabinet, and
no cap at all -- it has already been unscrewed and parked, so the bottle is
open.  The bottle spawns anywhere the placement stage of the long task accepts
it (the disc around ``place_center_xy``) and the bowl over its own
randomisation range, exactly as in the long task.

Success is the long task's last milestone, ``stage_poured``: at least
``pour_ratio_threshold`` of the particles inside the bowl with no more than
``max_spilled`` on the table, with none airborne, held for ten control steps.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.config.hands import register_task
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.cabinet_pouring.cabinet_pouring_long import DEFAULT_CONTROL_MODE, DEFAULT_ROBOT_UID_RIGHT, CabinetPouringLongEnv


ENV_ID = "CabinetPouringPourBeads-Sharpa-v1"

MAX_EPISODE_STEPS = 700


@register_task(
    "CabinetPouringPourBeads",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class CabinetPouringPourBeadsEnv(CabinetPouringLongEnv):
    """Open bottle plus the bowl; pour the particles across."""

    has_cabinet = False
    has_cap = False
    placement_region = None
    lock_bottle_during_cap_stages = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx)
            self._initialize_workspace(env_idx)

            source_xyz = self._placed_bottle_xyz(b)
            source_q = self._place_bottle(b, source_xyz)
            self._place_target(b)
            self._place_particles(b, source_xyz, source_q)

            # The stages this one starts from are already done.
            self._stage_door[env_idx] = True
            self._stage_bottle_out[env_idx] = True
            self._stage_bottle_placed[env_idx] = True
            self._stage_cap_turned[env_idx] = True
            self._stage_cap_placed[env_idx] = True

    def evaluate(self):
        with torch.device(self.device):
            source_pose = self.source.pose
            (
                num_in_target,
                num_in_source,
                num_spilled,
                num_airborne,
            ) = self._pour_counts(source_pose)
            tilt_cos, source_upright, settled = self._source_posture(source_pose)

            pour_ratio, pour_candidate, poured = self._pour_success_state(
                num_in_target, num_spilled, num_airborne
            )
            self._stage_poured |= poured
            return dict(
                num_in_target=num_in_target,
                num_in_source=num_in_source,
                num_spilled=num_spilled,
                num_airborne=num_airborne,
                pour_ratio=pour_ratio,
                pour_candidate=pour_candidate,
                pour_hold_count=self._pour_hold_count.clone(),
                tilt_cos=tilt_cos,
                source_upright=source_upright,
                settled=settled,
                poured=poured,
                stage_poured=self._stage_poured.clone(),
                success_condition=poured,
                success=poured,
            )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_pos=self.target.pose.p,
            pour_ratio=info["pour_ratio"][:, None],
            tilt_cos=info["tilt_cos"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                source_pose=self.source.pose.raw_pose,
                target_pose=self.target.pose.raw_pose,
                tcp_to_source_pos=self.source.pose.p - self.agent.tcp.pose.p,
                source_to_target_pos=self.target.pose.p - self.source.pose.p,
                num_spilled=info["num_spilled"][:, None].float(),
                stages=info["stage_poured"][:, None].float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_poured"].to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=DEFAULT_ROBOT_UID_RIGHT)
    parser.add_argument("--control_mode", default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/cabinet_pour_pour")
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
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Robot DoF", env.unwrapped.agent.robot.dof)

    env.reset(seed=args.seed)
    for step in range(args.steps):
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
        if step % 25 == 0:
            print(
                f"step {step}: src {int(info['num_in_source'])} "
                f"tgt {int(info['num_in_target'])} spill {int(info['num_spilled'])} "
                f"reward {float(reward):.1f}"
            )
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()

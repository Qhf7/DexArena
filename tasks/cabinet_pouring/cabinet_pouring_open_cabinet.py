"""Stage 1 of ``CabinetPouringLong-Sharpa-v1``: swing the cabinet door open.

Only the cabinet stands on the work table -- the bottle, its cap and the bowl
are all left out, since this stage never touches them.  The cabinet is placed
and randomised over exactly the range the long-horizon task uses, its door
starts shut, and everything else (table, robot start pose, cameras)
is inherited from ``CabinetPouringLongEnv``.

Success is the long task's first milestone, ``stage_door_open``, with the
same 80-degree acceptance angle as the full task, held for
five consecutive control steps as in every other task in the repo.
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

from tasks.cabinet_pouring.cabinet_pouring_long import DEFAULT_CONTROL_MODE, DEFAULT_ROBOT_UID_LEFT, CabinetPouringLongEnv


ENV_ID = "CabinetPouringOpenCabinet-Sharpa-v1"

MAX_EPISODE_STEPS = 600


@register_task(
    "CabinetPouringOpenCabinet",
    side="left",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class CabinetPouringOpenCabinetEnv(CabinetPouringLongEnv):
    """Cabinet only; open the door past 80 degrees."""

    has_bottle = False
    has_cap = False
    has_target = False
    placement_region = None
    lock_bottle_during_cap_stages = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx)
            self._initialize_workspace(env_idx)
            self._initialize_cabinet(env_idx, b, door_open=False)

    def evaluate(self):
        with torch.device(self.device):
            door_angle, door_open_now, door_open = self._door_status()
            self._stage_door |= door_open
            return dict(
                door_angle=door_angle,
                door_angle_deg=torch.rad2deg(door_angle),
                door_open_now=door_open_now,
                door_open=door_open,
                door_open_hold_count=self._door_open_hold_count.clone(),
                stage_door_open=self._stage_door.clone(),
                success_condition=door_open,
                success=door_open,
            )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            door_qpos=info["door_angle"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                door_pose=self.door_link.pose.raw_pose,
                cabinet_xy=self._cabinet_episode_xy,
                tcp_to_door_pos=self.door_link.pose.p - self.agent.tcp.pose.p,
                stages=info["stage_door_open"][:, None].float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_door_open"].to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=DEFAULT_ROBOT_UID_LEFT)
    parser.add_argument("--control_mode", default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="videos/cabinet_pour_door_open")
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
                f"step {step}: door {float(info['door_angle_deg']):.1f} deg  "
                f"reward {float(reward):.1f}"
            )
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()

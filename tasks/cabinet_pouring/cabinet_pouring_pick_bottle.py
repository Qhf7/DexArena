"""Stage 2 of ``CabinetPouringLong-Sharpa-v1``: take the bottle out and stand it on the table.

The cabinet and the (capped, filled) bottle are on the work table; the bowl is
left out because this stage never pours.  Both spawn exactly where the long
task puts them -- the cabinet over its own randomisation range, the bottle on
the cabinet's upper shelf -- and the door starts wide open, i.e. the state the
door-opening stage hands over.

Success reuses the long task's second milestone unchanged: the bottle must
first clear the cabinet's front face and then come to rest upright on the
tabletop inside the placement disc, held for five consecutive control steps.
The two are latched in order, so the bottle cannot count as placed unless it
really came out of the cabinet first.
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


ENV_ID = "CabinetPouringPickBottle-Sharpa-v1"

MAX_EPISODE_STEPS = 900


@register_task(
    "CabinetPouringPickBottle",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class CabinetPouringPickBottleEnv(CabinetPouringLongEnv):
    """Cabinet (door open) plus the bottle inside it; set the bottle down."""

    has_target = False
    placement_region = "bottle"
    lock_bottle_during_cap_stages = False

    def __init__(self, *args, **kwargs):
        if "bottle_stage_twins" in kwargs and not bool(
            kwargs["bottle_stage_twins"]
        ):
            raise ValueError(
                "CabinetPouringPickBottle-Sharpa-v1 is a strict A->B-stage subset "
                "and therefore requires bottle_stage_twins=True."
            )
        kwargs["bottle_stage_twins"] = True
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx)
            self._initialize_workspace(env_idx)
            cabinet_xy_offset = self._initialize_cabinet(env_idx, b, door_open=True)

            source_xyz = self._shelf_bottle_xyz(b, cabinet_xy_offset)
            source_q = self._place_bottle(b, source_xyz)
            self._place_cap_on_bottle(env_idx, b, source_xyz, source_q)
            self._reset_bottle_stage_twins(env_idx)
            self._place_particles(b, source_xyz, source_q)

            # The stage this one starts from is already done.
            self._stage_door[env_idx] = True

    def evaluate(self):
        with torch.device(self.device):
            (
                source_pose,
                tilt_cos,
                source_upright,
                settled,
                bottle_released,
                bottle_place_candidate,
                bottle_placed,
            ) = self._bottle_placement_transition_state()
            bottle_out = self._bottle_out_of_cabinet(source_pose)

            self._stage_bottle_out |= self._stage_door & bottle_out
            newly_bottle_placed = (
                ~self._stage_bottle_placed
                & self._stage_bottle_out
                & bottle_placed
            )
            self._capture_new_bottle_locks(newly_bottle_placed)
            self._stage_bottle_placed |= self._stage_bottle_out & bottle_placed
            success_condition = self._stage_bottle_placed.clone()

            door_angle = self.cabinet.get_qpos()[:, 0]
            return dict(
                door_angle=door_angle,
                door_angle_deg=torch.rad2deg(door_angle),
                tilt_cos=tilt_cos,
                source_upright=source_upright,
                settled=settled,
                bottle_out=bottle_out,
                bottle_place_candidate=bottle_place_candidate,
                bottle_released=bottle_released,
                bottle_place_hold_count=self._bottle_place_hold_count.clone(),
                bottle_placed=bottle_placed,
                stage_bottle_out=self._stage_bottle_out.clone(),
                stage_bottle_placed=self._stage_bottle_placed.clone(),
                bottle_locked=self._bottle_lock_mask(),
                bottle_rep=self._active_bottle_rep.clone(),
                success_condition=success_condition,
                success=success_condition,
            )

    def _get_obs_extra(self, info: dict):
        source_pose = self._source_pose()
        cap_pose = torch.cat([self._cap_p(), self._cap_q()], dim=-1)
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            door_qpos=info["door_angle"][:, None],
            tilt_cos=info["tilt_cos"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                source_pose=source_pose.raw_pose,
                cap_pose=cap_pose,
                door_pose=self.door_link.pose.raw_pose,
                cabinet_xy=self._cabinet_episode_xy,
                tcp_to_source_pos=source_pose.p - self.agent.tcp.pose.p,
                source_to_place_pos=torch.tensor(
                    self.place_center_xy, device=self.device
                ).expand(self.num_envs, -1)
                - source_pose.p[:, :2],
                stages=torch.stack(
                    [info["stage_bottle_out"], info["stage_bottle_placed"]], dim=-1
                ).float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_bottle_placed"].to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--robot_uids", default=DEFAULT_ROBOT_UID_RIGHT)
    parser.add_argument("--control_mode", default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", default="videos/cabinet_pour_bottle_pick_place"
    )
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
                f"step {step}: out {bool(info['bottle_out'])} "
                f"placed {bool(info['bottle_placed'])} reward {float(reward):.1f}"
            )
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()

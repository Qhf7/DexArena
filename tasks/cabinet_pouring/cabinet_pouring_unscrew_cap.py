"""Merged CabinetPour cap stage: unscrew, remove, and place in one episode.

This is the production stage-3/4 subtask of CabinetPouringLong-Sharpa-v1. It starts from
exactly the long task's fixed B bottle representation. After a half turn and a
short low-speed pause, the attached joint cap is replaced in-hand by the +2 mm
free cap. The operator then carries that same cap to the tabletop drop-off.
Detachment is an internal transition; only stable, hand-clear placement is
success.
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

from tasks.cabinet_pouring.cabinet_pouring_long import DEFAULT_CONTROL_MODE, CabinetPouringLongEnv


ENV_ID = "CabinetPouringUnscrewCap-Sharpa-v1"
MAX_EPISODE_STEPS = 1400
DEFAULT_CAP_REMOVE_PLACE_ROBOT_UID = (
    "panda_sharpa_hand_right_angled_mobile_base"
)


@register_task(
    "CabinetPouringUnscrewCap",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class CabinetPouringUnscrewCapEnv(CabinetPouringLongEnv):
    """Strict B-stage subset: unscrew and park the cap continuously."""

    has_cabinet = False
    has_target = False
    placement_region = "sequence"
    lock_bottle_during_cap_stages = True

    def __init__(self, *args, **kwargs):
        # Production plant validated for this task family.
        kwargs.setdefault("cap_material_friction", 0.8)
        kwargs.setdefault("cap_running_torque", 0.40)
        kwargs.setdefault("cap_breakaway_torque", 0.40)
        kwargs.setdefault("cap_joint_armature", 5e-5)

        # Backward-compatible CLI spelling. Representation B itself provides
        # the fixed root; accepting True avoids breaking deployed launch files.
        legacy_fixed_root = kwargs.pop("cap_contact_fixed_root", None)
        if legacy_fixed_root is not None and not bool(legacy_fixed_root):
            raise ValueError(
                f"{ENV_ID} is a strict B-stage subset and requires "
                "a fixed bottle root"
            )
        if "bottle_stage_twins" in kwargs and not bool(
            kwargs["bottle_stage_twins"]
        ):
            raise ValueError(
                f"{ENV_ID} requires bottle_stage_twins=True"
            )
        kwargs["bottle_stage_twins"] = True
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_episode_buffers(env_idx)
            self._initialize_workspace(env_idx)

            source_xyz = self._placed_bottle_xyz(b)
            source_q = self._place_bottle(b, source_xyz)
            self._place_cap_on_bottle(env_idx, b, source_xyz, source_q)
            self._reset_bottle_stage_twins(env_idx)
            self._place_particles(b, source_xyz, source_q)
            self._activate_fixed_bottle_at_reset(env_idx)

            # This subtask starts at the long task's accepted B boundary.
            self._stage_door[env_idx] = True
            self._stage_bottle_out[env_idx] = True
            self._stage_bottle_placed[env_idx] = True
            self._sync_placement_region_markers(env_idx)

    def evaluate(self):
        with torch.device(self.device):
            source_pose = self._source_pose()
            tilt_cos, source_upright, settled = self._source_posture(source_pose)
            cap_turned, cap_settled, cap_placed = self._cap_status()

            self._stage_cap_turned |= (
                self._stage_bottle_placed & cap_turned
            )
            newly_cap_placed = ~self._stage_cap_placed & cap_placed
            self._stage_cap_placed |= cap_placed
            self._swap_to_open_bottle(newly_cap_placed)
            self._sync_placement_region_markers()

            return dict(
                tilt_cos=tilt_cos,
                source_upright=source_upright,
                settled=settled,
                cap_twist=self._cap_twist.clone(),
                cap_twist_deg=torch.rad2deg(self._cap_twist),
                cap_detached=self._cap_detached.clone(),
                bottle_rep=self._active_bottle_rep.clone(),
                cap_settled=cap_settled,
                cap_turned=cap_turned,
                cap_turn_hold_count=self._cap_turn_hold_count.clone(),
                cap_placed=cap_placed,
                cap_place_hold_count=self._cap_place_hold_count.clone(),
                stage_cap_turned=self._stage_cap_turned.clone(),
                stage_cap_placed=self._stage_cap_placed.clone(),
                success_condition=cap_placed,
                success=cap_placed,
            )

    def _get_obs_extra(self, info: dict):
        source_pose = self._source_pose()
        cap_p = self._cap_p()
        cap_q = self._cap_q()
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            cap_pos=cap_p,
            cap_twist=info["cap_twist"][:, None],
            cap_detached=info["cap_detached"][:, None].float(),
            tilt_cos=info["tilt_cos"][:, None],
        )
        if "state" in self.obs_mode:
            obs.update(
                source_pose=source_pose.raw_pose,
                cap_pose=torch.cat([cap_p, cap_q], dim=-1),
                bottle_rep=info["bottle_rep"][:, None].float(),
                tcp_to_source_pos=source_pose.p - self.agent.tcp.pose.p,
                tcp_to_cap_pos=cap_p - self.agent.tcp.pose.p,
                cap_to_place_pos=torch.tensor(
                    self.cap_place_center_xy, device=self.device
                ).expand(self.num_envs, -1)
                - cap_p[:, :2],
                stages=torch.stack(
                    [
                        info["stage_cap_turned"],
                        info["stage_cap_placed"],
                    ],
                    dim=-1,
                ).float(),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["stage_cap_placed"].to(torch.float32)


def main(
    env_id: str = ENV_ID,
    output_dir: str = "videos/cabinet_pour_cap_remove_place",
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=env_id)
    parser.add_argument(
        "--robot_uids", default=DEFAULT_CAP_REMOVE_PLACE_ROBOT_UID
    )
    parser.add_argument("--control_mode", default=DEFAULT_CONTROL_MODE)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=output_dir)
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
        _, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        env.render()
        if step % 25 == 0:
            print(
                f"step {step}: twist {float(info['cap_twist_deg']):.1f} deg  "
                f"detached {bool(info['cap_detached'])} "
                f"placed {bool(info['cap_placed'])} "
                f"reward {float(reward):.1f}"
            )
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()


if __name__ == "__main__":
    main()

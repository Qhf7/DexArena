"""Subtask 3: turn the rail's wing screw a quarter turn by hand.

The panel (flat in the assembly zone) and the rail seated in its groove are fixed
templates. The rail carries a WING screw: a revolute head on the rail web with a
64 mm wing bar, thread resistance and a stop at 90 degrees. The task is to turn it
clockwise (seen from above) to the stop with the fingers. Success is the screw
angle alone; touching the head is a reward stage and, in the long task, the
"which hand did it" gate.

Task simplification: this replaces the YCB screwdriver + Phillips
recess version, measured on its 111 recorded demonstrations: open-loop action
replay reached success in 11/111 (8/111 on the last frame). The arm was never
near a joint limit while screwing (0/111 episodes within 0.1 rad on any joint,
one episode on joint 4), so posture was not the cause. The replay lost the TOOL:
in 24 probed episodes the screwdriver had drifted in the grasp by a median 13 mm
at the recorded engagement frame (recess capture 4 mm), the blade entered the
recess in 4/24 and the tool was dropped in 8/24. The quarter turn itself was made
by rolling the handle in the fingers (hand joints moved 0.61 rad, the wrist
0.03 rad), which no open-loop replay reproduces. Turning a fixed-axis part
directly by hand replays at 96% in this pipeline (NutBoltTurnNut-Sharpa-v1, 118/123),
hence the wing screw.

The rail with the wing screw is built HERE, not in furniture_rail_insert_groove:
that module is in the code fingerprint of the collected RailInsert dataset and
stays byte-identical. The body geometry (web, flanges, lips, slots) is the same
part; only the revolute head differs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np
import sapien
import sapien.physx as physx
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from tasks.config.hands import register_task
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.robots.panda_gripper_right_mobile_base import PandaGripperRightMobileBase
from tasks.robots.panda_sharpa_hand_right_mobile_base import PandaSharpaHandRightAngledMobileBase
from tasks.robots.panda_wuji_hand_right_mobile_base import PandaWujiHandRightMobileBase
from tasks.furniture_assembly.furniture_assembly_lay_flat_panel import (
    PLACE_REGION_HALF_SIZE,
    build_place_marker,
    DEFAULT_CONTROL_MODE,
    MOBILE_ROBOT_UIDS,
    PANEL_HALF,
    PLACE_CENTER,
    FurnitureAssemblyLayFlatPanelEnv,
    ROBOT_ROOT_POSE,
    build_drawer_panel,
    build_dexarena_base_camera,
    make_render_material,
    yaw_to_quat_wxyz,
    finger_contact_count,
)
from tasks.furniture_assembly.furniture_assembly_insert_rail import (
    RAIL_DENSITY,
    RAIL_HALF,
    RAIL_SCREW_LOCAL_POSITIONS,
    RAIL_SEATED_X_LOCAL,
    RAIL_SEATED_Z_LOCAL,
    RAIL_SHEET_T,
    SCREW_JOINT_DAMPING,
    SCREW_JOINT_FRICTION,
    SCREW_JOINT_LIMIT,
    apply_screw_thread_resistance,
)
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.config.layout_randomization import layout_rng, uniform_offset
from tasks.config.object_materials import object_material
from tasks.scene_builders import TableSceneBuilder
from tasks.config.sim_profiles import make_sim_config
from tasks.config.workspace_rig import (
    ARM_INIT_QPOS,
    PLATFORM_ROOT_X,
    set_work_table_pose,
)
from tasks.config.task_state import DeclarativeTaskStateMixin, TaskStateField

GRIPPER_ROBOT_UID = "panda_gripper_right_mobile_base"
ENV_IDS = (
    "FurnitureAssemblyFastenScrew-Sharpa-v1",
    "FurnitureAssemblyFastenScrew-Wuji-v1",
    "FurnitureAssemblyFastenScrew-XHand-v1",
    "FurnitureAssemblyFastenScrew-Gripper-v1",
)
DEFAULT_VIDEO_DIR = "tasks/assets/task_object_meshes/furniture/videos_screw_fasten"


# ---------------------------------------------------------------------------
# Wing screw geometry (rail-local metres; the head link's origin is on the web
# top at RAIL_SCREW_LOCAL_POSITIONS[0], z up, the wing bar along the rail at 0).
# ---------------------------------------------------------------------------
WING_SCREW_STEM_RADIUS = 0.005
WING_SCREW_STEM_HEIGHT = 0.016      # web top (-3.5 mm) -> 6.5 mm above the flange tops (+6 mm)
WING_SCREW_HUB_RADIUS = 0.012
WING_HALF = (0.032, 0.0045, 0.010)  # half extents: span (64 mm tip to tip), thickness, height
WING_SCREW_DENSITY = 1400.0         # glass-filled nylon knob, ~30 g in total
WING_TIP_LEVER = WING_HALF[0]

# "Screw fastened" -- the ONE predicate of stage 3, evaluated by this subtask and
# by the long task.
SCREW_TARGET_ROTATION = float(np.pi / 2.0)
SCREW_TARGET_TOLERANCE = 0.05       # rad (~3 deg) short of the joint stop still counts


def screw_fastened(screw_angle):
    """The wing sits at (or within SCREW_TARGET_TOLERANCE of) its 90-degree stop."""
    return screw_angle >= SCREW_TARGET_ROTATION - SCREW_TARGET_TOLERANCE


def _add_rail_body(builder, phys) -> None:
    """The slider-rail body as furniture_rail_insert_groove.build_slider_rail makes it:
    collision = web + two flanges (aluminium, RAIL_DENSITY); the lips and the stamped
    slots are visual only. Kept geometry-identical so the long task's rail is the part
    the RailInsert subtask seats."""
    steel = make_render_material([0.66, 0.68, 0.72, 1.0], roughness=0.28, metallic=0.92)
    dark = make_render_material([0.42, 0.44, 0.48, 1.0], roughness=0.42, metallic=0.85)
    hx, hy, hz = RAIL_HALF
    t = RAIL_SHEET_T / 2.0
    web_pose = sapien.Pose(p=[0.0, 0.0, -hz + t])
    builder.add_box_collision(pose=web_pose, half_size=[hx, hy, t], material=phys, density=RAIL_DENSITY)
    builder.add_box_visual(pose=web_pose, half_size=[hx, hy, t], material=steel)
    for sgn in (-1.0, 1.0):
        flange_pose = sapien.Pose(p=[0.0, sgn * (hy - t), 0.0])
        builder.add_box_collision(pose=flange_pose, half_size=[hx, t, hz], material=phys, density=RAIL_DENSITY)
        builder.add_box_visual(pose=flange_pose, half_size=[hx, t, hz], material=steel)
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, sgn * (hy - t), hz + 0.0003]),
            half_size=[hx, t * 1.4, 0.0004], material=dark,
        )
    for i in range(-4, 5):
        builder.add_box_visual(
            pose=sapien.Pose(p=[i * 0.021, 0.0, -hz + 2.0 * t + 0.0002]),
            half_size=[0.0055, 0.0035, 0.0002], material=dark,
        )


def _add_wing_screw_head_link(art, rail_link, phys):
    """Revolute wing screw on the rail web at RAIL_SCREW_LOCAL_POSITIONS[0].

    Stem through the channel, a round hub and a wing bar above the flange tops.
    Positive joint angle is CLOCKWISE seen from above (tightening), travel 0..90 deg,
    the same convention as the Phillips head it replaces.
    """
    knob = make_render_material([0.10, 0.10, 0.11, 1.0], roughness=0.60, metallic=0.10)
    brass = make_render_material([0.92, 0.78, 0.26, 1.0], roughness=0.50, metallic=0.30)
    head = art.create_link_builder(rail_link)
    head.set_name("screw_head")
    head.set_joint_name("screw_joint")
    cylinder_q = [float(np.cos(-np.pi / 4)), 0.0, float(np.sin(-np.pi / 4)), 0.0]     # cylinder axis x -> +z
    stem_pose = sapien.Pose(p=[0.0, 0.0, WING_SCREW_STEM_HEIGHT / 2.0], q=cylinder_q)
    head.add_cylinder_collision(pose=stem_pose, radius=WING_SCREW_STEM_RADIUS,
                                half_length=WING_SCREW_STEM_HEIGHT / 2.0, material=phys, density=WING_SCREW_DENSITY)
    head.add_cylinder_visual(pose=stem_pose, radius=WING_SCREW_STEM_RADIUS,
                             half_length=WING_SCREW_STEM_HEIGHT / 2.0, material=brass)
    hub_z = WING_SCREW_STEM_HEIGHT + WING_HALF[2]
    hub_pose = sapien.Pose(p=[0.0, 0.0, hub_z], q=cylinder_q)
    head.add_cylinder_collision(pose=hub_pose, radius=WING_SCREW_HUB_RADIUS, half_length=WING_HALF[2],
                                material=phys, density=WING_SCREW_DENSITY)
    head.add_cylinder_visual(pose=hub_pose, radius=WING_SCREW_HUB_RADIUS, half_length=WING_HALF[2], material=knob)
    wing_pose = sapien.Pose(p=[0.0, 0.0, hub_z])
    head.add_box_collision(pose=wing_pose, half_size=list(WING_HALF), material=phys, density=WING_SCREW_DENSITY)
    head.add_box_visual(pose=wing_pose, half_size=list(WING_HALF), material=knob)
    # A brass stripe along the wing top so its orientation reads from above.
    head.add_box_visual(pose=sapien.Pose(p=[0.0, 0.0, hub_z + WING_HALF[2] + 0.0003]),
                        half_size=[WING_HALF[0], 0.0015, 0.0003], material=brass)
    sx, sy, sz = RAIL_SCREW_LOCAL_POSITIONS[0]
    screw_axis_q = [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0]        # joint x -> -z
    head.set_joint_properties(
        "revolute",
        limits=[[0.0, SCREW_JOINT_LIMIT]],
        pose_in_parent=sapien.Pose(p=[sx, sy, sz], q=screw_axis_q),
        pose_in_child=sapien.Pose(q=screw_axis_q),
        friction=SCREW_JOINT_FRICTION,
        damping=SCREW_JOINT_DAMPING,
    )
    return head


def build_wing_screw_rail(
    scene: ManiSkillScene,
    name: str = "slider_rail",
    kinematic: bool = False,
):
    """The furniture rail carrying the wing screw (stage 3 of the family).

    Root link "rail_body" (the RailInsert part) + revolute link "screw_head" on
    joint "screw_joint" with the shared thread resistance.
    """
    art = scene.create_articulation_builder()
    art.disable_self_collisions = True
    builder = art.create_link_builder()          # the rail body is the root link
    builder.set_name("rail_body")
    phys = object_material("metal")
    _add_rail_body(builder, phys)
    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.5])
    _add_wing_screw_head_link(art, builder, phys)
    art.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.5])
    built = art.build(name=name, fix_root_link=bool(kinematic))
    apply_screw_thread_resistance(built)
    return built


@register_task(
    "FurnitureAssemblyFastenScrew",
    side="right",
    max_episode_steps=600,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    render_mode="rgb_array",
)
class FurnitureAssemblyFastenScrewEnv(DeclarativeTaskStateMixin, BaseEnv):
    # Task bookkeeping that physics cannot re-derive. Restoring a
    # mid-episode frame without these silently rewinds every milestone,
    # which breaks offline re-render, demo-init and end-frame QC.
    # APPEND ONLY -- reordering breaks older recordings.
    TASK_STATE_KEY = "furniture_wing_screw_task_state"
    TASK_STATE_FIELDS = (
        TaskStateField("touched_once", 1, torch.bool),
        TaskStateField("rotation_progress", 1, torch.float32),
        TaskStateField("success_counter", 1, torch.int32),
    )
    SUPPORTED_ROBOTS = sorted(MOBILE_ROBOT_UIDS)
    agent: Union[
        PandaGripperRightMobileBase,
        PandaSharpaHandRightAngledMobileBase,
        PandaWujiHandRightMobileBase,
    ]

    # Task layer: the family's single-sourced layout. The panel sits where the
    # long task leaves it after stage 1, anywhere the PanelLayFlat predicate
    # accepts; the per-episode offsets below are the family's rack half-ranges.
    PANEL_CENTER = PLACE_CENTER
    TARGET_ROTATION = SCREW_TARGET_ROTATION
    TARGET_TOLERANCE = SCREW_TARGET_TOLERANCE
    TIP_CONTACT_FORCE = 0.6
    SUCCESS_HOLD_STEPS = 1
    SUCCESS_REWARD = 10.0

    _reset_agent = FurnitureAssemblyLayFlatPanelEnv._reset_agent
    _load_lighting = FurnitureAssemblyLayFlatPanelEnv._load_lighting

    def __init__(
        self,
        *args,
        robot_uids: str = GRIPPER_ROBOT_UID,
        robot_init_qpos_noise: float = 0.01,
        panel_random_x: float = 0.030,
        panel_random_y: float = 0.025,
        panel_random_yaw_deg: float = 9.0,
        visual_detail: bool = True,
        terminate_on_success: bool = False,
        render_camera_eye=(0.75, -1.65, 0.92),
        render_camera_target=(-0.38, 0.02, 0.32),
        render_camera_fov: float = 1.0,
        render_width: int = 720,
        render_height: int = 720,
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.panel_random_x = float(panel_random_x)
        self.panel_random_y = float(panel_random_y)
        self.panel_random_yaw_deg = float(panel_random_yaw_deg)
        self.visual_detail = bool(visual_detail)
        self.terminate_on_success = bool(terminate_on_success)
        self.render_camera_eye = list(render_camera_eye)
        self.render_camera_target = list(render_camera_target)
        self.render_camera_fov = float(render_camera_fov)
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        super().__init__(
            *args, robot_uids=robot_uids, num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq, **kwargs,
        )

    @property
    def _default_sim_config(self):
        # Environment layer: the locked fleet profile.
        return make_sim_config(
            max_rigid_contact_count=2**21,
            max_rigid_patch_count=2**19,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.58, -1.41, 0.85], [-0.42, -0.04, 0.30])
        return [
            build_dexarena_base_camera(name="base_camera"),
            CameraConfig("external_camera", pose, 320, 240, np.deg2rad(58), 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(self.render_camera_eye, self.render_camera_target)
        return CameraConfig(
            "render_camera", pose, self.render_width, self.render_height,
            self.render_camera_fov, 0.01, 100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, ROBOT_ROOT_POSE)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # Environment layer: the work table belongs to the shared rig.
        set_work_table_pose(self)

        base_color = (0.725, 0.541, 0.369, 1.0)
        self.panel = build_drawer_panel(
            self.scene, base_color=base_color, visual_detail=self.visual_detail,
            kinematic=True,
        )
        self.place_marker = build_place_marker(self.scene, PLACE_REGION_HALF_SIZE)   # same zone marker as the long task
        self.rail = build_wing_screw_rail(self.scene, kinematic=True)
        # The screw is the rail's own revolute child (fixed rail root here).
        self.screw = self.rail
        self.rail_body = self.rail.links_map["rail_body"]
        self.screw_joint = self.rail.active_joints_map["screw_joint"]
        self.screw_head = self.rail.links_map["screw_head"]

        self.rotation_progress = torch.zeros((self.num_envs,), device=self.device)
        self.success_counter = torch.zeros(
            (self.num_envs,), dtype=torch.int32, device=self.device
        )
        self.touched_once = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

    def _contact_count(self, actor) -> torch.Tensor:
        """Distinct fingers in force contact with `actor` (collision-bearing links)."""
        return finger_contact_count(self.scene, self.agent, actor, self.TIP_CONTACT_FORCE)

    def _screw_touched(self) -> torch.Tensor:
        return self._contact_count(self.screw_head) >= 1.0

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            set_work_table_pose(self)
            self._reset_agent(b)

            # Per-episode layout from the EPISODE RNG (reproducible from the
            # recorded seed under the teleop reset): the panel, with the rail
            # seated on it, is offset and yawed within the assembly zone.
            offsets = [
                uniform_offset(layout_rng(self, int(i)), (self.panel_random_x, self.panel_random_y),
                               self.panel_random_yaw_deg)
                for i in env_idx
            ]
            panel_p = torch.tensor(
                [[self.PANEL_CENTER[0] + dx, self.PANEL_CENTER[1] + dy, PANEL_HALF[2]] for dx, dy, _ in offsets],
                device=self.device, dtype=torch.float32,
            )
            panel_q = yaw_to_quat_wxyz(torch.tensor([yaw for _, _, yaw in offsets], device=self.device, dtype=torch.float32))
            panel_pose = Pose.create_from_pq(p=panel_p, q=panel_q)
            self.panel.set_pose(panel_pose)
            identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(b, -1)
            marker_p = torch.tensor([[self.PANEL_CENTER[0], self.PANEL_CENTER[1], 0.0]], device=self.device, dtype=torch.float32).expand(b, -1)
            self.place_marker.set_pose(Pose.create_from_pq(p=marker_p, q=identity))

            rail_local = torch.tensor(
                [[RAIL_SEATED_X_LOCAL, 0.0, RAIL_SEATED_Z_LOCAL]], device=self.device
            ).expand(b, -1)
            self.rail.set_pose(panel_pose * Pose.create_from_pq(p=rail_local, q=identity))
            zeros = torch.zeros((b, 1), device=self.device)
            self.rail.set_qpos(zeros)
            self.rail.set_qvel(zeros.clone())

            self.rotation_progress[env_idx] = 0.0
            self.success_counter[env_idx] = 0
            self.touched_once[env_idx] = False

    def _before_control_step(self):
        # The screw angle is whatever the fingers have physically turned it to;
        # this hook only keeps the process latch.
        self.rotation_progress = self.rail.get_qpos()[:, 0].clone()
        self.touched_once |= self._screw_touched()

    def evaluate(self):
        touched = self._screw_touched()
        angle = self.rail.get_qpos()[:, 0]
        reached_angle = angle >= self.TARGET_ROTATION - self.TARGET_TOLERANCE
        self.success_counter = torch.where(
            reached_angle, self.success_counter + 1, torch.zeros_like(self.success_counter)
        )
        # Shared with the long task's stage 3 (SUCCESS_HOLD_STEPS is 1, so the
        # counter above is bookkeeping only).
        success = screw_fastened(angle)
        tcp_to_screw = torch.linalg.norm(self.screw_head.pose.p - self.agent.tcp.pose.p, dim=1)
        return {
            "success": success,
            "termination_success": success if self.terminate_on_success else torch.zeros_like(success),
            "is_screw_touched": touched,
            "touched_once": self.touched_once,
            "tcp_to_screw_distance": tcp_to_screw,
            "screw_angle": angle,
            "rotation_progress": angle / self.TARGET_ROTATION,
            "screw_position": self.screw_head.pose.p,
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "screw_pose": self.screw_head.pose.raw_pose,
            "tcp_to_screw": info["screw_position"] - self.agent.tcp.pose.p,
        }
        if self.obs_mode_struct.use_state:
            obs.update(
                panel_pose=self.panel.pose.raw_pose,
                rail_pose=self.rail.pose.raw_pose,
                screw_angle=info["screw_angle"].unsqueeze(1),
                rotation_progress=info["rotation_progress"].unsqueeze(1),
                is_screw_touched=info["is_screw_touched"].float().unsqueeze(1),
                touched_once=info["touched_once"].float().unsqueeze(1),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        reach = 1.0 - torch.tanh(6.0 * info["tcp_to_screw_distance"])
        touch = info["is_screw_touched"].float()
        turn = info["rotation_progress"]
        reward = reach + touch + 5.0 * turn
        reward = torch.where(
            info["success"], torch.full_like(reward, self.SUCCESS_REWARD), reward
        )
        info["reward_terms"] = {"reach": reach, "touch": touch, "turn": turn}
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs, action, info) / self.SUCCESS_REWARD


def hold_action(env, control_mode: str):
    action = torch.zeros(
        env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32
    )
    if "delta" in control_mode:
        return action
    qpos = env.unwrapped.agent.robot.get_qpos()
    dof = min(action.shape[-1], qpos.shape[-1])
    action[..., :dof] = qpos[..., :dof].reshape(action[..., :dof].shape)
    if env.unwrapped.robot_uids == GRIPPER_ROBOT_UID:
        action[..., -1] = 1.0
    return action


def run(args):
    env = gym.make(
        args.env_id, num_envs=args.num_envs, obs_mode=args.obs_mode,
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        render_mode="rgb_array" if args.record_video else args.render_mode,
        sim_backend=args.sim_backend, terminate_on_success=args.terminate_on_success,
        visual_detail=not args.no_visual_detail,
        render_camera_eye=args.cam_eye,
        render_camera_target=args.cam_target,
        render_camera_fov=args.cam_fov,
    )
    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordEpisode(
            env, output_dir=args.video_dir, save_video=True, save_trajectory=False,
            trajectory_name="screw_fasten_v2", video_fps=args.video_fps,
            max_steps_per_video=args.max_steps,
        )
    obs, info = env.reset(seed=args.seed)
    keep_still = hold_action(env, args.control_mode)
    for step in range(args.max_steps):
        obs, reward, terminated, truncated, info = env.step(keep_still)
        if args.render_mode == "human" and not args.record_video:
            env.render()
        if args.debug_print and (step + 1) % 10 == 0:
            print(
                f"step={step + 1:04d} angle_deg="
                f"{float(torch.rad2deg(info['screw_angle']).mean()):.1f} "
                f"touched={float(info['is_screw_touched'].float().mean()):.2f} "
                f"success={float(info['success'].float().mean()):.2f}"
            )
        if bool((torch.as_tensor(terminated) | torch.as_tensor(truncated)).all().item()):
            break
    env.close()


def parse_args():
    p = argparse.ArgumentParser(description="Turn the furniture rail's wing screw by 90 degrees.")
    p.add_argument("--env-id", default=ENV_IDS[0], choices=ENV_IDS)
    p.add_argument("--control-mode", default=DEFAULT_CONTROL_MODE,
                   choices=["pd_joint_pos", "pd_ee_delta_pose"])
    p.add_argument("--obs-mode", default="state")
    p.add_argument("--reward-mode", default="normalized_dense",
                   choices=["dense", "normalized_dense", "sparse"])
    p.add_argument("--sim-backend", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--render-mode", default="rgb_array", choices=["rgb_array", "human"])
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--terminate-on-success", action="store_true")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--cam-eye", type=float, nargs=3, default=[0.75, -1.65, 0.92])
    p.add_argument("--cam-target", type=float, nargs=3, default=[-0.38, 0.02, 0.32])
    p.add_argument("--cam-fov", type=float, default=1.0)
    p.add_argument("--no-visual-detail", action="store_true")
    p.add_argument("--debug-print", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

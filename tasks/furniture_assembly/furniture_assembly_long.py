"""Long-horizon dual-arm furniture assembly task.

All workpieces are present from reset and remain the same physical entities for the
whole episode.  The required order and hand assignment are:

1. the LEFT hand removes the panel from the rack and places it in the marked region;
2. the RIGHT hand picks up the rail and inserts it into the panel groove;
3. the RIGHT hand turns the rail's wing screw by 90 degrees (no screwdriver,
   see furniture_screw_fasten).

Once a stage is completed, that workpiece pose is latched.  This prevents later contact
from destroying a completed subtask without hiding or replacing any asset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import sapien.physx as physx
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from tasks.config.hands import register_task
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SceneConfig, SimConfig
from mani_skill.utils.wrappers.record import RecordEpisode

# Import all four dual-arm agents so their UIDs are registered before gym.make.
from tasks.robots.panda_dual_gripper_mobile_base import PandaDualGripperMobileBase
from tasks.robots.panda_dual_sharpa_hand_mobile_base import (
    PandaDualSharpaHandAngledMobileBase,
)
from tasks.robots.panda_dual_wuji_hand_mobile_base import PandaDualWujiHandMobileBase

from tasks.furniture_assembly.furniture_assembly_lay_flat_panel import (
    GRASP_LOCAL_POS,
    PLACE_CENTER as SHARED_PLACE_CENTER,
    RACK_CENTER as SHARED_RACK_CENTER,
    RAIL_PICK_CENTER as SHARED_RAIL_PICK_CENTER,
    PANEL_HALF,
    PANEL_IN_SLOT_Z,
    FurnitureAssemblyLayFlatPanelEnv,
    ROBOT_ROOT_POSE,
    axis_to_quat_wxyz,
    build_drawer_panel,
    build_panel_rack,
    build_place_marker,
    build_dexarena_base_camera,
    quat_mul,
    yaw_to_quat_wxyz,
    finger_contact_count,
    panel_place_quantities,
)
from tasks.furniture_assembly.furniture_assembly_insert_rail import (
    RailSeatClip,
    RAIL_GRASP_LOCAL,
    RAIL_REST_Z,
    rail_seat_quantities,
)
from tasks.furniture_assembly.furniture_assembly_fasten_screw import (
    SCREW_TARGET_ROTATION,
    SCREW_TARGET_TOLERANCE,
    build_wing_screw_rail,
    screw_fastened,
)
from tasks.robots.panda_dual_xhand_hand_mobile_base import PandaDualXHandHandMobileBase  # noqa: F401
from tasks.config.layout_randomization import episode_rng, uniform_offset, layout_rng
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.scene_builders import TableSceneBuilder
from tasks.config.sim_profiles import make_sim_config
from tasks.config.workspace_rig import (
    ARM_INIT_QPOS,
    PLATFORM_ROOT_X,
    set_work_table_pose,
)
from tasks.config.energy_fence import set_task_object_energy_fence
from tasks.config.task_state import DeclarativeTaskStateMixin, TaskStateField

ROBOT_UID_GRIPPER = "panda_dual_gripper_mobile_base"
ROBOT_UID_SHARPA = "panda_dual_sharpa_hand_angled_mobile_base"
ROBOT_UID_WUJI = "panda_dual_wuji_hand_angled_mobile_base"
ROBOT_UID_XHAND = "panda_dual_xhand_hand_angled_mobile_base"
ROBOT_UIDS = (
    ROBOT_UID_GRIPPER,
    ROBOT_UID_SHARPA,
    ROBOT_UID_WUJI,
    ROBOT_UID_XHAND,
)

ENV_IDS = (
    "FurnitureAssemblyLong-Sharpa-v1",
    "FurnitureAssemblyLong-Wuji-v1",
    "FurnitureAssemblyLong-XHand-v1",
    "FurnitureAssemblyLong-Gripper-v1",
)
DEFAULT_CONTROL_MODE = "pd_joint_pos"
DEFAULT_VIDEO_DIR = "tasks/assets/task_object_meshes/furniture/videos_furniture"
MAX_EPISODE_STEPS = 2400 # was 1200; three dual-arm stages at juice's 1800/5-stage pace
# Task layer: explicit masses (kg). Panel/rail come from their builders'
# material densities; the rack used to fall through to the engine default
# density, which happened to land near its real mass -- now stated so it is a
# decision, not a leftover.
TASK_OBJECT_MASS = {
    "drawer_panel": 0.1983, # hollow-core board 380x190x18 mm at 160 kg/m3
    "slider_rail": 0.0967, # aluminium light-duty drawer slide; the wing screw adds ~30 g
    "panel_rack": 1.917,            # steel fixture (kinematic)
}

STAGE_PLACE_PANEL = 0
STAGE_INSERT_RAIL = 1
STAGE_FASTEN_SCREW = 2
STAGE_DONE = 3
STAGE_NAMES = ("place_panel_left", "insert_rail_right", "fasten_screw_right", "done") # rail stage moved to the RIGHT hand (user)


# Per-episode layout half-ranges for the long task (m, m, deg) -- the subtasks' defaults:
# furniture_panel_lay_flat rack_random_x/y 0.030/0.025 + rack_random_yaw 9 deg,
# furniture_rail_insert_groove rail_random_x/y 0.045/0.035 (no yaw).
# (stage 3 has no free tool any more, so there is no third draw.)
LONG_RACK_RANDOM = (0.030, 0.025, 9.0)
LONG_RAIL_RANDOM = (0.045, 0.035, 0.0)


@register_task(
    "FurnitureAssemblyLong",
    side="both",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    render_mode="rgb_array",
)
class FurnitureAssemblyLongEnv(DeclarativeTaskStateMixin, BaseEnv):
    # Task bookkeeping that physics cannot re-derive. Restoring a
    # mid-episode frame without these silently rewinds every milestone,
    # which breaks offline re-render, demo-init and end-frame QC.
    # APPEND ONLY -- reordering breaks older recordings.
    TASK_STATE_KEY = "furniture_task_state"
    TASK_STATE_FIELDS = (
        TaskStateField("stage", 1, torch.long),
        TaskStateField("stage_counter", 1, torch.long),
        TaskStateField("left_panel_grasped_once", 1, torch.bool),
        TaskStateField("right_rail_grasped_once", 1, torch.bool),
        TaskStateField("right_screw_touched_once", 1, torch.bool),
        TaskStateField("rotation_progress", 1, torch.float32),
        TaskStateField("panel_lock_pose", 7, torch.float32),
        TaskStateField("rail_lock_pose", 7, torch.float32),
    )
    """Three-stage bimanual task with strict order; each stage predicate is the
    corresponding subtask's success predicate (which hand did it is recorded, not required)."""

    SUPPORTED_ROBOTS = list(ROBOT_UIDS)
    agent: Any

    # +Y is the left-arm side; -Y is the right-arm side for these dual mobile bases.
    # Task layer: shared with the subtasks, single-sourced on the rig.
    RACK_CENTER = SHARED_RACK_CENTER
    PLACE_CENTER = SHARED_PLACE_CENTER
    RAIL_PICK_CENTER = SHARED_RAIL_PICK_CENTER
    PLACE_REGION_HALF_SIZE = (0.225, 0.135)

    STAGE_HOLD_STEPS = 5
    TIP_CONTACT_FORCE = 0.6
    # the three stage predicates are the subtask modules' shared
    # functions (panel_place_quantities / rail_seat_quantities / screw_fastened);
    # no thresholds of their own live here any more.
    TARGET_ROTATION = SCREW_TARGET_ROTATION
    TARGET_TOLERANCE = SCREW_TARGET_TOLERANCE
    SUCCESS_REWARD = 20.0

    _load_lighting = FurnitureAssemblyLayFlatPanelEnv._load_lighting

    def __init__(
        self,
        *args,
        robot_uids: str = ROBOT_UID_SHARPA,
        robot_init_qpos_noise: float = 0.01,
        rail_seat_clip: bool = True,
        terminate_on_success: bool = False,
        render_width: int = 1440,
        render_height: int = 1440,
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if num_envs != 1:
            raise ValueError("FurnitureAssemblyLongEnv currently supports num_envs=1")
        if robot_uids not in ROBOT_UIDS:
            raise ValueError(
                f"Unsupported robot {robot_uids!r}; expected one of {sorted(ROBOT_UIDS)}"
            )
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.rail_seat_clip = bool(rail_seat_clip)
        self.terminate_on_success = bool(terminate_on_success)
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        if reconfiguration_freq is None:
            reconfiguration_freq = 1
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        # Environment layer: the locked fleet profile, not a private config.
        # Only the GPU buffer sizes (pure capacity, no physics) are task-owned.
        return make_sim_config(
            max_rigid_contact_count=2**21,
            max_rigid_patch_count=2**20,
        )

    @property
    def _default_sensor_configs(self):
        return [
            build_dexarena_base_camera(name="base_camera"),
            CameraConfig(
                "external_camera",
                sapien_utils.look_at([0.43, -1.31, 0.85], [-0.41, -0.04, 0.25]),
                320, 240, np.deg2rad(60.0), 0.01, 100.0,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            sapien_utils.look_at([0.61, -1.76, 1.16], [-0.41, -0.03, 0.43]),
            self.render_width,
            self.render_height,
            np.deg2rad(65.0),
            0.01,
            100.0,
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

        # Every asset is built now and is visible from the first frame.
        self.rack = build_panel_rack(self.scene)
        self.place_marker = build_place_marker(self.scene, self.PLACE_REGION_HALF_SIZE)
        self.panel = build_drawer_panel(self.scene, name="long_panel")
        # The rail carries the wing screw (furniture_screw_fasten); the screw is
        # the rail's own revolute child.
        self.rail = build_wing_screw_rail(self.scene, name="long_rail")
        self.screw = self.rail
        self.rail_body = self.rail.links_map["rail_body"]
        self.screw_joint = self.rail.active_joints_map["screw_joint"]
        self.screw_head = self.rail.links_map["screw_head"]
        for free_body in (self.panel, self.rail):
            set_task_object_energy_fence(free_body)
        self.rail_clip = RailSeatClip(self.scene, self.rail, enabled=self.rail_seat_clip)

        self.grasp_local = common.to_tensor(GRASP_LOCAL_POS, device=self.device).float()
        self.rail_grasp_local = common.to_tensor(
            RAIL_GRASP_LOCAL, device=self.device
        ).float()

        self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stage_counter = torch.zeros_like(self.stage)
        self.left_panel_grasped_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.right_rail_grasped_once = torch.zeros_like(self.left_panel_grasped_once)
        self.right_screw_touched_once = torch.zeros_like(self.left_panel_grasped_once)
        self.rotation_progress = torch.zeros(self.num_envs, device=self.device)
        self.panel_lock_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.rail_lock_pose = torch.zeros_like(self.panel_lock_pose)

    def _reset_agent(self, env_idx: torch.Tensor):
        batch_size = len(env_idx)
        if hasattr(self.agent, "get_reset_pos"):
            initial = self.agent.get_reset_pos(device=self.device)
        else:
            initial = torch.as_tensor(
                self.agent.keyframes["rest"].qpos,
                dtype=torch.float32,
                device=self.device,
            )
        qpos = initial.expand(batch_size, -1).clone()
        active_names = [joint.name for joint in self.agent.robot.get_active_joints()]
        arm_indices = [active_names.index(name) for name in self.agent.arm_joint_names]
        # Arm layer: the shared start pose, stamped on both arms (7 joints each).
        shared = torch.tensor(ARM_INIT_QPOS, dtype=torch.float32, device=self.device)
        for start in range(0, len(arm_indices), len(ARM_INIT_QPOS)):
            block = arm_indices[start:start + len(ARM_INIT_QPOS)]
            if len(block) == len(ARM_INIT_QPOS):
                qpos[:, torch.tensor(block, device=self.device)] = shared
        if self.robot_init_qpos_noise > 0:
            qpos[:, arm_indices] += common.to_tensor(
                self._episode_rng.normal(
                    0.0, self.robot_init_qpos_noise, (batch_size, len(arm_indices))
                ),
                device=self.device,
            )
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(ROBOT_ROOT_POSE)

    @staticmethod
    def _pose_from_raw(raw_pose: torch.Tensor) -> Pose:
        return Pose.create_from_pq(p=raw_pose[:, :3], q=raw_pose[:, 3:])

    def _side_links(self, side: str):
        links = list(getattr(self.agent, f"{side}_tip_links", ()))
        palm = getattr(self.agent, f"{side}_palm_link", None)
        if palm is not None:
            links.append(palm)
        return links

    def _side_contact_count(self, side: str, actor) -> torch.Tensor:
        """Distinct fingers of `side` in force contact with `actor` (collision-bearing links)."""
        return finger_contact_count(self.scene, self.agent, actor, self.TIP_CONTACT_FORCE, side=side)

    def _is_side_grasped(self, side: str, actor) -> torch.Tensor:
        grasped = self._side_contact_count(side, actor) >= 2.0
        native = getattr(self.agent, f"is_{side}_grasping", None)
        if callable(native):
            try:
                grasped |= native(actor).to(torch.bool)
            except Exception:
                pass
        return grasped

    def _all_contact_count(self, actor) -> torch.Tensor:
        return self._side_contact_count("left", actor) + self._side_contact_count(
            "right", actor
        )

    def _landmarks(self, actor, local_points: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                (
                    actor.pose
                    * Pose.create_from_pq(
                        p=local_points[index].expand(self.num_envs, -1)
                    )
                ).p
                for index in range(local_points.shape[0])
            ],
            dim=1,
        )

    def _panel_ready(self) -> torch.Tensor:
        """Stage 1 predicate = the PanelLayFlat subtask's, on the fixed PLACE_CENTER."""
        center = torch.tensor(self.PLACE_CENTER, device=self.device, dtype=torch.float32)
        center = center.view(1, 2).expand(self.num_envs, -1)
        q = panel_place_quantities(
            self.panel.pose, center, self._all_contact_count(self.panel),
            self.panel.linear_velocity, self.panel.angular_velocity,
            region_half_size=self.PLACE_REGION_HALF_SIZE)
        return q["placed_now"]

    def _rail_ready(self) -> torch.Tensor:
        """Stage 2 predicate = the RailInsert subtask's.

        the release term used to be measured against `self.rail`, an
        Articulation -- `get_pairwise_contact_forces` raises on it, the helper
        swallowed the exception, and "released" was True whatever the hand did.
        The rail body LINK is what the fingers touch."""
        q = rail_seat_quantities(
            self.rail.pose, self.panel.pose, self._all_contact_count(self.rail_body),
            self.rail.root_linear_velocity, self.rail.root_angular_velocity)
        return q["seated_now"]

    def _right_screw_touched(self) -> torch.Tensor:
        """Telemetry: a RIGHT-hand finger in force contact with the wing screw head."""
        return self._side_contact_count("right", self.screw_head) >= 1.0

    # ------------------------------------------------------------------
    # Completed-part hold.
    #
    # WHY THIS IS NOT A PER-SUBSTEP TELEPORT ANY MORE.
    #
    # `_lock_completed_parts` used to re-write pose + linear + angular velocity
    # of the finished panel and rail from `_before_simulation_step`, i.e. on
    # every PhysX substep (20 per control step at 400 Hz sim / 20 Hz control).
    # Each write re-created whatever penetration the solver had just resolved,
    # so a hand resting on a completed part had its depenetration impulse
    # re-applied every substep with nowhere for the energy to go except the
    # robot -- measured elsewhere in this repo at 31.8 rad/s peak robot qvel
    # against a 2.2 rad/s actuator limit.
    #
    # A kinematic body holds its pose exactly and generates no constraint to
    # resolve. The pose is written ONCE, on the stage transition that locks it.
    # ------------------------------------------------------------------
    def _hold_bodies(self):
        """(actor, per-env hold mask, locked raw pose) for each completed part."""
        # Only the panel. The rail is an articulation now (its screw is a child
        # link) and PhysX cannot flag a link kinematic; seated, it is form-closed
        # in the groove by its two pins and clipped in by RailSeatClip.
        return (
            (self.panel, self.stage >= STAGE_INSERT_RAIL, self.panel_lock_pose),
        )

    def _refresh_hold_state(self):
        """Reconcile the kinematic hold with the stage flags. No-op when unchanged."""
        if not hasattr(self, "stage"):
            return
        if self.gpu_sim_enabled:
            wants_hold = any(
                bool(mask.any()) for _, mask, _ in self._hold_bodies()
            )
            if not wants_hold:
                return
            raise RuntimeError(
                "Furniture holds completed parts with the PhysX kinematic flag, "
                "which GPU PhysX does not support changing after scene load. Run "
                "the task on CPU (num_envs=1). The previous per-substep teleport "
                "did work on GPU but re-applied a depenetration impulse on every "
                "substep a hand touched a locked part."
            )
        for actor, mask, lock_pose in self._hold_bodies():
            if actor is None:
                continue
            for i, raw in enumerate(actor._objs):
                component = raw.find_component_by_type(
                    physx.PhysxRigidDynamicComponent
                )
                if component is None:
                    continue
                want = bool(mask[i])
                if all(bool(x) for x in component.get_locked_motion_axes()) == want:
                    continue
                if want:
                    row = lock_pose[i].detach().cpu().numpy()
                    component.set_pose(sapien.Pose(p=row[:3], q=row[3:]))
                    component.set_linear_velocity([0.0, 0.0, 0.0])
                    component.set_angular_velocity([0.0, 0.0, 0.0])
                    # Lock the six DoF rather than setting `kinematic`. Both hold
                    # the body exactly, but flipping `kinematic` ZEROES whatever
                    # velocity the body carries at that instant, so a state
                    # restore that lands on a held body lost its recorded
                    # velocity (measured: `replay_recording --mode state` failed
                    # on recorded Screwing episodes). Locked axes keep the body
                    # dynamic, so its state still round-trips.
                component.set_locked_motion_axes([want] * 6)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            set_work_table_pose(self)
            # Per-episode layout, the FIRST episode-RNG draw, with the SAME half-ranges
            # the subtasks use by default (LONG_*_RANDOM). Draw order: rack, rail.
            # The panel starts in the rack slot and rides with the rack.
            rng = layout_rng(self, int(env_idx[0]))
            rdx, rdy, ryaw = uniform_offset(rng, LONG_RACK_RANDOM[:2], LONG_RACK_RANDOM[2])
            ldx, ldy, lyaw = uniform_offset(rng, LONG_RAIL_RANDOM[:2], LONG_RAIL_RANDOM[2])
            self._reset_agent(env_idx)
            batch_size = len(env_idx)
            identity = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=self.device
            ).expand(batch_size, -1)

            rack_yaw = torch.full((batch_size,), ryaw, device=self.device)
            rack_position = torch.tensor(
                [[self.RACK_CENTER[0] + rdx, self.RACK_CENTER[1] + rdy, 0.0]], device=self.device
            ).expand(batch_size, -1)
            self.rack.set_pose(Pose.create_from_pq(p=rack_position, q=yaw_to_quat_wxyz(rack_yaw)))

            panel_position = torch.tensor(
                [[self.RACK_CENTER[0] + rdx, self.RACK_CENTER[1] + rdy, PANEL_IN_SLOT_Z]],
                device=self.device,
            ).expand(batch_size, -1)
            upright = torch.full((batch_size,), np.pi / 2.0, device=self.device)
            panel_quaternion = quat_mul(
                yaw_to_quat_wxyz(rack_yaw),
                axis_to_quat_wxyz(upright, axis=0),
            )
            self.panel.set_pose(
                Pose.create_from_pq(p=panel_position, q=panel_quaternion)
            )
            self.panel.set_linear_velocity(torch.zeros((batch_size, 3), device=self.device))
            self.panel.set_angular_velocity(torch.zeros((batch_size, 3), device=self.device))

            marker_position = torch.tensor(
                [[*self.PLACE_CENTER, 0.0]], device=self.device
            ).expand(batch_size, -1)
            self.place_marker.set_pose(
                Pose.create_from_pq(p=marker_position, q=identity)
            )

            rail_position = torch.tensor(
                [[self.RAIL_PICK_CENTER[0] + ldx, self.RAIL_PICK_CENTER[1] + ldy, RAIL_REST_Z]], device=self.device
            ).expand(batch_size, -1)
            # Lay the rail parallel to the panel's long edge for an unambiguous pickup.
            rail_yaw = torch.full((batch_size,), lyaw, device=self.device)
            self.rail.set_pose(
                Pose.create_from_pq(p=rail_position, q=yaw_to_quat_wxyz(rail_yaw))
            )
            self.rail.set_root_linear_velocity(torch.zeros((batch_size, 3), device=self.device))
            self.rail.set_root_angular_velocity(torch.zeros((batch_size, 3), device=self.device))
            self.rail.set_qpos(torch.zeros((batch_size, 1), device=self.device))
            self.rail.set_qvel(torch.zeros((batch_size, 1), device=self.device))

            self.stage[env_idx] = STAGE_PLACE_PANEL
            self.stage_counter[env_idx] = 0
            self.left_panel_grasped_once[env_idx] = False
            self.right_rail_grasped_once[env_idx] = False
            self.right_screw_touched_once[env_idx] = False
            self.rotation_progress[env_idx] = 0.0
            self.rail_clip.release(env_idx.tolist() if hasattr(env_idx, "tolist") else env_idx)
            self.panel_lock_pose[env_idx] = self.panel.pose.raw_pose[env_idx]
            self.rail_lock_pose[env_idx] = self.rail.pose.raw_pose[env_idx]

    def _before_control_step(self):
        # Reconcile the kinematic hold (writes a pose only on a stage flip);
        # `_sync_screw_to_rail` is an animation, the screw being a puppet driven
        # by the measured tool roll rather than by contact. Neither runs per
        # PhysX substep any more -- see _refresh_hold_state.
        self._refresh_hold_state()

        # Which hand did it is recorded (HUD, obs, reward shaping) but no longer gates
        # a stage: each stage predicate is exactly its subtask's success
        # predicate, which has no such term on the single-arm rig.
        left_panel_grasped = self._is_side_grasped("left", self.panel)
        self.left_panel_grasped_once |= (
            (self.stage == STAGE_PLACE_PANEL) & left_panel_grasped
        )
        panel_now = (self.stage == STAGE_PLACE_PANEL) & self._panel_ready()
        self.stage_counter = torch.where(
            panel_now,
            self.stage_counter + 1,
            torch.where(
                self.stage == STAGE_PLACE_PANEL,
                torch.zeros_like(self.stage_counter),
                self.stage_counter,
            ),
        )
        panel_done = (
            (self.stage == STAGE_PLACE_PANEL)
            & (self.stage_counter >= self.STAGE_HOLD_STEPS)
        )
        if torch.any(panel_done):
            self.panel_lock_pose[panel_done] = self.panel.pose.raw_pose[panel_done]
            self.stage[panel_done] = STAGE_INSERT_RAIL
            self.stage_counter[panel_done] = 0
            self._refresh_hold_state()

        right_rail_grasped = self._is_side_grasped("right", self.rail_body)
        self.right_rail_grasped_once |= (
            (self.stage == STAGE_INSERT_RAIL) & right_rail_grasped
        )
        rail_now = (self.stage == STAGE_INSERT_RAIL) & self._rail_ready()
        self.stage_counter = torch.where(
            rail_now,
            self.stage_counter + 1,
            torch.where(
                self.stage == STAGE_INSERT_RAIL,
                torch.zeros_like(self.stage_counter),
                self.stage_counter,
            ),
        )
        rail_done = (
            (self.stage == STAGE_INSERT_RAIL)
            & (self.stage_counter >= self.STAGE_HOLD_STEPS)
        )
        if torch.any(rail_done):
            self.rail_lock_pose[rail_done] = self.rail.pose.raw_pose[rail_done]
            self.stage[rail_done] = STAGE_FASTEN_SCREW
            self.stage_counter[rail_done] = 0
            self._refresh_hold_state()
            # Snap-fit: the seated rail is clipped into its dado from here on.
            self.rail_clip.engage(self.rail.pose, torch.nonzero(rail_done).flatten().tolist())

        # The screw angle is PHYSICAL -- the fingers turn the rail's revolute
        # wing screw. No animation, no written qpos.
        self.right_screw_touched_once |= (
            (self.stage == STAGE_FASTEN_SCREW) & self._right_screw_touched()
        )
        self.rotation_progress = self.rail.get_qpos()[:, 0].clone()
        # The shared stage predicate alone, as in the ScrewFasten subtask.
        screw_done = (self.stage == STAGE_FASTEN_SCREW) & screw_fastened(self.rotation_progress)
        self.stage[screw_done] = STAGE_DONE

    def evaluate(self):
        right_touching = self._right_screw_touched()
        success = self.stage == STAGE_DONE
        return {
            "success": success,
            "termination_success": (
                success if self.terminate_on_success else torch.zeros_like(success)
            ),
            "stage": self.stage,
            "stage_one_hot": torch.nn.functional.one_hot(
                self.stage, num_classes=4
            ).float(),
            "panel_ready": self._panel_ready(),
            "rail_ready": self._rail_ready(),
            "left_panel_grasped_once": self.left_panel_grasped_once,
            "right_rail_grasped_once": self.right_rail_grasped_once,
            "right_screw_touched_once": self.right_screw_touched_once,
            "right_screw_touched": right_touching,
            "screw_angle": self.rail.get_qpos()[:, 0],
            "rotation_progress": self.rotation_progress / self.TARGET_ROTATION,
        }

    def _get_obs_extra(self, info: dict):
        left_tcp_pose = self.agent.left_tcp_pose
        right_tcp_pose = self.agent.right_tcp_pose
        panel_grasp = (
            self.panel.pose
            * Pose.create_from_pq(p=self.grasp_local.expand(self.num_envs, -1))
        ).p
        rail_grasp = (
            self.rail.pose
            * Pose.create_from_pq(p=self.rail_grasp_local.expand(self.num_envs, -1))
        ).p
        obs = {
            "left_tcp_pose": left_tcp_pose.raw_pose,
            "right_tcp_pose": right_tcp_pose.raw_pose,
            "stage_one_hot": info["stage_one_hot"],
            "panel_pose": self.panel.pose.raw_pose,
            "rail_pose": self.rail.pose.raw_pose,
            "screw_pose": self.screw_head.pose.raw_pose,
            "left_tcp_to_panel": panel_grasp - left_tcp_pose.p,
            "left_tcp_to_rail": rail_grasp - left_tcp_pose.p,
            "right_tcp_to_screw": self.screw_head.pose.p - right_tcp_pose.p,
        }
        if self.obs_mode_struct.use_state:
            obs.update(
                rack_pose=self.rack.pose.raw_pose,
                screw_angle=info["screw_angle"].unsqueeze(1),
                rotation_progress=info["rotation_progress"].unsqueeze(1),
                left_panel_grasped_once=info["left_panel_grasped_once"].float().unsqueeze(1),
                right_rail_grasped_once=info["right_rail_grasped_once"].float().unsqueeze(1),
                right_screw_touched_once=info["right_screw_touched_once"].float().unsqueeze(1),
                right_screw_touched=info["right_screw_touched"].float().unsqueeze(1),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        left_tcp = self.agent.left_tcp_pose.p
        right_tcp = self.agent.right_tcp_pose.p
        panel_grasp = (
            self.panel.pose
            * Pose.create_from_pq(p=self.grasp_local.expand(self.num_envs, -1))
        ).p
        rail_grasp = (
            self.rail.pose
            * Pose.create_from_pq(p=self.rail_grasp_local.expand(self.num_envs, -1))
        ).p
        panel_reach = 1.0 - torch.tanh(
            6.0 * torch.linalg.norm(left_tcp - panel_grasp, dim=1)
        )
        rail_reach = 1.0 - torch.tanh(
            6.0 * torch.linalg.norm(left_tcp - rail_grasp, dim=1)
        )
        screw_reach = 1.0 - torch.tanh(
            6.0 * torch.linalg.norm(right_tcp - self.screw_head.pose.p, dim=1)
        )
        reward = torch.where(
            self.stage == STAGE_PLACE_PANEL,
            panel_reach
            + self.left_panel_grasped_once.float()
            + 2.0 * info["panel_ready"].float(),
            torch.where(
                self.stage == STAGE_INSERT_RAIL,
                4.0
                + rail_reach
                + self.right_rail_grasped_once.float()
                + 3.0 * info["rail_ready"].float(),
                torch.where(
                    self.stage == STAGE_FASTEN_SCREW,
                    9.0
                    + screw_reach
                    + info["right_screw_touched_once"].float()
                    + 5.0 * info["rotation_progress"],
                    torch.full_like(panel_reach, self.SUCCESS_REWARD),
                ),
            ),
        )
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
    return action


def run(args):
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array" if args.record_video else args.render_mode,
        sim_backend=args.sim_backend,
        terminate_on_success=args.terminate_on_success,
    )
    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=args.video_dir,
            save_video=True,
            save_trajectory=False,
            trajectory_name="dual_furniture_assembly_long_v1",
            video_fps=20,
            max_steps_per_video=args.max_steps,
        )
    env.reset(seed=args.seed)
    action = hold_action(env, args.control_mode)
    for _ in range(args.max_steps):
        _, _, terminated, truncated, _ = env.step(action)
        if bool((torch.as_tensor(terminated) | torch.as_tensor(truncated)).all()):
            break
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual-arm, three-stage furniture assembly task."
    )
    parser.add_argument("--env-id", default=ENV_IDS[0], choices=ENV_IDS)
    parser.add_argument(
        "--control-mode",
        default=DEFAULT_CONTROL_MODE,
        choices=["pd_joint_pos", "pd_joint_delta_pos"],
    )
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument(
        "--reward-mode",
        default="normalized_dense",
        choices=["dense", "normalized_dense", "sparse"],
    )
    parser.add_argument("--sim-backend", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--render-mode", default="rgb_array", choices=["rgb_array", "human"])
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terminate-on-success", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

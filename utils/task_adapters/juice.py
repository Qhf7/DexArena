"""Juice family (fridge, blender lid, blender button, apple) under num_envs > 1.

The tasks were written for one env: one cached per-episode layout, reset
writes sized to ``num_envs``, and ``__init__`` guards. With ``num_envs == 1``
every path below defers to the task, so recorded episodes replay exactly as
before; with more envs each env gets its own layout, every reset write is
sized to the reset sub-batch, and quantities derived from the blender are read
off each env's live blender pose.
"""

from __future__ import annotations

import numpy as np
import torch

from mani_skill.utils import common
from mani_skill.utils.building.articulation_builder import ArticulationBuilder
from mani_skill.utils.structs.pose import Pose

from ._base import GpuLoadWritesMixin, MultiEnvInitMixin
from ._shims import full_reset_mask, patched


def _all_scenes(builder, scene_idxs=None):
    """``set_scene_idxs`` replacement: build the object in every sub-scene."""
    return builder


def blender_yaw(blender_q: torch.Tensor, nominal_quat) -> torch.Tensor:
    """Yaw of each env's live blender relative to the nominal geometry."""
    q_nom = torch.tensor(np.asarray(nominal_quat, dtype=np.float32), device=blender_q.device)
    w1, x1, y1, z1 = blender_q.unbind(-1)
    w2, x2, y2, z2 = q_nom[0], -q_nom[1], -q_nom[2], -q_nom[3]
    rw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    rz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return 2.0 * torch.atan2(rz, rw)


def live_blender_point(blender_p, blender_q, nominal, point_name: str):
    """``nominal.<point_name>`` carried along with each env's live blender pose:
    root + Rz(yaw) * (point - root), z untouched (the relation
    ``shifted_blender_geometry`` encodes)."""
    yaw = blender_yaw(blender_q, nominal.blender_quat)
    point = np.asarray(getattr(nominal, point_name), dtype=np.float64)
    offset = torch.tensor(
        point - np.asarray(nominal.blender_root, dtype=np.float64),
        dtype=torch.float32,
        device=blender_p.device,
    )
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        [
            blender_p[:, 0] + cos * offset[0] - sin * offset[1],
            blender_p[:, 1] + sin * offset[0] + cos * offset[1],
            torch.full_like(yaw, float(point[2])),
        ],
        dim=-1,
    )


# ---------------------------------------------------------------- fridge


class OpenFridgeMixin(MultiEnvInitMixin):
    def _load_scene(self, options: dict):
        if self.num_envs == 1:
            return super()._load_scene(options)
        import sapien
        from types import SimpleNamespace

        from tasks.juice_preparation.juice_preparation_open_fridge import (
            CUSTOM_DOOR_JOINT_NAME,
            CUSTOM_DOOR_LINK_NAME,
            CUSTOM_FRIDGE_ROOT_POS,
            CUSTOM_HANDLE_LINK_NAME,
            FRIDGE_DOOR_HINGE_FRICTION,
            SUCCESS_OPEN_ANGLE_DEG,
            TARGET_OPEN_ANGLE_DEG,
            TableSceneBuilder,
            configure_task_object_physics,
            door_success_qpos,
            set_work_table_pose,
        )

        # `JuicePreparationOpenFridgeEnv._load_scene` with the fridge built in every
        # sub-scene and the joint limits read from one of them.
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        set_work_table_pose(self)

        loader = self.scene.create_urdf_loader()
        loader.name = "custom_tabletop_mini_fridge"
        loader.fix_root_link = True
        loader.scale = 1.0
        loader.load_multiple_collisions_from_file = True
        parsed = loader.parse(str(self.fridge_urdf_path))
        builders = parsed["articulation_builders"]
        if len(builders) != 1:
            raise RuntimeError(f"Expected one refrigerator articulation, got {len(builders)}")
        builder = builders[0]
        builder.disable_self_collisions = True
        builder.initial_pose = sapien.Pose(p=CUSTOM_FRIDGE_ROOT_POS.tolist())
        self.fridge = builder.build(name="custom_tabletop_mini_fridge")
        configure_task_object_physics(self.fridge, "hard_plastic")
        self.fridge_geometry = SimpleNamespace(
            root_position=CUSTOM_FRIDGE_ROOT_POS.copy(),
            root_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            hinge_root=np.array([-0.173, 0.180, 0.0]),
            hinge_axis_root=np.array([0.0, 0.0, -1.0]),
            handle_root_closed=np.array([-0.258, -0.115, 0.285]),
            front_outward_root=np.array([-1.0, 0.0, 0.0]),
        )
        self.door_joint = self.fridge.active_joints_map[CUSTOM_DOOR_JOINT_NAME]
        self.door_link = self.fridge.links_map[CUSTOM_DOOR_LINK_NAME]
        self.handle_link = self.fridge.links_map[CUSTOM_HANDLE_LINK_NAME]
        self.door_joint.set_drive_properties(stiffness=0.0, damping=0.85, force_limit=85.0, mode="force")
        if hasattr(self.door_joint, "set_friction"):
            self.door_joint.set_friction(FRIDGE_DOOR_HINGE_FRICTION)

        qlimits = self.fridge.get_qlimits().detach().cpu().numpy()[0].reshape(-1, 2)
        if qlimits.shape[0] != 1:
            raise RuntimeError(f"Expected one refrigerator joint, got limits={qlimits}")
        self.door_lower = float(qlimits[0, 0])
        self.door_upper = float(qlimits[0, 1])
        self.target_open_qpos = door_success_qpos(self.door_lower, self.door_upper, TARGET_OPEN_ANGLE_DEG)
        self.success_open_qpos = door_success_qpos(self.door_lower, self.door_upper, SUCCESS_OPEN_ANGLE_DEG)
        print(f"[fridge asset] urdf={self.fridge_urdf_path} root_p={CUSTOM_FRIDGE_ROOT_POS.tolist()}")

    def _after_reconfigure(self, options: dict):
        super()._after_reconfigure(options)
        if self.num_envs > 1:
            # Allocated once: a partial reset must clear only its own envs' latches.
            self.hand_contact_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def _set_door_state(self, env_idx: torch.Tensor, door_qpos: float):
        b = len(env_idx)
        dof = self.fridge.get_qpos().shape[-1]
        qpos = torch.full((b, dof), float(door_qpos), dtype=torch.float32, device=self.device)
        self.fridge.set_qpos(qpos)
        self.fridge.set_qvel(torch.zeros_like(qpos))

    def _reset_agent_rows(self, env_idx: torch.Tensor):
        from tasks.juice_preparation.juice_preparation_open_fridge import apply_arm_init_qpos, arm_qpos_noise, episode_rng

        if hasattr(self.agent, "get_reset_pos"):
            initial = self.agent.get_reset_pos(device=self.device)
        else:
            initial = torch.as_tensor(self.agent.keyframes["rest"].qpos, dtype=torch.float32, device=self.device)
        qpos = apply_arm_init_qpos(self.agent, initial.expand(len(env_idx), -1).clone())
        for k, env_i in enumerate(env_idx.tolist()):
            # Each env draws from ITS OWN episode RNG.
            index, noise = arm_qpos_noise(
                episode_rng(self, int(env_i)), self.agent, 1, self.robot_init_qpos_noise, device=self.device
            )
            qpos[k, index] += noise[0]
        self.agent.reset(qpos)
        self._set_robot_root_pose()

    def _initialize_episode_batched(self, env_idx: torch.Tensor):
        from tasks.juice_preparation.juice_preparation_open_fridge import CUSTOM_FRIDGE_ROOT_POS, layout_rng, sample_juice_layout, set_work_table_pose

        self.table_scene.initialize(env_idx)
        set_work_table_pose(self)
        b = len(env_idx)
        offsets = np.zeros((b, 3), dtype=np.float64)
        for k, env_i in enumerate(env_idx.tolist()):
            offsets[k, :2] = sample_juice_layout(layout_rng(self, int(env_i)))["fridge"]
        self._reset_agent_rows(env_idx)
        root_p = torch.tensor(
            np.asarray(CUSTOM_FRIDGE_ROOT_POS, dtype=np.float64)[None] + offsets,
            dtype=torch.float32,
            device=self.device,
        )
        root_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand(b, 4)
        self.fridge.set_pose(Pose.create_from_pq(p=root_p, q=root_q))
        self._set_door_state(env_idx, self.door_lower)
        self.success_settle_counter[env_idx] = 0
        self.success_latched[env_idx] = False
        self.hand_contact_latched[env_idx] = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.num_envs == 1:
            return super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            self._initialize_episode_batched(env_idx)


class CloseFridgeMixin(OpenFridgeMixin):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.num_envs == 1:
            return super()._initialize_episode(env_idx, options)
        from tasks.juice_preparation.juice_preparation_close_fridge import INITIAL_OPEN_ANGLE_DEG, SUCCESS_CLOSED_ANGLE_DEG, door_success_qpos

        with torch.device(self.device):
            self._initialize_episode_batched(env_idx)
            initial_qpos = float(
                np.clip(self.door_lower + np.deg2rad(INITIAL_OPEN_ANGLE_DEG), self.door_lower, self.door_upper)
            )
            self._set_door_state(env_idx, initial_qpos)
            self.initial_open_qpos = initial_qpos
            self.success_closed_qpos = door_success_qpos(self.door_lower, self.door_upper, SUCCESS_CLOSED_ANGLE_DEG)
            self.hand_contact_latched[env_idx] = False


# ------------------------------------------------------------ blender lid


def _lid_seat_quantities_live(lid_pose, seat_target: torch.Tensor):
    """``lid_seat_quantities`` with one seat per env."""
    from tasks.juice_preparation.juice_preparation_place_lid import LID_TILT_TOL, LID_XY_TOL, LID_Z_TOL

    position = lid_pose.p
    to_seat = seat_target.to(position.dtype).reshape(-1, 3) - position
    xy_error = torch.linalg.norm(to_seat[:, :2], dim=1)
    z_error = torch.abs(to_seat[:, 2])
    rotation = lid_pose.to_transformation_matrix()[..., :3, :3]
    tilt_error = torch.acos(torch.clamp(rotation[..., 2, 2], -1.0, 1.0))
    seated = (xy_error <= LID_XY_TOL) & (z_error <= LID_Z_TOL) & (tilt_error <= LID_TILT_TOL)
    stable = torch.ones_like(seated)
    return position, to_seat, xy_error, z_error, tilt_error, seated, stable


class PlaceBlenderLidMixin:
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.num_envs == 1:
            return super()._initialize_episode(env_idx, options)
        from tasks.juice_preparation.juice_preparation_place_lid import (
            layout_rng,
            reset_robot_agent,
            sample_juice_layout,
            set_work_table_pose,
            shifted_blender_geometry,
        )

        with torch.device(self.device):
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)
            set_work_table_pose(self)
            layouts = [
                sample_juice_layout(
                    layout_rng(self, int(i)),
                    lid_half_xy=(self.lid_x_noise, self.lid_y_noise),
                    lid_yaw_half_deg=float(np.rad2deg(self.lid_yaw_noise)),
                )
                for i in env_idx.tolist()
            ]
            geometries = [shifted_blender_geometry(self._geometry_nominal, *lay["blender"]) for lay in layouts]
            self.geometry = geometries[0]
            reset_robot_agent(
                agent=self.agent,
                robot_uids=self.robot_uids,
                batch_size=batch_size,
                device=self.device,
                robot_init_qpos_noise=self.robot_init_qpos_noise,
                episode_rng=self._episode_rng,
            )
            t = lambda rows: torch.tensor(np.asarray(rows, dtype=np.float32), device=self.device)  # noqa: E731
            if hasattr(self.blender, "set_pose"):
                self.blender.set_pose(
                    Pose.create_from_pq(t([g.blender_root for g in geometries]), t([g.blender_quat for g in geometries]))
                )
            if self.blender_is_articulation:
                blender_zero = torch.zeros((batch_size, self.blender.get_qpos().shape[-1]), device=self.device)
                self.blender.set_qpos(blender_zero)
                self.blender.set_qvel(blender_zero.clone())

            lid_p = t([g.lid_table_position for g in geometries]).clone()
            yaw = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            for k, lay in enumerate(layouts):
                ldx, ldy, lyaw = lay["lid"]
                lid_p[k, 0] += ldx
                lid_p[k, 1] += ldy
                yaw[k] = lyaw
            lid_q = torch.zeros(batch_size, 4, dtype=torch.float32, device=self.device)
            lid_q[:, 0] = torch.cos(0.5 * yaw)
            lid_q[:, 3] = torch.sin(0.5 * yaw)
            zero_velocity = torch.zeros(batch_size, 3, dtype=torch.float32, device=self.device)
            self.lid.set_linear_velocity(zero_velocity)
            self.lid.set_angular_velocity(zero_velocity)
            self.lid.set_pose(Pose.create_from_pq(lid_p, lid_q))
            self.lid.set_linear_velocity(zero_velocity)
            self.lid.set_angular_velocity(zero_velocity)

            self.picked_once[env_idx] = False
            self.transported_once[env_idx] = False
            self.seated_once[env_idx] = False
            self.released_once[env_idx] = False
            self.success_latched[env_idx] = False
            self.success_counter[env_idx] = 0

    def _task_quantities(self):
        if self.num_envs == 1:
            return super()._task_quantities()
        import tasks.juice_preparation.juice_preparation_place_lid as module

        seat = live_blender_point(self.blender.pose.p, self.blender.pose.q, self._geometry_nominal, "seat_position")

        def live(lid_pose, seat_position, *, device):
            return _lid_seat_quantities_live(lid_pose, seat)

        with patched(module, "lid_seat_quantities", live):
            return super()._task_quantities()


# --------------------------------------------------------- blender button


class PressBlenderButtonMixin(GpuLoadWritesMixin):
    def _refresh_gpu_kinematics(self) -> None:
        """Push pending joint writes to the GPU sim and fetch the link poses;
        under physx_cuda ``set_qpos`` alone leaves the poses unchanged."""
        px = self.scene.px
        px.gpu_apply_articulation_qpos()
        px.gpu_apply_articulation_qvel()
        px.gpu_update_articulation_kinematics()
        self.scene._gpu_fetch_all()

    def _calibrate_button_geometry(self) -> None:
        if not self.gpu_sim_enabled:
            return super()._calibrate_button_geometry()
        orig = self.blender.set_qpos

        def set_qpos(value):
            orig(value)
            self._refresh_gpu_kinematics()

        with patched(self.blender, "set_qpos", set_qpos):
            super()._calibrate_button_geometry()

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate (N,3) or (3,) vectors ``v`` by wxyz quaternions ``q`` (N,4)."""
        w, xyz = q[:, :1], q[:, 1:]
        v = v.expand(q.shape[0], 3) if v.ndim == 1 else v
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v + w * t + torch.cross(xyz, t, dim=-1)

    def _write_button_world_frame(self, rows: torch.Tensor):
        """World-frame button quantities of envs ``rows`` from their live blender pose."""
        p = self.blender.pose.p[rows]
        q = self.blender.pose.q[rows]
        self.button_press_direction[rows] = self._quat_rotate(q, self._button_dir_local)
        self.button_released_surface_position[rows] = p + self._quat_rotate(q, self._button_surface_local)
        self.button_released_link_position[rows] = p + self._quat_rotate(q, self._button_link_local)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.num_envs == 1:
            return super()._initialize_episode(env_idx, options)
        from tasks.juice_preparation.juice_preparation_press_button import (
            ROBOT_ROOT_POSE,
            layout_rng,
            reset_robot_agent,
            sample_juice_layout,
            set_work_table_pose,
            shifted_blender_geometry,
        )

        # The task re-measures the button on every reset by driving every env's
        # button joint, which would disturb mid-episode envs. Measure once, on
        # the first (full) reset, in the blender's own frame; later resets
        # carry those numbers with each env's blender pose.
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            set_work_table_pose(self)
            dev = self.device
            ids = [int(i) for i in env_idx.tolist()]
            b = len(ids)
            geometries = [
                shifted_blender_geometry(self._geometry_nominal, *sample_juice_layout(layout_rng(self, i))["blender"])
                for i in ids
            ]
            reset_robot_agent(
                agent=self.agent,
                robot_uids=self.robot_uids,
                batch_size=b,
                device=dev,
                robot_init_qpos_noise=self.robot_init_qpos_noise,
                episode_rng=self._episode_rng,
                root_pose=ROBOT_ROOT_POSE,
            )
            t = lambda rows: torch.tensor(np.asarray(rows, dtype=np.float32), device=dev)  # noqa: E731
            if not hasattr(self, "_button_dir_local"):
                if b != self.num_envs:
                    raise RuntimeError("the first reset must cover every env (button calibration)")
                # Calibrate with every blender on ONE pose, so the measurement
                # and the single `self.geometry` it reads agree; then move them
                # apart below.
                self.geometry = geometries[0]
                self.blender.set_pose(
                    Pose.create_from_pq(t([geometries[0].blender_root] * b), t([geometries[0].blender_quat] * b))
                )
                self.blender.set_qpos(torch.zeros_like(self.blender.get_qpos()))
                self.blender.set_qvel(torch.zeros_like(self.blender.get_qvel()))
                self._calibrate_button_geometry()
                p0 = t([geometries[0].blender_root])
                q0 = t([geometries[0].blender_quat])
                q0_inv = q0 * torch.tensor([1.0, -1.0, -1.0, -1.0], device=dev)
                self._button_dir_local = self._quat_rotate(q0_inv, self.button_press_direction[:1])[0]
                self._button_surface_local = self._quat_rotate(q0_inv, self.button_released_surface_position[:1] - p0)[0]
                self._button_link_local = self._quat_rotate(q0_inv, self.button_released_link_position[:1] - p0)[0]
                self.button_press_direction = self.button_press_direction.clone()
                self.button_released_surface_position = self.button_released_surface_position.clone()
                self.button_released_link_position = self.button_released_link_position.clone()
                self.button_world_travel = self.button_world_travel.clone()

            self.blender.set_pose(
                Pose.create_from_pq(t([g.blender_root for g in geometries]), t([g.blender_quat for g in geometries]))
            )
            qpos = torch.zeros((b, self.blender.get_qpos().shape[-1]), device=dev)
            qpos[:, self.button_joint_index] = float(self.button_rest_qpos)
            self.blender.set_qpos(qpos)
            self.blender.set_qvel(torch.zeros_like(qpos))
            if hasattr(self.button_joint, "set_drive_target"):
                # Constant for every env: written under a full mask (the joint
                # setter mispairs indices under a partial one).
                with full_reset_mask(self.scene):
                    self.button_joint.set_drive_target(self.button_rest_qpos)
            self._write_button_world_frame(env_idx)
            self.button_hold_counter[env_idx] = 0
            self.success_latched[env_idx] = False

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        super().set_state_dict(state, env_idx)
        if self.num_envs > 1 and hasattr(self, "_button_dir_local"):
            restore_idx = (
                torch.arange(self.num_envs, device=self.device)
                if env_idx is None
                else torch.as_tensor(env_idx, dtype=torch.long, device=self.device).flatten()
            )
            # The restored blender sits where the DEMO had it.
            self._write_button_world_frame(restore_idx)


# ------------------------------------------------------------------ apple


def apple_inside_blender_live(apple_position, blender_p, blender_q, nominal, *, device):
    """``apple_inside_blender`` with one opening per env, read off the live blender."""
    from tasks.juice_preparation.juice_preparation_apple_to_blender import BLENDER_BOTTOM_CLEARANCE, BLENDER_INSIDE_RADIUS

    opening = live_blender_point(blender_p, blender_q, nominal, "seat_position")
    xy = torch.linalg.norm(apple_position[:, :2] - opening[:, :2], dim=1)
    bottom = float(nominal.blender_root[2] + nominal.blender_lower[2] + BLENDER_BOTTOM_CLEARANCE)
    return (
        (xy <= BLENDER_INSIDE_RADIUS)
        & (apple_position[:, 2] >= bottom)
        & (apple_position[:, 2] <= opening[:, 2] + 0.015)
    ), opening, xy


class TakeAppleToBlenderMixin(MultiEnvInitMixin):
    def _load_scene(self, options: dict):
        if self.num_envs == 1:
            return super()._load_scene(options)
        import tasks.juice_preparation.juice_preparation_apple_to_blender as module

        load_fridge = module.load_custom_tabletop_fridge

        def load_in_every_scene(*args, **kwargs):
            with patched(ArticulationBuilder, "set_scene_idxs", _all_scenes):
                return load_fridge(*args, **kwargs)

        with patched(module, "load_custom_tabletop_fridge", load_in_every_scene):
            super()._load_scene(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.num_envs == 1:
            return super()._initialize_episode(env_idx, options)
        import sapien

        from tasks.juice_preparation.juice_preparation_apple_to_blender import (
            FRIDGE_ROOT_P,
            ROBOT_ROOT_P,
            apply_arm_init_qpos,
            layout_rng,
            new_apple_motion_state,
            sample_juice_layout,
            set_work_table_pose,
            shifted_blender_geometry,
        )

        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            set_work_table_pose(self)
            b = len(env_idx)
            dev = self.device
            fridge_p = np.zeros((b, 3))
            apple_p = np.zeros((b, 3))
            apple_q = np.zeros((b, 4))
            blender_p = np.zeros((b, 3))
            blender_q = np.zeros((b, 4))
            for k, env_i in enumerate(env_idx.tolist()):
                layout = sample_juice_layout(layout_rng(self, int(env_i)))
                fdx, fdy = layout["fridge"]
                fridge_p[k] = np.asarray(FRIDGE_ROOT_P, dtype=np.float64) + np.array([fdx, fdy, 0.0])
                pose = self._apple_world_pose(fridge_p[k])
                apple_p[k], apple_q[k] = pose.p, pose.q
                geometry = shifted_blender_geometry(self._blender_geometry_nominal, *layout["blender"])
                blender_p[k], blender_q[k] = geometry.blender_root, geometry.blender_quat
            self._juice_layout = layout
            t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731

            if hasattr(self.agent, "get_reset_pos"):
                robot_qpos = self.agent.get_reset_pos(device=dev)
            else:
                robot_qpos = torch.as_tensor(self.agent.keyframes["rest"].qpos, dtype=torch.float32, device=dev)
            robot_qpos = apply_arm_init_qpos(self.agent, robot_qpos.to(torch.float32).expand(b, -1).clone())
            active_names = [joint.name for joint in self.agent.robot.get_active_joints()]
            arm_indices = [
                active_names.index(name) for name in getattr(self.agent, "arm_joint_names", []) if name in active_names
            ]
            if arm_indices and self.robot_init_qpos_noise > 0:
                index = torch.tensor(arm_indices, dtype=torch.long, device=dev)
                for k, env_i in enumerate(env_idx.tolist()):
                    noise = self._batched_episode_rng[int(env_i)].normal(
                        0.0, self.robot_init_qpos_noise, (len(arm_indices),)
                    )
                    robot_qpos[k, index] += common.to_tensor(noise, device=dev).to(torch.float32)
            self.agent.reset(robot_qpos)
            self.agent.robot.set_root_pose(sapien.Pose(p=ROBOT_ROOT_P))

            self.fridge.set_pose(Pose.create_from_pq(p=t(fridge_p), q=t([[1.0, 0.0, 0.0, 0.0]]).expand(b, 4)))
            door = float(np.deg2rad(np.clip(self.initial_door_angle_deg, 0.0, 110.0)))
            fridge_qpos = torch.full((b, self.fridge.get_qpos().shape[-1]), door, dtype=torch.float32, device=dev)
            self.fridge.set_qpos(fridge_qpos)
            self.fridge.set_qvel(torch.zeros_like(fridge_qpos))

            zero_velocity = torch.zeros((b, 3), dtype=torch.float32, device=dev)
            self.apple.set_pose(Pose.create_from_pq(p=t(apple_p), q=t(apple_q)))
            self.apple.set_linear_velocity(zero_velocity)
            self.apple.set_angular_velocity(zero_velocity)

            self.blender.set_pose(Pose.create_from_pq(t(blender_p), t(blender_q)))
            if self.blender_is_articulation:
                zeros = torch.zeros((b, self.blender.get_qpos().shape[-1]), dtype=torch.float32, device=dev)
                self.blender.set_qpos(zeros)
                self.blender.set_qvel(zeros)

            self.stable_counter[env_idx] = 0
            self.success_latched[env_idx] = False
            self.apple_left_fridge_latched[env_idx] = False
            self.apple_lifted_latched[env_idx] = False
            fresh = new_apple_motion_state(self.num_envs, dev)
            for key, value in fresh.items():
                self.apple_motion[key][env_idx] = value[env_idx]

    def evaluate(self):
        if self.num_envs == 1:
            return super().evaluate()
        import tasks.juice_preparation.juice_preparation_apple_to_blender as module

        def live(apple_position, blender_geometry, *, device):
            return apple_inside_blender_live(
                apple_position, self.blender.pose.p, self.blender.pose.q, self._blender_geometry_nominal, device=device
            )

        with patched(module, "apple_inside_blender", live):
            return super().evaluate()

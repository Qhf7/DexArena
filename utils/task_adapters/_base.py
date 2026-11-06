"""Mixins every adapted task class gets, plus the two generic GPU helpers."""

from __future__ import annotations

import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Pose

from utils.camera_configs import (
    DEXARENA_WRIST_CAMERA_NAME,
    build_dexarena_fixed_wrist_camera,
    resolve_wrist_camera_mount,
)

from ._shims import full_reset_mask, patched


class PixelObsMixin:
    """The observation the pixel agents consume.

    * the policy wrist camera is declared on the hand flange of
      ``WRIST_MOUNT_BY_TASK`` (a ``sensor_configs`` override can re-pose a
      camera but cannot create one or move its mount);
    * ``proprio`` needs ``tcp_pose``, so it is added to the rgb observation when
      the task does not expose it;
    * ``_pixel_reward_info_key`` replaces the task's sparse reward by ``info[key]``.
    """

    _pixel_env_id: str = ""
    _pixel_reward_info_key: str | None = None

    @property
    def _default_sensor_configs(self):
        configs = super()._default_sensor_configs
        configs = list(configs) if isinstance(configs, (list, tuple)) else [configs]
        configs = [c for c in configs if c.uid != DEXARENA_WRIST_CAMERA_NAME]
        try:
            mount = resolve_wrist_camera_mount(
                self.agent.robot, self._pixel_env_id, agent=self.agent
            )
        except RuntimeError:
            return configs  # no wrist flange on this robot variant
        return configs + [build_dexarena_fixed_wrist_camera(mount=mount)]

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)
        if "state" not in self.obs_mode and "tcp_pose" not in obs:
            # The palm (`agent.tcp`) first, as the checkpoints were trained; the
            # dual-hand agents also expose `tcp_pose` (midpoint, ~0.2 m away).
            tcp = getattr(self.agent, "tcp", None)
            obs["tcp_pose"] = (
                tcp.pose.raw_pose if tcp is not None else self.agent.tcp_pose.raw_pose
            )
        return obs

    def compute_sparse_reward(self, obs, action, info):
        key = self._pixel_reward_info_key
        if key is None:
            return super().compute_sparse_reward(obs, action, info)
        return info[key].to(torch.float32)


class NumEnvsTrampoline(BaseEnv):
    """Sits between the task class and ``BaseEnv`` in the adapted MRO.

    Tasks that only support one env refuse ``num_envs != 1`` at the top of
    their ``__init__``. ``MultiEnvInitMixin`` hands them ``num_envs=1`` and this
    class forwards the real count to ``BaseEnv``.
    """

    def __init__(self, *args, num_envs: int = 1, **kwargs):
        super().__init__(*args, num_envs=self._pixel_true_num_envs, **kwargs)


class MultiEnvInitMixin:
    _pixel_needs_trampoline = True

    def __init__(self, *args, num_envs: int = 1, **kwargs):
        self._pixel_true_num_envs = int(num_envs)
        super().__init__(*args, num_envs=1, **kwargs)


class GpuLoadWritesMixin:
    """Defer scene-load writes that need the GPU buffers.

    Under ``physx_cuda`` the GPU buffers do not exist until after ``_load_scene``,
    so a joint drive target written there raises ("cannot parse `typestr`").
    Targets are recorded at load and written in ``_after_reconfigure`` under a
    full reset mask; an actor pose becomes its ``initial_pose``. CPU untouched.
    """

    def _load_scene(self, options):
        if not self.gpu_sim_enabled:
            return super()._load_scene(options)
        from mani_skill.utils.structs.actor import Actor
        from mani_skill.utils.structs.articulation_joint import ArticulationJoint

        deferred: list[tuple] = []
        scene = self.scene

        def set_drive_target(joint, target):
            deferred.append((joint, "drive_target", target))

        def set_drive_velocity_target(joint, velocity):
            deferred.append((joint, "drive_velocity_target", velocity))

        def set_pose(actor, pose):
            actor.initial_pose = Pose.create(pose, device=scene.device)

        with patched(ArticulationJoint, "set_drive_target", set_drive_target), patched(
            ArticulationJoint, "set_drive_velocity_target", set_drive_velocity_target
        ), patched(Actor, "set_pose", set_pose):
            super()._load_scene(options)
        self._pixel_deferred_load_writes = deferred

    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        writes = getattr(self, "_pixel_deferred_load_writes", ())
        if writes:
            with full_reset_mask(self.scene):
                for joint, attr, value in writes:
                    setattr(joint, attr, value)
        self._pixel_deferred_load_writes = []

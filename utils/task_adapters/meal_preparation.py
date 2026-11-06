"""MealPreparation stages under physx_cuda and partial resets.

The per-episode caches (``_episode_*_pose``, ``box_lid_target_qpos``,
``_episode_open_lid_qpos``) are replaced by whatever a reset produces, so a
partial reset left them ``(b, ...)`` instead of ``(num_envs, ...)`` and the
next demo-state restore indexed them with env ids (CUDA index out of bounds).
The adapter keeps them full-size. The spoon's load-time pose write and the
box's partial restore need the GPU-specific handling of the base mixins.
"""

from __future__ import annotations

import torch

from ._base import GpuLoadWritesMixin
from ._shims import partial_gpu_reset, patched, scatter_rows, slice_full_writes

EPISODE_CACHES = (
    "_episode_box_pose",
    "_episode_bread_pose",
    "_episode_ball_pose",
    "_episode_spoon_pose",
    "_episode_plate_pose",
    "_episode_open_lid_qpos",
    "box_lid_target_qpos",
)


class MealPreparationMixin(GpuLoadWritesMixin):
    def _snapshot_caches(self):
        return {name: getattr(self, name, None) for name in EPISODE_CACHES}

    def _merge_caches(self, before: dict, env_idx):
        """Any cache a reset rewrote with ``len(env_idx)`` rows goes back to full size."""
        if self.num_envs == 1:
            return
        for name, old in before.items():
            new = getattr(self, name, None)
            if new is None or new is old or new.shape[0] == self.num_envs:
                continue
            setattr(self, name, scatter_rows(self, old, env_idx, new))

    def _initialize_actors(self, env_idx: torch.Tensor):
        before = self._snapshot_caches()
        super()._initialize_actors(env_idx)
        self._merge_caches(before, env_idx)

    def _set_box_lid_drive_target(self, target_qpos: torch.Tensor):
        before = getattr(self, "box_lid_target_qpos", None)
        super()._set_box_lid_drive_target(target_qpos)
        new = getattr(self, "box_lid_target_qpos", None)
        if new is not None and new.shape[0] != self.num_envs:
            rows = self.scene._reset_mask.nonzero().flatten()
            self.box_lid_target_qpos = scatter_rows(self, before, rows, new)

    def _seat_object_on_plate(self, obj, bbox, env_idx, xy_radius, fallback_z=0.035):
        cache = self._episode_plate_pose
        if cache.shape[0] == len(env_idx):
            return super()._seat_object_on_plate(obj, bbox, env_idx, xy_radius, fallback_z)
        # The base reads the cache as one row per env being reset.
        with patched(self, "_episode_plate_pose", cache[env_idx]):
            return super()._seat_object_on_plate(obj, bbox, env_idx, xy_radius, fallback_z)

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        restore_idx = (
            torch.arange(self.num_envs, device=self.device)
            if env_idx is None
            else torch.as_tensor(env_idx, dtype=torch.long, device=self.device).flatten()
        )
        before = self._snapshot_caches()
        # The fixed-root box is restored through full-size qpos/qvel tensors;
        # a partial GPU reset writes only the reset rows.
        with slice_full_writes(self, getattr(self, "box", None)):
            super().set_state_dict(state, env_idx)
        self._merge_caches(before, restore_idx)

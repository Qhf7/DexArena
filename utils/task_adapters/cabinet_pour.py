"""CabinetPour stage envs."""

from __future__ import annotations

import numpy as np
import sapien
import torch

from mani_skill.utils.structs.pose import Pose

# The milestone that is a stage env's own success counts the control step it
# first becomes true (the long task debounces it).
STAGE_HOLD_COUNT_ATTR = {
    "CabinetPouringOpenCabinet-Sharpa-v1": "_door_open_hold_count",
    "CabinetPouringPickBottle-Sharpa-v1": "_bottle_place_hold_count",
    "CabinetPouringUnscrewCap-Sharpa-v1": "_cap_place_hold_count",
    "CabinetPouringPourBeads-Sharpa-v1": "_pour_hold_count",
}


class CabinetPourStageMixin:
    def _qualify_control_steps(self, candidate, count_attr, last_step_attr, hold_steps):
        if count_attr == STAGE_HOLD_COUNT_ATTR.get(self._pixel_env_id):
            hold_steps = 1
        return super()._qualify_control_steps(candidate, count_attr, last_step_attr, hold_steps)


class BottlePickPlaceMixin(CabinetPourStageMixin):
    def _capture_new_bottle_locks(self, newly_placed: torch.Tensor):
        # Under GPU sim the swap to the root-fixed bottle twin writes render
        # visibility, which the batched renderer refuses once it exists. This
        # stage ends at that transition, so the twin is never needed there.
        if self.gpu_sim_enabled:
            return
        super()._capture_new_bottle_locks(newly_placed)


class CapRemovePlaceMixin(CabinetPourStageMixin):
    """Render the cap stage under the batched (GPU) renderer.

    The task toggles ``RenderBodyComponent.visibility`` at reset and detachment,
    which the batched render system forbids once built. Under GPU sim nothing
    writes visibility after load: the parked bottle representations stay
    visible, the three physical caps are hidden once at load, and a
    collision-free proxy with the same meshes follows whichever is active.
    """

    @property
    def _hide_by_pose_active(self) -> bool:
        return bool(self.gpu_sim_enabled)

    def _load_scene(self, options: dict):
        from tasks.cabinet_pouring.cabinet_pouring_long import (
            BOTTLE_ASSET_DIR,
            CAP_MARKER_BOXES,
            CAP_TWIN_PARK_Z,
            CabinetPouringLongEnv,
        )

        super()._load_scene(options)
        if not self._hide_by_pose_active:
            return
        for i in range(self.num_envs):
            for cap in (self.cap, self.cap_fixed, self.cap_twin):
                CabinetPouringLongEnv._set_body_visible(cap, i, False)
        self.cap_render_proxy = self._build_mesh_asset(
            asset_dir=BOTTLE_ASSET_DIR,
            name="source_cap_render_proxy",
            visual_glob="lid_vis/*.obj",
            collision_glob="lid_col_solid/*.obj",
            add_collision=False,
            scale=self.source_scale,
            # The mesh frame of the twin and of the URDF cap link (rpy x=pi/2).
            mesh_pose=sapien.Pose(q=[np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]),
            initial_pose=sapien.Pose(p=[0.0, 0.0, CAP_TWIN_PARK_Z]),
            body_type="kinematic",
            marker_boxes=CAP_MARKER_BOXES,
        )
        # Render-only, like the placement markers: recorded demo states do not
        # carry it and must still restore.
        self.remove_from_state_dict_registry(self.cap_render_proxy)

    def _set_body_visible(self, obj, i: int, visible: bool):
        if not self._hide_by_pose_active:
            super()._set_body_visible(obj, i, visible)

    def _set_render_visibility(self, body, i: int, value: float):
        if not self._hide_by_pose_active:
            super()._set_render_visibility(body, i, value)

    def _ghost_render_body(self, i: int):
        # None makes `_set_ghost_active` skip its visibility write; it still
        # applies the collision groups.
        if self._hide_by_pose_active:
            return None
        return super()._ghost_render_body(i)

    def _sync_cap_render_proxy(self):
        if not self._hide_by_pose_active:
            return
        self.cap_render_proxy.set_pose(Pose.create_from_pq(p=self._cap_p(), q=self._cap_q()))
        if self.scene._gpu_sim_initialized:
            self.scene.px.gpu_apply_rigid_dynamic_data()
            self.scene.px.gpu_fetch_rigid_dynamic_data()

    def _activate_fixed_bottle_at_reset(self, env_idx: torch.Tensor):
        """A -> B for the reset sub-batch only.

        The base ``_swap_to_fixed_bottle`` builds tensors over all envs, which
        a partial GPU reset rejects (shape mismatch); same writes, restricted
        to ``env_idx``.
        """
        from tasks.cabinet_pouring.cabinet_pouring_long import BOTTLE_DYNAMIC_PARK_P, BOTTLE_REP_FIXED

        if len(env_idx) == 0:
            return
        env_idx = torch.as_tensor(env_idx, dtype=torch.long, device=self.device)
        b = len(env_idx)
        source_pose = self.source.pose.raw_pose[env_idx].clone()
        source_qpos = self.bottle_screw.get_qpos()[env_idx].clone()

        self.bottle_screw_fixed.set_root_pose(Pose(source_pose))
        fixed_qpos = self.bottle_screw_fixed.get_qpos()[env_idx].clone()
        fixed_qpos[:, self._cap_screw_qindex_fixed] = source_qpos[:, self._cap_screw_qindex]
        self.bottle_screw_fixed.set_qpos(fixed_qpos)
        self.bottle_screw_fixed.set_qvel(torch.zeros_like(fixed_qpos))
        root_zero = torch.zeros((b, 3), device=self.device)
        self.bottle_screw_fixed.set_root_linear_velocity(root_zero)
        self.bottle_screw_fixed.set_root_angular_velocity(root_zero)

        dynamic_pose = self.bottle_screw.root_pose.raw_pose[env_idx].clone()
        dynamic_pose[:, :3] = torch.tensor(BOTTLE_DYNAMIC_PARK_P, device=self.device)
        self.bottle_screw.set_root_pose(Pose(dynamic_pose))
        self.bottle_screw.set_qvel(torch.zeros_like(source_qpos))

        self._active_bottle_rep[env_idx] = BOTTLE_REP_FIXED
        self._sync_bottle_representation_runtime(env_idx)
        self._capture_cap_twist_origins(env_idx)
        self._flush_bottle_representation_state()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # Re-arm B's joint-cap collisions: detachment disables them and the
        # base reset only re-arms A's, so later episodes would start with a cap
        # the fingers pass through.
        for i in torch.as_tensor(env_idx).reshape(-1).tolist():
            self._set_ghost_active(int(i), True)

    def evaluate(self):
        info = super().evaluate()
        # Last, after any detach swap: every render (step, reset, restored
        # state via get_obs) goes through evaluate first.
        self._sync_cap_render_proxy()
        return info

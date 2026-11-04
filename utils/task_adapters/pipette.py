"""PipetteTransfer under physx_cuda and num_envs > 1."""

from __future__ import annotations

import torch

from ._base import GpuLoadWritesMixin
from ._shims import partial_gpu_reset, slice_full_writes
from .state_dict import MergedStateDictMixin


class PipetteTransferMixin(MergedStateDictMixin, GpuLoadWritesMixin):
    """The plunger drive target is written at load (deferred past GPU init);
    the reset writes are sized to the sub-batch of a partial GPU reset."""

    def _initialize_episode(self, env_idx, options):
        with slice_full_writes(self, self.pipette):
            super()._initialize_episode(env_idx, options)

    def _reset_arms(self):
        from tasks.config.layout_randomization import arm_qpos_noise, episode_rng
        from tasks.pipette_transfer.pipette_transfer_long import ARM_INIT_QPOS

        qpos = self.agent.robot.get_qpos().clone()
        active = [j.name for j in self.agent.robot.get_active_joints()]
        idx = [active.index(n) for n in getattr(self.agent, "arm_joint_names", []) if n in active]
        shared = torch.tensor(ARM_INIT_QPOS, dtype=qpos.dtype, device=self.device)
        for start in range(0, len(idx), len(ARM_INIT_QPOS)):
            block = idx[start : start + len(ARM_INIT_QPOS)]
            if len(block) == len(ARM_INIT_QPOS):
                qpos[:, torch.tensor(block, device=self.device)] = shared
        # The noise of env i comes from env i's own episode RNG, and a partial
        # GPU reset writes only the rows of the envs being reset.
        if partial_gpu_reset(self):
            ids = self.scene._reset_mask.nonzero().flatten().tolist()
        else:
            ids = list(range(qpos.shape[0]))
        for k in ids:
            index, noise = arm_qpos_noise(
                episode_rng(self, k), self.agent, 1, self.robot_init_qpos_noise, device=self.device
            )
            qpos[k, index] += noise[0]
        if len(ids) != qpos.shape[0]:
            rows = torch.tensor(ids, device=self.device)
            self.agent.robot.set_qpos(qpos[rows])
            self.agent.robot.set_qvel(torch.zeros_like(qpos[rows]))
        else:
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_qvel(torch.zeros_like(self.agent.robot.get_qvel()))
        self.agent.controller.reset()

"""Shared controller/reset-pose helpers for dual Panda-arm impedance robots."""

import torch

from mani_skill.agents.controllers import (
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
)

from tasks.robots.franka_joint_impedance_controller import (
    FrankaJointImpedanceControllerConfig,
)


class DualArmImpedanceMixin:
    """Common arm-controller and reset-pose logic for dual Panda-arm robots.

    Subclasses must define ``side_names``, ``arm_joint_names``,
    ``arm_stiffness``, ``arm_damping``, ``arm_force_limit``, ``arm_impedance``,
    and ``_arm_joint_indices`` (set in ``_after_init``), plus a
    ``_side_reset_qpos(side)`` method returning a ``{joint_name: value}``
    mapping for everything on that side (arm + end effector).
    """

    def _side_reset_qpos(self, side: str) -> dict:
        raise NotImplementedError

    def get_reset_pos(self, device=None, dtype=torch.float32):
        device = self.device if device is None else device
        qpos_by_name = {}
        for side in self.side_names:
            qpos_by_name.update(self._side_reset_qpos(side))
        active_names = [joint.name for joint in self.robot.get_active_joints()]
        missing = [name for name in active_names if name not in qpos_by_name]
        if missing:
            raise KeyError(f"Missing qpos values for active joints: {missing}")
        return torch.tensor(
            [qpos_by_name[name] for name in active_names],
            device=device,
            dtype=dtype,
        )

    def _arm_pd_joint_pos(self, joint_names, gpu_fetch=True, gpu_apply=True):
        if self.arm_impedance:
            return FrankaJointImpedanceControllerConfig(
                joint_names,
                lower=None,
                upper=None,
                gpu_fetch=gpu_fetch,
                gpu_apply=gpu_apply,
            )
        return PDJointPosControllerConfig(
            joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

    def _arm_pd_ee_delta_pose(self, joint_names, ee_link):
        return PDEEPoseControllerConfig(
            joint_names=joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=ee_link,
            urdf_path=self.urdf_path,
        )

    def is_static(self, threshold: float = 0.2):
        arm_qvel = self.robot.get_qvel()[..., self._arm_joint_indices]
        return torch.max(torch.abs(arm_qvel), dim=1).values <= threshold

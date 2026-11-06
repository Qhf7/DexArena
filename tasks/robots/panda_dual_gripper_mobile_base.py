"""Two upright Panda arms with stock Franka grippers on mobile base."""

from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import Keyframe
from tasks.robots.panda_base import PandaAgent
from mani_skill.agents.controllers import (
    PDJointPosMimicControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

from tasks.robots.dual_arm_impedance_mixin import DualArmImpedanceMixin


@register_agent()
class PandaDualGripperMobileBase(DualArmImpedanceMixin, PandaAgent):
    """Stationary dual Panda with the stock two-finger Franka grippers."""

    uid = "panda_dual_gripper_mobile_base"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_dual_gripper_mobile_base.urdf"
    )
    urdf_path = str(_urdf_path.resolve())

    side_names = ("left", "right")
    single_arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    single_gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    single_ee_link_name = "panda_hand_tcp"

    left_arm_joint_names = [f"left_{name}" for name in single_arm_joint_names]
    right_arm_joint_names = [f"right_{name}" for name in single_arm_joint_names]
    left_gripper_joint_names = [
        f"left_{name}" for name in single_gripper_joint_names
    ]
    right_gripper_joint_names = [
        f"right_{name}" for name in single_gripper_joint_names
    ]
    arm_joint_names = left_arm_joint_names + right_arm_joint_names
    gripper_joint_names = left_gripper_joint_names + right_gripper_joint_names

    left_ee_link_name = f"left_{single_ee_link_name}"
    right_ee_link_name = f"right_{single_ee_link_name}"
    ee_link_name = right_ee_link_name

    left_finger_link_names = [
        "left_panda_leftfinger",
        "left_panda_rightfinger",
    ]
    right_finger_link_names = [
        "right_panda_leftfinger",
        "right_panda_rightfinger",
    ]
    left_finger_pad_link_names = [
        "left_panda_leftfinger_pad",
        "left_panda_rightfinger_pad",
    ]
    right_finger_pad_link_names = [
        "right_panda_leftfinger_pad",
        "right_panda_rightfinger_pad",
    ]

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            name: dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1)
            for name in left_finger_link_names + right_finger_link_names
        },
    )
    disable_self_collisions = True

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    arm_impedance = True

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    single_arm_joint_reset_pos = np.array(
        [0.0, 0.0, 0.0, -np.pi * 2 / 3, 0.0, np.pi, np.pi / 4]
    )
    single_gripper_joint_reset_pos = np.array([0.04, 0.04])

    keyframes = dict(
        rest=Keyframe(
            qpos=np.concatenate(
                [
                    single_arm_joint_reset_pos,
                    single_gripper_joint_reset_pos,
                    single_arm_joint_reset_pos,
                    single_gripper_joint_reset_pos,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    def _side_reset_qpos(self, side: str) -> dict:
        qpos_by_name = dict(
            zip(
                getattr(self, f"{side}_arm_joint_names"),
                self.single_arm_joint_reset_pos,
            )
        )
        qpos_by_name.update(
            zip(
                getattr(self, f"{side}_gripper_joint_names"),
                self.single_gripper_joint_reset_pos,
            )
        )
        return qpos_by_name

    def _gripper_pd_joint_pos(self, joint_names):
        return PDJointPosMimicControllerConfig(
            joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={joint_names[1]: {"joint": joint_names[0]}},
        )

    @property
    def _controller_configs(self):
        left_arm_joint = self._arm_pd_joint_pos(
            self.left_arm_joint_names, gpu_fetch=True, gpu_apply=False
        )
        right_arm_joint = self._arm_pd_joint_pos(
            self.right_arm_joint_names, gpu_fetch=False, gpu_apply=True
        )
        left_arm_ee = self._arm_pd_ee_delta_pose(
            self.left_arm_joint_names, self.left_ee_link_name
        )
        right_arm_ee = self._arm_pd_ee_delta_pose(
            self.right_arm_joint_names, self.right_ee_link_name
        )
        left_gripper = self._gripper_pd_joint_pos(self.left_gripper_joint_names)
        right_gripper = self._gripper_pd_joint_pos(self.right_gripper_joint_names)

        return deepcopy_dict(
            dict(
                pd_joint_pos=dict(
                    left_arm=left_arm_joint,
                    right_arm=right_arm_joint,
                    left_gripper=left_gripper,
                    right_gripper=right_gripper,
                ),
                pd_ee_delta_pose=dict(
                    left_arm=left_arm_ee,
                    right_arm=right_arm_ee,
                    left_gripper=left_gripper,
                    right_gripper=right_gripper,
                ),
            )
        )

    def _after_init(self):
        links = self.robot.get_links()
        self.left_tcp = sapien_utils.get_obj_by_name(links, self.left_ee_link_name)
        self.right_tcp = sapien_utils.get_obj_by_name(links, self.right_ee_link_name)
        self.tcp = self.right_tcp
        self.left_finger_links = [
            sapien_utils.get_obj_by_name(links, name)
            for name in self.left_finger_link_names
        ]
        self.right_finger_links = [
            sapien_utils.get_obj_by_name(links, name)
            for name in self.right_finger_link_names
        ]
        self.left_finger_pad_links = [
            sapien_utils.get_obj_by_name(links, name)
            for name in self.left_finger_pad_link_names
        ]
        self.right_finger_pad_links = [
            sapien_utils.get_obj_by_name(links, name)
            for name in self.right_finger_pad_link_names
        ]
        # For a stock gripper the TCP is the palm-equivalent reference frame,
        # and the two finger bodies are the available tip/contact links.
        self.left_palm_link = self.left_tcp
        self.right_palm_link = self.right_tcp
        self.palm_link = self.right_palm_link
        self.left_tip_links = self.left_finger_links
        self.right_tip_links = self.right_finger_links
        self.tip_links = self.left_tip_links + self.right_tip_links
        # Single-Panda-style aliases default to the right gripper.
        self.finger1_link, self.finger2_link = self.right_finger_links
        self.finger1pad_link, self.finger2pad_link = self.right_finger_pad_links
        active_names = [joint.name for joint in self.robot.get_active_joints()]
        self._arm_joint_indices = torch.tensor(
            [active_names.index(name) for name in self.arm_joint_names],
            device=self.device,
        )

    def _is_grasping_with_fingers(
        self, object: Actor, finger_links, min_force=0.5, max_angle=85
    ):
        left_force = self.scene.get_pairwise_contact_forces(finger_links[0], object)
        right_force = self.scene.get_pairwise_contact_forces(finger_links[1], object)
        left_force_norm = torch.linalg.norm(left_force, dim=1)
        right_force_norm = torch.linalg.norm(right_force, dim=1)
        left_direction = finger_links[0].pose.to_transformation_matrix()[..., :3, 1]
        right_direction = -finger_links[1].pose.to_transformation_matrix()[..., :3, 1]
        left_angle = common.compute_angle_between(left_direction, left_force)
        right_angle = common.compute_angle_between(right_direction, right_force)
        left_ok = torch.logical_and(
            left_force_norm >= min_force, torch.rad2deg(left_angle) <= max_angle
        )
        right_ok = torch.logical_and(
            right_force_norm >= min_force, torch.rad2deg(right_angle) <= max_angle
        )
        return torch.logical_and(left_ok, right_ok)

    def is_left_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        return self._is_grasping_with_fingers(
            object, self.left_finger_links, min_force, max_angle
        )

    def is_right_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        return self._is_grasping_with_fingers(
            object, self.right_finger_links, min_force, max_angle
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Return true when either the left or right gripper grasps ``object``."""
        return torch.logical_or(
            self.is_left_grasping(object, min_force, max_angle),
            self.is_right_grasping(object, min_force, max_angle),
        )

    @property
    def left_tcp_pos(self):
        return self.left_tcp.pose.p

    @property
    def right_tcp_pos(self):
        return self.right_tcp.pose.p

    @property
    def tcp_pos(self):
        return 0.5 * (self.left_tcp_pos + self.right_tcp_pos)

    @property
    def left_tcp_pose(self):
        return self.left_tcp.pose

    @property
    def right_tcp_pose(self):
        return self.right_tcp.pose

    @property
    def tcp_pose(self):
        return self.right_tcp_pose


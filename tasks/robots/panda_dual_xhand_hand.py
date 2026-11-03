"""Trapezoid base + 2 Panda arms + 2 RobotEra XHand1 hands.

The rig (base pose, arm mounts, arm gains, reset pose) is IDENTICAL to
``panda_dual_wuji_hand_trapezoid_base`` — the URDF is generated from it by
``scripts/build_dual_xhand_trapezoid_urdf.py``, which swaps only the hands, so
the operator stands against the same frozen operating station on every dual rig.

RobotEra ships right-hand assets only; the LEFT hand is the mirrored asset from
``data_gen/teleop/make_mirrored_hand.py`` (chirality pinned by
``tests/test_hand_retargeting_chirality.py``). Both hands keep the right hand's
joint sign convention, so one qpos vector means the same finger curl on both
sides and ``hand_controller_gains`` tables apply unchanged via ``_strip_side``.
"""

from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import Keyframe
from tasks.robots.panda_base import PandaAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent

from tasks.robots.franka_joint_impedance_controller import (
    FrankaJointImpedanceControllerConfig,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from tasks.robots.panda_xhand_hand_right import PandaXHandHandRight
from tasks.config.contact_materials import apply_xhand_robot_contact_materials
from tasks.robots.hand_controller_gains import (
    HAND_JOINT_MAX_VELOCITY,
    hand_urdf_config,
    xhand_marker_links,
    xhand_pad_links,
    XHAND_HAND_ARMATURE,
    XHAND_HAND_DAMPING,
    XHAND_HAND_FORCE_LIMIT,
    XHAND_HAND_FRICTIONLOSS,
    XHAND_HAND_KP,
    apply_joint_physics,
    gains_to_list,
)


class PandaDualXHandHand(PandaAgent):

    side_names = ("left", "right")

    single_arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    # Side-stripped names; same order as panda_xhand_hand_right.hand_joint_names
    # (thumb 3, index 3, mid/ring/pinky 2 each).
    single_hand_joint_names = [
        "hand_thumb_bend_joint",
        "hand_thumb_rota_joint1",
        "hand_thumb_rota_joint2",
        "hand_index_bend_joint",
        "hand_index_joint1",
        "hand_index_joint2",
        "hand_mid_joint1",
        "hand_mid_joint2",
        "hand_ring_joint1",
        "hand_ring_joint2",
        "hand_pinky_joint1",
        "hand_pinky_joint2",
    ]
    single_tip_link_names = [
        "hand_thumb_rota_tip",
        "hand_index_rota_tip",
        "hand_mid_tip",
        "hand_ring_tip",
        "hand_pinky_tip",
    ]
    single_ee_link_name = "hand_link"        # palm

    left_arm_joint_names = [f"left_{n}" for n in single_arm_joint_names]
    right_arm_joint_names = [f"right_{n}" for n in single_arm_joint_names]
    left_hand_joint_names = [f"left_{n}" for n in single_hand_joint_names]
    right_hand_joint_names = [f"right_{n}" for n in single_hand_joint_names]

    arm_joint_names = left_arm_joint_names + right_arm_joint_names
    hand_joint_names = left_hand_joint_names + right_hand_joint_names

    left_ee_link_name = f"left_{single_ee_link_name}"
    right_ee_link_name = f"right_{single_ee_link_name}"
    ee_link_name = right_ee_link_name

    left_tip_link_names = [f"left_{n}" for n in single_tip_link_names]
    right_tip_link_names = [f"right_{n}" for n in single_tip_link_names]
    tip_link_name = left_tip_link_names + right_tip_link_names

    # Same pad/shell split and marker handling as the single-arm agent, both hands.
    urdf_config = hand_urdf_config(xhand_pad_links("left_") + xhand_pad_links("right_"))

    # Self-collision ON, cleaned by the same two-clique exclusion table the
    # single-arm XHand agent uses (measured there: the palm's convex hull swallows
    # the index finger's second segment by 9.27 mm at rest; everything else is
    # clean). Cross-finger and distal/tip <-> palm contact stay ON. Same policy as
    # the dual wuji rig.
    disable_self_collisions = False

    # Per-side base bit; each side uses 2 consecutive group-2 bits (arm-mount,
    # palm <-> index_rota_link1). Clear of ManiSkill's reserved 29/30/31.
    _SELF_COLLISION_BASE_BIT = {"left": 0, "right": 2}

    # Arm plant: unchanged from the dual wuji rig.
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    arm_impedance = True

    @property
    def hand_stiffness(self):
        return gains_to_list(self.hand_joint_names, XHAND_HAND_KP)

    @property
    def hand_damping(self):
        return gains_to_list(self.hand_joint_names, XHAND_HAND_DAMPING)

    @property
    def hand_force_limit(self):
        return gains_to_list(self.hand_joint_names, XHAND_HAND_FORCE_LIMIT)

    single_arm_joint_reset_pos = np.array(
        [0.0, 0.0, 0.0, -np.pi * 2 / 3, 0.0, np.pi, np.pi / 4]
    )
    single_hand_joint_reset_pos = np.zeros(len(single_hand_joint_names))

    keyframes = dict(
        rest=Keyframe(
            qpos=np.concatenate([
                single_arm_joint_reset_pos, single_hand_joint_reset_pos,
                single_arm_joint_reset_pos, single_hand_joint_reset_pos,
            ]),
            pose=sapien.Pose(),
        )
    )

    def get_reset_pos(self, device=None, dtype=torch.float32):
        device = self.device if device is None else device
        qpos_by_name = {}
        for side in self.side_names:
            arm_names = getattr(self, f"{side}_arm_joint_names")
            hand_names = getattr(self, f"{side}_hand_joint_names")
            qpos_by_name.update(zip(arm_names, self.single_arm_joint_reset_pos))
            qpos_by_name.update(zip(hand_names, self.single_hand_joint_reset_pos))
        active = [j.name for j in self.robot.get_active_joints()]
        missing = [n for n in active if n not in qpos_by_name]
        if missing:
            raise KeyError(f"Missing qpos values for active joints: {missing}")
        return torch.tensor(
            [qpos_by_name[n] for n in active], device=device, dtype=dtype,
        )

    def _arm_pd_joint_pos(self, joint_names, gpu_fetch=True, gpu_apply=True):
        if self.arm_impedance:
            # Two arm controllers run per substep: the first fetches qpos once,
            # the last applies qf once (set_qf shares one buffer and
            # DictController iterates in insertion order).
            return FrankaJointImpedanceControllerConfig(
                joint_names,
                lower=None, upper=None,
                gpu_fetch=gpu_fetch,
                gpu_apply=gpu_apply,
            )
        return PDJointPosControllerConfig(
            joint_names,
            lower=None, upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

    def _arm_pd_ee_delta_pose(self, joint_names, ee_link):
        return PDEEPoseControllerConfig(
            joint_names=joint_names,
            pos_lower=-0.1, pos_upper=0.1,
            rot_lower=-0.1, rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=ee_link,
            urdf_path=self.urdf_path,
        )

    def _hand_pd_joint_pos(self, joint_names):
        idx = {n: i for i, n in enumerate(self.hand_joint_names)}
        return PDJointPosControllerConfig(
            joint_names,
            lower=None, upper=None,
            stiffness=[self.hand_stiffness[idx[n]] for n in joint_names],
            damping=[self.hand_damping[idx[n]] for n in joint_names],
            force_limit=[self.hand_force_limit[idx[n]] for n in joint_names],
            normalize_action=False,
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
        left_hand = self._hand_pd_joint_pos(self.left_hand_joint_names)
        right_hand = self._hand_pd_joint_pos(self.right_hand_joint_names)

        return deepcopy_dict(dict(
            pd_joint_pos=dict(
                left_arm=left_arm_joint,
                right_arm=right_arm_joint,
                left_hand=left_hand,
                right_hand=right_hand,
            ),
            pd_ee_delta_pose=dict(
                left_arm=left_arm_ee,
                right_arm=right_arm_ee,
                left_hand=left_hand,
                right_hand=right_hand,
            ),
        ))

    def _apply_frozen_contact_materials(self):
        return apply_xhand_robot_contact_materials(
            self.robot, xhand_pad_links("left_") + xhand_pad_links("right_")
        )

    def _after_init(self):
        links = self.robot.get_links()
        self.left_palm_link = sapien_utils.get_obj_by_name(links, self.left_ee_link_name)
        self.right_palm_link = sapien_utils.get_obj_by_name(links, self.right_ee_link_name)
        self.palm_link = self.right_palm_link
        self.left_tip_links = [sapien_utils.get_obj_by_name(links, n) for n in self.left_tip_link_names]
        self.right_tip_links = [sapien_utils.get_obj_by_name(links, n) for n in self.right_tip_link_names]
        self.tip_links = self.left_tip_links + self.right_tip_links
        self.tcp = self.right_palm_link
        if not self.disable_self_collisions:
            for side, base in self._SELF_COLLISION_BASE_BIT.items():
                PandaXHandHandRight._apply_self_collision_srdf(
                    links, prefix=f"{side}_", arm_link=f"{side}_panda_link7",
                    base_bit=base,
                )

        # Reflected rotor inertia + transmission dry friction, same tables the
        # single-arm XHand agent uses. Frozen physical properties (sim2real).
        apply_joint_physics(self.robot, self.hand_joint_names,
                            frictionloss=XHAND_HAND_FRICTIONLOSS,
                            armature=XHAND_HAND_ARMATURE,
                            max_velocity=HAND_JOINT_MAX_VELOCITY,
                            no_collision_links=(xhand_marker_links("left_")
                                                + xhand_marker_links("right_")))

    # Per-hand wrist cameras (palm link frame); same initial pose as the dual
    # wuji rig, to be tuned for the XHand palm geometry.
    wrist_camera_pose = sapien.Pose(
        p=[0.02, -0.06, 0.05],
        q=[0.6424, 0.2955, -0.2955, 0.6424],
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig("left_wrist_camera", self.wrist_camera_pose, 320, 240,
                         np.deg2rad(58), 0.01, 100, mount=self.left_palm_link),
            CameraConfig("right_wrist_camera", self.wrist_camera_pose, 320, 240,
                         np.deg2rad(58), 0.01, 100, mount=self.right_palm_link),
        ]

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., : len(self.arm_joint_names)]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def left_tcp_pos(self):
        tips = torch.stack([t.pose.p for t in self.left_tip_links], dim=0)
        return 0.7 * tips.mean(dim=0) + 0.3 * self.left_palm_link.pose.p

    @property
    def right_tcp_pos(self):
        tips = torch.stack([t.pose.p for t in self.right_tip_links], dim=0)
        return 0.7 * tips.mean(dim=0) + 0.3 * self.right_palm_link.pose.p

    @property
    def tcp_pos(self):
        return 0.5 * (self.left_tcp_pos + self.right_tcp_pos)

    @property
    def left_tcp_pose(self):
        palm_pose = self.left_palm_link.pose
        return palm_pose.__class__.create_from_pq(p=self.left_tcp_pos, q=palm_pose.q)

    @property
    def right_tcp_pose(self):
        palm_pose = self.right_palm_link.pose
        return palm_pose.__class__.create_from_pq(p=self.right_tcp_pos, q=palm_pose.q)

    @property
    def tcp_pose(self):
        palm_pose = self.right_palm_link.pose
        return palm_pose.__class__.create_from_pq(p=self.tcp_pos, q=palm_pose.q)


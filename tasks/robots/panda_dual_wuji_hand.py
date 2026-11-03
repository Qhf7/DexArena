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

from tasks.config.contact_materials import apply_wuji_robot_contact_materials
from tasks.robots.hand_controller_gains import (
    HAND_JOINT_MAX_VELOCITY,
    WUJI_HAND_FRICTIONLOSS,
    hand_urdf_config,
    wuji_pad_links,
    HAND_JOINT_MAX_VELOCITY,
    WUJI_HAND_FRICTIONLOSS,
    hand_urdf_config,
    wuji_pad_links,
    WUJI_HAND_DAMPING,
    WUJI_HAND_FORCE_LIMIT,
    WUJI_HAND_KP,
    gains_to_list,
    WUJI_HAND_ARMATURE,
    apply_joint_physics,
)


class PandaDualWujiHand(PandaAgent):

    side_names = ("left", "right")

    single_arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    # Joint names without side prefix; matches the single-arm wrapper order.
    single_hand_joint_names = [
        f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)
    ]
    single_tip_link_names = [f"finger{f}_tip_link" for f in range(1, 6)]
    single_ee_link_name = "palm_link"

    left_arm_joint_names  = [f"left_{n}"  for n in single_arm_joint_names]
    right_arm_joint_names = [f"right_{n}" for n in single_arm_joint_names]
    left_hand_joint_names  = [f"left_{n}"  for n in single_hand_joint_names]
    right_hand_joint_names = [f"right_{n}" for n in single_hand_joint_names]

    arm_joint_names  = left_arm_joint_names  + right_arm_joint_names
    hand_joint_names = left_hand_joint_names + right_hand_joint_names

    left_ee_link_name  = f"left_{single_ee_link_name}"
    right_ee_link_name = f"right_{single_ee_link_name}"
    ee_link_name = right_ee_link_name

    left_tip_link_names  = [f"left_{n}"  for n in single_tip_link_names]
    right_tip_link_names = [f"right_{n}" for n in single_tip_link_names]
    tip_link_name = left_tip_link_names + right_tip_link_names

    # Same pad/shell split as the single-arm agent, both hands (four-layer L2).
    urdf_config = hand_urdf_config(wuji_pad_links("left_") + wuji_pad_links("right_"))

    # Self-collision ON, cleaned by a hand-authored SRDF-style exclusion table (the wuji URDF
    # ships none). WHY it's needed: wuji's finger BASES are nested inside the palm shell by
    # design (finger_link1 sits ~77% inside the palm collision hull, 4.5mm deep) and the
    # concave palm STL loads as ONE convex hull (1.39x inflated), so the palm hull engulfs the
    # finger roots. The engine only auto-excludes DIRECT parent-child pairs (palm<->link1), so
    # the grandparent (palm<->link2), within-finger, and hand<->arm-mount pairs surface as
    # spurious rest self-collisions that fling the arm with SC on. We disable exactly those via
    # shared group-2 "ignore" bits (see _apply_self_collision_srdf). Verified: rest drift
    # 0.00cm, 0 residual self-contacts, full-fist closes.  tip<->palm and link4<->palm are KEPT
    # ON so the fingertip is still blocked from sinking into its own palm (max-fold tip
    # penetration 11mm->0.7mm), and CROSS-finger collision stays ON (prevents recording
    # interpenetrating finger poses in demos). Matches shadow/sharpa self-collision quality.
    disable_self_collisions = False

    # Per-side base group-2 bit for the exclusion cliques. Each hand uses 11 consecutive bits
    # (1 arm-mount + 5*(within-finger, base<->palm)); left 0-10, right 11-21. Stays clear of
    # ManiSkill's reserved bits (29 = disable_self_collisions, 30/31 = Fetch/robocasa base).
    _SELF_COLLISION_BASE_BIT = {"left": 0, "right": 11}

    # Arm gains follow ManiSkill upstream `panda` agent. Hand gains are
    # per-joint, sourced from `tasks.robots.hand_controller_gains`
    # (see docs/hand_controller_parameters.md).
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    # Joint impedance is the repo-default pd_joint_pos arm plant; the *_stiff
    # uid keeps the legacy stiff PD arm for pre-switch data and checkpoints.
    arm_impedance = True

    @property
    def hand_stiffness(self):
        return gains_to_list(self.hand_joint_names, WUJI_HAND_KP)

    @property
    def hand_damping(self):
        return gains_to_list(self.hand_joint_names, WUJI_HAND_DAMPING)

    @property
    def hand_force_limit(self):
        return gains_to_list(self.hand_joint_names, WUJI_HAND_FORCE_LIMIT)

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
        return apply_wuji_robot_contact_materials(
            self.robot, wuji_pad_links("left_") + wuji_pad_links("right_")
        )

    def _after_init(self):
        links = self.robot.get_links()
        self.left_palm_link  = sapien_utils.get_obj_by_name(links, self.left_ee_link_name)
        self.right_palm_link = sapien_utils.get_obj_by_name(links, self.right_ee_link_name)
        self.palm_link = self.right_palm_link
        self.left_tip_links  = [sapien_utils.get_obj_by_name(links, n) for n in self.left_tip_link_names]
        self.right_tip_links = [sapien_utils.get_obj_by_name(links, n) for n in self.right_tip_link_names]
        self.tip_links = self.left_tip_links + self.right_tip_links
        self.tcp = self.right_palm_link
        if not self.disable_self_collisions:
            self._apply_self_collision_srdf(links)


        # Vendor-MJCF reflected rotor inertia (armature; the wuji MJCF declares
        # NO frictionloss, so armature is the only transmission property to
        # port). Frozen physical property (sim2real).
        apply_joint_physics(self.robot, self.hand_joint_names,
                            armature=WUJI_HAND_ARMATURE,
                            frictionloss=WUJI_HAND_FRICTIONLOSS,
                            max_velocity=HAND_JOINT_MAX_VELOCITY)
    def _apply_self_collision_srdf(self, links):
        """Disable the structurally-nested self-collision pairs the wuji URDF lacks an SRDF
        for. Links sharing a group-2 bit never collide; each 'clique' below gets one private
        bit, so exactly its members mutually ignore and nothing else changes. KEEPS ON:
        cross-finger pairs (different bits) and tip/link4 <-> palm (not in the base<->palm
        clique) so the fingertip stays blocked from the palm."""
        by = {lk.name: lk for lk in links}

        def set_clique(names, bit):
            for n in names:
                lk = by.get(n)
                if lk is not None:
                    lk.set_collision_group_bit(group=2, bit_idx=bit, bit=1)

        for side, base in self._SELF_COLLISION_BASE_BIT.items():
            p = f"{side}_"
            bit = base
            # arm-mount: hand base <-> arm flange (rigidly fixed, always adjacent)
            set_clique([p + "palm_link", p + "hand_docking_link", p + "panda_link7"], bit)
            bit += 1
            for f in range(1, 6):
                # within-finger f (incl. tip): non-adjacent segments of one finger
                set_clique(
                    [p + f"finger{f}_link{j}" for j in range(1, 5)] + [p + f"finger{f}_tip_link"],
                    bit,
                )
                bit += 1
                # finger BASE (link1..3, nested in the palm shell) <-> palm; link4/tip stay ON
                set_clique([p + "palm_link"] + [p + f"finger{f}_link{j}" for j in range(1, 4)], bit)
                bit += 1

    # Per-hand wrist cameras (palm link frame). Same initial pose for both hands;
    # tune per hand if needed.
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

    # Per-hand wrist cameras (palm link frame). Same initial pose for both hands;
    # tune per hand if needed.
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


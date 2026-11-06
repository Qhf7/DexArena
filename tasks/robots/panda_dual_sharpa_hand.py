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

from tasks.robots.hand_controller_gains import (
    SHARPA_HAND_ARMATURE,
    SHARPA_HAND_DAMPING,
    SHARPA_HAND_FORCE_LIMIT,
    SHARPA_HAND_FRICTIONLOSS,
    SHARPA_HAND_KP,
    apply_joint_physics,
    gains_to_list,
)
from tasks.config.contact_materials import apply_sharpa_robot_contact_materials


class PandaDualSharpaHand(PandaAgent):

    side_names = ("left", "right")

    single_arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    # Joint names without side prefix; matches the single-arm wrapper order.
    single_hand_joint_names = [
        "thumb_CMC_FE", "thumb_CMC_AA",
        "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
        "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
        "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
        "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
        "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
    ]
    single_tip_link_names = [
        "thumb_fingertip", "index_fingertip", "middle_fingertip",
        "ring_fingertip", "pinky_fingertip",
    ]
    single_ee_link_name = "hand_C_MC"

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

    # Hand-layer contact params, kept IDENTICAL to the single-arm sharpa
    # (tasks/robots/panda_sharpa_hand_right.py): pads 1.2/1.2 (the unified
    # engine value the object_materials pair table is derived against) and
    # REAL pad patch radii (5 mm pressed / 2 mm light touch).
    #
    # TWO historical defects fixed here (measured on the dual rig):
    # 1. The material list only named the `*_fingertip` links — but the load-
    #    bearing grasp surfaces are the ELASTOMER pads and DP shells (what the
    #    single agent's grip_link_names covers), so every actual contact
    #    surface silently ran on the 0.3/0.3 fallback with patch 0: a
    #    low-friction grip on a tapered bottle is a squeeze-launcher.
    # 2. The values were the pre-unification 2.0/1.0 + patch 0.1 (a 10 cm
    #    torsional lever arm that welds grasped objects to the fingers).
    single_grip_link_names = [
        "thumb_elastomer", "index_elastomer", "middle_elastomer",
        "ring_elastomer", "pinky_elastomer",
        "thumb_DP", "index_DP", "middle_DP", "ring_DP", "pinky_DP",
    ]
    grip_link_names = (
        [f"left_{n}" for n in single_grip_link_names]
        + [f"right_{n}" for n in single_grip_link_names]
    )
    urdf_config = dict(
        _materials=dict(
            fingertip=dict(static_friction=1.2, dynamic_friction=1.2, restitution=0.0)
        ),
        link={
            name: dict(material="fingertip", patch_radius=0.005, min_patch_radius=0.002)
            for name in grip_link_names
        },
    )

    # Self-collision ON, like the single-arm sharpa. The fingers close into a full fist fine
    # with it on (verified: fist |cmd-actual| ~0.07-0.10 rad). Keeping SC on avoids recording
    # physically-impossible (interpenetrating) finger poses into demos. The ONE spurious pair
    # that overlaps at the neutral pose — hand_C_MC <-> thumb_MC, PER SIDE — is filtered out
    # in _after_init (verified: removes the ~1.75cm rest drift, closure unaffected).
    disable_self_collisions = False

    # `hand_C_MC` and `thumb_MC` collision meshes interpenetrate at the neutral (zero) pose on
    # EACH hand; ignoring just these two pairs (a shared group-2 bit per pair) removes the
    # spurious rest contact and keeps every other self-collision on. Mirrors the single-arm.
    SELF_COLLISION_IGNORE_PAIRS = [
        ("right_hand_C_MC", "right_thumb_MC"),
        ("left_hand_C_MC", "left_thumb_MC"),
    ]
    _SELF_COLLISION_IGNORE_BASE_BIT = 28  # group-2 bits 28+ for our per-pair ignore masks

    # Arm gains follow ManiSkill upstream `panda` agent. Hand gains are
    # per-joint, sourced from `tasks.robots.hand_controller_gains`
    # (each gain's source is stated in that module).
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    # Joint impedance is the repo-default pd_joint_pos arm plant; the *_stiff
    # uid keeps the legacy stiff PD arm for pre-switch data and checkpoints.
    arm_impedance = True

    # 1.0 = unscaled vendor kp (history 25.0 -> 3.0 -> 1.0). Same
    # story as the single-arm PandaSharpaHandRight (see its SIM_HAND_KP_SCALE
    # comment): once the dropped transmission physics was restored — dry
    # friction + armature (applied in _after_init below) + the ball_grasping
    # solver bump — the vendor kp is faithfully realized and no compensation
    # factor is needed. force_limit stays the real hardware torque ceiling.
    # --sharpa-kp-scale is an EXTRA multiplier on top of this.
    SIM_HAND_KP_SCALE = 1.0

    @property
    def hand_stiffness(self):
        return [g * self.SIM_HAND_KP_SCALE
                for g in gains_to_list(self.hand_joint_names, SHARPA_HAND_KP)]

    @property
    def hand_damping(self):
        s = self.SIM_HAND_KP_SCALE ** 0.5
        return [g * s for g in gains_to_list(self.hand_joint_names, SHARPA_HAND_DAMPING)]

    @property
    def hand_force_limit(self):
        # REAL hardware force limits (unscaled) -> demos reproducible on the real hand.
        return gains_to_list(self.hand_joint_names, SHARPA_HAND_FORCE_LIMIT)

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
        return apply_sharpa_robot_contact_materials(
            self.robot, self.grip_link_names
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

        # Disable the neutral-pose overlapping pairs on BOTH hands: each pair shares its own
        # group-2 bit on both links, so exactly that pair stops colliding and nothing else
        # changes. Removes the ~1.75cm rest drift while keeping all functional self-collision.
        for _i, (_a, _b) in enumerate(self.SELF_COLLISION_IGNORE_PAIRS):
            _bit = self._SELF_COLLISION_IGNORE_BASE_BIT + _i
            for _n in (_a, _b):
                _lk = sapien_utils.get_obj_by_name(links, _n)
                if _lk is not None:
                    _lk.set_collision_group_bit(group=2, bit_idx=_bit, bit=1)

        # Vendor-MJCF joint transmission physics (frozen, sim2real), BOTH hands:
        # Coulomb dry friction + reflected rotor inertia — same as the
        # single-arm agent (see PandaSharpaHandRight._after_init).
        apply_joint_physics(self.robot, self.hand_joint_names,
                            frictionloss=SHARPA_HAND_FRICTIONLOSS,
                            armature=SHARPA_HAND_ARMATURE)

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


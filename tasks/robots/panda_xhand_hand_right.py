import os
import xml.etree.ElementTree as ET
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

from tasks.config.contact_materials import apply_xhand_robot_contact_materials
from tasks.robots.hand_controller_gains import (
    HAND_JOINT_MAX_VELOCITY,
    hand_urdf_config,
    xhand_marker_links,
    xhand_pad_links,
    XHAND_HAND_DAMPING,
    XHAND_HAND_FORCE_LIMIT,
    XHAND_HAND_KP,
    XHAND_HAND_ARMATURE,
    XHAND_HAND_FRICTIONLOSS,
    gains_to_list,
    apply_joint_physics,
)


def _resolve_xhand_urdf(base: Path) -> str:
    """The committed composite URDF, unless XHAND_MOUNT_RPY="r p y" (radians) is
    set — then write a sibling URDF with that mount orientation and use it. Lets us
    dial the palm-facing direction without editing files, e.g.
        XHAND_MOUNT_RPY="0 0 -1.57" exported before the robot is built;
    bake the chosen value as the default mount rpy once found.
    Rewrites only the FIXED panda_to_xhand_joint (zero added DOF). The sibling lives
    in the same dir so the relative ../xhand_hand_meshes paths still resolve."""
    rpy = os.environ.get("XHAND_MOUNT_RPY")
    if not rpy:
        return str(base.resolve())
    r, p, y = (float(v) for v in rpy.replace(",", " ").split())
    tree = ET.parse(base)
    for j in tree.getroot().findall("joint"):
        if j.get("name") == "panda_to_xhand_joint":
            j.find("origin").set("rpy", f"{r} {p} {y}")
    out = base.parent / "panda_xhand_hand_right__mount.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"[xhand] mount rpy override -> ({r}, {p}, {y}); using {out.name}")
    return str(out.resolve())


@register_agent()
class PandaXHandHandRight(PandaAgent):
    """Franka Panda + RobotEra XHand1 (right), following the sharpa/wuji
    "palm-from-mount, no wrist DOF" paradigm: the arm delivers the palm 6D pose
    through a single fixed mount joint, and the 12-DOF fully-decoupled action
    space is fingers only. ee_link = palm (``right_hand_link``).

    Gain provenance (see tasks.robots.hand_controller_gains): force_limit is
    vendor-grade (official URDF v1.3 effort, 1.1/0.4 N*m two-class); kp /
    armature / frictionloss come from the tsingqingyun/xhand1 MuJoCo
    reproduction and are a solid-but-unvalidated starting point pending a sim
    tuning pass.
    """

    uid = "panda_xhand_hand_right"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets" / "urdf" / "panda_xhand_hand_right.urdf"
    )
    urdf_path = _resolve_xhand_urdf(_urdf_path)

    arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    # URDF order: thumb(3), index(bend + 2), mid(2), ring(2), pinky(2).
    hand_joint_names = [
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    ]

    ee_link_name = "right_hand_link"  # palm / root

    tip_link_name = [
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    ]

    # Pads (1.2/1.2 + patch radii) on the DISTAL PHALANGES, which are what
    # actually presses on an object; the 5x3x3 mm ``*_tip`` markers keep the pad
    # material (retargeting target == pad) but lose their collision in
    # _after_init -- they are 3.5e-8 kg and protrude 1.8-2.6 mm past the real pad,
    # so every pinch used to make first contact through them. Everything else is
    # stated at the 0.35 shell value (four-layer contract L2, trap 1).
    urdf_config = hand_urdf_config(xhand_pad_links("right_"))

    # Default OFF (training / other callers keep it off). When a caller flips it ON
    # (e.g. the MANUS teleop's --self-collision), _after_init installs a hand-authored
    # SRDF-style exclusion table via group-2 "ignore" bits — the XHand URDF ships no
    # SRDF. MEASURED (rest pose, SC on, no exclusions): exactly ONE
    # pathological pair, palm <-> index_rota_link1, interpenetrating 9.27 mm with a
    # 1183 N*s impulse that drove the hand 1.66 rad / the palm 70 mm off the reset
    # pose. It is structural, not dynamic: the concave palm STL loads as ONE inflated
    # convex hull and swallows the index finger's SECOND segment (the engine only
    # auto-excludes DIRECT parent-child pairs, and index has an extra bend joint, so
    # rota_link1 is a GRANDCHILD). Everything else is clean: mid/ring/pinky link1 are
    # direct palm children, and the only other rest pair (palm <-> panda_link7, rigid
    # through the flange) sits at +0.20 mm with zero impulse. Cross-finger contact and
    # every distal-segment/tip <-> palm pair stay ON, so a full fist still cannot sink
    # a fingertip into its own palm.
    disable_self_collisions = True

    # Group-2 bits for the exclusion cliques (2 bits; clear of ManiSkill's reserved
    # 29 = disable_self_collisions and 30/31 = Fetch/robocasa base).
    _SELF_COLLISION_BASE_BIT = 0

    # Arm gains follow ManiSkill upstream `panda`; joint impedance is the
    # repo-default arm plant (see panda_wuji_hand_right).
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

    keyframes = dict(
        rest=Keyframe(
            qpos=np.concatenate([
                np.array([0.0, 0.0, 0.0, -np.pi * 2 / 3, 0.0, np.pi, np.pi / 4]),
                np.zeros(len(hand_joint_names)),
            ]),
            pose=sapien.Pose(),
        )
    )

    @property
    def _controller_configs(self):
        if self.arm_impedance:
            arm_pd_joint_pos = FrankaJointImpedanceControllerConfig(
                self.arm_joint_names,
                lower=None, upper=None,
            )
        else:
            arm_pd_joint_pos = PDJointPosControllerConfig(
                self.arm_joint_names,
                lower=None, upper=None,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                normalize_action=False,
            )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1, pos_upper=0.1,
            rot_lower=-0.1, rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        hand_pd_joint_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=None, upper=None,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            normalize_action=False,
        )
        return deepcopy_dict(dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                hand=hand_pd_joint_pos,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                hand=hand_pd_joint_pos,
            ),
        ))

    # Per-hand wrist camera pose (palm link frame). INITIAL value — tune for the
    # XHand palm geometry once the mount orientation is finalized.
    wrist_camera_pose = sapien.Pose(
        p=[0.02, -0.06, 0.05],
        q=[0.6424, 0.2955, -0.2955, 0.6424],
    )

    def _apply_frozen_contact_materials(self):
        return apply_xhand_robot_contact_materials(
            self.robot, xhand_pad_links("right_")
        )

    def _after_init(self):
        links = self.robot.get_links()
        self.palm_link = sapien_utils.get_obj_by_name(links, self.ee_link_name)
        self.tip_links = [
            sapien_utils.get_obj_by_name(links, name) for name in self.tip_link_name
        ]
        self.tcp = self.palm_link
        if not self.disable_self_collisions:
            self._apply_self_collision_srdf(links, prefix="right_", arm_link="panda_link7",
                                            base_bit=self._SELF_COLLISION_BASE_BIT)

        # Reflected rotor inertia + transmission dry friction. Frozen physical
        # properties (sim2real). XHand values are QDD-defensible MJCF-conversion
        # defaults pending hardware identification.
        apply_joint_physics(self.robot, self.hand_joint_names,
                            frictionloss=XHAND_HAND_FRICTIONLOSS,
                            armature=XHAND_HAND_ARMATURE,
                            max_velocity=HAND_JOINT_MAX_VELOCITY,
                            no_collision_links=xhand_marker_links("right_"))

    @staticmethod
    def _apply_self_collision_srdf(links, prefix, arm_link, base_bit):
        """Ignore exactly the two structurally-adjacent XHand pairs (see the
        disable_self_collisions comment for the measurement). Links sharing a
        group-2 bit never collide; each clique gets its own private bit, so
        nothing else changes."""
        by = {lk.name: lk for lk in links}

        def clique(names, bit):
            for n in names:
                lk = by.get(n)
                if lk is not None:
                    lk.set_collision_group_bit(group=2, bit_idx=bit, bit=1)

        palm = f"{prefix}hand_link"
        # arm-mount: palm <-> arm flange (rigid through link8, can never separate)
        clique([palm, arm_link], base_bit)
        # index finger's second segment is nested INSIDE the palm's convex hull
        clique([palm, f"{prefix}hand_index_rota_link1"], base_bit + 1)

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                "wrist_camera",
                self.wrist_camera_pose,
                320,
                240,
                np.deg2rad(58),
                0.01,
                100,
                mount=self.palm_link,
            )
        ]

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., : len(self.arm_joint_names)]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.palm_link.pose.p


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


@register_agent()
class PandaSharpaHandRight(PandaAgent):
    uid = "panda_sharpa_hand_right"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets" / "urdf" / "panda_sharpa_hand_right.urdf"
    )
    urdf_path = str(_urdf_path.resolve())

    arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    hand_joint_names = [
        "right_thumb_CMC_FE", "right_thumb_CMC_AA",
        "right_thumb_MCP_FE", "right_thumb_MCP_AA",
        "right_thumb_IP",
        "right_index_MCP_FE", "right_index_MCP_AA",
        "right_index_PIP", "right_index_DIP",
        "right_middle_MCP_FE", "right_middle_MCP_AA",
        "right_middle_PIP", "right_middle_DIP",
        "right_ring_MCP_FE", "right_ring_MCP_AA",
        "right_ring_PIP", "right_ring_DIP",
        "right_pinky_CMC", "right_pinky_MCP_FE",
        "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
    ]

    ee_link_name = "right_hand_C_MC"

    tip_link_name = [
        "right_thumb_fingertip",
        "right_index_fingertip",
        "right_middle_fingertip",
        "right_ring_fingertip",
        "right_pinky_fingertip",
    ]

    # Links that ACTUALLY carry the fingertip collision geometry. The `*_fingertip`
    # frames in `tip_link_name` are collision-LESS reference sites (used for
    # retargeting/IK), so applying the grippy material there did nothing -- the real
    # contact surfaces (the rubber pads `*_elastomer` + the distal phalanges `*_DP`)
    # were left at the URDF default ~0.3 friction, which made the hand slippery and
    # unable to hold small objects. Target the pads/DP so the 2.0/1.0 fingertip
    # material actually bites (this fixes the "object slips out of the grasp" issue
    # across all sharpa tasks).
    grip_link_names = [
        "right_thumb_elastomer", "right_index_elastomer", "right_middle_elastomer",
        "right_ring_elastomer", "right_pinky_elastomer",
        "right_thumb_DP", "right_index_DP", "right_middle_DP",
        "right_ring_DP", "right_pinky_DP",
    ]

    urdf_config = dict(
        _materials=dict(
            fingertip=dict(static_friction=1.2, dynamic_friction=1.2, restitution=0.0)
        ),
        link={
            # patch_radius / min_patch_radius are the PhysX TORSIONAL friction lever
            # arm -- how hard it is to spin an object about the contact normal. The
            # old 0.1 (10 cm) invented an enormous anti-twist moment that welded
            # objects between the fingers. These are real soft-pad contact patch
            # sizes instead: 5 mm pressed, 2 mm on light touch.
            #
            # Friction 1.2/1.2 is the pad baseline shared by all three hands. Every
            # hand's fingertips are silicone-class, so the real pair friction is
            # essentially the same (silicone on a smooth surface, CoF ~1.0), and a
            # shared engine value makes the object friction table hand-independent
            # (mu_obj = 2*mu_real - 1.2; the combine rule was measured as average).
            # Static = dynamic until an inclined-plane measurement separates them.
            # Same value is duplicated in the compliant-pad path below and in
            # bottle_cap_unscrew's elastomer override; dual variants follow.
            name: dict(material="fingertip", patch_radius=0.005, min_patch_radius=0.002)
            for name in grip_link_names
        },
    )

    # Self-collision ON: the fingers can't interpenetrate the object/each other, so
    # recorded grasps are physically valid / real-robot-transferable. The ONLY spurious
    # rest-pose overlap (palm `right_hand_C_MC` <-> thumb metacarpal `right_thumb_MC`,
    # a convex-hull artifact) is filtered out per-pair in _after_init.
    disable_self_collisions = False

    # Arm gains follow ManiSkill upstream `panda` agent. Hand gains are
    # per-joint, sourced from `tasks.robots.hand_controller_gains`
    # (see docs/hand_controller_parameters.md).
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    # Joint impedance is the repo-default pd_joint_pos arm plant; the *_stiff
    # uid keeps the legacy stiff PD arm for pre-switch data and checkpoints.
    arm_impedance = True

    # Hand gains are the UNSCALED vendor MJCF values, baked so teleop-recording AND
    # training use IDENTICAL physics. FORCE LIMIT is the REAL hardware value: the sim
    # can never exert more joint torque than the hardware, so any recorded grip is
    # reproducible on the real hand.
    #
    # KP SCALE = 1.0. History: a black-box 25.0, then a calibrated 3.0,
    # finally 1.0 once ALL the real transmission physics the original port dropped
    # was restored. What 25x had been silently compensating, measured piece by piece:
    #   (a) joint dry friction (SHARPA_HAND_FRICTIONLOSS) — without it light contact
    #       backdrives the fingers open;
    #   (b) solver under-realization — the old 100Hz/15-1 TGS config realized only
    #       ~0.3x of the commanded stiffness in contact (ball_grasping now 200Hz/16-4);
    #   (c) armature (SHARPA_HAND_ARMATURE) — without the reflected rotor inertia the
    #       armature-derived kd left the joints ~4-5x OVERdamped and the drive
    #       constraint poorly conditioned in contact.
    # With (a)+(b)+(c) in place, vendor kp x1 measures: 100ms step settle, 1.1deg
    # mean sine lag @20Hz teleop, fist-squeeze 22N at onset+10% closure (>= the
    # converged high-res reference) — faithful stiffness, no compensation factor.
    # Every number in the hand plant is now a vendor/real value.
    SIM_HAND_KP_SCALE = 1.0

    # COMPLIANT (soft) fingertip pads — vendor-aligned. The elastomer pads are
    # near-massless (vendor URDF: mass 1E-4 kg / ~0 inertia). The vendor MuJoCo
    # model makes their CONTACT soft (`solref="0.06 0.9"`, a compliant pad), which
    # is why a ~0.1 g pad is stable there. Our port had dropped that and used a
    # HARD contact: a near-massless pad hitting a RIGID/IMMOVABLE object is ejected
    # (the depenetration impulse divided by ~0 mass, Δv=J/m, blows up) — this is the
    # "fingers get flung off when gripping a fixed object" symptom. PhysX-5 compliant
    # contacts are exposed in SAPIEN via a NEGATIVE restitution (magnitude = contact
    # spring stiffness); applied to the elastomer pads only (the DP phalanges keep
    # the hard material — they carry real mass, 4.55 g). Calibrated to ~halve the
    # contact-velocity spike while STILL blocking (no pass-through). Set 0.0 = hard
    # (revert). NOTE: this is a SHARED-agent change — it softens fingertip contact
    # for every sharpa task (ball_grasping etc.), toward the vendor/real behaviour;
    # re-verify those grips if it ever regresses.
    # DEFAULT HARD (0.0). The "fingers flung off" symptom turned out to be a
    # 90deg-angled-mount SELF-COLLISION (hand wrist/palm penetrating panda_link7,
    # ~2340 impulse at rest), NOT the fingertip contact — fixed by the mount<->
    # flange ignore pair on PandaSharpaHandRightAngled. So we keep the fingertip
    # contact HARD: a soft (negative-restitution) pad let the near-massless pad
    # PENETRATE a rigid object and then get geometrically TRAPPED (fingers stuck
    # in the cap; friction/stiffness-independent — measured). Hard = no penetration
    # = no trap. A negative value re-enables the compliant pad if ever needed.
    SIM_ELASTOMER_RESTITUTION = 0.0

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
        # REAL hardware force limits (unscaled) -> demos are reproducible on the real hand.
        return gains_to_list(self.hand_joint_names, SHARPA_HAND_FORCE_LIMIT)

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
    # Sharpa palm geometry if needed.
    wrist_camera_pose = sapien.Pose(
        p=[0.02, -0.06, 0.05],
        q=[0.6424, 0.2955, -0.2955, 0.6424],
    )

    # Pairs of links whose collision hulls overlap at REST (a few joints apart, same base
    # structure) and would otherwise spuriously push apart when self-collision is ON.
    # Measured: at the open/rest pose only `right_hand_C_MC` <-> `right_thumb_MC` carries a
    # non-zero contact impulse (~1.1); that constant force transmits through the fixed wrist
    # to the redundant 7-DOF arm and drives it in a deterministic circle. Disabling just
    # this pair (a shared group-2 "ignore" bit) lets self-collision stay ON for valid grasp
    # demos (fingertip-fingertip, fingers-vs-object all preserved) without the arm drift.
    SELF_COLLISION_IGNORE_PAIRS = [("right_hand_C_MC", "right_thumb_MC")]
    _SELF_COLLISION_IGNORE_BASE_BIT = 28  # group-2 bits 28+ for our per-pair ignore masks

    def _apply_frozen_contact_materials(self):
        return apply_sharpa_robot_contact_materials(
            self.robot, self.grip_link_names
        )

    def _after_init(self):
        links = self.robot.get_links()
        self.palm_link = sapien_utils.get_obj_by_name(links, self.ee_link_name)
        self.tip_links = [
            sapien_utils.get_obj_by_name(links, name) for name in self.tip_link_name
        ]
        self.tcp = self.palm_link

        # Surgically disable the spuriously-overlapping link pairs (see above). Each pair
        # gets its OWN group-2 bit set on BOTH its links, so exactly that pair stops
        # colliding and nothing else is affected. Harmless when self-collision is disabled.
        for _i, (_a, _b) in enumerate(self.SELF_COLLISION_IGNORE_PAIRS):
            _bit = self._SELF_COLLISION_IGNORE_BASE_BIT + _i
            for _n in (_a, _b):
                _lk = sapien_utils.get_obj_by_name(links, _n)
                if _lk is not None:
                    _lk.set_collision_group_bit(group=2, bit_idx=_bit, bit=1)

        # Vendor-MJCF joint transmission physics (frozen, sim2real): Coulomb
        # dry friction (resists backdrive — the physics the original gain port
        # dropped) + reflected rotor inertia (armature; also makes the
        # armature-derived kd consistent, restoring the design damping ratio).
        apply_joint_physics(self.robot, self.hand_joint_names,
                            frictionloss=SHARPA_HAND_FRICTIONLOSS,
                            armature=SHARPA_HAND_ARMATURE)

        # Vendor-aligned compliant fingertip pads (see SIM_ELASTOMER_RESTITUTION).
        # A negative restitution turns the elastomer contact into a soft spring, so
        # the near-massless pad is cushioned instead of ejected on rigid contact.
        if self.SIM_ELASTOMER_RESTITUTION < 0.0:
            soft = sapien.physx.PhysxMaterial(1.2, 1.2,
                                              float(self.SIM_ELASTOMER_RESTITUTION))
            for _art in self.robot._objs:
                for _lk in _art.get_links():
                    if _lk.get_name().endswith("elastomer"):
                        for _s in _lk.get_collision_shapes():
                            _s.set_physical_material(soft)

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



# ---------------------------------------------------------------------------
# 90-degree angled-mount variant (REAL two-segment elbow bracket)
# ---------------------------------------------------------------------------
# Motivation: with the stock straight mount the palm/finger axis is COLINEAR
# with the Franka flange +Z. So when the arm reset points the flange down the
# fingers point straight down, and getting "fingers forward / palm down" near
# the table demands a ~90deg wrist bend that slams j6 into its limit and stalls
# the differential-IK config branch (see docs / wrist-pitchup notes). Baking a
# 90deg bend into the mount makes "fingers forward, palm down" the wrist ZERO
# pose, re-centering the reachable wrist-rotation range on the useful posture.
#
# The rotation was measured (not guessed): at the rest keyframe the fingers lie
# along flange +Z and the palm normal is the palm-frame +X axis, horizontal in
# the flange frame. Rotating +90deg about the knuckle axis sends the fingers to
# a flange-horizontal direction and the palm normal to flange +Z (= world -Z =
# palm-down when the flange points down). Composed with the stock 135deg yaw the
# full parent->child mount rotation is rpy = (-1.4228, -1.5545, -2.4223).
_ANGLED_MOUNT_RPY = "-1.4228 -1.5545 -2.4223"

# --- Two-segment elbow bracket (replaces the zero-length prototype) --------
# The zero-length elbow was physically impossible: the hand's mount hub reaches
# ~21-25 mm BACK past the flange plane, so at xyz=0 it penetrated panda_link7 by
# 20.8 mm (wrist) / 14 mm (C_MC) — masked in sim by ignore bits, unbuildable in
# real life. The real bracket is two Ø40 mm barrels at 90°:
#   segment 1: axial, flange face -> elbow corner, length _ELBOW_L1
#   segment 2: horizontal, elbow corner -> hand mount face, length _ELBOW_L2
# Sizes were solved against the measured collision hulls:
#   L1 = 32 mm -> hand clears panda_link7 by +11.2 mm
#   L2 = 28 mm -> hand's back hub clears the segment-1 barrel by +8.1 mm
#   (segment-2 direction is fixed by the hand's mount-face normal; total hand
#    offset |t| = 42.5 mm — compact, and the ignore bits are GONE: collision
#    between the hand chain and panda_link7 is honest again.)
# The real machined bracket MUST reproduce: both lengths, the Ø40 barrels, the
# mount rpy above, and (once built) its true mass in _ELBOW_MASS.
_ELBOW_L1 = 0.032        # m, flange face -> segment-2 axis (along link8 +z)
_ELBOW_L2 = 0.028        # m, flange axis -> hand mount face (along _ELBOW_DW)
_ELBOW_DW = (0.7626, -0.6468, 0.0)  # segment-2 dir in link8 frame = mount
                                    # rotation applied to the hand's forward
                                    # axis, z-projected + normalized (true 90°)
_ELBOW_R = 0.020         # m, barrel radius (Ø40)
_ELBOW_MASS = 0.15       # kg, hollow-aluminium estimate — weigh & match the
                         # real bracket when it exists (it adds arm payload)

# Hand mount origin in the bracket/link8 frame (bracket->hand joint xyz).
_ELBOW_XYZ = (_ELBOW_L2 * _ELBOW_DW[0], _ELBOW_L2 * _ELBOW_DW[1], _ELBOW_L1)


def _generate_elbow_assets():
    """Generate the elbow visual + collision meshes (cached; link8 frame) and
    return (mesh_dir_relpath, inertial_dict). Visual = plates + barrels +
    corner sphere concatenated; collision = TWO convex hulls (one per segment,
    solid). A 1 mm air gap is left under the flange face so the bracket never
    parks a permanent zero-impulse contact against panda_link7's collar."""
    import trimesh
    urdf_dir = PandaSharpaHandRight._urdf_path.parent
    mdir = urdf_dir / "angled_mount"
    mdir.mkdir(exist_ok=True)
    vis_f, colA_f, colB_f = mdir / "elbow_visual.obj", mdir / "elbow_col_a.obj", mdir / "elbow_col_b.obj"
    dW = np.array(_ELBOW_DW)
    L1, L2, r = _ELBOW_L1, _ELBOW_L2, _ELBOW_R

    def _cyl(radius, z0, z1, axis):
        c = trimesh.creation.cylinder(radius=radius, height=z1 - z0, sections=48)
        c.apply_translation([0, 0, (z0 + z1) / 2])
        if axis is not None:                       # rotate +z onto `axis`
            c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], axis))
        return c

    if not (vis_f.exists() and colA_f.exists() and colB_f.exists()):
        plate1 = _cyl(0.030, 0.001, 0.006, None)               # Ø60 flange plate (1 mm gap)
        seg1 = _cyl(r, 0.004, L1, None)                        # vertical barrel
        corner = trimesh.creation.icosphere(radius=r, subdivisions=2)
        corner.apply_translation([0, 0, L1])                   # rounded elbow
        seg2 = _cyl(r, 0.0, L2 - 0.004, dW)                    # horizontal barrel
        seg2.apply_translation([0, 0, L1])
        plate2 = _cyl(0.023, L2 - 0.004, L2, dW)               # Ø46 hand plate
        plate2.apply_translation([0, 0, L1])
        trimesh.util.concatenate([plate1, seg1, corner, seg2, plate2]).export(str(vis_f))
        trimesh.Trimesh(vertices=np.vstack([plate1.vertices, seg1.vertices])
                        ).convex_hull.export(str(colA_f))
        trimesh.Trimesh(vertices=np.vstack([corner.vertices, seg2.vertices, plate2.vertices])
                        ).convex_hull.export(str(colB_f))
    # Inertial estimate: full convex fill of the bracket, scaled to _ELBOW_MASS.
    hull = trimesh.Trimesh(vertices=np.vstack([
        trimesh.load(str(f), process=False, force="mesh").vertices
        for f in (colA_f, colB_f)])).convex_hull
    hull.density = _ELBOW_MASS / hull.volume
    com = hull.center_mass
    I = hull.moment_inertia                                    # about COM, link frame
    inertial = dict(mass=_ELBOW_MASS, com=com,
                    ixx=I[0, 0], iyy=I[1, 1], izz=I[2, 2],
                    ixy=I[0, 1], ixz=I[0, 2], iyz=I[1, 2])
    return "angled_mount", inertial


def _generate_angled_urdf(base=None, out_name="panda_sharpa_hand_right_angled.urdf") -> str:
    """Derive an angled-mount URDF from ``base`` by replacing the single
    `panda_to_sharpa_hand_joint` with: a fixed `angle_bracket` link (visual +
    2 solid convex collisions + realistic inertial) bolted to panda_link8, and
    the hand fixed on the bracket at the elbow offset with the SAME mount
    rotation. BOTH joints are `fixed` — the bracket adds ZERO degrees of
    freedom. Single source of truth = the source URDF; regenerated on import.

    ``base`` defaults to the plain right-hand URDF; the mobile-base composite
    passes its own URDF (same mount-joint block, so the substitution is
    identical) to produce the angled x mobile combination."""
    rel, ine = _generate_elbow_assets()
    if base is None:
        base = PandaSharpaHandRight._urdf_path
    out = base.with_name(out_name)
    text = base.read_text()
    marker = '<joint name="panda_to_sharpa_hand_joint" type="fixed">'
    i = text.index(marker)
    j = text.index("</joint>", i) + len("</joint>")
    block = f"""<link name="angle_bracket">
    <inertial>
      <origin rpy="0 0 0" xyz="{ine['com'][0]:.5f} {ine['com'][1]:.5f} {ine['com'][2]:.5f}"/>
      <mass value="{ine['mass']:.3f}"/>
      <inertia ixx="{ine['ixx']:.3e}" iyy="{ine['iyy']:.3e}" izz="{ine['izz']:.3e}" ixy="{ine['ixy']:.3e}" ixz="{ine['ixz']:.3e}" iyz="{ine['iyz']:.3e}"/>
    </inertial>
    <visual><geometry><mesh filename="{rel}/elbow_visual.obj"/></geometry></visual>
    <collision><geometry><mesh filename="{rel}/elbow_col_a.obj"/></geometry></collision>
    <collision><geometry><mesh filename="{rel}/elbow_col_b.obj"/></geometry></collision>
  </link>
  <joint name="panda_to_bracket_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="panda_link8"/>
    <child link="angle_bracket"/>
  </joint>
  <joint name="panda_to_sharpa_hand_joint" type="fixed">
    <origin rpy="{_ANGLED_MOUNT_RPY}" xyz="{_ELBOW_XYZ[0]:.5f} {_ELBOW_XYZ[1]:.5f} {_ELBOW_XYZ[2]:.5f}"/>
    <parent link="angle_bracket"/>
    <child link="right_hand_wrist"/>
  </joint>"""
    text = text[:i] + block + text[j:]
    if not out.exists() or out.read_text() != text:
        out.write_text(text)
    return str(out.resolve())


@register_agent()
class PandaSharpaHandRightAngled(PandaSharpaHandRight):
    """Sharpa right hand on a 90deg angled flange mount (zero-length elbow).

    Identical DOF / joints / meshes / gains as `PandaSharpaHandRight`, so the
    sharpa retargeter and every downstream hand-frame consumer are unaffected;
    only the fixed flange->hand transform differs. Select in teleop with
    `--hand-model sharpa --robot-uids panda_sharpa_hand_right_angled`.
    """

    uid = "panda_sharpa_hand_right_angled"
    urdf_path = _generate_angled_urdf()

    # NO extra self-collision ignore pairs. The zero-length prototype needed
    # (wrist/C_MC <-> panda_link7) ignore bits because the hand hub physically
    # overlapped the flange collar (unbuildable). The real two-segment bracket
    # moves the hand +11.2 mm CLEAR of panda_link7, so hand-chain<->arm collision
    # is honest again (only the base palm<->thumb_MC pair remains). Keep the base
    # bit LOW anyway: bit 29 is ManiSkill's `disable_self_collisions` bit that
    # scene articulations (e.g. the bottle) also use — never let a pair land there.
    _SELF_COLLISION_IGNORE_BASE_BIT = 24

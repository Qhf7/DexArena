"""Dual upright Panda + Sharpa manipulators mounted on mobile-base."""

from pathlib import Path

import numpy as np
import sapien
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.registration import register_agent

from tasks.robots.panda_dual_sharpa_hand import (
    PandaDualSharpaHand,
)
from tasks.robots.panda_sharpa_hand_right import (
    _ANGLED_MOUNT_RPY,
    _ELBOW_XYZ,
    _generate_elbow_assets,
)


def _generate_dual_angled_urdf(base: Path, out_name: str) -> str:
    """Dual-arm version of the single agent's ``_generate_angled_urdf``: both
    ``{side}_panda_to_sharpa_hand_joint`` mounts are replaced by the SAME
    two-segment elbow bracket (identical hardware part on both flanges — the
    left hand is a mirrored-mesh hand, but the bracket itself is symmetric
    about its own segments, so one part number serves both arms)."""
    rel, ine = _generate_elbow_assets()
    out = base.with_name(out_name)
    text = base.read_text()
    for side in ("left", "right"):
        marker = f'<joint name="{side}_panda_to_sharpa_hand_joint" type="fixed">'
        i = text.index(marker)
        j = text.index("</joint>", i) + len("</joint>")
        block = f"""<link name="{side}_angle_bracket">
    <inertial>
      <origin rpy="0 0 0" xyz="{ine['com'][0]:.5f} {ine['com'][1]:.5f} {ine['com'][2]:.5f}"/>
      <mass value="{ine['mass']:.3f}"/>
      <inertia ixx="{ine['ixx']:.3e}" iyy="{ine['iyy']:.3e}" izz="{ine['izz']:.3e}" ixy="{ine['ixy']:.3e}" ixz="{ine['ixz']:.3e}" iyz="{ine['iyz']:.3e}"/>
    </inertial>
    <visual><geometry><mesh filename="{rel}/elbow_visual.obj"/></geometry></visual>
    <collision><geometry><mesh filename="{rel}/elbow_col_a.obj"/></geometry></collision>
    <collision><geometry><mesh filename="{rel}/elbow_col_b.obj"/></geometry></collision>
  </link>
  <joint name="{side}_panda_to_bracket_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="{side}_panda_link8"/>
    <child link="{side}_angle_bracket"/>
  </joint>
  <joint name="{side}_panda_to_sharpa_hand_joint" type="fixed">
    <origin rpy="{_ANGLED_MOUNT_RPY}" xyz="{_ELBOW_XYZ[0]:.5f} {_ELBOW_XYZ[1]:.5f} {_ELBOW_XYZ[2]:.5f}"/>
    <parent link="{side}_angle_bracket"/>
    <child link="{side}_hand_wrist"/>
  </joint>"""
        text = text[:i] + block + text[j:]
    if not out.exists() or out.read_text() != text:
        out.write_text(text)
    return str(out.resolve())


@register_agent()
class PandaDualSharpaHandMobileBase(PandaDualSharpaHand):
    """Stationary dual-arm robot whose root frame is the base's top center.

    The arm/hand joint names, controllers, keyframes, wrist cameras, and TCP
    helpers intentionally match :class:`PandaDualSharpaHand`.
    Only the mounting URDF changes, so tasks can switch robot UID without
    changing their action or observation layout.

    """

    uid = "panda_dual_sharpa_hand_mobile_base"
    # Keep the dual-arm root footprint identical to the left/right single-arm
    # mobile-base agents. Tasks use this scalar when positioning the shared
    # base relative to the table; spelling it out avoids silently depending on
    # their getattr(..., 1.0) compatibility fallback.
    mobile_base_horizontal_scale = 1.0
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_dual_sharpa_hand_mobile_base.urdf"
    )
    urdf_path = str(_urdf_path.resolve())



@register_agent()
class PandaDualSharpaHandAngledMobileBase(PandaDualSharpaHandMobileBase):
    """Both arms carry the 90-degree elbow bracket of the single angled agent
    (same DOF/joint names/gains as the straight dual — only the two fixed
    flange->hand transforms change, so retargeters and action layouts are
    unaffected). Built for two-hand tasks in the palm-down ergonomic zone,
    e.g. CabinetPourCapTurn: one hand braces the bottle, the other twists —
    the screw drive's thread friction makes single-handed cap turning
    physically co-rotate the bottle, exactly like reality."""

    uid = "panda_dual_sharpa_hand_angled_mobile_base"
    urdf_path = _generate_dual_angled_urdf(
        PandaDualSharpaHandMobileBase._urdf_path,
        "panda_dual_sharpa_hand_angled_mobile_base.urdf",
    )
    # The straight dual's rest is the ball-task posture: pre-curled fingers
    # (throw-catch cupping) that FOLD INTO the new bracket barrels (measured
    # 10-142 N of rest-pose self-contact), and a horizontal-stagger arm pose.
    # The angled agents' zero is the single angled rest — arms at the single
    # keyframe, hands FLAT (all zeros, like the single agent) — on BOTH arms.
    single_arm_joint_reset_pos = np.array(
        [0.0, 0.0, 0.0, -np.pi * 2 / 3, 0.0, np.pi, np.pi / 4]
    )
    single_hand_joint_reset_pos = np.zeros_like(
        PandaDualSharpaHand.single_hand_joint_reset_pos
    )
    keyframes = dict(
        rest=Keyframe(
            qpos=np.concatenate([
                single_arm_joint_reset_pos,
                single_hand_joint_reset_pos,
                single_arm_joint_reset_pos,
                single_hand_joint_reset_pos,
            ]),
            pose=sapien.Pose(),
        )
    )


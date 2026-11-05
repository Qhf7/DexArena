"""Upright Panda + LEFT Sharpa hand on the full-size mobile base.

Arm + left-chirality Sharpa hand, mounted on the dual robot's full-size trolley at
``xyz="0 0.22 0"`` -- i.e. left of the trolley centre, where the dual robot's left arm
sits. Dropped at the same root pose as ``panda_dual_sharpa_hand_mobile_base``, this arm is
kinematically identical joint-for-joint to that robot's left arm, so a single-arm subtask
composes into the long-horizon task without re-deriving reach. The ``*_right_mobile_base``
agent is the same robot with the right hand at ``0 -0.22 0``. The URDF is a committed asset
derived from ``panda_dual_sharpa_hand_mobile_base.urdf`` (right arm pruned, arm chain
de-prefixed); if that dual URDF changes, this one has to be brought back in step by hand.

Only the hand links carry the side prefix; the arm does not, exactly as in
``panda_sharpa_hand_right.urdf``. That keeps the controller groups (``arm``/``hand``), the
action layout and the 29-dim DOF count identical to the right agent.

Also defines the angled (90-degree elbow bracket) variant, mirroring
``panda_sharpa_hand_right_mobile_base.py``.
"""

from pathlib import Path

from mani_skill.agents.registration import register_agent

from tasks.robots.panda_sharpa_hand_right import (
    _ANGLED_MOUNT_RPY,
    _ELBOW_XYZ,
    PandaSharpaHandRight,
    _generate_elbow_assets,
)


def _generate_angled_left_urdf(base: Path, out_name: str) -> str:
    """Left-hand version of the right agent's ``_generate_angled_urdf``.

    Same substitution: the single ``panda_to_sharpa_hand_joint`` is replaced by a fixed
    ``angle_bracket`` link bolted to ``panda_link8`` plus the hand fixed on the bracket at
    the elbow offset with the SAME mount rotation. BOTH joints are ``fixed`` — the bracket
    adds ZERO degrees of freedom. Only the mount's child link differs
    (``left_hand_wrist``), since that is the one name in the block that carries a side.

    Written out here rather than reaching into the right agent's generator so the two
    robots stay independent -- same reason ``_generate_dual_angled_urdf`` is its own copy
    in ``panda_dual_sharpa_hand_mobile_base.py``. What IS shared is the bracket *part*
    (``_generate_elbow_assets`` / ``_ELBOW_XYZ`` / ``_ANGLED_MOUNT_RPY``): it is one
    symmetric piece of hardware, identical on either flange, so it stays single-source.
    """
    rel, ine = _generate_elbow_assets()
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
    <child link="left_hand_wrist"/>
  </joint>"""
    text = text[:i] + block + text[j:]
    if not out.exists() or out.read_text() != text:
        out.write_text(text)
    return str(out.resolve())


@register_agent()
class PandaSharpaHandLeftMobileBase(PandaSharpaHandRight):
    """Single LEFT Sharpa arm mounted where the dual robot's left arm sits."""

    uid = "panda_sharpa_hand_left_mobile_base"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_sharpa_hand_left_mobile_base.urdf"
    )
    urdf_path = str(_urdf_path.resolve())

    # Hand-side link/joint names carry `left_` instead of `right_`. Gains resolve
    # regardless (hand_controller_gains.gains_to_list strips the side prefix), so only
    # the name tables below need overriding -- controllers, keyframes, sensors, TCP
    # helpers and _after_init all come straight from the arm+hand base agent.
    #
    # Spelled out rather than derived from the right agent's tables: this robot's joint
    # order is its own contract (keyframes and recorded actions are consumed
    # positionally), and it should not silently follow an edit made for a different
    # robot. Order matches panda_sharpa_hand_right.py's table entry for entry.
    hand_joint_names = [
        "left_thumb_CMC_FE", "left_thumb_CMC_AA",
        "left_thumb_MCP_FE", "left_thumb_MCP_AA",
        "left_thumb_IP",
        "left_index_MCP_FE", "left_index_MCP_AA",
        "left_index_PIP", "left_index_DIP",
        "left_middle_MCP_FE", "left_middle_MCP_AA",
        "left_middle_PIP", "left_middle_DIP",
        "left_ring_MCP_FE", "left_ring_MCP_AA",
        "left_ring_PIP", "left_ring_DIP",
        "left_pinky_CMC", "left_pinky_MCP_FE",
        "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP",
    ]

    ee_link_name = "left_hand_C_MC"

    tip_link_name = [
        "left_thumb_fingertip",
        "left_index_fingertip",
        "left_middle_fingertip",
        "left_ring_fingertip",
        "left_pinky_fingertip",
    ]

    # The real contact surfaces (rubber pads + distal phalanges), not the collision-less
    # `*_fingertip` reference frames -- see panda_sharpa_hand_right.py for why.
    grip_link_names = [
        "left_thumb_elastomer", "left_index_elastomer", "left_middle_elastomer",
        "left_ring_elastomer", "left_pinky_elastomer",
        "left_thumb_DP", "left_index_DP", "left_middle_DP",
        "left_ring_DP", "left_pinky_DP",
    ]

    # Re-declared (not inherited): the parent builds this dict from ITS grip_link_names at
    # class-definition time, so it would otherwise apply the fingertip material to the
    # right hand's link names, which do not exist in this URDF. Values are the parent's --
    # see panda_sharpa_hand_right.py for why patch_radius is 5 mm / 2 mm.
    urdf_config = dict(
        _materials=dict(
            fingertip=dict(static_friction=1.2, dynamic_friction=1.2, restitution=0.0)
        ),
        link={
            name: dict(material="fingertip", patch_radius=0.005, min_patch_radius=0.002)
            for name in grip_link_names
        },
    )

    # Same convex-hull artifact as the right hand, on the left links. _after_init reads
    # this attribute, so no method override is needed.
    SELF_COLLISION_IGNORE_PAIRS = [("left_hand_C_MC", "left_thumb_MC")]

    # Mirrors the geometry documented in tasks/assets/README.md; task authors can
    # reference these to position the robot without measuring the URDF.
    mobile_base_horizontal_scale = 1.0
    mobile_base_height = 0.91964257
    mobile_base_arm_offset_y = 0.22



@register_agent()
class PandaSharpaHandLeftAngledMobileBase(PandaSharpaHandLeftMobileBase):
    """90-degree elbow mount on the full-size mobile base, left arm.

    Composition of the two variants: the trolley + upright arm come from the
    mobile-base composite URDF, and the flange->hand mount is replaced by the same
    two-segment elbow bracket the right angled agents use. "Fingers forward,
    palm down" is the wrist ZERO pose
    here too. Mirror of PandaSharpaHandRightAngledMobileBase; the bracket is one
    symmetric part, so only the mount's child link differs.
    """

    uid = "panda_sharpa_hand_left_angled_mobile_base"
    urdf_path = _generate_angled_left_urdf(
        base=PandaSharpaHandLeftMobileBase._urdf_path,
        out_name="panda_sharpa_hand_left_angled_mobile_base.urdf",
    )

    # NO extra self-collision ignore pairs beyond the inherited palm<->thumb_MC one: the
    # two-segment bracket moves the hand +11.2 mm CLEAR of panda_link7, so hand-chain
    # <-> arm collision is honest. Base bit stays LOW (24) -- bit 29 is ManiSkill's
    # `disable_self_collisions` bit that scene articulations also use. Same values as the
    # right angled agent; see panda_sharpa_hand_right.py for the measurement.
    _SELF_COLLISION_IGNORE_BASE_BIT = 24

    mobile_base_horizontal_scale = 1.0
    mobile_base_height = 0.91964257
    mobile_base_arm_offset_y = 0.22


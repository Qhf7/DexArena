"""Upright Panda + RIGHT RobotEra XHand1 on the full-size mobile base.

Arm + right XHand1, mounted on the dual robot's full-size trolley at
``xyz="0 -0.22 0"`` -- right of the trolley centre, where the dual robot's right arm
sits. At a given root pose this arm is kinematically identical, joint for joint, to
``panda_dual_xhand_hand_mobile_base``'s right arm (pinned by
``tests/test_xhand_mobile_base_rig.py``). The ``*_left_mobile_base`` agent is the same
robot with the mirrored left hand at ``0 0.22 0``.

The URDF is a GENERATED asset (``scripts/build_xhand_mobile_base_urdfs.py``): the
trolley + arm rig is copied verbatim from the frozen dual-wuji mobile URDF and only the
hand is swapped, so this rig is the same operating station every other family stands
against (four-layer contract, L3).

Everything else -- controllers, keyframes, gains, pad materials, self-collision
cliques, wrist camera -- is inherited unchanged from ``PandaXHandHandRight``; the link
and joint names are identical (arm chain unprefixed, hand chain ``right_``).
"""

from pathlib import Path

from mani_skill.agents.registration import register_agent

from tasks.robots.panda_xhand_hand_right import PandaXHandHandRight


@register_agent()
class PandaXHandHandRightMobileBase(PandaXHandHandRight):
    """Single RIGHT XHand1 arm mounted where the dual robot's right arm sits."""

    uid = "panda_xhand_hand_right_mobile_base"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_xhand_hand_right_mobile_base.urdf"
    )
    urdf_path = str(_urdf_path.resolve())

    # Mirrors the geometry documented in tasks/assets/README.md; task authors
    # can reference these to position the robot without measuring the URDF.
    mobile_base_horizontal_scale = 1.0
    mobile_base_height = 0.91964257
    mobile_base_arm_offset_y = -0.22



# ---------------------------------------------------------------------------
# 90-degree angled mount (project rule: every hand end-effector rides the elbow
# bracket). Same DOF / joint names / gains / keyframes as the straight agent --
# only the fixed flange->hand transform changes (tasks/robots/angled_mount.py).
# ---------------------------------------------------------------------------
from tasks.robots.angled_mount import generate_angled_urdf as _generate_angled_urdf_generic  # noqa: E402


@register_agent()
class PandaXHandHandRightAngledMobileBase(PandaXHandHandRightMobileBase):
    uid = "panda_xhand_hand_right_angled_mobile_base"
    urdf_path = _generate_angled_urdf_generic(
        PandaXHandHandRightMobileBase._urdf_path, "panda_xhand_hand_right_angled_mobile_base.urdf", "xhand", [("right", "")],
    )
    _SELF_COLLISION_IGNORE_BASE_BIT = getattr(PandaXHandHandRightMobileBase, "_SELF_COLLISION_IGNORE_BASE_BIT", 24)

"""Upright Panda + RIGHT Wuji hand on the full-size mobile base.

Arm + right-chirality Wuji hand, mounted on the dual robot's full-size trolley at
``xyz="0 -0.22 0"`` -- i.e. right of the trolley centre, where the dual robot's right arm
sits. At a given root pose this arm is kinematically identical, joint for joint, to
``panda_dual_wuji_hand_mobile_base``'s right arm. The ``*_left_mobile_base`` agent is the
same robot with the left hand at ``0 0.22 0``. The URDF is a committed asset derived from
``panda_dual_wuji_hand_mobile_base.urdf`` (left arm pruned, arm chain de-prefixed); if that
dual URDF changes, this one has to be brought back in step by hand.

CHANGED (was: compact 0.75-footprint trolley, arm centred at y = 0). Any layout
tuned against the old geometry has to be re-derived: the trolley is 10.2 cm longer in x
and the shoulder moved 22 cm in -y. In particular a base pose picked for the compact
trolley now leaves too little table clearance, so cabinet_pour picks the base pose by
trolley footprint instead (COMPACT_TROLLEY_BASE_POSE vs FULL_TROLLEY_BASE_POSE, keyed on
``mobile_base_horizontal_scale``).
"""

from pathlib import Path

from mani_skill.agents.registration import register_agent

from tasks.robots.panda_wuji_hand_right import PandaWujiHandRight


@register_agent()
class PandaWujiHandRightMobileBase(PandaWujiHandRight):
    """Single RIGHT Wuji arm mounted where the dual robot's right arm sits."""

    uid = "panda_wuji_hand_right_mobile_base"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_wuji_hand_right_mobile_base.urdf"
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
class PandaWujiHandRightAngledMobileBase(PandaWujiHandRightMobileBase):
    uid = "panda_wuji_hand_right_angled_mobile_base"
    urdf_path = _generate_angled_urdf_generic(
        PandaWujiHandRightMobileBase._urdf_path, "panda_wuji_hand_right_angled_mobile_base.urdf", "wuji", [("right", "")],
    )
    _SELF_COLLISION_IGNORE_BASE_BIT = getattr(PandaWujiHandRightMobileBase, "_SELF_COLLISION_IGNORE_BASE_BIT", 24)

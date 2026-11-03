"""Dual upright Panda + RobotEra XHand1 manipulators mounted on the mobile base.

Joint names, controllers, keyframes, wrist cameras and TCP helpers match
:class:`PandaDualXHandHand`; only the mounting URDF changes, which
preserves the action and observation layouts across the two variants -- the same
relationship ``panda_dual_wuji_hand_mobile_base`` has to its trapezoid sibling.

The URDF is GENERATED (``scripts/build_xhand_mobile_base_urdfs.py``): the trolley +
two upright Panda arms are copied verbatim from the frozen dual-wuji mobile URDF,
so this is the same operating station every other family stands against.
"""

from pathlib import Path

from mani_skill.agents.registration import register_agent

from tasks.robots.panda_dual_xhand_hand import (
    PandaDualXHandHand,
)


@register_agent()
class PandaDualXHandHandMobileBase(PandaDualXHandHand):
    """Stationary dual-arm XHand1 robot whose root frame is the trolley's top centre."""

    uid = "panda_dual_xhand_hand_mobile_base"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_dual_xhand_hand_mobile_base.urdf"
    )
    urdf_path = str(_urdf_path.resolve())



# ---------------------------------------------------------------------------
# 90-degree angled mount (project rule: every hand end-effector rides the elbow
# bracket). Same DOF / joint names / gains / keyframes as the straight agent --
# only the fixed flange->hand transform changes (tasks/robots/angled_mount.py).
# ---------------------------------------------------------------------------
from tasks.robots.angled_mount import generate_angled_urdf as _generate_angled_urdf_generic  # noqa: E402


@register_agent()
class PandaDualXHandHandAngledMobileBase(PandaDualXHandHandMobileBase):
    uid = "panda_dual_xhand_hand_angled_mobile_base"
    urdf_path = _generate_angled_urdf_generic(
        PandaDualXHandHandMobileBase._urdf_path, "panda_dual_xhand_hand_angled_mobile_base.urdf", "xhand", [("left", "left_"), ("right", "right_")],
    )
    _SELF_COLLISION_IGNORE_BASE_BIT = getattr(PandaDualXHandHandMobileBase, "_SELF_COLLISION_IGNORE_BASE_BIT", 24)

"""Dual upright Panda + Wuji manipulators mounted on mobile-base."""

from pathlib import Path

from mani_skill.agents.registration import register_agent

from tasks.robots.panda_dual_wuji_hand import (
    PandaDualWujiHand,
)


@register_agent()
class PandaDualWujiHandMobileBase(PandaDualWujiHand):
    """Stationary dual-arm robot whose root frame is the base's top center.

    Joint names, controllers, keyframes, wrist cameras, and TCP helpers match
    :class:`PandaDualWujiHand`. Only the mounting URDF changes,
    preserving the action and observation layouts across the two variants.

    """

    uid = "panda_dual_wuji_hand_mobile_base"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_dual_wuji_hand_mobile_base.urdf"
    )
    urdf_path = str(_urdf_path.resolve())



# ---------------------------------------------------------------------------
# 90-degree angled mount (project rule: every hand end-effector rides the elbow
# bracket). Same DOF / joint names / gains / keyframes as the straight agent --
# only the fixed flange->hand transform changes (tasks/robots/angled_mount.py).
# ---------------------------------------------------------------------------
from tasks.robots.angled_mount import generate_angled_urdf as _generate_angled_urdf_generic  # noqa: E402


@register_agent()
class PandaDualWujiHandAngledMobileBase(PandaDualWujiHandMobileBase):
    uid = "panda_dual_wuji_hand_angled_mobile_base"
    urdf_path = _generate_angled_urdf_generic(
        PandaDualWujiHandMobileBase._urdf_path, "panda_dual_wuji_hand_angled_mobile_base.urdf", "wuji", [("left", "left_"), ("right", "right_")],
    )
    _SELF_COLLISION_IGNORE_BASE_BIT = getattr(PandaDualWujiHandMobileBase, "_SELF_COLLISION_IGNORE_BASE_BIT", 24)

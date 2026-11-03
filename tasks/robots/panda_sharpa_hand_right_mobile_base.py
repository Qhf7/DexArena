"""Upright Panda + RIGHT Sharpa hand on the full-size mobile base.

Arm + right-chirality Sharpa hand, mounted on the dual robot's full-size trolley at
``xyz="0 -0.22 0"`` -- i.e. right of the trolley centre, where the dual robot's right arm
sits. At a given root pose this arm is kinematically identical, joint for joint, to
``panda_dual_sharpa_hand_mobile_base``'s right arm. The ``*_left_mobile_base`` agent is
the same robot with the left hand at ``0 0.22 0``. The URDF is a committed asset derived
from ``panda_dual_sharpa_hand_mobile_base.urdf`` (left arm pruned, arm chain de-prefixed);
if that dual URDF changes, this one has to be brought back in step by hand.

CHANGED (was: compact 0.75-footprint trolley, arm centred at y = 0). Any layout
tuned against the old geometry has to be re-derived: the trolley is 10.2 cm longer in x
and the shoulder moved 22 cm in -y. In particular a base pose picked for the compact
trolley now leaves too little table clearance, so cabinet_pour picks the base pose by
trolley footprint instead (COMPACT_TROLLEY_BASE_POSE vs FULL_TROLLEY_BASE_POSE, keyed on
``mobile_base_horizontal_scale``). The angled variant below inherits the new base
automatically (it splices its elbow bracket onto this URDF).
"""

from pathlib import Path

from mani_skill.agents.registration import register_agent

from tasks.robots.panda_sharpa_hand_right import (
    PandaSharpaHandRight,
    PandaSharpaHandRightAngled,
    _generate_angled_urdf,
)


@register_agent()
class PandaSharpaHandRightMobileBase(PandaSharpaHandRight):
    """Single RIGHT Sharpa arm mounted where the dual robot's right arm sits."""

    uid = "panda_sharpa_hand_right_mobile_base"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_sharpa_hand_right_mobile_base.urdf"
    )
    urdf_path = str(_urdf_path.resolve())

    # Mirrors the geometry documented in tasks/assets/README.md; task authors
    # can reference these to position the robot without measuring the URDF.
    mobile_base_horizontal_scale = 1.0
    mobile_base_height = 0.91964257
    mobile_base_arm_offset_y = -0.22



@register_agent()
class PandaSharpaHandRightAngledMobileBase(PandaSharpaHandRightAngled):
    """90-degree elbow mount on the compact mobile base.

    Composition of the two variants: the trolley + upright arm come from the
    mobile-base composite URDF, and the flange->hand mount is replaced by the
    same two-segment elbow bracket the table-mounted angled agent uses (see
    PandaSharpaHandRightAngled / docs/angled_mount_design.md). "Fingers
    forward, palm down" is the wrist ZERO pose here too — teleop wrist
    ergonomics match the angled table agent, not the straight mobile mount.
    Inherits the angled agent's honest self-collision config (no wrist<->link7
    ignore bits; base bit 24)."""

    uid = "panda_sharpa_hand_right_angled_mobile_base"
    urdf_path = _generate_angled_urdf(
        base=PandaSharpaHandRightMobileBase._urdf_path,
        out_name="panda_sharpa_hand_right_angled_mobile_base.urdf",
    )
    mobile_base_horizontal_scale = 1.0
    mobile_base_height = 0.91964257
    mobile_base_arm_offset_y = -0.22


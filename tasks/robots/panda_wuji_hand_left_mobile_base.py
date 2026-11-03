"""Upright Panda + LEFT Wuji hand on the full-size mobile base.

Arm + left-chirality Wuji hand, mounted on the dual robot's full-size trolley at
``xyz="0 0.22 0"`` -- i.e. left of the trolley centre, where the dual robot's left arm
sits. Dropped at the same root pose as ``panda_dual_wuji_hand_mobile_base``, this arm is
kinematically identical joint-for-joint to that robot's left arm, so a single-arm subtask
composes into the long-horizon task without re-deriving reach. The ``*_right_mobile_base``
agent is the same robot with the right hand at ``0 -0.22 0``. The URDF is a committed asset
derived from ``panda_dual_wuji_hand_mobile_base.urdf`` (right arm pruned, arm chain
de-prefixed); if that dual URDF changes, this one has to be brought back in step by hand.

Only the hand links carry the side prefix; the arm and its docking flange do not, exactly
as in ``panda_wuji_hand_right.urdf``. That keeps the controller groups (``arm``/``hand``),
the action layout and the 27-dim DOF count identical to the right agent.
"""

from pathlib import Path

from mani_skill.agents.registration import register_agent

from tasks.config.contact_materials import apply_wuji_robot_contact_materials
from tasks.robots.hand_controller_gains import (
    hand_urdf_config,
    wuji_pad_links,
)
from tasks.robots.panda_wuji_hand_right import PandaWujiHandRight


@register_agent()
class PandaWujiHandLeftMobileBase(PandaWujiHandRight):
    """Single LEFT Wuji arm mounted where the dual robot's left arm sits."""

    uid = "panda_wuji_hand_left_mobile_base"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_wuji_hand_left_mobile_base.urdf"
    )
    urdf_path = str(_urdf_path.resolve())

    # Hand-side link/joint names carry `left_` instead of `right_`. Gains resolve
    # regardless (hand_controller_gains.gains_to_list strips the side prefix), so only
    # the name tables below need overriding -- controllers, keyframes, sensors, TCP
    # helpers and _after_init all come straight from the arm+hand base agent.
    hand_joint_names = [
        f"left_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)
    ]

    ee_link_name = "left_palm_link"

    tip_link_name = [f"left_finger{f}_tip_link" for f in range(1, 6)]

    def _apply_frozen_contact_materials(self):
        return apply_wuji_robot_contact_materials(
            self.robot, wuji_pad_links("left_")
        )

    # Re-declared (not inherited): the parent builds this dict from ITS tip_link_name at
    # class-definition time, so it would otherwise apply the fingertip material to the
    # right hand's link names, which do not exist in this URDF.
    urdf_config = hand_urdf_config(wuji_pad_links("left_"))

    # Mirrors the geometry documented in tasks/assets/README.md; task authors can
    # reference these to position the robot without measuring the URDF.
    mobile_base_horizontal_scale = 1.0
    mobile_base_height = 0.91964257
    mobile_base_arm_offset_y = 0.22

    def _apply_self_collision_srdf(self, links):
        """Left-hand copy of the right agent's clique filter (same design, `left_` names).

        Deliberately duplicated rather than sharing a parameterised body with
        PandaWujiHandRight: these are two independent robots and neither should be able
        to break the other. Keep the two in sync by hand if the clique layout changes.

        Disables the structurally-nested self-collision pairs the wuji URDF lacks an SRDF
        for. Links sharing a group-2 bit never collide; each 'clique' below gets one
        private bit, so exactly its members mutually ignore and nothing else changes.
        KEEPS ON: cross-finger pairs (different bits) and tip/link4 <-> palm (not in the
        base<->palm clique) so the fingertip stays blocked from the palm. Note the
        arm-mount links (hand_docking_link, panda_link7) are UNPREFIXED in the single-arm
        URDF -- only the hand links carry `left_`, so the mount clique mixes prefixed and
        unprefixed names.
        """
        by = {lk.name: lk for lk in links}

        def set_clique(names, bit):
            for n in names:
                lk = by.get(n)
                if lk is not None:
                    lk.set_collision_group_bit(group=2, bit_idx=bit, bit=1)

        bit = self._SELF_COLLISION_BASE_BIT
        # arm-mount: hand base <-> arm flange (rigidly fixed, always adjacent)
        set_clique(["left_palm_link", "hand_docking_link", "panda_link7"], bit)
        bit += 1
        for f in range(1, 6):
            # within-finger f (incl. tip): non-adjacent segments of one finger
            set_clique(
                [f"left_finger{f}_link{j}" for j in range(1, 5)]
                + [f"left_finger{f}_tip_link"],
                bit,
            )
            bit += 1
            # finger BASE (link1..3, nested in the palm shell) <-> palm; link4/tip stay ON
            set_clique(
                ["left_palm_link"] + [f"left_finger{f}_link{j}" for j in range(1, 4)],
                bit,
            )
            bit += 1



# ---------------------------------------------------------------------------
# 90-degree angled mount (project rule: every hand end-effector rides the elbow
# bracket). Same DOF / joint names / gains / keyframes as the straight agent --
# only the fixed flange->hand transform changes (tasks/robots/angled_mount.py).
# ---------------------------------------------------------------------------
from tasks.robots.angled_mount import generate_angled_urdf as _generate_angled_urdf_generic  # noqa: E402


@register_agent()
class PandaWujiHandLeftAngledMobileBase(PandaWujiHandLeftMobileBase):
    uid = "panda_wuji_hand_left_angled_mobile_base"
    urdf_path = _generate_angled_urdf_generic(
        PandaWujiHandLeftMobileBase._urdf_path, "panda_wuji_hand_left_angled_mobile_base.urdf", "wuji", [("left", "")],
    )
    _SELF_COLLISION_IGNORE_BASE_BIT = getattr(PandaWujiHandLeftMobileBase, "_SELF_COLLISION_IGNORE_BASE_BIT", 24)

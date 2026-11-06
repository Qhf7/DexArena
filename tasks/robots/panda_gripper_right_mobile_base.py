"""Upright Panda with the stock Franka gripper as the RIGHT arm of the full-size trolley."""

from pathlib import Path

from mani_skill.agents.registration import register_agent

from tasks.robots.panda_impedance import PandaImpedanceDefault


@register_agent()
class PandaGripperRightMobileBase(PandaImpedanceDefault):
    """Single stock-gripper Panda at the dual rig's right mount (y = -0.22).

    The URDF is derived from ``panda_dual_gripper_mobile_base.urdf`` by
    ``scripts/build_gripper_mobile_base_urdfs.py``: same trolley, same arm chain, so this
    arm is the dual robot's right arm, joint for joint. Joint names, controller modes,
    gripper mimic action, TCP and grasp helpers are the stock ``panda`` ones.
    """

    uid = "panda_gripper_right_mobile_base"
    _urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "panda_gripper_right_mobile_base.urdf"
    )
    urdf_path = str(_urdf_path.resolve())

    mobile_base_horizontal_scale = 1.0
    mobile_base_height = 0.91964257
    mobile_base_arm_offset_y = -0.22

    def _after_init(self):
        super()._after_init()
        # Same convention as the dual gripper agent: the two finger links double as the
        # fingertip links the tasks read.
        self.finger_links = [self.finger1_link, self.finger2_link]
        self.tip_links = list(self.finger_links)
        self.palm_link = self.tcp

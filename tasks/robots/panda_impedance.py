"""Panda gripper agents backed by the Franka joint-impedance arm controller.

The controller itself lives in ``franka_joint_impedance_controller.py`` (shared
with the dex-hand wrappers); see that module's docstring for the control law and
the three GPU/stability/gravity fixes. Importing this module transparently
replaces the upstream ``panda`` agent's ``pd_joint_pos`` arm with joint
impedance. Every other control mode (``pd_ee_delta_pose``, ...) keeps the stock
stiff PD arm.

Action semantics are identical to ``panda`` pd_joint_pos (absolute joint targets
in radians, ``normalize_action=False``), so demos and checkpoints stay valid.
"""

from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam

from tasks.robots.panda_base import PandaStructureContactMixin

from tasks.robots.franka_joint_impedance_controller import (  # noqa: F401  (re-export)
    FRANKA_ARM_FORCE_LIMIT,
    FRANKA_FR3_DEPLOYMENT_KD,
    FRANKA_FR3_DEPLOYMENT_KP,
    FrankaJointImpedanceController,
    FrankaJointImpedanceControllerConfig,
)

# Real Franka Hand continuous grasping force (N). The stock ManiSkill gripper
# ships force_limit=100 N, which exceeds what the real hand can sustain
# (70 N continuous, 140 N short-peak) — demos recorded with it can squeeze in
# ways the real gripper cannot reproduce.
FRANKA_HAND_FORCE_LIMIT = 70.0

# Gripper PD gains, adopted from the official IsaacLab Franka config
# (isaaclab_assets/robots/franka.py, FRANKA_PANDA_CFG "panda_hand" actuator:
# stiffness=2000, damping=100). The real hand is a closed position servo that
# exposes no stiffness/damping, so there is no vendor value to port; IsaacLab's
# is the community de-facto standard AND structurally identical to ours
# (per-finger prismatic joint, implicit PD drive), so the numbers transfer
# directly. A welcome side effect of kp=2000: the theoretical force ceiling
# becomes kp * 0.04 m stroke = 80 N > 70 N, so the real-hardware force clamp
# above actually engages (with the old kp=1000 the ceiling was 40 N and the
# clamp was dead code). Known residual gap: under PD position control the grip
# force still scales with object width, whereas the real hand's grasp(force)
# commands it directly — fixing that needs a force-commanded gripper
# controller, tracked separately.
FRANKA_GRIPPER_STIFFNESS = 2000.0
FRANKA_GRIPPER_DAMPING = 100.0

# Real Franka Hand travel speed: 50 mm/s per finger (official product manual /
# datasheet). Neither ManiSkill nor SAPIEN applies the URDF's <limit velocity>
# to the PhysX joint (it reads back as the 100 m/s engine default), so without
# this cap the PD drive closes the fingers at ~0.8 m/s — a full 40 mm close in
# ~0.05 s versus ~0.8 s on hardware, which skews the grasp timing of every
# recorded demo. Applied per finger joint in _after_init below. (The URDF's own
# velocity="0.2" is 4x the datasheet value and is ignored on purpose.)
FRANKA_GRIPPER_MAX_VELOCITY = 0.05  # m/s


def _apply_gripper_joint_limits(robot):
    """Cap both finger joints at the real hand's travel speed."""
    for joint in robot.get_active_joints():
        if joint.name in ("panda_finger_joint1", "panda_finger_joint2"):
            for pj in joint._objs:
                pj.set_max_velocity(FRANKA_GRIPPER_MAX_VELOCITY)

# Gripper finger contact parameters, aligned with the shared dex-hand baseline:
# the stock fingertips are silicone-class pads, so they use the same 1.2/1.2
# pad friction as sharpa/wuji — this keeps tasks/config/object_materials.py
# hand-INDEPENDENT (effective mu = (1.2 + mu_obj)/2 for every end-effector).
# patch_radius drops from the stock 0.1 (a fake 10 cm anti-twist lever that
# welded grasped objects in place) to real contact-patch scale: the finger pad
# is ~17 mm wide, so 10 mm pressed / 5 mm light touch.
_GRIPPER_URDF_CONFIG = dict(
    _materials=dict(
        gripper=dict(static_friction=1.2, dynamic_friction=1.2, restitution=0.0)
    ),
    link=dict(
        panda_leftfinger=dict(
            material="gripper", patch_radius=0.01, min_patch_radius=0.005
        ),
        panda_rightfinger=dict(
            material="gripper", patch_radius=0.01, min_patch_radius=0.005
        ),
    ),
)


def _clamp_gripper_to_franka_hand(cfgs):
    """Apply the real-hand force ceiling and the IsaacLab PD gains to every
    control mode's gripper config."""
    for mode_cfg in cfgs.values():
        gripper = mode_cfg.get("gripper") if isinstance(mode_cfg, dict) else None
        if gripper is None:
            continue
        if hasattr(gripper, "force_limit"):
            gripper.force_limit = FRANKA_HAND_FORCE_LIMIT
        if hasattr(gripper, "stiffness"):
            gripper.stiffness = FRANKA_GRIPPER_STIFFNESS
        if hasattr(gripper, "damping"):
            gripper.damping = FRANKA_GRIPPER_DAMPING
    return cfgs


@register_agent(override=True)
class PandaImpedanceDefault(PandaStructureContactMixin, Panda):
    """Repo-default ``panda`` with an impedance pd_joint_pos arm."""

    uid = "panda"
    urdf_config = _GRIPPER_URDF_CONFIG

    @property
    def _controller_configs(self):
        cfgs = super()._controller_configs
        arm_cfg = cfgs["pd_joint_pos"]["arm"]
        # Keep the stock *stiff* absolute-position arm available under a separate
        # key (main's escape hatch for pre-impedance data/checkpoints).
        cfgs["pd_joint_pos_stiff"] = dict(
            arm=arm_cfg,
            gripper=cfgs["pd_joint_pos"]["gripper"],
        )
        cfgs["pd_joint_pos"]["arm"] = FrankaJointImpedanceControllerConfig(
            joint_names=arm_cfg.joint_names,
            lower=None,
            upper=None,
            gravity_compensation=True,
        )
        # BaseAgent.set_control_mode disables link gravity whenever
        # balance_passive_force (default True) is set.
        cfgs["pd_joint_pos"]["balance_passive_force"] = True
        return _clamp_gripper_to_franka_hand(cfgs)

    def _after_init(self):
        super()._after_init()
        _apply_gripper_joint_limits(self.robot)


@register_agent(override=True)
class PandaWristCamImpedanceDefault(PandaStructureContactMixin, PandaWristCam):
    """Wrist-camera ``panda_wristcam`` with the same joint-impedance arm.

    Only the ``*Vision-v1`` env variants use this agent; the state Open/Close
    track (incl. box-open) uses ``PandaImpedanceDefault`` (uid ``panda``). It is
    defined here purely so the shared ``tasks.articulated_open_close`` module
    imports cleanly. Additive local shim mirroring the upstream benchmark branch;
    ``PandaImpedanceDefault`` above is left untouched.
    """

    uid = "panda_wristcam"
    urdf_config = _GRIPPER_URDF_CONFIG

    @property
    def _controller_configs(self):
        cfgs = super()._controller_configs
        arm_cfg = cfgs["pd_joint_pos"]["arm"]
        cfgs["pd_joint_pos"]["arm"] = FrankaJointImpedanceControllerConfig(
            joint_names=arm_cfg.joint_names,
            lower=None,
            upper=None,
            gravity_compensation=True,
        )
        # BaseAgent.set_control_mode disables link gravity whenever
        # balance_passive_force (default True) is set.
        cfgs["pd_joint_pos"]["balance_passive_force"] = True
        return _clamp_gripper_to_franka_hand(cfgs)

    def _after_init(self):
        super()._after_init()
        _apply_gripper_joint_limits(self.robot)

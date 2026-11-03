"""Subtask S1 of the pipette task, single LEFT hand: take the parked pipette,
dispense into the test tube standing in the rack, put the pipette back.

Replaces the deleted dual-arm dispense env (left hand lifted the tube,
right hand pressed): the operator asked for a one-hand version where the tube is never
picked up. Success = plunger pressed >= 50 % with the tip inside the tube mouth, plunger
released, pipette back in the stand (up to 35 deg tilt) with the hand away. This is
also the long task's own second stage (the evaluator has no other
dispense rule), so this module only masks the cup and pins the rig.

Every object keeps the LONG TASK's own position -- this subtask overrides nothing, so it
is a strict geometric subset of the long task (operator note). The shared stand
sits at y=-0.17, 0.820 m from the left arm root and 0.733 m from the right.
"""
import torch
from tasks.config.hands import register_task
from mani_skill.utils.structs.pose import Pose
from tasks.pipette_transfer.pipette_transfer_long import PipetteSubtaskEnv, LEFT_ROBOT_UID, LEFT_ROBOT_UIDS

SUBTASK_ENV_ID = "PipetteTransferDispenseLiquid-Sharpa-v1"

@register_task(
    "PipetteTransferDispenseLiquid",
    side="left",
    max_episode_steps=1100,
    obs_mode="state",
    control_mode="pd_joint_pos",
    reward_mode="normalized_dense",
    render_mode="rgb_array",
)
class PipetteTransferDispenseLiquidEnv(PipetteSubtaskEnv):
    """Left hand only: grasp the parked pipette, press into the racked tube, park it again."""
    SUPPORTED_ROBOTS = list(LEFT_ROBOT_UIDS)
    TARGET_STAGE = 1
    REQUIRE_FINAL_PLACEMENT = True     # success = stage 3 (pressed in the tube AND parked)
    LOAD_CUP = False
    LOAD_PIPETTE = True
    LOAD_TUBE = True
    LOAD_RACK = True
    LOAD_PIPETTE_STAND = True

    def __init__(self, *a, **kw):
        kw.setdefault("robot_uids", LEFT_ROBOT_UID)
        super().__init__(*a, **kw)



if __name__ == "__main__":
    from tasks.pipette_transfer.pipette_transfer_long import run_cli
    run_cli(SUBTASK_ENV_ID, "videos/pipette_dispense")

"""Subtask S0 of the pipette task: aspirate in the sample cup, return the pipette.

User decision: S0 is a one-hand job, so it is registered on the single
RIGHT-arm trolley rig (`panda_sharpa_hand_right_angled_mobile_base`, dof 29) instead of
the dual model. The right arm sits where the dual build's right arm sits (trolley x
-0.870, y -0.220), so the cup / pipette / stand layout, the predicates and the
thresholds are the long task's, untouched. Wuji / XHand right-hand ids are registered
alongside, like every other single-hand subtask in the repo.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tasks.config.hands import register_task
from tasks.pipette_transfer.pipette_transfer_long import PipetteSubtaskEnv, RIGHT_ROBOT_UID, RIGHT_ROBOT_UIDS

SUBTASK_ENV_ID = "PipetteTransferAspirateLiquid-Sharpa-v1"

MAX_EPISODE_STEPS = 800


@register_task(
    "PipetteTransferAspirateLiquid",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode="pd_joint_pos",
    reward_mode="normalized_dense",
    render_mode="rgb_array",
)
class PipetteTransferAspirateLiquidEnv(PipetteSubtaskEnv):
    """Right hand only: grasp the parked pipette, aspirate in the cup, return it to the stand."""
    SUPPORTED_ROBOTS = list(RIGHT_ROBOT_UIDS)
    TARGET_STAGE = 0
    LOAD_CUP = True
    LOAD_PIPETTE = True
    LOAD_TUBE = False
    LOAD_RACK = False
    LOAD_PIPETTE_STAND = True

    def __init__(self, *a, **kw):
        kw.setdefault("robot_uids", RIGHT_ROBOT_UID)   # never fall back to the dual model
        super().__init__(*a, **kw)


if __name__ == "__main__":
    from tasks.pipette_transfer.pipette_transfer_long import run_cli
    run_cli(SUBTASK_ENV_ID,"videos/pipette_aspirate")

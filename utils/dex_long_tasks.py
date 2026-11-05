"""The ten long tasks as ordered stages, each stage tied to the subtask that owns it.

WHY THIS FILE
    The long-horizon experiment runs a long task and hands control to one subtask policy per
    stage. That needs three things the subtask registry does not have: the ORDER of the stages,
    the signal in `info` that says a stage is finished, and WHICH ARM the stage uses -- the
    subtask policies are 29-dim single-arm while nine of the ten long tasks run the 58-dim
    dual-arm robot, so every action has to be routed into one arm's slots.

WHERE THE CONTENT COMES FROM
    The stage order and the gate names were read out of the long task modules themselves and
    confirmed against a live probe of all ten environments: five tasks expose an integer `stage`, the rest latch booleans named `stage_*`.
    Screwing's order is bolt -> nut -> turn (the reset order of its latches), not the order the
    subtasks happen to be listed in.
"""
from __future__ import annotations

from dataclasses import dataclass

LEFT, RIGHT, DUAL = "left", "right", "dual"

#: Joint layout of panda_dual_sharpa_hand_angled_mobile_base, in controller order.
DUAL_SLOTS = {
    LEFT: {"arm": (0, 7), "hand": (14, 36)},
    RIGHT: {"arm": (7, 14), "hand": (36, 58)},
}
DUAL_ACTION_DIM = 58
SINGLE_ACTION_DIM = 29


@dataclass(frozen=True)
class LongStage:
    subtask: str          # env id of the subtask that owns this stage
    arm: str              # which arm acts: LEFT, RIGHT or DUAL (the stage drives both)
    gate_key: str         # key in `info` that reports this stage
    gate_value: float = 1.0   # `stage`-style gates compare >=, boolean gates compare truthy
    gate_kind: str = "flag"   # "flag" or "stage"


@dataclass(frozen=True)
class LongTask:
    env_id: str
    family: str
    robot_uids: str
    stages: tuple[LongStage, ...]

    @property
    def is_dual(self) -> bool:
        return "dual" in self.robot_uids

    @property
    def action_dim(self) -> int:
        return DUAL_ACTION_DIM if self.is_dual else SINGLE_ACTION_DIM


_DUAL = "panda_dual_sharpa_hand_angled_mobile_base"
_RIGHT_ARM = "panda_sharpa_hand_right_angled_mobile_base"

LONG_TASKS: tuple[LongTask, ...] = (
    LongTask("CabinetPouringLong-Sharpa-v1", "cabinet", _DUAL, (
        LongStage("CabinetPouringOpenCabinet-Sharpa-v1", LEFT, "stage_door_open"),
        LongStage("CabinetPouringPickBottle-Sharpa-v1", RIGHT, "stage_bottle_placed"),
        LongStage("CabinetPouringUnscrewCap-Sharpa-v1", RIGHT, "stage_cap_placed"),
        LongStage("CabinetPouringPourBeads-Sharpa-v1", RIGHT, "stage_poured"),
    )),
    LongTask("DrawerPlugLong-Sharpa-v1", "drawer", _DUAL, (
        LongStage("DrawerPlugOpenDrawer-Sharpa-v1", LEFT, "stage_drawer_open"),
        LongStage("DrawerPlugPickCharger-Sharpa-v1", RIGHT, "stage_charger_placed"),
        LongStage("DrawerPlugInsertPlug-Sharpa-v1", RIGHT, "stage_charger_inserted"),
    )),
    LongTask("FurnitureAssemblyLong-Sharpa-v1", "furniture", _DUAL, (
        LongStage("FurnitureAssemblyLayFlatPanel-Sharpa-v1", LEFT, "stage", 1, "stage"),
        LongStage("FurnitureAssemblyInsertRail-Sharpa-v1", RIGHT, "stage", 2, "stage"),
        LongStage("FurnitureAssemblyFastenScrew-Sharpa-v1", RIGHT, "stage", 3, "stage"),
    )),
    LongTask("MealPreparationLong-Sharpa-v1", "meal", _DUAL, (
        LongStage("MealPreparationOpenLunchBox-Sharpa-v1", LEFT, "stage_box_open"),
        LongStage("MealPreparationPlaceBread-Sharpa-v1", RIGHT, "stage_bread_done"),
        LongStage("MealPreparationScoopTomato-Sharpa-v1", RIGHT, "stage_ball_done"),
    )),
    LongTask("OfficeDeskLong-Sharpa-v1", "office", _DUAL, (
        # office's stage counter spends one value PER KEY: STAGE_TYPE_FIRST=1 through
        # len(TARGET_KEYS) are the keyboard stage, STAGE_CLICK_MOUSE=1+len(TARGET_KEYS), and
        # STAGE_DONE one past it (tasks/office_desk/office_desk_long.py). The first pilot gated the keyboard at >=2 and
        # passed it 3/3 after ONE key, on a subtask that scores 32% in-domain -- caught that way.
        LongStage("OfficeDeskOpenLaptop-Sharpa-v1", RIGHT, "stage", 1, "stage"),
        LongStage("OfficeDeskTypeKeys-Sharpa-v1", LEFT, "stage", 6, "stage"),
        LongStage("OfficeDeskClickMouse-Sharpa-v1", RIGHT, "stage", 7, "stage"),
    )),
    LongTask("PipetteTransferLong-Sharpa-v1", "pipette", _DUAL, (
        LongStage("PipetteTransferAspirateLiquid-Sharpa-v1", RIGHT, "stage", 1, "stage"),
        LongStage("PipetteTransferDispenseLiquid-Sharpa-v1", LEFT, "stage", 2, "stage"),
    )),
    LongTask("JuicePreparationLong-Sharpa-v1", "juice", _DUAL, (
        LongStage("JuicePreparationOpenFridge-Sharpa-v1", LEFT, "stage", 1, "stage"),
        LongStage("JuicePreparationAppleToBlender-Sharpa-v1", RIGHT, "stage", 2, "stage"),
        LongStage("JuicePreparationCloseFridge-Sharpa-v1", LEFT, "stage", 3, "stage"),
        LongStage("JuicePreparationPlaceLid-Sharpa-v1", RIGHT, "stage", 4, "stage"),
        LongStage("JuicePreparationPressButton-Sharpa-v1", RIGHT, "stage", 5, "stage"),
    )),
    LongTask("NutBoltLong-Sharpa-v1", "nut", _DUAL, (
        LongStage("NutBoltPickBolt-Sharpa-v1", LEFT, "stage_bolt_placed"),
        LongStage("NutBoltSeatNut-Sharpa-v1", RIGHT, "stage_nut_placed"),
        LongStage("NutBoltTurnNut-Sharpa-v1", RIGHT, "stage_turned"),
    )),
    LongTask("PackageOpeningLong-Sharpa-v1", "package", _DUAL, (
        LongStage("PackageOpeningOpenPackage-Sharpa-v1", DUAL, "stage", 1, "stage"),
        LongStage("PackageOpeningExtractApple-Sharpa-v1", RIGHT, "stage", 2, "stage"),
    )),
    # The only family whose subtasks both use the right hand, so it runs the single-arm robot
    # and needs no routing. Its stage machine spends stages 1-3 on the dustpan subtask.
    LongTask("TableCleaningLong-Sharpa-v1", "table", _RIGHT_ARM, (
        LongStage("TableCleaningPickDebris-Sharpa-v1", RIGHT, "stage", 1, "stage"),
        LongStage("TableCleaningDumpDustpan-Sharpa-v1", RIGHT, "stage", 4, "stage"),
    )),
)

BY_ENV = {t.env_id: t for t in LONG_TASKS}
BY_FAMILY = {t.family: t for t in LONG_TASKS}


def resolve_long_task(key: str) -> LongTask:
    if key in BY_ENV:
        return BY_ENV[key]
    if key in BY_FAMILY:
        return BY_FAMILY[key]
    raise KeyError(f"{key!r} is not a long task; choose from {sorted(BY_ENV)} or {sorted(BY_FAMILY)}")


def stage_reached(info, stage: LongStage) -> bool:
    """True when `info` says this stage is finished. Works for both gate kinds."""
    import numpy as np

    value = info.get(stage.gate_key)
    if value is None:
        raise KeyError(f"{stage.gate_key!r} missing from info; probe the env before evaluating")
    value = np.asarray(getattr(value, "cpu", lambda: value)()).reshape(-1)[0]
    return float(value) >= stage.gate_value if stage.gate_kind == "stage" else bool(value)


def route_action(action, arm: str, held, dual: bool):
    """Place a 29-dim single-arm action into the dual-arm action, holding the other arm.

    `held` is the current joint position of every actuated joint, so the arm that is not acting
    keeps its pose under absolute pd_joint_pos control instead of falling to zero.
    """
    import numpy as np

    action = np.asarray(action).reshape(-1)
    if not dual:
        return action
    out = np.asarray(held, dtype=np.float32).reshape(-1).copy()
    if arm == DUAL:
        if action.size != DUAL_ACTION_DIM:
            raise ValueError(f"dual stage needs {DUAL_ACTION_DIM} dims, got {action.size}")
        return action
    if action.size != SINGLE_ACTION_DIM:
        raise ValueError(f"single-arm stage needs {SINGLE_ACTION_DIM} dims, got {action.size}")
    slots = DUAL_SLOTS[arm]
    a0, a1 = slots["arm"]; h0, h1 = slots["hand"]
    out[a0:a1] = action[:7]
    out[h0:h1] = action[7:]
    return out

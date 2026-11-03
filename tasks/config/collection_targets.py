"""The 40 tasks demonstrations are collected on: 10 long-horizon tasks and their 30 stages.

Each task registers one environment id per end effector (tasks/config/hands.py), 160 ids
in all. Anything that enumerates collection targets reads this module.
"""
from __future__ import annotations

from tasks.config.hands import HANDS, env_id

#: The ten long-horizon tasks, one per family.
LONG_TASKS: tuple[str, ...] = (
    "CabinetPouringLong",
    "DrawerPlugLong",
    "FurnitureAssemblyLong",
    "JuicePreparationLong",
    "MealPreparationLong",
    "NutBoltLong",
    "OfficeDeskLong",
    "PackageOpeningLong",
    "PipetteTransferLong",
    "TableCleaningLong",
)

#: The thirty stage tasks. Each is a strict stage of its family's long task.
SUBTASKS: tuple[str, ...] = (
    "CabinetPouringOpenCabinet",
    "CabinetPouringPickBottle",
    "CabinetPouringPourBeads",
    "CabinetPouringUnscrewCap",
    "DrawerPlugInsertPlug",
    "DrawerPlugOpenDrawer",
    "DrawerPlugPickCharger",
    "FurnitureAssemblyFastenScrew",
    "FurnitureAssemblyInsertRail",
    "FurnitureAssemblyLayFlatPanel",
    "JuicePreparationAppleToBlender",
    "JuicePreparationCloseFridge",
    "JuicePreparationOpenFridge",
    "JuicePreparationPlaceLid",
    "JuicePreparationPressButton",
    "MealPreparationOpenLunchBox",
    "MealPreparationPlaceBread",
    "MealPreparationScoopTomato",
    "NutBoltPickBolt",
    "NutBoltSeatNut",
    "NutBoltTurnNut",
    "OfficeDeskClickMouse",
    "OfficeDeskOpenLaptop",
    "OfficeDeskTypeKeys",
    "PackageOpeningExtractApple",
    "PackageOpeningOpenPackage",
    "PipetteTransferAspirateLiquid",
    "PipetteTransferDispenseLiquid",
    "TableCleaningDumpDustpan",
    "TableCleaningPickDebris",
)

TASKS: tuple[str, ...] = LONG_TASKS + SUBTASKS


def env_ids(hand: str = "Sharpa", tasks: tuple[str, ...] = TASKS) -> tuple[str, ...]:
    """The environment ids of `tasks` for one end effector."""
    return tuple(env_id(task, hand) for task in tasks)


#: Every collection target: 40 tasks x 4 end effectors.
COLLECTION_TARGETS: tuple[str, ...] = tuple(env_id(task, hand) for hand in HANDS for task in TASKS)


def is_collection_target(env_id: str) -> bool:
    return str(env_id) in COLLECTION_TARGETS

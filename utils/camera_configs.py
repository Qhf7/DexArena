"""Camera geometry for every RoboDex long task.

The wrist camera is the same calibrated Marvin/Sharpa view on every task
(`DEXARENA_FIXED_WRIST_CAMERA_CONFIG`); only the hand flange it rides on varies
(`WRIST_MOUNT_BY_TASK`, resolved by `resolve_wrist_camera_mount`). The base
camera is per long task (`FAMILY_CAMERA_RIGS`) and shared by its subtasks.
`gym.make(..., sensor_configs=...)` can re-pose a camera but cannot create one
or move its mount, so the mount must be right in `_default_sensor_configs`.

Changing a pose changes `camera_config_hash` and invalidates every dataset
rendered under it; add a versioned entry instead of editing in place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

DEXARENA_BASE_CAMERA_NAME = "base_camera"
DEXARENA_BASE_CAMERA_WIDTH = 320
DEXARENA_BASE_CAMERA_HEIGHT = 240
DEXARENA_BASE_CAMERA_FOV_DEG = 58.0
DEXARENA_BASE_CAMERA_NEAR = 0.01
DEXARENA_BASE_CAMERA_FAR = 100
DEXARENA_BASE_CAMERA_EYE = (-0.5, -0.15, 0.6)
DEXARENA_BASE_CAMERA_TARGET = (-0.15, -0.15, 0.1)


@dataclass(frozen=True)
class BaseCameraConfig:
    """Pinhole parameters and a world-frame pose for a base camera."""

    eye_in_world_m: tuple[float, float, float]
    target_in_world_m: tuple[float, float, float]
    image_up_in_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    name: str = "base_camera"
    width: int = 256
    height: int = 256
    vertical_fov_deg: float = 90.0
    near_m: float = 0.01
    far_m: float = 100.0


@dataclass(frozen=True)
class WristCameraConfig:
    """Pinhole parameters and a wrist-link-frame pose."""

    eye_in_parent_m: tuple[float, float, float]
    forward_in_parent: tuple[float, float, float]
    image_up_in_parent: tuple[float, float, float]
    name: str = "task_wrist_camera"
    width: int = 256
    height: int = 256
    vertical_fov_deg: float = 90.0
    near_m: float = 0.01
    far_m: float = 2.0

    @property
    def target_in_parent_m(self) -> tuple[float, float, float]:
        return tuple(
            eye + direction
            for eye, direction in zip(
                self.eye_in_parent_m,
                self.forward_in_parent,
            )
        )


@dataclass(frozen=True)
class MarvinWristCameraConfig:
    """Robot integration and pinhole parameters for a Marvin wrist camera."""

    robot_uid: str
    urdf_filename: str
    mount_link_name: str
    eye_in_parent_m: tuple[float, float, float]
    forward_in_parent: tuple[float, float, float]
    image_up_in_parent: tuple[float, float, float]
    width: int = 640
    height: int = 360
    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 58.71550708558255
    near_m: float = 0.01
    far_m: float = 2.0

    @property
    def target_in_parent_m(self) -> tuple[float, float, float]:
        return tuple(
            eye + direction
            for eye, direction in zip(
                self.eye_in_parent_m,
                self.forward_in_parent,
            )
        )


_MARVIN_COMMON_MOUNT = {
    "eye_in_parent_m": (0.060, 0.0, 0.012),
    "forward_in_parent": (-0.17364817766693033, 0.0, 0.984807753012208),
    "image_up_in_parent": (-0.984807753012208, 0.0, -0.17364817766693033),
}

SHARPA_WRIST_CAMERA_V1 = MarvinWristCameraConfig(
    robot_uid="panda_sharpa_hand_right_angled_wristcam_v1",
    urdf_filename="panda_sharpa_hand_right_angled_marvin_wristcam_v1.urdf",
    mount_link_name="right_hand_wrist",
    **_MARVIN_COMMON_MOUNT,
)

WUJI_WRIST_CAMERA_V1 = MarvinWristCameraConfig(
    robot_uid="panda_wuji_hand_right_wristcam_v1",
    urdf_filename="panda_wuji_hand_right_marvin_wristcam_v1.urdf",
    mount_link_name="hand_docking_link",
    **_MARVIN_COMMON_MOUNT,
)

MARVIN_WRIST_CAMERA_CONFIGS: Mapping[str, MarvinWristCameraConfig] = (
    MappingProxyType(
        {
            SHARPA_WRIST_CAMERA_V1.robot_uid: SHARPA_WRIST_CAMERA_V1,
            WUJI_WRIST_CAMERA_V1.robot_uid: WUJI_WRIST_CAMERA_V1,
        }
    )
)


@dataclass(frozen=True)
class FamilyCameraRig:
    """The base view one long task's subtasks share; `stages` lists the subtasks
    rendered under it."""

    base: BaseCameraConfig
    version: str
    stages: tuple[str, ...]


# ---- base cameras, one per long task ----------------------------------------

# CabinetPour: 20 degrees down, ~40 cm back; the geometry the ACT image policies
# were validated against (`docs/camera_setup`).
CABINET_BASE_CAMERA = BaseCameraConfig(
    eye_in_world_m=(-0.3904819657242641, 0.1, 1.08),
    target_in_world_m=(-0.12114399236727436, 0.1, 0.34),
)

PACKAGE_OPENING_BASE_CAMERA = CABINET_BASE_CAMERA

# Since 2026-09-16 every family uses the CabinetPour pose, so the whole fleet
# sits in one visual domain. The retired bespoke poses are kept below because
# datasets were rendered under them. Cost of the unification at 128x128: the
# office mouse (3.4 px) and the ScrewingTurn nut (2.7 px) are at the edge of
# readability; a narrower FOV would throw the drawer pull out of frame.

# Office scene anchors (the retired office pose aimed at their centroid).
OFFICE_OBJECT_ANCHORS: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "OfficeDeskClickMouse-Sharpa-v1": (-0.100, -0.32, 0.080),
        "OfficeDeskOpenLaptop-Sharpa-v1": (-0.063, 0.00, 0.160),
        "OfficeDeskTypeKeys-Sharpa-v1": (-0.190, 0.00, 0.055),
    }
)
OFFICE_ANCHOR_CENTROID: tuple[float, float, float] = (-0.1177, -0.1067, 0.0983)

SCREWING_WORK_CENTER: tuple[float, float, float] = (-0.12, 0.0, 0.068)
SCREWING_PLACE_RADIUS = 0.10

# DrawerPlug scene constants, measured off the built env. The demos pull the
# drawer to the 0.498 m joint limit, not the 0.18 m success threshold.
DRAWER_PLUG_HANDLE_CLOSED: tuple[float, float, float] = (-0.122, 0.1833, 0.3291)
DRAWER_PLUG_SUCCESS_TRAVEL = 0.18
DRAWER_PLUG_FULL_TRAVEL = 0.498     # = drawer_open_limit, and the measured max


DEXARENA_FIXED_CAMERA_CONFIG_VERSION = "z1p08_overhead20_back40cm_v1"


# ---- retired poses -----------------------------------------------------------
# Kept because datasets were rendered under them (registered as retired presets
# in `utils/pixel_cameras.py`); not for new work.
RETIRED_OFFICE_CENTROID58 = BaseCameraConfig(
    eye_in_world_m=(0.25, -0.45, 0.60),
    target_in_world_m=OFFICE_ANCHOR_CENTROID,
    vertical_fov_deg=58.0,
)

RETIRED_SCREWING_SIDE_NEAR = BaseCameraConfig(
    eye_in_world_m=(-0.12, -0.38, 0.30),
    target_in_world_m=SCREWING_WORK_CENTER,
    vertical_fov_deg=58.0,
)

RETIRED_DRAWER_PLUG_HIGH38 = BaseCameraConfig(
    eye_in_world_m=(-0.37, -0.80, 1.25),
    target_in_world_m=(-0.37, 0.10, 0.31),
    vertical_fov_deg=38.0,
)

# The superseded z1p08 office pose named by the `office_fixed` / `office_task`
# presets (not the pose the fleet unified on).
LEGACY_OFFICE_BASE_CAMERA_Z1P08 = BaseCameraConfig(
    eye_in_world_m=(-0.626326980708465, -0.05, 1.08),
    target_in_world_m=(-0.2623567464422627, -0.05, 0.08),
)


# Every family currently uses the same base camera; the registry keeps `stages`
# checkable and gives a family that later needs its own pose a place for it.
FAMILY_CAMERA_RIGS: Mapping[str, FamilyCameraRig] = MappingProxyType(
    {
        "cabinet": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=(
                "CabinetPouringOpenCabinet-Sharpa-v1",
                "CabinetPouringPickBottle-Sharpa-v1",
                "CabinetPouringUnscrewCap-Sharpa-v1",
                "CabinetPouringPourBeads-Sharpa-v1",
            ),
        ),
        "package_opening": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=(
                "PackageOpeningOpenPackage-Sharpa-v1",
                "PackageOpeningExtractApple-Sharpa-v1",
            ),
        ),
        "office": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=(
                "OfficeDeskClickMouse-Sharpa-v1",
                "OfficeDeskOpenLaptop-Sharpa-v1",
                "OfficeDeskTypeKeys-Sharpa-v1",
            ),
        ),
        "screwing": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=(
                "NutBoltLong-Sharpa-v1",
                "NutBoltTurnNut-Sharpa-v1",
                "NutBoltPickBolt-Sharpa-v1",
                "NutBoltSeatNut-Sharpa-v1",
            ),
        ),
        "drawer_plug": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=(
                "DrawerPlugLong-Sharpa-v1",
                "DrawerPlugOpenDrawer-Sharpa-v1",
                "DrawerPlugPickCharger-Sharpa-v1",
                "DrawerPlugInsertPlug-Sharpa-v1",
            ),
        ),
        "meal_preparation": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=(
                "MealPreparationOpenLunchBox-Sharpa-v1",
                "MealPreparationPlaceBread-Sharpa-v1",
                "MealPreparationScoopTomato-Sharpa-v1",
            ),
        ),
        "furniture": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=(
                "FurnitureAssemblyLayFlatPanel-Sharpa-v1",
                "FurnitureAssemblyInsertRail-Sharpa-v1",
                "FurnitureAssemblyFastenScrew-Sharpa-v1",
            ),
        ),
        "pipette": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=(
                "PipetteTransferAspirateLiquid-Sharpa-v1",
                "PipetteTransferDispenseLiquid-Sharpa-v1",
            ),
        ),
        "juice": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=(
                "JuicePreparationOpenFridge-Sharpa-v1",
                "JuicePreparationCloseFridge-Sharpa-v1",
                "JuicePreparationAppleToBlender-Sharpa-v1",
                "JuicePreparationPressButton-Sharpa-v1",
                "JuicePreparationPlaceLid-Sharpa-v1",
            ),
        ),
        "table_clean": FamilyCameraRig(
            base=CABINET_BASE_CAMERA,
            version=DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
            stages=("TableCleaningDumpDustpan-Sharpa-v1", "TableCleaningPickDebris-Sharpa-v1"),
        ),
    }
)

FAMILY_BY_STAGE: Mapping[str, str] = MappingProxyType(
    {stage: family for family, rig in FAMILY_CAMERA_RIGS.items() for stage in rig.stages}
)


DEXARENA_FIXED_BASE_CAMERA_CONFIGS: Mapping[str, BaseCameraConfig] = MappingProxyType(
    {
        "office": LEGACY_OFFICE_BASE_CAMERA_Z1P08,
        "cabinet": CABINET_BASE_CAMERA,
        "package_opening": PACKAGE_OPENING_BASE_CAMERA,
    }
)


# ---- the one wrist camera ----------------------------------------------------

# The policy camera uses the calibrated Marvin/Sharpa extrinsics, with a
# square 256-pixel sensor and 90-degree vertical FOV for image-policy input.
DEXARENA_FIXED_WRIST_CAMERA_CONFIG = WristCameraConfig(
    eye_in_parent_m=SHARPA_WRIST_CAMERA_V1.eye_in_parent_m,
    forward_in_parent=SHARPA_WRIST_CAMERA_V1.forward_in_parent,
    image_up_in_parent=SHARPA_WRIST_CAMERA_V1.image_up_in_parent,
)

DEXARENA_WRIST_CAMERA_NAME = DEXARENA_FIXED_WRIST_CAMERA_CONFIG.name

# Which hand flange the wrist camera rides on, per env id (the side is the
# `robot_type` recorded in that stage's dataset, never inferred from the name).
# A dual-armed env must appear here; the generic fallback is ambiguous there.
WRIST_MOUNT_BY_TASK: Mapping[str, str] = MappingProxyType(
    {
        # Office
        "OfficeDeskOpenLaptop-Sharpa-v1": "right_hand_wrist",
        "OfficeDeskTypeKeys-Sharpa-v1": "left_hand_wrist",
        "OfficeDeskClickMouse-Sharpa-v1": "right_hand_wrist",
        # CabinetPour
        "CabinetPouringPickBottle-Sharpa-v1": "right_hand_wrist",
        "CabinetPouringUnscrewCap-Sharpa-v1": "right_hand_wrist",
        "CabinetPouringOpenCabinet-Sharpa-v1": "left_hand_wrist",
        "CabinetPouringPourBeads-Sharpa-v1": "right_hand_wrist",
        # PackageOpening
        "PackageOpeningExtractApple-Sharpa-v1": "right_hand_wrist",
        "PackageOpeningOpenPackage-Sharpa-v1": "right_hand_wrist",
        # Screwing (the bolt stage is left-handed).
        "NutBoltLong-Sharpa-v1": "right_hand_wrist",
        "NutBoltTurnNut-Sharpa-v1": "right_hand_wrist",
        "NutBoltSeatNut-Sharpa-v1": "right_hand_wrist",
        "NutBoltPickBolt-Sharpa-v1": "left_hand_wrist",
        # DrawerPlug (mixed-handed).
        "DrawerPlugLong-Sharpa-v1": "left_hand_wrist",
        "DrawerPlugOpenDrawer-Sharpa-v1": "left_hand_wrist",
        "DrawerPlugPickCharger-Sharpa-v1": "right_hand_wrist",
        "DrawerPlugInsertPlug-Sharpa-v1": "right_hand_wrist",
        # Remaining sharpa_hand stages.
        "MealPreparationOpenLunchBox-Sharpa-v1": "left_hand_wrist",
        "MealPreparationPlaceBread-Sharpa-v1": "right_hand_wrist",
        "FurnitureAssemblyLayFlatPanel-Sharpa-v1": "left_hand_wrist",
        "FurnitureAssemblyInsertRail-Sharpa-v1": "right_hand_wrist",
        "PipetteTransferAspirateLiquid-Sharpa-v1": "right_hand_wrist",
        "JuicePreparationOpenFridge-Sharpa-v1": "left_hand_wrist",
        "JuicePreparationCloseFridge-Sharpa-v1": "left_hand_wrist",
        "JuicePreparationAppleToBlender-Sharpa-v1": "right_hand_wrist",
        "JuicePreparationPressButton-Sharpa-v1": "right_hand_wrist",
        "TableCleaningDumpDustpan-Sharpa-v1": "right_hand_wrist",
        "TableCleaningPickDebris-Sharpa-v1": "right_hand_wrist",
        "JuicePreparationPlaceLid-Sharpa-v1": "right_hand_wrist",
        "FurnitureAssemblyFastenScrew-Sharpa-v1": "right_hand_wrist",
        "MealPreparationScoopTomato-Sharpa-v1": "right_hand_wrist",
        "PipetteTransferDispenseLiquid-Sharpa-v1": "left_hand_wrist",
    }
)

# Historical name, kept because `tasks/cabinet_pouring/cabinet_pouring_long.py` error messages quote it.
DEXARENA_FIXED_WRIST_MOUNT_BY_TASK = WRIST_MOUNT_BY_TASK


# ---- builders ----------------------------------------------------------------


def build_dexarena_base_camera(
    *,
    eye: Sequence[float] = DEXARENA_BASE_CAMERA_EYE,
    target: Sequence[float] = DEXARENA_BASE_CAMERA_TARGET,
    up: Sequence[float] = (0.0, 0.0, 1.0),
    width: int = DEXARENA_BASE_CAMERA_WIDTH,
    height: int = DEXARENA_BASE_CAMERA_HEIGHT,
    fov_deg: float = DEXARENA_BASE_CAMERA_FOV_DEG,
    near: float = DEXARENA_BASE_CAMERA_NEAR,
    far: float = DEXARENA_BASE_CAMERA_FAR,
    name: str = DEXARENA_BASE_CAMERA_NAME,
) -> CameraConfig:
    """Build the shared RoboDex base camera for RGB task observations."""

    pose = sapien_utils.look_at(list(eye), list(target), up=list(up))
    return CameraConfig(
        str(name),
        pose,
        int(width),
        int(height),
        np.deg2rad(float(fov_deg)),
        float(near),
        float(far),
    )


def _build_base_camera(
    config: BaseCameraConfig, *, width: int | None = None, height: int | None = None
) -> CameraConfig:
    return build_dexarena_base_camera(
        eye=config.eye_in_world_m,
        target=config.target_in_world_m,
        up=config.image_up_in_world,
        width=config.width if width is None else width,
        height=config.height if height is None else height,
        fov_deg=config.vertical_fov_deg,
        near=config.near_m,
        far=config.far_m,
        name=config.name,
    )


def build_dexarena_fixed_base_camera(scene: str) -> CameraConfig:
    """Build the `z1p08` fixed base view for an Office/Cabinet/PackageOpening scene.

    `scene="office"` returns the SUPERSEDED office pose; new office work wants
    `build_family_base_camera("office")`.
    """

    return _build_base_camera(DEXARENA_FIXED_BASE_CAMERA_CONFIGS[scene])


def build_family_base_camera(
    family: str, *, width: int | None = None, height: int | None = None
) -> CameraConfig:
    """The shared base view of one long task. Resolution may be overridden.

    The ACT image policies want 256; the RLPD pixels stack asks for 128 through
    `gym.make(..., sensor_configs=...)`, which resizes without moving the camera.
    """
    try:
        rig = FAMILY_CAMERA_RIGS[family]
    except KeyError:
        raise KeyError(
            f"No camera rig for long task {family!r}. "
            f"Known: {sorted(FAMILY_CAMERA_RIGS)}"
        ) from None
    return _build_base_camera(rig.base, width=width, height=height)


def build_dexarena_fixed_wrist_camera(
    *, mount, width: int | None = None, height: int | None = None
) -> CameraConfig:
    """The calibrated wrist view, on the link it was calibrated in."""

    config = DEXARENA_FIXED_WRIST_CAMERA_CONFIG
    pose = sapien_utils.look_at(
        list(config.eye_in_parent_m),
        list(config.target_in_parent_m),
        up=list(config.image_up_in_parent),
    )
    return CameraConfig(
        config.name,
        pose,
        int(config.width if width is None else width),
        int(config.height if height is None else height),
        np.deg2rad(config.vertical_fov_deg),
        config.near_m,
        config.far_m,
        mount=mount,
    )


def resolve_wrist_camera_mount(robot, env_id: str | None, *, agent=None):
    """The hand-flange link the policy wrist camera rides on.

    Resolution order: the `WRIST_MOUNT_BY_TASK` entry, the agent's own
    `wrist_camera_mount_link`, then the robot's lone `*_hand_wrist` link. It
    never falls back to the palm or the TCP (the old
    `links_map.get("wrist", self.agent.tcp)` always did, since no Sharpa robot
    has a link named `wrist`, and rendered the palm filling the frame).
    """
    links = robot.get_links()
    mount_name = WRIST_MOUNT_BY_TASK.get(str(env_id or ""))
    if mount_name is not None:
        mount = sapien_utils.get_obj_by_name(links, mount_name)
        if mount is None:
            raise RuntimeError(
                f"wrist camera mount {mount_name!r} (from "
                f"WRIST_MOUNT_BY_TASK[{env_id!r}]) is not a link of this robot. "
                f"Available: {[link.name for link in links]}"
            )
        return mount
    if agent is not None:
        mount = getattr(agent, "wrist_camera_mount_link", None)
        if mount is not None:
            return mount
    candidates = [link for link in links if link.name.endswith("_hand_wrist")]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        "cannot resolve a wrist camera mount "
        + (
            f"(ambiguous: {[c.name for c in candidates]})"
            if candidates
            else "(no *_hand_wrist link)"
        )
        + f". Add an entry for {env_id!r} to WRIST_MOUNT_BY_TASK."
    )

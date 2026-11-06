"""Single source of truth for the image-based observation pipeline.

Every data source (online env, eval env, replay rendering, offline HDF5) builds
its cameras from the same :class:`PixelObsConfig` and passes
``config.sensor_configs()`` to ``gym.make``. The config hashes itself; the hash
is written into every dataset and checked at training start.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    """One camera of the observation contract.

    ``sensor_name`` is the ManiSkill sensor uid (the key in ``sensor_data``);
    ``obs_key`` is the canonical name the algorithm sees (``cam_0`` / ``cam_1``).
    ``eye``/``target`` build the camera pose with ``sapien_utils.look_at``; for a
    mounted camera (the wrist) the pose is relative to its mount link, which is
    defined by the task and identified here by ``sensor_name``.
    """

    sensor_name: str
    obs_key: str
    # `None` for both: keep the pose/FOV the env declares, set only the resolution.
    eye: tuple[float, float, float] | None = None
    target: tuple[float, float, float] | None = None
    fov_deg: float = 90.0
    # Image-up direction for `look_at` (the calibrated wrist camera's up is not +Z).
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)



def _cabinet_fixed_cameras() -> tuple[CameraSpec, ...]:
    """The CabinetPour views, read from `utils/camera_configs.py` so the geometry
    has one source; resolution comes from `PixelObsConfig`."""
    from utils.camera_configs import (
        DEXARENA_FIXED_BASE_CAMERA_CONFIGS,
        DEXARENA_FIXED_WRIST_CAMERA_CONFIG,
    )

    base = DEXARENA_FIXED_BASE_CAMERA_CONFIGS["cabinet"]
    wrist = DEXARENA_FIXED_WRIST_CAMERA_CONFIG
    return (
        CameraSpec(
            sensor_name=base.name,
            obs_key="cam_0",
            eye=tuple(base.eye_in_world_m),
            target=tuple(base.target_in_world_m),
            up=tuple(base.image_up_in_world),
            fov_deg=float(base.vertical_fov_deg),
        ),
        CameraSpec(
            # `task_wrist_camera` is the image-policy view declared by the task.
            sensor_name=wrist.name,
            obs_key="cam_1",
            eye=tuple(wrist.eye_in_parent_m),
            target=tuple(wrist.target_in_parent_m),
            up=tuple(wrist.image_up_in_parent),
            fov_deg=float(wrist.vertical_fov_deg),
        ),
    )


CABINET_FIXED_CAMERAS: tuple[CameraSpec, ...] = _cabinet_fixed_cameras()


def _office_fixed_cameras() -> tuple[CameraSpec, ...]:
    """The calibrated rig in the office scene, retargeting the envs' existing
    `base_camera` and `wrist_camera`. Reference only; see CAMERA_PRESETS."""
    from utils.camera_configs import (
        DEXARENA_FIXED_BASE_CAMERA_CONFIGS,
        DEXARENA_FIXED_WRIST_CAMERA_CONFIG,
    )

    base = DEXARENA_FIXED_BASE_CAMERA_CONFIGS["office"]
    wrist = DEXARENA_FIXED_WRIST_CAMERA_CONFIG
    return (
        CameraSpec(
            sensor_name=base.name,
            obs_key="cam_0",
            eye=tuple(base.eye_in_world_m),
            target=tuple(base.target_in_world_m),
            up=tuple(base.image_up_in_world),
            fov_deg=float(base.vertical_fov_deg),
        ),
        CameraSpec(
            sensor_name="wrist_camera",
            obs_key="cam_1",
            eye=tuple(wrist.eye_in_parent_m),
            target=tuple(wrist.target_in_parent_m),
            up=tuple(wrist.image_up_in_parent),
            fov_deg=float(wrist.vertical_fov_deg),
        ),
    )


OFFICE_FIXED_CAMERAS: tuple[CameraSpec, ...] = _office_fixed_cameras()


def _office_task_cameras() -> tuple[CameraSpec, ...]:
    """Each office env's own base view (resolution only) plus the calibrated
    wrist view on `wrist_camera`. Kept to describe the 2026-09-12 datasets;
    prefer `office_rig`."""
    from utils.camera_configs import DEXARENA_FIXED_WRIST_CAMERA_CONFIG

    wrist = DEXARENA_FIXED_WRIST_CAMERA_CONFIG
    return (
        CameraSpec(sensor_name="base_camera", obs_key="cam_0"),
        CameraSpec(
            sensor_name="wrist_camera",
            obs_key="cam_1",
            eye=tuple(wrist.eye_in_parent_m),
            target=tuple(wrist.target_in_parent_m),
            up=tuple(wrist.image_up_in_parent),
            fov_deg=float(wrist.vertical_fov_deg),
        ),
    )


OFFICE_TASK_CAMERAS: tuple[CameraSpec, ...] = _office_task_cameras()


def _office_shared_cameras() -> tuple[CameraSpec, ...]:
    """One base view (office_laptop's) for all three office stages. Rejected
    (the mouse is out of frame); kept as a record."""
    from utils.camera_configs import DEXARENA_FIXED_WRIST_CAMERA_CONFIG

    wrist = DEXARENA_FIXED_WRIST_CAMERA_CONFIG
    return (
        CameraSpec(
            sensor_name="base_camera",
            obs_key="cam_0",
            eye=(0.12, -0.35, 0.55),
            target=(-0.063, 0.0, 0.16),
            fov_deg=58.0,
        ),
        CameraSpec(
            sensor_name="wrist_camera",
            obs_key="cam_1",
            eye=tuple(wrist.eye_in_parent_m),
            target=tuple(wrist.target_in_parent_m),
            up=tuple(wrist.image_up_in_parent),
            fov_deg=float(wrist.vertical_fov_deg),
        ),
    )


OFFICE_SHARED_CAMERAS: tuple[CameraSpec, ...] = _office_shared_cameras()


def _family_rig_cameras(family: str) -> tuple[CameraSpec, ...]:
    """One long task's rig from `FAMILY_CAMERA_RIGS`: its shared base pose plus
    the calibrated wrist view on `task_wrist_camera` (mounted on `*_hand_wrist`)."""
    from utils.camera_configs import (
        FAMILY_CAMERA_RIGS,
        DEXARENA_FIXED_WRIST_CAMERA_CONFIG,
        DEXARENA_WRIST_CAMERA_NAME,
    )

    base = FAMILY_CAMERA_RIGS[family].base
    wrist = DEXARENA_FIXED_WRIST_CAMERA_CONFIG
    return (
        CameraSpec(
            sensor_name=base.name,
            obs_key="cam_0",
            eye=tuple(base.eye_in_world_m),
            target=tuple(base.target_in_world_m),
            up=tuple(base.image_up_in_world),
            fov_deg=float(base.vertical_fov_deg),
        ),
        CameraSpec(
            sensor_name=DEXARENA_WRIST_CAMERA_NAME,
            obs_key="cam_1",
            eye=tuple(wrist.eye_in_parent_m),
            target=tuple(wrist.target_in_parent_m),
            up=tuple(wrist.image_up_in_parent),
            fov_deg=float(wrist.vertical_fov_deg),
        ),
    )


def _retired_rig_cameras(base_attr: str) -> tuple[CameraSpec, ...]:
    """A retired base pose + the calibrated wrist view, kept so datasets
    rendered under it stay loadable."""
    from utils import camera_configs as cc

    base = getattr(cc, base_attr)
    wrist = cc.DEXARENA_FIXED_WRIST_CAMERA_CONFIG
    return (
        CameraSpec(
            sensor_name=base.name,
            obs_key="cam_0",
            eye=tuple(base.eye_in_world_m),
            target=tuple(base.target_in_world_m),
            up=tuple(base.image_up_in_world),
            fov_deg=float(base.vertical_fov_deg),
        ),
        CameraSpec(
            sensor_name=cc.DEXARENA_WRIST_CAMERA_NAME,
            obs_key="cam_1",
            eye=tuple(wrist.eye_in_parent_m),
            target=tuple(wrist.target_in_parent_m),
            up=tuple(wrist.image_up_in_parent),
            fov_deg=float(wrist.vertical_fov_deg),
        ),
    )


OFFICE_RIG_CAMERAS: tuple[CameraSpec, ...] = _family_rig_cameras("office")


# Since 2026-09-16 every family uses the shared CabinetPour base pose; the
# retired bespoke poses are kept below.
SCREWING_RIG_CAMERAS: tuple[CameraSpec, ...] = _family_rig_cameras("screwing")
DRAWER_PLUG_RIG_CAMERAS: tuple[CameraSpec, ...] = _family_rig_cameras("drawer_plug")

# Retired geometry, kept loadable. See `_retired_rig_cameras`.
OFFICE_RIG_CENTROID58_CAMERAS = _retired_rig_cameras("RETIRED_OFFICE_CENTROID58")
SCREWING_RIG_SIDE_NEAR_CAMERAS = _retired_rig_cameras("RETIRED_SCREWING_SIDE_NEAR")
DRAWER_PLUG_RIG_HIGH38_CAMERAS = _retired_rig_cameras("RETIRED_DRAWER_PLUG_HIGH38")

CAMERA_PRESETS: dict[str, tuple[CameraSpec, ...]] = {
    # CabinetPour family (and every family since the 2026-09-16 unification).
    "cabinet_fixed": CABINET_FIXED_CAMERAS,
    # Office presets kept for reference / existing datasets; use `office_rig`.
    "office_fixed": OFFICE_FIXED_CAMERAS,
    "office_task": OFFICE_TASK_CAMERAS,
    "office_shared": OFFICE_SHARED_CAMERAS,
    # Per-family rigs: shared base pose + calibrated wrist view on `*_hand_wrist`.
    "screwing_rig": SCREWING_RIG_CAMERAS,
    "office_rig": OFFICE_RIG_CAMERAS,
    "drawer_plug_rig": DRAWER_PLUG_RIG_CAMERAS,
    # Retired 2026-09-16, kept so datasets rendered under them stay loadable.
    "office_rig_centroid58": OFFICE_RIG_CENTROID58_CAMERAS,
    "screwing_rig_side_near": SCREWING_RIG_SIDE_NEAR_CAMERAS,
    "drawer_plug_rig_high38": DRAWER_PLUG_RIG_HIGH38_CAMERAS,
}

# Changing the preset changes `config_hash`, i.e. it defines a different visual
# domain: datasets rendered under one preset cannot be mixed into another.
DEFAULT_CAMERAS = CABINET_FIXED_CAMERAS


# Proprioceptive fields kept in the observation; object state is never listed.
DEFAULT_PROPRIO_FIELDS: tuple[str, ...] = ("qpos", "qvel", "tcp_pose")

# Cameras the robot mounts on itself: rendered every step and dropped by the
# contract, so they are shrunk (the kwarg can resize a camera but not delete it).
AGENT_SENSOR_NAMES: tuple[str, ...] = ("wrist_camera",)

# Agents whose self-mounted cameras are not called `wrist_camera` (the dual-hand
# agent declares `left_wrist_camera` / `right_wrist_camera`). Keyed by env id.
AGENT_SENSOR_NAMES_BY_TASK: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "PackageOpeningOpenPackage-Sharpa-v1": (
            "left_wrist_camera",
            "right_wrist_camera",
        ),
        # ExtractAndPlaceApple runs the single right-hand robot (`wrist_camera`)
        # and is deliberately not listed.
    }
)

# Task cameras outside the contract, per env (a `sensor_configs` key naming a
# camera the env does not have raises).
UNUSED_TASK_SENSOR_NAMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "OfficeDeskTypeKeys-Sharpa-v1": ("external_camera",),
        "FurnitureAssemblyLayFlatPanel-Sharpa-v1": ("external_camera",),
        "FurnitureAssemblyInsertRail-Sharpa-v1": ("external_camera",),
        "JuicePreparationPressButton-Sharpa-v1": ("button_camera",),
    }
)


def shrink_sensors_for(env_id: str | None) -> tuple[str, ...]:
    """Cameras to shrink for ``env_id``: agent-mounted ones plus known spares
    (cameras a preset uses are skipped)."""
    key = str(env_id or "")
    agent_sensors = AGENT_SENSOR_NAMES_BY_TASK.get(key, AGENT_SENSOR_NAMES)
    return agent_sensors + UNUSED_TASK_SENSOR_NAMES.get(key, ())
SHRUNK_SENSOR_PIXELS = 4


@dataclass(frozen=True)
class PixelObsConfig:
    """Everything that determines what the encoder sees."""

    cameras: tuple[CameraSpec, ...] = DEFAULT_CAMERAS
    width: int = 128
    height: int = 128
    shader_pack: str = "default"
    frame_stack: int = 1
    proprio_fields: tuple[str, ...] = DEFAULT_PROPRIO_FIELDS
    obs_version: str = "rlpd_pixels_v1"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Camera resolution must be positive, got {self.width}x{self.height}.")
        if self.frame_stack < 1:
            raise ValueError(f"`frame_stack` must be >= 1, got {self.frame_stack}.")
        keys = [camera.obs_key for camera in self.cameras]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate camera obs keys: {keys}.")
        names = [camera.sensor_name for camera in self.cameras]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate camera sensor names: {names}.")

    # ---- keys ----------------------------------------------------------------

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(camera.obs_key for camera in self.cameras)

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Per-camera (H, W, C) of a single stored frame."""
        return (self.height, self.width, 3)

    def spec_for(self, obs_key: str) -> CameraSpec:
        for camera in self.cameras:
            if camera.obs_key == obs_key:
                return camera
        raise KeyError(f"No camera with obs_key={obs_key!r}; have {self.camera_keys}.")

    # ---- env construction ----------------------------------------------------

    def sensor_configs(self, *, shrink_sensors: Sequence[str] = ()) -> dict[str, Any]:
        """``sensor_configs`` kwarg for ``gym.make``: global resolution/shader,
        per-camera pose and FOV, and ``shrink_sensors`` shrunk to
        `SHRUNK_SENSOR_PIXELS` (not part of `config_hash`)."""
        from mani_skill.utils import sapien_utils  # local: keeps this module sim-free

        config: dict[str, Any] = {
            "width": int(self.width),
            "height": int(self.height),
            "shader_pack": str(self.shader_pack),
        }
        ours = {camera.sensor_name for camera in self.cameras}
        for name in shrink_sensors:
            if name in ours:
                continue
            config[name] = {
                "width": SHRUNK_SENSOR_PIXELS,
                "height": SHRUNK_SENSOR_PIXELS,
            }
        for camera in self.cameras:
            entry: dict[str, Any] = {
                "width": int(self.width),
                "height": int(self.height),
            }
            if camera.eye is not None and camera.target is not None:
                pose = sapien_utils.look_at(
                    list(camera.eye), list(camera.target), up=list(camera.up)
                )
                # `look_at` may return a batched pose; ManiSkill wants [p, q].
                position = np.asarray(pose.p, dtype=np.float64).reshape(-1)[:3]
                quaternion = np.asarray(pose.q, dtype=np.float64).reshape(-1)[:4]
                entry["pose"] = [*position.tolist(), *quaternion.tolist()]
                entry["fov"] = float(np.deg2rad(camera.fov_deg))
            config[camera.sensor_name] = entry
        return config

    # ---- hashing -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        """Stable 16-hex-char digest of everything that changes the pixels."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def replace(self, **changes: Any) -> "PixelObsConfig":
        return replace(self, **changes)

    def summary(self) -> str:
        cameras = ", ".join(f"{c.obs_key}={c.sensor_name}" for c in self.cameras)
        return (
            f"PixelObsConfig(hash={self.config_hash()}, {self.width}x{self.height}, "
            f"cameras=[{cameras}], shader={self.shader_pack}, "
            f"frame_stack={self.frame_stack}, proprio={list(self.proprio_fields)}, "
            f"version={self.obs_version})"
        )


DEFAULT_PIXEL_OBS_CONFIG = PixelObsConfig()


def pixel_obs_config_from_args(args) -> PixelObsConfig:
    """Build the config from a training ``Args`` dataclass (see rlpd_pixels_config)."""
    preset = getattr(args, "camera_preset", "cabinet_fixed")
    if preset not in CAMERA_PRESETS:
        raise ValueError(
            f"Unknown camera_preset {preset!r}; have {sorted(CAMERA_PRESETS)}."
        )
    return PixelObsConfig(
        cameras=CAMERA_PRESETS[preset],
        width=int(args.image_size),
        height=int(args.image_size),
        shader_pack=str(args.shader_pack),
        frame_stack=int(args.frame_stack),
        proprio_fields=DEFAULT_PROPRIO_FIELDS,
    )

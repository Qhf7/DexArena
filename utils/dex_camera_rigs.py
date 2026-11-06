"""Evaluation camera rigs, isolated from task and collection imports.

Reuse the shared calibration in utils.camera_configs. Families without a shared
base-camera target use measured scene centroids and the same projection rule.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from utils.camera_configs import (
    DEXARENA_FIXED_BASE_CAMERA_CONFIGS,
    DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
    DEXARENA_FIXED_WRIST_CAMERA_CONFIG,
    BaseCameraConfig,
    build_dexarena_base_camera,
    build_dexarena_fixed_wrist_camera,
)

RIG_VERSION = "family_rig_v1_z1p08_overhead20"
BASE_HEIGHT_M = 1.08
BASE_TILT_DEG = 20.0  # optical axis from the downward table normal

BASE_CAMERA_NAME = "base_camera"
WRIST_LINKS: tuple[tuple[str, str], ...] = (("left", "left_hand_wrist"), ("right", "right_hand_wrist"))

# Measured scene centroids (long task, reset over 5 seeds, x/y/z in metres); main's three
# families are taken from utils.camera_configs instead and are not listed here.
_MEASURED_TARGETS: dict[str, tuple[float, float, float]] = {
    "meal": (-0.1567, -0.1562, 0.024),
    "drawer_plug": (0.0271, -0.0059, 0.1421),
    "screwing": (-0.1043, -0.1443, 0.0131),
    "furniture": (-0.129, -0.0506, 0.0283),
    "pipette": (-0.1669, 0.0207, 0.0484),
    "juice": (-0.097, -0.0772, 0.0607),
    "table_clean": (-0.1222, -0.2759, 0.0135),
}

_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("MealPreparation", "meal"),
    ("CabinetPouring", "cabinet"),
    ("DrawerPlug", "drawer_plug"),
    ("NutBolt", "screwing"),
    ("OfficeDesk", "office"),
    ("FurnitureAssembly", "furniture"),
    ("PipetteTransfer", "pipette"),
    ("JuicePreparation", "juice"),
    ("PackageOpening", "package_opening"),
    ("TableCleaning", "table_clean"),
)

FAMILIES: tuple[str, ...] = tuple(sorted(set(_MEASURED_TARGETS) | set(DEXARENA_FIXED_BASE_CAMERA_CONFIGS)))


def family_of(env_id: str) -> str:
    for needle, family in _FAMILY_RULES:
        if env_id.startswith(needle):
            return family
    raise KeyError(f"{env_id}: no camera-rig family (add a rule in utils/dex_camera_rigs.py)")


def derive_eye(target: tuple[float, float, float]) -> tuple[float, float, float]:
    """main's rule: eye at BASE_HEIGHT_M, behind the target along -X so that the optical
    axis is BASE_TILT_DEG from the downward normal. Reproduces main's office and cabinet
    eyes from their targets to 1e-9 (tests)."""
    tx, ty, tz = (float(v) for v in target)
    return (tx - (BASE_HEIGHT_M - tz) * math.tan(math.radians(BASE_TILT_DEG)), ty, BASE_HEIGHT_M)


def base_camera_spec(family: str) -> BaseCameraConfig:
    if family in DEXARENA_FIXED_BASE_CAMERA_CONFIGS:
        return DEXARENA_FIXED_BASE_CAMERA_CONFIGS[family]
    target = _MEASURED_TARGETS[family]
    return BaseCameraConfig(eye_in_world_m=derive_eye(target), target_in_world_m=target, name=BASE_CAMERA_NAME)


def build_base_camera(family: str):
    spec = base_camera_spec(family)
    return build_dexarena_base_camera(
        eye=spec.eye_in_world_m, target=spec.target_in_world_m, up=spec.image_up_in_world,
        width=spec.width, height=spec.height, fov_deg=spec.vertical_fov_deg,
        near=spec.near_m, far=spec.far_m, name=BASE_CAMERA_NAME,
    )


def _agents(agent) -> list:
    return list(getattr(agent, "agents", None) or [agent])


def wrist_mounts(agent) -> list[tuple[str, Any]]:
    """[(camera name, wrist link)] for every arm the environment actually built."""
    mounts = []
    for sub in _agents(agent):
        links = sub.robot.links_map
        for side, link_name in WRIST_LINKS:
            link = links.get(link_name)
            if link is not None:
                mounts.append((f"{side}_wrist_camera", link))
    if not mounts:
        raise RuntimeError("no wrist link found on the robot; expected one of "
                           + ", ".join(name for _side, name in WRIST_LINKS))
    return mounts


def build_wrist_cameras(agent) -> list:
    cameras = []
    for name, link in wrist_mounts(agent):
        camera = build_dexarena_fixed_wrist_camera(mount=link)
        camera.uid = name
        cameras.append(camera)
    return cameras


def rig_cameras(env_unwrapped, family: str) -> list:
    return [build_base_camera(family)] + build_wrist_cameras(env_unwrapped.agent)


def rig_camera_names(env_unwrapped) -> list[str]:
    return [BASE_CAMERA_NAME] + [name for name, _link in wrist_mounts(env_unwrapped.agent)]


def describe(family: str, env_unwrapped=None) -> dict[str, Any]:
    """Describe the evaluation cameras for the environment manifest."""
    spec = base_camera_spec(family)
    out = {
        "rig_version": RIG_VERSION,
        "base_rule": DEXARENA_FIXED_CAMERA_CONFIG_VERSION,
        "family": family,
        "base_camera": {"name": BASE_CAMERA_NAME, **asdict(spec),
                        "source": "utils.camera_configs" if family in DEXARENA_FIXED_BASE_CAMERA_CONFIGS
                        else "derived: measured long-task scene centroid + main's rule"},
        "wrist_camera": {**asdict(DEXARENA_FIXED_WRIST_CAMERA_CONFIG), "links": dict(WRIST_LINKS)},
        "hidden_objects": "kept hidden (sensor semantics), guidance markers off",
    }
    if env_unwrapped is not None:
        out["cameras"] = rig_camera_names(env_unwrapped)
    return out

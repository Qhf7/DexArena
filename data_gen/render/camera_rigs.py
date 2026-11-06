"""One camera rig per task family, applied to every collection target of that family.

WHAT (one fixed-camera rule for every family; per-family tuning is a later step)
    base_camera              world-fixed. main's rule ``z1p08_overhead20_back40cm_v1``
                             (utils/camera_configs.py, PR #44/#45): 1.08 m high, image plane
                             20 deg from the tabletop, placed behind the target along -X;
                             256x256, vertical FOV 90 deg, near/far 0.01/100 m.
    {left,right}_wrist_camera   the calibrated Marvin mount (eye (0.06, 0, 0.012) in the wrist
                             link, forward tilted 10 deg off the link z axis) on
                             ``{side}_hand_wrist`` -- ONE PER ARM PRESENT: a single-arm subtask
                             gets the wrist of the arm it uses, a dual-arm long task gets both;
                             256x256, vertical FOV 90 deg, near/far 0.01/2 m.

    The three families main already fixed (office, cabinet, package_opening) use main's
    eye/target verbatim. The other seven get the same rule applied to a MEASURED target:
    the centroid of the family's LONG task scene content at reset (its scene is the
    superset of its subtasks'), 5 seeds, parked off-scene helpers excluded
    Every number here is a first placement: re-rendering is hours, re-collection is
    weeks, so the rig is tuned later, not during collection.

WHY ITS OWN MODULE
    Nothing under tasks/ imports this file, so it is in no environment's code-fingerprint
    closure: `blast_radius` is 0 and the 40 fingerprints do not move when a camera moves
    (measured). The frozen
    utils/camera_configs.py is READ, never edited.

WHAT IT IS NOT
    Not the cameras the environments were recorded with: those differ per family (and
    within Office / Furniture / Juice), and none of them is in the recorded data, which is
    state-only.
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
    raise KeyError(f"{env_id}: no camera-rig family (add a rule in data_gen/render/camera_rigs.py)")


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


def install_rig(env_cls, family: str) -> None:
    """Make ``env_cls`` render the family rig as its human render cameras.

    Same lever as the camera installer used at record time: a class
    property, in this process only; the task module on disk is untouched.
    """
    env_cls._default_human_render_camera_configs = property(lambda self: rig_cameras(self, family))


def describe(family: str, env_unwrapped=None) -> dict[str, Any]:
    """What went into the videos -- written to every video dataset's manifest."""
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

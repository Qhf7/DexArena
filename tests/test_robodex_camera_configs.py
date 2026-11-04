from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


def test_build_dexarena_base_camera_default_schema() -> None:
    from utils.camera_configs import (
        DEXARENA_BASE_CAMERA_FOV_DEG,
        DEXARENA_BASE_CAMERA_HEIGHT,
        DEXARENA_BASE_CAMERA_NAME,
        DEXARENA_BASE_CAMERA_WIDTH,
        build_dexarena_base_camera,
    )

    camera = build_dexarena_base_camera()

    assert camera.uid == DEXARENA_BASE_CAMERA_NAME
    assert camera.width == DEXARENA_BASE_CAMERA_WIDTH
    assert camera.height == DEXARENA_BASE_CAMERA_HEIGHT
    assert math.isclose(camera.fov, np.deg2rad(DEXARENA_BASE_CAMERA_FOV_DEG))
    assert camera.near == 0.01
    assert camera.far == 100


def test_build_dexarena_base_camera_allows_task_specific_pose() -> None:
    from utils.camera_configs import build_dexarena_base_camera

    camera = build_dexarena_base_camera(
        eye=[0.55, -0.55, 0.45],
        target=[0.04, 0.0, 0.08],
        width=160,
        height=120,
        fov_deg=62,
    )

    assert camera.uid == "base_camera"
    assert camera.width == 160
    assert camera.height == 120
    assert math.isclose(camera.fov, np.deg2rad(62))
    assert camera.pose is not None


def test_fixed_base_camera_profiles_use_the_accepted_projection() -> None:
    from utils.camera_configs import (
        DEXARENA_FIXED_BASE_CAMERA_CONFIGS,
        build_dexarena_fixed_base_camera,
    )

    assert tuple(DEXARENA_FIXED_BASE_CAMERA_CONFIGS) == (
        "office",
        "cabinet",
        "package_opening",
    )
    for scene, profile in DEXARENA_FIXED_BASE_CAMERA_CONFIGS.items():
        camera = build_dexarena_fixed_base_camera(scene)
        assert (camera.width, camera.height) == (256, 256)
        assert camera.fov == pytest.approx(math.pi / 2)
        assert profile.eye_in_world_m[2] == pytest.approx(1.08)


def test_fixed_wrist_camera_reuses_the_marvin_extrinsics() -> None:
    from utils.camera_configs import (
        DEXARENA_FIXED_WRIST_CAMERA_CONFIG,
        SHARPA_WRIST_CAMERA_V1,
    )

    fixed = DEXARENA_FIXED_WRIST_CAMERA_CONFIG
    assert fixed.eye_in_parent_m == SHARPA_WRIST_CAMERA_V1.eye_in_parent_m
    assert fixed.forward_in_parent == SHARPA_WRIST_CAMERA_V1.forward_in_parent
    assert fixed.image_up_in_parent == SHARPA_WRIST_CAMERA_V1.image_up_in_parent
    assert (fixed.width, fixed.height, fixed.vertical_fov_deg) == (256, 256, 90.0)

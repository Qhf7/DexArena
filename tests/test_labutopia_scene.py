"""Lab asset integrity and opt-in integration; GPU smoke requires prepared assets."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fetcher():
    spec = importlib.util.spec_from_file_location(
        "lab_fetch",
        ROOT / "tasks/scene_builders/fetch_labutopia_scene.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verified_cache_does_not_require_network(fetcher, tmp_path, monkeypatch):
    path = tmp_path / "room.usd"
    path.write_bytes(b"cached room data")
    monkeypatch.setattr(
        fetcher, "urlopen", lambda *a, **k: pytest.fail("unexpected network")
    )
    assert fetcher.fetch_file("room.usd", fetcher.sha256(path), tmp_path, {}) == path


def test_corrupt_download_does_not_replace_existing_file(
    fetcher, tmp_path, monkeypatch
):
    from io import BytesIO

    path = tmp_path / "room.usd"
    path.write_bytes(b"previous room")
    monkeypatch.setattr(
        fetcher, "urlopen", lambda *a, **k: BytesIO(b"corrupt download")
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetcher.fetch_file(
            "room.usd", "0" * 64, tmp_path, {"revision": "test", "directory": "assets"}
        )
    assert path.read_bytes() == b"previous room"
    assert not (tmp_path / "room.usd.part").exists()


def test_sources_are_pinned_and_do_not_fetch_task_instruments(fetcher):
    manifest = json.loads(fetcher.MANIFEST.read_text())
    assert len(manifest["revision"]) == 40
    assert manifest["room_prim"] == "/World/lab_015"
    for path, digest in manifest["sha256"].items():
        assert not Path(path).is_absolute() and ".." not in Path(path).parts
        assert len(digest) == 64
        assert (
            path in ("lab_003.usd", "SubUSDs/lab_015.usd")
            or "/materials/" in path
            or "/textures/" in path
        )


def test_lab_is_opt_in_without_changing_published_matrix():
    from utils.dex_scene_eval import (
        AVAILABLE_BACKGROUND_IDS,
        BACKGROUND_IDS,
        background_manifest,
    )
    from utils.scene_matrix import matrix

    assert "labutopia_lab003" in AVAILABLE_BACKGROUND_IDS
    assert "labutopia_lab003" not in BACKGROUND_IDS
    assert len(matrix()) == 90
    manifest = background_manifest("labutopia_lab003")
    assert manifest["mode"] == "prepared_workspace"
    assert manifest["native_height_delta_m"] == 0


GPU = pytest.mark.skipif(
    os.environ.get("ROBODEX_TEST_LABUTOPIA") != "1",
    reason="set ROBODEX_TEST_LABUTOPIA=1 on a GPU host with prepared lab/task assets",
)


@GPU
def test_room_has_no_geometry_on_the_task_table():
    import numpy as np

    from tasks.scene_builders.backgrounds import GLB_SCENE_DIR
    from tasks.scene_builders.labutopia import ROOM_OFFSET, TABLE_HEIGHT

    records = json.loads(
        (GLB_SCENE_DIR / "labutopia_lab003/room_geometry.json").read_text()
    )
    for record in records:
        bounds = np.asarray(record["bounds"])
        bounds[:, :2] *= -1  # room yaw is pi
        lo, hi = bounds.min(axis=0) + ROOM_OFFSET, bounds.max(axis=0) + ROOM_OFFSET
        above_table = lo[2] >= TABLE_HEIGHT - 0.02 and lo[2] < TABLE_HEIGHT + 0.7
        intersects_table = (
            hi[0] > -0.44 and lo[0] < 0.78 and hi[1] > -1.22 and lo[1] < 1.21
        )
        assert not (above_table and intersects_table), record["path"]


@GPU
def test_pipette_support_steps_and_repeatable_resets():
    import gymnasium as gym
    import numpy as np

    import tasks  # noqa: F401
    from utils.dex_scene_checks import inspect_native_scene
    from utils.dex_scene_eval import make_scene_eval_env, registered_task_settings

    env_id = "PipetteTransferAspirateLiquid-Sharpa-v1"
    settings = registered_task_settings(env_id)
    baseline = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        sim_backend="physx_cpu",
        robot_uids=settings["robot_uids"],
        control_mode="pd_joint_pos",
    )
    try:
        baseline.reset(seed=17)
        expected_pose = (
            baseline.unwrapped.agent.robot.pose.raw_pose.cpu().numpy().copy()
        )
        expected_qpos = baseline.unwrapped.agent.robot.qpos.cpu().numpy().copy()
        expected_cup = baseline.unwrapped.sample_cup.pose.raw_pose.cpu().numpy().copy()
    finally:
        baseline.close()
    env = make_scene_eval_env(
        env_id, scene_background="labutopia_lab003", obs_mode="rgb"
    )
    try:
        for seed in (17, 18, 17):
            obs, _ = env.reset(seed=seed)
            u = env.unwrapped
            assert u.test_tube is None and u.tube_rack is None
            assert len(u._native_scene_entities) == 2  # room + empty work table
            checks = inspect_native_scene(env)
            assert checks["errors"] == []
            if seed == 17:
                np.testing.assert_allclose(
                    u.agent.robot.pose.raw_pose.cpu(), expected_pose, atol=1e-6
                )
                np.testing.assert_allclose(
                    u.agent.robot.qpos.cpu(), expected_qpos, atol=1e-6
                )
                np.testing.assert_allclose(
                    u.sample_cup.pose.raw_pose.cpu(), expected_cup, atol=1e-6
                )
            for sensor in obs["sensor_data"].values():
                assert sensor["rgb"].float().std().item() > 1
            action = u.agent.robot.qpos.clone()
            for _ in range(10):
                env.step(action)
            assert inspect_native_scene(env)["errors"] == []
            assert np.isfinite(u.agent.robot.qpos.cpu()).all()
    finally:
        env.close()

"""GR00T modality configs for the dexterous-hand tasks (utils/groot_utils/modality_configs.py).

The dataset-side split (meta/modality.json) is generated from the task table, so every registered
subtask has one, its joint groups match the task's layout, and the video keys follow the camera
rig: one wrist view for a single-arm task, one per arm for a dual-arm task. The GR00T-side
ModalityConfig objects need gr00t installed and are not built here.
"""
from __future__ import annotations

import importlib

import pytest

from utils import dex_tasks
from utils.groot_utils import modality_configs as M


def test_every_dex_subtask_has_a_modality_json_matching_its_layout() -> None:
    for task in dex_tasks.DEX_TASKS:
        assert task.env_id in M.MODALITY_JSON
        entry = M.MODALITY_JSON[task.env_id]
        groups = {name: (start, end) for name, start, end in task.group_slices}
        assert {k: (v["start"], v["end"]) for k, v in entry["state"].items()} == groups
        assert {k: (v["start"], v["end"]) for k, v in entry["action"].items()} == groups
        assert max(end for _, end in groups.values()) == task.action_dim
        assert entry["annotation"] == {"human.task_description": {"original_key": "task_index"}}


def test_video_keys_follow_the_rig_one_wrist_per_arm() -> None:
    single = M.MODALITY_JSON["NutBoltTurnNut-Sharpa-v1"]["video"]
    assert list(single) == ["head", "wrist"]
    assert single["head"]["original_key"] == "observation.images.base_0_rgb"
    assert single["wrist"]["original_key"] == "observation.images.left_wrist_0_rgb"
    dual = M.MODALITY_JSON["PackageOpeningOpenPackage-Sharpa-v1"]["video"]
    assert list(dual) == ["head", "left_wrist", "right_wrist"]
    assert dual["right_wrist"]["original_key"] == "observation.images.right_wrist_0_rgb"


def test_dual_arm_task_uses_the_58_joint_layout() -> None:
    entry = M.MODALITY_JSON["PackageOpeningOpenPackage-Sharpa-v1"]
    assert list(entry["state"]) == ["left_arm", "right_arm", "left_hand", "right_hand"]
    assert entry["state"]["right_hand"] == {"start": 36, "end": 58}


def test_groot_video_keys_override_replaces_the_rig_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    # The package family dataset keeps the base view and the right wrist only.
    monkeypatch.setenv("GROOT_VIDEO_KEYS", "head:base_0_rgb,right_wrist:right_wrist_0_rgb")
    reloaded = importlib.reload(M)
    try:
        entry = reloaded.MODALITY_JSON["PackageOpeningOpenPackage-Sharpa-v1"]["video"]
        assert list(entry) == ["head", "right_wrist"]
        assert entry["right_wrist"]["original_key"] == "observation.images.right_wrist_0_rgb"
        # the override is global: single-arm entries change too, which is why it is only set
        # for the package family's training and evaluation processes
        assert list(reloaded.MODALITY_JSON["NutBoltTurnNut-Sharpa-v1"]["video"]) == ["head", "right_wrist"]
    finally:
        monkeypatch.delenv("GROOT_VIDEO_KEYS")
        importlib.reload(M)
    assert list(M.MODALITY_JSON["NutBoltTurnNut-Sharpa-v1"]["video"]) == ["head", "wrist"]


def test_write_modality_json_round_trips(tmp_path) -> None:
    out = M.write_modality_json("DrawerPlugOpenDrawer-Sharpa-v1", tmp_path)
    import json
    assert json.loads(out.read_text()) == M.MODALITY_JSON["DrawerPlugOpenDrawer-Sharpa-v1"]
    with pytest.raises(KeyError):
        M.write_modality_json("NotATask-v1", tmp_path)

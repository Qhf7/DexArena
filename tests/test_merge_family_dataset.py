"""Merging a family's subtask datasets into one long-horizon training set.

utils/groot_utils/merge_family_dataset.py concatenates the per-subtask LeRobot v2.1 datasets of
a family, renumbering episodes, frames and the task table, and records where each source landed
(meta/robodex_family.json). scripts/dex/ops/verify_family_labels.py then checks, source by
source, that every merged episode still carries the label the instruction table assigns it.
The fixtures reproduce the layout the renderer writes (read off the real datasets).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.dex_long_tasks import resolve_long_task
from utils.dex_tasks import assign_dataset_instructions, dex_instructions, resolve_dex_task

REPO = Path(__file__).resolve().parents[1]
FAMILY = "pipette"           # two single-arm subtasks, the smallest family
EPISODES = 8                 # the table assigns 2 episodes to each of the 4 phrasings
FRAMES = 3
VIDEO_KEYS = ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb")


def _write_subtask_dataset(root: Path, env_id: str) -> None:
    """A per-episode-labelled v2.1 dataset as replay_to_vla_dataset.py + relabel write it."""
    labels = assign_dataset_instructions(env_id, range(EPISODES))
    table = {name: i for i, name in enumerate(dex_instructions(env_id))}
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    for key in VIDEO_KEYS:
        (root / "videos" / "chunk-000" / key).mkdir(parents=True)
    (root / "meta" / "tasks.jsonl").write_text("".join(
        json.dumps({"task_index": i, "task": name}) + "\n" for name, i in table.items()))
    with (root / "meta" / "episodes.jsonl").open("w") as f, (root / "meta" / "episodes_stats.jsonl").open("w") as g:
        for ep in range(EPISODES):
            f.write(json.dumps({"episode_index": ep, "tasks": [labels[ep]], "length": FRAMES}) + "\n")
            g.write(json.dumps({"episode_index": ep, "stats": {"action": {"mean": [0.0] * 29}}}) + "\n")
    frame_base = 0
    for ep in range(EPISODES):
        pq.write_table(pa.table({
            "observation.state": pa.array([[float(ep)] * 29] * FRAMES, pa.list_(pa.float32())),
            "action": pa.array([[float(ep)] * 29] * FRAMES, pa.list_(pa.float32())),
            "episode_index": pa.array([ep] * FRAMES, pa.int64()),
            "frame_index": pa.array(range(FRAMES), pa.int64()),
            "index": pa.array(range(frame_base, frame_base + FRAMES), pa.int64()),
            "task_index": pa.array([table[labels[ep]]] * FRAMES, pa.int64()),
        }), root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        frame_base += FRAMES
        for key in VIDEO_KEYS:
            (root / "videos" / "chunk-000" / key / f"episode_{ep:06d}.mp4").write_bytes(b"video:%s:%d" % (env_id.encode(), ep))
    features = {key: {"dtype": "video", "shape": [256, 256, 3]} for key in VIDEO_KEYS}
    features.update({"observation.state": {"dtype": "float32", "shape": [29]}, "action": {"dtype": "float32", "shape": [29]}})
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "fps": 30, "features": features, "total_episodes": EPISODES,
        "total_frames": frame_base, "total_tasks": len(table), "total_videos": EPISODES * len(VIDEO_KEYS),
        "total_chunks": 1, "chunks_size": 1000, "splits": {"train": f"0:{EPISODES}"}}, indent=4))
    (root / "meta" / "modality.json").write_text(json.dumps({"state": {"single_arm": {"start": 0, "end": 7}}}))
    (root / "meta" / "stats.json").write_text("{}")
    (root / "meta" / "robodex_render.json").write_text(json.dumps(
        {"env_id": env_id, "instruction_assignment": "per_episode"}))


@pytest.fixture
def dex_out(tmp_path: Path) -> Path:
    long = resolve_long_task(FAMILY)
    for stage in long.stages:
        task = resolve_dex_task(stage.subtask)
        _write_subtask_dataset(tmp_path / "datasets" / f"{task.short_name}_rig_vla", task.env_id)
    return tmp_path


def _run(script: str, dex_out: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / script), *args], cwd=REPO, capture_output=True,
                          text=True, env={**os.environ, "DEX_OUT": str(dex_out), "PYTHONPATH": str(REPO)})


def test_merge_concatenates_the_family_and_the_labels_survive_verification(dex_out: Path) -> None:
    merged = _run("utils/groot_utils/merge_family_dataset.py", dex_out, "--family", FAMILY, "--out-tag", "v4")
    assert merged.returncode == 0, merged.stdout + merged.stderr
    root = dex_out / "datasets" / f"{FAMILY}_family_rig_v4_vla"
    stages = resolve_long_task(FAMILY).stages
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == EPISODES * len(stages)
    assert info["total_frames"] == EPISODES * FRAMES * len(stages)
    assert info["total_tasks"] == 4 * len(stages), "every phrasing of every subtask is a distinct instruction"
    assert info["splits"] == {"train": f"0:{EPISODES * len(stages)}"}

    manifest = json.loads((root / "meta" / "robodex_family.json").read_text())
    assert manifest["instruction_assignment"] == "per_episode"
    assert [r["env_id"] for r in manifest["source_ranges"]] == [s.subtask for s in stages]
    assert [r["first_episode"] for r in manifest["source_ranges"]] == [0, EPISODES]

    # frames are renumbered globally and the per-frame task_index points at the right label
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (root / "meta" / "tasks.jsonl").open()}
    frames = pd.concat(pd.read_parquet(p) for p in sorted((root / "data" / "chunk-000").glob("*.parquet")))
    assert frames["index"].tolist() == list(range(len(frames)))
    assert sorted(frames["episode_index"].unique()) == list(range(EPISODES * len(stages)))
    expected = {}
    for src in manifest["source_ranges"]:
        labels = assign_dataset_instructions(src["env_id"], range(src["episodes"]))
        for i in range(src["episodes"]):
            expected[src["first_episode"] + i] = labels[i]
    for ep, group in frames.groupby("episode_index"):
        assert {tasks[int(t)] for t in group["task_index"]} == {expected[int(ep)]}
    # videos travel with their episode
    assert (root / "videos" / "chunk-000" / VIDEO_KEYS[0] / f"episode_{EPISODES:06d}.mp4").read_bytes().startswith(
        b"video:" + stages[1].subtask.encode())

    verified = _run("scripts/dex/ops/verify_family_labels.py", dex_out, "--dataset", str(root))
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert "[ok]" in verified.stdout


def test_merge_refuses_to_overwrite_and_a_corrupted_label_fails_verification(dex_out: Path) -> None:
    first = _run("utils/groot_utils/merge_family_dataset.py", dex_out, "--family", FAMILY)
    assert first.returncode == 0, first.stdout + first.stderr
    again = _run("utils/groot_utils/merge_family_dataset.py", dex_out, "--family", FAMILY)
    assert again.returncode != 0 and "--overwrite" in (again.stdout + again.stderr)

    root = dex_out / "datasets" / f"{FAMILY}_family_rig_vla"
    lines = (root / "meta" / "episodes.jsonl").read_text().splitlines()
    rec = json.loads(lines[0]); rec["tasks"] = ["do something else"]
    lines[0] = json.dumps(rec)
    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    verified = _run("scripts/dex/ops/verify_family_labels.py", dex_out, "--dataset", str(root))
    assert verified.returncode == 1
    assert "unregistered" in verified.stdout

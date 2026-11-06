"""Relabelling a rendered dataset's episodes in place.

The fixtures reproduce the layout of the real rendered datasets, read off
`cabinet_pour_bottle_pick_place_fixcam_vla{,_v3.0}` on 2026-09-19: v3.0 keeps the
instruction in `meta/tasks.parquet`'s pandas index column, v2.1 in `meta/tasks.jsonl`,
and both carry a per-episode `tasks` list plus a per-frame `task_index`.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.dex_tasks import assign_dataset_instructions, dex_instructions
from utils.groot_utils.relabel_instructions import main as relabel_main

ENV_ID = "CabinetPouringPickBottle-Sharpa-v1"
EPISODES = 8
FRAMES_PER_EPISODE = 3


def _episode_frames() -> tuple[list[int], list[int]]:
    episodes = [index for index in range(EPISODES) for _ in range(FRAMES_PER_EPISODE)]
    return episodes, [0] * len(episodes)


def _make_v30(root: Path, instruction: str) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    # As LeRobot writes it: the instruction is the DataFrame's unnamed pandas index and
    # task_index its only column; DataFrame.to_parquet stores the index as pandas metadata.
    pd.DataFrame(
        {"task_index": pd.array([0], dtype="int64")}, index=pd.Index([instruction])
    ).to_parquet(root / "meta" / "tasks.parquet")
    pq.write_table(
        pa.table({
            "episode_index": pa.array(range(EPISODES), pa.int64()),
            "tasks": pa.array([[instruction]] * EPISODES, pa.list_(pa.string())),
            "length": pa.array([FRAMES_PER_EPISODE] * EPISODES, pa.int64()),
        }),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    episodes, task_index = _episode_frames()
    pq.write_table(
        pa.table({"episode_index": pa.array(episodes, pa.int64()),
                  "task_index": pa.array(task_index, pa.int64())}),
        root / "data" / "chunk-000" / "file-000.parquet",
    )


def _make_v21(root: Path, instruction: str) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": instruction}) + "\n"
    )
    (root / "meta" / "episodes.jsonl").write_text("\n".join(
        json.dumps({"episode_index": index, "tasks": [instruction],
                    "length": FRAMES_PER_EPISODE})
        for index in range(EPISODES)
    ) + "\n")
    for index in range(EPISODES):
        pq.write_table(
            pa.table({
                "episode_index": pa.array([index] * FRAMES_PER_EPISODE, pa.int64()),
                "task_index": pa.array([0] * FRAMES_PER_EPISODE, pa.int64()),
            }),
            root / "data" / "chunk-000" / f"episode_{index:06d}.parquet",
        )


def _run(root: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["relabel_instructions.py", "--dataset", str(root), "--env_id", ENV_ID, *extra],
    )
    relabel_main()


def _v30_labels(root: Path) -> dict[int, str]:
    # Read it the way LeRobot does; a table whose index did not survive is unusable to it.
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    assert all(isinstance(value, str) for value in tasks.index), (
        "LeRobot reads the instructions from the pandas index; it came back as integers"
    )
    assert list(tasks.columns) == ["task_index"]
    by_index = dict(zip(tasks["task_index"].tolist(), tasks.index.tolist()))
    episodes = pq.read_table(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()
    labels = {
        index: entry[0]
        for index, entry in zip(episodes["episode_index"], episodes["tasks"])
    }
    frames = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    for episode, task_index in zip(frames["episode_index"], frames["task_index"]):
        assert by_index[task_index] == labels[episode], "frame label disagrees with episode"
    return labels


def _v21_labels(root: Path) -> dict[int, str]:
    tasks = [json.loads(line) for line in
             (root / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
    by_index = {entry["task_index"]: entry["task"] for entry in tasks}
    labels = {}
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        labels[record["episode_index"]] = record["tasks"][0]
    for path in sorted((root / "data").rglob("*.parquet")):
        frames = pq.read_table(path).to_pydict()
        for episode, task_index in zip(frames["episode_index"], frames["task_index"]):
            assert by_index[task_index] == labels[episode], "frame label disagrees"
    return labels


@pytest.mark.parametrize("build,read", [(_make_v30, _v30_labels), (_make_v21, _v21_labels)])
def test_relabel_spreads_the_instructions_across_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build, read
) -> None:
    root = tmp_path / "dataset"
    build(root, dex_instructions(ENV_ID)[0])
    assert len(set(read(root).values())) == 1, "fixture starts with one instruction"

    _run(root, monkeypatch)

    labels = read(root)
    assert labels == assign_dataset_instructions(ENV_ID, range(EPISODES))
    assert set(Counter(labels.values())) == set(dex_instructions(ENV_ID))


@pytest.mark.parametrize("build,read", [(_make_v30, _v30_labels), (_make_v21, _v21_labels)])
def test_dry_run_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build, read
) -> None:
    root = tmp_path / "dataset"
    canonical = dex_instructions(ENV_ID)[0]
    build(root, canonical)
    _run(root, monkeypatch, "--dry-run")
    assert set(read(root).values()) == {canonical}
    assert not (root / "meta" / "robodex_render.json").exists()


def test_v30_tasks_table_round_trips_through_pandas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first version wrote tasks.parquet as a bare arrow table. Every column came back,
    but without pandas metadata the instruction strings stopped being the index and LeRobot
    could no longer map a frame's task_index to its instruction -- PI0.5 SFT failed with
    "Cannot extract task strings from LeRobot dataset metadata" on the first relabelled
    dataset it was pointed at."""
    root = tmp_path / "dataset"
    _make_v30(root, dex_instructions(ENV_ID)[0])
    _run(root, monkeypatch)
    table = pq.read_table(root / "meta" / "tasks.parquet")
    assert b"pandas" in (table.schema.metadata or {}), "index is only restored from pandas metadata"
    frame = pd.read_parquet(root / "meta" / "tasks.parquet")
    assert frame.index.tolist() == list(dex_instructions(ENV_ID))
    assert frame["task_index"].tolist() == [0, 1, 2, 3]
    assert frame.index.name is None, "LeRobot's index is unnamed"


def test_relabel_repairs_a_table_written_without_pandas_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The datasets relabelled by the first version are in exactly this state; re-running
    the tool has to put the index back rather than fail on the odd layout."""
    root = tmp_path / "dataset"
    _make_v30(root, dex_instructions(ENV_ID)[0])
    pq.write_table(
        pa.table({"task_index": pa.array(range(4), pa.int64()),
                  "__index_level_0__": pa.array(list(dex_instructions(ENV_ID)), pa.string())}),
        root / "meta" / "tasks.parquet",
    )
    broken = pd.read_parquet(root / "meta" / "tasks.parquet")
    assert broken.index.tolist() == [0, 1, 2, 3], "fixture reproduces the broken layout"
    _run(root, monkeypatch)
    repaired = pd.read_parquet(root / "meta" / "tasks.parquet")
    assert repaired.index.tolist() == list(dex_instructions(ENV_ID))
    assert list(repaired.columns) == ["task_index"]


def test_relabel_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    _make_v30(root, dex_instructions(ENV_ID)[0])
    _run(root, monkeypatch)
    first = _v30_labels(root)
    _run(root, monkeypatch)
    assert _v30_labels(root) == first


def test_relabel_records_what_it_did(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    _make_v30(root, dex_instructions(ENV_ID)[0])
    (root / "meta" / "robodex_render.json").write_text(
        json.dumps({"env_id": ENV_ID, "camera_profile": "robodex_fixed",
                    "instruction": dex_instructions(ENV_ID)[0], "instruction_index": 0}) + "\n"
    )
    _run(root, monkeypatch)
    manifest = json.loads((root / "meta" / "robodex_render.json").read_text())
    assert manifest["instruction_assignment"] == "per_episode"
    assert manifest["instructions"] == list(dex_instructions(ENV_ID))
    assert manifest["relabelled_from"] == "single_instruction"
    # The single-instruction fields must not survive: they would now be a lie.
    assert "instruction" not in manifest and "instruction_index" not in manifest
    # Unrelated render provenance is preserved.
    assert manifest["camera_profile"] == "robodex_fixed"


def test_an_unregistered_env_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    _make_v30(root, dex_instructions(ENV_ID)[0])
    monkeypatch.setattr(
        "sys.argv",
        ["relabel_instructions.py", "--dataset", str(root), "--env_id", "NoSuchTask-v1"],
    )
    with pytest.raises(KeyError, match="utils/dex_tasks.py"):
        relabel_main()


def test_a_directory_that_is_not_a_dataset_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        ["relabel_instructions.py", "--dataset", str(root), "--env_id", ENV_ID],
    )
    with pytest.raises(SystemExit, match="neither a v3.0 nor a v2.1"):
        relabel_main()

"""Check that a rendered dataset's labels are what utils/dex_tasks.py assigns.

Reads the three places a label lives (the tasks table, the per-episode metadata, the
per-frame task_index) and checks they agree with each other and with
``dataset_episode_instruction`` for every episode. Exit 0 if so, 1 with a reason if not.

    python scripts/dex/ops/verify_labels.py --dataset <root> --env_id <ENV>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dex_tasks import assign_dataset_instructions, dex_instructions  # noqa: E402


def _labels_v30(root: Path) -> tuple[dict[int, str], dict[int, str]]:
    # Read the tasks table the way LeRobot does: the instruction strings must come back as
    # the DataFrame index, or a frame's task_index cannot be mapped to its instruction.
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    if not all(isinstance(value, str) for value in tasks.index):
        raise SystemExit(
            f"[FAIL] {root.name} (v3.0): meta/tasks.parquet does not carry the instruction "
            "strings as its pandas index, which is where LeRobot reads them from -- the table "
            "was written without pandas metadata. Re-run relabel_instructions.py."
        )
    by_index = dict(zip(tasks["task_index"].tolist(), tasks.index.tolist()))
    episodes: dict[int, str] = {}
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "tasks"]).to_pydict()
        for index, entry in zip(table["episode_index"], table["tasks"]):
            episodes[int(index)] = entry[0]
    frames: dict[int, str] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"]).to_pydict()
        for episode, task_index in set(zip(table["episode_index"], table["task_index"])):
            label = by_index[task_index]
            if frames.setdefault(int(episode), label) != label:
                raise SystemExit(f"episode {episode} has frames with two different task_index values")
    return episodes, frames


def _labels_v21(root: Path) -> tuple[dict[int, str], dict[int, str]]:
    by_index = {}
    for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            by_index[record["task_index"]] = record["task"]
    episodes = {}
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            episodes[int(record["episode_index"])] = record["tasks"][0]
    frames: dict[int, str] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"]).to_pydict()
        for episode, task_index in set(zip(table["episode_index"], table["task_index"])):
            label = by_index[task_index]
            if frames.setdefault(int(episode), label) != label:
                raise SystemExit(f"episode {episode} has frames with two different task_index values")
    return episodes, frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--env_id", required=True)
    args = ap.parse_args()
    root = Path(args.dataset)

    if (root / "meta" / "tasks.parquet").is_file():
        layout, (episodes, frames) = "v3.0", _labels_v30(root)
    elif (root / "meta" / "tasks.jsonl").is_file():
        layout, (episodes, frames) = "v2.1", _labels_v21(root)
    else:
        raise SystemExit(f"{root}: not a LeRobot dataset")

    expected = assign_dataset_instructions(args.env_id, sorted(episodes))
    problems = []
    if frames != episodes:
        off = sorted(index for index in episodes if frames.get(index) != episodes[index])
        problems.append(f"per-frame task_index disagrees with episode metadata on {len(off)} episodes: {off[:5]}")
    if episodes != expected:
        off = sorted(index for index in episodes if episodes[index] != expected[index])
        problems.append(f"labels differ from the table's assignment on {len(off)} episodes: {off[:5]}")
    counts = Counter(episodes.values())
    unregistered = sorted(set(counts) - set(dex_instructions(args.env_id)))
    if unregistered:
        problems.append(f"unregistered instructions present: {unregistered}")

    split = "/".join(str(counts.get(text, 0)) for text in dex_instructions(args.env_id))
    name = root.name
    if problems:
        print(f"[FAIL] {name} ({layout}, {len(episodes)} eps, split {split})")
        for problem in problems:
            print(f"       {problem}")
        raise SystemExit(1)
    print(f"[ok]   {name} ({layout}, {len(episodes)} eps, split {split})")


if __name__ == "__main__":
    main()

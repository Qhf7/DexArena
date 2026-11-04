"""Check a merged family dataset's labels against the table, source by source.

A family dataset concatenates its subtasks' datasets (utils/groot_utils/merge_family_dataset.py,
build_package_family_dataset.py); meta/robodex_family.json records which merged episodes came
from which subtask (`source_ranges`). For every source this checks that each episode's label is
one of that subtask's registered instructions, that the label the merged tasks table gives a
frame agrees with the episode's, and that the split is the balanced one the table assigns
(25/25/25/25 for 100 episodes). Exit 0 if so, 1 with a reason if not.

    python scripts/dex/ops/verify_family_labels.py --dataset <root>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dex_tasks import assign_dataset_instructions, dex_instructions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    root = Path(args.dataset)
    manifest = json.loads((root / "meta" / "robodex_family.json").read_text())
    ranges = manifest.get("source_ranges")
    if not ranges:
        raise SystemExit(f"[FAIL] {root.name}: robodex_family.json has no source_ranges (merged by the old tool?)")

    by_index = {}
    for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line); by_index[record["task_index"]] = record["task"]
    episodes: dict[int, str] = {}
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line); episodes[int(record["episode_index"])] = record["tasks"][0]
    frames: dict[int, str] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"]).to_pydict()
        for episode, task_index in set(zip(table["episode_index"], table["task_index"])):
            label = by_index[task_index]
            if frames.setdefault(int(episode), label) != label:
                raise SystemExit(f"[FAIL] {root.name}: episode {episode} has frames with two task_index values")

    problems, splits = [], []
    for source in ranges:
        env_id, first, count = source["env_id"], int(source["first_episode"]), int(source["episodes"])
        registered = dex_instructions(env_id)
        # The source's own episode indices were 0..count-1 before the merge, so the table's
        # assignment for source episode i is what merged episode first+i must carry.
        expected = assign_dataset_instructions(env_id, range(count))
        labels = {i: episodes.get(first + i) for i in range(count)}
        unregistered = sorted({l for l in labels.values() if l not in registered})
        if unregistered:
            problems.append(f"{env_id}: unregistered instructions {unregistered}")
        off = [i for i in range(count) if labels[i] != expected[i]]
        if off:
            problems.append(f"{env_id}: {len(off)} episodes differ from the table's assignment (first: {off[:5]})")
        disagree = [i for i in range(count) if frames.get(first + i) != labels[i]]
        if disagree:
            problems.append(f"{env_id}: per-frame task_index disagrees with episode metadata on {len(disagree)} episodes")
        counts = Counter(labels.values())
        splits.append(env_id.split("-")[0] + " " + "/".join(str(counts.get(t, 0)) for t in registered))

    total = sum(int(s["episodes"]) for s in ranges)
    if total != len(episodes):
        problems.append(f"source_ranges cover {total} episodes but the dataset has {len(episodes)}")
    if problems:
        print(f"[FAIL] {root.name} ({len(episodes)} eps): " + "; ".join(splits))
        for problem in problems:
            print(f"       {problem}")
        raise SystemExit(1)
    print(f"[ok]   {root.name} ({len(episodes)} eps, assignment={manifest.get('instruction_assignment')}): " + "; ".join(splits))


if __name__ == "__main__":
    main()

"""Relabel a rendered dataset's episodes with per-episode instructions.

The datasets rendered earlier label every episode with one instruction, because
that is what the renderer wrote. The protocol since then gives each EPISODE one of the
task's registered instructions (utils/dex_tasks.py), so a 100-episode dataset carries all
four phrasings and training sees every one of them.

Nothing about the recording changes -- the videos, states and actions are already correct
and a re-render would reproduce them byte for byte. Only the language labels move, and they
live entirely in metadata plus one integer column:

    v3.0 (PI0.5)   meta/tasks.parquet            task_index -> instruction
                   meta/episodes/**/*.parquet    per-episode `tasks` list
                   data/**/*.parquet             per-frame `task_index`
    v2.1 (GR00T)   meta/tasks.jsonl              one JSON object per instruction
                   meta/episodes.jsonl           per-episode `tasks` list
                   data/**/episode_*.parquet     per-frame `task_index`

So this rewrites those files in place (atomically, file by file) instead of spending a day
of render time per task. Run with --dry-run first: it prints the assignment it would write
and touches nothing.

Usage:
    python utils/groot_utils/relabel_instructions.py --dataset <root> --env_id <ENV> --dry-run
    python utils/groot_utils/relabel_instructions.py --dataset <root> --env_id <ENV>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dex_tasks import (  # noqa: E402
    DEX_INSTRUCTIONS_VERSION,
    assign_dataset_instructions,
    resolve_dex_task,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, help="dataset root (v3.0 or v2.1)")
    ap.add_argument("--env_id", required=True, help="registered dex task (utils/dex_tasks.py)")
    ap.add_argument("--dry-run", action="store_true", help="print the assignment, write nothing")
    return ap.parse_args()


def _write_parquet_atomic(table: pa.Table, path: Path) -> None:
    """Replace `path` only once the new file is complete, so a crash cannot truncate it."""
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, path)


def _write_dataframe_atomic(frame: "pd.DataFrame", path: Path) -> None:
    """Like _write_parquet_atomic, through pandas so the index is stored as pandas metadata."""
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def _write_text_atomic(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _read_episode_lengths_v30(root: Path) -> dict[int, int]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No v3.0 episode metadata under {root / 'meta' / 'episodes'}.")
    table = pq.read_table(files, columns=["episode_index", "length"]).to_pydict()
    return dict(zip(table["episode_index"], table["length"]))


def _relabel_v30(root: Path, assignment: dict[int, str], instructions: list[str],
                 *, dry_run: bool) -> None:
    index_of = {text: position for position, text in enumerate(instructions)}

    # meta/tasks.parquet: LeRobot v3.0 keeps the instruction strings as the DataFrame's
    # (unnamed) pandas index and `task_index` as its one column, and reads it back with
    # pd.read_parquet. That index survives only through the pandas metadata DataFrame.to_parquet
    # stores in the file, so the table has to be written as a DataFrame -- written as a bare
    # arrow table the strings come back as an ordinary column, the index becomes 0..n-1, and
    # LeRobot can no longer map a frame's task_index to its instruction.
    tasks_path = root / "meta" / "tasks.parquet"
    existing = pd.read_parquet(tasks_path)
    if "task_index" not in existing.columns:
        raise SystemExit(f"{tasks_path} has no task_index column; refusing to guess its layout.")
    new_tasks = pd.DataFrame(
        {"task_index": pd.array(range(len(instructions)), dtype=existing["task_index"].dtype)},
        index=pd.Index(instructions, name=existing.index.name),
    )

    # meta/episodes/**: the per-episode `tasks` list.
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    episode_tables = []
    for path in episode_files:
        table = pq.read_table(path)
        indices = table.column("episode_index").to_pylist()
        tasks = pa.array([[assignment[int(index)]] for index in indices],
                         type=table.schema.field("tasks").type)
        episode_tables.append((path, table.set_column(
            table.schema.get_field_index("tasks"), "tasks", tasks)))

    # data/**: the per-frame `task_index`.
    data_files = sorted((root / "data").rglob("*.parquet"))
    data_tables = []
    for path in data_files:
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        field = table.schema.field("task_index")
        task_index = pa.array(
            [index_of[assignment[int(episode)]] for episode in episodes], type=field.type
        )
        data_tables.append((path, table.set_column(
            table.schema.get_field_index("task_index"), "task_index", task_index)))

    if dry_run:
        print(f"[dry-run] would rewrite {tasks_path.name}, {len(episode_tables)} episode "
              f"metadata file(s) and {len(data_tables)} data file(s)")
        return
    _write_dataframe_atomic(new_tasks, tasks_path)
    for path, table in episode_tables:
        _write_parquet_atomic(table, path)
    for path, table in data_tables:
        _write_parquet_atomic(table, path)
    print(f"[v3.0] rewrote {tasks_path.name}, {len(episode_tables)} episode metadata file(s), "
          f"{len(data_tables)} data file(s)")


def _relabel_v21(root: Path, assignment: dict[int, str], instructions: list[str],
                 *, dry_run: bool) -> None:
    index_of = {text: position for position, text in enumerate(instructions)}

    tasks_text = "".join(
        json.dumps({"task_index": position, "task": text}, ensure_ascii=False) + "\n"
        for position, text in enumerate(instructions)
    )
    episodes_path = root / "meta" / "episodes.jsonl"
    episode_lines = []
    for line in episodes_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["tasks"] = [assignment[int(record["episode_index"])]]
        episode_lines.append(json.dumps(record, ensure_ascii=False))
    episodes_text = "\n".join(episode_lines) + "\n"

    data_files = sorted((root / "data").rglob("*.parquet"))
    data_tables = []
    for path in data_files:
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        field = table.schema.field("task_index")
        task_index = pa.array(
            [index_of[assignment[int(episode)]] for episode in episodes], type=field.type
        )
        data_tables.append((path, table.set_column(
            table.schema.get_field_index("task_index"), "task_index", task_index)))

    if dry_run:
        print(f"[dry-run] would rewrite tasks.jsonl, episodes.jsonl "
              f"({len(episode_lines)} episodes) and {len(data_tables)} data file(s)")
        return
    _write_text_atomic(tasks_text, root / "meta" / "tasks.jsonl")
    _write_text_atomic(episodes_text, episodes_path)
    for path, table in data_tables:
        _write_parquet_atomic(table, path)
    print(f"[v2.1] rewrote tasks.jsonl, episodes.jsonl ({len(episode_lines)} episodes), "
          f"{len(data_tables)} data file(s)")


def main():
    args = parse_args()
    root = Path(args.dataset)
    task = resolve_dex_task(args.env_id)
    instructions = list(task.instructions)

    is_v30 = (root / "meta" / "tasks.parquet").is_file()
    is_v21 = (root / "meta" / "tasks.jsonl").is_file()
    if is_v30 == is_v21:
        raise SystemExit(f"{root} is neither a v3.0 nor a v2.1 LeRobot dataset.")

    if is_v30:
        episode_indices = sorted(_read_episode_lengths_v30(root))
    else:
        episode_indices = sorted(
            json.loads(line)["episode_index"]
            for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        )
    assignment = assign_dataset_instructions(args.env_id, episode_indices)

    counts = Counter(assignment.values())
    print(f"{root} ({'v3.0' if is_v30 else 'v2.1'}): {len(episode_indices)} episodes, "
          f"instructions v{DEX_INSTRUCTIONS_VERSION}")
    for text in instructions:
        print(f"  {counts.get(text, 0):4d}  {text}")
    if counts.get(instructions[0], 0) == len(episode_indices):
        raise SystemExit("Assignment is degenerate (one instruction for every episode).")

    if is_v30:
        _relabel_v30(root, assignment, instructions, dry_run=args.dry_run)
    else:
        _relabel_v21(root, assignment, instructions, dry_run=args.dry_run)

    if args.dry_run:
        return
    # Record what happened next to the dataset, so a relabelled dataset is not mistaken for
    # a freshly rendered one and the assignment can be checked later.
    manifest_path = root / "meta" / "robodex_render.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.pop("instruction", None)
    manifest.pop("instruction_index", None)
    manifest.update({
        "instruction_assignment": "per_episode",
        "instructions": instructions,
        "instructions_version": DEX_INSTRUCTIONS_VERSION,
        "relabelled_from": "single_instruction",
    })
    _write_text_atomic(json.dumps(manifest, indent=2) + "\n", manifest_path)
    print(f"[done] {manifest_path}")


if __name__ == "__main__":
    main()

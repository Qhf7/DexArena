"""Join a family's subtask datasets into one dataset for long-horizon training.

The long-horizon experiment asks for ONE checkpoint per long task, trained on every subtask of
that task together. Each subtask already has its own rendered dataset (100 episodes, the family's
fixed base camera plus the acting arm's wrist camera, 29 absolute joint targets), and within a
family those datasets share every feature: same camera fields, same widths, same fps. They differ
only in the language instruction, which is exactly the signal that has to tell the one policy
which stage it is doing.

So the merge is a concatenation: LeRobot's own `aggregate_datasets` joins the episodes and
renumbers the task table, and the modality config is copied from the first subtask because all of
them carry the same joint layout.

    python utils/groot_utils/merge_family_dataset.py --family juice [--out-tag v4] [--dst ...] [--overwrite]

The language labels are copied per frame from the sources, so a family merged from the
relabelled subtask datasets (each episode carrying one of its task's four registered
instructions, utils/dex_tasks.py) carries them too: `--out-tag v4` names that dataset
`<family>_family_rig_v4_vla`, and meta/robodex_family.json records the assignment and the
episode range each source occupies, so the labels can be verified against the table.

Not usable for package_opening: its OpenPackage subtask is dual-arm (58 dims, three camera
views) while ExtractAndPlaceApple is single-arm (29 dims, two views). Merging those would need a
dual-arm re-render of the single-arm subtask, which is a separate job.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dex_tasks import DEX_TASKS  # noqa: E402

# $DEX_OUT holds datasets/, runs/ and evals/ (docs/GROOT_DEXARENA.md); the merged dataset goes next to its sources.
DEX_OUT = Path(os.environ.get("DEX_OUT", "dex_out"))


def family_tasks(family: str):
    tasks = [t for t in DEX_TASKS if t.short_name.split("_")[0] == family]
    if not tasks:
        raise SystemExit(f"no subtasks for family {family!r}")
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True)
    ap.add_argument("--suffix", default="_rig")
    ap.add_argument("--out-tag", default="",
                    help="suffix for the merged dataset's name, e.g. v4 -> <family>_family_rig_v4_vla")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tasks = family_tasks(args.family)
    srcs, missing, widths = [], [], set()
    for t in tasks:
        for suffix in (args.suffix, "_fixcam", ""):
            d = DEX_OUT / "datasets" / f"{t.short_name}{suffix}_vla"
            if (d / "meta" / "info.json").exists():
                srcs.append((t, d)); break
        else:
            missing.append(t.short_name)
    if missing:
        raise SystemExit(f"missing datasets: {missing}")
    infos = []
    for _, d in srcs:
        info = json.loads((d / "meta" / "info.json").read_text())
        infos.append(info)
        widths.add((tuple(sorted(k for k in info["features"] if k.startswith("observation.images"))),
                    tuple(info["features"]["observation.state"]["shape"]),
                    tuple(info["features"]["action"]["shape"]), info["fps"]))
    if len(widths) > 1:
        raise SystemExit(f"{args.family}: subtasks disagree on cameras/widths/fps: {widths}")

    out_tag = f"_{args.out_tag}" if args.out_tag else ""
    dst = Path(args.dst) if args.dst else DEX_OUT / "datasets" / f"{args.family}_family{args.suffix}{out_tag}_vla"
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)

    import pandas as pd

    print(f"[merge] {args.family}: {len(srcs)} subtasks")
    for (t, d), info in zip(srcs, infos):
        print(f"   {t.short_name:42s} {info['total_episodes']:4d} episodes {info['total_frames']:7d} frames")
    video_keys = sorted(k for k in infos[0]["features"] if k.startswith("observation.images")
                        or k.startswith("next.observation.images"))
    (dst / "data" / "chunk-000").mkdir(parents=True)
    for key in video_keys:
        (dst / "videos" / "chunk-000" / key).mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)

    # Concatenate by copying each episode file under a new index. LeRobot 2.1 keeps the episode
    # index in the file name, in an `episode_index` column, and in meta/episodes.jsonl, and the
    # frame `index` column is global, so all three are renumbered as the episodes are appended.
    ep_out = 0
    frame_base = 0
    tasks_table: dict[str, int] = {}
    episodes_meta, episodes_stats = [], []
    source_ranges = []      # (env_id, first merged episode, count) per source, in merge order
    assignments = set()     # how each source labelled its episodes (meta/robodex_render.json)
    for (t, d), info in zip(srcs, infos):
        ep_first = ep_out
        render = d / "meta" / "robodex_render.json"
        if render.exists():
            assignments.add(json.loads(render.read_text()).get("instruction_assignment", "fixed"))
        src_tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
                     for l in (d / "meta" / "tasks.jsonl").open()}
        stats_by_ep = {json.loads(l)["episode_index"]: json.loads(l)
                       for l in (d / "meta" / "episodes_stats.jsonl").open()} \
            if (d / "meta" / "episodes_stats.jsonl").exists() else {}
        for line in (d / "meta" / "episodes.jsonl").open():
            rec = json.loads(line)
            src_idx = rec["episode_index"]
            frame = pd.read_parquet(d / "data" / "chunk-000" / f"episode_{src_idx:06d}.parquet")
            frame["episode_index"] = ep_out
            frame["index"] = range(frame_base, frame_base + len(frame))
            new_task_idx = []
            for ti in frame["task_index"]:
                name = src_tasks[int(ti)]
                new_task_idx.append(tasks_table.setdefault(name, len(tasks_table)))
            frame["task_index"] = new_task_idx
            frame.to_parquet(dst / "data" / "chunk-000" / f"episode_{ep_out:06d}.parquet", index=False)
            for key in video_keys:
                src_mp4 = d / "videos" / "chunk-000" / key / f"episode_{src_idx:06d}.mp4"
                if src_mp4.exists():
                    shutil.copy(src_mp4, dst / "videos" / "chunk-000" / key / f"episode_{ep_out:06d}.mp4")
            rec["episode_index"] = ep_out
            episodes_meta.append(rec)
            if src_idx in stats_by_ep:
                st = stats_by_ep[src_idx]; st["episode_index"] = ep_out; episodes_stats.append(st)
            frame_base += len(frame)
            ep_out += 1
        source_ranges.append({"env_id": t.env_id, "first_episode": ep_first, "episodes": ep_out - ep_first})
        print(f"   ... {t.short_name}: {ep_out} episodes / {frame_base} frames so far")

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for rec in episodes_meta:
            f.write(json.dumps(rec) + "\n")
    if episodes_stats:
        with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
            for st in episodes_stats:
                f.write(json.dumps(st) + "\n")
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        for name, idx in sorted(tasks_table.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": name}) + "\n")
    info = dict(infos[0])
    info.update(total_episodes=ep_out, total_frames=frame_base, total_tasks=len(tasks_table),
                total_videos=ep_out * len(video_keys), total_chunks=1,
                splits={"train": f"0:{ep_out}"})
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    shutil.copy(srcs[0][1] / "meta" / "modality.json", dst / "meta" / "modality.json")
    if (srcs[0][1] / "meta" / "stats.json").exists():
        shutil.copy(srcs[0][1] / "meta" / "stats.json", dst / "meta" / "stats.json")
    (dst / "meta" / "robodex_family.json").write_text(json.dumps({
        "family": args.family,
        "subtasks": [t.env_id for t, _ in srcs],
        "sources": [str(d) for _, d in srcs],
        "source_ranges": source_ranges,
        # "per_episode" when every source carried per-episode labels (the relabelled datasets);
        # anything else means at least one source still had a single instruction.
        "instruction_assignment": "per_episode" if assignments == {"per_episode"} else "mixed:" + ",".join(sorted(assignments) or ["unknown"]),
        "instructions": sorted(tasks_table, key=tasks_table.get),
    }, indent=2) + "\n")
    print(f"[merge] wrote {ep_out} episodes / {frame_base} frames / {len(tasks_table)} instructions -> {dst}")


if __name__ == "__main__":
    main()

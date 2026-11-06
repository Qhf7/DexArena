"""The package_opening family dataset: two subtasks with different embodiments made into one.

OpenPackage is dual-arm (58 joints, base + left wrist + right wrist) and ExtractAndPlaceApple
is right-arm only (29 joints, base + one wrist). One policy has to cover both, so the family
dataset lives in the dual-arm 58-joint layout and keeps the two views both subtasks can supply:
the family base camera and the RIGHT wrist.

    OpenPackage episodes         : state/action kept; the left-wrist video is dropped.
    ExtractAndPlaceApple episodes: right arm/hand copied into the right slots; the left arm and
                                   hand filled with the dual robot's initial pose, read from
                                   OpenPackage's first frame (it varies < 0.012 rad across
                                   episodes); its single wrist video becomes the right wrist.

The left-arm filler is never executed at evaluation time: the chained evaluator routes only the
acting arm's slots and holds the other arm, so what matters is that the policy sees a constant,
plausible left arm during the extraction stage -- which is also what the long task shows it.

    python utils/groot_utils/build_package_family_dataset.py [--overwrite]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.dex_tasks import resolve_dex_task  # noqa: E402

# $DEX_OUT holds datasets/, runs/ and evals/ (docs/GROOT_DEXARENA.md).
DEX_OUT = Path(os.environ.get("DEX_OUT", "dex_out"))
DUAL = DEX_OUT / "datasets" / "package_opening_open_package_rig_vla"
SINGLE = DEX_OUT / "datasets" / "package_opening_extract_apple_rig_vla"
# Language labels are copied per frame from the sources. With the relabelled subtask datasets
# (one of four registered instructions per episode, utils/dex_tasks.py) the family carries all
# eight; --collapse-canonical reproduces the earlier build, which folded every phrasing back
# to each subtask's canonical instruction to match the single-instruction specialists of the time.
CANONICAL = {"package_opening_open_package_rig_vla": resolve_dex_task("PackageOpeningOpenPackage-Sharpa-v1").prompt,
             "package_opening_extract_apple_rig_vla": resolve_dex_task("PackageOpeningExtractApple-Sharpa-v1").prompt}

L_ARM, R_ARM, L_HAND, R_HAND = (0, 7), (7, 14), (14, 36), (36, 58)
VIDEO_MAP_DUAL = {"observation.images.base_0_rgb": "observation.images.base_0_rgb",
                  "observation.images.right_wrist_0_rgb": "observation.images.right_wrist_0_rgb"}
VIDEO_MAP_SINGLE = {"observation.images.base_0_rgb": "observation.images.base_0_rgb",
                    "observation.images.left_wrist_0_rgb": "observation.images.right_wrist_0_rgb"}
VEC_KEYS = ("observation.state", "next.observation.state", "action")


def pad_29_to_58(v, rest):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    out = rest.copy()
    out[R_ARM[0]:R_ARM[1]] = v[:7]
    out[R_HAND[0]:R_HAND[1]] = v[7:29]
    return out


def vec_stats(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
            "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(), "count": [int(arr.shape[0])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-tag", default="", help="e.g. v4 -> package_family_rig_v4_vla")
    ap.add_argument("--collapse-canonical", action="store_true",
                    help="label every episode with its subtask's canonical instruction (the earlier build)")
    args = ap.parse_args()
    DST = DEX_OUT / "datasets" / f"package_family_rig{'_' + args.out_tag if args.out_tag else ''}_vla"
    if DST.exists():
        if not args.overwrite:
            raise SystemExit(f"{DST} exists; pass --overwrite")
        shutil.rmtree(DST)

    first = pd.read_parquet(DUAL / "data" / "chunk-000" / "episode_000000.parquet", columns=["observation.state"])
    rest = np.asarray(first["observation.state"].iloc[0], dtype=np.float32).copy()
    print("[package] left-arm filler:", np.round(rest[L_ARM[0]:L_ARM[1]], 3).tolist())

    out_video_keys = sorted(set(VIDEO_MAP_DUAL.values()))
    for prefix in ("", "next."):
        for key in out_video_keys:
            (DST / "videos" / "chunk-000" / f"{prefix}{key}").mkdir(parents=True)
    (DST / "data" / "chunk-000").mkdir(parents=True)
    (DST / "meta").mkdir(parents=True)

    ep_out = frame_base = 0
    tasks_table, episodes_meta, episodes_stats = {}, [], []
    sources, source_ranges, assignments = [], [], set()
    for src, video_map, pad in ((DUAL, VIDEO_MAP_DUAL, False), (SINGLE, VIDEO_MAP_SINGLE, True)):
        ep_first = ep_out
        render = src / "meta" / "robodex_render.json"
        if render.exists():
            assignments.add(json.loads(render.read_text()).get("instruction_assignment", "fixed"))
        src_tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (src / "meta" / "tasks.jsonl").open()}
        src_stats = {json.loads(l)["episode_index"]: json.loads(l) for l in (src / "meta" / "episodes_stats.jsonl").open()}
        n = 0
        for line in (src / "meta" / "episodes.jsonl").open():
            rec = json.loads(line); src_idx = rec["episode_index"]
            frame = pd.read_parquet(src / "data" / "chunk-000" / f"episode_{src_idx:06d}.parquet")
            if pad:
                for key in VEC_KEYS:
                    frame[key] = [pad_29_to_58(v, rest) for v in frame[key]]
            frame["episode_index"] = ep_out
            frame["index"] = range(frame_base, frame_base + len(frame))
            if args.collapse_canonical:
                labels = [CANONICAL[src.name] for _ in frame["task_index"]]
            else:
                labels = [src_tasks[int(ti)] for ti in frame["task_index"]]
            frame["task_index"] = [tasks_table.setdefault(name, len(tasks_table)) for name in labels]
            frame.to_parquet(DST / "data" / "chunk-000" / f"episode_{ep_out:06d}.parquet", index=False)
            for prefix in ("", "next."):
                for src_key, dst_key in video_map.items():
                    mp4 = src / "videos" / "chunk-000" / f"{prefix}{src_key}" / f"episode_{src_idx:06d}.mp4"
                    if mp4.exists():
                        shutil.copy(mp4, DST / "videos" / "chunk-000" / f"{prefix}{dst_key}" / f"episode_{ep_out:06d}.mp4")
            st = src_stats[src_idx]; st["episode_index"] = ep_out
            stats = st["stats"]
            for key in VEC_KEYS:                       # recompute the widened vectors
                stats[key] = vec_stats(np.stack(frame[key].values))
            for prefix in ("", "next."):               # rename / drop image stats to match
                for src_key, dst_key in video_map.items():
                    if f"{prefix}{src_key}" in stats and src_key != dst_key:
                        stats[f"{prefix}{dst_key}"] = stats.pop(f"{prefix}{src_key}")
                stats.pop(f"{prefix}observation.images.left_wrist_0_rgb", None) if not pad else None
            episodes_stats.append(st)
            rec["episode_index"] = ep_out; episodes_meta.append(rec)
            frame_base += len(frame); ep_out += 1; n += 1
        sources.append((str(src), n)); print(f"[package] {src.name}: {n} episodes -> cumulative {ep_out} / {frame_base} frames")
        env_id = "PackageOpeningOpenPackage-Sharpa-v1" if src is DUAL else "PackageOpeningExtractApple-Sharpa-v1"
        source_ranges.append({"env_id": env_id, "first_episode": ep_first, "episodes": n})

    with (DST / "meta" / "episodes.jsonl").open("w") as f:
        for rec in episodes_meta: f.write(json.dumps(rec) + "\n")
    with (DST / "meta" / "episodes_stats.jsonl").open("w") as f:
        for st in episodes_stats: f.write(json.dumps(st) + "\n")
    with (DST / "meta" / "tasks.jsonl").open("w") as f:
        for name, idx in sorted(tasks_table.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": name}) + "\n")

    info = json.loads((DUAL / "meta" / "info.json").read_text())
    for prefix in ("", "next."):
        info["features"].pop(f"{prefix}observation.images.left_wrist_0_rgb", None)
    info.update(total_episodes=ep_out, total_frames=frame_base, total_tasks=len(tasks_table),
                total_videos=ep_out * 2 * len(out_video_keys), total_chunks=1, splits={"train": f"0:{ep_out}"})
    (DST / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    modality = json.loads((DUAL / "meta" / "modality.json").read_text())
    modality["video"].pop("left_wrist")
    (DST / "meta" / "modality.json").write_text(json.dumps(modality, indent=4) + "\n")
    (DST / "meta" / "robodex_family.json").write_text(json.dumps({
        "family": "package", "layout": "dual_58_two_views",
        "views": ["base_0_rgb", "right_wrist_0_rgb"],
        "left_arm_filler": rest[L_ARM[0]:L_ARM[1]].tolist(), "left_hand_filler": rest[L_HAND[0]:L_HAND[1]].tolist(),
        "sources": sources, "source_ranges": source_ranges,
        "subtasks": [r["env_id"] for r in source_ranges],
        "instruction_assignment": "canonical" if args.collapse_canonical else ("per_episode" if assignments == {"per_episode"} else "mixed:" + ",".join(sorted(assignments) or ["unknown"])),
        "instructions": sorted(tasks_table, key=tasks_table.get)}, indent=2) + "\n")
    print(f"[package] wrote {ep_out} episodes / {frame_base} frames / {len(tasks_table)} instructions -> {DST}")


if __name__ == "__main__":
    main()

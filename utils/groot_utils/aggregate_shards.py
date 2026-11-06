"""Merge per-shard VLA LeRobot v3.0 datasets (from sharded replay renders) into one.

The renderer is run in slices (``--episode_start/--episodes``) in separate processes so a
renderer crash (SAPIEN's Vulkan occasionally fails with ErrorOutOfHostMemory on a
reconfigure) costs one slice, not the whole task. This joins the slices back in order.

Usage:
  python utils/groot_utils/aggregate_shards.py --shards_root <dir with shard_*/> --dst <dataset root>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards_root", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.aggregate import aggregate_datasets

    shards = sorted(p for p in Path(args.shards_root).glob("shard_*") if (p / "meta" / "info.json").exists())
    if not shards:
        raise SystemExit(f"no shard_*/meta/info.json under {args.shards_root}")
    dst = Path(args.dst)
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)
    expected = 0
    for shard in shards:
        info = json.loads((shard / "meta" / "info.json").read_text())
        expected += int(info["total_episodes"])
        print(f"[shard] {shard.name}: {info['total_episodes']} episodes / {info['total_frames']} frames")
    aggregate_datasets(
        repo_ids=[s.name for s in shards],
        aggr_repo_id=dst.name,
        roots=[s for s in shards],
        aggr_root=dst,
    )
    info = json.loads((dst / "meta" / "info.json").read_text())
    if int(info["total_episodes"]) != expected:
        raise SystemExit(f"aggregate has {info['total_episodes']} episodes, shards had {expected}")
    print(f"[done] {dst}: {info['total_episodes']} episodes / {info['total_frames']} frames, "
          f"codebase {info['codebase_version']}")


if __name__ == "__main__":
    main()

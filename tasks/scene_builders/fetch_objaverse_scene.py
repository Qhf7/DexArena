"""Download an Objaverse mesh for use as a background scene.

Objaverse uids are the same ids Sketchfab uses, so the uid can be read straight
off a model's Sketchfab URL:

    https://sketchfab.com/3d-models/minimalistic-workshop-environment-64d6f92fc5d44770843782de76e03e61
                                                                     ^-- uid

    ./scripts/scene_library.sh fetch 64d6f92fc5d44770843782de76e03e61
    python -m tasks.scene_builders.fetch_objaverse_scene 64d6f92fc5d44770843782de76e03e61

The file lands in the directory the GLB background adapter reads
(``ROBODEX_SCENE_GLB_DIR``, default ``~/.maniskill/data/scene_datasets/glb_scenes``).

On machines where huggingface.co is slow or blocked, point ``--endpoint`` (or
``HF_ENDPOINT``) at a mirror, e.g. ``https://hf-mirror.com``. This talks plain
HTTP rather than going through ``huggingface_hub`` precisely so the mirror is
honored no matter which client version is installed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from tasks.scene_builders.backgrounds import GLB_SCENE_DIR

REPO_PATH = "datasets/allenai/objaverse/resolve/main"
INDEX_FILE = "object-paths.json.gz"


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.rename(dst)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("uids", nargs="+", help="Objaverse/Sketchfab model uids")
    p.add_argument("--out", default=str(GLB_SCENE_DIR))
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    p.add_argument("--cache", default=None, help="where to keep the uid index (default: alongside --out)")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.endpoint.rstrip("/")

    # One index maps every uid to its path inside the repo; there is no way to
    # guess which shard a uid lives in.
    index_path = Path(args.cache or out) / INDEX_FILE
    if not index_path.exists():
        print(f"fetching uid index from {base} ...")
        download(f"{base}/{REPO_PATH}/{INDEX_FILE}", index_path)
    with gzip.open(index_path, "rt") as f:
        paths = json.load(f)

    missing = [u for u in args.uids if u not in paths]
    if missing:
        print(f"not in Objaverse: {', '.join(missing)}", file=sys.stderr)
        return 1

    for uid in args.uids:
        dst = out / f"{uid}.glb"
        download(f"{base}/{REPO_PATH}/{paths[uid]}", dst)
        print(f"{uid} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

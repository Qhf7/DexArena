"""Download the pinned LabUtopia room dependencies and prepare SAPIEN assets.

Run this file directly to prepare assets without importing the simulator:
    python tasks/scene_builders/fetch_labutopia_scene.py --out /path/to/glb_scenes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "labutopia_sources.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(relative: str, expected: str, cache: Path, manifest: dict) -> Path:
    path = cache / relative
    if path.is_file() and sha256(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"Rui-li023/LabUtopia/{manifest['revision']}/{manifest['directory']}/{relative}"
    )
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with (
            urlopen(
                f"https://raw.githubusercontent.com/{suffix}", timeout=120
            ) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        with temporary.open("rb") as stream:
            lfs = stream.read(128).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
        if lfs:
            with (
                urlopen(
                    f"https://media.githubusercontent.com/media/{suffix}", timeout=120
                ) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
        if sha256(temporary) != expected:
            raise ValueError(
                f"SHA256 mismatch for {relative}; refusing unverified scene data"
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default = (
        Path(os.environ.get("MS_ASSET_DIR", Path.home() / ".maniskill"))
        / "data/scene_datasets/glb_scenes"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("ROBODEX_SCENE_GLB_DIR", default)),
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Existing lab_003 directory; all input hashes are verified",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = json.loads(MANIFEST.read_text())
    output = args.out / "labutopia_lab003"
    source = args.source or output / "source"
    entries = list(manifest["sha256"].items())
    if args.source:
        for relative, expected in entries:
            path = source / relative
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Missing or incorrect pinned source: {path}")
    else:
        print(
            f"Downloading/verifying {len(entries)} LabUtopia files (CC BY-NC 4.0)",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda item: fetch_file(*item, source, manifest), entries))
    # A direct file invocation must not import tasks/__init__ or require SAPIEN.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "convert_labutopia", HERE / "convert_labutopia.py"
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    output.mkdir(parents=True, exist_ok=True)
    stats = converter.convert(source / "lab_003.usd", output / "room.glb")
    provenance = {
        **manifest,
        "conversion": stats,
        "tabletop_props": [],
        "room_sha256": sha256(output / "room.glb"),
    }
    (output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "ATTRIBUTION.md").write_text(
        "# LabUtopia laboratory assets\n\n"
        "Source: [Rui-li023/LabUtopia](https://github.com/Rui-li023/LabUtopia).\n"
        f"Revision: {manifest['revision']}. Room: lab_003, /World/lab_015.\n\n"
        "Assets: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).\n"
        "Changes: room-only USD to GLB conversion, mesh simplification, MDL to PBR materials.\n"
        "No task objects are included. The work table is provided separately by ManiSkill.\n"
    )
    print(f"Prepared {output / 'room.glb'}: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Write the manifest of a PI0.5 base checkpoint directory: one ``{name, bytes,
sha256}`` record per file under ``params/``, relative to the base directory.
``train/train_pi05.py --base-manifest`` checks the base against it before
fine-tuning (``utils.pi05_utils.checkpoint.verify_base``).

    python -m utils.pi05_utils.base_manifest --base /path/to/pi05_base --output /path/to/pi05_base-manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def base_manifest(base: Path) -> list[dict]:
    from utils.pi05_utils.data import sha

    params = Path(base).resolve() / "params"
    if not params.is_dir():
        raise FileNotFoundError(f"{params} is not a directory")
    files = sorted(path for path in params.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"no files under {params}")
    return [
        dict(
            name=path.relative_to(Path(base).resolve()).as_posix(),
            bytes=path.stat().st_size,
            sha256=sha(path),
        )
        for path in files
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="directory that contains params/")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    records = base_manifest(args.base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    total = sum(record["bytes"] for record in records)
    print(f"[base-manifest] {args.output}: {len(records)} files, {total / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

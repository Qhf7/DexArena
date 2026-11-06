"""Resolve dataset / checkpoint / run paths for the training entry points.

Absolute paths are used as-is. Relative paths are resolved against a root that
defaults to this repository, so nothing is tied to one machine's layout:

- ``ROBODEX_STORAGE_ROOT`` -- base for checkpoints and tensorboard runs
  (default: the repository root).
- ``ROBODEX_DATA_ROOT``    -- base for demonstration datasets
  (default: ``<storage root>/data``).
- ``ROBODEX_RUNS_ROOT``    -- base for ``runs/{run_name}``
  (default: ``<storage root>/runs``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

STORAGE_ROOT = Path(os.environ.get("ROBODEX_STORAGE_ROOT", REPO_ROOT)).expanduser()
DATA_ROOT = Path(os.environ.get("ROBODEX_DATA_ROOT", STORAGE_ROOT / "data")).expanduser()
RUNS_ROOT = Path(os.environ.get("ROBODEX_RUNS_ROOT", STORAGE_ROOT / "runs")).expanduser()

DATA_PATH_FIELDS = (
    "demo_data_path",
    "bc_data_path",
    "demo_init_data_path",
    "warmup_data_path",
    "critic_warmup_data_path",
)
CHECKPOINT_PATH_FIELDS = (
    "checkpoint",
    "pretrained_checkpoint",
    "fqe_checkpoint",
    "bc_checkpoint",
    "base_policy_path",
)
SEQUENCE_PATH_FIELDS = ("extra_npz_paths",)


def _expand(path: str | os.PathLike[str], root: Path) -> str:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return str(resolved)
    return str(root / resolved)


def data_path(*parts: str | os.PathLike[str]) -> str:
    if not parts:
        return str(DATA_ROOT)
    return _expand(Path(*map(str, parts)), DATA_ROOT)


def runs_path(*parts: str | os.PathLike[str]) -> str:
    return str(RUNS_ROOT.joinpath(*map(str, parts)))


def ensure_parent_dir(path: str | os.PathLike[str]) -> str:
    """Create the parent directory required to write ``path``."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    return str(output)


def storage_path(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None
    return _expand(path, STORAGE_ROOT)


def demo_path(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None
    return _expand(path, DATA_ROOT)


def checkpoint_path(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None
    return _expand(path, STORAGE_ROOT)


def checkpoint_dir(path: str | os.PathLike[str]) -> str:
    checkpoint = checkpoint_path(path)
    if checkpoint is None:
        raise ValueError("checkpoint path cannot be None")
    return str(Path(checkpoint).parent)


def resolve_training_paths(args: Any) -> Any:
    for field in DATA_PATH_FIELDS:
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                setattr(args, field, demo_path(value))
    for field in CHECKPOINT_PATH_FIELDS:
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                setattr(args, field, checkpoint_path(value))
    for field in SEQUENCE_PATH_FIELDS:
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                setattr(args, field, tuple(demo_path(path) for path in value))
    return args

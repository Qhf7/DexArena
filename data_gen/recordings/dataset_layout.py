"""Where an exported LeRobot dataset lives: ``<lerobot root>/<hand>/<env id>``.

``<hand>`` is the end effector token of the env id in lower case (``sharpa``, ``wuji``,
``xhand``, ``gripper``), so one root holds every hand's datasets side by side and one hand's
datasets can be selected by directory. Every tool that derives a dataset directory from an env
id uses `dataset_root`; every tool that discovers datasets under a root uses `iter_dataset_roots`.
"""
from __future__ import annotations

from pathlib import Path

from tasks.config.hands import HANDS, hand_of

HAND_DIRS: dict[str, str] = {hand: hand.lower() for hand in HANDS}


def hand_dir(env_id: str) -> str:
    """The directory an env id's datasets live under: its hand token in lower case."""
    return HAND_DIRS[hand_of(env_id)]


def dataset_root(base: str | Path, env_id: str) -> Path:
    return Path(base) / hand_dir(env_id) / str(env_id)


def collection_root(dataset_dir: str | Path) -> Path:
    """The inverse of `dataset_root`: the root a dataset directory sits under."""
    parent = Path(dataset_dir).parent
    return parent.parent if parent.name in HAND_DIRS.values() else parent


def iter_dataset_roots(base: str | Path, require_replay: bool = True) -> list[Path]:
    """Every LeRobot dataset under `base`, i.e. every ``<base>/<hand>/<env id>``."""
    base = Path(base)
    if not base.is_dir():
        return []
    found: list[Path] = []
    for hand in sorted(HAND_DIRS.values()):
        hand_root = base / hand
        if not hand_root.is_dir():
            continue
        found += sorted(
            d for d in hand_root.iterdir()
            if d.is_dir() and (d / "meta" / "info.json").exists()
            and (not require_replay or (d / "replay").exists())
        )
    return found

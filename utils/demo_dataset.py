from __future__ import annotations

import csv
import glob
import inspect
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from utils.paths import demo_path


LeRobotDataset = None
DEFAULT_FEATURES: dict[str, Any] = {}
_LEROBOT_IMPORT_ERROR: Exception | None = None


@dataclass(frozen=True)
class EnvStateField:
    category: str
    name: str
    tail_shape: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.category}.{self.name}"

    @property
    def flat_dim(self) -> int:
        if len(self.tail_shape) == 0:
            return 1
        return int(np.prod(self.tail_shape, dtype=np.int64))


@dataclass(frozen=True)
class DatasetSchema:
    obs_dim: int
    action_dim: int
    env_state_fields: tuple[EnvStateField, ...]
    image_fields: tuple[tuple[str, tuple[int, int, int]], ...] = ()

    @property
    def env_state_dim(self) -> int:
        return int(sum(field.flat_dim for field in self.env_state_fields))

    @property
    def env_state_names(self) -> list[str]:
        names: list[str] = []
        for field in self.env_state_fields:
            if len(field.tail_shape) == 0:
                names.append(field.key)
                continue
            for flat_idx in np.ndindex(field.tail_shape):
                suffix = ".".join(str(idx) for idx in flat_idx)
                names.append(f"{field.key}.{suffix}")
        return names


@dataclass(frozen=True)
class ExportSummary:
    dataset_root: Path
    repo_id: str
    num_source_files: int
    num_episodes_added: int
    num_frames_added: int
    total_episodes: int
    total_frames: int
    obs_dim: int
    action_dim: int
    env_state_dim: int


def sort_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: (path.parent.as_posix(), path.name))


def _resolve_input_paths(data_path: str) -> list[Path]:
    data_path = demo_path(data_path)
    contains_glob = any(char in data_path for char in "*?[]")
    if contains_glob:
        matched_paths = [Path(path).expanduser().resolve() for path in glob.glob(os.path.expanduser(data_path))]
        if not matched_paths:
            raise FileNotFoundError(f"Demo path glob did not match any files: {data_path}")
        return list(dict.fromkeys(sort_paths(matched_paths)))

    path = Path(data_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Demo path does not exist: {data_path}")
    return [path.resolve()]


def maybe_import_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyArrow is required to read LeRobot datasets. Install it with "
            "`python -m pip install pyarrow`."
        ) from exc
    return pq


def load_export_settings(dataset_root: Path) -> dict[str, Any]:
    candidate_paths = [
        dataset_root / "export_settings.json",
        dataset_root.parent / "export_settings.json",
    ]
    for path in candidate_paths:
        if path.exists():
            return json.loads(path.read_text())
    return {}


def is_lerobot_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta" / "info.json").exists()
        and (path / "data").exists()
    )

def require_strict_recording_certificate(
    dataset_root: str | Path,
    features: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reject uncertified datasets emitted by the strict teleop recorder."""
    if "episode.source_hash" not in features:
        return None
    from data_gen.teleop.lerobot_integrity import assert_valid_certificate

    try:
        return assert_valid_certificate(dataset_root)
    except Exception as exc:
        raise RuntimeError(
            f"Strict recording dataset `{dataset_root}` failed its integrity gate: {exc}"
        ) from exc



def _find_ancestor_lerobot_root(path: Path) -> Optional[Path]:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if is_lerobot_dataset_root(candidate):
            return candidate.resolve()
    return None


def resolve_lerobot_dataset_roots(data_path: str) -> list[Path]:
    dataset_roots: list[Path] = []
    for candidate in _resolve_input_paths(data_path):
        ancestor_root = _find_ancestor_lerobot_root(candidate)
        if ancestor_root is not None:
            dataset_roots.append(ancestor_root)
            continue

        if not candidate.is_dir():
            continue

        if is_lerobot_dataset_root(candidate):
            dataset_roots.append(candidate.resolve())
            continue

        for info_path in candidate.rglob("meta/info.json"):
            dataset_root = info_path.parent.parent.resolve()
            if is_lerobot_dataset_root(dataset_root):
                dataset_roots.append(dataset_root)

    return list(dict.fromkeys(sort_paths(dataset_roots)))


def _require_lerobot(cache_root: Path | None = None) -> None:
    global LeRobotDataset
    global DEFAULT_FEATURES
    global _LEROBOT_IMPORT_ERROR

    if LeRobotDataset is not None:
        return

    if cache_root is not None:
        hf_home = (cache_root / ".hf_cache").resolve()
        datasets_cache = hf_home / "datasets"
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as ImportedLeRobotDataset
        from lerobot.datasets.utils import DEFAULT_FEATURES as ImportedDefaultFeatures
    except ImportError as exc:  # pragma: no cover
        _LEROBOT_IMPORT_ERROR = exc
        LeRobotDataset = None
        DEFAULT_FEATURES = {}
        raise RuntimeError(
            "LeRobot is not installed. Install it with `python -m pip install \"lerobot==0.4.0\"`."
        ) from exc

    LeRobotDataset = ImportedLeRobotDataset
    DEFAULT_FEATURES = ImportedDefaultFeatures


def require_lerobot(cache_root: Path | None = None) -> None:
    _require_lerobot(cache_root=cache_root)


def _require_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyArrow is required to replace existing LeRobot episodes by seed. "
            "Install it with `python -m pip install pyarrow`."
        ) from exc
    return pq


def _normalize_frame_value(value: Any, feature: dict[str, Any]) -> Any:
    expected_dtype = str(feature["dtype"])
    expected_shape = tuple(int(dim) for dim in feature["shape"])

    if expected_dtype == "string":
        if isinstance(value, str):
            return value
        return str(value)

    if expected_dtype in {"image", "video"}:
        return value

    array_value = np.asarray(value, dtype=np.dtype(expected_dtype))
    if array_value.shape == expected_shape:
        return array_value.copy()

    if array_value.ndim == 0 and expected_shape == (1,):
        return array_value.reshape(1).copy()

    try:
        return array_value.reshape(expected_shape).copy()
    except ValueError as exc:
        raise RuntimeError(
            "Failed to normalize LeRobot frame value with expected "
            f"dtype={expected_dtype} and shape={expected_shape}; got value "
            f"with shape {array_value.shape}."
        ) from exc


def _extract_seed_scalar(value: Any) -> int:
    array_value = np.asarray(value, dtype=np.int64).reshape(-1)
    if array_value.size != 1:
        raise RuntimeError(
            "Expected `episode.seed` to store exactly one scalar per frame, got "
            f"shape {np.asarray(value).shape}."
        )
    return int(array_value[0])


def _load_episode_tasks(dataset_root: Path, pq: Any) -> dict[int, str | None]:
    episode_meta_paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_meta_paths:
        raise RuntimeError(f"No episode metadata parquet files found under `{dataset_root}`.")

    episode_meta = pq.read_table(episode_meta_paths, columns=["episode_index", "tasks"]).to_pydict()
    task_by_episode: dict[int, str | None] = {}
    raw_tasks = episode_meta.get("tasks", [])
    for idx, episode_index_value in enumerate(episode_meta.get("episode_index", [])):
        task_list = raw_tasks[idx] if idx < len(raw_tasks) else None
        task_by_episode[int(episode_index_value)] = (
            str(task_list[0]) if isinstance(task_list, list) and len(task_list) > 0 else None
        )
    return task_by_episode


def _write_episode_manifest(dataset_root: Path) -> None:
    try:
        pq = _require_pyarrow_parquet()
    except RuntimeError:
        return

    episode_meta_paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    data_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not episode_meta_paths or not data_paths:
        return

    episode_meta = pq.read_table(episode_meta_paths).to_pydict()
    data = pq.read_table(data_paths, columns=["episode_index", "episode.seed"]).to_pydict()

    episode_indices = np.asarray(data["episode_index"], dtype=np.int64)
    seed_values = np.asarray([_extract_seed_scalar(value) for value in data["episode.seed"]], dtype=np.int64)

    seed_by_episode: dict[int, int] = {}
    for episode_index in np.unique(episode_indices).tolist():
        mask = episode_indices == int(episode_index)
        unique_seed = np.unique(seed_values[mask])
        if unique_seed.size != 1:
            raise RuntimeError(
                f"Episode {int(episode_index)} in `{dataset_root}` contains multiple seeds: "
                f"{unique_seed.tolist()}."
            )
        seed_by_episode[int(episode_index)] = int(unique_seed[0])

    manifest_rows: list[dict[str, Any]] = []
    raw_tasks = episode_meta.get("tasks", [])
    raw_lengths = episode_meta.get("length", [])
    for idx, episode_index_value in enumerate(episode_meta.get("episode_index", [])):
        episode_index = int(episode_index_value)
        task_list = raw_tasks[idx] if idx < len(raw_tasks) else None
        task_label = (
            str(task_list[0]) if isinstance(task_list, list) and len(task_list) > 0 else ""
        )
        episode_length = int(raw_lengths[idx]) if idx < len(raw_lengths) else -1
        manifest_rows.append(
            {
                "episode_index": episode_index,
                "seed": seed_by_episode.get(episode_index, -1),
                "length": episode_length,
                "task": task_label,
            }
        )

    manifest_path = dataset_root / "episodes.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_index", "seed", "length", "task"])
        writer.writeheader()
        writer.writerows(manifest_rows)


def _replace_existing_episodes_by_seed(
    *,
    dataset_root: Path,
    repo_id: str,
    fps: int,
    robot_type: str,
    features: dict[str, dict[str, Any]],
    replace_episode_seed: int,
    use_videos: bool = False,
) -> int:
    pq = _require_pyarrow_parquet()
    task_by_episode = _load_episode_tasks(dataset_root, pq)

    data_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not data_paths:
        return 0

    table = pq.read_table(data_paths)
    data = table.to_pydict()
    if "episode_index" not in data or "frame_index" not in data:
        raise RuntimeError(
            f"Existing LeRobot dataset at `{dataset_root}` is missing episode/frame indices."
        )
    if "episode.seed" not in data:
        raise RuntimeError(
            f"Existing LeRobot dataset at `{dataset_root}` does not contain `episode.seed`, "
            "so seed-based replacement is not possible."
        )

    episode_indices = np.asarray(data["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(data["frame_index"], dtype=np.int64)
    seed_values = np.asarray([_extract_seed_scalar(value) for value in data["episode.seed"]], dtype=np.int64)

    episode_ids_to_remove: set[int] = set()
    for episode_index in np.unique(episode_indices).tolist():
        mask = episode_indices == int(episode_index)
        unique_seeds = np.unique(seed_values[mask])
        if unique_seeds.size != 1:
            raise RuntimeError(
                f"Episode {int(episode_index)} in `{dataset_root}` contains multiple seeds: "
                f"{unique_seeds.tolist()}."
            )
        if int(unique_seeds[0]) == int(replace_episode_seed):
            episode_ids_to_remove.add(int(episode_index))

    if not episode_ids_to_remove:
        return 0

    order = np.lexsort((frame_indices, episode_indices))
    feature_columns = set(features.keys())
    print(
        f"Replacing {len(episode_ids_to_remove)} existing episode(s) with seed "
        f"{int(replace_episode_seed)} in `{dataset_root}`."
    )

    rebuilt_dataset_root = Path(
        tempfile.mkdtemp(prefix=f"{dataset_root.name}.seed-rebuild.", dir=dataset_root.parent)
    )
    shutil.rmtree(rebuilt_dataset_root)
    backup_dataset_root: Path | None = None

    try:
        recreated_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(fps),
            root=rebuilt_dataset_root,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )

        current_episode_index: int | None = None
        for ordered_idx in order.tolist():
            episode_index = int(episode_indices[ordered_idx])
            if episode_index in episode_ids_to_remove:
                continue

            if current_episode_index is None:
                current_episode_index = episode_index
            elif episode_index != current_episode_index:
                recreated_dataset.save_episode()
                current_episode_index = episode_index

            frame: dict[str, Any] = {}
            for column_name, column_values in data.items():
                if column_name not in feature_columns:
                    continue
                frame[column_name] = _normalize_frame_value(
                    column_values[ordered_idx],
                    features[column_name],
                )
            task_label = task_by_episode.get(episode_index)
            if task_label is None:
                raise RuntimeError(
                    f"Missing task metadata for episode {episode_index} in `{dataset_root}`."
                )
            frame["task"] = task_label
            recreated_dataset.add_frame(frame)

        if current_episode_index is not None:
            recreated_dataset.save_episode()
        recreated_dataset.finalize()

        backup_dataset_root = Path(
            tempfile.mkdtemp(prefix=f"{dataset_root.name}.seed-backup.", dir=dataset_root.parent)
        )
        shutil.rmtree(backup_dataset_root)
        shutil.move(str(dataset_root), str(backup_dataset_root))
        try:
            shutil.move(str(rebuilt_dataset_root), str(dataset_root))
        except Exception:
            if backup_dataset_root.exists() and not dataset_root.exists():
                shutil.move(str(backup_dataset_root), str(dataset_root))
            raise
        _write_episode_manifest(dataset_root)
        shutil.rmtree(backup_dataset_root)
        backup_dataset_root = None
    finally:
        if rebuilt_dataset_root.exists():
            shutil.rmtree(rebuilt_dataset_root)
        if backup_dataset_root is not None and backup_dataset_root.exists():
            shutil.rmtree(backup_dataset_root)

    return len(episode_ids_to_remove)


def _inspect_env_state_fields_from_sequence(
    env_state_sequence: Sequence[dict[str, Any]],
) -> tuple[EnvStateField, ...]:
    if len(env_state_sequence) == 0:
        return ()

    first_state = env_state_sequence[0]
    fields: list[EnvStateField] = []
    for category in sorted(first_state.keys()):
        category_group = first_state[category]
        if not isinstance(category_group, dict):
            raise RuntimeError(
                "Expected environment-state categories to be dictionaries, got "
                f"{type(category_group)!r} for `{category}`."
            )
        for name in sorted(category_group.keys()):
            values = normalize_env_state_leaf(category_group[name])
            fields.append(
                EnvStateField(
                    category=category,
                    name=name,
                    tail_shape=tuple(int(dim) for dim in values.shape),
                )
            )
    return tuple(fields)


def _flatten_env_state_sequence_from_dicts(
    env_state_sequence: Sequence[dict[str, Any]],
    env_state_fields: tuple[EnvStateField, ...],
) -> np.ndarray | None:
    if len(env_state_fields) == 0:
        return None

    sequences: list[np.ndarray] = []
    for field in env_state_fields:
        frames: list[np.ndarray] = []
        for step_idx, env_state in enumerate(env_state_sequence):
            if field.category not in env_state:
                raise RuntimeError(
                    f"Missing env-state category `{field.category}` at step {step_idx}."
                )
            category_group = env_state[field.category]
            if field.name not in category_group:
                raise RuntimeError(
                    f"Missing env-state field `{field.key}` at step {step_idx}."
                )
            values = normalize_env_state_leaf(category_group[field.name])
            if tuple(int(dim) for dim in values.shape) != field.tail_shape:
                raise RuntimeError(
                    "Environment-state layout changed within an episode for "
                    f"`{field.key}` at step {step_idx}."
                )
            frames.append(values.reshape(1, -1))
        sequences.append(np.concatenate(frames, axis=0))

    if len(sequences) == 0:
        return None
    return np.concatenate(sequences, axis=1)


def _build_features(
    schema: DatasetSchema,
    *,
    include_success: bool,
    include_seed: bool,
    include_source_hash: bool,
    use_videos: bool = False,
    next_images: bool = True,
) -> dict[str, dict[str, Any]]:
    state_names = [f"obs_{idx}" for idx in range(schema.obs_dim)]
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (schema.obs_dim,),
            "names": state_names,
        },
        "next.observation.state": {
            "dtype": "float32",
            "shape": (schema.obs_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (schema.action_dim,),
            "names": [f"action_{idx}" for idx in range(schema.action_dim)],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        # Environment signals remain literal; an operator ending RECORD does
        # not get rewritten into a fake termination/truncation.
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.terminated": {"dtype": "bool", "shape": (1,), "names": None},
        "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
        # Dataset boundary is separate from the environment's terminal flags.
        "next.recording_boundary": {"dtype": "bool", "shape": (1,), "names": None},
    }
    if include_source_hash:
        features["episode.source_hash"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
    if schema.env_state_dim > 0:
        env_feature = {
            "dtype": "float32",
            "shape": (schema.env_state_dim,),
            "names": schema.env_state_names,
        }
        features["observation.environment_state"] = env_feature
        features["next.observation.environment_state"] = dict(env_feature)
    for key, shape in schema.image_fields:
        image_feature = {
            "dtype": "video" if use_videos else "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
        features[key] = image_feature
        if next_images:
            # Offline-rendered video datasets skip the `next.` copy: it would double the
            # video bytes for a frame the consumer already has as the next row.
            features[f"next.{key}"] = dict(image_feature)
    if include_success:
        features["next.success"] = {"dtype": "bool", "shape": (1,), "names": None}
    if include_seed:
        features["episode.seed"] = {"dtype": "int64", "shape": (1,), "names": None}
    return features


def _custom_features_from_dataset(dataset: Any) -> dict[str, Any]:
    default_keys = set(DEFAULT_FEATURES.keys())
    return {key: value for key, value in dataset.features.items() if key not in default_keys}


def _create_or_resume_dataset(
    *,
    dataset_root: Path,
    repo_id: str,
    fps: int,
    robot_type: str,
    features: dict[str, dict[str, Any]],
    overwrite: bool,
    replace_episode_seed: int | None = None,
    use_videos: bool = False,
) -> Any:
    _require_lerobot(cache_root=dataset_root)
    if overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)

    if dataset_root.exists():
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        existing_features = _custom_features_from_dataset(dataset)
        if existing_features != features:
            raise RuntimeError(
                f"Existing LeRobot dataset at `{dataset_root}` has incompatible features. "
                "Delete it or rerun with `--overwrite`."
            )
        if int(dataset.fps) != int(fps):
            raise RuntimeError(
                f"Existing LeRobot dataset at `{dataset_root}` uses fps={dataset.fps}, expected {fps}."
            )
        if str(dataset.meta.robot_type) != robot_type:
            raise RuntimeError(
                f"Existing LeRobot dataset at `{dataset_root}` uses robot_type={dataset.meta.robot_type}, "
                f"expected {robot_type}."
            )
        del dataset
        if replace_episode_seed is not None:
            _replace_existing_episodes_by_seed(
                dataset_root=dataset_root,
                repo_id=repo_id,
                fps=fps,
                robot_type=robot_type,
                features=features,
                replace_episode_seed=int(replace_episode_seed),
                use_videos=use_videos,
            )
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        return dataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        root=dataset_root,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
    )


def _group_flat_env_state_names(
    flat_names: Sequence[str],
) -> tuple[tuple[EnvStateField, tuple[tuple[tuple[int, ...], int], ...]], ...]:
    ordered_field_keys: list[tuple[str, str]] = []
    grouped_entries: dict[tuple[str, str], list[tuple[tuple[int, ...], int]]] = {}

    for col_idx, name in enumerate(flat_names):
        parts = str(name).split(".")
        if len(parts) < 2:
            raise RuntimeError(f"Malformed env-state name `{name}`.")
        category = parts[0]
        key = parts[1]
        index_tuple = tuple(int(part) for part in parts[2:])
        field_key = (category, key)
        if field_key not in grouped_entries:
            ordered_field_keys.append(field_key)
            grouped_entries[field_key] = []
        grouped_entries[field_key].append((index_tuple, col_idx))

    grouped_fields: list[tuple[EnvStateField, tuple[tuple[tuple[int, ...], int], ...]]] = []
    for category, key in ordered_field_keys:
        entries = grouped_entries[(category, key)]
        if len(entries) == 1 and entries[0][0] == ():
            tail_shape: tuple[int, ...] = ()
        else:
            rank = max(len(index_tuple) for index_tuple, _ in entries)
            shape = [0] * rank
            for index_tuple, _ in entries:
                if len(index_tuple) != rank:
                    raise RuntimeError(f"Inconsistent env-state rank for `{category}.{key}`.")
                for axis, index in enumerate(index_tuple):
                    shape[axis] = max(shape[axis], index + 1)
            tail_shape = tuple(shape)
        grouped_fields.append(
            (
                EnvStateField(category=category, name=key, tail_shape=tail_shape),
                tuple(entries),
            )
        )
    return tuple(grouped_fields)


def _get_env_state_keys_from_flat_names(flat_names: Sequence[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    grouped_names: dict[str, list[str]] = {}
    for field, _ in _group_flat_env_state_names(flat_names):
        grouped_names.setdefault(field.category, [])
        grouped_names[field.category].append(field.name)
    return tuple((category, tuple(names)) for category, names in grouped_names.items())


def normalize_env_state_leaf(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim >= 2 and array.shape[0] == 1:
        array = np.squeeze(array, axis=0)
    return array.copy()


def normalize_env_state_batch(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim >= 2 and array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    return array.copy()


def unflatten_env_state_batch(
    flat_values: np.ndarray,
    flat_names: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    flat_values = np.asarray(flat_values, dtype=np.float32)
    if flat_values.ndim == 1:
        flat_values = flat_values.reshape(1, -1)
    if flat_values.ndim != 2:
        raise RuntimeError(f"Expected rank-2 env-state batch, got shape {flat_values.shape}.")
    if flat_values.shape[1] != len(flat_names):
        raise RuntimeError(
            "Flattened environment-state width does not match metadata names: "
            f"values={flat_values.shape[1]}, names={len(flat_names)}."
        )

    state: dict[str, dict[str, np.ndarray]] = {}
    for field, entries in _group_flat_env_state_names(flat_names):
        state.setdefault(field.category, {})
        if len(field.tail_shape) == 0:
            col_idx = entries[0][1]
            state[field.category][field.name] = normalize_env_state_batch(
                flat_values[:, col_idx : col_idx + 1]
            )
            continue

        values = np.zeros((flat_values.shape[0],) + field.tail_shape, dtype=np.float32)
        for index_tuple, col_idx in entries:
            values[(slice(None),) + index_tuple] = flat_values[:, col_idx]
        state[field.category][field.name] = normalize_env_state_batch(values)
    return state


def unflatten_env_state(
    flat_values: np.ndarray,
    flat_names: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    batch_state = unflatten_env_state_batch(flat_values, flat_names)
    state: dict[str, dict[str, np.ndarray]] = {}
    for category, category_values in batch_state.items():
        state[category] = {}
        for name, values in category_values.items():
            state[category][name] = values[0].copy()
    return state


def sort_traj_keys(h5_file: h5py.File) -> list[str]:
    def key_fn(key: str) -> tuple[int, int | str]:
        suffix = key.split("_")[-1]
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, key)

    return sorted([key for key in h5_file.keys() if key.startswith("traj_")], key=key_fn)


def flatten_traj_batch(array: Any) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)


def format_dim_counts(dim_counts: dict[tuple[int, int], int]) -> str:
    if not dim_counts:
        return "none"
    parts = []
    for (obs_dim, act_dim), count in sorted(dim_counts.items()):
        parts.append(f"obs/action={obs_dim}/{act_dim} ({count} trajs)")
    return ", ".join(parts)


def resolve_demo_h5_paths(data_path: str) -> list[Path]:
    if h5py is None:
        raise RuntimeError(
            "h5py is required to read legacy ManiSkill trajectory files. "
            "Install it with `python -m pip install h5py`."
        )

    h5_paths: list[Path] = []
    for candidate in _resolve_input_paths(data_path):
        if candidate.is_file():
            if candidate.suffix.lower() != ".h5":
                raise RuntimeError(f"Demo path must point to an .h5 file: {candidate}")
            h5_paths.append(candidate.resolve())
            continue
        if candidate.is_dir():
            h5_paths.extend(path.resolve() for path in sorted(candidate.rglob("*.h5")))
            continue
        raise RuntimeError(f"Unsupported demo path type: {candidate}")

    h5_paths = list(dict.fromkeys(h5_paths))
    if not h5_paths:
        raise RuntimeError(f"No valid .h5 demo files found from path: {data_path}")
    return h5_paths


def _get_env_state_step_count(traj_group: h5py.Group) -> int:
    if "env_states" not in traj_group:
        return 0

    state_steps: Optional[int] = None
    states_root = traj_group["env_states"]
    for category in states_root.keys():
        category_group = states_root[category]
        for name in category_group.keys():
            value = category_group[name]
            if value.shape[0] <= 0:
                return 0
            state_steps = value.shape[0] if state_steps is None else min(state_steps, value.shape[0])
    return 0 if state_steps is None else int(state_steps)


def _get_env_state_keys(traj_group: h5py.Group) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if "env_states" not in traj_group:
        return tuple()
    keys = []
    states_root = traj_group["env_states"]
    for category in sorted(states_root.keys()):
        category_group = states_root[category]
        keys.append((category, tuple(sorted(category_group.keys()))))
    return tuple(keys)


def _feature_flat_dim(feature: dict[str, Any]) -> int:
    shape = tuple(int(dim) for dim in feature.get("shape", ()))
    if len(shape) == 0:
        return 1
    return int(np.prod(shape, dtype=np.int64))


def _load_h5_demo_dataset(
    data_path: str,
    *,
    expected_obs_dim: int,
    expected_act_dim: int,
    load_bc_data: bool,
    load_reset_states: bool,
) -> "LoadedDemoDataset":
    h5_paths = resolve_demo_h5_paths(data_path)
    dataset = LoadedDemoDataset(data_path=data_path, source_paths=h5_paths)

    bc_obs_list: list[np.ndarray] = []
    bc_action_list: list[np.ndarray] = []
    reset_obs_list: list[np.ndarray] = []
    reset_progress_list: list[np.ndarray] = []
    reset_state_lists: dict[str, dict[str, list[np.ndarray]]] = {}
    expected_reset_state_keys: Optional[tuple[tuple[str, tuple[str, ...]], ...]] = None

    for h5_path in h5_paths:
        file_matched = False
        with h5py.File(h5_path, "r") as h5_file:
            for traj_key in sort_traj_keys(h5_file):
                traj_group = h5_file[traj_key]
                if "obs" not in traj_group or "actions" not in traj_group:
                    continue

                traj_obs = flatten_traj_batch(traj_group["obs"])
                traj_actions = flatten_traj_batch(traj_group["actions"])
                obs_steps = max(0, traj_obs.shape[0] - 1)
                valid_steps = min(obs_steps, traj_actions.shape[0])
                if load_reset_states:
                    valid_steps = min(valid_steps, _get_env_state_step_count(traj_group))
                if valid_steps <= 0:
                    continue

                traj_obs_dim = traj_obs.shape[1]
                traj_act_dim = traj_actions.shape[1]
                if traj_obs_dim != expected_obs_dim or traj_act_dim != expected_act_dim:
                    dims = (traj_obs_dim, traj_act_dim)
                    dataset.mismatched_dim_counts[dims] = dataset.mismatched_dim_counts.get(dims, 0) + 1
                    continue

                traj_reset_obs = traj_obs[:valid_steps]
                traj_progress = np.linspace(0.0, 1.0, num=valid_steps, endpoint=True, dtype=np.float32)

                if load_bc_data:
                    bc_obs_list.append(traj_reset_obs)
                    bc_action_list.append(traj_actions[:valid_steps])
                    dataset.matched_bc_trajectories += 1
                    file_matched = True

                if load_reset_states:
                    if "env_states" not in traj_group:
                        dataset.missing_env_states_trajectories += 1
                        continue

                    traj_reset_state_keys = _get_env_state_keys(traj_group)
                    if expected_reset_state_keys is None:
                        expected_reset_state_keys = traj_reset_state_keys
                        reset_state_lists = {
                            category: {name: [] for name in names}
                            for category, names in expected_reset_state_keys
                        }
                    if traj_reset_state_keys != expected_reset_state_keys:
                        dataset.mismatched_reset_state_keys += 1
                        continue

                    states_root = traj_group["env_states"]
                    for category, names in expected_reset_state_keys:
                        for name in names:
                            values = np.asarray(states_root[category][name][:valid_steps])
                            if values.ndim == 1:
                                values = values[:, None]
                            reset_state_lists[category][name].append(normalize_env_state_batch(values))
                    reset_obs_list.append(traj_reset_obs)
                    reset_progress_list.append(traj_progress)
                    dataset.matched_reset_trajectories += 1
                    file_matched = True

        if file_matched:
            dataset.matched_source_paths.append(h5_path)

    if load_bc_data:
        if not bc_obs_list:
            raise RuntimeError(
                "No valid BC samples matched env obs/action dims "
                f"{expected_obs_dim}/{expected_act_dim} from path {data_path}. "
                f"Available trajectory dims: {format_dim_counts(dataset.mismatched_dim_counts)}"
            )
        dataset.bc_obs = torch.from_numpy(np.concatenate(bc_obs_list, axis=0)).float()
        dataset.bc_actions = torch.from_numpy(np.concatenate(bc_action_list, axis=0)).float()

    if load_reset_states:
        if not reset_obs_list or not reset_state_lists:
            raise RuntimeError(
                "No valid reset states matched env obs/action dims "
                f"{expected_obs_dim}/{expected_act_dim} from path {data_path}."
            )
        dataset.reset_obs = torch.from_numpy(np.concatenate(reset_obs_list, axis=0)).float()
        dataset.demo_progress = torch.from_numpy(np.concatenate(reset_progress_list, axis=0)).float()
        dataset.reset_env_states = {}
        for category, category_items in reset_state_lists.items():
            dataset.reset_env_states[category] = {}
            for name, value_list in category_items.items():
                dataset.reset_env_states[category][name] = torch.from_numpy(
                    np.concatenate(value_list, axis=0)
                ).float()

    return dataset


def _load_lerobot_demo_dataset(
    data_path: str,
    *,
    expected_obs_dim: int,
    expected_act_dim: int,
    load_bc_data: bool,
    load_reset_states: bool,
) -> "LoadedDemoDataset":
    dataset_roots = resolve_lerobot_dataset_roots(data_path)
    if not dataset_roots:
        raise RuntimeError(f"No LeRobot dataset roots found from path: {data_path}")

    pq = maybe_import_pyarrow()
    dataset = LoadedDemoDataset(data_path=data_path, source_paths=dataset_roots)

    bc_obs_list: list[np.ndarray] = []
    bc_action_list: list[np.ndarray] = []
    reset_obs_list: list[np.ndarray] = []
    reset_progress_list: list[np.ndarray] = []
    reset_state_lists: dict[str, dict[str, list[np.ndarray]]] = {}
    expected_reset_state_keys: Optional[tuple[tuple[str, tuple[str, ...]], ...]] = None

    for dataset_root in dataset_roots:
        info_path = dataset_root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        features = info.get("features", {})
        require_strict_recording_certificate(dataset_root, features)

        obs_feature = features.get("observation.state")
        action_feature = features.get("action")
        if obs_feature is None or action_feature is None:
            raise RuntimeError(
                f"LeRobot dataset `{dataset_root}` is missing `observation.state` or `action` features."
            )

        obs_dim = _feature_flat_dim(obs_feature)
        act_dim = _feature_flat_dim(action_feature)
        episode_count_hint = max(1, int(info.get("total_episodes", 0) or 0))
        if obs_dim != expected_obs_dim or act_dim != expected_act_dim:
            dims = (obs_dim, act_dim)
            dataset.mismatched_dim_counts[dims] = dataset.mismatched_dim_counts.get(dims, 0) + episode_count_hint
            continue

        env_state_feature = features.get("observation.environment_state")
        env_state_names = list(env_state_feature.get("names") or []) if env_state_feature is not None else []
        data_paths = sorted((dataset_root / "data").rglob("*.parquet"))
        if not data_paths:
            continue

        columns = ["episode_index", "frame_index", "action", "observation.state"]
        if load_reset_states and env_state_feature is not None:
            columns.append("observation.environment_state")
        data = pq.read_table(data_paths, columns=columns).to_pydict()
        raw_episode_indices = np.asarray(data.get("episode_index", []), dtype=np.int64)
        if raw_episode_indices.size == 0:
            continue

        raw_frame_indices = np.asarray(data.get("frame_index", []), dtype=np.int64)
        if raw_frame_indices.size != raw_episode_indices.size:
            raw_frame_indices = np.arange(raw_episode_indices.size, dtype=np.int64)

        order = np.lexsort((raw_frame_indices, raw_episode_indices))
        episode_indices = raw_episode_indices[order]
        actions = np.asarray([data["action"][i] for i in order], dtype=np.float32)
        observations = np.asarray([data["observation.state"][i] for i in order], dtype=np.float32)
        env_state_values = None
        if load_reset_states and env_state_feature is not None and env_state_names:
            env_state_values = np.asarray(
                [data["observation.environment_state"][i] for i in order],
                dtype=np.float32,
            )

        episode_ids = np.unique(episode_indices).tolist()
        root_reset_keys = None
        if load_reset_states:
            if env_state_feature is None or not env_state_names:
                dataset.missing_env_states_trajectories += len(episode_ids)
            else:
                root_reset_keys = _get_env_state_keys_from_flat_names(env_state_names)
                if expected_reset_state_keys is None:
                    expected_reset_state_keys = root_reset_keys
                    reset_state_lists = {
                        category: {name: [] for name in names}
                        for category, names in expected_reset_state_keys
                    }
                elif root_reset_keys != expected_reset_state_keys:
                    dataset.mismatched_reset_state_keys += len(episode_ids)
                    root_reset_keys = None

        file_matched = False
        for episode_id in episode_ids:
            mask = episode_indices == int(episode_id)
            episode_obs = observations[mask]
            episode_actions = actions[mask]
            if episode_obs.shape[0] == 0 or episode_actions.shape[0] == 0:
                continue

            if load_bc_data:
                bc_obs_list.append(episode_obs)
                bc_action_list.append(episode_actions)
                dataset.matched_bc_trajectories += 1
                file_matched = True

            if load_reset_states:
                if env_state_values is None or root_reset_keys is None or expected_reset_state_keys is None:
                    continue

                episode_states = unflatten_env_state_batch(env_state_values[mask], env_state_names)
                episode_progress = np.linspace(
                    0.0,
                    1.0,
                    num=episode_obs.shape[0],
                    endpoint=True,
                    dtype=np.float32,
                )
                for category, names in expected_reset_state_keys:
                    for name in names:
                        reset_state_lists[category][name].append(
                            normalize_env_state_batch(episode_states[category][name])
                        )
                reset_obs_list.append(episode_obs)
                reset_progress_list.append(episode_progress)
                dataset.matched_reset_trajectories += 1
                file_matched = True

        if file_matched:
            dataset.matched_source_paths.append(dataset_root)

    if load_bc_data:
        if not bc_obs_list:
            raise RuntimeError(
                "No valid BC samples matched env obs/action dims "
                f"{expected_obs_dim}/{expected_act_dim} from path {data_path}. "
                f"Available trajectory dims: {format_dim_counts(dataset.mismatched_dim_counts)}"
            )
        dataset.bc_obs = torch.from_numpy(np.concatenate(bc_obs_list, axis=0)).float()
        dataset.bc_actions = torch.from_numpy(np.concatenate(bc_action_list, axis=0)).float()

    if load_reset_states:
        if not reset_obs_list or not reset_state_lists:
            raise RuntimeError(
                "No valid reset states matched env obs/action dims "
                f"{expected_obs_dim}/{expected_act_dim} from path {data_path}."
            )
        dataset.reset_obs = torch.from_numpy(np.concatenate(reset_obs_list, axis=0)).float()
        dataset.demo_progress = torch.from_numpy(np.concatenate(reset_progress_list, axis=0)).float()
        dataset.reset_env_states = {}
        for category, category_items in reset_state_lists.items():
            dataset.reset_env_states[category] = {}
            for name, value_list in category_items.items():
                dataset.reset_env_states[category][name] = torch.from_numpy(
                    np.concatenate(value_list, axis=0)
                ).float()

    return dataset


@dataclass
class LoadedDemoDataset:
    data_path: str
    source_paths: list[Path]
    matched_source_paths: list[Path] = field(default_factory=list)
    mismatched_dim_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    matched_bc_trajectories: int = 0
    matched_reset_trajectories: int = 0
    missing_env_states_trajectories: int = 0
    mismatched_reset_state_keys: int = 0
    bc_obs: Optional[torch.Tensor] = None
    bc_actions: Optional[torch.Tensor] = None
    reset_obs: Optional[torch.Tensor] = None
    reset_env_states: Optional[dict[str, dict[str, torch.Tensor]]] = None
    demo_progress: Optional[torch.Tensor] = None

    @property
    def h5_paths(self) -> list[Path]:
        return self.source_paths

    @property
    def matched_h5_paths(self) -> list[Path]:
        return self.matched_source_paths

    @property
    def num_bc_samples(self) -> int:
        if self.bc_obs is None:
            return 0
        return int(self.bc_obs.shape[0])

    @property
    def num_reset_samples(self) -> int:
        if self.reset_obs is not None:
            return int(self.reset_obs.shape[0])
        if self.reset_env_states is None:
            return 0
        for category_items in self.reset_env_states.values():
            for value in category_items.values():
                return int(value.shape[0])
        return 0


@dataclass(frozen=True)
class DemoEpisode:
    observations: torch.Tensor
    actions: torch.Tensor
    source_path: Path
    episode_id: str

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def load_demo_dataset(
    data_path: str,
    *,
    expected_obs_dim: int,
    expected_act_dim: int,
    load_bc_data: bool = False,
    load_reset_states: bool = False,
) -> LoadedDemoDataset:
    lerobot_roots = resolve_lerobot_dataset_roots(data_path)
    if lerobot_roots:
        return _load_lerobot_demo_dataset(
            data_path,
            expected_obs_dim=expected_obs_dim,
            expected_act_dim=expected_act_dim,
            load_bc_data=load_bc_data,
            load_reset_states=load_reset_states,
        )
    return _load_h5_demo_dataset(
        data_path,
        expected_obs_dim=expected_obs_dim,
        expected_act_dim=expected_act_dim,
        load_bc_data=load_bc_data,
        load_reset_states=load_reset_states,
    )


def _load_h5_demo_episodes(
    data_path: str,
    *,
    expected_obs_dim: int,
    expected_act_dim: int,
) -> list[DemoEpisode]:
    h5_paths = resolve_demo_h5_paths(data_path)
    episodes: list[DemoEpisode] = []
    mismatched_dim_counts: dict[tuple[int, int], int] = {}

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as h5_file:
            for traj_key in sort_traj_keys(h5_file):
                traj_group = h5_file[traj_key]
                if "obs" not in traj_group or "actions" not in traj_group:
                    continue

                traj_obs = flatten_traj_batch(traj_group["obs"])
                traj_actions = flatten_traj_batch(traj_group["actions"])
                obs_steps = max(0, traj_obs.shape[0] - 1)
                valid_steps = min(obs_steps, traj_actions.shape[0])
                if valid_steps <= 0:
                    continue

                traj_obs = traj_obs[:valid_steps]
                traj_actions = traj_actions[:valid_steps]
                if traj_obs.shape[1] != expected_obs_dim or traj_actions.shape[1] != expected_act_dim:
                    dims = (int(traj_obs.shape[1]), int(traj_actions.shape[1]))
                    mismatched_dim_counts[dims] = mismatched_dim_counts.get(dims, 0) + 1
                    continue

                episodes.append(
                    DemoEpisode(
                        observations=torch.from_numpy(traj_obs).float(),
                        actions=torch.from_numpy(traj_actions).float(),
                        source_path=h5_path,
                        episode_id=traj_key,
                    )
                )

    if not episodes:
        raise RuntimeError(
            "No valid demo episodes matched env obs/action dims "
            f"{expected_obs_dim}/{expected_act_dim} from path {data_path}. "
            f"Available trajectory dims: {format_dim_counts(mismatched_dim_counts)}"
        )
    return episodes


def _load_lerobot_demo_episodes(
    data_path: str,
    *,
    expected_obs_dim: int,
    expected_act_dim: int,
) -> list[DemoEpisode]:
    dataset_roots = resolve_lerobot_dataset_roots(data_path)
    if not dataset_roots:
        raise RuntimeError(f"No LeRobot dataset roots found from path: {data_path}")

    pq = maybe_import_pyarrow()
    episodes: list[DemoEpisode] = []
    mismatched_dim_counts: dict[tuple[int, int], int] = {}

    for dataset_root in dataset_roots:
        info_path = dataset_root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        features = info.get("features", {})
        require_strict_recording_certificate(dataset_root, features)

        obs_feature = features.get("observation.state")
        action_feature = features.get("action")
        if obs_feature is None or action_feature is None:
            raise RuntimeError(
                f"LeRobot dataset `{dataset_root}` is missing `observation.state` or `action` features."
            )

        obs_dim = _feature_flat_dim(obs_feature)
        act_dim = _feature_flat_dim(action_feature)
        episode_count_hint = max(1, int(info.get("total_episodes", 0) or 0))
        if obs_dim != expected_obs_dim or act_dim != expected_act_dim:
            dims = (obs_dim, act_dim)
            mismatched_dim_counts[dims] = mismatched_dim_counts.get(dims, 0) + episode_count_hint
            continue

        data_paths = sorted((dataset_root / "data").rglob("*.parquet"))
        if not data_paths:
            continue

        columns = ["episode_index", "frame_index", "action", "observation.state"]
        data = pq.read_table(data_paths, columns=columns).to_pydict()
        raw_episode_indices = np.asarray(data.get("episode_index", []), dtype=np.int64)
        if raw_episode_indices.size == 0:
            continue

        raw_frame_indices = np.asarray(data.get("frame_index", []), dtype=np.int64)
        if raw_frame_indices.size != raw_episode_indices.size:
            raw_frame_indices = np.arange(raw_episode_indices.size, dtype=np.int64)

        order = np.lexsort((raw_frame_indices, raw_episode_indices))
        episode_indices = raw_episode_indices[order]
        actions = np.asarray([data["action"][i] for i in order], dtype=np.float32)
        observations = np.asarray([data["observation.state"][i] for i in order], dtype=np.float32)

        for episode_id in np.unique(episode_indices).tolist():
            mask = episode_indices == int(episode_id)
            episode_obs = observations[mask]
            episode_actions = actions[mask]
            if episode_obs.shape[0] == 0 or episode_actions.shape[0] == 0:
                continue

            episodes.append(
                DemoEpisode(
                    observations=torch.from_numpy(episode_obs).float(),
                    actions=torch.from_numpy(episode_actions).float(),
                    source_path=dataset_root,
                    episode_id=str(int(episode_id)),
                )
            )

    if not episodes:
        raise RuntimeError(
            "No valid demo episodes matched env obs/action dims "
            f"{expected_obs_dim}/{expected_act_dim} from path {data_path}. "
            f"Available trajectory dims: {format_dim_counts(mismatched_dim_counts)}"
        )
    return episodes


def load_demo_episodes(
    data_path: str,
    *,
    expected_obs_dim: int,
    expected_act_dim: int,
) -> list[DemoEpisode]:
    lerobot_roots = resolve_lerobot_dataset_roots(data_path)
    if lerobot_roots:
        return _load_lerobot_demo_episodes(
            data_path,
            expected_obs_dim=expected_obs_dim,
            expected_act_dim=expected_act_dim,
        )
    return _load_h5_demo_episodes(
        data_path,
        expected_obs_dim=expected_obs_dim,
        expected_act_dim=expected_act_dim,
    )


class DiffusionSequenceDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[DemoEpisode],
        *,
        n_obs_steps: int,
        horizon: int,
        drop_n_last_frames: int = 0,
    ) -> None:
        if n_obs_steps <= 0:
            raise ValueError(f"`n_obs_steps` must be positive, got {n_obs_steps}.")
        if horizon <= 0:
            raise ValueError(f"`horizon` must be positive, got {horizon}.")
        if drop_n_last_frames < 0:
            raise ValueError(f"`drop_n_last_frames` must be non-negative, got {drop_n_last_frames}.")

        self.episodes = list(episodes)
        self.n_obs_steps = int(n_obs_steps)
        self.horizon = int(horizon)
        self.drop_n_last_frames = int(drop_n_last_frames)
        self.sample_index: list[tuple[int, int]] = []

        for episode_idx, episode in enumerate(self.episodes):
            if episode.length <= 0:
                continue
            usable_length = max(0, episode.length - self.drop_n_last_frames)
            for step_idx in range(usable_length):
                self.sample_index.append((episode_idx, step_idx))

        if not self.sample_index:
            raise RuntimeError(
                "No diffusion training windows were created. Check the demo dataset length and "
                "`drop_n_last_frames`."
            )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_idx, step_idx = self.sample_index[index]
        episode = self.episodes[episode_idx]
        episode_len = episode.length

        obs_indices = torch.arange(
            step_idx - self.n_obs_steps + 1,
            step_idx + 1,
            dtype=torch.long,
        ).clamp_(0, episode_len - 1)
        action_start = step_idx - self.n_obs_steps + 1
        action_indices = torch.arange(action_start, action_start + self.horizon, dtype=torch.long)
        action_is_pad = torch.logical_or(action_indices < 0, action_indices >= episode_len)
        action_indices = action_indices.clamp_(0, episode_len - 1)

        return {
            "observation.state": episode.observations.index_select(0, obs_indices),
            "action": episode.actions.index_select(0, action_indices),
            "action_is_pad": action_is_pad,
        }


class LeRobotEpisodeRecorder:
    def __init__(
        self,
        *,
        dataset_root: str | Path,
        repo_id: str,
        fps: int,
        robot_type: str,
        task: str | None = None,
        include_env_state: bool = True,
        include_success: bool = True,
        include_seed: bool = True,
        include_source_hash: bool = False,
        overwrite: bool = False,
        replace_episode_seed: int | None = None,
        use_videos: bool = False,
        next_images: bool = True,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"`fps` must be positive, got {fps}.")

        self.dataset_root = Path(dataset_root).resolve()
        self.repo_id = str(repo_id)
        self.fps = int(fps)
        self.robot_type = str(robot_type)
        self.default_task = task
        self.include_env_state = bool(include_env_state)
        self.include_success = bool(include_success)
        self.include_seed = bool(include_seed)
        self.include_source_hash = bool(include_source_hash)
        self.overwrite = bool(overwrite)
        self.replace_episode_seed = (
            None if replace_episode_seed is None else int(replace_episode_seed)
        )
        self.use_videos = bool(use_videos)
        self.next_images = bool(next_images)

        self._schema: DatasetSchema | None = None
        self._dataset = None
        self._num_episodes_added = 0
        self._num_frames_added = 0
        self._finalized = False

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("Recorder is already finalized.")

    def _ensure_dataset(self, schema: DatasetSchema) -> None:
        if self._dataset is not None:
            if self._schema != schema:
                raise RuntimeError(
                    "Episode schema does not match the existing LeRobot dataset. "
                    f"Expected obs_dim={self._schema.obs_dim}, action_dim={self._schema.action_dim}, "
                    f"env_state_dim={self._schema.env_state_dim}, image_fields={self._schema.image_fields}; "
                    f"got obs_dim={schema.obs_dim}, action_dim={schema.action_dim}, "
                    f"env_state_dim={schema.env_state_dim}, image_fields={schema.image_fields}."
                )
            return

        features = _build_features(
            schema,
            include_success=self.include_success,
            include_seed=self.include_seed,
            include_source_hash=self.include_source_hash,
            use_videos=self.use_videos,
            next_images=self.next_images,
        )
        self._dataset = _create_or_resume_dataset(
            dataset_root=self.dataset_root,
            repo_id=self.repo_id,
            fps=self.fps,
            robot_type=self.robot_type,
            features=features,
            overwrite=self.overwrite,
            replace_episode_seed=self.replace_episode_seed,
            use_videos=self.use_videos,
        )
        self._schema = schema

    def add_episode(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        task: str | None = None,
        success: np.ndarray | None = None,
        episode_seed: int | None = None,
        source_hash: int | None = None,
        env_state_sequence: Sequence[dict[str, Any]] | None = None,
        image_sequences: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self._ensure_open()

        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        terminated = np.asarray(terminated, dtype=np.bool_).reshape(-1)
        truncated = np.asarray(truncated, dtype=np.bool_).reshape(-1)
        if self.include_source_hash and source_hash is None:
            raise RuntimeError(
                "Strict LeRobot export requires a non-null episode source hash."
            )
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("Observation sequence contains non-finite values.")
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("Action sequence contains non-finite values.")
        if not np.all(np.isfinite(rewards)):
            raise RuntimeError("Reward sequence contains non-finite values.")


        if len(obs.shape) != 2:
            raise RuntimeError(f"`obs` must be rank-2, got {obs.shape}.")
        if len(actions.shape) != 2:
            raise RuntimeError(f"`actions` must be rank-2, got {actions.shape}.")
        if actions.shape[0] == 0:
            raise RuntimeError("Cannot export an episode with zero transitions.")
        if obs.shape[0] != actions.shape[0] + 1:
            raise RuntimeError(
                f"Inconsistent episode lengths: obs={obs.shape[0]}, actions={actions.shape[0]}."
            )
        if len(rewards) != len(actions):
            raise RuntimeError(
                f"`rewards` length {len(rewards)} does not match action length {len(actions)}."
            )
        if len(terminated) != len(actions) or len(truncated) != len(actions):
            raise RuntimeError("Done-flag lengths do not match the action length.")

        env_state_fields: tuple[EnvStateField, ...] = ()
        env_state = None
        if self.include_env_state and env_state_sequence is not None:
            if len(env_state_sequence) != obs.shape[0]:
                raise RuntimeError(
                    "Environment-state sequence length does not match observation length: "
                    f"env_state={len(env_state_sequence)}, obs={obs.shape[0]}."
                )
            env_state_fields = _inspect_env_state_fields_from_sequence(env_state_sequence)
            env_state = _flatten_env_state_sequence_from_dicts(
                env_state_sequence,
                env_state_fields,
            )

        normalized_images: dict[str, np.ndarray] = {}
        image_fields: list[tuple[str, tuple[int, int, int]]] = []
        for key, values in (image_sequences or {}).items():
            images = np.asarray(values)
            if images.ndim != 4 or images.shape[-1] != 3:
                raise RuntimeError(
                    f"Image sequence `{key}` must have shape (T,H,W,3), got {images.shape}."
                )
            if images.shape[0] != obs.shape[0]:
                raise RuntimeError(
                    f"Image sequence `{key}` length {images.shape[0]} does not match "
                    f"observation length {obs.shape[0]}."
                )
            if images.dtype != np.uint8:
                raise RuntimeError(
                    f"Image sequence `{key}` must use uint8 pixels, got {images.dtype}."
                )
            normalized_images[str(key)] = np.ascontiguousarray(images)
            image_fields.append(
                (
                    str(key),
                    tuple(int(dim) for dim in images.shape[1:]),
                )
            )

        schema = DatasetSchema(
            obs_dim=int(obs.shape[1]),
            action_dim=int(actions.shape[1]),
            env_state_fields=env_state_fields,
            image_fields=tuple(sorted(image_fields)),
        )
        self._ensure_dataset(schema)

        success_values = None
        if self.include_success:
            if success is None:
                success_values = np.zeros((len(actions),), dtype=np.bool_)
            else:
                success_values = np.asarray(success, dtype=np.bool_).reshape(-1)
                if len(success_values) != len(actions):
                    raise RuntimeError(
                        f"`success` length {len(success_values)} does not match action length {len(actions)}."
                    )

        task_string = task or self.default_task or self.repo_id
        add_frame_accepts_task = "task" in inspect.signature(
            self._dataset.add_frame
        ).parameters
        stored_seed = -1 if episode_seed is None else int(episode_seed)
        stored_source_hash = -1 if source_hash is None else int(source_hash)
        for step_idx in range(actions.shape[0]):
            terminated_value = bool(terminated[step_idx])
            truncated_value = bool(truncated[step_idx])
            done_value = bool(terminated_value or truncated_value)
            frame: dict[str, Any] = {
                "observation.state": obs[step_idx],
                "next.observation.state": obs[step_idx + 1],
                "action": actions[step_idx],
                "next.reward": np.asarray([rewards[step_idx]], dtype=np.float32),
                "next.done": np.asarray([done_value], dtype=np.bool_),
                "next.terminated": np.asarray([terminated_value], dtype=np.bool_),
                "next.truncated": np.asarray([truncated_value], dtype=np.bool_),
                "next.recording_boundary": np.asarray(
                    [step_idx == actions.shape[0] - 1], dtype=np.bool_
                ),
            }
            if self.include_source_hash:
                frame["episode.source_hash"] = np.asarray(
                    [stored_source_hash], dtype=np.int64
                )
            if env_state is not None:
                frame["observation.environment_state"] = env_state[step_idx]
                frame["next.observation.environment_state"] = env_state[step_idx + 1]
            if self.include_success and success_values is not None:
                frame["next.success"] = np.asarray([bool(success_values[step_idx])], dtype=np.bool_)
            if self.include_seed:
                frame["episode.seed"] = np.asarray([stored_seed], dtype=np.int64)
            for key, images in normalized_images.items():
                frame[key] = images[step_idx]
                if self.next_images:
                    frame[f"next.{key}"] = images[step_idx + 1]
            if add_frame_accepts_task:
                # Newer LeRobot versions add the task column internally.
                self._dataset.add_frame(frame, task=task_string)
            else:
                # Older LeRobot versions expect task to be part of the frame.
                frame["task"] = task_string
                self._dataset.add_frame(frame)

        self._dataset.save_episode()
        self._num_episodes_added += 1
        self._num_frames_added += int(actions.shape[0])

    def finalize(self) -> ExportSummary:
        if not self._finalized and self._dataset is not None:
            # LeRobot v0.3 exposed ``finalize``; newer versions persist metadata
            # during ``save_episode`` and no longer provide that method.
            finalize_dataset = getattr(self._dataset, "finalize", None)
            if callable(finalize_dataset):
                finalize_dataset()
            _write_episode_manifest(self.dataset_root)
        self._finalized = True

        if self._schema is None:
            return ExportSummary(
                dataset_root=self.dataset_root,
                repo_id=self.repo_id,
                num_source_files=0,
                num_episodes_added=0,
                num_frames_added=0,
                total_episodes=0,
                total_frames=0,
                obs_dim=0,
                action_dim=0,
                env_state_dim=0,
            )

        dataset_meta = getattr(self._dataset, "meta", None)
        total_episodes = int(getattr(dataset_meta, "total_episodes", self._num_episodes_added))
        total_frames = int(getattr(dataset_meta, "total_frames", self._num_frames_added))
        return ExportSummary(
            dataset_root=self.dataset_root,
            repo_id=self.repo_id,
            num_source_files=0,
            num_episodes_added=self._num_episodes_added,
            num_frames_added=self._num_frames_added,
            total_episodes=total_episodes,
            total_frames=total_frames,
            obs_dim=self._schema.obs_dim,
            action_dim=self._schema.action_dim,
            env_state_dim=self._schema.env_state_dim,
        )


class DemoStateInitializer:
    def __init__(
        self,
        *,
        reset_env_states: dict[str, dict[str, torch.Tensor]],
        reset_obs: Optional[torch.Tensor] = None,
        demo_progress: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        init_prob: float = 1.0,
        min_prob: float = 0.0,
        decay: bool = True,
        decay_steps: int = 1,
        curriculum_mode: str = "uniform",
        curriculum_start_frac: float = 0.2,
        curriculum_steps: int = 1,
    ):
        self.reset_env_states = reset_env_states
        self.reset_obs = reset_obs
        self.demo_progress = demo_progress
        self.rng = np.random.default_rng(seed)
        self.init_prob = float(init_prob)
        self.min_prob = float(min_prob)
        self.decay = bool(decay)
        self.decay_steps = max(1, int(decay_steps))
        self.curriculum_mode = str(curriculum_mode).strip().lower()
        self.curriculum_start_frac = float(np.clip(curriculum_start_frac, 0.0, 1.0))
        self.curriculum_steps = max(1, int(curriculum_steps))
        self.current_prob = float(np.clip(self.init_prob, 0.0, 1.0))
        self.current_min_progress = 0.0
        self.num_samples = 0
        for category_items in self.reset_env_states.values():
            for value in category_items.values():
                self.num_samples = int(value.shape[0])
                break
            if self.num_samples > 0:
                break
        if self.num_samples <= 0:
            raise ValueError("DemoStateInitializer requires at least one reset sample.")
        if self.curriculum_mode not in {"uniform", "progress"}:
            raise ValueError(
                f"Unsupported demo-init curriculum mode `{self.curriculum_mode}`. "
                "Expected `uniform` or `progress`."
            )
        if self.curriculum_mode == "progress":
            if self.demo_progress is None:
                raise ValueError("Progress curriculum requires per-sample demo progress values.")
            if int(self.demo_progress.shape[0]) != self.num_samples:
                raise ValueError(
                    "Demo progress length does not match reset-state count: "
                    f"{int(self.demo_progress.shape[0])} vs {self.num_samples}."
                )
            demo_progress_np = np.asarray(self.demo_progress.detach().cpu(), dtype=np.float32).reshape(-1)
            sort_order = np.argsort(demo_progress_np, kind="stable")
            self.sorted_demo_progress = demo_progress_np[sort_order]
            self.sorted_sample_inds = sort_order
        else:
            self.sorted_demo_progress = None
            self.sorted_sample_inds = None

    def set_train_progress(self, global_step: int) -> None:
        if not self.decay:
            self.current_prob = float(np.clip(self.init_prob, 0.0, 1.0))
        else:
            progress = min(max(global_step / max(1, self.decay_steps), 0.0), 1.0)
            prob = self.init_prob * (1.0 - progress)
            self.current_prob = float(np.clip(max(self.min_prob, prob), 0.0, 1.0))

        if self.curriculum_mode != "progress":
            self.current_min_progress = 0.0
            return

        curriculum_progress = min(max(global_step / max(1, self.curriculum_steps), 0.0), 1.0)
        visible_fraction = self.curriculum_start_frac + curriculum_progress * (1.0 - self.curriculum_start_frac)
        self.current_min_progress = float(np.clip(1.0 - visible_fraction, 0.0, 1.0))

    def get_current_prob(self) -> float:
        return self.current_prob

    def get_current_curriculum_min_progress(self) -> float:
        return self.current_min_progress

    def _sample_demo_mask(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0 or self.current_prob <= 0.0:
            return np.zeros(batch_size, dtype=bool)
        if self.current_prob >= 1.0:
            return np.ones(batch_size, dtype=bool)
        return self.rng.random(batch_size) < self.current_prob

    def _sample_demo_indices(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        if self.curriculum_mode != "progress":
            return self.rng.integers(0, self.num_samples, size=count, dtype=np.int64)

        sorted_demo_progress = self.sorted_demo_progress
        sorted_sample_inds = self.sorted_sample_inds
        assert sorted_demo_progress is not None
        assert sorted_sample_inds is not None
        eligible_start = int(np.searchsorted(sorted_demo_progress, self.current_min_progress, side="left"))
        eligible_inds = sorted_sample_inds[eligible_start:]
        if eligible_inds.size == 0:
            eligible_inds = sorted_sample_inds[-1:]
        sampled_pos = self.rng.integers(0, eligible_inds.shape[0], size=count, dtype=np.int64)
        return eligible_inds[sampled_pos]

    def build_reset_options(
        self,
        *,
        device: torch.device,
        num_envs: int,
        options: Optional[dict] = None,
    ) -> Optional[dict]:
        merged = {} if options is None else dict(options)
        if "reset_to_env_states" in merged:
            return merged

        env_idx = merged.get("env_idx")
        if env_idx is None:
            batch_size = int(num_envs)
        else:
            batch_size = int(torch.as_tensor(env_idx).numel())
        if batch_size <= 0:
            return merged

        demo_mask = self._sample_demo_mask(batch_size)
        if not demo_mask.any():
            return merged

        sample_inds = torch.as_tensor(self._sample_demo_indices(int(demo_mask.sum())), dtype=torch.long)
        if env_idx is None:
            selected_env_idx = torch.arange(batch_size, dtype=torch.long)
        else:
            selected_env_idx = torch.as_tensor(env_idx, dtype=torch.long).view(-1)
        selected_env_idx = selected_env_idx[torch.as_tensor(demo_mask, dtype=torch.bool)]

        env_states: dict[str, dict[str, torch.Tensor]] = {}
        for category, category_items in self.reset_env_states.items():
            env_states[category] = {}
            for name, values in category_items.items():
                selected_values = values.index_select(0, sample_inds)
                if selected_values.ndim >= 2 and selected_values.shape[1] == 1:
                    selected_values = selected_values.squeeze(1)
                env_states[category][name] = selected_values.to(
                    device=device,
                    non_blocking=True,
                )

        reset_payload = {"env_states": env_states}
        if self.reset_obs is not None:
            reset_payload["obs"] = self.reset_obs.index_select(0, sample_inds).to(
                device=device,
                non_blocking=True,
            )
        merged["env_idx"] = selected_env_idx.to(device=device)
        merged["reset_to_env_states"] = reset_payload
        return merged


class DemoInitManiSkillVectorEnv(ManiSkillVectorEnv):
    def __init__(self, *args, demo_state_initializer: Optional[DemoStateInitializer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_state_initializer = demo_state_initializer

    def reset(self, *, seed=None, options=None):
        if self.demo_state_initializer is not None:
            base_options = {} if options is None else dict(options)
            demo_options = self.demo_state_initializer.build_reset_options(
                device=self.base_env.device,
                num_envs=self.num_envs,
                options=base_options,
            )
            if demo_options is None or "reset_to_env_states" not in demo_options:
                return super().reset(seed=seed, options=base_options)

            base_env_idx = base_options.get("env_idx")
            demo_env_idx = demo_options.get("env_idx")
            base_batch_size = self.num_envs if base_env_idx is None else int(torch.as_tensor(base_env_idx).numel())
            demo_batch_size = 0 if demo_env_idx is None else int(torch.as_tensor(demo_env_idx).numel())

            if demo_batch_size <= 0:
                return super().reset(seed=seed, options=base_options)
            if demo_batch_size == base_batch_size:
                return super().reset(seed=seed, options=demo_options)

            super().reset(seed=seed, options=base_options)
            return super().reset(seed=None, options=demo_options)
        return super().reset(seed=seed, options=options)

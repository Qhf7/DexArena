# General ACT image trainer v2: zero expected dimensions mean auto-infer.
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import importlib
import inspect
import json
import re
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Custom environments are registered explicitly through ``--task-module``.
# Importing a submodule first executes tasks/__init__.py; without this guard it
# imports every RoboDex task and can silently replace ManiSkill's stock Panda
# agents (notably panda_wristcam) with the repository impedance variants.  A
# checkpoint must be evaluated with the same controller that produced its
# demonstrations, so avoid that process-global registration side effect.
os.environ.setdefault("ROBODEX_SKIP_AUTO_TASK_IMPORTS", "1")

import gymnasium as gym
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import swanlab
except Exception:
    swanlab = None

from train.train_configs.ACT_config import parse_args


@dataclass(frozen=True)
class JointPosResidualSpec:
    """Policy-only residual parameterization for absolute joint actions."""

    enabled: bool = False
    action_start: int = 0
    qpos_start: int = 0
    dim: int = 7
    scale: float = 0.4

    def validate(self, action_dim: int, qpos_dim: int) -> None:
        if not self.enabled:
            return
        if self.dim <= 0:
            raise ValueError("joint_pos_residual_dim must be positive")
        if self.scale <= 0:
            raise ValueError("joint_pos_residual_scale must be positive")
        if self.action_start < 0 or self.action_start + self.dim > action_dim:
            raise ValueError(
                "Residual action slice is out of bounds: "
                f"start={self.action_start} dim={self.dim} action_dim={action_dim}"
            )
        if self.qpos_start < 0 or self.qpos_start + self.dim > qpos_dim:
            raise ValueError(
                "Residual qpos slice is out of bounds: "
                f"start={self.qpos_start} dim={self.dim} qpos_dim={qpos_dim}"
            )

    @property
    def action_slice(self) -> slice:
        return slice(self.action_start, self.action_start + self.dim)

    @property
    def qpos_slice(self) -> slice:
        return slice(self.qpos_start, self.qpos_start + self.dim)


def actions_to_policy(
    absolute_actions: np.ndarray,
    qpos: np.ndarray,
    spec: JointPosResidualSpec,
) -> np.ndarray:
    actions = np.asarray(absolute_actions, dtype=np.float32).copy()
    if not spec.enabled:
        return actions
    qpos = np.asarray(qpos, dtype=np.float32)
    spec.validate(actions.shape[-1], qpos.shape[-1])
    actions[..., spec.action_slice] = (
        actions[..., spec.action_slice] - qpos[..., spec.qpos_slice]
    ) / float(spec.scale)
    return actions


def policy_to_env_action(
    policy_action: np.ndarray,
    qpos: np.ndarray,
    spec: JointPosResidualSpec,
) -> np.ndarray:
    action = np.asarray(policy_action, dtype=np.float32).copy()
    if not spec.enabled:
        return action
    qpos = np.asarray(qpos, dtype=np.float32)
    spec.validate(action.shape[-1], qpos.shape[-1])
    action[..., spec.action_slice] = (
        qpos[..., spec.qpos_slice]
        + float(spec.scale) * action[..., spec.action_slice]
    )
    return action


def robot_qpos_from_env(env: Any) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    agent = getattr(base_env, "agent", None)
    robot = getattr(agent, "robot", None)
    if robot is None or not hasattr(robot, "get_qpos"):
        raise RuntimeError("Cannot read robot qpos from the evaluation environment")
    value = robot.get_qpos()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    qpos = np.asarray(value, dtype=np.float32)
    if qpos.ndim >= 2:
        qpos = qpos[0]
    return qpos.reshape(-1)


def load_aligned_lerobot_states(
    source_dataset: str,
    derived_dataset_root: Path,
    obs_key: str = "observation.state",
) -> np.ndarray:
    source = Path(source_dataset).expanduser().resolve()
    parquet_files = sorted((source / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {source / 'data'}")

    frames = []
    for path in parquet_files:
        frame = pd.read_parquet(path)
        if obs_key not in frame.columns:
            raise KeyError(f"Dataset {path} does not contain {obs_key!r}")
        frames.append(frame)
    source_frame = pd.concat(frames, ignore_index=True)
    states = np.stack(source_frame[obs_key].to_numpy()).astype(np.float32)

    global_index_path = Path(derived_dataset_root) / "global_index.npy"
    if not global_index_path.is_file():
        raise FileNotFoundError(
            "Residual image training requires global_index.npy for source-state alignment"
        )
    requested = np.asarray(np.load(global_index_path), dtype=np.int64).reshape(-1)

    if "index" in source_frame.columns:
        source_indices = source_frame["index"].to_numpy(dtype=np.int64)
        if len(np.unique(source_indices)) != len(source_indices):
            raise RuntimeError("Source dataset index column is not unique")
        row_by_index = {int(index): row for row, index in enumerate(source_indices)}
        try:
            rows = np.asarray(
                [row_by_index[int(index)] for index in requested], dtype=np.int64
            )
        except KeyError as error:
            raise KeyError(
                f"Derived global index {int(error.args[0])} is absent from source dataset"
            ) from error
    else:
        if requested.size and (requested.min() < 0 or requested.max() >= len(states)):
            raise IndexError("Derived global_index.npy is outside the source dataset")
        rows = requested

    aligned = np.asarray(states[rows], dtype=np.float32)
    if len(aligned) != len(requested):
        raise RuntimeError("Residual state alignment changed the frame count")
    return aligned


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def build_run_name(args) -> str:
    if args.exp_name is None:
        name = Path(__file__).stem
        run_name = f"{args.env_id}__{name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    return "eval_" + run_name if args.evaluate else run_name


def set_random_seed(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(args) -> torch.device:
    if args.cuda and torch.cuda.is_available():
        if args.gpu_id is not None:
            return torch.device(f"cuda:{args.gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_apply_gpu_id(
    backend: Optional[str],
    gpu_id: Optional[int],
    gpu_backends: set[str],
) -> Optional[str]:
    if (
        backend is None
        or gpu_id is None
        or ":" in backend
        or backend not in gpu_backends
    ):
        return backend
    return f"{backend}:{gpu_id}"


def parse_value(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_env_kwargs(items: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"env_kwarg must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        result[key] = parse_value(value)
    return result


def parse_camera_map(items: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items:
        left, right = item.split("=", 1)
        result[left.strip()] = right.strip()
    return result


def parse_int_map(items: Sequence[str], label: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{label} must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        value = int(raw_value)
        if value < 0:
            raise ValueError(f"{label} values must be non-negative, got {item!r}")
        result[key.strip()] = value
    return result


def import_task_modules(task_module: Optional[str]) -> None:
    if not task_module:
        return
    for module in task_module.split(","):
        module = module.strip()
        if module:
            importlib.import_module(module)


def make_env(args, capture_video: bool):
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    import_task_modules(args.task_module)

    sim_backend = args.sim_backend or (
        "physx_cuda" if torch.cuda.is_available() and args.cuda else "physx_cpu"
    )
    render_backend = (
        "sapien_cuda" if torch.cuda.is_available() and args.cuda else "sapien_cpu"
    )

    env_kwargs: Dict[str, Any] = dict(
        obs_mode=args.obs_mode,
        render_mode="rgb_array" if capture_video else None,
        sim_backend=maybe_apply_gpu_id(
            sim_backend, args.gpu_id, {"physx_cuda", "gpu", "cuda"}
        ),
        render_backend=maybe_apply_gpu_id(
            render_backend, args.gpu_id, {"sapien_cuda", "gpu", "cuda"}
        ),
        max_episode_steps=args.num_eval_steps,
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids
    if args.eval_sensor_size is not None:
        env_kwargs["sensor_configs"] = {
            "width": int(args.eval_sensor_size),
            "height": int(args.eval_sensor_size),
        }

    env_kwargs.update(parse_env_kwargs(args.env_kwarg))

    make_kwargs = dict(env_kwargs)
    make_kwargs["num_envs"] = 1
    if args.eval_reconfiguration_freq is not None:
        make_kwargs["reconfiguration_freq"] = args.eval_reconfiguration_freq

    env = gym.make(
        args.env_id,
        **make_kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env, env_kwargs


def to_np(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_bool(value: Any) -> bool:
    if value is None:
        return False
    array = to_np(value).reshape(-1)
    return bool(array.size and array.astype(bool).any())


def scalar_float(value: Any) -> float:
    if value is None:
        return float("nan")
    array = to_np(value).astype(np.float32).reshape(-1)
    return float(array.mean()) if array.size else float("nan")


def format_action(env: gym.Env, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)

    if low.shape == action.shape:
        return np.clip(action, low, high)
    if low.size == action.size:
        return np.clip(action.reshape(low.shape), low, high)
    if low.ndim >= 2 and low.shape[-1] == action.size:
        return np.clip(np.broadcast_to(action, low.shape).copy(), low, high)
    return np.clip(action, low.min(), high.max())


def extract_frame(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for nested in value.values():
            frame = extract_frame(nested)
            if frame is not None:
                return frame
        return None
    if isinstance(value, (list, tuple)):
        for nested in value:
            frame = extract_frame(nested)
            if frame is not None:
                return frame
        return None

    array = to_np(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] < 3:
        return None

    array = array[..., :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(array.max()) <= 1.5:
            array *= 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def safe_render(env: gym.Env) -> Optional[np.ndarray]:
    try:
        return extract_frame(env.render())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraInfo:
    name: str
    file: str
    height: int
    width: int


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: int
    start: int
    end: int
    success: bool


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    num_frames: int
    raw_state_dim: int
    state_dim: int
    state_indices: Tuple[int, ...]
    action_dim: int
    cameras: Tuple[CameraInfo, ...]


def parse_state_slice(spec: str, raw_dim: int) -> Tuple[int, ...]:
    """Parse an index/range specification such as ``all`` or ``0:25,30``."""
    normalized_spec = str(spec).strip().lower()
    if normalized_spec in {"", "all", "*", ":"}:
        return tuple(range(raw_dim))

    indices: List[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid state slice token: {token!r}")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if parts[1] else raw_dim
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step == 0:
                raise ValueError(f"State slice step cannot be zero: {token!r}")
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(token))

    if not indices:
        raise ValueError("offline_state_slice selected no dimensions")

    normalized = tuple(
        index + raw_dim if index < 0 else index
        for index in indices
    )
    if any(index < 0 or index >= raw_dim for index in normalized):
        raise ValueError(
            f"offline_state_slice={spec!r} is out of bounds for raw_dim={raw_dim}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"offline_state_slice contains duplicate indices: {spec!r}")
    return normalized


class Stats:
    def __init__(self, obs_mean, obs_std, action_mean, action_std):
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Stats":
        return cls(
            data["obs_mean"],
            data["obs_std"],
            data["action_mean"],
            data["action_std"],
        )


def parse_camera_output_shape(shape: Sequence[int]) -> Tuple[int, int]:
    """Return ``(height, width)`` from image or full-array metadata.

    Supported metadata layouts:
      - HWC
      - CHW
      - NHWC
      - NCHW

    Some preprocessing versions store the complete NumPy array shape in
    ``dataset_info.json`` (for example ``[11152, 128, 128, 3]``), while older
    versions store only the per-frame image shape. Both are valid here.
    """

    dims = tuple(int(value) for value in shape)

    if len(dims) == 4:
        # The leading dimension is the number of frames.
        dims = dims[1:]

    if len(dims) != 3:
        raise ValueError(
            "Camera output_shape must be HWC/CHW or NHWC/NCHW, "
            f"got {tuple(int(value) for value in shape)}"
        )

    if dims[0] in (1, 3, 4):
        # CHW
        return dims[1], dims[2]

    if dims[-1] in (1, 3, 4):
        # HWC
        return dims[0], dims[1]

    raise ValueError(
        "Cannot infer camera layout from output_shape="
        f"{dims}; expected HWC or CHW after removing an optional "
        "leading frame dimension"
    )


def load_dataset_info(
    root_path: str,
    requested_cameras: Sequence[str],
    offline_state_slice: str,
    expected_raw_state_dim: int,
    expected_policy_state_dim: int,
) -> DatasetInfo:
    root = Path(root_path).expanduser().resolve()
    info_path = root / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")

    with open(info_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    available: Dict[str, CameraInfo] = {}
    for item in raw["cameras"]:
        height, width = parse_camera_output_shape(item["output_shape"])
        available[item["name"]] = CameraInfo(
            name=item["name"],
            file=item["output_file"],
            height=height,
            width=width,
        )

    names = list(requested_cameras) if requested_cameras else list(available)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            f"Requested cameras do not exist: {missing}; available={list(available)}"
        )

    state = np.load(root / "state.npy", mmap_mode="r")
    action = np.load(root / "action.npy", mmap_mode="r")
    cameras = tuple(available[name] for name in names)

    if state.ndim != 2 or action.ndim != 2:
        raise ValueError("state.npy and action.npy must both be 2-D")
    if len(state) != len(action):
        raise ValueError("state/action frame counts do not match")

    raw_state_dim = int(state.shape[1])
    if int(expected_raw_state_dim) > 0 and raw_state_dim != int(
        expected_raw_state_dim
    ):
        raise RuntimeError(
            f"Raw offline state dimension changed: got {raw_state_dim}, "
            f"expected {expected_raw_state_dim}."
        )

    state_indices = parse_state_slice(offline_state_slice, raw_state_dim)
    if (
        int(expected_policy_state_dim) > 0
        and len(state_indices) != int(expected_policy_state_dim)
    ):
        raise RuntimeError(
            f"Policy state slice {offline_state_slice!r} selects "
            f"{len(state_indices)} dims, expected {expected_policy_state_dim}."
        )

    for camera in cameras:
        images = np.load(root / camera.file, mmap_mode="r")

        if images.ndim != 4:
            raise ValueError(
                f"Camera {camera.name} array must be 4-D, got {images.shape}"
            )

        if images.shape[0] != len(state):
            raise ValueError(
                f"Camera {camera.name} frame count is {images.shape[0]}, "
                f"but state/action contain {len(state)} frames"
            )

        # The preprocessing script currently writes NHWC uint8 arrays.
        expected_frame_shape = (camera.height, camera.width, 3)
        if tuple(images.shape[1:]) != expected_frame_shape:
            raise ValueError(
                f"Camera {camera.name} per-frame shape is {images.shape[1:]}, "
                f"expected {expected_frame_shape}; full shape={images.shape}"
            )

        if images.dtype != np.uint8:
            raise ValueError(
                f"Camera {camera.name} must be uint8, got {images.dtype}"
            )

    return DatasetInfo(
        root=root,
        num_frames=len(state),
        raw_state_dim=raw_state_dim,
        state_dim=len(state_indices),
        state_indices=state_indices,
        action_dim=int(action.shape[1]),
        cameras=cameras,
    )


def parse_csv_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return bool(default)
        return bool(value)
    low = str(value).strip().lower()
    if low in {"1", "true", "yes", "y", "success", "succeeded"}:
        return True
    if low in {"0", "false", "no", "n", "failure", "failed"}:
        return False
    return bool(default)


def load_episodes(
    info: DatasetInfo,
    filter_success: bool,
    include_ids: Sequence[int],
    exclude_ids: Sequence[int],
) -> List[EpisodeInfo]:
    table_path = info.root / "episodes.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing {table_path}")
    table = pd.read_csv(table_path)

    required = {"episode_index", "start", "end_exclusive"}
    missing_columns = sorted(required.difference(table.columns))
    if missing_columns:
        raise ValueError(
            f"episodes.csv is missing columns {missing_columns}; "
            f"available={list(table.columns)}"
        )

    has_success = "success" in table.columns
    if filter_success and not has_success:
        print(
            "[WARN] episodes.csv has no success column; "
            "filter_success cannot remove failures, so all episodes are kept"
        )

    include = set(int(value) for value in include_ids)
    exclude = set(int(value) for value in exclude_ids)
    available = set(table["episode_index"].astype(int).tolist())

    missing = sorted(include - available)
    if missing:
        raise ValueError(
            f"Included episode IDs do not exist: {missing}; available={sorted(available)}"
        )

    episodes: List[EpisodeInfo] = []
    skipped: List[int] = []

    for row in table.itertuples(index=False):
        episode_id = int(row.episode_index)
        success = parse_csv_bool(
            getattr(row, "success", True),
            default=True,
        )

        if include and episode_id not in include:
            continue
        if episode_id in exclude:
            continue
        if not include and filter_success and has_success and not success:
            skipped.append(episode_id)
            continue

        start = int(row.start)
        end = int(row.end_exclusive)
        if not 0 <= start < end <= info.num_frames:
            raise ValueError(
                f"Episode {episode_id} has invalid bounds [{start}, {end})"
            )
        episodes.append(EpisodeInfo(episode_id, start, end, success))

    if not episodes:
        raise RuntimeError("No episodes remain after filtering")

    print(f"[DATA] loaded_episode_ids={[episode.episode_id for episode in episodes]}")
    if skipped:
        print(f"[DATA] skipped_unsuccessful={skipped}")
    return episodes


def split_episodes(
    episodes: Sequence[EpisodeInfo],
    ratio: float,
    seed: int,
) -> Tuple[List[EpisodeInfo], List[EpisodeInfo]]:
    episodes = list(episodes)
    if ratio <= 0 or len(episodes) <= 1:
        return episodes, []

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    num_validation = max(
        1,
        min(len(episodes) - 1, int(round(len(episodes) * ratio))),
    )
    validation_indices = set(order[:num_validation].tolist())
    train = [
        episode
        for index, episode in enumerate(episodes)
        if index not in validation_indices
    ]
    validation = [
        episode
        for index, episode in enumerate(episodes)
        if index in validation_indices
    ]
    return train, validation


def compute_stats(
    info: DatasetInfo,
    episodes: Sequence[EpisodeInfo],
    state_std_floor: float,
    action_std_floor: float,
    residual_states: Optional[np.ndarray] = None,
    residual_spec: Optional[JointPosResidualSpec] = None,
) -> Stats:
    state = np.load(info.root / "state.npy", mmap_mode="r")
    action = np.load(info.root / "action.npy", mmap_mode="r")
    states = np.concatenate(
        [
            np.asarray(state[episode.start:episode.end], dtype=np.float32)[
                :, info.state_indices
            ]
            for episode in episodes
        ],
        axis=0,
    )
    action_parts: List[np.ndarray] = []
    for episode in episodes:
        part = np.asarray(
            action[episode.start:episode.end], dtype=np.float32
        )
        if residual_spec is not None and residual_spec.enabled:
            if residual_states is None:
                raise ValueError("Residual action stats require aligned source states")
            part = actions_to_policy(
                part,
                residual_states[episode.start:episode.end],
                residual_spec,
            )
        action_parts.append(part)
    actions = np.concatenate(action_parts, axis=0)
    state_std = np.maximum(
        states.std(0),
        float(state_std_floor),
    )
    action_std = np.maximum(
        actions.std(0),
        float(action_std_floor),
    )
    return Stats(
        states.mean(0),
        state_std,
        actions.mean(0),
        action_std,
    )


def random_shift(image: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return image
    _, height, width = image.shape
    padded = F.pad(
        image.unsqueeze(0),
        (pad, pad, pad, pad),
        mode="replicate",
    )[0]
    top = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    left = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    return padded[:, top:top + height, left:left + width]


class ACTImageDataset(Dataset):
    def __init__(
        self,
        info: DatasetInfo,
        episodes: Sequence[EpisodeInfo],
        stats: Stats,
        chunk_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        training: bool,
        random_shift_pad: int,
        action_padding_mode: str,
        min_chunk_valid_steps: int,
        residual_states: Optional[np.ndarray] = None,
        residual_spec: Optional[JointPosResidualSpec] = None,
    ):
        self.info = info
        self.episodes = list(episodes)
        self.stats = stats
        self.chunk_size = int(chunk_size)
        self.training = bool(training)
        self.random_shift_pad = int(random_shift_pad)
        self.action_padding_mode = str(action_padding_mode)
        self.min_chunk_valid_steps = int(min_chunk_valid_steps)
        self.residual_states = residual_states
        self.residual_spec = residual_spec
        self.image_mean = torch.tensor(
            image_mean, dtype=torch.float32
        ).view(3, 1, 1)
        self.image_std = torch.tensor(
            image_std, dtype=torch.float32
        ).view(3, 1, 1)

        self.index: List[Tuple[int, int]] = []
        for episode in self.episodes:
            minimum = min(
                self.min_chunk_valid_steps,
                episode.end - episode.start,
            )
            last_start = episode.end - minimum
            self.index.extend(
                (frame, episode.end)
                for frame in range(episode.start, last_start + 1)
            )

        self.state = None
        self.action = None
        self.images = None
        self._open()

    def _open(self) -> None:
        if self.state is None:
            self.state = np.load(self.info.root / "state.npy", mmap_mode="r")
        if self.action is None:
            self.action = np.load(self.info.root / "action.npy", mmap_mode="r")
        if self.images is None:
            self.images = {
                camera.name: np.load(
                    self.info.root / camera.file,
                    mmap_mode="r",
                )
                for camera in self.info.cameras
            }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["state"] = None
        state["action"] = None
        state["images"] = None
        return state

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        self._open()
        frame, episode_end = self.index[item]

        raw_state = np.array(self.state[frame], dtype=np.float32, copy=True)
        state = raw_state[np.asarray(self.info.state_indices, dtype=np.int64)]
        state = (state - self.stats.obs_mean) / self.stats.obs_std

        end = min(frame + self.chunk_size, episode_end)
        actions = np.array(
            self.action[frame:end],
            dtype=np.float32,
            copy=True,
        )
        if self.residual_spec is not None and self.residual_spec.enabled:
            if self.residual_states is None:
                raise RuntimeError("Residual image dataset has no aligned source states")
            actions = actions_to_policy(
                actions,
                self.residual_states[frame:end],
                self.residual_spec,
            )
        valid = len(actions)
        if valid <= 0:
            raise RuntimeError(f"Empty action chunk at frame {frame}")
        if valid < self.chunk_size:
            if self.action_padding_mode == "repeat_last":
                padding = np.repeat(
                    actions[-1:],
                    self.chunk_size - valid,
                    axis=0,
                )
            elif self.action_padding_mode == "zero":
                padding = np.zeros(
                    (
                        self.chunk_size - valid,
                        actions.shape[1],
                    ),
                    dtype=np.float32,
                )
            else:
                raise ValueError(
                    f"Unknown action_padding_mode={self.action_padding_mode!r}"
                )
            actions = np.concatenate([actions, padding], axis=0)
        actions = (
            actions - self.stats.action_mean
        ) / self.stats.action_std

        mask = np.zeros(self.chunk_size, dtype=np.float32)
        mask[:valid] = 1.0

        image_tensors: Dict[str, torch.Tensor] = {}
        for camera in self.info.cameras:
            image = np.array(
                self.images[camera.name][frame],
                dtype=np.uint8,
                copy=True,
            )
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if self.training:
                image = random_shift(image, self.random_shift_pad)
            image = (image - self.image_mean) / self.image_std
            image_tensors[camera.name] = image

        return {
            "state": torch.from_numpy(state.astype(np.float32)),
            "images": image_tensors,
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "mask": torch.from_numpy(mask),
        }


def normalize_action_dims(
    dims: Sequence[int],
    action_dim: int,
) -> Tuple[int, ...]:
    if not dims:
        return tuple(range(action_dim))
    normalized: List[int] = []
    for value in dims:
        index = int(value)
        if index < 0:
            index += action_dim
        if index < 0 or index >= action_dim:
            raise ValueError(
                f"static_action_dims contains {value}, "
                f"but action_dim={action_dim}"
            )
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError("static_action_dims contains duplicates")
    return tuple(normalized)


def build_static_sample_weights(
    info: DatasetInfo,
    index: Sequence[Tuple[int, int]],
    chunk_size: int,
    action_dims: Sequence[int],
    action_threshold: float,
    static_weight: float,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Build per-anchor weights without deleting trajectory frames."""
    if float(static_weight) >= 1.0:
        return None, {
            "static_windows": 0.0,
            "active_windows": float(len(index)),
            "static_fraction": 0.0,
        }

    dims = normalize_action_dims(action_dims, info.action_dim)
    action = np.load(info.root / "action.npy", mmap_mode="r")
    weights: List[float] = []
    static_count = 0
    active_count = 0

    for frame, episode_end in index:
        end = min(frame + int(chunk_size), episode_end)
        chunk = np.asarray(
            action[frame:end, dims],
            dtype=np.float32,
        )
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        has_motion = bool(
            np.any(
                np.linalg.norm(chunk, axis=1)
                > float(action_threshold)
            )
        )
        if has_motion:
            weights.append(1.0)
            active_count += 1
        else:
            weights.append(float(static_weight))
            static_count += 1

    tensor = torch.as_tensor(weights, dtype=torch.double)
    if tensor.numel() == 0 or float(tensor.sum()) <= 0:
        raise RuntimeError(
            "Static-window weighting produced no positive sample weights"
        )

    total = static_count + active_count
    summary = {
        "static_windows": float(static_count),
        "active_windows": float(active_count),
        "static_fraction": (
            float(static_count / total) if total else 0.0
        ),
    }
    return tensor, summary


def build_dataset_sample_weights(
    info: DatasetInfo,
    index: Sequence[Tuple[int, int]],
    filename: Optional[str],
) -> Optional[torch.Tensor]:
    if not filename:
        return None
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = info.root / path
    values = np.load(path, mmap_mode="r")
    if values.ndim != 1 or len(values) != info.num_frames:
        raise ValueError(
            f"Sample weights {path} must have shape ({info.num_frames},), "
            f"got {values.shape}"
        )
    selected = np.asarray(
        [float(values[frame]) for frame, _ in index],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0):
        raise ValueError(f"Sample weights {path} must be finite and positive")
    return torch.as_tensor(selected, dtype=torch.double)


# ---------------------------------------------------------------------------
# Spatial multi-camera ACT
# ---------------------------------------------------------------------------

def spatial_sincos_position_embedding(
    height: int,
    width: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if hidden_dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by 4 for 2-D position encoding")

    quarter = hidden_dim // 4
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(quarter, 1)))

    x_phase = grid_x.reshape(-1, 1) * omega.reshape(1, -1)
    y_phase = grid_y.reshape(-1, 1) * omega.reshape(1, -1)
    position = torch.cat(
        [
            torch.sin(x_phase),
            torch.cos(x_phase),
            torch.sin(y_phase),
            torch.cos(y_phase),
        ],
        dim=1,
    )
    return position.to(dtype=dtype)


class SharedSpatialImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int, channels: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        input_channels = 3

        for output_channels in channels:
            output_channels = int(output_channels)
            groups = min(8, output_channels)
            while output_channels % groups != 0:
                groups -= 1
            layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, output_channels),
                    nn.GELU(),
                ]
            )
            input_channels = output_channels

        self.backbone = nn.Sequential(*layers)
        self.projection = nn.Conv2d(
            input_channels,
            hidden_dim,
            kernel_size=1,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        feature_map = self.projection(self.backbone(image))
        batch, hidden_dim, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.output_norm(tokens)
        position = spatial_sincos_position_embedding(
            height,
            width,
            hidden_dim,
            feature_map.device,
            feature_map.dtype,
        )
        tokens = tokens + position.unsqueeze(0)
        return tokens, height, width


class ResNet18SpatialImageEncoder(nn.Module):
    """ImageNet-initialized ResNet-18 exposed as spatial ACT tokens."""

    def __init__(self, hidden_dim: int, pretrained: bool):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.projection = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        feature_map = self.projection(self.backbone(image))
        _, hidden_dim, height, width = feature_map.shape
        tokens = self.output_norm(feature_map.flatten(2).transpose(1, 2))
        position = spatial_sincos_position_embedding(
            height,
            width,
            hidden_dim,
            feature_map.device,
            feature_map.dtype,
        )
        return tokens + position.unsqueeze(0), height, width


class ACTImageLatestPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        camera_names: Sequence[str],
        image_sizes: Mapping[str, Sequence[int]],
        chunk_size: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        nheads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        image_encoder_backbone: str = "conv",
        image_encoder_pretrained: bool = False,
        image_encoder_channels: Sequence[int] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.image_sizes = {
            str(name): (int(size[0]), int(size[1]))
            for name, size in image_sizes.items()
        }
        self.chunk_size = int(chunk_size)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.nheads = int(nheads)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.fusion_layers = int(fusion_layers)
        self.dropout = float(dropout)
        self.image_encoder_backbone = str(image_encoder_backbone)
        self.image_encoder_pretrained = bool(image_encoder_pretrained)
        self.image_encoder_channels = tuple(
            int(value) for value in image_encoder_channels
        )

        if not self.camera_names:
            raise ValueError("At least one camera is required")
        if self.hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4")

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if self.image_encoder_backbone == "conv":
            self.image_encoder = SharedSpatialImageEncoder(
                self.hidden_dim,
                self.image_encoder_channels,
            )
        elif self.image_encoder_backbone == "resnet18":
            self.image_encoder = ResNet18SpatialImageEncoder(
                self.hidden_dim,
                self.image_encoder_pretrained,
            )
        else:
            raise ValueError(
                f"Unknown image_encoder_backbone={self.image_encoder_backbone!r}"
            )

        self.state_embedding = nn.Parameter(
            torch.randn(1, self.hidden_dim) * 0.02
        )
        self.camera_embeddings = nn.Parameter(
            torch.randn(len(self.camera_names), self.hidden_dim) * 0.02
        )

        fusion_layer = nn.TransformerEncoderLayer(
            self.hidden_dim,
            self.nheads,
            self.hidden_dim * 4,
            self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            fusion_layer,
            self.fusion_layers,
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        self.action_encoder = nn.Linear(
            self.action_dim,
            self.hidden_dim,
        )
        self.posterior_pos = nn.Parameter(
            torch.randn(self.chunk_size + 1, self.hidden_dim) * 0.02
        )
        posterior_layer = nn.TransformerEncoderLayer(
            self.hidden_dim,
            self.nheads,
            self.hidden_dim * 4,
            self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.posterior = nn.TransformerEncoder(
            posterior_layer,
            self.num_encoder_layers,
        )
        self.mu_head = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar_head = nn.Linear(
            self.hidden_dim,
            self.latent_dim,
        )

        self.latent_proj = nn.Linear(
            self.latent_dim,
            self.hidden_dim,
        )
        self.action_queries = nn.Parameter(
            torch.randn(self.chunk_size, self.hidden_dim) * 0.02
        )
        decoder_layer = nn.TransformerEncoderLayer(
            self.hidden_dim,
            self.nheads,
            self.hidden_dim * 4,
            self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            self.num_decoder_layers,
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def model_config(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "camera_names": list(self.camera_names),
            "image_sizes": {
                name: list(size)
                for name, size in self.image_sizes.items()
            },
            "chunk_size": self.chunk_size,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "nheads": self.nheads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "fusion_layers": self.fusion_layers,
            "dropout": self.dropout,
            "image_encoder_backbone": self.image_encoder_backbone,
            # Checkpoints already contain every backbone tensor.  Never trigger
            # a network download merely to load/evaluate a trained checkpoint.
            "image_encoder_pretrained": False,
            "image_encoder_channels": list(self.image_encoder_channels),
        }

    def encode_observation(
        self,
        state: torch.Tensor,
        images: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        state_token = (
            self.state_encoder(state) + self.state_embedding[0]
        ).unsqueeze(1)
        tokens = [state_token]
        missing = [name for name in self.camera_names if name not in images]
        if missing:
            raise KeyError(f"Missing cameras {missing}; got {list(images.keys())}")

        # The backbone is shared across cameras/history frames.  One combined
        # call is exactly the same mapping and substantially reduces launch
        # overhead for batch-one closed-loop inference.
        batch = state.shape[0]
        stacked_images = torch.cat(
            [images[name] for name in self.camera_names], dim=0
        )
        camera_tokens, _, _ = self.image_encoder(stacked_images)
        tokens_per_camera = camera_tokens.shape[1]
        camera_tokens = camera_tokens.reshape(
            len(self.camera_names),
            batch,
            tokens_per_camera,
            self.hidden_dim,
        )
        for camera_index in range(len(self.camera_names)):
            tokens.append(
                camera_tokens[camera_index]
                + self.camera_embeddings[camera_index].view(1, 1, -1)
            )

        fused = self.fusion(torch.cat(tokens, dim=1))
        return self.fusion_norm(fused[:, 0])

    def encode_posterior(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat(
            [
                context.unsqueeze(1),
                self.action_encoder(actions),
            ],
            dim=1,
        )
        tokens = tokens + self.posterior_pos.unsqueeze(0)

        padding_mask = None
        if mask is not None:
            valid = torch.cat(
                [
                    torch.ones(
                        (mask.shape[0], 1),
                        dtype=torch.bool,
                        device=mask.device,
                    ),
                    mask > 0.5,
                ],
                dim=1,
            )
            padding_mask = ~valid

        hidden = self.posterior(
            tokens,
            src_key_padding_mask=padding_mask,
        )[:, 0]
        mu = self.mu_head(hidden)
        logvar = torch.clamp(
            self.logvar_head(hidden),
            -10.0,
            10.0,
        )
        return mu, logvar

    def decode(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        condition = context + self.latent_proj(latent)
        tokens = (
            self.action_queries.unsqueeze(0)
            + condition.unsqueeze(1)
        )
        return self.action_head(self.decoder(tokens))

    def forward(
        self,
        state: torch.Tensor,
        images: Mapping[str, torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        sample_latent: bool = True,
    ):
        context = self.encode_observation(state, images)
        zero_latent = torch.zeros(
            (state.shape[0], self.latent_dim),
            dtype=context.dtype,
            device=context.device,
        )
        prior_prediction = self.decode(context, zero_latent)

        if actions is None:
            return prior_prediction, prior_prediction, None, None

        mu, logvar = self.encode_posterior(
            context,
            actions,
            mask,
        )
        if sample_latent:
            latent = (
                mu
                + torch.randn_like(mu)
                * torch.exp(0.5 * logvar)
            )
        else:
            latent = mu
        posterior_prediction = self.decode(context, latent)
        return posterior_prediction, prior_prediction, mu, logvar

    @torch.no_grad()
    def act_chunk(
        self,
        state: torch.Tensor,
        images: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.forward(state, images, actions=None)[0]


def kl_div(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (
        -0.5
        * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
    ).mean()


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    action_groups: Sequence[Sequence[int]] = (),
    group_weights: Sequence[float] = (),
) -> torch.Tensor:
    if action_groups:
        weights = (
            tuple(float(value) for value in group_weights)
            if group_weights
            else tuple(1.0 for _ in action_groups)
        )
        losses: List[torch.Tensor] = []
        for indices in action_groups:
            index = torch.as_tensor(
                tuple(int(value) for value in indices),
                dtype=torch.long,
                device=prediction.device,
            )
            per_step = F.l1_loss(
                prediction.index_select(-1, index),
                target.index_select(-1, index),
                reduction="none",
            ).mean(-1)
            losses.append(
                (per_step * mask).sum()
                / mask.sum().clamp_min(1.0)
            )
        weight_tensor = torch.as_tensor(
            weights,
            dtype=prediction.dtype,
            device=prediction.device,
        )
        return (
            torch.stack(losses) * weight_tensor
        ).sum() / weight_tensor.sum().clamp_min(1e-8)

    per_step = F.l1_loss(
        prediction,
        target,
        reduction="none",
    ).mean(-1)
    return (
        (per_step * mask).sum()
        / mask.sum().clamp_min(1.0)
    )


def resolve_action_loss_groups(
    specs: Sequence[str],
    action_dim: int,
) -> Tuple[Tuple[int, ...], ...]:
    groups: List[Tuple[int, ...]] = []
    for spec in specs:
        indices = parse_state_slice(str(spec), action_dim)
        if not indices:
            raise ValueError(f"Empty action loss group: {spec!r}")
        groups.append(tuple(indices))
    return tuple(groups)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_ckpt(
    path: Path,
    model: ACTImageLatestPolicy,
    optimizer,
    stats: Stats,
    args,
    step: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_rng = np.random.get_state()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),
            "stats": stats.to_dict(),
            "args": vars(args),
            "step": int(step),
            "extra": extra or {},
            "model_config": model.model_config(),
            "policy_state_spec": {
                "mode": args.policy_state_mode,
                "online_state_paths": list(args.online_state_paths),
                "state_adapter": args.state_adapter,
                "env_state_keys": list(args.env_state_keys),
                "offline_state_slice": args.offline_state_slice,
                "expected_raw_state_dim": args.expected_raw_state_dim,
                "expected_policy_state_dim": args.expected_policy_state_dim,
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": {
                    "bit_generator": numpy_rng[0],
                    # This environment's torch serializer does not support
                    # uint32 storages. MT19937 values fit losslessly in int64.
                    "state": torch.from_numpy(
                        numpy_rng[1].astype(np.int64, copy=True)
                    ),
                    "position": int(numpy_rng[2]),
                    "has_gauss": int(numpy_rng[3]),
                    "cached_gaussian": float(numpy_rng[4]),
                },
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        },
        path,
    )


def load_ckpt(
    path: str,
    device: torch.device,
):
    checkpoint = torch.load(path, map_location=device)
    model = ACTImageLatestPolicy(
        **checkpoint["model_config"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    stats = Stats.from_dict(checkpoint["stats"])
    return model, stats, checkpoint


def log_scalar(
    writer: SummaryWriter,
    args,
    key: str,
    value: float,
    step: int,
) -> None:
    writer.add_scalar(key, value, step)
    if args.track and swanlab is not None:
        swanlab.log({key: value}, step=step)


# ---------------------------------------------------------------------------
# Online observation conversion
# ---------------------------------------------------------------------------

def flatten_state_dict(data: Mapping[str, Any]) -> np.ndarray:
    from mani_skill.utils import common

    try:
        value = common.flatten_state_dict(
            data,
            use_torch=True,
        )
    except TypeError:
        value = common.flatten_state_dict(data)
    array = to_np(value).astype(np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return array.reshape(-1)


def flatten_observation_value(value: Any) -> np.ndarray:
    array = to_np(value).astype(np.float32)
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    return array.reshape(-1)


def get_by_path(root: Any, path: str) -> Any:
    current = root
    for part in str(path).split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(
                    f"Path {path!r} is missing component {part!r}; "
                    f"available={list(current.keys())}"
                )
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            raise KeyError(
                f"Cannot resolve component {part!r} in path {path!r} "
                f"from object type {type(current).__name__}"
            )
    return current


_STATE_ADAPTER_CACHE: Dict[str, Callable[..., Any]] = {}


def load_state_adapter(spec: str) -> Callable[..., Any]:
    if spec in _STATE_ADAPTER_CACHE:
        return _STATE_ADAPTER_CACHE[spec]
    if ":" not in spec:
        raise ValueError(
            "state_adapter must use MODULE:FUNCTION, "
            f"got {spec!r}"
        )
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"State adapter {spec!r} is not callable")
    _STATE_ADAPTER_CACHE[spec] = function
    return function


def call_state_adapter(
    function: Callable[..., Any],
    *,
    obs: Any,
    env: Optional[gym.Env],
    args: Any,
) -> np.ndarray:
    signature = inspect.signature(function)
    parameters = signature.parameters
    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    kwargs: Dict[str, Any] = {}
    if has_var_kwargs or "obs" in parameters:
        kwargs["obs"] = obs
    elif "observation" in parameters:
        kwargs["observation"] = obs

    if has_var_kwargs or "env" in parameters:
        kwargs["env"] = env
    if has_var_kwargs or "args" in parameters:
        kwargs["args"] = args

    try:
        signature.bind(**kwargs)
    except TypeError:
        positional = [
            parameter
            for parameter in parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional) >= 3:
            value = function(obs, env, args)
        elif len(positional) == 2:
            value = function(obs, env)
        elif len(positional) == 1:
            value = function(obs)
        else:
            value = function()
    else:
        value = function(**kwargs)

    return flatten_observation_value(value)


def concat_paths(
    obs: Mapping[str, Any],
    paths: Sequence[str],
) -> np.ndarray:
    if not paths:
        raise ValueError("No online state paths were provided")
    parts = [
        flatten_observation_value(get_by_path(obs, path))
        for path in paths
    ]
    return np.concatenate(parts, axis=0).astype(
        np.float32,
        copy=False,
    )


def common_state_candidates(
    obs: Mapping[str, Any],
    env_state_keys: Sequence[str],
) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []

    def add(name: str, builder: Callable[[], np.ndarray]) -> None:
        try:
            value = np.asarray(builder(), dtype=np.float32).reshape(-1)
        except (KeyError, TypeError, ValueError, AttributeError):
            return
        if value.size == 0:
            return
        if any(
            existing.size == value.size
            and np.array_equal(existing, value)
            for _, existing in candidates
        ):
            return
        candidates.append((name, value))

    add(
        "agent.qpos+agent.qvel+extra.tcp_pose",
        lambda: concat_paths(
            obs,
            ("agent.qpos", "agent.qvel", "extra.tcp_pose"),
        ),
    )
    add(
        "agent.qpos+agent.qvel",
        lambda: concat_paths(
            obs,
            ("agent.qpos", "agent.qvel"),
        ),
    )
    add(
        "agent.qpos",
        lambda: concat_paths(obs, ("agent.qpos",)),
    )
    add(
        "agent",
        lambda: flatten_state_dict({"agent": obs["agent"]}),
    )

    keys = tuple(env_state_keys) if env_state_keys else ("agent", "extra")
    add(
        "agent_extra",
        lambda: flatten_state_dict(
            {key: obs[key] for key in keys}
        ),
    )
    return candidates


def obs_to_state(
    obs: Any,
    args: Any,
    expected_dim: Optional[int] = None,
    env: Optional[gym.Env] = None,
) -> np.ndarray:
    """Construct the online policy state without task-specific hard-coding."""
    if not isinstance(obs, Mapping):
        raise TypeError(
            "Visual ACT expects a dictionary observation; "
            f"got {type(obs).__name__}"
        )

    if args.state_adapter:
        state = call_state_adapter(
            load_state_adapter(args.state_adapter),
            obs=obs,
            env=env,
            args=args,
        )
        source = f"adapter:{args.state_adapter}"
    elif args.policy_state_mode == "paths":
        state = concat_paths(obs, args.online_state_paths)
        source = "paths:" + ",".join(args.online_state_paths)
    elif args.policy_state_mode == "proprioception":
        state = concat_paths(
            obs,
            ("agent.qpos", "agent.qvel", "extra.tcp_pose"),
        )
        source = "proprioception"
    elif args.policy_state_mode == "agent":
        if "agent" not in obs:
            raise KeyError(
                f"agent is missing; available={list(obs.keys())}"
            )
        state = flatten_state_dict({"agent": obs["agent"]})
        source = "agent"
    elif args.policy_state_mode == "agent_extra":
        keys = (
            tuple(args.env_state_keys)
            if args.env_state_keys
            else ("agent", "extra")
        )
        missing = [key for key in keys if key not in obs]
        if missing:
            raise KeyError(
                f"env_state_keys missing from observation: {missing}; "
                f"available={list(obs.keys())}"
            )
        state = flatten_state_dict(
            {key: obs[key] for key in keys}
        )
        source = "agent_extra:" + ",".join(keys)
    elif args.policy_state_mode == "auto":
        if args.online_state_paths:
            explicit = concat_paths(obs, args.online_state_paths)
            if expected_dim is None or explicit.size == int(expected_dim):
                return explicit.astype(np.float32, copy=False)

        candidates = common_state_candidates(
            obs,
            args.env_state_keys,
        )
        if expected_dim is not None:
            matching = [
                (name, value)
                for name, value in candidates
                if value.size == int(expected_dim)
            ]
            if matching:
                source, state = matching[0]
            else:
                dimensions = {
                    name: int(value.size)
                    for name, value in candidates
                }
                raise RuntimeError(
                    "Auto policy-state construction found no layout matching "
                    f"model state_dim={expected_dim}. Candidates={dimensions}. "
                    "Use --online-state-paths, --policy-state-mode, or "
                    "--state-adapter."
                )
        elif candidates:
            source, state = candidates[0]
        else:
            raise RuntimeError(
                "Auto policy-state construction found no usable state fields"
            )
    else:
        raise ValueError(
            f"Unknown policy_state_mode={args.policy_state_mode!r}"
        )

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if expected_dim is not None and state.size != int(expected_dim):
        raise RuntimeError(
            "Online state dimension does not match dataset/model:\n"
            f"  source={source}\n"
            f"  online={state.size}\n"
            f"  model={expected_dim}\n"
            "Use --online-state-paths, --policy-state-mode, or "
            "--state-adapter to match the offline dataset semantics."
        )
    return state


def available_rgb_sensors(obs: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    sensor_data = obs.get("sensor_data")
    if isinstance(sensor_data, Mapping):
        names.extend(str(name) for name in sensor_data.keys())

    rgb = obs.get("rgb")
    if isinstance(rgb, Mapping):
        names.extend(str(name) for name in rgb.keys())

    for name, payload in obs.items():
        if name in {"sensor_data", "rgb", "agent", "extra"}:
            continue
        if isinstance(payload, Mapping) and (
            "rgb" in payload or "Color" in payload
        ):
            names.append(str(name))
    return sorted(set(names))


def normalize_camera_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    for suffix in ("camera", "rgb", "sensor", "image"):
        normalized = normalized.replace(suffix, "")
    return normalized


CAMERA_ALIAS_GROUPS: Tuple[Tuple[str, ...], ...] = (
    (
        "base",
        "base_camera",
        "base_0_rgb",
        "head",
        "head_camera",
        "external",
        "external_camera",
    ),
    (
        "wrist",
        "wrist_camera",
        "hand",
        "hand_camera",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    ),
)


def resolve_sensor_name(
    obs: Mapping[str, Any],
    dataset_camera_name: str,
    explicit_name: Optional[str],
    auto_match: bool,
) -> str:
    available = available_rgb_sensors(obs)
    requested = explicit_name or dataset_camera_name

    if requested in available:
        return requested

    lower_matches = [
        name for name in available
        if name.lower() == requested.lower()
    ]
    if len(lower_matches) == 1:
        return lower_matches[0]

    if not auto_match:
        raise KeyError(
            f"Sensor {requested!r} not found; available={available}"
        )

    requested_normalized = normalize_camera_name(requested)
    normalized_matches = [
        name
        for name in available
        if normalize_camera_name(name) == requested_normalized
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    for group in CAMERA_ALIAS_GROUPS:
        group_normalized = {
            normalize_camera_name(value)
            for value in group
        }
        if requested_normalized not in group_normalized:
            continue
        alias_matches = [
            name
            for name in available
            if normalize_camera_name(name) in group_normalized
        ]
        if len(alias_matches) == 1:
            return alias_matches[0]

    raise KeyError(
        f"Could not map dataset camera {dataset_camera_name!r} "
        f"to an environment sensor. Requested={requested!r}; "
        f"available={available}. Use "
        "--camera-map DATASET_CAMERA=ENV_SENSOR."
    )


def extract_sensor_rgb(
    obs: Mapping[str, Any],
    sensor_name: str,
) -> np.ndarray:
    payload = None

    if (
        "sensor_data" in obs
        and isinstance(obs["sensor_data"], Mapping)
        and sensor_name in obs["sensor_data"]
    ):
        payload = obs["sensor_data"][sensor_name]
    elif (
        "rgb" in obs
        and isinstance(obs["rgb"], Mapping)
    ):
        payload = obs["rgb"].get(sensor_name)
    elif sensor_name in obs:
        payload = obs[sensor_name]

    if payload is None:
        raise KeyError(
            f"Sensor {sensor_name!r} not found; "
            f"available={available_rgb_sensors(obs)}"
        )

    if isinstance(payload, Mapping):
        if "rgb" in payload:
            payload = payload["rgb"]
        elif "Color" in payload:
            payload = payload["Color"]
        else:
            raise KeyError(
                f"Sensor {sensor_name!r} has no rgb/Color field; "
                f"keys={list(payload.keys())}"
            )

    image = to_np(payload)
    if image.ndim == 4:
        image = image[0]
    if (
        image.ndim == 3
        and image.shape[0] in (3, 4)
        and image.shape[-1] not in (3, 4)
    ):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(
            f"Invalid RGB shape for {sensor_name}: {image.shape}"
        )

    image = image[..., :3]
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size and float(image.max()) <= 1.5:
            image *= 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def resize_rgb(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    mode: str,
    letterbox_value: int,
) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    pil = Image.fromarray(image, mode="RGB")
    resample = (
        Image.Resampling.BILINEAR
        if hasattr(Image, "Resampling")
        else Image.BILINEAR
    )

    if mode == "stretch":
        output = pil.resize(
            (target_width, target_height),
            resample=resample,
        )
    elif mode == "letterbox":
        scale = min(
            target_width / source_width,
            target_height / source_height,
        )
        new_width = max(
            1,
            int(round(source_width * scale)),
        )
        new_height = max(
            1,
            int(round(source_height * scale)),
        )
        resized = pil.resize(
            (new_width, new_height),
            resample=resample,
        )
        output = Image.new(
            "RGB",
            (target_width, target_height),
            color=(
                letterbox_value,
                letterbox_value,
                letterbox_value,
            ),
        )
        output.paste(
            resized,
            (
                (target_width - new_width) // 2,
                (target_height - new_height) // 2,
            ),
        )
    elif mode == "center_crop":
        scale = max(
            target_width / source_width,
            target_height / source_height,
        )
        new_width = max(
            target_width,
            int(round(source_width * scale)),
        )
        new_height = max(
            target_height,
            int(round(source_height * scale)),
        )
        resized = pil.resize(
            (new_width, new_height),
            resample=resample,
        )
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        output = resized.crop(
            (
                left,
                top,
                left + target_width,
                top + target_height,
            )
        )
    else:
        raise ValueError(
            f"Unknown resize mode: {mode}"
        )

    return np.ascontiguousarray(
        np.asarray(output, dtype=np.uint8)
    )


def obs_to_tensors(
    obs: Any,
    model: ACTImageLatestPolicy,
    stats: Stats,
    args,
    device: torch.device,
    env: Optional[gym.Env] = None,
    obs_history: Optional[Sequence[Any]] = None,
):
    state = obs_to_state(
        obs,
        args,
        expected_dim=model.state_dim,
        env=env,
    )

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    obs_mean = torch.as_tensor(
        stats.obs_mean,
        dtype=torch.float32,
        device=device,
    )
    obs_std = torch.as_tensor(
        stats.obs_std,
        dtype=torch.float32,
        device=device,
    )
    state_tensor = (
        state_tensor - obs_mean
    ) / obs_std

    if not isinstance(obs, Mapping):
        raise TypeError(
            "Image ACT requires a dictionary observation"
        )

    camera_mapping = parse_camera_map(
        args.camera_map
    )
    history_offsets = parse_int_map(
        args.camera_history_offsets,
        "camera_history_offsets",
    )
    mean = torch.tensor(
        args.image_mean,
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        args.image_std,
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)

    images: Dict[str, torch.Tensor] = {}
    for camera_name in model.camera_names:
        source_obs = obs
        history_offset = history_offsets.get(camera_name, 0)
        if obs_history:
            source_obs = obs_history[
                min(history_offset, len(obs_history) - 1)
            ]
        sensor_name = resolve_sensor_name(
            source_obs,
            dataset_camera_name=camera_name,
            explicit_name=camera_mapping.get(camera_name),
            auto_match=args.camera_auto_match,
        )
        image = extract_sensor_rgb(source_obs, sensor_name)
        height, width = model.image_sizes[
            camera_name
        ]
        image = resize_rgb(
            image,
            height,
            width,
            args.resize_mode,
            args.letterbox_value,
        )
        tensor = (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.float32,
            )
            / 255.0
        )
        images[camera_name] = (
            tensor - mean
        ) / std

    return state_tensor, images


def check_environment(
    model: ACTImageLatestPolicy,
    stats: Stats,
    args,
    device: torch.device,
) -> None:
    env, env_kwargs = make_env(
        args,
        capture_video=False,
    )
    try:
        obs, _ = env.reset(
            seed=(
                args.eval_seed
                if args.eval_seed is not None
                else args.seed
            )
        )
        state, images = obs_to_tensors(
            obs,
            model,
            stats,
            args,
            device,
            env=env,
        )
        action_dim = int(
            np.prod(env.action_space.shape)
        )
        if action_dim != model.action_dim:
            raise RuntimeError(
                f"Environment action_dim={action_dim}, "
                f"dataset/model action_dim={model.action_dim}"
            )

        print(
            f"[ENV CHECK] state={tuple(state.shape)} "
            f"action_dim={action_dim} mode={args.policy_state_mode}"
        )
        print(
            "[ENV CHECK] policy_state="
            f"mode={args.policy_state_mode} "
            f"paths={list(args.online_state_paths)} "
            f"adapter={args.state_adapter}"
        )
        for name, image in images.items():
            print(
                f"[ENV CHECK] camera={name} "
                f"shape={tuple(image.shape)}"
            )
        print(
            f"[ENV CHECK] env_cfg="
            f"{json.dumps(env_kwargs, default=str)}"
        )
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_recorded_initial_raw_states(
    dataset_path: str,
    episode_ids: Sequence[int],
    frame_indices: Sequence[int] = (),
) -> Dict[int, Any]:
    """Load selected replay raw states for RoboDex episodes."""
    source = Path(dataset_path).expanduser().resolve()
    parquet_paths = sorted((source / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {source / 'data'}")

    columns = ["episode_index", "frame_index", "episode.source_hash"]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in parquet_paths],
        ignore_index=True,
    )
    frame = frame.sort_values(["episode_index", "frame_index"])
    first_rows = frame.groupby("episode_index", sort=True).first()

    from data_gen.recordings.episode_store import load_pending_episode

    selected_frames = (
        tuple(int(value) for value in frame_indices)
        if frame_indices
        else tuple(0 for _ in episode_ids)
    )
    if len(selected_frames) != len(episode_ids):
        raise ValueError("frame_indices must match episode_ids length")

    result: Dict[int, Any] = {}
    for episode_id, frame_index in zip(episode_ids, selected_frames):
        episode_id = int(episode_id)
        if episode_id not in first_rows.index:
            raise KeyError(
                f"Episode {episode_id} is absent from recorded dataset {source}"
            )
        source_hash = int(
            first_rows.loc[episode_id, "episode.source_hash"]
        )
        sidecar = source / "replay" / f"source_{source_hash:015x}"
        episode = load_pending_episode(sidecar)
        raw_states = episode.get("raw_state_sequence", ())
        if not raw_states:
            raise RuntimeError(
                f"Episode {episode_id} has no raw_state_sequence in {sidecar}"
            )
        if frame_index >= len(raw_states):
            raise IndexError(
                f"Episode {episode_id} raw-state frame {frame_index} is out of "
                f"range for sequence length {len(raw_states)}"
            )
        result[episode_id] = raw_states[frame_index]
    return result

@torch.no_grad()
def evaluate(
    model: ACTImageLatestPolicy,
    stats: Stats,
    args,
    device: torch.device,
    run_dir: Path,
    step: int,
) -> Dict[str, float]:
    env, _ = make_env(
        args,
        capture_video=args.capture_video,
    )
    model.eval()
    residual_spec = JointPosResidualSpec(
        enabled=args.joint_pos_residual,
        action_start=args.joint_pos_residual_action_start,
        qpos_start=args.joint_pos_residual_qpos_start,
        dim=args.joint_pos_residual_dim,
        scale=args.joint_pos_residual_scale,
    )

    action_mean = torch.as_tensor(
        stats.action_mean,
        dtype=torch.float32,
        device=device,
    )
    action_std = torch.as_tensor(
        stats.action_std,
        dtype=torch.float32,
        device=device,
    )

    action_chunk_size = (
        model.chunk_size
        if args.action_chunk_size == 0
        else int(args.action_chunk_size)
    )
    if action_chunk_size > model.chunk_size:
        raise ValueError(
            f"action_chunk_size={action_chunk_size} exceeds the model's "
            f"trained chunk_size={model.chunk_size}"
        )

    def predict_chunk(
        current_obs: Any,
        current_history: Sequence[Any],
    ) -> np.ndarray:
        state, images = obs_to_tensors(
            current_obs,
            model,
            stats,
            args,
            device,
            env=env,
            obs_history=current_history,
        )
        chunk = model.act_chunk(
            state,
            images,
        )[0]
        chunk = (
            chunk * action_std
            + action_mean
        )
        return (
            chunk.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    video_dir = (
        Path(args.video_dir)
        if args.video_dir
        else run_dir / "videos"
    )
    if args.capture_video:
        video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    base_seed = (
        args.eval_seed
        if args.eval_seed is not None
        else args.seed + 10000
    )
    rows: List[Dict[str, Any]] = []
    recorded_episode_ids: Tuple[int, ...] = ()
    recorded_initial_states: Dict[int, Any] = {}
    if args.eval_initial_state_dataset:
        recorded_episode_ids = (
            tuple(int(value) for value in args.eval_initial_episode_ids)
            if args.eval_initial_episode_ids
            else tuple(range(args.num_eval_episodes))
        )
        recorded_initial_states = load_recorded_initial_raw_states(
            args.eval_initial_state_dataset,
            recorded_episode_ids,
            args.eval_initial_frame_indices,
        )
        print(
            "[EVAL] using recorded initial raw states "
            f"dataset={Path(args.eval_initial_state_dataset).expanduser().resolve()} "
            f"episodes={list(recorded_episode_ids)} "
            f"frames={list(args.eval_initial_frame_indices) if args.eval_initial_frame_indices else [0] * len(recorded_episode_ids)}"
        )

    try:
        for episode in range(
            args.num_eval_episodes
        ):
            obs, info = env.reset(
                seed=base_seed + episode
            )
            recorded_episode_id: Optional[int] = None
            if recorded_initial_states:
                recorded_episode_id = recorded_episode_ids[episode]
                from data_gen.recordings.replay_recording import _set_raw_state

                _set_raw_state(
                    env,
                    recorded_initial_states[recorded_episode_id],
                )
                # A recording may begin after teleoperation setup has already
                # consumed many environment steps.  Preserve its physical/task
                # state but start a fresh evaluation horizon instead of
                # immediately truncating at the recorded clock value.
                elapsed_steps = getattr(env.unwrapped, "_elapsed_steps", None)
                if torch.is_tensor(elapsed_steps):
                    elapsed_steps.zero_()
                elif elapsed_steps is not None:
                    setattr(env.unwrapped, "_elapsed_steps", 0)
                obs = env.unwrapped.get_obs()
                info = env.unwrapped.evaluate()
            observation_history: List[Any] = [obs]
            maximum_history_offset = max(
                parse_int_map(
                    args.camera_history_offsets,
                    "camera_history_offsets",
                ).values(),
                default=0,
            )
            action_buffer: List[np.ndarray] = []
            chunk_history: List[
                Tuple[int, np.ndarray]
            ] = []
            frames: List[np.ndarray] = []

            episode_return = 0.0
            success_once = False
            success_end = False
            first_success_step = -1
            length = 0
            stage_once = {
                key: False
                for key in args.stage_keys
            }
            metric_final: Dict[str, float] = {}
            metric_min: Dict[str, float] = {}
            metric_max: Dict[str, float] = {}

            for timestep in range(
                args.num_eval_steps
            ):
                if (
                    args.capture_video
                    and episode
                    < args.max_eval_videos
                ):
                    frame = safe_render(env)
                    if frame is not None:
                        frames.append(frame)

                if args.inference_mode == "chunk":
                    if not action_buffer:
                        action_buffer = [
                            value.copy()
                            for value in predict_chunk(
                                obs,
                                observation_history,
                            )[
                                :action_chunk_size
                            ]
                        ]
                    action = action_buffer.pop(0)

                elif args.inference_mode == "replan":
                    action = predict_chunk(
                        obs,
                        observation_history,
                    )[0]

                elif args.inference_mode == "ensemble":
                    chunk = predict_chunk(
                        obs,
                        observation_history,
                    )
                    chunk_history.append(
                        (timestep, chunk)
                    )
                    chunk_history = [
                        (start, old_chunk)
                        for start, old_chunk
                        in chunk_history
                        if start
                        <= timestep
                        < start + len(old_chunk)
                    ]

                    candidates = np.stack(
                        [
                            old_chunk[
                                timestep - start
                            ]
                            for start, old_chunk
                            in chunk_history
                        ],
                        axis=0,
                    )
                    ages = np.asarray(
                        [
                            timestep - start
                            for start, _
                            in chunk_history
                        ],
                        dtype=np.float32,
                    )
                    weights = np.exp(
                        -float(
                            args.ensemble_decay
                        )
                        * ages
                    )
                    weights /= max(
                        float(weights.sum()),
                        1e-8,
                    )
                    action = np.sum(
                        candidates
                        * weights[:, None],
                        axis=0,
                    ).astype(np.float32)

                    for dimension in (
                        args.ensemble_latest_action_dims
                    ):
                        dimension = int(
                            dimension
                        )
                        if dimension < 0:
                            dimension += (
                                action.shape[0]
                            )
                        if (
                            0
                            <= dimension
                            < action.shape[0]
                        ):
                            action[
                                dimension
                            ] = candidates[
                                -1,
                                dimension,
                            ]
                else:
                    raise ValueError(
                        "Unknown inference_mode="
                        f"{args.inference_mode}"
                    )

                if residual_spec.enabled:
                    action = policy_to_env_action(
                        action,
                        robot_qpos_from_env(env),
                        residual_spec,
                    )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    format_action(env, action)
                )
                observation_history.insert(0, obs)
                del observation_history[maximum_history_offset + 1 :]
                episode_return += scalar_float(
                    reward
                )
                length = timestep + 1

                success_now = any(
                    scalar_bool(info.get(key, False))
                    for key in args.success_keys
                )
                if (
                    success_now
                    and not success_once
                ):
                    success_once = True
                    first_success_step = (
                        timestep
                    )
                success_end = success_now

                for key in args.stage_keys:
                    if (
                        key in info
                        and scalar_bool(info[key])
                    ):
                        stage_once[key] = True

                for key in args.metric_keys:
                    if key not in info:
                        continue
                    value = scalar_float(
                        info[key]
                    )
                    if not math.isfinite(value):
                        continue
                    metric_final[key] = value
                    metric_min[key] = min(
                        metric_min.get(
                            key,
                            value,
                        ),
                        value,
                    )
                    metric_max[key] = max(
                        metric_max.get(
                            key,
                            value,
                        ),
                        value,
                    )

                if success_now and args.terminate_on_success:
                    break
                if (
                    scalar_bool(terminated)
                    or scalar_bool(truncated)
                ):
                    break

            if (
                args.capture_video
                and episode
                < args.max_eval_videos
                and frames
                and imageio is not None
            ):
                imageio.mimsave(
                    video_dir
                    / (
                        f"{args.inference_mode}"
                        f"_step_{step}"
                        f"_ep_{episode}.mp4"
                    ),
                    frames,
                    fps=args.video_fps,
                )

            row: Dict[str, Any] = {
                "episode_id": episode,
                "seed": base_seed + episode,
                "return_": episode_return,
                "length": length,
                "success_once": success_once,
                "success_end": success_end,
                "first_success_step": (
                    first_success_step
                ),
                "failure_type": (
                    "success"
                    if success_once
                    else "failure"
                ),
            }
            if recorded_episode_id is not None:
                row["recorded_initial_episode_id"] = recorded_episode_id
            row.update(
                {
                    f"stage_{key}_once": value
                    for key, value
                    in stage_once.items()
                }
            )
            row.update(
                {
                    f"final_{key}": value
                    for key, value
                    in metric_final.items()
                }
            )
            row.update(
                {
                    f"min_{key}": value
                    for key, value
                    in metric_min.items()
                }
            )
            row.update(
                {
                    f"max_{key}": value
                    for key, value
                    in metric_max.items()
                }
            )
            rows.append(row)

            print(
                f"[EVAL] step={step} "
                f"ep={episode+1}/"
                f"{args.num_eval_episodes} "
                f"return={episode_return:.3f} "
                f"len={length} "
                f"success_once={success_once}"
            )
    finally:
        env.close()

    log_dir = run_dir / (
        "eval_logs_eval_only"
        if args.evaluate
        else "eval_logs"
    )
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    eval_suffix = ""
    if args.evaluate:
        if args.inference_mode == "chunk":
            eval_suffix += (
                f"_action_chunk_{action_chunk_size}"
            )
        elif args.inference_mode == "ensemble":
            decay = format(
                float(args.ensemble_decay),
                ".6g",
            ).replace("-", "m").replace(".", "p")
            eval_suffix += f"_decay_{decay}"
        eval_suffix += (
            f"_seed_{base_seed}"
            f"_n_{args.num_eval_episodes}"
        )

    csv_path = (
        log_dir
        / (
            f"eval_{args.inference_mode}"
            f"{eval_suffix}"
            f"_step_{step}.csv"
        )
    )
    if rows:
        keys = sorted(
            {
                key
                for row in rows
                for key in row
            }
        )
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=keys,
            )
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"[EVAL] results saved to "
            f"{csv_path}"
        )

    metrics: Dict[
        str,
        List[float],
    ] = defaultdict(list)
    for row in rows:
        metrics["return_mean"].append(
            float(row["return_"])
        )
        metrics["episode_len_mean"].append(
            float(row["length"])
        )
        metrics["success_once_mean"].append(
            float(row["success_once"])
        )
        metrics[
            "success_at_end_mean"
        ].append(
            float(row["success_end"])
        )

    for key in args.stage_keys:
        metrics[
            f"stage/{key}_once_rate"
        ].extend(
            [
                float(
                    row.get(
                        f"stage_{key}_once",
                        False,
                    )
                )
                for row in rows
            ]
        )

    return {
        key: float(np.mean(values))
        for key, values in metrics.items()
        if values
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def move_batch(
    batch,
    device: torch.device,
):
    state = batch["state"].to(
        device,
        non_blocking=True,
    )
    images = {
        name: image.to(
            device,
            non_blocking=True,
        )
        for name, image
        in batch["images"].items()
    }
    actions = batch["actions"].to(
        device,
        non_blocking=True,
    )
    mask = batch["mask"].to(
        device,
        non_blocking=True,
    )
    return state, images, actions, mask


def main() -> None:
    args = parse_args()
    run_name = build_run_name(args)
    set_random_seed(args)
    device = resolve_device(args)

    run_dir = (
        Path(args.output_dir)
        / run_name
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        run_dir / "resolved_args.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(args),
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    writer = SummaryWriter(
        str(run_dir)
    )

    if args.track:
        if swanlab is None:
            print(
                "[WARN] SwanLab unavailable; "
                "disabling tracking"
            )
        else:
            swanlab.init(
                project=(
                    args.swanlab_project_name
                ),
                entity=args.swanlab_entity,
                config=vars(args),
                name=run_name,
                save_code=True,
                group=args.swanlab_group,
                tags=[
                    "act",
                    "image",
                    "spatial_tokens",
                    "configurable_policy_state",
                    "offline_imitation",
                ],
            )

    if args.evaluate:
        model, stats, checkpoint = (
            load_ckpt(
                args.checkpoint,
                device,
            )
        )
        print(
            f"[LOAD] checkpoint="
            f"{args.checkpoint} "
            f"step="
            f"{checkpoint.get('step', 0)}"
        )
        check_environment(
            model,
            stats,
            args,
            device,
        )
        metrics = evaluate(
            model,
            stats,
            args,
            device,
            Path(
                args.checkpoint
            ).resolve().parent,
            int(
                checkpoint.get(
                    "step",
                    0,
                )
            ),
        )
        for key, value in metrics.items():
            print(
                f"eval_{key}={value}"
            )
            log_scalar(
                writer,
                args,
                f"eval/{key}",
                value,
                int(
                    checkpoint.get(
                        "step",
                        0,
                    )
                ),
            )
        writer.close()
        return

    info = load_dataset_info(
        args.demo_data_path,
        args.camera_names,
        args.offline_state_slice,
        args.expected_raw_state_dim,
        args.expected_policy_state_dim,
    )
    residual_spec = JointPosResidualSpec(
        enabled=args.joint_pos_residual,
        action_start=args.joint_pos_residual_action_start,
        qpos_start=args.joint_pos_residual_qpos_start,
        dim=args.joint_pos_residual_dim,
        scale=args.joint_pos_residual_scale,
    )
    residual_states: Optional[np.ndarray] = None
    if residual_spec.enabled:
        residual_states = load_aligned_lerobot_states(
            args.joint_pos_residual_source_dataset,
            info.root,
            args.joint_pos_residual_source_obs_key,
        )
        residual_spec.validate(info.action_dim, residual_states.shape[1])
        if len(residual_states) != info.num_frames:
            raise RuntimeError(
                "Aligned source-state frame count does not match image dataset: "
                f"{len(residual_states)} != {info.num_frames}"
            )
        print(
            "[ACTION] pd_joint_pos residual parameterization: "
            f"action[{residual_spec.action_start}:"
            f"{residual_spec.action_start + residual_spec.dim}] "
            f"= (absolute_target - source_obs[{residual_spec.qpos_start}:"
            f"{residual_spec.qpos_start + residual_spec.dim}]) / "
            f"{residual_spec.scale}; policy state remains image-only"
        )
    action_loss_groups = resolve_action_loss_groups(
        args.action_loss_groups,
        info.action_dim,
    )
    action_loss_group_weights = (
        tuple(float(value) for value in args.action_loss_group_weights)
        if args.action_loss_group_weights
        else tuple(1.0 for _ in action_loss_groups)
    )
    if action_loss_groups:
        print(
            "[LOSS] action groups="
            f"{[list(group) for group in action_loss_groups]} "
            f"weights={list(action_loss_group_weights)}"
        )
    episodes = load_episodes(
        info,
        args.filter_success,
        args.include_episode_ids,
        args.exclude_episode_ids,
    )
    (
        train_episodes,
        validation_episodes,
    ) = split_episodes(
        episodes,
        args.validation_ratio,
        args.seed,
    )
    stats = compute_stats(
        info,
        train_episodes,
        state_std_floor=args.state_std_floor,
        action_std_floor=args.action_std_floor,
        residual_states=residual_states,
        residual_spec=residual_spec,
    )

    train_dataset = ACTImageDataset(
        info,
        train_episodes,
        stats,
        args.chunk_size,
        args.image_mean,
        args.image_std,
        training=True,
        random_shift_pad=(
            args.random_shift_pad
        ),
        action_padding_mode=args.action_padding_mode,
        min_chunk_valid_steps=args.min_chunk_valid_steps,
        residual_states=residual_states,
        residual_spec=residual_spec,
    )
    sample_weights, sampling_summary = build_static_sample_weights(
        info,
        train_dataset.index,
        chunk_size=args.chunk_size,
        action_dims=args.static_action_dims,
        action_threshold=args.static_action_threshold,
        static_weight=args.static_sample_weight,
    )
    dataset_sample_weights = build_dataset_sample_weights(
        info,
        train_dataset.index,
        args.sample_weight_file,
    )
    if dataset_sample_weights is not None:
        sample_weights = (
            dataset_sample_weights
            if sample_weights is None
            else sample_weights * dataset_sample_weights
        )
        print(
            "[DATA] phase/sample weights: "
            f"file={args.sample_weight_file} "
            f"min={float(dataset_sample_weights.min()):.3f} "
            f"mean={float(dataset_sample_weights.mean()):.3f} "
            f"max={float(dataset_sample_weights.max()):.3f}"
        )
    train_sampler = None
    if sample_weights is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        print(
            "[DATA] static-window sampling: "
            f"weight={args.static_sample_weight} "
            f"threshold={args.static_action_threshold} "
            f"dims={list(normalize_action_dims(args.static_action_dims, info.action_dim))} "
            f"static_fraction={sampling_summary['static_fraction']:.3f}"
        )

    # The iterator is created after CUDA/SAPIEN environment validation below.
    # Linux's default ``fork`` start method is unsafe once those runtimes have
    # initialized background threads and GPU state, and previously caused a
    # random DataLoader-worker SIGSEGV around step 6400.  Spawn clean worker
    # interpreters instead; ACTImageDataset.__getstate__ deliberately drops its
    # mmap handles so every spawned worker reopens them independently.
    worker_process_kwargs = (
        {"multiprocessing_context": "spawn"}
        if args.num_workers > 0
        else {}
    )
    if args.num_workers > 0:
        print(
            f"[DATA] workers={args.num_workers} "
            "multiprocessing_context=spawn"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_dataset) >= args.batch_size),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker,
        **worker_process_kwargs,
    )

    validation_loader = None
    if validation_episodes:
        validation_dataset = (
            ACTImageDataset(
                info,
                validation_episodes,
                stats,
                args.chunk_size,
                args.image_mean,
                args.image_std,
                training=False,
                random_shift_pad=0,
                action_padding_mode=args.action_padding_mode,
                min_chunk_valid_steps=args.min_chunk_valid_steps,
                residual_states=residual_states,
                residual_spec=residual_spec,
            )
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(
                device.type == "cuda"
            ),
            persistent_workers=(
                args.num_workers > 0
            ),
            worker_init_fn=seed_worker,
            **worker_process_kwargs,
        )

    camera_names = [
        camera.name
        for camera in info.cameras
    ]
    image_sizes = {
        camera.name: (
            camera.height,
            camera.width,
        )
        for camera in info.cameras
    }

    if args.checkpoint:
        (
            model,
            checkpoint_stats,
            checkpoint,
        ) = load_ckpt(
            args.checkpoint,
            device,
        )
        if (
            model.state_dim
            != info.state_dim
        ):
            raise RuntimeError(
                "Checkpoint state_dim "
                "does not match dataset"
            )
        if (
            model.action_dim
            != info.action_dim
        ):
            raise RuntimeError(
                "Checkpoint action_dim "
                "does not match dataset"
            )
        if tuple(model.camera_names) != tuple(camera_names):
            raise RuntimeError(
                "Checkpoint camera list does not match dataset"
            )
        if model.chunk_size != args.chunk_size:
            raise RuntimeError(
                f"Checkpoint chunk_size={model.chunk_size} does not match "
                f"--chunk-size={args.chunk_size}"
            )
        if model.image_sizes != image_sizes:
            raise RuntimeError(
                f"Checkpoint image sizes {model.image_sizes} do not match "
                f"dataset image sizes {image_sizes}"
            )
        stats = checkpoint_stats
        train_dataset.stats = stats
        if validation_episodes:
            validation_dataset.stats = (
                stats
            )
        start_step = int(
            checkpoint.get(
                "step",
                0,
            )
        )
    else:
        model = ACTImageLatestPolicy(
            state_dim=info.state_dim,
            action_dim=info.action_dim,
            camera_names=camera_names,
            image_sizes=image_sizes,
            chunk_size=args.chunk_size,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            nheads=args.nheads,
            num_encoder_layers=(
                args.num_encoder_layers
            ),
            num_decoder_layers=(
                args.num_decoder_layers
            ),
            fusion_layers=(
                args.fusion_layers
            ),
            dropout=args.dropout,
            image_encoder_backbone=args.image_encoder_backbone,
            image_encoder_pretrained=args.image_encoder_pretrained,
            image_encoder_channels=(
                args.image_encoder_channels
            ),
        ).to(device)
        checkpoint = None
        start_step = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if (
        checkpoint is not None
        and checkpoint.get(
            "optimizer_state_dict"
        ) is not None
    ):
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if args.check_env_before_training:
        check_environment(
            model,
            stats,
            args,
            device,
        )

    total_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )
    with torch.no_grad():
        dummy_state = torch.zeros(
            1,
            info.state_dim,
            device=device,
        )
        dummy_images = {
            camera.name: torch.zeros(
                1,
                3,
                camera.height,
                camera.width,
                device=device,
            )
            for camera in info.cameras
        }
        dummy_context = (
            model.encode_observation(
                dummy_state,
                dummy_images,
            )
        )
    print(
        f"[DATA] frames={info.num_frames} "
        f"raw_state_dim={info.raw_state_dim} "
        f"policy_state_dim={info.state_dim} "
        f"action_dim={info.action_dim} "
        f"cameras={camera_names}"
    )
    print(
        "[STATE] offline selection="
        f"{args.offline_state_slice} indices={list(info.state_indices)}"
    )
    print(
        "[STATE] online construction="
        f"mode={args.policy_state_mode} "
        f"paths={list(args.online_state_paths)} "
        f"adapter={args.state_adapter}"
    )
    print(
        f"[DATA] train_episodes="
        f"{[x.episode_id for x in train_episodes]}"
    )
    print(
        f"[DATA] train_anchors={len(train_dataset)} "
        f"min_chunk_valid_steps={args.min_chunk_valid_steps}"
    )
    if validation_episodes:
        print(
            f"[DATA] validation_episodes="
            f"{[x.episode_id for x in validation_episodes]}"
        )
    print(
        f"[MODEL] total_params="
        f"{total_params:,} "
        f"context_shape="
        f"{tuple(dummy_context.shape)}"
    )
    print(
        f"[RUN] run_dir="
        f"{run_dir}"
    )

    global_step = start_step
    best_success = -1.0
    best_return = -float("inf")
    loader_iterator = iter(
        train_loader
    )
    start_time = time.time()

    while global_step < args.steps:
        try:
            batch = next(
                loader_iterator
            )
        except StopIteration:
            loader_iterator = iter(
                train_loader
            )
            batch = next(
                loader_iterator
            )

        global_step += 1
        model.train()
        (
            state,
            images,
            actions,
            mask,
        ) = move_batch(
            batch,
            device,
        )

        (
            posterior_prediction,
            prior_prediction,
            mu,
            logvar,
        ) = model(
            state,
            images,
            actions,
            mask,
            sample_latent=True,
        )
        posterior_l1 = masked_l1(
            posterior_prediction,
            actions,
            mask,
            action_loss_groups,
            action_loss_group_weights,
        )
        prior_l1 = masked_l1(
            prior_prediction,
            actions,
            mask,
            action_loss_groups,
            action_loss_group_weights,
        )
        kl = kl_div(
            mu,
            logvar,
        )
        loss = (
            posterior_l1
            + args.prior_l1_weight
            * prior_l1
            + args.kl_weight
            * kl
        )

        optimizer.zero_grad(
            set_to_none=True
        )
        loss.backward()
        grad_norm = (
            nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
        )
        optimizer.step()

        if (
            global_step
            % args.log_freq
            == 0
        ):
            samples_per_second = int(
                (
                    global_step
                    - start_step
                )
                * args.batch_size
                / max(
                    time.time()
                    - start_time,
                    1e-6,
                )
            )
            prior_action_std = float(
                prior_prediction.detach()
                .std(
                    dim=(0, 1),
                    unbiased=False,
                )
                .mean()
                .item()
            )
            expert_action_std = float(
                actions.detach()
                .std(
                    dim=(0, 1),
                    unbiased=False,
                )
                .mean()
                .item()
            )
            values = {
                "train/loss": float(
                    loss.item()
                ),
                "train/posterior_l1": float(
                    posterior_l1.item()
                ),
                "train/prior_l1": float(
                    prior_l1.item()
                ),
                "train/kl_loss": float(
                    kl.item()
                ),
                "train/grad_norm": float(
                    grad_norm
                ),
                "train/prior_action_std": (
                    prior_action_std
                ),
                "train/expert_action_std": (
                    expert_action_std
                ),
                "charts/SPS": float(
                    samples_per_second
                ),
            }
            for key, value in values.items():
                log_scalar(
                    writer,
                    args,
                    key,
                    value,
                    global_step,
                )
            print(
                f"[TRAIN] step="
                f"{global_step}/"
                f"{args.steps} "
                f"loss="
                f"{loss.item():.5f} "
                f"post_l1="
                f"{posterior_l1.item():.5f} "
                f"prior_l1="
                f"{prior_l1.item():.5f} "
                f"kl="
                f"{kl.item():.5f} "
                f"pred_std="
                f"{prior_action_std:.4f} "
                f"expert_std="
                f"{expert_action_std:.4f} "
                f"grad="
                f"{float(grad_norm):.3f} "
                f"SPS="
                f"{samples_per_second}"
            )

        if (
            validation_loader is not None
            and global_step
            % args.eval_every
            == 0
        ):
            model.eval()
            validation_posterior: List[
                float
            ] = []
            validation_prior: List[
                float
            ] = []
            with torch.no_grad():
                for validation_batch in (
                    validation_loader
                ):
                    (
                        validation_state,
                        validation_images,
                        validation_actions,
                        validation_mask,
                    ) = move_batch(
                        validation_batch,
                        device,
                    )
                    (
                        validation_post,
                        validation_prior_pred,
                        _,
                        _,
                    ) = model(
                        validation_state,
                        validation_images,
                        validation_actions,
                        validation_mask,
                        sample_latent=False,
                    )
                    validation_posterior.append(
                        float(
                            masked_l1(
                                validation_post,
                                validation_actions,
                                validation_mask,
                                action_loss_groups,
                                action_loss_group_weights,
                            ).item()
                        )
                    )
                    validation_prior.append(
                        float(
                            masked_l1(
                                validation_prior_pred,
                                validation_actions,
                                validation_mask,
                                action_loss_groups,
                                action_loss_group_weights,
                            ).item()
                        )
                    )

            mean_post = float(
                np.mean(
                    validation_posterior
                )
            )
            mean_prior = float(
                np.mean(
                    validation_prior
                )
            )
            log_scalar(
                writer,
                args,
                "val/posterior_l1",
                mean_post,
                global_step,
            )
            log_scalar(
                writer,
                args,
                "val/prior_l1",
                mean_prior,
                global_step,
            )
            print(
                f"[VAL] step="
                f"{global_step} "
                f"post_l1="
                f"{mean_post:.5f} "
                f"prior_l1="
                f"{mean_prior:.5f}"
            )

        if (
            args.num_eval_episodes > 0
            and (
                global_step % args.eval_every == 0
                or global_step == args.steps
            )
        ):
            metrics = evaluate(
                model,
                stats,
                args,
                device,
                run_dir,
                global_step,
            )
            for key, value in metrics.items():
                print(
                    f"eval_{key}="
                    f"{value}"
                )
                log_scalar(
                    writer,
                    args,
                    f"eval/{key}",
                    value,
                    global_step,
                )

            success = metrics.get(
                "success_once_mean",
                0.0,
            )
            episode_return = metrics.get(
                "return_mean",
                -float("inf"),
            )
            if (
                args.save_model
                and success
                > best_success
            ):
                best_success = success
                save_ckpt(
                    run_dir
                    / "ckpt_best_success.pt",
                    model,
                    optimizer,
                    stats,
                    args,
                    global_step,
                    {"metrics": metrics},
                )
            if (
                args.save_model
                and episode_return
                > best_return
            ):
                best_return = (
                    episode_return
                )
                save_ckpt(
                    run_dir
                    / "ckpt_best_return.pt",
                    model,
                    optimizer,
                    stats,
                    args,
                    global_step,
                    {"metrics": metrics},
                )

        if (
            args.save_model
            and global_step
            % args.save_every
            == 0
        ):
            save_ckpt(
                run_dir
                / f"ckpt_{global_step}.pt",
                model,
                optimizer,
                stats,
                args,
                global_step,
            )

    if args.save_model:
        save_ckpt(
            run_dir
            / "ckpt_final.pt",
            model,
            optimizer,
            stats,
            args,
            global_step,
        )
    writer.close()


if __name__ == "__main__":
    main()

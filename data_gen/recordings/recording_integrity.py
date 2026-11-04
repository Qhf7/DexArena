"""Strict integrity checks shared by demonstration recording and export.

The recording pipeline is intentionally fail-closed: a partial episode is not
training data.  This module owns the cardinality, schema, and finiteness checks
used before a pending episode is made discoverable and again when it is loaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PENDING_SCHEMA_VERSION = 4
RECORDING_CONTRACT = "teleop-pending-lerobot-replay-v2"
RECORDING_CONTEXT_KEY = "__recording_context__"
class EpisodeIntegrityError(RuntimeError):
    """Raised when an episode cannot be exported or replayed losslessly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_schema(value: Any) -> tuple[str, tuple[int, ...]]:
    array = np.asarray(value)
    if array.dtype == np.dtype("O"):
        raise EpisodeIntegrityError("object-dtype values are not replay-safe")
    return str(array.dtype), tuple(int(dim) for dim in array.shape)


def tree_schema(value: Any, *, path: str = "root") -> Any:
    """Return a deterministic nested schema or raise for unsupported leaves."""
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), tree_schema(value[key], path=f"{path}.{key}"))
                for key in sorted(value, key=lambda item: str(item))
            ),
        )
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(tree_schema(item, path=f"{path}[{index}]") for index, item in enumerate(value)),
        )
    if isinstance(value, list):
        return (
            "list",
            tuple(tree_schema(item, path=f"{path}[{index}]") for index, item in enumerate(value)),
        )
    try:
        dtype, shape = _array_schema(value)
    except Exception as exc:
        raise EpisodeIntegrityError(f"unsupported value at {path}: {type(value).__name__}") from exc
    return "array", dtype, shape


def assert_tree_finite(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            assert_tree_finite(value[key], path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_tree_finite(item, path=f"{path}[{index}]")
        return
    array = np.asarray(value)
    if array.dtype == np.dtype("O"):
        raise EpisodeIntegrityError(f"object-dtype value at {path}")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise EpisodeIntegrityError(f"non-finite numeric value at {path}")


def validate_tree_sequence(
    name: str,
    sequence: Sequence[Any] | None,
    *,
    expected_len: int,
    required: bool,
) -> None:
    if sequence is None:
        if required:
            raise EpisodeIntegrityError(f"missing required {name}")
        return
    if len(sequence) != expected_len:
        raise EpisodeIntegrityError(
            f"{name} length {len(sequence)} != required {expected_len}"
        )
    if not sequence:
        return
    reference = tree_schema(sequence[0], path=f"{name}[0]")
    assert_tree_finite(sequence[0], path=f"{name}[0]")
    for index, value in enumerate(sequence[1:], start=1):
        current = tree_schema(value, path=f"{name}[{index}]")
        if current != reference:
            raise EpisodeIntegrityError(
                f"{name} schema changed at frame {index}: {current!r} != {reference!r}"
            )
        assert_tree_finite(value, path=f"{name}[{index}]")


def validate_episode_payload(
    *,
    obs: Any,
    actions: Any,
    rewards: Any,
    terminated: Any,
    truncated: Any,
    success: Any,
    env_state_sequence: Sequence[Any] | None,
    raw_state_sequence: Sequence[Any] | None,
    info_sequence: Sequence[Any] | None,
    image_sequences: Mapping[str, Any] | None,
    require_env_state: bool,
    require_raw_state: bool,
    require_info: bool = True,
    expected_image_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the exact T/T+1 contract and return normalized dimensions."""
    obs_array = np.asarray(obs)
    action_array = np.asarray(actions)
    if obs_array.ndim != 2:
        raise EpisodeIntegrityError(f"obs must be rank-2, got {obs_array.shape}")
    if action_array.ndim != 2:
        raise EpisodeIntegrityError(f"actions must be rank-2, got {action_array.shape}")
    action_count = int(action_array.shape[0])
    if action_count < 1:
        raise EpisodeIntegrityError("episode must contain at least one action")
    if obs_array.shape[0] != action_count + 1:
        raise EpisodeIntegrityError(
            f"obs length {obs_array.shape[0]} != actions+1 ({action_count + 1})"
        )
    if not np.all(np.isfinite(obs_array)):
        raise EpisodeIntegrityError("obs contains non-finite values")
    if not np.all(np.isfinite(action_array)):
        raise EpisodeIntegrityError("actions contains non-finite values")

    for name, values in (
        ("rewards", rewards),
        ("terminated", terminated),
        ("truncated", truncated),
        ("success", success),
    ):
        array = np.asarray(values).reshape(-1)
        if array.shape[0] != action_count:
            raise EpisodeIntegrityError(
                f"{name} length {array.shape[0]} != actions length {action_count}"
            )
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise EpisodeIntegrityError(f"{name} contains non-finite values")

    frame_count = action_count + 1
    validate_tree_sequence(
        "environment_state",
        env_state_sequence,
        expected_len=frame_count,
        required=require_env_state,
    )
    validate_tree_sequence(
        "raw_state",
        raw_state_sequence,
        expected_len=frame_count,
        required=require_raw_state,
    )
    if require_raw_state and raw_state_sequence is not None:
        for index, frame in enumerate(raw_state_sequence):
            if not isinstance(frame, Mapping):
                raise EpisodeIntegrityError(f"raw_state[{index}] is not a mapping")
            context = frame.get(RECORDING_CONTEXT_KEY)
            if not isinstance(context, Mapping) or "elapsed_steps" not in context:
                raise EpisodeIntegrityError(
                    f"raw_state[{index}] is missing {RECORDING_CONTEXT_KEY}.elapsed_steps"
                )
            elapsed = np.asarray(context["elapsed_steps"])
            if elapsed.size == 0 or not np.issubdtype(elapsed.dtype, np.number):
                raise EpisodeIntegrityError(f"raw_state[{index}] has invalid elapsed_steps")
            if not np.all(np.isfinite(elapsed)):
                raise EpisodeIntegrityError(f"raw_state[{index}] has non-finite elapsed_steps")
    validate_tree_sequence(
        "info",
        info_sequence,
        expected_len=action_count,
        required=require_info,
    )

    actual_image_keys = sorted(str(key) for key in (image_sequences or {}))
    if expected_image_keys is not None:
        expected_keys = sorted(str(key) for key in expected_image_keys)
        if actual_image_keys != expected_keys:
            raise EpisodeIntegrityError(
                f"camera schema changed: {actual_image_keys!r} != {expected_keys!r}"
            )
    for key in actual_image_keys:
        frames = np.asarray(image_sequences[key])
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise EpisodeIntegrityError(
                f"image sequence {key!r} must be (T+1,H,W,3), got {frames.shape}"
            )
        if frames.shape[0] != frame_count:
            raise EpisodeIntegrityError(
                f"image sequence {key!r} length {frames.shape[0]} != {frame_count}"
            )
        if frames.dtype != np.uint8:
            raise EpisodeIntegrityError(
                f"image sequence {key!r} must be uint8, got {frames.dtype}"
            )

    return {
        "num_actions": action_count,
        "num_frames": frame_count,
        "obs_dim": int(obs_array.shape[1]),
        "action_dim": int(action_array.shape[1]),
        "image_keys": actual_image_keys,
    }

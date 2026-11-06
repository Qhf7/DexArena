"""Zero-error replay verification for strict demonstration recordings.

State replay restores every recorded T+1 raw simulator state and checks that
both state_dict and policy observation reproduce exactly. Action replay restores
the initial raw state, reapplies the exact actions passed to env.step, and checks
all post-step states/observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch

from data_gen.recordings.episode_store import load_pending_episode
from data_gen.recordings.lerobot_integrity import (
    assert_valid_certificate,
    verify_replay_sidecar,
)
from data_gen.recordings.recording_integrity import (
    EpisodeIntegrityError,
    RECORDING_CONTEXT_KEY,
)


from data_gen.recordings.episode_store import restore_seed

# Observation and state helpers shared by the recording tools: the exact
# observation.state / image split the recordings were written with, numpy copies
# of simulator state trees, and RGB frame conversion for the renderers.
def to_numpy_copy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def clone_tree_to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clone_tree_to_numpy(sub_value) for key, sub_value in value.items()}
    return to_numpy_copy(value)


def _squeeze_single_env(value: Any) -> np.ndarray:
    array = to_numpy_copy(value).astype(np.float32, copy=False)
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array, dtype=np.float32).reshape(-1).copy()


# Keys that LeRobot uses for per-camera image features. We use one stable key
# per sensor so a dataset can be re-opened across sessions without schema drift.
def _image_feature_key(camera_name: str) -> str:
    return f"observation.images.{camera_name}"


def _flatten_proprio_obs(value: Any, parts: list[np.ndarray]) -> None:
    """Recursively concatenate every numeric leaf of ``value`` into ``parts``.

    The recorded ``observation.state`` is a flat float32 vector. When
    the env is in obs_mode="state" the env already returns a flat tensor and
    this helper is bypassed; when it's in obs_mode="rgb"/"rgbd" the env returns
    a nested dict (``agent``/``extra``/``sensor_param``/``sensor_data``). The
    caller has already stripped the sensor subtrees, so what reaches us here is
    pure proprioception that we just flatten in a stable (sorted-key) order.
    """
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            _flatten_proprio_obs(value[key], parts)
        return
    if value is None:
        return
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except Exception:
            return
    if array.dtype == np.dtype("O"):
        return
    try:
        array = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError):
        return
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    parts.append(np.asarray(array, dtype=np.float32).reshape(-1))


def _extract_state_and_images(obs: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Split a single-env obs into ``(state_vector, {camera_key: rgb_image})``.

    Works for both shapes a recording can have:

      * ``obs_mode="state"``: ``obs`` is a tensor / ndarray, no images.
      * ``obs_mode="rgb"`` / ``"rgbd"`` / ``"rgb+state"``: ``obs`` is a nested
        dict from ManiSkill containing ``sensor_data/<cam>/rgb`` plus the usual
        ``agent`` + ``extra`` proprio subtrees (and ``sensor_param/...`` camera
        intrinsics, which we drop — they're per-env constants, not a per-step
        observation the policy should depend on).

    Returns ``(state, images)`` where ``state`` is a 1-D float32 vector (the
    flattened proprio), and ``images`` maps the LeRobot feature key (e.g.
    ``observation.images.base_camera``) to a single uint8 (H,W,3) frame.
    """
    if not isinstance(obs, dict):
        return _squeeze_single_env(obs), {}

    images: dict[str, np.ndarray] = {}
    sensor_data = obs.get("sensor_data")
    if isinstance(sensor_data, dict):
        for camera_name, payload in sensor_data.items():
            if not isinstance(payload, dict):
                continue
            rgb = payload.get("rgb")
            if rgb is None:
                continue
            if torch.is_tensor(rgb):
                frame = rgb.detach().cpu().numpy()
            else:
                frame = np.asarray(rgb)
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise EpisodeIntegrityError(
                    f"camera {camera_name!r} RGB has invalid shape {frame.shape}"
                )
            images[_image_feature_key(str(camera_name))] = np.ascontiguousarray(
                frame.astype(np.uint8, copy=False)
            )

    parts: list[np.ndarray] = []
    for key in sorted(obs.keys()):
        if key in ("sensor_data", "sensor_param"):
            continue
        _flatten_proprio_obs(obs[key], parts)
    if not parts:
        state = np.zeros((0,), dtype=np.float32)
    else:
        state = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if state.size == 0:
        raise EpisodeIntegrityError("observation.state is empty")
    if not np.all(np.isfinite(state)):
        raise EpisodeIntegrityError("observation.state contains non-finite values")
    return state, images


def to_numpy_rgb_image(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, dict):
        for key in ("rgb", "Color", "color", "image"):
            if key in image:
                return to_numpy_rgb_image(image[key])
        return None
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] < 3:
        return None
    image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _tree_to_torch(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, Mapping):
        return {key: _tree_to_torch(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_torch(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_torch(item, device) for item in value]
    return torch.as_tensor(np.asarray(value), device=device)


def _assert_tree_exact(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise EpisodeIntegrityError(f"{path}: expected mapping, got {type(actual).__name__}")
        actual_keys = sorted(str(key) for key in actual)
        expected_keys = sorted(str(key) for key in expected)
        if actual_keys != expected_keys:
            raise EpisodeIntegrityError(
                f"{path}: keys differ: {actual_keys!r} != {expected_keys!r}"
            )
        actual_by_name = {str(key): value for key, value in actual.items()}
        expected_by_name = {str(key): value for key, value in expected.items()}
        for key in expected_keys:
            _assert_tree_exact(
                actual_by_name[key],
                expected_by_name[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise EpisodeIntegrityError(f"{path}: sequence structure differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_tree_exact(actual_item, expected_item, path=f"{path}[{index}]")
        return
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape or actual_array.dtype != expected_array.dtype:
        raise EpisodeIntegrityError(
            f"{path}: leaf schema mismatch actual={actual_array.dtype}{actual_array.shape}, "
            f"expected={expected_array.dtype}{expected_array.shape}"
        )
    if not np.array_equal(actual_array, expected_array):
        mismatch = np.flatnonzero(actual_array.reshape(-1) != expected_array.reshape(-1))
        first = int(mismatch[0])
        actual_flat = actual_array.reshape(-1)
        expected_flat = expected_array.reshape(-1)
        detail = f"first[{first}]={actual_flat[first]!r} expected={expected_flat[first]!r}"
        if np.issubdtype(actual_array.dtype, np.number):
            max_abs = float(np.max(np.abs(actual_array - expected_array)))
            detail += f" max_abs={max_abs:.9g}"
        raise EpisodeIntegrityError(f"{path}: leaf values differ; {detail}")


def _assert_observation_exact(
    obs: Any,
    expected_state: np.ndarray,
    expected_images: Mapping[str, np.ndarray],
    *,
    frame_index: int,
) -> None:
    state, images = _extract_state_and_images(obs)
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    expected_state = np.asarray(expected_state, dtype=np.float32).reshape(-1)
    if state.shape != expected_state.shape:
        raise EpisodeIntegrityError(
            f"frame {frame_index}: observation.state shape {state.shape} != "
            f"{expected_state.shape}"
        )
    absolute_error = np.abs(state - expected_state)
    if not np.all(absolute_error <= 1e-6):
        mismatch = np.flatnonzero(absolute_error > 1e-6)
        first = int(mismatch[0])
        max_abs = float(np.max(absolute_error))
        raise EpisodeIntegrityError(
            f"frame {frame_index}: observation.state differs at {len(mismatch)} values; "
            f"first[{first}]={state[first]!r} expected={expected_state[first]!r}; "
            f"max_abs={max_abs:.9g}"
        )
    if sorted(images) != sorted(expected_images):
        raise EpisodeIntegrityError(
            f"frame {frame_index}: camera keys differ during replay"
        )
    for key, expected in expected_images.items():
        if not np.array_equal(np.asarray(images[key], dtype=np.uint8), expected):
            raise EpisodeIntegrityError(
                f"frame {frame_index}: camera pixels differ for {key}"
            )


def _reconstruct_action(flat: np.ndarray, encoding: Mapping[str, Any]) -> Any:
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    kind = encoding.get("kind")
    if kind == "array":
        dtype = np.dtype(str(encoding["source_dtype"]))
        return flat.astype(dtype, copy=False).reshape(tuple(encoding["shape"]))
    if kind == "dict":
        action: dict[str, np.ndarray] = {}
        for entry in encoding.get("entries", []):
            start = int(entry["start"])
            stop = int(entry["stop"])
            dtype = np.dtype(str(entry["source_dtype"]))
            action[str(entry["key"])] = (
                flat[start:stop].astype(dtype, copy=False).reshape(tuple(entry["shape"]))
            )
        if not action:
            raise EpisodeIntegrityError("action encoding has no dict entries")
        return action
    raise EpisodeIntegrityError(f"unsupported action encoding: {kind!r}")


def _make_env(identity: Mapping[str, Any]) -> gym.Env:
    import tasks  # noqa: F401  registers every environment id

    env_id = str(identity.get("env_id") or "")
    if not env_id:
        raise EpisodeIntegrityError("recording identity is missing env_id")

    raw_kwargs = identity.get("env_kwargs") or "{}"
    if isinstance(raw_kwargs, str):
        env_kwargs = json.loads(raw_kwargs)
    elif isinstance(raw_kwargs, Mapping):
        env_kwargs = dict(raw_kwargs)
    else:
        raise EpisodeIntegrityError("identity env_kwargs is not a JSON object")
    if not isinstance(env_kwargs, dict):
        raise EpisodeIntegrityError("identity env_kwargs is not a JSON object")

    robot_uids = identity.get("robot_uids")
    if robot_uids and "robot_uids" not in env_kwargs:
        env_kwargs["robot_uids"] = str(robot_uids)

    common: dict[str, Any] = {
        "num_envs": 1,
        "render_mode": "rgb_array",
        "max_episode_steps": 1_000_000,
    }
    for key in ("obs_mode", "control_mode", "reward_mode"):
        value = identity.get(key)
        if value not in (None, "", "unknown"):
            common[key] = str(value)
    return gym.make(env_id, **common, **env_kwargs)


def _set_raw_state(env: gym.Env, raw_state: Any) -> None:
    import torch

    if not isinstance(raw_state, Mapping):
        raise EpisodeIntegrityError("raw state is not a mapping")
    base = env.unwrapped
    setter = getattr(base, "set_state_dict", None)
    if not callable(setter):
        raise EpisodeIntegrityError("environment has no set_state_dict")
    restorable = dict(raw_state)
    context = restorable.pop(RECORDING_CONTEXT_KEY, None)
    device = getattr(base, "device", "cpu")
    setter(_tree_to_torch(restorable, device))
    if context is not None:
        if not isinstance(context, Mapping) or "elapsed_steps" not in context:
            raise EpisodeIntegrityError("recording context has no elapsed_steps")
        target = getattr(base, "_elapsed_steps", None)
        if target is None:
            raise EpisodeIntegrityError("environment has no _elapsed_steps clock")
        restored = torch.as_tensor(context["elapsed_steps"], device=device)
        if torch.is_tensor(target):
            target.copy_(restored.to(dtype=target.dtype).reshape_as(target))
        else:
            setattr(base, "_elapsed_steps", restored)


def _current_raw_state(env: gym.Env) -> Any:
    base = env.unwrapped
    getter = getattr(base, "get_state_dict", None)
    if not callable(getter):
        raise EpisodeIntegrityError("environment has no get_state_dict")
    raw_state = clone_tree_to_numpy(getter())
    elapsed_steps = getattr(base, "_elapsed_steps", None)
    if isinstance(raw_state, dict) and elapsed_steps is not None:
        raw_state[RECORDING_CONTEXT_KEY] = {
            "elapsed_steps": clone_tree_to_numpy(elapsed_steps),
        }
    return raw_state


def _current_observation(env: gym.Env) -> Any:
    getter = getattr(env.unwrapped, "get_obs", None)
    if not callable(getter):
        raise EpisodeIntegrityError("environment has no get_obs")
    return getter()


def verify_episode_replay(
    episode: Mapping[str, Any],
    *,
    mode: str = "state",
    render: bool = False,
) -> dict[str, int]:
    meta = episode["meta"]
    identity = meta.get("identity_manifest") or {}
    raw_states = episode.get("raw_state_sequence")
    if raw_states is None:
        raise EpisodeIntegrityError("recording has no raw T+1 states")
    action_encoding = identity.get("action_encoding")
    if mode in ("action", "both") and not isinstance(action_encoding, Mapping):
        raise EpisodeIntegrityError("recording has no reversible action encoding")

    obs = np.asarray(episode["obs"], dtype=np.float32)
    image_sequences = episode.get("image_sequences") or {}
    env = _make_env(identity)
    state_frames = 0
    action_steps = 0
    try:
        env.reset(seed=restore_seed(episode))
        if mode in ("state", "both"):
            for frame_index, raw_state in enumerate(raw_states):
                _set_raw_state(env, raw_state)
                _assert_tree_exact(
                    _current_raw_state(env),
                    raw_state,
                    path=f"raw_state[{frame_index}]",
                )
                expected_images = {
                    key: np.asarray(values[frame_index], dtype=np.uint8)
                    for key, values in image_sequences.items()
                }
                _assert_observation_exact(
                    _current_observation(env),
                    obs[frame_index],
                    expected_images,
                    frame_index=frame_index,
                )
                if render:
                    env.render()
                state_frames += 1

        if mode in ("action", "both"):
            env.reset(seed=restore_seed(episode))
            _set_raw_state(env, raw_states[0])
            for step_index, flat_action in enumerate(episode["actions"]):
                action = _reconstruct_action(flat_action, action_encoding)
                device = getattr(env.unwrapped, "device", "cpu")
                action = _tree_to_torch(action, device)
                next_obs, _reward, _terminated, _truncated, _info = env.step(action)
                _assert_tree_exact(
                    _current_raw_state(env),
                    raw_states[step_index + 1],
                    path=f"action_replay.raw_state[{step_index + 1}]",
                )
                expected_images = {
                    key: np.asarray(values[step_index + 1], dtype=np.uint8)
                    for key, values in image_sequences.items()
                }
                _assert_observation_exact(
                    next_obs,
                    obs[step_index + 1],
                    expected_images,
                    frame_index=step_index + 1,
                )
                if render:
                    env.render()
                action_steps += 1
    finally:
        env.close()
    return {"state_frames": state_frames, "action_steps": action_steps}


def _episodes_from_target(
    target: Path,
    source_hashes: Sequence[str],
) -> list[dict[str, Any]]:
    if (target / "meta" / "info.json").exists():
        assert_valid_certificate(target)
        replay_root = target / "replay"
        candidates = sorted(replay_root.glob("source_*"))
        wanted = {value.lower().removeprefix("0x") for value in source_hashes}
        if wanted:
            candidates = [
                path for path in candidates if path.name.removeprefix("source_") in wanted
            ]
        if not candidates:
            raise EpisodeIntegrityError(f"no matching replay sidecars under {replay_root}")
        for path in candidates:
            verify_replay_sidecar(path)
        return [load_pending_episode(path) for path in candidates]
    verify_replay_sidecar(target) if (target / "manifest.json").exists() else None
    return [load_pending_episode(target)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify zero-error state/action replay for a strict recording."
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Pending episode, replay/source_* sidecar, or certified LeRobot dataset root.",
    )
    parser.add_argument(
        "--source-hash",
        action="append",
        default=[],
        help="Limit a dataset-root replay to one source hash (repeatable).",
    )
    parser.add_argument("--mode", choices=("state", "action", "both"), default="state")
    parser.add_argument("--render", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    episodes = _episodes_from_target(args.target.resolve(), args.source_hash)
    state_frames = 0
    action_steps = 0
    for episode in episodes:
        result = verify_episode_replay(episode, mode=args.mode, render=args.render)
        state_frames += result["state_frames"]
        action_steps += result["action_steps"]
    print(
        f"[ReplayVerify] PASS episodes={len(episodes)} "
        f"state_frames={state_frames} action_steps={action_steps} mode={args.mode}",
        flush=True,
    )


if __name__ == "__main__":
    main()

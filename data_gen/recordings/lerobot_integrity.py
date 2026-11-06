"""Lossless LeRobot export sidecars and post-export verification.

The standard policy columns remain directly consumable by LeRobot, while the
``replay/`` sidecar preserves the complete T/T+1 simulator record needed for
action replay and audit.  A dataset is considered trainable only after its
certificate reaches ``status=valid``.
"""

from __future__ import annotations

import io
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data_gen.recordings.recording_integrity import EpisodeIntegrityError, sha256_file


CERTIFICATE_RELATIVE_PATH = Path("meta") / "recording_integrity.json"
REPLAY_SCHEMA_VERSION = 1


def _artifact_hashes(dataset_root: Path) -> dict[str, str]:
    """Hash every training/replay artifact covered by the certificate."""
    artifacts: dict[str, str] = {}
    for relative_root in ("meta", "data", "videos", "replay"):
        root = dataset_root / relative_root
        if not root.exists():
            continue
        for path in sorted(
            candidate for candidate in root.rglob("*") if candidate.is_file()
        ):
            if path == certificate_path(dataset_root):
                continue
            artifacts[str(path.relative_to(dataset_root))] = sha256_file(path)
    return artifacts


def episode_source_hash(episode: Mapping[str, Any]) -> int:
    """Stable signed-int64-safe identity derived from the exact source payload."""
    digest = str(episode["meta"]["payload_sha256"]["episode.npz"])
    return int(digest[:15], 16)


def certificate_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / CERTIFICATE_RELATIVE_PATH


def write_certificate(
    dataset_root: str | Path,
    *,
    status: str,
    source_hashes: Sequence[int],
    diagnostics: Mapping[str, Any] | None = None,
) -> Path:
    target = certificate_path(dataset_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": str(status),
        "source_hashes": [int(value) for value in source_hashes],
        "artifacts_sha256": _artifact_hashes(Path(dataset_root)),
        "diagnostics": dict(diagnostics or {}),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def assert_valid_certificate(dataset_root: str | Path) -> dict[str, Any]:
    path = certificate_path(dataset_root)
    if not path.exists():
        raise EpisodeIntegrityError(
            f"strict LeRobot dataset has no integrity certificate: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != REPLAY_SCHEMA_VERSION:
        raise EpisodeIntegrityError(
            f"unsupported LeRobot integrity schema: {payload.get('schema_version')!r}"
        )
    if payload.get("status") != "valid":
        raise EpisodeIntegrityError(
            f"LeRobot integrity certificate is not valid: {payload.get('status')!r}"
        )
    artifacts = payload.get("artifacts_sha256")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise EpisodeIntegrityError("integrity certificate has no artifact checksums")
    root = Path(dataset_root)
    actual_artifacts = _artifact_hashes(root)
    certified_names = {str(name) for name in artifacts}
    actual_names = set(actual_artifacts)
    if certified_names != actual_names:
        missing = sorted(certified_names - actual_names)
        unexpected = sorted(actual_names - certified_names)
        raise EpisodeIntegrityError(
            "certified artifact set changed: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    for relative, expected in artifacts.items():
        actual = actual_artifacts[str(relative)]
        if actual != expected:
            raise EpisodeIntegrityError(
                f"certified artifact checksum mismatch: {root / str(relative)}"
            )
    return payload


def stage_replay_sidecar(dataset_root: str | Path, episode: Mapping[str, Any]) -> Path:
    """Copy the exact pending payload into the exported dataset atomically."""
    dataset_root = Path(dataset_root)
    source_hash = episode_source_hash(episode)
    replay_root = dataset_root / "replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    final_dir = replay_root / f"source_{source_hash:015x}"
    if final_dir.exists():
        existing = verify_replay_sidecar(final_dir)
        if int(existing.get("source_hash", -1)) != source_hash:
            raise EpisodeIntegrityError(f"replay sidecar identity collision: {final_dir}")
        source_dir = Path(episode["dir"])
        for name, expected in existing.get("files", {}).items():
            source = source_dir / name
            if not source.is_file() or sha256_file(source) != expected:
                raise EpisodeIntegrityError(
                    f"existing replay sidecar differs from pending source: {source}"
                )
        return final_dir

    temp_dir = replay_root / f".source_{source_hash:015x}.tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=False, exist_ok=False)
    source_dir = Path(episode["dir"])
    copied: list[str] = []
    for name in (
        "episode.npz",
        "env_states.pkl",
        "raw_states.pkl",
        "infos.pkl",
        "images.npz",
        "meta.json",
    ):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, temp_dir / name)
            copied.append(name)
    manifest = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_hash": source_hash,
        "episode_seed": int(episode["episode_seed"]),
        "identity_manifest": dict(episode["meta"].get("identity_manifest") or {}),
        "files": {
            name: sha256_file(temp_dir / name) for name in sorted(copied)
        },
    }
    (temp_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temp_dir.replace(final_dir)
    return final_dir


def _column_array(data: Mapping[str, Any], key: str, dtype: Any) -> np.ndarray:
    return np.asarray(data[key], dtype=dtype)


def _decode_image_cell(cell: Any, dataset_root: Path) -> np.ndarray:
    if isinstance(cell, Mapping):
        raw = cell.get("bytes")
        if raw is None and cell.get("path"):
            raw = (dataset_root / str(cell["path"])).read_bytes()
        if raw is None:
            raise EpisodeIntegrityError("LeRobot image cell has neither bytes nor path")
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(cell)
    if array.ndim == 3 and array.shape[-1] == 3:
        return array.astype(np.uint8, copy=False)
    raise EpisodeIntegrityError(f"unsupported LeRobot image cell: {type(cell).__name__}")


def verify_lerobot_roundtrip(
    dataset_root: str | Path,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare every newly exported LeRobot transition with its pending source."""
    from utils.demo_dataset import (
        _flatten_env_state_sequence_from_dicts,
        _inspect_env_state_fields_from_sequence,
        maybe_import_pyarrow,
    )

    dataset_root = Path(dataset_root)
    pq = maybe_import_pyarrow()
    parquet_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise EpisodeIntegrityError(f"LeRobot export has no parquet data: {dataset_root}")
    columns = [
        "episode_index",
        "frame_index",
        "episode.source_hash",
        "episode.seed",
        "observation.state",
        "next.observation.state",
        "action",
        "next.reward",
        "next.done",
        "next.terminated",
        "next.truncated",
        "next.recording_boundary",
        "next.success",
    ]
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    features = info.get("features", {})
    if "observation.environment_state" in features:
        columns.extend(
            [
                "observation.environment_state",
                "next.observation.environment_state",
            ]
        )
    # Image features live in the parquet (one PNG cell per frame) and round-trip exactly.
    # Video features live in videos/*.mp4, are lossy by construction (AV1) and round-trip
    # through the decoder with a PSNR floor instead -- see _verify_video_feature.
    image_keys = sorted(
        key
        for key, feature in features.items()
        if key.startswith("observation.images.") and feature.get("dtype") == "image"
    )
    video_keys = sorted(
        key
        for key, feature in features.items()
        if key.startswith("observation.images.") and feature.get("dtype") == "video"
    )
    columns.extend(image_keys)
    columns.extend(f"next.{key}" for key in image_keys if f"next.{key}" in features)
    table = pq.read_table(parquet_paths, columns=columns).to_pydict()
    video_roundtrip: dict[str, dict[str, Any]] = {}
    source_values = np.asarray(table["episode.source_hash"], dtype=np.int64).reshape(-1)
    frame_values = np.asarray(table["frame_index"], dtype=np.int64).reshape(-1)

    verified_transitions = 0
    for episode in episodes:
        source_hash = episode_source_hash(episode)
        indices = np.flatnonzero(source_values == source_hash)
        if indices.size != int(episode["actions"].shape[0]):
            raise EpisodeIntegrityError(
                f"source {source_hash:015x}: LeRobot rows={indices.size}, "
                f"expected={episode['actions'].shape[0]}"
            )
        indices = indices[np.argsort(frame_values[indices])]

        def exact(column: str, expected: Any, dtype: Any) -> None:
            actual = np.asarray(
                [np.asarray(table[column][int(i)]).reshape(-1) for i in indices],
                dtype=dtype,
            )
            wanted = np.asarray(expected, dtype=dtype)
            if actual.shape != wanted.shape or not np.array_equal(actual, wanted):
                raise EpisodeIntegrityError(
                    f"source {source_hash:015x}: roundtrip mismatch in {column}; "
                    f"actual={actual.shape}, expected={wanted.shape}"
                )

        obs = np.asarray(episode["obs"], dtype=np.float32)
        terminated = np.asarray(episode["terminated"], dtype=np.bool_).reshape(-1, 1)
        truncated = np.asarray(episode["truncated"], dtype=np.bool_).reshape(-1, 1)
        exact("observation.state", obs[:-1], np.float32)
        exact("next.observation.state", obs[1:], np.float32)
        exact("action", episode["actions"], np.float32)
        exact("next.reward", np.asarray(episode["rewards"]).reshape(-1, 1), np.float32)
        exact("next.terminated", terminated, np.bool_)
        exact("next.truncated", truncated, np.bool_)
        exact("next.done", np.logical_or(terminated, truncated), np.bool_)
        boundary = np.zeros_like(terminated)
        boundary[-1, 0] = True
        exact("next.recording_boundary", boundary, np.bool_)
        exact("next.success", np.asarray(episode["success"]).reshape(-1, 1), np.bool_)
        exact(
            "episode.seed",
            np.full((indices.size, 1), int(episode["episode_seed"]), np.int64),
            np.int64,
        )

        env_sequence = episode.get("env_state_sequence")
        if env_sequence is not None:
            fields = _inspect_env_state_fields_from_sequence(env_sequence)
            flattened = _flatten_env_state_sequence_from_dicts(env_sequence, fields)
            exact("observation.environment_state", flattened[:-1], np.float32)
            exact("next.observation.environment_state", flattened[1:], np.float32)

        for key, images in (episode.get("image_sequences") or {}).items():
            image_values = np.asarray(images, dtype=np.uint8)
            if key in video_keys:
                episode_index = int(table["episode_index"][int(indices[0])])
                for feature_key, wanted in ((key, image_values[:-1]), (f"next.{key}", image_values[1:])):
                    if feature_key not in features:
                        continue
                    stats = _verify_video_feature(dataset_root, info, feature_key, episode_index, wanted)
                    agg = video_roundtrip.setdefault(feature_key, {"episodes": 0, "frames": 0,
                                                                   "min_psnr_db": float("inf"),
                                                                   "max_mean_abs_diff": 0.0})
                    agg["episodes"] += 1
                    agg["frames"] += int(wanted.shape[0])
                    agg["min_psnr_db"] = min(agg["min_psnr_db"], stats["min_psnr_db"])
                    agg["max_mean_abs_diff"] = max(agg["max_mean_abs_diff"], stats["mean_abs_diff"])
                continue
            if key not in image_keys:
                raise EpisodeIntegrityError(f"source {source_hash:015x}: {key} is not an exported feature")
            current = np.stack(
                [_decode_image_cell(table[key][int(i)], dataset_root) for i in indices]
            )
            if not np.array_equal(current, image_values[:-1]):
                raise EpisodeIntegrityError(f"source {source_hash:015x}: image mismatch in {key}")
            if f"next.{key}" in features:
                following = np.stack(
                    [_decode_image_cell(table[f"next.{key}"][int(i)], dataset_root) for i in indices]
                )
                if not np.array_equal(following, image_values[1:]):
                    raise EpisodeIntegrityError(
                        f"source {source_hash:015x}: image mismatch in next.{key}"
                    )
        verified_transitions += int(indices.size)

    result = {
        "verified_episodes": len(episodes),
        "verified_transitions": verified_transitions,
        "parquet_sha256": {
            str(path.relative_to(dataset_root)): sha256_file(path)
            for path in parquet_paths
        },
    }
    if video_roundtrip:
        result["video_roundtrip"] = {"min_psnr_db_required": VIDEO_ROUNDTRIP_MIN_PSNR_DB, **video_roundtrip}
    return result


# The floor catches a broken encode, not a misaligned one: alignment is checked by the
# decoder against the frame timestamps (_decode_frames_pyav), and a wrong-but-adjacent
# frame of a still scene would pass any PSNR floor anyway. AV1 at LeRobot's default crf=30
# sits at 32-35 dB on rendered 256x256 frames and dipped to 29.9 dB on one fast-motion wrist
# frame (PickAndPlaceBread ep 51), so 30 was too tight; a corrupt or empty
# frame is far below 25.
VIDEO_ROUNDTRIP_MIN_PSNR_DB = 25.0


def _episode_video_source(dataset_root: Path, info: Mapping[str, Any], feature_key: str,
                          episode_index: int) -> tuple[Path, float]:
    """(mp4 path, timestamp of this episode's first frame) from the v3 episodes ledger."""
    from utils.demo_dataset import maybe_import_pyarrow

    pq = maybe_import_pyarrow()
    ledgers = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not ledgers:
        raise EpisodeIntegrityError(f"LeRobot export has no episodes ledger: {dataset_root}")
    wanted = ["episode_index"] + [f"videos/{feature_key}/{field}"
                                  for field in ("chunk_index", "file_index", "from_timestamp")]
    rows = pq.read_table(ledgers, columns=wanted).to_pydict()
    hits = [i for i, value in enumerate(rows["episode_index"]) if int(value) == int(episode_index)]
    if len(hits) != 1:
        raise EpisodeIntegrityError(
            f"episode {episode_index}: {len(hits)} ledger rows for {feature_key}, expected 1")
    row = hits[0]
    template = str(info.get("video_path") or "")
    if not template:
        raise EpisodeIntegrityError("info.json declares video features but no video_path")
    path = dataset_root / template.format(
        video_key=feature_key,
        chunk_index=int(rows[f"videos/{feature_key}/chunk_index"][row]),
        file_index=int(rows[f"videos/{feature_key}/file_index"][row]),
    )
    return path, float(rows[f"videos/{feature_key}/from_timestamp"][row])


def _decode_frames_pyav(path: Path, start: float, count: int, fps: float) -> np.ndarray:
    """`count` RGB frames of `path` starting at timestamp `start`, decoded with PyAV.

    LeRobot's own decoder wants torchvision's VideoReader, which this torchvision build
    lacks; PyAV is what LeRobot encodes with, so it is always present. LeRobot packs
    several episodes into one mp4 and keys every second frame (g=2), so a backward seek
    to `start` lands at most one frame early and the scan below is short.
    """
    import av

    tolerance = 0.25 / fps
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        container.seek(int(max(start - 1.0 / fps, 0.0) / time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            stamp = float(frame.pts * frame.time_base)
            if stamp < start - tolerance:
                continue
            expected = start + len(frames) / fps
            if abs(stamp - expected) > tolerance:
                raise EpisodeIntegrityError(
                    f"{path.name}: frame at {stamp:.4f}s where {expected:.4f}s was expected "
                    f"(episode start {start:.4f}s, frame {len(frames)})")
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise EpisodeIntegrityError(f"{path.name}: decoded {len(frames)} frames from {start:.4f}s, expected {count}")
    return np.stack(frames)


def _verify_video_feature(dataset_root: Path, info: Mapping[str, Any], feature_key: str,
                          episode_index: int, wanted: np.ndarray) -> dict[str, float]:
    """Decode this episode's frames of one video feature and hold them to a PSNR floor."""
    fps = float(info["fps"])
    path, start = _episode_video_source(dataset_root, info, feature_key, episode_index)
    if not path.exists():
        raise EpisodeIntegrityError(f"episode {episode_index}: video file missing: {path}")
    frames = _decode_frames_pyav(path, start, int(wanted.shape[0]), fps)
    if frames.shape != wanted.shape:
        raise EpisodeIntegrityError(
            f"episode {episode_index}: {feature_key} decoded {frames.shape}, rendered {wanted.shape}")
    diff = frames.astype(np.float32) - wanted.astype(np.float32)
    mse = np.mean(np.square(diff), axis=(1, 2, 3))
    psnr = np.where(mse > 0, 10.0 * np.log10((255.0 ** 2) / np.maximum(mse, 1e-12)), np.inf)
    min_psnr = float(psnr.min())
    if min_psnr < VIDEO_ROUNDTRIP_MIN_PSNR_DB:
        worst = int(np.argmin(psnr))
        raise EpisodeIntegrityError(
            f"episode {episode_index}: {feature_key} frame {worst} decodes at {min_psnr:.1f} dB "
            f"(< {VIDEO_ROUNDTRIP_MIN_PSNR_DB} dB): wrong frame or broken encode, not compression")
    return {"min_psnr_db": min_psnr, "mean_abs_diff": float(np.abs(diff).mean())}


def verify_replay_sidecar(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("files", {}).items():
        actual = sha256_file(path / name)
        if actual != expected:
            raise EpisodeIntegrityError(
                f"replay sidecar checksum mismatch for {path / name}"
            )
    return manifest


def sidecars_in_episode_order(dataset_root: str | Path) -> list[Path]:
    """Replay sidecars in the dataset's own episode order (read back from the parquet), so a
    re-export or a derived dataset keeps episode i as episode i."""
    from utils.demo_dataset import maybe_import_pyarrow

    dataset_root = Path(dataset_root)
    pq = maybe_import_pyarrow()
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise EpisodeIntegrityError(f"{dataset_root}: no parquet data")
    table = pq.read_table(files, columns=["episode_index", "episode.source_hash"]).to_pydict()
    order: dict[int, int] = {}
    for index, source in zip(table["episode_index"], table["episode.source_hash"]):
        index, source = int(index), int(np.asarray(source).reshape(-1)[0])
        if order.setdefault(index, source) != source:
            raise EpisodeIntegrityError(f"{dataset_root.name}: episode {index} carries two source hashes")
    out = []
    for index in range(len(order)):
        sidecar = dataset_root / "replay" / f"source_{order[index]:015x}"
        if not sidecar.is_dir():
            raise EpisodeIntegrityError(f"{dataset_root.name}: episode {index} has no replay sidecar {sidecar.name}")
        out.append(sidecar)
    return out

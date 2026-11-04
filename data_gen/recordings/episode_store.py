"""On-disk staging area for recorded demonstration episodes (the "pending" store).

The recording session buffers each auto-segmented episode and writes it here as a
self-contained folder *before* any decision is made about keeping it. A separate
review tool (``data_gen.recordings.review_demos``) later replays/inspects each pending
episode and exports only the kept ones into the final, directly-trainable LeRobot
dataset.

Design constraints (so the final export needs ZERO post-processing):
  * The pending store is deliberately NOT a LeRobot dataset root. It contains no
    ``meta/info.json`` + ``data/`` layout, so ``resolve_lerobot_dataset_roots``
    used by the training scripts will never mistake it for a dataset.
  * Each episode keeps exactly what ``LeRobotEpisodeRecorder.add_episode`` needs
    (obs / actions / rewards / terminated / truncated / success / episode_seed /
    env_state_sequence), so the review tool can reproduce the current
    ``--record-demo`` output byte-for-byte in format.
  * Optionally each episode also keeps the RAW, restorable ManiSkill state_dict
    per frame, so a future offline re-render (option "B": replay with a different
    camera / rebuild observations) can be added without re-collecting anything.
"""

from __future__ import annotations

import json
import pickle
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data_gen.recordings.recording_integrity import (
    PENDING_SCHEMA_VERSION,
    RECORDING_CONTRACT,
    EpisodeIntegrityError,
    sha256_file,
    validate_episode_payload,
)


def _validate_identity_manifest(value: Mapping[str, Any] | None) -> dict[str, Any]:
    manifest = dict(value or {})
    if manifest.get("contract") != RECORDING_CONTRACT:
        raise EpisodeIntegrityError(
            f"recording contract must be {RECORDING_CONTRACT!r}"
        )
    for key in ("env_id", "task", "robot_uids", "control_mode", "obs_mode"):
        if not str(manifest.get(key, "")).strip():
            raise EpisodeIntegrityError(f"recording identity is missing {key!r}")
    try:
        parsed_env_kwargs = json.loads(str(manifest.get("env_kwargs", "")))
    except json.JSONDecodeError as exc:
        raise EpisodeIntegrityError(
            f"recording identity has invalid env_kwargs JSON: {exc}"
        ) from exc
    if not isinstance(parsed_env_kwargs, dict):
        raise EpisodeIntegrityError(
            "recording identity env_kwargs must decode to a JSON object"
        )
    action_dim = int(manifest.get("action_dim", 0))
    if action_dim <= 0 or not isinstance(manifest.get("action_encoding"), Mapping):
        raise EpisodeIntegrityError(
            "recording identity is missing a valid action encoding"
        )
    manifest["action_dim"] = action_dim
    manifest["env_kwargs"] = json.dumps(
        parsed_env_kwargs, sort_keys=True, separators=(",", ":")
    )
    return manifest


def _now_stamp() -> str:
    """Session id: the start time to the second. The store appends the env id, so two
    sessions of different tasks started in the same second do not collide."""
    return time.strftime("%Y%m%d_%H%M%S")


def _write_video(frames: Sequence[np.ndarray], out_dir: Path, name: str, fps: int) -> str | None:
    """Write ``frames`` to ``out_dir/name.mp4``. Returns the file name or None."""
    usable = [np.asarray(f) for f in frames if f is not None]
    if not usable:
        return None
    try:
        from mani_skill.utils.visualization.misc import images_to_video

        images_to_video(usable, str(out_dir), name, fps=int(fps), verbose=False)
        return f"{name}.mp4"
    except Exception as exc:  # pragma: no cover - video is best-effort
        print(f"[PendingStore] Warning: failed to write video {name}.mp4: {exc}", flush=True)
        return None


def _slug_camera_key(key: str) -> str:
    """Turn 'observation.images.base_camera' into a tidy mp4 stem 'cam_base_camera'."""
    stem = key
    for prefix in ("observation.images.", "observation.image.", "images.", "image."):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_")
    return f"cam_{cleaned or 'unknown'}"


@dataclass
class PendingEpisodeStore:
    """Writes one folder per buffered episode under ``root/session_id``."""

    root: Path
    env_id: str
    control_mode: str
    robot_type: str
    repo_prefix: str
    dataset_root: Path
    record_lerobot_fps: int
    record_env_state: bool = True
    record_raw_state: bool = True
    identity_manifest: Mapping[str, Any] | None = None
    save_video: bool = True
    video_fps: int = 10
    session_id: str = field(default_factory=_now_stamp)
    _count: int = 0

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.dataset_root = Path(self.dataset_root)
        self.session_dir = self.root / f"session_{self.session_id}_{self.env_id}"

    @property
    def num_pending(self) -> int:
        return self._count

    def quarantine_episode(
        self,
        *,
        episode_seed: int,
        reason: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> Path:
        """Persist a failure marker without making the episode exportable."""
        quarantine_root = self.session_dir / "quarantine"
        quarantine_root.mkdir(parents=True, exist_ok=True)
        target = quarantine_root / (
            f"episode_seed_{int(episode_seed)}_{time.time_ns()}"
        )
        target.mkdir(parents=False, exist_ok=False)
        payload = {
            "schema_version": PENDING_SCHEMA_VERSION,
            "integrity_status": "quarantined",
            "episode_seed": int(episode_seed),
            "reason": str(reason),
            "diagnostics": dict(diagnostics or {}),
            "identity_manifest": dict(self.identity_manifest or {}),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (target / "reason.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"[PendingStore] QUARANTINED -> {target}: {reason}", flush=True)
        return target

    def add_episode(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        success: np.ndarray,
        episode_seed: int,
        env_state_sequence: Sequence[dict[str, Any]] | None,
        raw_state_sequence: Sequence[dict[str, Any]] | None,
        frames: Sequence[np.ndarray] | None,
        auto_success: bool,
        info_sequence: Sequence[dict[str, Any]] | None = None,
        # Per-camera RGB streams must be uint8 (T+1,H,W,3) arrays with the same
        # leading dimension as obs. None means the episode is state-only.
        image_sequences: Mapping[str, np.ndarray] | None = None,
        # The seed ManiSkill drew for the episode (``env._episode_seed``); None
        # for recordings that predate it. See ``restore_seed``.
        env_episode_seed: int | None = None,
    ) -> Path:
        """Persist a single buffered episode. Returns its folder path."""
        identity_manifest = _validate_identity_manifest(self.identity_manifest)
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        integrity = validate_episode_payload(
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            success=success,
            env_state_sequence=env_state_sequence,
            raw_state_sequence=raw_state_sequence,
            info_sequence=info_sequence,
            image_sequences=image_sequences,
            require_env_state=bool(self.record_env_state),
            require_raw_state=bool(self.record_raw_state),
            require_info=True,
            expected_image_keys=sorted(str(key) for key in (image_sequences or {})),
        )

        self.session_dir.mkdir(parents=True, exist_ok=True)
        final_ep_dir = self.session_dir / f"episode_{self._count:04d}"
        if final_ep_dir.exists():
            raise EpisodeIntegrityError(f"pending episode already exists: {final_ep_dir}")
        ep_dir = self.session_dir / (
            f".episode_{self._count:04d}.tmp-{uuid.uuid4().hex}"
        )
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ep_dir / "episode.npz",
            obs=obs,
            actions=actions,
            rewards=np.asarray(rewards, dtype=np.float32).reshape(-1),
            terminated=np.asarray(terminated, dtype=np.bool_).reshape(-1),
            truncated=np.asarray(truncated, dtype=np.bool_).reshape(-1),
            success=np.asarray(success, dtype=np.bool_).reshape(-1),
            episode_seed=np.asarray([int(episode_seed)], dtype=np.int64),
        )

        has_env_state = bool(self.record_env_state and env_state_sequence is not None)
        if has_env_state:
            with (ep_dir / "env_states.pkl").open("wb") as fh:
                pickle.dump(list(env_state_sequence), fh)

        has_raw_state = bool(self.record_raw_state and raw_state_sequence is not None)
        if has_raw_state:
            with (ep_dir / "raw_states.pkl").open("wb") as fh:
                pickle.dump(list(raw_state_sequence), fh)

        has_info = info_sequence is not None
        if has_info:
            with (ep_dir / "infos.pkl").open("wb") as fh:
                pickle.dump(list(info_sequence), fh)

        video_name = None
        if self.save_video and frames:
            video_name = _write_video(frames, ep_dir, "video", self.video_fps)

        # Per-camera RGB streams: drop to a single compressed npz so review_demos
        # can pull them straight back when exporting to LeRobot. We also keep a
        # tiny manifest (key -> (H,W,3)) in meta.json so the review tool can sanity-
        # check the schema without unpacking the array file.
        image_keys: list[tuple[str, tuple[int, int, int]]] = []
        # Per-camera preview videos (`cam_<key>.mp4`) -- best-effort, written
        # alongside images.npz so reviewers can flip through each viewpoint
        # without unpacking the array file. The LeRobot export still reads
        # images.npz, so these mp4s are purely a preview convenience.
        camera_videos: dict[str, str] = {}
        has_images = False
        if image_sequences:
            arrays_to_save: dict[str, np.ndarray] = {}
            for key, values in image_sequences.items():
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
                images = np.ascontiguousarray(images.astype(np.uint8, copy=False))
                arrays_to_save[str(key)] = images
                image_keys.append((str(key), tuple(int(d) for d in images.shape[1:])))
            if arrays_to_save:
                np.savez_compressed(ep_dir / "images.npz", **arrays_to_save)
                has_images = True
                if self.save_video:
                    cam_fps = max(int(self.record_lerobot_fps), int(self.video_fps), 1)
                    for key, arr in arrays_to_save.items():
                        stem = _slug_camera_key(key)
                        written = _write_video(list(arr), ep_dir, stem, cam_fps)
                        if written:
                            camera_videos[key] = written

        final_success = bool(np.asarray(success, dtype=np.bool_).reshape(-1)[-1]) if len(actions) else False
        checksum_names = ["episode.npz"]
        for optional_name in (
            "env_states.pkl",
            "raw_states.pkl",
            "infos.pkl",
            "images.npz",
        ):
            if (ep_dir / optional_name).exists():
                checksum_names.append(optional_name)
        meta = {
            "schema_version": PENDING_SCHEMA_VERSION,
            "integrity_status": "valid",
            "env_id": self.env_id,
            "control_mode": self.control_mode,
            "robot_type": self.robot_type,
            "repo_prefix": self.repo_prefix,
            "dataset_root": str(self.dataset_root),
            "record_lerobot_fps": int(self.record_lerobot_fps),
            "record_env_state": has_env_state,
            "record_raw_state": has_raw_state,
            "has_raw_state": has_raw_state,
            "has_info": has_info,
            "has_images": has_images,
            # Sorted so that two episodes recorded with the same camera setup
            # produce IDENTICAL ``image_keys`` lists, which is what
            # `_export_group` uses to bucket compatible episodes together.
            "image_keys": sorted(image_keys),
            "num_actions": integrity["num_actions"],
            "num_frames": integrity["num_frames"],
            "obs_dim": integrity["obs_dim"],
            "action_dim": integrity["action_dim"],
            "episode_seed": int(episode_seed),
            **({"env_episode_seed": int(env_episode_seed)} if env_episode_seed is not None else {}),
            "auto_success": bool(auto_success),
            "final_success": final_success,
            "video": video_name,
            # Per-camera preview videos: key -> filename, e.g.
            #   "observation.images.base_camera" -> "cam_base_camera.mp4"
            # Empty dict (instead of None) so consumers can iterate unconditionally.
            "camera_videos": dict(sorted(camera_videos.items())),
            "identity_manifest": identity_manifest,
            "payload_sha256": {
                name: sha256_file(ep_dir / name) for name in sorted(checksum_names)
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "pending",
        }
        (ep_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        ep_dir.replace(final_ep_dir)

        self._count += 1
        cam_count = len(meta.get("camera_videos") or {})
        print(
            f"[PendingStore] Buffered episode -> {final_ep_dir} "
            f"({meta['num_actions']} actions, success={final_success}, "
            f"auto={auto_success}, video={'yes' if video_name else 'no'}, "
            f"camera_videos={cam_count}).",
            flush=True,
        )
        return final_ep_dir


def load_pending_episode(ep_dir: Path) -> dict[str, Any]:
    """Load one pending episode folder into a dict usable for LeRobot export."""
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    if int(meta.get("schema_version", 0)) != PENDING_SCHEMA_VERSION:
        raise EpisodeIntegrityError(
            f"unsupported pending schema at {ep_dir}; strict replay requires "
            f"schema {PENDING_SCHEMA_VERSION}"
        )
    if meta.get("integrity_status") != "valid":
        raise EpisodeIntegrityError(
            f"pending episode is not valid: {meta.get('integrity_status')!r}"
        )
    meta["identity_manifest"] = _validate_identity_manifest(
        meta.get("identity_manifest")
    )
    for name, expected in (meta.get("payload_sha256") or {}).items():
        path = ep_dir / name
        if not path.exists():
            raise EpisodeIntegrityError(f"missing pending payload: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise EpisodeIntegrityError(
                f"checksum mismatch for {path}: {actual} != {expected}"
            )
    with np.load(ep_dir / "episode.npz") as data:
        arrays = {key: data[key] for key in data.files}

    env_state_sequence = None
    env_states_path = ep_dir / "env_states.pkl"
    if meta.get("record_env_state") and env_states_path.exists():
        with env_states_path.open("rb") as fh:
            env_state_sequence = pickle.load(fh)

    raw_state_sequence = None
    raw_states_path = ep_dir / "raw_states.pkl"
    if meta.get("record_raw_state") and raw_states_path.exists():
        with raw_states_path.open("rb") as fh:
            raw_state_sequence = pickle.load(fh)

    info_sequence = None
    infos_path = ep_dir / "infos.pkl"
    if meta.get("has_info") and infos_path.exists():
        with infos_path.open("rb") as fh:
            info_sequence = pickle.load(fh)

    image_sequences: dict[str, np.ndarray] | None = None
    images_path = ep_dir / "images.npz"
    if meta.get("has_images") and images_path.exists():
        with np.load(images_path) as data:
            image_sequences = {key: data[key].copy() for key in data.files}

    validate_episode_payload(
        obs=arrays["obs"],
        actions=arrays["actions"],
        rewards=arrays["rewards"],
        terminated=arrays["terminated"],
        truncated=arrays["truncated"],
        success=arrays["success"],
        env_state_sequence=env_state_sequence,
        raw_state_sequence=raw_state_sequence,
        info_sequence=info_sequence,
        image_sequences=image_sequences,
        require_env_state=bool(meta.get("record_env_state")),
        require_raw_state=bool(meta.get("record_raw_state")),
        require_info=bool(meta.get("has_info")),
        expected_image_keys=[key for key, _shape in meta.get("image_keys", [])],
    )

    return {
        "dir": ep_dir,
        "meta": meta,
        "obs": arrays["obs"],
        "actions": arrays["actions"],
        "rewards": arrays["rewards"],
        "terminated": arrays["terminated"],
        "truncated": arrays["truncated"],
        "success": arrays["success"],
        "episode_seed": int(arrays["episode_seed"].reshape(-1)[0]),
        "env_episode_seed": meta.get("env_episode_seed"),
        "env_state_sequence": env_state_sequence,
        "raw_state_sequence": raw_state_sequence,
        "info_sequence": info_sequence,
        "image_sequences": image_sequences,
        "video": (ep_dir / meta["video"]) if meta.get("video") else None,
    }


def restore_seed(meta, default=None) -> int:
    """The seed a consumer must pass to ``env.reset`` to rebuild this episode's scene.

    Accepts an episode meta dict or a loaded episode record (which carries the
    seeds at top level and the meta under ``"meta"``).

    ``episode_seed`` is the recorder's own counter (it is what ``--episode-seed``
    started at, plus one per episode); the recording loop never passes it to
    ``reset``. ``env_episode_seed`` is what ManiSkill actually drew for the
    episode (``env._episode_seed``); older recordings carry it back-filled
    where a scene body depends on it. Anything placed at scene
    build from the episode RNG -- TableClean's static bin -- restores only from
    the latter; everything else lives in the state dict and does not care.
    """
    nested = meta.get("meta") if isinstance(meta.get("meta"), Mapping) else {}
    for key in ("env_episode_seed", "episode_seed"):
        for source in (meta, nested):
            value = source.get(key)
            if value is not None:
                return int(value)
    if default is None:
        raise KeyError("episode_seed")
    return int(default)


def discover_pending_episodes(root: Path) -> list[Path]:
    """Return all pending episode folders under ``root`` (sorted, stable order)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        meta_path.parent
        for meta_path in root.rglob("meta.json")
        if meta_path.parent.name.startswith("episode_")
        and not meta_path.parent.name.startswith(".")
        and (meta_path.parent / "episode.npz").exists()
    )

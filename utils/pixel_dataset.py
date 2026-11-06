"""HDF5 schema and reader for offline image datasets::

    /episodes/{i}/
        cam_0        uint8   (T + 1, H, W, 3)   observations, incl. the terminal one
        cam_1        uint8   (T + 1, H, W, 3)
        proprio      float32 (T + 1, D)
        action       float32 (T, A)
        reward       float32 (T,)
        done         bool    (T,)
        truncated    bool    (T,)
        attrs: success (bool), num_transitions (int)
    /meta/
        attrs: camera_config_hash, control_mode, obs_version, env_id, robot_uids,
               done_convention, schema_version

The frame-like arrays hold ``T + 1`` rows so the terminal observation is stored;
``o``/``o'`` share frames by index like the online buffer. ``camera_config_hash``
must match the current :class:`PixelObsConfig`, otherwise loading is an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from utils.pixel_cameras import PixelObsConfig


SCHEMA_VERSION = "pixel_episodes_v1"
# What `done=1` means; asserted against the online convention at training start.
DONE_CONVENTION = "terminated_only"  # time-limit truncation is NOT a done


class PixelEpisodeWriter:
    """Append episodes to the schema above."""

    def __init__(
        self,
        path: str | Path,
        config: PixelObsConfig,
        *,
        control_mode: str,
        env_id: str,
        robot_uids: str | None = None,
        compression: str | None = "gzip",
        append: bool = False,
    ):
        """``append=True`` continues an existing file (rendering in chunks);
        refuses one written under a different camera config, env or convention."""
        import h5py  # local import: only the data pipeline needs h5py

        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.compression = compression
        if append and self.path.exists():
            self.file = h5py.File(self.path, "a")
            expected = {
                "camera_config_hash": config.config_hash(),
                "control_mode": str(control_mode),
                "obs_version": str(config.obs_version),
                "env_id": str(env_id),
                "robot_uids": str(robot_uids or ""),
                "done_convention": DONE_CONVENTION,
                "schema_version": SCHEMA_VERSION,
            }
            attrs = self.file["meta"].attrs
            mismatched = {
                key: (str(attrs.get(key)), value)
                for key, value in expected.items()
                if str(attrs.get(key)) != value
            }
            keys = sorted(int(key) for key in self.file["episodes"])
            if mismatched or keys != list(range(len(keys))):
                self.file.close()
                raise ValueError(
                    f"Cannot append to {self.path}: meta (file, now) {mismatched}, "
                    f"episode keys contiguous={keys == list(range(len(keys)))}"
                )
            self.num_episodes = len(keys)
            return
        self.file = h5py.File(self.path, "w")
        self.file.create_group("episodes")
        meta = self.file.create_group("meta")
        meta.attrs["camera_config_hash"] = config.config_hash()
        meta.attrs["camera_config"] = str(config.to_dict())
        meta.attrs["control_mode"] = str(control_mode)
        meta.attrs["obs_version"] = str(config.obs_version)
        meta.attrs["env_id"] = str(env_id)
        meta.attrs["robot_uids"] = str(robot_uids or "")
        meta.attrs["done_convention"] = DONE_CONVENTION
        meta.attrs["schema_version"] = SCHEMA_VERSION
        self.num_episodes = 0

    def add_episode(
        self,
        observations: Mapping[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        truncated: np.ndarray,
        success: bool,
    ) -> None:
        """``observations`` holds ``T + 1`` frames; the transition arrays hold ``T``."""
        num_transitions = int(actions.shape[0])
        group = self.file["episodes"].create_group(str(self.num_episodes))
        for key in self.config.camera_keys:
            frames = np.asarray(observations[key], dtype=np.uint8)
            if frames.shape[0] != num_transitions + 1:
                raise ValueError(
                    f"{key}: expected {num_transitions + 1} frames (T+1), got {frames.shape[0]}."
                )
            group.create_dataset(key, data=frames, compression=self.compression)
        proprio = np.asarray(observations["proprio"], dtype=np.float32)
        if proprio.shape[0] != num_transitions + 1:
            raise ValueError(
                f"proprio: expected {num_transitions + 1} rows (T+1), got {proprio.shape[0]}."
            )
        group.create_dataset("proprio", data=proprio, compression=self.compression)
        group.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
        group.create_dataset("reward", data=np.asarray(rewards, dtype=np.float32).reshape(-1))
        group.create_dataset("done", data=np.asarray(dones, dtype=bool).reshape(-1))
        group.create_dataset("truncated", data=np.asarray(truncated, dtype=bool).reshape(-1))
        group.attrs["success"] = bool(success)
        group.attrs["num_transitions"] = num_transitions
        self.num_episodes += 1

    def close(self) -> None:
        self.file["meta"].attrs["num_episodes"] = self.num_episodes
        self.file.close()

    def __enter__(self) -> "PixelEpisodeWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class OfflineImageDataset:
    """Frozen prior data for RLPD's offline half of the batch: uint8 frames on
    CPU with the same ``o``/``o'`` index sharing as the online buffer."""

    def __init__(
        self,
        path: str | Path,
        config: PixelObsConfig,
        *,
        expected_action_dim: int,
        expected_proprio_dim: int,
        expected_control_mode: str | None = None,
        success_only: bool = False,
        device: torch.device | str = "cpu",
    ):
        import h5py

        self.path = Path(path).expanduser()
        self.config = config
        self.storage_device = torch.device(device)
        self.camera_keys = config.camera_keys

        frames: dict[str, list[np.ndarray]] = {key: [] for key in self.camera_keys}
        proprio_frames: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        truncateds: list[np.ndarray] = []
        obs_ids: list[np.ndarray] = []
        next_ids: list[np.ndarray] = []
        self.num_episodes = 0
        self.num_success_episodes = 0
        # Per-episode transition counts, in load order (the delta-action
        # rewrite needs the episode starts).
        episode_transition_counts: list[int] = []
        boundaries = 0
        boundaries_with_done = 0
        frame_offset = 0

        with h5py.File(self.path, "r") as file:
            meta = dict(file["meta"].attrs)
            self.camera_config_hash = str(meta.get("camera_config_hash", ""))
            self.control_mode = str(meta.get("control_mode", ""))
            self.obs_version = str(meta.get("obs_version", ""))
            self.done_convention = str(meta.get("done_convention", ""))
            self.env_id = str(meta.get("env_id", ""))

            expected_hash = config.config_hash()
            if self.camera_config_hash != expected_hash:
                raise RuntimeError(
                    "camera_config_hash mismatch: dataset "
                    f"{self.camera_config_hash!r} vs current config {expected_hash!r}. The "
                    "offline and online visual pipelines must be bit-identical; re-render "
                    "the dataset or restore the config it was rendered with.\n"
                    f"current config: {config.summary()}"
                )
            if expected_control_mode is not None and self.control_mode != expected_control_mode:
                raise RuntimeError(
                    f"control_mode mismatch: dataset {self.control_mode!r} vs env "
                    f"{expected_control_mode!r}. Recorded actions would mean something else."
                )
            if self.done_convention != DONE_CONVENTION:
                raise RuntimeError(
                    f"done_convention mismatch: dataset {self.done_convention!r} vs online "
                    f"{DONE_CONVENTION!r}."
                )

            for name in sorted(file["episodes"], key=int):
                group = file["episodes"][name]
                success = bool(group.attrs.get("success", False))
                if success_only and not success:
                    continue
                episode_actions = np.asarray(group["action"], dtype=np.float32)
                num_transitions = int(episode_actions.shape[0])
                if num_transitions == 0:
                    continue
                for key in self.camera_keys:
                    frames[key].append(np.asarray(group[key], dtype=np.uint8))
                proprio_frames.append(np.asarray(group["proprio"], dtype=np.float32))
                actions.append(episode_actions)
                rewards.append(np.asarray(group["reward"], dtype=np.float32).reshape(-1, 1))
                episode_dones = np.asarray(group["done"], dtype=np.float32).reshape(-1, 1)
                dones.append(episode_dones)
                episode_truncated = np.asarray(group["truncated"], dtype=bool).reshape(-1)
                truncateds.append(episode_truncated.astype(np.float32).reshape(-1, 1))
                obs_ids.append(frame_offset + np.arange(num_transitions, dtype=np.int64))
                next_ids.append(frame_offset + np.arange(1, num_transitions + 1, dtype=np.int64))
                frame_offset += num_transitions + 1
                boundary = episode_truncated | episode_dones.reshape(-1).astype(bool)
                boundaries += int(boundary.sum())
                boundaries_with_done += int(
                    (boundary & episode_dones.reshape(-1).astype(bool)).sum()
                )
                # Online, `done = terminated = info["success"]`, so an episode's
                # final transition is done exactly when the episode succeeded.
                final_done = bool(episode_dones.reshape(-1)[-1])
                if final_done != success:
                    raise RuntimeError(
                        f"Episode {name} of {self.path.name} ends with done="
                        f"{int(final_done)} but success={int(success)}. The dataset's "
                        "`done` does not mean `terminated on success`, which is the "
                        f"convention online collection uses ({DONE_CONVENTION!r}); "
                        "mixing the two teaches the critic two terminal semantics "
                        "for the same state."
                    )
                episode_transition_counts.append(num_transitions)
                self.num_episodes += 1
                self.num_success_episodes += int(success)

        if self.num_episodes == 0:
            raise RuntimeError(f"No usable episodes in {self.path} (success_only={success_only}).")

        self.frames = {
            key: torch.from_numpy(np.concatenate(frames[key], axis=0)).to(self.storage_device)
            for key in self.camera_keys
        }
        self.proprio_frames = torch.from_numpy(np.concatenate(proprio_frames, axis=0)).to(
            self.storage_device
        )
        self.actions = torch.from_numpy(np.concatenate(actions, axis=0)).to(self.storage_device)
        self.rewards = torch.from_numpy(np.concatenate(rewards, axis=0)).to(self.storage_device)
        self.dones = torch.from_numpy(np.concatenate(dones, axis=0)).to(self.storage_device)
        # Kept alongside `dones` so the trainer can restore the bootstrap on
        # exactly the time-limit rows, as the state path does.
        self.truncated = torch.from_numpy(np.concatenate(truncateds, axis=0)).to(
            self.storage_device
        )
        self.obs_frame_id = torch.from_numpy(np.concatenate(obs_ids, axis=0)).to(
            self.storage_device
        )
        self.next_frame_id = torch.from_numpy(np.concatenate(next_ids, axis=0)).to(
            self.storage_device
        )
        self.episode_boundaries = boundaries
        self.episode_boundaries_with_done = boundaries_with_done
        self.episode_transition_counts = list(episode_transition_counts)

        if self.actions.shape[1] != expected_action_dim:
            raise RuntimeError(
                f"Dataset action dim {self.actions.shape[1]} != env action dim {expected_action_dim}."
            )
        if self.proprio_frames.shape[1] != expected_proprio_dim:
            raise RuntimeError(
                f"Dataset proprio dim {self.proprio_frames.shape[1]} != "
                f"env proprio dim {expected_proprio_dim}."
            )
        height, width, channels = config.image_shape
        for key in self.camera_keys:
            if tuple(self.frames[key].shape[1:]) != (height, width, channels):
                raise RuntimeError(
                    f"Dataset {key} shape {tuple(self.frames[key].shape[1:])} != "
                    f"config {(height, width, channels)}."
                )

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @property
    def terminal_done_fraction(self) -> float:
        if self.episode_boundaries == 0:
            return float("nan")
        return self.episode_boundaries_with_done / self.episode_boundaries

    def _gather(self, frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        obs = {key: self.frames[key][frame_ids] for key in self.camera_keys}
        obs["proprio"] = self.proprio_frames[frame_ids]
        return obs

    def sample(self, batch_size: int) -> dict[str, object]:
        indices = torch.randint(0, len(self), (batch_size,), device=self.storage_device)
        return {
            "observations": self._gather(self.obs_frame_id[indices]),
            "next_observations": self._gather(self.next_frame_id[indices]),
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "is_offline": torch.ones((batch_size, 1), device=self.storage_device),
        }

    def summary(self) -> str:
        return (
            f"OfflineImageDataset({self.path.name}): {len(self)} transitions, "
            f"{self.num_episodes} episodes ({self.num_success_episodes} successful), "
            f"hash={self.camera_config_hash}, control_mode={self.control_mode}, "
            f"terminal_done_fraction={self.terminal_done_fraction:.3f}"
        )


def image_demo_init_action(path: str | Path) -> np.ndarray:
    """Mean first-frame action across a rendered image dataset, the ``a_{-1}``
    seed (image analogue of :func:`utils.delta_action.demo_init_action`). Read
    from the file directly because the delta wrapper must exist before the
    dataset can be loaded.
    """
    import h5py

    first_actions: list[np.ndarray] = []
    with h5py.File(Path(path).expanduser(), "r") as file:
        for name in sorted(file["episodes"], key=int):
            actions = file["episodes"][name]["action"]
            if actions.shape[0] > 0:
                first_actions.append(np.asarray(actions[0], dtype=np.float32))
    if not first_actions:
        raise RuntimeError(f"No episodes with actions in {path}.")
    return np.mean(np.stack(first_actions, axis=0), axis=0)

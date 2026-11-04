"""Image replay buffer for pixel-based RL.

Frames are stored as uint8 on the CPU in their own ring; every transition stores
two frame ids (``o`` and ``o'``), so consecutive steps of one sub-env share a
frame and ``T`` steps store ``T + 1`` frames. Frames and transitions are written
monotonically, so the samplable transitions form a contiguous suffix tracked by
``valid_start``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch


class ImageReplayBuffer:
    """FIFO buffer over dict observations, with optional 50/50 offline mixing."""

    def __init__(
        self,
        capacity: int,
        camera_keys: Sequence[str],
        image_shape: tuple[int, int, int],
        proprio_dim: int,
        action_dim: int,
        *,
        device: torch.device | str = "cpu",
        offline: Optional["object"] = None,
        offline_mix_fraction: float = 0.5,
        frame_slack: float = 0.02,
    ):
        self.capacity = int(capacity)
        self.camera_keys = tuple(camera_keys)
        self.image_shape = tuple(int(dim) for dim in image_shape)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        # Storage device for the buffer itself; batches are moved at sample time.
        self.storage_device = torch.device(device)

        # One extra frame per episode plus a small margin.
        self.frame_capacity = self.capacity + max(int(self.capacity * frame_slack), 64)

        height, width, channels = self.image_shape
        self.frames = {
            key: torch.zeros(
                (self.frame_capacity, height, width, channels),
                dtype=torch.uint8,
                device=self.storage_device,
            )
            for key in self.camera_keys
        }
        self.proprio_frames = torch.zeros(
            (self.frame_capacity, self.proprio_dim),
            dtype=torch.float32,
            device=self.storage_device,
        )

        self.actions = torch.zeros(
            (self.capacity, self.action_dim), dtype=torch.float32, device=self.storage_device
        )
        self.rewards = torch.zeros(
            (self.capacity, 1), dtype=torch.float32, device=self.storage_device
        )
        self.dones = torch.zeros(
            (self.capacity, 1), dtype=torch.float32, device=self.storage_device
        )
        self.obs_frame_id = torch.zeros(
            (self.capacity,), dtype=torch.int64, device=self.storage_device
        )
        self.next_frame_id = torch.zeros(
            (self.capacity,), dtype=torch.int64, device=self.storage_device
        )

        self.frame_count = 0  # frames ever written
        self.trans_count = 0  # transitions ever written
        self.valid_start = 0  # oldest still-samplable transition index
        # Frame id holding each sub-env's current observation; -1 right after a
        # reset, when the next `add` must store the fresh observation itself.
        self.current_frame_id: torch.Tensor | None = None

        # `done` bookkeeping for the done-convention check.
        self.episode_boundaries = 0
        self.episode_boundaries_with_done = 0

        self.offline = offline
        self.offline_mix_fraction = float(offline_mix_fraction) if offline is not None else 0.0
        if not 0.0 <= self.offline_mix_fraction <= 1.0:
            raise ValueError(
                f"`offline_mix_fraction` must be in [0, 1], got {self.offline_mix_fraction}."
            )

    # ---- capacity accounting -------------------------------------------------

    @property
    def size(self) -> int:
        return int(self.trans_count - self.valid_start)

    def storage_bytes(self) -> int:
        total = sum(frame.numel() * frame.element_size() for frame in self.frames.values())
        total += self.proprio_frames.numel() * self.proprio_frames.element_size()
        for tensor in (self.actions, self.rewards, self.dones, self.obs_frame_id, self.next_frame_id):
            total += tensor.numel() * tensor.element_size()
        return int(total)

    @property
    def terminal_done_fraction(self) -> float:
        """Share of episode-final transitions stored with ``done=1``."""
        if self.episode_boundaries == 0:
            return float("nan")
        return self.episode_boundaries_with_done / self.episode_boundaries

    # ---- writing -------------------------------------------------------------

    def _store_frames(self, obs: Mapping[str, torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
        """Append ``obs[rows]`` to the frame ring; return their global frame ids."""
        count = int(rows.numel())
        ids = torch.arange(
            self.frame_count, self.frame_count + count, device=self.storage_device
        )
        slots = ids % self.frame_capacity
        for key in self.camera_keys:
            source = obs[key].index_select(0, rows.to(obs[key].device))
            self.frames[key][slots] = source.to(self.storage_device, dtype=torch.uint8)
        proprio = obs["proprio"]
        self.proprio_frames[slots] = proprio.index_select(0, rows.to(proprio.device)).to(
            self.storage_device, dtype=torch.float32
        )
        self.frame_count += count
        return ids

    def add(
        self,
        obs: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: Mapping[str, torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Append one vectorized env step (``num_envs`` transitions).

        ``next_obs`` must be the *true* next observation (the terminal frame on a
        done step, not the auto-reset frame). ``terminated`` is what goes into the
        TD backup as ``done``; ``truncated`` only ends the frame chain.
        """
        num_envs = int(actions.shape[0])
        if self.current_frame_id is None:
            self.current_frame_id = torch.full(
                (num_envs,), -1, dtype=torch.int64, device=self.storage_device
            )
        elif self.current_frame_id.numel() != num_envs:
            raise ValueError(
                f"Buffer was primed with {self.current_frame_id.numel()} sub-envs, got {num_envs}."
            )

        # Only the first step of an episode has to store its own `o`; every other
        # step reuses the frame stored as the previous step's `o'`.
        fresh_rows = torch.nonzero(self.current_frame_id < 0, as_tuple=False).flatten()
        if fresh_rows.numel() > 0:
            self.current_frame_id[fresh_rows] = self._store_frames(obs, fresh_rows)
        obs_ids = self.current_frame_id.clone()

        all_rows = torch.arange(num_envs, device=self.storage_device)
        next_ids = self._store_frames(next_obs, all_rows)

        slots = (
            torch.arange(num_envs, device=self.storage_device) + self.trans_count
        ) % self.capacity
        self.actions[slots] = actions.reshape(num_envs, self.action_dim).to(
            self.storage_device, dtype=torch.float32
        )
        self.rewards[slots] = rewards.reshape(num_envs, 1).to(
            self.storage_device, dtype=torch.float32
        )
        terminated = terminated.reshape(num_envs).to(self.storage_device)
        truncated = truncated.reshape(num_envs).to(self.storage_device)
        self.dones[slots] = terminated.reshape(num_envs, 1).float()
        self.obs_frame_id[slots] = obs_ids
        self.next_frame_id[slots] = next_ids
        self.trans_count += num_envs

        boundary = terminated.bool() | truncated.bool()
        self.episode_boundaries += int(boundary.sum().item())
        self.episode_boundaries_with_done += int((boundary & terminated.bool()).sum().item())
        self.current_frame_id = torch.where(
            boundary, torch.full_like(next_ids, -1), next_ids
        )

        self._advance_valid_start()

    def _advance_valid_start(self) -> None:
        self.valid_start = max(self.valid_start, self.trans_count - self.capacity)
        oldest_alive_frame = self.frame_count - self.frame_capacity
        while self.valid_start < self.trans_count:
            slot = self.valid_start % self.capacity
            if int(self.obs_frame_id[slot].item()) >= oldest_alive_frame:
                break
            self.valid_start += 1

    # ---- sampling ------------------------------------------------------------

    def _gather(self, frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        slots = frame_ids % self.frame_capacity
        obs = {key: self.frames[key][slots] for key in self.camera_keys}
        obs["proprio"] = self.proprio_frames[slots]
        return obs

    def _sample_online(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = torch.randint(
            self.valid_start, self.trans_count, (batch_size,), device=self.storage_device
        )
        slots = indices % self.capacity
        return {
            "observations": self._gather(self.obs_frame_id[slots]),
            "next_observations": self._gather(self.next_frame_id[slots]),
            "actions": self.actions[slots],
            "rewards": self.rewards[slots],
            "dones": self.dones[slots],
            "is_offline": torch.zeros((batch_size, 1), device=self.storage_device),
        }

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> dict[str, object]:
        """RLPD symmetric sampling: ``offline_mix_fraction`` of the batch from the
        frozen demo dataset, the rest from online replay."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty image replay buffer.")
        device = torch.device(device) if device is not None else self.storage_device

        num_offline = (
            min(max(int(round(batch_size * self.offline_mix_fraction)), 0), batch_size)
            if self.offline is not None
            else 0
        )
        num_online = batch_size - num_offline

        pieces = []
        if num_online > 0:
            pieces.append(self._sample_online(num_online))
        if num_offline > 0:
            pieces.append(self.offline.sample(num_offline))

        batch = pieces[0] if len(pieces) == 1 else _concat_batches(pieces)
        return _batch_to_device(batch, device)


def _concat_batches(pieces: Sequence[dict[str, object]]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for key in ("observations", "next_observations"):
        merged[key] = {
            inner: torch.cat([piece[key][inner] for piece in pieces], dim=0)
            for inner in pieces[0][key]
        }
    for key in ("actions", "rewards", "dones", "is_offline"):
        merged[key] = torch.cat([piece[key] for piece in pieces], dim=0)
    return merged


def _batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                inner: tensor.to(device, non_blocking=True) for inner, tensor in value.items()
            }
        else:
            moved[key] = value.to(device, non_blocking=True)
    return moved


def shuffle_batch(batch: dict[str, object], permutation: torch.Tensor) -> dict[str, object]:
    """Reorder a batch (used to reshuffle a mega-batch before slicing it)."""
    shuffled: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            shuffled[key] = {inner: tensor[permutation] for inner, tensor in value.items()}
        else:
            shuffled[key] = value[permutation]
    return shuffled


def slice_batch(batch: dict[str, object], start: int, stop: int) -> dict[str, object]:
    sliced: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            sliced[key] = {inner: tensor[start:stop] for inner, tensor in value.items()}
        else:
            sliced[key] = value[start:stop]
    return sliced


def observation_grid(
    batch: Mapping[str, object],
    camera_keys: Sequence[str],
    *,
    rows: int = 4,
) -> Optional["torch.Tensor"]:
    """Tile a training batch's frames (after augmentation) into one HWC uint8
    image: one row per sample, cameras left to right, offline samples on top
    when the batch carries `is_offline`."""
    observations = batch["observations"]
    frames = [observations[key] for key in camera_keys]
    if not frames:
        return None
    batch_size = frames[0].shape[0]
    rows = max(1, min(int(rows), batch_size))

    is_offline = batch.get("is_offline")
    if is_offline is not None:
        flags = is_offline.reshape(-1) > 0.5
        offline_rows = torch.nonzero(flags, as_tuple=False).flatten()[: rows // 2]
        online_rows = torch.nonzero(~flags, as_tuple=False).flatten()[: rows - rows // 2]
        indices = torch.cat([offline_rows, online_rows])
    else:
        indices = torch.arange(rows, device=frames[0].device)
    if indices.numel() == 0:
        return None

    picked = [frame[indices].detach().to("cpu", torch.uint8) for frame in frames]
    # (R, H, W, C) per camera -> concat cameras on W, samples on H.
    return torch.cat([torch.cat(list(sample), dim=1) for sample in zip(*picked)], dim=0)

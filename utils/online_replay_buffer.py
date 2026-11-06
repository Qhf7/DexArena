"""FIFO replay buffer for online (off-policy) fine-tuning.

Collects transitions from a vectorized ManiSkill environment and serves
mini-batches with the same key layout the offline agents consume
(`observations / actions / rewards / next_observations / dones`).

Supports RLPD-style symmetric sampling: an optional fraction of every batch is
drawn from a frozen offline demo dataset (`OfflineReplayDataset`). Keeping demo
transitions in the critic updates is the main stabilizer against the early
Q-overestimation / "unlearning" drop that happens when an offline-pretrained
agent starts collecting its own (initially off-distribution) data.
"""

from __future__ import annotations

from typing import Optional

import torch

from utils.offline_replay_buffer import OfflineReplayDataset


_BATCH_KEYS = ("observations", "actions", "rewards", "next_observations", "dones")


class OnlineReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        *,
        offline: Optional[OfflineReplayDataset] = None,
        offline_mix_fraction: float = 0.5,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device

        self.observations = torch.zeros((self.capacity, self.obs_dim), device=device)
        self.actions = torch.zeros((self.capacity, self.action_dim), device=device)
        self.rewards = torch.zeros((self.capacity, 1), device=device)
        self.next_observations = torch.zeros((self.capacity, self.obs_dim), device=device)
        self.dones = torch.zeros((self.capacity, 1), device=device)

        self.ptr = 0
        self.size = 0

        self.offline = offline
        self.offline_mix_fraction = (
            float(offline_mix_fraction) if offline is not None else 0.0
        )
        if not 0.0 <= self.offline_mix_fraction <= 1.0:
            raise ValueError(
                f"`offline_mix_fraction` must be in [0, 1], got {self.offline_mix_fraction}."
            )

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Append one vectorized step (`num_envs` transitions) to the buffer."""
        n = int(obs.shape[0])
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        self.observations[idx] = obs.to(self.device, dtype=torch.float32)
        self.actions[idx] = actions.to(self.device, dtype=torch.float32)
        self.rewards[idx] = rewards.reshape(n, 1).to(self.device, dtype=torch.float32)
        self.next_observations[idx] = next_obs.to(self.device, dtype=torch.float32)
        self.dones[idx] = dones.reshape(n, 1).to(self.device, dtype=torch.float32)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    def _sample_online(self, batch_size: int) -> dict[str, torch.Tensor]:
        inds = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "observations": self.observations[inds],
            "actions": self.actions[inds],
            "rewards": self.rewards[inds],
            "next_observations": self.next_observations[inds],
            "dones": self.dones[inds],
        }

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty online replay buffer.")
        device = device or self.device

        if self.offline is not None and self.offline_mix_fraction > 0.0:
            num_offline = int(round(batch_size * self.offline_mix_fraction))
            num_offline = min(max(num_offline, 0), batch_size)
        else:
            num_offline = 0
        num_online = batch_size - num_offline

        pieces: list[dict[str, torch.Tensor]] = []
        # `bc_mask`: rows the BC anchor may regress onto (offline rows from
        # successful episodes).
        masks: list[torch.Tensor] = []
        if num_online > 0:
            pieces.append(self._sample_online(num_online))
            masks.append(torch.zeros((num_online, 1), device=self.device))
        if num_offline > 0:
            off = self.offline.sample(num_offline, self.device, include_extra=False)
            pieces.append({key: off[key] for key in _BATCH_KEYS})
            masks.append(off["episode_success"])

        if len(pieces) == 1:
            batch = dict(pieces[0])
        else:
            batch = {
                key: torch.cat([piece[key] for piece in pieces], dim=0)
                for key in _BATCH_KEYS
            }
        batch["bc_mask"] = torch.cat(masks, dim=0) if len(masks) > 1 else masks[0]
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

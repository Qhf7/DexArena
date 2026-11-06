from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from utils.demo_dataset import (
    maybe_import_pyarrow,
    require_strict_recording_certificate,
    resolve_lerobot_dataset_roots,
)
from utils.paths import demo_path


def _feature_dim(feature: dict) -> int:
    shape = tuple(int(dim) for dim in feature.get("shape", ()))
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _stack_column(values, dtype: np.dtype) -> np.ndarray:
    return np.asarray([np.asarray(value).reshape(-1) for value in values], dtype=dtype)


@dataclass
class OfflineReplayDataset:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    next_actions: torch.Tensor
    dones: torch.Tensor
    truncated: torch.Tensor
    episode_indices: torch.Tensor
    source_paths: list[Path]
    is_extra: torch.Tensor
    # Discounted return-to-go per transition; only populated when the loader is
    # called with a `gamma` (used as the SARSA/FQE Monte-Carlo ground truth).
    returns_to_go: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        num_samples = int(self.observations.shape[0])
        tensors = {
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "next_actions": self.next_actions,
            "dones": self.dones,
            "truncated": self.truncated,
            "episode_indices": self.episode_indices,
            "is_extra": self.is_extra,
        }
        if self.returns_to_go is not None:
            tensors["returns_to_go"] = self.returns_to_go
        for name, tensor in tensors.items():
            if int(tensor.shape[0]) != num_samples:
                raise ValueError(
                    f"Replay tensor `{name}` has {tensor.shape[0]} samples; expected {num_samples}."
                )
        if self.rewards.ndim != 2 or self.rewards.shape[1] != 1:
            raise ValueError(f"`rewards` must have shape [N, 1], got {tuple(self.rewards.shape)}.")
        if self.dones.ndim != 2 or self.dones.shape[1] != 1:
            raise ValueError(f"`dones` must have shape [N, 1], got {tuple(self.dones.shape)}.")

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def episode_success(self) -> torch.Tensor:
        """[N, 1] flag: did this row's episode ever collect reward? Keeps the
        BC anchor off failed demonstrations. Cached on first use."""
        cached = getattr(self, "_episode_success", None)
        if cached is not None:
            return cached
        episodes = self.episode_indices
        positive = (self.rewards.reshape(-1) > 0).float()
        per_episode = torch.zeros(
            int(episodes.max().item()) + 1 if episodes.numel() else 0
        ).scatter_reduce(0, episodes, positive, reduce="amax", include_self=True)
        cached = per_episode[episodes].reshape(-1, 1)
        self._episode_success = cached
        return cached

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def num_episodes(self) -> int:
        return int(torch.unique(self.episode_indices).numel())

    def observation_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.observations.mean(dim=0)
        std = self.observations.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, std

    def action_stats(self, *, successful_only: bool = True) -> torch.Tensor:
        """Per-dimension std of the demonstrated actions (successful episodes by
        default), floored at a tenth of the median. Used for
        ``target_entropy_demo_scale``; not by the BC anchor."""
        actions = self.actions
        if successful_only:
            mask = self.episode_success.reshape(-1) > 0
            if bool(mask.any()):
                actions = actions[mask]
        std = actions.std(dim=0, unbiased=False)
        return std.clamp_min(std.median() * 0.1).clamp_min(1e-6)

    @property
    def num_extra_transitions(self) -> int:
        return int(self.is_extra.sum().item())

    @property
    def has_extra_transitions(self) -> bool:
        return self.num_extra_transitions > 0

    def _sample_indices(
        self,
        batch_size: int,
        *,
        include_extra: bool = True,
        extra_fraction: Optional[float] = None,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

        if not include_extra:
            pool = torch.nonzero(~self.is_extra, as_tuple=False).flatten()
            if pool.numel() == 0:
                raise RuntimeError("Cannot sample demo-only batch: replay has no demo transitions.")
            return pool[torch.randint(0, pool.numel(), (batch_size,))]

        if extra_fraction is None or not self.has_extra_transitions:
            return torch.randint(0, len(self), (batch_size,))

        extra_fraction = float(extra_fraction)
        if not 0.0 <= extra_fraction <= 1.0:
            raise ValueError(f"`extra_fraction` must be in [0, 1], got {extra_fraction}.")

        extra_pool = torch.nonzero(self.is_extra, as_tuple=False).flatten()
        demo_pool = torch.nonzero(~self.is_extra, as_tuple=False).flatten()
        if extra_pool.numel() == 0:
            num_extra = 0
        elif demo_pool.numel() == 0:
            num_extra = batch_size
        else:
            num_extra = int(round(batch_size * extra_fraction))
        num_extra = min(max(num_extra, 0), batch_size)
        num_demo = batch_size - num_extra

        pieces = []
        if num_demo > 0:
            if demo_pool.numel() == 0:
                raise RuntimeError("Cannot sample demo portion: replay has no demo transitions.")
            pieces.append(demo_pool[torch.randint(0, demo_pool.numel(), (num_demo,))])
        if num_extra > 0:
            pieces.append(extra_pool[torch.randint(0, extra_pool.numel(), (num_extra,))])
        indices = torch.cat(pieces, dim=0)
        return indices[torch.randperm(indices.numel())]

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        *,
        include_extra: bool = True,
        extra_fraction: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        indices = self._sample_indices(
            batch_size,
            include_extra=include_extra,
            extra_fraction=extra_fraction,
        )
        batch = {
            "observations": self.observations[indices].to(device=device, non_blocking=True),
            "actions": self.actions[indices].to(device=device, non_blocking=True),
            "rewards": self.rewards[indices].to(device=device, non_blocking=True),
            "next_observations": self.next_observations[indices].to(
                device=device, non_blocking=True
            ),
            "next_actions": self.next_actions[indices].to(device=device, non_blocking=True),
            "dones": self.dones[indices].to(device=device, non_blocking=True),
            "truncated": self.truncated[indices].to(device=device, non_blocking=True),
            "is_extra": self.is_extra[indices].to(device=device, non_blocking=True),
            "episode_success": self.episode_success[indices].to(
                device=device, non_blocking=True
            ),
        }
        if self.returns_to_go is not None:
            batch["returns_to_go"] = self.returns_to_go[indices].to(
                device=device, non_blocking=True
            )
        return batch


def _load_extra_npz_transitions(
    npz_path: str,
    expected_obs_dim: Optional[int],
    expected_action_dim: Optional[int],
) -> dict[str, np.ndarray]:
    """Load explicit (s, a, r, s', done) transitions from an .npz file.

    Used for counterfactual-branch transitions (one off-trajectory env step per
    state): these have a genuine, explicitly-stored next state but NO valid next
    action, so they are only ever consumed by a V-bootstrap target (IQL/AWR).
    `next_actions` is set to a placeholder (the action itself) and must not be
    read by a SARSA/FQE consumer.
    """
    npz_path = demo_path(npz_path)
    data = np.load(npz_path)
    required = {"observations", "actions", "rewards", "next_observations", "dones"}
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError(f"`{npz_path}` is missing transition fields: {sorted(missing)}.")
    obs = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    next_obs = np.asarray(data["next_observations"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
    dones = np.asarray(data["dones"], dtype=np.float32).reshape(-1, 1)
    if "truncated" in data.files:
        truncated = np.asarray(data["truncated"], dtype=np.float32).reshape(-1, 1)
    else:
        truncated = (1.0 - dones).astype(np.float32)
    n = obs.shape[0]
    if any(arr.shape[0] != n for arr in (actions, next_obs, rewards, dones, truncated)):
        raise RuntimeError(f"`{npz_path}` transition arrays have mismatched lengths.")
    if expected_obs_dim is not None and obs.shape[1] != int(expected_obs_dim):
        raise RuntimeError(
            f"`{npz_path}` obs_dim={obs.shape[1]} != expected {expected_obs_dim}."
        )
    if expected_action_dim is not None and actions.shape[1] != int(expected_action_dim):
        raise RuntimeError(
            f"`{npz_path}` action_dim={actions.shape[1]} != expected {expected_action_dim}."
        )
    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_obs,
        "next_actions": actions.copy(),  # placeholder: branches have no valid a'
        "dones": dones,
        "truncated": truncated,
    }


def load_lerobot_replay_dataset(
    data_path: str,
    *,
    expected_obs_dim: Optional[int] = None,
    expected_action_dim: Optional[int] = None,
    gamma: Optional[float] = None,
    extra_npz_paths: Optional[Sequence[str]] = None,
    success_only: bool = False,
    sparse_reward: bool = False,
) -> OfflineReplayDataset:
    dataset_roots = resolve_lerobot_dataset_roots(data_path)
    if not dataset_roots:
        raise RuntimeError(f"No LeRobot dataset roots found from path: {data_path}")

    pq = maybe_import_pyarrow()
    obs_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    reward_batches: list[np.ndarray] = []
    next_obs_batches: list[np.ndarray] = []
    next_action_batches: list[np.ndarray] = []
    done_batches: list[np.ndarray] = []
    truncated_batches: list[np.ndarray] = []
    episode_batches: list[np.ndarray] = []
    is_extra_batches: list[np.ndarray] = []
    rtg_batches: list[np.ndarray] = []
    matched_paths: list[Path] = []
    global_episode_offset = 0
    num_dropped_failed = 0
    num_dropped_tail = 0

    strict_required_features = {
        "observation.state",
        "next.observation.state",
        "action",
        "next.reward",
        "next.done",
        "next.terminated",
        "next.truncated",
        "next.recording_boundary",
        "next.success",
    }
    strict_columns = [
        "episode_index",
        "frame_index",
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
    legacy_required_features = {
        "observation.state",
        "action",
        "next.reward",
        "next.done",
        "next.truncated",
        "next.success",
    }
    legacy_columns = [
        "episode_index",
        "frame_index",
        "observation.state",
        "action",
        "next.reward",
        "next.done",
        "next.truncated",
        "next.success",
    ]

    for dataset_root in dataset_roots:
        info = json.loads((dataset_root / "meta" / "info.json").read_text())
        features = info.get("features", {})
        strict_dataset = "episode.source_hash" in features
        require_strict_recording_certificate(dataset_root, features)
        required_features = (
            strict_required_features if strict_dataset else legacy_required_features
        )
        columns = strict_columns if strict_dataset else legacy_columns
        missing_features = required_features.difference(features)
        if missing_features:
            raise RuntimeError(
                f"LeRobot dataset `{dataset_root}` is missing CQL fields: "
                f"{sorted(missing_features)}."
            )

        obs_dim = _feature_dim(features["observation.state"])
        action_dim = _feature_dim(features["action"])
        if expected_obs_dim is not None and obs_dim != int(expected_obs_dim):
            raise RuntimeError(
                f"Dataset `{dataset_root}` has obs_dim={obs_dim}; expected {expected_obs_dim}."
            )
        if expected_action_dim is not None and action_dim != int(expected_action_dim):
            raise RuntimeError(
                f"Dataset `{dataset_root}` has action_dim={action_dim}; "
                f"expected {expected_action_dim}."
            )

        data_paths = sorted((dataset_root / "data").rglob("*.parquet"))
        if not data_paths:
            continue
        data = pq.read_table(data_paths, columns=columns).to_pydict()

        raw_episode_indices = np.asarray(data["episode_index"], dtype=np.int64).reshape(-1)
        raw_frame_indices = np.asarray(data["frame_index"], dtype=np.int64).reshape(-1)
        order = np.lexsort((raw_frame_indices, raw_episode_indices))
        episode_indices = raw_episode_indices[order]
        observations = _stack_column(data["observation.state"], np.float32)[order]
        actions = _stack_column(data["action"], np.float32)[order]
        rewards = _stack_column(data["next.reward"], np.float32)[order]
        dones = _stack_column(data["next.done"], np.bool_)[order]
        truncated = _stack_column(data["next.truncated"], np.bool_)[order]
        success = _stack_column(data["next.success"], np.bool_)[order]
        if strict_dataset:
            next_observations = _stack_column(
                data["next.observation.state"], np.float32
            )[order]
            terminated = _stack_column(data["next.terminated"], np.bool_)[order]
            recording_boundary = _stack_column(
                data["next.recording_boundary"], np.bool_
            )[order]
            if not np.array_equal(dones, np.logical_or(terminated, truncated)):
                raise RuntimeError(
                    f"Dataset `{dataset_root}` has inconsistent "
                    "done/terminated/truncated flags."
                )
        else:
            next_observations = None
            recording_boundary = None

        root_matched = False
        for local_episode_id in np.unique(episode_indices):
            mask = episode_indices == local_episode_id
            episode_obs = observations[mask]
            episode_actions = actions[mask]
            episode_rewards = rewards[mask]
            episode_truncated = truncated[mask]
            episode_success = success[mask].copy()
            if strict_dataset:
                assert next_observations is not None
                assert recording_boundary is not None
                next_episode_obs = next_observations[mask]
                # Stop training bootstrap at a recording boundary without
                # rewriting the literal environment terminal flags.
                episode_dones = np.logical_or(
                    dones[mask], recording_boundary[mask]
                )
            else:
                # Legacy LeRobot rows have no exact post-action observation.
                # Preserve the historical shift/drop behavior only for them.
                next_episode_obs = np.empty_like(episode_obs)
                if episode_obs.shape[0] > 1:
                    next_episode_obs[:-1] = episode_obs[1:]
                next_episode_obs[-1] = episode_obs[-1]
                episode_dones = episode_success.copy()
            if episode_obs.shape[0] == 0:
                continue

            # Demo selection: keep only successful episodes (any frame reaches
            # `next.success`). Failed/timeout demos never enter any batch.
            if success_only and not bool(episode_success.any()):
                num_dropped_failed += 1
                continue

            # a'[i] = a[i+1]. A strict boundary or legacy tail handling masks
            # the final placeholder so it cannot leak into a target.
            next_episode_actions = np.empty_like(episode_actions)
            if episode_actions.shape[0] > 1:
                next_episode_actions[:-1] = episode_actions[1:]
            next_episode_actions[-1] = episode_actions[-1]
            if not strict_dataset and not bool(episode_dones[-1]):
                episode_obs = episode_obs[:-1]
                episode_actions = episode_actions[:-1]
                episode_rewards = episode_rewards[:-1]
                next_episode_obs = next_episode_obs[:-1]
                next_episode_actions = next_episode_actions[:-1]
                episode_truncated = episode_truncated[:-1]
                episode_success = episode_success[:-1]
                episode_dones = episode_dones[:-1]
                num_dropped_tail += 1
                if episode_obs.shape[0] == 0:
                    continue
                episode_dones[-1] = True


            obs_batches.append(episode_obs)
            action_batches.append(episode_actions)
            reward_batches.append(episode_rewards)
            next_obs_batches.append(next_episode_obs)
            next_action_batches.append(next_episode_actions)
            done_batches.append(episode_dones)
            truncated_batches.append(episode_truncated)
            episode_batches.append(
                np.full(
                    episode_obs.shape[0],
                    global_episode_offset,
                    dtype=np.int64,
                )
            )
            is_extra_batches.append(np.zeros(episode_obs.shape[0], dtype=np.bool_))

            if gamma is not None:
                # G_t = r_t + gamma * (0 if done_t else G_{t+1}); the exact value
                # of the behavior policy that SARSA/FQE should converge to.
                # When `sparse_reward`, use the terminal success signal as the
                # per-step reward (matches env `compute_sparse_reward = success`),
                # so RTG becomes gamma^(steps-to-success) — the quantity an online
                # PPO critic under sparse reward is meant to fit.
                rtg_rewards = (
                    episode_success.astype(np.float32).reshape(-1, 1)
                    if sparse_reward
                    else episode_rewards
                )
                episode_rtg = np.zeros_like(rtg_rewards)
                running = 0.0
                for i in range(rtg_rewards.shape[0] - 1, -1, -1):
                    bootstrap = 0.0 if bool(episode_dones[i]) else running
                    running = float(rtg_rewards[i, 0]) + gamma * bootstrap
                    episode_rtg[i, 0] = running
                rtg_batches.append(episode_rtg)

            global_episode_offset += 1
            root_matched = True

        if root_matched:
            matched_paths.append(dataset_root)

    num_extra = 0
    for npz_path in extra_npz_paths or []:
        extra = _load_extra_npz_transitions(npz_path, expected_obs_dim, expected_action_dim)
        n = extra["observations"].shape[0]
        obs_batches.append(extra["observations"])
        action_batches.append(extra["actions"])
        reward_batches.append(extra["rewards"])
        next_obs_batches.append(extra["next_observations"])
        next_action_batches.append(extra["next_actions"])
        done_batches.append(extra["dones"])
        truncated_batches.append(extra["truncated"])
        # Each branch is an independent single-step "episode"; one shared id per file
        # keeps the episode count sane (episode indices are not used during sampling).
        episode_batches.append(np.full(n, global_episode_offset, dtype=np.int64))
        is_extra_batches.append(np.ones(n, dtype=np.bool_))
        global_episode_offset += 1
        if gamma is not None:
            # Single-step branch: return-to-go is just its immediate reward.
            rtg_batches.append(extra["rewards"].astype(np.float32))
        num_extra += n
        print(f"[replay] extra branch transitions from `{npz_path}`: {n}")

    if not obs_batches:
        raise RuntimeError(f"No replay transitions were loaded from `{data_path}`.")

    observations = torch.from_numpy(np.concatenate(obs_batches, axis=0)).float()
    actions = torch.from_numpy(np.concatenate(action_batches, axis=0)).float()
    rewards = torch.from_numpy(np.concatenate(reward_batches, axis=0)).float()
    next_observations = torch.from_numpy(np.concatenate(next_obs_batches, axis=0)).float()
    next_actions = torch.from_numpy(np.concatenate(next_action_batches, axis=0)).float()
    dones = torch.from_numpy(np.concatenate(done_batches, axis=0)).float()
    truncated = torch.from_numpy(np.concatenate(truncated_batches, axis=0)).float()
    episode_indices_tensor = torch.from_numpy(np.concatenate(episode_batches, axis=0)).long()
    is_extra = torch.from_numpy(np.concatenate(is_extra_batches, axis=0)).bool()
    returns_to_go = (
        torch.from_numpy(np.concatenate(rtg_batches, axis=0)).float()
        if gamma is not None and rtg_batches
        else None
    )

    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise RuntimeError("Replay dataset contains non-finite observations or actions.")
    if not torch.isfinite(rewards).all():
        raise RuntimeError("Replay dataset contains non-finite rewards.")

    num_terminal = int(dones.sum().item())
    print(
        f"[replay] episodes={global_episode_offset} transitions={observations.shape[0]} "
        f"terminal_or_boundary={num_terminal} "
        f"dropped_legacy_tails={num_dropped_tail} "
        f"dropped_failed_episodes={num_dropped_failed} "
        f"extra_branch_transitions={num_extra}"
    )

    return OfflineReplayDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        next_actions=next_actions,
        dones=dones,
        truncated=truncated,
        episode_indices=episode_indices_tensor,
        source_paths=matched_paths,
        is_extra=is_extra,
        returns_to_go=returns_to_go,
    )

"""Residual (delta) action parameterization for absolute-joint-position control.

``pd_joint_pos`` on the Sharpa agents takes absolute joint targets in radians,
so a zero-initialized SAC actor emits the midpoint of every joint range and its
unit exploration noise spans radians. Here the policy emits ``u`` in ``[-1, 1]``
and the env receives ``a_t = clip(a_{t-1} + u * delta, joint_low, joint_high)``;
``u = 0`` holds the previous command.

Nothing on disk changes: datasets keep their absolute actions,
``to_delta_dataset`` derives ``u`` in memory and ``DeltaActionWrapper`` converts
back before the action reaches the env.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from utils.demo_dataset import maybe_import_pyarrow, resolve_lerobot_dataset_roots
from utils.offline_replay_buffer import OfflineReplayDataset


# Chosen off a fidelity/exploration sweep on the cabinet_pour_door_open demos.
DEFAULT_DELTA = 0.2

def demo_init_action(
    data_path: str, expected_action_dim: Optional[int] = None
) -> np.ndarray:
    """Mean first-frame action across demo episodes, the ``a_{-1}`` seed.

    Not the reset qpos: the hand PD carries a standing target-to-position
    offset, so seeding there would open every episode by closing a gap no demo
    contains.
    """
    pq = maybe_import_pyarrow()
    roots = resolve_lerobot_dataset_roots(data_path)
    if not roots:
        raise RuntimeError(f"No LeRobot dataset roots found from path: {data_path}")

    first_actions: list[np.ndarray] = []
    for root in roots:
        paths = sorted((root / "data").rglob("*.parquet"))
        if not paths:
            continue
        table = pq.read_table(
            paths, columns=["episode_index", "frame_index", "action"]
        ).to_pydict()
        episodes = np.asarray(table["episode_index"], dtype=np.int64)
        frames = np.asarray(table["frame_index"], dtype=np.int64)
        actions = np.asarray(
            [np.asarray(a).reshape(-1) for a in table["action"]], dtype=np.float32
        )
        for episode in np.unique(episodes):
            mask = np.nonzero(episodes == episode)[0]
            first_actions.append(actions[mask[np.argmin(frames[mask])]])

    if not first_actions:
        raise RuntimeError(f"No demo transitions found under `{data_path}`.")

    init_action = np.mean(np.stack(first_actions), axis=0).astype(np.float32)
    if expected_action_dim is not None and init_action.shape[0] != int(
        expected_action_dim
    ):
        raise RuntimeError(
            f"Demo action_dim={init_action.shape[0]} != expected "
            f"{expected_action_dim}; dataset and control_mode disagree."
        )
    return init_action


class DeltaActionWrapper(gym.Wrapper):
    """Expose a ``[-1, 1]`` delta action space over an absolute-position env.

    Sits outside ``ManiSkillVectorEnv`` so auto-reset stays the vector env's
    business and ``RecordEpisode`` still sees absolute joint targets.
    """

    def __init__(
        self,
        env,
        delta: float,
        init_action: np.ndarray,
        *,
        augment_obs: bool = True,
    ):
        super().__init__(env)
        # `augment_obs=False` is the image path: the caller appends the command
        # block to proprio itself.
        self.augment_obs = bool(augment_obs)
        if float(delta) <= 0.0:
            raise ValueError(f"`delta` must be positive, got {delta}.")

        device = getattr(env, "device", None)
        if device is None:
            device = env.unwrapped.device
        low = torch.as_tensor(
            env.single_action_space.low, dtype=torch.float32, device=device
        )
        high = torch.as_tensor(
            env.single_action_space.high, dtype=torch.float32, device=device
        )
        action_dim = int(low.numel())
        init = torch.as_tensor(init_action, dtype=torch.float32, device=device)
        if init.numel() != action_dim:
            raise ValueError(
                f"`init_action` has {init.numel()} dims, env action space has "
                f"{action_dim}."
            )

        self.low, self.high = low, high
        # The env's absolute bounds; `single_action_space` advertises the delta space.
        self.absolute_action_low = low.detach().cpu().numpy()
        self.absolute_action_high = high.detach().cpu().numpy()
        self.scale = (high - low) / 2.0
        self.bias = (high + low) / 2.0
        self.delta = float(delta)
        self.init_action = torch.clamp(init, low, high)

        self.num_envs = int(env.num_envs)
        self.a_prev = self.init_action.expand(self.num_envs, -1).clone()

        self.single_action_space = gym.spaces.Box(
            -1.0, 1.0, (action_dim,), dtype=np.float32
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )
        if self.augment_obs:
            obs_dim = int(np.prod(env.single_observation_space.shape)) + action_dim
            self.single_observation_space = gym.spaces.Box(
                -np.inf, np.inf, (obs_dim,), dtype=np.float32
            )
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space, self.num_envs
            )

        # The command that produced the observation `step` just returned
        # (`a_prev` is rewound for auto-reset sub-envs; this one is not).
        self.a_prev_pre_reset = self.a_prev.clone()

    def normalized(self, command: torch.Tensor) -> torch.Tensor:
        """Absolute command -> the [-1, 1] block that goes into the observation."""
        return (command - self.bias) / self.scale

    def _augment(self, obs):
        """Append the normalized previous command (``a_t`` depends on ``a_{t-1}``);
        no-op when `augment_obs=False`."""
        if not self.augment_obs:
            return obs
        return torch.cat([obs, self.normalized(self.a_prev)], dim=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.a_prev = self.init_action.expand(self.num_envs, -1).clone()
        self.a_prev_pre_reset = self.a_prev.clone()
        return self._augment(obs), info

    def step(self, action: torch.Tensor):
        action = torch.as_tensor(action, device=self.a_prev.device)
        absolute = torch.clamp(
            self.a_prev + action * self.delta, self.low, self.high
        )
        obs, reward, terminated, truncated, info = self.env.step(absolute)
        self.a_prev = absolute
        self.a_prev_pre_reset = absolute.clone()

        # `final_observation` (true o' of auto-reset sub-envs) gets the command
        # that produced it, before `a_prev` is rewound below.
        if "final_observation" in info:
            info["final_observation"] = self._augment(info["final_observation"])

        done = torch.logical_or(terminated, truncated)
        if bool(done.any()):
            self.a_prev = torch.where(
                done.reshape(-1, 1), self.init_action, self.a_prev
            )
        return self._augment(obs), reward, terminated, truncated, info


def to_delta_dataset(
    replay: OfflineReplayDataset,
    *,
    delta: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
    init_action: np.ndarray,
) -> OfflineReplayDataset:
    """Rewrite a loaded replay in place into the delta parameterization.

    ``action_low`` / ``action_high`` are the *absolute* (env) bounds, not the
    ``[-1, 1]`` bounds the wrapper advertises. Operates on the in-memory
    tensors only; the dataset on disk is never touched.
    """
    low = torch.as_tensor(action_low, dtype=torch.float32)
    high = torch.as_tensor(action_high, dtype=torch.float32)
    scale = (high - low) / 2.0
    bias = (high + low) / 2.0
    init = torch.clamp(torch.as_tensor(init_action, dtype=torch.float32), low, high)

    actions = replay.actions
    if actions.shape[1] != low.numel():
        raise RuntimeError(
            f"Replay action_dim={actions.shape[1]} != env action_dim={low.numel()}."
        )

    # Transitions are concatenated episode by episode, so an episode starts
    # wherever the episode index changes.
    episodes = replay.episode_indices
    starts = torch.ones_like(episodes, dtype=torch.bool)
    starts[1:] = episodes[1:] != episodes[:-1]

    # Encode against the command the wrapper would be holding, not the recorded
    # previous action, so clipping is self-correcting.
    absolute = actions.numpy()
    low_np, high_np = low.numpy(), high.numpy()
    init_np = init.numpy()
    is_start = starts.numpy()

    held = np.empty_like(absolute)  # a_{t-1} the policy is holding
    applied = np.empty_like(absolute)  # a_t after clipping
    deltas_np = np.empty_like(absolute)
    current = init_np.copy()
    for i in range(absolute.shape[0]):
        if is_start[i]:
            current = init_np.copy()
        held[i] = current
        step = np.clip((absolute[i] - current) / float(delta), -1.0, 1.0)
        current = np.clip(current + step * float(delta), low_np, high_np)
        deltas_np[i] = step
        applied[i] = current

    deltas = torch.from_numpy(deltas_np)
    previous = torch.from_numpy(held)
    reached = torch.from_numpy(applied)

    replay.observations = torch.cat(
        [replay.observations, (previous - bias) / scale], dim=1
    )
    replay.next_observations = torch.cat(
        [replay.next_observations, (reached - bias) / scale], dim=1
    )
    replay.actions = deltas
    # a'[i] = a[i+1], an episode's last row masked, mirroring the loader.
    next_deltas = torch.empty_like(deltas)
    next_deltas[:-1] = deltas[1:]
    next_deltas[-1] = deltas[-1]
    ends = torch.ones_like(episodes, dtype=torch.bool)
    ends[:-1] = episodes[:-1] != episodes[1:]
    next_deltas[ends] = deltas[ends]
    replay.next_actions = next_deltas

    tracking = np.abs(applied - absolute)
    clipped = float((np.abs(deltas_np) >= 1.0 - 1e-6).mean())
    print(
        f"[delta] delta={delta} | clipped {100.0 * clipped:.3f}% of entries | "
        f"tracking error mean {tracking.mean():.6f} p99 {np.percentile(tracking, 99):.6f} "
        f"max {tracking.max():.4f} rad | obs_dim "
        f"{replay.observations.shape[1] - actions.shape[1]} -> "
        f"{replay.observations.shape[1]}"
    )
    return replay


def delta_command_recurrence(
    absolute: np.ndarray,
    *,
    delta: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
    init_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay the wrapper's recurrence over one episode's absolute actions.

    Returns ``(deltas, held, applied)``: the ``[-1, 1]`` action, the command
    held beforehand (``a_{t-1}``) and the command reached after clipping (``a_t``).
    """
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    init = np.clip(np.asarray(init_action, dtype=np.float32), low, high)

    held = np.empty_like(absolute)
    applied = np.empty_like(absolute)
    deltas = np.empty_like(absolute)
    current = init.copy()
    for i in range(absolute.shape[0]):
        held[i] = current
        step = np.clip((absolute[i] - current) / float(delta), -1.0, 1.0)
        current = np.clip(current + step * float(delta), low, high)
        deltas[i] = step
        applied[i] = current
    return deltas, held, applied


def to_delta_image_dataset(
    dataset,
    *,
    delta: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
    init_action: np.ndarray,
) -> None:
    """Rewrite an :class:`~utils.pixel_dataset.OfflineImageDataset` in place.

    The image analogue of :func:`to_delta_dataset`: the command block is
    appended to ``proprio`` per frame (``T + 1`` rows: ``held[0]`` followed by
    ``applied[0..T-1]``).
    """
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    scale = (high - low) / 2.0
    bias = (high + low) / 2.0

    actions = dataset.actions.cpu().numpy()
    if actions.shape[1] != low.size:
        raise RuntimeError(
            f"Dataset action_dim={actions.shape[1]} != env action_dim={low.size}."
        )
    counts = list(dataset.episode_transition_counts)
    if sum(counts) != actions.shape[0]:
        raise RuntimeError(
            f"Episode transition counts sum to {sum(counts)} but the dataset holds "
            f"{actions.shape[0]} transitions."
        )

    new_actions = np.empty_like(actions)
    frame_commands: list[np.ndarray] = []
    cursor = 0
    for num_transitions in counts:
        chunk = actions[cursor : cursor + num_transitions]
        deltas, held, applied = delta_command_recurrence(
            chunk,
            delta=delta,
            action_low=low,
            action_high=high,
            init_action=init_action,
        )
        new_actions[cursor : cursor + num_transitions] = deltas
        # T + 1 frame commands: what a_prev was at each stored frame.
        frame_commands.append(np.concatenate([held[:1], applied], axis=0))
        cursor += num_transitions

    commands = np.concatenate(frame_commands, axis=0)
    if commands.shape[0] != dataset.proprio_frames.shape[0]:
        raise RuntimeError(
            f"Built {commands.shape[0]} frame commands for "
            f"{dataset.proprio_frames.shape[0]} proprio frames."
        )
    normalized = torch.from_numpy((commands - bias) / scale).to(
        dataset.proprio_frames.device
    )
    dataset.proprio_frames = torch.cat([dataset.proprio_frames, normalized], dim=1)
    dataset.actions = torch.from_numpy(new_actions).to(dataset.actions.device)

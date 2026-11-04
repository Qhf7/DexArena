"""Network primitives shared by the offline and offline-to-online RL algorithms.

These used to live in ``algo/cql.py`` (and were duplicated in ``bc.py`` /
``td3bc.py`` / ``rlpd.py``), which made CQL an implicit dependency of every other
agent. Attribute names are unchanged, so checkpoints written before this module
existed still load.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    """Plain MLP with ReLU activations and orthogonal init."""
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        linear = nn.Linear(current_dim, int(hidden_dim))
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.ReLU()])
        current_dim = int(hidden_dim)
    output = nn.Linear(current_dim, int(output_dim))
    nn.init.orthogonal_(output.weight, gain=1.0)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def build_backbone(input_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    """``build_mlp`` without the output layer, for heads that branch."""
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        linear = nn.Linear(current_dim, int(hidden_dim))
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.ReLU()])
        current_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed diagonal-Gaussian policy, actions scaled to [low, high]."""

    def __init__(
        self,
        obs_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        action_dim = int(action_low.numel())
        self.backbone = build_backbone(obs_dim, hidden_dims)
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.zeros_(self.log_std.bias)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        if torch.any(action_scale <= 0):
            raise ValueError("Every continuous action dimension must have a positive range.")
        self.register_buffer("action_scale", action_scale.float())
        self.register_buffer("action_bias", action_bias.float())

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        log_prob = distribution.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale * (1.0 - squashed.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(obs)
        normalized_action = ((action - self.action_bias) / self.action_scale).clamp(
            -1.0 + 1e-6, 1.0 - 1e-6
        )
        pre_tanh = torch.atanh(normalized_action)
        distribution = Normal(mean, log_std.exp())
        log_prob = distribution.log_prob(pre_tanh)
        log_prob -= torch.log(
            self.action_scale * (1.0 - normalized_action.pow(2)) + 1e-6
        )
        return log_prob.sum(dim=-1, keepdim=True)


class DeterministicActor(nn.Module):
    """pi(s) -> action, tanh-squashed and scaled to [action_low, action_high]."""

    def __init__(
        self,
        obs_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        action_dim = int(action_low.numel())
        self.network = build_mlp(obs_dim, hidden_dims, action_dim)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        if torch.any(action_scale <= 0):
            raise ValueError("Every continuous action dimension must have a positive range.")
        self.register_buffer("action_scale", action_scale.float())
        self.register_buffer("action_bias", action_bias.float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(obs)) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    """Q(s, a) -> scalar."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.network = build_mlp(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([obs, action], dim=-1))


class ValueNetwork(nn.Module):
    """V(s) -> scalar."""

    def __init__(self, obs_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.network = build_mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class RunningMeanStd(nn.Module):
    """Welford running statistics, used to standardize the proprio vector.

    The state-based agents normalize observations with the demo dataset's mean
    and std. An image-based run may have no state demo data at all (and its
    proprio block is a different vector from the state observation anyway), so
    it accumulates the statistics from collected transitions instead. Registered
    as buffers, so they travel with the checkpoint and evaluation reproduces
    training.

    Callers must update it during collection only, never inside a gradient
    update: shifting the normalization between the rollout and the update that
    consumes it changes pi_old's inputs after the fact, which for PPO silently
    corrupts the importance ratio.
    """

    def __init__(self, dim: int, epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(float(epsilon)))

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        batch = batch.detach().float().to(self.mean.device)
        batch = batch.reshape(-1, batch.shape[-1])
        batch_count = torch.tensor(float(batch.shape[0]), device=self.mean.device)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta.pow(2) * self.count * batch_count / total) / total
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total)

    def normalize(self, batch: torch.Tensor) -> torch.Tensor:
        # `.to(self.mean.device)` for the same reason `update` above does it.
        # Leaving it out made this class correct only when the caller happened
        # to have moved the batch already; the physx_cpu half of the RLPD-pixel
        # fleet did not, and every one of those runs died here on 2026-09-18.
        # A normalizer that works on one simulator backend and not another is a
        # trap regardless of which caller is "at fault".
        return (batch.to(self.mean.device) - self.mean) / self.var.clamp_min(
            1e-8
        ).sqrt()

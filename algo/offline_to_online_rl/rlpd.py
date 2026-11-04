"""RLPD (Reinforcement Learning with Prior Data), self-contained.

A faithful, single-file re-implementation of the algorithm from
``ikostrikov/rlpd`` (Ball et al., 2023), tuned to the repo's *Adroit Binary*
recipe (online RL from scratch with a fixed expert/demo buffer mixed 50/50 into
every batch). Everything the algorithm needs lives here on purpose — the actor,
the critic ensemble and the SAC update — so the method reads top to bottom
without hopping into the other agent modules.

The recipe = SAC + four ingredients:
  1. **Critic ensemble** of ``num_qs`` Q-networks (default 10).
  2. **Random subset min** for the TD target: at each update a fresh subset of
     ``num_min_qs`` (default 2) critics is drawn and their elementwise min is the
     bootstrap value (REDQ-style, fights overestimation under high UTD).
  3. **LayerNorm** inside every critic — the key stabilizer that lets the agent
     bootstrap on its own off-distribution early data without Q divergence.
  4. **Automatic entropy tuning** (``target_entropy = -action_dim / 2``), with
     ``backup_entropy`` off by default (the Adroit Binary setting).

The 50/50 online/offline symmetric sampling is handled outside this file by
``utils.online_replay_buffer.OnlineReplayBuffer`` (``offline_mix_fraction=0.5``).
There is no BC anchor and no warm-start: canonical RLPD trains from scratch.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.networks import SquashedGaussianActor
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path



class EnsembleLinear(nn.Module):
    """``num_qs`` independent Linear layers evaluated in one batched matmul.

    Weights are stored as ``[num_qs, in, out]`` and applied to a stacked input
    ``[num_qs, batch, in]`` via a single ``baddbmm`` — one kernel instead of a
    Python loop over the heads. Each head is initialized exactly as the loop
    version's ``nn.Linear`` (orthogonal weight, zero bias).
    """

    def __init__(self, num_qs: int, in_features: int, out_features: int, *, gain: float):
        super().__init__()
        weight = torch.empty(num_qs, in_features, out_features)
        for i in range(num_qs):
            head = torch.empty(out_features, in_features)  # match nn.Linear layout
            nn.init.orthogonal_(head, gain=gain)
            weight[i] = head.t()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(num_qs, 1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [num_qs, batch, in]
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleLayerNorm(nn.Module):
    """Per-head LayerNorm over the feature dim of a ``[num_qs, batch, dim]`` tensor.

    Each head keeps its own affine parameters (``[num_qs, 1, dim]``), matching a
    separate ``nn.LayerNorm`` per critic in the loop version.
    """

    def __init__(self, num_qs: int, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_qs, 1, dim))
        self.bias = nn.Parameter(torch.zeros(num_qs, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class EnsembleCritic(nn.Module):
    """Ensemble of ``num_qs`` LayerNorm Q-networks Q(s, a) -> scalar.

    Vectorized: instead of holding ``num_qs`` separate MLPs and looping in
    Python, the heads share a stack of ``EnsembleLinear`` / ``EnsembleLayerNorm``
    layers so every head is evaluated in one batched matmul per layer.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        num_qs: int,
        *,
        layer_norm: bool,
    ):
        super().__init__()
        self.num_qs = int(num_qs)
        layers: list[nn.Module] = []
        current_dim = int(obs_dim + action_dim)
        for hidden_dim in hidden_dims:
            layers.append(
                EnsembleLinear(self.num_qs, current_dim, int(hidden_dim), gain=np.sqrt(2))
            )
            if layer_norm:
                layers.append(EnsembleLayerNorm(self.num_qs, int(hidden_dim)))
            layers.append(nn.ReLU())
            current_dim = int(hidden_dim)
        layers.append(EnsembleLinear(self.num_qs, current_dim, 1, gain=1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return all heads stacked as ``[num_qs, batch, 1]``."""
        sa = torch.cat([obs, action], dim=-1)
        sa = sa.unsqueeze(0).expand(self.num_qs, -1, -1).contiguous()
        return self.net(sa)

    def subset_min(
        self, obs: torch.Tensor, action: torch.Tensor, num_min_qs: int
    ) -> torch.Tensor:
        """Elementwise min over a random subset of ``num_min_qs`` heads.

        REDQ-style target: draw a fresh subset each call so the bootstrap value
        is a pessimistic estimate from a different pair of critics every update.
        """
        all_q = self(obs, action)  # [num_qs, batch, 1]
        idx = torch.randperm(self.num_qs, device=all_q.device)[:num_min_qs]
        return all_q[idx].min(dim=0).values


class RLPDAgent(nn.Module):
    def __init__(
        self,
        args,
        *,
        obs_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        run_name: str,
        action_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.args = args
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_low.numel())
        self.register_buffer("obs_mean", obs_mean.float())
        self.register_buffer("obs_std", obs_std.float().clamp_min(1e-6))
        self.register_buffer("action_low", action_low.float())
        self.register_buffer("action_high", action_high.float())
        # Per-dimension scale of the demonstrated actions; only
        # `target_entropy_demo_scale` reads it.
        measured_action_std = action_std is not None
        if not measured_action_std:
            action_std = torch.ones_like(action_low)
        if not measured_action_std and getattr(
            args, "target_entropy_demo_scale", None
        ) is not None:
            raise ValueError(
                "`target_entropy_demo_scale` needs the demonstrated action scale; "
                "pass `action_std` or set `target_entropy` explicitly."
            )
        elif int(action_std.numel()) != self.action_dim:
            raise ValueError(
                f"`action_std` has {action_std.numel()} dims, action space has "
                f"{self.action_dim}."
            )
        self.register_buffer("action_std", action_std.float().clamp_min(1e-6))

        hidden_dims = tuple(int(dim) for dim in args.hidden_dims)
        self.actor = SquashedGaussianActor(
            self.obs_dim, self.action_low, self.action_high, hidden_dims
        )
        self.critic = EnsembleCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            args.num_qs,
            layer_norm=args.critic_layer_norm,
        )
        self.critic_target = EnsembleCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            args.num_qs,
            layer_norm=args.critic_layer_norm,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.actor_learning_rate
        )
        self.q_optimizer = optim.Adam(
            self.critic.parameters(), lr=args.critic_learning_rate
        )

        initial_alpha = max(float(args.init_temperature), 1e-8)
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(initial_alpha), dtype=torch.float32)
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=args.alpha_learning_rate
        )
        # `target_entropy_demo_scale`: entropy of a Gaussian whose per-dim std is
        # `scale` x the demonstrated std, instead of the `-action_dim/2` default.
        if args.target_entropy is not None:
            self.target_entropy = float(args.target_entropy)
        elif getattr(args, "target_entropy_demo_scale", None) is not None:
            scale = float(args.target_entropy_demo_scale)
            self.target_entropy = float(
                (
                    0.5 * math.log(2.0 * math.pi * math.e)
                    + torch.log(scale * self.action_std)
                ).sum()
            )
            print(
                f"[rlpd] target_entropy={self.target_entropy:.2f} derived from the "
                f"demo action scale (x{scale}); default would have been "
                f"{-self.action_dim / 2.0:.2f}"
            )
        else:
            self.target_entropy = -float(self.action_dim) / 2.0

        writer = SummaryWriter(runs_path(run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(f"|{key}|{value}|" for key, value in asdict(args).items()),
        )
        self.logger = Logger(log_swanlab=args.track, tensorboard=writer)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.args.normalize_observations:
            return obs
        return (obs - self.obs_mean) / self.obs_std

    @torch.no_grad()
    def select_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        normalized_obs = self.normalize_obs(obs)
        if deterministic:
            return self.actor.deterministic(normalized_obs)
        return self.actor.sample(normalized_obs)[0]

    def update(
        self,
        batch: dict[str, torch.Tensor],
        update_step: int,
        update_actor: bool = True,
    ) -> dict[str, float]:
        """One gradient step.

        Each call updates the critic ensemble and soft-updates the targets.
        REDQ/RLPD convention: under a UTD ratio of G the training loop calls
        this G times per collected transition but sets ``update_actor=True`` on
        only the last call, so the policy and temperature are updated once while
        the critic is updated G times (keeps the actor from chasing a noisy,
        rapidly-moving critic).
        """
        obs = self.normalize_obs(batch["observations"])
        next_obs = self.normalize_obs(batch["next_observations"])
        actions = batch["actions"]
        rewards = batch["rewards"] * self.args.reward_scale + self.args.reward_bias
        dones = batch["dones"]

        alpha = self.alpha.detach()

        # ---- critic: random-subset-min TD target over the target ensemble ----
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            target_q = self.critic_target.subset_min(
                next_obs, next_actions, self.args.num_min_qs
            )
            if self.args.backup_entropy:
                target_q = target_q - alpha * next_log_prob
            target_q = rewards + (1.0 - dones) * self.args.gamma * target_q

        all_q = self.critic(obs, actions)  # [num_qs, B, 1]
        # Regress every head onto the same target (mean over heads, sum is
        # equivalent up to a constant scale on the critic LR).
        q_loss = F.mse_loss(all_q, target_q.unsqueeze(0).expand_as(all_q))

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.args.max_grad_norm
        )
        self.q_optimizer.step()

        actor_loss_value = float("nan")
        alpha_loss_value = float("nan")
        log_prob_value = float("nan")
        bc_loss_value = float("nan")
        bc_action_err_value = float("nan")
        bc_rows_value = float("nan")
        if update_actor:
            # ---- actor: maximize ensemble-mean Q with entropy regularization ----
            self.critic.requires_grad_(False)
            sampled_actions, log_prob, mean_actions = self.actor.sample(obs)
            actor_q = self.critic(obs, sampled_actions).mean(dim=0)
            actor_loss = (self.alpha.detach() * log_prob - actor_q).mean()

            # ---- optional behavior anchor on the successful demo rows ----
            # Scaled by |Q| as in TD3+BC so `bc_coef` is independent of the reward
            # scale; regresses the deterministic action.
            if self.args.bc_coef > 0.0 and "bc_mask" in batch:
                mask = batch["bc_mask"]
                denom = mask.sum().clamp(min=1.0)
                # Raw squared error, not divided by the demo action std
                # (normalizing scales this term's gradient by ~600x).
                squared = ((mean_actions - actions) ** 2).mean(dim=-1, keepdim=True)
                bc_loss = (squared * mask).sum() / denom
                q_scale = actor_q.abs().mean().detach().clamp(min=1e-3)
                actor_loss = actor_loss + self.args.bc_coef * q_scale * bc_loss
                bc_loss_value = float(bc_loss.item())
                bc_action_err_value = float(
                    ((mean_actions - actions).abs().mean(dim=-1, keepdim=True) * mask).sum()
                    / denom
                )
                bc_rows_value = float(mask.mean().item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()
            self.critic.requires_grad_(True)

            actor_loss_value = float(actor_loss.item())
            log_prob_value = float(log_prob.mean().item())

            # ---- temperature: drive entropy toward target_entropy ----
            if self.args.autotune_entropy:
                alpha_loss = -(
                    self.log_alpha * (log_prob.detach() + self.target_entropy)
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_value = float(alpha_loss.item())

        # ---- soft target update (every critic step) ----
        with torch.no_grad():
            for online_param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.lerp_(online_param.data, self.args.tau)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "entropy_alpha": float(self.alpha.detach().item()),
            "policy_log_prob": log_prob_value,
            "bc_loss": bc_loss_value,
            # Mean |pi(s) - a_demo| per dimension on the anchored rows.
            "bc_action_err": bc_action_err_value,
            "bc_rows": bc_rows_value,
            "q_mean": float(all_q.mean().item()),
            # Spread of Q across states; a critic collapsed to a constant still
            # shows a healthy mean.
            "q_std": float(all_q.mean(dim=0).std().item()),
            "target_q": float(target_q.mean().item()),
            "target_q_abs_max": float(target_q.abs().max().item()),
            "q_grad_norm": float(q_grad_norm),
        }

    def save(self, path: str, global_step: int) -> None:
        path = ensure_parent_dir(path)
        torch.save(
            {
                "agent": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "global_step": int(global_step),
                "args": asdict(self.args),
            },
            path,
        )
        print(f"model saved to {path}")

    def load(self, path: str, *, load_optimizers: bool) -> int:
        checkpoint = torch.load(path, map_location=self.obs_mean.device)
        state = checkpoint["agent"]
        if "action_std" not in state:
            # Checkpoint written before the `action_std` buffer existed; it only
            # feeds `target_entropy_demo_scale`, so keep the constructor's value.
            state = dict(state, action_std=self.action_std)
        self.load_state_dict(state)
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        return int(checkpoint.get("global_step", 0))

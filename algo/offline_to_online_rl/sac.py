"""Soft Actor-Critic for online fine-tuning of an offline-pretrained policy.

Structurally identical to the offline IQL/CQL agents in this repo (it reuses
`SquashedGaussianActor` + twin `QNetwork` from ``algo.networks``), so an IQL
checkpoint can warm-start the actor, both Q-networks and their targets directly
(``load_pretrained``). The only IQL-specific weight, the value network ``vf``, is
dropped.

The update is plain SAC:
  - critic: r + gamma * (1 - done) * (min_i Q_target_i(s', a') - alpha * log pi(a'|s'))
  - actor:  E[ alpha * log pi(a|s) - min_i Q_i(s, a) ]
  - alpha:  optional autotuning toward target_entropy = -action_dim
plus an optional, decaying behavior-cloning anchor (MSE to demo actions) for
extra early-stage stability.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.networks import QNetwork, SquashedGaussianActor
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path


class SACAgent(nn.Module):
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
    ):
        super().__init__()
        self.args = args
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_low.numel())
        self.register_buffer("obs_mean", obs_mean.float())
        self.register_buffer("obs_std", obs_std.float().clamp_min(1e-6))
        self.register_buffer("action_low", action_low.float())
        self.register_buffer("action_high", action_high.float())

        hidden_dims = tuple(int(dim) for dim in args.hidden_dims)
        self.actor = SquashedGaussianActor(
            self.obs_dim, self.action_low, self.action_high, hidden_dims
        )
        self.qf1 = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf2 = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf1_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf2_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.qf1_target.requires_grad_(False)
        self.qf2_target.requires_grad_(False)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.actor_learning_rate
        )
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=args.critic_learning_rate,
        )

        initial_alpha = max(float(args.entropy_alpha), 1e-8)
        self.log_entropy_alpha = nn.Parameter(
            torch.tensor(np.log(initial_alpha), dtype=torch.float32)
        )
        self.entropy_optimizer = optim.Adam(
            [self.log_entropy_alpha], lr=args.alpha_learning_rate
        )
        self.target_entropy = (
            -float(self.action_dim)
            if args.target_entropy is None
            else float(args.target_entropy)
        )

        writer = SummaryWriter(runs_path(run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(f"|{key}|{value}|" for key, value in asdict(args).items()),
        )
        self.logger = Logger(log_swanlab=args.track, tensorboard=writer)

    @property
    def entropy_alpha(self) -> torch.Tensor:
        if self.args.autotune_entropy:
            return self.log_entropy_alpha.exp()
        return torch.as_tensor(
            self.args.entropy_alpha,
            dtype=self.obs_mean.dtype,
            device=self.obs_mean.device,
        )

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

    def _bc_weight(self, update_step: int) -> float:
        weight = float(self.args.actor_bc_weight)
        if weight <= 0.0:
            return 0.0
        if not self.args.bc_decay:
            return weight
        progress = min(max(update_step / max(1, self.args.bc_decay_steps), 0.0), 1.0)
        return max(float(self.args.actor_bc_min_weight), weight * (1.0 - progress))

    def update(
        self,
        batch: dict[str, torch.Tensor],
        update_step: int,
        demo_batch: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, float]:
        obs = self.normalize_obs(batch["observations"])
        next_obs = self.normalize_obs(batch["next_observations"])
        actions = batch["actions"]
        rewards = batch["rewards"] * self.args.reward_scale + self.args.reward_bias
        dones = batch["dones"]

        alpha = self.entropy_alpha.detach()
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            next_q = torch.minimum(
                self.qf1_target(next_obs, next_actions),
                self.qf2_target(next_obs, next_actions),
            )
            target_q = rewards + (1.0 - dones) * self.args.gamma * (
                next_q - alpha * next_log_prob
            )

        q1_data = self.qf1(obs, actions)
        q2_data = self.qf2(obs, actions)
        q_loss = F.mse_loss(q1_data, target_q) + F.mse_loss(q2_data, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = nn.utils.clip_grad_norm_(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            self.args.max_grad_norm,
        )
        self.q_optimizer.step()

        actor_loss_value = float("nan")
        alpha_loss_value = float("nan")
        log_prob_value = float("nan")
        bc_term_value = float("nan")
        bc_weight_value = 0.0
        actor_q_lambda_value = float("nan")
        if update_step % self.args.policy_update_interval == 0:
            self.qf1.requires_grad_(False)
            self.qf2.requires_grad_(False)
            sampled_actions, log_prob, _ = self.actor.sample(obs)
            actor_q = torch.minimum(
                self.qf1(obs, sampled_actions),
                self.qf2(obs, sampled_actions),
            )
            # TD3+BC normalization: scale Q by its running magnitude so the Q term
            # stays O(1) regardless of (over)estimated Q growth. Without this the
            # raw Q term dominates and the actor chases an inflated, off-distribution
            # critic — the offline->online "unlearning" failure mode.
            actor_q_lambda = self.args.actor_q_weight / actor_q.abs().mean().detach().clamp_min(
                1e-6
            )
            actor_loss = (
                self.entropy_alpha.detach() * log_prob - actor_q_lambda * actor_q
            ).mean()
            actor_q_lambda_value = float(actor_q_lambda.item())

            bc_weight_value = self._bc_weight(update_step)
            if bc_weight_value > 0.0 and demo_batch is not None:
                demo_obs = self.normalize_obs(demo_batch["observations"])
                demo_actions = demo_batch["actions"]
                bc_term = F.mse_loss(self.actor.deterministic(demo_obs), demo_actions)
                actor_loss = actor_loss + bc_weight_value * bc_term
                bc_term_value = float(bc_term.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()
            self.qf1.requires_grad_(True)
            self.qf2.requires_grad_(True)

            actor_loss_value = float(actor_loss.item())
            log_prob_value = float(log_prob.mean().item())

            if self.args.autotune_entropy:
                alpha_loss = -(
                    self.log_entropy_alpha
                    * (log_prob.detach() + self.target_entropy)
                ).mean()
                self.entropy_optimizer.zero_grad()
                alpha_loss.backward()
                self.entropy_optimizer.step()
                with torch.no_grad():
                    self.log_entropy_alpha.clamp_(max=float(np.log(10.0)))
                alpha_loss_value = float(alpha_loss.item())

        if update_step % self.args.target_update_interval == 0:
            with torch.no_grad():
                for online, target in (
                    (self.qf1, self.qf1_target),
                    (self.qf2, self.qf2_target),
                ):
                    for online_param, target_param in zip(
                        online.parameters(), target.parameters()
                    ):
                        target_param.data.lerp_(online_param.data, self.args.tau)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "entropy_alpha": float(self.entropy_alpha.detach().item()),
            "policy_log_prob": log_prob_value,
            "bc_term": bc_term_value,
            "bc_weight": bc_weight_value,
            "actor_q_lambda": actor_q_lambda_value,
            "q1_data": float(q1_data.mean().item()),
            "q2_data": float(q2_data.mean().item()),
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
                "entropy_optimizer": self.entropy_optimizer.state_dict(),
                "global_step": int(global_step),
                "args": asdict(self.args),
            },
            path,
        )
        print(f"model saved to {path}")

    def load(self, path: str, *, load_optimizers: bool) -> int:
        checkpoint = torch.load(path, map_location=self.obs_mean.device)
        self.load_state_dict(checkpoint["agent"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            self.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer"])
        return int(checkpoint.get("global_step", 0))

    def load_pretrained(self, path: str) -> int:
        """Warm-start from an offline (IQL/CQL) checkpoint.

        Loads every matching parameter/buffer non-strictly: the actor, both
        Q-networks, their targets and the obs-normalization stats transfer over;
        IQL's value network (`vf.*`) is an unexpected key and is ignored, while
        SAC's `log_entropy_alpha` is a missing key and keeps its init.
        """
        checkpoint = torch.load(path, map_location=self.obs_mean.device)
        state = checkpoint["agent"] if "agent" in checkpoint else checkpoint
        missing, unexpected = self.load_state_dict(state, strict=False)
        ignored = [k for k in unexpected if not k.startswith("vf.")]
        unexpected_critical = ignored
        missing_critical = [k for k in missing if k != "log_entropy_alpha"]
        if missing_critical or unexpected_critical:
            print(
                "[SAC warm-start] non-trivial key mismatch — "
                f"missing={missing_critical}, unexpected={unexpected_critical}"
            )
        else:
            print(
                "[SAC warm-start] loaded actor + twin Q + targets + obs stats "
                f"from {path} (ignored vf.*, kept fresh entropy alpha)."
            )
        return int(checkpoint.get("global_step", 0))

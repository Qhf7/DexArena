"""Minimal SARSA / Fitted-Q-Evaluation agent (policy evaluation only).

Decoupled from any policy improvement: this fits a single Q-network to the
behavior policy with the SARSA target

    Q(s, a) <- r + (1 - done) * gamma * Q_target(s', a')

where ``a'`` is the *next action actually taken in the demo* (from the
transition), never sampled from a policy. Because ``a'`` is always in the data
distribution, the bootstrap target never queries out-of-distribution actions, so
the regression is stable by construction. Its fixed point equals the discounted
return-to-go of each demo, which the trainer checks against a Monte-Carlo
ground truth.

There is intentionally no actor, no conservative (CQL) term, no entropy term,
and no OOD action sampling here.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.networks import QNetwork
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path


class FQEAgent(nn.Module):
    def __init__(
        self,
        args,
        *,
        obs_dim: int,
        action_dim: int,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        run_name: str,
    ):
        super().__init__()
        self.args = args
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.register_buffer("obs_mean", obs_mean.float())
        self.register_buffer("obs_std", obs_std.float().clamp_min(1e-6))

        hidden_dims = tuple(int(dim) for dim in args.hidden_dims)
        # A single Q (faithful to Q(s', a')); a' is in-distribution so there is
        # no overestimation to fight and no need for a twin-min critic.
        self.qf = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf_target.load_state_dict(self.qf.state_dict())
        self.qf_target.requires_grad_(False)

        self.q_optimizer = optim.Adam(
            self.qf.parameters(), lr=args.critic_learning_rate
        )

        writer = SummaryWriter(runs_path(run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(f"|{key}|{value}|" for key, value in asdict(args).items()),
        )
        self.logger = Logger(log_swanlab=args.track, tensorboard=writer)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.args.normalize_observations:
            return obs
        return (obs - self.obs_mean) / self.obs_std

    def update(self, batch: dict[str, torch.Tensor], global_step: int) -> dict[str, float]:
        obs = self.normalize_obs(batch["observations"])
        next_obs = self.normalize_obs(batch["next_observations"])
        actions = batch["actions"]
        next_actions = batch["next_actions"]
        rewards = batch["rewards"] * self.args.reward_scale + self.args.reward_bias
        dones = batch["dones"]

        with torch.no_grad():
            next_q = self.qf_target(next_obs, next_actions)
            target_q = rewards + (1.0 - dones) * self.args.gamma * next_q

        q_data = self.qf(obs, actions)
        q_loss = F.mse_loss(q_data, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = nn.utils.clip_grad_norm_(self.qf.parameters(), self.args.max_grad_norm)
        self.q_optimizer.step()

        if global_step % self.args.target_update_interval == 0:
            with torch.no_grad():
                for online_param, target_param in zip(
                    self.qf.parameters(), self.qf_target.parameters()
                ):
                    target_param.data.lerp_(online_param.data, self.args.tau)

        return {
            "q_loss": float(q_loss.item()),
            "q_bellman_loss": float(q_loss.item()),
            "q_data": float(q_data.mean().item()),
            "target_q": float(target_q.mean().item()),
            "q_grad_norm": float(q_grad_norm),
        }

    @torch.no_grad()
    def predict_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.qf(self.normalize_obs(obs), action)

    def save(self, path: str, global_step: int) -> None:
        path = ensure_parent_dir(path)
        torch.save(
            {
                "agent": self.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
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
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        return int(checkpoint.get("global_step", 0))

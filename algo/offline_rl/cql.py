from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.networks import QNetwork, SquashedGaussianActor
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path


class CQLAgent(nn.Module):
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
    def select_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        normalized_obs = self.normalize_obs(obs)
        if deterministic:
            return self.actor.deterministic(normalized_obs)
        return self.actor.sample(normalized_obs)[0]

    def _repeat_obs(self, obs: torch.Tensor, num_actions: int) -> torch.Tensor:
        return (
            obs.unsqueeze(1)
            .expand(-1, int(num_actions), -1)
            .reshape(-1, obs.shape[-1])
        )

    def _conservative_losses(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        num_actions = int(self.args.cql_num_actions)
        repeated_obs = self._repeat_obs(obs, num_actions)
        repeated_next_obs = self._repeat_obs(next_obs, num_actions)

        random_actions = torch.empty(
            batch_size * num_actions,
            self.action_dim,
            device=obs.device,
            dtype=obs.dtype,
        ).uniform_(0.0, 1.0)
        random_actions = self.action_low + random_actions * (
            self.action_high - self.action_low
        )

        with torch.no_grad():
            current_actions, current_log_prob, _ = self.actor.sample(repeated_obs)
            next_actions, next_log_prob, _ = self.actor.sample(repeated_next_obs)

        def evaluate_candidates(q_network: QNetwork) -> torch.Tensor:
            q_random = q_network(repeated_obs, random_actions).view(batch_size, num_actions)
            q_current = q_network(repeated_obs, current_actions).view(batch_size, num_actions)
            q_next = q_network(repeated_obs, next_actions).view(batch_size, num_actions)

            if self.args.cql_importance_sample:
                uniform_log_prob = -torch.log(
                    self.action_high - self.action_low
                ).sum()
                q_random = q_random - uniform_log_prob
                q_current = q_current - current_log_prob.view(batch_size, num_actions)
                q_next = q_next - next_log_prob.view(batch_size, num_actions)
            return torch.cat([q_random, q_current, q_next], dim=1)

        temperature = float(self.args.cql_temperature)
        cql_q1 = evaluate_candidates(self.qf1)
        cql_q2 = evaluate_candidates(self.qf2)
        cql1 = (
            torch.logsumexp(cql_q1 / temperature, dim=1).mean() * temperature
            - q1_data.mean()
        )
        cql2 = (
            torch.logsumexp(cql_q2 / temperature, dim=1).mean() * temperature
            - q2_data.mean()
        )
        return cql1, cql2

    def update(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
        actor_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        obs = self.normalize_obs(batch["observations"])
        next_obs = self.normalize_obs(batch["next_observations"])
        actions = batch["actions"]
        rewards = batch["rewards"] * self.args.reward_scale + self.args.reward_bias
        dones = batch["dones"]
        is_extra = batch.get("is_extra")

        alpha = self.entropy_alpha.detach()
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            next_q = torch.minimum(
                self.qf1_target(next_obs, next_actions),
                self.qf2_target(next_obs, next_actions),
            )
            not_done = 1.0 - dones
            if is_extra is not None and not self.args.bootstrap_extra_transitions:
                not_done = not_done * (~is_extra).float().view_as(not_done)
            target_q = rewards + not_done * self.args.gamma * (
                next_q - alpha * next_log_prob
            )

        q1_data = self.qf1(obs, actions)
        q2_data = self.qf2(obs, actions)
        q1_bellman_loss = F.mse_loss(q1_data, target_q)
        q2_bellman_loss = F.mse_loss(q2_data, target_q)
        cql1_loss, cql2_loss = self._conservative_losses(
            obs, next_obs, q1_data, q2_data
        )
        cql_loss = float(self.args.cql_alpha) * (cql1_loss + cql2_loss)
        q_loss = q1_bellman_loss + q2_bellman_loss + cql_loss

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
        actor_q_lambda_value = float("nan")
        if global_step % self.args.policy_update_interval == 0:
            policy_batch = actor_batch if actor_batch is not None else batch
            policy_obs = self.normalize_obs(policy_batch["observations"])
            policy_actions = policy_batch["actions"]
            self.qf1.requires_grad_(False)
            self.qf2.requires_grad_(False)
            sampled_actions, log_prob, _ = self.actor.sample(policy_obs)
            # Persistent behavior-cloning anchor: keep the policy close to the data
            # distribution on every update so the bootstrapped next_actions stay
            # in-distribution and the critic does not overestimate OOD actions.
            # NLL drives log_std to its floor (policy variance collapse), which
            # pins policy_log_prob high and poisons the importance-sampled
            # conservative term; the MSE anchor on the deterministic action trains
            # only the mean and leaves the spread untouched (TD3+BC style).
            if self.args.bc_loss_type == "nll":
                bc_term = -self.actor.log_prob(policy_obs, policy_actions).mean()
            else:
                bc_term = F.mse_loss(self.actor.deterministic(policy_obs), policy_actions)
            if global_step <= self.args.bc_warmup_steps:
                # Warmup: pure BC (plus the entropy term, which is 0 by default).
                actor_loss = (self.entropy_alpha.detach() * log_prob).mean() + (
                    self.args.actor_bc_weight * bc_term
                )
            else:
                actor_q = torch.minimum(
                    self.qf1(policy_obs, sampled_actions),
                    self.qf2(policy_obs, sampled_actions),
                )
                # TD3+BC normalization: scale Q by its magnitude so the BC term and
                # the Q term stay on comparable scales regardless of Q growth.
                actor_q_lambda = self.args.actor_q_weight / actor_q.abs().mean().detach().clamp_min(
                    1e-6
                )
                actor_loss = (
                    self.entropy_alpha.detach() * log_prob - actor_q_lambda * actor_q
                ).mean() + self.args.actor_bc_weight * bc_term
                actor_q_lambda_value = float(actor_q_lambda.item())

            bc_term_value = float(bc_term.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()
            self.qf1.requires_grad_(True)
            self.qf2.requires_grad_(True)

            actor_loss_value = float(actor_loss.item())
            log_prob_value = float(log_prob.mean().item())
            if self.args.autotune_entropy and global_step > self.args.bc_warmup_steps:
                alpha_loss = -(
                    self.log_entropy_alpha
                    * (log_prob.detach() + self.target_entropy)
                ).mean()
                self.entropy_optimizer.zero_grad()
                alpha_loss.backward()
                self.entropy_optimizer.step()
                # Safety cap: keep the temperature from blowing up if autotuning is
                # ever re-enabled and the policy log-prob spikes.
                with torch.no_grad():
                    self.log_entropy_alpha.clamp_(max=float(np.log(10.0)))
                alpha_loss_value = float(alpha_loss.item())

        if global_step % self.args.target_update_interval == 0:
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
            "q1_bellman_loss": float(q1_bellman_loss.item()),
            "q2_bellman_loss": float(q2_bellman_loss.item()),
            "cql1_loss": float(cql1_loss.item()),
            "cql2_loss": float(cql2_loss.item()),
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "entropy_alpha": float(self.entropy_alpha.detach().item()),
            "policy_log_prob": log_prob_value,
            "bc_term": bc_term_value,
            "actor_q_lambda": actor_q_lambda_value,
            "q1_data": float(q1_data.mean().item()),
            "q2_data": float(q2_data.mean().item()),
            "target_q": float(target_q.mean().item()),
            "target_q_abs_max": float(target_q.abs().max().item()),
            "critic_extra_frac": (
                float(is_extra.float().mean().item()) if is_extra is not None else 0.0
            ),
            "q_grad_norm": float(q_grad_norm),
        }

    def update_bc(self, batch: dict[str, torch.Tensor], global_step: int) -> dict[str, float]:
        obs = self.normalize_obs(batch["observations"])
        actions = batch["actions"]

        if self.args.bc_loss_type == "nll":
            bc_loss = -self.actor.log_prob(obs, actions).mean()
            pred_actions = self.actor.deterministic(obs)
            bc_mse = F.mse_loss(pred_actions, actions)
        else:
            pred_actions = self.actor.deterministic(obs)
            bc_mse = F.mse_loss(pred_actions, actions)
            bc_loss = bc_mse

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.args.max_grad_norm
        )
        self.actor_optimizer.step()

        with torch.no_grad():
            action_abs_error = torch.abs(pred_actions - actions).mean()

        return {
            "bc_loss": float(bc_loss.item()),
            "bc_mse": float(bc_mse.item()),
            "bc_action_abs_error": float(action_abs_error.item()),
            "actor_grad_norm": float(actor_grad_norm),
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

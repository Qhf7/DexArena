"""TD3+BC for offline learning on the DexArena tasks.

A faithful, self-contained reimplementation of TD3+BC
(https://github.com/sfujim/TD3_BC), adapted to this repo's observation/action
dimensions and the LeRobot offline replay buffer. Deliberately standalone: it
does NOT reuse the actor objective from ``algo.offline_rl.cql`` / ``algo.offline_rl.iql``.

The recipe (Fujimoto & Gu, 2021):
  - Deterministic actor pi(s) (tanh-squashed to the action range).
  - Twin critics Q1, Q2 with clipped double-Q targets and target networks.
  - Target policy smoothing: a' = pi_target(s') + clipped noise, clamped to range.
  - Delayed (policy_freq) actor + target updates.
  - Actor loss: -lambda * Q1(s, pi(s)) + MSE(pi(s), a_demo), with the adaptive
    lambda = alpha / mean(|Q|) keeping the RL and BC terms on comparable scales.
  - State normalization (handled here via obs_mean / obs_std).
"""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.networks import DeterministicActor, QNetwork
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path


class TD3BCAgent(nn.Module):
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
        self.actor = DeterministicActor(
            self.obs_dim, self.action_low, self.action_high, hidden_dims
        )
        self.actor_target = DeterministicActor(
            self.obs_dim, self.action_low, self.action_high, hidden_dims
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad_(False)

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

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        # TD3+BC's policy is deterministic; `deterministic` is accepted only for a
        # uniform interface with the stochastic agents in this repo.
        return self.actor(self.normalize_obs(obs))

    def update(self, batch: dict[str, torch.Tensor], global_step: int) -> dict[str, float]:
        obs = self.normalize_obs(batch["observations"])
        next_obs = self.normalize_obs(batch["next_observations"])
        actions = batch["actions"]
        rewards = batch["rewards"] * self.args.reward_scale + self.args.reward_bias
        dones = batch["dones"]

        # --- Critic update (every step) with target policy smoothing. ---
        with torch.no_grad():
            # Noise is specified in normalized action units (as in the reference)
            # and scaled into each dim's physical range.
            noise = (
                torch.randn_like(actions) * self.args.policy_noise
            ).clamp(-self.args.noise_clip, self.args.noise_clip) * self.actor.action_scale
            next_actions = (self.actor_target(next_obs) + noise).clamp(
                self.action_low, self.action_high
            )
            target_q = torch.minimum(
                self.qf1_target(next_obs, next_actions),
                self.qf2_target(next_obs, next_actions),
            )
            target_q = rewards + (1.0 - dones) * self.args.gamma * target_q

        q1_data = self.qf1(obs, actions)
        q2_data = self.qf2(obs, actions)
        q1_loss = F.mse_loss(q1_data, target_q)
        q2_loss = F.mse_loss(q2_data, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = nn.utils.clip_grad_norm_(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            self.args.max_grad_norm,
        )
        self.q_optimizer.step()

        # --- Delayed actor + target update (every policy_freq steps). ---
        actor_loss_value = float("nan")
        actor_q_term_value = float("nan")
        bc_loss_value = float("nan")
        actor_q_lambda_value = float("nan")
        bc_abs_error_value = float("nan")
        bc_gripper_abs_error_value = float("nan")
        if global_step % self.args.policy_freq == 0:
            pred_actions = self.actor(obs)
            actor_q = self.qf1(obs, pred_actions)
            # TD3+BC lambda normalization: scale Q by its running magnitude so the
            # RL term stays O(alpha) regardless of Q growth, keeping it balanced
            # against the (unit-weighted) BC term.
            lmbda = self.args.alpha / actor_q.abs().mean().detach().clamp_min(1e-6)
            bc_loss = F.mse_loss(pred_actions, actions)
            actor_q_term = -lmbda * actor_q.mean()
            actor_loss = actor_q_term + bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()

            with torch.no_grad():
                for online, target in (
                    (self.actor, self.actor_target),
                    (self.qf1, self.qf1_target),
                    (self.qf2, self.qf2_target),
                ):
                    for online_param, target_param in zip(
                        online.parameters(), target.parameters()
                    ):
                        target_param.data.lerp_(online_param.data, self.args.tau)

            actor_loss_value = float(actor_loss.item())
            actor_q_term_value = float(actor_q_term.item())
            bc_loss_value = float(bc_loss.item())
            actor_q_lambda_value = float(lmbda.item())
            with torch.no_grad():
                bc_abs_error_value = float(
                    torch.abs(pred_actions - actions).mean().item()
                )
                bc_gripper_abs_error_value = float(
                    torch.abs(pred_actions[:, -1] - actions[:, -1]).mean().item()
                )

        return {
            "q_loss": float(q_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "q1_data": float(q1_data.mean().item()),
            "q2_data": float(q2_data.mean().item()),
            "target_q": float(target_q.mean().item()),
            "target_q_abs_max": float(target_q.abs().max().item()),
            "actor_loss": actor_loss_value,
            "actor_q_term": actor_q_term_value,
            "bc_loss": bc_loss_value,
            "actor_q_lambda": actor_q_lambda_value,
            "bc_action_abs_error": bc_abs_error_value,
            "bc_gripper_abs_error": bc_gripper_abs_error_value,
            "q_grad_norm": float(q_grad_norm),
        }

    def save(self, path: str, global_step: int) -> None:
        path = ensure_parent_dir(path)
        torch.save(
            {
                "agent": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "global_step": int(global_step),
                "args": asdict(self.args),
            },
            path,
        )
        print(f"model saved to {path}")

    def load_actor_from_bc(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.obs_mean.device)
        bc_state = checkpoint["agent"]

        checks = {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "action_low": self.action_low,
            "action_high": self.action_high,
        }
        for key, current in checks.items():
            loaded = bc_state.get(key)
            if loaded is None or tuple(loaded.shape) != tuple(current.shape):
                raise RuntimeError(
                    f"BC checkpoint `{path}` has incompatible `{key}` shape."
                )
            if not torch.allclose(loaded.to(current.device), current, atol=1e-5, rtol=1e-4):
                print(
                    f"[td3bc] warning: BC checkpoint `{key}` differs from current "
                    "TD3+BC value; make sure the dataset/control mode match."
                )

        actor_state = {
            key.removeprefix("actor."): value
            for key, value in bc_state.items()
            if key.startswith("actor.")
        }
        missing, unexpected = self.actor.load_state_dict(actor_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "BC actor checkpoint is incompatible with TD3+BC actor: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.actor_target.load_state_dict(self.actor.state_dict())
        print(f"initialized TD3+BC actor from BC checkpoint `{path}`")

    def load(self, path: str, *, load_optimizers: bool) -> int:
        checkpoint = torch.load(path, map_location=self.obs_mean.device)
        self.load_state_dict(checkpoint["agent"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        return int(checkpoint.get("global_step", 0))

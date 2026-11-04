"""One-step RL / Advantage-Weighted Regression on a frozen behavior Q.

This is the *policy improvement* half, decoupled from policy evaluation. It takes
the Q-network already fit (and validated) by the FQE/SARSA stage (`algo/offline_rl/fqe.py`),
freezes it, and extracts an improved policy without ever querying an
out-of-distribution action:

    V(s)   <- expectile regression toward  Q_frozen(s, a)      (a from data)
    pi     <- argmax  E[ exp(beta * (Q_frozen(s,a) - V(s))) * log pi(a|s) ]

Because the frozen Q already absorbed the discounted future during FQE training,
there is NO temporal-difference bootstrap here: V is simply the expectile of the
frozen Q over the behavior action distribution, and the actor is advantage-
weighted behavior cloning. Consequently `update()` only ever touches `(s, a)` --
it does not read next-observations, next-actions, or rewards -- and every Q query
is on an in-distribution dataset action, so no OOD value is ever used.

Reuses `ValueNetwork`, `SquashedGaussianActor` and `QNetwork` from
`algo/networks.py`. There is intentionally no critic optimizer and no target network.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim

from algo.networks import QNetwork, SquashedGaussianActor, ValueNetwork
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path


class AWRAgent(nn.Module):
    def __init__(
        self,
        args,
        *,
        obs_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        frozen_q: QNetwork,
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
        # Frozen behavior-Q from FQE: stored as a submodule (so .to(device) and
        # checkpointing carry it), but never optimized and never grad-tracked.
        self.qf_frozen = frozen_q
        self.qf_frozen.eval()
        self.qf_frozen.requires_grad_(False)

        self.vf = ValueNetwork(self.obs_dim, hidden_dims)
        self.actor = SquashedGaussianActor(
            self.obs_dim, self.action_low, self.action_high, hidden_dims
        )

        self.v_optimizer = optim.Adam(self.vf.parameters(), lr=args.value_learning_rate)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.actor_learning_rate
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
        normalized_obs = self.normalize_obs(obs)
        if deterministic:
            return self.actor.deterministic(normalized_obs)
        return self.actor.sample(normalized_obs)[0]

    def update(self, batch: dict[str, torch.Tensor], global_step: int) -> dict[str, float]:
        # Only (s, a) are used: the frozen Q is the value target, no bootstrap.
        obs = self.normalize_obs(batch["observations"])
        actions = batch["actions"]

        with torch.no_grad():
            q_data = self.qf_frozen(obs, actions)

        # --- Value update: expectile regression toward the frozen Q. ---
        value = self.vf(obs)
        adv = q_data - value
        expectile = float(self.args.iql_expectile)
        expectile_weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        v_loss = (expectile_weight * adv.pow(2)).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        v_grad_norm = nn.utils.clip_grad_norm_(self.vf.parameters(), self.args.max_grad_norm)
        self.v_optimizer.step()

        # --- Policy update: advantage-weighted regression on dataset actions. ---
        actor_loss_value = float("nan")
        weight_mean_value = float("nan")
        log_prob_value = float("nan")
        policy_adv = adv.detach()
        if global_step % self.args.policy_update_interval == 0:
            with torch.no_grad():
                policy_adv = q_data - self.vf(obs)
                weight = torch.exp(self.args.iql_beta * policy_adv).clamp(
                    max=self.args.iql_adv_clip
                )
            log_prob = self.actor.log_prob(obs, actions)
            actor_loss = -(weight * log_prob).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()

            actor_loss_value = float(actor_loss.item())
            weight_mean_value = float(weight.mean().item())
            log_prob_value = float(log_prob.mean().item())

        return {
            "v_loss": float(v_loss.item()),
            "actor_loss": actor_loss_value,
            "awr_weight_mean": weight_mean_value,
            "policy_log_prob": log_prob_value,
            "q_data": float(q_data.mean().item()),
            "v_value": float(value.mean().item()),
            "adv_mean": float(policy_adv.mean().item()),
            "v_grad_norm": float(v_grad_norm),
        }

    def save(self, path: str, global_step: int) -> None:
        path = ensure_parent_dir(path)
        # Full state (incl. frozen Q + obs stats) so eval reloads are self-contained.
        torch.save(
            {
                "agent": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "v_optimizer": self.v_optimizer.state_dict(),
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
            self.v_optimizer.load_state_dict(checkpoint["v_optimizer"])
        return int(checkpoint.get("global_step", 0))

"""Pure Behavior Cloning (BC) for the DexArena tasks.

A self-contained BC baseline: a deterministic MLP policy trained by MSE to
imitate the demonstration actions. Deliberately standalone — it does NOT reuse
the training helpers from the value-based agents; only the network
primitives are shared via ``algo.networks``.

No critic, no target networks, no rewards: BC only consumes (s, a) pairs from the
(successful) demonstrations. The last action dimension is the gripper; it can be
upweighted in the MSE via ``bc_gripper_weight`` when gripper open/close is hard to
fit (default 1.0 = no weighting).
"""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim

from algo.networks import DeterministicActor
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path


class BCAgent(nn.Module):
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
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.learning_rate
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
        # BC's policy is deterministic; `deterministic` is accepted only for a
        # uniform interface with the other agents in this repo.
        return self.actor(self.normalize_obs(obs))

    def update(self, batch: dict[str, torch.Tensor], global_step: int) -> dict[str, float]:
        obs = self.normalize_obs(batch["observations"])
        actions = batch["actions"]

        pred_actions = self.actor(obs)
        sq_error = (pred_actions - actions).pow(2)
        gripper_weight = float(self.args.bc_gripper_weight)
        if self.action_dim > 0 and gripper_weight != 1.0:
            weights = torch.ones(
                self.action_dim, dtype=sq_error.dtype, device=sq_error.device
            )
            weights[-1] = gripper_weight
            bc_loss = (sq_error * weights).mean()
        else:
            bc_loss = sq_error.mean()

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.args.max_grad_norm
        )
        self.actor_optimizer.step()

        with torch.no_grad():
            bc_mse = sq_error.mean()
            action_abs_error = torch.abs(pred_actions - actions).mean()
            gripper_pred = pred_actions[:, -1]
            gripper_demo = actions[:, -1]
            metrics = {
                "bc_loss": float(bc_loss.item()),
                "bc_mse": float(bc_mse.item()),
                "action_abs_error": float(action_abs_error.item()),
                "gripper_abs_error": float(
                    torch.abs(gripper_pred - gripper_demo).mean().item()
                ),
                "gripper_pred_mean": float(gripper_pred.mean().item()),
                "gripper_demo_mean": float(gripper_demo.mean().item()),
                "actor_grad_norm": float(actor_grad_norm),
            }
            if self.action_dim > 1:
                metrics["arm_action_norm"] = float(
                    torch.linalg.norm(pred_actions[:, :-1], dim=-1).mean().item()
                )
                metrics["demo_arm_action_norm"] = float(
                    torch.linalg.norm(actions[:, :-1], dim=-1).mean().item()
                )
        return metrics

    def save(self, path: str, global_step: int) -> None:
        path = ensure_parent_dir(path)
        torch.save(
            {
                "agent": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
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
        return int(checkpoint.get("global_step", 0))

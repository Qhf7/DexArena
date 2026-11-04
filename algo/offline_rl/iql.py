from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.networks import QNetwork, SquashedGaussianActor, ValueNetwork
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path


class IQLAgent(nn.Module):
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
        # IQL keeps no target value network; V is bootstrapped only into Q.
        self.vf = ValueNetwork(self.obs_dim, hidden_dims)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.actor_learning_rate
        )
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=args.critic_learning_rate,
        )
        self.v_optimizer = optim.Adam(
            self.vf.parameters(), lr=args.value_learning_rate
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

    def _bc_terms(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_prob: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred_actions = self.actor.deterministic(obs)
        sq_error = (pred_actions - actions).pow(2)
        if self.action_dim > 0 and self.args.bc_gripper_weight != 1.0:
            weights = torch.ones(self.action_dim, dtype=sq_error.dtype, device=sq_error.device)
            weights[-1] = float(self.args.bc_gripper_weight)
            bc_mse = (sq_error * weights).mean()
        else:
            bc_mse = sq_error.mean()

        if log_prob is None:
            log_prob = self.actor.log_prob(obs, actions)
        bc_nll = -log_prob.mean()
        bc_loss = bc_nll if self.args.bc_loss_type == "nll" else bc_mse

        with torch.no_grad():
            action_abs_error = torch.abs(pred_actions - actions).mean()
            pred_norm = torch.linalg.norm(pred_actions, dim=-1).mean()
            demo_norm = torch.linalg.norm(actions, dim=-1).mean()
            metrics = {
                "actor_bc_term": float(bc_loss.item()),
                "actor_bc_mse": float(bc_mse.item()),
                "actor_bc_nll": float(bc_nll.item()),
                "actor_action_abs_error": float(action_abs_error.item()),
                "actor_action_norm": float(pred_norm.item()),
                "actor_demo_action_norm": float(demo_norm.item()),
            }
            if self.action_dim > 1:
                pred_arm_norm = torch.linalg.norm(pred_actions[:, :-1], dim=-1).mean()
                demo_arm_norm = torch.linalg.norm(actions[:, :-1], dim=-1).mean()
                metrics.update(
                    {
                        "actor_arm_action_norm": float(pred_arm_norm.item()),
                        "actor_demo_arm_action_norm": float(demo_arm_norm.item()),
                    }
                )
            if self.action_dim > 0:
                gripper_pred = pred_actions[:, -1]
                gripper_demo = actions[:, -1]
                metrics.update(
                    {
                        "actor_gripper_abs_error": float(
                            torch.abs(gripper_pred - gripper_demo).mean().item()
                        ),
                        "actor_gripper_pred_mean": float(gripper_pred.mean().item()),
                        "actor_gripper_demo_mean": float(gripper_demo.mean().item()),
                    }
                )
        return bc_loss, metrics

    def update(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
        actor_batch: dict[str, torch.Tensor] | None = None,
        bc_anchor_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        obs = self.normalize_obs(batch["observations"])
        next_obs = self.normalize_obs(batch["next_observations"])
        actions = batch["actions"]
        rewards = batch["rewards"] * self.args.reward_scale + self.args.reward_bias
        dones = batch["dones"]
        is_extra = batch.get("is_extra")

        # --- Value update: expectile regression toward min target-Q(s, a). ---
        with torch.no_grad():
            q_target = torch.minimum(
                self.qf1_target(obs, actions),
                self.qf2_target(obs, actions),
            )
        value = self.vf(obs)
        adv = q_target - value
        expectile = float(self.args.iql_expectile)
        # |tau - 1{adv<0}| * adv^2: upweights positive advantages when tau>0.5.
        expectile_weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        v_elem = expectile_weight * adv.pow(2)
        # V is the value of the data (behavior) policy; counterfactual branch
        # actions must not redefine it. Mixing low branch-Q into the expectile
        # regression at demo states drags V down and collapses the return scale,
        # so by default fit V on demo rows only.
        if is_extra is not None and self.args.value_demo_only:
            v_mask = (~is_extra).to(v_elem.dtype).view_as(v_elem)
            v_loss = (v_elem * v_mask).sum() / v_mask.sum().clamp_min(1.0)
        else:
            v_loss = v_elem.mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        v_grad_norm = nn.utils.clip_grad_norm_(self.vf.parameters(), self.args.max_grad_norm)
        self.v_optimizer.step()

        # --- Q update: bootstrap with V(s') only (never queries OOD actions). ---
        with torch.no_grad():
            next_value = self.vf(next_obs)
            not_done = 1.0 - dones
            if is_extra is not None and not self.args.bootstrap_extra_transitions:
                not_done = not_done * (~is_extra).float().view_as(not_done)
            target_q = rewards + not_done * self.args.gamma * next_value
            # When branches DO bootstrap (so their Q lands on the same return
            # scale as V and can earn a positive advantage), clamp their target
            # to cap any V(s') extrapolation on off-manifold branch next states.
            # Branches are excluded from the V regression above, so this target
            # cannot feed back into V and re-trigger the runaway it once caused.
            if (
                is_extra is not None
                and self.args.bootstrap_extra_transitions
                and self.args.extra_target_q_clip is not None
            ):
                extra = is_extra.view_as(target_q)
                clipped = target_q.clamp(max=float(self.args.extra_target_q_clip))
                target_q = torch.where(extra, clipped, target_q)
        q1_data = self.qf1(obs, actions)
        q2_data = self.qf2(obs, actions)
        q1_loss = F.mse_loss(q1_data, target_q)
        q2_loss = F.mse_loss(q2_data, target_q)
        q_loss = q1_loss + q2_loss
        demo_mask = None
        extra_mask = None
        if is_extra is not None:
            extra_mask = is_extra.view_as(target_q)
            demo_mask = ~extra_mask

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = nn.utils.clip_grad_norm_(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            self.args.max_grad_norm,
        )
        self.q_optimizer.step()

        # --- Policy update: advantage-weighted regression on dataset actions. ---
        actor_loss_value = float("nan")
        awr_loss_value = float("nan")
        actor_bc_metrics = {}
        weight_mean_value = float("nan")
        log_prob_value = float("nan")
        awr_pos_frac_value = float("nan")
        branch_imitated_frac_value = float("nan")
        branch_weight_mean_value = float("nan")
        demo_weight_mean_value = float("nan")
        policy_adv = adv.detach()
        if global_step % self.args.policy_update_interval == 0:
            policy_batch = actor_batch if actor_batch is not None else batch
            policy_obs = self.normalize_obs(policy_batch["observations"])
            policy_actions = policy_batch["actions"]
            policy_is_extra = policy_batch.get("is_extra")
            with torch.no_grad():
                policy_q = torch.minimum(
                    self.qf1_target(policy_obs, policy_actions),
                    self.qf2_target(policy_obs, policy_actions),
                )
                policy_adv = policy_q - self.vf(policy_obs)
                weight = torch.exp(self.args.iql_beta * policy_adv).clamp(
                    max=self.args.iql_adv_clip
                )
                # Only imitate advantage-improving actions. This is what lets the
                # decisive gripper-close branches (positive advantage) enter the
                # policy while random/bad branch perturbations (adv<=0) drop out.
                pos_mask = (policy_adv > 0).to(weight.dtype)
                if self.args.awr_positive_adv_only:
                    weight = weight * pos_mask
                if self.args.normalize_awr_weights:
                    weight = weight / weight.mean().clamp_min(1e-6)
            log_prob = self.actor.log_prob(policy_obs, policy_actions)
            awr_loss = -(weight * log_prob).mean()

            if bc_anchor_batch is not None:
                anchor_obs = self.normalize_obs(bc_anchor_batch["observations"])
                bc_loss, actor_bc_metrics = self._bc_terms(
                    anchor_obs, bc_anchor_batch["actions"]
                )
            else:
                # The BC anchor must not imitate the perturbed branch actions, or it
                # degrades into noisy BC. When the actor batch contains branch rows,
                # restrict the anchor to demo rows (is_extra==False).
                if (
                    self.args.bc_demo_only
                    and policy_is_extra is not None
                    and bool(policy_is_extra.any())
                ):
                    demo_rows = (~policy_is_extra.bool()).view(-1)
                    if bool(demo_rows.any()):
                        bc_loss, actor_bc_metrics = self._bc_terms(
                            policy_obs[demo_rows], policy_actions[demo_rows]
                        )
                    else:
                        bc_loss = policy_obs.new_zeros(())
                        actor_bc_metrics = {}
                else:
                    bc_loss, actor_bc_metrics = self._bc_terms(
                        policy_obs, policy_actions, log_prob=log_prob
                    )

            if global_step <= self.args.bc_warmup_steps:
                actor_loss = bc_loss
            else:
                actor_loss = awr_loss + self.args.actor_bc_weight * bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()

            actor_loss_value = float(actor_loss.item())
            awr_loss_value = float(awr_loss.item())
            weight_mean_value = float(weight.mean().item())
            log_prob_value = float(log_prob.mean().item())
            with torch.no_grad():
                awr_pos_frac_value = float(pos_mask.mean().item())
                flat_weight = weight.view(-1)
                if policy_is_extra is not None:
                    extra_rows = policy_is_extra.bool().view(-1)
                    pos_rows = (policy_adv > 0).view(-1)
                    n_extra = int(extra_rows.sum().item())
                    if n_extra > 0:
                        branch_imitated_frac_value = float(
                            (extra_rows & pos_rows).float().sum().item() / n_extra
                        )
                        branch_weight_mean_value = float(
                            flat_weight[extra_rows].mean().item()
                        )
                    demo_rows_all = ~extra_rows
                    if bool(demo_rows_all.any()):
                        demo_weight_mean_value = float(
                            flat_weight[demo_rows_all].mean().item()
                        )

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

        metrics = {
            "v_loss": float(v_loss.item()),
            "q_loss": float(q_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": actor_loss_value,
            "actor_awr_loss": awr_loss_value,
            "awr_weight_mean": weight_mean_value,
            "awr_pos_frac": awr_pos_frac_value,
            "branch_imitated_frac": branch_imitated_frac_value,
            "branch_weight_mean": branch_weight_mean_value,
            "demo_weight_mean": demo_weight_mean_value,
            "policy_log_prob": log_prob_value,
            "v_value": float(value.mean().item()),
            "adv_mean": float(policy_adv.mean().item()),
            "q1_data": float(q1_data.mean().item()),
            "q2_data": float(q2_data.mean().item()),
            "target_q": float(target_q.mean().item()),
            "target_q_abs_max": float(target_q.abs().max().item()),
            "critic_extra_frac": (
                float(is_extra.float().mean().item()) if is_extra is not None else 0.0
            ),
            "v_grad_norm": float(v_grad_norm),
            "q_grad_norm": float(q_grad_norm),
        }
        metrics.update(actor_bc_metrics)
        if demo_mask is not None and bool(demo_mask.any().item()):
            demo_td = (q1_data[demo_mask] - target_q[demo_mask]).pow(2)
            metrics.update(
                {
                    "demo_reward": float(rewards[demo_mask].mean().item()),
                    "demo_target_q": float(target_q[demo_mask].mean().item()),
                    "demo_q1_data": float(q1_data[demo_mask].mean().item()),
                    "demo_q1_loss": float(demo_td.mean().item()),
                }
            )
        if extra_mask is not None and bool(extra_mask.any().item()):
            extra_td = (q1_data[extra_mask] - target_q[extra_mask]).pow(2)
            metrics.update(
                {
                    "extra_reward": float(rewards[extra_mask].mean().item()),
                    "extra_target_q": float(target_q[extra_mask].mean().item()),
                    "extra_q1_data": float(q1_data[extra_mask].mean().item()),
                    "extra_q1_loss": float(extra_td.mean().item()),
                }
            )
        return metrics

    def update_bc(self, batch: dict[str, torch.Tensor], global_step: int) -> dict[str, float]:
        obs = self.normalize_obs(batch["observations"])
        actions = batch["actions"]

        bc_loss, bc_metrics = self._bc_terms(obs, actions)

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.args.max_grad_norm
        )
        self.actor_optimizer.step()

        return {
            "bc_loss": float(bc_loss.item()),
            "bc_mse": bc_metrics["actor_bc_mse"],
            "bc_action_abs_error": bc_metrics["actor_action_abs_error"],
            "bc_gripper_abs_error": bc_metrics.get("actor_gripper_abs_error", float("nan")),
            "actor_grad_norm": float(actor_grad_norm),
        }

    def save(self, path: str, global_step: int) -> None:
        path = ensure_parent_dir(path)
        torch.save(
            {
                "agent": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
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
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            self.v_optimizer.load_state_dict(checkpoint["v_optimizer"])
        return int(checkpoint.get("global_step", 0))

"""OPT (Online Pre-Training) for offline-to-online RL, with an IQL backbone.

Implements the three-stage scheme of Shin et al., *Online Pre-Training for
Offline-to-Online Reinforcement Learning* (ICML 2025, arXiv:2507.08387),
instantiated on IQL (their Section 4.2 "Extending OPT to IQL"):

  1. Offline pre-training  -- done elsewhere (``algo.offline_rl.iql`` / ``train_iql``).
  2. Online pre-training    -- a *fresh* value function ``Q^on`` (twin Q + V) is
     trained while the policy and the offline value function ``Q^off`` stay
     frozen, using the one-step meta-adaptation objective (paper Eq. 3):
         L(psi) = L_off(psi) + L_on(psi - meta_lr * grad L_off(psi))
  3. Online fine-tuning     -- both value functions and the policy are updated.
     The AWR policy weight uses a kappa-scheduled *combined* advantage
         A = (1 - kappa) * A^off + kappa * A^on
     so early updates lean on the stable ``Q^off`` and later ones on the
     faster-adapting ``Q^on``.

This module is self-contained: it only borrows the network *primitives*
(``QNetwork`` / ``SquashedGaussianActor`` from ``algo.networks``) and defines its own
``ValueNetwork`` and all IQL/OPT update math here, rather than importing
``IQLAgent``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call

from algo.networks import QNetwork, SquashedGaussianActor, build_mlp
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.network = build_mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


# The submodule names of the online value function whose parameters take the
# meta-adaptation (virtual) gradient step. Order is fixed for stable zipping.
_ON_MODULES = ("qf_on1", "qf_on2", "vf_on")


class OPTAgent(nn.Module):
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

        # Offline value function Q^off / V^off (warm-started from an IQL ckpt).
        self.qf_off1 = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf_off2 = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf_off1_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf_off2_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.vf_off = ValueNetwork(self.obs_dim, hidden_dims)

        # Online value function Q^on / V^on (fresh, random init -- no offline bias).
        self.qf_on1 = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf_on2 = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf_on1_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.qf_on2_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims)
        self.vf_on = ValueNetwork(self.obs_dim, hidden_dims)

        for online, target in self._target_pairs():
            target.load_state_dict(online.state_dict())
            target.requires_grad_(False)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.actor_learning_rate
        )
        self.q_off_optimizer = optim.Adam(
            list(self.qf_off1.parameters()) + list(self.qf_off2.parameters()),
            lr=args.critic_learning_rate,
        )
        self.v_off_optimizer = optim.Adam(
            self.vf_off.parameters(), lr=args.value_learning_rate
        )
        # A single optimizer over {qf_on1, qf_on2, vf_on}: the meta-adaptation
        # step takes one joint virtual gradient step over exactly these params.
        self.on_optimizer = optim.Adam(
            self._on_parameters(), lr=args.critic_learning_rate
        )

        writer = SummaryWriter(runs_path(run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(f"|{key}|{value}|" for key, value in asdict(args).items()),
        )
        self.logger = Logger(log_swanlab=args.track, tensorboard=writer)

    # ------------------------------------------------------------------ utils
    def _off_target_pairs(self):
        return (
            (self.qf_off1, self.qf_off1_target),
            (self.qf_off2, self.qf_off2_target),
        )

    def _on_target_pairs(self):
        return (
            (self.qf_on1, self.qf_on1_target),
            (self.qf_on2, self.qf_on2_target),
        )

    def _target_pairs(self):
        return self._off_target_pairs() + self._on_target_pairs()

    def _on_modules(self):
        return tuple(getattr(self, name) for name in _ON_MODULES)

    def _on_parameters(self):
        params = []
        for module in self._on_modules():
            params += list(module.parameters())
        return params

    def _on_named_params(self) -> list[tuple[str, str, torch.Tensor]]:
        named = []
        for mname in _ON_MODULES:
            for pname, p in getattr(self, mname).named_parameters():
                named.append((mname, pname, p))
        return named

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.args.normalize_observations:
            return obs
        return (obs - self.obs_mean) / self.obs_std

    @torch.no_grad()
    def select_action(
        self, obs: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        normalized_obs = self.normalize_obs(obs)
        if deterministic:
            return self.actor.deterministic(normalized_obs)
        return self.actor.sample(normalized_obs)[0]

    def _soft_update_targets(self, pairs=None) -> None:
        pairs = self._target_pairs() if pairs is None else pairs
        with torch.no_grad():
            for online, target in pairs:
                for op, tp in zip(online.parameters(), target.parameters()):
                    tp.data.lerp_(op.data, self.args.tau)

    def _bc_weight(self, step: int) -> float:
        weight = float(self.args.actor_bc_weight)
        if weight <= 0.0:
            return 0.0
        if not self.args.bc_decay:
            return weight
        progress = min(max(step / max(1, self.args.bc_decay_steps), 0.0), 1.0)
        return max(float(self.args.actor_bc_min_weight), weight * (1.0 - progress))

    def _bc_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Gripper-weighted MSE anchor toward demo actions (matches IQL)."""
        pred = self.actor.deterministic(obs)
        sq_error = (pred - actions).pow(2)
        if self.action_dim > 0 and self.args.bc_gripper_weight != 1.0:
            weights = torch.ones(
                self.action_dim, dtype=sq_error.dtype, device=sq_error.device
            )
            weights[-1] = float(self.args.bc_gripper_weight)
            return (sq_error * weights).mean()
        return sq_error.mean()

    # ------------------------------------------------------- IQL critic losses
    def _critic_losses(
        self,
        q1_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        q2_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        v_fn: Callable[[torch.Tensor], torch.Tensor],
        q1t_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        q2t_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IQL value (expectile) + Q (TD with V-bootstrap) losses.

        The forwards are passed as callables so the same math serves both the
        usual module forward and the functional (meta-adapted-params) forward.
        Targets are detached, exactly like IQL: V regresses to the detached
        min target-Q, and the Q target bootstraps the detached V(s').
        """
        with torch.no_grad():
            q_target = torch.minimum(q1t_fn(obs, actions), q2t_fn(obs, actions))
        value = v_fn(obs)
        adv = q_target - value
        expectile = float(self.args.iql_expectile)
        weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        v_loss = (weight * adv.pow(2)).mean()

        with torch.no_grad():
            next_value = v_fn(next_obs)
            target_q = rewards + (1.0 - dones) * self.args.gamma * next_value
        q_loss = F.mse_loss(q1_fn(obs, actions), target_q) + F.mse_loss(
            q2_fn(obs, actions), target_q
        )
        return v_loss, q_loss

    def _scaled(self, batch: dict[str, torch.Tensor]):
        obs = self.normalize_obs(batch["observations"])
        next_obs = self.normalize_obs(batch["next_observations"])
        actions = batch["actions"]
        rewards = batch["rewards"] * self.args.reward_scale + self.args.reward_bias
        dones = batch["dones"]
        return obs, next_obs, actions, rewards, dones

    # ----------------------------------------------- Phase B: online pre-train
    def online_pretrain_update(
        self, batch_off: dict[str, torch.Tensor], batch_on: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Train Q^on/V^on only, via the Eq. 3 one-step meta-adaptation.

        Policy and Q^off are untouched. ``first_order=True`` drops the
        second-order term (FOMAML) to save memory/compute.
        """
        off = self._scaled(batch_off)
        on = self._scaled(batch_on)

        # --- L_off(psi): inner loss on offline data with current online params.
        v_off, q_off = self._critic_losses(
            self.qf_on1, self.qf_on2, self.vf_on,
            self.qf_on1_target, self.qf_on2_target,
            *off,
        )
        l_off = v_off + q_off

        named = self._on_named_params()
        params = [p for (_, _, p) in named]
        create_graph = not bool(self.args.first_order)
        grads = torch.autograd.grad(
            l_off, params, create_graph=create_graph, retain_graph=True
        )

        # --- Virtual step psi' = psi - meta_lr * grad, split back per module.
        meta_lr = float(self.args.meta_lr)
        adapted: dict[str, dict[str, torch.Tensor]] = {m: {} for m in _ON_MODULES}
        for (mname, pname, p), g in zip(named, grads):
            adapted[mname][pname] = p - meta_lr * g

        def fcall(mname: str):
            module = getattr(self, mname)
            return lambda *inp: functional_call(module, adapted[mname], tuple(inp))

        # --- L_on(psi'): outer loss on online data at the adapted params.
        v_on, q_on = self._critic_losses(
            fcall("qf_on1"), fcall("qf_on2"), fcall("vf_on"),
            self.qf_on1_target, self.qf_on2_target,
            *on,
        )
        l_on = v_on + q_on

        l_meta = l_off + l_on
        self.on_optimizer.zero_grad()
        l_meta.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self._on_parameters(), self.args.max_grad_norm
        )
        self.on_optimizer.step()
        # Phase B keeps Q^off (and its target) frozen -- only Q^on targets move.
        self._soft_update_targets(self._on_target_pairs())

        return {
            "opt_meta_loss": float(l_meta.item()),
            "opt_v_loss_off": float(v_off.item()),
            "opt_q_loss_off": float(q_off.item()),
            "opt_v_loss_on": float(v_on.item()),
            "opt_q_loss_on": float(q_on.item()),
            "opt_grad_norm": float(grad_norm),
        }

    # ------------------------------------------- a single IQL critic-set update
    def _critic_update(
        self,
        prefix: str,
        obs,
        next_obs,
        actions,
        rewards,
        dones,
        q_opt: optim.Optimizer,
        v_opt: optim.Optimizer,
    ) -> dict[str, float]:
        qf1 = getattr(self, f"qf_{prefix}1")
        qf2 = getattr(self, f"qf_{prefix}2")
        vf = getattr(self, f"vf_{prefix}")
        qf1_t = getattr(self, f"qf_{prefix}1_target")
        qf2_t = getattr(self, f"qf_{prefix}2_target")

        v_loss, _ = self._critic_losses(
            qf1, qf2, vf, qf1_t, qf2_t, obs, next_obs, actions, rewards, dones
        )
        v_opt.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(vf.parameters(), self.args.max_grad_norm)
        v_opt.step()

        # Recompute Q loss after the V step so its bootstrap uses the fresh V.
        _, q_loss = self._critic_losses(
            qf1, qf2, vf, qf1_t, qf2_t, obs, next_obs, actions, rewards, dones
        )
        q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(qf1.parameters()) + list(qf2.parameters()), self.args.max_grad_norm
        )
        q_opt.step()
        return {
            f"{prefix}_v_loss": float(v_loss.item()),
            f"{prefix}_q_loss": float(q_loss.item()),
        }

    def _advantage(self, prefix: str, obs, actions) -> torch.Tensor:
        qf1_t = getattr(self, f"qf_{prefix}1_target")
        qf2_t = getattr(self, f"qf_{prefix}2_target")
        vf = getattr(self, f"vf_{prefix}")
        q = torch.minimum(qf1_t(obs, actions), qf2_t(obs, actions))
        return q - vf(obs)

    # ----------------------------------------------- Phase C: online fine-tune
    def finetune_update(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
        kappa: float,
        demo_batch: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, float]:
        """Update BOTH value functions and the policy; AWR uses combined adv."""
        obs, next_obs, actions, rewards, dones = self._scaled(batch)

        metrics: dict[str, float] = {}
        metrics.update(
            self._critic_update(
                "off", obs, next_obs, actions, rewards, dones,
                self.q_off_optimizer, self.v_off_optimizer,
            )
        )
        # The online critic set shares one optimizer (zero/backward/step twice;
        # the disjoint V/Q grads make this equivalent to two optimizers).
        metrics.update(
            self._critic_update(
                "on", obs, next_obs, actions, rewards, dones,
                self.on_optimizer, self.on_optimizer,
            )
        )

        actor_loss_value = float("nan")
        awr_loss_value = float("nan")
        bc_term_value = float("nan")
        bc_weight_value = 0.0
        weight_mean_value = float("nan")
        adv_value = float("nan")
        if global_step % self.args.policy_update_interval == 0:
            with torch.no_grad():
                adv_off = self._advantage("off", obs, actions)
                adv_on = self._advantage("on", obs, actions)
                adv = (1.0 - kappa) * adv_off + kappa * adv_on
                weight = torch.exp(self.args.iql_beta * adv).clamp(
                    max=self.args.iql_adv_clip
                )
                if self.args.awr_positive_adv_only:
                    weight = weight * (adv > 0).to(weight.dtype)
                if self.args.normalize_awr_weights:
                    weight = weight / weight.mean().clamp_min(1e-6)
            log_prob = self.actor.log_prob(obs, actions)
            awr_loss = -(weight * log_prob).mean()

            actor_loss = awr_loss
            bc_weight_value = self._bc_weight(global_step)
            if bc_weight_value > 0.0 and demo_batch is not None:
                demo_obs = self.normalize_obs(demo_batch["observations"])
                bc_term = self._bc_loss(demo_obs, demo_batch["actions"])
                actor_loss = actor_loss + bc_weight_value * bc_term
                bc_term_value = float(bc_term.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()

            actor_loss_value = float(actor_loss.item())
            awr_loss_value = float(awr_loss.item())
            weight_mean_value = float(weight.mean().item())
            adv_value = float(adv.mean().item())

        self._soft_update_targets()

        metrics.update(
            {
                "kappa": float(kappa),
                "actor_loss": actor_loss_value,
                "actor_awr_loss": awr_loss_value,
                "awr_weight_mean": weight_mean_value,
                "adv_mean": adv_value,
                "bc_term": bc_term_value,
                "bc_weight": bc_weight_value,
            }
        )
        return metrics

    # ------------------------------------------------------------- checkpoints
    def save(self, path: str, global_step: int) -> None:
        path = ensure_parent_dir(path)
        torch.save(
            {
                "agent": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_off_optimizer": self.q_off_optimizer.state_dict(),
                "v_off_optimizer": self.v_off_optimizer.state_dict(),
                "on_optimizer": self.on_optimizer.state_dict(),
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
            self.q_off_optimizer.load_state_dict(checkpoint["q_off_optimizer"])
            self.v_off_optimizer.load_state_dict(checkpoint["v_off_optimizer"])
            self.on_optimizer.load_state_dict(checkpoint["on_optimizer"])
        return int(checkpoint.get("global_step", 0))

    def load_pretrained(self, path: str) -> int:
        """Warm-start the offline value function + policy from an IQL ckpt.

        Remaps IQL's ``qf1/qf2/vf`` (and targets) onto this agent's
        ``qf_off1/qf_off2/vf_off``; ``actor`` and the obs-norm buffers transfer
        directly. ``Q^on/V^on`` keep their fresh random init.
        """
        checkpoint = torch.load(path, map_location=self.obs_mean.device)
        state = checkpoint["agent"] if "agent" in checkpoint else checkpoint
        rename = {
            "qf1.": "qf_off1.",
            "qf2.": "qf_off2.",
            "qf1_target.": "qf_off1_target.",
            "qf2_target.": "qf_off2_target.",
            "vf.": "vf_off.",
        }
        remapped = {}
        for key, value in state.items():
            new_key = key
            for old, new in rename.items():
                if key.startswith(old):
                    new_key = new + key[len(old):]
                    break
            remapped[new_key] = value

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        # Q^on/V^on are expected to be "missing" (kept at init); flag anything else.
        on_prefixes = ("qf_on1.", "qf_on2.", "qf_on1_target.", "qf_on2_target.", "vf_on.")
        missing_critical = [
            k for k in missing if not k.startswith(on_prefixes)
        ]
        if missing_critical or unexpected:
            print(
                "[OPT warm-start] non-trivial key mismatch — "
                f"missing={missing_critical}, unexpected={unexpected}"
            )
        else:
            print(
                "[OPT warm-start] loaded actor + Q^off (twin Q + targets) + V^off "
                f"+ obs stats from {path} (Q^on/V^on kept fresh)."
            )
        return int(checkpoint.get("global_step", 0))

from __future__ import annotations

import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algo.diffusion import (
    EtaFixed,
    Normalizer,
    VPGDiffusion,
    build_backbone,
    build_critic,
)
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path

log = logging.getLogger(__name__)

class PPODiffusion(VPGDiffusion):
    """PPO over the denoising chain (port of diffusion_ppo.py)."""

    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef=None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_adv = norm_adv
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.gamma_denoising = gamma_denoising
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

    def loss(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss: bool = False,
        reward_horizon: int = 4,
        chunk_start: int = 0,
    ):
        newlogprobs, eta = self.get_logprobs(obs, chains, get_ent=True)
        entropy_loss = -eta.mean()
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)

        newlogprobs = newlogprobs[:, chunk_start : chunk_start + reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :, chunk_start : chunk_start + reward_horizon, :]
        newlogprobs = newlogprobs.mean(dim=(-1, -2)).view(-1)
        oldlogprobs = oldlogprobs.mean(dim=(-1, -2)).view(-1)

        if use_bc_loss:
            with torch.no_grad():
                base_samples = self.forward(
                    cond=obs, deterministic=False, return_chain=True, use_base_policy=True
                )
            bc_logprobs = self.get_logprobs(obs, base_samples.chains)
            bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
            bc_loss = -bc_logprobs.mean()
        else:
            bc_loss = 0.0

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantage_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        advantage_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=advantage_min, max=advantage_max)

        advantages = advantages.repeat_interleave(self.ft_denoising_steps)
        discount = torch.tensor(
            [self.gamma_denoising**i for i in reversed(range(self.ft_denoising_steps))]
        ).to(self.device)
        discount = discount.repeat(len(advantages) // self.ft_denoising_steps)
        advantages = advantages * discount

        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        t = torch.arange(self.ft_denoising_steps).float().to(self.device) / max(self.ft_denoising_steps - 1, 1)
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (math.exp(self.clip_ploss_coef_rate) - 1)
        else:
            clip_ploss_coef = torch.tensor([self.clip_ploss_coef]).to(self.device)
        clip_ploss_coef = clip_ploss_coef.repeat(len(advantages) // self.ft_denoising_steps)

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_ploss_coef).float().mean().item()

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalues = self.critic(obs).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(newvalues - oldvalues, -self.clip_vloss_coef, self.clip_vloss_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        return (pg_loss, entropy_loss, v_loss, clipfrac, approx_kl.item(), ratio.mean().item(), bc_loss, eta.mean().item())


# =====================================================================================
# PPO fine-tuning agent
# =====================================================================================


class DPPOAgent(nn.Module):
    """PPO fine-tuning of the diffusion denoising chain (mirrors PPOAgent)."""

    def __init__(self, args, *, obs_dim: int, action_dim: int, normalizer_stats: dict, run_name: str, device):
        super().__init__()
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.backbone = getattr(args, "backbone", "mlp")

        self.normalizer = Normalizer(
            normalizer_stats, obs_mode=args.obs_normalization_mode, action_mode=args.action_normalization_mode
        ).to(device)

        actor = build_backbone(args, obs_dim=obs_dim, action_dim=action_dim)
        critic = build_critic(args, obs_dim=obs_dim)
        eta = EtaFixed(base_eta=args.eta_base) if args.use_ddim else None
        self.model = PPODiffusion(
            actor=actor,
            critic=critic,
            ft_denoising_steps=args.ft_denoising_steps,
            min_sampling_denoising_std=args.min_sampling_denoising_std,
            min_logprob_denoising_std=args.min_logprob_denoising_std,
            eta=eta,
            learn_eta=args.learn_eta,
            gamma_denoising=args.gamma_denoising,
            clip_ploss_coef=args.clip_ploss_coef,
            clip_ploss_coef_base=args.clip_ploss_coef_base,
            clip_ploss_coef_rate=args.clip_ploss_coef_rate,
            clip_vloss_coef=args.clip_vloss_coef,
            norm_adv=args.norm_adv,
            horizon_steps=args.horizon,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=str(device),
            denoising_steps=args.denoising_steps,
            predict_epsilon=args.predict_epsilon,
            use_ddim=args.use_ddim,
            ddim_steps=args.ddim_steps,
            final_action_clip_value=args.final_action_clip_value,
        ).to(device)

        # Separate optimizers for actor (fine-tuned copy) and critic
        self.actor_optimizer = optim.AdamW(
            self.model.actor_ft.parameters(), lr=args.actor_lr, weight_decay=args.actor_weight_decay
        )
        self.critic_optimizer = optim.AdamW(
            self.model.critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_weight_decay
        )

        writer = SummaryWriter(runs_path(run_name))
        self.logger = Logger(log_swanlab=args.track, tensorboard=writer)

    # ---- checkpoint loading (from BC pretrain) ----
    def load_base_policy(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        ckpt_backbone = ckpt.get("backbone", "mlp")
        if ckpt_backbone != self.backbone:
            raise ValueError(
                f"Base-policy backbone mismatch: checkpoint is '{ckpt_backbone}' but this "
                f"DPPO agent uses '{self.backbone}'. Re-run fine-tuning with --backbone "
                f"{ckpt_backbone} (or pretrain with --backbone {self.backbone})."
            )
        ckpt_k = ckpt.get("denoising_steps")
        if ckpt_k is not None and int(ckpt_k) != int(self.args.denoising_steps):
            raise ValueError(
                f"Base-policy denoising_steps mismatch: checkpoint was trained with K={ckpt_k} "
                f"but this DPPO agent samples with K={self.args.denoising_steps}. Sampling with "
                f"a different K silently produces garbage actions — pretrain the base with "
                f"--denoising_steps {self.args.denoising_steps} or fine-tune with --denoising_steps {ckpt_k}."
            )
        # Prefer the EMA-averaged weights (match BC eval; better base) when available;
        # fall back to raw for old checkpoints without EMA. Keys are identical since
        # ema_model is a deepcopy of the BC model.
        state = ckpt["model"]
        src = "model(raw)"
        if getattr(self.args, "use_ema_base", True) and ckpt.get("ema_model") is not None:
            state = ckpt["ema_model"]
            src = "ema_model"
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self.model.actor_ft = copy.deepcopy(self.model.actor).to(self.device)
        for param in self.model.actor_ft.parameters():
            param.requires_grad_(True)
        self.actor_optimizer = optim.AdamW(
            self.model.actor_ft.parameters(), lr=self.args.actor_lr, weight_decay=self.args.actor_weight_decay
        )
        if "normalizer" in ckpt:
            self.normalizer.load_state_dict(ckpt["normalizer"])
        log.info("Loaded base policy from %s [%s] (missing=%d unexpected=%d)", path, src, len(missing), len(unexpected))

    # ---- rollout-time helpers ----
    def normalize_cond(self, cond: dict) -> dict:
        return {"state": self.normalizer.normalize_obs(cond["state"])}

    @torch.no_grad()
    def act(self, cond_raw: dict, *, deterministic: bool = False):
        cond = self.normalize_cond(cond_raw)
        samples = self.model(cond, deterministic=deterministic, return_chain=True)
        chains = samples.chains  # (B, K_ft+1, horizon, Da) normalized
        value = self.model.critic(cond).view(-1)
        logprob = self.model.get_logprobs(cond, chains)  # (B*K_ft, horizon, Da)
        B = chains.shape[0]
        logprob = logprob.reshape(B, self.model.ft_denoising_steps, self.args.horizon, self.action_dim)
        start = self.args.n_obs_steps - 1
        action_norm = samples.trajectories[:, start : start + self.args.n_action_steps]  # (B, Ta_exec, Da)
        action_env = self.normalizer.unnormalize_action(action_norm)
        return {
            "action_env": action_env,
            "chains": chains,
            "logprob": logprob,
            "value": value,
        }

    @torch.no_grad()
    def sample_action_chunk(self, cond_raw: dict, *, deterministic: bool = True) -> torch.Tensor:
        cond = self.normalize_cond(cond_raw)
        samples = self.model(cond, deterministic=deterministic, return_chain=False)
        start = self.args.n_obs_steps - 1  # see `act`: skip the To-1 stale head of the horizon
        action_norm = samples.trajectories[:, start : start + self.args.n_action_steps]
        return self.normalizer.unnormalize_action(action_norm)

    @torch.no_grad()
    def get_value(self, cond_raw: dict) -> torch.Tensor:
        cond = self.normalize_cond(cond_raw)
        return self.model.critic(cond).view(-1)

    def update(self, buffer: dict) -> dict:
        args = self.args
        N = buffer["returns"].shape[0]
        inds = np.arange(N)
        minibatch = max(1, N // args.num_minibatches)

        metrics = {}
        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, N, minibatch):
                mb = inds[start : start + minibatch]
                mb_t = torch.as_tensor(mb, device=self.device)
                obs = {"state": self.normalizer.normalize_obs(buffer["obs_state"][mb_t])}
                pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, eta_mean = self.model.loss(
                    obs,
                    buffer["chains"][mb_t],
                    buffer["returns"][mb_t],
                    buffer["values"][mb_t],
                    buffer["advantages"][mb_t],
                    buffer["logprob"][mb_t],
                    use_bc_loss=args.use_bc_loss,
                    reward_horizon=args.n_action_steps,
                    chunk_start=args.n_obs_steps - 1,
                )
                loss = (
                    pg_loss
                    + args.vf_coef * v_loss
                    + args.ent_coef * entropy_loss
                    + args.bc_coef * bc_loss
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), args.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                metrics = {
                    "losses/pg_loss": float(pg_loss.item()),
                    "losses/v_loss": float(v_loss.item()),
                    "losses/entropy": float(entropy_loss.item()),
                    "losses/approx_kl": float(approx_kl),
                    "losses/clipfrac": float(clipfrac),
                    "losses/ratio": float(ratio),
                    "losses/bc_loss": float(bc_loss) if args.use_bc_loss else 0.0,
                }
            if args.target_kl is not None and metrics.get("losses/approx_kl", 0.0) > args.target_kl:
                break
        return metrics

    def save(self, path: str, *, iteration: int, global_step: int):
        path = ensure_parent_dir(path)
        payload = {
            "model": self.model.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "backbone": self.backbone,
            "iteration": int(iteration),
            "global_step": int(global_step),
        }
        torch.save(payload, path)
        print(f"DPPO checkpoint saved to {path}")

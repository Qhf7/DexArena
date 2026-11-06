"""RLPD from pixels: the state-based RLPD agent with a visual encoder.

Observation path::

    images --(random shift)--> encoder --> z ; concat([z, proprio]) --> actor/critic

The encoder is trained by the critic loss only (the actor consumes
``z.detach()``), ``next_obs`` is encoded under ``no_grad``, and augmentation is
applied to the whole batch. Only the critic MLP has a Polyak target (DrQ-v2
style); there is no target encoder.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.encoders import EncoderConfig, VisualEncoder, augment_obs, build_encoder

# `RunningMeanStd` lives in algo/networks.py; re-exported here for callers.
from algo.networks import RunningMeanStd, SquashedGaussianActor
from algo.offline_to_online_rl.rlpd import EnsembleCritic
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path
from utils.pixel_cameras import PixelObsConfig


class RLPDPixelAgent(nn.Module):
    def __init__(
        self,
        args,
        *,
        pixel_config: PixelObsConfig,
        encoder_config: EncoderConfig,
        proprio_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        run_name: str,
    ):
        super().__init__()
        self.args = args
        self.pixel_config = pixel_config
        self.encoder_config = encoder_config.validate()
        self.camera_keys = pixel_config.camera_keys
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_low.numel())
        self.register_buffer("action_low", action_low.float())
        self.register_buffer("action_high", action_high.float())

        if encoder_config.grad_source == "bc_loss":
            raise ValueError(
                "grad_source='bc_loss' trains the encoder from a behaviour-cloning "
                "objective; it belongs to the image-BC baseline, not to RLPD."
            )
        if encoder_config.sharing == "separate_per_head" and encoder_config.grad_source != "actor_and_critic":
            raise ValueError(
                "sharing='separate_per_head' gives the actor its own encoder, which then "
                "has no gradient unless grad_source='actor_and_critic'."
            )

        image_shape = pixel_config.image_shape
        self.encoder: VisualEncoder = build_encoder(
            encoder_config, self.camera_keys, image_shape
        )
        self.actor_encoder: VisualEncoder | None = None
        if encoder_config.sharing == "separate_per_head":
            self.actor_encoder = build_encoder(encoder_config, self.camera_keys, image_shape)
        print(
            f"[rlpd_pixel] encoder={encoder_config.arch} latent_dim={self.encoder.latent_dim} "
            f"proprio_dim={self.proprio_dim} grad_source={encoder_config.grad_source} "
            f"sharing={encoder_config.sharing} augmentation={encoder_config.augmentation}"
        )

        self.feature_dim = self.encoder.latent_dim + self.proprio_dim
        hidden_dims = tuple(int(dim) for dim in args.hidden_dims)
        self.actor = SquashedGaussianActor(
            self.feature_dim, self.action_low, self.action_high, hidden_dims
        )
        self.critic = EnsembleCritic(
            self.feature_dim,
            self.action_dim,
            hidden_dims,
            args.num_qs,
            layer_norm=args.critic_layer_norm,
        )
        self.critic_target = EnsembleCritic(
            self.feature_dim,
            self.action_dim,
            hidden_dims,
            args.num_qs,
            layer_norm=args.critic_layer_norm,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        self.proprio_stats = RunningMeanStd(self.proprio_dim)
        self.normalize_proprio = bool(args.normalize_proprio)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_learning_rate)
        self.q_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_learning_rate)
        # Separate encoder optimizer so the grad_source switch is explicit.
        self.encoder_optimizer = self._build_encoder_optimizer(self.encoder, args)
        self.actor_encoder_optimizer = None
        if self.actor_encoder is not None:
            self.actor_encoder_optimizer = self._build_encoder_optimizer(
                self.actor_encoder, args
            )

        # Freeze -> unfreeze warmup for a pre-trained backbone.
        self.backbone_warmup_steps = int(getattr(args, "backbone_warmup_steps", 0))
        self._backbone_unfrozen = self.backbone_warmup_steps <= 0
        if not self._backbone_unfrozen:
            for encoder in self._encoders():
                encoder.set_backbone_trainable(False)
            print(
                f"[rlpd_pixel] backbone frozen for the first "
                f"{self.backbone_warmup_steps} updates, then fine-tuned at "
                f"lr={args.encoder_learning_rate * args.backbone_lr_scale:.2e}"
            )

        # Initial backbone weights, to log how far fine-tuning drifts (not saved).
        self.register_buffer("_backbone_reference", None, persistent=False)
        self._backbone_reference_norm = 1.0
        self._backbone_drift = float("nan")
        self._drift_interval = max(int(getattr(args, "backbone_drift_log_interval", 1000)), 1)
        backbone_params = self.encoder.parameter_groups()["backbone"]
        if backbone_params and not encoder_config.freeze_backbone:
            with torch.no_grad():
                reference = torch.cat([p.detach().flatten() for p in backbone_params])
            self._backbone_reference = reference.clone()
            self._backbone_reference_norm = float(reference.norm().clamp_min(1e-8))
            print(
                f"[rlpd_pixel] fine-tuning {reference.numel():,} backbone parameters "
                f"against a {sum(p.numel() for p in self.encoder.parameter_groups()['head']):,}"
                f"-parameter head; watch losses/backbone_drift"
            )

        initial_alpha = max(float(args.init_temperature), 1e-8)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha), dtype=torch.float32))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.alpha_learning_rate)
        self.target_entropy = (
            -float(self.action_dim) / 2.0
            if args.target_entropy is None
            else float(args.target_entropy)
        )

        writer = SummaryWriter(runs_path(run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(f"|{key}|{value}|" for key, value in asdict(args).items()),
        )
        writer.add_text("pixel_obs_config", pixel_config.summary())
        writer.add_text("encoder_config", str(asdict(encoder_config)))
        self.logger = Logger(log_swanlab=args.track, tensorboard=writer)

    # ---- encoder optimization -----------------------------------------------

    def _encoders(self) -> list[VisualEncoder]:
        encoders = [self.encoder]
        if self.actor_encoder is not None:
            encoders.append(self.actor_encoder)
        return encoders

    def _build_encoder_optimizer(self, encoder: VisualEncoder, args):
        """One Adam over two parameter groups: pre-trained backbone (scaled lr)
        and fresh head. Backbone parameters are registered even while frozen."""
        if self.encoder_config.grad_source == "frozen":
            return None
        groups = encoder.parameter_groups()
        backbone_lr = args.encoder_learning_rate * float(args.backbone_lr_scale)
        param_groups = []
        if groups["head"]:
            param_groups.append(
                {"params": groups["head"], "lr": args.encoder_learning_rate, "name": "head"}
            )
        if groups["backbone"] and not self.encoder_config.freeze_backbone:
            param_groups.append(
                {"params": groups["backbone"], "lr": backbone_lr, "name": "backbone"}
            )
        return optim.Adam(param_groups) if param_groups else None

    @torch.no_grad()
    def _update_backbone_drift(self, update_step: int) -> float:
        """Relative L2 distance of the backbone from its initial weights,
        recomputed every ``backbone_drift_log_interval`` updates."""
        if self._backbone_reference is None:
            return float("nan")
        if update_step % self._drift_interval != 0 and not np.isnan(self._backbone_drift):
            return self._backbone_drift
        current = torch.cat(
            [p.detach().flatten() for p in self.encoder.parameter_groups()["backbone"]]
        )
        self._backbone_drift = float(
            (current - self._backbone_reference).norm() / self._backbone_reference_norm
        )
        return self._backbone_drift

    def _maybe_unfreeze_backbone(self, update_step: int) -> None:
        if self._backbone_unfrozen or update_step < self.backbone_warmup_steps:
            return
        for encoder in self._encoders():
            encoder.set_backbone_trainable(True)
        self._backbone_unfrozen = True
        print(f"[rlpd_pixel] backbone unfrozen at update {update_step}")

    # ---- observation plumbing -----------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def normalize_obs_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        if not self.normalize_proprio:
            return proprio
        return self.proprio_stats.normalize(proprio)

    @torch.no_grad()
    def update_proprio_stats(self, proprio: torch.Tensor) -> None:
        if self.normalize_proprio:
            self.proprio_stats.update(proprio)

    def _features(
        self,
        obs: Mapping[str, torch.Tensor],
        encoder: VisualEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images and return ``(z, [z, proprio])``."""
        latent = encoder({key: obs[key] for key in self.camera_keys})
        proprio = self.normalize_obs_proprio(obs["proprio"])
        return latent, torch.cat([latent, proprio], dim=-1)

    def _augment(self, obs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if not self.encoder_config.augmentation:
            return obs
        return augment_obs(obs, self.camera_keys, self.encoder_config.augment_pad)

    @torch.no_grad()
    def select_action(
        self, obs: Mapping[str, torch.Tensor], deterministic: bool = False
    ) -> torch.Tensor:
        """Act from a raw (unaugmented) observation dict; critic is not involved."""
        encoder = self.actor_encoder if self.actor_encoder is not None else self.encoder
        _, features = self._features(obs, encoder)
        if deterministic:
            return self.actor.deterministic(features)
        return self.actor.sample(features)[0]

    # ---- one gradient step ---------------------------------------------------

    def update(
        self,
        batch: Mapping[str, object],
        update_step: int,
        update_actor: bool = True,
    ) -> dict[str, float]:
        obs = self._augment(batch["observations"])
        next_obs = self._augment(batch["next_observations"])
        # Kept so the trainer can log the augmented frames the encoder saw.
        self._last_augmented = {"observations": obs, "is_offline": batch.get("is_offline")}
        actions = batch["actions"]
        rewards = batch["rewards"] * self.args.reward_scale + self.args.reward_bias
        dones = batch["dones"]
        is_offline = batch.get("is_offline")

        self._maybe_unfreeze_backbone(update_step)
        frozen_encoder = self.encoder_config.grad_source == "frozen"

        # ---- encode ----------------------------------------------------------
        if frozen_encoder:
            with torch.no_grad():
                latent, features = self._features(obs, self.encoder)
        else:
            latent, features = self._features(obs, self.encoder)
        # The o' branch must not touch the encoder, BatchNorm statistics included.
        with torch.no_grad(), self.encoder.frozen_norm_stats():
            _, next_features = self._features(next_obs, self.encoder)

        alpha = self.alpha.detach()

        # ---- critic: random-subset-min TD target -----------------------------
        with torch.no_grad():
            actor_next_features = next_features
            if self.actor_encoder is not None:
                # Same o' rule for the per-head actor encoder.
                with self.actor_encoder.frozen_norm_stats():
                    _, actor_next_features = self._features(next_obs, self.actor_encoder)
            next_actions, next_log_prob, _ = self.actor.sample(actor_next_features)
            target_q = self.critic_target.subset_min(
                next_features, next_actions, self.args.num_min_qs
            )
            if self.args.backup_entropy:
                target_q = target_q - alpha * next_log_prob
            target_q = rewards + (1.0 - dones) * self.args.gamma * target_q

        all_q = self.critic(features, actions)  # [num_qs, B, 1]
        q_loss = F.mse_loss(all_q, target_q.unsqueeze(0).expand_as(all_q))

        self.q_optimizer.zero_grad(set_to_none=True)
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        encoder_grad_norm = float("nan")
        # Reported separately: a combined norm is dominated by the backbone.
        backbone_grad_norm = float("nan")
        head_grad_norm = float("nan")
        if self.encoder_optimizer is not None:
            groups = self.encoder.parameter_groups()
            if groups["backbone"] and any(p.grad is not None for p in groups["backbone"]):
                backbone_grad_norm = float(
                    torch.norm(
                        torch.stack(
                            [
                                p.grad.detach().norm()
                                for p in groups["backbone"]
                                if p.grad is not None
                            ]
                        )
                    )
                )
            if groups["head"] and any(p.grad is not None for p in groups["head"]):
                head_grad_norm = float(
                    torch.norm(
                        torch.stack(
                            [p.grad.detach().norm() for p in groups["head"] if p.grad is not None]
                        )
                    )
                )
            encoder_grad_norm = float(
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            )
        self.q_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        actor_loss_value = float("nan")
        alpha_loss_value = float("nan")
        log_prob_value = float("nan")
        if update_actor:
            # ---- actor -------------------------------------------------------
            # With grad_source="critic_only" the latent is detached: no actor
            # gradient reaches the encoder.
            if self.actor_encoder is not None:
                _, actor_features = self._features(obs, self.actor_encoder)
            elif self.encoder_config.grad_source == "actor_and_critic":
                _, actor_features = self._features(obs, self.encoder)
            else:
                actor_features = torch.cat(
                    [latent.detach(), self.normalize_obs_proprio(obs["proprio"])], dim=-1
                )

            self.critic.requires_grad_(False)
            sampled_actions, log_prob, _ = self.actor.sample(actor_features)
            actor_q = self.critic(actor_features, sampled_actions).mean(dim=0)
            actor_loss = (self.alpha.detach() * log_prob - actor_q).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            if self.encoder_config.grad_source == "actor_and_critic":
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad(set_to_none=True)
                if self.actor_encoder_optimizer is not None:
                    self.actor_encoder_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()
            if self.encoder_config.grad_source == "actor_and_critic":
                if self.actor_encoder_optimizer is not None:
                    nn.utils.clip_grad_norm_(
                        self.actor_encoder.parameters(), self.args.max_grad_norm
                    )
                    self.actor_encoder_optimizer.step()
                elif self.encoder_optimizer is not None:
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
                    self.encoder_optimizer.step()
            self.critic.requires_grad_(True)

            actor_loss_value = float(actor_loss.item())
            log_prob_value = float(log_prob.mean().item())

            if self.args.autotune_entropy:
                alpha_loss = -(
                    self.log_alpha * (log_prob.detach() + self.target_entropy)
                ).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_value = float(alpha_loss.item())

        # ---- soft target update (critic MLP only; DrQ-v2 convention) ---------
        with torch.no_grad():
            for online_param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.lerp_(online_param.data, self.args.tau)

        metrics = {
            "q_loss": float(q_loss.item()),
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "entropy_alpha": float(self.alpha.detach().item()),
            "policy_log_prob": log_prob_value,
            "q_mean": float(all_q.mean().item()),
            "target_q": float(target_q.mean().item()),
            "target_q_abs_max": float(target_q.abs().max().item()),
            "q_grad_norm": float(q_grad_norm),
            "encoder_grad_norm": encoder_grad_norm,
            "encoder_backbone_grad_norm": backbone_grad_norm,
            "encoder_head_grad_norm": head_grad_norm,
            "backbone_drift": self._update_backbone_drift(update_step),
        }
        metrics.update(self._diagnostics(all_q, target_q, latent, is_offline))
        return metrics

    @torch.no_grad()
    def _diagnostics(
        self,
        all_q: torch.Tensor,
        target_q: torch.Tensor,
        latent: torch.Tensor,
        is_offline: torch.Tensor | None,
    ) -> dict[str, float]:
        """Diagnostics for why a run is stuck."""
        mean_q = all_q.mean(dim=0)  # [B, 1]
        td_error = (mean_q - target_q).abs()
        metrics = {
            # Spread across heads; zero means the ensemble has collapsed.
            "critic_ensemble_std": float(all_q.std(dim=0).mean().item()),
            "latent_l2_norm": float(latent.detach().norm(dim=-1).mean().item()),
            "td_error": float(td_error.mean().item()),
        }
        if is_offline is not None:
            offline_mask = is_offline.reshape(-1) > 0.5
            online_mask = ~offline_mask
            if bool(offline_mask.any()) and bool(online_mask.any()):
                q_offline = float(mean_q.reshape(-1)[offline_mask].mean().item())
                q_online = float(mean_q.reshape(-1)[online_mask].mean().item())
                td_offline = float(td_error.reshape(-1)[offline_mask].mean().item())
                td_online = float(td_error.reshape(-1)[online_mask].mean().item())
                metrics.update(
                    {
                        "q_mean_offline": q_offline,
                        "q_mean_online": q_online,
                        # A widening gap signals a visual mismatch between demos
                        # and online data.
                        "q_offline_minus_online": q_offline - q_online,
                        "td_error_offline": td_offline,
                        "td_error_online": td_online,
                        "td_error_offline_over_online": td_offline / max(td_online, 1e-8),
                    }
                )
        metrics.update(self.encoder.diagnostics())
        return metrics

    # ---- checkpoints ---------------------------------------------------------

    def save(self, path: str, global_step: int) -> None:
        path = ensure_parent_dir(path)
        checkpoint = {
            "agent": self.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "global_step": int(global_step),
            "args": asdict(self.args),
            "encoder_config": asdict(self.encoder_config),
            "pixel_obs_config": self.pixel_config.to_dict(),
            "camera_config_hash": self.pixel_config.config_hash(),
        }
        if self.encoder_optimizer is not None:
            checkpoint["encoder_optimizer"] = self.encoder_optimizer.state_dict()
        if self.actor_encoder_optimizer is not None:
            checkpoint["actor_encoder_optimizer"] = self.actor_encoder_optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"model saved to {path}")

    def load(self, path: str, *, load_optimizers: bool) -> int:
        checkpoint = torch.load(path, map_location=self.action_low.device, weights_only=False)
        stored_hash = checkpoint.get("camera_config_hash")
        if stored_hash is not None and stored_hash != self.pixel_config.config_hash():
            raise RuntimeError(
                f"Checkpoint was trained with camera_config_hash={stored_hash!r} but the "
                f"current config hashes to {self.pixel_config.config_hash()!r}."
            )
        self.load_state_dict(checkpoint["agent"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
            if self.encoder_optimizer is not None and "encoder_optimizer" in checkpoint:
                self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            if (
                self.actor_encoder_optimizer is not None
                and "actor_encoder_optimizer" in checkpoint
            ):
                self.actor_encoder_optimizer.load_state_dict(
                    checkpoint["actor_encoder_optimizer"]
                )
        return int(checkpoint.get("global_step", 0))

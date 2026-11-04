"""PPO from pixels, with a behaviour-cloning anchor (`train/train_ppo_pixels.py`).

The state-based agent in :mod:`algo.ppo` with a visual encoder in front of each
head, for tasks where the observation is two RGB cameras plus proprioception::

    images --> encoder --> z ; concat([z, normalized proprio]) --> actor / critic

Four losses, and they touch disjoint things:

===================  =======================  ==================================
loss                 data                     parameters it updates
===================  =======================  ==================================
clipped surrogate    online rollout           actor encoder, actor MLP, log-std
entropy bonus        online rollout           log-std only
value regression     rollout GAE returns      critic encoder, critic MLP
behaviour cloning    the rendered demos       actor encoder, actor MLP
===================  =======================  ==================================

Three rules that are easy to get wrong and produce no error:

1. **The policy branch never sees augmented images.** ``logprob_old`` was
   computed during the rollout on the raw frame; augmenting the same state in the
   update changes the input the ratio is measured on, so ``pi_new/pi_old`` stops
   being a ratio of two policies at one state. The value branch and the BC batch
   do get random shift -- neither is an importance ratio. ``ratio_err`` in the
   metrics is the guard: at the first minibatch of the first epoch the ratio must
   be exactly 1.
2. **Nothing the policy depends on may move between collection and the update.**
   Freezing the proprio statistics *inside* :meth:`update` is not enough: run
   5744 updated them once per rollout step, so ``pi_old`` had been evaluated
   under 200 different normalizations and ``ratio_err`` came out at 5-20 instead
   of ~0. Every iteration then tripped ``target_kl`` on its first minibatch and
   the run took ~350 gradient steps in 22 hours. The trainer now accumulates the
   rollout's proprio and updates the statistics once, after the update; and
   ``recompute_old_logprob`` re-derives ``logprob_old`` in minibatch-shaped
   chunks so the epoch-0 ratio is 1 by construction even under GPU numerics.
   ``rollout_logprob_err`` keeps reporting what that recomputation had to repair.
3. **The actor's mean is bounded by a tanh**, because on this robot family
   ``normalize_action=False`` -- an action *is* a joint target in radians, and an
   unbounded linear head starts out commanding 0 rad on every joint, which is
   outside some joint ranges entirely. The span of that tanh is the policy's
   step-size scale in action space: mapped onto the full joint range (half-span
   ~1.09 rad here) one Adam step at 3e-4 moved the mean 0.04 rad = 1.3 sigma,
   which is 22x the displacement ``target_kl=0.05`` allows for a whole update.
   ``action_scale_source="demos"`` shrinks the span to what the demos use.

Note what is *not* an option on this robot: anchoring the mean on the measured
``qpos``. The demos hold their joint targets 0.25 rad away from qpos on average
(p99 1.18; over 1 rad on the finger joints) -- that offset is how the stiff PD
controller produces grip force -- so a small residual around ``qpos`` cannot
represent them. The only anchor at the demos' own 0.004 rad/step scale is the
previous *command*, which is what ``utils.delta_action`` does and what demo-state
resets rule out (the wrapper rewinds ``a_prev`` on every reset).

Unlike the RLPD agent there is no Q ensemble, no target network and no
temperature: the demos enter only through the BC loss here and through
demo-state resets in the trainer.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from algo.encoders import EncoderConfig, VisualEncoder, augment_obs, build_encoder
from algo.networks import RunningMeanStd
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path
from utils.pixel_cameras import PixelObsConfig


# The log-std is a free parameter; clamp it so a bad update cannot make the
# policy deterministic (no exploration, exploding ratios) or absurdly wide.
LOG_STD_MIN = -8.0
LOG_STD_MAX = 1.0


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def index_obs(obs: Mapping[str, torch.Tensor], index) -> dict[str, torch.Tensor]:
    """Index every field of an observation dict with the same rows."""
    return {key: value[index] for key, value in obs.items()}


def _build_trunk(
    input_dim: int, hidden_dims: Sequence[int], output_dim: int, output_std: float
) -> nn.Sequential:
    """PPO's MLP: Tanh activations, orthogonal init, small-gain output layer."""
    layers: list[nn.Module] = []
    current = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([layer_init(nn.Linear(current, int(hidden))), nn.Tanh()])
        current = int(hidden)
    layers.append(layer_init(nn.Linear(current, int(output_dim)), std=output_std))
    return nn.Sequential(*layers)


class PixelActorCritic(nn.Module):
    """Two encoders (or one, shared) feeding a Gaussian actor and a value head."""

    def __init__(
        self,
        *,
        pixel_config: PixelObsConfig,
        encoder_config: EncoderConfig,
        proprio_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_dims: Sequence[int] = (256, 256, 256),
        logstd_init: float = -3.5,
        normalize_proprio: bool = True,
        mean_low: torch.Tensor | None = None,
        mean_high: torch.Tensor | None = None,
    ):
        super().__init__()
        self.pixel_config = pixel_config
        self.encoder_config = encoder_config.validate()
        self.camera_keys = pixel_config.camera_keys
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_low.numel())
        self.normalize_proprio = bool(normalize_proprio)

        image_shape = pixel_config.image_shape
        # `encoder` is the critic's. `actor_encoder` is None when the two heads
        # share one, mirroring the layout of algo/offline_to_online_rl/rlpd_pixel.py.
        self.encoder: VisualEncoder = build_encoder(encoder_config, self.camera_keys, image_shape)
        self.actor_encoder: VisualEncoder | None = None
        if encoder_config.sharing == "separate_per_head":
            self.actor_encoder = build_encoder(encoder_config, self.camera_keys, image_shape)

        self.feature_dim = self.encoder.latent_dim + self.proprio_dim
        self.actor_mean = _build_trunk(self.feature_dim, hidden_dims, self.action_dim, 0.01)
        self.critic = _build_trunk(self.feature_dim, hidden_dims, 1, 1.0)
        self.actor_logstd = nn.Parameter(
            torch.full((1, self.action_dim), float(logstd_init), dtype=torch.float32)
        )

        self.proprio_stats = RunningMeanStd(self.proprio_dim)

        action_low = action_low.float()
        action_high = action_high.float()
        if torch.any(action_high - action_low <= 0):
            raise ValueError("Every action dimension must have a positive range.")
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

        # The tanh's span. It defaults to the action limits, but a policy only
        # ever needs to reach the region the demos use, and a narrower span is a
        # proportionally smaller step in action space for the same parameter
        # step -- which is the whole trust-region budget here. Never allowed
        # outside the limits.
        mean_low = action_low if mean_low is None else mean_low.float().to(action_low.device)
        mean_high = action_high if mean_high is None else mean_high.float().to(action_low.device)
        mean_low = torch.maximum(mean_low, action_low)
        mean_high = torch.minimum(mean_high, action_high)
        action_scale = (mean_high - mean_low) / 2.0
        if torch.any(action_scale <= 0):
            raise ValueError("Every action dimension must have a positive mean range.")
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", (mean_high + mean_low) / 2.0)

        span_ratio = float(
            (action_scale / ((action_high - action_low) / 2.0)).mean().item()
        )
        print(
            f"[ppo_pixel] encoder={encoder_config.arch} sharing={encoder_config.sharing} "
            f"latent_dim={self.encoder.latent_dim} proprio_dim={self.proprio_dim} "
            f"action_dim={self.action_dim} sigma_init={float(np.exp(logstd_init)):.4f} "
            f"mean_span={span_ratio:.3f}x joint range"
        )

    # ---- observation plumbing -------------------------------------------------

    @property
    def policy_encoder(self) -> VisualEncoder:
        return self.actor_encoder if self.actor_encoder is not None else self.encoder

    def encoders(self) -> list[VisualEncoder]:
        encoders = [self.encoder]
        if self.actor_encoder is not None:
            encoders.append(self.actor_encoder)
        return encoders

    def normalize_proprio_vec(self, proprio: torch.Tensor) -> torch.Tensor:
        if not self.normalize_proprio:
            return proprio
        return self.proprio_stats.normalize(proprio)

    @torch.no_grad()
    def update_proprio_stats(self, proprio: torch.Tensor) -> None:
        """Collection-time only -- see rule 2 in the module docstring."""
        if self.normalize_proprio:
            self.proprio_stats.update(proprio)

    def _features(
        self, obs: Mapping[str, torch.Tensor], encoder: VisualEncoder
    ) -> torch.Tensor:
        latent = encoder({key: obs[key] for key in self.camera_keys})
        return torch.cat([latent, self.normalize_proprio_vec(obs["proprio"])], dim=-1)

    # ---- policy / value -------------------------------------------------------

    def action_mean(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Mean joint target, bounded into [action_low, action_high]."""
        features = self._features(obs, self.policy_encoder)
        return self.action_bias + self.action_scale * torch.tanh(self.actor_mean(features))

    def distribution(self, obs: Mapping[str, torch.Tensor]) -> Normal:
        mean = self.action_mean(obs)
        log_std = self.actor_logstd.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return Normal(mean, log_std.exp())

    def get_value(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self._features(obs, self.encoder))

    def get_action_and_value(
        self, obs: Mapping[str, torch.Tensor], action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout-time forward: one raw observation in, everything stored out."""
        distribution = self.distribution(obs)
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(1),
            distribution.entropy().sum(1),
            self.get_value(obs),
        )

    def evaluate_actions(
        self, obs: Mapping[str, torch.Tensor], action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy branch of the update. No value here: it runs on augmented obs."""
        distribution = self.distribution(obs)
        return distribution.log_prob(action).sum(1), distribution.entropy().sum(1)

    def get_action(
        self, obs: Mapping[str, torch.Tensor], deterministic: bool = False
    ) -> torch.Tensor:
        if deterministic:
            return self.action_mean(obs)
        return self.distribution(obs).sample()

    @torch.no_grad()
    def initialize_action_bias(self, mean_action: torch.Tensor) -> None:
        """Seed the actor's output bias so it starts at the demos' mean pose.

        With the default small-gain output layer the untrained mean is
        ``action_bias``, the midpoint of every joint range -- a pose no demo ever
        visits. Inverting the tanh squash for the demos' mean action costs
        nothing and makes the very first rollout (and the BC warmup's starting
        point) a plausible configuration instead of a sprawl.
        """
        output_layer = self.actor_mean[-1]
        target = (mean_action.to(self.action_bias.device) - self.action_bias) / self.action_scale
        output_layer.bias.copy_(torch.atanh(target.clamp(-0.999, 0.999)))


class PPOPixelAgent(nn.Module):
    """PPO update, BC anchor, logging and checkpoints for :class:`PixelActorCritic`.

    ``bc_dataset`` is a :class:`utils.pixel_dataset.OfflineImageDataset` -- the
    *rendered* demos. Its frames are read directly rather than through
    ``sample()``, which would also gather ``next_observations`` that nothing in
    an on-policy update uses.
    """

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
        bc_dataset=None,
        mean_low: torch.Tensor | None = None,
        mean_high: torch.Tensor | None = None,
    ):
        super().__init__()
        self.args = args
        self.pixel_config = pixel_config
        self.encoder_config = encoder_config
        self.camera_keys = pixel_config.camera_keys

        self.actor_critic = PixelActorCritic(
            pixel_config=pixel_config,
            encoder_config=encoder_config,
            proprio_dim=proprio_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dims=tuple(int(dim) for dim in args.hidden_dims),
            logstd_init=float(args.actor_logstd_init),
            normalize_proprio=bool(args.normalize_proprio),
            mean_low=mean_low,
            mean_high=mean_high,
        )

        if encoder_config.grad_source == "frozen":
            for encoder in self.actor_critic.encoders():
                encoder.requires_grad_(False)
            print("[ppo_pixel] encoders frozen: only the MLP heads and log-std train")

        # Two learning rates. The critic regresses a [0, 1] target and wants the
        # usual 3e-4; the actor's step lands in *radians of joint target* and has
        # to stay well inside sigma, or every update spends its whole KL budget
        # on the first minibatch (run 5744: 0.04 rad = 1.3 sigma per step).
        # With `sharing="shared"` the one encoder sits in the critic group and
        # the actor rate covers only the MLP and the log-std.
        actor_lr = float(
            args.learning_rate
            if getattr(args, "actor_learning_rate", None) is None
            else args.actor_learning_rate
        )
        self.actor_learning_rate = actor_lr
        actor_modules = [self.actor_critic.actor_mean]
        if self.actor_critic.actor_encoder is not None:
            actor_modules.append(self.actor_critic.actor_encoder)
        actor_ids = {id(p) for module in actor_modules for p in module.parameters()}
        actor_ids.add(id(self.actor_critic.actor_logstd))
        actor_params, critic_params = [], []
        for parameter in self.actor_critic.parameters():
            if not parameter.requires_grad:
                continue
            (actor_params if id(parameter) in actor_ids else critic_params).append(parameter)
        self.optimizer = optim.Adam(
            [
                {"params": actor_params, "lr": actor_lr, "name": "actor"},
                {"params": critic_params, "lr": args.learning_rate, "name": "critic"},
            ],
            lr=args.learning_rate,
            eps=1e-5,
        )
        print(
            f"[ppo_pixel] actor lr={actor_lr:g} ({len(actor_params)} tensors), "
            f"critic lr={args.learning_rate:g} ({len(critic_params)} tensors)"
        )

        writer = SummaryWriter(runs_path(run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(f"|{key}|{value}|" for key, value in _args_items(args)),
        )
        writer.add_text("pixel_obs_config", pixel_config.summary())
        writer.add_text("encoder_config", str(asdict(encoder_config)))
        self.logger = Logger(log_swanlab=args.track, tensorboard=writer)

        self.bc_dataset = bc_dataset
        self.use_bc = bool(args.use_bc_loss and bc_dataset is not None and len(bc_dataset) > 0)
        self.bc_coef = float(args.bc_coef)
        self.bc_min_coef = float(args.bc_min_coef)
        self.bc_decay = bool(args.bc_decay)
        self.bc_loss_type = str(args.bc_loss_type)
        bc_batch_size = int(args.bc_batch_size)
        self.bc_batch_size = bc_batch_size if bc_batch_size > 0 else args.minibatch_size
        self.bc_decay_steps = max(
            1, int(float(args.bc_decay_steps) * int(args.total_timesteps))
        )
        self._last_augmented: dict[str, torch.Tensor] | None = None
        self.bc_train_indices, self.bc_val_indices = self._split_bc_episodes(
            float(getattr(args, "bc_val_fraction", 0.0))
        )

    def _split_bc_episodes(self, val_fraction: float):
        """Hold out whole demo *episodes* for the BC validation error.

        Splitting by transition would put frames from the same demo on both
        sides and report a fit, not a generalization gap -- and the gap is the
        question, because the training error alone cannot say whether the actor
        is too small or already memorizing (run 5744: 0.0052 rad after 320
        epochs over 16009 frames, with no held-out number to compare it to).
        """
        if not self.use_bc:
            return None, None
        dataset = self.bc_dataset
        counts = list(getattr(dataset, "episode_transition_counts", []))
        device = dataset.actions.device
        all_indices = torch.arange(len(dataset), device=device)
        if val_fraction <= 0.0 or len(counts) < 4:
            return all_indices, None
        episode_of = torch.repeat_interleave(
            torch.arange(len(counts), device=device),
            torch.as_tensor(counts, device=device),
        )
        generator = np.random.default_rng(int(getattr(self.args, "seed", 0) or 0))
        order = generator.permutation(len(counts))
        num_val = max(1, int(round(val_fraction * len(counts))))
        val_episodes = torch.as_tensor(
            np.sort(order[:num_val]).copy(), device=device, dtype=torch.long
        )
        is_val = torch.isin(episode_of, val_episodes)
        train_indices = all_indices[~is_val]
        val_indices = all_indices[is_val]
        print(
            f"[ppo_pixel] BC split: {len(counts) - num_val} train episodes "
            f"({int(train_indices.numel())} frames) / {num_val} val episodes "
            f"({int(val_indices.numel())} frames)"
        )
        return train_indices, val_indices

    # ---- behaviour cloning ----------------------------------------------------

    def _get_bc_coef(self, global_step: int) -> float:
        if not self.use_bc or self.bc_coef <= 0:
            return 0.0
        if not self.bc_decay:
            return self.bc_coef
        progress = min(max(global_step / self.bc_decay_steps, 0.0), 1.0)
        return max(self.bc_min_coef, self.bc_coef * (1.0 - progress))

    def _gather_bc(
        self, device: torch.device, indices: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        dataset = self.bc_dataset
        frame_ids = dataset.obs_frame_id[indices]
        obs = {
            key: dataset.frames[key][frame_ids].to(device, non_blocking=True)
            for key in self.camera_keys
        }
        obs["proprio"] = dataset.proprio_frames[frame_ids].to(device, non_blocking=True)
        return obs, dataset.actions[indices].to(device, non_blocking=True)

    def _sample_bc_batch(
        self, device: torch.device, batch_size: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        pool = self.bc_train_indices
        positions = torch.randint(
            0, int(pool.numel()), (min(int(batch_size), int(pool.numel())),), device=pool.device
        )
        return self._gather_bc(device, pool[positions])

    @torch.no_grad()
    def bc_validation_error(self, device: torch.device, max_frames: int = 4096) -> float:
        """Mean |mean - a_demo| in radians on the held-out demos, un-augmented.

        Un-augmented on purpose: the rollout sees raw frames, so this is the
        error the policy actually acts with. ``nan`` when nothing is held out.
        """
        pool = self.bc_val_indices
        if pool is None or int(pool.numel()) == 0:
            return float("nan")
        was_training = self.actor_critic.training
        self.actor_critic.eval()
        if int(pool.numel()) > max_frames:
            step = int(pool.numel()) // max_frames
            pool = pool[::step][:max_frames]
        total, count = 0.0, 0
        chunk = max(1, int(self.bc_batch_size))
        for start in range(0, int(pool.numel()), chunk):
            indices = pool[start : start + chunk]
            obs, actions = self._gather_bc(device, indices)
            error = (self.actor_critic.action_mean(obs) - actions).abs().mean()
            total += float(error.item()) * int(indices.numel())
            count += int(indices.numel())
        if was_training:
            self.actor_critic.train()
        return total / max(count, 1)

    def _compute_bc_loss(
        self, bc_obs: Mapping[str, torch.Tensor], bc_actions: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Returns (loss, mean |mean - a_demo| per dim).

        The action error is reported un-normalized, in radians, because that is
        the quantity with a physical meaning -- and because dividing the *loss*
        by the demo action std is what wrecked the RLPD anchor (it scaled the
        gradient by 634x and turned the run into noiseless BC).
        """
        mean_action = self.actor_critic.action_mean(bc_obs)
        action_err = float((mean_action - bc_actions).abs().mean().item())
        if self.bc_loss_type == "nll":
            log_std = self.actor_critic.actor_logstd.clamp(LOG_STD_MIN, LOG_STD_MAX)
            distribution = Normal(mean_action, log_std.expand_as(mean_action).exp())
            return -distribution.log_prob(bc_actions).sum(1).mean(), action_err
        return ((mean_action - bc_actions) ** 2).mean(), action_err

    def bc_pretrain(self, num_steps: int, log_every: int = 500) -> None:
        """Pure-BC warmup on the demos, before any environment interaction.

        Runs on its own optimizer so PPO does not inherit Adam moments shaped by
        a different objective. Only the actor path is in the graph, so the critic
        and its encoder are untouched; with ``bc_loss_type="nll"`` the log-std is
        trained too, which is the point of that option.
        """
        if not self.use_bc or num_steps <= 0:
            return
        args = self.args
        device = next(self.actor_critic.parameters()).device
        parameters = [
            p
            for p in (
                list(self.actor_critic.actor_mean.parameters())
                + list(self.actor_critic.policy_encoder.parameters())
                + [self.actor_critic.actor_logstd]
            )
            if p.requires_grad
        ]
        optimizer = optim.Adam(parameters, lr=float(args.bc_warmup_learning_rate))
        batch_size = int(args.bc_warmup_batch_size)
        self.actor_critic.train()

        # The warmup runs before the first rollout, so the proprio statistics are
        # still at their prior (mean 0, var 1). Prime them from the demos instead,
        # or BC would learn against one normalization and PPO would immediately
        # switch to another.
        if self.actor_critic.normalize_proprio:
            self.actor_critic.proprio_stats.update(self.bc_dataset.proprio_frames.to(device))

        loss_item = 0.0
        action_err = float("nan")
        val_err = float("nan")
        for step in range(int(num_steps)):
            bc_obs, bc_actions = self._sample_bc_batch(device, batch_size)
            loss, action_err = self._compute_bc_loss(self._augment(bc_obs), bc_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            loss_item = float(loss.item())
            if step % log_every == 0:
                val_err = self.bc_validation_error(device)
                self.logger.add_scalar("bc_pretrain/loss", loss_item, step)
                self.logger.add_scalar("bc_pretrain/action_err", action_err, step)
                if np.isfinite(val_err):
                    self.logger.add_scalar("bc_pretrain/val_action_err", val_err, step)
                print(
                    f"bc_pretrain {step}/{num_steps}: loss={loss_item:.5f} "
                    f"action_err={action_err:.5f} val_action_err={val_err:.5f} rad"
                )
        val_err = self.bc_validation_error(device)
        print(
            f"BC pretrain done: {num_steps} steps, loss={loss_item:.5f}, "
            f"action_err={action_err:.5f} rad, val_action_err={val_err:.5f} rad"
        )
        if np.isfinite(val_err) and np.isfinite(action_err) and action_err > 0:
            # The number that says which problem you have: a ratio near 1 means
            # the actor is capacity-bound (make it bigger), a large ratio means
            # it is memorizing 62 demos (more data / augmentation / less capacity).
            print(f"BC generalization gap: val/train = {val_err / action_err:.2f}x")

    # ---- helpers --------------------------------------------------------------

    def _augment(self, obs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if not self.encoder_config.augmentation:
            return obs
        return augment_obs(obs, self.camera_keys, self.encoder_config.augment_pad)

    @torch.no_grad()
    def select_action(
        self, obs: Mapping[str, torch.Tensor], deterministic: bool = False
    ) -> torch.Tensor:
        action = self.actor_critic.get_action(obs, deterministic=deterministic)
        return self.clip_action(action)

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            action.detach(), self.actor_critic.action_low, self.actor_critic.action_high
        )

    def update_proprio_stats(self, proprio: torch.Tensor) -> None:
        self.actor_critic.update_proprio_stats(proprio)

    # ---- one PPO update -------------------------------------------------------

    def update(
        self,
        start_time: float,
        rollout_time: float,
        global_step: int,
        b_obs: Mapping[str, torch.Tensor],
        b_logprobs: torch.Tensor,
        b_actions: torch.Tensor,
        b_advantages: torch.Tensor,
        b_returns: torch.Tensor,
        b_values: torch.Tensor,
    ) -> dict[str, float]:
        args = self.args
        update_time = time.time()
        self.actor_critic.train()

        device = b_actions.device
        batch_size = int(b_actions.shape[0])
        b_inds = np.arange(batch_size)
        clipfracs: list[float] = []
        bc_coef = self._get_bc_coef(global_step)
        bc_loss_item = 0.0
        bc_action_err = float("nan")
        ratio_err = float("nan")
        rollout_logprob_err = float("nan")
        approx_kl = torch.zeros((), device=device)
        old_approx_kl = torch.zeros((), device=device)
        v_loss = torch.zeros((), device=device)
        pg_loss = torch.zeros((), device=device)
        entropy_loss = torch.zeros((), device=device)
        # Defined up here because the KL early stop can fire on the very first
        # minibatch, before any optimizer step has produced one.
        grad_norm = torch.zeros((), device=device)
        stop_early = False
        policy_frozen = False
        num_steps_taken = 0
        num_policy_steps = 0

        if getattr(args, "recompute_old_logprob", True):
            # Re-derive log pi_old here, in minibatch-shaped chunks, from the
            # same parameters that collected the data. No update has happened
            # yet, so this is the same quantity the rollout stored -- but it is
            # now computed under exactly the batch shape and the normalization
            # the epochs below will use, which makes the epoch-0 ratio 1 by
            # construction. `rollout_logprob_err` reports what that repaired.
            with torch.no_grad():
                recomputed = torch.empty_like(b_logprobs)
                for start in range(0, batch_size, args.minibatch_size):
                    chunk = slice(start, min(start + args.minibatch_size, batch_size))
                    recomputed[chunk], _ = self.actor_critic.evaluate_actions(
                        index_obs(b_obs, chunk), b_actions[chunk]
                    )
                rollout_logprob_err = float(
                    ((recomputed - b_logprobs).exp() - 1.0).abs().max().item()
                )
                b_logprobs = recomputed
            # Recomputing makes `ratio_err` 0 whatever happens, so this is the
            # only thing left that can notice the rollout and the update having
            # drifted apart. Say so once rather than let it sit in a scalar.
            if rollout_logprob_err > 0.1 and not getattr(self, "_warned_logprob_drift", False):
                self._warned_logprob_drift = True
                print(
                    f"[ppo_pixel] WARNING: rollout_logprob_err={rollout_logprob_err:.3g}. "
                    "The policy is not the same function of the state at update time as it "
                    "was during collection -- check that nothing it depends on (proprio "
                    "statistics, augmentation, a buffer) moves inside the rollout loop."
                )

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = torch.as_tensor(b_inds[start:end], device=device)
                mb_obs = index_obs(b_obs, mb_inds)

                if not policy_frozen:
                    # Policy branch: raw frames only (rule 1).
                    newlogprob, entropy = self.actor_critic.evaluate_actions(
                        mb_obs, b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        )
                        if epoch == 0 and start == 0:
                            # G1 invariant: before any gradient step, pi_new IS
                            # pi_old. Anything above ~1e-5 means the update sees a
                            # different observation than the rollout did
                            # (augmentation leaking into this branch, or proprio
                            # statistics that moved).
                            ratio_err = float((ratio - 1.0).abs().max().item())

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        # The KL budget for this batch is spent. That is the
                        # early stop doing its job -- but it must not also stop
                        # the critic, which has no trust region and needs this
                        # data (run 5744 ended at explained_variance -6.6 because
                        # the two were coupled). Under `kl_stop_scope="policy"`
                        # the actor and its BC anchor freeze and the value head
                        # keeps training on the remaining minibatches.
                        stop_early = True
                        policy_frozen = True
                        if getattr(args, "kl_stop_scope", "policy") != "policy":
                            break

                # Value branch: augmented frames, its own encoder. Runs whether
                # or not the policy is frozen.
                mb_value_obs = self._augment(mb_obs)
                self._last_augmented = dict(mb_value_obs)
                newvalue = self.actor_critic.get_value(mb_value_obs).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = args.vf_coef * v_loss

                if not policy_frozen:
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()
                    loss = loss + pg_loss - args.ent_coef * entropy_loss

                    # The BC anchor is a policy loss: once the KL budget is
                    # spent it stops with the surrogate, or the actor would keep
                    # moving outside the trust region the early stop just drew.
                    if bc_coef > 0.0 and self.use_bc:
                        bc_obs, bc_actions = self._sample_bc_batch(device, self.bc_batch_size)
                        bc_loss, bc_action_err = self._compute_bc_loss(
                            self._augment(bc_obs), bc_actions
                        )
                        bc_loss_item = float(bc_loss.item())
                        loss = loss + bc_coef * bc_loss
                    num_policy_steps += 1

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), args.max_grad_norm
                )
                self.optimizer.step()
                num_steps_taken += 1

            if stop_early and getattr(args, "kl_stop_scope", "policy") != "policy":
                break

        update_time = time.time() - update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        policy_std = float(
            self.actor_critic.actor_logstd.detach().clamp(LOG_STD_MIN, LOG_STD_MAX).exp().mean().item()
        )

        metrics = {
            "value_loss": float(v_loss.item()),
            "policy_loss": float(pg_loss.item()),
            "entropy": float(entropy_loss.item()),
            "bc_loss": bc_loss_item,
            "bc_coef": bc_coef,
            "bc_action_err": bc_action_err,
            "old_approx_kl": float(old_approx_kl.item()),
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else float("nan"),
            "explained_variance": float(explained_var),
            "policy_std": policy_std,
            "grad_norm": float(grad_norm),
            # Must be ~0; see the note where it is computed.
            "ratio_err": ratio_err,
            # What the rollout's stored log-probs were off by. `ratio_err` can
            # only stay at 0 because this one absorbs the drift, so this is the
            # number that tells you the collection-time policy and the
            # update-time policy have gone out of sync.
            "rollout_logprob_err": rollout_logprob_err,
            "early_stopped": float(stop_early),
            # How much of the configured update actually ran. `policy_steps`
            # going to 0 is what a dead run looks like from the inside.
            "grad_steps": float(num_steps_taken),
            "policy_steps": float(num_policy_steps),
        }
        for key, value in metrics.items():
            if np.isfinite(value):
                self.logger.add_scalar(f"losses/{key}", value, global_step)
        for group in self.optimizer.param_groups:
            self.logger.add_scalar(
                f"charts/learning_rate_{group.get('name', 'all')}", group["lr"], global_step
            )
        sps = int(global_step / max(time.time() - start_time, 1e-6))
        self.logger.add_scalar("charts/SPS", sps, global_step)
        self.logger.add_scalar("time/update_time", update_time, global_step)
        self.logger.add_scalar("time/rollout_time", rollout_time, global_step)
        self.logger.add_scalar(
            "time/rollout_fps", args.num_envs * args.num_steps / max(rollout_time, 1e-6), global_step
        )
        metrics["sps"] = float(sps)
        return metrics

    # ---- checkpoints ----------------------------------------------------------

    def save(self, path: str, global_step: int) -> None:
        path = ensure_parent_dir(path)
        torch.save(
            {
                "agent": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": int(global_step),
                "args": dict(_args_items(self.args)),
                "encoder_config": asdict(self.encoder_config),
                "pixel_obs_config": self.pixel_config.to_dict(),
                "camera_config_hash": self.pixel_config.config_hash(),
            },
            path,
        )
        print(f"model saved to {path}")

    def load(self, path: str, *, load_optimizer: bool) -> int:
        checkpoint = torch.load(
            path, map_location=self.actor_critic.action_low.device, weights_only=False
        )
        stored_hash = checkpoint.get("camera_config_hash")
        if stored_hash is not None and stored_hash != self.pixel_config.config_hash():
            raise RuntimeError(
                f"Checkpoint was trained with camera_config_hash={stored_hash!r} but the "
                f"current config hashes to {self.pixel_config.config_hash()!r}: the policy "
                "would be evaluated on a different visual domain than it was trained on."
            )
        self.actor_critic.load_state_dict(checkpoint["agent"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint.get("global_step", 0))


def _args_items(args) -> list[tuple[str, object]]:
    if is_dataclass(args):
        return sorted(asdict(args).items())
    return sorted(vars(args).items())

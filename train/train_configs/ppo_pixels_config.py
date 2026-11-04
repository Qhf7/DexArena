"""Config for image-based PPO (`train/train_ppo_pixels.py`).

The observation pipeline (cameras, resolution, shader, proprio fields) and the
encoder switches are the *same* objects the image-based RLPD run uses, so a
dataset rendered for one is usable by the other and `camera_config_hash` means
the same thing in both. Everything else is PPO's own: on-policy rollouts, GAE,
a clipped surrogate, and the two ways the demonstrations enter the loop --
a behaviour-cloning anchor on the actor, and demo-state resets.

Three defaults differ from `rlpd_pixels_config.py` on purpose:

* `control_mode="pd_joint_pos"` with **no** delta-action wrapper. The actor's
  mean is bounded into the joint limits by a tanh, so it commands absolute joint
  targets directly. Dropping the residual parameterization is what makes
  demo-state resets safe: the delta wrapper rewinds `a_prev` to `init_action` on
  every reset, which would pair a robot restored into the middle of a demo with
  a command anchored at the initial pose.
* The reward is the env's raw sparse {0, 1} success indicator -- no scale, no
  bias. Returns therefore live in [0, 1], so the value function needs no
  normalization and the value loss cannot dominate the policy loss.
* `actor_logstd_init=-3.5` (sigma ~ 0.03 rad). Actions are joint targets in
  radians (`normalize_action=False` on this robot family), and the demos move
  0.0042 rad per step on average (p99 0.040). PPO's usual -0.2 is sigma = 0.82
  rad, which would shake the arm apart before a single gradient step lands.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tyro

from algo.encoders import EncoderConfig
from train.train_configs.rlpd_pixels_config import (
    encoder_config_from_args,
    pixel_config_from_args,
)
from utils.paths import resolve_training_paths
from utils.pixel_cameras import CAMERA_PRESETS, PixelObsConfig


# PPO's actor loss does not flow through the critic, so the RLPD invariant that
# motivates "critic_only" (an actor inflating Q by distorting the features) has
# no analogue here; those settings would simply leave the actor's encoder with
# no gradient at all.
PPO_GRAD_SOURCES = ("actor_and_critic", "frozen")


@dataclass
class Args:
    # Experiment metadata
    # Run name; also the directory under runs/. None -> derived from env + algo.
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    # CUDA device index for torch and for the ManiSkill GPU backends.
    gpu_id: Optional[int] = None
    # Mirror metrics (and the obs image grid) to SwanLab as well as TensorBoard.
    track: bool = False
    swanlab_project_name: str = "ManiSkill"
    swanlab_entity: Optional[str] = None
    # Record eval videos. This renders an extra camera; some CabinetPour stages
    # cannot record at all (see docs/ and scripts/rlpd/). Door-open can.
    capture_video: bool = False
    save_model: bool = True
    # Evaluation-only mode: load `checkpoint`, run one eval, exit.
    evaluate: bool = False
    checkpoint: Optional[str] = None

    # Environment
    env_id: str = tyro.MISSING
    robot_uids: Optional[str] = "panda_sharpa_hand_left_angled_mobile_base"
    # Absolute joint-position control. The actor commands joint targets directly.
    control_mode: Optional[str] = "pd_joint_pos"
    # "physx_cuda" (vectorized) or "physx_cpu" (forces num_envs=1). None -> auto.
    sim_backend: Optional[str] = None
    # The {0, 1} success indicator. Used raw: no reward_scale, no reward_bias.
    reward_mode: str = "sparse"
    # Time limit for a *training* episode, in env steps.
    max_episode_steps: int = 400

    # Visual observation pipeline (single source of truth: utils/pixel_cameras.py)
    image_size: int = 128
    # "cabinet_fixed" is the CabinetPour preset (base + wrist), the one the
    # rendered door-open demos carry. Changing it changes `camera_config_hash`
    # and the BC dataset is refused on load.
    camera_preset: str = "cabinet_fixed"
    shader_pack: str = "default"
    frame_stack: int = 1

    # Encoder (algo/encoders)
    encoder_arch: str = "d4pg"
    latent_dim_per_cam: int = 64
    # PPO trains each encoder from the loss of the head that owns it; see
    # PPO_GRAD_SOURCES. "frozen" is the ablation arm that trains neither.
    encoder_grad_source: str = "actor_and_critic"
    # "separate_per_head": the actor and the critic each own an encoder, trained
    # by their own loss. That is the default because it removes any question of
    # the value loss overwhelming the policy gradient in a shared trunk.
    # "shared" is the cheaper ManiSkill-ppo_rgb-style arm (one encoder, both
    # losses), at 2x less encoder compute per update.
    encoder_sharing: str = "separate_per_head"
    # DrQ random shift. Applied to the value branch and to the BC batch, never
    # to the policy branch -- see train_ppo_pixels.py's header for why.
    augmentation: bool = True
    augment_pad: int = 4
    freeze_backbone: bool = False
    update_bn_stats: bool = True
    encoder_norm_layer: str = "group"
    encoder_norm_groups: int = 4
    encoder_pretrained_path: Optional[str] = None
    latent_layer_norm: bool = True

    # PPO core
    total_timesteps: int = 5_000_000
    # Stop after this many env steps, leaving every schedule (lr anneal, BC decay,
    # demo-init decay, curriculum) on the `total_timesteps` clock. None = run it all.
    stop_at_step: Optional[int] = None
    learning_rate: float = 3e-4
    # Separate rate for the actor (MLP + log-std + its encoder). The critic
    # regresses a [0, 1] target and is happy at 3e-4; an actor step is measured
    # in radians of joint target and has to stay well inside sigma. Measured on
    # run 5744: one Adam step at 3e-4 moved the mean 0.04 rad = 1.3 sigma, so
    # every update blew its whole KL budget on the first minibatch. None = same
    # as `learning_rate`.
    actor_learning_rate: Optional[float] = 3e-5
    anneal_lr: bool = False
    # Parallel collection envs. Each one renders both policy cameras every step,
    # so this is the VRAM/throughput knob; measure before raising it.
    num_envs: int = 64
    # Rollout length per env per iteration. Need not reach `max_episode_steps`:
    # the last step bootstraps from V(next_obs), and a shorter rollout means a
    # smaller image batch and more frequent updates.
    num_steps: int = 200
    num_minibatches: int = 32
    # One pass. With sigma ~ 0.03 rad over 29 dims the KL budget for a batch is
    # spent in a handful of minibatches anyway; a second epoch only adds
    # value-head steps, which `kl_stop_scope="policy"` already keeps running.
    update_epochs: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    # 0 by default: sigma is a learned parameter, and an entropy bonus on 29
    # joint dims pushes it up until the arm shakes. The BC anchor and the clip
    # range are what keep the policy from collapsing early.
    ent_coef: float = 0.0
    anneal_ent_coef: bool = False
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # KL early stop, as a budget for the whole update. Read it per dimension:
    # KL = sum_i (dmu_i)^2 / (2 sigma^2), so 0.2 over 29 dims at sigma = 0.03
    # allows an RMS mean move of 0.0035 rad per joint. 0.05 (the old default)
    # allowed 0.0018 rad and was tripped by the very first minibatch of every
    # iteration in run 5744.
    target_kl: Optional[float] = 0.2
    # What the early stop stops. "policy" freezes the actor and its BC anchor
    # and lets the value head keep training on the rest of the batch; "all"
    # abandons the update, which couples the critic to the actor's trust region
    # (run 5744 finished at explained_variance -6.6 that way).
    kl_stop_scope: str = "policy"
    # Re-derive log pi_old at the top of each update instead of trusting the
    # value the rollout stored. Makes the epoch-0 ratio exactly 1 even under
    # GPU numerics; `losses/rollout_logprob_err` still reports the drift.
    recompute_old_logprob: bool = True
    # log sigma at init; sigma ~ 0.03 rad. See the module docstring.
    actor_logstd_init: float = -3.5
    # Span of the tanh on the actor's mean: "joint_limits" or "demos" (the demo
    # actions' per-joint min/max, padded by `action_scale_margin` of their span
    # and always clipped into the limits). The span is the policy's step-size
    # scale in action space; on door-open the demos use 0.56x the joint range.
    action_scale_source: str = "joint_limits"
    action_scale_margin: float = 0.1
    hidden_dims: Tuple[int, ...] = (512, 512, 512, 512)
    # Standardize proprio with statistics accumulated during collection. Images
    # are never normalized this way; they reach the encoder as uint8/255.
    normalize_proprio: bool = True

    # Behaviour cloning anchor. `bc_data_path` is the *rendered* HDF5 dataset
    # (utils.pixel_dataset schema) -- BC needs images, so the LeRobot state demos
    # cannot serve here.
    use_bc_loss: bool = True
    bc_data_path: Optional[str] = None
    bc_success_only: bool = False
    # Where the demo frames live during training. They are read every minibatch,
    # so keeping them on the training device avoids a host->device copy per
    # sample; the door-open set is ~1.6 GiB. "cpu" trades speed for VRAM.
    bc_data_device: str = "auto"
    # Weight on the BC loss at step 0, linearly decayed to `bc_min_coef` over
    # `bc_decay_steps * total_timesteps`. Do NOT normalize the loss by the demo
    # action std: on delta-encoded cabinet demos that scaled the anchor by 634x
    # and the run degenerated into noiseless BC (see the RLPD ablations).
    bc_coef: float = 1.0
    bc_min_coef: float = 0.05
    bc_decay: bool = True
    bc_decay_steps: float = 0.5
    # Demo frames per PPO minibatch. <=0 uses `minibatch_size`, which doubles
    # the encoder work per update for no measured benefit.
    bc_batch_size: int = 256
    # "mse" on the (bounded) actor mean, or "nll" of the demo action under the
    # full Gaussian, which also trains sigma.
    bc_loss_type: str = "mse"
    # Pure-BC gradient steps before PPO starts. Without them the actor's mean is
    # the middle of the joint ranges and the first rollouts are a flailing arm
    # that never sees a reward.
    bc_warmup_steps: int = 20_000
    bc_warmup_batch_size: int = 256
    bc_warmup_learning_rate: float = 3e-4
    # Fraction of demo *episodes* held out of every BC batch, to report
    # `bc_pretrain/val_action_err` next to the training error. Without it the
    # training error cannot distinguish "the actor is too small" from "the actor
    # has memorized 62 demos", which is the decision that sets `hidden_dims`.
    bc_val_fraction: float = 0.1
    # Seed the actor's output bias so an untrained policy commands the demos'
    # mean pose instead of the joint-range midpoint.
    init_action_bias_from_demos: bool = True

    # Demo-state resets (the reverse curriculum). `demo_init_data_path` is the
    # LeRobot dataset, which is where the physical `environment_state` lives;
    # the rendered HDF5 carries images but no env states.
    use_demo_init_states: bool = True
    # Drop demo states recorded after the episode's first success (see
    # train_ppo_pixels.drop_post_success_states). Off by default: the measured
    # door_open / screwing_turn runs sampled them.
    demo_init_drop_post_success: bool = False
    # Curriculum clock runs `mult` times faster while train success is at or
    # above this rate. mult = 1 disables it (the measured runs).
    demo_init_fast_advance_success: float = 0.8
    demo_init_fast_advance_mult: float = 1.0
    # Clock rate while train success is BELOW `demo_init_advance_success`.
    # 0 = hard stop (the original gate).
    demo_init_stall_pace: float = 0.0
    demo_init_data_path: Optional[str] = None
    demo_init_prob: float = 1.0
    demo_init_min_prob: float = 0.1
    demo_init_decay: bool = True
    # Fractions of `total_timesteps`.
    demo_init_decay_steps: float = 0.7
    # "progress" starts every episode near the end of a demo and walks the
    # eligible window back to the demo's start; "uniform" samples any frame.
    demo_init_curriculum: str = "progress"
    demo_init_curriculum_start_frac: float = 0.2
    demo_init_curriculum_steps: float = 0.5
    # Both schedules above advance on a *curriculum clock*, not on `global_step`.
    # The clock only ticks while the recent training success rate is at least
    # `demo_init_advance_success`. On a wall-clock schedule the window walks back
    # to the demo start whether or not the policy can finish from where it
    # already starts -- run 5744's train success fell 0.157 -> 0.001 in lockstep
    # with the decay. 0 restores the old step-driven behaviour.
    demo_init_advance_success: float = 0.5
    # Episodes averaged for that gate.
    demo_init_success_window: int = 200

    # Eval
    eval_reconfiguration_freq: Optional[int] = 0
    eval_partial_reset: bool = False
    # Cross-scene generalization eval (docs/howto_cross_scene_eval.md): swap the
    # task's table/ground for one of the native RoboCasa kitchen surfaces in
    # `utils.dex_scene_eval.BACKGROUND_IDS`. Only the scene changes -- the eval
    # env keeps the `camera_preset` the checkpoint was trained on, unlike
    # `make_scene_eval_env`, which also swaps in the 256x256 `robodex_fixed` rig.
    # Requires `sim_backend=physx_cpu` (one env, episodes run serially).
    scene_background: Optional[str] = None
    # Diagnostic for a `scene_background` run: re-enable the ORIGINAL table's
    # collision shapes (still invisible) next to the native surface, whose top is
    # within 1 mm of it. The policy then sees the new scene but objects rest on the
    # training-time physics, which separates "failed because it looks different"
    # from "failed because the native tabletop collides differently".
    scene_keep_original_collision: bool = False
    # Where to save the background's `eval_background_manifest` (json), if anywhere.
    scene_manifest_path: Optional[str] = None
    # Save the first eval observation (policy cameras side by side) as an image.
    eval_first_frame_path: Optional[str] = None
    num_eval_envs: int = 25
    num_eval_episodes: int = 50
    num_eval_steps: int = 400
    video_fps: int = 20

    # Logging and checkpoints, in env steps
    log_freq: int = 25_000
    log_image_freq: int = 100_000
    eval_freq: int = 100_000
    save_freq: int = 200_000

    # Filled in at runtime by `validate_args`
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def validate_args(args: Args) -> Args:
    if args.evaluate and args.checkpoint is None:
        raise ValueError("`--checkpoint` is required when `--evaluate` is set.")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    if args.robot_uids == "":
        raise ValueError("`robot_uids` must be non-empty when provided.")
    for name in ("num_envs", "num_steps", "num_minibatches", "update_epochs", "total_timesteps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"`{name}` must be positive, got {getattr(args, name)}.")
    args.batch_size = int(args.num_envs * args.num_steps)
    if args.batch_size % args.num_minibatches != 0:
        raise ValueError(
            f"num_envs*num_steps={args.batch_size} is not divisible by "
            f"num_minibatches={args.num_minibatches}."
        )
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = max(1, args.total_timesteps // args.batch_size)

    if args.num_steps > args.max_episode_steps:
        raise ValueError(
            f"`num_steps` ({args.num_steps}) exceeds `max_episode_steps` "
            f"({args.max_episode_steps}); a rollout longer than the episode only "
            "wastes storage."
        )
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(f"`gamma` must be in [0, 1], got {args.gamma}.")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError(f"`gae_lambda` must be in [0, 1], got {args.gae_lambda}.")
    if args.clip_coef <= 0.0:
        raise ValueError(f"`clip_coef` must be positive, got {args.clip_coef}.")
    if args.actor_learning_rate is not None and args.actor_learning_rate <= 0.0:
        raise ValueError(
            f"`actor_learning_rate` must be positive, got {args.actor_learning_rate}."
        )
    if args.kl_stop_scope not in {"policy", "all"}:
        raise ValueError(
            f"`kl_stop_scope` must be 'policy' or 'all', got {args.kl_stop_scope!r}."
        )
    if args.action_scale_source not in {"joint_limits", "demos"}:
        raise ValueError(
            "`action_scale_source` must be 'joint_limits' or 'demos', got "
            f"{args.action_scale_source!r}."
        )
    if args.action_scale_margin < 0.0:
        raise ValueError(
            f"`action_scale_margin` must be non-negative, got {args.action_scale_margin}."
        )
    if args.action_scale_source == "demos" and not args.use_bc_loss:
        raise ValueError(
            "`--action-scale-source demos` reads the per-joint action range out of the "
            "rendered demo HDF5, so it needs `--use-bc-loss` with `--bc-data-path`."
        )
    if not 0.0 <= args.bc_val_fraction < 1.0:
        raise ValueError(
            f"`bc_val_fraction` must be in [0, 1), got {args.bc_val_fraction}."
        )
    if not args.hidden_dims or any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError(f"`hidden_dims` must contain positive dims, got {args.hidden_dims}.")
    if args.image_size <= 0:
        raise ValueError(f"`image_size` must be positive, got {args.image_size}.")
    if args.scene_background is not None:
        from utils.dex_scene_eval import BACKGROUND_IDS

        if args.scene_background not in BACKGROUND_IDS:
            raise ValueError(
                f"`scene_background` must be one of {BACKGROUND_IDS}, got {args.scene_background!r}."
            )
        if not args.evaluate:
            raise ValueError("`scene_background` is an evaluation-only setting; pass `--evaluate`.")
        if args.sim_backend != "physx_cpu":
            raise ValueError("`scene_background` requires `--sim-backend physx_cpu`.")
    if args.camera_preset not in CAMERA_PRESETS:
        raise ValueError(
            f"`camera_preset` must be one of {sorted(CAMERA_PRESETS)}, got {args.camera_preset!r}."
        )
    if args.frame_stack != 1:
        raise NotImplementedError(
            "Only frame_stack=1 is implemented; stacking would have to key frames "
            "by episode in the rollout buffer."
        )
    if args.bc_data_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"`bc_data_device` must be 'auto', 'cpu' or 'cuda', got {args.bc_data_device!r}."
        )
    if args.encoder_grad_source not in PPO_GRAD_SOURCES:
        raise ValueError(
            f"`encoder_grad_source` must be one of {PPO_GRAD_SOURCES} for PPO, got "
            f"{args.encoder_grad_source!r}. PPO's actor loss does not pass through "
            "the critic, so 'critic_only' would leave the actor's encoder without "
            "any gradient, and 'bc_loss' belongs to the image-BC baseline."
        )
    if args.use_bc_loss:
        if args.bc_data_path is None:
            raise ValueError(
                "`--use_bc_loss` needs `--bc_data_path` pointing at a *rendered* "
                "image dataset (utils.pixel_dataset HDF5); the LeRobot state demos "
                "carry no frames."
            )
        if args.bc_loss_type not in {"mse", "nll"}:
            raise ValueError(f"`bc_loss_type` must be 'mse' or 'nll', got {args.bc_loss_type!r}.")
        if args.bc_coef < 0 or args.bc_min_coef < 0:
            raise ValueError("BC coefficients must be non-negative.")
        if not 0.0 < args.bc_decay_steps <= 1.0:
            raise ValueError(
                f"`bc_decay_steps` is a fraction of total_timesteps, got {args.bc_decay_steps}."
            )
    elif args.bc_warmup_steps > 0:
        raise ValueError(
            "`--bc_warmup_steps` pretrains the actor on the demo dataset, which is "
            "loaded only when `--use_bc_loss` is set. Enable it, or set warmup to 0."
        )
    if args.use_demo_init_states:
        if args.demo_init_data_path is None:
            raise ValueError(
                "`--use_demo_init_states` needs `--demo_init_data_path` pointing at "
                "the LeRobot demo dataset: restoring a state needs "
                "`observation.environment_state`, which the rendered HDF5 does not have."
            )
        if args.demo_init_curriculum not in {"uniform", "progress"}:
            raise ValueError(
                f"`demo_init_curriculum` must be 'uniform' or 'progress', got "
                f"{args.demo_init_curriculum!r}."
            )
        if not 0.0 <= args.demo_init_min_prob <= args.demo_init_prob <= 1.0:
            raise ValueError(
                "Demo-init probabilities must satisfy 0 <= min_prob <= init_prob <= 1, got "
                f"min={args.demo_init_min_prob}, init={args.demo_init_prob}."
            )
        for name in ("demo_init_decay_steps", "demo_init_curriculum_steps"):
            if not 0.0 < getattr(args, name) <= 1.0:
                raise ValueError(
                    f"`{name}` is a fraction of total_timesteps, got {getattr(args, name)}."
                )
        if not 0.0 <= args.demo_init_advance_success <= 1.0:
            raise ValueError(
                "`demo_init_advance_success` is a success rate in [0, 1], got "
                f"{args.demo_init_advance_success}."
            )
        if args.demo_init_success_window <= 0:
            raise ValueError(
                f"`demo_init_success_window` must be positive, got {args.demo_init_success_window}."
            )
    for name in ("log_freq", "log_image_freq", "eval_freq", "save_freq"):
        if getattr(args, name) <= 0:
            raise ValueError(f"`{name}` must be positive, got {getattr(args, name)}.")
    # Fails fast on an impossible encoder combination.
    encoder_config_from_args(args)
    return args


def parse_args(overrides: Optional[Dict[str, Any]] = None) -> Args:
    args = tyro.cli(Args)
    if overrides:
        for key, value in overrides.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown arg override: {key}")
            setattr(args, key, value)
    resolve_training_paths(args)
    return validate_args(args)


__all__ = [
    "Args",
    "EncoderConfig",
    "PixelObsConfig",
    "encoder_config_from_args",
    "parse_args",
    "pixel_config_from_args",
    "validate_args",
]

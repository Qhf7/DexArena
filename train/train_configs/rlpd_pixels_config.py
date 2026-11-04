"""Config for image-based RLPD (`train/train_rlpd_pixels.py`).

Defaults follow RLPD_PIXELS_SPEC §8 (128x128, two cameras, latent 64 per
camera, random shift with pad 4, UTD 8, batch 256, 10-head ensemble with a
random-pair min, LayerNorm on, entropy backup off, target entropy -dim(A)/2).

Two deliberate differences from ``rlpd_config.py``:

* ``demo_data_path`` defaults to ``None``. A pixels run is fully functional with
  no prior data at all — that is the "no demo data" configuration — in which
  case ``offline_mix_fraction`` is forced to 0 and the algorithm degenerates to
  high-UTD SAC-from-pixels with the RLPD critic. Pointing at a rendered HDF5
  dataset (``utils.pixel_dataset``) turns the 50/50 symmetric sampling back on.
* ``reward_mode`` defaults to ``"sparse"``: the pixels experiments are the
  sparse binary-reward setting of §0, and the simulator's ground-truth success
  is the reward (no reward classifier, §0).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tyro

from algo.encoders import EncoderConfig
from utils.paths import resolve_training_paths
from utils.pixel_cameras import (
    CAMERA_PRESETS,
    PixelObsConfig,
    pixel_obs_config_from_args,
)


@dataclass
class Args:
    # Experiment metadata
    # Run name; also the directory under runs/. None -> derived from env + algo.
    exp_name: Optional[str] = None
    # RNG seed for torch/numpy/env resets and for the augmentation shifts.
    seed: int = 1
    # Force deterministic cuDNN kernels. Costs a little speed, buys reproducibility.
    torch_deterministic: bool = True
    # Train on GPU when one is visible; False forces CPU (impractical for pixels).
    cuda: bool = True
    # Which CUDA device to use. None -> device 0 / whatever CUDA_VISIBLE_DEVICES gives.
    gpu_id: Optional[int] = None
    # Mirror metrics (and the obs/augmented_batch image grid) to SwanLab as well
    # as the local TensorBoard writer.
    track: bool = False
    # SwanLab project / entity the run is filed under. Only read when `track`.
    swanlab_project_name: str = "ManiSkill"
    swanlab_entity: Optional[str] = None
    # Record eval-episode videos. This renders an *extra* camera on top of the
    # policy cameras; some CabinetPour stages cannot record at all (see docs).
    capture_video: bool = False
    # Write checkpoints every `save_freq` env steps.
    save_model: bool = True
    # Evaluation-only mode: load `checkpoint`, run one eval, exit without training.
    evaluate: bool = False
    # Resume an interrupted run (full optimizer state). RLPD has no warm start.
    checkpoint: Optional[str] = None

    # Environment
    # ManiSkill task id. Must match the task the demo HDF5 was rendered from.
    env_id: str = tyro.MISSING
    # Robot to instantiate. None -> the env's own default.
    robot_uids: Optional[str] = None
    # Controller. `delta_action` below requires `pd_joint_pos`.
    control_mode: Optional[str] = "pd_ee_delta_pose"
    # Task-specific object variant index, when the env exposes one. None -> env default.
    # "physx_gpu" (vectorized) or "physx_cpu" (forces num_envs=1). None -> auto.
    sim_backend: Optional[str] = None
    # Reward the env returns. "sparse" is the {0, 1} success indicator these
    # experiments are defined on; the simulator's success flag *is* the reward.
    reward_mode: str = "sparse"
    # Replace the task's sparse reward by `info[reward_info_key]` (e.g.
    # `door_open_now` on CabinetPouringOpenCabinet-Sharpa-v1: the instantaneous predicate
    # instead of the latched stage flag). None keeps the task's reward.
    reward_info_key: Optional[str] = None
    # Time limit for a *training* episode, in env steps.
    max_episode_steps: int = 300
    # Residual (delta) joint-position control, as used by the state-based runs on
    # the CabinetPour tasks. The wrapper converts a [-1, 1] action into an
    # absolute joint target `a_t = clip(a_{t-1} + u * delta)`, so `a_{t-1}` has to
    # be part of the observation or the MDP is not Markov -- on the image path it
    # is appended to `proprio` (images are untouched). The offline dataset is
    # rewritten into the same parameterization by `to_delta_image_dataset`;
    # skipping that would put absolute joint targets in the offline half of the
    # batch and deltas in the online half, which nothing would report.
    delta_action: bool = False
    # Per-step bound on |a_t - a_{t-1}|, in radians. 0.2 is what the cabinet
    # demos' fidelity/exploration sweep picked.
    delta_action_scale: float = 0.2

    # Prior data (optional). HDF5 in the `utils.pixel_dataset` schema, rendered
    # with the *same* PixelObsConfig — the camera_config_hash is checked on load.
    demo_data_path: Optional[str] = None
    # Keep only episodes whose recorded outcome is success.
    success_only: bool = False
    # RLPD symmetric sampling: fraction of every batch drawn from the demos.
    offline_mix_fraction: float = 0.5

    # Visual observation pipeline (single source of truth: utils/pixel_cameras.py)
    # Square render resolution, per camera. Also fixes the encoder's input size.
    image_size: int = 128
    # Named set of camera poses/FOVs to render the policy observation from.
    # "ball_grasp_p0" is the pose pair the P0 gate was run against: the ball is
    # ~20 px in the third-person view and stays readable through the grasp, and
    # the wrist camera looks along the approach axis instead of across the hand.
    # "task_default" reproduces tasks/ball_grasping.py exactly and fails that gate
    # (see utils/pixel_cameras.py). Changing the preset changes the
    # camera_config_hash, i.e. it defines a different visual domain -- the demo
    # HDF5 has to be re-rendered, and the loader refuses a mismatched one.
    camera_preset: str = "cabinet_fixed"
    # SAPIEN shading pipeline used to render the policy cameras. "default" is the
    # fast rasterizer; ray-traced packs look better and are far slower. It is part
    # of the camera_config_hash, so demos and online frames cannot disagree on it.
    shader_pack: str = "default"
    # Number of consecutive frames concatenated per camera. Only 1 is implemented
    # (see validate_args); >1 would need the replay buffer to key frames by episode.
    frame_stack: int = 1

    # Encoder (spec §3.1-§3.3)
    # "d4pg" = the 4-layer CNN trained from scratch; "resnet10" = the SERL
    # ResNet-10 + spatial softmax, optionally loaded from `encoder_pretrained_path`.
    encoder_arch: str = "d4pg"
    # Latent width produced per camera; the encoder output is their concatenation
    # (2 cameras x 64 = 128), then proprio is appended for the actor/critic input.
    latent_dim_per_cam: int = 64
    # Which loss the encoder gets gradients from. "critic_only" is the spec's
    # invariant §1.2: the actor sees `z.detach()`, so only the TD loss shapes the
    # features and the actor cannot collapse them.
    encoder_grad_source: str = "critic_only"
    # "shared" = one encoder feeding actor and critic (spec default).
    # "separate_per_head" gives the actor its own copy, which only makes sense
    # with `encoder_grad_source=actor_and_critic` and doubles the encoder cost.
    encoder_sharing: str = "shared"
    # DrQ random-shift augmentation on every image entering the *critic* path.
    # This is the main regularizer for high-UTD training from pixels.
    augmentation: bool = True
    # Shift magnitude in pixels: the frame is edge-padded by this much and a
    # random crop back to `image_size` is taken. 4 on 128x128 is the DrQ-v2 recipe.
    augment_pad: int = 4
    # Hold the ResNet backbone fixed for the whole run (only the head trains).
    # This is the §9.3 ablation arm, not the fine-tuning setup.
    freeze_backbone: bool = False
    # Whether BatchNorm running statistics keep updating. `requires_grad_(False)`
    # does NOT stop them -- only `.eval()` does -- so with a frozen BatchNorm
    # backbone this is the difference between adapting to rendered images (True)
    # and reusing the ImageNet statistics (False). A no-op under GroupNorm.
    update_bn_stats: bool = True
    # Normalization inside the ResNet: "group", "batch" or "none".
    # "group" is the default because the released SERL / HIL-SERL ResNet-10 is
    # GroupNorm with 4 groups (spec §3.3 assumes BatchNorm; the actual checkpoint
    # is not). It also keeps the two §9.3 ResNet arms -- frozen pre-trained and
    # frozen random -- architecturally identical, which invariant §1.5 requires.
    # Ignored by the D4PG encoder.
    encoder_norm_layer: str = "group"
    # GroupNorm group count. 4 is what the released checkpoint was trained with;
    # a mismatch loads without error and then normalizes over the wrong partition.
    encoder_norm_groups: int = 4
    # Local path to a ResNet-10 backbone state dict (this repo's parameter names).
    # None -> random initialization.
    encoder_pretrained_path: Optional[str] = None
    # LayerNorm + tanh on the latent. Mandated for D4PG (§3.2), applied to the
    # ResNet head too so latent scale is comparable across encoder variants.
    latent_layer_norm: bool = True
    # Adam learning rate for the encoder head (and for the backbone before
    # `backbone_lr_scale` is applied).
    encoder_learning_rate: float = 3e-4
    # Fine-tuning a pre-trained backbone (`--encoder_arch resnet10` with an
    # `--encoder_pretrained_path` and WITHOUT `--freeze_backbone`). Both knobs are
    # no-ops for the from-scratch D4PG encoder, which reports no backbone group.
    #
    # The backbone trains at `encoder_learning_rate * backbone_lr_scale`. At a
    # shared rate the critic's early gradient noise overwrites the ImageNet
    # features within a few thousand updates and the run quietly becomes the
    # random-ResNet control group of §9.3 instead of the pre-trained arm.
    backbone_lr_scale: float = 0.1
    # Updates to hold the backbone frozen before fine-tuning starts, so the fresh
    # head and the critic MLP adapt to the given features first (§11: the TD
    # signal is nearly zero early under a sparse reward).
    backbone_warmup_steps: int = 0
    # How often (in updates) to recompute the `backbone_drift` metric -- the
    # distance from the loaded weights, i.e. how far fine-tuning has moved them.
    # It touches all ~5M backbone parameters, so it is cached between recomputations.
    backbone_drift_log_interval: int = 1000

    # Eval
    # Episodes between re-sampling the eval scene layout (object placement,
    # geometry). 1 = a fresh layout every episode, i.e. the honest generalization
    # number; 0 = one fixed layout for the whole run. Must be 0 to use
    # `eval_partial_reset` with >1 eval env, since a reconfigure resets all subenvs.
    eval_reconfiguration_freq: Optional[int] = 1
    # Let finished eval sub-envs restart while others keep running. Faster, but
    # it biases the episode mix toward short episodes; off for honest numbers.
    eval_partial_reset: bool = False
    # Eval envs stepped in parallel, and total episodes averaged per eval. Each
    # eval env renders both policy cameras, so this is the eval VRAM knob.
    num_eval_envs: int = 25
    num_eval_episodes: int = 50
    # Time limit for an *eval* episode, in env steps.
    num_eval_steps: int = 300
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
    # Frame rate of the recorded eval videos.
    video_fps: int = 20

    # Online collection. `utd_ratio` gradient updates run per *env step* (the
    # same convention as train_rlpd.py), so with num_envs=8 and utd_ratio=8 the
    # per-transition UTD is 1. Invariant §1.6: whatever value is used here must
    # be used for the state-based re-run it is compared against.
    num_envs: int = 8
    # Training length, counted in env steps (not gradient updates).
    total_env_steps: int = 1_000_000
    # Pure-collection warmup (env transitions) with a random policy, before any
    # gradient update.
    start_steps: int = 5_000
    # During that random warmup, sample joint targets locally around the current
    # qpos instead of uniformly over the full joint limits.
    joint_warmup_delta: float = 0.05
    # Online replay capacity in transitions, stored as uint8 frames in host RAM.
    # 2e5 transitions x ~96 KB = ~19 GB at 128x128x3 with 2 cameras.
    buffer_size: int = 200_000
    # Gradient updates per env step; see the block comment above.
    utd_ratio: int = 8

    # SAC core
    # Transitions per gradient update, split by `offline_mix_fraction`.
    batch_size: int = 256
    # Standardize the proprio vector by the demo mean/std. Images are never
    # normalized this way -- they go to the encoder as uint8/255.
    normalize_proprio: bool = True
    # Affine remap of the env reward, `r -> reward_scale * r + reward_bias`,
    # applied identically online and offline. `reward_bias=-1` turns a sparse
    # {0, 1} goal into a per-step cost, which makes the return prefer finishing early.
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Discount. Sets the effective horizon ~1/(1-gamma).
    gamma: float = 0.99
    # Polyak rate for the target critic: theta_targ <- tau*theta + (1-tau)*theta_targ.
    tau: float = 0.005
    # Width/depth of the actor and of every critic head, on top of [latent, proprio].
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    # Adam learning rates for the three optimizers (the encoder has its own above).
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    # Global gradient-norm clip applied to each optimizer separately.
    max_grad_norm: float = 10.0

    # RLPD critic ensemble
    # Q-heads in the ensemble, and how many are drawn at random per TD target to
    # take the min over (REDQ-style). LayerNorm on the heads is the stabilizer
    # that makes this safe at high UTD; turning it off is not a supported variant.
    num_qs: int = 10
    num_min_qs: int = 2
    critic_layer_norm: bool = True
    # Include the entropy term in the TD backup. RLPD's binary-reward recipe
    # leaves it off.
    backup_entropy: bool = False

    # Auto-tune the entropy temperature alpha toward `target_entropy`.
    autotune_entropy: bool = True
    # Starting alpha. With `autotune_entropy=False` it is the fixed value.
    init_temperature: float = 1.0
    # None -> -action_dim / 2 (the RLPD default), computed in the agent.
    target_entropy: Optional[float] = None

    # Logging and checkpoints, in env-step units
    # How often scalars and the obs/augmented_batch image grid are written.
    log_freq: int = 2_000
    # How often the policy is evaluated, and how often a checkpoint is written.
    eval_freq: int = 50_000
    save_freq: int = 100_000


def encoder_config_from_args(args: Args) -> EncoderConfig:
    return EncoderConfig(
        arch=args.encoder_arch,
        latent_dim_per_cam=args.latent_dim_per_cam,
        grad_source=args.encoder_grad_source,
        sharing=args.encoder_sharing,
        augmentation=args.augmentation,
        augment_pad=args.augment_pad,
        freeze_backbone=args.freeze_backbone,
        update_bn_stats=args.update_bn_stats,
        norm_layer=args.encoder_norm_layer,
        norm_groups=args.encoder_norm_groups,
        pretrained_path=args.encoder_pretrained_path,
        latent_layer_norm=args.latent_layer_norm,
    ).validate()


def pixel_config_from_args(args: Args) -> PixelObsConfig:
    return pixel_obs_config_from_args(args)


def validate_args(args: Args) -> Args:
    if args.evaluate and args.checkpoint is None:
        raise ValueError("`--checkpoint` is required when `--evaluate` is set.")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    if args.robot_uids == "":
        raise ValueError("`robot_uids` must be non-empty when provided.")
    if args.batch_size <= 0 or args.total_env_steps <= 0:
        raise ValueError("`batch_size` and `total_env_steps` must be positive.")
    if args.num_envs <= 0:
        raise ValueError(f"`num_envs` must be positive, got {args.num_envs}.")
    if args.utd_ratio <= 0:
        raise ValueError(f"`utd_ratio` must be positive, got {args.utd_ratio}.")
    if args.buffer_size < args.batch_size:
        raise ValueError("`buffer_size` must be >= `batch_size`.")
    if args.image_size <= 0:
        raise ValueError(f"`image_size` must be positive, got {args.image_size}.")
    if args.camera_preset not in CAMERA_PRESETS:
        raise ValueError(
            f"`camera_preset` must be one of {sorted(CAMERA_PRESETS)}, got {args.camera_preset!r}."
        )
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
    if args.frame_stack != 1:
        raise NotImplementedError(
            "Only frame_stack=1 is implemented: the scene starts at rest, so a "
            "single frame is Markov enough; "
            "stacking would need the replay buffer to key frames by episode, which "
            "is a deliberate, separate change."
        )
    if not 0.0 <= args.offline_mix_fraction <= 1.0:
        raise ValueError(
            f"`offline_mix_fraction` must be in [0, 1], got {args.offline_mix_fraction}."
        )
    if args.demo_data_path is None and args.offline_mix_fraction > 0.0:
        print(
            "[config] no --demo_data_path given: offline_mix_fraction "
            f"{args.offline_mix_fraction} -> 0.0 (online-only run; RLPD's symmetric "
            "sampling is inactive, the critic ensemble / LayerNorm / UTD recipe is not)."
        )
        args.offline_mix_fraction = 0.0
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(f"`gamma` must be in [0, 1], got {args.gamma}.")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError(f"`tau` must be in (0, 1], got {args.tau}.")
    if args.num_qs <= 0:
        raise ValueError(f"`num_qs` must be positive, got {args.num_qs}.")
    if not 0 < args.num_min_qs <= args.num_qs:
        raise ValueError(
            f"`num_min_qs` must be in (0, num_qs={args.num_qs}], got {args.num_min_qs}."
        )
    if args.init_temperature <= 0.0:
        raise ValueError(f"`init_temperature` must be positive, got {args.init_temperature}.")
    if not args.hidden_dims or any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError(f"`hidden_dims` must contain positive dims, got {args.hidden_dims}.")
    if args.joint_warmup_delta <= 0.0:
        raise ValueError(
            f"`joint_warmup_delta` must be positive, got {args.joint_warmup_delta}."
        )
    for name in ("log_freq", "eval_freq", "save_freq"):
        if getattr(args, name) <= 0:
            raise ValueError(f"`{name}` must be positive, got {getattr(args, name)}.")
    if args.delta_action:
        if args.control_mode != "pd_joint_pos":
            raise ValueError(
                f"`delta_action` parameterizes absolute joint targets; it needs "
                f"control_mode='pd_joint_pos', got {args.control_mode!r}."
            )
        if args.delta_action_scale <= 0.0:
            raise ValueError(
                f"`delta_action_scale` must be positive, got {args.delta_action_scale}."
            )
        if args.demo_data_path is None:
            raise ValueError(
                "`delta_action` seeds a_{-1} from the mean first action of the demos, "
                "so it needs `--demo_data_path`."
            )
    if not 0.0 <= args.backbone_lr_scale <= 1.0:
        raise ValueError(
            f"`backbone_lr_scale` must be in [0, 1], got {args.backbone_lr_scale}. "
            "Above 1 would train the pre-trained backbone faster than the randomly "
            "initialized head."
        )
    if args.backbone_warmup_steps < 0:
        raise ValueError(
            f"`backbone_warmup_steps` must be non-negative, got {args.backbone_warmup_steps}."
        )
    if args.encoder_arch == "resnet10" and not args.freeze_backbone:
        if args.encoder_pretrained_path is None:
            print(
                "[config] --encoder_arch resnet10 without --encoder_pretrained_path and "
                "without --freeze_backbone: this trains a ResNet-10 from scratch. That is "
                "neither arm of the spec's §9.3 ablation (which has *frozen* pre-trained "
                "and *frozen* random). Pass --encoder_pretrained_path to fine-tune from "
                "ImageNet."
            )
        elif args.encoder_norm_layer == "batch":
            print(
                "[config] fine-tuning with --encoder_norm_layer batch. The released "
                "ResNet-10 weights are GroupNorm; loading them here will fail the norm "
                "check in load_pretrained_backbone. Use --encoder_norm_layer group "
                "(--encoder_norm_groups 4) unless these are your own BatchNorm weights."
            )
    if args.freeze_backbone and args.backbone_warmup_steps > 0:
        raise ValueError(
            "`--freeze_backbone` pins the backbone for the whole run, so "
            "`--backbone_warmup_steps` would never take effect. Drop one of them: "
            "the warmup is for fine-tuning runs."
        )
    # Fails fast on an impossible encoder combination (e.g. a per-head encoder
    # with critic-only gradients).
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

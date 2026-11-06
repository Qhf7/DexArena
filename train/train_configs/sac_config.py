from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tyro

from utils.paths import demo_path, resolve_training_paths, runs_path


@dataclass
class Args:
    # Experiment metadata
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    gpu_id: Optional[int] = None
    track: bool = False
    swanlab_project_name: str = "ManiSkill"
    swanlab_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = True
    evaluate: bool = False
    # SAC's own checkpoint to resume from (full optimizer state). For warm-starting
    # from an offline IQL/CQL run use `--pretrained-checkpoint` instead.
    checkpoint: Optional[str] = None
    # Offline-pretrained checkpoint to warm-start actor + twin Q + targets from.
    pretrained_checkpoint: Optional[str] = runs_path("ball_grasp_iql", "final_ckpt.pt")

    # Environment and offline demo data (demos are mixed into online batches)
    env_id: str = tyro.MISSING
    control_mode: Optional[str] = "pd_joint_pos"
    demo_data_path: str = demo_path("teleop_lerobot")
    sim_backend: Optional[str] = None
    reward_mode: str = "sparse"
    max_episode_steps: int = 200

    # Eval (reuses train_iql.make_eval_env / evaluate_policy)
    eval_reconfiguration_freq: Optional[int] = 1
    eval_partial_reset: bool = False
    num_eval_envs: int = 8
    num_eval_episodes: int = 16
    num_eval_steps: int = 200
    video_fps: int = 20

    # Online collection
    num_envs: int = 16
    total_env_steps: int = 1_000_000
    # Pure-collection warmup (env transitions) before any gradient update.
    start_steps: int = 1_000
    buffer_size: int = 1_000_000
    # Gradient updates performed per vectorized env step (UTD = updates_per_iter / num_envs).
    updates_per_iter: int = 16
    # RLPD-style symmetric sampling: fraction of every batch drawn from offline demos.
    offline_mix_fraction: float = 0.5

    # SAC core
    batch_size: int = 256
    normalize_observations: bool = True
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    gamma: float = 0.99
    tau: float = 0.005
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_grad_norm: float = 10.0
    policy_update_interval: int = 1
    target_update_interval: int = 1

    # Entropy temperature. Autotuning is disabled by default: on these tasks it
    # formed a positive-feedback loop (variance collapse -> log_prob spike ->
    # alpha blow-up -> Q divergence, see cql_config). A small fixed alpha keeps
    # mild online exploration without that risk.
    autotune_entropy: bool = False
    entropy_alpha: float = 0.05
    target_entropy: Optional[float] = None

    # Actor objective (TD3+BC style, mirrors cql_config). `actor_q_weight` scales
    # the magnitude-normalized Q term; `actor_bc_weight` keeps a persistent
    # behavior-cloning anchor toward demo actions so the policy stays in the data
    # distribution while the online critic is still noisy (anti-unlearning).
    actor_q_weight: float = 2.5
    actor_bc_weight: float = 1.0
    actor_bc_min_weight: float = 0.0
    bc_decay: bool = True
    bc_decay_steps: int = 200_000

    # Logging and checkpoints, in env-step units
    log_freq: int = 2_000
    eval_freq: int = 20_000
    save_freq: int = 100_000


def validate_args(args: Args) -> Args:
    if args.evaluate and args.checkpoint is None and args.pretrained_checkpoint is None:
        raise ValueError(
            "`--checkpoint` or `--pretrained-checkpoint` is required when `--evaluate` is set."
        )
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    if args.batch_size <= 0 or args.total_env_steps <= 0:
        raise ValueError("`batch_size` and `total_env_steps` must be positive.")
    if args.num_envs <= 0:
        raise ValueError(f"`num_envs` must be positive, got {args.num_envs}.")
    if args.updates_per_iter <= 0:
        raise ValueError(f"`updates_per_iter` must be positive, got {args.updates_per_iter}.")
    if args.buffer_size < args.batch_size:
        raise ValueError("`buffer_size` must be >= `batch_size`.")
    if not 0.0 <= args.offline_mix_fraction <= 1.0:
        raise ValueError(
            f"`offline_mix_fraction` must be in [0, 1], got {args.offline_mix_fraction}."
        )
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(f"`gamma` must be in [0, 1], got {args.gamma}.")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError(f"`tau` must be in (0, 1], got {args.tau}.")
    if args.actor_bc_weight < 0.0 or args.actor_bc_min_weight < 0.0:
        raise ValueError("`actor_bc_weight` and `actor_bc_min_weight` must be non-negative.")
    if args.actor_q_weight < 0.0:
        raise ValueError(f"`actor_q_weight` must be non-negative, got {args.actor_q_weight}.")
    if not args.hidden_dims or any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError(f"`hidden_dims` must contain positive dimensions, got {args.hidden_dims}.")
    positive_intervals = {
        "policy_update_interval": args.policy_update_interval,
        "target_update_interval": args.target_update_interval,
        "log_freq": args.log_freq,
        "eval_freq": args.eval_freq,
        "save_freq": args.save_freq,
        "bc_decay_steps": args.bc_decay_steps,
    }
    for name, value in positive_intervals.items():
        if value <= 0:
            raise ValueError(f"`{name}` must be positive, got {value}.")
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

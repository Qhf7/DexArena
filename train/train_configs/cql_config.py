from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tyro

from utils.paths import demo_path, resolve_training_paths


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
    checkpoint: Optional[str] = None

    # Environment and offline data
    env_id: str = tyro.MISSING
    control_mode: Optional[str] = "pd_joint_pos"
    demo_data_path: str = demo_path("teleop_lerobot")
    extra_npz_paths: Optional[Tuple[str, ...]] = None
    sim_backend: Optional[str] = None
    eval_reconfiguration_freq: Optional[int] = 1
    # Must stay False while eval_reconfiguration_freq != 0 with >1 eval env:
    # ManiSkillVectorEnv forbids partial resets together with auto-reconfiguration
    # (mirrors train_ppo's eval setup).
    eval_partial_reset: bool = False
    num_eval_envs: int = 4
    num_eval_episodes: int = 4
    num_eval_steps: int = 5000
    video_fps: int = 20
    reward_mode: str = "sparse"

    # Offline training
    bc_only: bool = False
    bc_loss_type: str = "mse"
    total_updates: int = 500_000
    batch_size: int = 256
    actor_batch_size: Optional[int] = None
    actor_demo_only: bool = True
    critic_extra_fraction: Optional[float] = 0.5
    bootstrap_extra_transitions: bool = False
    normalize_observations: bool = True
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    gamma: float = 0.99
    tau: float = 0.005
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-4
    alpha_learning_rate: float = 3e-4
    # Persistent BC anchor on the actor (TD3+BC style). `actor_q_weight` is the
    # TD3+BC alpha that scales the (normalized) Q term; `actor_bc_weight` weights
    # the behavior-cloning negative log-likelihood term kept on every update.
    actor_bc_weight: float = 1.0
    actor_q_weight: float = 2.5
    max_grad_norm: float = 10.0
    policy_update_interval: int = 1
    target_update_interval: int = 1

    # SAC entropy. Autotuning is disabled by default: with the persistent BC
    # anchor it is unnecessary, and on this task it formed a positive-feedback
    # loop (policy variance collapse -> log_prob spike -> alpha blow-up -> Q
    # divergence). entropy_alpha=0.0 then disables the entropy term entirely.
    autotune_entropy: bool = False
    entropy_alpha: float = 0.0
    target_entropy: Optional[float] = None
    bc_warmup_steps: int = 10_000

    # Conservative Q-learning
    cql_alpha: float = 5.0
    cql_temperature: float = 1.0
    cql_num_actions: int = 10
    cql_importance_sample: bool = True

    # Logging and checkpoints, in optimizer updates
    log_freq: int = 100
    eval_freq: int = 100_000
    save_freq: int = 100_000


def validate_args(args: Args) -> Args:
    if args.evaluate and args.checkpoint is None:
        raise ValueError("`--checkpoint` is required when `--evaluate` is set.")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    if args.batch_size <= 0 or args.total_updates <= 0:
        raise ValueError("`batch_size` and `total_updates` must be positive.")
    if args.actor_batch_size is not None and args.actor_batch_size <= 0:
        raise ValueError(f"`actor_batch_size` must be positive, got {args.actor_batch_size}.")
    if args.critic_extra_fraction is not None and not 0.0 <= args.critic_extra_fraction <= 1.0:
        raise ValueError(
            f"`critic_extra_fraction` must be in [0, 1], got {args.critic_extra_fraction}."
        )
    if args.bc_loss_type not in {"mse", "nll"}:
        raise ValueError(f"`bc_loss_type` must be 'mse' or 'nll', got {args.bc_loss_type}.")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(f"`gamma` must be in [0, 1], got {args.gamma}.")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError(f"`tau` must be in (0, 1], got {args.tau}.")
    if args.cql_alpha < 0.0:
        raise ValueError(f"`cql_alpha` must be non-negative, got {args.cql_alpha}.")
    if args.actor_bc_weight < 0.0 or args.actor_q_weight < 0.0:
        raise ValueError(
            "`actor_bc_weight` and `actor_q_weight` must be non-negative, got "
            f"{args.actor_bc_weight}, {args.actor_q_weight}."
        )
    if args.cql_temperature <= 0.0 or args.cql_num_actions <= 0:
        raise ValueError("`cql_temperature` and `cql_num_actions` must be positive.")
    if not args.hidden_dims or any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError(f"`hidden_dims` must contain positive dimensions, got {args.hidden_dims}.")
    positive_intervals = {
        "policy_update_interval": args.policy_update_interval,
        "target_update_interval": args.target_update_interval,
        "log_freq": args.log_freq,
        "eval_freq": args.eval_freq,
        "save_freq": args.save_freq,
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

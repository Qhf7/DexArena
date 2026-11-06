from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tyro

from utils.paths import demo_path, resolve_training_paths, runs_path


@dataclass
class Args:
    # Experiment metadata
    exp_name: Optional[str] = "ball_grasp_bc_rollout_iql_online"
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
    checkpoint: Optional[str] = runs_path("ball_grasp_bc_rollout_iql", "final_ckpt.pt")
    load_optimizer_state: bool = False

    # Environment and offline demo data. Demos are mixed into online batches to
    # reduce unlearning, and optionally serve as the actor BC anchor.
    env_id: str = tyro.MISSING
    control_mode: Optional[str] = "pd_joint_pos"
    demo_data_path: str = demo_path("teleop_lerobot")
    success_only: bool = True
    sim_backend: Optional[str] = None
    reward_mode: str = "normalized_dense"
    max_episode_steps: int = 300

    # Eval
    eval_reconfiguration_freq: Optional[int] = 1
    eval_partial_reset: bool = False
    num_eval_envs: int = 100
    num_eval_episodes: int = 100
    num_eval_steps: int = 300
    video_fps: int = 20

    # Online collection
    num_envs: int = 16
    total_env_steps: int = 1_000_000
    start_steps: int = 0
    buffer_size: int = 1_000_000
    updates_per_iter: int = 16
    offline_mix_fraction: float = 0.5

    # IQL core
    batch_size: int = 256
    actor_batch_size: Optional[int] = None
    bc_loss_type: str = "mse"
    bc_warmup_steps: int = 0
    actor_bc_weight: float = 0
    bc_gripper_weight: float = 2.0
    normalize_observations: bool = True
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    gamma: float = 0.99
    tau: float = 0.005
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    max_grad_norm: float = 10.0
    policy_update_interval: int = 1
    target_update_interval: int = 1

    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    iql_adv_clip: float = 100.0
    normalize_awr_weights: bool = True
    awr_positive_adv_only: bool = False

    # These flags are no-ops for normal online batches, but IQLAgent expects
    # them because the offline trainer can also consume branch transitions.
    actor_demo_only: bool = False
    bc_demo_only: bool = True
    critic_extra_fraction: Optional[float] = None
    bootstrap_extra_transitions: bool = True
    extra_target_q_clip: Optional[float] = None
    value_demo_only: bool = False

    # Logging and checkpoints, in env-step units
    log_freq: int = 2_000
    eval_freq: int = 20_000
    save_freq: int = 100_000


def validate_args(args: Args) -> Args:
    if args.evaluate and args.checkpoint is None:
        raise ValueError("`--checkpoint` is required when `--evaluate` is set.")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    if args.batch_size <= 0 or args.total_env_steps <= 0:
        raise ValueError("`batch_size` and `total_env_steps` must be positive.")
    if args.actor_batch_size is not None and args.actor_batch_size <= 0:
        raise ValueError(f"`actor_batch_size` must be positive, got {args.actor_batch_size}.")
    if args.num_envs <= 0:
        raise ValueError(f"`num_envs` must be positive, got {args.num_envs}.")
    if args.start_steps < 0:
        raise ValueError(f"`start_steps` must be non-negative, got {args.start_steps}.")
    if args.updates_per_iter <= 0:
        raise ValueError(f"`updates_per_iter` must be positive, got {args.updates_per_iter}.")
    if args.buffer_size < args.batch_size:
        raise ValueError("`buffer_size` must be >= `batch_size`.")
    if not 0.0 <= args.offline_mix_fraction <= 1.0:
        raise ValueError(
            f"`offline_mix_fraction` must be in [0, 1], got {args.offline_mix_fraction}."
        )
    if args.bc_loss_type not in {"mse", "nll"}:
        raise ValueError(f"`bc_loss_type` must be 'mse' or 'nll', got {args.bc_loss_type}.")
    if args.bc_warmup_steps < 0:
        raise ValueError(f"`bc_warmup_steps` must be non-negative, got {args.bc_warmup_steps}.")
    if args.actor_bc_weight < 0.0:
        raise ValueError(f"`actor_bc_weight` must be non-negative, got {args.actor_bc_weight}.")
    if args.bc_gripper_weight <= 0.0:
        raise ValueError(f"`bc_gripper_weight` must be positive, got {args.bc_gripper_weight}.")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(f"`gamma` must be in [0, 1], got {args.gamma}.")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError(f"`tau` must be in (0, 1], got {args.tau}.")
    if not 0.5 <= args.iql_expectile < 1.0:
        raise ValueError(f"`iql_expectile` must be in [0.5, 1), got {args.iql_expectile}.")
    if args.iql_beta <= 0.0:
        raise ValueError(f"`iql_beta` must be positive, got {args.iql_beta}.")
    if args.iql_adv_clip <= 0.0:
        raise ValueError(f"`iql_adv_clip` must be positive, got {args.iql_adv_clip}.")
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

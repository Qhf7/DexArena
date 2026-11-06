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
    # OPT's own checkpoint to resume from (full optimizer state). For warm-starting
    # from an offline IQL run use `--pretrained-checkpoint` instead.
    checkpoint: Optional[str] = None
    # Offline-pretrained IQL checkpoint: warm-starts actor + Q^off/V^off + obs stats.
    pretrained_checkpoint: Optional[str] = runs_path("ball_grasp_iql", "final_ckpt.pt")

    # Environment and offline demo data (demos are mixed into online batches)
    env_id: str = tyro.MISSING
    control_mode: Optional[str] = "pd_joint_pos"
    demo_data_path: str = demo_path("teleop_lerobot")
    # Keep only successful demonstrations (any frame reaches `next.success`).
    # Failed/timeout demos are dropped before any batch is sampled. Disable with
    # `--no-success-only` to mix the full dataset into online batches.
    success_only: bool = True
    sim_backend: Optional[str] = None
    reward_mode: str = "sparse"
    max_episode_steps: int = 300

    # Eval (reuses train_iql.make_eval_env / evaluate_policy)
    eval_reconfiguration_freq: Optional[int] = 1
    eval_partial_reset: bool = False
    num_eval_envs: int = 20
    num_eval_episodes: int = 100
    num_eval_steps: int = 300
    video_fps: int = 20

    # Online collection
    num_envs: int = 16
    total_env_steps: int = 1_000_000
    # Pure-collection warmup (env transitions) before any gradient update.
    start_steps: int = 1_000
    buffer_size: int = 1_000_000
    # Gradient updates performed per vectorized env step.
    updates_per_iter: int = 16
    # RLPD-style symmetric sampling: fraction of every batch drawn from offline demos.
    offline_mix_fraction: float = 0.5

    # --- OPT: online pre-training (Phase B) ---
    # Env steps collected with the frozen offline policy before fine-tuning.
    opt_env_steps: int = 25_000
    # Gradient steps to meta-train the fresh Q^on/V^on (paper uses ~2x env steps).
    opt_grad_steps: int = 50_000
    # Inner virtual-step size in the Eq. 3 meta-adaptation objective.
    meta_lr: float = 3e-4
    # Drop the second-order term (FOMAML) to save memory/compute.
    first_order: bool = False

    # --- OPT: fine-tuning combined advantage (Phase C) ---
    # AWR uses A = (1-kappa)*A_off + kappa*A_on; kappa is annealed start->end
    # over `kappa_schedule_steps` gradient updates (lean on stable Q^off early,
    # the faster-adapting Q^on later).
    kappa_start: float = 0.1
    kappa_end: float = 0.9
    kappa_schedule_steps: int = 200_000

    # IQL core
    batch_size: int = 256
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

    # Persistent, decaying BC anchor toward demo actions (anti-unlearning).
    actor_bc_weight: float = 1.0
    actor_bc_min_weight: float = 0.0
    bc_gripper_weight: float = 5.0
    bc_decay: bool = True
    bc_decay_steps: int = 200_000

    # Logging and checkpoints, in env-step units
    log_freq: int = 10_000
    eval_freq: int = 100_000
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
    if args.opt_env_steps < 0 or args.opt_grad_steps < 0:
        raise ValueError("`opt_env_steps` and `opt_grad_steps` must be non-negative.")
    if args.opt_env_steps >= args.total_env_steps:
        raise ValueError("`opt_env_steps` must be < `total_env_steps`.")
    if args.meta_lr <= 0.0:
        raise ValueError(f"`meta_lr` must be positive, got {args.meta_lr}.")
    if not 0.0 <= args.kappa_start <= 1.0 or not 0.0 <= args.kappa_end <= 1.0:
        raise ValueError("`kappa_start` and `kappa_end` must be in [0, 1].")
    if args.kappa_schedule_steps <= 0:
        raise ValueError(
            f"`kappa_schedule_steps` must be positive, got {args.kappa_schedule_steps}."
        )
    if not 0.0 <= args.offline_mix_fraction <= 1.0:
        raise ValueError(
            f"`offline_mix_fraction` must be in [0, 1], got {args.offline_mix_fraction}."
        )
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
    if args.actor_bc_weight < 0.0 or args.actor_bc_min_weight < 0.0:
        raise ValueError("`actor_bc_weight` and `actor_bc_min_weight` must be non-negative.")
    if args.bc_gripper_weight <= 0.0:
        raise ValueError(f"`bc_gripper_weight` must be positive, got {args.bc_gripper_weight}.")
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

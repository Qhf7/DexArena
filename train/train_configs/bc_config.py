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
    # None keeps the environment's own default robot.
    robot_uids: Optional[str] = None
    control_mode: Optional[str] = "pd_joint_pos"
    demo_data_path: str = demo_path("teleop_lerobot")
    sim_backend: Optional[str] = None
    reward_mode: str = "sparse"
    # Drop the 7D `object_pose` feature from the flattened state observation.
    drop_object_pose: bool = False
    # Keep only successful demonstrations (any frame reaches `next.success`).
    # Failed/timeout demos are dropped before any batch is sampled. Disable with
    # `--no-success-only` to train on the full dataset.
    success_only: bool = False

    # Eval
    eval_reconfiguration_freq: Optional[int] = 1
    eval_partial_reset: bool = False
    num_eval_envs: int = 100
    num_eval_episodes: int = 100
    num_eval_steps: int = 300
    video_fps: int = 20

    # Offline training. Pure BC converges quickly, so the default budget is
    # smaller than the RL agents'; override on the command line if needed.
    total_updates: int = 100_000
    batch_size: int = 256
    normalize_observations: bool = True
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    learning_rate: float = 3e-4
    max_grad_norm: float = 10.0

    # BC specifics: weight on the gripper (last) action dimension in the MSE.
    # 1.0 = no weighting (plain BC); raise it when gripper open/close is hard to fit.
    bc_gripper_weight: float = 1.0

    # Logging and checkpoints, in optimizer updates
    log_freq: int = 100
    eval_freq: int = 10_000
    save_freq: int = 50_000


def validate_args(args: Args) -> Args:
    if args.evaluate and args.checkpoint is None:
        raise ValueError("`--checkpoint` is required when `--evaluate` is set.")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    if args.batch_size <= 0 or args.total_updates <= 0:
        raise ValueError("`batch_size` and `total_updates` must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError(f"`learning_rate` must be positive, got {args.learning_rate}.")
    if args.bc_gripper_weight <= 0.0:
        raise ValueError(f"`bc_gripper_weight` must be positive, got {args.bc_gripper_weight}.")
    if not args.hidden_dims or any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError(f"`hidden_dims` must contain positive dimensions, got {args.hidden_dims}.")
    positive_intervals = {
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

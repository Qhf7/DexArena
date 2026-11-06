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

    # Frozen behavior-Q from the FQE/SARSA stage (required unless --evaluate with
    # a self-contained AWR checkpoint, which already embeds the frozen Q).
    fqe_checkpoint: Optional[str] = None

    # Environment and offline data
    env_id: str = tyro.MISSING
    control_mode: Optional[str] = "pd_joint_pos"
    demo_data_path: str = demo_path("teleop_lerobot")
    sim_backend: Optional[str] = None
    eval_reconfiguration_freq: Optional[int] = 1
    eval_partial_reset: bool = False
    num_eval_envs: int = 4
    num_eval_episodes: int = 4
    num_eval_steps: int = 500
    video_fps: int = 20
    reward_mode: str = "sparse"

    # Offline training
    total_updates: int = 200_000
    batch_size: int = 256
    normalize_observations: bool = True
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    actor_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    max_grad_norm: float = 10.0
    policy_update_interval: int = 1

    # Advantage-weighted regression. `iql_expectile` is the asymmetry of the
    # value expectile regression toward the frozen Q (tau in [0.5, 1); higher
    # pushes V toward the upper expectile of Q). `iql_beta` is the inverse
    # temperature of the advantage weighting; `iql_adv_clip` caps the
    # exponentiated weight for numerical stability.
    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    iql_adv_clip: float = 100.0

    # Logging and checkpoints, in optimizer updates
    log_freq: int = 100
    eval_freq: int = 20_000
    save_freq: int = 50_000


def validate_args(args: Args) -> Args:
    if args.evaluate and args.checkpoint is None:
        raise ValueError("`--checkpoint` is required when `--evaluate` is set.")
    if not args.evaluate and args.fqe_checkpoint is None and args.checkpoint is None:
        raise ValueError(
            "`--fqe_checkpoint` is required for training (the frozen behavior Q)."
        )
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    if args.batch_size <= 0 or args.total_updates <= 0:
        raise ValueError("`batch_size` and `total_updates` must be positive.")
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

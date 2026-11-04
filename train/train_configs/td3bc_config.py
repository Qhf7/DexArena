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
    # Optional pure-BC checkpoint used only to initialize the actor before TD3+BC
    # training. This is different from `checkpoint`, which resumes a full TD3+BC
    # agent including critics and optimizers.
    bc_checkpoint: Optional[str] = None

    # Environment and offline data
    env_id: str = tyro.MISSING
    control_mode: Optional[str] = "pd_ee_delta_pose"
    demo_data_path: str = demo_path("teleop_lerobot")
    sim_backend: Optional[str] = None
    reward_mode: str = "normalized_dense"
    # Keep only successful demonstrations (any frame reaches `next.success`).
    # Failed/timeout demos are dropped before any batch is sampled. Disable with
    # `--no-success-only` to train on the full dataset.
    success_only: bool = True

    # Eval
    eval_reconfiguration_freq: Optional[int] = 1
    eval_partial_reset: bool = False
    num_eval_envs: int = 20
    num_eval_episodes: int = 20
    num_eval_steps: int = 300
    video_fps: int = 20

    # Offline training
    total_updates: int = 500_000
    batch_size: int = 256
    normalize_observations: bool = True
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    gamma: float = 0.99
    tau: float = 0.005
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    max_grad_norm: float = 10.0

    # TD3+BC specifics. `policy_freq` delays the actor + target updates; the
    # target-policy-smoothing noise (`policy_noise`, clipped to `noise_clip`) is
    # given in normalized action units and scaled into each dim's range. `alpha`
    # is the coefficient of the adaptive lambda = alpha / mean(|Q|) that balances
    # the RL term against the unit-weighted BC term.
    policy_freq: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    alpha: float = 2.5

    # Logging and checkpoints, in optimizer updates
    log_freq: int = 100
    eval_freq: int = 50_000
    save_freq: int = 100_000


def validate_args(args: Args) -> Args:
    if args.evaluate and args.checkpoint is None:
        raise ValueError("`--checkpoint` is required when `--evaluate` is set.")
    if args.checkpoint is not None and args.bc_checkpoint is not None:
        raise ValueError("Use either `--checkpoint` or `--bc-checkpoint`, not both.")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    if args.batch_size <= 0 or args.total_updates <= 0:
        raise ValueError("`batch_size` and `total_updates` must be positive.")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(f"`gamma` must be in [0, 1], got {args.gamma}.")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError(f"`tau` must be in (0, 1], got {args.tau}.")
    if args.alpha < 0.0:
        raise ValueError(f"`alpha` must be non-negative, got {args.alpha}.")
    if args.policy_noise < 0.0 or args.noise_clip < 0.0:
        raise ValueError("`policy_noise` and `noise_clip` must be non-negative.")
    if not args.hidden_dims or any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError(f"`hidden_dims` must contain positive dimensions, got {args.hidden_dims}.")
    positive_intervals = {
        "policy_freq": args.policy_freq,
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

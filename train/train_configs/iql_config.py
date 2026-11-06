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
    # Optional .npz files of explicit (s,a,r,s',done) counterfactual-branch
    # transitions: one off-trajectory env step taken from a recorded demo state.
    # The IQL critic learns their value via the r+gamma*V(s') target; they have
    # no valid next action, so only the V-bootstrap path may consume them.
    extra_npz_paths: Optional[Tuple[str, ...]] = None
    sim_backend: Optional[str] = None
    # Keep only successful demonstrations (any frame reaches `next.success`).
    # Failed/timeout demos are dropped before any batch is sampled. Disable with
    # `--no-success-only` to train on the full dataset.
    success_only: bool = True
    eval_reconfiguration_freq: Optional[int] = 1
    # Must stay False while eval_reconfiguration_freq != 0 with >1 eval env:
    # ManiSkillVectorEnv forbids partial resets together with auto-reconfiguration
    # (mirrors train_ppo's eval setup).
    eval_partial_reset: bool = False
    num_eval_envs: int = 20
    num_eval_episodes: int = 100
    num_eval_steps: int = 300
    video_fps: int = 20
    reward_mode: str = "sparse"

    # Offline training
    bc_only: bool = False
    bc_loss_type: str = "mse"
    # Actor-side BC stabilization. IQL's policy step is still weighted behavior
    # cloning; on staged manipulation tasks the critic can overweight early reach
    # actions and underweight gripper/lift actions. A short pure-BC warmup plus a
    # persistent BC anchor keeps the extracted policy faithful to the full demo.
    bc_warmup_steps: int = 20_000
    actor_bc_weight: float = 1.0
    bc_gripper_weight: float = 5.0
    total_updates: int = 500_000
    batch_size: int = 256
    actor_batch_size: Optional[int] = None
    # When extra branch transitions are loaded, keep policy extraction on the
    # original demonstrations by default. The branch transitions are still useful
    # for Q/V, but directly imitating their perturbed actions often makes the
    # final policy look like a noisy BC policy.
    actor_demo_only: bool = True
    # When the actor batch includes branch transitions (actor_demo_only=False),
    # the BC anchor must NOT plain-imitate the perturbed branch actions or it
    # degrades into noisy BC. Keep the anchor on demo rows (is_extra==False) only.
    # No-op when the actor is demo-only.
    bc_demo_only: bool = True
    # Fraction of critic batch sampled from extra .npz branch transitions. `None`
    # means natural replay ratio; with K=8 branches that is usually too branch-heavy.
    critic_extra_fraction: Optional[float] = 0.5
    # Extra branch transitions are one-step counterfactual probes. Their next
    # states can be off the demo manifold, so bootstrapping through V(s') can make
    # Q/V self-amplify. By default, use branch transitions as immediate-reward
    # action-ranking samples: target_q = r for extra rows, while demos still use
    # r + gamma * V(s').
    bootstrap_extra_transitions: bool = False
    # Cap the bootstrapped target for branch rows (only used when
    # bootstrap_extra_transitions=True). Bounds any V(s') extrapolation on
    # off-manifold branch next states. ~max demo return-to-go is a safe value.
    extra_target_q_clip: Optional[float] = None
    # Fit V (expectile value) on demo rows only. Branch actions provide Q action
    # spread but must not redefine the data-policy value, or they drag V down and
    # collapse the return scale. No-op when no branch transitions are loaded.
    value_demo_only: bool = True
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

    # Implicit Q-Learning. `iql_expectile` is the asymmetry of the value
    # expectile regression (tau in [0.5, 1); higher pushes V toward the upper
    # expectile of Q, i.e. more optimistic). `iql_beta` is the inverse
    # temperature of the advantage-weighted regression used for policy
    # extraction; `iql_adv_clip` caps the exponentiated advantage weight to
    # keep the policy loss numerically stable.
    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    iql_adv_clip: float = 100.0
    normalize_awr_weights: bool = True
    # Zero the AWR weight on non-improving actions (adv<=0) so only advantage-
    # improving actions are imitated. With branch transitions in the actor batch
    # this is what selects the decisive high-advantage actions (e.g. timely
    # gripper close) and drops the random bad perturbations.
    awr_positive_adv_only: bool = False

    # Logging and checkpoints, in optimizer updates
    log_freq: int = 100
    eval_freq: int = 50_000
    save_freq: int = 50_000


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
    if args.extra_target_q_clip is not None and args.extra_target_q_clip <= 0.0:
        raise ValueError(
            f"`extra_target_q_clip` must be positive, got {args.extra_target_q_clip}."
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

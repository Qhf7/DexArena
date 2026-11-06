import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tyro

from utils.delta_action import DEFAULT_DELTA
from utils.paths import demo_path, resolve_training_paths


@dataclass
class Args:
    # Experiment metadata
    # Run name; also the directory under runs/. None -> derived from env + algo.
    exp_name: Optional[str] = None
    # RNG seed for torch/numpy/env resets. Changing it is the only difference
    # between "seeds" of the same configuration.
    seed: int = 1
    # Force deterministic cuDNN kernels. Costs a little speed, buys reproducibility.
    torch_deterministic: bool = True
    # Train on GPU when one is visible; False forces CPU.
    cuda: bool = True
    # Which CUDA device to use. None -> device 0 / whatever CUDA_VISIBLE_DEVICES gives.
    gpu_id: Optional[int] = None
    # Mirror metrics to SwanLab in addition to the local TensorBoard writer.
    track: bool = False
    # SwanLab project / entity the run is filed under. Only read when `track`.
    swanlab_project_name: str = "ManiSkill"
    swanlab_entity: Optional[str] = None
    # Record eval-episode videos. Needs a GPU renderer and costs eval wall time.
    capture_video: bool = False
    # Write checkpoints every `save_freq` env steps.
    save_model: bool = True
    # Evaluation-only mode: load `checkpoint`, run one eval, exit without training.
    evaluate: bool = False
    # RLPD trains from scratch (no offline pretraining / warm-start). `--checkpoint`
    # is only for resuming an interrupted RLPD run (full optimizer state).
    checkpoint: Optional[str] = None

    # Environment and offline demo data (demos are mixed 50/50 into online batches)
    # ManiSkill task id to train and evaluate on.
    env_id: str = tyro.MISSING
    # None -> infer from the LeRobot dataset's meta/info.json when possible.
    # Use `--robot-uids panda_sharpa_hand_right` to force Sharpa explicitly.
    robot_uids: Optional[str] = None
    # ``pd_joint_pos`` is backed by the repo's Franka joint-impedance arm.
    control_mode: Optional[str] = "pd_ee_delta_pose"
    # LeRobot dataset root (or glob) holding the prior data. Must have been
    # collected on the same env/robot/control mode as the online env.
    demo_data_path: str = demo_path("teleop_lerobot")
    # Keep only episodes whose recorded outcome is success. Failed demos still
    # carry useful coverage for a Q-learner, so this is off by default.
    success_only: bool = False
    # Task-specific object variant index, when the env exposes one. None -> env default.
    # "physx_gpu" (vectorized) or "physx_cpu" (forces num_envs=1). None -> auto.
    sim_backend: Optional[str] = None
    # Which reward the env returns. "sparse" is the {0, 1} goal indicator the
    # cabinet runs use; "normalized_dense" is the shaped one.
    reward_mode: str = "normalized_dense"
    # Time limit for a *training* episode, in env steps.
    max_episode_steps: int = 300

    # Residual action parameterization (see utils/delta_action.py). Required for
    # `normalize_action=False` controllers such as the Sharpa agents'
    # `pd_joint_pos`, where the action is an absolute joint target in radians:
    # there a zero-initialized / off-distribution actor emits the midpoint of
    # every joint range instead of a no-op, and the Gaussian's unit pre-tanh
    # sigma becomes ~1.8 rad of per-step noise. Leave off for controllers that
    # already expose a normalized delta space (`pd_ee_delta_pose`).
    delta_action: bool = False
    # Per-step bound on |a_t - a_{t-1}|, in action units; see DEFAULT_DELTA for
    # how the cabinet demos' fidelity/exploration sweep picked it.
    delta_action_scale: float = DEFAULT_DELTA

    # Eval (reuses train_iql.make_eval_env / evaluate_policy)
    # Episodes between re-sampling the eval scene layout (object placement,
    # geometry). 1 = a fresh layout every episode, i.e. the honest generalization
    # number; 0 = one fixed layout for the whole run. Must be 0 to use
    # `eval_partial_reset` with >1 eval env, since a reconfigure resets all subenvs.
    eval_reconfiguration_freq: Optional[int] = 1
    # Let finished eval sub-envs restart while others keep running. Faster, but
    # it biases the episode mix toward short episodes; off for honest numbers.
    eval_partial_reset: bool = False
    # Eval envs stepped in parallel, and total episodes averaged per eval.
    num_eval_envs: int = 100
    num_eval_episodes: int = 100
    # Time limit for an *eval* episode, in env steps.
    num_eval_steps: int = 300
    # Frame rate of the recorded eval videos.
    video_fps: int = 20

    # Online collection. RLPD/Adroit Binary collects from a single env so the UTD
    # ratio below is exact: with num_envs=1, each env step yields one transition
    # and triggers `utd_ratio` gradient updates. Raising num_envs collects more
    # data per step and thus *lowers* the effective per-transition UTD.
    num_envs: int = 1
    # Training length, counted in env steps (not gradient updates).
    total_env_steps: int = 1_000_000
    # Pure-collection warmup (env transitions) before any gradient update.
    start_steps: int = 5_000
    # During random warmup, sample the impedance arm target locally around the
    # current qpos instead of uniformly over the full Franka joint limits.
    joint_warmup_delta: float = 0.05
    # Online replay capacity in transitions. The offline demos live in a separate
    # buffer and are never evicted.
    buffer_size: int = 1_000_000
    # UTD (update-to-data) ratio: critic gradient updates per collected transition.
    # The Adroit Binary recipe uses 20. Matching the official SAC learner, the
    # critic is updated on every mini-batch of the utd_ratio loop while the actor
    # and temperature are updated once per block (on the last mini-batch).
    utd_ratio: int = 20
    # RLPD-style symmetric sampling: fraction of every batch drawn from offline demos.
    offline_mix_fraction: float = 0.5

    # SAC core
    # Transitions per gradient update, split by `offline_mix_fraction`.
    batch_size: int = 256
    # Standardize observations by the demo mean/std (the stats are frozen at
    # startup, so online data cannot shift them mid-run).
    normalize_observations: bool = True
    # Affine remap of the env reward, `r -> reward_scale * r + reward_bias`,
    # applied identically online and offline. `reward_bias=-1` turns a sparse
    # {0, 1} goal into a per-step cost, which makes the return prefer finishing early.
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Discount. Sets the effective horizon ~1/(1-gamma); the cabinet runs use 0.995.
    gamma: float = 0.99
    # Polyak rate for the target critic: theta_targ <- tau*theta + (1-tau)*theta_targ.
    tau: float = 0.005
    # Width/depth of the actor and of every critic head.
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    # Adam learning rates for the three optimizers.
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    # Global gradient-norm clip applied to each optimizer separately.
    max_grad_norm: float = 10.0

    # Behavior anchor on the offline half of each batch: adds
    # `bc_coef * mean|Q| * ||pi(s) - a_demo||^2` to the actor loss, restricted to
    # demo rows whose episode succeeded. 0 = off (canonical RLPD). 0.4 is the
    # TD3+BC-equivalent weight and is what held CabinetPouringOpenCabinet-Sharpa-v1 at eval
    # success 1.0; 4 is already past the failure boundary (the anchor drowns the
    # `-Q` term and the run never explores). See algo/offline_to_online_rl/rlpd.py.
    bc_coef: float = 0.0

    # RLPD-specific critic. A LayerNorm Q-ensemble of `num_qs` heads; each TD
    # target uses a fresh random subset of `num_min_qs` heads (REDQ-style min).
    # LayerNorm is the key stabilizer that makes the high-UTD ensemble safe.
    num_qs: int = 10
    num_min_qs: int = 2
    critic_layer_norm: bool = True
    # Adroit Binary disables the entropy term in the TD backup.
    backup_entropy: bool = False

    # Entropy temperature. RLPD auto-tunes alpha toward target_entropy; LayerNorm
    # + the critic ensemble keep Q bounded, so the autotune positive-feedback
    # failure that motivated fixing alpha in cql/sac does not arise here.
    autotune_entropy: bool = True
    # Starting alpha. With `autotune_entropy=False` it is the fixed value.
    init_temperature: float = 1.0
    # target_entropy=None -> -action_dim / 2 (RLPD default), computed in the agent.
    target_entropy: Optional[float] = None
    # Task-agnostic alternative to hand-tuning `target_entropy`: set the target
    # to the entropy of a Gaussian whose per-dimension std is this multiple of
    # the demonstrated action std (measured on the successful demos). 2.5 leaves
    # the policy meaningfully wider than the demos without asking for noise the
    # task cannot absorb. Requires demo data; mutually exclusive with an
    # explicit `target_entropy`.
    target_entropy_demo_scale: Optional[float] = None

    # Logging and checkpoints, in env-step units
    # How often scalars are written to TensorBoard/SwanLab.
    log_freq: int = 2_000
    # How often the policy is evaluated, and how often a checkpoint is written.
    eval_freq: int = 100_000
    save_freq: int = 100_000


def _is_lerobot_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta" / "info.json").exists()
        and (path / "data").exists()
    )


def _find_ancestor_lerobot_root(path: Path) -> Optional[Path]:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if _is_lerobot_dataset_root(candidate):
            return candidate.resolve()
    return None


def _resolve_lerobot_dataset_roots(data_path: str) -> list[Path]:
    contains_glob = any(char in data_path for char in "*?[]")
    raw_paths = (
        [Path(path).expanduser().resolve() for path in glob.glob(data_path)]
        if contains_glob
        else [Path(data_path).expanduser()]
    )
    roots: list[Path] = []
    for raw_path in raw_paths:
        if not raw_path.exists():
            continue
        ancestor = _find_ancestor_lerobot_root(raw_path)
        if ancestor is not None:
            roots.append(ancestor)
            continue
        if raw_path.is_dir():
            for info_path in raw_path.rglob("meta/info.json"):
                root = info_path.parent.parent.resolve()
                if _is_lerobot_dataset_root(root):
                    roots.append(root)
    return list(dict.fromkeys(sorted(roots, key=lambda path: path.as_posix())))


def resolve_robot_uids_from_dataset(args: Args) -> Args:
    if args.robot_uids is not None or args.demo_data_path is None:
        return args

    roots = _resolve_lerobot_dataset_roots(args.demo_data_path)
    robot_types = set()
    for root in roots:
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        robot_type = info.get("robot_type")
        if robot_type:
            robot_types.add(str(robot_type))

    if not robot_types:
        return args
    if len(robot_types) > 1:
        raise ValueError(
            "Cannot infer one `robot_uids` from multiple dataset robot types: "
            f"{sorted(robot_types)}. Pass `--robot-uids` explicitly."
        )

    args.robot_uids = next(iter(robot_types))
    return args


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
    if args.joint_warmup_delta <= 0.0:
        raise ValueError(
            f"`joint_warmup_delta` must be positive, got {args.joint_warmup_delta}."
        )
    if args.delta_action and args.delta_action_scale <= 0.0:
        raise ValueError(
            f"`delta_action_scale` must be positive, got {args.delta_action_scale}."
        )
    if args.target_entropy is not None and args.target_entropy_demo_scale is not None:
        raise ValueError(
            "Set either `target_entropy` or `target_entropy_demo_scale`, not both."
        )
    if (
        args.target_entropy_demo_scale is not None
        and args.target_entropy_demo_scale <= 0.0
    ):
        raise ValueError(
            "`target_entropy_demo_scale` must be positive, got "
            f"{args.target_entropy_demo_scale}."
        )
    if args.bc_coef < 0.0:
        raise ValueError(f"`bc_coef` must be non-negative, got {args.bc_coef}.")
    if args.bc_coef > 0.0 and args.offline_mix_fraction <= 0.0:
        raise ValueError(
            "`bc_coef` anchors the actor on offline demo rows, but "
            "`offline_mix_fraction` is 0 so no batch contains any."
        )
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
    if args.num_qs <= 0:
        raise ValueError(f"`num_qs` must be positive, got {args.num_qs}.")
    if not 0 < args.num_min_qs <= args.num_qs:
        raise ValueError(
            f"`num_min_qs` must be in (0, num_qs={args.num_qs}], got {args.num_min_qs}."
        )
    if args.init_temperature <= 0.0:
        raise ValueError(f"`init_temperature` must be positive, got {args.init_temperature}.")
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
    resolve_robot_uids_from_dataset(args)
    return validate_args(args)

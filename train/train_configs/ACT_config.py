# General ACT image config v2: zero expected dimensions mean auto-infer.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import tyro


InferenceMode = Literal["chunk", "replan", "ensemble"]
ResizeMode = Literal["letterbox", "stretch", "center_crop"]
PolicyStateMode = Literal[
    "auto",
    "paths",
    "proprioception",
    "agent",
    "agent_extra",
]
ActionPaddingMode = Literal["repeat_last", "zero"]


@dataclass
class Args:
    # ------------------------------------------------------------------
    # Experiment
    # ------------------------------------------------------------------
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    gpu_id: Optional[int] = None
    track: bool = False
    swanlab_project_name: str = "ManiSkill"
    swanlab_entity: Optional[str] = None
    swanlab_group: str = "ACT-Image-General"
    capture_video: bool = False
    save_model: bool = True
    evaluate: bool = False
    evaluate_only: bool = False
    checkpoint: Optional[str] = None

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env_id: str = tyro.MISSING
    task_module: Optional[str] = None
    robot_uids: Optional[str] = None
    control_mode: Optional[str] = "pd_joint_delta_pos"
    obs_mode: str = "rgb"
    sim_backend: Optional[str] = None
    eval_reconfiguration_freq: Optional[int] = 1
    num_eval_steps: int = 400
    eval_max_steps: Optional[int] = None
    num_eval_episodes: int = 20
    eval_seed: Optional[int] = None
    video_dir: Optional[str] = None
    max_eval_videos: int = 5
    video_fps: int = 20
    eval_sensor_size: Optional[int] = None
    """Override all evaluation policy cameras to a square size."""
    camera_preset: Optional[str] = None
    """Install the fixed image-policy cameras of this preset (`CAMERA_PRESETS` in utils/pixel_cameras.py)
    as the environment's sensors, at image_size pixels; the released ACT/DP checkpoints need
    `cabinet_fixed`. Takes precedence over eval_sensor_size."""
    image_size: int = 256
    """Side of the square images rendered by the camera_preset sensors."""
    eval_initial_state_dataset: Optional[str] = None
    """Optional RoboDex dataset whose recorded episode-start raw states seed eval."""
    eval_initial_episode_ids: Tuple[int, ...] = ()
    """Recorded episode IDs used in order; empty selects 0..num_eval_episodes-1."""
    eval_initial_frame_indices: Tuple[int, ...] = ()
    """Raw-state frame per recorded episode; empty uses frame zero."""
    env_kwarg: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Online policy-state construction
    # ------------------------------------------------------------------
    policy_state_mode: PolicyStateMode = "auto"
    """
    auto:
      Try the configured paths first, then common robot-state layouts, and
      choose the first layout whose dimension matches the trained model.

    paths:
      Concatenate fields listed in online_state_paths. Dot notation is
      supported, for example agent.qpos extra.tcp_pose.

    proprioception:
      Legacy convenience layout:
      agent.qpos + agent.qvel + extra.tcp_pose.

    agent:
      Flatten the complete observation["agent"] mapping.

    agent_extra:
      Flatten the top-level mappings selected by env_state_keys. If empty,
      flatten agent + extra.
    """

    online_state_paths: Tuple[str, ...] = ()
    """Explicit dot paths used by policy_state_mode=paths/auto."""

    state_adapter: Optional[str] = None
    """
    Optional custom function in MODULE:FUNCTION form. It may accept
    (obs), (obs, env), or keyword arguments obs=..., env=..., args=..., and
    must return a one-dimensional NumPy-compatible policy-state vector.
    This is the escape hatch for tasks/robot-specific state transforms.
    """

    env_state_keys: Tuple[str, ...] = ()
    camera_map: Tuple[str, ...] = ()
    """Optional DATASET_CAMERA=ENV_SENSOR mappings."""
    camera_history_offsets: Tuple[str, ...] = ()
    """Optional DATASET_CAMERA=PAST_STEPS mappings for visual history tokens."""
    camera_auto_match: bool = True

    # ------------------------------------------------------------------
    # Preprocessed dataset
    # ------------------------------------------------------------------
    demo_data_path: str = "datasets/act_image/demo_128"
    dataset_path: Optional[str] = None
    camera_names: Tuple[str, ...] = ()
    """Empty uses all cameras listed in dataset_info.json."""
    filter_success: bool = True
    include_episode_ids: Tuple[int, ...] = ()
    exclude_episode_ids: Tuple[int, ...] = ()
    validation_ratio: float = 0.0

    # Offline policy-state selection.
    # "all" uses every state dimension. Other examples: "0:25", "0:7,9".
    offline_state_slice: str = "all"

    # Set either value to 0 to infer it from the preprocessed dataset.
    expected_raw_state_dim: int = 0
    expected_policy_state_dim: int = 0

    # Floors prevent nearly constant state/action dimensions from exploding
    # after normalization.
    state_std_floor: float = 1e-3
    action_std_floor: float = 1e-2

    # Optional policy-only residual parameterization for an absolute
    # ``pd_joint_pos`` environment. Demonstrations remain unchanged on disk.
    joint_pos_residual: bool = False
    joint_pos_residual_source_dataset: Optional[str] = None
    """LeRobot source providing observation.state/qpos for image-only labels."""
    joint_pos_residual_source_obs_key: str = "observation.state"
    joint_pos_residual_action_start: int = 0
    joint_pos_residual_qpos_start: int = 0
    joint_pos_residual_dim: int = 7
    joint_pos_residual_scale: float = 0.4

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    resize_mode: ResizeMode = "letterbox"
    letterbox_value: int = 0
    random_shift_pad: int = 0

    # ------------------------------------------------------------------
    # Dataset sampling
    # ------------------------------------------------------------------
    action_padding_mode: ActionPaddingMode = "repeat_last"
    min_chunk_valid_steps: int = 5
    """
    Do not use the final anchors whose future action target contains fewer than
    this many real steps. This reduces over-sampling of very short terminal
    chunks when chunk_size is large.
    """

    # Set static_sample_weight below 1 to down-weight chunks containing no
    # meaningful motion. 1.0 disables weighted sampling.
    static_sample_weight: float = 1.0
    static_action_threshold: float = 1e-3
    static_action_dims: Tuple[int, ...] = ()
    """
    Action dimensions used to decide whether a chunk is static.
    Empty means all action dimensions. For a 7-D delta-pose + gripper action,
    use 0 1 2 3 4 5 to ignore the gripper command.
    """
    sample_weight_file: Optional[str] = None
    """Optional per-frame NumPy weights, relative to demo_data_path."""

    # ------------------------------------------------------------------
    # ACT
    # ------------------------------------------------------------------
    steps: int = 20000
    batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 10.0
    chunk_size: int = 30
    kl_weight: float = 0.001
    prior_l1_weight: float = 0.1
    """
    loss = posterior_l1 + prior_l1_weight * prior_l1 + kl_weight * KL.
    prior_l1 trains the z=0 inference path directly.
    """
    action_loss_groups: Tuple[str, ...] = ()
    """Optional action slices such as 0:7 7:29, averaged as separate groups."""
    action_loss_group_weights: Tuple[float, ...] = ()
    """Optional positive weights matching action_loss_groups."""

    hidden_dim: int = 256
    latent_dim: int = 32
    nheads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 4
    fusion_layers: int = 2
    dropout: float = 0.1
    image_encoder_backbone: str = "conv"
    """Shared visual backbone: ``conv`` or torchvision ``resnet18``."""
    image_encoder_pretrained: bool = False
    """Initialize ResNet-18 from ImageNet-1K weights before training."""
    image_encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)

    # ------------------------------------------------------------------
    # Evaluation / logging
    # ------------------------------------------------------------------
    inference_mode: InferenceMode = "ensemble"
    action_chunk_size: int = 0
    """
    Number of predicted actions to execute before replanning in ``chunk``
    mode. Zero executes the model's full trained chunk. This is an inference-
    only horizon and does not change the checkpoint's architectural
    ``chunk_size``.
    """
    ensemble_decay: float = 0.05
    ensemble_latest_action_dims: Tuple[int, ...] = (-1,)
    eval_every: int = 2000
    save_every: int = 2000
    log_freq: int = 100
    output_dir: str = "runs"
    check_env_before_training: bool = True

    success_keys: Tuple[str, ...] = ("success", "task_success")
    terminate_on_success: bool = False
    stage_keys: Tuple[str, ...] = ()
    metric_keys: Tuple[str, ...] = ()


def validate_args(args: Args) -> Args:
    if args.evaluate_only:
        args.evaluate = True
    if args.evaluate and args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluation")
    if args.dataset_path is not None:
        args.demo_data_path = args.dataset_path
    if args.eval_max_steps is not None:
        args.num_eval_steps = args.eval_max_steps

    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError("gpu_id must be non-negative")
    if args.eval_sensor_size is not None and args.eval_sensor_size <= 0:
        raise ValueError("eval_sensor_size must be positive")
    if args.camera_preset is not None:
        from utils.pixel_cameras import CAMERA_PRESETS

        if args.camera_preset not in CAMERA_PRESETS:
            raise ValueError(
                f"Unknown camera_preset {args.camera_preset!r}; have {sorted(CAMERA_PRESETS)}"
            )
        if args.image_size <= 0:
            raise ValueError("image_size must be positive")
    if (
        args.eval_initial_episode_ids
        and len(args.eval_initial_episode_ids) < args.num_eval_episodes
    ):
        raise ValueError(
            "eval_initial_episode_ids must contain at least num_eval_episodes values"
        )
    if (
        args.eval_initial_frame_indices
        and len(args.eval_initial_frame_indices) < args.num_eval_episodes
    ):
        raise ValueError(
            "eval_initial_frame_indices must contain at least "
            "num_eval_episodes values"
        )
    if any(value < 0 for value in args.eval_initial_frame_indices):
        raise ValueError("eval_initial_frame_indices must be non-negative")
    if args.steps <= 0 and not args.evaluate:
        raise ValueError("steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if args.action_chunk_size < 0:
        raise ValueError("action_chunk_size must be non-negative")
    if args.action_loss_group_weights and (
        len(args.action_loss_group_weights) != len(args.action_loss_groups)
    ):
        raise ValueError(
            "action_loss_group_weights must match action_loss_groups length"
        )
    if any(value <= 0 for value in args.action_loss_group_weights):
        raise ValueError("action_loss_group_weights must be positive")
    if not 0.0 <= args.validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    if args.hidden_dim % args.nheads != 0:
        raise ValueError("hidden_dim must be divisible by nheads")
    if args.hidden_dim % 4 != 0:
        raise ValueError(
            "hidden_dim must be divisible by 4 for 2-D position encoding"
        )
    if args.image_encoder_backbone not in {"conv", "resnet18"}:
        raise ValueError(
            "image_encoder_backbone must be one of: conv, resnet18"
        )
    if args.image_encoder_pretrained and args.image_encoder_backbone != "resnet18":
        raise ValueError(
            "image_encoder_pretrained is only supported with resnet18"
        )
    if args.prior_l1_weight < 0:
        raise ValueError("prior_l1_weight must be non-negative")
    if args.kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    if len(args.image_mean) != 3 or len(args.image_std) != 3:
        raise ValueError("image_mean/image_std must contain 3 values")
    if any(value <= 0 for value in args.image_std):
        raise ValueError("image_std values must be positive")
    if args.random_shift_pad < 0:
        raise ValueError("random_shift_pad must be non-negative")
    if args.expected_raw_state_dim < 0 or args.expected_policy_state_dim < 0:
        raise ValueError("expected state dimensions must be non-negative")
    if args.state_std_floor <= 0 or args.action_std_floor <= 0:
        raise ValueError("normalization std floors must be positive")
    if args.joint_pos_residual:
        if not args.joint_pos_residual_source_dataset:
            raise ValueError(
                "joint_pos_residual_source_dataset is required when "
                "joint_pos_residual is enabled"
            )
        if args.joint_pos_residual_action_start < 0:
            raise ValueError("joint_pos_residual_action_start must be non-negative")
        if args.joint_pos_residual_qpos_start < 0:
            raise ValueError("joint_pos_residual_qpos_start must be non-negative")
        if args.joint_pos_residual_dim <= 0:
            raise ValueError("joint_pos_residual_dim must be positive")
        if args.joint_pos_residual_scale <= 0:
            raise ValueError("joint_pos_residual_scale must be positive")
    if not 0.0 <= args.static_sample_weight <= 1.0:
        raise ValueError("static_sample_weight must be in [0, 1]")
    if args.static_action_threshold < 0:
        raise ValueError("static_action_threshold must be non-negative")
    if args.min_chunk_valid_steps <= 0:
        raise ValueError("min_chunk_valid_steps must be positive")
    if args.min_chunk_valid_steps > args.chunk_size:
        raise ValueError(
            "min_chunk_valid_steps cannot exceed chunk_size"
        )

    if args.policy_state_mode == "paths" and not args.online_state_paths:
        raise ValueError(
            "policy_state_mode=paths requires --online-state-paths"
        )

    include = set(args.include_episode_ids)
    exclude = set(args.exclude_episode_ids)
    overlap = include & exclude
    if overlap:
        raise ValueError(
            f"Episode IDs appear in both include and exclude: {sorted(overlap)}"
        )

    for item in args.camera_map:
        if "=" not in item:
            raise ValueError(
                f"camera_map must use DATASET_CAMERA=ENV_SENSOR, got {item!r}"
            )
    if not args.success_keys:
        raise ValueError("success_keys must contain at least one key")
    return args


def parse_args(overrides: Optional[Dict[str, Any]] = None) -> Args:
    args = tyro.cli(Args)
    if overrides:
        for key, value in overrides.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown argument override: {key}")
            setattr(args, key, value)
    return validate_args(args)

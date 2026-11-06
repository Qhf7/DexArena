from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tyro

from utils.paths import resolve_training_paths


@dataclass
class Args:
    # Experiment metadata
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    gpu_id: Optional[int] = None
    """CUDA device index to use for both PyTorch and ManiSkill GPU backends; defaults to the current CUDA default device"""
    track: bool = True
    """if toggled, this experiment will be tracked with SwanLab"""
    swanlab_project_name: str = "ManiSkill"
    """the swanlab project name"""
    swanlab_entity: Optional[str] = None
    """the entity (team) of swanlab's project"""
    capture_video: bool = True
    """whether to capture evaluation videos"""
    save_model: bool = True
    """whether to save checkpoints into `runs/{run_name}`"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given checkpoint"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Environment / dataset
    env_id: str = "CabinetPouringOpenCabinet-Sharpa-v1"
    """the id of the environment"""
    robot_uids: Optional[str] = None
    """robot uid override passed to the environment"""
    control_mode: Optional[str] = "pd_joint_pos"
    """the control mode to use for the environment"""
    reward_mode: Optional[str] = None
    """environment reward mode for evaluation (e.g. `sparse`); `None` uses the env default. BC training ignores reward."""
    sim_backend: Optional[str] = None
    """simulation backend override; defaults to `physx_cuda` when CUDA is available, else `physx_cpu`"""
    reconfiguration_freq: Optional[int] = None
    """training-time env reconfiguration frequency, used only for dimensionality checks"""
    eval_reconfiguration_freq: Optional[int] = 1
    """evaluation env reconfiguration frequency"""
    eval_partial_reset: bool = False
    """whether to let evaluation envs partially reset on termination"""
    num_eval_envs: int = 10
    """the number of parallel evaluation environments"""
    num_eval_steps: int = 300
    """max evaluation rollout length in environment steps"""
    num_eval_episodes: int = 10
    """number of evaluation episodes to run"""
    demo_data_path: str = "teleop_lerobot"
    """offline demonstration dataset path (LeRobot root, parent directory, glob, or legacy .h5 path)"""
    validation_ratio: float = 0.1
    """holdout fraction for episode-level validation split"""

    # Diffusion sequence settings
    n_obs_steps: int = 2
    """number of observation steps in the conditioning window"""
    horizon: int = 16
    """number of actions predicted by the diffusion model"""
    n_action_steps: int = 8
    """number of actions executed before resampling the next action chunk"""
    use_ema: bool = True
    """maintain an exponential moving average of the diffusion model weights (standard
    diffusion-policy practice); sampling, evaluation and checkpoints use the EMA weights"""
    ema_decay_max: float = 0.9999
    """upper bound of the EMA decay"""
    ema_inv_gamma: float = 1.0
    """EMA decay warmup schedule: decay = 1 - (1 + step/inv_gamma)^-power, capped at ema_decay_max"""
    ema_power: float = 0.75
    """power of the EMA decay warmup schedule"""
    drop_n_last_frames: Optional[int] = None
    """drop this many tail timesteps from each episode when building training windows; defaults to LeRobot's horizon-based rule"""
    obs_normalization_mode: str = "min_max"
    """observation normalization mode: `mean_std` or `min_max`; LeRobot diffusion defaults to `min_max` for state"""
    action_normalization_mode: str = "min_max"
    """action normalization mode: `mean_std` or `min_max`; LeRobot diffusion defaults to `min_max` for action"""
    do_mask_loss_for_padding: bool = False
    """mask padded action targets in the training loss; LeRobot diffusion defaults to `False`"""

    # Optimization
    num_epochs: int = 200
    """number of training epochs"""
    batch_size: int = 256
    """offline training batch size"""
    num_workers: int = 4
    """dataloader workers"""
    learning_rate: float = 1e-4
    """optimizer learning rate"""
    betas: Tuple[float, float] = (0.95, 0.999)
    """AdamW beta coefficients"""
    weight_decay: float = 1e-6
    """AdamW weight decay"""
    max_grad_norm: float = 10.0
    """gradient clipping norm"""
    lr_scheduler: str = "cosine"
    """learning-rate schedule: `constant` or `cosine`"""
    lr_warmup_steps: int = 500
    """number of warmup optimizer steps"""

    # Backbone selection (white-box diffusion_2 stack)
    backbone: str = "mlp"
    """noise-prediction backbone: `mlp` (DiffusionMLP) or `unet` (Conv1d U-Net). Must match the
    DPPO fine-tuning run's `--backbone` for the checkpoint to load."""

    # MLP backbone (used when backbone == 'mlp')
    time_dim: int = 16
    """sinusoidal timestep embedding size for the MLP backbone"""
    mlp_dims: Tuple[int, ...] = (2560,) * 11
    """hidden widths of the DiffusionMLP noise network; default is the widest ablation
    network (2560 x 11 residual, ~68.6M, best MLP on ShadowHandBallGrasp). Width matters
    more than depth; even at U-Net param scale the MLP reaches ~0.19 vs U-Net's ~0.31."""
    cond_mlp_dims: Optional[Tuple[int, ...]] = None
    """optional conditioning MLP widths; `None` concatenates the flat state directly"""
    residual_style: bool = True
    """use ResidualMLP blocks instead of a plain MLP (requires odd len(mlp_dims))"""
    use_layernorm: bool = True
    """apply LayerNorm inside the MLP backbone / critic"""
    critic_mlp_dims: Tuple[int, ...] = (256, 256)
    """critic hidden widths (unused by BC; kept so the checkpoint's args inform DPPO)"""

    # White-box DDPM (hand-rolled cosine schedule; replaces the diffusers scheduler for DPPO compat)
    denoising_steps: int = 100
    """number of DDPM denoising steps in the white-box core"""
    predict_epsilon: bool = True
    """if True the network predicts noise (epsilon); else it predicts the clean sample"""
    final_action_clip_value: Optional[float] = 1.0
    """clamp the final denoised action chunk to +/- this value; `None` disables"""

    # Model
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    """feature dimensions in the temporal U-Net down path"""
    kernel_size: int = 5
    """temporal convolution kernel size"""
    n_groups: int = 8
    """group norm group count"""
    diffusion_step_embed_dim: int = 128
    """diffusion timestep embedding size"""
    use_film_scale_modulation: bool = True
    """whether to use FiLM scale modulation in addition to bias modulation"""
    noise_scheduler_type: str = "DDPM"
    """noise scheduler type: `DDPM` or `DDIM`"""
    num_train_timesteps: int = 100
    """number of forward diffusion timesteps"""
    beta_schedule: str = "squaredcos_cap_v2"
    """beta schedule passed to diffusers"""
    beta_start: float = 0.0001
    """starting beta value"""
    beta_end: float = 0.02
    """ending beta value"""
    prediction_type: str = "epsilon"
    """diffusion prediction target: `epsilon` or `sample`"""
    clip_sample: bool = True
    """whether to clip the normalized reverse-diffusion samples"""
    clip_sample_range: float = 1.0
    """normalized clipping range used by the scheduler"""
    num_inference_steps: Optional[int] = None
    """number of reverse-diffusion steps at inference time; defaults to `num_train_timesteps`"""

    # Logging / checkpointing
    log_freq: int = 10
    """log training loss every N optimizer steps"""
    eval_freq: int = 25
    """run validation + environment evaluation every N epochs"""
    save_freq: int = 25
    """save intermediate checkpoints every N epochs"""


def validate_args(args: Args) -> Args:
    if args.checkpoint is None and args.evaluate:
        raise ValueError("`--checkpoint` is required when `--evaluate` is set.")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    if args.n_obs_steps <= 0:
        raise ValueError(f"`n_obs_steps` must be positive, got {args.n_obs_steps}.")
    if args.horizon <= 0:
        raise ValueError(f"`horizon` must be positive, got {args.horizon}.")
    if args.n_action_steps <= 0 or args.n_action_steps > args.horizon:
        raise ValueError(
            f"`n_action_steps` must be in [1, horizon], got {args.n_action_steps} with horizon={args.horizon}."
        )
    max_executable_steps = args.horizon - args.n_obs_steps + 1
    if args.n_action_steps > max_executable_steps:
        raise ValueError(
            "`n_action_steps` must satisfy LeRobot diffusion's rollout constraint "
            f"`n_action_steps <= horizon - n_obs_steps + 1`. Got n_action_steps={args.n_action_steps}, "
            f"horizon={args.horizon}, n_obs_steps={args.n_obs_steps}."
        )
    if args.validation_ratio < 0.0 or args.validation_ratio >= 1.0:
        raise ValueError(f"`validation_ratio` must be in [0, 1), got {args.validation_ratio}.")
    if args.obs_normalization_mode not in {"mean_std", "min_max"}:
        raise ValueError(
            f"`obs_normalization_mode` must be `mean_std` or `min_max`, got {args.obs_normalization_mode}."
        )
    if args.action_normalization_mode not in {"mean_std", "min_max"}:
        raise ValueError(
            f"`action_normalization_mode` must be `mean_std` or `min_max`, got {args.action_normalization_mode}."
        )
    if args.lr_scheduler not in {"constant", "cosine"}:
        raise ValueError(f"`lr_scheduler` must be `constant` or `cosine`, got {args.lr_scheduler}.")
    if args.backbone not in {"mlp", "unet"}:
        raise ValueError(f"`backbone` must be `mlp` or `unet`, got {args.backbone}.")
    if args.backbone not in {"mlp", "unet"}:
        raise ValueError(f"`backbone` must be `mlp` or `unet`, got {args.backbone}.")
    if args.noise_scheduler_type not in {"DDPM", "DDIM"}:
        raise ValueError(
            f"`noise_scheduler_type` must be `DDPM` or `DDIM`, got {args.noise_scheduler_type}."
        )
    if args.prediction_type not in {"epsilon", "sample"}:
        raise ValueError(f"`prediction_type` must be `epsilon` or `sample`, got {args.prediction_type}.")
    if args.backbone == "unet":
        downsampling_factor = 2 ** len(args.down_dims)
        if args.horizon % downsampling_factor != 0:
            raise ValueError(
                "The diffusion `horizon` must be divisible by the temporal U-Net downsampling factor. "
                f"Got horizon={args.horizon} and down_dims={args.down_dims}."
            )
    if args.backbone == "unet":
        downsampling_factor = 2 ** len(args.down_dims)
        if args.horizon % downsampling_factor != 0:
            raise ValueError(
                "The diffusion `horizon` must be divisible by the temporal U-Net downsampling factor. "
                f"Got horizon={args.horizon} and down_dims={args.down_dims}."
            )
    if args.drop_n_last_frames is None:
        args.drop_n_last_frames = args.horizon - args.n_action_steps - args.n_obs_steps + 1
    if args.drop_n_last_frames < 0:
        raise ValueError(f"`drop_n_last_frames` must be non-negative, got {args.drop_n_last_frames}.")
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

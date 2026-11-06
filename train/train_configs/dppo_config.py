"""Configuration for the DPPO module (BC pretrain + PPO fine-tune).

Two dataclasses mirror the repo's existing config style (``ppo_config.py`` /
``diffusion_config.py``): :class:`PretrainArgs` for ``train/pretrain_dppo.py`` and
:class:`FinetuneArgs` for ``train/train_dppo.py``. They share a common set of
model/diffusion/normalizer fields so a BC checkpoint loads cleanly into fine-tuning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import tyro

from utils.paths import resolve_training_paths


@dataclass
class _Common:
    # Experiment metadata
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    cuda: bool = True
    gpu_id: Optional[int] = None
    """CUDA device index for both PyTorch and ManiSkill GPU backends"""
    track: bool = True
    """log to swanlab / tensorboard"""
    swanlab_project_name: str = "ManiSkill"
    swanlab_entity: Optional[str] = None
    save_model: bool = True

    # Environment identity (shared so obs/action dims are derived consistently)
    env_id: str = tyro.MISSING
    control_mode: Optional[str] = "pd_ee_delta_pose"
    reward_mode: Optional[str] = "sparse"
    """environment reward mode (e.g. `sparse`); `None` uses the env default"""
    sim_backend: Optional[str] = None

    # --- Diffusion horizons (To / Ta / prediction horizon) ---
    n_obs_steps: int = 2
    """To: observation history length used for conditioning"""
    horizon: int = 16
    """Ta-pred: number of actions the diffusion model predicts per call"""
    n_action_steps: int = 8
    """number of predicted actions actually executed in the env per decision"""

    # --- DDPM / DDIM ---
    denoising_steps: int = 20
    """K: number of denoising steps in the diffusion chain"""
    ft_denoising_steps: int = 5
    """number of (last) denoising steps fine-tuned / treated as the MDP"""
    predict_epsilon: bool = True
    use_ddim: bool = False
    ddim_steps: Optional[int] = None
    final_action_clip_value: Optional[float] = 1.0
    """clamp the final sampled (normalized) action to [-v, v]; None disables"""

    # --- Network ---
    backbone: str = "mlp"
    """noise-prediction backbone: 'mlp' (DiffusionMLP) or 'unet' (Conv1d U-Net)"""
    mlp_dims: Tuple[int, ...] = (2560,) * 11
    """match the BC backbone (2560 x 11 residual) so an MLP base policy loads cleanly"""
    cond_mlp_dims: Optional[Tuple[int, ...]] = None
    critic_mlp_dims: Tuple[int, ...] = (256, 256, 256)
    time_dim: int = 16
    residual_style: bool = True
    use_layernorm: bool = True

    # --- U-Net backbone (used only when backbone == 'unet') ---
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    """feature dimensions in the temporal U-Net down path"""
    kernel_size: int = 5
    """temporal convolution kernel size"""
    n_groups: int = 8
    """group-norm group count"""
    diffusion_step_embed_dim: int = 128
    """U-Net diffusion timestep embedding size"""
    use_film_scale_modulation: bool = True
    """FiLM scale modulation (in addition to bias) in the U-Net conditional blocks"""

    # --- Normalization ---
    obs_normalization_mode: str = "min_max"
    """either 'min_max' or 'mean_std'"""
    action_normalization_mode: str = "min_max"


@dataclass
class PretrainArgs(_Common):
    # Data
    demo_data_path: str = "teleop_lerobot"
    """LeRobot demo dataset root / parent / glob"""
    validation_ratio: float = 0.1
    drop_n_last_frames: Optional[int] = None

    # Optimization
    num_epochs: int = 200
    batch_size: int = 256
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 10.0

    # Logging / checkpointing
    log_freq: int = 10
    save_freq: int = 25
    checkpoint: Optional[str] = None
    """resume-from checkpoint"""


@dataclass
class FinetuneArgs(_Common):
    # Pretrained policy (optional: omit to start from random init + identity normalizer)
    base_policy_path: Optional[str] = None
    use_ema_base: bool = True
    """load the BC checkpoint's EMA weights (`ema_model`) as the DPPO base instead of the
    raw weights; falls back to raw for old checkpoints without EMA"""

    # Parallel envs / rollout
    num_envs: int = 64
    num_eval_envs: int = 32
    num_eval_episodes: int = 16
    total_timesteps: int = 5_000_000
    num_steps: int = 40
    """decision steps per env per rollout iteration (each executes n_action_steps sim steps)"""
    num_eval_steps: int = 400
    max_episode_steps: Optional[int] = 400
    """env episode truncation; defaults to num_steps * n_action_steps when unset"""
    partial_reset: bool = False
    eval_partial_reset: bool = False
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    capture_video: bool = True
    save_train_video_freq: Optional[int] = None

    # GAE / PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 10
    norm_adv: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0
    target_kl: Optional[float] = None
    reward_scale: float = 0.1
    """multiplies env reward to keep the value function in range; tune so the scaled
    per-episode return lands in ~0-10 ."""
    use_bc_loss: bool = True
    """enable the BC anchor (forward-KL toward the frozen base policy)"""
    bc_coef: float = 0.03
    """weight of the BC-anchor loss; only used when use_bc_loss=True"""

    # PPO-over-denoising specifics
    gamma_denoising: float = 0.99
    clip_ploss_coef: float = 0.03
    clip_ploss_coef_base: float = 1e-3
    clip_ploss_coef_rate: float = 3.0
    clip_vloss_coef: Optional[float] = 2.0
    min_sampling_denoising_std: float = 0.1
    min_logprob_denoising_std: float = 0.1

    # Optimizers
    actor_lr: float = 1e-5
    actor_weight_decay: float = 0.0
    critic_lr: float = 1e-3
    critic_weight_decay: float = 0.0

    # Eta (DDIM only)
    learn_eta: bool = False
    eta_base: float = 1.0

    # Eval / checkpointing
    eval_freq: int = 10
    save_freq: int = 25
    checkpoint: Optional[str] = None
    evaluate: bool = False

    # filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def _apply_overrides(args, overrides: Optional[Dict[str, Any]]):
    if overrides:
        for key, value in overrides.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown arg override: {key}")
            setattr(args, key, value)
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    resolve_training_paths(args)
    return args


def parse_pretrain_args(overrides: Optional[Dict[str, Any]] = None) -> PretrainArgs:
    return _apply_overrides(tyro.cli(PretrainArgs), overrides)


def parse_finetune_args(overrides: Optional[Dict[str, Any]] = None) -> FinetuneArgs:
    return _apply_overrides(tyro.cli(FinetuneArgs), overrides)

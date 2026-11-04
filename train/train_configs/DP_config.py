"""Image DP configuration; shared image/environment options follow ACT's adapter."""
from dataclasses import dataclass
from typing import Tuple
import tyro
from train.train_configs.ACT_config import Args as ImageArgs, validate_args as validate_image_args


@dataclass
class Args(ImageArgs):
    chunk_size: int = 16
    action_chunk_size: int = 4
    inference_mode: str = "chunk"
    batch_size: int = 16
    num_workers: int = 4
    validation_ratio: float = 0.1
    min_chunk_valid_steps: int = 1
    image_encoder_backbone: str = "resnet18"
    image_encoder_pretrained: bool = True
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    control_mode: str = "pd_joint_pos"
    num_eval_episodes: int = 0
    check_env_before_training: bool = False
    save_every: int = 1000
    eval_every: int = 1000
    swanlab_group: str = "Diffusion-Image"
    denoising_steps: int = 100
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    visual_feature_dim: int = 128
    ema_power: float = 0.75
    ema_decay_max: float = 0.9999
    lr_warmup_steps: int = 500
    max_val_batches: int = 32
    torch_threads: int = 4


def parse_args():
    args = validate_image_args(tyro.cli(Args))
    if args.denoising_steps < 2 or not args.down_dims or args.n_groups <= 0:
        raise ValueError("Invalid diffusion configuration")
    if args.chunk_size % (2 ** (len(args.down_dims) - 1)):
        raise ValueError("chunk_size must be divisible by the U-Net downsampling factor")
    if not 1 <= args.action_chunk_size <= args.chunk_size:
        raise ValueError("action_chunk_size must be in [1, chunk_size]")
    if args.inference_mode != "chunk":
        raise ValueError("Image DP uses chunk inference")
    if any(d % args.n_groups for d in args.down_dims):
        raise ValueError("down_dims must be divisible by n_groups")
    if args.image_encoder_backbone != "resnet18" or args.joint_pos_residual:
        raise ValueError("Image DP supports ResNet18 and absolute actions")
    if args.track:
        raise ValueError("Use TensorBoard/metrics.jsonl; SwanLab is not wired into this entry point")
    if args.static_sample_weight != 1 or args.sample_weight_file or args.action_loss_groups:
        raise ValueError("Weighted sampling/action groups are not supported by this DP entry point")
    if min(args.save_every, args.eval_every, args.log_freq, args.max_val_batches, args.torch_threads) <= 0:
        raise ValueError("Logging/checkpoint/validation intervals and torch_threads must be positive")
    if not 0 < args.ema_decay_max < 1 or args.ema_power <= 0:
        raise ValueError("Invalid EMA configuration")
    if args.evaluate and args.num_eval_episodes <= 0:
        raise ValueError("Evaluation requires --num-eval-episodes greater than zero")
    if args.learning_rate <= 0 or args.lr_warmup_steps < 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer configuration")
    return args

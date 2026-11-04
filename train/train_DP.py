"""Train/evaluate image diffusion using existing prepared multicamera NumPy datasets."""
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('ROBODEX_SKIP_AUTO_TASK_IMPORTS', '1')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights, resnet18
from algo.diffusion import DiffusionModel, DiffusionUNet
from train import train_ACT as shared
from train.train_configs.DP_config import parse_args


def replace_batch_norm(module):
    """Replace BatchNorm with GroupNorm for stable small-batch training and EMA."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            norm = nn.GroupNorm(32, child.num_features)
            with torch.no_grad():
                norm.weight.copy_(child.weight)
                norm.bias.copy_(child.bias)
            setattr(module, name, norm)
        else:
            replace_batch_norm(child)


class DiffusionImagePolicy(nn.Module):
    """RGB-history and current-state conditioned diffusion policy."""

    def __init__(self, args, state_dim, action_dim, cameras, device, pretrained=None):
        super().__init__()
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.chunk_size = args.chunk_size
        self.camera_names = tuple(camera["name"] for camera in cameras)
        self.image_sizes = {
            camera["name"]: (camera["height"], camera["width"])
            for camera in cameras
        }
        use_pretrained = (
            args.image_encoder_pretrained if pretrained is None else pretrained
        )
        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT if use_pretrained else None
        )
        replace_batch_norm(backbone)
        self.encoder = nn.Sequential(
            *list(backbone.children())[:-2],
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(512 * 4, args.visual_feature_dim),
            nn.LayerNorm(args.visual_feature_dim),
        )
        cond_dim = state_dim + len(cameras) * args.visual_feature_dim
        unet = DiffusionUNet(
            action_dim=action_dim, global_cond_dim=cond_dim, args=args
        )
        self.diffusion = DiffusionModel(
            network=unet,
            horizon_steps=args.chunk_size,
            obs_dim=cond_dim,
            action_dim=action_dim,
            device=str(device),
            denoising_steps=args.denoising_steps,
            predict_epsilon=True,
            denoised_clip_value=1.0,
            final_action_clip_value=1.0,
        )
        self.to(device)

    def encode(self, state, images):
        features = [self.encoder(images[name]) for name in self.camera_names]
        return {"state": torch.cat([state, *features], dim=-1)}

    def loss(self, state, images, actions, mask):
        cond = self.encode(state, images)
        noise = torch.randn_like(actions)
        timestep = torch.randint(
            self.diffusion.denoising_steps,
            (len(actions),),
            device=actions.device,
        )
        noisy = self.diffusion.q_sample(actions, timestep, noise)
        predicted_noise = self.diffusion.network(noisy, timestep, cond)
        squared = (predicted_noise - noise).square().mean(-1)
        return (squared * mask).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def act_chunk(self, state, images):
        return self.diffusion(
            self.encode(state, images), deterministic=True
        ).trajectories


def normalization(info, episodes, args):
    stats = shared.compute_stats(info, episodes, args.state_std_floor, args.action_std_floor)
    action = np.load(info.root / 'action.npy', mmap_mode='r')
    lo = np.minimum.reduce([action[e.start:e.end].min(0) for e in episodes])
    hi = np.maximum.reduce([action[e.start:e.end].max(0) for e in episodes])
    # Stats adapter expresses min/max action normalization as (x - midpoint)/half_range.
    stats.action_mean = ((lo + hi) / 2).astype(np.float32)
    stats.action_std = np.maximum((hi - lo) / 2, args.action_std_floor).astype(np.float32)
    return stats


def make_dataset(info, episodes, stats, args, training):
    return shared.ACTImageDataset(info, episodes, stats, args.chunk_size, args.image_mean,
                                  args.image_std, training, args.random_shift_pad,
                                  'repeat_last', args.min_chunk_valid_steps)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get('format') != 'robodex_diffusion_image_v1':
        raise ValueError('Not an image diffusion checkpoint')
    args = SimpleNamespace(**ckpt['args'])
    model = DiffusionImagePolicy(args, ckpt['state_dim'], ckpt['action_dim'], ckpt['cameras'], device, pretrained=False)
    model.load_state_dict(ckpt['ema_model'])
    model.eval()
    return model, shared.Stats.from_dict(ckpt['stats']), ckpt


def save_checkpoint(path, model, ema, optimizer, args, stats, cameras, step, split):
    payload = dict(format='robodex_diffusion_image_v1', args=vars(args), stats=stats.to_dict(),
                   cameras=cameras, state_dim=model.state_dim, action_dim=model.action_dim,
                   step=step, model=model.state_dict(), ema_model=ema.state_dict(),
                   optimizer=optimizer.state_dict(), split=split,
                   rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                            python=random.getstate(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    print(f'[SAVE] {path}', flush=True)


@torch.no_grad()
def ema_update(ema, model, step, args):
    decay = min(args.ema_decay_max, max(0., 1 - (1 + step) ** (-args.ema_power)))
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    shared.set_random_seed(args)
    device = shared.resolve_device(args)
    if args.evaluate:
        model, stats, ckpt = load_checkpoint(args.checkpoint, device)
        # Restore the observation contract; inference horizon and evaluation settings remain CLI-controlled.
        for key in ('camera_map', 'camera_history_offsets', 'image_mean', 'image_std', 'resize_mode',
                    'letterbox_value', 'policy_state_mode', 'online_state_paths', 'state_adapter', 'env_state_keys'):
            setattr(args, key, ckpt['args'][key])
        shared.check_environment(model, stats, args, device)
        print(json.dumps(shared.evaluate(model, stats, args, device, Path(args.checkpoint).resolve().parent, ckpt['step']), indent=2))
        return

    run_dir = Path(args.output_dir) / (args.exp_name or f'diffusion_image_{args.seed}_{int(time.time())}')
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'resolved_args.json').exists():
        raise FileExistsError(f'Use a new experiment directory: {run_dir}')
    (run_dir / 'resolved_args.json').write_text(json.dumps(vars(args), indent=2))
    info = shared.load_dataset_info(args.demo_data_path, args.camera_names, args.offline_state_slice,
                                    args.expected_raw_state_dim, args.expected_policy_state_dim)
    episodes = shared.load_episodes(info, args.filter_success, args.include_episode_ids, args.exclude_episode_ids)
    train_eps, val_eps = shared.split_episodes(episodes, args.validation_ratio, args.seed)
    split = {'train': [e.episode_id for e in train_eps], 'validation': [e.episode_id for e in val_eps]}
    (run_dir / 'split.json').write_text(json.dumps(split, indent=2))
    stats = normalization(info, train_eps, args)
    cameras = [dict(name=c.name, height=c.height, width=c.width) for c in info.cameras]
    model = DiffusionImagePolicy(args, info.state_dim, info.action_dim, cameras, device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step = 0
    if args.checkpoint:
        _, old_stats, ckpt = load_checkpoint(args.checkpoint, device)
        for key in ('chunk_size', 'denoising_steps', 'down_dims', 'visual_feature_dim', 'diffusion_step_embed_dim',
                    'kernel_size', 'n_groups', 'use_film_scale_modulation', 'demo_data_path', 'image_mean', 'image_std'):
            if ckpt['args'][key] != vars(args)[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if ckpt['split'] != split or ckpt['cameras'] != cameras:
            raise ValueError('Resume dataset split/cameras mismatch')
        stats = old_stats
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        rng = ckpt['rng']
        torch.set_rng_state(rng['torch'].cpu())
        np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])
        if rng['cuda'] is not None and device.type == 'cuda':
            torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
        if start_step >= args.steps:
            raise ValueError('steps must exceed checkpoint step')
    train_ds = make_dataset(info, train_eps, stats, args, True)
    val_ds = make_dataset(info, val_eps, stats, args, False) if val_eps else None
    options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                   pin_memory=device.type == 'cuda', worker_init_fn=shared.seed_worker,
                   persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **options)
    val_loader = DataLoader(val_ds, shuffle=False, **options) if val_ds else None
    writer = SummaryWriter(str(run_dir))
    if args.check_env_before_training:
        shared.check_environment(ema, stats, args, device)
    print(f'[TRAIN] device={device} train_windows={len(train_ds)} val_windows={len(val_ds) if val_ds else 0} '
          f'parameters={sum(p.numel() for p in model.parameters())} cameras={cameras}', flush=True)
    iterator = iter(train_loader)
    started = time.monotonic()
    with (run_dir / 'metrics.jsonl').open('a', buffering=1) as metrics_file:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            progress = max(0., (step - args.lr_warmup_steps) / max(1, args.steps - args.lr_warmup_steps))
            scale = min(1., step / max(1, args.lr_warmup_steps)) * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate * scale
            model.train()
            loss = model.loss(*shared.move_batch(batch, device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            ema_update(ema, model, step, args)
            if step == start_step + 1 or step % args.log_freq == 0:
                metrics = dict(step=step, loss=float(loss), grad_norm=float(grad), lr=optimizer.param_groups[0]['lr'],
                               elapsed_seconds=time.monotonic() - started)
                if device.type == 'cuda':
                    metrics['peak_gpu_memory_gb'] = torch.cuda.max_memory_allocated(device) / 1024**3
                print(json.dumps(metrics), flush=True)
                metrics_file.write(json.dumps(metrics) + '\n')
                for key in ('loss', 'grad_norm', 'lr'):
                    writer.add_scalar(f'train/{key}', metrics[key], step)
            if step % args.eval_every == 0 or step == args.steps:
                # Validation must not change the training/sampling RNG stream.
                with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
                    torch.manual_seed(args.seed + 10000)
                    values = []
                    if val_loader:
                        with torch.no_grad():
                            for index, val_batch in enumerate(val_loader):
                                if index >= args.max_val_batches:
                                    break
                                values.append(float(ema.loss(*shared.move_batch(val_batch, device))))
                    if values:
                        value = float(np.mean(values))
                        writer.add_scalar('val/ema_noise_mse', value, step)
                        metrics_file.write(json.dumps(dict(step=step, val_ema_noise_mse=value)) + '\n')
                        print(f'[VAL] step={step} ema_noise_mse={value:.6f}', flush=True)
                    if args.num_eval_episodes:
                        result = shared.evaluate(ema, stats, args, device, run_dir, step)
                        metrics_file.write(json.dumps(dict(step=step, eval=result)) + '\n')
            if (step % args.save_every == 0 or step == args.steps) and args.save_model:
                save_checkpoint(run_dir / f'ckpt_{step}.pt', model, ema, optimizer, args, stats, cameras, step, split)
    writer.close()
    (run_dir / 'TRAINING_COMPLETE').write_text('ok\n')


if __name__ == '__main__':
    main()

from collections import defaultdict
import math
import os
from pathlib import Path
import random
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np
import swanlab
import torch
from torch.utils.data import DataLoader

from algo.diffusion import DiffusionAgent, ObsHistory, compute_normalizer_stats, make_identity_normalizer_stats
from train.train_configs.diffusion_config import parse_args

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks  # noqa: F401
from utils.demo_dataset import DiffusionSequenceDataset, load_demo_episodes
from utils.paths import runs_path


def build_run_name(args) -> str:
    if args.exp_name is None:
        exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    if args.evaluate:
        run_name = "eval_" + run_name
    return run_name


def set_random_seed(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


def maybe_apply_gpu_id(backend: str | None, gpu_id: int | None, gpu_backends: set[str]) -> str | None:
    if backend is None or gpu_id is None or ":" in backend:
        return backend
    if backend not in gpu_backends:
        return backend
    return f"{backend}:{gpu_id}"


def resolve_torch_device(args) -> torch.device:
    if not (torch.cuda.is_available() and args.cuda):
        return torch.device("cpu")
    device_str = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
    return torch.device(device_str)


def resolve_sim_backend(args) -> str:
    if args.sim_backend is not None:
        return maybe_apply_gpu_id(args.sim_backend, args.gpu_id, {"physx_cuda", "gpu", "cuda"})
    default_backend = "physx_cuda" if torch.cuda.is_available() else "physx_cpu"
    return maybe_apply_gpu_id(default_backend, args.gpu_id, {"physx_cuda", "gpu", "cuda"})


def resolve_render_backend(args, capture_video: bool) -> str | None:
    if not capture_video:
        return "none"
    return maybe_apply_gpu_id("sapien_cuda", args.gpu_id, {"sapien_cuda", "gpu", "cuda"})


def make_eval_env(args, run_name: str):
    capture_video = bool(args.capture_video)
    if capture_video and not torch.cuda.is_available():
        print("CUDA render device not available; disabling evaluation video capture.")
        capture_video = False
    sim_backend = resolve_sim_backend(args)
    num_eval_envs = max(1, min(int(args.num_eval_envs), int(args.num_eval_episodes)))
    if sim_backend == "physx_cpu" and num_eval_envs > 1:
        print(
            "Parallel diffusion evaluation requires a GPU-backed ManiSkill sim backend. "
            "Falling back to `num_eval_envs=1` on CPU."
        )
        num_eval_envs = 1

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array" if capture_video else None,
        sim_backend=sim_backend,
        render_backend=resolve_render_backend(args, capture_video),
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.reward_mode is not None:
        env_kwargs["reward_mode"] = args.reward_mode
    env_kwargs["max_episode_steps"] = args.num_eval_steps

    env = gym.make(
        args.env_id,
        num_envs=num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if capture_video:
        eval_output_dir = runs_path(run_name, "videos")
        if args.evaluate and args.checkpoint is not None:
            eval_output_dir = str(Path(args.checkpoint).parent / "test_videos")
        print(f"Saving eval videos to {eval_output_dir}")
        env = RecordEpisode(
            env,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=int(getattr(env.unwrapped, "control_freq", 20)),
        )
    env = ManiSkillVectorEnv(
        env,
        num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )
    return env, env_kwargs


def split_demo_episodes(episodes, validation_ratio: float, seed: int):
    if len(episodes) == 1 or validation_ratio <= 0.0:
        return list(episodes), []

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(episodes))
    num_val_episodes = int(math.floor(len(episodes) * validation_ratio))
    if num_val_episodes <= 0:
        num_val_episodes = 1
    if num_val_episodes >= len(episodes):
        num_val_episodes = len(episodes) - 1

    val_indices = set(indices[:num_val_episodes].tolist())
    train_episodes = [episode for idx, episode in enumerate(episodes) if idx not in val_indices]
    val_episodes = [episode for idx, episode in enumerate(episodes) if idx in val_indices]
    return train_episodes, val_episodes


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate_policy(agent, eval_envs, args, device, action_low: torch.Tensor, action_high: torch.Tensor):
    agent.eval()
    num_envs = int(getattr(eval_envs, "num_envs", 1))
    obs_dim = int(np.array(eval_envs.single_observation_space.shape).prod())
    history = ObsHistory(num_envs, args.n_obs_steps, obs_dim, device)

    obs, _ = eval_envs.reset(seed=args.seed)
    history.reset(obs.to(device))
    metrics = defaultdict(list)
    num_episodes = 0
    num_decision_steps = max(1, args.num_eval_steps // args.n_action_steps)

    for _ in range(num_decision_steps):
        action_env = agent.sample_action_chunk(history.cond, deterministic=True)
        for a_i in range(args.n_action_steps):
            a = torch.clamp(action_env[:, a_i], action_low, action_high)
            next_obs, _, term, trunc, infos = eval_envs.step(a)
            next_obs = next_obs.to(device)
            done = torch.logical_or(term, trunc).to(device)
            history.append(next_obs, mask=~done)
            if "final_info" in infos:
                fmask = infos["_final_info"].to(device)
                num_episodes += int(fmask.sum().item())
                if "episode" in infos["final_info"]:
                    for key, value in infos["final_info"]["episode"].items():
                        # index on the value's own device: physx_cpu returns CPU tensors while fmask lives on `device`
                        metrics[key].extend(value[fmask.to(value.device)].detach().float().cpu().tolist())
                reward_terms = infos["final_info"].get("reward_terms")
                if reward_terms is not None:
                    for name, term_values in reward_terms.items():
                        if torch.is_tensor(term_values):
                            metrics[f"rewards/{name}"].extend(
                                term_values[fmask.to(term_values.device)].detach().float().cpu().tolist())
            if done.any():
                history.reset(next_obs, env_mask=done)

    reduced_metrics = {key: float(np.mean(values)) for key, values in metrics.items() if values}
    reduced_metrics["episodes"] = float(num_episodes)
    return reduced_metrics


if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args)
    set_random_seed(args)

    device = resolve_torch_device(args)
    eval_envs, env_kwargs = make_eval_env(args, run_name)
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.array(eval_envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(eval_envs.single_action_space.shape))
    action_low = torch.from_numpy(eval_envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(eval_envs.single_action_space.high).to(device)

    train_episodes = []
    val_episodes = []
    normalizer_stats = make_identity_normalizer_stats(obs_dim, action_dim)

    if not args.evaluate:
        episodes = load_demo_episodes(
            args.demo_data_path,
            expected_obs_dim=obs_dim,
            expected_act_dim=action_dim,
        )
        train_episodes, val_episodes = split_demo_episodes(episodes, args.validation_ratio, args.seed)
        if not train_episodes:
            raise RuntimeError("No training episodes remained after the validation split.")

        normalizer_stats = compute_normalizer_stats(train_episodes)
        train_dataset = DiffusionSequenceDataset(
            train_episodes,
            n_obs_steps=args.n_obs_steps,
            horizon=args.horizon,
            drop_n_last_frames=args.drop_n_last_frames,
        )
        val_dataset = (
            DiffusionSequenceDataset(
                val_episodes,
                n_obs_steps=args.n_obs_steps,
                horizon=args.horizon,
                drop_n_last_frames=args.drop_n_last_frames,
            )
            if val_episodes
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            if val_dataset is not None
            else None
        )
        total_training_steps = max(1, len(train_loader) * args.num_epochs)
        print(
            "Loaded diffusion demos: "
            f"{len(episodes)} episodes total, "
            f"{len(train_episodes)} train / {len(val_episodes)} val, "
            f"{len(train_dataset)} train windows"
            + (f", {len(val_dataset)} val windows" if val_dataset is not None else "")
        )
    else:
        train_loader = None
        val_loader = None
        total_training_steps = 1

    max_episode_steps = gym_utils.find_max_episode_steps_value(eval_envs._env)
    if args.track:
        config = vars(args).copy()
        config["env_cfg"] = dict(
            **env_kwargs,
            num_envs=int(getattr(eval_envs, "num_envs", args.num_eval_envs)),
            env_id=args.env_id,
            env_horizon=max_episode_steps,
            partial_reset=args.eval_partial_reset,
        )
        if not args.evaluate:
            config["train_episodes"] = len(train_episodes)
            config["val_episodes"] = len(val_episodes)
            config["train_windows"] = len(train_loader.dataset)
            config["val_windows"] = 0 if val_loader is None else len(val_loader.dataset)
        swanlab.init(
            project=args.swanlab_project_name,
            entity=args.swanlab_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="Diffusion",
            tags=["diffusion", "offline_imitation", "state"],
        )

    agent = DiffusionAgent(
        args,
        obs_dim=obs_dim,
        action_dim=action_dim,
        normalizer_stats=normalizer_stats,
        run_name=run_name,
        device=device,
    ).to(device)

    resume_epoch = 0
    global_step = 0
    if args.checkpoint is not None:
        resume_info = agent.load(args.checkpoint, load_optimizer=not args.evaluate)
        resume_epoch = int(resume_info.get("epoch", 0))
        global_step = int(resume_info.get("global_step", 0))

    if args.evaluate:
        print("Running diffusion evaluation")
        eval_metrics = evaluate_policy(agent, eval_envs, args, device, action_low, action_high)
        print(f"Evaluated {int(eval_metrics.pop('episodes', 0.0))} episodes")
        for key, value in eval_metrics.items():
            print(f"eval_{key}_mean={value}")
            agent.logger.add_scalar(f"eval/{key}", value, global_step)
        agent.logger.close()
        eval_envs.close()
        raise SystemExit(0)

    print("Running diffusion training")
    for epoch in range(resume_epoch + 1, args.num_epochs + 1):
        epoch_losses = []
        for batch in train_loader:
            global_step += 1
            batch = move_batch_to_device(batch, device)
            step_metrics = agent.update(batch)
            epoch_losses.append(step_metrics["loss"])

            if global_step % args.log_freq == 0:
                agent.logger.add_scalar("train/loss", step_metrics["loss"], global_step)
                agent.logger.add_scalar("train/grad_norm", step_metrics["grad_norm"], global_step)

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        agent.logger.add_scalar("train/epoch_loss", epoch_loss, global_step)
        print(f"Epoch {epoch}/{args.num_epochs}: train_loss={epoch_loss:.6f}")

        should_eval = (epoch % args.eval_freq == 0) or (epoch == 1) or (epoch == args.num_epochs)
        if should_eval:
            val_loss = float("nan") if val_loader is None else agent.evaluate_loss(val_loader, device)
            agent.logger.add_scalar("val/loss", val_loss, global_step)
            print(f"Epoch {epoch}/{args.num_epochs}: val_loss={val_loss:.6f}")

            eval_metrics = evaluate_policy(agent, eval_envs, args, device, action_low, action_high)
            print(f"Evaluated {int(eval_metrics.pop('episodes', 0.0))} episodes")
            for key, value in eval_metrics.items():
                agent.logger.add_scalar(f"eval/{key}", value, global_step)
                print(f"eval_{key}_mean={value}")

        should_save = args.save_model and ((epoch % args.save_freq == 0) or (epoch == args.num_epochs))
        if should_save:
            model_path = runs_path(
                run_name,
                f"ckpt_{epoch}.pt" if epoch != args.num_epochs else "final_ckpt.pt",
            )
            agent.save(model_path, epoch=epoch, global_step=global_step)

    agent.logger.close()
    eval_envs.close()

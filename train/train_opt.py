"""OPT (Online Pre-Training) fine-tuning with an IQL backbone.

Three-stage offline-to-online RL (Shin et al., ICML 2025, arXiv:2507.08387):
the offline stage is a separate IQL run (`train_iql.py`); this script runs the
two online stages, warm-started from that IQL checkpoint.

  Phase B (online pre-training): freeze the policy and the offline value
    function Q^off, collect `opt_env_steps` transitions with the frozen policy,
    and meta-train a fresh Q^on/V^on for `opt_grad_steps` gradient steps.
  Phase C (online fine-tuning): update both value functions and the policy;
    the AWR policy weight uses a kappa-scheduled combined advantage.

Reuses env construction and evaluation helpers from `train_iql` / `train_sac`.
"""

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

import mani_skill.envs  # noqa: F401

import tasks  # noqa: F401  (registers every RoboDex env)
from algo.offline_to_online_rl.opt import OPTAgent
from train.train_configs.opt_config import parse_args
from utils.offline_replay_buffer import load_lerobot_replay_dataset
from utils.online_replay_buffer import OnlineReplayBuffer
from utils.paths import runs_path
from utils.train_utils import (
    build_run_name,
    evaluate_policy,
    log_train_episode_metrics,
    make_eval_env,
    make_train_env,
    resolve_device,
    set_seed,
)


def kappa_at(args, ft_step: int) -> float:
    progress = min(max(ft_step / max(1, args.kappa_schedule_steps), 0.0), 1.0)
    return args.kappa_start + (args.kappa_end - args.kappa_start) * progress


def collect_step(agent, train_envs, buffer, obs, device, deterministic=False):
    """Step the (frozen or live) policy once and store the transition."""
    with torch.no_grad():
        action = agent.select_action(obs.to(device), deterministic=deterministic)
    next_obs, reward, terminations, truncations, infos = train_envs.step(action)
    real_next_obs = next_obs.clone()
    if "final_info" in infos and "final_observation" in infos:
        done_mask = infos["_final_info"]
        real_next_obs[done_mask] = infos["final_observation"][done_mask]
    buffer.add(obs, action, reward, real_next_obs, terminations.float())
    return next_obs, infos


def log_metrics(agent, metrics, env_step, start_time, header):
    for key, value in metrics.items():
        if np.isfinite(value):
            agent.logger.add_scalar(f"losses/{key}", value, env_step)
    sps = env_step / max(time.time() - start_time, 1e-6)
    agent.logger.add_scalar("charts/steps_per_second", sps, env_step)
    print(f"{header} | " + ", ".join(
        f"{k}={v:.4f}" for k, v in metrics.items() if np.isfinite(v)
    ))


def run_eval(agent, eval_envs, args, device, env_step, tag=""):
    eval_metrics = evaluate_policy(agent, eval_envs, args, device)
    print(f"[eval{tag}] step={env_step} "
          f"episodes={int(eval_metrics.pop('episodes', 0.0))}")
    for key, value in eval_metrics.items():
        print(f"  eval_{key}={value}")
        agent.logger.add_scalar(f"eval/{key}", value, env_step)
    agent.train()
    return eval_metrics


if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args, "train_opt")
    set_seed(args)
    device = resolve_device(args)

    eval_envs, env_kwargs = make_eval_env(args, run_name)
    if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
        raise TypeError("OPT currently supports only continuous Box action spaces.")
    obs_dim = int(np.prod(eval_envs.single_observation_space.shape))
    action_dim = int(np.prod(eval_envs.single_action_space.shape))
    action_low = torch.from_numpy(eval_envs.single_action_space.low).float()
    action_high = torch.from_numpy(eval_envs.single_action_space.high).float()

    replay = None
    if args.evaluate:
        obs_mean = torch.zeros(obs_dim)
        obs_std = torch.ones(obs_dim)
    else:
        replay = load_lerobot_replay_dataset(
            args.demo_data_path,
            expected_obs_dim=obs_dim,
            expected_action_dim=action_dim,
            success_only=args.success_only,
        )
        obs_mean, obs_std = replay.observation_stats()
        print(
            "Loaded offline demo dataset for mixing: "
            f"{len(replay)} transitions, {replay.num_episodes} episodes, "
            f"obs_dim={replay.obs_dim}, action_dim={replay.action_dim}"
        )

    if args.track:
        config = vars(args).copy()
        config["env_cfg"] = dict(
            **env_kwargs,
            num_envs=int(getattr(eval_envs, "num_envs", args.num_eval_envs)),
            env_id=args.env_id,
        )
        swanlab.init(
            project=args.swanlab_project_name,
            entity=args.swanlab_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="OPT",
            tags=["opt", "online_pretrain", "iql", "state"],
        )

    agent = OPTAgent(
        args,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        obs_mean=obs_mean,
        obs_std=obs_std,
        run_name=run_name,
    ).to(device)

    # Resume a full OPT run (with optimizers) takes precedence; otherwise
    # warm-start actor + Q^off/V^off + obs stats from the offline IQL checkpoint.
    if args.checkpoint is not None:
        agent.load(args.checkpoint, load_optimizers=not args.evaluate)
    elif args.pretrained_checkpoint is not None:
        agent.load_pretrained(args.pretrained_checkpoint)

    if args.evaluate:
        metrics = run_eval(agent, eval_envs, args, device, 0, tag="-only")
        agent.logger.close()
        eval_envs.close()
        raise SystemExit(0)

    train_envs, num_envs = make_train_env(args)
    buffer = OnlineReplayBuffer(
        args.buffer_size,
        obs_dim,
        action_dim,
        device,
        offline=replay,
        offline_mix_fraction=args.offline_mix_fraction,
    )
    use_bc_anchor = args.actor_bc_weight > 0.0

    # Sanity baseline: the warm-started policy before any online update. With a
    # good IQL checkpoint this should already show non-trivial success and must
    # match the IQL run -- the key check that load_pretrained's key remap is right.
    init_metrics = run_eval(agent, eval_envs, args, device, 0, tag="-warmstart")
    print(f"[warm-start] success={init_metrics.get('success', float('nan')):.3f}")
    agent.train()

    obs, _ = train_envs.reset(seed=args.seed)
    env_step = 0
    start_time = time.time()

    # ---------------------------------------------------------- Phase B: pretrain
    print(f"=== Phase B: online pre-training "
          f"({args.opt_env_steps} env steps, {args.opt_grad_steps} grad steps) ===")
    while env_step < args.opt_env_steps:
        obs, infos = collect_step(
            agent, train_envs, buffer, obs, device, deterministic=False
        )
        log_train_episode_metrics(agent, infos, env_step + num_envs)
        env_step += num_envs

    for grad_step in range(1, args.opt_grad_steps + 1):
        batch_on = buffer._sample_online(args.batch_size)
        batch_off = replay.sample(args.batch_size, device, include_extra=False)
        metrics = agent.online_pretrain_update(batch_off, batch_on)
        if grad_step % args.log_freq == 0 or grad_step == 1:
            for key, value in metrics.items():
                if np.isfinite(value):
                    agent.logger.add_scalar(f"losses/{key}", value, env_step)
            print(f"[pretrain] grad_step {grad_step}/{args.opt_grad_steps}: "
                  f"meta_loss={metrics['opt_meta_loss']:.4f}, "
                  f"q_on={metrics['opt_q_loss_on']:.4f}, "
                  f"v_on={metrics['opt_v_loss_on']:.4f}")

    run_eval(agent, eval_envs, args, device, env_step, tag="-after-pretrain")

    # ------------------------------------------------------- Phase C: fine-tune
    print(f"=== Phase C: online fine-tuning to {args.total_env_steps} env steps ===")
    update_step = 0
    next_log = env_step + args.log_freq
    next_eval = env_step + args.eval_freq
    next_save = env_step + args.save_freq

    while env_step < args.total_env_steps:
        obs, infos = collect_step(
            agent, train_envs, buffer, obs, device, deterministic=False
        )
        log_train_episode_metrics(agent, infos, env_step + num_envs)
        env_step += num_envs

        if buffer.size >= args.batch_size:
            for _ in range(args.updates_per_iter):
                update_step += 1
                kappa = kappa_at(args, update_step)
                batch = buffer.sample(args.batch_size, device)
                demo_batch = (
                    replay.sample(args.batch_size, device, include_extra=False)
                    if use_bc_anchor
                    else None
                )
                metrics = agent.finetune_update(
                    batch, update_step, kappa, demo_batch=demo_batch
                )

        if env_step >= next_log and update_step > 0:
            agent.logger.add_scalar("charts/update_step", update_step, env_step)
            log_metrics(
                agent, metrics, env_step, start_time,
                f"env_step {env_step}/{args.total_env_steps} (updates={update_step})",
            )
            next_log += args.log_freq

        if env_step >= next_eval:
            run_eval(agent, eval_envs, args, device, env_step)
            next_eval += args.eval_freq

        if args.save_model and env_step >= next_save:
            agent.save(runs_path(run_name, f"ckpt_{env_step}.pt"), env_step)
            next_save += args.save_freq

    if args.save_model:
        agent.save(runs_path(run_name, "final_ckpt.pt"), env_step)

    agent.logger.close()
    train_envs.close()
    eval_envs.close()

"""Online SAC fine-tuning, warm-started from an offline IQL/CQL checkpoint.

Collects transitions from a vectorized ManiSkill environment with the
(pretrained, hence non-random) stochastic policy, stores them in a FIFO replay
buffer, and runs off-policy SAC updates that mix in offline demos (RLPD-style).
Reuses the env construction and evaluation helpers from ``train_iql``.
"""

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

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks  # noqa: F401  (registers every RoboDex env)
from algo.offline_to_online_rl.sac import SACAgent
from train.train_configs.sac_config import parse_args
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


if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args, "train_sac")
    set_seed(args)
    device = resolve_device(args)

    eval_envs, env_kwargs = make_eval_env(args, run_name)
    if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
        raise TypeError("SAC currently supports only continuous Box action spaces.")
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
            group="SAC",
            tags=["sac", "online_finetune", "state"],
        )

    agent = SACAgent(
        args,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        obs_mean=obs_mean,
        obs_std=obs_std,
        run_name=run_name,
    ).to(device)

    # Resume a SAC run (with optimizers) takes precedence; otherwise warm-start
    # actor + twin Q + targets + obs stats from the offline checkpoint.
    if args.checkpoint is not None:
        agent.load(args.checkpoint, load_optimizers=not args.evaluate)
    elif args.pretrained_checkpoint is not None:
        agent.load_pretrained(args.pretrained_checkpoint)

    if args.evaluate:
        metrics = evaluate_policy(agent, eval_envs, args, device)
        print(f"Evaluated {int(metrics.pop('episodes', 0.0))} episodes")
        for key, value in metrics.items():
            print(f"eval_{key}_mean={value}")
            agent.logger.add_scalar(f"eval/{key}", value, 0)
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

    # Sanity baseline: how good is the warm-started policy before any update.
    init_metrics = evaluate_policy(agent, eval_envs, args, device)
    print(
        f"[warm-start eval] success={init_metrics.get('success', float('nan')):.3f} "
        f"return={init_metrics.get('return', init_metrics.get('episode_return', float('nan')))}"
    )
    for key, value in init_metrics.items():
        if key != "episodes":
            agent.logger.add_scalar(f"eval/{key}", value, 0)
    agent.train()

    action_low_dev = agent.action_low
    action_high_dev = agent.action_high

    obs, _ = train_envs.reset(seed=args.seed)
    env_step = 0
    update_step = 0
    next_log = args.log_freq
    next_eval = args.eval_freq
    next_save = args.save_freq
    start_time = time.time()

    while env_step < args.total_env_steps:
        with torch.no_grad():
            if env_step < args.start_steps:
                rand = torch.rand((num_envs, action_dim), device=device)
                action = action_low_dev + rand * (action_high_dev - action_low_dev)
            else:
                action = agent.select_action(obs.to(device), deterministic=False)

        next_obs, reward, terminations, truncations, infos = train_envs.step(action)

        # Use the true terminal observation (not the auto-reset obs) as next_obs,
        # and bootstrap on truncation (time limit) but not on real termination.
        real_next_obs = next_obs.clone()
        if "final_info" in infos and "final_observation" in infos:
            done_mask = infos["_final_info"]
            real_next_obs[done_mask] = infos["final_observation"][done_mask]
        buffer.add(obs, action, reward, real_next_obs, terminations.float())

        log_train_episode_metrics(agent, infos, env_step + num_envs)
        obs = next_obs
        env_step += num_envs

        if env_step >= args.start_steps and buffer.size >= args.batch_size:
            for _ in range(args.updates_per_iter):
                update_step += 1
                batch = buffer.sample(args.batch_size, device)
                demo_batch = (
                    replay.sample(args.batch_size, device, include_extra=False)
                    if use_bc_anchor
                    else None
                )
                metrics = agent.update(batch, update_step, demo_batch=demo_batch)

        if env_step >= next_log and update_step > 0:
            for key, value in metrics.items():
                if np.isfinite(value):
                    agent.logger.add_scalar(f"losses/{key}", value, env_step)
            sps = env_step / max(time.time() - start_time, 1e-6)
            agent.logger.add_scalar("charts/steps_per_second", sps, env_step)
            agent.logger.add_scalar("charts/update_step", update_step, env_step)
            print(
                f"env_step {env_step}/{args.total_env_steps} "
                f"(updates={update_step}): q_loss={metrics['q_loss']:.4f}, "
                f"actor_loss={metrics['actor_loss']:.4f}, "
                f"alpha={metrics['entropy_alpha']:.4f}, "
                f"q1={metrics['q1_data']:.3f}"
            )
            next_log += args.log_freq

        if env_step >= next_eval:
            eval_metrics = evaluate_policy(agent, eval_envs, args, device)
            print(f"Evaluated {int(eval_metrics.pop('episodes', 0.0))} episodes")
            for key, value in eval_metrics.items():
                print(f"eval_{key}_mean={value}")
                agent.logger.add_scalar(f"eval/{key}", value, env_step)
            agent.train()
            next_eval += args.eval_freq

        if args.save_model and env_step >= next_save:
            agent.save(runs_path(run_name, f"ckpt_{env_step}.pt"), env_step)
            next_save += args.save_freq

    if args.save_model:
        agent.save(runs_path(run_name, "final_ckpt.pt"), env_step)

    agent.logger.close()
    train_envs.close()
    eval_envs.close()

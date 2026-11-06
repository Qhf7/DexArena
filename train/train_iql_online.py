"""Online IQL fine-tuning from an offline IQL checkpoint.

This is the direct IQL counterpart to ``train_sac.py``: it keeps the IQL
critic/value/policy losses, collects new transitions online, and trains from a
replay buffer that can mix online data with frozen offline demonstrations.
"""

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

from algo.offline_rl.iql import IQLAgent
import tasks  # noqa: F401  (registers every RoboDex env)
from train.train_configs.iql_online_config import parse_args
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


def run_eval(agent, eval_envs, args, device, env_step, tag=""):
    eval_metrics = evaluate_policy(agent, eval_envs, args, device)
    print(
        f"[eval{tag}] step={env_step} "
        f"episodes={int(eval_metrics.pop('episodes', 0.0))}"
    )
    for key, value in eval_metrics.items():
        print(f"  eval_{key}={value}")
        agent.logger.add_scalar(f"eval/{key}", value, env_step)
    agent.train()
    return eval_metrics


if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args, "train_iql_online")
    set_seed(args)
    device = resolve_device(args)

    eval_envs, env_kwargs = make_eval_env(args, run_name)
    if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
        raise TypeError("IQL online currently supports only continuous Box action spaces.")
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
            "Loaded offline demo dataset for IQL online mixing: "
            f"{len(replay)} transitions, {replay.num_episodes} episodes, "
            f"obs_dim={replay.obs_dim}, action_dim={replay.action_dim}, "
            f"success_only={args.success_only}"
        )

    if args.track:
        config = vars(args).copy()
        config["env_cfg"] = dict(
            **env_kwargs,
            num_envs=int(getattr(eval_envs, "num_envs", args.num_eval_envs)),
            env_id=args.env_id,
        )
        if replay is not None:
            config["offline_transitions"] = len(replay)
            config["offline_episodes"] = replay.num_episodes
        swanlab.init(
            project=args.swanlab_project_name,
            entity=args.swanlab_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="IQL-Online",
            tags=["iql", "online_finetune", "state"],
        )

    agent = IQLAgent(
        args,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        obs_mean=obs_mean,
        obs_std=obs_std,
        run_name=run_name,
    ).to(device)

    if args.checkpoint is not None:
        ckpt_step = agent.load(
            args.checkpoint,
            load_optimizers=bool(args.load_optimizer_state and not args.evaluate),
        )
        print(f"Loaded IQL checkpoint `{args.checkpoint}` at offline_step={ckpt_step}")

    if args.evaluate:
        run_eval(agent, eval_envs, args, device, 0, tag="-only")
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

    init_metrics = run_eval(agent, eval_envs, args, device, 0, tag="-warmstart")
    print(
        f"[warm-start] success={init_metrics.get('success', float('nan')):.3f} "
        f"return={init_metrics.get('return', init_metrics.get('episode_return', float('nan')))}"
    )

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
                actor_batch_size = args.actor_batch_size or args.batch_size
                critic_batch = buffer.sample(args.batch_size, device)
                actor_batch = buffer.sample(actor_batch_size, device)
                bc_anchor_batch = (
                    replay.sample(actor_batch_size, device, include_extra=False)
                    if args.actor_bc_weight > 0.0
                    else None
                )
                metrics = agent.update(
                    critic_batch,
                    update_step,
                    actor_batch=actor_batch,
                    bc_anchor_batch=bc_anchor_batch,
                )

        if env_step >= next_log and update_step > 0:
            for key, value in metrics.items():
                if np.isfinite(value):
                    agent.logger.add_scalar(f"losses/{key}", value, env_step)
            sps = env_step / max(time.time() - start_time, 1e-6)
            agent.logger.add_scalar("charts/steps_per_second", sps, env_step)
            agent.logger.add_scalar("charts/update_step", update_step, env_step)
            print(
                f"env_step {env_step}/{args.total_env_steps} "
                f"(updates={update_step}): v_loss={metrics['v_loss']:.4f}, "
                f"q_loss={metrics['q_loss']:.4f}, "
                f"actor_loss={metrics['actor_loss']:.4f}, "
                f"awr_w={metrics['awr_weight_mean']:.3f}, "
                f"q1={metrics['q1_data']:.3f}"
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

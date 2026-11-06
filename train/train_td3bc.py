"""Offline TD3+BC training on the DexArena tasks.

Self-contained: defines its own env/eval helpers rather than importing from
``train_iql``. By default trains on the *successful* demonstrations only
(``--success-only``, on by default); failed/timeout demos are filtered out by the
replay loader. Logs critic/actor diagnostics under ``losses/*`` and policy
performance under ``eval/*`` (see the file header of ``algo/offline_rl/td3bc.py``).
"""

from pathlib import Path
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

from algo.offline_rl.td3bc import TD3BCAgent
import tasks  # noqa: F401  (registers every RoboDex env)
from train.train_configs.td3bc_config import parse_args
from utils.offline_replay_buffer import load_lerobot_replay_dataset
from utils.paths import runs_path
from utils.train_utils import (
    build_run_name,
    evaluate_policy,
    make_eval_env,
    resolve_device,
    set_seed,
)




if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args, "train_td3bc")
    set_seed(args)
    device = resolve_device(args)

    eval_envs, env_kwargs = make_eval_env(args, run_name)
    if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
        raise TypeError("TD3+BC currently supports only continuous Box action spaces.")
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
            "Loaded offline replay dataset: "
            f"{len(replay)} transitions, {replay.num_episodes} episodes, "
            f"obs_dim={replay.obs_dim}, action_dim={replay.action_dim}, "
            f"success_only={args.success_only}"
        )
        dataset_action_min = replay.actions.amin(dim=0)
        dataset_action_max = replay.actions.amax(dim=0)
        tolerance = 1e-5
        if torch.any(dataset_action_min < action_low - tolerance) or torch.any(
            dataset_action_max > action_high + tolerance
        ):
            raise RuntimeError(
                "Dataset actions fall outside the evaluation environment action bounds. "
                "Check that the dataset and TD3+BC environment use the same control mode."
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
            group="TD3BC",
            tags=["td3bc", "offline_rl", "state"],
        )

    agent = TD3BCAgent(
        args,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        obs_mean=obs_mean,
        obs_std=obs_std,
        run_name=run_name,
    ).to(device)

    global_step = 0
    if args.checkpoint is not None:
        global_step = agent.load(args.checkpoint, load_optimizers=not args.evaluate)
    elif args.bc_checkpoint is not None:
        agent.load_actor_from_bc(args.bc_checkpoint)

    if args.evaluate:
        metrics = evaluate_policy(agent, eval_envs, args, device)
        print(f"Evaluated {int(metrics.pop('episodes', 0.0))} episodes")
        for key, value in metrics.items():
            print(f"eval_{key}_mean={value}")
            agent.logger.add_scalar(f"eval/{key}", value, global_step)
        agent.logger.close()
        eval_envs.close()
        raise SystemExit(0)

    agent.train()
    start_time = time.time()
    initial_step = global_step
    while global_step < args.total_updates:
        global_step += 1
        batch = replay.sample(args.batch_size, device)
        metrics = agent.update(batch, global_step)

        if global_step % args.log_freq == 0:
            for key, value in metrics.items():
                if np.isfinite(value):
                    agent.logger.add_scalar(f"losses/{key}", value, global_step)
            updates_per_second = (global_step - initial_step) / max(
                time.time() - start_time, 1e-6
            )
            agent.logger.add_scalar(
                "charts/updates_per_second", updates_per_second, global_step
            )
            print(
                f"Update {global_step}/{args.total_updates}: "
                f"q_loss={metrics['q_loss']:.4f}, "
                f"bc_loss={metrics['bc_loss']:.6f}, "
                f"q1={metrics['q1_data']:.3f}"
            )

        if global_step % args.eval_freq == 0 or global_step == args.total_updates:
            eval_metrics = evaluate_policy(agent, eval_envs, args, device)
            print(f"Evaluated {int(eval_metrics.pop('episodes', 0.0))} episodes")
            for key, value in eval_metrics.items():
                print(f"eval_{key}_mean={value}")
                agent.logger.add_scalar(f"eval/{key}", value, global_step)
            agent.train()

        if args.save_model and (
            global_step % args.save_freq == 0 or global_step == args.total_updates
        ):
            filename = (
                "final_ckpt.pt"
                if global_step == args.total_updates
                else f"ckpt_{global_step}.pt"
            )
            agent.save(runs_path(run_name, filename), global_step)

    agent.logger.close()
    eval_envs.close()

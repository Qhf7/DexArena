"""Online RLPD training for ball grasping, from scratch.

Collects transitions from a vectorized ManiSkill environment with the current
(initially random) stochastic policy, stores them in a FIFO replay buffer, and
runs high-UTD SAC updates whose every batch mixes 50/50 with a frozen offline
demo dataset (RLPD symmetric sampling). Unlike ``train_sac``, there is no
offline warm-start and no BC anchor — canonical RLPD learns from scratch.
Reuses the env construction and evaluation helpers from ``train_iql``.
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
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks  # noqa: F401  (registers every RoboDex env)
from algo.offline_to_online_rl.rlpd import RLPDAgent
from train.train_configs.rlpd_config import parse_args
from utils.delta_action import demo_init_action, to_delta_dataset
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
    run_name = build_run_name(args, "train_rlpd")
    set_seed(args)
    device = resolve_device(args)

    # The residual wrapper needs the demos' first command as its `a_{-1}` seed,
    # and it changes the observation the env reports, so it has to be resolved
    # before any env is built.
    init_action = (
        demo_init_action(args.demo_data_path) if args.delta_action else None
    )

    eval_envs, env_kwargs = make_eval_env(args, run_name, init_action=init_action)
    if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
        raise TypeError("RLPD currently supports only continuous Box action spaces.")
    obs_dim = int(np.prod(eval_envs.single_observation_space.shape))
    action_dim = int(np.prod(eval_envs.single_action_space.shape))
    action_low = torch.from_numpy(eval_envs.single_action_space.low).float()
    action_high = torch.from_numpy(eval_envs.single_action_space.high).float()

    replay = None
    action_std = None
    if args.evaluate:
        obs_mean = torch.zeros(obs_dim)
        obs_std = torch.ones(obs_dim)
    else:
        # Under `delta_action` the wrapper appends the previous command to the
        # observation, so the dataset on disk is `action_dim` columns narrower
        # than what the agent sees; `to_delta_dataset` adds the same block back.
        replay = load_lerobot_replay_dataset(
            args.demo_data_path,
            expected_obs_dim=obs_dim - action_dim if args.delta_action else obs_dim,
            expected_action_dim=action_dim,
            success_only=args.success_only,
        )
        if args.delta_action:
            replay = to_delta_dataset(
                replay,
                delta=args.delta_action_scale,
                action_low=eval_envs.absolute_action_low,
                action_high=eval_envs.absolute_action_high,
                init_action=init_action,
            )
        # A time-limit row is not a terminal state -- the episode was cut by the
        # harness, not by the MDP -- so its target must bootstrap instead of being
        # pinned at the immediate reward. Unconditional: there is no task here for
        # which anchoring V=0 at a timeout is the correct target.
        timeouts = int((replay.dones * replay.truncated).sum().item())
        replay.dones = replay.dones * (1.0 - replay.truncated)
        print(f"[replay] bootstrap restored on {timeouts} timeout transitions")
        obs_mean, obs_std = replay.observation_stats()
        # Scale of the demonstrated motion; only the entropy target is expressed
        # in these units (see OfflineReplayDataset.action_stats -- the BC anchor
        # used to be, and measurably should not be).
        action_std = replay.action_stats()
        print(
            "Loaded offline demo dataset for mixing: "
            f"{len(replay)} transitions, {replay.num_episodes} episodes, "
            f"obs_dim={replay.obs_dim}, action_dim={replay.action_dim}, "
            f"robot_uids={args.robot_uids}, success_only={args.success_only}"
        )
        dataset_action_min = replay.actions.amin(dim=0)
        dataset_action_max = replay.actions.amax(dim=0)
        tolerance = 1e-5
        if torch.any(dataset_action_min < action_low - tolerance) or torch.any(
            dataset_action_max > action_high + tolerance
        ):
            raise RuntimeError(
                "Dataset actions fall outside the environment action bounds. "
                "Check that the dataset, robot_uids, and control_mode match."
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
            group="RLPD",
            tags=["rlpd", "online", "state"],
        )

    agent = RLPDAgent(
        args,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        obs_mean=obs_mean,
        obs_std=obs_std,
        run_name=run_name,
        action_std=action_std,
    ).to(device)

    # Resume an interrupted RLPD run (with optimizers). No offline warm-start:
    # canonical RLPD trains from scratch.
    if args.checkpoint is not None:
        agent.load(args.checkpoint, load_optimizers=not args.evaluate)

    if args.evaluate:
        metrics = evaluate_policy(agent, eval_envs, args, device)
        print(f"Evaluated {int(metrics.pop('episodes', 0.0))} episodes")
        for key, value in metrics.items():
            print(f"eval_{key}_mean={value}")
            agent.logger.add_scalar(f"eval/{key}", value, 0)
        agent.logger.close()
        eval_envs.close()
        raise SystemExit(0)

    train_envs, num_envs = make_train_env(args, init_action=init_action)
    buffer = OnlineReplayBuffer(
        args.buffer_size,
        obs_dim,
        action_dim,
        device,
        offline=replay,
        offline_mix_fraction=args.offline_mix_fraction,
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
                # The residual space is already bounded per step, so uniform
                # sampling is the well-scaled warmup there. The branch below is
                # for raw `pd_joint_pos`, where a uniform draw is an absolute
                # joint target anywhere in the limits.
                if args.control_mode == "pd_joint_pos" and not args.delta_action:
                    arm_dim = min(7, action_dim)
                    current_arm_qpos = obs[:, :arm_dim].to(device)
                    local_noise = (
                        torch.rand((num_envs, arm_dim), device=device) * 2.0 - 1.0
                    ) * args.joint_warmup_delta
                    action[:, :arm_dim] = torch.clamp(
                        current_arm_qpos + local_noise,
                        action_low_dev[:arm_dim],
                        action_high_dev[:arm_dim],
                    )
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
            # UTD ratio: `utd_ratio` critic updates per collected transition;
            # actor + temperature updated once per block, on the last mini-batch.
            # Draw all `utd_ratio` mini-batches in one big `sample` call and slice
            # them out, instead of paying the per-sample overhead `utd_ratio`
            # times. The big batch is shuffled first so each slice keeps the
            # buffer's ~50/50 online/offline mix (sample() concatenates the two
            # sources, so contiguous slices would otherwise be single-source).
            mega_size = args.batch_size * args.utd_ratio
            mega = buffer.sample(mega_size, device)
            perm = torch.randperm(mega_size, device=device)
            mega = {
                key: value[perm].view(args.utd_ratio, args.batch_size, *value.shape[1:])
                for key, value in mega.items()
            }
            for i in range(args.utd_ratio):
                update_step += 1
                batch = {key: value[i] for key, value in mega.items()}
                metrics = agent.update(
                    batch, update_step, update_actor=(i == args.utd_ratio - 1)
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
                f"(updates={update_step}): q_loss={metrics['q_loss']:.4f}, "
                f"actor_loss={metrics['actor_loss']:.4f}, "
                f"alpha={metrics['entropy_alpha']:.4f}, "
                f"q_mean={metrics['q_mean']:.3f}"
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

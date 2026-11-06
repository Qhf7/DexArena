"""Offline pure-BC training.

Shared env/eval helpers come from ``utils.train_utils``; only the optional
object-pose projection (``--drop-object-pose``) is BC-specific and lives here.
With ``--success-only`` the replay loader drops failed/timeout demos. Logs
imitation diagnostics under ``losses/*`` and policy performance under ``eval/*``
(see the file header of ``algo/offline_rl/bc.py``).
"""

import functools
from pathlib import Path
import sys
import time
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np
import swanlab
import torch

import mani_skill.envs  # noqa: F401

from algo.offline_rl.bc import BCAgent
import tasks  # noqa: F401  (registers every RoboDex env)
from train.train_configs.bc_config import parse_args
from utils.offline_replay_buffer import load_lerobot_replay_dataset
from utils.paths import runs_path
from utils.train_utils import (
    build_run_name,
    evaluate_policy,
    make_eval_env,
    resolve_device,
    set_seed,
)


OBJECT_POSE_START = 29
OBJECT_POSE_DIM = 7



def make_obs_keep_indices(args, raw_obs_dim: int) -> Optional[torch.Tensor]:
    if not args.drop_object_pose:
        return None
    object_pose_end = OBJECT_POSE_START + OBJECT_POSE_DIM
    if raw_obs_dim < object_pose_end:
        raise ValueError(
            "`--drop-object-pose` expects PandaBallGrasp state observations with "
            f"at least {object_pose_end} dims, got obs_dim={raw_obs_dim}."
        )
    keep_indices = []
    for idx in range(raw_obs_dim):
        if not OBJECT_POSE_START <= idx < object_pose_end:
            keep_indices.append(idx)
    return torch.tensor(keep_indices, dtype=torch.long)


def project_observation(
    obs: torch.Tensor, keep_indices: Optional[torch.Tensor]
) -> torch.Tensor:
    if keep_indices is None:
        return obs
    return obs.index_select(-1, keep_indices.to(device=obs.device))


def project_replay_observations(replay, keep_indices: Optional[torch.Tensor]) -> None:
    if keep_indices is None:
        return
    replay.observations = project_observation(replay.observations, keep_indices)
    replay.next_observations = project_observation(replay.next_observations, keep_indices)




if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args, "train_bc")
    set_seed(args)
    device = resolve_device(args)

    eval_envs, env_kwargs = make_eval_env(args, run_name)
    if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
        raise TypeError("BC currently supports only continuous Box action spaces.")
    raw_obs_dim = int(np.prod(eval_envs.single_observation_space.shape))
    obs_keep_indices = make_obs_keep_indices(args, raw_obs_dim)
    obs_projector = functools.partial(project_observation, keep_indices=obs_keep_indices)
    obs_dim = (
        int(obs_keep_indices.numel())
        if obs_keep_indices is not None
        else raw_obs_dim
    )
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
            expected_obs_dim=raw_obs_dim,
            expected_action_dim=action_dim,
            success_only=args.success_only,
        )
        project_replay_observations(replay, obs_keep_indices)
        obs_mean, obs_std = replay.observation_stats()
        print(
            "Loaded offline replay dataset: "
            f"{len(replay)} transitions, {replay.num_episodes} episodes, "
            f"obs_dim={replay.obs_dim}, raw_obs_dim={raw_obs_dim}, "
            f"action_dim={replay.action_dim}, "
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
                "Check that the dataset and BC environment use the same control mode."
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
        config["raw_obs_dim"] = raw_obs_dim
        config["obs_dim"] = obs_dim
        swanlab.init(
            project=args.swanlab_project_name,
            entity=args.swanlab_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="BC",
            tags=["bc", "imitation", "state"],
        )

    agent = BCAgent(
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

    if args.evaluate:
        metrics = evaluate_policy(
            agent, eval_envs, args, device, obs_projector=obs_projector
        )
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
                f"bc_loss={metrics['bc_loss']:.6f}, "
                f"bc_mse={metrics['bc_mse']:.6f}, "
                f"abs_err={metrics['action_abs_error']:.6f}"
            )

        if global_step % args.eval_freq == 0 or global_step == args.total_updates:
            eval_metrics = evaluate_policy(
            agent, eval_envs, args, device, obs_projector=obs_projector
        )
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

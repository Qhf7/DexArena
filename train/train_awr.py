"""Frozen-Q AWR training: decoupled policy improvement on the FQE behavior-Q.

Loads the Q-network validated by the FQE/SARSA stage (`train/train_fqe.py`),
freezes it, and extracts an improved policy via one-step advantage-weighted
regression (`algo/offline_rl/awr.py`). No critic is trained here. The evaluation
harness (`make_eval_env`/`evaluate_policy`) comes from `utils/train_utils.py`.

Usage:
  python -m train.train_awr \
      --fqe_checkpoint <fqe-run>/final_ckpt.pt \
      --demo_data_path <lerobot-demo-root> \
      --exp_name ballgrasp_awr --track
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
import tasks  # noqa: F401  (registers every RoboDex env)

from algo.offline_rl.awr import AWRAgent
from algo.networks import QNetwork
from train.train_configs.awr_config import parse_args
from utils.offline_replay_buffer import load_lerobot_replay_dataset
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


def load_frozen_q(checkpoint_path: str, obs_dim: int, action_dim: int, device):
    """Rebuild the FQE Q-network and its obs-normalization stats from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["agent"]
    hidden_dims = tuple(int(d) for d in ckpt["args"]["hidden_dims"])
    frozen_q = QNetwork(obs_dim, action_dim, hidden_dims)
    qf_state = {k[len("qf."):]: v for k, v in state.items() if k.startswith("qf.")}
    if not qf_state:
        raise RuntimeError(f"`{checkpoint_path}` has no `qf.*` weights (not an FQE checkpoint?).")
    frozen_q.load_state_dict(qf_state)
    frozen_q.eval()
    frozen_q.requires_grad_(False)
    # Obs normalization MUST match the frozen Q's training, or the Q sees OOD inputs.
    obs_mean = state["obs_mean"].float()
    obs_std = state["obs_std"].float()
    return frozen_q, obs_mean, obs_std


if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args, "train_awr")
    set_seed(args)
    device = resolve_device(args)

    eval_envs, env_kwargs = make_eval_env(args, run_name)
    if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
        raise TypeError("AWR currently supports only continuous Box action spaces.")
    obs_dim = int(np.prod(eval_envs.single_observation_space.shape))
    action_dim = int(np.prod(eval_envs.single_action_space.shape))
    action_low = torch.from_numpy(eval_envs.single_action_space.low).float()
    action_high = torch.from_numpy(eval_envs.single_action_space.high).float()

    # The frozen Q + its obs stats come from the FQE checkpoint, NOT the buffer:
    # re-deriving stats from a (possibly augmented) buffer would feed OOD inputs
    # to the frozen Q. When reloading a self-contained AWR checkpoint for eval,
    # the embedded qf_frozen/obs stats are used instead.
    if args.fqe_checkpoint is not None:
        frozen_q, obs_mean, obs_std = load_frozen_q(
            args.fqe_checkpoint, obs_dim, action_dim, device
        )
    else:
        frozen_q = QNetwork(obs_dim, action_dim, tuple(int(d) for d in args.hidden_dims))
        obs_mean = torch.zeros(obs_dim)
        obs_std = torch.ones(obs_dim)

    replay = None
    if not args.evaluate:
        replay = load_lerobot_replay_dataset(
            args.demo_data_path,
            expected_obs_dim=obs_dim,
            expected_action_dim=action_dim,
        )
        print(
            "Loaded offline replay dataset: "
            f"{len(replay)} transitions, {replay.num_episodes} episodes, "
            f"obs_dim={replay.obs_dim}, action_dim={replay.action_dim}, "
            f"sources={len(replay.source_paths)}"
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
            group="AWR",
            tags=["awr", "one_step_rl", "offline_rl", "state"],
        )

    agent = AWRAgent(
        args,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        obs_mean=obs_mean,
        obs_std=obs_std,
        frozen_q=frozen_q,
        run_name=run_name,
    ).to(device)

    global_step = 0
    if args.checkpoint is not None:
        global_step = agent.load(args.checkpoint, load_optimizers=not args.evaluate)

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
                f"v_loss={metrics['v_loss']:.4f}, "
                f"actor_loss={metrics['actor_loss']:.4f}, "
                f"adv_mean={metrics['adv_mean']:.4f}"
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

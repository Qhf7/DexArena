"""Standalone SARSA / Fitted-Q-Evaluation training script.

Pure offline policy evaluation of the demo's behavior policy: fits a Q-network
with the SARSA target ``r + gamma * Q(s', a')`` (a' from the transition) and
checks the result against the Monte-Carlo discounted return-to-go. No
environment is created and no policy is trained -- this validates that a stable,
numerically-correct Q can be fit on the demos before adding any policy
improvement.

Usage:
  python -m train.train_fqe \
      --demo_data_path <lerobot-demo-root> \
      --exp_name ballgrasp_fqe --track
"""

from pathlib import Path
import random
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import swanlab
import torch

from algo.offline_rl.fqe import FQEAgent
from train.train_configs.fqe_config import parse_args
from utils.offline_replay_buffer import load_lerobot_replay_dataset
from utils.paths import runs_path
from utils.train_utils import build_run_name, resolve_device, set_seed


@torch.no_grad()
def monte_carlo_check(agent, replay, args, device) -> dict[str, float]:
    """Compare predicted Q(s, a) against the demo discounted return-to-go."""
    batch = replay.sample(args.mc_eval_batch_size, device)
    q_pred = agent.predict_q(batch["observations"], batch["actions"]).flatten()
    mc = batch["returns_to_go"].flatten()
    mse = torch.mean((q_pred - mc) ** 2)
    q_centered = q_pred - q_pred.mean()
    mc_centered = mc - mc.mean()
    corr = (q_centered * mc_centered).sum() / (
        q_centered.norm() * mc_centered.norm() + 1e-8
    )
    return {
        "q_vs_mc_mse": float(mse.item()),
        "q_vs_mc_corr": float(corr.item()),
        "mc_return_mean": float(mc.mean().item()),
        "q_pred_mean": float(q_pred.mean().item()),
    }


if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args, "train_fqe")
    set_seed(args)
    device = resolve_device(args)

    replay = load_lerobot_replay_dataset(args.demo_data_path, gamma=args.gamma)
    obs_dim = replay.obs_dim
    action_dim = replay.action_dim
    obs_mean, obs_std = replay.observation_stats()
    print(
        "Loaded offline replay dataset: "
        f"{len(replay)} transitions, {replay.num_episodes} episodes, "
        f"obs_dim={obs_dim}, action_dim={action_dim}, "
        f"sources={len(replay.source_paths)}"
    )
    if replay.returns_to_go is None:
        raise RuntimeError("returns_to_go missing; loader must be called with a gamma.")

    if args.track:
        config = vars(args).copy()
        config["offline_transitions"] = len(replay)
        config["offline_episodes"] = replay.num_episodes
        swanlab.init(
            project=args.swanlab_project_name,
            entity=args.swanlab_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="FQE",
            tags=["fqe", "sarsa", "policy_evaluation", "offline_rl", "state"],
        )

    agent = FQEAgent(
        args,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_mean=obs_mean,
        obs_std=obs_std,
        run_name=run_name,
    ).to(device)

    agent.train()
    start_time = time.time()
    global_step = 0
    while global_step < args.total_updates:
        global_step += 1
        batch = replay.sample(args.batch_size, device)
        metrics = agent.update(batch, global_step)

        if global_step % args.log_freq == 0:
            for key, value in metrics.items():
                if np.isfinite(value):
                    agent.logger.add_scalar(f"losses/{key}", value, global_step)
            updates_per_second = global_step / max(time.time() - start_time, 1e-6)
            agent.logger.add_scalar(
                "charts/updates_per_second", updates_per_second, global_step
            )
            print(
                f"Update {global_step}/{args.total_updates}: "
                f"q_loss={metrics['q_loss']:.4f}, "
                f"q_data={metrics['q_data']:.3f}, "
                f"target_q={metrics['target_q']:.3f}"
            )

        if global_step % args.mc_eval_freq == 0 or global_step == args.total_updates:
            mc_metrics = monte_carlo_check(agent, replay, args, device)
            for key, value in mc_metrics.items():
                agent.logger.add_scalar(f"eval/{key}", value, global_step)
            print(
                f"  [MC check] corr={mc_metrics['q_vs_mc_corr']:.4f}, "
                f"mse={mc_metrics['q_vs_mc_mse']:.4f}, "
                f"q_mean={mc_metrics['q_pred_mean']:.3f}, "
                f"mc_mean={mc_metrics['mc_return_mean']:.3f}"
            )

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

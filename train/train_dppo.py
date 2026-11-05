from collections import defaultdict
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

from algo.dppo import DPPOAgent
from algo.diffusion import ObsHistory, make_identity_normalizer_stats
from train.train_configs.dppo_config import parse_finetune_args

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks  # noqa: F401
from utils.paths import runs_path


# ---------------------------------------------------------------------------------
# Setup helpers (mirrors train_ppo.py)
# ---------------------------------------------------------------------------------


def build_run_name(args) -> str:
    if args.exp_name is None:
        exp_name = os.path.basename(__file__)[: -len(".py")]
        return f"{args.env_id}__{exp_name}__{args.seed}__{int(time.time())}"
    return args.exp_name


def resolve_torch_device(args) -> torch.device:
    if not (torch.cuda.is_available() and args.cuda):
        return torch.device("cpu")
    return torch.device(f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda")


def _apply_gpu_id(backend, gpu_id, gpu_backends):
    if backend is None or gpu_id is None or ":" in backend or backend not in gpu_backends:
        return backend
    return f"{backend}:{gpu_id}"


def resolve_sim_backend(args) -> str:
    return _apply_gpu_id("physx_cuda", args.gpu_id, {"physx_cuda", "gpu", "cuda"})


def resolve_render_backend(args) -> str:
    return _apply_gpu_id("sapien_cuda", args.gpu_id, {"sapien_cuda", "gpu", "cuda"})


def make_env(args, *, num_envs, reconfiguration_freq, record_dir=None, max_steps_per_video=None):
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend=resolve_sim_backend(args),
        render_backend=resolve_render_backend(args),
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.reward_mode is not None:
        env_kwargs["reward_mode"] = args.reward_mode
    env_kwargs["max_episode_steps"] = args.env_max_episode_steps
    env = gym.make(args.env_id, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if record_dir is not None:
        env = RecordEpisode(
            env, output_dir=record_dir, save_trajectory=False,
            max_steps_per_video=max_steps_per_video,
            video_fps=int(getattr(env.unwrapped, "control_freq", 20)),
        )
    return env


def collect_rollout(agent, env, history, args, device, *, action_low, action_high, deterministic=False):
    N = args.num_envs
    To, Do, Da = args.n_obs_steps, agent.obs_dim, agent.action_dim
    K1 = agent.model.ft_denoising_steps + 1
    Kft = agent.model.ft_denoising_steps
    H = args.horizon

    obs_state = torch.zeros((args.num_steps, N, To, Do), device=device)
    chains = torch.zeros((args.num_steps, N, K1, H, Da), device=device)
    logprob = torch.zeros((args.num_steps, N, Kft, H, Da), device=device)
    values = torch.zeros((args.num_steps, N), device=device)
    rewards = torch.zeros((args.num_steps, N), device=device)
    dones = torch.zeros((args.num_steps, N), device=device)

    # Zero-delta "hold" action for finished envs (no-op under delta control modes).
    hold_action = torch.clamp(torch.zeros(N, Da, device=device), action_low, action_high)

    ep_metrics = defaultdict(list)  # episode stats harvested from auto-reset (return, success_once, ...)
    for step in range(args.num_steps):
        cond_raw = history.cond  # (N, To, Do)
        out = agent.act(cond_raw, deterministic=deterministic)

        obs_state[step] = cond_raw["state"]
        chains[step] = out["chains"]
        logprob[step] = out["logprob"]
        values[step] = out["value"]

        action_env = out["action_env"]  # (N, n_action_steps, Da)
        step_reward = torch.zeros(N, device=device)
        active = torch.ones(N, dtype=torch.bool, device=device)  # not yet done this chunk
        macro_done = torch.zeros(N, dtype=torch.bool, device=device)
        cur_obs = cond_raw["state"][:, -1]
        for a_i in range(args.n_action_steps):
            a = torch.clamp(action_env[:, a_i], action_low, action_high)
            # Finished envs hold position to minimize perturbation of the new episode.
            a = torch.where(active.unsqueeze(-1), a, hold_action)
            next_obs, r, term, trunc, infos = env.step(a)
            next_obs = next_obs.to(device)
            done = torch.logical_or(term, trunc).to(device)
            # Reward counts only steps the env actually executed (incl. the done step).
            step_reward = step_reward + r.to(device).float() * active.float() * args.reward_scale
            history.append(next_obs, mask=active & (~done))

            if "final_info" in infos:
                final_info = infos["final_info"]
                fmask = infos["_final_info"].to(device)
                if "episode" in final_info:
                    for k, v in final_info["episode"].items():
                        ep_metrics[k].extend(v[fmask].detach().float().cpu().tolist())

            macro_done = macro_done | (active & done)
            active = active & (~done)
            cur_obs = next_obs

        rewards[step] = step_reward
        dones[step] = macro_done.float()
        # Envs that finished this chunk have auto-reset; refresh their history from the
        # current obs so the next decision starts cleanly on the new episode.
        if macro_done.any():
            history.reset(cur_obs, env_mask=macro_done)

    with torch.no_grad():
        bootstrap_value = agent.get_value(history.cond)

    return {
        "obs_state": obs_state,
        "chains": chains,
        "logprob": logprob,
        "values": values,
        "rewards": rewards,
        "dones": dones,
        "bootstrap_value": bootstrap_value,
        "ep_metrics": ep_metrics,
    }


@torch.no_grad()
def evaluate_policy(agent, eval_env, eval_history, args, device, *, action_low, action_high):
    """Deterministic eval rollout. Returns a dict of mean episode metrics."""
    obs, _ = eval_env.reset(seed=args.seed)
    eval_history.reset(obs.to(device))
    metrics = defaultdict(list)
    num_decision_steps = max(1, args.num_eval_steps // args.n_action_steps)
    for _ in range(num_decision_steps):
        action_env = agent.sample_action_chunk(eval_history.cond, deterministic=True)
        for a_i in range(args.n_action_steps):
            a = torch.clamp(action_env[:, a_i], action_low, action_high)
            next_obs, _, term, trunc, infos = eval_env.step(a)
            next_obs = next_obs.to(device)
            done = torch.logical_or(term, trunc).to(device)
            eval_history.append(next_obs, mask=~done)
            if "final_info" in infos:
                fmask = infos["_final_info"].to(device)
                if "episode" in infos["final_info"]:
                    for k, v in infos["final_info"]["episode"].items():
                        metrics[k].extend(v[fmask].detach().float().cpu().tolist())
            if done.any():
                eval_history.reset(next_obs, env_mask=done)
    return {k: float(np.mean(v)) for k, v in metrics.items() if v}


def compute_gae(rollout, args, device):
    rewards = rollout["rewards"]
    values = rollout["values"]
    dones = rollout["dones"]
    bootstrap_value = rollout["bootstrap_value"]

    num_steps, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=device)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_nonterminal = 1.0 - dones[t]
            next_values = bootstrap_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def flatten_buffer(rollout, advantages, returns):
    def flat(x):
        return x.reshape(-1, *x.shape[2:])

    return {
        "obs_state": flat(rollout["obs_state"]),
        "chains": flat(rollout["chains"]),
        "logprob": flat(rollout["logprob"]),
        "values": flat(rollout["values"]).reshape(-1),
        "advantages": flat(advantages).reshape(-1),
        "returns": flat(returns).reshape(-1),
    }


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_finetune_args()

    # Derived sizes
    args.env_max_episode_steps = (
        args.max_episode_steps if args.max_episode_steps is not None else args.num_steps * args.n_action_steps
    )
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // (args.batch_size * args.n_action_steps)
    run_name = build_run_name(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = resolve_torch_device(args)

    record_dir = runs_path(run_name, "train_videos") if args.capture_video else None
    env = make_env(
        args, num_envs=args.num_envs, reconfiguration_freq=args.reconfiguration_freq,
        record_dir=record_dir, max_steps_per_video=args.env_max_episode_steps,
    )
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.array(env.single_observation_space.shape).prod())
    action_dim = int(np.prod(env.single_action_space.shape))
    action_low = torch.from_numpy(env.single_action_space.low).to(device)
    action_high = torch.from_numpy(env.single_action_space.high).to(device)
    print(f"Env {args.env_id}: obs_dim={obs_dim}, action_dim={action_dim}")

    # Normalizer: from BC checkpoint stats, demos, or identity (ball-grasp w/o demos).
    if args.base_policy_path is None:
        normalizer_stats = make_identity_normalizer_stats(obs_dim, action_dim)
        print("No base policy -> identity normalizer + random init.")
    else:
        # Stats will be overwritten by the checkpoint's normalizer in load_base_policy;
        # start from identity so buffers exist with the right shapes.
        normalizer_stats = make_identity_normalizer_stats(obs_dim, action_dim)

    agent = DPPOAgent(
        args, obs_dim=obs_dim, action_dim=action_dim,
        normalizer_stats=normalizer_stats, run_name=run_name, device=device,
    ).to(device)
    if args.base_policy_path is not None:
        agent.load_base_policy(args.base_policy_path)

    if args.track:
        config = vars(args).copy()
        config["obs_dim"] = obs_dim
        config["action_dim"] = action_dim
        swanlab.init(
            project=args.swanlab_project_name, entity=args.swanlab_entity,
            sync_tensorboard=False, config=config, name=run_name, save_code=True,
            group="DPPO", tags=["dppo", "finetune", "ppo", "diffusion"],
        )

    # Eval env (deterministic policy; episodes terminate on success/failure).
    eval_env = make_env(
        args, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq,
    )
    eval_env = ManiSkillVectorEnv(
        eval_env, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True
    )
    eval_history = ObsHistory(args.num_eval_envs, args.n_obs_steps, obs_dim, device)

    history = ObsHistory(args.num_envs, args.n_obs_steps, obs_dim, device)
    global_step = 0
    start_time = time.time()

    print(f"Running DPPO fine-tuning for {args.num_iterations} iterations")
    for iteration in range(1, args.num_iterations + 1):
        agent.eval()

        # Periodic evaluation (always runs on the first iteration).
        if iteration % args.eval_freq == 1:
            eval_metrics = evaluate_policy(
                agent, eval_env, eval_history, args, device,
                action_low=action_low, action_high=action_high,
            )
            for k, v in eval_metrics.items():
                agent.logger.add_scalar(f"eval/{k}", v, global_step)
            if eval_metrics:
                print(f"  eval @ iter {iteration}: " + ", ".join(f"{k}={v:.3f}" for k, v in eval_metrics.items()))
            if args.evaluate:
                break

        obs, _ = env.reset(seed=args.seed + iteration)
        history.reset(obs.to(device))

        rollout = collect_rollout(
            agent, env, history, args, device, action_low=action_low, action_high=action_high
        )
        global_step += args.batch_size * args.n_action_steps

        advantages, returns = compute_gae(rollout, args, device)
        buffer = flatten_buffer(rollout, advantages, returns)

        agent.train()
        metrics = agent.update(buffer)

        # Logging
        sps = int(global_step / (time.time() - start_time))
        agent.logger.add_scalar("charts/SPS", sps, global_step)
        agent.logger.add_scalar("charts/iteration", iteration, global_step)
        for key, value in metrics.items():
            agent.logger.add_scalar(key, value, global_step)
        ep_metrics = rollout["ep_metrics"]
        for k, v in ep_metrics.items():
            if v:
                agent.logger.add_scalar(f"train/{k}", float(np.mean(v)), global_step)
        agent.logger.add_scalar("charts/mean_reward", float(rollout["rewards"].mean().item()), global_step)
        ep_ret = float(np.mean(ep_metrics["return"])) if ep_metrics.get("return") else float("nan")
        succ = float(np.mean(ep_metrics["success_once"])) if ep_metrics.get("success_once") else float("nan")
        print(
            f"iter {iteration}/{args.num_iterations} step={global_step} SPS={sps} "
            f"pg={metrics.get('losses/pg_loss', float('nan')):.4f} "
            f"v={metrics.get('losses/v_loss', float('nan')):.4f} "
            f"ep_ret={ep_ret:.3f} succ={succ:.3f}"
        )

        if args.save_model and (iteration % args.save_freq == 0 or iteration == args.num_iterations):
            model_path = runs_path(run_name, f"ckpt_{iteration}.pt")
            agent.save(model_path, iteration=iteration, global_step=global_step)

        agent.model.step()  # anneal (no-op under default config)

    agent.logger.close()
    env.close()
    eval_env.close()

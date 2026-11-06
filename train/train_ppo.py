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
import torch

from train.train_configs.ppo_config import parse_args
import swanlab

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

try:
    # Optional GPU-IK patch, not shipped with the repository; only IK-based control modes
    # on the GPU simulation backend need it.
    from utils.maniskill_patches import patch_maniskill_gpu_ik

    patch_maniskill_gpu_ik()
except ModuleNotFoundError:
    pass

from algo.ppo import PPOAgent
import tasks  # noqa: F401
from utils.demo_dataset import (
    DemoInitManiSkillVectorEnv,
    DemoStateInitializer,
    format_dim_counts,
    load_demo_dataset,
)
from utils.paths import runs_path


def maybe_load_demo_dataset(
    *,
    data_path,
    expected_obs_dim,
    expected_act_dim,
    load_bc_data,
    load_reset_states,
    label,
):
    if not data_path or (not load_bc_data and not load_reset_states):
        return None

    dataset = load_demo_dataset(
        data_path,
        expected_obs_dim=expected_obs_dim,
        expected_act_dim=expected_act_dim,
        load_bc_data=load_bc_data,
        load_reset_states=load_reset_states,
    )
    if load_bc_data and dataset.bc_obs is not None and dataset.bc_actions is not None:
        print(
            f"Loaded BC dataset from {data_path}: "
            f"{dataset.num_bc_samples} samples across {dataset.matched_bc_trajectories} trajectories, "
            f"obs_dim={dataset.bc_obs.shape[1]}, act_dim={dataset.bc_actions.shape[1]}, "
            f"sources={len(dataset.matched_source_paths)}"
        )
    if load_reset_states and dataset.reset_obs is not None:
        print(
            f"Loaded {label} reset states from {data_path}: "
            f"{dataset.num_reset_samples} states across {dataset.matched_reset_trajectories} trajectories, "
            f"obs_dim={dataset.reset_obs.shape[1]}, sources={len(dataset.matched_source_paths)}"
        )
    if dataset.mismatched_dim_counts:
        print(
            "Skipped demo trajectories with non-matching obs/action dims while loading for "
            f"env obs/action={expected_obs_dim}/{expected_act_dim}: "
            f"{format_dim_counts(dataset.mismatched_dim_counts)}"
        )
    if load_reset_states and dataset.missing_env_states_trajectories:
        print(
            "Skipped reset-state loading for "
            f"{dataset.missing_env_states_trajectories} trajectories without `env_states`."
        )
    if load_reset_states and dataset.mismatched_reset_state_keys:
        print(
            "Skipped reset-state loading for "
            f"{dataset.mismatched_reset_state_keys} trajectories with mismatched env-state keys."
        )
    return dataset


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
    return maybe_apply_gpu_id("physx_cuda", args.gpu_id, {"physx_cuda", "gpu", "cuda"})


def resolve_render_backend(args) -> str:
    return maybe_apply_gpu_id("sapien_cuda", args.gpu_id, {"sapien_cuda", "gpu", "cuda"})


if __name__ == "__main__":
    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    if args.evaluate:
        run_name = "eval_" + run_name


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = resolve_torch_device(args)

    # env setup
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend=resolve_sim_backend(args),
        render_backend=resolve_render_backend(args),
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.reward_mode is not None:
        env_kwargs["reward_mode"] = args.reward_mode
    assert args.num_steps == args.num_eval_steps
    env_kwargs["max_episode_steps"] = args.num_steps
    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        video_fps = int(envs.unwrapped.control_freq)
        eval_output_dir = runs_path(run_name, "videos")
        if args.evaluate:
            eval_output_dir = str(Path(args.checkpoint).parent / "test_videos")
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=runs_path(run_name, "train_videos"),
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=video_fps,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=video_fps,
        )
    train_vector_env_cls = DemoInitManiSkillVectorEnv if args.use_demo_init_states else ManiSkillVectorEnv
    envs = train_vector_env_cls(
        envs,
        args.num_envs,
        ignore_terminations=not args.partial_reset,
        record_metrics=True,
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    expected_obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    expected_act_dim = int(np.prod(envs.single_action_space.shape))

    bc_dataset = None
    use_bc_dataset = bool(args.use_bc_loss and args.bc_coef > 0 and args.bc_data_path)
    demo_init_path = args.demo_init_data_path or args.bc_data_path
    use_demo_init_dataset = bool(args.use_demo_init_states and demo_init_path)
    if use_bc_dataset or use_demo_init_dataset:
        if use_bc_dataset and args.bc_data_path == demo_init_path and use_demo_init_dataset:
            shared_dataset = maybe_load_demo_dataset(
                data_path=args.bc_data_path,
                expected_obs_dim=expected_obs_dim,
                expected_act_dim=expected_act_dim,
                load_bc_data=True,
                load_reset_states=True,
                label="demo-init",
            )
            bc_dataset = shared_dataset
            demo_init_dataset = shared_dataset
        else:
            if use_bc_dataset:
                bc_dataset = maybe_load_demo_dataset(
                    data_path=args.bc_data_path,
                    expected_obs_dim=expected_obs_dim,
                    expected_act_dim=expected_act_dim,
                    load_bc_data=True,
                    load_reset_states=False,
                    label="bc",
                )
            else:
                bc_dataset = None
            if use_demo_init_dataset:
                demo_init_dataset = maybe_load_demo_dataset(
                    data_path=demo_init_path,
                    expected_obs_dim=expected_obs_dim,
                    expected_act_dim=expected_act_dim,
                    load_bc_data=False,
                    load_reset_states=True,
                    label="demo-init",
                )
            else:
                demo_init_dataset = None
    else:
        demo_init_dataset = None

    if args.use_demo_init_states:
        if demo_init_dataset is None or demo_init_dataset.reset_env_states is None:
            raise RuntimeError("Demo-state initialization is enabled but no reset-state dataset was loaded.")
        # articulation state keys are robot-uid-based; demos recorded on another plant
        # variant of the same robot (e.g. panda -> panda_smooth) need a key remap
        if args.robot_uids is not None:
            arts = demo_init_dataset.reset_env_states.get("articulations")
            if arts is not None and args.robot_uids not in arts and len(arts) == 1:
                old_key = next(iter(arts))
                arts[args.robot_uids] = arts.pop(old_key)
                print(f"Remapped demo-init articulation state key '{old_key}' -> '{args.robot_uids}'")
        demo_init_decay_steps = max(1, int(args.total_timesteps * float(args.demo_init_decay_steps)))
        envs.demo_state_initializer = DemoStateInitializer(
            reset_env_states=demo_init_dataset.reset_env_states,
            reset_obs=demo_init_dataset.reset_obs,
            demo_progress=demo_init_dataset.demo_progress,
            seed=args.seed,
            init_prob=args.demo_init_prob,
            min_prob=args.demo_init_min_prob,
            decay=args.demo_init_decay,
            decay_steps=demo_init_decay_steps,
            curriculum_mode=args.demo_init_curriculum,
            curriculum_start_frac=args.demo_init_curriculum_start_frac,
            curriculum_steps=max(1, int(args.total_timesteps * float(args.demo_init_curriculum_steps))),
        )
        envs.demo_state_initializer.set_train_progress(global_step=0)

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    if not args.evaluate:
        print("Running training")
    else:
        print("Running evaluation")
    if args.track:
        config = vars(args)
        # env_kwargs may already carry reward_mode (when --reward-mode is set);
        # log the effective mode without duplicating the keyword.
        log_env_kwargs = {k: v for k, v in env_kwargs.items() if k != "reward_mode"}
        effective_reward_mode = env_kwargs.get("reward_mode", "normalized_dense")
        config["env_cfg"] = dict(
            **log_env_kwargs,
            num_envs=args.num_envs,
            env_id=args.env_id,
            reward_mode=effective_reward_mode,
            env_horizon=max_episode_steps,
            partial_reset=args.partial_reset,
        )
        config["eval_env_cfg"] = dict(
            **log_env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            reward_mode=effective_reward_mode,
            env_horizon=max_episode_steps,
            partial_reset=False,
        )
        swanlab.init(
            project=args.swanlab_project_name,
            entity=args.swanlab_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="PPO",
            tags=["ppo", "walltime_efficient"],
        )

    agent = PPOAgent(args, envs, run_name, bc_dataset=bc_dataset).to(device)
    if not args.evaluate and args.bc_warmup_steps > 0:
        agent.bc_pretrain(args.bc_warmup_steps)
        if args.save_model:
            agent.save(runs_path(run_name, "ckpt_bc_pretrain.pt"))
    if args.use_demo_init_states:
        agent.logger.add_scalar(
            "charts/demo_init_prob",
            envs.demo_state_initializer.get_current_prob(),
            0,
        )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print("####")
    print(
        f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}"
    )
    print(
        f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}"
    )
    print("####")
    action_space_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(envs.single_action_space.high).to(device)
    initial_ent_coef = args.ent_coef

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_action = agent.actor_critic.get_action(eval_obs, deterministic=True)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                agent.logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = runs_path(run_name, f"ckpt_{iteration}.pt")
            agent.save(model_path)

        if args.anneal_lr or args.anneal_ent_coef:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations

        # Annealing the learning rate if instructed to do so.
        if args.anneal_lr:
            lrnow = frac * args.learning_rate
            agent.optimizer.param_groups[0]["lr"] = lrnow

        # Annealing entropy regularization coefficient if instructed to do so.
        if args.anneal_ent_coef:
            args.ent_coef = frac * initial_ent_coef

        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.actor_critic.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            # TRY NOT TO MODIFY: execute the game and log data.
            if args.use_demo_init_states:
                envs.demo_state_initializer.set_train_progress(global_step)
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    agent.logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                reward_terms = final_info.get("reward_terms")
                if reward_terms is not None:
                    for name, term_values in reward_terms.items():
                        if torch.is_tensor(term_values):
                            agent.logger.add_scalar(
                                f"train/rewards/{name}",
                                term_values[done_mask].detach().float().mean().item(),
                                global_step,
                            )
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = \
                        agent.actor_critic.get_value(infos["final_observation"][done_mask]).view(-1)
        rollout_time = time.time() - rollout_time
        if args.use_demo_init_states:
            agent.logger.add_scalar(
                "charts/demo_init_prob",
                envs.demo_state_initializer.get_current_prob(),
                global_step,
            )
            agent.logger.add_scalar(
                "charts/demo_init_curriculum_min_progress",
                envs.demo_state_initializer.get_current_curriculum_min_progress(),
                global_step,
            )
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.actor_critic.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.update(start_time, rollout_time, global_step, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)

    if not args.evaluate:
        if args.save_model:
            model_path = runs_path(run_name, "final_ckpt.pt")
            agent.save(model_path)
        agent.logger.close()
    envs.close()
    eval_envs.close()

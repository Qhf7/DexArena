"""Online PPO from pixels, anchored on demonstrations (see `algo/ppo_pixel.py`).

Written for `CabinetPouringOpenCabinet-Sharpa-v1`: a sparse {0, 1} reward that terminates the
episode on the first success, two 128x128 cameras (`cabinet_fixed`) plus
proprioception, and absolute `pd_joint_pos` control. Under that reward, PPO from
a random policy would never see a non-zero return, so the 62 human demos enter
the loop twice:

* **BC anchor** -- a behaviour-cloning term on the actor, from the *rendered*
  demo HDF5 (images + proprio + actions), plus an optional pure-BC warmup before
  the first rollout.
* **demo-state resets** -- a fraction of episodes start from a physical state
  restored out of the *LeRobot* demos (`observation.environment_state`), sampled
  by a reverse curriculum that begins near the end of each demo and walks back to
  its start. Restoring a state re-renders the cameras, so no stored frames are
  needed for this half, which is why the two paths read different datasets.

Both schedules decay, so training ends on the task's own initial distribution
with the anchor reduced to `bc_min_coef`.

What differs from `train_ppo.py` (state PPO), beyond the observation:

* **Only time-limit truncations bootstrap.** ManiSkill sets
  `terminated = info["success"]`, and success here is absorbing and worth exactly
  1, so adding `gamma * V(s_final)` at a terminated step would inflate the return
  past the reward the task can ever pay. The state script bootstraps at every
  done; this one masks that to `truncated & ~terminated`.
* **No reward scale or bias.** The return is `gamma^t_success` or 0, so values
  live in [0, 1] and the value loss cannot swamp the policy gradient.
* **The rollout keeps images as uint8** on the training device:
  `num_envs * num_steps * 2 * 128 * 128 * 3` bytes (64 x 200 -> 2.4 GiB).

Example (the tuned door-open configuration is the PPO-from-pixels command in
`train/TRAIN_EVAL.md`)::

    python train/train_ppo_pixels.py \\
        --env-id CabinetPouringOpenCabinet-Sharpa-v1 --camera-preset cabinet_fixed \\
        --bc-data-path <...>/pixels/door_open_demos_62_128.h5 \\
        --demo-init-data-path <...>/cabinet_pour_door_open_for_bc \\
        --exp-name cabinet_pour_door_open_ppo_pixels
"""

from collections import deque
import json
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
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks  # noqa: F401  (registers every RoboDex env)
from algo.ppo_pixel import PPOPixelAgent, index_obs
from train.train_configs.ppo_pixels_config import (
    encoder_config_from_args,
    parse_args,
    pixel_config_from_args,
)
from utils.demo_dataset import (
    DemoInitManiSkillVectorEnv,
    DemoStateInitializer,
    load_demo_dataset,
    resolve_lerobot_dataset_roots,
)
from utils.image_replay_buffer import observation_grid
from utils.paths import runs_path
from utils.pixel_dataset import OfflineImageDataset
from utils.pixel_env import SuccessAsSparseReward, make_pixel_eval_env, make_pixel_train_env
from utils.pixel_observation import (
    assert_obs_contract,
    extract_pixel_obs,
    report_env_obs_keys,
)
from utils.train_utils import (
    build_run_name,
    evaluate_policy,
    log_train_episode_metrics,
    resolve_device,
    set_seed,
)


def lerobot_feature_dims(data_path: str) -> tuple[int, int]:
    """(observation.state dim, action dim) of a LeRobot demo dataset.

    ``load_demo_dataset`` filters trajectories whose dims do not match what the
    *env* reports, which works for the state agents. Here the env's observation
    is a dict of images, so there is no state dim to hand it -- read the one the
    dataset was recorded with instead of hard-coding a number that would silently
    drop every episode if the recording changed.
    """
    roots = resolve_lerobot_dataset_roots(data_path)
    if not roots:
        raise RuntimeError(
            f"No LeRobot dataset found under {data_path!r}. Demo-state resets need the "
            "LeRobot demos (they carry `observation.environment_state`), not the "
            "rendered image HDF5."
        )
    info = json.loads((Path(roots[0]) / "meta" / "info.json").read_text())
    features = info["features"]
    return (
        int(features["observation.state"]["shape"][0]),
        int(features["action"]["shape"][0]),
    )


def drop_post_success_states(dataset, data_path: str) -> None:
    """Remove the demo states recorded AFTER the episode's first success.

    Every DexArena recording runs ~20 frames past success. Restoring one gives a
    1-step episode that pays reward 1 whatever the action -- no learning signal
    -- and those frames are exactly where the reverse curriculum starts. They
    are also the only frames recorded in a post-success REPRESENTATION:
    CabinetPourBottlePickPlace's tail is in bottle rep B, and restoring it flips
    render visibility, which the GPU batched renderer refuses ("Modifying a
    scene ... not allowed", fleet run 2026-09-20).

    `next.success[k]` describes the state at frame k+1, so frame j is dropped
    when any next.success before j is set. Progress is re-spread over what is
    left, so `--demo-init-curriculum-start-frac 0.2` still means "the last 20%
    of the way TO success".
    """
    import pyarrow.parquet as pq

    roots = resolve_lerobot_dataset_roots(data_path)
    if len(roots) != 1:
        raise RuntimeError("--demo-init-drop-post-success expects exactly one LeRobot root.")
    table = pq.read_table(
        sorted((Path(roots[0]) / "data").rglob("*.parquet")),
        columns=["episode_index", "frame_index", "next.success"],
    ).to_pydict()
    episode = np.asarray(table["episode_index"], dtype=np.int64)
    frame = np.asarray(table["frame_index"], dtype=np.int64)
    success = np.asarray(table["next.success"], dtype=np.float32).reshape(-1) > 0.5
    order = np.lexsort((frame, episode))          # the loader's own ordering
    episode, success = episode[order], success[order]
    if episode.shape[0] != int(dataset.num_reset_samples):
        raise RuntimeError(
            f"{episode.shape[0]} recorded frames vs {dataset.num_reset_samples} loaded reset "
            "states: the loader skipped episodes, so the success column cannot be aligned."
        )
    keep = np.ones(episode.shape[0], dtype=bool)
    progress = np.zeros(episode.shape[0], dtype=np.float32)
    for e in np.unique(episode):
        idx = np.nonzero(episode == e)[0]
        seen = np.concatenate([[False], np.cumsum(success[idx])[:-1] > 0])
        keep[idx] = ~seen
        kept = idx[~seen]
        progress[kept] = np.linspace(0.0, 1.0, num=len(kept), endpoint=True, dtype=np.float32)
    keep_t = torch.from_numpy(keep)
    for items in dataset.reset_env_states.values():
        for name in list(items):
            items[name] = items[name][keep_t]
    if dataset.reset_obs is not None:
        dataset.reset_obs = dataset.reset_obs[keep_t]
    dataset.demo_progress = torch.from_numpy(progress[keep])
    print(
        f"demo-init: dropped {int((~keep).sum())} post-success states "
        f"({(~keep).mean():.1%}); {int(keep.sum())} remain"
    )


def build_demo_state_initializer(args, action_dim: int):
    """Load the demo env states and wrap them in the reverse-curriculum sampler."""
    state_dim, demo_action_dim = lerobot_feature_dims(args.demo_init_data_path)
    if demo_action_dim != action_dim:
        raise RuntimeError(
            f"Demo action dim {demo_action_dim} != env action dim {action_dim}: the "
            "demos were recorded on a different robot or controller."
        )
    dataset = load_demo_dataset(
        args.demo_init_data_path,
        expected_obs_dim=state_dim,
        expected_act_dim=action_dim,
        load_bc_data=False,
        load_reset_states=True,
    )
    if dataset.reset_env_states is None:
        raise RuntimeError(
            f"No env states loaded from {args.demo_init_data_path}; demo-state resets "
            "need `observation.environment_state` in the recording."
        )
    # Articulation state keys are robot-uid-based. Demos recorded on another
    # variant of the same robot need the key remapped or `set_state_dict` sees a
    # name it does not own.
    if args.robot_uids is not None:
        articulations = dataset.reset_env_states.get("articulations")
        if (
            articulations is not None
            and args.robot_uids not in articulations
            and len(articulations) == 1
        ):
            old_key = next(iter(articulations))
            articulations[args.robot_uids] = articulations.pop(old_key)
            print(f"Remapped demo-init articulation state key {old_key!r} -> {args.robot_uids!r}")

    if args.demo_init_drop_post_success:
        drop_post_success_states(dataset, args.demo_init_data_path)

    print(
        f"Loaded demo-init states from {args.demo_init_data_path}: "
        f"{dataset.num_reset_samples} states across {dataset.matched_reset_trajectories} "
        f"trajectories (state_dim={state_dim})"
    )
    if dataset.missing_env_states_trajectories:
        print(
            f"Skipped {dataset.missing_env_states_trajectories} trajectories without `env_states`."
        )
    total = int(args.total_timesteps)
    return DemoStateInitializer(
        reset_env_states=dataset.reset_env_states,
        # The pixels agent never consumes a stored observation: restoring the
        # state re-renders the cameras, and the recorded 79-dim state vector is
        # not the observation this policy sees anyway.
        reset_obs=None,
        demo_progress=dataset.demo_progress,
        seed=args.seed,
        init_prob=args.demo_init_prob,
        min_prob=args.demo_init_min_prob,
        decay=args.demo_init_decay,
        decay_steps=max(1, int(total * float(args.demo_init_decay_steps))),
        curriculum_mode=args.demo_init_curriculum,
        curriculum_start_frac=args.demo_init_curriculum_start_frac,
        curriculum_steps=max(1, int(total * float(args.demo_init_curriculum_steps))),
    )


def allocate_rollout_storage(args, pixel_config, proprio_dim, action_dim, device):
    """(T, N, ...) storage. Frames stay uint8 -- float32 would be 4x this."""
    height, width, channels = pixel_config.image_shape
    obs = {
        key: torch.zeros(
            (args.num_steps, args.num_envs, height, width, channels),
            dtype=torch.uint8,
            device=device,
        )
        for key in pixel_config.camera_keys
    }
    obs["proprio"] = torch.zeros(
        (args.num_steps, args.num_envs, proprio_dim), dtype=torch.float32, device=device
    )
    image_bytes = sum(t.numel() for k, t in obs.items() if k != "proprio")
    print(f"rollout image storage: {image_bytes / 2**30:.2f} GiB ({args.num_steps} x {args.num_envs})")
    return obs


if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args, "train_ppo_pixels")
    set_seed(args)
    device = resolve_device(args)

    pixel_config = pixel_config_from_args(args)
    encoder_config = encoder_config_from_args(args)
    print(pixel_config.summary())

    # Reward := success on both envs, so a return is 0 or 1 by construction and
    # train/eval returns mean the same thing (see SuccessAsSparseReward).
    eval_envs, env_kwargs = make_pixel_eval_env(
        args, pixel_config, run_name, env_wrappers=(SuccessAsSparseReward,)
    )
    if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
        raise TypeError("PPO supports only continuous Box action spaces.")
    action_dim = int(np.prod(eval_envs.single_action_space.shape))
    action_low = torch.from_numpy(eval_envs.single_action_space.low).float()
    action_high = torch.from_numpy(eval_envs.single_action_space.high).float()
    control_mode = str(eval_envs.unwrapped.control_mode)

    # Print what the rgb env actually returns, then assert the assembled
    # observation against the contract, instead of assuming either.
    raw_obs, _ = eval_envs.reset(seed=args.seed)
    print(report_env_obs_keys(raw_obs, pixel_config))
    probe_obs = extract_pixel_obs(raw_obs, pixel_config)
    assert_obs_contract(probe_obs, pixel_config)
    proprio_dim = int(probe_obs["proprio"].shape[1])
    print(
        f"observation contract OK: cameras={list(pixel_config.camera_keys)} "
        f"{pixel_config.image_shape}, proprio_dim={proprio_dim}, action_dim={action_dim}, "
        f"control_mode={control_mode}"
    )

    bc_dataset = None
    if args.use_bc_loss:
        bc_device = device if args.bc_data_device != "cpu" else torch.device("cpu")
        if args.bc_data_device == "auto" and device.type != "cuda":
            bc_device = torch.device("cpu")
        bc_dataset = OfflineImageDataset(
            args.bc_data_path,
            pixel_config,
            expected_action_dim=action_dim,
            expected_proprio_dim=proprio_dim,
            expected_control_mode=control_mode,
            success_only=args.bc_success_only,
            device=bc_device,
        )
        print(f"{bc_dataset.summary()} (held on {bc_device})")

    demo_state_initializer = (
        build_demo_state_initializer(args, action_dim) if args.use_demo_init_states else None
    )

    if args.track:
        config = dict(sorted(vars(args).items()))
        config["env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_envs,
            env_id=args.env_id,
        )
        config["camera_config_hash"] = pixel_config.config_hash()
        swanlab.init(
            project=args.swanlab_project_name,
            entity=args.swanlab_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="PPO",
            tags=["ppo", "pixels", args.encoder_arch, args.env_id],
        )

    # Span of the tanh on the actor's mean. Narrowing it to the region the demos
    # actually command is the cheapest way to shrink the policy's step size in
    # action space (see `algo/ppo_pixel.py`, rule 3); the margin keeps a joint
    # the demos barely move from being pinned to a point.
    mean_low = mean_high = None
    if args.action_scale_source == "demos":
        demo_actions = bc_dataset.actions.float()
        low, high = demo_actions.min(dim=0).values, demo_actions.max(dim=0).values
        pad = float(args.action_scale_margin) * (high - low).clamp_min(1e-3)
        mean_low, mean_high = (low - pad).cpu(), (high + pad).cpu()

    agent = PPOPixelAgent(
        args,
        pixel_config=pixel_config,
        encoder_config=encoder_config,
        proprio_dim=proprio_dim,
        action_low=action_low,
        action_high=action_high,
        run_name=run_name,
        bc_dataset=bc_dataset,
        mean_low=mean_low,
        mean_high=mean_high,
    ).to(device)

    global_step = 0
    if args.checkpoint is not None:
        global_step = agent.load(args.checkpoint, load_optimizer=not args.evaluate)
        print(f"resumed from {args.checkpoint} at env step {global_step}")

    obs_projector = lambda observation: extract_pixel_obs(observation, pixel_config)  # noqa: E731

    if args.evaluate:
        eval_projector = obs_projector
        if args.eval_first_frame_path:
            # What the policy sees on the first eval step, cameras side by side: for a
            # `--scene-background` run, the evidence that the new scene is in view.
            saved = []

            def eval_projector(observation):
                projected = obs_projector(observation)
                if not saved:
                    from pathlib import Path as _Path

                    from PIL import Image

                    frame = torch.cat(
                        [projected[key][0] for key in pixel_config.camera_keys], dim=1
                    )
                    _Path(args.eval_first_frame_path).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame.cpu().numpy()).save(args.eval_first_frame_path)
                    saved.append(True)
                return projected

        metrics = evaluate_policy(agent, eval_envs, args, device, obs_projector=eval_projector)
        print(f"Evaluated {int(metrics.pop('episodes', 0.0))} episodes")
        for key, value in metrics.items():
            print(f"eval_{key}_mean={value}")
            agent.logger.add_scalar(f"eval/{key}", value, global_step)
        agent.logger.close()
        eval_envs.close()
        raise SystemExit(0)

    if args.checkpoint is None and bc_dataset is not None:
        if args.init_action_bias_from_demos:
            agent.actor_critic.initialize_action_bias(bc_dataset.actions.float().mean(dim=0))
            print("actor output bias seeded from the demos' mean action")
        agent.bc_pretrain(args.bc_warmup_steps)
        if args.save_model and args.bc_warmup_steps > 0:
            agent.save(runs_path(run_name, "ckpt_bc_pretrain.pt"), 0)
            metrics = evaluate_policy(agent, eval_envs, args, device, obs_projector=obs_projector)
            print(f"BC warmup eval over {int(metrics.pop('episodes', 0.0))} episodes")
            for key, value in metrics.items():
                print(f"bc_warmup_eval_{key}_mean={value}")
                agent.logger.add_scalar(f"bc_warmup_eval/{key}", value, 0)

    train_envs, num_envs, _ = make_pixel_train_env(
        args,
        pixel_config,
        vector_env_cls=(
            DemoInitManiSkillVectorEnv if demo_state_initializer is not None else ManiSkillVectorEnv
        ),
        env_wrappers=(SuccessAsSparseReward,),
    )
    # The demo-init schedules run off this clock, not off `global_step`: it only
    # advances while the policy is finishing episodes from where the curriculum
    # already starts them (see `demo_init_advance_success`).
    curriculum_step = global_step
    recent_successes: deque[float] = deque(maxlen=int(args.demo_init_success_window))
    if demo_state_initializer is not None:
        train_envs.demo_state_initializer = demo_state_initializer
        demo_state_initializer.set_train_progress(global_step=curriculum_step)

    obs_storage = allocate_rollout_storage(args, pixel_config, proprio_dim, action_dim, device)
    actions = torch.zeros((args.num_steps, num_envs, action_dim), device=device)
    logprobs = torch.zeros((args.num_steps, num_envs), device=device)
    rewards = torch.zeros((args.num_steps, num_envs), device=device)
    dones = torch.zeros((args.num_steps, num_envs), device=device)
    values = torch.zeros((args.num_steps, num_envs), device=device)

    raw_obs, _ = train_envs.reset(seed=args.seed)
    next_obs = obs_projector(raw_obs)
    next_done = torch.zeros(num_envs, device=device)
    start_time = time.time()
    next_log = global_step + args.log_freq
    next_image_log = global_step + args.log_image_freq
    next_eval = global_step + args.eval_freq
    next_save = global_step + args.save_freq
    initial_ent_coef = args.ent_coef
    best_eval_score = (-1.0, -1.0)
    metrics: dict[str, float] = {}

    print("####")
    print(
        f"num_iterations={args.num_iterations} num_envs={num_envs} "
        f"batch_size={args.batch_size} minibatch_size={args.minibatch_size} "
        f"update_epochs={args.update_epochs}"
    )
    print("####")

    # A resumed run continues its iteration count, so lr annealing picks up where
    # it stopped instead of jumping back to the initial rate.
    first_iteration = 1 + global_step // args.batch_size
    for iteration in range(first_iteration, args.num_iterations + 1):
        if args.anneal_lr or args.anneal_ent_coef:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            if args.anneal_lr:
                # Per group: the actor and the critic run at different rates and
                # annealing only `param_groups[0]` would silently reset one of
                # them to the other's base value.
                for group in agent.optimizer.param_groups:
                    base = (
                        agent.actor_learning_rate
                        if group.get("name") == "actor"
                        else args.learning_rate
                    )
                    group["lr"] = frac * base
            if args.anneal_ent_coef:
                args.ent_coef = frac * initial_ent_coef

        # ---- rollout ---------------------------------------------------------
        agent.eval()
        final_values = torch.zeros((args.num_steps, num_envs), device=device)
        rollout_time = time.time()
        for step in range(args.num_steps):
            global_step += num_envs
            for key, storage in obs_storage.items():
                storage[step] = next_obs[key]
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.actor_critic.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            if demo_state_initializer is not None:
                # Read before `step`, because the auto-reset inside it is where a
                # demo state gets sampled.
                demo_state_initializer.set_train_progress(curriculum_step)
            next_raw_obs, reward, terminations, truncations, infos = train_envs.step(
                agent.clip_action(action)
            )
            rewards[step] = reward.view(-1)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)

            # Bootstrap only where the time limit cut the episode. A terminated
            # step is a real terminal state (success), worth exactly its reward.
            timeout_mask = torch.logical_and(truncations, ~terminations)
            if "final_observation" in infos and bool(timeout_mask.any()):
                final_obs = extract_pixel_obs(infos["final_observation"], pixel_config)
                with torch.no_grad():
                    final_values[step, timeout_mask] = (
                        agent.actor_critic.get_value(index_obs(final_obs, timeout_mask)).view(-1)
                    )

            episode_stats = log_train_episode_metrics(agent, infos, global_step)
            if episode_stats is not None and episode_stats["num_episodes"] > 0:
                recent_successes.extend(
                    [episode_stats["success"]] * episode_stats["num_episodes"]
                )
            next_obs = obs_projector(next_raw_obs)
        rollout_time = time.time() - rollout_time

        # ---- GAE -------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.actor_critic.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # `final_values[t]` is non-zero only on time-limit steps, where
                # `next_not_done` is 0 and the stored `nextvalues` belongs to the
                # already-reset episode.
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                )
            returns = advantages + values

        b_obs = {key: value.reshape((-1,) + value.shape[2:]) for key, value in obs_storage.items()}
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        metrics = agent.update(
            start_time,
            rollout_time,
            global_step,
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
        )

        # Only now. The whole rollout was collected, GAE'd and updated under one
        # fixed normalization; folding this iteration's proprio in before the
        # update would make `pi_old` and `pi_new` two different functions of the
        # same state, which is what wrecked run 5744 (ratio_err 5-20, every
        # update killed by target_kl on its first minibatch).
        agent.update_proprio_stats(obs_storage["proprio"])

        train_success_rate = (
            float(np.mean(recent_successes)) if recent_successes else float("nan")
        )
        agent.logger.add_scalar("charts/train_success_rate", train_success_rate, global_step)
        if demo_state_initializer is not None:
            # The curriculum clock advances only when the policy is already
            # succeeding from the window it is being given. `nan` (no episode
            # has finished yet) does not advance it.
            if (
                args.demo_init_advance_success <= 0.0
                or (
                    np.isfinite(train_success_rate)
                    and train_success_rate >= args.demo_init_advance_success
                )
            ):
                # Success-adaptive pace. The clock only ever advanced one batch
                # per iteration, so a task the policy already solves from the
                # current window (drawer_plug_open sat at succ 1.0) spent its
                # budget waiting for the schedule. The 0.5 gate below still
                # stops it the moment the policy falls behind.
                pace = (
                    args.demo_init_fast_advance_mult
                    if np.isfinite(train_success_rate)
                    and train_success_rate >= args.demo_init_fast_advance_success
                    else 1.0
                )
                curriculum_step += int(args.batch_size * pace)
            elif args.demo_init_stall_pace > 0.0:
                # Below the gate: slow down, do not stop. With the post-success
                # states dropped the success rate is honest, and door_open /
                # cap_remove_place sat at 0.10 from step 100k -- a hard stop there
                # also freezes `demo_init_prob` at ~1, so the policy never gets
                # to practise from the real initial state that eval measures.
                curriculum_step += int(args.batch_size * args.demo_init_stall_pace)
            agent.logger.add_scalar("charts/curriculum_step", curriculum_step, global_step)
            agent.logger.add_scalar(
                "charts/demo_init_prob", demo_state_initializer.get_current_prob(), global_step
            )
            agent.logger.add_scalar(
                "charts/demo_init_curriculum_min_progress",
                demo_state_initializer.get_current_curriculum_min_progress(),
                global_step,
            )

        if global_step >= next_log:
            print(
                f"iter {iteration}/{args.num_iterations} step {global_step}/{args.total_timesteps}: "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f} "
                f"bc={metrics['bc_coef']:.3f}x{metrics['bc_loss']:.5f} "
                f"kl={metrics['approx_kl']:.4f} sigma={metrics['policy_std']:.4f} "
                # steps: how many of the configured minibatches the policy
                # actually got. `0/N` means the KL budget was spent before the
                # first one and this iteration taught the actor nothing.
                f"steps={metrics['policy_steps']:.0f}/{args.num_minibatches * args.update_epochs} "
                f"ratio_err={metrics['ratio_err']:.2e} "
                f"lp_err={metrics['rollout_logprob_err']:.2e} "
                f"succ={train_success_rate:.3f} {metrics['sps']:.0f} steps/s"
            )
            next_log += args.log_freq

        if global_step >= next_image_log and agent._last_augmented is not None:
            grid = observation_grid(
                {"observations": agent._last_augmented}, pixel_config.camera_keys
            )
            if grid is not None:
                agent.logger.add_image("obs/augmented_batch", grid.numpy(), global_step)
            next_image_log += args.log_image_freq

        if global_step >= next_eval:
            eval_metrics = evaluate_policy(
                agent, eval_envs, args, device, obs_projector=obs_projector
            )
            print(f"Evaluated {int(eval_metrics.pop('episodes', 0.0))} episodes")
            for key, value in eval_metrics.items():
                print(f"eval_{key}_mean={value}")
                agent.logger.add_scalar(f"eval/{key}", value, global_step)
            agent.train()
            next_eval += args.eval_freq
            # Keep the best evaluated policy. Ranked on success_once, then
            # success_at_end; a tie goes to the LATER checkpoint, which has seen
            # more of the curriculum. One 50-episode eval is only good to about
            # +-0.12 (measured on ScrewingTurn), so "best of N evals" is biased
            # upward -- the number to report is the seeded re-evaluation of this
            # file, not `best.json`.
            score = (
                float(eval_metrics.get("success_once", 0.0)),
                float(eval_metrics.get("success_at_end", 0.0)),
            )
            if args.save_model and score >= best_eval_score:
                best_eval_score = score
                agent.save(runs_path(run_name, "ckpt_best.pt"), global_step)
                Path(runs_path(run_name, "best.json")).write_text(
                    json.dumps(
                        dict(
                            global_step=int(global_step),
                            success_once=score[0],
                            success_at_end=score[1],
                        ),
                        indent=2,
                    )
                )
                print(f"best checkpoint so far: step {global_step} success_once={score[0]:.3f}")

        if args.save_model and global_step >= next_save:
            agent.save(runs_path(run_name, f"ckpt_{global_step}.pt"), global_step)
            next_save += args.save_freq

        if args.stop_at_step is not None and global_step >= args.stop_at_step:
            # Stop early WITHOUT touching the schedules: lr annealing, the BC
            # decay and both demo-init schedules are fractions of
            # `total_timesteps`, so shrinking that instead would be a different
            # (5x faster) schedule, not a shorter run of the measured one.
            print(f"stop_at_step={args.stop_at_step} reached at env step {global_step}")
            break

    if args.save_model:
        agent.save(runs_path(run_name, "final_ckpt.pt"), global_step)
    agent.logger.close()
    train_envs.close()
    eval_envs.close()

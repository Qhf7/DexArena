"""Online RLPD from pixels for ball grasping (RLPD_PIXELS_SPEC).

Same loop as ``train_rlpd.py`` — collect with the current stochastic policy,
run high-UTD updates on a critic ensemble, mix in prior data 50/50 — except the
observation is two 128x128 RGB cameras plus a proprioceptive vector, and the
agent carries a visual encoder trained by the critic loss.

Prior data is *optional*: with no ``--demo_data_path`` the run is online-only
(``offline_mix_fraction`` is forced to 0) and everything else about the recipe
is unchanged. With a dataset, its ``camera_config_hash`` must match the current
visual config and its ``done`` convention must match the online one, both
checked before the first gradient step (§5.4, §5.5).

Examples::

    # online-only, D4PG encoder from scratch (the main baseline)
    python train/train_rlpd_pixels.py --env_id CabinetPouringOpenCabinet-Sharpa-v1 \\
        --num_envs 8 --utd_ratio 8 --total_env_steps 1000000 \\
        --exp_name open_cabinet_rlpd_pixels
"""

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

import tasks  # noqa: F401  (registers every RoboDex env)
from algo.offline_to_online_rl.rlpd_pixel import RLPDPixelAgent
from train.train_configs.rlpd_pixels_config import (
    encoder_config_from_args,
    parse_args,
    pixel_config_from_args,
)
from utils.delta_action import DeltaActionWrapper, to_delta_image_dataset
from utils.image_replay_buffer import ImageReplayBuffer, observation_grid
from utils.paths import runs_path
from utils.pixel_dataset import (
    DONE_CONVENTION,
    OfflineImageDataset,
    image_demo_init_action,
)
from utils.pixel_env import make_pixel_eval_env, make_pixel_train_env
from utils.pixel_observation import (
    assert_obs_contract,
    extract_pixel_obs,
    report_env_obs_keys,
)
from utils.train_utils import (
    obs_to_device,
    build_run_name,
    evaluate_policy,
    log_train_episode_metrics,
    resolve_device,
    set_seed,
)


def with_prev_action(obs, command):
    """Append the normalized previous command to `proprio` (images untouched).

    Under `delta_action` the executed target is `a_t = clip(a_{t-1} + u*delta)`,
    so `a_{t-1}` is part of the state -- without it the MDP is not Markov. The
    state agent gets this block from `DeltaActionWrapper._augment`; on the image
    path the observation reaching the wrapper is still ManiSkill's raw rgb dict,
    so the block is added here, where proprio is assembled, and by
    `to_delta_image_dataset` on the offline half. Both must agree.
    """
    if command is None:
        return obs
    merged = dict(obs)
    merged["proprio"] = torch.cat([obs["proprio"], command.to(obs["proprio"].device)], dim=-1)
    return merged


def true_next_obs(next_obs_raw, infos, pixel_config, command=None):
    """Observation contract for ``o'``, using the terminal frame on done steps.

    ``ManiSkillVectorEnv`` auto-resets, so ``next_obs`` of a finished sub-env is
    already the *reset* observation; the real ``o'`` is in ``final_observation``.
    Bootstrapping from the reset frame would teach the critic that success looks
    like a fresh episode.
    """
    # `command` is the command that produced this o' (pre-reset), which is the
    # right one for both branches below.
    next_obs = with_prev_action(extract_pixel_obs(next_obs_raw, pixel_config), command)
    if "final_observation" not in infos:
        return next_obs
    done_mask = infos["_final_observation"]
    if not bool(done_mask.any().item()):
        return next_obs
    final_obs = with_prev_action(
        extract_pixel_obs(infos["final_observation"], pixel_config), command
    )
    for key, value in next_obs.items():
        # Clone: the env owns the tensors behind `next_obs`.
        merged = value.clone()
        merged[done_mask] = final_obs[key][done_mask]
        next_obs[key] = merged
    return next_obs


if __name__ == "__main__":
    args = parse_args()
    run_name = build_run_name(args, "train_rlpd_pixels")
    set_seed(args)
    device = resolve_device(args)

    pixel_config = pixel_config_from_args(args)
    encoder_config = encoder_config_from_args(args)
    print(pixel_config.summary())

    eval_envs, env_kwargs = make_pixel_eval_env(args, pixel_config, run_name)
    if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
        raise TypeError("RLPD supports only continuous Box action spaces.")
    init_action = (
        image_demo_init_action(args.demo_data_path) if args.delta_action else None
    )
    if args.delta_action:
        # Outside ManiSkillVectorEnv (it owns auto-reset), and with
        # `augment_obs=False`: the observation here is still the raw rgb dict, so
        # the command block is appended later, where proprio is built.
        eval_envs = DeltaActionWrapper(
            eval_envs,
            delta=args.delta_action_scale,
            init_action=init_action,
            augment_obs=False,
        )
    action_dim = int(np.prod(eval_envs.single_action_space.shape))
    action_low = torch.from_numpy(eval_envs.single_action_space.low).float()
    action_high = torch.from_numpy(eval_envs.single_action_space.high).float()
    control_mode = str(eval_envs.unwrapped.control_mode)

    # Spec §2: print what the rgb env actually returns instead of assuming, then
    # assert the assembled observation against the contract.
    raw_obs, _ = eval_envs.reset(seed=args.seed)
    print(report_env_obs_keys(raw_obs, pixel_config))
    probe_obs = extract_pixel_obs(raw_obs, pixel_config)
    assert_obs_contract(probe_obs, pixel_config)
    # The contract is asserted on the bare observation; the delta command block
    # is appended afterwards, so the agent's proprio is that much wider.
    proprio_dim = int(probe_obs["proprio"].shape[1]) + (action_dim if args.delta_action else 0)
    print(
        f"observation contract OK: cameras={list(pixel_config.camera_keys)} "
        f"{pixel_config.image_shape}, proprio_dim={proprio_dim}, action_dim={action_dim}, "
        f"control_mode={control_mode}"
    )

    offline = None
    if args.demo_data_path is not None:
        offline = OfflineImageDataset(
            args.demo_data_path,
            pixel_config,
            expected_action_dim=action_dim,
            expected_control_mode=control_mode,
            success_only=args.success_only,
            # The stored proprio is `action_dim` columns narrower than what the
            # agent sees; `to_delta_image_dataset` appends the same block.
            expected_proprio_dim=proprio_dim - (action_dim if args.delta_action else 0),
        )
        if args.delta_action:
            to_delta_image_dataset(
                offline,
                delta=args.delta_action_scale,
                action_low=eval_envs.absolute_action_low,
                action_high=eval_envs.absolute_action_high,
                init_action=init_action,
            )
            print(
                f"[replay] rewritten into the delta parameterization "
                f"(proprio {offline.proprio_frames.shape[1]}, actions in [-1, 1])"
            )
        # A time-limit row is not a terminal state -- the episode was cut by the
        # harness, not by the MDP -- so its target must bootstrap instead of being
        # pinned at the immediate reward. Masked by `truncated`, so a row that
        # ended on success keeps done=1, matching online collection.
        timeouts = int((offline.dones * offline.truncated).sum().item())
        offline.dones = offline.dones * (1.0 - offline.truncated)
        print(f"[replay] bootstrap restored on {timeouts} timeout transitions")
        print(offline.summary())

    if args.track:
        config = vars(args).copy()
        config["env_cfg"] = dict(
            **env_kwargs,
            num_envs=int(getattr(eval_envs, "num_envs", args.num_eval_envs)),
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
            group="RLPD",
            tags=["rlpd", "online", "pixels", args.encoder_arch],
        )

    agent = RLPDPixelAgent(
        args,
        pixel_config=pixel_config,
        encoder_config=encoder_config,
        proprio_dim=proprio_dim,
        action_low=action_low,
        action_high=action_high,
        run_name=run_name,
    ).to(device)

    if args.checkpoint is not None:
        agent.load(args.checkpoint, load_optimizers=not args.evaluate)

    def obs_projector(observation, envs=None):
        """Raw env obs -> the agent's observation.

        Reads the delta wrapper's *current* `a_prev` at call time: after
        `step` it has already been rewound for the sub-envs that reset, so it
        pairs with the observation being projected.
        """
        projected = extract_pixel_obs(observation, pixel_config)
        if not args.delta_action:
            return projected
        wrapper = envs if envs is not None else eval_envs
        return with_prev_action(projected, wrapper.normalized(wrapper.a_prev))

    if args.evaluate:
        eval_projector = obs_projector
        if args.eval_first_frame_path:
            # Save what the policy actually sees on the first eval step, cameras
            # side by side. For a `--scene-background` run this is the evidence
            # that the new scene is inside the policy's field of view at all --
            # without it, "same success rate as the original scene" is ambiguous.
            saved = []

            def eval_projector(observation, envs=None):
                projected = obs_projector(observation, envs)
                if not saved:
                    from PIL import Image

                    frame = torch.cat(
                        [projected[key][0] for key in pixel_config.camera_keys], dim=1
                    )
                    Path(args.eval_first_frame_path).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame.cpu().numpy()).save(args.eval_first_frame_path)
                    saved.append(True)
                return projected

        metrics = evaluate_policy(agent, eval_envs, args, device, obs_projector=eval_projector)
        print(f"Evaluated {int(metrics.pop('episodes', 0.0))} episodes")
        for key, value in metrics.items():
            print(f"eval_{key}_mean={value}")
            agent.logger.add_scalar(f"eval/{key}", value, 0)
        agent.logger.close()
        eval_envs.close()
        raise SystemExit(0)

    train_envs, num_envs, _ = make_pixel_train_env(args, pixel_config)
    if args.delta_action:
        train_envs = DeltaActionWrapper(
            train_envs,
            delta=args.delta_action_scale,
            init_action=init_action,
            augment_obs=False,
        )
    train_projector = lambda observation: obs_projector(observation, train_envs)  # noqa: E731
    train_command = (
        (lambda: train_envs.normalized(train_envs.a_prev_pre_reset))
        if args.delta_action
        else (lambda: None)
    )
    buffer = ImageReplayBuffer(
        args.buffer_size,
        pixel_config.camera_keys,
        pixel_config.image_shape,
        proprio_dim,
        action_dim,
        device="cpu",  # host RAM; batches move to the GPU at sample time (§3.5)
        offline=offline,
        offline_mix_fraction=args.offline_mix_fraction,
    )
    print(
        f"online image buffer: capacity={args.buffer_size}, "
        f"{buffer.storage_bytes() / 2**30:.1f} GiB of host RAM"
    )

    agent.train()
    action_low_dev = agent.action_low
    action_high_dev = agent.action_high

    raw_obs, _ = train_envs.reset(seed=args.seed)
    obs = train_projector(raw_obs)
    env_step = 0
    update_step = 0
    metrics: dict[str, float] = {}
    next_log = args.log_freq
    next_eval = args.eval_freq
    next_save = args.save_freq
    # Best-by-eval checkpoint. `ckpt_<step>.pt` is a periodic snapshot and says
    # nothing about quality; with sparse rewards the last step is routinely not
    # the best one, so the run's most useful artifact has to be tracked as it
    # happens -- once training has moved on, the weights are gone.
    #
    # Ranked on `success_once` (success at ANY point in the episode), which is
    # the metric these tasks latch and the one the state runs were read on.
    # Ties break on mean return, so among equally-successful policies the one
    # that got there in fewer steps wins; a tie on BOTH keeps the earlier
    # checkpoint rather than churning the file every eval.
    best_eval = {"success_once": float("-inf"), "return": float("-inf"), "env_step": -1}
    done_convention_checked = offline is None
    start_time = time.time()

    while env_step < args.total_env_steps:
        with torch.no_grad():
            if env_step < args.start_steps:
                rand = torch.rand((num_envs, action_dim), device=device)
                action = action_low_dev + rand * (action_high_dev - action_low_dev)
                if args.control_mode == "pd_joint_pos" and not args.delta_action:
                    arm_dim = min(7, action_dim)
                    current_arm_qpos = obs["proprio"][:, :arm_dim].to(device)
                    local_noise = (
                        torch.rand((num_envs, arm_dim), device=device) * 2.0 - 1.0
                    ) * args.joint_warmup_delta
                    action[:, :arm_dim] = torch.clamp(
                        current_arm_qpos + local_noise,
                        action_low_dev[:arm_dim],
                        action_high_dev[:arm_dim],
                    )
            else:
                # `obs_to_device`, exactly as the eval path does
                # (utils/train_utils.py:341). `extract_pixel_obs` leaves tensors
                # on the device the ENV produced them on, which is the GPU for
                # physx_cuda and the CPU for physx_cpu -- so omitting this worked
                # by accident on 19 of the 27 fleet tasks and killed all 8
                # physx_cpu ones at the first post-`start_steps` action with
                # `Expected all tensors to be on the same device, but found at
                # least two devices, cuda:0 and cpu!` (2026-09-18).
                action = agent.select_action(
                    obs_to_device(obs, device), deterministic=False
                )

        next_raw_obs, reward, terminations, truncations, infos = train_envs.step(action)
        # o' pairs with the command that produced it, i.e. before the auto-reset
        # rewound `a_prev` for the sub-envs that finished.
        next_obs = true_next_obs(next_raw_obs, infos, pixel_config, train_command())

        buffer.add(obs, action, reward, next_obs, terminations, truncations)
        agent.update_proprio_stats(obs["proprio"])

        log_train_episode_metrics(agent, infos, env_step + num_envs)
        # The next iteration's `o` uses the post-reset command instead.
        obs = train_projector(next_raw_obs)
        env_step += num_envs

        # Spec §5.5. Both halves use `done = terminated = info["success"]`: online
        # because that is what the buffer stores, offline because
        # `OfflineImageDataset` refuses to load an episode whose final `done`
        # disagrees with its success flag. So this is a report, not an assertion --
        # the two *fractions* legitimately differ, since the demos are all
        # successful while the online success rate starts at zero. Comparing them
        # directly would fail every run for the first hundred thousand steps.
        if not done_convention_checked and buffer.episode_boundaries >= num_envs * 4:
            print(
                f"[done-convention] {DONE_CONVENTION!r} on both halves | "
                f"episode-final done: offline={offline.terminal_done_fraction:.2f} "
                f"(= its success rate) online={buffer.terminal_done_fraction:.2f} "
                f"(= the policy's success rate so far)"
            )
            done_convention_checked = True

        if env_step >= args.start_steps and buffer.size >= args.batch_size:
            # `utd_ratio` critic updates per env step; the actor and the
            # temperature are updated once per block, on the last mini-batch.
            for i in range(args.utd_ratio):
                update_step += 1
                batch = buffer.sample(args.batch_size, device)
                metrics = agent.update(
                    batch, update_step, update_actor=(i == args.utd_ratio - 1)
                )

        if env_step >= next_log and update_step > 0:
            for key, value in metrics.items():
                if np.isfinite(value):
                    agent.logger.add_scalar(f"losses/{key}", value, env_step)
            grid = observation_grid(
                getattr(agent, "_last_augmented", {}) or {}, pixel_config.camera_keys
            ) if getattr(agent, "_last_augmented", None) else None
            if grid is not None:
                # Rows: offline samples above, online below; columns: cam_0, cam_1.
                agent.logger.add_image("obs/augmented_batch", grid.numpy(), env_step)
            sps = env_step / max(time.time() - start_time, 1e-6)
            agent.logger.add_scalar("charts/steps_per_second", sps, env_step)
            agent.logger.add_scalar("charts/update_step", update_step, env_step)
            agent.logger.add_scalar("charts/buffer_size", buffer.size, env_step)
            print(
                f"env_step {env_step}/{args.total_env_steps} "
                f"(updates={update_step}): q_loss={metrics['q_loss']:.4f}, "
                f"actor_loss={metrics['actor_loss']:.4f}, "
                f"alpha={metrics['entropy_alpha']:.4f}, "
                f"q_mean={metrics['q_mean']:.3f}, "
                f"ens_std={metrics['critic_ensemble_std']:.3f}, "
                f"|z|={metrics['latent_l2_norm']:.2f}, {sps:.1f} steps/s"
            )
            next_log += args.log_freq

        if env_step >= next_eval:
            eval_metrics = evaluate_policy(
                agent, eval_envs, args, device, obs_projector=obs_projector
            )
            print(f"Evaluated {int(eval_metrics.pop('episodes', 0.0))} episodes")
            for key, value in eval_metrics.items():
                print(f"eval_{key}_mean={value}")
                agent.logger.add_scalar(f"eval/{key}", value, env_step)
            score = eval_metrics.get("success_once")
            if args.save_model and score is not None:
                ret = float(eval_metrics.get("return", float("-inf")))
                better = (
                    score > best_eval["success_once"]
                    or (score == best_eval["success_once"] and ret > best_eval["return"])
                )
                if better:
                    best_eval = {
                        "success_once": float(score),
                        "return": ret,
                        "env_step": int(env_step),
                        "update_step": int(update_step),
                    }
                    agent.save(runs_path(run_name, "best_ckpt.pt"), env_step)
                    # Written AFTER the checkpoint, so a crash between the two
                    # leaves a stale-but-valid .pt and no json claiming a score
                    # for weights that were never written.
                    Path(runs_path(run_name, "best_ckpt.json")).write_text(
                        json.dumps(best_eval, indent=2)
                    )
                    print(
                        f"new best eval: success_once={score:.3f} return={ret:.3f} "
                        f"at env_step {env_step} -> best_ckpt.pt"
                    )
                else:
                    print(
                        f"eval success_once={score:.3f} did not beat "
                        f"{best_eval['success_once']:.3f} from env_step "
                        f"{best_eval['env_step']}; best_ckpt.pt unchanged"
                    )
                agent.logger.add_scalar("eval/best_success_once",
                                        best_eval["success_once"], env_step)
            agent.train()
            next_eval += args.eval_freq

        if args.save_model and env_step >= next_save:
            agent.save(runs_path(run_name, f"ckpt_{env_step}.pt"), env_step)
            next_save += args.save_freq

    if args.save_model:
        agent.save(runs_path(run_name, "final_ckpt.pt"), env_step)
        print(
            f"best eval over the run: success_once={best_eval['success_once']:.3f} "
            f"at env_step {best_eval['env_step']} (best_ckpt.pt)"
            if best_eval["env_step"] >= 0
            else "no eval ran; best_ckpt.pt was never written"
        )

    agent.logger.close()
    train_envs.close()
    eval_envs.close()

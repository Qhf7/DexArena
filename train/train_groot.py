"""GR00T baseline — finetune + eval entry point (mirrors ``train/train_dppo.py``).

Runs in the unified env (``gr00t`` + ManiSkill in one venv). Two modes:

* **Finetune (default):** dispatches GR00T's ``launch_finetune.py`` via ``torchrun`` on
  ``--num_gpus`` GPUs to finetune ``--base_model_path`` on a LeRobot ``--dataset_path``.
* **Eval (``--evaluate``):** builds an ``obs_mode="rgb"`` ManiSkill env and constructs a
  :class:`GrootAgent` that loads the ``--model_path`` checkpoint **in-process** (no server) —
  or a :class:`ZeroActionAgent` stub for harness smoke-tests — and runs ``evaluate_policy`` to
  report ``eval_success_once``, directly comparable to DP / PPO / DPPO.

Usage:
    # Finetune (4 GPUs):
    CUDA_VISIBLE_DEVICES=1,4,6,7 python train/train_groot.py \
        --dataset_path <vla_dataset> --env_id <env_id> \
        --output_dir runs/groot_ft --num_gpus 4

    # Eval a finetuned checkpoint in-process:
    python train/train_groot.py --evaluate --model_path runs/groot_ft/checkpoint-10000 \
        --env_id <task> --num_eval_episodes 50 --num_eval_steps <task-horizon>

    # Harness smoke-test (no checkpoint; proves rgb env + eval loop):
    python train/train_groot.py --evaluate --stub --num_eval_episodes 2 --num_eval_steps 20
"""

from collections import defaultdict
import json
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

from algo.groot import GrootAgent, ZeroActionAgent
from utils.dex_tasks import draw_instruction
from utils.groot_utils.modality_configs import MODALITY_JSON
from train.train_configs.groot_config import parse_args

# ManiSkill specific imports
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

try:
    # Optional GPU-IK patch. Not committed on all branches; only needed for IK-based control
    # modes on the GPU sim backend. Harmless to skip for joint-position control.
    from utils.maniskill_patches import patch_maniskill_gpu_ik

    patch_maniskill_gpu_ik()
except ModuleNotFoundError:
    pass

import tasks  # noqa: F401  (registers every environment id)


# ---------------------------------------------------------------------------------
# Setup helpers (mirror train_dppo.py)
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
    return _apply_gpu_id(getattr(args, "sim_backend", None) or "physx_cuda", args.gpu_id, {"physx_cuda", "gpu", "cuda"})


def resolve_render_backend(args) -> str:
    return _apply_gpu_id("sapien_cuda", args.gpu_id, {"sapien_cuda", "gpu", "cuda"})


def make_eval_env(args, *, num_envs, reconfiguration_freq, record_dir=None, max_steps_per_video=None):
    """Build the eval env with ``obs_mode="rgb"`` (GR00T consumes camera images)."""
    env_kwargs = dict(
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend=resolve_sim_backend(args),
        render_backend=resolve_render_backend(args),
    )
    if getattr(args, "camera_size", None) is not None:
        env_kwargs["sensor_configs"] = dict(width=args.camera_size, height=args.camera_size)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.reward_mode is not None:
        env_kwargs["reward_mode"] = args.reward_mode
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids
    if args.obj_id is not None and "specific" in args.env_id.lower():
        env_kwargs["obj_id"] = args.obj_id
    # The policy only ever saw the scene as the dataset builder rendered it, so any
    # env option used at build time has to be repeated here. Missing them silently
    # changes the observation distribution: with the goal marker left hidden, a
    # vision-conditioned policy has no goal in its input at all.
    for key, value in (json.loads(args.eval_env_extra) if args.eval_env_extra else {}).items():
        env_kwargs[key] = value
    env_kwargs["max_episode_steps"] = args.env_max_episode_steps
    from utils.dex_cameras import install_camera_profile
    install_camera_profile(args.env_id, getattr(args, "camera_profile", "none"))
    if getattr(args, "scene_background", None):
        from utils.dex_scene_eval import make_scene_eval_env
        env = make_scene_eval_env(args.env_id, scene_background=args.scene_background,
                                  num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **env_kwargs)
        args.scene_background_manifest = env.unwrapped.eval_background_manifest
        if record_dir is not None:
            destination = Path(record_dir).parent / "scene_background.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(args.scene_background_manifest, indent=2) + "\n")
    else:
        args.scene_background_manifest = None
        env = gym.make(args.env_id, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if record_dir is not None:
        env = RecordEpisode(
            env, output_dir=record_dir, save_trajectory=False,
            max_steps_per_video=max_steps_per_video,
            video_fps=int(getattr(env.unwrapped, "control_freq", 20)),
        )
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    return env


def _success_from_info(info) -> bool:
    """True if any env reports task success this step (ManiSkill convention)."""
    success = info.get("success")
    if success is None:
        return False
    if torch.is_tensor(success):
        return bool(success.any().item())
    return bool(np.asarray(success).any())


@torch.no_grad()
def _flag_from_info(info, key):
    """Read a boolean env flag out of info, tolerating its absence.

    Not every task variant defines is_grasped (the gripper-less ones use
    obj_tip_dist instead), so a missing key means "no signal", not False.
    """
    v = info.get(key)
    if v is None:
        return False
    import torch as _t
    return bool(_t.as_tensor(v).any().item())

def resolve_eval_instructions(args):
    """The instructions this evaluation may run under, canonical first.

    GR00T's language input used to be ``--exp_name``, i.e. whatever string the run happened to
    be named after. It now comes from the shared table (utils/dex_tasks.py) so a checkpoint is
    evaluated on the language it was trained on, and -- under the default ``registered``
    sampling -- on the task's other phrasings too.
    """
    from utils.dex_tasks import DEX_TASKS_BY_ENV, validate_dex_instruction

    task = DEX_TASKS_BY_ENV.get(args.env_id)
    if task is None:  # envs outside the table (the YCB/gripper lineage) keep free text
        return (getattr(args, "prompt", None) or args.exp_name or "task",)
    prompt = getattr(args, "prompt", None)
    if prompt is not None:
        validate_dex_instruction(args.env_id, prompt)
    if args.instruction_sampling == "canonical":
        return (prompt or task.prompt,)
    if args.instruction_sampling != "registered":
        raise ValueError(
            f"--instruction_sampling must be 'registered' or 'canonical'; "
            f"got {args.instruction_sampling!r}."
        )
    return task.instructions


def evaluate_policy(agent, env, args, device, *, action_low, action_high, instructions=None):
    """Serial per-episode eval (GR00T eval uses a single env).

    Each episode: reset, then repeatedly query GR00T for an action chunk and execute it
    receding-horizon until the env terminates/truncates or the step budget is hit. Reports
    ``success_once`` (success at any point in the episode) and mean return — the same metric
    family used by dppo/ppo so the comparison row is apples-to-apples.
    """
    H = agent.action_horizon
    assert args.num_eval_envs == 1, (
        "GR00T eval runs a single env (agents emit batch-1 action chunks); "
        f"got --num_eval_envs {args.num_eval_envs}."
    )
    instructions = tuple(instructions or resolve_eval_instructions(args))
    metrics = defaultdict(list)
    records = []
    for ep in range(args.num_eval_episodes):
        episode_seed = args.seed + ep
        # One instruction per episode, drawn from the episode's own seed so the wording varies
        # without the run becoming irreproducible (utils/dex_tasks.draw_instruction).
        instruction = draw_instruction(args.env_id, episode_seed, instructions)
        if hasattr(agent, "set_instruction"):
            agent.set_instruction(instruction)
        obs, _ = env.reset(seed=episode_seed)
        agent.reset_rollout(num_envs=args.num_eval_envs)
        ep_success = False
        # success_once conflates two phases: did it pick the object up at all,
        # and did it get the object to the goal. Track them separately so a
        # failure can be attributed to grasping or to placing.
        ep_grasped = False
        ep_placed = False
        ep_return = 0.0
        steps = 0
        done = False
        while steps < args.num_eval_steps and not done:
            chunk = agent.predict_action_chunk(obs)  # (1, <=H, action_dim)
            for a_i in range(min(H, chunk.shape[1])):
                action = torch.clamp(chunk[:, a_i], action_low, action_high)
                obs, reward, term, trunc, info = env.step(action)
                ep_return += float(torch.as_tensor(reward).float().mean().item())
                ep_success = ep_success or _success_from_info(info)
                ep_grasped = ep_grasped or _flag_from_info(info, "is_grasped")
                ep_placed = ep_placed or _flag_from_info(info, "is_obj_placed")
                steps += 1
                done = bool(torch.logical_or(term, trunc).any().item())
                if done or steps >= args.num_eval_steps:
                    break
        # Per-episode line so a successful run can be found among the recordings;
        # the summary alone cannot tell you which episode to watch.
        print(f"[episode] idx={len(metrics['success_once'])} "
              f"success={int(bool(ep_success))} grasped={int(bool(ep_grasped))} "
              f"placed={int(bool(ep_placed))} return={ep_return:.2f} steps={steps}",
              flush=True)
        records.append({
            "episode_index": ep, "environment_seed": int(episode_seed),
            "instruction": instruction,
            "success": bool(ep_success), "grasped": bool(ep_grasped), "placed": bool(ep_placed),
            "return": float(ep_return), "episode_length": int(steps),
            # RecordEpisode names videos by episode order, so this is the matching file.
            "video": f"{ep}.mp4",
        })
        metrics["success_once"].append(float(ep_success))
        metrics["grasped_once"].append(float(ep_grasped))
        metrics["placed_once"].append(float(ep_placed))
        metrics["return"].append(ep_return)
        metrics["episode_len"].append(steps)
    summary = {k: float(np.mean(v)) for k, v in metrics.items() if v}
    summary["num_episodes"] = len(records)
    summary["num_successes"] = int(sum(r["success"] for r in records))
    evaluate_policy.last_records = records
    return summary


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------


def finetune(args):
    """Finetune GR00T by dispatching its ``launch_finetune.py`` via torchrun (multi-GPU).

    GR00T finetuning is an ``accelerate``/``torchrun`` distributed job over the ``gr00t`` package;
    we build the command and ``exec`` it so this file is the single finetune entry point. Set
    ``GROOT_BACKBONE_PATH`` and ``CUDA_VISIBLE_DEVICES`` in the environment before calling.
    """
    if not args.dataset_path:
        raise ValueError("finetune needs --dataset_path (a LeRobot v2.1 VLA dataset).")
    gr_dir = REPO_ROOT / "libs" / "Isaac-GR00T"
    output_dir = args.output_dir or f"runs/{build_run_name(args)}"
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.num_gpus}", f"--master_port={args.master_port}",
        str(gr_dir / "gr00t" / "experiment" / "launch_finetune.py"),
        "--base_model_path", str(args.base_model_path),
        "--dataset_path", str(args.dataset_path),
        "--embodiment_tag", str(args.embodiment_tag),
        "--num_gpus", str(args.num_gpus),
        "--global_batch_size", str(args.global_batch_size),
        "--output_dir", str(output_dir),
        "--max_steps", str(args.max_steps),
        "--save_steps", str(args.save_steps),
        "--save_total_limit", str(args.save_total_limit),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
    ]
    if args.learning_rate is not None:
        cmd += ["--learning_rate", str(args.learning_rate)]
    if args.dataloader_num_workers is not None:
        cmd += ["--dataloader_num_workers", str(args.dataloader_num_workers)]
    if args.resume_from_checkpoint:
        # GR00T's flag is boolean: it resumes from the newest checkpoint-* under
        # --output_dir. The path we were given only selects that run directory.
        cmd += ["--resume_from_checkpoint"]
    # Modality config: a single env-keyed module (utils/groot_utils/modality_configs.py) that
    # self-registers the env named by $GROOT_MODALITY_ENV on import (launch_finetune imports it).
    modality_path = args.modality_config_path or str(
        REPO_ROOT / "utils" / "groot_utils" / "modality_configs.py")
    cmd += ["--modality_config_path", modality_path]
    # Default to the env's own config, but let an explicitly-set GROOT_MODALITY_ENV win —
    # that is the only way to train a variant config (e.g. the "-vision" entry that drops the
    # goal dims from state) without inventing a second env id. Overwriting unconditionally
    # silently turned a vision-only ablation into a duplicate of the baseline.
    os.environ.setdefault("GROOT_MODALITY_ENV", str(args.env_id))
    if args.tune_llm:
        cmd += ["--tune_llm"]
    if args.tune_visual:
        cmd += ["--tune_visual"]
    # Backbone tuning defaults to ON (GR00T's own convention), which costs ~64 GB and
    # will not fit a 40 GB card. It is the easiest thing to get wrong by omission, so
    # state it plainly rather than leaving it to be inferred from the exec line.
    _tuning = args.tune_llm or args.tune_visual
    _mode = (f"full finetune (llm={args.tune_llm}, visual={args.tune_visual})"
             if _tuning else "frozen backbone, action head only")
    print(f"[train_groot] backbone: {_mode}; expect ~{64 if _tuning else 13} GB GPU memory")
    print("[train_groot] finetune (exec):", " ".join(cmd))
    os.execvp(sys.executable, cmd)  # replace this process with torchrun; does not return


def main():
    args = parse_args()
    if not args.evaluate:
        finetune(args)  # execs torchrun; never returns
        return
    args.env_max_episode_steps = (
        args.max_episode_steps if args.max_episode_steps is not None else args.num_eval_steps
    )
    run_name = build_run_name(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = resolve_torch_device(args)

    out_dir = Path(args.eval_output_dir) if args.eval_output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    if args.capture_video:
        record_dir = str(out_dir / "videos") if out_dir is not None else f"runs/{run_name}/eval_videos"
    else:
        record_dir = None
    env = make_eval_env(
        args, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq,
        record_dir=record_dir, max_steps_per_video=args.env_max_episode_steps,
    )
    assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    action_dim = int(np.prod(env.single_action_space.shape))
    action_low = torch.from_numpy(env.single_action_space.low).to(device)
    action_high = torch.from_numpy(env.single_action_space.high).to(device)
    print(f"Env {args.env_id}: obs_mode=rgb, action_dim={action_dim}, num_eval_envs={args.num_eval_envs}")
    instructions = resolve_eval_instructions(args)
    print(f"[eval] instruction_sampling={args.instruction_sampling} over {len(instructions)}: "
          + ", ".join(repr(text) for text in instructions))

    if args.stub:
        print("Using ZeroActionAgent stub (harness smoke-test; no GR00T server).")
        agent = ZeroActionAgent(args, action_dim=action_dim, run_name=run_name, device=device).to(device)
    else:
        print(f"Loading GrootAgent in-process from {args.model_path}")
        # A policy trained with the goal carried in its state (the converter's
        # --goal_in_state) needs the same three dims at eval time, but the env's
        # proprio vector stops at the gripper. Rebuild the layout and append the
        # goal-site position so the observation matches what it was trained on.
        state_layout, goal_state_fn = None, None
        modality_state = MODALITY_JSON.get(os.environ.get("GROOT_MODALITY_ENV", args.env_id), {}).get("state", {})
        if any(key.endswith("hand") for key in modality_state):
            # Dexterous hand (single or dual arm): the state is every actuated joint in
            # controller order, matching the dataset built with --proprio controller_qpos.
            from utils.image_observation import extract_controller_qpos

            state_layout = tuple((k, v["start"], v["end"]) for k, v in modality_state.items())

            def goal_state_fn(obs, _env=env):
                return extract_controller_qpos(obs, env=_env)

            print(f"[eval] dex state layout: {state_layout}")
        elif "goal" in modality_state:
            from utils.image_observation import extract_franka_proprio

            state_layout = (("single_arm", 0, 7), ("gripper", 7, 8), ("goal", 8, 11))

            def goal_state_fn(obs, _env=env):
                proprio = np.asarray(extract_franka_proprio(obs), dtype=np.float32).reshape(-1)
                goal = _env.unwrapped.goal_site.pose.p[0].cpu().numpy().astype(np.float32)
                return np.concatenate([proprio, goal])

            print("[eval] state includes goal: proprio 8 -> 11 dims")
        agent = GrootAgent(args, action_dim=action_dim, run_name=run_name, device=device,
                           prompt=instructions[0],
                           state_layout=state_layout, state_fn=goal_state_fn).to(device)

    if args.track:
        import swanlab

        config = vars(args).copy()
        config["action_dim"] = action_dim
        swanlab.init(
            project=args.swanlab_project_name, entity=args.swanlab_entity,
            sync_tensorboard=False, config=config, name=run_name, save_code=True,
            group="GROOT", tags=["groot", "eval", "baseline"],
        )

    agent.eval()
    eval_metrics = evaluate_policy(agent, env, args, device, action_low=action_low,
                                   action_high=action_high, instructions=instructions)
    for k, v in eval_metrics.items():
        agent.logger.add_scalar(f"eval/{k}", v, 0)
    print("eval results: " + ", ".join(f"eval_{k}={v:.4f}" for k, v in eval_metrics.items()))
    if out_dir is not None:
        # Machine-readable copy of what the log says, so videos can be sorted by outcome
        # without re-parsing stdout.
        with open(out_dir / "episodes.jsonl", "w") as fh:
            for record in getattr(evaluate_policy, "last_records", []):
                fh.write(json.dumps(record) + "\n")
        summary = dict(eval_metrics)
        summary.update({
            "env_id": args.env_id, "model_path": str(args.model_path), "seed": int(args.seed),
            "num_eval_episodes": int(args.num_eval_episodes), "max_episode_steps": int(args.env_max_episode_steps),
            "action_horizon": int(args.action_horizon), "sim_backend": args.sim_backend,
            "camera_size": args.camera_size, "camera_profile": getattr(args, "camera_profile", "none"),
            "instruction_sampling": args.instruction_sampling,
            "instructions": list(instructions),
            "modality_env": os.environ.get("GROOT_MODALITY_ENV", args.env_id),
            "scene_background": getattr(args, "scene_background_manifest", None),
        })
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[eval] wrote {out_dir}/summary.json and episodes.jsonl")

    agent.logger.close()
    env.close()


if __name__ == "__main__":
    main()

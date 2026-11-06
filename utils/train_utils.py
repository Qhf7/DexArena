"""Shared setup and evaluation helpers for the RL training entry points.

Every ``train/train_*.py`` script used to carry its own copy of these (or import
them from ``train/train_iql.py``, which made one algorithm's script a dependency
of the others). The implementations here are the superset versions: they log the
richer per-episode metrics and tolerate configs that do not define every field.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
from pathlib import Path
import random
import time
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from utils.paths import runs_path


# Episode metrics collected from `info`; absent keys are skipped, so a task only
# needs to expose the ones it actually computes.
BOOL_KEYS = ("success", "placed", "settled", "near_goal", "loose_grasp", "lifted")
SCALAR_MIN_KEYS = ("obj_to_goal_dist", "tip_min_dist", "tip_mean_dist", "obj_vel", "arm_qvel")
SCALAR_MAX_KEYS = ("lift_height", "num_close_tips")


def build_run_name(args, algo_tag: str) -> str:
    # FQE has no eval mode and builds no environment, so neither field is required.
    prefix = "eval_" if getattr(args, "evaluate", False) else ""
    if args.exp_name is not None:
        return prefix + args.exp_name
    env_id = getattr(args, "env_id", None)
    stem = f"{env_id}__{algo_tag}" if env_id is not None else algo_tag
    return f"{prefix}{stem}__{args.seed}__{int(time.time())}"


def set_seed(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


def maybe_apply_gpu_id(backend: str | None, gpu_id: int | None, names: set[str]):
    if backend is None or gpu_id is None or ":" in backend or backend not in names:
        return backend
    return f"{backend}:{gpu_id}"


def resolve_device(args) -> torch.device:
    if not (args.cuda and torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device(f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda")


def resolve_sim_backend(args) -> str:
    sim_backend = args.sim_backend
    if sim_backend is None:
        sim_backend = "physx_cuda" if torch.cuda.is_available() else "physx_cpu"
    return maybe_apply_gpu_id(sim_backend, args.gpu_id, {"physx_cuda", "gpu", "cuda"})


# One eval video is accumulated frame-by-frame in host RAM before it is
# encoded, so its total size is a hard budget, not a preference.
_VIDEO_RAM_BUDGET_BYTES = 4 * 1024 ** 3
# libx264 refuses 19200x10800; stay well inside what it accepts.
_VIDEO_TILED_PIXEL_CAP = 4096 * 2304
# Never let the budget *raise* a camera above this: at num_envs=1 the two caps
# above are slack, and a bigger render buys nothing but Vulkan work per frame.
_VIDEO_MAX_TILE_HEIGHT = 360


def apply_optional_env_kwargs(args, env_kwargs: dict) -> None:
    robot_uids = getattr(args, "robot_uids", None)
    if robot_uids is not None:
        env_kwargs["robot_uids"] = robot_uids
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode


def eval_video_output_dir(args, run_name: str) -> str:
    active_checkpoint = getattr(args, "checkpoint", None) or getattr(
        args, "pretrained_checkpoint", None
    )
    if args.evaluate and active_checkpoint is not None:
        return str(Path(active_checkpoint).parent / "test_videos")
    return runs_path(run_name, "videos")


def video_camera_override(num_envs: int, steps_per_video: int) -> dict:
    """Human-render camera size that keeps the tiled eval video encodable.

    ``RecordEpisode`` tiles all ``num_envs`` frames into one image and holds the
    whole video in host RAM, so the per-env camera is sized from the tile count
    and the video length (16:9, floor 128x72).
    """
    grid_rows = max(1, int(math.sqrt(num_envs)))
    grid_cols = math.ceil(num_envs / grid_rows)
    # Two independent caps: bytes held in RAM for one video, and the encoder's
    # frame-size limit. Whichever binds first wins.
    ram_pixels = _VIDEO_RAM_BUDGET_BYTES / max(1, steps_per_video) / 3
    per_env_pixels = min(ram_pixels, _VIDEO_TILED_PIXEL_CAP) / max(1, grid_rows * grid_cols)
    height = int(math.sqrt(per_env_pixels * 9.0 / 16.0))
    height = min(_VIDEO_MAX_TILE_HEIGHT, max(72, height - (height % 2)))
    width = max(128, int(height * 16.0 / 9.0) // 2 * 2)
    return dict(width=width, height=height)


def _maybe_wrap_delta_action(env, args, init_action):
    """Wrap in the residual action space when the config asks for it.

    Applied outside ``ManiSkillVectorEnv`` (and outside ``RecordEpisode``) so
    partial resets stay the vector env's business and recorded trajectories
    keep absolute joint targets. Configs without the field are unaffected.
    """
    if not getattr(args, "delta_action", False):
        return env
    if init_action is None:
        raise ValueError(
            "`delta_action` is on but no `init_action` was passed; derive it with "
            "`utils.delta_action.demo_init_action(args.demo_data_path)`."
        )
    from utils.delta_action import DeltaActionWrapper

    return DeltaActionWrapper(
        env,
        getattr(args, "delta_action_scale", 0.1),
        init_action,
    )


def make_eval_env(args, run_name: str, *, init_action=None):
    sim_backend = resolve_sim_backend(args)
    capture_video = bool(args.capture_video and torch.cuda.is_available())
    num_envs = max(1, min(args.num_eval_envs, args.num_eval_episodes))
    if sim_backend == "physx_cpu":
        num_envs = 1

    env_kwargs = {
        "obs_mode": "state",
        "render_mode": "rgb_array" if capture_video else None,
        "sim_backend": sim_backend,
        "render_backend": maybe_apply_gpu_id(
            "sapien_cuda" if capture_video else "none",
            args.gpu_id,
            {"sapien_cuda", "gpu", "cuda"},
        ),
        "reward_mode": args.reward_mode,
        "max_episode_steps": args.num_eval_steps,
    }
    apply_optional_env_kwargs(args, env_kwargs)
    if capture_video:
        # Override the env's own human-render camera: its size is chosen for a
        # single nice-looking frame, not for `num_envs` of them tiled into one
        # image and buffered in RAM. See video_camera_override.
        env_kwargs["human_render_camera_configs"] = video_camera_override(
            num_envs, args.num_eval_steps
        )
        print(
            f"[video] {num_envs} eval envs -> human render camera "
            f"{env_kwargs['human_render_camera_configs']['width']}x"
            f"{env_kwargs['human_render_camera_configs']['height']} per env"
        )

    env = gym.make(
        args.env_id,
        num_envs=num_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if capture_video:
        env = RecordEpisode(
            env,
            output_dir=eval_video_output_dir(args, run_name),
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=args.video_fps,
        )
    env = ManiSkillVectorEnv(
        env,
        num_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )
    env = _maybe_wrap_delta_action(env, args, init_action)
    return env, env_kwargs


def make_train_env(args, *, init_action=None):
    sim_backend = resolve_sim_backend(args)
    num_envs = 1 if sim_backend == "physx_cpu" else args.num_envs

    env_kwargs = {
        "obs_mode": "state",
        "render_mode": None,
        "sim_backend": sim_backend,
        "render_backend": "none",
        "reward_mode": args.reward_mode,
        "max_episode_steps": args.max_episode_steps,
    }
    apply_optional_env_kwargs(args, env_kwargs)

    # reconfiguration_freq=0 so geometry stays fixed; ManiSkillVectorEnv then
    # auto-resets done sub-envs (partial reset) for continuous online collection.
    env = gym.make(args.env_id, num_envs=num_envs, reconfiguration_freq=0, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, num_envs, ignore_terminations=False, record_metrics=True
    )
    env = _maybe_wrap_delta_action(env, args, init_action)
    return env, num_envs


def log_train_episode_metrics(agent, infos, env_step) -> Optional[dict]:
    """Log the episodes that finished on this step; return their success rate.

    The return value is what a success-gated curriculum needs (the pixels PPO
    trainer advances its demo-init schedule on it); callers that only want the
    logging can ignore it.
    """
    if "final_info" not in infos:
        return None
    done_mask = infos["_final_info"]
    if not bool(done_mask.any().item()):
        return None
    final_info = infos["final_info"]
    episode = final_info.get("episode", {})
    for key, value in episode.items():
        agent.logger.add_scalar(
            f"train/{key}", value[done_mask].float().mean().item(), env_step
        )
    if "success" in final_info:
        agent.logger.add_scalar(
            "train/success",
            final_info["success"][done_mask].float().mean().item(),
            env_step,
        )
    # `success_once` is the one to gate on: success here latches within an
    # episode, and `success_at_end` would also demand the policy hold it.
    for key in ("success_once", "success"):
        if key in episode:
            return {
                "success": float(episode[key][done_mask].float().mean().item()),
                "num_episodes": int(done_mask.sum().item()),
            }
    if "success" in final_info:
        return {
            "success": float(final_info["success"][done_mask].float().mean().item()),
            "num_episodes": int(done_mask.sum().item()),
        }
    return None


def obs_to_device(obs, device):
    """Move an observation to ``device``.

    Observations are a tensor for the state-based agents and a dict of tensors
    for the pixel agents (``cam_*`` / ``proprio``).
    """
    if isinstance(obs, Mapping):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in obs.items()
        }
    return obs.to(device)


def obs_device(obs) -> torch.device:
    if isinstance(obs, Mapping):
        for value in obs.values():
            if torch.is_tensor(value):
                return value.device
        raise ValueError("Observation dict contains no tensors.")
    return obs.device


def _step_value(infos, key, num_envs: int, reduce: str):
    """Per-env value of ``infos[key]``, using the pre-reset copy where an env is done.

    ``ManiSkillVectorEnv.step`` replaces the info dict with the auto-reset one
    on steps where an env ended and stashes the real terminal info under
    ``final_info``; reading ``infos[key]`` there would report the fresh episode.
    """
    src = infos.get(key)
    if src is None:
        return None

    def _reduce(tensor):
        tensor = tensor.detach().view(num_envs, -1)
        return tensor.bool().any(dim=-1) if reduce == "bool" else tensor.float().mean(dim=-1)

    value = _reduce(src)
    final_info = infos.get("final_info")
    done_mask = infos.get("_final_info")
    if final_info is not None and done_mask is not None and key in final_info:
        final_value = _reduce(final_info[key]).to(value.device)
        value = torch.where(done_mask.to(value.device), final_value, value)
    return value


@torch.no_grad()
def evaluate_policy(
    agent,
    envs,
    args,
    device,
    obs_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    """Roll out the deterministic policy and aggregate per-episode metrics.

    ``obs_projector`` lets a caller reshape observations before they reach the
    agent (BC uses it to drop the object-pose block).
    """
    agent.eval()
    obs, _ = envs.reset(seed=args.seed)
    if obs_projector is not None:
        obs = obs_projector(obs)
    metrics = defaultdict(list)
    completed = 0
    num_envs = int(getattr(envs, "num_envs", 1))
    # Accumulators are created on the device the infos arrive on: under
    # physx_cpu the images are on the GPU while infos stay on the CPU.
    bool_once: dict[str, torch.Tensor] = {}
    info_device: torch.device | None = None
    scalar_min: dict[str, torch.Tensor] = {}
    scalar_max: dict[str, torch.Tensor] = {}
    max_steps = args.num_eval_steps * max(
        1, math.ceil(args.num_eval_episodes / num_envs)
    )

    for _ in range(max_steps):
        action = agent.select_action(obs_to_device(obs, device), deterministic=True)
        obs, _, _, _, infos = envs.step(action)
        if obs_projector is not None:
            obs = obs_projector(obs)
        for key in BOOL_KEYS:
            value = _step_value(infos, key, num_envs, "bool")
            if value is None:
                continue
            if info_device is None:
                info_device = value.device
                # `success` is reported for every task; create it with the
                # first info tensor so it lands on the same device.
                bool_once.setdefault(
                    "success",
                    torch.zeros(num_envs, dtype=torch.bool, device=info_device),
                )
            if key not in bool_once:
                bool_once[key] = torch.zeros(num_envs, dtype=torch.bool, device=value.device)
            bool_once[key] |= value.to(bool_once[key].device)
        for key in SCALAR_MIN_KEYS:
            value = _step_value(infos, key, num_envs, "float")
            if value is None:
                continue
            scalar_min[key] = (
                value if key not in scalar_min
                else torch.minimum(scalar_min[key], value.to(scalar_min[key].device))
            )
        for key in SCALAR_MAX_KEYS:
            value = _step_value(infos, key, num_envs, "float")
            if value is None:
                continue
            scalar_max[key] = (
                value if key not in scalar_max
                else torch.maximum(scalar_max[key], value.to(scalar_max[key].device))
            )
        if "final_info" not in infos:
            continue
        done_mask = infos["_final_info"]
        if not bool(done_mask.any().item()):
            continue

        remaining = args.num_eval_episodes - completed
        done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()[:remaining]
        selected_mask = torch.zeros_like(done_mask, dtype=torch.bool)
        selected_mask[done_indices] = True
        completed += int(selected_mask.sum().item())

        final_info = infos["final_info"]
        # `ManiSkillVectorEnv(record_metrics=True)` publishes its own
        # `success_once`/`fail_once` in `final_info["episode"]`; `bool_once` wins.
        derived_once = {f"{key}_once" for key in bool_once}
        for key, value in final_info["episode"].items():
            if key in derived_once:
                continue
            metrics[key].append(value[selected_mask].detach().float().mean().item())
        for key, value in bool_once.items():
            metrics[f"{key}_once"].append(
                value[selected_mask].detach().float().mean().item()
            )
        for key, value in scalar_min.items():
            metrics[f"min_{key}"].append(
                value[selected_mask].detach().float().mean().item()
            )
        for key, value in scalar_max.items():
            metrics[f"max_{key}"].append(
                value[selected_mask].detach().float().mean().item()
            )
        for key in BOOL_KEYS:
            if key in final_info:
                metrics[f"{key}_final"].append(
                    final_info[key][selected_mask].detach().float().mean().item()
                )
        for key in (*SCALAR_MIN_KEYS, *SCALAR_MAX_KEYS):
            if key in final_info:
                metrics[f"final_{key}"].append(
                    final_info[key][selected_mask].detach().float().view(-1).mean().item()
                )
        if "success_once" in metrics:
            # Treat task success as "achieved at any point in the episode". With
            # ignored terminations, the final frame can drift out of success.
            metrics["success"].append(metrics["success_once"][-1])
        for value in bool_once.values():
            value[done_mask] = False
        for value in scalar_min.values():
            value[done_mask] = float("inf")
        for value in scalar_max.values():
            value[done_mask] = -float("inf")
        if completed >= args.num_eval_episodes:
            break

    result = {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in metrics.items()
    }
    result["episodes"] = float(completed)
    return result

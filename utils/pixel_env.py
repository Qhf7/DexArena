"""ManiSkill environment builders for the image-based agents.

The same environments as the state-based builders in :mod:`utils.train_utils`,
built with ``obs_mode="rgb"`` and the cameras of one
:class:`~utils.pixel_cameras.PixelObsConfig`, so collection, eval and offline
rendering see identical cameras.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from utils.pixel_cameras import PixelObsConfig, shrink_sensors_for
from utils.train_utils import (
    apply_optional_env_kwargs,
    eval_video_output_dir,
    maybe_apply_gpu_id,
    resolve_sim_backend,
)


_PRIVATE_TEMPDIR_SET = False


def _use_private_tempdir() -> None:
    """Point ``tempfile`` at a per-user directory.

    Several tasks stage assets under ``tempfile.gettempdir()`` and rebuild the
    directory on every construction; on a shared node a tree staged by another
    user cannot be deleted (``/tmp`` is sticky) and the task fails with
    ``PermissionError``.
    """
    global _PRIVATE_TEMPDIR_SET
    if _PRIVATE_TEMPDIR_SET:
        return
    import getpass
    import tempfile
    from pathlib import Path

    path = Path(tempfile.gettempdir()) / f"robodex_{getpass.getuser()}"
    path.mkdir(mode=0o700, exist_ok=True)
    tempfile.tempdir = str(path)
    _PRIVATE_TEMPDIR_SET = True


def install_pixel_env(env_id: str, *, reward_info_key: str | None = None) -> type:
    """Register, in place of ``env_id``'s task class, the adapted subclass the
    pixel agents train and evaluate through (``utils.task_adapters``). The task
    files themselves stay as every other algorithm sees them. Idempotent; call
    before ``gym.make``.
    """
    from mani_skill.utils.registration import REGISTERED_ENVS

    from utils.task_adapters import adapt

    _use_private_tempdir()
    entry = REGISTERED_ENVS[env_id]
    base_cls = entry.cls
    if "_pixel_base" in base_cls.__dict__:
        base_cls = base_cls._pixel_base
    entry.cls = adapt(env_id, base_cls, reward_info_key=reward_info_key)
    return entry.cls


def _pixel_env_kwargs(args, config: PixelObsConfig, *, max_episode_steps: int) -> dict[str, Any]:
    sim_backend = resolve_sim_backend(args)
    env_kwargs: dict[str, Any] = {
        "obs_mode": "rgb",
        "sim_backend": sim_backend,
        # Rendering the cameras needs a GPU rasterizer.
        "render_backend": maybe_apply_gpu_id(
            "sapien_cuda", args.gpu_id, {"sapien_cuda", "gpu", "cuda"}
        ),
        "reward_mode": args.reward_mode,
        "max_episode_steps": int(max_episode_steps),
        # Shrink the cameras the robot mounts on itself; the observation
        # contract drops their pixels anyway.
        "sensor_configs": config.sensor_configs(
            shrink_sensors=shrink_sensors_for(args.env_id)
        ),
    }
    apply_optional_env_kwargs(args, env_kwargs)
    return env_kwargs


class SuccessAsSparseReward(gym.Wrapper):
    """Sparse reward := this step's ``info["success"]``, the termination signal.

    Several stage envs pay their sparse reward from a latched stage flag while
    ``success`` reads a live or debounced condition, so reward can outlive the
    episode boundary (DrawerPlugOpen returns 5 per success). PPO opts in so
    reward and termination agree.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, info["success"].to(reward.dtype), terminated, truncated, info


def make_pixel_train_env(
    args, config: PixelObsConfig, *, vector_env_cls=ManiSkillVectorEnv, env_wrappers=()
):
    """Vectorized collection env; sub-envs auto-reset on termination/truncation.

    ``vector_env_cls`` swaps in a ``ManiSkillVectorEnv`` subclass (PPO passes
    ``DemoInitManiSkillVectorEnv`` so auto-resets can restore a demo state).
    """
    install_pixel_env(args.env_id, reward_info_key=getattr(args, "reward_info_key", None))
    env_kwargs = _pixel_env_kwargs(args, config, max_episode_steps=args.max_episode_steps)
    env_kwargs["render_mode"] = None
    num_envs = 1 if env_kwargs["sim_backend"] == "physx_cpu" else args.num_envs

    env = gym.make(args.env_id, num_envs=num_envs, reconfiguration_freq=0, **env_kwargs)
    for wrapper in env_wrappers:
        env = wrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    # Terminations are honored: an episode ends on its first success frame, the
    # same convention the recorded demos carry.
    env = vector_env_cls(env, num_envs, record_metrics=True)
    return env, num_envs, env_kwargs


def _make_scene_background_env(args, scene_background: str, num_envs: int, env_kwargs: dict[str, Any]):
    """Eval env on a native RoboCasa surface (docs/howto_cross_scene_eval.md).

    Only the background is installed (not `make_scene_eval_env`, which also
    swaps the camera rig), so the cameras stay those of `env_kwargs`.
    """
    import json
    from pathlib import Path

    from mani_skill.utils.registration import REGISTERED_ENVS

    from utils.dex_scene_eval import install_eval_background

    original_class = REGISTERED_ENVS[args.env_id].cls
    try:
        manifest = install_eval_background(args.env_id, scene_background)
        env = gym.make(
            args.env_id,
            num_envs=num_envs,
            reconfiguration_freq=args.eval_reconfiguration_freq,
            **env_kwargs,
        )
    finally:
        # A later gym.make in this process must build the original task.
        REGISTERED_ENVS[args.env_id].cls = original_class
    if getattr(args, "scene_keep_original_collision", False):
        import sapien

        if args.eval_reconfiguration_freq:
            raise ValueError("scene_keep_original_collision needs eval_reconfiguration_freq=0: "
                             "a reconfigure rebuilds the scene with the table disabled again.")
        table = env.unwrapped.scene.actors["table-workspace"]
        for entity in table._objs:
            body = entity.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
            for shape in body.get_collision_shapes():
                # The groups the native clones get in dex_scene_eval._clone_collision.
                shape.set_collision_groups([1, 1, 0, 0])
        manifest["original_table"] = "render_disabled_collision_RESTORED_diagnostic"
        print("[scene] diagnostic: original table collision restored (still not rendered)")
    print(f"[scene] {args.env_id}: background={scene_background} ({manifest['label']}), "
          f"{manifest['render_entity_count']} scene entities, "
          f"{len(manifest['hidden_background_entities'])} props hidden")
    manifest_path = getattr(args, "scene_manifest_path", None)
    if manifest_path:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
    return env


def make_pixel_eval_env(args, config: PixelObsConfig, run_name: str, *, env_wrappers=()):
    install_pixel_env(args.env_id, reward_info_key=getattr(args, "reward_info_key", None))
    env_kwargs = _pixel_env_kwargs(args, config, max_episode_steps=args.num_eval_steps)
    capture_video = bool(args.capture_video and torch.cuda.is_available())
    env_kwargs["render_mode"] = "rgb_array" if capture_video else None
    num_envs = max(1, min(args.num_eval_envs, args.num_eval_episodes))
    if env_kwargs["sim_backend"] == "physx_cpu":
        num_envs = 1
    if capture_video:
        # RecordEpisode tiles all `num_envs` frames into one image and buffers
        # the whole video in RAM; size the camera accordingly.
        from utils.train_utils import video_camera_override

        env_kwargs["human_render_camera_configs"] = video_camera_override(
            num_envs, args.num_eval_steps
        )
        print(
            f"[video] {num_envs} eval envs -> human render camera "
            f"{env_kwargs['human_render_camera_configs']['width']}x"
            f"{env_kwargs['human_render_camera_configs']['height']} per env"
        )

    scene_background = getattr(args, "scene_background", None)
    if scene_background is None:
        env = gym.make(
            args.env_id,
            num_envs=num_envs,
            reconfiguration_freq=args.eval_reconfiguration_freq,
            **env_kwargs,
        )
    else:
        env = _make_scene_background_env(args, scene_background, num_envs, env_kwargs)
    for wrapper in env_wrappers:
        env = wrapper(env)
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
        env, num_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True
    )
    return env, env_kwargs

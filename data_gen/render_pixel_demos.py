"""Render recorded LeRobot demonstrations into an image dataset.

Each recorded environment state is pinned into the simulator and the policy
cameras are captured; nothing is simulated forward. The recordings store both
``observation.environment_state`` and ``next.observation.environment_state``,
so ``o`` and ``o'`` are each rendered from their own recorded state. Frames per
episode are ``T + 1`` (states ``0..T-1`` plus the terminal next state), matching
``utils/pixel_dataset.py``.

Example::

    python -m data_gen.render_pixel_demos \\
      --demo-path /mnt/ceph2/DexArena/data/CabinetPouringOpenCabinet-Sharpa-v1 \\
      --task-name CabinetPouringOpenCabinet-Sharpa-v1 \\
      --env-id CabinetPouringOpenCabinet-Sharpa-v1 \\
      --robot-uids panda_sharpa_hand_left_angled_mobile_base \\
      --control-mode pd_joint_pos \\
      --camera-preset cabinet_fixed --image-size 128 \\
      --out data/cabinet_door_pixels/demos.h5 --save-video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401

import tasks  # noqa: F401  (registers every DexArena env)
from data_gen.recordings.recording_integrity import RECORDING_CONTEXT_KEY
from train.train_configs.rlpd_pixels_config import Args
from utils.pixel_cameras import (
    shrink_sensors_for,
    CAMERA_PRESETS,
    pixel_obs_config_from_args,
)
from utils.demo_dataset import unflatten_env_state
from utils.pixel_dataset import PixelEpisodeWriter
from utils.pixel_env import install_pixel_env
from utils.pixel_observation import (
    assert_obs_contract,
    extract_pixel_obs,
    report_env_obs_keys,
)


ENV_STATE = "observation.environment_state"
NEXT_ENV_STATE = "next.observation.environment_state"


def sync_joint_controller_state(base: Any) -> None:
    """Point joint-controller targets at qpos after pinning a recorded state."""
    agent = getattr(base, "agent", None)
    controllers = getattr(getattr(agent, "controller", None), "controllers", None)
    if agent is None or not isinstance(controllers, dict):
        return
    qpos = agent.robot.get_qpos()
    for controller in controllers.values():
        indices = torch.as_tensor(
            controller.active_joint_indices, dtype=torch.long, device=qpos.device
        ).reshape(-1)
        current = qpos.index_select(1, indices).clone()
        for attr in ("_start_qpos", "_target_qpos"):
            if hasattr(controller, attr):
                setattr(controller, attr, current.clone())


def restore_elapsed_steps(base: Any, context: dict[str, np.ndarray]) -> None:
    """Rewind the episode clock the recorder saved alongside the sim state."""
    if "elapsed_steps" not in context:
        raise RuntimeError(f"Recorded {RECORDING_CONTEXT_KEY} has no elapsed_steps field.")
    target = getattr(base, "_elapsed_steps", None)
    if target is None:
        raise RuntimeError("Environment has no _elapsed_steps clock to restore.")
    restored = torch.as_tensor(context["elapsed_steps"], device=base.device)
    if torch.is_tensor(target):
        target.copy_(restored.to(dtype=target.dtype).reshape_as(target))
    else:
        base._elapsed_steps = restored


def set_initial_state(env: gym.Env, flat_state: np.ndarray, names: list[str]) -> None:
    """Pin a recorded (flattened) environment state into the simulator."""
    base = env.unwrapped
    template = base.get_state_dict()
    recorded = unflatten_env_state(flat_state, names)
    context = recorded.pop(RECORDING_CONTEXT_KEY, None)
    prepared: dict[str, Any] = {}
    dropped: list[str] = []
    for category, values in recorded.items():
        reference = template.get(category)
        if torch.is_tensor(reference):
            # A flat tensor hung straight off the state dict (CabinetPour's latch
            # vector) was flattened as {"value": ...}; set_state_dict unwraps it.
            prepared[category] = {
                name: torch.as_tensor(value, dtype=reference.dtype, device=base.device)
                for name, value in values.items()
            }
            continue
        if reference is None:
            dropped.extend(f"{category}/{name}" for name in values)
            continue
        prepared[category] = {}
        for name, value in values.items():
            if name not in reference:
                dropped.append(f"{category}/{name}")
                continue
            dtype = reference[name].dtype if torch.is_tensor(reference[name]) else torch.float32
            prepared[category][name] = torch.as_tensor(value, dtype=dtype, device=base.device)
    if dropped:
        raise RuntimeError(f"Recorded state has fields absent from the env: {sorted(dropped)}.")
    base.set_state_dict(prepared)
    if context is not None:
        restore_elapsed_steps(base, context)
    sync_joint_controller_state(base)


def load_episodes(
    root: Path,
    task_name: str | None,
    task_index: int | None,
    limit: int | None,
    skip: int = 0,
):
    """Group a LeRobot dataset into per-episode arrays, filtered by task.

    The door-open directory holds 62 recorded demos (task
    ``CabinetPouringOpenCabinet-Sharpa-v1``) merged with 100 BC rollouts (task
    ``CabinetPouringOpenCabinet-Sharpa-v1``); the filter is what keeps the two apart.
    """
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    features = info.get("features", {})
    state_names = list((features.get(ENV_STATE) or {}).get("names") or [])
    if not state_names:
        raise RuntimeError(
            f"{root} has no `{ENV_STATE}` feature; demos cannot be re-rendered "
            "without recorded environment states."
        )
    if NEXT_ENV_STATE not in features:
        raise RuntimeError(
            f"{root} has no `{NEXT_ENV_STATE}`. Without it the terminal frame of "
            "each episode cannot be rendered, and under a sparse reward that is "
            "the one transition whose o' carries signal."
        )

    wanted_index = task_index
    if task_name is not None:
        tasks = [
            json.loads(line)
            for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        matches = [entry["task_index"] for entry in tasks if entry["task"] == task_name]
        if not matches:
            raise RuntimeError(
                f"Task {task_name!r} not in {root/'meta'/'tasks.jsonl'}; have "
                f"{[entry['task'] for entry in tasks]}"
            )
        wanted_index = matches[0]

    columns = [
        "episode_index",
        "frame_index",
        "action",
        ENV_STATE,
        NEXT_ENV_STATE,
        "next.reward",
        "next.terminated",
        "next.truncated",
        "next.success",
    ]
    if "task_index" in features or wanted_index is not None:
        columns.append("task_index")
    if "episode.seed" in features:
        columns.append("episode.seed")
    available = set(features) | {"episode_index", "frame_index", "index", "task_index"}
    columns = [name for name in columns if name in available]

    table = pq.read_table(sorted((root / "data").rglob("*.parquet")), columns=columns)
    data = table.to_pydict()

    def column(name, dtype):
        if name not in data:
            return None
        return np.asarray(
            [np.asarray(value).reshape(-1) for value in data[name]], dtype=dtype
        )

    episode_index = column("episode_index", np.int64).reshape(-1)
    task_column = column("task_index", np.int64)
    order = np.argsort(
        np.asarray(data["frame_index"], dtype=np.int64).reshape(-1), kind="stable"
    )
    order = order[np.argsort(episode_index[order], kind="stable")]

    fields = {
        "action": column("action", np.float32),
        "state": column(ENV_STATE, np.float32),
        "next_state": column(NEXT_ENV_STATE, np.float32),
        "reward": column("next.reward", np.float32),
        "terminated": column("next.terminated", bool),
        "truncated": column("next.truncated", bool),
        "success": column("next.success", bool),
        "seed": column("episode.seed", np.int64),
    }

    episodes = []
    matched = 0
    for index in dict.fromkeys(episode_index[order].tolist()):
        rows = order[episode_index[order] == index]
        if wanted_index is not None and task_column is not None:
            if int(np.ravel(task_column[rows[0]])[0]) != int(wanted_index):
                continue
        matched += 1
        if matched <= skip:
            continue
        episode = {name: (value[rows] if value is not None else None) for name, value in fields.items()}
        episode["index"] = int(index)
        episodes.append(episode)
        if limit is not None and len(episodes) >= limit:
            break
    return episodes, state_names, info


def build_args(cli) -> Args:
    return Args(
        env_id=cli.env_id,
        robot_uids=cli.robot_uids,
        control_mode=cli.control_mode,
        reward_mode=cli.reward_mode,
        gpu_id=cli.gpu_id,
        image_size=cli.image_size,
        camera_preset=cli.camera_preset,
        num_envs=1,
        max_episode_steps=cli.max_episode_steps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task-name", default=None, help="keep only this LeRobot task")
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="skip the first N matching episodes (with --append: render in chunks)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "continue an existing --out file instead of overwriting it. One "
            "process per chunk: a single long render exhausts Vulkan resources "
            "on lyg0383 (getMemoryFdKHR / createFenceUnique failures)."
        ),
    )
    parser.add_argument("--env-id", default="CabinetPouringOpenCabinet-Sharpa-v1")
    parser.add_argument("--robot-uids", default="panda_sharpa_hand_left_angled_mobile_base")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument(
        "--sim-backend",
        default="physx_cuda",
        choices=("physx_cuda", "physx_cpu"),
        help=(
            "Physics backend for the replay env. Rendering never steps physics "
            "-- it restores a recorded state and takes a picture -- so this only "
            "decides whether SAPIEN builds a BATCHED render system. Use "
            "physx_cpu for any task whose recorded frames change a render "
            "body's visibility, e.g. CabinetPouringPickBottle-Sharpa-v1, whose "
            "terminal frame is stored in the root-fixed bottle representation: "
            "restoring it flips visibility, and the batched renderer refuses "
            "with \"Modifying a scene ... is not allowed after creating the "
            "batched render system\". Recording and replay never hit this because "
            "they already run on CPU. Otherwise prefer physx_cuda: on lyg0383 "
            "the non-batched (physx_cpu) render path dies after a few thousand "
            "captures with \"vk::Device::createFenceUnique: "
            "ErrorOutOfHostMemory\"."
        ),
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--camera-preset", default="cabinet_fixed", choices=sorted(CAMERA_PRESETS))
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--save-video", action="store_true", help="P0 artifact for episode 0")
    parser.add_argument("--video-dir", default="videos/pixel_demos")
    cli = parser.parse_args(argv)

    args = build_args(cli)
    config = pixel_obs_config_from_args(args)
    print(config.summary())

    root = Path(cli.demo_path).expanduser()
    episodes, state_names, info = load_episodes(
        root, cli.task_name, cli.task_index, cli.limit, cli.skip
    )
    print(
        f"[demos] {len(episodes)} episodes from {root.name} "
        f"(filter: {cli.task_name or cli.task_index or 'none'}, skip {cli.skip})"
    )
    if not episodes:
        raise SystemExit("No episodes matched the filter.")

    install_pixel_env(cli.env_id)
    env = gym.make(
        cli.env_id,
        num_envs=1,
        obs_mode="rgb",
        robot_uids=cli.robot_uids,
        control_mode=cli.control_mode,
        sim_backend=cli.sim_backend,
        # Always the GPU rasterizer: only the PHYSICS backend moves, so the
        # pixels stay the ones online collection will produce.
        render_backend="sapien_cuda",
        reward_mode=cli.reward_mode,
        max_episode_steps=cli.max_episode_steps,
        # Each demo pins its own recorded state, but the *scene* (which cabinet
        # asset, where the trolley sits) is chosen at reconfiguration, so it has
        # to be rebuilt per episode from that episode's seed.
        reconfiguration_freq=1,
        sensor_configs=config.sensor_configs(
            shrink_sensors=shrink_sensors_for(cli.env_id)
        ),
    )
    base = env.unwrapped
    control_mode = str(base.control_mode)

    raw, _ = env.reset(seed=0)
    print(report_env_obs_keys(raw, config))
    probe = extract_pixel_obs(raw, config)
    assert_obs_contract(probe, config)
    proprio_dim = int(probe["proprio"].shape[1])
    print(f"[render] proprio_dim={proprio_dim} control_mode={control_mode}")

    writer = PixelEpisodeWriter(
        cli.out,
        config,
        control_mode=control_mode,
        env_id=cli.env_id,
        robot_uids=cli.robot_uids,
        compression=None if cli.compression == "none" else cli.compression,
        append=cli.append,
    )
    first_episode = writer.num_episodes

    def capture() -> dict[str, np.ndarray]:
        obs = extract_pixel_obs(base.get_obs(), config)
        return {key: value[0].detach().cpu().numpy() for key, value in obs.items()}

    written = 0
    successes = 0
    video_frames: list[np.ndarray] = []
    try:
        for episode in episodes:
            seed = (
                int(np.ravel(episode["seed"][0])[0])
                if episode["seed"] is not None
                else episode["index"]
            )
            env.reset(seed=seed)
            num_transitions = int(episode["action"].shape[0])

            frames: dict[str, list[np.ndarray]] = {
                key: [] for key in (*config.camera_keys, "proprio")
            }
            for step in range(num_transitions):
                set_initial_state(env, episode["state"][step], state_names)
                for key, value in capture().items():
                    frames[key].append(value)
            # Terminal frame: the recorded o' of the last transition.
            set_initial_state(env, episode["next_state"][num_transitions - 1], state_names)
            for key, value in capture().items():
                frames[key].append(value)

            if cli.save_video and not video_frames:
                video_frames = [
                    np.concatenate(
                        [frames[key][t] for key in config.camera_keys], axis=1
                    )
                    for t in range(len(frames["proprio"]))
                ]

            success = bool(episode["success"].any()) if episode["success"] is not None else False
            writer.add_episode(
                observations={key: np.stack(values, axis=0) for key, values in frames.items()},
                actions=episode["action"],
                rewards=episode["reward"].reshape(-1),
                # DONE_CONVENTION="terminated_only": a time-limit truncation is
                # not a terminal state and must not cut the TD backup.
                dones=episode["terminated"].reshape(-1),
                truncated=episode["truncated"].reshape(-1),
                success=success,
            )
            written += 1
            successes += int(success)
            if written % 10 == 0 or written == len(episodes):
                print(
                    f"[render] {written}/{len(episodes)} episodes, "
                    f"{successes} successful ({successes / written:.1%})"
                )
    finally:
        writer.close()
        env.close()

    if video_frames:
        from mani_skill.utils.visualization.misc import images_to_video

        Path(cli.video_dir).mkdir(parents=True, exist_ok=True)
        images_to_video(
            video_frames, cli.video_dir, f"{cli.env_id}_demo0", fps=20, verbose=False
        )
        print(f"[render] P0 artifact: {cli.video_dir}/{cli.env_id}_demo0.mp4")

    print(
        f"[render] wrote {written} episodes ({successes} successful) to {cli.out} "
        f"as episodes {first_episode}..{writer.num_episodes - 1}; file now holds "
        f"{writer.num_episodes}\n"
        f"[render] camera_config_hash={config.config_hash()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

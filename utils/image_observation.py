from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


BASE_CAMERA_ALIASES = ("base_camera", "base_0_rgb", "head_camera")
# The published rig names one wrist camera per arm; a single-arm task has exactly one of
# these, a dual-arm task both (utils/dex_camera_rigs.py).
LEFT_WRIST_CAMERA_ALIASES = ("left_wrist_camera", "left_wrist_0_rgb")
RIGHT_WRIST_CAMERA_ALIASES = ("right_wrist_camera", "right_wrist_0_rgb")
# Any single wrist view, whichever arm: the rig names it by side, older datasets used
# task_wrist_camera, and the robot's own runtime "wrist_camera" is the last resort.
WRIST_CAMERA_ALIASES = (
    "task_wrist_camera", "right_wrist_camera", "left_wrist_camera",
    "hand_camera", "left_wrist_0_rgb", "wrist_camera",
)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _squeeze_single_env(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    return array


def _normalize_rgb(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        for key in ("rgb", "Color", "color", "image"):
            if key in value:
                return _normalize_rgb(value[key])
        raise KeyError(f"Camera entry has no RGB value. Keys: {sorted(value)}")

    image = _squeeze_single_env(value)
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC or CHW RGB image, got shape {image.shape}.")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] < 3:
        raise ValueError(f"Expected at least three image channels, got shape {image.shape}.")
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _find_camera_entry(observation: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    containers = []
    for key in ("sensor_data", "observation", "images", "image"):
        value = observation.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    containers.append(observation)

    for container in containers:
        for alias in aliases:
            if alias in container:
                return container[alias]
    for container in containers[:-1]:
        for value in container.values():
            if not isinstance(value, Mapping):
                continue
            for alias in aliases:
                if alias in value:
                    return value[alias]

    raise KeyError(
        f"Could not find camera aliases {aliases} in observation keys "
        f"{sorted(str(key) for key in observation)}."
    )


def extract_camera_rgb(
    observation: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> np.ndarray:
    return _normalize_rgb(_find_camera_entry(observation, aliases))


def _qpos_from_observation(observation: Mapping[str, Any]) -> np.ndarray | None:
    agent = observation.get("agent")
    if isinstance(agent, Mapping) and "qpos" in agent:
        return _squeeze_single_env(agent["qpos"]).astype(np.float32, copy=False)
    if "qpos" in observation:
        return _squeeze_single_env(observation["qpos"]).astype(np.float32, copy=False)
    return None


def _qpos_from_env(env: Any) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    agent = getattr(base_env, "agent", None)
    robot = getattr(agent, "robot", None)
    if robot is None:
        raise ValueError("Environment does not expose `unwrapped.agent.robot`.")
    return _squeeze_single_env(robot.get_qpos()).astype(np.float32, copy=False)


def extract_franka_proprio(
    observation: Mapping[str, Any],
    *,
    env: Any | None = None,
) -> np.ndarray:
    qpos = _qpos_from_observation(observation)
    if qpos is None:
        if env is None:
            raise KeyError("Observation has no agent qpos and no environment was provided.")
        qpos = _qpos_from_env(env)

    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.size < 9:
        raise ValueError(f"Expected Panda qpos with at least 9 values, got {qpos.size}.")
    gripper_width = np.asarray([qpos[-2:].sum()], dtype=np.float32)
    return np.concatenate([qpos[:7], gripper_width]).astype(np.float32, copy=False)


def controller_joint_order(env: Any) -> list[str]:
    """Joint names in the order the active controller consumes them.

    A ``pd_joint_pos`` action on a multi-controller agent (arm + hand) is the
    concatenation of each sub-controller's joint targets, so this is also the order
    in which a VLA state should read the joints back: state[i] and action[i] then
    refer to the same joint.
    """
    base_env = getattr(env, "unwrapped", env)
    controller = base_env.agent.controller
    subs = getattr(controller, "controllers", None)
    if isinstance(subs, Mapping):
        return [joint.name for sub in subs.values() for joint in sub.joints]
    return [joint.name for joint in controller.joints]


def extract_controller_qpos(observation: Mapping[str, Any] | None, *, env: Any) -> np.ndarray:
    """Full joint state (every actuated joint) in controller order.

    The dexterous-hand embodiments have no "gripper width"; their proprio is the
    raw joint vector, reordered from the articulation's own joint order into the
    controller's so it lines up with the action vector.
    """
    base_env = getattr(env, "unwrapped", env)
    robot = base_env.agent.robot
    qpos = _squeeze_single_env(robot.get_qpos()).astype(np.float32, copy=False).reshape(-1)
    active = [joint.name for joint in robot.get_active_joints()]
    order = controller_joint_order(env)
    missing = [name for name in order if name not in active]
    if missing:
        raise ValueError(f"Controller joints {missing} are not active joints of the robot.")
    index = [active.index(name) for name in order]
    return qpos[index].astype(np.float32, copy=False)


def extract_goal_position(env: Any) -> np.ndarray:
    """World-frame goal xyz for tasks whose goal is not inferable from the views."""
    goal_site = getattr(getattr(env, "unwrapped", env), "goal_site", None)
    if goal_site is None:
        raise ValueError("Environment exposes no `goal_site` to read a goal position from.")
    return _squeeze_single_env(goal_site.pose.p).astype(np.float32, copy=False)


def extract_proprio(
    observation: Mapping[str, Any],
    *,
    env: Any | None = None,
    proprio_kind: str = "franka_gripper",
) -> np.ndarray:
    """The policy-facing proprio vector for an embodiment (see TaskProfile.proprio_kind)."""
    if proprio_kind == "franka_gripper":
        return extract_franka_proprio(observation, env=env)
    if proprio_kind == "controller_qpos":
        if env is None:
            raise ValueError("`controller_qpos` proprio needs the environment for the joint order.")
        return extract_controller_qpos(observation, env=env)
    raise ValueError(f"Unknown proprio_kind {proprio_kind!r}.")


def extract_franka_vla_observation(
    observation: Mapping[str, Any],
    *,
    env: Any | None = None,
    prompt: str,
    include_goal: bool = False,
    proprio_kind: str = "franka_gripper",
) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise TypeError(
            "VLA recording requires a dictionary observation. "
            f"Got {type(observation).__name__}; create the environment with obs_mode='rgb'."
        )
    state = extract_proprio(observation, env=env, proprio_kind=proprio_kind)
    if include_goal:
        if env is None:
            raise ValueError("`include_goal` needs the environment to read the goal position.")
        state = np.concatenate([state, extract_goal_position(env)]).astype(np.float32, copy=False)
    return {
        "observation.state": state,
        "observation.images.base_0_rgb": extract_camera_rgb(
            observation,
            BASE_CAMERA_ALIASES,
        ),
        "observation.images.left_wrist_0_rgb": extract_camera_rgb(
            observation,
            WRIST_CAMERA_ALIASES,
        ),
        "task": str(prompt),
    }

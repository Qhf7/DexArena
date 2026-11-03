"""The four end effectors every task registers, and the env id / robot uid rule.

An env id is ``<Task>-<Hand>-v1``; the robot uid follows from the hand and the arm side
the task is performed with (``left``, ``right`` or ``both``).
"""
from __future__ import annotations

from mani_skill.utils.registration import register_env

HANDS: tuple[str, ...] = ("Sharpa", "Wuji", "XHand", "Gripper")
SIDES: tuple[str, ...] = ("left", "right", "both")

ROBOT_UIDS: dict[str, dict[str, str]] = {
    "Sharpa": {
        "left": "panda_sharpa_hand_left_angled_mobile_base",
        "right": "panda_sharpa_hand_right_angled_mobile_base",
        "both": "panda_dual_sharpa_hand_angled_mobile_base",
    },
    "Wuji": {
        "left": "panda_wuji_hand_left_angled_mobile_base",
        "right": "panda_wuji_hand_right_angled_mobile_base",
        "both": "panda_dual_wuji_hand_angled_mobile_base",
    },
    "XHand": {
        "left": "panda_xhand_hand_left_angled_mobile_base",
        "right": "panda_xhand_hand_right_angled_mobile_base",
        "both": "panda_dual_xhand_hand_angled_mobile_base",
    },
    "Gripper": {
        "left": "panda_gripper_left_mobile_base",
        "right": "panda_gripper_right_mobile_base",
        "both": "panda_dual_gripper_mobile_base",
    },
}


def env_id(task: str, hand: str) -> str:
    if hand not in HANDS:
        raise ValueError(f"unknown hand {hand!r}; expected one of {HANDS}")
    return f"{task}-{hand}-v1"


def robot_uid(hand: str, side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}; expected one of {SIDES}")
    return ROBOT_UIDS[hand][side]


def task_of(env_id: str) -> str:
    """``DrawerPlugOpenDrawer-Sharpa-v1`` -> ``DrawerPlugOpenDrawer``."""
    task, _hand = _split(env_id)
    return task


def hand_of(env_id: str) -> str:
    """``DrawerPlugOpenDrawer-Sharpa-v1`` -> ``Sharpa``."""
    _task, hand = _split(env_id)
    return hand


def _split(env_id: str) -> tuple[str, str]:
    parts = str(env_id).split("-")
    if len(parts) != 3 or parts[2] != "v1" or parts[1] not in HANDS:
        raise ValueError(f"{env_id!r} is not a <Task>-<Hand>-v1 environment id")
    return parts[0], parts[1]


def register_task(task: str, side: str, max_episode_steps: int, **kwargs):
    """Register ``cls`` under ``<task>-<Hand>-v1`` for every hand, each with its robot uid."""

    def decorator(cls):
        for hand in HANDS:
            register_env(
                env_id(task, hand),
                max_episode_steps=max_episode_steps,
                robot_uids=robot_uid(hand, side),
                **kwargs,
            )(cls)
        return cls

    return decorator

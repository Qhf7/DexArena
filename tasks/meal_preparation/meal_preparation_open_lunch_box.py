"""Stage 1 of ``MealPreparationLong-<Hand>-v1``: open or close the hinged box.

This standalone stage contains the box and the common tabletop scene.  For
``task_mode="open"`` the lid starts closed and the policy must open it past
``open_threshold``.  For ``task_mode="close"`` the lid starts open and the
policy must return it below ``close_threshold``.  The box articulation,
camera, robot layout, and dense/sparse reward logic are inherited from
``MealPreparationBase``.

Success is the first long-horizon milestone, represented by ``is_open`` and
``stage_id`` for the open variant or ``is_closed`` and ``stage_id`` for the
close variant.  The
stage is registered separately for Sharpa, Wuji, Soft, and their supported
mobile-base agents so Stage 1 data can be collected independently and reused
by the full dual-arm task.
"""

from tasks.config.hands import register_task

from tasks.meal_preparation.meal_preparation_long import (
    DEFAULT_CONTROL_MODE,
    OPEN_SUBTASK_ROBOT_UID,
    _OpenCloseBoxTask,
)


# HORIZON, measured against the recorded human demonstrations.
# Collection is not bound by this -- `runner.make_env` overrides it with
# 1e6 -- so the value only ever binds TRAINING and EVALUATION: it is the
# time budget a policy is given. Set below what a person actually takes, it
# truncates every rollout before the task can be finished, and the reported
# success rate is measured on an impossible budget.
# 3/3 successful demos exceeded 120; p95 = 204 steps.



@register_task(
    "MealPreparationOpenLunchBox",
    side="left",
    max_episode_steps=300,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class MealPreparationOpenLunchBoxEnv(_OpenCloseBoxTask):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", OPEN_SUBTASK_ROBOT_UID)
        super().__init__(*args, **kwargs)


"""Standalone task: pick and place the bread.

This standalone stage starts from the output distribution of Stage 2: the box
is open and the round food has already been transferred to the plate.  The
only remaining manipulation is for the right hand to grasp the RoboCasa bread
and place it on the plate.  The common table scene, assets, object-on-plate
geometry, and robot layout are inherited from ``MealPreparationBase``.

Success only requires the bread to remain in the plate region for the shared
20-control-step dwell.  The grasp/lift/transport markers remain available as
diagnostics, but deliberately do not gate this standalone task: they are noisy
proxies for the outcome and used to reject valid placements.  The long-horizon
task keeps its process constraint in ``MealPreparationBase``.
"""

from tasks.config.hands import register_task

from tasks.meal_preparation.meal_preparation_long import (
    DEFAULT_CONTROL_MODE,
    SUBTASK_ROBOT_UID,
    _PickAndPlaceBreadTask,
)


class _StableBreadOnPlateTask(_PickAndPlaceBreadTask):
    """Relax the standalone predicate to the requested physical outcome.

    ``_PickAndPlaceBreadTask.evaluate`` still applies ``_qualify_stage``, so
    ``bread_info["on_plate"]`` must remain true for 20 consecutive control
    steps before success latches.  This keeps both required outcome properties:
    the loaf is geometrically in the plate and it stays there for the dwell.
    """

    def _bread_stage_success_info(self, open_threshold, bread_xy_threshold):
        info = super()._bread_stage_success_info(
            open_threshold,
            bread_xy_threshold,
        )
        info["success_condition"] = info["bread_info"]["on_plate"]
        return info


# HORIZON, measured against the recorded human demonstrations.
# Collection is not bound by this -- `runner.make_env` overrides it with
# 1e6 -- so the value only ever binds TRAINING and EVALUATION: it is the
# time budget a policy is given. Set below what a person actually takes, it
# truncates every rollout before the task can be finished, and the reported
# success rate is measured on an impossible budget.
# 4/4 successful demos exceeded 120; p95 = 321 steps.



@register_task(
    "MealPreparationPlaceBread",
    side="right",
    max_episode_steps=400,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class MealPreparationPlaceBreadEnv(_StableBreadOnPlateTask):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", SUBTASK_ROBOT_UID)
        super().__init__(*args, **kwargs)


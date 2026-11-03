"""Stage 3 of ``DualMealPreparation-v1``: scoop the tomato with the spoon onto the plate.

The scene starts from the accepted output of Stage 2: the box lid is open and
the bread is already on the plate; the tomato sits against the right wall of
the box floor and the serving spoon lies on the table, grip toward the operator.
The policy must pick up the spoon, push it along the box wall under the tomato,
lift, and tip the tomato onto the plate.

Success (``_ball_stage_success_info``): tomato on the plate AND the process
latches -- it started in the box, it was in the spoon's bowl with the bowl
facing up and off the floor (``ever_ball_scooped``), the right hand was on the
spoon handle while it was (``ever_right_hand_used_spoon``), it was carried
(``ever_ball_transport``) and it left the box (``ever_ball_left_box``). A tomato
carried by hand never satisfies the bowl test.

replaces ``meal_preparation_pick_and_place_tomato`` (hand pick).
"""
from pathlib import Path

import numpy as np
import sapien

from tasks.config.hands import register_task
from tasks.meal_preparation.meal_preparation_long import (  # noqa: F401  (the factors stay importable from here)
    DEFAULT_CONTROL_MODE,
    SPOON_LENGTH_FACTOR,
    SPOON_MASS_BEFORE_RESIZE,
    TOMATO_SIZE_FACTOR,
    _ScaledFoodAssetsMixin,
    _build_scaled_spoon_actor,
    ROBOCASA_OBJECT_SCALES,
    SPOON_GRIP_SLEEVE,
    SPOON_LENGTH_SCALE,
    SUBTASK_ROBOT_UID,
    _ScoopTomatoTask,
    _build_robocasa_actor,
    _collision_min_z_under_quat,
    _make_material,
    _configure_task_object_physics,
    _mul_scale3,
    _require_robocasa_actor_asset,
    _set_component_mass,
    object_material,
)


class _ScaledScoopTomatoTask(_ScaledFoodAssetsMixin, _ScoopTomatoTask):
    """S3 with only the spoon length and tomato size changed; the builds live in the base
    module (`_ScaledFoodAssetsMixin`) so the long task uses the very same ones."""



@register_task(
    "MealPreparationScoopTomato",
    side="right",
    max_episode_steps=1200,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class MealPreparationScoopTomatoEnv(_ScaledScoopTomatoTask):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", SUBTASK_ROBOT_UID)
        super().__init__(*args, **kwargs)


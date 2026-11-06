"""Stage 2 of package opening: extract the apple, place it, and release it."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.config.hands import register_task

from tasks.package_opening.package_opening_long import (
    DEFAULT_CONTROL_MODE,
    PackageOpeningSubtaskEnv,
    SUBTASK_ROBOT_UID,
    run_env_cli,
)

ENV_ID = "PackageOpeningExtractApple-Sharpa-v1"
# HORIZON, measured against the recorded human demonstrations.
# Collection is not bound by this -- `runner.make_env` overrides it with
# 1e6 -- so the value only ever binds TRAINING and EVALUATION: it is the
# time budget a policy is given. Set below what a person actually takes, it
# truncates every rollout before the task can be finished, and the reported
# success rate is measured on an impossible budget.
# 2/3 successful demos exceeded 500; p95 = 643 steps.
MAX_EPISODE_STEPS = 700


@register_task(
    "PackageOpeningExtractApple",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="normalized_dense",
    render_mode="rgb_array",
)
class PackageOpeningExtractAppleEnv(PackageOpeningSubtaskEnv):
    """Start with an opened package and complete apple extraction and placement."""

    TARGET_STAGE = 1
    LOAD_PACKAGE = True
    LOAD_APPLE = True
    LOAD_PLATE = True


if __name__ == "__main__":
    run_env_cli(ENV_ID, "videos/package_opening_extract_apple")

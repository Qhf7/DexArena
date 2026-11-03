"""Stage 1 of package opening: open both package lids beyond 90 degrees.

registered on the same dual-arm rig as the long task -- the user
framed the task as "open the box with both hands", and a subtask must be a
subset of the long task, not a different rig. Single-arm builds remain
selectable through ``robot_uids`` (SUPPORTED_ROBOTS is inherited).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.config.hands import register_task

from tasks.package_opening.package_opening_long import (
    DEFAULT_CONTROL_MODE,
    PackageOpeningSubtaskEnv,
    LONG_TASK_ROBOT_UID,
    run_env_cli,
)

ENV_ID = "PackageOpeningOpenPackage-Sharpa-v1"
MAX_EPISODE_STEPS = 500


@register_task(
    "PackageOpeningOpenPackage",
    side="both",
    max_episode_steps=MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="normalized_dense",
    render_mode="rgb_array",
)
class PackageOpeningOpenPackageEnv(PackageOpeningSubtaskEnv):
    """Start with the package closed and open both lids past 90 degrees with both hands."""

    TARGET_STAGE = 0
    LOAD_PACKAGE = True
    LOAD_APPLE = False
    LOAD_PLATE = False


if __name__ == "__main__":
    run_env_cli(ENV_ID, "videos/package_opening_open_package")

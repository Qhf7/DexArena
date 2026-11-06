"""The operator rig: robot platform, arm roots and work table, locked together.

Every long-horizon task and every subtask must present the operator with the
SAME physical setup, or the reachable envelope shifts from task to task and a
motion that works in one family is impossible in another. CabinetPour is the
reference the whole family set is aligned to.

The invariant is stated against the table because the table is what the objects
sit on:

    arm_root_x == table_center_x + ARM_TO_TABLE_CENTER_X          (-1.050)
    arm_root_y == +/- ARM_LATERAL_Y                               (0.220)

Equivalently the arm roots sit 0.445 m OUTSIDE the table's near edge (the table
is 1.210 m deep in x, so near_edge = center - 0.605).

Measured drift that this module exists to prevent:

    family                arm_x    table_center   arm - center    verdict
    CabinetPour          -0.870        +0.180        -1.050       reference
    DrawerPlug           -0.870        +0.180        -1.050       ok
    MealPreparation      -0.870        +0.180        -1.050       ok
    PackageOpening       -0.870        +0.180        -1.050       ok
    office               -1.145        -0.120        -1.025       25 mm off
    juice                -1.145        -0.120        -1.025       25 mm off
    TableClean           -0.615        -0.120        -0.495       555 mm off

Second pass: the relative invariant held for 33 of 37 environments
but split them across TWO absolute anchors -- juice/office/table_clean on the
stock table centre, the other five families on the shifted one. See the
ABSOLUTE anchor block below for why that mattered and what it is now.

Note that a correct rig is NECESSARY but not SUFFICIENT for reachability: juice
had a rig only 25 mm off yet was completely unusable, because its objects were
placed 0.53-0.73 m from the table's near edge while every other family places
them at 0.12-0.33 m. The rig fixes the operator's posture; the object layout
still has to put the task inside the arm's working band (measured 0.54-0.90 m
from the arm root, with CabinetPour's farthest grasped object at 0.901 m).
"""

from __future__ import annotations

# Arm root x minus table centre x. Negative: the arms stand back from the table.
ARM_TO_TABLE_CENTER_X = -1.050

# Lateral offset of each arm root from the platform centreline.
ARM_LATERAL_Y = 0.220

# Depth of the shared work table in x, for the near-edge form of the invariant.
TABLE_DEPTH_X = 1.210

# ---------------------------------------------------------------------------
# ABSOLUTE anchor.
#
# The relative invariant above (arm root 1.050 m behind the table centre) makes
# every task FEEL the same, but it does not pin the rig in world coordinates,
# and two things outside the task layer are expressed in world coordinates:
# the play-space origin of the recording setup (default [-0.4, 0, -0.5]) and
# every task's camera ``look_at`` literals. With juice /
# office / table_clean anchored on the stock table centre (-0.120) and
# CabinetPour / Screwing / DrawerPlug / MealPreparation / PackageOpening on the
# shifted one (+0.180), the operator's body stood 0.30 m differently against
# the table depending on which family was loaded.
#
# CabinetPour + Screwing are the reference, so their absolute numbers are the
# rig: the ManiSkill stock table is pushed +0.30 m away from the operator and
# the trolley stands at -0.870.
WORK_TABLE_X_OFFSET = 0.30

# ManiSkill's TableSceneBuilder builds its table centred here before the shift.
STOCK_TABLE_CENTER_X = -0.12

# Work-table centre and trolley/arm-root x, in world coordinates. Every task in
# the eight long-horizon families must land on exactly these.
# Rounded so the derivation cannot leave a 1e-16 tail that breaks an exact
# comparison against the literals CabinetPour/Screwing were accepted with.
TABLE_CENTER_X = round(STOCK_TABLE_CENTER_X + WORK_TABLE_X_OFFSET, 6)     # +0.18
PLATFORM_ROOT_X = round(TABLE_CENTER_X + ARM_TO_TABLE_CENTER_X, 6)        # -0.87

# Reachability band, measured from the arm root to an object's nearest surface.
# Below the lower bound the arm cannot fold tightly enough; above the upper one
# it cannot reach. CabinetPour spans 0.637 (target container) to 0.901 (bottle).
REACH_MIN = 0.54
REACH_MAX = 0.90


def arm_root_x(table_center_x: float) -> float:
    """The only correct arm-root x for a table centred at ``table_center_x``."""
    return float(table_center_x) + ARM_TO_TABLE_CENTER_X


def table_near_edge_x(table_center_x: float) -> float:
    return float(table_center_x) - 0.5 * TABLE_DEPTH_X


def check_rig(arm_x: float, table_center_x: float, tol: float = 1e-3) -> None:
    """Raise if a scene's platform has drifted off the shared rig."""
    want = arm_root_x(table_center_x)
    if abs(float(arm_x) - want) > tol:
        raise ValueError(
            f"workspace rig drift: arm_root_x={arm_x:+.4f} but the table centred "
            f"at {table_center_x:+.4f} requires {want:+.4f} "
            f"(off by {float(arm_x) - want:+.4f} m). See tasks/config/workspace_rig.py."
        )


# ---------------------------------------------------------------------------
# Shared arm start pose.
#
# The operator's hands map onto whatever posture the arms reset into, so a task
# that starts from a different pose feels different from the first frame. This
# is the pose CabinetPour, Screwing, DrawerPlug and TableClean already use;
# juice, meal_preparation, office and package_opening were falling through to
# the stock Panda "rest" keyframe instead, which sits 0.77 rad away at joint 6
# and 0.50 rad away at joint 4 -- a visibly different wrist attitude.
#
#     reference  [-0.500, -0.200, 0.000, -2.600, 0.000, 2.400, 0.785]
#     stock rest [ 0.000,  0.000, 0.000, -2.094, 0.000, 3.142, 0.785]
# ---------------------------------------------------------------------------
ARM_INIT_QPOS = (-0.5, -0.2, 0.0, -2.6, 0.0, 2.4, 0.7853981633974483)


def apply_arm_init_qpos(agent, qpos):
    """Overwrite only the ARM joints of a reset pose with the shared start pose.

    Hand joints are left exactly as the agent's own reset pose defines them, so
    a task that cups the fingers around an object at reset keeps doing so.
    Works for one arm or two; raises if the arm joint count is not a multiple
    of seven, which would mean the caller passed the wrong agent.
    """
    import torch

    arm_names = list(agent.arm_joint_names)
    if len(arm_names) % len(ARM_INIT_QPOS) != 0:
        raise RuntimeError(
            f"expected a multiple of {len(ARM_INIT_QPOS)} arm joints, "
            f"found {len(arm_names)}: {arm_names}"
        )
    active = [str(j.name) for j in agent.robot.get_active_joints()]
    out = qpos.clone()
    per_arm = len(ARM_INIT_QPOS)
    for slot, name in enumerate(arm_names):
        if name not in active:
            raise RuntimeError(f"arm joint {name!r} is not an active joint")
        out[..., active.index(name)] = float(ARM_INIT_QPOS[slot % per_arm])
    return out


def platform_root_pose():
    """Root pose of the mobile-base trolley, shared by every task family.

    The trolley URDF mounts each arm at ``(0, +-ARM_LATERAL_Y, 0)`` off this
    root, so putting the root here puts both arm roots on the rig.
    """
    import sapien

    return sapien.Pose(p=[PLATFORM_ROOT_X, 0.0, 0.0])


def armless_base_root_pose(side: str = "right"):
    """Root pose for a legacy robot that has NO trolley in its URDF.

    Those robots' root link IS ``panda_link0``, so it has to be placed where a
    trolley-mounted arm root would land, or the operator faces a different rig
    the moment they switch to that build. The missing trolley MESH is a
    separate, visual-only gap.
    """
    import sapien

    y = ARM_LATERAL_Y if str(side) == "left" else -ARM_LATERAL_Y
    return sapien.Pose(p=[PLATFORM_ROOT_X, y, 0.0])


def work_table_pose(table_height: float):
    """Pose that puts a ``TableSceneBuilder`` table on the shared rig.

    ``table_height`` is the builder's own ``table_scene.table_height``; the
    quaternion is the builder's yaw of +90 deg, which the tasks must reproduce
    because they overwrite the whole pose, not just its x.
    """
    import numpy as np
    import sapien

    return sapien.Pose(
        p=[TABLE_CENTER_X, 0.0, -float(table_height)],
        q=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
    )


def set_work_table_pose(env) -> bool:
    """Move ``env.table_scene``'s table onto the shared rig. Returns True if set.

    ``TableSceneBuilder.initialize`` restores its own stock pose on EVERY reset,
    so this has to be called after every ``table_scene.initialize(env_idx)`` as
    well as once at load time. Under GPU simulation the load-time call is a
    no-op (static poses cannot be written before the scene is initialised) and
    the reset-time call does the work.

    This is deliberately unconditional in ``robot_uids``: a table that moves
    with the robot build lets a mirror process and a teleop process that were
    given slightly different ``--robot-uids`` show the operator two different
    tables, 0.30 m apart, with nothing raising.
    """
    table_scene = getattr(env, "table_scene", None)
    if table_scene is None:
        return False
    if getattr(env, "gpu_sim_enabled", False) and not env.scene._gpu_sim_initialized:
        return False
    table_scene.table.set_pose(work_table_pose(table_scene.table_height))
    return True

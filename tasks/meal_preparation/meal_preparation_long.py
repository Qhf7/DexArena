"""MealPreparation long-horizon task and shared scene implementation.

The full task is a dual-arm sequence with three latched stages:

  Stage 1 -- the left arm opens the hinged box.
  Stage 2 -- the right arm picks up the BREAD (the large object) and places it
  on the plate.
  Stage 3 -- the right arm SCOOPS the TOMATO out of the box with the serving
  spoon and puts it on the plate (the spoon stage is back, merged
  from the user-tested probe tasks/_probe_meal_pizza.py).

Stage 3 was a hand pick-and-place for a while because the
296 mm spoon of the first version could not be held against the scoop torque
by rigid finger pads. What made the scoop workable (measured with the sub-step
kinematic harness and confirmed by the operator): a 441 mm spoon gripped at a
fat rubber end (the hand sits ~35 cm behind the bowl), the tomato against the
box's right wall so the scoop is a straight push along that wall, the box 0.16 m
closer to the arms, and the spoon lying along x with the grip toward the
operator. The 68 mm tomato is the size the scoop was tuned on.

The common base in this file owns the table scene, box articulation, RoboCasa
objects, robot-compatible layout, pre-collection locks, observations, process
markers, and physical predicates shared by every stage.  The standalone stage
modules inherit this implementation and initialize their scenes from the
corresponding predecessor distribution, while the dual environment executes
all three stages in one episode.

Each standalone stage is initialized from the predecessor stage's accepted
outcome distribution.  Stage 1 starts with a closed box; Stage 2 starts with
the box open and everything else untouched; Stage 3 starts with the box open
and the bread already on the plate.  The common asset and physics
implementation stays in this module so the standalone stages and the
long-horizon environment use identical geometry and predicates.

Success is process-aware: the box must be opened by the left arm, the bread
must be grasped and carried before it counts as placed, and the tomato must
START inside the box and be grasped and carried out of it -- so opening the box
stays a real prerequisite rather than decoration.  The stage markers are
latched in order so a policy cannot skip an earlier stage by reaching only the
final object poses.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np
import sapien
import sapien.physx as physx
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MEAL_PREP_ASSET_DIR = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "task_object_meshes"
    / "MealPreparation_Assets"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tasks.config.sim_profiles import make_sim_config
except Exception:
    spec = importlib.util.spec_from_file_location(
        "robodex_task_config_sim_profiles",
        REPO_ROOT / "tasks" / "config" / "sim_profiles.py",
    )
    if spec is None or spec.loader is None:
        raise
    sim_profiles = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sim_profiles
    spec.loader.exec_module(sim_profiles)
    make_sim_config = sim_profiles.make_sim_config
from tasks.config.object_materials import object_material
from tasks.config.workspace_rig import (
    PLATFORM_ROOT_X,
    apply_arm_init_qpos,
    armless_base_root_pose,
    platform_root_pose,
    set_work_table_pose,
)
from tasks.config.mass_properties import set_mass as set_realizable_mass
from tasks.config.mass_properties import _pose_matrix, _shape_points
from tasks.robots.panda_sharpa_hand_right import PandaSharpaHandRight
from tasks.robots.panda_wuji_hand_right import PandaWujiHandRight
from tasks.robots.panda_sharpa_hand_right_mobile_base import (
    PandaSharpaHandRightAngledMobileBase,
    PandaSharpaHandRightMobileBase,
)
from tasks.robots.panda_sharpa_hand_left_mobile_base import (
    PandaSharpaHandLeftAngledMobileBase,
    PandaSharpaHandLeftMobileBase,
)
from tasks.robots.panda_wuji_hand_left_mobile_base import PandaWujiHandLeftMobileBase
from tasks.robots.panda_wuji_hand_right_mobile_base import PandaWujiHandRightMobileBase
from tasks.robots.panda_xhand_hand_left_mobile_base import PandaXHandHandLeftMobileBase  # noqa: F401
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.robots.panda_dual_xhand_hand_mobile_base import PandaDualXHandHandMobileBase  # noqa: F401
from tasks.robots.panda_dual_sharpa_hand_mobile_base import (
    PandaDualSharpaHandAngledMobileBase,
    PandaDualSharpaHandMobileBase,
)
from tasks.robots.panda_dual_wuji_hand_mobile_base import PandaDualWujiHandMobileBase


AVAILABLE_MOBILE_BASE_ROBOT_UIDS = {
    "panda_sharpa_hand_left_mobile_base",
    "panda_sharpa_hand_left_angled_mobile_base",
    "panda_sharpa_hand_right_mobile_base",
    "panda_sharpa_hand_right_angled_mobile_base",
    "panda_wuji_hand_left_angled_mobile_base",
    "panda_wuji_hand_right_angled_mobile_base",
    "panda_xhand_hand_left_angled_mobile_base",
    "panda_xhand_hand_right_angled_mobile_base",
    "panda_dual_sharpa_hand_mobile_base",
    "panda_dual_sharpa_hand_angled_mobile_base",
    "panda_dual_wuji_hand_angled_mobile_base",
    "panda_dual_xhand_hand_angled_mobile_base",
    "panda_gripper_left_mobile_base",
    "panda_gripper_right_mobile_base",
    "panda_dual_gripper_mobile_base",
}
MOBILE_BASE_ROBOT_UIDS = AVAILABLE_MOBILE_BASE_ROBOT_UIDS

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.io_utils import load_json
from mani_skill.utils import common, sapien_utils
from tasks.config.hands import register_task
from tasks.scene_builders.table import TableSceneBuilder
from tasks.config.guidance_markers import pop_guidance_marker_flag
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply
from mani_skill.utils.structs.types import Array
from mani_skill.utils.wrappers.record import RecordEpisode


DEXTEROUS_HAND_ROBOT_UIDS = {
    "panda_sharpa_hand_right",
    "panda_wuji_hand_right",
} | AVAILABLE_MOBILE_BASE_ROBOT_UIDS

DUAL_HAND_ROBOT_UIDS = {
    uid
    for uid in AVAILABLE_MOBILE_BASE_ROBOT_UIDS
    if uid.startswith("panda_dual_")
}

# Legacy builds without a trolley in their URDF (their root link IS
# panda_link0) predate the shared rig. They used to
# sit at ManiSkill's stock -0.615, i.e. 555 mm off the rig with the arm root
# INSIDE the table footprint, while the object layout below was already shifted
# to the rig -- so those env ids were unusable rather than merely different.
# They now resolve through _rig_root_pose() like every other build.
DEFAULT_ROBOT_ROOT_POSE = [-0.615, 0, 0]

OPEN_SUBTASK_ROBOT_UID = "panda_sharpa_hand_left_angled_mobile_base"
RIGHT_SUBTASK_ROBOT_UID = "panda_sharpa_hand_right_angled_mobile_base"
# Backward-compatible name used by the Stage 2/3 registration modules.
SUBTASK_ROBOT_UID = RIGHT_SUBTASK_ROBOT_UID
LONG_TASK_ROBOT_UID = "panda_dual_sharpa_hand_angled_mobile_base"
DEFAULT_CONTROL_MODE = "pd_joint_pos"
SUPPORTED_CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")
MOBILE_BASE_ROOT_POSE = [-0.87, 0, 0]
SINGLE_MOBILE_BASE_ROOT_POSE = MOBILE_BASE_ROOT_POSE
DUAL_MOBILE_BASE_ROOT_POSE = MOBILE_BASE_ROOT_POSE
WORK_TABLE_X_OFFSET = 0.30

# Task-object material classes are the only friction choices owned here.  The
# resulting engine coefficients come from tasks/config/object_materials.py;
# robot, fingertip, table, and solver parameters remain untouched.
# Task-object physical specification: (material class, dominant contact pair,
# real-world mass in kg).
#
# The dominant pair says which contact the single PhysX scalar is solved for.
# PhysX combines materials with `average`, so one scalar per surface cannot make
# every pair correct at once (see tasks/config/object_materials.py). Objects that
# are picked up are solved for the fingertip pad; objects that live on the
# tabletop are solved for the tabletop.
#
# Real-world mass references:
#   meal / pizza box body   0.15-0.60 kg (hollow board)
#   box lid                 0.05-0.25 kg
#   ceramic dinner plate    0.30-0.50 kg
#   steel tablespoon        30-80 g
#   bread roll              0.10-0.35 kg
#   meatball / food ball    20-100 g
TASK_OBJECT_MATERIAL_CLASSES = {
    "box": "cardboard",
    "spoon": "metal",
    "plate": "ceramic",
    "bread": "bread",
    "ball_food": "cooked_food",
}

TASK_OBJECT_DOMINANT_PAIR = {
    # "table" for the box body is NOT about sliding: its root is fixed, so the
    # tabletop pair never loads. It is about what goes INSIDE it. The pad solve
    # for cardboard clamps to mu = 0, which would put the bread on a
    # frictionless wall (bread-box pair 0.00); the table solve gives mu = 0.70,
    # so bread-box becomes 0.35 and tomato-box 0.45 against a real ~0.45.
    "box": "table",
    "box_lid": "pad",      # the lid is the part the hand actually grips
    "plate": "table",      # kinematic, only ever contacts the table and food
    "spoon": "pad",        # the tool the hand grips (its rubber grip carries its own class)
    "bread": "pad",
    "ball_food": "pad",
}

# Masses are chosen against each object's ACTUAL modelled size, so the implied
# density also lands in a real range. They were re-set together with the scales
# below: the previous values belonged to objects 2-5x oversized,
# and keeping them would have left a 35 mm cherry tomato weighing 200 g (rho
# ~8900, i.e. steel) and a 100 mm bagel weighing 400 g.
#   box       0.23 m PP food container            0.15 kg
#   box_lid   its lid                             0.06 kg
#   plate     0.26 m ceramic dinner plate         0.35 kg  (unchanged, already real)
#   spoon     0.20 m stainless table spoon        0.06 kg  (unchanged)
#   bread     0.10 m bagel                        0.10 kg
#   ball_food 0.035 m cherry tomato               0.02 kg
TASK_OBJECT_MASS = {
    "box": 0.15,
    "box_lid": 0.06,
    "plate": 0.35,
    "bread": 0.10,
    "ball_food": 0.16,     # 68 mm tomato
    "spoon": 0.08,         # 0.44 m stainless serving spoon with a deep bowl
}

# Object scales, set to REAL sizes. Measured collision AABBs
# before / after, against the real object each one is meant to be:
#
#   object      was        now      real reference
#   spoon      0.944 m -> 0.296 m   serving spoon; raised from 0.200 m after
#                                     the operator reported the hand could not
#                                     hold it -- a 0.20 m spoon's handle is a
#                                     ~10 x 19 mm ribbon, below what the sharpa
#                                     pads can pinch. 0.296 m puts the handle
#                                     at ~14 x 28 mm.
#   plate      0.621 m -> 0.260 m   dinner plate 0.22-0.28 m
#   bread      0.202 m -> 0.100 m   bagel 0.095-0.11 m
#   ball_food  0.084 m -> 0.035 m   cherry tomato 0.03-0.04 m
#   box lid    0.520 m -> 0.299 m   food container lid over a 0.230 m body
#
# The spoon and the food are COUPLED: the point of the tool-using stage is to
# scoop the food with the spoon, and a 0.20 m spoon has a ~0.045 x 0.030 m
# bowl, which an 0.084 m beefsteak tomato cannot sit in. Shrinking the spoon to
# a real one therefore forces the food down to a cherry tomato, which is what
# the ball_food scale below does.
#
# Two things move WITH the spoon and must never be edited independently of it:
# the scoop-contact thresholds (they measure from the spoon's ORIGIN, so they
# are a fraction of its length -- see SPOON_BALL_CONTACT_THRESHOLD) and the
# layout, since a 4.7x smaller spoon no longer spans the gap it used to.

ROBOCASA_OBJECT_CANDIDATES = {
    "plate": ("plate",),
    "spoon": ("spoon",),
    "bread": ("waffle", "bagel", "bread"),
    "ball_food": ("tomato", "egg", "tangerine", "lime", "lemon", "mushroom"),
}

# Prefer flat bread-like food and smaller non-apple round food variants when RoboCasa objects are available.
ROBOCASA_OBJECT_PREFERRED_MODELS = {
    "spoon": ("spoon_16",),
    "bread": ("bagel_5", "bagel_0", "waffle_4", "waffle_0", "bread_9", "bread_3"),
    "ball_food": ("tomato_2", "tomato_8", "tangerine_5", "egg_5", "mushroom_4"),
}

ROBOCASA_OBJECT_REGISTRIES = {
    "plate": (("aigen_objs/plate", 1.65), ("objaverse/plate", 1.35)),
    "spoon": (("objaverse/spoon", 1.5),), # spoon_16 vendored under MealPreparation_Assets; 1.5 = the scale the scoop was tuned at
    "bread": (
        ("objaverse/waffle", (1.00, 1.00, 1.0)),
        ("objaverse/bagel", (0.90, 0.90, 1.0)),
        ("aigen_objs/bread", (0.80, 0.80, 1.0)),
        ("objaverse/bread", (0.70, 0.70, 1.0)),
    ),
    "ball_food": (
        ("objaverse/tomato", 0.90),
        ("objaverse/egg", 0.85),
        ("objaverse/tangerine", 0.90),
        ("objaverse/lime", 0.85),
        ("objaverse/lemon", 0.80),
        ("objaverse/mushroom", 0.90),
    ),
}

# Scale that made each object oversized, kept so the derived thresholds below
# can state what they were tuned against.
ROBOCASA_LEGACY_OBJECT_SCALES = {
    "plate": 2.52,
    "bread": 2.52,
    "ball_food": 1.55,
    "spoon": 1.95,
}
# the spoon stage is back (see the module docstring). The plate grows
# to a 0.31 m platter so the bread (0.10 x 0.11) and the tomato sit on it side
# by side; the tomato is the 68 mm one the scoop harness was tuned on (a 35 mm
# one rolled out of the bowl); the spoon is RoboCasa spoon_16 shortened along
# its length only (SPOON_LENGTH_SCALE) with a round rubber grip at the end.
ROBOCASA_OBJECT_SCALES = {
    "plate": 1.25,      # 0.621 m -> 0.308 m platter
    "bread": 1.25,      # 0.202 m -> 0.100 m
    "ball_food": 1.28,  # 0.084 m -> 0.068 m
    "spoon": 1.95,      # spoon_16 -> 0.519 m before SPOON_LENGTH_SCALE
}

MEAL_PREP_LAYOUT_X_SHIFT = -0.12 + WORK_TABLE_X_OFFSET
MEAL_PREP_BOX_PULL_X = -0.15          # box + tomato toward the arms (scoop reach); -0.16 put the near wall 5 mm inside the 0.54 m band
MEAL_PREP_TABLE_PULL_X = -0.08        # plate + bread toward the arms
BALL_IN_BOX_OFFSET = (0.068, -0.090)  # tomato against the box's right (-y) wall, forward half
SPOON_SPAWN_X = -0.268                # + MEAL_PREP_LAYOUT_X_SHIFT: grip capsule >= 0.54 m from the right arm root after settling (reach band floor; tests/test_workspace_rig.py measures the settled hulls)
SPOON_SPAWN_Y = -0.26                 # between the bread lane (-0.12) and the plate (-0.55)

PLATE_RADIUS = 0.154

# ---- spoon (merged from the scoop probe) ----
# spoon_16's mesh is authored with its length on local z and its bowl normal on
# local -y. It lies ALONG X with the handle toward the arm (asset refquat yawed
# -90 deg): the one layout in which a 0.49 m spoon clears the box, the plate and
# the tomato and keeps its grip inside the reach band. The operator confirmed
# "handle toward me" is the easy grip (docs/meal_prep_spoon_probe_notes_zh.md 3.1d).
SPOON_SPAWN_QUAT = [-0.5, 0.5, 0.5, -0.5]
SPOON_BOWL_LOCAL_NORMAL = [0.0, -1.0, 0.0]
# Shortened along its LENGTH axis only (asset local z) to a 441 mm serving spoon:
# sub-step kinematic scoop harness, tomato at the box's right wall, straight
# push: 0.70 -> 3/7 pitches, 0.85 -> 7/7 (8..30 deg), 1.00 -> 4/7.
SPOON_LENGTH_SCALE = 0.85
# Round rubber grip (36 mm diameter) on the last 90 mm of the handle, sticking
# 25 mm past its end: the operator scoops with the hand far back on the handle.
SPOON_GRIP_SLEEVE = dict(shape="capsule", center=(0.0, -0.019, 0.200), radius=0.018, half_length=0.045,
                         quat=(0.7071068, 0.0, -0.7071068, 0.0), material="rubber")
# Handle centreline (bowl back -> grip end) for "the hand is on the spoon".
SPOON_HANDLE_SEGMENT_LOCAL = ((0.0, -0.019, -0.110), (0.0, -0.019, 0.263))
# Distance from the BOWL CENTRE (measured on the built hulls) to the tomato's
# origin that counts as "in the spoon" (bowl half-width ~53 mm, tomato r 34 mm).
SPOON_BOWL_CONTACT_THRESHOLD = 0.075
SPOON_UP_DOT_THRESHOLD = 0.45           # bowl opening pointing up enough to carry
RIGHT_HAND_SPOON_THRESHOLD = 0.075      # fingertip to the handle segment
APPLE_RADIUS = 0.056

PARTNET_BOX_MODEL_ID = os.environ.get("MEAL_PREP_BOX_MODEL_ID", "100426")
PARTNET_BOX_MODEL_IDS = ("100426", "100658", "100194", "100202", "100197")
PARTNET_BOX_ALIASES_BY_MODEL_ID = {
    "100658": ("100658", "be3f5a83e94251a7c1a0f8bf0959443"),
    "100194": ("100194", "59af77ad277cf2806ae51efdae196e9a"),
    "100426": ("100426", "3e18df9aea04c8cdbc7fb81194b2b6a8"),
    "100202": ("100202", "6c0d654636fc54545c0af2849a181c0"),
    "100197": ("100197", "622026621d781235889463094c9fcbb0"),
}
PARTNET_BOX_CONFIGS = {
    "100658": {
        "inner_bottom_raw_z": -0.305,
        "description": "multi-flap hollow box with a usable bottom surface",
    },
    "100194": {
        "inner_bottom_raw_z": -0.1623,
        "description": "large multi-flap gift box candidate",
    },
    "100426": {
        "inner_bottom_raw_z": -0.18815,
        "description": "stable two-lid box candidate; apple rests on the lower bottom surface",
    },
}
PARTNET_BOX_SCALE = 0.35
PARTNET_STABLE_BOX_MODEL_IDS = {"100426"}
PARTNET_BOX_LID_DRIVE_STIFFNESS = 0.0
PARTNET_BOX_LID_DRIVE_DAMPING = 0.30      # N*m*s/rad
PARTNET_BOX_LID_DRIVE_FORCE_LIMIT = 5.0   # N*m, structural ceiling
# ⚠️ `joint.set_friction` takes a SETTING, not newton-metres (the same trap the
# drawer and the fridge hinge were recalibrated for). Measured breakaway torque
# on THIS hinge, by re-applying qf before every substep (set_qf survives only
# one substep):
#       setting 0.08 -> 0.077 N*m     setting 0.30 -> 0.151 N*m
#       setting 1.00 -> 0.410 N*m
# and measured sag of a lid left at 93 deg for 30 s:
#       0.08 -> -8.78 deg    0.30 -> -0.89 deg    1.00 -> 0.00 deg
#
# At the old 0.08 the lid crossed back under the 90 deg open threshold after
# ~15 s on its own, so every stage that requires "the box is open" became
# impossible partway through an episode -- which is why bread placed on the
# plate stopped counting, and why the box looked shut by the time the operator
# got to it. 1.00 holds it dead still; 0.410 N*m is 2.7 N at the lid edge, i.e.
# still a light push for the arm, and it is a moulded PP container hinge, not
# the cardboard crease the 0.02-0.15 N*m note below was written for.
PARTNET_BOX_LID_DRIVE_FRICTION = 1.00     # setting, = 0.410 N*m breakaway
# A real box crease resists 0.02-0.15 N*m. The previous 1.5 N*m exceeded the
# lid's own gravity torque, so the lid never fell under its own weight and the
# hinge behaved like a brake rather than a fold.
PARTNET_BOX_LID_LOCK_STIFFNESS = 1500.0
PARTNET_BOX_LID_LOCK_DAMPING = 120.0
PARTNET_BOX_LID_LOCK_FORCE_LIMIT = 3000.0
PARTNET_BOX_LID_LOCK_FRICTION = 8.0

PARTNET_BOX_TABLE_CLEARANCE = 0.005
ROBOCASA_TABLE_CLEARANCE = 0.003
BALL_IN_BOX_CLEARANCE = 0.0008
# Procedural food container. Was 2.0, which built a 0.400 m body under a
# 0.520 m lid -- a crate, not a meal box, and the single largest thing on the
# desk. 1.15 gives a 0.230 m body under a 0.299 m lid, i.e. a real meal-prep
# container footprint. The body DEPTH is set separately below because the
# asset's own aspect ratio (0.15 deep per 0.20 wide) is a cube, while a real
# container of that footprint is 0.08-0.12 m deep.
DEXMIMICGEN_BOX_SCALE = 1.15
PROCEDURAL_BOX_WALL = 0.01 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_APPLE_SUPPORT_THICKNESS = 0.0025 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_LX = 0.20 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_LY = 0.20 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_LZ = 0.10 * DEXMIMICGEN_BOX_SCALE   # 0.115 m deep, see above
PROCEDURAL_BOX_LID_LOWER_LX = 0.18 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_LID_LOWER_LY = 0.18 * DEXMIMICGEN_BOX_SCALE
# Lid PANEL thicknesses. These were 0.020 / 0.030 (i.e. 50 mm of lid on a
# 115 mm deep box, 75 mm at the probe's larger scale) -- and the hinge is
# parked at `box_lz + lid_total_lz + clearance`, so the whole lid thickness
# became a hinge STAND-OFF. Closed, the lid hangs off that hinge and covers
# it; OPEN, it swings away and the stand-off is a visible air gap between
# the box and its lid (measured 57.9 mm with the lid at 75 deg). Tightening
# the clearances only bought 1.5 mm of it -- the lid's own thickness is the
# term that matters. A real food-container lid is a few mm of panel.
PROCEDURAL_BOX_LID_LOWER_LZ = 0.004 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_LID_UPPER_LX = 0.26 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_LID_UPPER_LY = 0.26 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_LID_UPPER_LZ = 0.006 * DEXMIMICGEN_BOX_SCALE
# Hinge stand-off from the box. These two constants ARE the visible air gap
# between the box and its lid: the hinge is parked REAR_CLEARANCE behind the
# rear wall and VERTICAL_CLEARANCE above the rim, so a closed lid floats and an
# open one looks detached (measured 4.6 mm / 13.8 mm at scale 1.15).
#
# They were sized as if the lid could swing DOWN through the rim. It cannot:
# no part of the lid lies behind the hinge (the brim's rear edge sits exactly
# at the hinge y), so every point of the lid can only rise as qpos grows. The
# clearance therefore bought nothing and cost the gap. 1 mm * scale is kept so
# the closed lid does not rest in a zero-gap contact against the rim.
# Verified after the change: closed gap 1.2 mm, and the lid still tracks its
# drive target over the full 0 -> 1.85 rad range with no interpenetration.
PROCEDURAL_BOX_LID_REAR_CLEARANCE = 0.001 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_LID_VERTICAL_CLEARANCE = 0.001 * DEXMIMICGEN_BOX_SCALE
PROCEDURAL_BOX_LID_DENSITY = 45.0
PROCEDURAL_BOX_CLOSED_QPOS = 0.06
# Hinge geometry that fills the stand-off between the rim and the revolute axis
# (see _build_procedural_lidded_box). Real container hinges are a few mm of
# knuckle; the leaf is the rear wall carried up to the axis.
PROCEDURAL_BOX_HINGE_KNUCKLE_RADIUS = 0.006
PROCEDURAL_BOX_HINGE_LEAF_THICKNESS = 0.003

# The lid angle that counts as "the box is open". Was 1.05 rad (60 deg): the
# operator asked for a full right angle, both because 60 deg is not visibly
# open and because a lid left at 60 deg falls shut again on its own. Every
# stage that starts from an already-open box seeds its lid from this same
# constant (plus a margin), so raising it moves the entry states with it.
BOX_OPEN_ANGLE = float(np.pi / 2)
# Seeded lid angle for stages that BEGIN with the box open. The margin keeps
# the entry state off the acceptance boundary.
BOX_OPEN_ENTRY_MARGIN = 0.15
BOX_EXIT_MARGIN = 0.025
BALL_IN_BOX_Z_TOLERANCE = 0.004
BOX_MANIPULATION_QPOS_DELTA = 0.01
BOX_HAND_CONTACT_THRESHOLD = 0.34
# Scoop-contact radii, measured from the spoon's ORIGIN (mid-handle) to the
# food's origin, so they are a fraction of the spoon's LENGTH, not an absolute
# hand-scale distance: the accepted values 0.115 / 0.095 were tuned against the
# 4.95 spoon scale. Deriving them keeps that geometry exactly while the spoon
# becomes a real one -- had they stayed absolute, a 0.20 m spoon would satisfy
# "food is within 0.115 m of the spoon" with the food anywhere near the whole
# utensil, and the scoop check would have stopped meaning anything.
# Distance from the BOWL (not the spoon's origin) to the food that counts as
# "the food is in the spoon". Measured on the built asset: bowl radius 25.9 mm,
# cherry tomato radius 17.5 mm, so food actually sitting in the bowl is within
# ~43 mm of the bowl centre. 0.055 leaves a working margin without accepting
# food that is merely somewhere near the spoon.
# "On the plate" acceptance radii, measured from the PLATE'S ORIGIN, so they
# are a fraction of the plate's radius: the accepted 0.19 / 0.26 were tuned
# against the 2.52 plate scale (a 0.62 m plate). Held absolute against a real
# 0.26 m plate they would accept food sitting 0.06-0.13 m OFF the rim, i.e. on
# the bare table, so they are derived from the scale like the spoon radii.
PLATE_BALL_XY_THRESHOLD = round(
    0.19 * ROBOCASA_OBJECT_SCALES["plate"] / ROBOCASA_LEGACY_OBJECT_SCALES["plate"], 4
)
PLATE_BREAD_XY_THRESHOLD = round(
    0.26 * ROBOCASA_OBJECT_SCALES["plate"] / ROBOCASA_LEGACY_OBJECT_SCALES["plate"], 4
)

BREAD_HAND_CONTACT_THRESHOLD = 0.08
BREAD_LIFT_THRESHOLD = 0.025
# Curl fraction that counts as "the hand is holding it", on the rebuilt
# ``_hand_closure`` scale (0 = open, 1 = fully curled). Contact stops the
# fingers short of their commanded target, so a firm grasp of the loaf lands
# near 0.47 and of the tomato near 0.54; 0.30 keeps ~1.6x margin. It never
# stands alone -- the object must also be touched AND lifted.
BREAD_GRASP_CLOSURE_THRESHOLD = 0.30
BREAD_TRANSPORT_DISTANCE = 0.045
BREAD_TRANSPORT_MIN_STEPS = 3

# Width of ``meal_preparation_task_state``: the tomato and bread process blocks
# (9 + 7 fields), the box-opening latch, the two initial heights, the three
# stage latches and the debounce/eval bookkeeping.
MEAL_PREP_TASK_STATE_DIM = 32 # + ever_right_hand_used_spoon
MEAL_PREP_EPISODE_POSES_DIM = 15  # box, bread, ball, plate, spoon (xyz each)

# Tomato pick-and-place (stage 3).  Mirrors the bread gates, with the contact and
# lift thresholds scaled to the smaller object: the tomato measures 35 mm across
# and 20 g against the loaf's much larger footprint, so bread's 80 mm contact
# radius would count a hand that is nowhere near it.
# ``_min_side_tip_distance`` measures FINGERTIP-LINK ORIGIN to OBJECT ORIGIN,
# not surface to surface: the tip link's origin sits inside the fingertip body,
# roughly 38 mm out from the object surface it is pressing on. A tomato being
# genuinely carried therefore reads 56-60 mm, so the original 0.050 could never
# be satisfied while actually holding it -- measured: closure and lift both
# passed for the whole carry while `ball_touched` stayed False, and the stage
# never latched. 0.075 clears the measured 56 mm with margin and stays well
# under the 0.080 the much larger bread uses.
BALL_HAND_CONTACT_THRESHOLD = 0.075
BALL_LIFT_THRESHOLD = 0.020
BALL_GRASP_CLOSURE_THRESHOLD = 0.30
BALL_TRANSPORT_DISTANCE = 0.045
BALL_TRANSPORT_MIN_STEPS = 3

# Gap left under an object seated on the plate by ``_seat_object_on_plate``.
# Just enough that it rests rather than starts interpenetrating; anything more
# and it drops onto a dished surface and skates.
PLATE_SEAT_CLEARANCE = 0.0005


def _iter_existing_roots(paths):
    seen = set()
    for path in paths:
        if path is None:
            continue
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if path.exists():
                yield path
        except PermissionError:
            continue


def _partnet_asset_roots():
    roots = [
        MEAL_PREP_ASSET_DIR / "partnet_mobility" / "dataset",
        os.environ.get("MEAL_PREP_BOX_ASSET_DIR"),
        os.environ.get("PARTNET_MOBILITY_ASSET_DIR"),
    ]
    yield from _iter_existing_roots(roots)


def _find_partnet_box_urdf():
    urdf_names = ("mobility_cvx.urdf", "mobility_fixed.urdf", "mobility.urdf")
    requested_model_ids = [PARTNET_BOX_MODEL_ID]
    requested_model_ids.extend(model_id for model_id in PARTNET_BOX_MODEL_IDS if model_id not in requested_model_ids)
    for root in _partnet_asset_roots():
        candidates = []
        for model_id in requested_model_ids:
            aliases = PARTNET_BOX_ALIASES_BY_MODEL_ID.get(model_id, (model_id,))
            if root.name in aliases:
                candidates.append(root)
            candidates.extend(root / alias for alias in aliases)
        for model_dir in candidates:
            for urdf_name in urdf_names:
                urdf_path = model_dir / urdf_name
                if urdf_path.exists():
                    return urdf_path
    return None


def _partnet_box_pose_z(urdf_path: Path, scale: float = 0.35):
    bbox_path = urdf_path.with_name("bounding_box.json")
    if not bbox_path.exists():
        return 0.071
    try:
        bbox = load_json(bbox_path)
        bbox_min_z = float(bbox["min"][2])
    except Exception:
        return 0.071
    return (-bbox_min_z * scale) + PARTNET_BOX_TABLE_CLEARANCE


def _partnet_model_id_from_path(urdf_path: Path):
    model_name = urdf_path.parent.name
    for model_id, aliases in PARTNET_BOX_ALIASES_BY_MODEL_ID.items():
        if model_name in aliases:
            return model_id
    return model_name


def _rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _yaw_quat(yaw: float):
    half = yaw / 2
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _load_obj_vertices(obj_path: Path):
    vertices = []
    try:
        with open(obj_path, "rt", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z, *_ = line.split()
                    vertices.append([float(x), float(y), float(z)])
    except OSError:
        return None
    if not vertices:
        return None
    return np.array(vertices, dtype=np.float32)


def _partnet_body_link_name(urdf_path: Path):
    semantics_path = urdf_path.with_name("semantics.txt")
    if semantics_path.exists():
        for line in semantics_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[-1] == "box_body":
                return parts[0]
    return None


def _partnet_fixed_rotation_for_link(root, link_name: str | None):
    if link_name is None:
        return np.eye(3, dtype=np.float32)
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        child = joint.find("child")
        if child is None or child.get("link") != link_name:
            continue
        origin = joint.find("origin")
        if origin is None or origin.get("rpy") is None:
            return np.eye(3, dtype=np.float32)
        rpy = np.array([float(v) for v in origin.get("rpy").split()], dtype=np.float32)
        return _rpy_to_matrix(rpy)
    return np.eye(3, dtype=np.float32)


def _partnet_body_vertices(urdf_path: Path):
    body_link = _partnet_body_link_name(urdf_path)
    if body_link is None:
        return None
    root = ET.parse(urdf_path).getroot()
    rotation = _partnet_fixed_rotation_for_link(root, body_link)
    vertices = []
    for link in root.findall("link"):
        if link.get("name") != body_link:
            continue
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            mesh_vertices = _load_obj_vertices(urdf_path.parent / mesh.get("filename"))
            if mesh_vertices is None:
                continue
            origin_xyz = np.zeros(3, dtype=np.float32)
            origin = visual.find("origin")
            if origin is not None and origin.get("xyz") is not None:
                origin_xyz = np.array([float(v) for v in origin.get("xyz").split()], dtype=np.float32)
            vertices.append((mesh_vertices + origin_xyz) @ rotation.T)
    if not vertices:
        return None
    return np.concatenate(vertices, axis=0)


def _partnet_visual_mesh_specs_for_link(urdf_path: Path, link_name: str):
    root = ET.parse(urdf_path).getroot()
    rotation = _partnet_fixed_rotation_for_link(root, link_name)
    specs = []
    for link in root.findall("link"):
        if link.get("name") != link_name:
            continue
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            origin_xyz = np.zeros(3, dtype=np.float32)
            origin = visual.find("origin")
            if origin is not None and origin.get("xyz") is not None:
                origin_xyz = np.array([float(v) for v in origin.get("xyz").split()], dtype=np.float32)
            specs.append(
                {
                    "filename": urdf_path.parent / mesh.get("filename"),
                    "origin_xyz": origin_xyz @ rotation.T,
                }
            )
    return specs


def _partnet_visual_bottom_offset_z(urdf_path: Path):
    vertices = _partnet_body_vertices(urdf_path)
    if vertices is None:
        return 0.0
    return float(-vertices[:, 2].min() * PARTNET_BOX_SCALE)


def _partnet_box_pose_z_from_vertices(urdf_path: Path, scale: float = PARTNET_BOX_SCALE):
    vertices = _partnet_body_vertices(urdf_path)
    if vertices is None:
        return _partnet_box_pose_z(urdf_path, scale)
    return float(-vertices[:, 2].min() * scale + PARTNET_BOX_TABLE_CLEARANCE)


def _partnet_box_bottom_top_z(urdf_path: Path, box_pose_z: float = 0.0, scale: float = PARTNET_BOX_SCALE):
    """Return the inner bottom height in the box's local frame.

    ``box_pose_z`` is kept as a compatibility argument for callers from the
    earlier implementation.  The returned value is deliberately local: all
    placement predicates add the current articulation root z exactly once.
    """
    model_id = _partnet_model_id_from_path(urdf_path)
    if model_id in PARTNET_BOX_CONFIGS:
        return float(PARTNET_BOX_CONFIGS[model_id]["inner_bottom_raw_z"] * scale)

    bottom_mesh = urdf_path.parent / "textured_objs" / "original-1.obj"
    vertices = _load_obj_vertices(bottom_mesh)
    if vertices is None:
        return box_pose_z + 0.07
    root = ET.parse(urdf_path).getroot()
    fixed_joint = None
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            fixed_joint = joint.find("origin")
            break
    rpy = np.zeros(3, dtype=np.float32)
    if fixed_joint is not None and fixed_joint.get("rpy") is not None:
        rpy = np.array([float(v) for v in fixed_joint.get("rpy").split()], dtype=np.float32)
    rotation = _rpy_to_matrix(rpy)
    world_vertices = (vertices @ rotation.T) * scale
    return float(world_vertices[:, 2].min())


def _mesh_aabb(mesh_path: Path):
    vertices = _load_obj_vertices(mesh_path)
    if vertices is None:
        return None
    return vertices.min(axis=0), vertices.max(axis=0)


def _collision_box_specs_for_mesh(urdf_path: Path, origin_xyz, mesh_filename: str):
    mesh_path = urdf_path.parent / mesh_filename
    aabb = _mesh_aabb(mesh_path)
    if aabb is None:
        return None
    mesh_min, mesh_max = aabb
    origin_xyz = np.array(origin_xyz, dtype=np.float32)
    box_min = origin_xyz + mesh_min
    box_max = origin_xyz + mesh_max
    center = (box_min + box_max) / 2.0
    size = np.maximum(box_max - box_min, 1e-4)
    return center, size


def _build_stable_partnet_box_urdf(urdf_path: Path):
    model_id = _partnet_model_id_from_path(urdf_path)
    if model_id not in PARTNET_STABLE_BOX_MODEL_IDS:
        return urdf_path

    cache_dir = MEAL_PREP_ASSET_DIR / "_generated" / "partnet" / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    stable_urdf_path = cache_dir / "mobility_stable.urdf"
    if stable_urdf_path.exists() and stable_urdf_path.stat().st_mtime >= urdf_path.stat().st_mtime:
        return stable_urdf_path

    root = ET.parse(urdf_path).getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and not Path(filename).is_absolute():
            mesh.set("filename", str((urdf_path.parent / filename).resolve()))

    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", f"{PARTNET_BOX_LID_DRIVE_DAMPING:.6f}")
        dynamics.set("friction", f"{PARTNET_BOX_LID_DRIVE_FRICTION:.6f}")

    for link_name in ("link_0", "link_1"):
        link = root.find(f"./link[@name='{link_name}']")
        if link is None:
            continue
        collision_specs = []
        for collision in list(link.findall("collision")):
            mesh = collision.find("./geometry/mesh")
            if mesh is None or mesh.get("filename") is None:
                continue
            origin_xyz = np.zeros(3, dtype=np.float32)
            origin = collision.find("origin")
            if origin is not None and origin.get("xyz") is not None:
                origin_xyz = np.array([float(v) for v in origin.get("xyz").split()], dtype=np.float32)
            spec = _collision_box_specs_for_mesh(urdf_path, origin_xyz, mesh.get("filename"))
            if spec is not None:
                collision_specs.append(spec)
        if not collision_specs:
            continue

        for collision in list(link.findall("collision")):
            link.remove(collision)

        combined_min = np.full(3, np.inf, dtype=np.float32)
        combined_max = np.full(3, -np.inf, dtype=np.float32)
        for center, size in collision_specs:
            half = size / 2.0
            combined_min = np.minimum(combined_min, center - half)
            combined_max = np.maximum(combined_max, center + half)
        center = (combined_min + combined_max) / 2.0
        size = np.maximum(combined_max - combined_min, 1e-4)

        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz="{} {} {}".format(*[f"{v:.8f}" for v in center]))
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", size="{} {} {}".format(*[f"{v:.8f}" for v in size]))

    tree = ET.ElementTree(root)
    tree.write(stable_urdf_path, encoding="utf-8", xml_declaration=True)
    return stable_urdf_path


def _robocasa_asset_roots():
    robocasa_roots = [
        MEAL_PREP_ASSET_DIR / "robocasa",
        os.environ.get("ROBOCASA_ASSET_DIR"),
    ]
    roots = []
    for root in robocasa_roots:
        if root is None:
            continue
        root = Path(root).expanduser()
        roots.extend(
            [
                root,
                root / "objects",
                root / "objects" / "objaverse",
                root / "objects" / "aigen_objs",
                root / "objects" / "lightwheel",
            ]
        )
    yield from _iter_existing_roots(roots)


def _robocasa_search_paths_for_message():
    paths = [
        MEAL_PREP_ASSET_DIR / "robocasa",
        os.environ.get("ROBOCASA_ASSET_DIR"),
    ]
    return [str(Path(path).expanduser()) for path in paths if path is not None]


def _require_robocasa_actor_asset(kind: str):
    asset_info = _find_robocasa_actor_asset(kind)
    if asset_info is None:
        search_paths = "\n  - ".join(_robocasa_search_paths_for_message())
        raise FileNotFoundError(
            f"RoboCasa asset for '{kind}' is required by MealPreparation but was not found.\n"
            "Set ROBOCASA_ASSET_DIR to a RoboCasa models/assets directory containing objaverse, aigen_objs, "
            "objects/objaverse, or objects/aigen_objs.\n"
            f"Searched:\n  - {search_paths}"
        )
    return asset_info


def _as_scale3(scale):
    if isinstance(scale, (float, int)):
        return (float(scale), float(scale), float(scale))
    return tuple(float(v) for v in scale)


def _mul_scale3(left, right):
    left = _as_scale3(left)
    right = _as_scale3(right)
    return tuple(a * b for a, b in zip(left, right))


def _valid_robocasa_model_xml(model_xml: Path):
    try:
        ET.parse(model_xml)
    except ET.ParseError:
        return False
    return True


def _parse_float_tuple(value: str | None, default):
    if value is None:
        return np.array(default, dtype=np.float32)
    return np.array([float(v) for v in value.split()], dtype=np.float32)


def _robocasa_reg_bbox(asset_info: dict, scale=1.0):
    if "mjcf_path" not in asset_info:
        return None
    try:
        root = ET.parse(asset_info["mjcf_path"]).getroot()
    except ET.ParseError:
        return None
    bbox_geom = None
    for geom in root.findall(".//geom"):
        if geom.get("name") == "reg_bbox":
            bbox_geom = geom
            break
    if bbox_geom is None:
        return None

    pos = _parse_float_tuple(bbox_geom.get("pos"), [0.0, 0.0, 0.0])
    size = _parse_float_tuple(bbox_geom.get("size"), [0.0, 0.0, 0.0])
    full_scale = np.array(_mul_scale3(asset_info.get("registry_scale", 1.0), scale), dtype=np.float32)
    bbox_min = (pos - size) * full_scale
    bbox_max = (pos + size) * full_scale
    return bbox_min, bbox_max


def _robocasa_tabletop_z(asset_info: dict, scale=1.0, clearance=ROBOCASA_TABLE_CLEARANCE, fallback=0.03):
    bbox = _robocasa_reg_bbox(asset_info, scale)
    if bbox is None:
        return fallback
    bbox_min, _ = bbox
    return float(clearance - bbox_min[2])


def _find_robocasa_mjcf_asset(kind: str):
    for root in _robocasa_asset_roots():
        objects_root = root / "objects" if (root / "objects").exists() else root
        search_roots = [objects_root]
        if root != objects_root:
            search_roots.append(root)

        preferred_models = ROBOCASA_OBJECT_PREFERRED_MODELS.get(kind, ())
        if preferred_models:
            preferred_xmls = []
            for search_root in search_roots:
                preferred_xmls.extend(
                    path
                    for path in search_root.rglob("model.xml")
                    if "__macosx" not in {part.lower() for part in path.parts}
                    and path.parent.name.lower() in {model.lower() for model in preferred_models}
                )
            for preferred in preferred_models:
                for model_xml in sorted(preferred_xmls):
                    if model_xml.parent.name.lower() != preferred.lower():
                        continue
                    if not _valid_robocasa_model_xml(model_xml):
                        continue
                    category = model_xml.parent.parent.name
                    registry_scale = 1.0
                    for registry_dir, scale in ROBOCASA_OBJECT_REGISTRIES[kind]:
                        if registry_dir.split("/")[-1].lower() == category.lower():
                            registry_scale = scale
                            break
                    return {
                        "asset_id": category,
                        "asset_dir": str(model_xml.parent),
                        "mjcf_path": str(model_xml),
                        "registry": "preferred_model",
                        "category": category,
                        "registry_scale": registry_scale,
                    }

        for registry_dir, registry_scale in ROBOCASA_OBJECT_REGISTRIES[kind]:
            model_xmls = []
            for search_root in search_roots:
                category_dir = search_root / registry_dir
                if not category_dir.exists():
                    continue
                model_xmls.extend(
                    path
                    for path in category_dir.rglob("model.xml")
                    if "__macosx" not in {part.lower() for part in path.parts}
                )
            model_xmls = sorted(model_xmls)
            preferred_models = ROBOCASA_OBJECT_PREFERRED_MODELS.get(kind, ())
            preferred_xmls = [
                path
                for preferred in preferred_models
                for path in model_xmls
                if path.parent.name.lower() == preferred.lower()
            ]
            search_xmls = preferred_xmls + [path for path in model_xmls if path not in preferred_xmls]
            for model_xml in search_xmls:
                if not _valid_robocasa_model_xml(model_xml):
                    continue
                category = registry_dir.split("/")[-1]
                return {
                    "asset_id": category,
                    "asset_dir": str(model_xml.parent),
                    "mjcf_path": str(model_xml),
                    "registry": registry_dir.split("/")[0],
                    "category": category,
                    "registry_scale": registry_scale,
                }
        for asset_name in ROBOCASA_OBJECT_CANDIDATES[kind]:
            model_xmls = sorted(
                path
                for path in root.rglob("model.xml")
                if "__macosx" not in {part.lower() for part in path.parts}
                and any(asset_name.lower() in part.lower() for part in path.parent.parts)
            )
            preferred_models = ROBOCASA_OBJECT_PREFERRED_MODELS.get(kind, ())
            preferred_xmls = [
                path
                for preferred in preferred_models
                for path in model_xmls
                if path.parent.name.lower() == preferred.lower()
            ]
            search_xmls = preferred_xmls + [path for path in model_xmls if path not in preferred_xmls]
            for model_xml in search_xmls:
                if not _valid_robocasa_model_xml(model_xml):
                    continue
                return {
                    "asset_id": asset_name,
                    "asset_dir": str(model_xml.parent),
                    "mjcf_path": str(model_xml),
                    "registry": "path_search",
                    "category": asset_name,
                    "registry_scale": 1.0,
                }
    return None


def _find_mesh_files(asset_dir: Path):
    mesh_exts = {".obj", ".glb", ".gltf", ".ply", ".stl", ".dae"}
    mesh_files = sorted(
        [
            path
            for path in asset_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in mesh_exts
            and "__macosx" not in {part.lower() for part in path.parts}
        ],
        key=lambda p: (
            p.name.lower() not in {"model.obj", "textured.obj", "base.obj"},
            len(p.parts),
            p.name,
        ),
    )
    if not mesh_files:
        return None, None

    def pick(priority):
        for token in priority:
            for path in mesh_files:
                name = path.name.lower()
                if token.startswith(".") and path.suffix.lower() == token:
                    return path
                if not token.startswith(".") and token in name:
                    return path
        return mesh_files[0]

    visual = pick(("textured.obj", "model.obj", "base.obj", ".glb", ".gltf", ".obj"))
    collision = pick(("collision.obj", "collision", "model.obj", "textured.obj"))
    return visual, collision


def _find_robocasa_mesh_asset(kind: str):
    for root in _robocasa_asset_roots():
        for asset_name in ROBOCASA_OBJECT_CANDIDATES[kind]:
            matches = [
                path
                for path in root.rglob("*")
                if path.is_dir() and asset_name.lower() in path.name.lower()
            ]
            for asset_dir in sorted(matches, key=lambda p: (len(p.parts), p.name)):
                visual, collision = _find_mesh_files(asset_dir)
                if visual is not None and collision is not None:
                    return {
                        "asset_id": asset_name,
                        "asset_dir": str(asset_dir),
                        "visual_path": str(visual),
                        "collision_path": str(collision),
                    }
    return None


def _collision_min_z_under_quat(actor, quat) -> float:
    """Lowest point of an actor's COLLISION hulls once `quat` is applied (meshes, boxes, capsules)."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    w, x, y, z = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    lo = np.inf
    for raw in actor._objs:
        component = raw if hasattr(raw, "get_collision_shapes") else (
            raw.find_component_by_type(physx.PhysxRigidDynamicComponent)
            or raw.find_component_by_type(physx.PhysxRigidStaticComponent)
        )
        if component is None:
            continue
        for shape in component.get_collision_shapes():
            sp = shape.get_local_pose()
            Rs = sp.to_transformation_matrix()[:3, :3]
            if hasattr(shape, "vertices"):
                pts = np.asarray(shape.vertices) * np.asarray(shape.scale)
            elif hasattr(shape, "half_size"):
                h = np.asarray(shape.half_size)
                pts = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            elif hasattr(shape, "half_length"):
                hl, r = float(shape.half_length), float(shape.radius)
                pts = np.array([[xx, sy * r, sz * r] for xx in (-hl, hl) for sy in (-1, 1) for sz in (-1, 1)])
            else:
                continue
            pts = pts @ Rs.T + sp.p
            lo = min(lo, float((pts @ R.T)[:, 2].min()))
    return 0.0 if not np.isfinite(lo) else lo


def _make_material(color):
    return sapien.render.RenderMaterial(base_color=color, roughness=0.6, specular=0.25)


def _build_mesh_actor(
    scene,
    asset_info,
    name,
    scale=1.0,
    initial_pose=None,
    body_type="dynamic",
):
    builder = scene.create_actor_builder()
    builder.add_multiple_convex_collisions_from_file(
        filename=asset_info["collision_path"],
        scale=[scale] * 3,
        material=physx.PhysxMaterial(1.0, 0.8, 0.0),
        density=500,
    )
    builder.add_visual_from_file(filename=asset_info["visual_path"], scale=[scale] * 3)
    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    if body_type == "static":
        return builder.build_static(name=name)
    if body_type == "kinematic":
        return builder.build_kinematic(name=name)
    return builder.build(name=name)


def _build_robocasa_mjcf_actor(
    scene,
    asset_info,
    name,
    scale=1.0,
    initial_pose=None,
    body_type="dynamic",
):
    loader = scene.create_mjcf_loader()
    loader.visual_groups = [1]
    loader.scale = np.array(_mul_scale3(asset_info["registry_scale"], scale))
    mjcf_path = Path(asset_info["mjcf_path"])
    parsed = loader.parse(str(mjcf_path), package_dir=str(mjcf_path.parent))
    actor_builders = parsed["actor_builders"]
    articulation_builders = parsed["articulation_builders"]
    if len(actor_builders) + len(articulation_builders) != 1:
        raise RuntimeError(f"Expected one RoboCasa model in {mjcf_path}, got {len(actor_builders)} actors and {len(articulation_builders)} articulations.")
    if articulation_builders:
        builder = articulation_builders[0]
        if initial_pose is not None:
            builder.set_initial_pose(initial_pose)
        return builder.build(name=name, fix_root_link=body_type == "static")

    builder = actor_builders[0]
    if name == "spoon" and SPOON_GRIP_SLEEVE is not None:
        sleeve = SPOON_GRIP_SLEEVE
        grip_mat = _make_material([0.12, 0.12, 0.14, 1.0])
        pose = sapien.Pose(list(sleeve["center"]), list(sleeve["quat"]))
        builder.add_capsule_collision(pose=pose, radius=sleeve["radius"], half_length=sleeve["half_length"],
                                      material=object_material(sleeve["material"]))
        builder.add_capsule_visual(pose=pose, radius=sleeve["radius"], half_length=sleeve["half_length"], material=grip_mat)
    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    if body_type == "static":
        return builder.build_static(name=name)
    if body_type == "kinematic":
        return builder.build_kinematic(name=name)
    return builder.build_dynamic(name=name)


def _find_robocasa_actor_asset(kind: str):
    return _find_robocasa_mjcf_asset(kind) or _find_robocasa_mesh_asset(kind)


def _build_robocasa_actor(scene, asset_info, name, scale, initial_pose, body_type="dynamic"):
    if "mjcf_path" in asset_info:
        return _build_robocasa_mjcf_actor(
            scene,
            asset_info,
            name=name,
            scale=scale,
            initial_pose=initial_pose,
            body_type=body_type,
        )
    return _build_mesh_actor(
        scene,
        asset_info,
        name=name,
        scale=scale,
        initial_pose=initial_pose,
        body_type=body_type,
    )


def _set_component_mass(component, mass: float) -> None:
    """Set an explicit mass with a realizable inertia (see tasks/config/mass_properties)."""
    set_realizable_mass(component, mass)


def _lid_link_names(obj) -> set:
    """Names of the links driven by an active joint (i.e. the lid)."""
    if not isinstance(obj, Articulation):
        return set()
    names = set()
    for articulation in obj._objs:
        for joint in articulation.get_active_joints():
            child = joint.child_link
            if child is not None:
                names.add(str(child.name))
    return names


def _configure_task_object_physics(obj, material_kind: str):
    """Apply the explicit material, mass and energy fences to a task object.

    Nothing here may fall through to an engine default: the material class, the
    dominant contact pair and the mass are all stated in the tables above.
    """
    material_class = TASK_OBJECT_MATERIAL_CLASSES[material_kind]
    lid_names = _lid_link_names(obj) if material_kind == "box" else set()
    base_material = object_material(
        material_class, dominant=TASK_OBJECT_DOMINANT_PAIR[material_kind]
    )
    lid_material = object_material(
        material_class, dominant=TASK_OBJECT_DOMINANT_PAIR["box_lid"]
    ) if lid_names else base_material
    base_mass = TASK_OBJECT_MASS[material_kind]
    lid_mass = TASK_OBJECT_MASS["box_lid"]
    structs = obj.get_links() if isinstance(obj, Articulation) else (obj,)
    for struct in structs:
        for raw in struct._objs:
            component = raw
            if not hasattr(component, "get_collision_shapes"):
                component = raw.find_component_by_type(
                    physx.PhysxRigidDynamicComponent
                )
                if component is None:
                    component = raw.find_component_by_type(
                        sapien.physx.PhysxRigidStaticComponent
                    )
            if component is None:
                continue
            # The lid is identified by RAW sapien link name: a merged
            # ManiSkill Link carries the un-prefixed name, the joint's child
            # link carries the scene-prefixed one, so the two never compare
            # equal at the struct level.
            is_lid = str(getattr(raw, "name", "")) in lid_names
            for shape in component.get_collision_shapes():
                shape.set_physical_material(
                    lid_material if is_lid else base_material
                )
            # Explicit mass. Without this the body keeps whatever the mesh
            # volume times the PhysX default density (1000 kg/m3) produced,
            # which gave a 13.45 kg meal box and a 1.5 kg spoon. Articulation
            # links are PhysxArticulationLinkComponent, so test for the mass
            # interface rather than for PhysxRigidDynamicComponent.
            if hasattr(component, "mass") and hasattr(component, "inertia"):
                _set_component_mass(
                    component, lid_mass if is_lid else base_mass
                )
            # Solver-energy ceilings are task-object safeguards, never robot,
            # hand, table, or global-Plant parameters.
            if hasattr(component, "set_max_depenetration_velocity"):
                component.set_max_depenetration_velocity(0.5)
                component.set_max_linear_velocity(3.0)
    return obj


def _build_procedural_lidded_box(
    scene, scene_idxs, name, visual_urdf_path: Path | None = None, initial_pose=None
):
    builder = scene.create_articulation_builder()
    builder.set_scene_idxs(scene_idxs)
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    phys_mat = physx.PhysxMaterial(1.0, 0.8, 0.0)

    base = builder.create_link_builder(parent=None)
    base.set_name("box_base")
    wall = PROCEDURAL_BOX_WALL
    box_lx, box_ly, box_lz = PROCEDURAL_BOX_LX, PROCEDURAL_BOX_LY, PROCEDURAL_BOX_LZ
    bottom_mat = _make_material([0.70, 0.50, 0.30, 1.0])
    inner_mat = _make_material([0.86, 0.72, 0.52, 1.0])
    for pose, half_size in [
        (sapien.Pose([0, 0, wall / 2]), [box_lx / 2, box_ly / 2, wall / 2]),
        (sapien.Pose([0, box_ly / 2, box_lz / 2]), [box_lx / 2, wall / 2, box_lz / 2]),
        (sapien.Pose([0, -box_ly / 2, box_lz / 2]), [box_lx / 2, wall / 2, box_lz / 2]),
        (sapien.Pose([box_lx / 2, 0, box_lz / 2]), [wall / 2, box_ly / 2, box_lz / 2]),
        (sapien.Pose([-box_lx / 2, 0, box_lz / 2]), [wall / 2, box_ly / 2, box_lz / 2]),
    ]:
        base.add_box_collision(pose=pose, half_size=half_size, material=phys_mat)
        base.add_box_visual(pose=pose, half_size=half_size, material=bottom_mat)
    base.add_box_collision(
        pose=sapien.Pose([0, 0, wall + PROCEDURAL_BOX_APPLE_SUPPORT_THICKNESS / 2]),
        half_size=[
            box_lx / 2 - wall,
            box_ly / 2 - wall,
            PROCEDURAL_BOX_APPLE_SUPPORT_THICKNESS / 2,
        ],
        material=phys_mat,
    )
    base.add_box_visual(
        pose=sapien.Pose([0, 0, wall + PROCEDURAL_BOX_APPLE_SUPPORT_THICKNESS / 2]),
        half_size=[
            box_lx / 2 - wall,
            box_ly / 2 - wall,
            PROCEDURAL_BOX_APPLE_SUPPORT_THICKNESS / 2,
        ],
        material=inner_mat,
    )

    lid_mat = _make_material([0.78, 0.57, 0.34, 1.0])
    lid = builder.create_link_builder(parent=base)
    lid.set_name("box_lid")
    lower_half_size = [
        PROCEDURAL_BOX_LID_LOWER_LX / 2,
        PROCEDURAL_BOX_LID_LOWER_LY / 2,
        PROCEDURAL_BOX_LID_LOWER_LZ / 2,
    ]
    upper_half_size = [
        PROCEDURAL_BOX_LID_UPPER_LX / 2,
        PROCEDURAL_BOX_LID_UPPER_LY / 2,
        PROCEDURAL_BOX_LID_UPPER_LZ / 2,
    ]
    lid_total_lz = PROCEDURAL_BOX_LID_LOWER_LZ + PROCEDURAL_BOX_LID_UPPER_LZ
    hinge_y = box_ly / 2 + PROCEDURAL_BOX_LID_REAR_CLEARANCE
    hinge_z = box_lz + lid_total_lz + PROCEDURAL_BOX_LID_VERTICAL_CLEARANCE
    lid_center_y_from_hinge = -PROCEDURAL_BOX_LID_UPPER_LY / 2
    lid_center_z = -lid_total_lz / 2
    lower_pose = sapien.Pose(
        [
            0,
            lid_center_y_from_hinge,
            lid_center_z + PROCEDURAL_BOX_LID_UPPER_LZ / 2,
        ]
    )
    upper_pose = sapien.Pose(
        [
            0,
            lid_center_y_from_hinge,
            lid_center_z - PROCEDURAL_BOX_LID_LOWER_LZ / 2,
        ]
    )
    lid.add_box_collision(
        pose=lower_pose,
        half_size=lower_half_size,
        material=phys_mat,
        density=PROCEDURAL_BOX_LID_DENSITY,
    )
    lid.add_box_visual(
        pose=lower_pose, half_size=lower_half_size, material=lid_mat
    )
    lid.add_box_collision(
        pose=upper_pose,
        half_size=upper_half_size,
        material=phys_mat,
        density=PROCEDURAL_BOX_LID_DENSITY,
    )
    lid.add_box_visual(
        pose=upper_pose, half_size=upper_half_size, material=lid_mat
    )
    # ------------------------------------------------------------------
    # Hinge geometry (user report: "the joint between lid and box
    # is a void"). The revolute axis is parked `lid_total_lz + clearance` above
    # the rim so the lid can hang under it and plug the mouth when closed, but
    # a joint has no body: with the lid open the operator looked at an air gap
    # of exactly that stand-off between the rim and the lid's lower edge.
    #   * knuckle: a cylinder on the LID, centred on the axis, so the lid is
    #     visibly (and physically) pinned to the axis line;
    #   * back leaf: a plate on the BASE rising from the rim to just under the
    #     knuckle, directly behind the axis, so nothing can be seen (or pass)
    #     through the stand-off. Its top stops 1.5 mm short of the knuckle and
    #     it lies behind the axis line, so neither the closed lid (all of it in
    #     front of the axis) nor the open lid (all of it above the axis behind
    #     the line) ever touches it -- verified by sweeping the hinge 0..1.85.
    # ------------------------------------------------------------------
    knuckle_r = PROCEDURAL_BOX_HINGE_KNUCKLE_RADIUS
    knuckle_mat = _make_material([0.45, 0.45, 0.47, 1.0])
    lid.add_cylinder_collision(
        pose=sapien.Pose([0.0, 0.0, 0.0]), radius=knuckle_r, half_length=box_lx / 2,
        material=phys_mat, density=PROCEDURAL_BOX_LID_DENSITY,
    )
    lid.add_cylinder_visual(
        pose=sapien.Pose([0.0, 0.0, 0.0]), radius=knuckle_r, half_length=box_lx / 2, material=knuckle_mat
    )
    leaf_t = PROCEDURAL_BOX_HINGE_LEAF_THICKNESS
    leaf_y0 = hinge_y + 0.0005                       # just behind the axis line
    leaf_top = hinge_z - knuckle_r - 0.0015          # 1.5 mm under the knuckle
    leaf_half = [box_lx / 2, leaf_t / 2, (leaf_top - box_lz) / 2]
    leaf_pose = sapien.Pose([0.0, leaf_y0 + leaf_t / 2, (leaf_top + box_lz) / 2])
    base.add_box_collision(pose=leaf_pose, half_size=leaf_half, material=phys_mat)
    # The VISUAL leaf continues up to the axis height (its top disappears inside
    # the knuckle), so the 1.5 mm running clearance under the knuckle is not a
    # visible slit. Collision stays at leaf_top: the lid never reaches below the
    # axis behind the line, so the visual-only part can never be walked through.
    leaf_vis_top = hinge_z + 0.003                   # ends inside the knuckle barrel
    leaf_vis_half = [box_lx / 2, leaf_t / 2, (leaf_vis_top - box_lz) / 2]
    leaf_vis_pose = sapien.Pose([0.0, leaf_y0 + leaf_t / 2, (leaf_vis_top + box_lz) / 2])
    base.add_box_visual(pose=leaf_vis_pose, half_size=leaf_vis_half, material=bottom_mat)
    lid.set_joint_name("box_lid_hinge")
    lid.set_joint_properties(
        type="revolute",
        limits=[[0.0, 1.85]],
        # The lid extends toward -Y from the rear hinge.  Use the -X hinge
        # axis so increasing qpos lifts the front edge away from the box;
        # the previous +X convention made the lid sweep down through the rim.
        pose_in_parent=sapien.Pose(
            [0.0, hinge_y, hinge_z],
            [0.0, 0.0, 0.0, 1.0],
        ),
        pose_in_child=sapien.Pose(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ),
        friction=0.04,
        damping=0.12,
    )
    return builder.build(name=name, fix_root_link=True)


class MealPreparationBase(BaseEnv):
    SUPPORTED_ROBOTS = sorted(DEXTEROUS_HAND_ROBOT_UIDS)
    SUPPORTED_CONTROL_MODES = SUPPORTED_CONTROL_MODES
    SUPPORTED_REWARD_MODES = ("sparse", "none")
    # 1 s at the 20 Hz control rate. Was 5 steps (0.25 s); the operator asked
    # for a full second so that brushing the lid past the angle cannot latch
    # the stage -- the lid has to be left open.
    success_hold_steps = 20
    # CabinetPour-style asset switches.  Subclasses override these before the
    # BaseEnv constructor builds the scene.
    has_box = True
    has_plate = True
    has_spoon = True
    has_bread = True
    has_ball_food = True
    agent: Union[
        PandaSharpaHandRight,
        PandaWujiHandRight,
        PandaSharpaHandLeftMobileBase,
        PandaSharpaHandLeftAngledMobileBase,
        PandaSharpaHandRightMobileBase,
        PandaSharpaHandRightAngledMobileBase,
        PandaWujiHandLeftMobileBase,
        PandaWujiHandRightMobileBase,
        PandaDualSharpaHandMobileBase,
        PandaDualSharpaHandAngledMobileBase,
        PandaDualWujiHandMobileBase,
    ]

    def __init__(
        self,
        *args,
        robot_uids=SUBTASK_ROBOT_UID,
        robot_init_qpos_noise=0.02,
        box_asset_source="procedural",
        initial_box_lid_qpos=0.0,
        lock_box_lid_until_opening=False,
        render_camera_view="overview",
        layout_noise=0.025,
        plate_noise=0.035,
        stage_spawn_noise=0.018,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if "sim_config" in kwargs and kwargs["sim_config"] is not None:
            raise ValueError(
                "MealPreparation freezes the Plant configuration in "
                "tasks/config/sim_profiles.py; runtime sim_config overrides are not allowed"
            )
        if box_asset_source == "partnet" and PARTNET_BOX_MODEL_ID in PARTNET_STABLE_BOX_MODEL_IDS:
            box_asset_source = "hybrid"
        if robot_uids not in DEXTEROUS_HAND_ROBOT_UIDS:
            raise ValueError(
                f"robot_uids must be one of {tuple(DEXTEROUS_HAND_ROBOT_UIDS)}, got {robot_uids!r}"
            )
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.box_asset_source = box_asset_source
        self.initial_box_lid_qpos = initial_box_lid_qpos
        self.lock_box_lid_until_opening = lock_box_lid_until_opening
        self.render_camera_view = render_camera_view
        self.layout_noise = float(layout_noise)
        self.plate_noise = float(plate_noise)
        self.stage_spawn_noise = float(stage_spawn_noise)
        self.box_lid_lock_active = lock_box_lid_until_opening
        self.ball_food_lock_active = lock_box_lid_until_opening
        self.tabletop_object_lock_active = lock_box_lid_until_opening
        self.box_lid_target_qpos = None
        self.box_asset_info = None
        self.box_urdf_path = None
        self.box_bottom_top_z = None
        self.loaded_box_asset_source = "unresolved"
        self.robocasa_asset_info = {}
        self.fallback_asset_info = {}
        self.initial_ball_in_box = None
        self.ever_ball_left_box = None
        self.ever_ball_manipulated = None
        self.ever_ball_scooped = None
        self.ever_ball_transport = None
        self.ball_transport_steps = None
        self.ball_grasp_start_xy = None
        self.current_ball_scooped = None
        self.ever_bread_manipulated = None
        self.ever_bread_grasped = None
        self.ever_bread_transport = None
        self.bread_transport_steps = None
        self.bread_grasp_start_xy = None
        self.current_bread_grasped = None
        self.ever_left_hand_opened_box = None
        self.initial_bread_z = None
        self.initial_ball_z = None
        self.stage_box_open = None
        self.stage_ball_done = None
        self.stage_bread_done = None
        self._stage_hold_count = None
        self._stage_last_eval_step = None
        self._marker_last_update_step = None
        self._success_hold_count = None
        self._success_last_eval_step = None
        # LAYOUT. The old spread was authored around objects 2-5x
        # oversized: the 0.944 m spoon reached back toward the arm from y=+0.62
        # and the 0.62 m plate covered a quarter of the desk. At real sizes the
        # same coordinates put the plate 1.14 m and the spoon 0.93 m from the
        # arm root that has to work them -- outside anything this arm reaches.
        #
        # Everything is now placed on an arc around the arm root that OWNS it
        # (box = left hand, bread/plate/spoon = right hand), inside the same
        # working band CabinetPour was accepted with: nearest surface 0.637 m,
        # farthest grasped object 0.901 m. Measured after the move, nearest
        # surface to the operating arm root:
        #     box 0.67   bread 0.66   plate 0.74   spoon 0.75   (ball rides the box)
        # Pairwise centre spacing is at least the sum of the two footprint
        # radii, so nothing starts intersecting anything else.
        # (scoop probe, user-tested): box and tomato 0.16 m closer to
        # the arms (the scoop kept dragging the wrist into its limits), plate and
        # bread 0.08 m closer; the tomato starts against the box's RIGHT wall so
        # the scoop is a straight +x push along that wall (33 deg IK margin vs
        # <10 deg for a diagonal corner scoop).
        self.box_pose_xyz = torch.tensor([-0.24 + MEAL_PREP_BOX_PULL_X + MEAL_PREP_LAYOUT_X_SHIFT, 0.08, 0.0])
        # Plate at -0.60, i.e. 0.35 m towards the operator's centre, at their
        # request (+y is the operator's left; they stand at -x facing +x).
        self.plate_pose_xyz = torch.tensor([-0.20 + MEAL_PREP_TABLE_PULL_X + MEAL_PREP_LAYOUT_X_SHIFT, -0.55, 0.011]) # 5 cm toward the arm; a 0.31 m platter at -0.60 had its far half past 0.90 m
        self.bread_pose_xyz = torch.tensor([-0.35 + MEAL_PREP_TABLE_PULL_X + MEAL_PREP_LAYOUT_X_SHIFT, -0.12, 0.032])
        self.ball_pose_xyz = torch.tensor([
            -0.24 + MEAL_PREP_BOX_PULL_X + BALL_IN_BOX_OFFSET[0] + MEAL_PREP_LAYOUT_X_SHIFT,
            0.08 + BALL_IN_BOX_OFFSET[1], 0.165,
        ])
        self.spoon_pose_xyz = torch.tensor([SPOON_SPAWN_X + MEAL_PREP_LAYOUT_X_SHIFT, SPOON_SPAWN_Y, 0.020])
        self.spoon_pose_q = torch.tensor(SPOON_SPAWN_QUAT, dtype=torch.float32)
        self._episode_spoon_pose = None
        self._scene_layout_offsets = None
        self._scene_plate_pose_xyz = None
        self._scene_box_pose_xyz = None
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        # Fleet-wide physics/control timing, solver iterations, contact offset,
        # determinism, and fallback material all come from the shared profile.
        # Meal Preparation owns only scene spacing here.
        return make_sim_config(spacing=3)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.45 + WORK_TABLE_X_OFFSET, -0.55, 0.45],
            target=[0.05 + WORK_TABLE_X_OFFSET, 0.0, 0.07],
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        box_x = float(self.box_pose_xyz[0])
        if self.render_camera_view == "box_inside":
            pose = sapien_utils.look_at([-0.55 + WORK_TABLE_X_OFFSET, 0.02, 0.24], [box_x, 0.0, 0.07])
            return [CameraConfig("render_camera", pose, 768, 768, 0.95, 0.01, 100)]
        if self.render_camera_view == "box_overhead":
            pose = sapien_utils.look_at([-0.55 + WORK_TABLE_X_OFFSET, 0.02, 0.55], [box_x, 0.04, 0.07])
            return [CameraConfig("render_camera", pose, 768, 768, 0.70, 0.01, 100)]
        if self.render_camera_view == "box_front_close":
            pose = sapien_utils.look_at([-0.38 + WORK_TABLE_X_OFFSET, 0.02, 0.16], [box_x, 0.02, 0.08])
            return [CameraConfig("render_camera", pose, 768, 768, 0.90, 0.01, 100)]
        if self.render_camera_view == "wide_overview":
            pose = sapien_utils.look_at([-1.85 + WORK_TABLE_X_OFFSET, -1.35, 1.00], [box_x, 0.0, 0.20])
            return [CameraConfig("render_camera", pose, 768, 768, 1.55, 0.01, 100)]
        pose = sapien_utils.look_at([-1.55 + WORK_TABLE_X_OFFSET, -1.20, 0.85], [box_x, 0.0, 0.22])
        return [CameraConfig("render_camera", pose, 768, 768, 1.35, 0.01, 100)]

    def _rig_root_pose(self) -> sapien.Pose:
        """Robot root pose on the shared rig, for EVERY supported build.

        Deliberately total rather than "mobile bases only": a rig that depends
        on robot_uids lets a mirror process and a teleop process launched with
        slightly different --robot-uids show the operator two scenes 0.30 m
        apart, with nothing raising.
        """
        uid = self.robot_uids if isinstance(self.robot_uids, str) else ""
        if uid in MOBILE_BASE_ROBOT_UIDS:
            return platform_root_pose()
        # Trolley-less single-arm build: its root IS panda_link0, so it has to
        # be placed where a trolley-mounted right arm root would land.
        return armless_base_root_pose("left" if "left" in uid else "right")

    def _set_work_table_pose(self):
        set_work_table_pose(self)

    def _load_scene(self, options: dict):
        # Static assets receive their per-environment random poses here.  GPU
        # simulation forbids moving static actors after scene construction.
        rng = np.random.default_rng(20260813)
        layout = rng.uniform(-self.layout_noise, self.layout_noise, size=(self.num_envs, 2))
        plate = rng.uniform(-self.plate_noise, self.plate_noise, size=(self.num_envs, 2))
        self._scene_layout_offsets = torch.as_tensor(layout, dtype=torch.float32, device=self.device)
        self._scene_plate_pose_xyz = self.plate_pose_xyz.expand(self.num_envs, -1).clone().to(self.device)
        self._scene_plate_pose_xyz[:, :2] += torch.as_tensor(plate, dtype=torch.float32, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._set_work_table_pose()
        self._load_meal_assets()

    def _load_meal_assets(self):
        self.box = self._load_box() if self.has_box else None
        self.plate = self._build_plate() if self.has_plate else None
        self.spoon = self._build_spoon() if self.has_spoon else None
        self.bread = self._build_bread() if self.has_bread else None
        self.ball_food = self._build_ball_food() if self.has_ball_food else None

    def _load_box(self):
        box_articulations = []
        urdf_path = None
        visual_urdf_path = None
        if self.box_asset_source in {"auto", "hybrid"}:
            visual_urdf_path = _find_partnet_box_urdf()
        if self.box_asset_source == "partnet":
            urdf_path = _find_partnet_box_urdf()
            if urdf_path is not None:
                raw_urdf_path = urdf_path
                urdf_path = _build_stable_partnet_box_urdf(urdf_path)
                box_pose_z = _partnet_box_pose_z_from_vertices(raw_urdf_path, scale=PARTNET_BOX_SCALE)
                self.box_pose_xyz = torch.tensor(
                    [0.03 + WORK_TABLE_X_OFFSET, 0.0, box_pose_z]
                )
                self.box_urdf_path = urdf_path
                model_id = _partnet_model_id_from_path(raw_urdf_path)
                self.box_bottom_top_z = _partnet_box_bottom_top_z(
                    raw_urdf_path, box_pose_z, scale=PARTNET_BOX_SCALE
                )
                print(
                    f"[MealPreparation] PartNet box z pose set to {box_pose_z:.4f} "
                    f"from bbox {urdf_path.with_name('bounding_box.json')}"
                )
                print(
                    f"[MealPreparation] PartNet box inner bottom local z set to "
                    f"{self.box_bottom_top_z:.4f} for model {model_id}"
                )
        if urdf_path is None:
            self.box_pose_xyz = self.box_pose_xyz.clone()
            self.box_pose_xyz[2] = 0.0
            # The food rests on the thin inner support, not on the top of the
            # outer bottom wall.  Using the support top avoids an initial
            # 5-mm collision overlap with the procedural box.
            self.box_bottom_top_z = (
                PROCEDURAL_BOX_WALL + PROCEDURAL_BOX_APPLE_SUPPORT_THICKNESS
            )
        for i in range(self.num_envs):
            scene_idxs = [i]
            box_pose = self.box_pose_xyz.clone()
            if self._scene_layout_offsets is not None:
                box_pose[:2] += self._scene_layout_offsets[i].to(box_pose.device)
            self._scene_box_pose_xyz = (
                self.box_pose_xyz.to(self.device).expand(self.num_envs, -1).clone()
                if self._scene_box_pose_xyz is None
                else self._scene_box_pose_xyz
            )
            self._scene_box_pose_xyz[i] = box_pose.to(self.device)
            if urdf_path is not None:
                loader = self.scene.create_urdf_loader()
                loader.fix_root_link = True
                loader.scale = PARTNET_BOX_SCALE
                loader.load_multiple_collisions_from_file = True
                builder = loader.parse(str(urdf_path))["articulation_builders"][0]
                builder.set_scene_idxs(scene_idxs)
                builder.initial_pose = sapien.Pose(box_pose.tolist())
                box = builder.build(name=f"partnet_box_{i}", fix_root_link=True)
                self.loaded_box_asset_source = "partnet"
                self.box_asset_info = {
                    "model_id": _partnet_model_id_from_path(raw_urdf_path),
                    "urdf_path": str(urdf_path),
                    "source_urdf_path": str(raw_urdf_path),
                    "scale": PARTNET_BOX_SCALE,
                    "inner_bottom_top_z": self.box_bottom_top_z,
                }
            else:
                box = _build_procedural_lidded_box(
                    self.scene,
                    scene_idxs,
                    f"procedural_box_{i}",
                    visual_urdf_path=visual_urdf_path,
                    initial_pose=sapien.Pose(box_pose.tolist()),
                )
                self.loaded_box_asset_source = "hybrid" if visual_urdf_path is not None else "procedural"
                if visual_urdf_path is not None:
                    self.box_asset_info = {
                        "model_id": _partnet_model_id_from_path(visual_urdf_path),
                        "visual_urdf_path": str(visual_urdf_path),
                        "physics": "procedural_hollow_box",
                        "visual": "partnet_body_mesh",
                    }
            box_articulations.append(box)
        # merge_links=True is required: without it ManiSkill leaves the merged
        # handle's `links` as None, so the very next call
        # (_configure_task_object_physics -> get_links()) raises TypeError and
        # every MealPreparation env fails to construct. Same convention as
        # drawer_plug.py / office_laptop.py; valid here because every per-scene
        # box comes from the same asset and therefore has the same link count.
        self.box = Articulation.merge(box_articulations, "meal_box", merge_links=True)
        _configure_task_object_physics(self.box, "box")
        self._configure_box_lid_drives()
        if self.loaded_box_asset_source == "procedural":
            print(
                "[MealPreparation] Using procedural hollow hinged box. "
                "Set MEAL_PREP_BOX_ASSET_DIR or PARTNET_MOBILITY_ASSET_DIR to use a local PartNet/SAPIEN box."
            )
        elif self.loaded_box_asset_source == "hybrid":
            print(f"[MealPreparation] Using hybrid box asset: {self.box_asset_info}")
        else:
            print(f"[MealPreparation] Using PartNet-Mobility box asset: {self.box_asset_info}")
        return self.box

    def _configure_box_lid_drives(
        self,
        stiffness=PARTNET_BOX_LID_DRIVE_STIFFNESS,
        damping=PARTNET_BOX_LID_DRIVE_DAMPING,
        force_limit=PARTNET_BOX_LID_DRIVE_FORCE_LIMIT,
        friction=PARTNET_BOX_LID_DRIVE_FRICTION,
    ):
        if self._box_lid_dof() == 0:
            return
        for articulation in self.box._objs:
            for joint in articulation.get_active_joints():
                joint.set_drive_properties(
                    stiffness,
                    damping,
                    force_limit=force_limit,
                    mode="force",
                )
                joint.set_friction(friction)

    def _set_box_lid_drive_target(self, target_qpos: torch.Tensor):
        dof = self._box_lid_dof()
        if dof == 0:
            return
        self.box_lid_target_qpos = target_qpos.clone()
        zero_velocity = torch.zeros_like(target_qpos)
        if self.gpu_sim_enabled:
            data_index_dtype = getattr(
                self.box, "_data_index", torch.tensor([], device=self.device)
            ).dtype
            joint_indices = torch.arange(dof, device=self.device, dtype=data_index_dtype)
            self.box.set_joint_drive_targets(target_qpos, joint_indices=joint_indices)
            self.box.set_joint_drive_velocity_targets(
                zero_velocity,
                joint_indices=joint_indices,
            )
            self.scene.px.gpu_apply_articulation_target_position()
            self.scene.px.gpu_apply_articulation_target_velocity()
        else:
            raw_articulations = list(self.box._objs)
            if target_qpos.shape[0] == len(raw_articulations):
                scene_indices = list(range(len(raw_articulations)))
            else:
                reset_mask = self.scene._reset_mask.detach().cpu().tolist()
                scene_indices = [
                    i for i, is_reset in enumerate(reset_mask) if is_reset
                ]
                if len(scene_indices) != target_qpos.shape[0]:
                    raise RuntimeError(
                        "CPU box drive target batch does not match the reset mask"
                    )
            for target_row, scene_idx in enumerate(scene_indices):
                joints = raw_articulations[scene_idx].get_active_joints()
                for joint_idx, joint in enumerate(joints[:dof]):
                    joint.set_drive_target(
                        float(target_qpos[target_row, joint_idx].item())
                    )
                    joint.set_drive_velocity_target(
                        float(zero_velocity[target_row, joint_idx].item())
                    )

    def _box_lid_dof(self):
        if not hasattr(self, "box") or self.box is None:
            return 0
        dof = 0
        for articulation in getattr(self.box, "_objs", []):
            try:
                dof = max(dof, len(articulation.get_active_joints()))
            except Exception:
                continue
        return dof

    def _build_plate(self):
        asset_info = _require_robocasa_actor_asset("plate")
        self.robocasa_asset_info["plate"] = asset_info
        self.plate_pose_xyz = self._robocasa_pose_on_table(
            "plate", asset_info, self.plate_pose_xyz, body_type="static"
        )
        # The asset metadata determines the final tabletop z.  Build the
        # static actor with the already sampled per-environment xy layout.
        self._scene_plate_pose_xyz[:, 2] = self.plate_pose_xyz[2]
        print(f"[MealPreparation] Using RoboCasa plate asset: {asset_info['asset_dir']}")
        builder_pose = self._scene_plate_pose_xyz
        plate = _build_robocasa_actor(
            self.scene,
            asset_info,
            name="pizza_plate",
            scale=ROBOCASA_OBJECT_SCALES["plate"],
            initial_pose=Pose.create_from_pq(builder_pose),
            body_type="static",
        )
        return _configure_task_object_physics(plate, "plate")

    def _build_spoon(self):
        asset_info = _require_robocasa_actor_asset("spoon")
        self.robocasa_asset_info["spoon"] = asset_info
        print(f"[MealPreparation] Using RoboCasa spoon asset: {asset_info['asset_dir']}")
        spoon = _build_robocasa_actor(
            self.scene,
            asset_info,
            name="spoon",
            scale=_mul_scale3(ROBOCASA_OBJECT_SCALES["spoon"], (1.0, 1.0, SPOON_LENGTH_SCALE)),   # length axis = local z
            initial_pose=sapien.Pose(p=self.spoon_pose_xyz.tolist(), q=self.spoon_pose_q.tolist()),
        )
        # Spawn height from the BUILT hulls under the spawn quaternion (the asset
        # table's tabletop z assumes the mesh's own frame, which the yawed spoon
        # is not in; the literal 0.020 put the bowl inside the tabletop).
        lowest = _collision_min_z_under_quat(spoon, self.spoon_pose_q.tolist())
        self.spoon_pose_xyz = self.spoon_pose_xyz.clone()
        self.spoon_pose_xyz[2] = float(-lowest + 0.0005)
        print(f"[MealPreparation] spoon pose set to ({self.spoon_pose_xyz[0]:.3f}, {self.spoon_pose_xyz[1]:.3f}, {self.spoon_pose_xyz[2]:.3f})")
        spoon.set_pose(sapien.Pose(p=self.spoon_pose_xyz.tolist(), q=self.spoon_pose_q.tolist()))
        return spoon

    def _build_bread(self):
        asset_info = _require_robocasa_actor_asset("bread")
        self.robocasa_asset_info["bread"] = asset_info
        self.bread_pose_xyz = self._robocasa_pose_on_table("bread", asset_info, self.bread_pose_xyz)
        print(f"[MealPreparation] Using RoboCasa bread asset: {asset_info['asset_dir']}")
        bread = _build_robocasa_actor(
            self.scene,
            asset_info,
            name="bread",
            scale=ROBOCASA_OBJECT_SCALES["bread"],
            initial_pose=sapien.Pose(self.bread_pose_xyz.tolist()),
        )
        return _configure_task_object_physics(bread, "bread")

    def _build_ball_food(self):
        asset_info = _require_robocasa_actor_asset("ball_food")
        self.robocasa_asset_info["ball_food"] = asset_info
        if self.has_box:
            # Keep the initial food pose in the same world frame as the box.
            # This matters for PartNet candidates whose calibrated box origin
            # is not the default procedural-box origin.
            self.ball_pose_xyz[:2] = self.box_pose_xyz[:2]
            self.ball_pose_xyz = self._robocasa_pose_in_box("ball_food", asset_info, self.ball_pose_xyz)
        else:
            self.ball_pose_xyz = self._robocasa_pose_on_table("ball_food", asset_info, self.ball_pose_xyz)
        print(f"[MealPreparation] Using RoboCasa round food asset for ball_food: {asset_info['asset_dir']}")
        ball_food = _build_robocasa_actor(
            self.scene,
            asset_info,
            name="round_ball_food",
            scale=ROBOCASA_OBJECT_SCALES["ball_food"],
            initial_pose=sapien.Pose(self.ball_pose_xyz.tolist()),
        )
        return _configure_task_object_physics(ball_food, "ball_food")

    def _robocasa_pose_in_box(self, kind: str, asset_info: dict, pose_xyz: torch.Tensor):
        pose_xyz = pose_xyz.clone()
        scale = ROBOCASA_OBJECT_SCALES[kind]
        bbox = _robocasa_reg_bbox(asset_info, scale)
        if bbox is None or self.box_bottom_top_z is None:
            z = _robocasa_tabletop_z(asset_info, scale, fallback=float(pose_xyz[2]))
        else:
            bbox_min, _ = bbox
            box_root_z = float(self.box_pose_xyz[2])
            z = float(
                box_root_z
                + self.box_bottom_top_z
                + BALL_IN_BOX_CLEARANCE
                - bbox_min[2]
            )
        pose_xyz[2] = z
        print(
            f"[MealPreparation] {kind} pose set inside box to "
            f"({pose_xyz[0]:.3f}, {pose_xyz[1]:.3f}, {pose_xyz[2]:.3f})"
        )
        return pose_xyz

    def _robocasa_pose_on_table(self, kind: str, asset_info: dict, pose_xyz: torch.Tensor, body_type="dynamic"):
        pose_xyz = pose_xyz.clone()
        scale = ROBOCASA_OBJECT_SCALES[kind]
        z = _robocasa_tabletop_z(asset_info, scale, fallback=float(pose_xyz[2]))
        pose_xyz[2] = z
        print(
            f"[MealPreparation] {kind} pose set to "
            f"({pose_xyz[0]:.3f}, {pose_xyz[1]:.3f}, {pose_xyz[2]:.3f}); "
            f"body_type={body_type}"
        )
        return pose_xyz

    def _scene_plate_pose(self):
        if self.plate is None:
            return self.plate_pose_xyz.to(self.device).expand(self.num_envs, -1)
        return self.plate.pose.p

    def _sample_episode_xy_noise(self, batch_size: int, scale: float):
        if scale <= 0:
            return torch.zeros((batch_size, 2), device=self.device)
        return torch.stack(
            [
                torch.as_tensor(
                    self._batched_episode_rng[i].uniform(-scale, scale, size=2),
                    device=self.device,
                    dtype=torch.float32,
                )
                for i in range(batch_size)
            ],
            dim=0,
        )

    def _sample_episode_xy_disk(self, batch_size: int, radius: float):
        """Uniformly sample a predecessor outcome inside a radial predicate."""
        if radius <= 0:
            return torch.zeros((batch_size, 2), device=self.device)
        samples = []
        for i in range(batch_size):
            theta = self._batched_episode_rng[i].uniform(0.0, 2.0 * np.pi)
            radial = radius * np.sqrt(self._batched_episode_rng[i].uniform(0.0, 1.0))
            samples.append([radial * np.cos(theta), radial * np.sin(theta)])
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.box_lid_lock_active = self.lock_box_lid_until_opening
        self.ball_food_lock_active = self.lock_box_lid_until_opening
        self.tabletop_object_lock_active = self.lock_box_lid_until_opening
        if self.box_lid_lock_active:
            self._configure_box_lid_drives(
                stiffness=PARTNET_BOX_LID_LOCK_STIFFNESS,
                damping=PARTNET_BOX_LID_LOCK_DAMPING,
                force_limit=PARTNET_BOX_LID_LOCK_FORCE_LIMIT,
                friction=PARTNET_BOX_LID_LOCK_FRICTION,
            )
        else:
            self._configure_box_lid_drives()
        self._initialize_actors(env_idx)
        self._initialize_agent(env_idx)
        # The hold is carried by the bodies' kinematic flag, so it has to be
        # (re-)applied after the actors are placed. Setting the *_lock_active
        # flags alone does nothing now -- that used to be enough only because
        # the per-substep teleport hooks read them every substep.
        self._set_lock_bodies_kinematic(
            self.ball_food_lock_active or self.tabletop_object_lock_active
        )

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self._set_work_table_pose()
            layout_offset = self._scene_layout_offsets[env_idx]
            dof = self._box_lid_dof()
            zero_qpos = torch.zeros((b, dof), device=self.device)
            if dof > 0:
                zero_qpos[:] = self.initial_box_lid_qpos
            if self.box is not None:
                self.box.set_qpos(zero_qpos)
                self.box.set_qvel(torch.zeros_like(zero_qpos))
                self._set_box_lid_drive_target(zero_qpos)
            box_pose = (
                self._scene_box_pose_xyz[env_idx].clone()
                if self._scene_box_pose_xyz is not None
                else self.box_pose_xyz.to(self.device).expand(b, -1).clone()
            )
            self._episode_box_pose = box_pose.clone()

            bread_pose = self.bread_pose_xyz.to(self.device).expand(b, -1).clone()
            bread_pose[:, :2] += layout_offset
            bread_pose[:, :2] += self._sample_episode_xy_noise(b, self.stage_spawn_noise)
            self._episode_bread_pose = bread_pose.clone()
            if self.bread is not None:
                self.bread.set_pose(Pose.create_from_pq(bread_pose))
            ball_pose = self.ball_pose_xyz.to(self.device).expand(b, -1).clone()
            ball_pose[:, :2] += layout_offset
            ball_pose[:, :2] += self._sample_episode_xy_noise(b, self.stage_spawn_noise * 0.6)
            self._episode_ball_pose = ball_pose.clone()
            if self.ball_food is not None:
                self.ball_food.set_pose(Pose.create_from_pq(ball_pose))
            # The plate is a static actor in GPU simulation and cannot be moved
            # after scene construction. Keep its calibrated pose fixed; the
            # box and dynamic objects still receive per-episode layout noise.
            spoon_pose = self.spoon_pose_xyz.to(self.device).expand(b, -1).clone()
            spoon_pose[:, :2] += layout_offset
            spoon_pose[:, :2] += self._sample_episode_xy_noise(b, self.stage_spawn_noise * 0.5)
            self._episode_spoon_pose = spoon_pose.clone()
            if self.spoon is not None:
                self.spoon.set_pose(Pose.create_from_pq(spoon_pose, self.spoon_pose_q.to(self.device).expand(b, -1)))
            self._episode_plate_pose = self._scene_plate_pose()[env_idx].clone()
            zero_vel = torch.zeros((b, 3), device=self.device)
            for actor in (self.ball_food, self.spoon, self.bread):
                if actor is not None:
                    actor.set_linear_velocity(zero_vel)
                    actor.set_angular_velocity(zero_vel)
            self._reset_stage_markers(env_idx)

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            if self.robot_uids in DUAL_HAND_ROBOT_UIDS:
                init_qpos = self.agent.get_reset_pos(device=self.device)
            else:
                init_qpos = torch.tensor(self.agent.keyframes["rest"].qpos, device=self.device)
            init_qpos = init_qpos.expand(b, -1).clone()
            # Shared teleop start pose (tasks/config/workspace_rig.py);
            # the per-arm noise below is applied on top of it.
            init_qpos = apply_arm_init_qpos(self.agent, init_qpos)
            arm_joint_names = list(self.agent.arm_joint_names)
            active_joint_names = [joint.name for joint in self.agent.robot.get_active_joints()]
            arm_indices = [
                active_joint_names.index(joint_name)
                for joint_name in arm_joint_names
                if joint_name in active_joint_names
            ]
            if arm_indices:
                noise = torch.stack(
                    [
                        torch.as_tensor(
                            self._batched_episode_rng[i].normal(
                                0, self.robot_init_qpos_noise, size=(len(arm_indices),)
                            ),
                            device=self.device,
                            dtype=init_qpos.dtype,
                        )
                        for i in range(b)
                    ],
                    dim=0,
                )
                init_qpos[:, arm_indices] += noise
            self.agent.reset(init_qpos)
            self.agent.robot.set_root_pose(self._rig_root_pose())

    def _box_lid_qpos(self):
        if self.box is None:
            return torch.zeros((self.num_envs,), device=self.device)
        if self._box_lid_dof() == 0:
            return torch.zeros((self.num_envs,), device=self.device)
        return torch.mean(self.box.qpos, dim=1)

    def _box_lid_qpos_all(self):
        if self.box is None:
            return torch.zeros((self.num_envs, 0), device=self.device)
        if self._box_lid_dof() == 0:
            return torch.zeros((self.num_envs, 0), device=self.device)
        return self.box.qpos

    def _box_lid_open_progress(self):
        lid_qpos_all = self._box_lid_qpos_all()
        if lid_qpos_all.shape[-1] == 0:
            return torch.zeros((self.num_envs,), device=self.device)
        return torch.min(lid_qpos_all, dim=1).values

    def _box_lid_close_progress(self):
        lid_qpos_all = self._box_lid_qpos_all()
        if lid_qpos_all.shape[-1] == 0:
            return torch.zeros((self.num_envs,), device=self.device)
        return torch.max(lid_qpos_all, dim=1).values

    def _set_box_lid_qpos(self, env_idx: torch.Tensor, lid_qpos: float):
        if self.box is None:
            return
        b = len(env_idx)
        dof = self._box_lid_dof()
        if dof == 0:
            return
        qpos = torch.zeros((b, dof), device=self.device)
        qpos[:] = lid_qpos
        self.box.set_qpos(qpos)
        self.box.set_qvel(torch.zeros_like(qpos))
        self._set_box_lid_drive_target(qpos)

    def _set_box_lid_qpos_values(self, env_idx: torch.Tensor, lid_qpos: torch.Tensor):
        """Set a per-environment lid pose for a standalone-stage reset."""
        if self.box is None:
            return
        dof = self._box_lid_dof()
        if dof == 0:
            return
        lid_qpos = torch.as_tensor(lid_qpos, device=self.device, dtype=torch.float32)
        if lid_qpos.ndim == 0:
            lid_qpos = lid_qpos.expand(len(env_idx))
        qpos = lid_qpos[:, None].expand(-1, dof).clone()
        self.box.set_qpos(qpos)
        self.box.set_qvel(torch.zeros_like(qpos))
        self._set_box_lid_drive_target(qpos)

    # ------------------------------------------------------------------
    # Collection-time hold.
    #
    # WHY THIS IS NOT A PER-STEP TELEPORT ANY MORE.
    #
    # The previous implementation re-wrote pose + velocity of the free task
    # objects (and qpos + qvel of the lid) from BOTH _before_simulation_step
    # and _after_simulation_step. At 400 Hz sim / 20 Hz control that is 40
    # forced writes per control step. Whenever the hand touched a held object,
    # every write re-created the penetration the solver had just resolved, so
    # the depenetration impulse was re-applied each substep -- and because the
    # object could not move, all of that energy went into the robot.
    #
    # Measured on MealPreparationLong-Sharpa-v1 (fingertip overlapping a
    # held object, 40 control steps, peak robot |qvel|):
    #     no lock, same overlap ......... 14.1 rad/s   (one-off, self-resolving)
    #     per-substep teleport .......... 31.8 rad/s   <- ~12x the Franka limit
    #     one write per control step .... 50.9 rad/s   <- WORSE: bigger jumps
    #     kinematic hold (this code) .... 17.0 rad/s   <- lock adds ~2.8
    #
    # A kinematic body holds its pose exactly, with no constraint violation to
    # resolve, so the hold no longer injects energy. Poses are written ONCE at
    # engage time instead of every substep.
    # ------------------------------------------------------------------
    def _lock_bodies(self):
        """Actors that are frozen in place while the collection hold is on."""
        return tuple(
            actor for actor in (self.ball_food, self.spoon, self.bread)
            if actor is not None
        )

    def _set_lock_bodies_kinematic(self, kinematic: bool) -> None:
        """Engage/release the hold. Name kept for the existing call sites; the
        mechanism is a six-DoF motion lock, not the PhysX kinematic flag."""
        if not kinematic and self.gpu_sim_enabled:
            # Releasing a hold that was never engaged is a no-op. Only refuse
            # when something actually asks to ENGAGE it under GPU PhysX --
            # otherwise the unconditional raise below fires on every GPU reset
            # even with lock_box_lid_until_opening=False (the default), which
            # made the whole family unusable with num_envs>1.
            return
        if self.gpu_sim_enabled:
            raise RuntimeError(
                "The collection hold toggles rigid bodies between kinematic and "
                "dynamic, which GPU PhysX does not support after scene load. The "
                "hold is a single-env collection mechanism; run it on CPU "
                "(num_envs=1). The previous per-substep teleport implementation "
                "worked on GPU but injected an impulse blast on every contact "
                "(31.8 rad/s peak robot qvel vs a 2.2 rad/s actuator limit)."
            )
        for actor in self._lock_bodies():
            for raw in actor._objs:
                component = raw.find_component_by_type(
                    physx.PhysxRigidDynamicComponent
                )
                if component is not None:
                    # Lock the six DoF rather than setting `kinematic`. Both hold
                    # the body exactly, but flipping `kinematic` ZEROES whatever
                    # velocity the body carries at that instant, so a state
                    # restore that lands on a held body lost its recorded
                    # velocity (measured: `replay_recording --mode state` failed
                    # on recorded Screwing episodes). Locked axes keep the body
                    # dynamic, so its state still round-trips.
                    component.set_locked_motion_axes([bool(kinematic)] * 6)

    def _lock_box_lid_closed(self):
        """Write the closed lid pose ONCE; the stiff lock drive holds it after."""
        if self.box is None:
            return
        dof = self._box_lid_dof()
        if dof == 0:
            return
        qpos = torch.zeros((self.num_envs, dof), device=self.device)
        qpos[:] = PROCEDURAL_BOX_CLOSED_QPOS
        self.box.set_qpos(qpos)
        self.box.set_qvel(torch.zeros_like(qpos))
        self._set_box_lid_drive_target(qpos)

    def release_box_lid_lock(self):
        was_locked = (
            self.box_lid_lock_active
            or self.ball_food_lock_active
            or self.tabletop_object_lock_active
        )
        if was_locked and not self.gpu_sim_enabled:
            self._set_lock_bodies_kinematic(False)
        self.box_lid_lock_active = False
        self.ball_food_lock_active = False
        self.tabletop_object_lock_active = False
        self._configure_box_lid_drives()
        dof = self._box_lid_dof()
        if dof > 0:
            qpos = torch.zeros((self.num_envs, dof), device=self.device)
            qpos[:] = PROCEDURAL_BOX_CLOSED_QPOS
            self._set_box_lid_drive_target(qpos)

    def start_expert_collection(self):
        self.release_box_lid_lock()

    def lock_box_lid_closed_for_collection(self):
        self.box_lid_lock_active = True
        self.ball_food_lock_active = True
        self.tabletop_object_lock_active = True
        self._configure_box_lid_drives(
            stiffness=PARTNET_BOX_LID_LOCK_STIFFNESS,
            damping=PARTNET_BOX_LID_LOCK_DAMPING,
            force_limit=PARTNET_BOX_LID_LOCK_FORCE_LIMIT,
            friction=PARTNET_BOX_LID_LOCK_FRICTION,
        )
        self._lock_box_lid_closed()
        self._place_lock_bodies()
        self._set_lock_bodies_kinematic(True)

    def _place_lock_bodies(self):
        """Put the held objects at their episode poses. Called ONCE per engage."""
        zero = torch.zeros((self.num_envs, 3), device=self.device)
        if self.ball_food is not None:
            ball_pose = getattr(
                self, "_episode_ball_pose",
                self.ball_pose_xyz.to(self.device).expand(self.num_envs, -1),
            )
            self.ball_food.set_pose(Pose.create_from_pq(ball_pose))
            self.ball_food.set_linear_velocity(zero)
            self.ball_food.set_angular_velocity(zero)
        if self.bread is not None:
            bread_xyz = getattr(
                self, "_episode_bread_pose",
                self.bread_pose_xyz.to(self.device).expand(self.num_envs, -1),
            )
            self.bread.set_pose(Pose.create_from_pq(bread_xyz))
            self.bread.set_linear_velocity(zero)
            self.bread.set_angular_velocity(zero)
        if self.gpu_sim_enabled:
            self.scene.px.gpu_apply_rigid_dynamic_data()

    def _box_inner_bounds(self):
        if self.box is None:
            zeros = torch.zeros((self.num_envs, 2), device=self.device)
            return zeros, zeros, torch.zeros((self.num_envs,), device=self.device), torch.zeros((self.num_envs,), device=self.device)
        box_xy = self.box.pose.p[:, :2]
        min_xy = box_xy + torch.tensor(
            [-PROCEDURAL_BOX_LX / 2 + PROCEDURAL_BOX_WALL, -PROCEDURAL_BOX_LY / 2 + PROCEDURAL_BOX_WALL],
            device=self.device,
        )
        max_xy = box_xy + torch.tensor(
            [PROCEDURAL_BOX_LX / 2 - PROCEDURAL_BOX_WALL, PROCEDURAL_BOX_LY / 2 - PROCEDURAL_BOX_WALL],
            device=self.device,
        )
        min_z = self.box.pose.p[:, 2] + self.box_bottom_top_z
        max_z = self.box.pose.p[:, 2] + PROCEDURAL_BOX_LZ
        return min_xy, max_xy, min_z, max_z

    def _ball_in_box(self, margin=0.0):
        if self.box is None or self.ball_food is None:
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        min_xy, max_xy, min_z, max_z = self._box_inner_bounds()
        p = self.ball_food.pose.p
        bbox = self._ball_food_bbox()
        if bbox is None:
            radius = APPLE_RADIUS
            xy_ok = torch.all(p[:, :2] >= min_xy + radius - margin, dim=1) & torch.all(
                p[:, :2] <= max_xy - radius + margin, dim=1
            )
            z_ok = (p[:, 2] >= min_z + radius - margin) & (p[:, 2] <= max_z - radius + margin)
            return xy_ok & z_ok
        bbox_min, bbox_max = bbox
        bbox_min_t = torch.as_tensor(bbox_min, device=self.device, dtype=p.dtype)
        bbox_max_t = torch.as_tensor(bbox_max, device=self.device, dtype=p.dtype)
        obj_min = p + bbox_min_t
        obj_max = p + bbox_max_t
        xy_ok = torch.all(obj_min[:, :2] >= min_xy - margin, dim=1) & torch.all(
            obj_max[:, :2] <= max_xy + margin, dim=1
        )
        z_ok = (obj_min[:, 2] >= min_z - BALL_IN_BOX_Z_TOLERANCE - margin) & (
            obj_max[:, 2] <= max_z + margin
        )
        return xy_ok & z_ok

    def _ball_food_bbox(self):
        asset_info = self.robocasa_asset_info.get("ball_food")
        if asset_info is None:
            return None
        return _robocasa_reg_bbox(asset_info, ROBOCASA_OBJECT_SCALES["ball_food"])

    def _ball_left_box(self):
        return ~self._ball_in_box(margin=BOX_EXIT_MARGIN)

    def _min_side_tip_distance_to_handle(self, side="right"):
        """Min fingertip distance to the spoon HANDLE SEGMENT (bowl back -> grip end)."""
        if self.robot_uids in DUAL_HAND_ROBOT_UIDS:
            tip_links = self.agent.left_tip_links if side == "left" else self.agent.right_tip_links
        else:
            tip_links = self.agent.tip_links
        tips = torch.stack([tip_link.pose.p for tip_link in tip_links], dim=1)
        a_l = torch.tensor(SPOON_HANDLE_SEGMENT_LOCAL[0], dtype=torch.float32, device=self.device).expand(self.num_envs, -1)
        b_l = torch.tensor(SPOON_HANDLE_SEGMENT_LOCAL[1], dtype=torch.float32, device=self.device).expand(self.num_envs, -1)
        a = self.spoon.pose.p + quaternion_apply(self.spoon.pose.q, a_l)
        b = self.spoon.pose.p + quaternion_apply(self.spoon.pose.q, b_l)
        ab = (b - a)[:, None, :]
        ap = tips - a[:, None, :]
        s = torch.clamp((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9), 0.0, 1.0)
        closest = a[:, None, :] + s[..., None] * ab
        return torch.min(torch.linalg.norm(tips - closest, dim=-1), dim=1).values

    def _spoon_bowl_center_local(self):
        """Bowl centre in the spoon frame, measured ONCE from the built collision
        hulls: the bowl is the wide end of the longest axis, its centre 45 mm
        back from the tip. Follows whatever spoon model/scale is loaded."""
        cached = getattr(self, "_spoon_bowl_center_local_cache", None)
        if cached is not None:
            return cached
        pts = []
        raw = self.spoon._objs[0]
        component = raw if hasattr(raw, "get_collision_shapes") else raw.find_component_by_type(physx.PhysxRigidDynamicComponent)
        for shape in component.get_collision_shapes():
            if not hasattr(shape, "vertices"):      # the grip capsule is not part of the bowl
                continue
            sp = shape.get_local_pose()
            pts.append((np.asarray(shape.vertices) * np.asarray(shape.scale)) @ sp.to_transformation_matrix()[:3, :3].T + sp.p)
        pts = np.concatenate(pts)
        ext = pts.max(0) - pts.min(0)
        axis = int(np.argmax(ext))
        other = [i for i in range(3) if i != axis]
        lo, hi = pts[:, axis].min(), pts[:, axis].max()

        def width_at(v):
            m = np.abs(pts[:, axis] - v) < 0.01
            return (pts[m].max(0) - pts[m].min(0))[other].max() if m.any() else 0.0

        tip = hi if width_at(hi - 0.03) > width_at(lo + 0.03) else lo
        sign = 1.0 if tip == hi else -1.0
        centre = np.zeros(3)
        centre[axis] = tip - sign * 0.045
        self._spoon_bowl_center_local_cache = torch.tensor(centre, dtype=torch.float32, device=self.device)
        return self._spoon_bowl_center_local_cache

    def _spoon_bowl_center_world(self):
        local = self._spoon_bowl_center_local().expand(self.num_envs, -1)
        return self.spoon.pose.p + quaternion_apply(self.spoon.pose.q, local)

    def _bread_bbox(self):
        asset_info = self.robocasa_asset_info.get("bread")
        if asset_info is None:
            return None
        return _robocasa_reg_bbox(asset_info, ROBOCASA_OBJECT_SCALES["bread"])

    @staticmethod
    def _actor_local_collision_points(actor):
        """Every collision vertex of `actor`, in the actor's own frame."""
        raw = actor._objs[0]
        component = raw.find_component_by_type(
            physx.PhysxRigidDynamicComponent
        ) or raw.find_component_by_type(physx.PhysxRigidStaticComponent)
        if component is None:
            return None
        chunks = []
        for shape in component.get_collision_shapes():
            transform = sapien.Pose(
                p=shape.local_pose.p, q=shape.local_pose.q
            ).to_transformation_matrix()
            if hasattr(shape, "half_size"):
                half = np.asarray(shape.half_size)
                verts = np.array([
                    [sx * half[0], sy * half[1], sz * half[2]]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
                ])
            elif hasattr(shape, "vertices"):
                verts = np.asarray(shape.vertices) * np.asarray(shape.scale)
            else:
                continue
            chunks.append((transform[:3, :3] @ verts.T).T + transform[:3, 3])
        return np.concatenate(chunks, 0) if chunks else None

    def _plate_floor_geometry(self):
        """(top z of the plate's flat floor, radius of that flat area), cached.

        Both are read off the plate's own collision mesh rather than guessed: the
        dish has a raised rim, and anything seated out on that rim slides.
        """
        cached = getattr(self, "_plate_floor_cache", None)
        if cached is not None:
            return cached
        points = self._actor_local_collision_points(self.plate)
        if points is None:
            result = (0.0, 0.0)
        else:
            radial = np.linalg.norm(points[:, :2], axis=1)
            outer = float(radial.max())
            # Walk outwards and stop where the top surface starts climbing into
            # the rim (measured on this asset: flat to ~70 mm, +7 mm by 110 mm).
            inner = radial[radial < 0.35 * outer]
            floor_z = float(points[radial < 0.35 * outer][:, 2].max()) if inner.size else 0.0
            flat = 0.35 * outer
            for limit in np.arange(0.40 * outer, outer, 0.02 * outer):
                band = points[radial < limit]
                if band.size and float(band[:, 2].max()) - floor_z > 0.002:
                    break
                flat = limit
            result = (floor_z, float(flat))
        self._plate_floor_cache = result
        return result

    def _seat_object_on_plate(self, obj, bbox, env_idx, xy_radius, fallback_z=0.035):
        """Put `obj` down on the plate -- the accepted OUTPUT of a placement stage.

        Used by the standalone stages to build their entry distribution from the
        previous stage's accepted output rather than from a hand-written pose.

        Both the height and the scatter are derived from the two meshes. They
        used to be a fixed ``BALL_IN_BOX_CLEARANCE`` above the plate ORIGIN and a
        scatter of ``xy_radius`` -- the placement SUCCESS radius. For the bread
        that meant spawning 7.3 mm above the plate floor and up to 109 mm off
        centre while the loaf's own footprint radius is 56.5 mm, so it landed
        half on the rim and skated: measured 3 of 16 seeds drifting past 50 mm
        and one leaving the plate entirely at 0.65 m/s. The success radius is not
        a spawn radius.
        """
        b = len(env_idx)
        pose = self._episode_plate_pose.clone()
        floor_z, flat_radius = self._plate_floor_geometry()
        points = self._actor_local_collision_points(obj)
        if points is None or bbox is None:
            pose[:, :2] += self._sample_episode_xy_disk(b, xy_radius)
            pose[:, 2] = self._episode_plate_pose[:, 2] + fallback_z
        else:
            drop = float(points[:, 2].min())                 # origin -> lowest point
            footprint = float(np.linalg.norm(points[:, :2], axis=1).max())
            # Keep the whole footprint on the flat floor; a hair of clearance
            # only, so the object seats instead of falling.
            scatter = max(0.0, min(xy_radius, flat_radius - footprint))
            pose[:, :2] += self._sample_episode_xy_disk(b, scatter)
            pose[:, 2] = (
                self._episode_plate_pose[:, 2] + floor_z + PLATE_SEAT_CLEARANCE - drop
            )
        obj.set_pose(Pose.create_from_pq(pose))
        obj.set_linear_velocity(torch.zeros((b, 3), device=self.device))
        obj.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def _place_bread_on_plate(self, env_idx: torch.Tensor):
        if self.bread is None:
            return
        self._seat_object_on_plate(
            self.bread, self._bread_bbox(), env_idx,
            PLATE_BREAD_XY_THRESHOLD, fallback_z=0.045,
        )
        self.initial_bread_z[env_idx] = self.bread.pose.p[env_idx, 2]
        self.bread_grasp_start_xy[env_idx] = self.bread.pose.p[env_idx, :2]

    def _min_side_tip_distance(self, obj: Actor, side="right"):
        if self.robot_uids in DUAL_HAND_ROBOT_UIDS:
            tip_links = self.agent.left_tip_links if side == "left" else self.agent.right_tip_links
        else:
            tip_links = self.agent.tip_links
        tip_positions = torch.stack([tip_link.pose.p for tip_link in tip_links], dim=1)
        tip_dist = torch.linalg.norm(tip_positions - obj.pose.p[:, None, :], dim=-1)
        return torch.min(tip_dist, dim=1).values

    def _reset_stage_markers(self, env_idx: torch.Tensor):
        b = len(env_idx)
        if self.initial_ball_in_box is None or self.initial_ball_in_box.shape[0] != self.num_envs:
            self.initial_ball_in_box = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.ever_ball_left_box = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.ever_ball_manipulated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.ever_ball_scooped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.ever_ball_transport = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.ever_right_hand_used_spoon = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.ball_transport_steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
            self.ball_grasp_start_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
            self.current_ball_scooped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.initial_ball_z = torch.zeros((self.num_envs,), device=self.device)
            self.ever_bread_manipulated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.ever_bread_grasped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.ever_bread_transport = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.bread_transport_steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
            self.bread_grasp_start_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
            self.current_bread_grasped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.ever_left_hand_opened_box = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.initial_bread_z = torch.zeros((self.num_envs,), device=self.device)
            self.stage_box_open = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.stage_ball_done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.stage_bread_done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self._stage_hold_count = torch.zeros(
                (self.num_envs, 3), dtype=torch.int32, device=self.device
            )
            self._stage_last_eval_step = torch.zeros(
                (self.num_envs, 3), dtype=torch.int64, device=self.device
            )
            self._marker_last_update_step = torch.zeros(
                (self.num_envs,), dtype=torch.int64, device=self.device
            )
            self._success_hold_count = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
            self._success_last_eval_step = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)

        self.initial_ball_in_box[env_idx] = self._ball_in_box()[env_idx]
        self.ever_ball_left_box[env_idx] = False
        self.ever_ball_manipulated[env_idx] = False
        self.ever_ball_scooped[env_idx] = False
        self.ever_ball_transport[env_idx] = False
        self.ever_right_hand_used_spoon[env_idx] = False
        self.ball_transport_steps[env_idx] = 0
        if self.ball_food is not None:
            self.ball_grasp_start_xy[env_idx] = self.ball_food.pose.p[env_idx, :2]
            self.initial_ball_z[env_idx] = self.ball_food.pose.p[env_idx, 2]
        else:
            self.ball_grasp_start_xy[env_idx] = 0.0
            self.initial_ball_z[env_idx] = 0.0
        self.current_ball_scooped[env_idx] = False
        self.ever_bread_manipulated[env_idx] = False
        self.ever_bread_grasped[env_idx] = False
        self.ever_bread_transport[env_idx] = False
        self.bread_transport_steps[env_idx] = 0
        if self.bread is not None:
            self.bread_grasp_start_xy[env_idx] = self.bread.pose.p[env_idx, :2]
        else:
            self.bread_grasp_start_xy[env_idx] = 0.0
        self.current_bread_grasped[env_idx] = False
        self.ever_left_hand_opened_box[env_idx] = False
        self.initial_bread_z[env_idx] = (
            self.bread.pose.p[env_idx, 2] if self.bread is not None else 0.0
        )
        self.stage_box_open[env_idx] = False
        self.stage_ball_done[env_idx] = False
        self.stage_bread_done[env_idx] = False
        self._stage_hold_count[env_idx] = 0
        self._stage_last_eval_step[env_idx] = -1
        self._marker_last_update_step[env_idx] = -1
        self._success_hold_count[env_idx] = 0
        self._success_last_eval_step[env_idx] = -1

    def _qualify_stage(
        self, stage_index: int, success_condition: torch.Tensor
    ) -> torch.Tensor:
        """Require the same raw stage predicate for five control steps."""
        step = self._elapsed_steps
        previous = self._stage_hold_count[:, stage_index]
        last_step = self._stage_last_eval_step[:, stage_index]
        new_step = step != last_step
        next_count = torch.where(
            success_condition,
            torch.clamp(previous + 1, max=self.success_hold_steps),
            torch.zeros_like(previous),
        )
        self._stage_hold_count[:, stage_index] = torch.where(
            new_step, next_count, previous
        )
        self._stage_last_eval_step[:, stage_index] = torch.where(
            new_step, step, last_step
        )
        return self._stage_hold_count[:, stage_index] >= self.success_hold_steps

    def _latch_success(self, success_condition: torch.Tensor) -> torch.Tensor:
        """Auxiliary hold used by the non-sequential close-box variant."""
        step = self._elapsed_steps
        new_step = step != self._success_last_eval_step
        next_count = torch.where(
            success_condition,
            torch.clamp(self._success_hold_count + 1, max=self.success_hold_steps),
            torch.zeros_like(self._success_hold_count),
        )
        self._success_hold_count = torch.where(new_step, next_count, self._success_hold_count)
        self._success_last_eval_step = step.clone()
        return self._success_hold_count >= self.success_hold_steps

    def _reset_success_hold(self):
        if self._success_hold_count is not None:
            self._success_hold_count.zero_()
            self._success_last_eval_step.fill_(-1)

    def _box_open_stage_info(self, open_threshold: float):
        lid_qpos_all = self._box_lid_qpos_all()
        if lid_qpos_all.shape[-1] == 0:
            box_open = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
        else:
            box_open = torch.all(lid_qpos_all >= open_threshold, dim=1)
        return {
            "box_open": box_open,
            "success_condition": box_open & self.ever_left_hand_opened_box,
        }

    def _ball_stage_success_info(
        self,
        open_threshold: float,
        ball_xy_threshold: float,
        bread_xy_threshold: float,
    ):
        """Stage 3: the tomato, by hand, out of the box and onto the plate.

        This replaced a spoon-scooping stage. Scooping was not
        reachable with a rigid-pad hand: holding a 296 mm spoon whose bowl sits
        140 mm from the grip needs ~110 mN*m of torque about the pinch axis, and
        rigid pads carry that through the contact PATCH, which for convex-on-
        convex contact is a point -- so the transmissible torque is ~0. The
        tomato is a 35 mm sphere with its centre of mass at its centre, so it
        needs no such torque and is a genuinely easy pick-and-place.

        The box requirement is deliberately kept: the tomato still starts inside
        the box and must leave it, so opening the box remains a real
        prerequisite rather than decoration.
        """
        open_info = self._box_open_stage_info(open_threshold)
        bread_info = self._object_on_plate_info(
            self.bread, xy_threshold=bread_xy_threshold, z_margin=0.07
        )
        ball_info = self._object_on_plate_info(
            self.ball_food, xy_threshold=ball_xy_threshold, z_margin=0.06
        )
        is_robot_static = self.agent.is_static(0.2)     # reported only, see below
        process_ok = (
            self.initial_ball_in_box
            & self.ever_ball_left_box
            & self.ever_ball_scooped
            & self.ever_ball_transport
            & self.ever_right_hand_used_spoon
        )
        # "On the plate, and it stays there for a second." Nothing else.
        #
        # The dwell is what `_qualify_stage` already enforces: the condition must
        # hold for `success_hold_steps` = 20 consecutive control steps = 1.0 s at
        # 20 Hz. Three terms used to ride along and made that second almost
        # impossible to survive:
        #   * the object's `is_static` reads PhysX rigid-body velocity, which is
        #     CORRUPTED while anything is touching it (measured elsewhere in this
        #     repo: 0.09 m/s and 3.75 rad/s reported for a body whose pose moves
        #     0.0002 m/s and 0.0009 rad/s). Fingers still resting on the tomato
        #     were enough to hold it False.
        #   * `is_robot_static` demanded the WHOLE arm stay under 0.2 rad/s for
        #     the full second, which a hand-driven teleop arm never does.
        #   * a live re-check of `box_open` could be undone by a bumped lid.
        # A single bad frame reset the 20-step counter to zero, which is exactly
        # the "triggers very unreliably" the operator reported.
        #
        # The dwell subsumes all of it: an object that holds the plate region for
        # a second IS at rest, in the only sense the task cares about, and needs
        # no velocity read to prove it. Geometry is kept (both objects must
        # actually be on the plate); only the velocity terms are gone.
        condition = (
            self.stage_bread_done
            & bread_info["on_plate"]
            & ball_info["on_plate"]
            & process_ok
        )
        return {
            **open_info,
            "ball_info": ball_info,
            "bread_info": bread_info,
            "is_robot_static": is_robot_static,
            "process_ok": process_ok,
            "success_condition": condition,
        }

    def _bread_stage_success_info(
        self,
        open_threshold: float,
        bread_xy_threshold: float,
    ):
        """Stage 2: the bread (the LARGE object) onto the plate.

        Its predecessor is the box-open milestone; earlier the bread
        was the third stage and inherited the spoon stage instead.
        """
        open_info = self._box_open_stage_info(open_threshold)
        bread_info = self._object_on_plate_info(
            self.bread, xy_threshold=bread_xy_threshold, z_margin=0.07
        )
        is_robot_static = self.agent.is_static(0.2)     # reported only
        process_ok = self.ever_bread_grasped & self.ever_bread_transport
        # On the plate, held for the 1.0 s `_qualify_stage` dwell. The object /
        # robot velocity terms were removed -- see
        # `_ball_stage_success_info` for the measurements behind that.
        condition = (
            self.stage_box_open
            & bread_info["on_plate"]
            & process_ok
        )
        return {
            **open_info,
            "bread_info": bread_info,
            "is_robot_static": is_robot_static,
            "process_ok": process_ok,
            "success_condition": condition,
        }

    def _ball_stage_id(self):
        stage = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        stage = torch.where(self.initial_ball_in_box, torch.ones_like(stage), stage)
        stage = torch.where(self.ever_ball_scooped, torch.full_like(stage, 2), stage)
        stage = torch.where(self.ever_ball_left_box, torch.full_like(stage, 3), stage)
        stage = torch.where(self.ever_ball_transport, torch.full_like(stage, 4), stage)
        stage = torch.where(self.stage_ball_done, torch.full_like(stage, 5), stage)
        return stage

    def _bread_stage_id(self):
        stage = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        stage = torch.where(self.ever_bread_manipulated, torch.ones_like(stage), stage)
        stage = torch.where(self.ever_bread_grasped, torch.full_like(stage, 2), stage)
        stage = torch.where(self.ever_bread_transport, torch.full_like(stage, 3), stage)
        return stage

    def _dual_stage_id(self):
        # Public id is the completed long-horizon milestone.  The detailed
        # process markers remain available in ``info`` and block stage skips.
        stage = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        stage = torch.where(self.stage_box_open, torch.ones_like(stage), stage)
        stage = torch.where(self.stage_bread_done, torch.full_like(stage, 2), stage)
        stage = torch.where(self.stage_ball_done, torch.full_like(stage, 3), stage)
        return stage

    def _open_close_stage_id(self, open_threshold, close_threshold, task_mode):
        stage = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        if task_mode == "open":
            stage = torch.where(self._box_lid_open_progress() > BOX_MANIPULATION_QPOS_DELTA, torch.ones_like(stage), stage)
            stage = torch.where(self._box_lid_open_progress() >= open_threshold, torch.full_like(stage, 2), stage)
        else:
            stage = torch.where(self._box_lid_close_progress() < open_threshold, torch.ones_like(stage), stage)
            stage = torch.where(self._box_lid_close_progress() <= close_threshold, torch.full_like(stage, 2), stage)
        return stage

    def _update_stage_markers(self):
        if self.initial_ball_in_box is None:
            return
        step = self._elapsed_steps
        new_step = step != self._marker_last_update_step
        if self.box is None:
            lid_qpos_all = torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
        else:
            lid_qpos_all = self._box_lid_qpos_all()
        lid_motion = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if lid_qpos_all.shape[-1] > 0:
            lid_motion = torch.any(lid_qpos_all > BOX_MANIPULATION_QPOS_DELTA, dim=1)
        left_near_box = (
            self._min_side_tip_distance(self.box, side="left") < BOX_HAND_CONTACT_THRESHOLD
            if self.box is not None else torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        )
        if self.ball_food is not None:
            ball_touched = (
                self._min_side_tip_distance(self.ball_food, side="right")
                < BALL_HAND_CONTACT_THRESHOLD
            )
            ball_lifted = (
                self.ball_food.pose.p[:, 2] > self.initial_ball_z + BALL_LIFT_THRESHOLD
            )
        else:
            ball_touched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            ball_lifted = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if self.bread is not None:
            bread_touched = self._min_side_tip_distance(self.bread, side="right") < BREAD_HAND_CONTACT_THRESHOLD
            bread_lifted = self.bread.pose.p[:, 2] > self.initial_bread_z + BREAD_LIFT_THRESHOLD
        else:
            bread_touched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            bread_lifted = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # Stage 3 is a TOOL stage again: the tomato counts as
        # "scooped" while it sits in the spoon's bowl, the bowl faces up and the
        # tomato is off the box floor -- a hand carrying it past the spoon does
        # not qualify (bowl-centre distance, not spoon-origin distance).
        if self.spoon is not None and self.ball_food is not None:
            bowl_dist = torch.linalg.norm(self.ball_food.pose.p - self._spoon_bowl_center_world(), dim=1)
            bowl_normal_world = quaternion_apply(
                self.spoon.pose.q,
                torch.tensor(SPOON_BOWL_LOCAL_NORMAL, dtype=torch.float32, device=self.device).expand(self.num_envs, -1),
            )
            bowl_up = bowl_normal_world[:, 2] > SPOON_UP_DOT_THRESHOLD
            ball_scooped_now = (bowl_dist < SPOON_BOWL_CONTACT_THRESHOLD) & bowl_up & ball_lifted
            right_near_spoon = self._min_side_tip_distance_to_handle("right") < RIGHT_HAND_SPOON_THRESHOLD
        else:
            ball_scooped_now = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            right_near_spoon = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        bread_grasp_closure = (
            self._hand_closure("right") >= BREAD_GRASP_CLOSURE_THRESHOLD
        )
        bread_grasped_now = bread_touched & bread_grasp_closure & bread_lifted
        self.current_ball_scooped = ball_scooped_now
        self.current_bread_grasped = bread_grasped_now

        # ORDER: box open -> BREAD (large) -> TOMATO (small).
        #
        # These trackers record WHAT PHYSICALLY HAPPENED -- "this object was
        # grasped", "it was carried" -- and are deliberately NOT gated on the
        # predecessor milestone. They used to be, and it made the long task
        # unwinnable for any operator who overlapped two stages: lifting the
        # tomato out of the box before the bread milestone latched recorded
        # nothing, and nothing could be recovered afterwards because the tomato
        # was already out of the box and could not leave it a second time.
        # Measured: bread placed, tomato placed, success False forever.
        #
        # Stage ORDER is still enforced, in the only place it belongs -- the
        # success conditions, which each require the predecessor latch
        # (`_ball_stage_success_info` needs `stage_bread_done`). Gating the
        # trackers as well bought no ordering and also broke the strict-subset
        # contract: a standalone stage pre-latches its inherited milestones, so
        # the same motion was booked differently in the subtask and in the long
        # task.
        #
        # Tomato: grasped by hand inside the box, lifted out, carried to the
        # plate. `ever_ball_left_box` still requires a grasp first, so a tomato
        # merely knocked out of the box does not count as extracted.
        next_ball_manipulated = self.ever_ball_manipulated | (
            ball_touched | ball_lifted
        )
        newly_ball_scooped = (
            new_step
            & ball_scooped_now
            & ~self.ever_ball_scooped
        )
        if self.ball_food is not None:
            self.ball_grasp_start_xy = torch.where(
                newly_ball_scooped[:, None],
                self.ball_food.pose.p[:, :2],
                self.ball_grasp_start_xy,
            )
            ball_displaced = (
                torch.linalg.norm(
                    self.ball_food.pose.p[:, :2] - self.ball_grasp_start_xy, dim=1
                )
                > BALL_TRANSPORT_DISTANCE
            )
        else:
            ball_displaced = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
        next_ball_scooped = self.ever_ball_scooped | ball_scooped_now
        next_ball_transport_steps = torch.where(
            ball_scooped_now & ball_displaced,
            torch.clamp(self.ball_transport_steps + 1, max=BALL_TRANSPORT_MIN_STEPS),
            torch.zeros_like(self.ball_transport_steps),
        )
        next_ball_transport = self.ever_ball_transport | (
            next_ball_transport_steps >= BALL_TRANSPORT_MIN_STEPS
        )
        next_ball_left = self.ever_ball_left_box | (
            next_ball_scooped & self._ball_left_box()
        )
        self.ever_ball_manipulated = torch.where(
            new_step, next_ball_manipulated, self.ever_ball_manipulated
        )
        self.ever_ball_scooped = torch.where(
            new_step, next_ball_scooped, self.ever_ball_scooped
        )
        self.ball_transport_steps = torch.where(
            new_step, next_ball_transport_steps, self.ball_transport_steps
        )
        self.ever_ball_transport = torch.where(
            new_step, next_ball_transport, self.ever_ball_transport
        )
        self.ever_ball_left_box = torch.where(
            new_step, next_ball_left, self.ever_ball_left_box
        )
        self.ever_right_hand_used_spoon = torch.where(
            new_step, self.ever_right_hand_used_spoon | (ball_scooped_now & right_near_spoon), self.ever_right_hand_used_spoon
        )

        next_bread_manipulated = self.ever_bread_manipulated | (
            bread_touched | bread_lifted
        )
        newly_grasped = (
            new_step
            & bread_grasped_now
            & ~self.ever_bread_grasped
        )
        if self.bread is not None:
            self.bread_grasp_start_xy = torch.where(
                newly_grasped[:, None],
                self.bread.pose.p[:, :2],
                self.bread_grasp_start_xy,
            )
            bread_displaced = (
                torch.linalg.norm(
                    self.bread.pose.p[:, :2] - self.bread_grasp_start_xy, dim=1
                )
                > BREAD_TRANSPORT_DISTANCE
            )
        else:
            bread_displaced = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
        next_bread_grasped = self.ever_bread_grasped | bread_grasped_now
        next_transport_steps = torch.where(
            bread_grasped_now & bread_displaced,
            torch.clamp(self.bread_transport_steps + 1, max=BREAD_TRANSPORT_MIN_STEPS),
            torch.zeros_like(self.bread_transport_steps),
        )
        next_bread_transport = self.ever_bread_transport | (
            next_transport_steps >= BREAD_TRANSPORT_MIN_STEPS
        )
        self.ever_bread_manipulated = torch.where(
            new_step, next_bread_manipulated, self.ever_bread_manipulated
        )
        self.ever_bread_grasped = torch.where(
            new_step, next_bread_grasped, self.ever_bread_grasped
        )
        self.bread_transport_steps = torch.where(
            new_step, next_transport_steps, self.bread_transport_steps
        )
        self.ever_bread_transport = torch.where(
            new_step, next_bread_transport, self.ever_bread_transport
        )
        next_opened = self.ever_left_hand_opened_box | (
            left_near_box & lid_motion
        )
        self.ever_left_hand_opened_box = torch.where(
            new_step, next_opened, self.ever_left_hand_opened_box
        )
        self._marker_last_update_step = torch.where(
            new_step, step, self._marker_last_update_step
        )

    # ------------------------------------------------------------------ #
    # env-state (offline re-render / RL-from-demo-states)
    # ------------------------------------------------------------------ #
    def get_state_dict(self):
        """Persist physical state plus every task-side sequential latch."""
        state = super().get_state_dict()
        state["meal_preparation_task_state"] = torch.cat(
            [
                self.initial_ball_in_box[:, None].float(),
                self.ever_ball_left_box[:, None].float(),
                self.ever_ball_manipulated[:, None].float(),
                self.ever_ball_scooped[:, None].float(),
                self.ever_ball_transport[:, None].float(),
                self.ball_transport_steps[:, None].float(),
                self.ball_grasp_start_xy.float(),
                self.current_ball_scooped[:, None].float(),
                self.initial_ball_z[:, None].float(),
                self.ever_bread_manipulated[:, None].float(),
                self.ever_bread_grasped[:, None].float(),
                self.ever_bread_transport[:, None].float(),
                self.bread_transport_steps[:, None].float(),
                self.bread_grasp_start_xy.float(),
                self.current_bread_grasped[:, None].float(),
                self.ever_left_hand_opened_box[:, None].float(),
                self.initial_bread_z[:, None].float(),
                self.stage_box_open[:, None].float(),
                self.stage_bread_done[:, None].float(),
                self.stage_ball_done[:, None].float(),
                self._stage_hold_count.float(),
                self._stage_last_eval_step.float(),
                self._marker_last_update_step[:, None].float(),
                self._success_hold_count[:, None].float(),
                self._success_last_eval_step[:, None].float(),
                self.ever_right_hand_used_spoon[:, None].float(),
            ],
            dim=-1,
        )
        state["meal_preparation_lock_state"] = torch.tensor(
            [
                self.box_lid_lock_active,
                self.ball_food_lock_active,
                self.tabletop_object_lock_active,
            ],
            dtype=torch.float32,
            device=self.device,
        ).expand(self.num_envs, -1)
        state["meal_preparation_episode_poses"] = torch.cat(
            [
                self._episode_box_pose,
                self._episode_bread_pose,
                self._episode_ball_pose,
                self._episode_plate_pose,
                self._episode_spoon_pose,
            ],
            dim=-1,
        ).float()
        # The lid drive target and this episode's open angle are per-episode
        # values that live only in Python: `_set_box_lid_drive_target` writes the
        # first at runtime and `_initialize_actors` samples the second per reset.
        # Restoring a mid-episode frame without them left the box driving toward
        # another episode's target. Separate keys rather than widening
        # `episode_poses`, so recordings made before this still load unchanged.
        for key, attr in (
            ("meal_preparation_lid_target", "box_lid_target_qpos"),
            ("meal_preparation_lid_open", "_episode_open_lid_qpos"),
        ):
            value = getattr(self, attr, None)
            if value is None:
                continue
            state[key] = common.to_tensor(
                value, device=self.device
            ).reshape(self.num_envs, -1).float()
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop("meal_preparation_task_state", None)
        lock_state = state.pop("meal_preparation_lock_state", None)
        episode_poses = state.pop("meal_preparation_episode_poses", None)
        lid_target = state.pop("meal_preparation_lid_target", None)
        lid_open = state.pop("meal_preparation_lid_open", None)
        # the lidded box is a FIXED-root articulation, and a PhysX root
        # pose write on it is not idempotent (each set_root_pose reads back ~4e-9 m
        # higher), which broke the exact record->replay contract by one float32
        # ulp. Its root never moves, so restore only its joint state and leave the
        # root pose untouched.
        fixed_boxes = {}
        arts = state.get("articulations")
        if isinstance(arts, dict) and self.box is not None:
            arts = dict(arts)
            state["articulations"] = arts
            for name in list(arts):
                if name.startswith(("procedural_box", "partnet_box")):
                    fixed_boxes[name] = arts.pop(name)
        if env_idx is None:
            restore_idx = torch.arange(self.num_envs, device=self.device)
            super().set_state_dict(state)
        else:
            restore_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            ).flatten()
            super().set_state_dict(state, restore_idx)
        if fixed_boxes:
            value = common.to_tensor(next(iter(fixed_boxes.values())), device=self.device)
            value = value.reshape(len(restore_idx), -1)
            dof = (value.shape[-1] - 13) // 2
            qpos = self.box.get_qpos().clone()
            qvel = self.box.get_qvel().clone()
            qpos[restore_idx] = value[:, 13:13 + dof].to(qpos.dtype)
            qvel[restore_idx] = value[:, 13 + dof:13 + 2 * dof].to(qvel.dtype)
            self.box.set_qpos(qpos)
            self.box.set_qvel(qvel)

        def _state_tensor(value):
            if isinstance(value, dict):
                value = value.get("value")
            if value is None:
                return None
            return common.to_tensor(value, device=self.device).reshape(
                len(restore_idx), -1
            )

        for value, attr in ((lid_target, "box_lid_target_qpos"),
                            (lid_open, "_episode_open_lid_qpos")):
            value = _state_tensor(value)
            if value is None:
                continue          # recorded before these keys existed
            current = getattr(self, attr, None)
            if current is None:
                # Never driven in this episode yet: adopt the recorded layout.
                setattr(self, attr, value)
                continue
            # Keep the live tensor's SHAPE, not just its values -- these are
            # (num_envs, dof) and a squeezed (num_envs,) copy silently changes
            # what every downstream broadcast means.
            current = current.clone()
            current[restore_idx] = value.reshape(current[restore_idx].shape).to(
                current.dtype
            )
            setattr(self, attr, current)

        task_state = _state_tensor(task_state)
        if task_state is not None:
            if task_state.shape[-1] != MEAL_PREP_TASK_STATE_DIM:
                raise ValueError(
                    f"meal_preparation_task_state must have "
                    f"{MEAL_PREP_TASK_STATE_DIM} fields, got {task_state.shape[-1]}"
                )
            self.initial_ball_in_box[restore_idx] = task_state[:, 0].bool()
            self.ever_ball_left_box[restore_idx] = task_state[:, 1].bool()
            self.ever_ball_manipulated[restore_idx] = task_state[:, 2].bool()
            self.ever_ball_scooped[restore_idx] = task_state[:, 3].bool()
            self.ever_ball_transport[restore_idx] = task_state[:, 4].bool()
            self.ball_transport_steps[restore_idx] = task_state[:, 5].to(torch.int32)
            self.ball_grasp_start_xy[restore_idx] = task_state[:, 6:8]
            self.current_ball_scooped[restore_idx] = task_state[:, 8].bool()
            self.initial_ball_z[restore_idx] = task_state[:, 9]
            self.ever_bread_manipulated[restore_idx] = task_state[:, 10].bool()
            self.ever_bread_grasped[restore_idx] = task_state[:, 11].bool()
            self.ever_bread_transport[restore_idx] = task_state[:, 12].bool()
            self.bread_transport_steps[restore_idx] = task_state[:, 13].to(
                torch.int32
            )
            self.bread_grasp_start_xy[restore_idx] = task_state[:, 14:16]
            self.current_bread_grasped[restore_idx] = task_state[:, 16].bool()
            self.ever_left_hand_opened_box[restore_idx] = task_state[:, 17].bool()
            self.initial_bread_z[restore_idx] = task_state[:, 18]
            self.stage_box_open[restore_idx] = task_state[:, 19].bool()
            self.stage_bread_done[restore_idx] = task_state[:, 20].bool()
            self.stage_ball_done[restore_idx] = task_state[:, 21].bool()
            self._stage_hold_count[restore_idx] = task_state[:, 22:25].to(
                torch.int32
            )
            self._stage_last_eval_step[restore_idx] = task_state[:, 25:28].to(
                torch.int64
            )
            self._marker_last_update_step[restore_idx] = task_state[:, 28].to(
                torch.int64
            )
            self._success_hold_count[restore_idx] = task_state[:, 29].to(
                torch.int32
            )
            self._success_last_eval_step[restore_idx] = task_state[:, 30].to(
                torch.int64
            )
            self.ever_right_hand_used_spoon[restore_idx] = task_state[:, 31].bool()

            if env_idx is not None:
                # A demo-state restore -- reset(options={"reset_to_env_states"})
                # -- starts a FRESH episode clock at zero while the registers we
                # just restored say "last evaluated at step N". The reset-time
                # get_info() then reads as a new control step and re-derives the
                # counts from that single frame:
                # bread_transport_steps and ball_transport_steps went 3 -> 0, so
                # a mid-demo frame came back having lost its transport progress.
                # Rebase the registers onto the new clock so that first evaluate
                # leaves the restored counts alone. Same fix CabinetPour and
                # Screwing already carry.
                elapsed = self._elapsed_steps[restore_idx]
                self._stage_last_eval_step[restore_idx] = elapsed[:, None].to(
                    self._stage_last_eval_step.dtype
                ).expand_as(self._stage_last_eval_step[restore_idx])
                self._marker_last_update_step[restore_idx] = elapsed.to(
                    self._marker_last_update_step.dtype
                )
                self._success_last_eval_step[restore_idx] = elapsed.to(
                    self._success_last_eval_step.dtype
                )

        episode_poses = _state_tensor(episode_poses)
        if episode_poses is not None:
            if episode_poses.shape[-1] != MEAL_PREP_EPISODE_POSES_DIM:
                raise ValueError(
                    f"meal_preparation_episode_poses must have {MEAL_PREP_EPISODE_POSES_DIM} fields"
                )
            self._episode_box_pose[restore_idx] = episode_poses[:, 0:3]
            self._episode_bread_pose[restore_idx] = episode_poses[:, 3:6]
            self._episode_ball_pose[restore_idx] = episode_poses[:, 6:9]
            self._episode_plate_pose[restore_idx] = episode_poses[:, 9:12]
            if self._episode_spoon_pose is None:
                self._episode_spoon_pose = torch.zeros((self.num_envs, 3), device=self.device)
            self._episode_spoon_pose[restore_idx] = episode_poses[:, 12:15]

        lock_state = _state_tensor(lock_state)
        if lock_state is not None:
            if lock_state.shape[-1] != 3:
                raise ValueError("meal_preparation_lock_state must have 3 fields")
            # The pre-collection locks are intentionally global: they alter
            # articulation drives and simulation callbacks for the whole env.
            expected = lock_state[0]
            if not torch.all(lock_state == expected):
                raise ValueError(
                    "Meal Preparation lock state must be identical across envs"
                )
            self.box_lid_lock_active = bool(expected[0].item())
            self.ball_food_lock_active = bool(expected[1].item())
            self.tabletop_object_lock_active = bool(expected[2].item())
            if self.box_lid_lock_active:
                self._configure_box_lid_drives(
                    stiffness=PARTNET_BOX_LID_LOCK_STIFFNESS,
                    damping=PARTNET_BOX_LID_LOCK_DAMPING,
                    force_limit=PARTNET_BOX_LID_LOCK_FORCE_LIMIT,
                    friction=PARTNET_BOX_LID_LOCK_FRICTION,
                )
            else:
                self._configure_box_lid_drives()
            # The hold is now carried by the bodies' kinematic flag, not by a
            # per-substep teleport, so restoring the flag has to restore that
            # too -- otherwise a replayed episode silently loses the hold.
            if not self.gpu_sim_enabled:
                self._set_lock_bodies_kinematic(
                    self.ball_food_lock_active or self.tabletop_object_lock_active
                )

    def _hand_closure(self, side="right"):
        """Mean finger curl: 0 = hand open, 1 = every flexion joint at its limit.

        This used to average ``(q - lower) / (upper - lower)`` over ALL 22 hand
        joints, which is not a grip measure. The six ABDUCTION joints sit at
        exactly 0.500 in the rest pose and stay there however hard the hand
        squeezes, so they pinned the average near the middle and squashed its
        range to 0.171 (open) .. 0.864 (fully curled). A 0.52 gate on that scale
        means "curl every flexion joint past half its mechanical limit", which a
        hand wrapped around a loaf never reaches. Measured on a firm scripted
        grasp: 0.495 for the bread and 0.541 for the tomato against a 0.52 gate
        -- so the bread stage could not latch at all and the tomato was a coin
        flip. That is exactly the "I put it on the plate and nothing happened"
        report, for both subtasks.

        Abduction is now excluded, and flexion is measured from the open pose
        (q ~= 0 for every flexion joint on these hands) toward the joint's own
        limit, so the number means what its name says over the full 0..1 range.
        """
        names = getattr(self.agent, f"{side}_hand_joint_names", None)
        if names is None:
            names = getattr(self.agent, "hand_joint_names", [])
        # Abduction spreads the fingers sideways; it carries no grip information.
        names = [name for name in names if not name.endswith("_AA")]
        if not names:
            return torch.zeros((self.num_envs,), device=self.device)
        active = [joint.name for joint in self.agent.robot.get_active_joints()]
        indices = [active.index(name) for name in names if name in active]
        if not indices:
            return torch.zeros((self.num_envs,), device=self.device)
        qpos = self.agent.robot.get_qpos()[:, indices]
        limits = self.agent.robot.get_qlimits()
        if limits.ndim == 3:
            limits = limits[0]
        limits = limits[indices]
        lower, upper = limits[:, 0], limits[:, 1]
        # Curl toward whichever limit the joint actually flexes into, so a hand
        # whose flexion sign is negative reads the same way.
        span = torch.where(upper.abs() >= lower.abs(), upper, lower)
        curl = qpos / torch.where(span.abs() < 1e-6, torch.ones_like(span), span)
        return torch.mean(torch.clamp(curl, 0.0, 1.0), dim=1)

    def _object_on_plate_info(self, obj: Actor, xy_threshold=0.075, z_margin=0.04):
        if self.plate is None or obj is None:
            zeros = torch.zeros((self.num_envs,), device=self.device)
            false = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            return dict(
                xy_dist_to_plate=zeros,
                z_above_plate=zeros,
                height_ok=false,
                is_static=false,
                on_plate=false,
            )
        plate_xy = self.plate.pose.p[:, :2]
        obj_xy = obj.pose.p[:, :2]
        xy_dist = torch.linalg.norm(obj_xy - plate_xy, axis=1)
        z_above_plate = obj.pose.p[:, 2] - self.plate.pose.p[:, 2]
        height_ok = (z_above_plate > 0.0) & (z_above_plate < z_margin)
        is_static = obj.is_static(lin_thresh=0.05, ang_thresh=0.5)
        on_plate = (xy_dist <= xy_threshold) & height_ok
        return dict(
            xy_dist_to_plate=xy_dist,
            z_above_plate=z_above_plate,
            height_ok=height_ok,
            is_static=is_static,
            on_plate=on_plate,
        )

    def _dexterous_object_reach_info(self, obj: Actor, object_radius=0.03):
        tip_positions = torch.stack([tip_link.pose.p for tip_link in self.agent.tip_links], dim=1)
        obj_tip_vec = tip_positions - obj.pose.p[:, None, :]
        obj_tip_dist = torch.linalg.norm(obj_tip_vec, dim=-1)
        closest_tip_surface_dist = torch.clamp(
            torch.min(obj_tip_dist, dim=1).values - object_radius, min=0.0
        )
        num_close_tips = (obj_tip_dist < object_radius + 0.03).float().sum(dim=1)
        loose_grasp = (num_close_tips >= 2.0) & (closest_tip_surface_dist < 0.025)
        return dict(
            obj_tip_dist=obj_tip_dist,
            closest_tip_surface_dist=closest_tip_surface_dist,
            num_close_tips=num_close_tips,
            loose_grasp=loose_grasp,
        )

    def _tcp_pose(self, side="right"):
        if side == "left" and self.robot_uids in DUAL_HAND_ROBOT_UIDS:
            if hasattr(self.agent, "left_tcp_pose"):
                return self.agent.left_tcp_pose
            return Pose.create_from_pq(self.agent.left_tcp_pos)
        if side == "right" and self.robot_uids in DUAL_HAND_ROBOT_UIDS:
            if hasattr(self.agent, "right_tcp_pose"):
                return self.agent.right_tcp_pose
            return Pose.create_from_pq(self.agent.right_tcp_pos)
        return self.agent.tcp.pose

    def _tcp_pos(self, side="right"):
        return self._tcp_pose(side).p

    def _side_object_reach_info(self, obj: Actor, side="right", object_radius=0.03):
        if self.robot_uids not in DUAL_HAND_ROBOT_UIDS:
            return self._dexterous_object_reach_info(obj, object_radius=object_radius)
        tip_links = self.agent.left_tip_links if side == "left" else self.agent.right_tip_links
        tip_positions = torch.stack([tip_link.pose.p for tip_link in tip_links], dim=1)
        obj_tip_vec = tip_positions - obj.pose.p[:, None, :]
        obj_tip_dist = torch.linalg.norm(obj_tip_vec, dim=-1)
        closest_tip_surface_dist = torch.clamp(
            torch.min(obj_tip_dist, dim=1).values - object_radius, min=0.0
        )
        num_close_tips = (obj_tip_dist < object_radius + 0.03).float().sum(dim=1)
        loose_grasp = (num_close_tips >= 2.0) & (closest_tip_surface_dist < 0.025)
        return dict(
            obj_tip_dist=obj_tip_dist,
            closest_tip_surface_dist=closest_tip_surface_dist,
            num_close_tips=num_close_tips,
            loose_grasp=loose_grasp,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.box is not None:
            obs.update(box_qpos=self._box_lid_qpos().unsqueeze(-1), box_qpos_all=self._box_lid_qpos_all())
        if self.plate is not None:
            obs.update(plate_pose=self.plate.pose.raw_pose)
        if self.robot_uids in DUAL_HAND_ROBOT_UIDS:
            obs.update(
                left_tcp_pose=self._tcp_pose("left").raw_pose,
                right_tcp_pose=self._tcp_pose("right").raw_pose,
            )
        if self.obs_mode_struct.use_state:
            obs.update(
            )
            if self.box is not None:
                obs.update(box_pose=self.box.pose.raw_pose, tcp_to_box_pos=self.box.pose.p - self.agent.tcp.pose.p)
            if self.spoon is not None:
                obs.update(spoon_pose=self.spoon.pose.raw_pose)
            if self.bread is not None:
                obs.update(bread_pose=self.bread.pose.raw_pose)
            if self.ball_food is not None:
                obs.update(ball_food_pose=self.ball_food.pose.raw_pose)
        return obs


class _OpenCloseBoxTask(MealPreparationBase):
    has_box = True
    has_plate = False
    has_spoon = True
    has_bread = False
    has_ball_food = False

    def __init__(
        self,
        *args,
        task_mode="open",
        open_threshold=BOX_OPEN_ANGLE,
        close_threshold=0.18,
        **kwargs,
    ):
        if task_mode not in ("open", "close"):
            raise ValueError(f"task_mode must be 'open' or 'close', got {task_mode!r}")
        self.task_mode = task_mode
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        kwargs.setdefault("lock_box_lid_until_opening", False)
        if task_mode == "open":
            kwargs.setdefault(
                "initial_box_lid_qpos", PROCEDURAL_BOX_CLOSED_QPOS
            )
        super().__init__(*args, **kwargs)

    def _initialize_actors(self, env_idx: torch.Tensor):
        super()._initialize_actors(env_idx)
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.box.qpos.shape[-1]
            if dof > 0 and self.task_mode == "close":
                qpos = torch.zeros((b, dof), device=self.device)
                qpos[:] = self.open_threshold + 0.25
                self.box.set_qpos(qpos)

    def evaluate(self):
        self._update_stage_markers()
        lid_qpos_all = self._box_lid_qpos_all()
        lid_qpos = self._box_lid_qpos()
        if lid_qpos_all.shape[-1] == 0:
            is_open = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            is_closed = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        else:
            is_open = torch.all(lid_qpos_all >= self.open_threshold, dim=1)
            is_closed = torch.all(lid_qpos_all <= self.close_threshold, dim=1)
        target = (
            torch.full_like(lid_qpos, self.open_threshold)
            if self.task_mode == "open"
            else torch.full_like(lid_qpos, self.close_threshold)
        )
        progress = lid_qpos if self.task_mode == "open" else -lid_qpos
        if self.task_mode == "open":
            open_info = self._box_open_stage_info(self.open_threshold)
            success_condition = open_info["success_condition"]
            stage_qualified = self._qualify_stage(0, success_condition)
            self.stage_box_open |= stage_qualified
            success = self.stage_box_open
        else:
            success_condition = is_closed
            success = self._latch_success(success_condition)
        return dict(
            success=success,
            success_condition=success_condition,
            box_lid_qpos=lid_qpos,
            box_lid_qpos_all=lid_qpos_all,
            box_lid_target_qpos=target,
            box_lid_progress=progress,
            stage_id=self._open_close_stage_id(
                self.open_threshold, self.close_threshold, self.task_mode
            ),
            is_open=is_open,
            is_closed=is_closed,
            open_process_ok=self.ever_left_hand_opened_box,
            stage_box_open=self.stage_box_open,
        )

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs.update(open_close_stage_id=info["stage_id"].unsqueeze(-1))
        return obs

    def compute_sparse_reward(self, obs: Any, action: Array, info: dict):
        success_reward = info["success"].to(dtype=torch.float32)
        info["reward_terms"] = {"success_reward": success_reward}
        return success_reward

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        if self.task_mode == "open":
            lid_progress = self._box_lid_open_progress()
            qpos_error = torch.clamp(self.open_threshold - lid_progress, min=0.0)
        else:
            lid_progress = self._box_lid_close_progress()
            qpos_error = torch.clamp(lid_progress - self.close_threshold, min=0.0)
        reaching_dist = torch.linalg.norm(self.agent.tcp.pose.p - self.box.pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(5.0 * reaching_dist)
        articulation_reward = 1 - torch.tanh(3.0 * qpos_error)
        reward = reaching_reward + 2.0 * articulation_reward
        reward[info["success"]] = 5.0
        info["reward_terms"] = {
            "reaching_reward": reaching_reward,
            "articulation_reward": articulation_reward,
        }
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0


class _ScoopTomatoTask(MealPreparationBase):
    """Stage 3 standalone: SCOOP the tomato out of the open box with the spoon and
    put it on the plate (the earlier hand pick-and-place is gone).
    Old text kept for the record:

    Replaced the spoon-scooping stage -- see
    ``_ball_stage_success_info`` for why scooping is not reachable with a
    rigid-pad hand. The tomato still starts INSIDE the box, so the open-box
    milestone stays a real physical prerequisite.
    """

    has_box = True
    has_plate = True
    has_spoon = True
    has_bread = True
    has_ball_food = True

    def __init__(self, *args, open_threshold=BOX_OPEN_ANGLE,
                 plate_ball_xy_threshold=PLATE_BALL_XY_THRESHOLD,
                 plate_bread_xy_threshold=PLATE_BREAD_XY_THRESHOLD, **kwargs):
        self.open_threshold = open_threshold
        self.plate_ball_xy_threshold = plate_ball_xy_threshold
        self.plate_bread_xy_threshold = plate_bread_xy_threshold
        # Entry distribution = the accepted output of stage 2: lid open, bread
        # already on the plate, tomato still in the box.
        kwargs.setdefault("initial_box_lid_qpos",
                          open_threshold + BOX_OPEN_ENTRY_MARGIN)
        kwargs.setdefault("lock_box_lid_until_opening", False)
        super().__init__(*args, **kwargs)

    def _initialize_actors(self, env_idx: torch.Tensor):
        super()._initialize_actors(env_idx)
        open_qpos = self.open_threshold + torch.stack(
            [
                torch.as_tensor(
                    self._batched_episode_rng[i].uniform(0.0, 0.12),
                    device=self.device,
                    dtype=torch.float32,
                )
                for i in range(len(env_idx))
            ],
            dim=0,
        )
        self._set_box_lid_qpos_values(env_idx, open_qpos)
        self._episode_open_lid_qpos = open_qpos
        self._place_bread_on_plate(env_idx)
        self.initial_ball_in_box[env_idx] = self._ball_in_box()[env_idx]
        # Milestones inherited from stages 1 and 2.
        self.ever_left_hand_opened_box[env_idx] = True
        self.stage_box_open[env_idx] = True
        self.ever_bread_manipulated[env_idx] = True
        self.ever_bread_grasped[env_idx] = True
        self.ever_bread_transport[env_idx] = True
        self.stage_bread_done[env_idx] = True
        self.ever_ball_left_box[env_idx] = False
        self.ever_ball_scooped[env_idx] = False
        self.ever_ball_transport[env_idx] = False
        self.ever_right_hand_used_spoon[env_idx] = False

    def evaluate(self):
        self._update_stage_markers()
        stage_info = self._ball_stage_success_info(
            self.open_threshold,
            self.plate_ball_xy_threshold,
            self.plate_bread_xy_threshold,
        )
        ball_info = stage_info["ball_info"]
        ball_left_box = self._ball_left_box()
        success_condition = stage_info["success_condition"]
        stage_qualified = self._qualify_stage(2, success_condition)
        self.stage_ball_done |= stage_qualified
        success = self.stage_ball_done
        info = dict(
            success=success,
            success_condition=success_condition,
            ball_left_box=ball_left_box,
            initial_ball_in_box=self.initial_ball_in_box,
            ever_ball_left_box=self.ever_ball_left_box,
            ever_ball_scooped=self.ever_ball_scooped,
            ever_right_hand_used_spoon=self.ever_right_hand_used_spoon,
            ever_ball_transport=self.ever_ball_transport,
            ball_grasped=self.current_ball_scooped,
            stage_id=self._ball_stage_id(),
            ball_process_ok=stage_info["process_ok"],
            box_open=stage_info["box_open"],
            stage_box_open=self.stage_box_open,
            stage_bread_done=self.stage_bread_done,
            stage_ball_done=self.stage_ball_done,
            is_robot_static=stage_info["is_robot_static"],
            ball_to_plate_xy_dist=ball_info["xy_dist_to_plate"],
            ball_z_above_plate=ball_info["z_above_plate"],
            is_ball_static=ball_info["is_static"],
            ball_on_plate=ball_info["on_plate"],
            bread_on_plate=stage_info["bread_info"]["on_plate"],
        )
        info.update(
            {
                f"ball_{key}": value
                for key, value in self._dexterous_object_reach_info(
                    self.ball_food, object_radius=APPLE_RADIUS
                ).items()
            }
        )
        return info

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs.update(
            ball_food_pose=self.ball_food.pose.raw_pose,
            ball_on_plate=info["ball_on_plate"].unsqueeze(-1),
            ball_process_ok=info["ball_process_ok"].unsqueeze(-1),
            ball_stage_id=info["stage_id"].unsqueeze(-1),
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                ball_to_plate_pos=self.plate.pose.p - self.ball_food.pose.p,
                ball_left_box=info["ball_left_box"].unsqueeze(-1),
                initial_ball_in_box=info["initial_ball_in_box"].unsqueeze(-1),
                ever_ball_left_box=info["ever_ball_left_box"].unsqueeze(-1),
                ever_ball_scooped=info["ever_ball_scooped"].unsqueeze(-1),
                ever_right_hand_used_spoon=info["ever_right_hand_used_spoon"].unsqueeze(-1),
                ever_ball_transport=info["ever_ball_transport"].unsqueeze(-1),
                ball_grasped=info["ball_grasped"].unsqueeze(-1),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: Array, info: dict):
        success_reward = info["success"].to(dtype=torch.float32)
        info["reward_terms"] = {"success_reward": success_reward}
        return success_reward

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_to_ball_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - self.ball_food.pose.p, axis=1
        )
        reach_reward = 1 - torch.tanh(5.0 * tcp_to_ball_dist)
        plate_reward = 1 - torch.tanh(8.0 * info["ball_to_plate_xy_dist"])
        process_reward = (
            0.5 * info["initial_ball_in_box"].to(torch.float32)
            + 0.5 * info["ever_ball_scooped"].to(torch.float32)
            + 0.5 * info["ever_ball_left_box"].to(torch.float32)
            + 0.5 * info["ever_ball_transport"].to(torch.float32)
        )
        reward = reach_reward + 2.0 * plate_reward + process_reward
        reward[info["success"]] = 5.0
        info["reward_terms"] = {
            "reach_reward": reach_reward,
            "plate_reward": plate_reward,
            "process_reward": process_reward,
        }
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0

class _PickAndPlaceBreadTask(MealPreparationBase):
    has_box = True
    has_plate = True
    has_spoon = True
    has_bread = True
    has_ball_food = True

    def __init__(
        self,
        *args,
        open_threshold=BOX_OPEN_ANGLE,
        plate_xy_threshold=PLATE_BREAD_XY_THRESHOLD,
        **kwargs,
    ):
        self.open_threshold = open_threshold
        self.plate_xy_threshold = plate_xy_threshold
        # Entry distribution = the accepted output of stage 1: the lid is open
        # and nothing else has been moved yet. The tomato stays in the box; it
        # is stage 3's job (order changed to bread-then-tomato).
        kwargs.setdefault("initial_box_lid_qpos",
                          open_threshold + BOX_OPEN_ENTRY_MARGIN)
        kwargs.setdefault("lock_box_lid_until_opening", False)
        super().__init__(*args, **kwargs)

    def _initialize_actors(self, env_idx: torch.Tensor):
        super()._initialize_actors(env_idx)
        with torch.device(self.device):
            b = len(env_idx)
            open_qpos = self.open_threshold + torch.stack(
                [
                    torch.as_tensor(
                        self._batched_episode_rng[i].uniform(0.0, 0.12),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    for i in range(b)
                ],
                dim=0,
            )
            self._set_box_lid_qpos_values(env_idx, open_qpos)
            self._episode_open_lid_qpos = open_qpos
            del b
            self.initial_ball_in_box[env_idx] = self._ball_in_box()[env_idx]
            self.ever_left_hand_opened_box[env_idx] = True
            self.stage_box_open[env_idx] = True

    def evaluate(self):
        self._update_stage_markers()
        stage_info = self._bread_stage_success_info(
            self.open_threshold,
            self.plate_xy_threshold,
        )
        bread_info = stage_info["bread_info"]
        success_condition = stage_info["success_condition"]
        stage_qualified = self._qualify_stage(1, success_condition)
        self.stage_bread_done |= stage_qualified
        success = self.stage_bread_done
        reach_info = self._dexterous_object_reach_info(self.bread, object_radius=0.12)
        return dict(
            success=success,
            success_condition=success_condition,
            bread_to_plate_xy_dist=bread_info["xy_dist_to_plate"],
            bread_z_above_plate=bread_info["z_above_plate"],
            bread_on_plate=bread_info["on_plate"],
            is_bread_static=bread_info["is_static"],
            ever_bread_manipulated=self.ever_bread_manipulated,
            ever_bread_grasped=self.ever_bread_grasped,
            ever_bread_transport=self.ever_bread_transport,
            bread_transport_steps=self.bread_transport_steps,
            stage_id=self._bread_stage_id(),
            box_open=stage_info["box_open"],
            # Live gate diagnostics. The placement gates and the PROCESS gates
            # are separate: bread sitting perfectly on the plate still fails if
            # it was never grasped and carried there. Reported only.
            bread_xy_to_plate=bread_info["xy_dist_to_plate"],
            robot_static=self.agent.is_static(0.2),
            hand_closure_right=self._hand_closure("right"),
            bread_process_ok=stage_info["process_ok"],
            stage_box_open=self.stage_box_open,
            stage_bread_done=self.stage_bread_done,
            is_robot_static=stage_info["is_robot_static"],
            **{f"bread_{key}": value for key, value in reach_info.items()},
        )

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs.update(
            bread_pose=self.bread.pose.raw_pose,
            bread_on_plate=info["bread_on_plate"].unsqueeze(-1),
            ever_bread_manipulated=info["ever_bread_manipulated"].unsqueeze(-1),
            ever_bread_grasped=info["ever_bread_grasped"].unsqueeze(-1),
            ever_bread_transport=info["ever_bread_transport"].unsqueeze(-1),
            bread_stage_id=info["stage_id"].unsqueeze(-1),
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                tcp_to_bread_pos=self.bread.pose.p - self.agent.tcp.pose.p,
                bread_to_plate_pos=self.plate.pose.p - self.bread.pose.p,
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: Array, info: dict):
        success_reward = info["success"].to(dtype=torch.float32)
        info["reward_terms"] = {"success_reward": success_reward}
        return success_reward

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_to_bread_dist = torch.linalg.norm(self.agent.tcp.pose.p - self.bread.pose.p, axis=1)
        reach_reward = 1 - torch.tanh(5.0 * tcp_to_bread_dist)
        plate_reward = 1 - torch.tanh(8.0 * info["bread_to_plate_xy_dist"])
        finger_reward = 1 - torch.tanh(12.0 * info["bread_closest_tip_surface_dist"])
        process_reward = (
            info["ever_bread_manipulated"].to(torch.float32)
            + info["ever_bread_grasped"].to(torch.float32)
            + info["ever_bread_transport"].to(torch.float32)
        ) / 3.0
        reward = 0.5 * reach_reward + 0.5 * finger_reward + 2.0 * plate_reward + process_reward
        reward[info["success"]] = 6.0
        info["reward_terms"] = {
            "reach_reward": reach_reward,
            "finger_reward": finger_reward,
            "plate_reward": plate_reward,
            "process_reward": process_reward,
        }
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0


# Backward-compatible base names for scripts that imported these classes from
# the original monolithic module. Public registered environments live in the
# dedicated sub-task modules.
OpenCloseBoxEnv = _OpenCloseBoxTask
ScoopTomatoEnv = _ScoopTomatoTask
PickAndPlaceTomatoEnv = _ScoopTomatoTask   # legacy name
PickAndPlaceBreadEnv = _PickAndPlaceBreadTask


# ---- assets shared by the ScoopTomato subtask and the long task (alignment) ----
# The spoon is 0.6x as long (its fixed-size grip sleeve moves with the mesh, the mass is
# kept) and the tomato 0.6x as large. A subtask must be a strict stage of its long task, so
# both build the same assets from this one place.

SPOON_LENGTH_FACTOR = 0.6
TOMATO_SIZE_FACTOR = 0.6
SPOON_MASS_BEFORE_RESIZE = 0.1371392458677292

def _build_scaled_spoon_actor(scene, asset_info, scale, initial_pose):
    """Build the same spoon while moving its fixed-size grip with the mesh."""
    loader = scene.create_mjcf_loader()
    loader.visual_groups = [1]
    loader.scale = np.array(_mul_scale3(asset_info["registry_scale"], scale))
    mjcf_path = Path(asset_info["mjcf_path"])
    parsed = loader.parse(str(mjcf_path), package_dir=str(mjcf_path.parent))
    actor_builders = parsed["actor_builders"]
    articulation_builders = parsed["articulation_builders"]
    if len(actor_builders) + len(articulation_builders) != 1:
        raise RuntimeError(
            f"Expected one RoboCasa spoon model in {mjcf_path}, got "
            f"{len(actor_builders)} actors and "
            f"{len(articulation_builders)} articulations."
        )
    if articulation_builders:
        builder = articulation_builders[0]
        builder.set_initial_pose(initial_pose)
        return builder.build(name="spoon", fix_root_link=False)

    builder = actor_builders[0]
    sleeve = SPOON_GRIP_SLEEVE
    center = list(sleeve["center"])
    center[2] *= SPOON_LENGTH_FACTOR
    pose = sapien.Pose(center, list(sleeve["quat"]))
    builder.add_capsule_collision(
        pose=pose,
        radius=sleeve["radius"],
        half_length=sleeve["half_length"],
        material=object_material(sleeve["material"]),
    )
    builder.add_capsule_visual(
        pose=pose,
        radius=sleeve["radius"],
        half_length=sleeve["half_length"],
        material=_make_material([0.12, 0.12, 0.14, 1.0]),
    )
    builder.set_initial_pose(initial_pose)
    spoon = builder.build_dynamic(name="spoon")
    component = spoon._objs[0].find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )
    _set_component_mass(component, SPOON_MASS_BEFORE_RESIZE)
    return spoon


class _ScaledFoodAssetsMixin:
    """The 0.6x spoon (sleeve moved, mass kept) and 0.6x tomato, for the ScoopTomato
    subtask and the long task alike. Precedes the family base in the MRO."""

    def _build_spoon(self):
        asset_info = _require_robocasa_actor_asset("spoon")
        self.robocasa_asset_info["spoon"] = asset_info
        print(f"[MealPreparation] Using RoboCasa spoon asset: {asset_info['asset_dir']}")
        spoon = _build_scaled_spoon_actor(
            self.scene,
            asset_info,
            scale=_mul_scale3(
                ROBOCASA_OBJECT_SCALES["spoon"],
                (1.0, 1.0, SPOON_LENGTH_SCALE * SPOON_LENGTH_FACTOR),
            ),
            initial_pose=sapien.Pose(
                p=self.spoon_pose_xyz.tolist(),
                q=self.spoon_pose_q.tolist(),
            ),
        )
        lowest = _collision_min_z_under_quat(
            spoon, self.spoon_pose_q.tolist()
        )
        self.spoon_pose_xyz = self.spoon_pose_xyz.clone()
        self.spoon_pose_xyz[2] = float(-lowest + 0.0005)
        print(
            f"[MealPreparation] spoon pose set to "
            f"({self.spoon_pose_xyz[0]:.3f}, {self.spoon_pose_xyz[1]:.3f}, "
            f"{self.spoon_pose_xyz[2]:.3f})"
        )
        spoon.set_pose(
            sapien.Pose(
                p=self.spoon_pose_xyz.tolist(),
                q=self.spoon_pose_q.tolist(),
            )
        )
        return spoon

    def _build_ball_food(self):
        asset_info = _require_robocasa_actor_asset("ball_food")
        self.robocasa_asset_info["ball_food"] = asset_info
        if self.has_box:
            self.ball_pose_xyz[:2] = self.box_pose_xyz[:2]
            self.ball_pose_xyz = self._robocasa_pose_in_box(
                "ball_food", asset_info, self.ball_pose_xyz
            )
        else:
            self.ball_pose_xyz = self._robocasa_pose_on_table(
                "ball_food", asset_info, self.ball_pose_xyz
            )
        print(
            "[MealPreparation] Using RoboCasa round food asset for "
            f"ball_food: {asset_info['asset_dir']}"
        )
        ball_food = _build_robocasa_actor(
            self.scene,
            asset_info,
            name="round_ball_food",
            scale=ROBOCASA_OBJECT_SCALES["ball_food"] * TOMATO_SIZE_FACTOR,
            initial_pose=sapien.Pose(self.ball_pose_xyz.tolist()),
        )
        return _configure_task_object_physics(ball_food, "ball_food")


#: One budget for the three long-task stages: OpenCloseBox 300 + PickAndPlaceBread 400 +
#: ScoopTomato 1200. The earlier 360 was shorter than the ScoopTomato stage alone.
LONG_TASK_MAX_EPISODE_STEPS = 1900


class DualMealPreparationEnv(_ScaledFoodAssetsMixin, MealPreparationBase):
    def __init__(
        self,
        *args,
        open_threshold=BOX_OPEN_ANGLE,
        plate_ball_xy_threshold=PLATE_BALL_XY_THRESHOLD,
        plate_bread_xy_threshold=PLATE_BREAD_XY_THRESHOLD,
        **kwargs,
    ):
        self.open_threshold = open_threshold
        self.plate_ball_xy_threshold = plate_ball_xy_threshold
        self.plate_bread_xy_threshold = plate_bread_xy_threshold
        kwargs.setdefault("initial_box_lid_qpos", PROCEDURAL_BOX_CLOSED_QPOS)
        kwargs.setdefault("lock_box_lid_until_opening", False)
        super().__init__(*args, **kwargs)
        if self.robot_uids not in DUAL_HAND_ROBOT_UIDS:
            raise ValueError(
                f"DualMealPreparationEnv requires a dual-hand robot, got {self.robot_uids!r}"
            )

    def evaluate(self):
        self._update_stage_markers()
        open_info = self._box_open_stage_info(self.open_threshold)
        stage_one_qualified = self._qualify_stage(
            0, open_info["success_condition"]
        )
        self.stage_box_open |= stage_one_qualified

        # ORDER: open box (left) -> bread, the LARGE object (right) -> tomato,
        # the SMALL object (right). Changed: stage 2 used to be a
        # spoon scoop, which a rigid-pad hand cannot perform.
        bread_info = self._bread_stage_success_info(
            self.open_threshold, self.plate_bread_xy_threshold
        )
        stage_two_qualified = self._qualify_stage(
            1, bread_info["success_condition"]
        )
        self.stage_bread_done |= stage_two_qualified

        final_info = self._ball_stage_success_info(
            self.open_threshold,
            self.plate_ball_xy_threshold,
            self.plate_bread_xy_threshold,
        )
        stage_three_qualified = self._qualify_stage(
            2, final_info["success_condition"]
        )
        self.stage_ball_done |= stage_three_qualified

        lid_qpos = self._box_lid_qpos()
        lid_qpos_all = self._box_lid_qpos_all()
        box_open = open_info["box_open"]
        bread_stage_info = bread_info
        bread_info = final_info["bread_info"]
        ball_info = final_info["ball_info"]
        is_robot_static = final_info["is_robot_static"]
        open_process_ok = self.ever_left_hand_opened_box
        bread_process_ok = bread_stage_info["process_ok"]
        ball_process_ok = final_info["process_ok"]
        bread_done = bread_stage_info["success_condition"]
        ball_done = final_info["success_condition"]
        success_condition = final_info["success_condition"]
        success = self.stage_ball_done

        left_box_dist = torch.linalg.norm(self._tcp_pos("left") - self.box.pose.p, axis=1)
        right_bread_dist = torch.linalg.norm(self._tcp_pos("right") - self.bread.pose.p, axis=1)
        right_ball_dist = torch.linalg.norm(self._tcp_pos("right") - self.ball_food.pose.p, axis=1)

        left_box_reach = self._side_object_reach_info(self.box, side="left", object_radius=0.12)
        right_ball_reach = self._side_object_reach_info(
            self.ball_food, side="right", object_radius=APPLE_RADIUS
        )
        right_bread_reach = self._side_object_reach_info(self.bread, side="right", object_radius=0.12)

        return dict(
            success=success,
            success_condition=success_condition,
            box_open=box_open,
            box_lid_qpos=lid_qpos,
            box_lid_qpos_all=lid_qpos_all,
            ball_done=ball_done,
            bread_done=bread_done,
            open_process_ok=open_process_ok,
            bread_process_ok=bread_process_ok,
            ball_process_ok=ball_process_ok,
            stage_box_open=self.stage_box_open,
            stage_ball_done=self.stage_ball_done,
            stage_bread_done=self.stage_bread_done,
            initial_ball_in_box=self.initial_ball_in_box,
            ever_ball_left_box=self.ever_ball_left_box,
            ever_ball_manipulated=self.ever_ball_manipulated,
            ever_ball_scooped=self.ever_ball_scooped,
            ever_right_hand_used_spoon=self.ever_right_hand_used_spoon,
            ever_ball_transport=self.ever_ball_transport,
            ever_bread_manipulated=self.ever_bread_manipulated,
            ever_bread_grasped=self.ever_bread_grasped,
            ever_bread_transport=self.ever_bread_transport,
            bread_transport_steps=self.bread_transport_steps,
            ever_left_hand_opened_box=self.ever_left_hand_opened_box,
            stage_id=self._dual_stage_id(),
            ball_on_plate=ball_info["on_plate"],
            bread_on_plate=bread_info["on_plate"],
            is_ball_static=ball_info["is_static"],
            is_bread_static=bread_info["is_static"],
            is_robot_static=is_robot_static,
            ball_to_plate_xy_dist=ball_info["xy_dist_to_plate"],
            bread_to_plate_xy_dist=bread_info["xy_dist_to_plate"],
            ball_z_above_plate=ball_info["z_above_plate"],
            bread_z_above_plate=bread_info["z_above_plate"],
            left_tcp_to_box_dist=left_box_dist,
            right_tcp_to_bread_dist=right_bread_dist,
            right_tcp_to_ball_dist=right_ball_dist,
            **{f"left_box_{key}": value for key, value in left_box_reach.items()},
            **{f"right_ball_{key}": value for key, value in right_ball_reach.items()},
            **{f"right_bread_{key}": value for key, value in right_bread_reach.items()},
        )

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs.update(
            box_open=info["box_open"].unsqueeze(-1),
            stage_box_open=info["stage_box_open"].unsqueeze(-1),
            stage_ball_done=info["stage_ball_done"].unsqueeze(-1),
            stage_bread_done=info["stage_bread_done"].unsqueeze(-1),
            ball_on_plate=info["ball_on_plate"].unsqueeze(-1),
            bread_on_plate=info["bread_on_plate"].unsqueeze(-1),
            ball_done=info["ball_done"].unsqueeze(-1),
            bread_done=info["bread_done"].unsqueeze(-1),
            open_process_ok=info["open_process_ok"].unsqueeze(-1),
            bread_process_ok=info["bread_process_ok"].unsqueeze(-1),
            ball_process_ok=info["ball_process_ok"].unsqueeze(-1),
            ever_bread_grasped=info["ever_bread_grasped"].unsqueeze(-1),
            ever_bread_transport=info["ever_bread_transport"].unsqueeze(-1),
            dual_stage_id=info["stage_id"].unsqueeze(-1),
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                left_tcp_to_box_pos=self.box.pose.p - self._tcp_pos("left"),
                right_tcp_to_ball_pos=self.ball_food.pose.p - self._tcp_pos("right"),
                right_tcp_to_bread_pos=self.bread.pose.p - self._tcp_pos("right"),
                ball_to_plate_pos=self.plate.pose.p - self.ball_food.pose.p,
                bread_to_plate_pos=self.plate.pose.p - self.bread.pose.p,
                initial_ball_in_box=info["initial_ball_in_box"].unsqueeze(-1),
                ever_ball_left_box=info["ever_ball_left_box"].unsqueeze(-1),
                ever_ball_scooped=info["ever_ball_scooped"].unsqueeze(-1),
                ever_right_hand_used_spoon=info["ever_right_hand_used_spoon"].unsqueeze(-1),
                ever_ball_transport=info["ever_ball_transport"].unsqueeze(-1),
                ever_bread_manipulated=info["ever_bread_manipulated"].unsqueeze(-1),
                ever_bread_grasped=info["ever_bread_grasped"].unsqueeze(-1),
                ever_bread_transport=info["ever_bread_transport"].unsqueeze(-1),
                ever_left_hand_opened_box=info["ever_left_hand_opened_box"].unsqueeze(-1),
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: Array, info: dict):
        milestones = torch.stack(
            [
                info["stage_box_open"],
                info["stage_ball_done"],
                info["stage_bread_done"],
            ], dim=-1,
        ).to(torch.float32)
        reward = milestones.sum(dim=-1)
        info["reward_terms"] = {
            "stage_1_reward": milestones[:, 0],
            "stage_2_reward": milestones[:, 1],
            "stage_3_reward": milestones[:, 2],
            "success_reward": info["success"].to(torch.float32),
        }
        return reward

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        lid_error = torch.clamp(self.open_threshold - self._box_lid_open_progress(), min=0.0)
        open_reward = 1 - torch.tanh(3.0 * lid_error)
        left_reach_reward = 1 - torch.tanh(5.0 * info["left_tcp_to_box_dist"])
        right_ball_reward = 1 - torch.tanh(5.0 * info["right_tcp_to_ball_dist"])
        ball_plate_reward = 1 - torch.tanh(8.0 * info["ball_to_plate_xy_dist"])
        right_bread_reward = 1 - torch.tanh(5.0 * info["right_tcp_to_bread_dist"])
        bread_finger_reward = 1 - torch.tanh(12.0 * info["right_bread_closest_tip_surface_dist"])
        bread_plate_reward = 1 - torch.tanh(8.0 * info["bread_to_plate_xy_dist"])

        stage_bonus = (
            info["box_open"].to(torch.float32)
            + info["ball_done"].to(torch.float32)
            + info["bread_done"].to(torch.float32)
            + info["open_process_ok"].to(torch.float32)
            + info["bread_process_ok"].to(torch.float32)
            + info["ball_process_ok"].to(torch.float32)
            + info["ever_bread_grasped"].to(torch.float32)
            + info["ever_bread_transport"].to(torch.float32)
            + info["ever_ball_scooped"].to(torch.float32)
            + info["ever_ball_transport"].to(torch.float32)
        )
        reward = (
            0.5 * left_reach_reward
            + 1.0 * open_reward
            + 0.5 * right_ball_reward
            + 1.5 * ball_plate_reward
            + 0.5 * right_bread_reward
            + 0.5 * bread_finger_reward
            + 1.5 * bread_plate_reward
            + stage_bonus
        )
        reward[info["success"]] = 10.0
        info["reward_terms"] = {
            "left_reach_reward": left_reach_reward,
            "open_reward": open_reward,
            "right_ball_reward": right_ball_reward,
            "ball_plate_reward": ball_plate_reward,
            "right_bread_reward": right_bread_reward,
            "bread_finger_reward": bread_finger_reward,
            "bread_plate_reward": bread_plate_reward,
            "stage_bonus": stage_bonus,
        }
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10.0

    def _bread_stage_success_info(self, open_threshold, bread_xy_threshold):
        """Same relaxation as ``_StableBreadOnPlateTask``: the loaf in the plate
        region for the dwell is the outcome; the grasp / transport markers stay
        diagnostics (``bread_process_ok`` is still reported by ``evaluate``).

        The subtask pre-latches the box milestone, so its ``on_plate``-only
        condition already implies "box open"; here the milestone is real and
        stays in the condition, which keeps the stage order.
        """
        info = super()._bread_stage_success_info(open_threshold, bread_xy_threshold)
        info["success_condition"] = self.stage_box_open & info["bread_info"]["on_plate"]
        return info


@register_task(
    "MealPreparationLong",
    side="both",
    max_episode_steps=LONG_TASK_MAX_EPISODE_STEPS,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    reward_mode="sparse",
    render_mode="rgb_array",
)
class MealPreparationLongEnv(DualMealPreparationEnv):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", LONG_TASK_ROBOT_UID)
        super().__init__(*args, **kwargs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="MealPreparationOpenLunchBox-Sharpa-v1")
    parser.add_argument("--task_mode", default="open", choices=["open", "close"])
    parser.add_argument("--video_dir", default="meal_preparation_videos/open_close_box")
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--num_steps", default=120, type=int)
    parser.add_argument("--sim_backend", default="physx_cuda")
    parser.add_argument(
        "--render_camera_view",
        default="overview",
        choices=["overview", "wide_overview", "box_inside", "box_overhead", "box_front_close"],
    )
    args = parser.parse_args()

    env_kwargs = dict(
        num_envs=args.num_envs,
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
    )
    env_kwargs["render_camera_view"] = args.render_camera_view
    if "OpenCloseBox" in args.env_id:
        env_kwargs["task_mode"] = args.task_mode

    env = gym.make(
        args.env_id,
        **env_kwargs,
    )
    env = RecordEpisode(
        env,
        output_dir=args.video_dir,
        save_trajectory=True,
        save_video=True,
        max_steps_per_video=args.num_steps,
        video_fps=20,
    )
    obs, _ = env.reset(seed=0)
    done = False
    step = 0
    while not done and step < args.num_steps:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = bool(terminated.item() or truncated.item())
        step += 1
    env.close()

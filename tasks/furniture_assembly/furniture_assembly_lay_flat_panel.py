"""PanelLayFlat: subtask 1 of DualFurnitureAssembly, multi-end-effector, single arm.

A wooden drawer side panel stands vertically in a slotted rack. The arm pulls it out of the
slot and lays it flat on the table with the machined groove facing UP. Subtask 2 (rail into
groove) starts from the state this task ends in.

Registered environments (tasks/config/hands.py; the long task does this stage with the LEFT arm)
    FurnitureAssemblyLayFlatPanel-Sharpa-v1   robot_uids="panda_sharpa_hand_left_angled_mobile_base"
    FurnitureAssemblyLayFlatPanel-Wuji-v1     robot_uids="panda_wuji_hand_left_angled_mobile_base"
    FurnitureAssemblyLayFlatPanel-XHand-v1    robot_uids="panda_xhand_hand_left_angled_mobile_base"
    FurnitureAssemblyLayFlatPanel-Gripper-v1  robot_uids="panda_gripper_left_mobile_base"

ACTION  (DEFAULT_CONTROL_MODE = "pd_joint_delta_pos")
    Joint space. Nothing in this file is end-effector specific, so any control_mode the agent
    exposes works; only the default registration changed.
      pd_joint_delta_pos  action in [-1, 1] -> +-0.1 rad per joint per control step.
                          Zeros mean "hold". This is the RL default.
      pd_joint_pos        RAW target joint angles (Panda registers this with
                          normalize_action=False), so zeros command every joint to 0 rad and
                          the arm snaps. Hold = current qpos. Use this for recording and
                          replaying demonstrations; it is what replay_trajectory targets.
    Panda's trailing gripper DoF is normalized either way: +1 open, -1 closed.

SUCCESS  (stateless, so trajectory replay reproduces it exactly)
    All of the following hold for SETTLE_STEPS consecutive control steps:
      panel_normal_z  >= FLATNESS_THRESHOLD   groove faces up, not merely horizontal
      |panel_z - h|   <= REST_Z_TOLERANCE     resting on the table, h = PANEL_HALF[2]
      panel footprint inside the rectangular assembly zone, with yaw error <= 20 deg
      contact_count   == 0                    the robot has let go
      panel velocity  <  thresholds            it has stopped moving
    Deliberately loose: no target yaw, no exact target pose. A panel lying flat on the table
    cannot still be in the slot, so "was extracted" is implied and is not checked separately.

STATE  (obs_mode="state"; see _get_obs_extra for the exact ordering)
    always            tcp_pose(7) panel_pose(7) tcp_to_grasp(3) place_center(3)
    state only        all_tip_positions(3*n_tip) groove_landmarks(6) groove_axis(3)
                      panel_normal(3) panel_lin_vel(3) panel_ang_vel(3) rack_pose(7)
                      panel_to_place(3) contact_count(1) gripper_opening(1)
                      extract_progress(1) flatness(1) height_above_rack(1)
                      is_grasped(1) is_out_of_rack(1) is_released(1) is_static(1)
                      is_flat_on_table(1) in_place_region(1)
    groove_landmarks and groove_axis are exported for subtasks 2 and 3.

REWARD  five stages, prefix-product gated, matching the ButtonPress convention:
    r_reach -> r_grasp -> r_extract -> r_flat -> r_settle
    Milestone latches keep an earlier stage at 1 once passed. Without them r_grasp collapses
    the moment the robot correctly releases the panel, i.e. exactly at success. The latches
    are reward-only and never feed into `success`.

ASSET NOTES
    The panel is real geometry: panel_collision_boxes() is a 10-box convex decomposition
    carving a full-length groove plus two square blind holes. SAPIEN cannot give a dynamic
    body a concave collision shape, and hand-authored boxes beat COACD for a shape this
    regular. Subtasks 2 and 3 import build_drawer_panel and the geometry constants from here.

    Free-standing the panel on its 16 mm edge is not viable: the CoM sits 75 mm above an 8 mm
    half-base, so it tips at atan(8/75) = 6.1 deg and solver noise alone knocks it over. The
    slot holds it with pure contact, no kinematic freezing.

    Friction is physical (wood on wood ~0.5). High friction belongs on the fingertip material,
    not on the workpiece.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np
import sapien
import sapien.physx as physx
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from tasks.config.hands import register_task
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SceneConfig, SimConfig
from mani_skill.utils.wrappers.record import RecordEpisode

from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.robots.panda_sharpa_hand_left_mobile_base import PandaSharpaHandLeftAngledMobileBase  # noqa: F401
from tasks.robots.panda_wuji_hand_left_mobile_base import PandaWujiHandLeftMobileBase  # noqa: F401
from tasks.config.object_materials import object_material
from tasks.scene_builders import TableSceneBuilder
from tasks.config.sim_profiles import make_sim_config
from tasks.config.workspace_rig import (
    ARM_INIT_QPOS,
    PLATFORM_ROOT_X,
    TABLE_CENTER_X,
    set_work_table_pose,
)
from tasks.config.energy_fence import set_task_object_energy_fence
from tasks.robots.panda_gripper_left_mobile_base import PandaGripperLeftMobileBase
from tasks.robots.panda_sharpa_hand_right_mobile_base import (
    PandaSharpaHandRightAngledMobileBase,
)
from tasks.robots.panda_wuji_hand_right_mobile_base import PandaWujiHandRightMobileBase
from tasks.config.task_state import DeclarativeTaskStateMixin, TaskStateField

try:
    from tasks.camera_configs import build_dexarena_base_camera
except Exception:
    def build_dexarena_base_camera(
        *,
        eye=(-0.5, -0.15, 0.6),
        target=(-0.15, -0.15, 0.1),
        width=320,
        height=240,
        fov_deg=58.0,
        near=0.01,
        far=100,
        name="base_camera",
    ) -> CameraConfig:
        return CameraConfig(
            str(name),
            sapien_utils.look_at(list(eye), list(target)),
            int(width),
            int(height),
            np.deg2rad(float(fov_deg)),
            float(near),
            float(far),
        )


GRIPPER_ROBOT_UID = "panda_gripper_left_mobile_base"
GRIPPER_ROBOT_UIDS = {"panda_gripper_left_mobile_base", "panda_gripper_right_mobile_base"}
HAND_ROBOT_UIDS = {
    # The long task does this stage with the LEFT hand (stage_names:
    # "place_panel_left"); the left variants are the strict-subset entries.
    "panda_sharpa_hand_left_angled_mobile_base",
    "panda_wuji_hand_left_angled_mobile_base",
    "panda_xhand_hand_left_angled_mobile_base",
    "panda_sharpa_hand_right_angled_mobile_base",
    "panda_wuji_hand_right_angled_mobile_base",
    "panda_xhand_hand_right_angled_mobile_base",
}
MOBILE_ROBOT_UIDS = {*GRIPPER_ROBOT_UIDS, *HAND_ROBOT_UIDS}

ENV_IDS = (
    "FurnitureAssemblyLayFlatPanel-Sharpa-v1",
    "FurnitureAssemblyLayFlatPanel-Wuji-v1",
    "FurnitureAssemblyLayFlatPanel-XHand-v1",
    "FurnitureAssemblyLayFlatPanel-Gripper-v1",
)
DEFAULT_CONTROL_MODE = "pd_joint_pos"
DEFAULT_VIDEO_DIR = "tasks/assets/task_object_meshes/furniture/videos_panel_lay_flat"
# Arm layer: the trolley/arm-root x is the shared rig's, never a local number.
ROBOT_ROOT_POSE = sapien.Pose(p=[PLATFORM_ROOT_X, 0.0, 0.0])


# ===========================================================================
# Shared asset geometry. Subtasks 2 and 3 import from here.
# ===========================================================================
PANEL_HALF = (0.190, 0.095, 0.009)   # 380 x 190 x 18 mm drawer side panel
GROOVE_HALF_W = 0.0145 # 29 mm dado (user: 2.5 mm play per side for the 24 mm rail; was 40)
GROOVE_DEPTH = 0.008 # 8 mm (was 5) -- tall enough to steer a rail being slid along it
# slide-in groove (user: cannot align to 3 deg by eye, and the pins/holes
# were invisible under the rail). No holes any more: the groove is CLOSED at its +x
# end by a stop wall and OPEN at -x through a flared mouth; the rail is slid in
# from the mouth until it hits the stop. Yaw is straightened by the mouth walls
# (+-15 deg drop-in), x by the stop (push until it stops), y by the groove walls.
GROOVE_STOP_X = 0.170                # inner face of the stop wall (panel frame); wall fills 0.170..+hx
GROOVE_MOUTH_FLARE = 0.045           # the groove half-width grows by this at the -x edge: 40 -> 130 mm mouth
GROOVE_MOUTH_LENGTH = 0.140          # flare runs 140 mm in from the edge (atan(45/140) = 17.8 deg walls; 90 mm lost the 15-20 deg capture once the groove went to 2.5 mm play)
# (user: "the panel cannot be picked up at all"): the pad pair is the
# calibrated wood-on-pad 0.9 and was not the problem -- the mass was. Sharpa's
# thumb saturates at ~3.3 N, so pinching an 18 mm edge carries at most
# 2 * 0.9 * 3.3 = 5.9 N (0.6 kg) at full squeeze and about half at a teleop-
# typical squeeze; the 0.81 kg MDF panel could not be lifted by any grip that
# does not wrap it. Flat-pack drawer sides are hollow-core / honeycomb board
# (IKEA class): ~240 kg/m3 puts this 380 x 190 x 18 mm panel at 0.30 kg.
# (user: "can pick it up now, but it sometimes slides down"): 160 kg/m3
# -> 0.20 kg, the light end of hollow-core board, i.e. ~2 N of pinch friction needed at
# the calibrated 0.9 pad pair instead of 3.3 N.
PANEL_DENSITY = 160.0                # hollow-core board -> 0.20 kg for this decomposition

RACK_BASE_HALF = (0.210, 0.075, 0.007)
RACK_SLOT_HALF_W = 0.0105            # 1.5 mm clearance per side on an 18 mm panel
RACK_WALL_HALF_T = 0.008
RACK_WALL_H = 0.065                  # insertion depth
RACK_SLOT_FLOOR_Z = 2.0 * RACK_BASE_HALF[2]
RACK_TOP_Z = RACK_SLOT_FLOOR_Z + RACK_WALL_H

# Panel-local landmarks reused downstream.
GROOVE_LOCAL_Z = PANEL_HALF[2] - GROOVE_DEPTH
# Two groove landmarks exported to the subtasks instead of the old blind holes: the
# mouth centre (where the rail goes in) and the stop face (where it ends up).
GROOVE_LANDMARKS_LOCAL = ((-PANEL_HALF[0], 0.0, GROOVE_LOCAL_Z), (GROOVE_STOP_X, 0.0, GROOVE_LOCAL_Z))
# Grasp reference: mid-length, high on the 150 mm dimension so it clears the rack walls.
GRASP_LOCAL_POS = (0.0, 0.060, 0.0)
# Panel centre height when standing in the slot.
PANEL_IN_SLOT_Z = RACK_SLOT_FLOOR_Z + PANEL_HALF[1] + 0.0015


# ---------------------------------------------------------------------------
# Task layer: the furniture family's world layout, in ONE place.
#
# Anchored on the shared rig (tasks/config/workspace_rig.py): the work table is
# centred at TABLE_CENTER_X = +0.180 and the trolley stands at PLATFORM_ROOT_X
# = -0.870. Every x below is "distance in front of the table centre", so the
# whole family moves with the rig instead of carrying absolute literals.
#
# Earlier the long task used x = -0.36 and the subtasks x = -0.245
# against a private -1.145 trolley, i.e. the same panel sat 0.115 m apart in
# the two envs and neither was on the rig.
# ---------------------------------------------------------------------------
# (operator: "to reach the screw I have to straighten the whole rear
# half of the arm"): measured from the right arm root (-0.870, -0.220, 0.333) the
# screw head sat 0.869 m away (subtask) / 0.844 (long task) and the panel 0.900 --
# at the 0.90 m band ceiling. The whole family moves 0.070 m toward the arms and
# 0.060 m to the operator's right, in ONE place, and the rail's screw hole moves
# to its near end (see RAIL_SCREW_LOCAL_POSITIONS). Measured after, same probes:
#   screw head 0.869 -> 0.744 (subtask), 0.844 -> 0.710 (long task)
#   panel 0.900 -> 0.824 | rail 0.920 -> 0.843 | tool 0.887 -> 0.827
#   rack, left arm 0.877 -> 0.812 | racked panel, left arm 0.841 -> 0.773
#   left-hand subtask variants: rail 0.965 -> 0.936, panel 0.900 -> 0.855
# The pull-in is capped at 0.070 m by the four-layer gate, NOT by preference: the
# gate measures the nearest collision vertex, and the 380 mm panel's near edge
# reaches the 0.54 m floor at 0.550 m here (0.480 at a 0.140 m pull-in).
IN_FRONT_OF_TABLE_CENTRE = 0.310      # the family's working lane
FAMILY_Y_SHIFT = -0.060               # whole family, toward the operator's right
RACK_CENTER = (TABLE_CENTER_X - IN_FRONT_OF_TABLE_CENTRE, 0.260 + FAMILY_Y_SHIFT)
PLACE_CENTER = (TABLE_CENTER_X - IN_FRONT_OF_TABLE_CENTRE, 0.000 + FAMILY_Y_SHIFT)
RAIL_PICK_CENTER = (TABLE_CENTER_X - IN_FRONT_OF_TABLE_CENTRE, -0.210 + FAMILY_Y_SHIFT) # 40 mm further to the operator's right (-y), on the right arm's line
TOOL_PICK_CENTER = (TABLE_CENTER_X - IN_FRONT_OF_TABLE_CENTRE + 0.020, -0.280 + FAMILY_Y_SHIFT)


# ===========================================================================
# Helpers
# ===========================================================================
def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    """Batched yaw angles -> wxyz quaternions."""
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    quat[:, 0] = torch.cos(yaw * 0.5)
    quat[:, 3] = torch.sin(yaw * 0.5)
    return quat


def axis_to_quat_wxyz(angle: torch.Tensor, axis: int) -> torch.Tensor:
    """Batched rotation about a single principal axis -> wxyz quaternions."""
    quat = torch.zeros((*angle.shape, 4), device=angle.device, dtype=angle.dtype)
    quat[..., 0] = torch.cos(angle * 0.5)
    quat[..., axis + 1] = torch.sin(angle * 0.5)
    return quat


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Batched wxyz quaternion product q1 * q2."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def make_render_material(
    rgba, roughness: float = 0.7, metallic: float = 0.0, specular: float = 0.3, textures=None
) -> sapien.render.RenderMaterial:
    """Build a RenderMaterial by attribute assignment; the constructor takes no keywords here.

    `metallic` is what separates wood from steel visually. Leaving it at the default is why
    every part in the first version of this asset rendered with the same plastic highlight.
    """
    material = sapien.render.RenderMaterial()
    material.base_color = np.asarray(rgba, dtype=np.float32)
    material.roughness = float(roughness)
    material.metallic = float(metallic)
    try:
        material.specular = float(specular)
    except Exception:
        pass
    for attr, path in (textures or {}).items():
        try:
            if path is not None and Path(path).is_file():
                setattr(material, attr, sapien.render.RenderTexture2D(str(path)))
        except Exception:
            pass
    return material


# Optional PBR maps. Present -> used; absent -> silently falls back to a flat base colour.
TEXTURE_DIR = Path(__file__).resolve().parents[1] / "assets" / "textures"
WOOD_TEXTURES = {
    "base_color_texture": TEXTURE_DIR / "wood_albedo.png",
    "normal_texture": TEXTURE_DIR / "wood_normal.png",
    "roughness_texture": TEXTURE_DIR / "wood_roughness.png",
}


# ===========================================================================
# Asset builders
# ===========================================================================
FINGER_KEYS = (("thumb",), ("index",), ("middle", "mid_"), ("ring",), ("pinky", "little"), ("finger1", "leftfinger"), ("finger2", "rightfinger"), ("finger3",), ("finger4",), ("finger5",))


def finger_contact_groups(agent, side: str = None) -> dict:
    """Hand links that CARRY collision geometry, grouped by finger.

    every furniture grasp/release test read contact forces on
    ``agent.tip_links`` -- on Sharpa the ``*_fingertip`` frames have 0 collision
    shapes, so "grasped" was False forever (and "released" True forever). With
    the screw stage now requiring a real engagement latch that made step 3
    unwinnable in teleop. Same fix as table_clean: count DISTINCT fingers whose
    collision-bearing links push on the object.
    """
    cache = getattr(agent, "_furniture_finger_groups", None)
    if cache is None:
        cache = {}
        agent._furniture_finger_groups = cache
    if side in cache:
        return cache[side]
    groups = {}
    for link in agent.robot.get_links():
        name = link.name.lower()
        if side is not None and not name.startswith(f"{side}_"):
            continue
        if len(link._objs[0].get_collision_shapes()) == 0:
            continue
        finger = next((i for i, keys in enumerate(FINGER_KEYS) if any(k in name for k in keys)), None)
        if finger is None:
            continue                      # palm / wrist shell: touching it is not a grasp
        groups.setdefault(finger, []).append(link)
    cache[side] = groups
    return groups


def finger_contact_count(scene, agent, actor, min_force: float, side: str = None):
    """(N,) number of distinct fingers in force contact with `actor` (0.0 = released)."""
    count = None
    for links in finger_contact_groups(agent, side).values():
        touching = None
        for link in links:
            try:
                force = scene.get_pairwise_contact_forces(link, actor)
            except Exception:
                continue
            hit = torch.linalg.norm(force, dim=1) > min_force
            touching = hit if touching is None else (touching | hit)
        if touching is not None:
            count = touching.float() if count is None else count + touching.float()
    if count is None:
        count = torch.zeros(agent.robot.get_qpos().shape[0], device=agent.robot.get_qpos().device)
    return count


def panel_collision_boxes() -> list:
    """[(centre, half_size), ...] -- 4 convex boxes: base slab, two straight side walls
    (from the end of the mouth flare to +x), and the stop wall closing the groove at +x.
    The two flared mouth walls are convex prisms, see panel_mouth_wedges().
    """
    hx, hy, hz = PANEL_HALF
    floor = hz - GROOVE_DEPTH
    boxes = [([0.0, 0.0, 0.5 * (-hz + floor)], [hx, hy, 0.5 * (floor + hz)])]
    wall_cz, wall_hz = 0.5 * (floor + hz), 0.5 * (hz - floor)
    x0 = -hx + GROOVE_MOUTH_LENGTH
    wall_cx, wall_hx = 0.5 * (x0 + hx), 0.5 * (hx - x0)
    wall_hy = 0.5 * (hy - GROOVE_HALF_W)
    for sgn in (-1.0, 1.0):
        boxes.append(([wall_cx, sgn * (GROOVE_HALF_W + wall_hy), wall_cz], [wall_hx, wall_hy, wall_hz]))
    stop_cx, stop_hx = 0.5 * (GROOVE_STOP_X + hx), 0.5 * (hx - GROOVE_STOP_X)
    boxes.append(([stop_cx, 0.0, wall_cz], [stop_hx, GROOVE_HALF_W, wall_hz]))
    return boxes


def panel_mouth_wedges() -> list:
    """Two convex prisms (8 vertices each, panel frame) forming the flared mouth walls."""
    hx, hy, hz = PANEL_HALF
    floor = hz - GROOVE_DEPTH
    x0 = -hx + GROOVE_MOUTH_LENGTH
    out = []
    for sgn in (-1.0, 1.0):
        quad = [(-hx, sgn * (GROOVE_HALF_W + GROOVE_MOUTH_FLARE)), (-hx, sgn * hy), (x0, sgn * hy), (x0, sgn * GROOVE_HALF_W)]
        out.append(np.array([[x, y, z] for z in (floor, hz) for (x, y) in quad], dtype=np.float64))
    return out


_GENERATED_DIR = Path(__file__).resolve().parents[1] / "assets" / "task_object_meshes" / "furniture" / "generated"


def mouth_wedge_files() -> list:
    """OBJ files of the mouth prisms, regenerated whenever the constants change."""
    import trimesh
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for k, verts in enumerate(panel_mouth_wedges()):
        path = _GENERATED_DIR / f"groove_mouth_wedge_{k}.obj"
        mesh = trimesh.convex.convex_hull(verts)
        stamp = f"# {np.round(verts, 6).tolist()}\n"
        if not path.exists() or not path.read_text().startswith(stamp):
            path.write_text(stamp + trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_texture=False))
        files.append(str(path))
    return files


def build_drawer_panel(
    scene: ManiSkillScene,
    name: str = "drawer_panel",
    static_friction: float = None,
    dynamic_friction: float = None,
    base_color=(0.725, 0.541, 0.369, 1.0),
    use_textures: bool = True,
    visual_detail: bool = True,
    kinematic: bool = False,
):
    """Drawer side panel with a real slide-in groove: flared mouth at -x, stop wall at +x.

    Everything added under `visual_detail` is visual-only, so it costs render time but not a
    single extra collision pair. Coplanar visuals are pushed out by >=0.3 mm to avoid
    z-fighting against the collision boxes they sit on.
    """
    builder = scene.create_actor_builder()
    # Task layer: MDF drawer panel -> the shared table's "wood" class. The
    # panel is carried, so the pad pair is the dominant one.
    phys = (
        object_material("wood")
        if static_friction is None
        else physx.PhysxMaterial(static_friction, dynamic_friction, 0.0)
    )
    face = make_render_material(
        base_color, roughness=0.72, textures=WOOD_TEXTURES if use_textures else None
    )
    cut = make_render_material([c * 0.76 for c in base_color[:3]] + [1.0], roughness=0.86)
    edge = make_render_material([c * 0.60 for c in base_color[:3]] + [1.0], roughness=0.62)
    routed = make_render_material([c * 0.88 for c in base_color[:3]] + [1.0], roughness=0.45)
    label = make_render_material([0.93, 0.92, 0.88, 1.0], roughness=0.65)
    ink = make_render_material([0.10, 0.10, 0.12, 1.0], roughness=0.8)

    for idx, (centre, half) in enumerate(panel_collision_boxes()):
        pose = sapien.Pose(p=centre)
        builder.add_box_collision(
            pose=pose, half_size=half, material=phys, density=PANEL_DENSITY
        )
        builder.add_box_visual(
            pose=pose, half_size=half, material=cut if idx == 3 else face   # the stop wall reads darker
        )
    for path in mouth_wedge_files():
        builder.add_convex_collision_from_file(filename=path, material=phys, density=PANEL_DENSITY)
        builder.add_visual_from_file(filename=path, material=face)

    hx, hy, hz = PANEL_HALF
    # Routed groove floor, slightly darker, so the channel and its closed end read from above.
    floor_z = hz - GROOVE_DEPTH
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.5 * (-hx + GROOVE_STOP_X), 0.0, floor_z + 0.0003]),
        half_size=[0.5 * (GROOVE_STOP_X + hx), GROOVE_HALF_W - 0.0005, 0.0003], material=cut,
    )
    # Edge banding on all four sides; the veneer edge is the most obvious real-wood cue.
    for sgn in (-1.0, 1.0):
        builder.add_box_visual(
            pose=sapien.Pose(p=[sgn * (hx + 0.0006), 0.0, 0.0]),
            half_size=[0.0006, hy, hz], material=edge,
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, sgn * (hy + 0.0006), 0.0]),
            half_size=[hx, 0.0006, hz], material=edge,
        )

    if not visual_detail:
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.5])
        return builder.build_kinematic(name=name) if kinematic else builder.build(name=name)

    # Polished strip along each groove lip; reads as the router's eased edge.
    # (user: the lip strips floated across the mouth, where the wall has
    # flared away, so hand and rail passed through a strip with no collision under
    # it). Straight strips now stop at the mouth, and angled strips follow the flare
    # edges on top of the (solid) wedge walls.
    x0 = -hx + GROOVE_MOUTH_LENGTH
    flare_len = float(np.hypot(GROOVE_MOUTH_LENGTH, GROOVE_MOUTH_FLARE))
    for sgn in (-1.0, 1.0):
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.5 * (x0 + hx), sgn * (GROOVE_HALF_W + 0.002), hz + 0.0003]),
            half_size=[0.5 * (hx - x0), 0.002, 0.0003], material=routed,
        )
        d = np.array([GROOVE_MOUTH_LENGTH, -sgn * GROOVE_MOUTH_FLARE]) / flare_len     # edge direction, mouth -> groove
        n = np.array([GROOVE_MOUTH_FLARE, sgn * GROOVE_MOUTH_LENGTH]) / flare_len       # into the wall
        mid = np.array([0.5 * (-hx + x0), sgn * (GROOVE_HALF_W + 0.5 * GROOVE_MOUTH_FLARE)]) + 0.002 * n
        yaw = float(np.arctan2(d[1], d[0]))
        builder.add_box_visual(
            pose=sapien.Pose(p=[mid[0], mid[1], hz + 0.0003], q=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
            half_size=[0.5 * flare_len, 0.002, 0.0003], material=routed,
        )

    # Printed part label with a stencilled barcode, sitting on the +y wall top face.
    label_x, label_y, label_z = -0.100, 0.045, hz + 0.0004
    builder.add_box_visual(
        pose=sapien.Pose(p=[label_x, label_y, label_z]),
        half_size=[0.021, 0.012, 0.0002], material=label,
    )
    for i in range(6):
        builder.add_box_visual(
            pose=sapien.Pose(p=[label_x - 0.014 + i * 0.0056, label_y, label_z + 0.0003]),
            half_size=[0.0011, 0.008, 0.0001], material=ink,
        )

    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.5])
    return builder.build_kinematic(name=name) if kinematic else builder.build(name=name)


def build_panel_rack(
    scene: ManiSkillScene, name: str = "panel_rack", visual_detail: bool = True
):
    """Static slotted rack: base plate plus two walls forming the slot.

    The slot lead-in chamfer is the one addition that is NOT visual-only. It is a real pair of
    angled collision boxes, because it stops the panel catching on the square wall lip during
    extraction and makes re-insertion possible for a future task. The rack is static, so the
    extra shapes cost nothing dynamically.
    """
    builder = scene.create_actor_builder()
    # Task layer: powder-coated steel rack -> "metal". It is a fixture, so it
    # is solved for the table pair rather than the pad pair.
    phys = object_material("metal", dominant="table")
    body = make_render_material([0.29, 0.31, 0.34, 1.0], roughness=0.55, metallic=0.35)
    plate = make_render_material([0.21, 0.23, 0.26, 1.0], roughness=0.70, metallic=0.30)
    pad = make_render_material([0.12, 0.14, 0.16, 1.0], roughness=0.9)
    bolt = make_render_material([0.62, 0.64, 0.67, 1.0], roughness=0.30, metallic=0.90)
    decal = make_render_material([0.85, 0.68, 0.15, 1.0], roughness=0.55)

    bx, by, bz = RACK_BASE_HALF
    base_pose = sapien.Pose(p=[0.0, 0.0, bz])
    builder.add_box_collision(pose=base_pose, half_size=[bx, by, bz], material=phys)
    builder.add_box_visual(pose=base_pose, half_size=[bx, by, bz], material=plate)

    wall_cz = RACK_SLOT_FLOOR_Z + RACK_WALL_H / 2.0
    wall_cy = RACK_SLOT_HALF_W + RACK_WALL_HALF_T
    wall_half = [bx, RACK_WALL_HALF_T, RACK_WALL_H / 2.0]
    chamfer_angle = np.deg2rad(22.0)
    chamfer_q = [float(np.cos(chamfer_angle / 2)), float(np.sin(chamfer_angle / 2)), 0.0, 0.0]

    for sgn in (-1.0, 1.0):
        wall_pose = sapien.Pose(p=[0.0, sgn * wall_cy, wall_cz])
        builder.add_box_collision(pose=wall_pose, half_size=wall_half, material=phys)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half, material=body)

        # Visual-only liner; no collision shape, so it cannot create false contacts.
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, sgn * (RACK_SLOT_HALF_W - 0.0004), wall_cz]),
            half_size=[bx * 0.98, 0.0004, RACK_WALL_H * 0.45],
            material=pad,
        )

        # Lead-in chamfer, flared away from the slot. Rotating about x by -angle for the +y
        # wall (and +angle for -y) tips each block's top outward.
        q = [chamfer_q[0], -sgn * chamfer_q[1], 0.0, 0.0]
        builder.add_box_collision(
            pose=sapien.Pose(p=[0.0, sgn * (wall_cy + 0.0015), RACK_TOP_Z + 0.004], q=q),
            half_size=[bx, RACK_WALL_HALF_T, 0.006],
            material=phys,
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, sgn * (wall_cy + 0.0015), RACK_TOP_Z + 0.004], q=q),
            half_size=[bx, RACK_WALL_HALF_T, 0.006],
            material=body,
        )

        if not visual_detail:
            continue

        # Bolt heads fixing each wall to the base plate.
        for fx in (-0.75, -0.25, 0.25, 0.75):
            builder.add_box_visual(
                pose=sapien.Pose(p=[fx * bx, sgn * wall_cy, RACK_SLOT_FLOOR_Z + 0.0016]),
                half_size=[0.0055, 0.0055, 0.0016],
                material=bolt,
            )
        # Outer stiffening ribs at both ends of each wall.
        for ex in (-1.0, 1.0):
            builder.add_box_visual(
                pose=sapien.Pose(
                    p=[ex * (bx - 0.012), sgn * (wall_cy + 0.009), wall_cz - 0.008]
                ),
                half_size=[0.010, 0.0025, RACK_WALL_H * 0.30],
                material=body,
            )

    if visual_detail:
        # Yellow hazard stripe along the front face of the base plate.
        for i in range(9):
            builder.add_box_visual(
                pose=sapien.Pose(p=[-bx + 0.02 + i * 0.0375, -by - 0.0006, bz]),
                half_size=[0.010, 0.0006, bz * 0.55],
                material=decal,
            )
        # Rubber feet, inset so they read against the plate edge.
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                builder.add_box_visual(
                    pose=sapien.Pose(p=[sx * (bx - 0.018), sy * (by - 0.014), 0.0018]),
                    half_size=[0.014, 0.010, 0.0020],
                    material=pad,
                )

    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
    # Kinematic, not static: the rack pose is re-randomized every episode, and on the GPU
    # backend a static actor's pose is uploaded once at build time and set_pose() is ignored.
    # Kinematic bodies are equally immovable but can be repositioned.
    rack = builder.build_kinematic(name=name)
    # Task layer: explicit mass (kinematic fixture, so dynamically inert).
    for raw in rack._objs:
        comp = raw.find_component_by_type(physx.PhysxRigidDynamicComponent)
        if comp is not None:
            comp.mass = 1.917
    return rack


PLACE_REGION_HALF_SIZE = (0.225, 0.135)   # assembly zone half extents (shared by the long task and the subtasks)

# ---------------------------------------------------------------------------
# "Panel placed" -- the ONE predicate of stage 1, evaluated by this subtask and
# by the long task. Before, the long task kept its own copy: centre
# within (35, 40) mm of the fixed PLACE_CENTER and 12 mm rest tolerance, while
# the subtask asked for the whole footprint inside the zone around a per-episode
# randomised centre with 10 mm. Same intent, two drifting texts.
# ---------------------------------------------------------------------------
PANEL_FLATNESS_COS = 0.95        # SIGNED panel z-axis, so groove-down can never pass
PANEL_YAW_TOL = np.deg2rad(20.0)
PANEL_REST_Z_TOL = 0.010         # |panel centre z - PANEL_HALF[2]|
PANEL_LIN_VEL_TOL = 0.030
PANEL_ANG_VEL_TOL = 0.400
PANEL_SETTLE_STEPS = 5


def panel_place_quantities(panel_pose, place_center_xy, contact_count, linear_velocity,
                           angular_velocity, region_half_size=PLACE_REGION_HALF_SIZE) -> dict:
    """Everything "the panel lies flat in the assembly zone, let go of and still" needs.

    `place_center_xy` is (N, 2); `contact_count` is the number of distinct fingers
    touching the panel (any hand). `placed_now` is the stage predicate itself.
    """
    rot = panel_pose.to_transformation_matrix()[..., :3, :3]
    normal = rot[..., :3, 2]
    x_axis = rot[..., :3, 0]
    y_axis = rot[..., :3, 1]
    to_place = place_center_xy - panel_pose.p[:, :2]
    # Axis-aligned world XY bounds of the rotated panel. Requiring the full footprint to
    # fit prevents centre-only success while an edge hangs outside the assembly zone.
    footprint_half_x = torch.abs(x_axis[:, 0]) * PANEL_HALF[0] + torch.abs(y_axis[:, 0]) * PANEL_HALF[1]
    footprint_half_y = torch.abs(x_axis[:, 1]) * PANEL_HALF[0] + torch.abs(y_axis[:, 1]) * PANEL_HALF[1]
    footprint_inside = (
        (torch.abs(to_place[:, 0]) + footprint_half_x <= region_half_size[0])
        & (torch.abs(to_place[:, 1]) + footprint_half_y <= region_half_size[1])
    )
    flatness = torch.clamp(normal[:, 2], -1.0, 1.0)          # SIGNED
    yaw_alignment = torch.abs(x_axis[:, 0])
    rest_z_error = torch.abs(panel_pose.p[:, 2] - PANEL_HALF[2])
    linear_speed = torch.linalg.norm(linear_velocity, dim=1)
    angular_speed = torch.linalg.norm(angular_velocity, dim=1)
    is_flat_on_table = (flatness >= PANEL_FLATNESS_COS) & (rest_z_error <= PANEL_REST_Z_TOL)
    in_place_region = footprint_inside & (yaw_alignment >= float(np.cos(PANEL_YAW_TOL)))
    is_released = contact_count <= 0.0
    is_static = (linear_speed <= PANEL_LIN_VEL_TOL) & (angular_speed <= PANEL_ANG_VEL_TOL)
    return {
        "flatness": flatness,
        "yaw_alignment": yaw_alignment,
        "footprint_inside": footprint_inside,
        "rest_z_error": rest_z_error,
        "linear_speed": linear_speed,
        "angular_speed": angular_speed,
        "contact_count": contact_count,
        "is_flat_on_table": is_flat_on_table,
        "in_place_region": in_place_region,
        "is_released": is_released,
        "is_static": is_static,
        "placed_now": is_flat_on_table & in_place_region & is_released & is_static,
    }


def build_place_marker(scene: ManiSkillScene, half_size, name: str = "place_zone"):
    """Visual-only rectangular assembly zone; all panel corners must end inside it."""
    builder = scene.create_actor_builder()
    paint = make_render_material([0.20, 0.62, 0.38, 1.0], roughness=0.95)
    fill = make_render_material([0.20, 0.55, 0.35, 0.18], roughness=0.95)

    hx, hy = float(half_size[0]), float(half_size[1])
    for x, y, sx, sy in (
        (0.0, -hy, hx, 0.003), (0.0, hy, hx, 0.003),
        (-hx, 0.0, 0.003, hy), (hx, 0.0, 0.003, hy),
    ):
        builder.add_box_visual(
            pose=sapien.Pose(p=[x, y, 0.0006]), half_size=[sx, sy, 0.0006],
            material=paint,
        )
    # Faint fill so the zone reads as a region rather than just an outline.
    builder.add_box_visual(
        pose=sapien.Pose(p=[0.0, 0.0, 0.0003]),
        half_size=[hx, hy, 0.0003],
        material=fill,
    )
    # Centre cross-hair.
    for q in ([1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.0, 0.7071]):
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, 0.0, 0.0006], q=q),
            half_size=[0.020, 0.0025, 0.0006],
            material=paint,
        )

    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -1.0])
    return builder.build_kinematic(name=name)


# ===========================================================================
# Environment
# ===========================================================================
@register_task(
    "FurnitureAssemblyLayFlatPanel",
    side="left",
    max_episode_steps=600,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    render_mode="rgb_array",
)
class FurnitureAssemblyLayFlatPanelEnv(DeclarativeTaskStateMixin, BaseEnv):
    # Task bookkeeping that physics cannot re-derive. Restoring a
    # mid-episode frame without these silently rewinds every milestone,
    # which breaks offline re-render, demo-init and end-frame QC.
    # APPEND ONLY -- reordering breaks older recordings.
    TASK_STATE_KEY = "furniture_panel_task_state"
    TASK_STATE_FIELDS = (
        TaskStateField("grasp_achieved", 1, torch.bool),
        TaskStateField("extract_achieved", 1, torch.bool),
        TaskStateField("settle_counter", 1, torch.int32),
        TaskStateField("place_center", 3, torch.float32),
    )
    """Extract a vertical drawer panel from a slotted rack and lay it flat, groove up."""

    SUPPORTED_ROBOTS = sorted(MOBILE_ROBOT_UIDS)
    agent: Union[
        PandaGripperLeftMobileBase,
        PandaSharpaHandRightAngledMobileBase,
        PandaWujiHandRightMobileBase,
    ]

    # Task layer: the family's world layout comes from ONE table (see
    # FURNITURE_LAYOUT above), so a subtask cannot sit at a different working
    # distance from the stage it is supposed to reproduce.
    RACK_SPAWN_CENTER = RACK_CENTER
    PLACE_SPAWN_CENTER = PLACE_CENTER

    # Success thresholds.
    PLACE_REGION_HALF_SIZE = (0.225, 0.135)
    # Success thresholds live in panel_place_quantities (shared with the long task);
    # these aliases only feed the dense reward and the docstring.
    YAW_ALIGNMENT_THRESHOLD = float(np.cos(PANEL_YAW_TOL))
    FLATNESS_THRESHOLD = PANEL_FLATNESS_COS
    REST_Z_TOLERANCE = PANEL_REST_Z_TOL
    LIN_VEL_THRESHOLD = PANEL_LIN_VEL_TOL
    ANG_VEL_THRESHOLD = PANEL_ANG_VEL_TOL
    SETTLE_STEPS = PANEL_SETTLE_STEPS
    TIP_CONTACT_FORCE = 0.6          # N, per fingertip link, to count as "touching"

    # Reward weights; every r_i is in [0, 1] and terms compose multiplicatively.
    W_REACH, W_GRASP, W_EXTRACT, W_FLAT, W_SETTLE = 1.0, 1.0, 2.0, 2.0, 4.0
    SUCCESS_REWARD = 10.0
    LIVING_PENALTY = 0.01
    ARM_QVEL_PENALTY = 5e-4
    DROP_PENALTY = 2.0

    external_camera_eye = [0.58, -1.41, 0.85]
    external_camera_target = [-0.42, -0.04, 0.34]

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.01,
        rack_random_x: float = 0.030,
        rack_random_y: float = 0.025,
        rack_random_yaw: float = np.deg2rad(9.0),
        # default 0 -- the long task places the panel on the FIXED
        # PLACE_CENTER, so a subtask that randomises the target is not its subset.
        place_random_x: float = 0.0,
        place_random_y: float = 0.0,
        panel_color_jitter: bool = True,
        visual_detail: bool = True,
        show_place_marker: bool = True,
        terminate_on_success: bool = False,
        render_width: int = 720,
        render_height: int = 720,
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if render_width <= 0 or render_height <= 0:
            raise ValueError("Render width and height must be positive integers.")
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.rack_random_x = float(rack_random_x)
        self.rack_random_y = float(rack_random_y)
        self.rack_random_yaw = float(rack_random_yaw)
        self.place_random_x = float(place_random_x)
        self.place_random_y = float(place_random_y)
        self.panel_color_jitter = bool(panel_color_jitter)
        self.visual_detail = bool(visual_detail)
        # The family-wide switch can only take cues away, never add them back.
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        self.show_place_marker = bool(show_place_marker) and self.show_guidance_markers
        self.terminate_on_success = bool(terminate_on_success)
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @property
    def _default_sim_config(self) -> SimConfig:
        # Environment layer: the locked fleet profile (400 Hz, solver 24/8, 1 mm
        # contact offset -- the slot clearance here is 1.8 mm per side, so the
        # locked offset already resolves it).
        return make_sim_config(
            max_rigid_contact_count=2**21,
            max_rigid_patch_count=2**19,
        )

    @property
    def _default_sensor_configs(self):
        external_pose = sapien_utils.look_at(
            eye=self.external_camera_eye, target=self.external_camera_target
        )
        return [
            build_dexarena_base_camera(name="base_camera"),
            CameraConfig("external_camera", external_pose, 320, 240, np.deg2rad(58.0), 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=sapien_utils.look_at(
                eye=[0.68, -1.71, 0.92], target=[-0.45, -0.04, 0.36]
            ),
            width=self.render_width,
            height=self.render_height,
            fov=1.0,
            near=0.01,
            far=100.0,
        )

    def _load_lighting(self, options: dict):
        # The groove and the blind holes are only readable through shading, so the default
        # flat lighting hides exactly the features this task is about.
        self.scene.set_ambient_light([0.28, 0.28, 0.30])
        self.scene.add_directional_light(
            [0.4, 0.5, -1.0], [2.0, 1.95, 1.85],
            shadow=True, shadow_scale=2.5, shadow_map_size=2048,
        )
        self.scene.add_directional_light([-0.6, -0.3, -0.8], [0.55, 0.58, 0.65])
        self.scene.add_point_light([0.3, -0.5, 0.7], [1.2, 1.1, 1.0], shadow=False)

    def _load_agent(self, options: dict):
        super()._load_agent(options, ROBOT_ROOT_POSE)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # Environment layer: the work table belongs to the shared rig.
        set_work_table_pose(self)

        base_color = (0.725, 0.541, 0.369, 1.0)
        if self.panel_color_jitter:
            import colorsys

            rng = getattr(self, "_episode_rng", None) or np.random.RandomState(0)
            base_color = colorsys.hsv_to_rgb(
                float(rng.uniform(0.055, 0.095)),
                float(rng.uniform(0.35, 0.55)),
                float(rng.uniform(0.55, 0.80)),
            ) + (1.0,)

        self.panel = build_drawer_panel(
            self.scene, base_color=base_color, visual_detail=self.visual_detail
        )
        set_task_object_energy_fence(self.panel)
        self.rack = build_panel_rack(self.scene, visual_detail=self.visual_detail)
        self.place_marker = (
            build_place_marker(self.scene, self.PLACE_REGION_HALF_SIZE)
            if self.show_place_marker
            else None
        )

        self.grasp_local_pos = common.to_tensor(GRASP_LOCAL_POS, device=self.device).float()
        self.groove_landmarks_local = common.to_tensor(
            GROOVE_LANDMARKS_LOCAL, device=self.device
        ).float()

        self.place_center = torch.zeros((self.num_envs, 3), device=self.device)
        # Reward-side milestone latches only. `success` never reads these.
        self.grasp_achieved = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.extract_achieved = torch.zeros_like(self.grasp_achieved)
        self.settle_counter = torch.zeros(
            (self.num_envs,), dtype=torch.int32, device=self.device
        )

    def _reset_agent(self, b: int):
        if self.robot_uids in MOBILE_ROBOT_UIDS:
            qpos = torch.tensor(
                self.agent.keyframes["rest"].qpos, device=self.device, dtype=torch.float32
            ).expand(b, -1).clone()
            active = [j.name for j in self.agent.robot.get_active_joints()]
            arm_idx = [
                active.index(n)
                for n in getattr(self.agent, "arm_joint_names", [])
                if n in active
            ]
            if arm_idx:
                arm_idx = torch.tensor(arm_idx, device=self.device, dtype=torch.long)
                # Arm layer: the shared start pose. The stock "rest" keyframe
                # sits 0.77 rad away at joint 6, which the operator feels from
                # the first frame as a different wrist attitude.
                qpos[:, arm_idx] = torch.tensor(
                    ARM_INIT_QPOS, device=self.device, dtype=torch.float32
                ).expand(b, -1).clone()
                qpos[:, arm_idx] += common.to_tensor(
                    self._episode_rng.normal(0.0, self.robot_init_qpos_noise, (b, len(arm_idx))),
                    device=self.device,
                )
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(ROBOT_ROOT_POSE)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            set_work_table_pose(self)
            self._reset_agent(b)

            def rand(scale: float) -> torch.Tensor:
                return (torch.rand((b,), device=self.device) * 2.0 - 1.0) * float(scale)

            # Rack. yaw 0 puts the slot along world x, so the panel long axis is along x.
            rack_pos = torch.zeros((b, 3), device=self.device)
            rack_pos[:, 0] = self.RACK_SPAWN_CENTER[0] + rand(self.rack_random_x)
            rack_pos[:, 1] = self.RACK_SPAWN_CENTER[1] + rand(self.rack_random_y)
            rack_yaw = rand(self.rack_random_yaw)
            self.rack.set_pose(Pose.create_from_pq(p=rack_pos, q=yaw_to_quat_wxyz(rack_yaw)))

            # Panel standing in the slot: q = Rz(rack_yaw) * Rx(90deg + lean).
            # The small lean makes it settle against one wall instead of balancing ambiguously.
            lean = rand(np.deg2rad(1.5))
            q_panel = quat_mul(
                yaw_to_quat_wxyz(rack_yaw),
                axis_to_quat_wxyz(torch.full_like(lean, np.pi / 2.0) + lean, axis=0),
            )
            panel_pos = rack_pos.clone()
            panel_pos[:, 2] = PANEL_IN_SLOT_Z
            # In-slot jitter in rack-local coordinates, rotated into world.
            dx, dy = rand(0.012), rand(0.0010)
            cy, sy = torch.cos(rack_yaw), torch.sin(rack_yaw)
            panel_pos[:, 0] += cy * dx - sy * dy
            panel_pos[:, 1] += sy * dx + cy * dy

            self.panel.set_pose(Pose.create_from_pq(p=panel_pos, q=q_panel))
            self.panel.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.panel.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            # Drop zone.
            place = torch.zeros((b, 3), device=self.device)
            place[:, 0] = self.PLACE_SPAWN_CENTER[0] + rand(self.place_random_x)
            place[:, 1] = self.PLACE_SPAWN_CENTER[1] + rand(self.place_random_y)
            self.place_center[env_idx] = place
            if self.place_marker is not None:
                self.place_marker.set_pose(Pose.create_from_pq(p=place))

            self.grasp_achieved[env_idx] = False
            self.extract_achieved[env_idx] = False
            self.settle_counter[env_idx] = 0

    # ------------------------------------------------------------------
    # Geometry accessors, exported for subtasks 2 and 3
    # ------------------------------------------------------------------
    @property
    def grasp_position(self) -> torch.Tensor:
        """(N, 3) world position of the nominal grasp point on the panel."""
        offset = Pose.create_from_pq(p=self.grasp_local_pos.expand(self.num_envs, -1))
        return (self.panel.pose * offset).p

    @property
    def groove_landmarks(self) -> torch.Tensor:
        """(N, 2, 3) world positions of the groove mouth centre and the stop face."""
        return torch.stack(
            [
                (
                    self.panel.pose
                    * Pose.create_from_pq(p=self.groove_landmarks_local[i].expand(self.num_envs, -1))
                ).p
                for i in range(2)
            ],
            dim=1,
        )

    # ------------------------------------------------------------------
    # End-effector abstraction (same proxy pattern as ButtonPress)
    # ------------------------------------------------------------------
    def _tip_links(self) -> list:
        if self.robot_uids in HAND_ROBOT_UIDS and hasattr(self.agent, "tip_links"):
            return list(self.agent.tip_links)
        links = [
            getattr(self.agent, a, None)
            for a in ("finger1pad_link", "finger2pad_link", "finger1_link", "finger2_link")
        ]
        links = [l for l in links if l is not None]
        return links[:2]

    def _tip_positions(self) -> torch.Tensor:
        links = self._tip_links()
        if links:
            return torch.stack([l.pose.p for l in links], dim=1)
        return self.agent.tcp.pose.p[:, None, :]

    def _contact_count(self) -> torch.Tensor:
        """How many DISTINCT fingers carry contact force against the panel (collision-bearing links)."""
        return finger_contact_count(self.scene, self.agent, self.panel, self.TIP_CONTACT_FORCE)

    def _gripper_opening(self) -> torch.Tensor:
        if self.robot_uids == GRIPPER_ROBOT_UID:
            return torch.clamp(self.agent.robot.get_qpos()[..., -2:], min=0.0).sum(dim=1)
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Single source of truth for evaluate() and the reward
    # ------------------------------------------------------------------
    def _task_quantities(self) -> dict:
        pose = self.panel.pose
        rot = pose.to_transformation_matrix()[..., :3, :3]
        panel_normal = rot[..., :3, 2]     # panel local +z, i.e. the groove side
        groove_axis = rot[..., :3, 0]      # panel local +x, i.e. along the groove

        grasp_point = self.grasp_position
        tips = self._tip_positions()
        reach_dist = torch.minimum(
            torch.linalg.norm(self.agent.tcp.pose.p - grasp_point, dim=1),
            torch.linalg.norm(tips - grasp_point[:, None, :], dim=2).min(dim=1).values,
        )

        # Lowest point of the panel slab. More reliable than centre height for "clear of the
        # slot", which stays high while a corner is still inside.
        sin_tilt = torch.sqrt(torch.clamp(1.0 - panel_normal[:, 2] ** 2, min=0.0))
        panel_lowest = pose.p[:, 2] - PANEL_HALF[1] * sin_tilt - PANEL_HALF[2]

        to_place = self.place_center - pose.p
        placed = panel_place_quantities(
            pose, self.place_center[:, :2], self._contact_count(),
            self.panel.linear_velocity, self.panel.angular_velocity,
            region_half_size=self.PLACE_REGION_HALF_SIZE)
        return {
            **placed,
            "panel_pos": pose.p,
            "panel_normal": panel_normal,
            "groove_axis": groove_axis,
            "grasp_point": grasp_point,
            "tip_positions": tips,
            "reach_dist": reach_dist,
            "gripper_opening": self._gripper_opening(),
            "panel_lowest": panel_lowest,
            "height_above_rack": panel_lowest - RACK_TOP_Z,
            # Scaled so it is exactly 1.0 when panel_lowest reaches RACK_TOP_Z, i.e. at the
            # same instant the extract milestone latches. Otherwise the latch would step the
            # term discontinuously from 0.69 to 1.0.
            "extract_progress": torch.clamp(
                (panel_lowest - RACK_SLOT_FLOOR_Z - 0.005) / (RACK_WALL_H - 0.005), 0.0, 1.0
            ),
            "place_dist": torch.linalg.norm(to_place[:, :2], dim=1),
            "arm_qvel": self.agent.robot.get_qvel()[
                ..., : len(getattr(self.agent, "arm_joint_names", [])) or 7
            ],
        }

    def evaluate(self):
        q = self._task_quantities()

        is_grasped = q["contact_count"] >= 2.0
        if self.robot_uids == GRIPPER_ROBOT_UID:
            native = getattr(self.agent, "is_grasping", None)
            if callable(native):
                try:
                    is_grasped = is_grasped | native(self.panel).to(torch.bool)
                except Exception:
                    pass

        is_released = q["is_released"]
        is_out_of_rack = q["height_above_rack"] > -0.002
        is_static = q["is_static"]
        is_flat_on_table = q["is_flat_on_table"]
        in_place_region = q["in_place_region"]

        # Success is a pure function of the current state, so replayed trajectories reproduce
        # it exactly. A panel lying flat on the table cannot still be in the slot, so there is
        # no need to also check that it was extracted.
        settled_now = q["placed_now"]          # == is_flat_on_table & in_place_region & is_released & is_static
        self.settle_counter = torch.where(
            settled_now, self.settle_counter + 1, torch.zeros_like(self.settle_counter)
        )
        success = self.settle_counter >= self.SETTLE_STEPS

        # Reward-only latches.
        self.grasp_achieved |= is_grasped
        self.extract_achieved |= is_grasped & is_out_of_rack

        return {
            "success": success,
            "termination_success": success if self.terminate_on_success
            else torch.zeros_like(success),
            "settle_progress": torch.clamp(
                self.settle_counter.float() / float(self.SETTLE_STEPS), 0.0, 1.0
            ),
            "is_grasped": is_grasped,
            "is_released": is_released,
            "is_out_of_rack": is_out_of_rack,
            "is_static": is_static,
            "is_flat_on_table": is_flat_on_table,
            "in_place_region": in_place_region,
            "grasp_achieved": self.grasp_achieved,
            "extract_achieved": self.extract_achieved,
            "dropped": q["panel_pos"][:, 2] < 0.002,
            "reach_dist": q["reach_dist"],
            "contact_count": q["contact_count"],
            "gripper_opening": q["gripper_opening"],
            "extract_progress": q["extract_progress"],
            "height_above_rack": q["height_above_rack"],
            "flatness": q["flatness"],
            "place_dist": q["place_dist"],
            "yaw_alignment": q["yaw_alignment"],
            "footprint_inside": q["footprint_inside"],
            "rest_z_error": q["rest_z_error"],
            "panel_normal": q["panel_normal"],
            "groove_axis": q["groove_axis"],
            "grasp_point": q["grasp_point"],
            "tcp_to_grasp": q["grasp_point"] - self.agent.tcp.pose.p,
            "panel_pose": self.panel.pose.raw_pose,
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "panel_pose": info["panel_pose"],
            "tcp_to_grasp": info["tcp_to_grasp"],
            "place_center": self.place_center,
        }
        if self.obs_mode_struct.use_state:
            obs.update(
                all_tip_positions=self._tip_positions().reshape(self.num_envs, -1),
                groove_landmarks=self.groove_landmarks.reshape(self.num_envs, -1),
                groove_axis=info["groove_axis"],
                panel_normal=info["panel_normal"],
                panel_linear_velocity=self.panel.linear_velocity,
                panel_angular_velocity=self.panel.angular_velocity,
                rack_pose=self.rack.pose.raw_pose,
                panel_to_place=self.place_center - self.panel.pose.p,
                contact_count=info["contact_count"].unsqueeze(1),
                gripper_opening=info["gripper_opening"].unsqueeze(1),
                extract_progress=info["extract_progress"].unsqueeze(1),
                flatness=info["flatness"].unsqueeze(1),
                height_above_rack=info["height_above_rack"].unsqueeze(1),
                is_grasped=info["is_grasped"].float().unsqueeze(1),
                is_out_of_rack=info["is_out_of_rack"].float().unsqueeze(1),
                is_released=info["is_released"].float().unsqueeze(1),
                is_static=info["is_static"].float().unsqueeze(1),
                is_flat_on_table=info["is_flat_on_table"].float().unsqueeze(1),
                in_place_region=info["in_place_region"].float().unsqueeze(1),
                yaw_alignment=info["yaw_alignment"].unsqueeze(1),
                footprint_inside=info["footprint_inside"].float().unsqueeze(1),
            )
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: Array, info: dict) -> torch.Tensor:
        q = self._task_quantities()
        ones = torch.ones((self.num_envs,), device=self.device)

        r_reach = 1.0 - torch.tanh(6.0 * q["reach_dist"])
        r_grasp = torch.clamp(q["contact_count"] / 2.0, 0.0, 1.0)
        r_extract = q["extract_progress"]
        r_flat = torch.clamp((q["flatness"] + 1.0) * 0.5, 0.0, 1.0) ** 2
        r_settle = (
            0.4 * (1.0 - torch.tanh(4.0 * q["place_dist"]))
            + 0.3 * (1.0 - torch.tanh(12.0 * q["rest_z_error"]))
            + 0.3 * info["settle_progress"]
        )

        # Latch passed milestones to 1 so a later stage cannot collapse the prefix product.
        # Without this r_grasp drops to zero the instant the robot correctly lets go, which is
        # exactly the moment it should be rewarded most.
        r_reach = torch.where(info["grasp_achieved"], ones, r_reach)
        r_grasp = torch.where(info["extract_achieved"], ones, r_grasp)
        r_extract = torch.where(info["extract_achieved"], ones, r_extract)
        r_flat = torch.where(info["is_flat_on_table"], ones, r_flat)

        prefix = r_reach
        term_reach = self.W_REACH * prefix
        prefix = prefix * r_grasp
        term_grasp = self.W_GRASP * prefix
        prefix = prefix * r_extract
        term_extract = self.W_EXTRACT * prefix
        prefix = prefix * r_flat
        term_flat = self.W_FLAT * prefix
        prefix = prefix * r_settle
        term_settle = self.W_SETTLE * prefix

        reward = term_reach + term_grasp + term_extract + term_flat + term_settle
        contact_weight = 1.0 + 4.0 * torch.clamp(q["contact_count"] / 2.0, 0.0, 1.0)
        reward = reward - self.ARM_QVEL_PENALTY * contact_weight * torch.mean(
            q["arm_qvel"] ** 2, dim=1
        )
        reward = reward - self.LIVING_PENALTY
        reward = reward - self.DROP_PENALTY * info["dropped"].float()
        reward = torch.where(
            info["success"], torch.full_like(reward, self.SUCCESS_REWARD), reward
        )

        info["reward_terms"] = {
            "r_reach": r_reach, "r_grasp": r_grasp, "r_extract": r_extract,
            "r_flat": r_flat, "r_settle": r_settle,
            "term_reach": term_reach, "term_grasp": term_grasp,
            "term_extract": term_extract, "term_flat": term_flat, "term_settle": term_settle,
        }
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs, action, info) / self.SUCCESS_REWARD


# ===========================================================================
# CLI: smoke test and demo/video recording
# ===========================================================================
def tensor_mean(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(np.asarray(value).mean())


def hold_action(env, control_mode: str) -> torch.Tensor:
    """Action that keeps the robot where it is, for the zero-action smoke test.

    Only delta modes can express "hold" as zeros. `pd_joint_pos` on Panda is registered with
    normalize_action=False, so its action IS the raw target joint angle vector; sending zeros
    commands every joint to 0 rad and the arm snaps across the workspace. Hold = current qpos.
    """
    action = torch.zeros(
        env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32
    )
    # Substring test rather than a fixed list, so pd_joint_target_delta_pos and friends are
    # covered too.
    if "delta" in control_mode:
        return action
    qpos = env.unwrapped.agent.robot.get_qpos()
    dof = min(action.shape[-1], qpos.shape[-1])
    action[..., :dof] = qpos[..., :dof].reshape(action[..., :dof].shape)
    if env.unwrapped.robot_uids == GRIPPER_ROBOT_UID:
        # The gripper mimic controller normalizes to [-1, 1]; +1 keeps it open.
        action[..., -1] = 1.0
    return action


def run(args: argparse.Namespace) -> None:
    render_mode = "rgb_array" if args.record_video else args.render_mode
    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=render_mode,
        sim_backend=args.sim_backend,
        terminate_on_success=args.terminate_on_success,
        panel_color_jitter=not args.no_color_jitter,
        visual_detail=not args.no_visual_detail,
        show_place_marker=not args.no_place_marker,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    if args.record_video or args.save_trajectory:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=args.video_dir,
            save_video=args.record_video,
            save_trajectory=args.save_trajectory,
            trajectory_name="panel_lay_flat_v1_trajectory",
            video_fps=args.video_fps,
            max_steps_per_video=args.max_steps,
        )

    print(f"env_id={args.env_id}  dof={env.unwrapped.agent.robot.dof}")
    print("obs space", env.observation_space)
    print("act space", env.action_space)

    obs, info = env.reset(seed=args.seed)
    for episode in range(args.episodes):
        # Recomputed per episode: for absolute joint modes the hold target is the reset qpos.
        keep_still = hold_action(env, args.control_mode)
        step = 0
        while step < args.max_steps:
            action = (
                torch.as_tensor(env.action_space.sample(), device=env.unwrapped.device)
                if args.random_actions
                else keep_still
            )
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if args.render_mode == "human" and not args.record_video:
                env.render()
            if args.debug_print and step % args.debug_interval == 0:
                print(
                    "step={:04d} rew={:+.3f} succ={:.2f} reach={:.4f} contacts={:.1f} "
                    "above_rack={:+.4f} flat={:+.3f} place_d={:.4f} rest_z={:.4f}".format(
                        step, tensor_mean(reward), tensor_mean(info["success"]),
                        tensor_mean(info["reach_dist"]), tensor_mean(info["contact_count"]),
                        tensor_mean(info["height_above_rack"]), tensor_mean(info["flatness"]),
                        tensor_mean(info["place_dist"]), tensor_mean(info["rest_z_error"]),
                    )
                )
            if bool((torch.as_tensor(terminated) | torch.as_tensor(truncated)).all().item()):
                break
        print(
            "episode={} steps={} success={:.2f} grasp={:.2f} extract={:.2f} flat={:+.3f}".format(
                episode + 1, step, tensor_mean(info["success"]),
                tensor_mean(info["grasp_achieved"]), tensor_mean(info["extract_achieved"]),
                tensor_mean(info["flatness"]),
            )
        )
        if episode + 1 < args.episodes:
            obs, info = env.reset()
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PanelLayFlat (DualFurnitureAssembly subtask 1).")
    p.add_argument("--env-id", "--env_id", default="FurnitureAssemblyLayFlatPanel-Gripper-v1", choices=ENV_IDS)
    p.add_argument(
        "--control-mode", "--control_mode", default=DEFAULT_CONTROL_MODE,
        choices=["pd_joint_delta_pos", "pd_joint_pos", "pd_joint_target_delta_pos",
                 "pd_ee_delta_pose", "pd_ee_delta_pos"],
    )
    p.add_argument("--obs-mode", "--obs_mode", default="state")
    p.add_argument("--reward-mode", "--reward_mode", default="normalized_dense",
                   choices=["dense", "normalized_dense", "sparse"])
    p.add_argument("--sim-backend", "--sim_backend", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--render-mode", "--render_mode", default="rgb_array",
                   choices=["rgb_array", "human"])
    p.add_argument("--num-envs", "--num_envs", type=int, default=1)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", "--max_steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--terminate-on-success", "--terminate_on_success", action="store_true")
    p.add_argument("--random-actions", "--random_actions", action="store_true")
    p.add_argument("--record-video", "--record_video", action="store_true")
    p.add_argument("--save-trajectory", "--save_trajectory", action="store_true")
    p.add_argument("--video-dir", "--video_dir", default=DEFAULT_VIDEO_DIR)
    p.add_argument("--render-width", "--render_width", type=int, default=720)
    p.add_argument("--render-height", "--render_height", type=int, default=720)
    p.add_argument("--video-fps", "--video_fps", type=int, default=20)
    p.add_argument("--no-color-jitter", "--no_color_jitter", action="store_true")
    p.add_argument("--no-visual-detail", "--no_visual_detail", action="store_true",
                   help="Strip the decorative visual shapes; keeps physics identical.")
    p.add_argument("--no-place-marker", "--no_place_marker", action="store_true")
    p.add_argument("--debug-print", "--debug_print", action="store_true")
    p.add_argument("--debug-interval", "--debug_interval", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

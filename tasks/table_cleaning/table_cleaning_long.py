"""``TableCleaningLong``: single right-arm table clean.

This module owns the long-horizon task and the scene, sensors and predicates its
subtasks share. Each subtask lives in its own module and subclasses
``TableCleanBase`` from here, following the repo layout used by ``cabinet_pour``:

    tasks/table_cleaning/table_cleaning_long.py                 TableCleaningLong-Sharpa-v1              (this file)
    tasks/table_cleaning/table_cleaning_pick_debris.py     TableCleaningPickDebris-Sharpa-v1    subtask 1
    tasks/table_cleaning/table_cleaning_dump_dustpan.py    TableCleaningDumpDustpan-Sharpa-v1   subtask 2

The robot is the right arm of the shared rig with any of the four end effectors
(tasks/config/hands.py). Grasping is a per-finger contact-count test, not
``is_grasping``: on a dexterous hand it counts distinct fingers in force contact,
on the parallel gripper each finger is one group.

    Subtask 1  TableCleaningPickDebris-Sharpa-v1   pick each piece of debris BY HAND and
                                         drop it into the dustpan
    Subtask 2  TableCleaningDumpDustpan-Sharpa-v1  pick up the dustpan by its handle,
                                         pour into the bin, stow the dustpan
    Long task  TableCleaningLong-Sharpa-v1             both, back to back

Why the redesign (was: brush-sweep the debris)
-----------------------------------------------------------
The brush was the same failure class as the spoon and the scissors: holding a
455 mm cantilever by a handle demands a torque about the grasp axis that a
rigid fingertip pad can only carry through contact SEPARATION, never through a
contact patch, and the measured demand (549 mN*m) was 2.5x the hand's capacity
(221 mN*m). The operator could not reliably pick it up. The user's
decision: pick the pieces by hand, and keep the dustpan stage -- with the
dustpan's thin handle thickened by a grip sleeve so it can be held (see
sweep_assets.build_dustpan). Debris are 20 mm cubes (11 g), three of them: a
10 mm cube is below what a teleoperated dexterous hand can pinch reliably.

Stages are latched and monotonic: a stage flag, once set, stays set for the
episode. This keeps the dense reward non-decreasing across a stage boundary.

Every geometric constant comes from sweep_assets.py, where it is recorded with
the script that measured it. Nothing here re-derives geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import sapien
import sapien.physx as physx
from tasks.config.layout_randomization import (
    LAYOUT_STREAM_SALT,
    layout_rng,
    ARM_INIT_QPOS_NOISE_STD,
    arm_qpos_noise,
    episode_rng,
    quat_mul_wxyz,
    scatter_xy,
    uniform_offset,
    yaw_quat_wxyz,
)
from tasks.config.mass_properties import set_mass as set_realizable_mass
import torch
import base64
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from tasks.config.hands import register_task
from mani_skill.utils.structs.pose import Pose

from tasks.config.sim_profiles import make_sim_config
from tasks.config.workspace_rig import (
    ARM_INIT_QPOS as RIG_ARM_INIT_QPOS,
    TABLE_CENTER_X,
    platform_root_pose,
    set_work_table_pose,
)
from tasks.scene_builders import TableSceneBuilder
from tasks.robots.panda_sharpa_hand_right import (  # noqa: F401
    PandaSharpaHandRight,
    PandaSharpaHandRightAngled,
)
from tasks.robots.panda_sharpa_hand_right_mobile_base import (  # noqa: F401
    PandaSharpaHandRightAngledMobileBase,
)
from tasks.robots.panda_wuji_hand_right_mobile_base import PandaWujiHandRightMobileBase  # noqa: F401
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.config.energy_fence import set_task_object_energy_fence
from tasks.config.object_materials import object_material

# ==========================================================================
# RoboCasa MJCF absolute-path patcher
# --------------------------------------------------------------------------
# Absolute-path patcher for RoboCasa MJCF assets.
#
# Kept here so the table-clean task modules are self-contained.
# ==========================================================================
_FALLBACK_TEXTURE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF"
    "gAI/0+3HAAAAAElFTkSuQmCC"
)


def _resolve_reference(model_dir: Path, compiler_dir: str | None, filename: str) -> Path:
    path = Path(filename)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    if compiler_dir:
        base = Path(compiler_dir)
        candidates.append((base / path) if base.is_absolute() else (model_dir / base / path))
    candidates.append(model_dir / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(model_dir.rglob(path.name))
    return matches[0].resolve() if matches else candidates[-1].resolve()


def prepare_robocasa_mjcf(model_dir: Path, cache_tag: str) -> Path:
    """Create an MJCF copy whose mesh/texture references are absolute."""
    model_dir = model_dir.resolve()
    source_xml = model_dir / "model.xml"
    if not source_xml.exists():
        raise FileNotFoundError(source_xml)

    # The cache directory is keyed by PROCESS as well as by tag. It used to be a
    # fixed shared path that this function rmtree'd and rebuilt, so two processes
    # preparing the same asset raced: one deleted the tree the other was writing
    # into, and the loser died with `IsADirectoryError` or a missing model.xml.
    # Measured verifying environments three at a time -- three of
    # fifty-six failed, and only ever the TableClean ones. Rebuilding per process
    # costs an XML rewrite and one small PNG, and cannot race.
    cache_dir = (
        Path(tempfile.gettempdir())
        / "robodex_robocasa_assets"
        / f"{cache_tag}.{os.getpid()}"
    )
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    fallback_texture = cache_dir / "fallback_texture.png"
    fallback_texture.write_bytes(_FALLBACK_TEXTURE)

    tree = ET.parse(source_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    mesh_dir = compiler.get("meshdir") if compiler is not None else None
    texture_dir = compiler.get("texturedir") if compiler is not None else None

    missing_meshes: list[str] = []
    fallback_textures: list[str] = []

    for mesh in root.findall(".//asset/mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        resolved = _resolve_reference(model_dir, mesh_dir, filename)
        if not resolved.exists():
            missing_meshes.append(filename)
        else:
            mesh.set("file", str(resolved))

    for texture in root.findall(".//asset/texture"):
        filename = texture.get("file")
        if not filename:
            continue
        resolved = _resolve_reference(model_dir, texture_dir, filename)
        if not resolved.exists():
            resolved = fallback_texture
            fallback_textures.append(filename)
        texture.set("file", str(resolved))

    if missing_meshes:
        raise FileNotFoundError(
            "Missing MJCF meshes:\n" + "\n".join(f"  - {x}" for x in missing_meshes)
        )
    if fallback_textures:
        print(f"[texture fallback] {model_dir.name}: missing={fallback_textures}")

    output_xml = cache_dir / "model.xml"
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"[asset] patched MJCF: {output_xml}")
    return cache_dir


# ==========================================================================
# Asset set: brush, dustpan, bin, debris
# --------------------------------------------------------------------------
# Confirmed asset set for the single-arm table-clean (Sweep) task.
#
# Every constant here was measured in SAPIEN, not taken from a spec sheet, and
# every asset was verified to simulate. Provenance for each number is the script
# named beside it in `tools/`.
#
#     brush     Objaverse 387894f1  "Deck Brush"   CC BY 4.0
#     dustpan   Objaverse 21901bdd  "Shovel"       CC BY-SA 4.0
#     debris    RoboCasa  SugarCube001             CC BY 4.0 (NVIDIA)
#     bin       Objaverse bb540fc9  "Office Bin"   CC BY 4.0
#
# Collision, and why it differs per asset
# ---------------------------------------
# PhysX allows non-convex triangle collision only on static/kinematic bodies, so
# anything that gets picked up needs convex decomposition. That splits the set:
#
#     brush, dustpan   dynamic -> CoACD  (`pip install coacd`)
#     bin              static  -> exact triangle mesh, 1 shape, no decomposition
#     debris           RoboCasa MJCF ships its own convex collision already
#
# A single convex hull spans the dustpan's mouth and fills the cavity, which is
# why CoACD is mandatory rather than an optimisation. See tools/verify_assets.py.
#
# Where the assets live
# ---------------------
# Downloaded meshes go under `tasks/assets/task_object_meshes/TableClean/`, the per-task asset folder
# for this task (repo convention: each task owns a folder under `tasks/assets/`).
# Attribution and licensing: `tasks/assets/task_object_meshes/TableClean/ATTRIBUTION.md`. The GLBs
# themselves are gitignored; fetch them with tools/fetch_assets.sh rather than
# committing them.
# ==========================================================================
# `parent` is tasks/.
TASK_ASSET_DIR = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "task_object_meshes"
    / "TableClean"
)
GLB_CACHE = TASK_ASSET_DIR / "glb_cache"
ROBOCASA = TASK_ASSET_DIR / "robocasa" / "lightwheel"

# --------------------------------------------------------------------------
# Objaverse UIDs. Renders: renders/verified/{TOOL__brush_deck, dustpan_A,
# trashcan, SUGAR5__macro5}.png
# --------------------------------------------------------------------------
UID_BRUSH = "387894f112f44d82a0dbbe6aa3906454"
UID_DUSTPAN = "21901bdd81f649e89af087ae6f67ca7e"
UID_BIN = "bb540fc9fcb04a5aa13a5d9b72918f8f"
DEBRIS_REF = "sugar_cube/SugarCube001"

# CoACD settings that produced the verified decompositions. threshold=0.03 is
# below the 0.05 default: the dustpan's thin entry lip is exactly the feature a
# looser threshold smooths away, and losing it costs the whole task.
COACD_PARAMS = dict(threshold=0.03, max_convex_hull=64, preprocess_resolution=64)

# --------------------------------------------------------------------------
# Measured geometry. tools/verify_tool.py, verify_sweep.py, verify_debris.py
# --------------------------------------------------------------------------
# NOTE: this normalises the brush's OVERALL length (handle included), so most
# of it is handle. At 0.16 the bristle head was only 71 mm across against the
# dustpan's 240 mm mouth, which looks and behaves undersized. 0.30 gives a
# 128 mm head -- about half the pan width, so two passes fill it.
# Scale is a genuine trade-off, both ends measured:
#   0.16  bristle head 71 mm (small vs the 240 mm pan mouth) BUT the whole tool
#         is 0.24 m and the arm grasps and lifts it reliably.
#   0.30  head 152 mm, which looks right -- but the tool becomes 0.455 m long
#         with a 0.48 m head-to-grip span, so the head swings into the bin and
#         the scripted grasp stops working entirely.
# 0.20 keeps a usable head without the swing radius blowing up.
BRUSH_WIDTH = 0.20              # normalise longest horizontal dim to this
BRUSH_MASS = 0.20              # real hand/deck brush 100-200 g
BRUSH_HULLS = 4                 # observed CoACD output
# Bristle contact patch, i.e. the part below 25 mm that actually pushes debris.
# Debris spread wider than this slips around the brush -- that is a 0/3 sweep,
# and it is the single most common way to mis-size the debris zone.
BRUSH_SWEEP_FACE = 0.099        # measured at BRUSH_WIDTH = 0.20
BRUSH_SWEEP_DEPTH = 0.031
# Handle: 3 graspable bins, 32-36 mm across, standing up to 76 mm off the table.
# Panda's usable span is ~68 mm (80 mm open, minus margin), so this fits with
# room to spare, and it is elevated -- a top-down pinch, not a flat-rod scrape.
BRUSH_HANDLE_WIDTH = 0.060      # scales with BRUSH_WIDTH; still < 68 mm span
BRUSH_HANDLE_TOP_Z = 0.143

# Settled pose, measured after 400 steps. The GLB's authored frame stands the
# brush upright, so spawning at the mesh-derived lift makes it topple 0.195 m
# on every reset: wasted settling, a randomised grasp pose, and a lift baseline
# that is 0.195 m too high (so "lifted 5 cm" never becomes true). Spawn it
# already lying down instead.
BRUSH_REST_Z = -0.0048
BRUSH_REST_QUAT = (0.8633, -0.0147, -0.5044, -0.0063)

# Head and grip in the brush's OWN frame, measured once from the mesh (see
# tools/ for the profile). Runtime geometry heuristics kept misidentifying the
# head -- they reported it 5 cm from the grasp when it is really 48 cm away,
# which sent the arm sweeping with the handle instead of the bristles.
#
# The grip is deliberately on the handle CLOSE to the head (local z = -0.11,
# cross-section 34 mm, well inside the 68 mm gripper span). Grasping the far end
# gives a 0.48 m lever arm: the head swings into the bin and nothing is
# controllable.
BRUSH_HEAD_LOCAL = (0.0, -0.150, -0.171)     # bristle face, lowest point
BRUSH_GRIP_LOCAL = (0.0, -0.083, -0.110)     # handle, near the head
BRUSH_LEN = 0.455                             # head-to-handle-end, for layout

DUSTPAN_WIDTH = 0.24
DUSTPAN_MASS = 0.15            # real moulded-PP dustpan 100-200 g
DUSTPAN_HULLS = 22
# The mouth is the asset's local -y edge; local +y is the 72 mm handle wall.
# yaw=+90 deg maps local -y onto world +x, i.e. mouth facing an incoming sweep.
DUSTPAN_MOUTH_YAW_DEG = 90.0
PAN_REST_Z = 0.0189             # settles slightly above its spawn lift
DUSTPAN_LIP_H = 0.0108          # 10.8 mm measured at the mouth edge
# Sweeping the brush UP to clear this lip is wrong: it raises the bristles off
# the debris and pushing stops (0/3 at every speed). Keep the brush down and let
# the debris tip over the lip. See tools/verify_integrated.py.
DUSTPAN_BRUSH_LIFT = 0.0

BIN_WIDTH = 0.22
BIN_HEIGHT_AT_WIDTH = 0.32      # tall: budget the lift, or floor-stand the bin
BIN_STATIC = True

DEBRIS_SCALE = 1.0
DEBRIS_SIZE = 0.010             # 10 mm cube
DEBRIS_REST_Z = 0.0049          # settled height, 5 instances, identical
DEBRIS_MASS = 0.0014           # 10 mm sugar cube, rho 1590 -> 1.4 g
DEBRIS_HULLS = 2                # cheapest collision in the whole survey
DEBRIS_COUNT = 5

# --- hand-pick variant (tasks/table_cleaning/table_cleaning_long.py) -------------------
# The brush-sweep task was retired; the operator now picks the pieces up by
# hand. A 10 mm cube is below what a teleoperated dexterous hand pinches
# reliably, so the pieces are scaled x2 (20 mm, rho 1590 -> 11 g) and there
# are three of them.
PICK_DEBRIS_COUNT = 3
PICK_DEBRIS_SCALE = 2.0
PICK_DEBRIS_MASS = 0.011            # kg, 20 mm sugar cube at rho 1590
PICK_DEBRIS_SPAWN_SPAN = 0.10       # m, spread of the three pieces

# Grip sleeve on the dustpan handle. Measured on the cooked collision hulls
#: the handle rises from the pan at local (-0.035, 0, 0.021) to
# (-0.111, 0, 0.051), a 14-19 mm bar 82 mm long -- pinchable, but too thin for
# the wrap grasp that carries the pan's 0.15 N*m about the grip when it is
# lifted and tipped. The sleeve is a 26 mm capsule along that axis (a rubber
# grip on a real dustpan handle); same actor, so it never self-collides.
DUSTPAN_GRIP_CENTER = (-0.073, 0.0, 0.036)
DUSTPAN_GRIP_QUAT = (0.9823, 0.0, 0.1874, 0.0)   # wxyz, +21.6 deg about y: x -> handle axis
DUSTPAN_GRIP_HALF_LENGTH = 0.041
DUSTPAN_GRIP_RADIUS = 0.013
# Centre of mass pinned over the platform (see tasks/table_cleaning/table_cleaning_long.py _load_scene);
# the platform's flat contact strip is x in [+0.02, +0.12] of the pan frame.
DUSTPAN_CMASS_LOCAL = (0.045, 0.0, -0.005)

# Keep the debris spawn box inside the brush's sweep face, or plan multi-pass.
DEBRIS_SPAWN_SPAN = 0.080       # < BRUSH_SWEEP_FACE (0.099)


def _yaw_q(deg: float) -> np.ndarray:
    a = np.deg2rad(deg) / 2.0
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw])


def glb_frame(uid: str, target_width: float):
    """(scale, z_up_quat, table_lift, extents) for an Objaverse GLB.

    glTF is Y-up by convention; a +90 deg rotation about X stands it Z-up.
    Requires trimesh only at load time, not per step.
    """
    import trimesh
    m = trimesh.load(GLB_CACHE / f"{uid}.glb", process=False, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    vz = v @ rot.T
    ext = vz.max(0) - vz.min(0)
    s = target_width / max(ext[0], ext[1])
    q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    return s, q, float(-vz.min(0)[2] * s), ext * s


def build_brush(scene, position, yaw_deg=90.0, kinematic=False):
    """Deck brush as a dynamic (grasped) body. CoACD collision."""
    s, q, lift, ext = glb_frame(UID_BRUSH, BRUSH_WIDTH)
    q_tot = _qmul(_yaw_q(yaw_deg), q)
    b = scene.create_actor_builder()
    path = str(GLB_CACHE / f"{UID_BRUSH}.glb")
    b.add_visual_from_file(filename=path, scale=[s] * 3, pose=sapien.Pose(q=q_tot))
    b.add_multiple_convex_collisions_from_file(
        filename=path, scale=[s] * 3, pose=sapien.Pose(q=q_tot), density=400,
        material=object_material("hard_plastic", dominant="pad"),
        decomposition="coacd", decomposition_params=COACD_PARAMS)
    b.initial_pose = sapien.Pose(p=[position[0], position[1], lift + 0.002])
    return (b.build_kinematic(name="brush") if kinematic
            else b.build_dynamic(name="brush")), lift


def build_dustpan(scene, position, yaw_deg=DUSTPAN_MOUTH_YAW_DEG):
    """Dustpan as a dynamic body (it must be liftable). CoACD collision."""
    s, q, lift, ext = glb_frame(UID_DUSTPAN, DUSTPAN_WIDTH)
    q_tot = _qmul(_yaw_q(yaw_deg), q)
    b = scene.create_actor_builder()
    path = str(GLB_CACHE / f"{UID_DUSTPAN}.glb")
    b.add_visual_from_file(filename=path, scale=[s] * 3, pose=sapien.Pose(q=q_tot))
    b.add_multiple_convex_collisions_from_file(
        filename=path, scale=[s] * 3, pose=sapien.Pose(q=q_tot), density=500,
        material=object_material("hard_plastic", dominant="pad"),
        decomposition="coacd", decomposition_params=COACD_PARAMS)
    # Grip sleeve (see DUSTPAN_GRIP_* above): collision + visual on the handle.
    grip_pose = sapien.Pose(p=list(DUSTPAN_GRIP_CENTER), q=list(DUSTPAN_GRIP_QUAT))
    b.add_capsule_collision(pose=grip_pose, radius=DUSTPAN_GRIP_RADIUS,
                            half_length=DUSTPAN_GRIP_HALF_LENGTH,
                            material=object_material("hard_plastic", dominant="pad"),
                            density=500)
    b.add_capsule_visual(pose=grip_pose, radius=DUSTPAN_GRIP_RADIUS,
                         half_length=DUSTPAN_GRIP_HALF_LENGTH,
                         material=sapien.render.RenderMaterial(base_color=[0.12, 0.12, 0.14, 1.0], roughness=0.9))
    b.initial_pose = sapien.Pose(p=[position[0], position[1], lift + 0.002])
    return b.build_dynamic(name="dustpan"), lift, ext


BIN_FLOOR_SLAB_HALF_H = 0.020    # m; the slab reaches 40 mm below the visual floor (no tunnelling at pour speeds)


def bin_floor_frame(uid: str = UID_BIN, target_width: float = BIN_WIDTH):
    """(floor_z_local, floor_radius) of the bin's INTERIOR floor, in the actor's
    frame (same scale/rotation as glb_frame). Measured from the mesh: the
    floor is the lowest band of vertices near the axis; its radius is how far
    that band reaches before the wall rises."""
    import trimesh
    m = trimesh.load(GLB_CACHE / f"{uid}.glb", process=False, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    vz = v @ rot.T
    ext = vz.max(0) - vz.min(0)
    sc = target_width / max(ext[0], ext[1])
    vz = vz * sc
    r = np.linalg.norm(vz[:, :2], axis=1)
    near_axis = vz[r < 0.25 * target_width]
    floor_z = float(near_axis[:, 2].min())
    band = vz[np.abs(vz[:, 2] - floor_z) < 0.002]
    floor_r = float(np.linalg.norm(band[:, :2], axis=1).max())
    return floor_z, floor_r


def build_bin(scene, position, width=BIN_WIDTH):
    """Trash bin as a STATIC body: the exact triangle mesh for the walls plus an
    explicit floor slab.

    the triangle mesh alone has NO working interior floor -- a
    cube released 3 cm above it (0.49 m/s) fell straight through and came to
    rest on the table under the bin (z 0.0098 instead of 0.018), with zero
    debris-bin contact force. Every recorded pour (5/5 episodes, 0901 human
    included) ended with all three pieces under the visual floor, i.e.
    invisible. The floor is now four rotated boxes (a 16-gon disk) whose top
    face is the mesh's interior floor and which reach 40 mm down.

    Returns (actor, lift, ext, floor_z_world): ``floor_z_world`` is the height
    a resting piece's underside sits at.
    """
    s, q, lift, ext = glb_frame(UID_BIN, width)
    floor_z_local, floor_r = bin_floor_frame(UID_BIN, width)
    b = scene.create_actor_builder()
    path = str(GLB_CACHE / f"{UID_BIN}.glb")
    material = object_material("hard_plastic", dominant="table")
    b.add_visual_from_file(filename=path, scale=[s] * 3, pose=sapien.Pose(q=q))
    b.add_nonconvex_collision_from_file(
        filename=path, scale=[s] * 3, pose=sapien.Pose(q=q), density=500,
        material=material)
    # Floor slab: 4 boxes at 0/45/90/135 deg, half-width r*tan(22.5 deg), so their
    # union covers the whole floor disk out to the wall. Top face = floor_z_local.
    hz = BIN_FLOOR_SLAB_HALF_H
    for k in range(4):
        yaw = k * np.pi / 4.0
        b.add_box_collision(
            pose=sapien.Pose(p=[0.0, 0.0, floor_z_local - hz],
                             q=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
            half_size=[floor_r * 0.99, floor_r * 0.99 * np.tan(np.pi / 8), hz],
            material=material)
    z0 = lift + 0.002
    b.initial_pose = sapien.Pose(p=[position[0], position[1], z0])
    actor = b.build_static(name="trashcan")
    _attach_bin_wall_backstop(actor, floor_z_local, floor_r, ext, lift)
    return actor, lift, ext, float(z0 + floor_z_local)


BIN_WALL_SEGMENTS = 24
BIN_WALL_RADIAL_HALF_T = 0.006
BIN_WALL_BOTTOM_OVERLAP = 0.004
BIN_WALL_TOP_INSET = 0.005


def _attach_bin_wall_backstop(actor, floor_z_local, floor_r, ext, lift):
    """Thick, tapered inner wall panels so debris cannot tunnel through the side
    wall either. Measured on the mesh + floor-slab bin: cubes fired at
    the wall at 0.5-2.5 m/s escaped 1/24 times; with these panels 0/24.

    Added to the DumpDustpan subtask only (by another agent); moved
    here so the long task and both subtasks share one bin. Same
    geometry, same shape order.
    """
    import sapien.physx as physx
    wall_bottom = floor_z_local - BIN_WALL_BOTTOM_OVERLAP
    wall_top = float(ext[2]) - float(lift)
    wall_height = wall_top - wall_bottom
    top_r = float(min(ext[0], ext[1])) / 2 - BIN_WALL_TOP_INSET
    tilt = np.arctan2(top_r - floor_r, wall_height)
    inner_mid_r = (floor_r + top_r) / 2 * np.cos(np.pi / BIN_WALL_SEGMENTS)
    radial_half = BIN_WALL_RADIAL_HALF_T
    centre_r = inner_mid_r + radial_half * np.cos(tilt)
    centre_z = (wall_bottom + wall_top) / 2 - radial_half * np.sin(tilt)
    tangent_half = (centre_r + radial_half) * np.tan(np.pi / BIN_WALL_SEGMENTS) * 1.05
    panel_half_h = wall_height / (2 * np.cos(tilt))
    for obj in actor._objs:
        component = obj.find_component_by_type(physx.PhysxRigidStaticComponent)
        if component is None:
            continue
        shapes = component.get_collision_shapes()
        if not shapes:
            continue
        material = shapes[0].get_physical_material()
        for k in range(BIN_WALL_SEGMENTS):
            yaw = 2 * np.pi * k / BIN_WALL_SEGMENTS
            cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
            ct, st = np.cos(tilt / 2), np.sin(tilt / 2)
            shape = physx.PhysxCollisionShapeBox(
                np.asarray([radial_half, tangent_half, panel_half_h], dtype=np.float32), material)
            shape.set_local_pose(sapien.Pose(
                p=[centre_r * np.cos(yaw), centre_r * np.sin(yaw), centre_z],
                q=[cy * ct, -sy * st, cy * st, sy * ct]))
            component.attach(shape)


def build_debris(scene, positions, prepare_mjcf, mass=DEBRIS_MASS, scale=DEBRIS_SCALE):
    """RoboCasa sugar cubes. MJCF already ships convex collision -- no CoACD.

    `prepare_mjcf` is prepare_robocasa_mjcf (defined above), passed in
    so this module stays import-light.
    """
    import sapien.physx as physx
    from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject

    patched = prepare_mjcf(ROBOCASA / DEBRIS_REF, "sweep_debris")
    out = []
    material = object_material("sugar_crystal", dominant="table")
    for i, (x, y) in enumerate(positions):
        # NOTE: MJCFObject stores `density`/`friction` but never applies them
        # (robocasa objects.py). Both are therefore set explicitly after build.
        t = MJCFObject(scene=scene, name=f"debris{i}_t", mjcf_path=str(patched),
                       scale=scale)
        b = t.actor_builder
        b.set_scene_idxs([0])
        b.initial_pose = sapien.Pose(p=[x, y, scale * DEBRIS_SIZE / 2 + 0.004])
        a = b.build_dynamic(name=f"debris_{i}")
        for o in a._objs:
            c = o.find_component_by_type(physx.PhysxRigidDynamicComponent)
            if c is not None:
                set_realizable_mass(c, mass)
                for shape in c.get_collision_shapes():
                    shape.set_physical_material(material)
        out.append(a)
    return out


# --------------------------------------------------------------------------
# Verification status. Update this table, not your memory.
# --------------------------------------------------------------------------
VERIFIED = {
    "brush":   "stable (drift 0, v 0); 3 graspable handle bins; 3/3 sweep "
               "@0.31 m/s (kinematic drive)",
    "dustpan": "dynamic via CoACD, 22 hulls; 3/3 dropped in; 3/3 swept in "
               "@0.68 m/s with a flat blade",
    "bin": "static exact mesh + explicit floor slab (the mesh "
               "floor did not collide; pieces rested on the table under it)",
    "debris":  "stability PASS (best score of 26 assets); 5 instances settle "
               "identically, zero residual velocity",
}

NOT_YET_VERIFIED = {
    "grasp":        "no scripted Panda grasp of the brush has been run; "
                    "graspability is inferred from handle geometry only",
    "sweep_rate":   "sweep results are single-seed. Real debris peaked at 2/3 "
                    "and showed no size trend -- treat as noisy, not a rate",
    "dump":         "lifting the loaded dustpan and emptying it into the bin "
                    "is untested end to end",
    "bin_height":   f"bin stands {BIN_HEIGHT_AT_WIDTH} m at {BIN_WIDTH} m wide; "
                    "confirm the loaded-dustpan lift stays in Panda range",
}



# --------------------------------------------------------------------------
# Layout (task layer). Everything is stated against the shared rig's table
# centre so the family moves with the rig. Distances below are from the RIGHT
# arm base at (PLATFORM_ROOT_X, -0.22) to each object's nearest surface,
# measured after the move -- the old layout had the debris 1.04-1.09 m
# out, past the 0.90 m reach limit, which did not matter for a brush sweep but
# is fatal for picking them up by hand.
# --------------------------------------------------------------------------
ROBOT_ROOT = platform_root_pose()
ARM_INIT_QPOS = list(RIG_ARM_INIT_QPOS)

DEBRIS_CENTER = (TABLE_CENTER_X - 0.30, -0.30)      # pieces spread 8 cm around this
DUSTPAN_POS = (TABLE_CENTER_X - 0.30, -0.52)        # mouth toward the debris
BIN_POS = (TABLE_CENTER_X - 0.30, 0.04)             # on the far side of the arm

# Per-episode layout randomization: uniform half-ranges around the nominal
# layout, every draw from the episode RNG (tasks/config/layout_randomization.py).
PAN_RANDOM_XY = (0.020, 0.020)          # m
PAN_RANDOM_YAW_DEG = 10.0
BIN_RANDOM_XY = (0.020, 0.020)          # m; the bin is static, so this is drawn at scene build
DEBRIS_SCATTER_HALF_SPAN = PICK_DEBRIS_SPAWN_SPAN / 2   # square around DEBRIS_CENTER
DEBRIS_MIN_SEPARATION = 0.035           # m between piece centres (pieces are 20 mm)
# Reward-shaping target for the set-down (dense reward only; success accepts a
# set-down anywhere on the table, see _set_down).
PAN_STOW_XY = (TABLE_CENTER_X - 0.30, -0.20)

LIFT_H = 0.05                       # counts as "picked up" / back down
RELEASE_SPEED = 0.02                # this still => actually let go
POUR_TILT = np.deg2rad(70.0)        # dustpan rotated past this => pouring
HOLD_STEPS = 8                      # consecutive steps before the pick stage latches
# (user): the dump stage and the long task succeed on ONE condition --
# every piece resting on the bin floor for DUMP_HOLD_STEPS consecutive steps
# (0.5 s). Grasp / pour / set-down remain milestones for the HUD and reward only.
DUMP_HOLD_STEPS = 10
BIN_FLOOR_BAND = 0.05               # m above the bin floor a resting piece's centre may be (pieces may stack)

ENV_ID = "TableCleaningLong-Sharpa-v1"
MAX_EPISODE_STEPS = 1200            # 60 s: 3 hand-picks + a pour + a set-down
ENV_IDS = ("TableCleaningPickDebris-Sharpa-v1", "TableCleaningDumpDustpan-Sharpa-v1", ENV_ID)

ROBOT_UID_SHARPA = "panda_sharpa_hand_right_angled_mobile_base"
ROBOT_UID_WUJI = "panda_wuji_hand_right_angled_mobile_base"
ROBOT_UID_XHAND = "panda_xhand_hand_right_angled_mobile_base"
ROBOT_UID_GRIPPER = "panda_gripper_right_mobile_base"
ROBOT_UIDS = (ROBOT_UID_SHARPA, ROBOT_UID_WUJI, ROBOT_UID_XHAND, ROBOT_UID_GRIPPER)


def _debris_xy(n: int, span: float, center, seed: int = 0):
    rng = np.random.default_rng(seed)
    cx, cy = center
    out = []
    for i in range(n):
        ang = 2 * np.pi * i / n + rng.uniform(-0.25, 0.25)
        rad = span * 0.5 * (0.35 + 0.65 * ((i + 1) / n))
        out.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    return out


class TableCleanBase(BaseEnv):
    """Scene, sensors and shared predicates. Subclasses own the stage machine."""

    SUPPORTED_ROBOTS = list(ROBOT_UIDS)

    def __init__(self, *args, robot_uids=ROBOT_UID_SHARPA,
                 num_envs=1,
                 debris_count=PICK_DEBRIS_COUNT,
                 robot_init_qpos_noise=ARM_INIT_QPOS_NOISE_STD, reconfiguration_freq=None, **kwargs):
        # ManiSkill only *warns* when robot_uids is outside SUPPORTED_ROBOTS,
        # so an unknown robot would load and then silently misbehave. Reject it.
        if robot_uids not in self.SUPPORTED_ROBOTS:
            raise ValueError(
                f"{type(self).__name__} supports {self.SUPPORTED_ROBOTS}; got {robot_uids!r}.")
        # Single-env only: the episode-layout helpers emit one pose each.
        if int(num_envs) != 1:
            raise ValueError(
                f"{type(self).__name__} supports num_envs=1 only (got "
                f"{num_envs}): its episode-layout helpers emit a single pose."
            )
        self.debris_count = int(debris_count)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        if self.robot_init_qpos_noise < 0:
            raise ValueError("robot_init_qpos_noise must be non-negative")
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        super().__init__(*args, robot_uids=robot_uids, num_envs=num_envs,
                         reconfiguration_freq=reconfiguration_freq, **kwargs)

    # ------------------------------------------------------------------ sim
    @property
    def _default_sim_config(self):
        return make_sim_config(
            found_lost_pairs_capacity=2 ** 25,
            max_rigid_contact_count=2 ** 21,
            max_rigid_patch_count=2 ** 20,
        )

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig("base_camera",
                         sapien_utils.look_at(
                             eye=[TABLE_CENTER_X + 0.45, -0.55, 0.45],
                             target=[TABLE_CENTER_X - 0.30, -0.20, 0.05]),
                         128, 128, np.deg2rad(60), 0.01, 100.0),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig("render_camera",
                            sapien_utils.look_at(
                                eye=[TABLE_CENTER_X + 0.55, -0.85, 0.62],
                                target=[TABLE_CENTER_X - 0.30, -0.20, 0.06]),
                            960, 720, np.deg2rad(54), 0.01, 100.0)

    def _load_agent(self, options):
        super()._load_agent(options, ROBOT_ROOT)

    # ---------------------------------------------------------------- scene
    def _load_scene(self, options):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        set_work_table_pose(self)

        self.pan, self.pan_lift, self.pan_ext = build_dustpan(
            self.scene, DUSTPAN_POS)
        self._set_mass(self.pan, DUSTPAN_MASS)
        # (user): the pan must REST ON ITS PLATFORM. With the volume-
        # based centre of mass (x = -0.011, pulled back by the grip sleeve) the
        # pan tipped 29 deg onto its handle at rest. A real dustpan's mass is in
        # the moulded body; the foam grip is light -- so the centre of mass is
        # pinned over the platform.
        for o in self.pan._objs:
            c = o.find_component_by_type(physx.PhysxRigidDynamicComponent)
            if c is not None:
                c.cmass_local_pose = sapien.Pose(list(DUSTPAN_CMASS_LOCAL))

        self.bin, self.bin_lift, self.bin_ext, self.bin_floor_z = build_bin(
            self.scene, self._bin_layout_xy())

        self._measure_pan_interior()
        self.debris = build_debris(
            self.scene,
            _debris_xy(self.debris_count, PICK_DEBRIS_SPAWN_SPAN, DEBRIS_CENTER),
            prepare_robocasa_mjcf,
            mass=PICK_DEBRIS_MASS,
            scale=PICK_DEBRIS_SCALE,
        )

        for free_body in (self.pan, *self.debris):
            set_task_object_energy_fence(free_body)

        self._alloc_buffers()

    def _bin_layout_xy(self):
        """Where the bin is built this episode.

        The bin is a STATIC body, so its per-episode offset has to be drawn when the
        scene is (re)built -- and every reset rebuilds it (reconfiguration_freq=1 for
        the single env this task supports). ManiSkill seeds the episode RNG before
        reconfiguring and re-seeds it afterwards, so a draw from the episode RNG here
        would be repeated verbatim by the first draw of `_initialize_episode`. The
        offset therefore comes from a stream derived from the episode seed."""
        seed = getattr(self, "_episode_seed", None)
        if seed is None:
            return BIN_POS
        rng = layout_rng(self, 0, salt=LAYOUT_STREAM_SALT + 2)   # the static bin: its own stream
        dx, dy, _ = uniform_offset(rng, BIN_RANDOM_XY)
        return (BIN_POS[0] + dx, BIN_POS[1] + dy)

    def _measure_pan_interior(self):
        """Cache the dustpan's usable floor, in the pan's own local frame.

        The pan's bounding box is not its interior: it spans the handle and
        the raised walls, so only the flat floor band actually holds anything.
        """
        comp = self.pan._objs[0].find_component_by_type(
            physx.PhysxRigidDynamicComponent)
        pts = []
        for sh in comp.get_collision_shapes():
            if not hasattr(sh, "get_vertices"):
                continue
            v = np.asarray(sh.get_vertices(), dtype=np.float64) \
                * np.asarray(sh.get_scale(), dtype=np.float64)
            sp = sh.get_local_pose()
            q = np.asarray(sp.q, dtype=np.float64)
            w0, x0, y0, z0 = q / max(np.linalg.norm(q), 1e-12)
            R = np.array([
                [1 - 2*(y0*y0 + z0*z0), 2*(x0*y0 - z0*w0), 2*(x0*z0 + y0*w0)],
                [2*(x0*y0 + z0*w0), 1 - 2*(x0*x0 + z0*z0), 2*(y0*z0 - x0*w0)],
                [2*(x0*z0 - y0*w0), 2*(y0*z0 + x0*w0), 1 - 2*(x0*x0 + y0*y0)]])
            pts.append((v @ R.T) + np.asarray(sp.p, dtype=np.float64))
        L = np.concatenate(pts, axis=0)
        floor = L[L[:, 2] < L[:, 2].min() + 0.012]
        m = 0.012 + PICK_DEBRIS_SCALE * DEBRIS_SIZE / 2     # cube half-width + clearance
        self._pan_floor = dict(
            x=(float(floor[:, 0].min()) + m, float(floor[:, 0].max()) - m),
            y=(float(floor[:, 1].min()) + m, float(floor[:, 1].max()) - m),
            z=float(floor[:, 2].max()) + 0.007)
        # True local bounds of the cooked collision geometry (the grip sleeve
        # is excluded: it sticks up above the pan and would inflate the box).
        body = L[L[:, 2] < L[:, 2].min() + 0.06]
        self._pan_lo = body.min(axis=0)
        self._pan_hi = body.max(axis=0)

    def _place_in_pan(self, b, env_idx=None):
        """Seat the debris on the dustpan floor, in the PAN's frame (episode RNG)."""
        f = self._pan_floor
        first = 0 if env_idx is None else int(torch.as_tensor(env_idx).reshape(-1)[0])
        rng = layout_rng(self, first, salt=LAYOUT_STREAM_SALT + 1)   # the pan load: its own stream
        n = len(self.debris)
        xs = np.linspace(f["x"][0], f["x"][1], n)
        pose = self.pan.pose
        R = pose.to_transformation_matrix()[0, :3, :3].detach().cpu().numpy()
        origin = pose.p[0].detach().cpu().numpy()
        for i, d in enumerate(self.debris):
            local = np.array([
                xs[i],
                float(rng.uniform(f["y"][0], f["y"][1])) * 0.75,
                f["z"] + PICK_DEBRIS_SCALE * DEBRIS_SIZE / 2], dtype=np.float64)
            world = origin + R @ local
            p = torch.tensor(world, dtype=torch.float32,
                             device=self.device).view(1, 3).expand(b, -1).clone()
            d.set_pose(Pose.create_from_pq(p, d.pose.q))
            zero = torch.zeros(b, 3, dtype=torch.float32, device=self.device)
            d.set_linear_velocity(zero)
            d.set_angular_velocity(zero)

    def _set_mass(self, actor, mass):
        """Explicit mass with a realizable inertia (tasks/config/mass_properties)."""
        for o in actor._objs:
            c = o.find_component_by_type(physx.PhysxRigidDynamicComponent)
            if c is not None:
                set_realizable_mass(c, mass)

    def _alloc_buffers(self):
        z = lambda: torch.zeros(self.num_envs, dtype=torch.bool,  # noqa: E731
                                device=self.device)
        self.stage = torch.zeros(self.num_envs, dtype=torch.long,
                                 device=self.device)
        self.hold = torch.zeros(self.num_envs, dtype=torch.int32,
                                device=self.device)
        self.f_picked = z()
        self.f_grasp_pan = z()
        self.f_poured = z()
        self.f_stow_pan = z()
        self.success_latched = z()

    FLAG_NAMES = ("f_picked", "f_grasp_pan", "f_poured", "f_stow_pan")

    def _reset_buffers(self, env_idx):
        self.stage[env_idx] = 0
        self.hold[env_idx] = 0
        for f in self.FLAG_NAMES + ("success_latched",):
            getattr(self, f)[env_idx] = False

    # ----------------------------------------------------------- episode
    def _reset_robot(self, b):
        """Reset from the agent's own rest keyframe, arm on the shared start pose."""
        kf = self.agent.keyframes["rest"].qpos
        qpos = torch.tensor(kf, dtype=torch.float32,
                            device=self.device).view(1, -1).expand(b, -1).clone()
        qpos[:, :7] = torch.tensor(ARM_INIT_QPOS, dtype=torch.float32,
                                   device=self.device)
        qpos[:, 7:] = 0.0
        for k in range(b):
            index, noise = arm_qpos_noise(
                episode_rng(self, k), self.agent, 1, self.robot_init_qpos_noise, device=self.device)
            qpos[k, index] += noise[0]
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(ROBOT_ROOT)

    def _place(self, actor, xy, z, b, yaw_deg=None, yaw_delta_deg=0.0):
        """Place at (xy, z). `yaw_deg` sets an absolute yaw; otherwise the built
        orientation is kept and turned by `yaw_delta_deg` about the vertical."""
        p = torch.tensor([xy[0], xy[1], z], dtype=torch.float32,
                         device=self.device).view(1, 3).expand(b, -1).clone()
        if yaw_deg is None:
            q = actor.pose.q
            if yaw_delta_deg:
                built = q[0].detach().cpu().numpy().astype(np.float64)
                q = torch.tensor(quat_mul_wxyz(yaw_quat_wxyz(np.deg2rad(yaw_delta_deg)), built),
                                 dtype=torch.float32, device=self.device).view(1, 4).expand(b, -1)
        else:
            a = np.deg2rad(yaw_deg) / 2
            q = torch.tensor([np.cos(a), 0.0, 0.0, np.sin(a)],
                             dtype=torch.float32,
                             device=self.device).view(1, 4).expand(b, -1)
        actor.set_pose(Pose.create_from_pq(p, q))
        zero = torch.zeros(b, 3, dtype=torch.float32, device=self.device)
        actor.set_linear_velocity(zero)
        actor.set_angular_velocity(zero)

    def _place_scene_start(self, env_idx):
        """The long task's reset: dustpan at rest, debris loose on the table."""
        b = len(env_idx)
        self.table_scene.initialize(env_idx)
        set_work_table_pose(self)
        # Per-episode layout (single env): pan offset + yaw, then the pieces scattered
        # around DEBRIS_CENTER with a minimum separation, each with a random yaw.
        # Drawn BEFORE the robot reset so the layout is the first episode-RNG draw.
        rng = layout_rng(self, int(env_idx.reshape(-1)[0]))
        self._reset_robot(b)
        dx, dy, yaw = uniform_offset(rng, PAN_RANDOM_XY, PAN_RANDOM_YAW_DEG)
        self._place(self.pan, (DUSTPAN_POS[0] + dx, DUSTPAN_POS[1] + dy), PAN_REST_Z, b,
                    yaw_delta_deg=float(np.rad2deg(yaw)))
        rest_z = PICK_DEBRIS_SCALE * DEBRIS_SIZE / 2
        spots = scatter_xy(rng, self.debris_count, DEBRIS_CENTER,
                           DEBRIS_SCATTER_HALF_SPAN, DEBRIS_MIN_SEPARATION)
        for d, (x, y) in zip(self.debris, spots):
            self._place(d, (x, y), rest_z + 0.004, b, yaw_delta_deg=float(rng.uniform(0.0, 360.0)))
        self._reset_buffers(env_idx)

    # -------------------------------------------------------- predicates
    _FINGER_KEYS = (("thumb", "finger1"), ("index", "finger2"), ("middle", "mid_", "finger3"),
                    ("ring", "finger4"), ("pinky", "finger5"))

    def _finger_contact_links(self):
        """Hand links that actually CARRY collision geometry, grouped by finger.

        the old test read contact forces on ``agent.tip_links`` --
        on Sharpa those ``*_fingertip`` frames are collision-LESS reference
        sites (0 shapes), so ``_grasped`` was False forever: a piece carried
        over the pan counted as "placed" and the pan could never be "grasped".
        Grouping by finger keeps the "N distinct fingers" meaning."""
        # Keyed on the LIVE robot. With num_envs=1 every reset() reconfigures
        # (reconfiguration_freq=1) and rebuilds the agent; a link list cached
        # on the env keeps pointing into the dead scene and reads 0 N forever.
        # measured on the 0901 human DumpDustpan episodes -- index
        # and thumb carried 9-15 N on the pan while pan_grasped stayed False in
        # every frame, so stage 2 could never latch and the task was unwinnable.
        robot = self.agent.robot
        cached = getattr(self, "_finger_links_cache", None)
        if cached is not None and cached[0] is robot:
            return cached[1]
        groups = {}
        fingers = [getattr(self.agent, name, None) for name in ("finger1_link", "finger2_link")]
        if all(finger is not None for finger in fingers):
            # Parallel gripper: each finger (with its pad) is one group.
            for index, finger in enumerate(fingers):
                groups[index] = [
                    link for link in self.agent.robot.get_links()
                    if link.name.startswith(finger.name) and len(link._objs[0].get_collision_shapes()) > 0
                ]
            self._finger_links_cache = (robot, groups)
            return groups
        for link in self.agent.robot.get_links():
            if not link.name.startswith(("right_", "left_")):
                continue
            if len(link._objs[0].get_collision_shapes()) == 0:
                continue
            finger = next((i for i, keys in enumerate(self._FINGER_KEYS)
                           if any(k in link.name for k in keys)), None)
            if finger is None:
                continue          # palm / wrist shell: touching it is not a grasp
            groups.setdefault(finger, []).append(link)
        self._finger_links_cache = (robot, groups)
        return groups

    def _grasped(self, actor, min_force=0.4, min_fingers=2):
        """Multi-finger grasp test for a dexterous hand: at least ``min_fingers``
        DISTINCT fingers in force contact with the object through their
        collision-bearing links (BaseAgent.is_grasping assumes a two-finger
        gripper and would silently report "never grasping")."""
        groups = self._finger_contact_links()
        if not groups:
            return torch.zeros(self.num_envs, dtype=torch.bool,
                               device=self.device)
        n = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        for links in groups.values():
            f = torch.zeros(self.num_envs, device=self.device)
            for link in links:
                f = f + torch.linalg.norm(
                    self.scene.get_pairwise_contact_forces(link, actor), dim=-1)
            n = n + (f >= min_force).int()
        return n >= min_fingers

    def _any_debris_grasped(self):
        held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for d in self.debris:
            held |= self._grasped(d, min_fingers=1)
        return held

    PAN_FLOOR_BAND = 0.09       # m above the pan floor a resting piece's centre can be (pieces may stack)
    PAN_STATIC_STEP = 0.003     # m of motion in the pan frame per step that still counts as resting

    def _debris_in_pan(self, resting=False):
        """Count debris inside the dustpan, in the PAN's own frame (pose-relative,
        so a lifted, tilted pan still reports what it is carrying).

        the z window used to reach 6 cm ABOVE the pan's rim, so a
        piece carried across the pan in the hand counted as "in the pan" --
        with a Sharpa grasp test that was blind (see _finger_contact_links),
        the pick stage latched on 1 placed + 1 carried piece. A piece now has
        to sit within PAN_FLOOR_BAND of the floor, and with ``resting=True``
        it must also be still in the pan frame."""
        pose = self.pan.pose
        R = pose.to_transformation_matrix()[..., :3, :3]
        lo = torch.tensor(self._pan_lo, dtype=torch.float32, device=self.device)
        hi = torch.tensor(self._pan_hi, dtype=torch.float32, device=self.device)
        n = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        prev = getattr(self, "_debris_local_prev", None)
        if prev is None or len(prev) != len(self.debris):
            prev = [None] * len(self.debris)
        for k, d in enumerate(self.debris):
            rel = (d.pose.p - pose.p).unsqueeze(-1)
            local = torch.matmul(R.transpose(-1, -2), rel).squeeze(-1)
            inside = ((local[:, 0] > lo[0]) & (local[:, 0] < hi[0]) &
                      (local[:, 1] > lo[1]) & (local[:, 1] < hi[1]) &
                      (local[:, 2] > lo[2] - 0.005) &
                      (local[:, 2] < lo[2] + self.PAN_FLOOR_BAND))
            if resting:
                still = (torch.linalg.norm(local - prev[k], dim=1) < self.PAN_STATIC_STEP
                         if prev[k] is not None else torch.zeros_like(inside))
                inside = inside & still
            prev[k] = local.detach().clone()
            n = n + inside.float()
        self._debris_local_prev = prev
        return n

    def _debris_in_bin(self):
        """Pieces anywhere inside the bin's volume (obs / HUD count)."""
        bin_p = self.bin.pose.p
        r = float(min(self.bin_ext[0], self.bin_ext[1])) * 0.45
        top = float(self.bin_ext[2])
        n = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for d in self.debris:
            dp = d.pose.p
            rad = torch.linalg.norm(dp[:, :2] - bin_p[:, :2], dim=1)
            n = n + ((rad < r) & (dp[:, 2] < top) & (dp[:, 2] > -0.05)).float()
        return n

    def _debris_on_bin_floor(self):
        """Pieces down on the bin floor: inside the footprint and with their centre
        within BIN_FLOOR_BAND of the floor (a piece on top of another still counts).
        Pose-only, so it restores exactly and does not depend on contact reports."""
        bin_p = self.bin.pose.p
        r = float(min(self.bin_ext[0], self.bin_ext[1])) * 0.45
        lo, hi = self.bin_floor_z - 0.02, self.bin_floor_z + BIN_FLOOR_BAND
        n = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for d in self.debris:
            dp = d.pose.p
            rad = torch.linalg.norm(dp[:, :2] - bin_p[:, :2], dim=1)
            n = n + ((rad < r) & (dp[:, 2] > lo) & (dp[:, 2] < hi)).float()
        return n

    def _tilt(self, actor):
        R = actor.pose.to_transformation_matrix()[..., :3, :3]
        return torch.acos(torch.clamp(R[..., 2, 2], -1.0, 1.0))

    def _speed(self, actor):
        return torch.linalg.norm(actor.linear_velocity, dim=1)

    def _set_down(self, actor, rest_z):
        """Let go of and resting on the table again: not grasped, still, and back
        within LIFT_H of its rest height. WHERE on the table does not matter.

        this used to demand a release within STOW_RADIUS of an
        unmarked PAN_STOW_XY, 32 cm from the pan's rest spot. 0/7 recorded
        episodes (4 human, 0901) came within that radius -- operators put the
        pan back where they found it. The stow spot guarded against "never
        lifted" in an earlier ordering; today f_stow_pan is gated on f_poured,
        which already requires a >= LIFT_H lift, so the spot buys nothing."""
        lift = actor.pose.p[:, 2] - rest_z
        return ((lift < LIFT_H) & (~self._grasped(actor))
                & (self._speed(actor) < RELEASE_SPEED))

    def _tcp_to(self, actor):
        """Distance from the mean fingertip to an object."""
        tips = getattr(self.agent, "tip_links", None)
        if tips:
            pts = torch.stack([l.pose.p for l in tips], dim=0).mean(dim=0)
        else:
            pts = self.agent.tcp.pose.p
        return torch.linalg.norm(pts - actor.pose.p, dim=1)

    def _latch(self, cond, steps=HOLD_STEPS):
        """Advance `hold` while cond holds; return True once it sticks."""
        self.hold = torch.where(cond, self.hold + 1, torch.zeros_like(self.hold))
        return self.hold >= steps

    # --------------------------------------------------- shared stage steps
    def _update_pick(self, pick_target):
        """Stage 1 (shared by the long task and subtask 1): at least
        ``pick_target`` pieces resting in the pan with the hand off them."""
        n_pan = self._debris_in_pan(resting=True)
        picked_now = (n_pan >= pick_target) & (~self._any_debris_grasped())
        self.f_picked |= picked_now
        return n_pan, picked_now

    def _update_dump(self, dump_target, gate):
        """Milestones 2-4 (shared by the long task and subtask 2): HUD and
        reward only now -- success is `_dump_success`."""
        n_pan = self._debris_in_pan()
        n_bin = self._debris_in_bin()
        grasped = self._grasped(self.pan)
        lift = self.pan.pose.p[:, 2] - PAN_REST_Z
        self.f_grasp_pan |= gate & grasped & (lift >= LIFT_H) & (n_pan >= 1)
        # Poured = the pieces reach the bin WHILE the pan is up. Without the
        # lift term (when the stow spot went away) a pan lifted once,
        # set back down and emptied piece by piece BY HAND would have latched
        # "poured" and then "stowed" on the same step. One arm cannot hold the
        # pan up and hand-carry pieces at the same time, so this is the pour.
        self.f_poured |= self.f_grasp_pan & (lift >= LIFT_H) & (n_bin >= dump_target)
        self.f_stow_pan |= self.f_poured & self._set_down(self.pan, PAN_REST_Z)
        return n_pan, n_bin, grasped, lift

    def _dump_success(self, gate):
        """The one success condition of the dump stage and of the long task:
        every piece on the bin floor for DUMP_HOLD_STEPS consecutive steps,
        latched. `gate` is the pick-stage latch (preset True in subtask 2)."""
        n_floor = self._debris_on_bin_floor()
        all_in = gate & (n_floor >= float(len(self.debris)))
        self.success_latched |= self._latch(all_in, DUMP_HOLD_STEPS)
        return n_floor, all_in

    # ------------------------------------------------------------------ #
    # env-state (offline re-render / RL-from-demo-states)
    # ------------------------------------------------------------------ #
    TASK_STATE_DIM = 2 + len(FLAG_NAMES) + 1

    def get_state_dict(self):
        state = super().get_state_dict()
        state["table_clean_task_state"] = torch.cat(
            [self.stage[:, None].float(), self.hold[:, None].float()]
            + [getattr(self, f)[:, None].float() for f in self.FLAG_NAMES]
            + [self.success_latched[:, None].float()],
            dim=-1,
        )
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        """Restore task state.

        ``env_idx`` is NOT optional in practice: ManiSkill's
        ``reset(options={"reset_to_env_states": ...})`` -- the PPO
        reverse-curriculum path and the offline RGB re-render path -- calls
        ``set_state_dict(env_states, env_idx)``. Without the parameter the whole
        family raised ``TypeError`` there while frame-0 action replay, which
        calls this with one argument, kept passing.
        """
        state = dict(state)
        task_state = state.pop("table_clean_task_state", None)
        if env_idx is None:
            restore_idx = torch.arange(self.num_envs, device=self.device)
            super().set_state_dict(state)
        else:
            restore_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            ).flatten()
            super().set_state_dict(state, restore_idx)
        # Fixed-base set_root_pose/get_state drifts root z by -2^-26 per
        # restore; re-pin the build pose so set->get stays idempotent.
        self.agent.robot.set_root_pose(ROBOT_ROOT)
        if isinstance(task_state, dict):
            # A flattened LeRobot environment state restores a top-level tensor
            # as {"value": tensor}; raw_states.pkl keeps the tensor directly.
            task_state = task_state.get("value")
        if task_state is None:
            return
        task_state = torch.as_tensor(
            task_state, device=self.device
        ).reshape(len(restore_idx), self.TASK_STATE_DIM)
        self.stage[restore_idx] = task_state[:, 0].to(torch.long)
        self.hold[restore_idx] = task_state[:, 1].to(torch.int32)
        for k, f in enumerate(self.FLAG_NAMES):
            getattr(self, f)[restore_idx] = task_state[:, 2 + k].bool()
        self.success_latched[restore_idx] = task_state[
            :, 2 + len(self.FLAG_NAMES)
        ].bool()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / self.MAX_REWARD

    def compute_sparse_reward(self, obs, action, info):
        return info["success"].float()

    # Width of the stage one-hot. A SUBCLASS WITH MORE STAGES MUST RAISE THIS
    # (it is read by _get_obs_extra through the instance, so an override on
    # the subclass takes effect).
    STAGE_ONE_HOT_CLASSES = 2

    def _get_obs_extra(self, info):
        stage = self.stage
        n = int(self.STAGE_ONE_HOT_CLASSES)
        if int(stage.max()) >= n:
            raise AssertionError(
                f"{type(self).__name__}: stage {int(stage.max())} exceeds "
                f"STAGE_ONE_HOT_CLASSES={n}; raise the class attribute"
            )
        return {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "pan_pose": self.pan.pose.raw_pose,
            "bin_pos": self.bin.pose.p,
            "debris_pos": torch.cat([d.pose.p for d in self.debris], dim=-1),
            "n_in_pan": info["n_in_pan"].unsqueeze(1),
            "n_in_bin": info["n_in_bin"].unsqueeze(1),
            "stage_one_hot": torch.nn.functional.one_hot(
                stage, num_classes=n).float(),
        }


# ==========================================================================
# Long-horizon task -- both subtasks, back to back
# ==========================================================================
@register_task(
    "TableCleaningLong",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
)
class TableCleaningLongEnv(TableCleanBase):
    """Both subtasks end to end, four latched stages.

    `stage` runs 0..4 and is exposed as a one-hot for stage-conditioned
    policies. Reward is the sum of per-stage terms gated on the preceding stage
    flag, so it never decreases at a boundary.
    """

    MAX_REWARD = 20.0
    STAGE_ONE_HOT_CLASSES = 5
    STAGE_NAMES = ("pick_debris", "grasp_pan", "pour", "stow_pan", "done")

    def __init__(self, *args, pick_target=None, dump_target=None,
                 debris_count=PICK_DEBRIS_COUNT, **kwargs):
        # (user): every piece has to be in the pan, not count-1.
        self._pick_target = (max(1, int(debris_count))
                             if pick_target is None else int(pick_target))
        # dump_target only scales the "poured" milestone; success needs every piece.
        self._dump_target = (max(1, int(debris_count) - 1)
                             if dump_target is None else int(dump_target))
        super().__init__(*args, debris_count=debris_count, **kwargs)

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            self._place_scene_start(env_idx)

    def evaluate(self):
        n_pan, picked_now = self._update_pick(self._pick_target)
        n_pan, n_bin, g_pan, lift_p = self._update_dump(
            self._dump_target, gate=self.f_picked)
        self.stage = sum(getattr(self, f).long() for f in self.FLAG_NAMES)
        n_floor, all_in = self._dump_success(gate=self.f_picked)
        return {
            "success": self.success_latched,
            "n_in_pan": n_pan, "n_in_bin": n_bin, "n_on_bin_floor": n_floor,
            "all_in_bin": all_in,
            "stage": self.stage,
            "picked": self.f_picked, "grasp_pan": self.f_grasp_pan,
            "poured": self.f_poured, "stow_pan": self.f_stow_pan,
            "pan_grasped": g_pan, "pan_lift": lift_p,
            "subtask1_done": self.f_picked,
            "subtask2_done": self.success_latched,
        }

    def compute_dense_reward(self, obs, action, info):
        r = torch.zeros(self.num_envs, device=self.device)
        n = max(1, self.debris_count)
        r = r + 3.0 * (info["n_in_pan"] / n) * (~self.f_picked).float()
        r = r + 3.0 * self.f_picked.float()
        r = r + (1.0 - torch.tanh(5.0 * self._tcp_to(self.pan))) \
            * self.f_picked.float() * (~self.f_grasp_pan).float()
        r = r + 1.5 * self.f_grasp_pan.float()
        r = r + 4.0 * (info["n_in_bin"] / n)
        r = r + 1.5 * self.f_stow_pan.float()
        return torch.where(info["success"],
                           torch.full_like(r, self.MAX_REWARD), r)


def main():
    import argparse
    import gymnasium as gym

    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default=ENV_ID, choices=ENV_IDS)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    env = gym.make(args.env_id, num_envs=1, obs_mode="state",
                   control_mode="pd_joint_pos", render_mode="rgb_array")
    env.reset(seed=0)
    e = env.unwrapped
    q = e.agent.robot.get_qpos()
    a = torch.cat([q[..., s.active_joint_indices]
                   for s in e.agent.controller.controllers.values()], -1).float()
    for _ in range(args.steps):
        _, _, _, _, info = env.step(a)
    print({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()})
    env.close()


if __name__ == "__main__":
    main()

"""RailInsertGroove: subtask 2 of DualFurnitureAssembly, multi-end-effector, single arm.

The drawer panel is already lying flat on the table with its groove facing up (the terminal
state of subtask 1). The arm picks a steel slider rail up off the table and seats it in the
groove: in through the flared mouth, along the walls, home against the stop (no pins/holes).

Registered environments (tasks/config/hands.py; the long task does this stage with the RIGHT arm)
    FurnitureAssemblyInsertRail-Sharpa-v1   robot_uids="panda_sharpa_hand_right_angled_mobile_base"
    FurnitureAssemblyInsertRail-Wuji-v1     robot_uids="panda_wuji_hand_right_angled_mobile_base"
    FurnitureAssemblyInsertRail-XHand-v1    robot_uids="panda_xhand_hand_right_angled_mobile_base"
    FurnitureAssemblyInsertRail-Gripper-v1  robot_uids="panda_gripper_right_mobile_base"

ACTION
    Same as subtask 1: DEFAULT_CONTROL_MODE = "pd_joint_delta_pos". See panel_lay_flat_v1 for
    the pd_joint_pos normalize_action caveat.

THE PANEL IS A FIXED TEMPLATE
    Subtask 2 isolates rail pickup and insertion, so the panel is a kinematic fixture at the
    centre of subtask 1's assembly region. It does not translate, rotate, or receive velocity
    from rail contact. The later dual-arm task can restore a dynamic panel when panel holding
    is itself part of the desired skill.

SUCCESS  (stateless; measured in the PANEL's frame, so panel drift does not corrupt it)
    Held for SETTLE_STEPS consecutive control steps:
      |z_local - RAIL_SEATED_Z_LOCAL| <= SEAT_DEPTH_TOL   rail bottomed out (2 mm window)
      dot(rail_axis, groove_axis) >= RAIL_SEAT_AXIS_COS screw boss at the stop-wall end
      |x_local - RAIL_SEATED_X_LOCAL| <= SEAT_X_TOL      rail pushed home against the stop wall
      dot(rail_up, panel_up)          >= RAIL_UP_COS      rail is not upside down
      contact_count                   == 0                the robot let go
      rail and panel both below the velocity thresholds

    The depth test alone is nearly sufficient, because z_local is quantised by geometry:
        pins seated in the holes            0.0090
        in the groove, pins on the floor    0.0125   <- pins stand 3.5 mm proud
        resting on the panel's top face     0.0175
    A +-2 mm window around 0.0090 cannot be reached unless the pins are in the holes. The one
    loophole is an upside-down rail, which rests on its flange edges at exactly 0.0090 too;
    the signed rail_up test closes it. The pin test is kept as an explicit, readable check.

STATE  (obs_mode="state")
    always      tcp_pose(7) rail_pose(7) panel_pose(7) tcp_to_rail_grasp(3)
                rail_to_seated(3) seated_pose(7)
    state only  all_tip_positions(3*n_tip) groove_landmarks(6) rail_ends(6)
                groove_axis(3) panel_normal(3) rail_axis(3) rail_up(3)
                rail_local_pos(3) rail_lin_vel(3) rail_ang_vel(3)
                contact_count(1) gripper_opening(1) seat_depth_error(1)
                stop_error(1) axis_align(1) up_align(1) lateral_error(1)
                is_grasped(1) is_lifted(1) is_aligned(1) is_seated(1) is_released(1)
                is_static(1)
REWARD  five stages, prefix-product gated, same convention as subtask 1:
    r_reach -> r_grasp -> r_lift -> r_align -> r_seat
    Milestone latches are reward-only and never feed into `success`.

ASSET NOTES
    The rail is a real C-section: web plus two flanges plus two pins, five convex boxes,
    2.5 mm sheet steel at 7850 kg/m^3 -> about 0.30 kg. The channel interior is genuinely open,
    which subtask 3 needs: the screwdriver tip has to reach the screw bosses on the web floor,
    and a solid box envelope would stop it at the flange tops.

    The rail spawns resting directly on the table, pins down and channel up, so the pins are
    already pointing the right way and the arm never has to flip it. No nest fixture is needed.
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

from tasks.robots.panda_gripper_right_mobile_base import PandaGripperRightMobileBase
from tasks.robots.panda_sharpa_hand_right_mobile_base import PandaSharpaHandRightAngledMobileBase
from tasks.robots.panda_wuji_hand_right_mobile_base import PandaWujiHandRightMobileBase

# The panel asset, its geometry constants and the shared helpers all live in subtask 1, so the
# two tasks can never drift apart.
from tasks.furniture_assembly.furniture_assembly_lay_flat_panel import (
    PLACE_REGION_HALF_SIZE,
    build_place_marker,
    DEFAULT_CONTROL_MODE,
    GROOVE_HALF_W,
    GROOVE_LOCAL_Z,
    MOBILE_ROBOT_UIDS,
    PANEL_HALF,
    PLACE_CENTER,
    RAIL_PICK_CENTER,
    FurnitureAssemblyLayFlatPanelEnv,
    ROBOT_ROOT_POSE,
    build_drawer_panel,
    build_dexarena_base_camera,
    make_render_material,
    quat_mul,
    yaw_to_quat_wxyz,
    GROOVE_STOP_X,
    GROOVE_LANDMARKS_LOCAL,
    finger_contact_count,
)
from tasks.robots.panda_xhand_hand_right_mobile_base import PandaXHandHandRightMobileBase  # noqa: F401
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.config.object_materials import object_material
from tasks.scene_builders import TableSceneBuilder
from tasks.config.sim_profiles import make_sim_config
from tasks.config.workspace_rig import (
    ARM_INIT_QPOS,
    PLATFORM_ROOT_X,
    set_work_table_pose,
)
from tasks.config.energy_fence import set_task_object_energy_fence
from tasks.config.task_state import DeclarativeTaskStateMixin, TaskStateField

GRIPPER_ROBOT_UID = "panda_gripper_right_mobile_base"
ENV_IDS = (
    "FurnitureAssemblyInsertRail-Sharpa-v1",
    "FurnitureAssemblyInsertRail-Wuji-v1",
    "FurnitureAssemblyInsertRail-XHand-v1",
    "FurnitureAssemblyInsertRail-Gripper-v1",
)
DEFAULT_VIDEO_DIR = "tasks/assets/task_object_meshes/furniture/videos_rail_insert"


# ===========================================================================
# Rail and anchor geometry. Subtask 3 imports the rail parts from here.
# ===========================================================================
RAIL_HALF = (0.145, 0.012, 0.006)    # 290 x 24 x 12 mm outer envelope
RAIL_SHEET_T = 0.0025                # 2.5 mm sheet; contact_offset must stay below half of it
# (user: "the rail is too slippery to pick up"): the pair friction is
# the calibrated steel-on-pad 0.8 and was not the problem -- the MASS was. Sharpa's
# thumb saturates at ~3.3 N, so a two-sided pinch carries at most
# 2 * 0.8 * 3.3 = 5.3 N (0.54 kg) at full squeeze and roughly half of that at a
# teleop-typical squeeze: a 0.28 kg steel slide sat right at the limit. Light-
# duty slides are extruded aluminium; that puts this section at ~0.10 kg.
RAIL_DENSITY = 2700.0                # aluminium light-duty slide -> ~0.10 kg for this section
RAIL_WEB_TOP_Z = -RAIL_HALF[2] + RAIL_SHEET_T                      # -0.0035
# One screw location on the web floor, shared with subtask 3.
# the screw hole moves from 55 mm to 110 mm from the rail's centre,
# i.e. 35 mm in from its near end -- where a real slider rail is actually
# screwed down. With the family's new lane this puts the screw head 0.737 m
# from the right arm root instead of 0.869 (operator: "fastening the screw
# means straightening the whole rear half of the arm").
RAIL_SCREW_LOCAL_POSITIONS = ((-0.110, 0.0, RAIL_WEB_TOP_Z),)
# The fastening screw is a real revolute child of the rail. It used
# to be a separate fixed-base articulation that a per-control-step animation
# teleported onto the rail and whose angle was WRITTEN from the measured tool
# roll -- i.e. exactly the forced re-pose the space-time rule forbids, and the
# cross recess in its head drove nothing. Now the head rides on the rail in every
# stage (so it is visible while the rail is inserted), and the ONLY thing that
# turns it is a blade pressing on the walls of its cross recess.
# (user: cannot see the cross, and the blade wanders out after a few
# degrees): 18 mm head that stands 2 mm proud of the rail flanges, a 5 mm deep cross so
# the blade is captured over most of its length, and 0.5 mm of play a side against the
# 2.4 mm blade (~10 deg of rotational slop before the walls bite).
SCREW_HEAD_RADIUS = 0.010            # m, 18 mm head
SCREW_HEAD_HALF_H = 0.0070           # m, 11 mm tall: top 2 mm above the 12 mm flanges
SCREW_SLOT_WIDTH = 0.0044            # m, cross channel; the 3.0 mm blade keeps 0.7 mm play a side
                                     # (operator: "the screwdriver is too hard to seat" -> +0.2 mm per side)
SCREW_SLOT_DEPTH = 0.0080            # m, from the top face
SCREW_JOINT_LIMIT = float(np.pi / 2.0)
SCREW_JOINT_FRICTION = 0.0 # coefficient, calibrated: breakaway ~0.001 N.m = thread resistance, so a bit rubbing the recess mouth cannot spin a free screw (drive-torque sweep, see docs)
SCREW_JOINT_DAMPING = 0.005          # N*m*s, keeps it from ringing after the blade lets go
SCREW_HEAD_CAP_HALF_H = 0.0004    # visual-only bright cap on each wing top (see _add_screw_head_link)
SCREW_SLOT_LEAD_IN = 0.0040       # 45-deg chamfer on the recess mouth: the channel opens from 4.4 to 12.4 mm at the top.
                                  # second round (operator: "the screwdriver is too hard to seat") 2.5 -> 4.0 mm:
                                  # on the bench a 12 deg mis-rolled, 1.5 mm off-axis, 6 deg tilted entry went from "sits on
                                  # the wings at 0.4 mm, never engages" to "drops 1.9 mm in, engages, turns the screw 81.7 deg".
# Thread resistance: `joint.set_friction` is a coefficient on the joint's
# constraint force and the free-hanging 15 g head carries none, so it cannot give a
# dry torque (drive sweep: breakaway 0.001 N.m at every coefficient). A zero-velocity
# damper with a force limit is a Coulomb torque: the screw needs SCREW_THREAD_TORQUE
# to turn, which a bit rubbing the recess mouth cannot supply but a blade on the
# channel walls (2.5 N at 6 mm) can. ~1/10 of the Sharpa handle-grip torque budget.
SCREW_THREAD_TORQUE = 0.015       # N.m
SCREW_THREAD_DAMPING = 5.0        # N.m.s/rad, saturates within ~3 mrad/s
RAIL_GRASP_LOCAL = (0.0, 0.0, 0.002)

# Seated rail centre expressed in the panel frame: groove floor plus half the rail height.
RAIL_SEATED_Z_LOCAL = GROOVE_LOCAL_Z + RAIL_HALF[2]                 # 0.0090
# Same quantity when the pins bottom out on the groove floor instead of entering the holes.
RAIL_UNSEATED_Z_LOCAL = PANEL_HALF[2] + RAIL_HALF[2]                 # 0.0150: rail lying on the wall tops
# Half-width of the "seated" window. Must leave real margin below the unseated rung, otherwise
# a rail merely resting on its pins would pass. 1.5 mm tolerance -> 2.0 mm margin.
# was 0.0015. Widened as far as the geometry allows: the pins
# stand 3.5 mm proud, so an UNSEATED rail resting on its pins sits 3.5 mm
# above the seat and the window must stay clear of that rung (assert
# below). 2 mm keeps a 1.5 mm margin; the groove guides a seated rail to
# within a few tenths anyway -- the real difficulty is the 6 mm pin/hole
# alignment, not depth.
SEAT_DEPTH_TOL = 0.002
_SEAT_MARGIN = RAIL_UNSEATED_Z_LOCAL - RAIL_SEATED_Z_LOCAL - SEAT_DEPTH_TOL
assert _SEAT_MARGIN > 0.001, (   # >= 1 mm of daylight between the window and the unseated rung
    f"SEAT_DEPTH_TOL={SEAT_DEPTH_TOL} leaves only {_SEAT_MARGIN * 1000:.2f} mm before the "
    "unseated rung; shrink the tolerance or make the pins stand prouder."
)

# The rail spawns resting directly on the table, pins DOWN and channel opening UP. The pins
# stand 3.5 mm proud, so they, not the web, touch the table; the web floats 3.5 mm up. This is
# the natural pre-grasp pose and already points the pins the right way for insertion, so the
# arm never has to flip the rail. Tipping angle about the weak axis is 32 deg, plenty stable.
RAIL_REST_Z = RAIL_HALF[2]                                            # 0.0060: flat rail on the table
# Slide-in seat: x is set by the stop wall, y by the groove walls.
RAIL_SEATED_X_LOCAL = GROOVE_STOP_X - RAIL_HALF[0]                    # +0.025: rail end against the stop
SEAT_X_TOL = 0.008                                                    # "pushed home" window
SEAT_Y_TOL = 0.004                                                    # groove play is +-2.5 mm; 4 leaves margin for the flange lips

# ---------------------------------------------------------------------------
# "Rail seated" -- the ONE predicate of stage 2, evaluated by this subtask and by
# the long task. Before, the long task had its own copy with up >=
# 0.95 (here 0.97) and an axis-direction term this subtask did not have, so a
# rail slid in screw-end first (180 deg about the panel normal) passed the
# subtask and failed the long task.
# ---------------------------------------------------------------------------
RAIL_SEAT_UP_COS = 0.97          # SIGNED dot(rail z, panel normal); blocks an upside-down rail
RAIL_SEAT_AXIS_COS = 0.97        # dot(rail x, groove x): the screw boss must end up at the stop-wall end
RAIL_SEAT_LIN_VEL_TOL = 0.030
RAIL_SEAT_ANG_VEL_TOL = 0.400
RAIL_SEAT_SETTLE_STEPS = 5


def rail_seat_quantities(rail_pose, panel_pose, contact_count, root_linear_velocity,
                         root_angular_velocity) -> dict:
    """Everything "the rail is home in the groove, let go of and still" needs, all in
    the PANEL frame so a panel that slid a few millimetres does not matter.
    `seated_now` is the stage predicate itself."""
    panel_rot = panel_pose.to_transformation_matrix()[..., :3, :3]
    rail_rot = rail_pose.to_transformation_matrix()[..., :3, :3]
    local_pos = (panel_pose.inv() * rail_pose).p
    seat_depth_error = torch.abs(local_pos[:, 2] - RAIL_SEATED_Z_LOCAL)
    stop_error = torch.abs(local_pos[:, 0] - RAIL_SEATED_X_LOCAL)     # distance from 'pushed home against the stop'
    lateral_error = torch.abs(local_pos[:, 1])
    up_align = torch.clamp(torch.sum(rail_rot[..., :3, 2] * panel_rot[..., :3, 2], dim=1), -1.0, 1.0)
    axis_align = torch.clamp(torch.sum(rail_rot[..., :3, 0] * panel_rot[..., :3, 0], dim=1), -1.0, 1.0)
    is_upright = up_align >= RAIL_SEAT_UP_COS
    is_axis_aligned = axis_align >= RAIL_SEAT_AXIS_COS
    is_seated = (
        (seat_depth_error <= SEAT_DEPTH_TOL)
        & (stop_error <= SEAT_X_TOL)
        & (lateral_error <= SEAT_Y_TOL)
        & is_upright
        & is_axis_aligned
    )
    rail_linear_speed = torch.linalg.norm(root_linear_velocity, dim=1)
    rail_angular_speed = torch.linalg.norm(root_angular_velocity, dim=1)
    is_released = contact_count <= 0.0
    is_static = (rail_linear_speed <= RAIL_SEAT_LIN_VEL_TOL) & (rail_angular_speed <= RAIL_SEAT_ANG_VEL_TOL)
    return {
        "rail_local_pos": local_pos,
        "seat_depth_error": seat_depth_error,
        "stop_error": stop_error,
        "lateral_error": lateral_error,
        "up_align": up_align,
        "axis_align": axis_align,
        "contact_count": contact_count,
        "rail_linear_speed": rail_linear_speed,
        "rail_angular_speed": rail_angular_speed,
        "is_upright": is_upright,
        "is_axis_aligned": is_axis_aligned,
        "is_seated": is_seated,
        "is_released": is_released,
        "is_static": is_static,
        "seated_now": is_seated & is_released & is_static,
    }



# ===========================================================================
# Asset builders
# ===========================================================================
def build_slider_rail(
    scene: ManiSkillScene,
    name: str = "slider_rail",
    static_friction: float = None,
    dynamic_friction: float = None,
    visual_detail: bool = True,
    kinematic: bool = False,
    show_screw_bosses: bool = True,
):
    """Steel drawer slide member: C-section with two locating pins on the underside.

    Friction is steel-on-wood (~0.3), not the inflated values the first prototype used. The
    open channel is real collision geometry, not just a visual, because subtask 3 needs the
    screwdriver tip to reach the web floor between the flanges.
    """
    art = scene.create_articulation_builder()
    art.disable_self_collisions = True
    builder = art.create_link_builder()          # the rail body is the root link
    builder.set_name("rail_body")
    # Task layer: light-duty aluminium drawer slide -> the shared table's "metal" class.
    phys = (
        object_material("metal")
        if static_friction is None
        else physx.PhysxMaterial(static_friction, dynamic_friction, 0.0)
    )
    steel = make_render_material([0.66, 0.68, 0.72, 1.0], roughness=0.28, metallic=0.92)
    dark = make_render_material([0.42, 0.44, 0.48, 1.0], roughness=0.42, metallic=0.85)
    screw = make_render_material([0.96, 0.82, 0.30, 1.0], roughness=0.55, metallic=0.25)   # matte gold: reads gold, not a mirror of the panel

    hx, hy, hz = RAIL_HALF
    t = RAIL_SHEET_T / 2.0

    # Web (channel floor).
    web_pose = sapien.Pose(p=[0.0, 0.0, -hz + t])
    builder.add_box_collision(
        pose=web_pose, half_size=[hx, hy, t], material=phys, density=RAIL_DENSITY
    )
    builder.add_box_visual(pose=web_pose, half_size=[hx, hy, t], material=steel)

    # Two upstanding flanges.
    for sgn in (-1.0, 1.0):
        flange_pose = sapien.Pose(p=[0.0, sgn * (hy - t), 0.0])
        builder.add_box_collision(
            pose=flange_pose, half_size=[hx, t, hz], material=phys, density=RAIL_DENSITY
        )
        builder.add_box_visual(pose=flange_pose, half_size=[hx, t, hz], material=steel)

    # Locating pins, 3.5 mm proud. Their protrusion is what quantises the seated height and
    # makes the success test unambiguous; see the module docstring.
        # Screw bosses on the web floor: subtask 3's target.
        if show_screw_bosses:
            for sx, sy, sz in RAIL_SCREW_LOCAL_POSITIONS[1:]:
                cylinder_q = [
                    float(np.cos(-np.pi / 4)), 0.0,
                    float(np.sin(-np.pi / 4)), 0.0,
                ]
                builder.add_cylinder_visual(
                    pose=sapien.Pose(p=[sx, sy, sz + 0.002], q=cylinder_q),
                    radius=0.006, half_length=0.002, material=screw,
                )
                builder.add_cylinder_collision(
                    pose=sapien.Pose(p=[sx, sy, sz + 0.002], q=cylinder_q),
                    radius=0.006, half_length=0.002, material=phys, density=7800.0,
                )
                for yaw in (0.0, np.pi / 2.0):
                    builder.add_box_visual(
                        pose=sapien.Pose(
                            p=[sx, sy, sz + 0.00415],
                            q=[float(np.cos(yaw / 2)), 0.0, 0.0,
                               float(np.sin(yaw / 2))],
                        ),
                        half_size=[0.0045, 0.00075, 0.00018], material=dark,
                    )
        # Rolled-over lip along the top of each flange.
        for sgn in (-1.0, 1.0):
            builder.add_box_visual(
                pose=sapien.Pose(p=[0.0, sgn * (hy - t), hz + 0.0003]),
                half_size=[hx, t * 1.4, 0.0004], material=dark,
            )
        # Stamped ventilation slots down the web.
        for i in range(-4, 5):
            builder.add_box_visual(
                pose=sapien.Pose(p=[i * 0.021, 0.0, -hz + 2.0 * t + 0.0002]),
                half_size=[0.0055, 0.0035, 0.0002], material=dark,
            )

    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.5])
    _add_screw_head_link(art, builder, phys, screw)
    art.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.5])
    built = art.build(name=name, fix_root_link=bool(kinematic))
    apply_screw_thread_resistance(built)
    return built


def apply_screw_thread_resistance(rail_articulation) -> None:
    """Coulomb thread torque on the screw joint (see SCREW_THREAD_TORQUE)."""
    joints = rail_articulation.active_joints_map if hasattr(rail_articulation, "active_joints_map") else {}
    joint = joints.get("screw_joint") if joints else None
    targets = [joint] if joint is not None else []
    if not targets:
        for raw in getattr(rail_articulation, "_objs", []):
            targets += [j for j in raw.get_active_joints() if j.name.endswith("screw_joint")]
    for j in targets:
        j.set_drive_properties(stiffness=0.0, damping=SCREW_THREAD_DAMPING,
                               force_limit=SCREW_THREAD_TORQUE, mode="force")
        j.set_drive_velocity_target(0.0)


def _add_screw_head_link(art, rail_link, phys, screw_render_material):
    """Revolute screw head on the rail web at RAIL_SCREW_LOCAL_POSITIONS[0].

    A recess cannot be subtracted from a convex primitive, so the head is a floor
    disc plus four quadrant wings that leave the cross channel EMPTY; the wings
    are inscribed in the head circle. A driver blade in the channel pushes on the
    wing walls, and that contact is the only torque path to the joint.
    """
    dark = make_render_material([0.08, 0.09, 0.10, 1.0], roughness=0.55, metallic=0.55)
    head = art.create_link_builder(rail_link)
    head.set_name("screw_head")
    head.set_joint_name("screw_joint")
    cylinder_q = [float(np.cos(-np.pi / 4)), 0.0, float(np.sin(-np.pi / 4)), 0.0]     # cylinder primitives: axis x -> +z
    # the revolute axis points DOWN (-z), so a positive joint angle is
    # CLOCKWISE seen from above = tightening, like a real screw. (Before, positive was
    # counter-clockwise; the operator turned it 'the wrong way' to the stop.)
    screw_axis_q = [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0]        # joint x -> -z
    floor_half_h = SCREW_HEAD_HALF_H - SCREW_SLOT_DEPTH / 2.0
    floor_z = -SCREW_HEAD_HALF_H + floor_half_h
    wing_half_h = SCREW_SLOT_DEPTH / 2.0
    wing_z = SCREW_HEAD_HALF_H - wing_half_h
    inner = SCREW_SLOT_WIDTH / 2.0
    outer = SCREW_HEAD_RADIUS / float(np.sqrt(2.0))
    wing_half = (outer - inner) / 2.0
    wing_c = inner + wing_half
    floor_pose = sapien.Pose(p=[0.0, 0.0, floor_z], q=cylinder_q)
    head.add_cylinder_collision(pose=floor_pose, radius=SCREW_HEAD_RADIUS, half_length=floor_half_h,
                                material=phys, density=7800.0)
    head.add_cylinder_visual(pose=floor_pose, radius=SCREW_HEAD_RADIUS, half_length=floor_half_h, material=dark)
    # Visual split ("I cannot see the cross to aim at it"): the recess
    # is only 3.4 mm wide and 5 mm deep, so from the operator's 35-60 deg view the
    # dark floor is hidden behind the wing walls and a one-colour wing makes the
    # cross vanish. Wing BODIES are painted recess-dark and only a thin cap on
    # each wing top carries the bright head colour, so the top view reads as four
    # bright quadrants around a dark cross at any elevation. Collision unchanged.
    cap_half_h = SCREW_HEAD_CAP_HALF_H
    lead = SCREW_SLOT_LEAD_IN
    # Each wing = lower block (full size, floor .. top-lead) + upper block inset by
    # `lead` on both channel-facing sides + two 45-deg chamfer plates bridging the
    # inset, so every channel mouth is a V that funnels the blade. All convex.
    lower_half_h = (2.0 * wing_half_h - lead) / 2.0
    lower_z = wing_z - wing_half_h + lower_half_h
    upper_half_h = lead / 2.0
    upper_z = wing_z + wing_half_h - upper_half_h
    plate_half_len = lead / np.sqrt(2.0) + 0.0002      # chamfer segment is lead*sqrt2 long
    plate_half_t = 0.0004
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            lower_pose = sapien.Pose(p=[sx * wing_c, sy * wing_c, lower_z])
            head.add_box_collision(pose=lower_pose, half_size=[wing_half, wing_half, lower_half_h],
                                   material=phys, density=7800.0)
            head.add_box_visual(pose=lower_pose, half_size=[wing_half, wing_half, lower_half_h], material=dark)
            u_half = wing_half - lead / 2.0
            upper_pose = sapien.Pose(p=[sx * (wing_c + lead / 2.0), sy * (wing_c + lead / 2.0), upper_z])
            head.add_box_collision(pose=upper_pose, half_size=[u_half, u_half, upper_half_h],
                                   material=phys, density=7800.0)
            head.add_box_visual(pose=upper_pose, half_size=[u_half, u_half, upper_half_h - cap_half_h], material=dark)
            head.add_box_visual(pose=sapien.Pose(p=[sx * (wing_c + lead / 2.0), sy * (wing_c + lead / 2.0), upper_z + upper_half_h - cap_half_h]),
                                half_size=[u_half, u_half, cap_half_h], material=screw_render_material)
            # chamfer plate on the wall facing the y-channel (normal ~ (-sx, 0, +1)/sqrt2) and the x-channel
            for axis in ("x", "y"):
                if axis == "x":   # wall at x = sx*inner, chamfer runs from (inner, top-lead) up/out to (inner+lead, top)
                    # the plate's CHANNEL-FACING face must lie on the chamfer plane: shift it into the
                    # wing by its half thickness along the plane normal (-sx,0,1)/sqrt2, otherwise its
                    # lower corner protrudes 0.6 mm into the 4 mm channel and the blade jams 3 mm down
                    n_shift = plate_half_t / np.sqrt(2.0)
                    centre = [sx * (inner + lead / 2.0 + n_shift), sy * 0.5 * (inner - 0.0003 + outer), wing_z + wing_half_h - lead / 2.0 - n_shift]
                    half = [plate_half_len, 0.5 * (outer - inner + 0.0003), plate_half_t]   # runs into the wing corner so no ledge is left at 45 deg
                    q = [float(np.cos(-sx * np.pi / 8.0)), 0.0, float(np.sin(-sx * np.pi / 8.0)), 0.0]   # about y by -sx*45 deg: plate normal (-sx,0,1)/sqrt2
                else:
                    n_shift = plate_half_t / np.sqrt(2.0)
                    centre = [sx * 0.5 * (inner - 0.0003 + outer), sy * (inner + lead / 2.0 + n_shift), wing_z + wing_half_h - lead / 2.0 - n_shift]
                    half = [0.5 * (outer - inner + 0.0003), plate_half_len, plate_half_t]
                    q = [float(np.cos(sy * np.pi / 8.0)), float(np.sin(sy * np.pi / 8.0)), 0.0, 0.0]   # about x by sy*45 deg: plate normal (0,-sy,1)/sqrt2
                plate_pose = sapien.Pose(p=centre, q=q)
                head.add_box_collision(pose=plate_pose, half_size=half, material=phys, density=7800.0)
                head.add_box_visual(pose=plate_pose, half_size=half, material=dark)
    sx, sy, sz = RAIL_SCREW_LOCAL_POSITIONS[0]
    head.set_joint_properties(
        "revolute",
        limits=[[0.0, SCREW_JOINT_LIMIT]],
        pose_in_parent=sapien.Pose(p=[sx, sy, sz + SCREW_HEAD_HALF_H], q=screw_axis_q),
        pose_in_child=sapien.Pose(q=screw_axis_q),
        friction=SCREW_JOINT_FRICTION,
        damping=SCREW_JOINT_DAMPING,
    )
    return head


RECESS_ENGAGE_DEPTH = 0.0015 # blade tip at least this far below the head top (was 3.5 mm -- the operator turned the screw to its stop while the blade sat ~3 mm deep and never got the latch)
RECESS_ENGAGE_LATERAL = 0.0040   # and within this of the screw axis


def blade_in_recess(tip_world, head_pose):
    """(in_recess, depth, lateral) of a batched blade-tip position w.r.t. the screw head frame."""
    rel = (head_pose.inv() * Pose.create_from_pq(p=tip_world)).p
    depth = SCREW_HEAD_HALF_H - rel[:, 2]
    lateral = torch.linalg.norm(rel[:, :2], dim=1)
    return (depth >= RECESS_ENGAGE_DEPTH) & (lateral <= RECESS_ENGAGE_LATERAL), depth, lateral


class ScrewEngagementLock:
    """Hard blade-in-recess constraint (user request: "the tip wanders
    out of the cross after a few degrees -- make the contact constraint firm").

    While the Phillips blade tip is inside the cross recess, a WORLD-anchored PhysX
    D6 drive on the screwdriver locks it laterally (+-0.5 mm) and in tilt (+-7 deg)
    to the screw axis. Roll about the axis and axial motion stay free, so the
    torque path is unchanged (blade faces pushing the wing walls) and the tool can
    be pulled straight out; pulling the tip 2 mm above the head top releases the
    lock. It only ever toggles a constraint -- it writes no pose, no velocity, no
    joint state (space-time continuity rule).

    Why world-anchored: a drive whose parent is an articulation link freezes the
    child actor in SAPIEN 3.0.3 (measured: 0 deg roll, tool pinned), so the anchor
    is the head's world pose captured at engagement. The rail is fixed (subtask)
    or seated on its pins (long task), so the head axis does not move meanwhile.

    CPU-only: PhysX drives are not created on the GPU backend; the lock is a no-op
    there and the recess works on contact alone.
    """

    ENGAGE_LATERAL = 0.0015          # tip within 1.5 mm of the screw axis
    ENGAGE_MIN_DEPTH = RECESS_ENGAGE_DEPTH   # past the chamfer mouth: a blade resting in the V must not lock
    ENGAGE_TILT_COS = float(np.cos(np.radians(10.0)))
    LATERAL_PLAY = 0.0005
    TILT_PLAY = 0.05                 # rad, swing cone (+-3 deg): 6.5 mm blade x tan 3 deg = 0.34 mm < 0.5 mm slot play, so a tilted blade cannot wedge and lift a loose rail
    RELEASE_ABOVE_TOP = 0.002

    # frame0 (world anchor) = head pose * R_y(+90 deg): x points DOWN the screw axis
    HEAD_LOCAL = sapien.Pose(q=[float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0])

    def __init__(self, scene, tool_actor, tip_local, head_half_h: float, enabled: bool = True):
        self.active = bool(enabled) and not bool(getattr(scene, "gpu_sim_enabled", False))
        self.head_half_h = float(head_half_h)
        self.engaged = []
        self.drives = []
        if not self.active:
            return
        tip = [float(v) for v in tip_local]
        # frame1 (tool) at the blade tip, x = -x_tool = the blade axis pointing into the head
        tool_local = sapien.Pose(p=tip, q=[0.0, 0.0, 1.0, 0.0])
        for i, sub_scene in enumerate(scene.sub_scenes):
            comp = tool_actor._objs[i].find_component_by_type(physx.PhysxRigidDynamicComponent)
            drive = sub_scene.create_drive(None, sapien.Pose(), comp, tool_local)
            # axial: from 2.5 mm above the head top (release band) down to the recess floor
            drive.set_limit_x(-(self.head_half_h + self.RELEASE_ABOVE_TOP + 0.0005), -0.0005, 0.0, 0.0)
            drive.set_limit_y(-self.LATERAL_PLAY, self.LATERAL_PLAY, 0.0, 0.0)
            drive.set_limit_z(-self.LATERAL_PLAY, self.LATERAL_PLAY, 0.0, 0.0)
            drive.set_limit_cone(self.TILT_PLAY, self.TILT_PLAY, 0.0, 0.0)   # twist (roll) is left free
            drive.disable()
            self.drives.append(drive)
            self.engaged.append(False)

    def reset(self, env_idx=None):
        if not self.active:
            return
        idx = range(len(self.drives)) if env_idx is None else [int(i) for i in env_idx]
        for i in idx:
            self.drives[i].disable()
            self.engaged[i] = False

    def update(self, head_pose, tool_pose):
        """Call once per control step (before the physics substeps)."""
        if not self.active:
            return
        hp = head_pose.raw_pose.detach().cpu().numpy()
        tp = tool_pose.raw_pose.detach().cpu().numpy()
        for i, drive in enumerate(self.drives):
            head = sapien.Pose(p=hp[i, :3].tolist(), q=hp[i, 3:].tolist()) * self.HEAD_LOCAL
            tool = sapien.Pose(p=tp[i, :3].tolist(), q=tp[i, 3:].tolist()) * drive.pose_in_child
            rel = head.inv() * tool
            axial = float(rel.p[0])                                # -(tip height above head centre)
            lateral = float(np.hypot(rel.p[1], rel.p[2]))
            rel_x_axis = rel.to_transformation_matrix()[:3, 0]
            tilt_cos = float(rel_x_axis[0])
            if not self.engaged[i]:
                inside = (
                    lateral <= self.ENGAGE_LATERAL
                    and -(self.head_half_h - self.ENGAGE_MIN_DEPTH) <= axial <= 0.0
                    and tilt_cos >= self.ENGAGE_TILT_COS
                )
                if inside:
                    drive.set_pose_in_parent(head)
                    drive.enable()
                    self.engaged[i] = True
            else:
                # Release is judged against the ANCHOR, not the live head: a free
                # rail (long task) can be lifted a few mm by a wedged blade, and
                # measuring against the lifted head would deadlock the tool at
                # its axial limit with the lock still on.
                anchored = drive.pose_in_parent.inv() * tool
                if float(anchored.p[0]) < -(self.head_half_h + self.RELEASE_ABOVE_TOP):
                    drive.disable()
                    self.engaged[i] = False


class RailSeatClip:
    """Snap-fit for the seated rail (long task).

    Real drawer slides click into their dado; ours is only form-closed sideways by
    the two pins, so a Phillips blade twisted against the wing walls can lift the
    97 g rail straight out of the groove when the driver is pulled back (measured:
    rail hanging on the blade 30 mm up). Once the rail is judged seated, a
    world-anchored D6 drive on the rail root pins it within +-0.3 mm / +-1 deg of
    the seated pose. The screw joint stays free (the root link is the drive's
    CHILD; a link as PARENT freezes the other body -- see ScrewEngagementLock).
    Toggles a constraint only; writes no state. CPU-only (no-op on GPU).
    """

    LINEAR_PLAY = 0.0003
    ANGULAR_PLAY = 0.0175   # rad, ~1 deg

    def __init__(self, scene, rail_articulation, enabled: bool = True):
        self.active = bool(enabled) and not bool(getattr(scene, "gpu_sim_enabled", False))
        self.drives = []
        self.engaged = []
        if not self.active:
            return
        for i, sub_scene in enumerate(scene.sub_scenes):
            root = rail_articulation._objs[i].get_links()[0]
            comp = root.entity.find_component_by_type(physx.PhysxArticulationLinkComponent)
            drive = sub_scene.create_drive(None, sapien.Pose(), comp, sapien.Pose())
            for axis in ("x", "y", "z"):
                getattr(drive, f"set_limit_{axis}")(-self.LINEAR_PLAY, self.LINEAR_PLAY, 0.0, 0.0)
            drive.set_limit_cone(self.ANGULAR_PLAY, self.ANGULAR_PLAY, 0.0, 0.0)
            drive.set_limit_twist(-self.ANGULAR_PLAY, self.ANGULAR_PLAY, 0.0, 0.0)
            drive.disable()
            self.drives.append(drive)
            self.engaged.append(False)

    def engage(self, rail_pose, env_idx=None):
        if not self.active:
            return
        rp = rail_pose.raw_pose.detach().cpu().numpy()
        idx = range(len(self.drives)) if env_idx is None else [int(i) for i in env_idx]
        for i in idx:
            if self.engaged[i]:
                continue
            self.drives[i].set_pose_in_parent(sapien.Pose(p=rp[i, :3].tolist(), q=rp[i, 3:].tolist()))
            self.drives[i].enable()
            self.engaged[i] = True

    def release(self, env_idx=None):
        if not self.active:
            return
        idx = range(len(self.drives)) if env_idx is None else [int(i) for i in env_idx]
        for i in idx:
            self.drives[i].disable()
            self.engaged[i] = False


# ===========================================================================
# Environment
# ===========================================================================
@register_task(
    "FurnitureAssemblyInsertRail",
    side="right",
    max_episode_steps=700,
    obs_mode="state",
    control_mode=DEFAULT_CONTROL_MODE,
    render_mode="rgb_array",
)
class FurnitureAssemblyInsertRailEnv(DeclarativeTaskStateMixin, BaseEnv):
    # Task bookkeeping that physics cannot re-derive. Restoring a
    # mid-episode frame without these silently rewinds every milestone,
    # which breaks offline re-render, demo-init and end-frame QC.
    # APPEND ONLY -- reordering breaks older recordings.
    TASK_STATE_KEY = "furniture_rail_task_state"
    TASK_STATE_FIELDS = (
        TaskStateField("grasp_achieved", 1, torch.bool),
        TaskStateField("lift_achieved", 1, torch.bool),
        TaskStateField("align_achieved", 1, torch.bool),
        TaskStateField("settle_counter", 1, torch.int32),
    )
    """Slide a steel slider rail into the drawer panel's groove through its mouth until it meets the stop."""

    SUPPORTED_ROBOTS = sorted(MOBILE_ROBOT_UIDS)
    agent: Union[
        PandaGripperRightMobileBase,
        PandaSharpaHandRightAngledMobileBase,
        PandaWujiHandRightMobileBase,
    ]

    # Task layer: the family's single-sourced layout (see
    # furniture_panel_lay_flat.FURNITURE layout block). The panel sits where the
    # long task leaves it after stage 1; the rail spawns on the long task's own
    # rail lane.
    PANEL_SPAWN_CENTER = PLACE_CENTER
    RAIL_SPAWN_CENTER = RAIL_PICK_CENTER

    # Success thresholds.
    # Success thresholds live in rail_seat_quantities (shared with the long task);
    # these aliases only feed the dense reward.
    SEAT_DEPTH_TOL = SEAT_DEPTH_TOL  # module constant; see the assert next to the ladder
    RAIL_UP_COS = RAIL_SEAT_UP_COS
    AXIS_ALIGN_COS = RAIL_SEAT_AXIS_COS
    LIFT_CLEAR_HEIGHT = 0.030        # above the table before the rail counts as picked up
    LIN_VEL_THRESHOLD = RAIL_SEAT_LIN_VEL_TOL
    ANG_VEL_THRESHOLD = RAIL_SEAT_ANG_VEL_TOL
    SETTLE_STEPS = RAIL_SEAT_SETTLE_STEPS
    TIP_CONTACT_FORCE = 0.6

    # Reward weights.
    W_REACH, W_GRASP, W_LIFT, W_ALIGN, W_SEAT = 1.0, 1.0, 1.0, 3.0, 4.0
    SUCCESS_REWARD = 10.0
    LIVING_PENALTY = 0.01
    ARM_QVEL_PENALTY = 5e-4
    DROP_PENALTY = 2.0

    external_camera_eye = [0.48, -0.61, 0.50]
    external_camera_target = [-0.03, -0.04, 0.06]

    # Borrowed verbatim from subtask 1 so the two tasks cannot drift apart.
    _reset_agent = FurnitureAssemblyLayFlatPanelEnv._reset_agent
    _tip_links = FurnitureAssemblyLayFlatPanelEnv._tip_links
    _tip_positions = FurnitureAssemblyLayFlatPanelEnv._tip_positions
    _gripper_opening = FurnitureAssemblyLayFlatPanelEnv._gripper_opening
    _load_lighting = FurnitureAssemblyLayFlatPanelEnv._load_lighting

    @property
    def _default_sim_config(self) -> SimConfig:
        # Environment layer: the locked fleet profile. The thinnest feature here
        # is the 2.5 mm rail sheet and the tightest gap the 1.5 mm groove
        # clearance; a private 0.6 mm offset used to resolve those a little
        # finer, but a per-task contact offset breaks the one thing this
        # profile exists to guarantee -- identical transition dynamics across
        # every task, so sub-task policies compose into the long task.
        return make_sim_config(
            max_rigid_contact_count=2**21,
            max_rigid_patch_count=2**19,
        )

    def __init__(
        self,
        *args,
        robot_uids: str = GRIPPER_ROBOT_UID,
        robot_init_qpos_noise: float = 0.01,
        rail_random_x: float = 0.045,
        rail_random_y: float = 0.035,
        # was +/-180 deg. The long task spawns the rail at yaw 0 on
        # this lane; a random yaw here is outside that input distribution AND
        # at +/-148 deg the 380 mm rail overlapped the panel at reset, so the
        # solver threw it 5 cm on the first step (measured settle 52.8 mm).
        rail_random_yaw: float = 0.0,
        panel_color_jitter: bool = True,
        visual_detail: bool = True,
        terminate_on_success: bool = False,
        render_camera_eye=(0.68, -1.71, 0.92),
        render_camera_target=(-0.45, -0.04, 0.36),
        render_camera_fov: float = 1.0,
        render_width: int = 720,
        render_height: int = 720,
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.rail_random_x = float(rail_random_x)
        self.rail_random_y = float(rail_random_y)
        self.rail_random_yaw = float(rail_random_yaw)
        self.panel_color_jitter = bool(panel_color_jitter)
        self.visual_detail = bool(visual_detail)
        self.terminate_on_success = bool(terminate_on_success)
        self.render_camera_eye = list(render_camera_eye)
        self.render_camera_target = list(render_camera_target)
        self.render_camera_fov = float(render_camera_fov)
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        self.show_guidance_markers = pop_guidance_marker_flag(kwargs)
        super().__init__(
            *args, robot_uids=robot_uids, num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq, **kwargs,
        )

    @property
    def panel_rest_z(self) -> float:
        """Panel centre height resting flat on the table."""
        return PANEL_HALF[2]

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
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
                eye=self.render_camera_eye, target=self.render_camera_target
            ),
            width=self.render_width, height=self.render_height,
            fov=self.render_camera_fov, near=0.01, far=100.0,
        )

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
            self.scene, base_color=base_color, visual_detail=self.visual_detail,
            kinematic=True,
        )
        # the long task builds its rail with show_screw_bosses=False (the
        # articulated screw is the visible one). A subtask must be a subset of the long
        # task, so the rail here is the same body: same shapes, same 281 g.
        # The long task's panel lies in the assembly zone marker; the subtask shows the
        # same marker under the same panel (user: strict subset).
        self.place_marker = build_place_marker(self.scene, PLACE_REGION_HALF_SIZE)
        self.rail = build_slider_rail(self.scene, visual_detail=self.visual_detail,
                                      show_screw_bosses=False)
        set_task_object_energy_fence(self.rail)
        self.rail_body = self.rail.links_map["rail_body"]
        self.screw_joint = self.rail.active_joints_map["screw_joint"]
        self.screw_head = self.rail.links_map["screw_head"]

        self.rail_grasp_local = common.to_tensor(RAIL_GRASP_LOCAL, device=self.device).float()
        self.groove_landmarks_local = common.to_tensor(GROOVE_LANDMARKS_LOCAL, device=self.device).float()
        self.rail_ends_local = common.to_tensor(((-RAIL_HALF[0], 0.0, 0.0), (RAIL_HALF[0], 0.0, 0.0)), device=self.device).float()

        self.grasp_achieved = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.lift_achieved = torch.zeros_like(self.grasp_achieved)
        self.align_achieved = torch.zeros_like(self.grasp_achieved)
        self.settle_counter = torch.zeros(
            (self.num_envs,), dtype=torch.int32, device=self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            set_work_table_pose(self)
            self._reset_agent(b)

            def rand(scale: float) -> torch.Tensor:
                return (torch.rand((b,), device=self.device) * 2.0 - 1.0) * float(scale)

            # --- fixed template panel: flat, groove up at subtask 1's target centre ---
            panel_pos = torch.zeros((b, 3), device=self.device)
            panel_pos[:, 0] = self.PANEL_SPAWN_CENTER[0]
            panel_pos[:, 1] = self.PANEL_SPAWN_CENTER[1]
            panel_pos[:, 2] = self.panel_rest_z
            panel_yaw = torch.zeros((b,), device=self.device)
            panel_quat = yaw_to_quat_wxyz(panel_yaw)
            self.panel.set_pose(Pose.create_from_pq(p=panel_pos, q=panel_quat))
            marker_p = torch.tensor([[self.PANEL_SPAWN_CENTER[0], self.PANEL_SPAWN_CENTER[1], 0.0]], device=self.device, dtype=torch.float32).expand(b, -1)
            self.place_marker.set_pose(Pose.create_from_pq(p=marker_p, q=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(b, -1)))
            # --- rail resting directly on the table, pins down, channel up ---
            rail_pos = torch.zeros((b, 3), device=self.device)
            rail_pos[:, 0] = self.RAIL_SPAWN_CENTER[0] + rand(self.rail_random_x)
            rail_pos[:, 1] = self.RAIL_SPAWN_CENTER[1] + rand(self.rail_random_y)
            rail_pos[:, 2] = RAIL_REST_Z
            rail_yaw = rand(self.rail_random_yaw)
            self.rail.set_pose(Pose.create_from_pq(p=rail_pos, q=yaw_to_quat_wxyz(rail_yaw)))
            self.rail.set_root_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.rail.set_root_angular_velocity(torch.zeros((b, 3), device=self.device))
            self.rail.set_qpos(torch.zeros((b, 1), device=self.device))
            self.rail.set_qvel(torch.zeros((b, 1), device=self.device))

            self.grasp_achieved[env_idx] = False
            self.lift_achieved[env_idx] = False
            self.align_achieved[env_idx] = False
            self.settle_counter[env_idx] = 0

    # ------------------------------------------------------------------
    # Geometry accessors
    # ------------------------------------------------------------------
    @property
    def rail_grasp_position(self) -> torch.Tensor:
        offset = Pose.create_from_pq(p=self.rail_grasp_local.expand(self.num_envs, -1))
        return (self.rail.pose * offset).p

    @property
    def groove_landmarks(self) -> torch.Tensor:
        """(N, 2, 3) world positions of the groove mouth centre and the stop face."""
        return torch.stack(
            [(self.panel.pose * Pose.create_from_pq(p=self.groove_landmarks_local[i].expand(self.num_envs, -1))).p for i in range(2)],
            dim=1,
        )

    @property
    def rail_ends(self) -> torch.Tensor:
        """(N, 2, 3) world positions of the two rail ends (-x end enters first, +x end meets the stop)."""
        return torch.stack(
            [(self.rail.pose * Pose.create_from_pq(p=self.rail_ends_local[i].expand(self.num_envs, -1))).p for i in range(2)],
            dim=1,
        )

    @property
    def seated_pose(self) -> Pose:
        """Where the rail should end up, expressed in world coordinates."""
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 0] = RAIL_SEATED_X_LOCAL
        offset[:, 2] = RAIL_SEATED_Z_LOCAL
        return self.panel.pose * Pose.create_from_pq(p=offset)

    def _contact_count(self, actor) -> torch.Tensor:
        """Distinct fingers in force contact with `actor` (collision-bearing links, see finger_contact_count)."""
        return finger_contact_count(self.scene, self.agent, actor, self.TIP_CONTACT_FORCE)

    # ------------------------------------------------------------------
    # Single source of truth
    # ------------------------------------------------------------------
    def _task_quantities(self) -> dict:
        panel_rot = self.panel.pose.to_transformation_matrix()[..., :3, :3]
        panel_normal, groove_axis = panel_rot[..., :3, 2], panel_rot[..., :3, 0]
        rail_rot = self.rail.pose.to_transformation_matrix()[..., :3, :3]
        rail_up, rail_axis = rail_rot[..., :3, 2], rail_rot[..., :3, 0]

        # Everything that defines success is measured in the panel frame, so the panel sliding
        # a few millimetres does not invalidate a correct insertion.
        seat = rail_seat_quantities(
            self.rail.pose, self.panel.pose, self._contact_count(self.rail_body),
            self.rail.root_linear_velocity, self.rail.root_angular_velocity)
        local_pos = seat["rail_local_pos"]

        grasp_point = self.rail_grasp_position
        tips = self._tip_positions()
        reach_dist = torch.minimum(
            torch.linalg.norm(self.agent.tcp.pose.p - grasp_point, dim=1),
            torch.linalg.norm(tips - grasp_point[:, None, :], dim=2).min(dim=1).values,
        )

        seated = self.seated_pose.p
        return {
            **seat,
            "rail_pos": self.rail.pose.p,
            "panel_pos": self.panel.pose.p,
            "rail_up": rail_up,
            "rail_axis": rail_axis,
            "panel_normal": panel_normal,
            "groove_axis": groove_axis,
            "rail_to_seated": seated - self.rail.pose.p,
            "seated_dist": torch.linalg.norm(seated - self.rail.pose.p, dim=1),
            "grasp_point": grasp_point,
            "reach_dist": reach_dist,
            "panel_contact_count": self._contact_count(self.panel),
            "gripper_opening": self._gripper_opening(),
            "rail_height_above_table": self.rail.pose.p[:, 2] - RAIL_REST_Z,
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
                    is_grasped = is_grasped | native(self.rail_body).to(torch.bool)
                except Exception:
                    pass
        is_released = q["is_released"]
        is_lifted = is_grasped & (q["rail_height_above_table"] > self.LIFT_CLEAR_HEIGHT)
        is_upright = q["is_upright"]
        is_aligned = is_upright & q["is_axis_aligned"]
        # is_seated is the shared predicate and now carries the axis
        # term too -- a rail slid in screw-end first no longer passes here.
        is_seated = q["is_seated"]
        is_static = q["is_static"]

        settled_now = q["seated_now"]
        self.settle_counter = torch.where(
            settled_now, self.settle_counter + 1, torch.zeros_like(self.settle_counter)
        )
        success = self.settle_counter >= self.SETTLE_STEPS

        # Reward-only latches.
        self.grasp_achieved |= is_grasped
        self.lift_achieved |= is_lifted
        self.align_achieved |= is_lifted & is_aligned

        return {
            "success": success,
            "termination_success": success if self.terminate_on_success
            else torch.zeros_like(success),
            "settle_progress": torch.clamp(
                self.settle_counter.float() / float(self.SETTLE_STEPS), 0.0, 1.0
            ),
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
            "is_upright": is_upright,
            "is_aligned": is_aligned,
            "is_seated": is_seated,
            "is_released": is_released,
            "is_static": is_static,
            "grasp_achieved": self.grasp_achieved,
            "lift_achieved": self.lift_achieved,
            "align_achieved": self.align_achieved,
            "dropped": q["rail_pos"][:, 2] < 0.002,
            "reach_dist": q["reach_dist"],
            "contact_count": q["contact_count"],
            "panel_contact_count": q["panel_contact_count"],
            "gripper_opening": q["gripper_opening"],
            "rail_height_above_table": q["rail_height_above_table"],
            "seat_depth_error": q["seat_depth_error"],
            "lateral_error": q["lateral_error"],
            "stop_error": q["stop_error"],
            "axis_align": q["axis_align"],
            "up_align": q["up_align"],
            "seated_dist": q["seated_dist"],
            "rail_to_seated": q["rail_to_seated"],
            "rail_local_pos": q["rail_local_pos"],
            "rail_up": q["rail_up"],
            "rail_axis": q["rail_axis"],
            "panel_normal": q["panel_normal"],
            "groove_axis": q["groove_axis"],
            "grasp_point": q["grasp_point"],
            "rail_pose": self.rail.pose.raw_pose,
            "panel_pose": self.panel.pose.raw_pose,
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "rail_pose": info["rail_pose"],
            "panel_pose": info["panel_pose"],
            "tcp_to_rail_grasp": info["grasp_point"] - self.agent.tcp.pose.p,
            "rail_to_seated": info["rail_to_seated"],
            "seated_pose": self.seated_pose.raw_pose,
        }
        if self.obs_mode_struct.use_state:
            obs.update(
                all_tip_positions=self._tip_positions().reshape(self.num_envs, -1),
                groove_landmarks=self.groove_landmarks.reshape(self.num_envs, -1),
                rail_ends=self.rail_ends.reshape(self.num_envs, -1),
                groove_axis=info["groove_axis"],
                panel_normal=info["panel_normal"],
                rail_axis=info["rail_axis"],
                rail_up=info["rail_up"],
                rail_local_pos=info["rail_local_pos"],
                rail_linear_velocity=self.rail.root_linear_velocity,
                rail_angular_velocity=self.rail.root_angular_velocity,
                contact_count=info["contact_count"].unsqueeze(1),
                gripper_opening=info["gripper_opening"].unsqueeze(1),
                seat_depth_error=info["seat_depth_error"].unsqueeze(1),
                stop_error=info["stop_error"].unsqueeze(1),
                axis_align=info["axis_align"].unsqueeze(1),
                up_align=info["up_align"].unsqueeze(1),
                lateral_error=info["lateral_error"].unsqueeze(1),
                is_grasped=info["is_grasped"].float().unsqueeze(1),
                is_lifted=info["is_lifted"].float().unsqueeze(1),
                is_aligned=info["is_aligned"].float().unsqueeze(1),
                is_seated=info["is_seated"].float().unsqueeze(1),
                is_released=info["is_released"].float().unsqueeze(1),
                is_static=info["is_static"].float().unsqueeze(1),
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
        r_lift = torch.clamp(q["rail_height_above_table"] / self.LIFT_CLEAR_HEIGHT, 0.0, 1.0)
        # Alignment above the groove: attitude first, then the lateral/longitudinal offset.
        r_align = (
            0.35 * torch.clamp((q["up_align"] + 1.0) * 0.5, 0.0, 1.0) ** 2
            + 0.25 * torch.clamp((q["axis_align"] + 1.0) * 0.5, 0.0, 1.0) ** 2
            + 0.40 * (1.0 - torch.tanh(8.0 * torch.linalg.norm(
                q["rail_to_seated"][:, :2], dim=1)))
        )
        # Seating: pin registration dominates, then the depth, then holding still after release.
        r_seat = (
            0.45 * (1.0 - torch.tanh(25.0 * q["stop_error"]))
            + 0.30 * (1.0 - torch.tanh(60.0 * q["seat_depth_error"]))
            + 0.25 * info["settle_progress"]
        )

        # Milestone latches: reward-only, never part of `success`. Without them r_grasp
        # collapses the instant the robot correctly releases a seated rail.
        r_reach = torch.where(info["grasp_achieved"], ones, r_reach)
        r_grasp = torch.where(info["is_seated"], ones, r_grasp)
        r_lift = torch.where(info["lift_achieved"], ones, r_lift)
        r_align = torch.where(info["is_seated"], ones, r_align)

        prefix = r_reach
        term_reach = self.W_REACH * prefix
        prefix = prefix * r_grasp
        term_grasp = self.W_GRASP * prefix
        prefix = prefix * r_lift
        term_lift = self.W_LIFT * prefix
        prefix = prefix * r_align
        term_align = self.W_ALIGN * prefix
        prefix = prefix * r_seat
        term_seat = self.W_SEAT * prefix

        reward = term_reach + term_grasp + term_lift + term_align + term_seat
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
            "r_reach": r_reach, "r_grasp": r_grasp, "r_lift": r_lift,
            "r_align": r_align, "r_seat": r_seat,
            "term_reach": term_reach, "term_grasp": term_grasp, "term_lift": term_lift,
            "term_align": term_align, "term_seat": term_seat,
        }
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs, action, info) / self.SUCCESS_REWARD


# ===========================================================================
# CLI
# ===========================================================================
def tensor_mean(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(np.asarray(value).mean())


def hold_action(env, control_mode: str) -> torch.Tensor:
    """See panel_lay_flat_v1.hold_action: zeros only mean 'hold' for delta control modes."""
    action = torch.zeros(
        env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32
    )
    if "delta" in control_mode:
        return action
    qpos = env.unwrapped.agent.robot.get_qpos()
    dof = min(action.shape[-1], qpos.shape[-1])
    action[..., :dof] = qpos[..., :dof].reshape(action[..., :dof].shape)
    if env.unwrapped.robot_uids == GRIPPER_ROBOT_UID:
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
        render_camera_fov=args.cam_fov,
        render_width=args.render_width,
        render_height=args.render_height,
        **({"render_camera_eye": args.cam_eye} if args.cam_eye else {}),
        **({"render_camera_target": args.cam_target} if args.cam_target else {}),
    )
    if args.record_video or args.save_trajectory:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordEpisode(
            env, output_dir=args.video_dir, save_video=args.record_video,
            save_trajectory=args.save_trajectory,
            trajectory_name="rail_insert_v1_trajectory",
            video_fps=args.video_fps, max_steps_per_video=args.max_steps,
        )

    print(f"env_id={args.env_id}  dof={env.unwrapped.agent.robot.dof}")
    print("obs space", env.observation_space)
    print("act space", env.action_space)

    obs, info = env.reset(seed=args.seed)
    for episode in range(args.episodes):
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
                    "lift={:+.4f} pin_err={:.4f} depth_err={:.4f} axis={:.3f} up={:+.3f}".format(
                        step, tensor_mean(reward), tensor_mean(info["success"]),
                        tensor_mean(info["reach_dist"]), tensor_mean(info["contact_count"]),
                        tensor_mean(info["rail_height_above_table"]),
                        tensor_mean(info["stop_error"]), tensor_mean(info["seat_depth_error"]),
                        tensor_mean(info["axis_align"]), tensor_mean(info["up_align"]),
                    )
                )
            if bool((torch.as_tensor(terminated) | torch.as_tensor(truncated)).all().item()):
                break
        print(
            "episode={} steps={} success={:.2f} grasp={:.2f} lift={:.2f} align={:.2f}".format(
                episode + 1, step, tensor_mean(info["success"]),
                tensor_mean(info["grasp_achieved"]), tensor_mean(info["lift_achieved"]),
                tensor_mean(info["align_achieved"]),
            )
        )
        if episode + 1 < args.episodes:
            obs, info = env.reset()
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RailInsertGroove (DualFurnitureAssembly subtask 2).")
    p.add_argument("--env-id", "--env_id", default="FurnitureAssemblyInsertRail-Gripper-v1", choices=ENV_IDS)
    p.add_argument("--control-mode", "--control_mode", default=DEFAULT_CONTROL_MODE,
                   choices=["pd_joint_delta_pos", "pd_joint_pos", "pd_joint_target_delta_pos",
                            "pd_ee_delta_pose", "pd_ee_delta_pos"])
    p.add_argument("--obs-mode", "--obs_mode", default="state")
    p.add_argument("--reward-mode", "--reward_mode", default="normalized_dense",
                   choices=["dense", "normalized_dense", "sparse"])
    p.add_argument("--sim-backend", "--sim_backend", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--render-mode", "--render_mode", default="rgb_array",
                   choices=["rgb_array", "human"])
    p.add_argument("--num-envs", "--num_envs", type=int, default=1)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", "--max_steps", type=int, default=350)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--terminate-on-success", "--terminate_on_success", action="store_true")
    p.add_argument("--random-actions", "--random_actions", action="store_true")
    p.add_argument("--record-video", "--record_video", action="store_true")
    p.add_argument("--save-trajectory", "--save_trajectory", action="store_true")
    p.add_argument("--video-dir", "--video_dir", default=DEFAULT_VIDEO_DIR)
    p.add_argument("--cam-eye", "--cam_eye", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"))
    p.add_argument("--cam-target", "--cam_target", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"))
    p.add_argument("--cam-fov", "--cam_fov", type=float, default=1.0)
    p.add_argument("--render-width", "--render_width", type=int, default=720)
    p.add_argument("--render-height", "--render_height", type=int, default=720)
    p.add_argument("--video-fps", "--video_fps", type=int, default=20)
    p.add_argument("--no-color-jitter", "--no_color_jitter", action="store_true")
    p.add_argument("--no-visual-detail", "--no_visual_detail", action="store_true")
    p.add_argument("--debug-print", "--debug_print", action="store_true")
    p.add_argument("--debug-interval", "--debug_interval", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

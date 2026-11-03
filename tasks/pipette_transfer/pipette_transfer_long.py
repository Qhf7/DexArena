"""Dual-hand pipette transfer task with procedural laboratory objects.

The full task is the two subtasks in order, with the SAME predicates:
S0 -- the right hand takes the parked pipette, aspirates in the fixed sample cup and
parks it again; S1 -- the left hand takes it again, dispenses into the test tube
standing in the rack and parks it again. The tube is never picked up. The pipette
plunger is a real prismatic URDF joint; no fluid simulation is used.
"""
from __future__ import annotations
import argparse, sys
from dataclasses import dataclass
from pathlib import Path

# Match package_opening: direct execution must make the repository root
# importable before importing tasks.robots or any task module.
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

# Keep task listing lightweight, matching package_opening.py.  Do this before
# importing Gymnasium, SAPIEN, ManiSkill, or robot definitions.
if __name__ == "__main__" and "--list" in sys.argv:
    print(f"{ENV_ID if 'ENV_ID' in globals() else 'PipetteTransferLong-Sharpa-v1'} subtasks (2)")
    print("\n1. aspirate_pipette")
    print("   Right hand takes the parked pipette, aspirates in the fixed sample cup, then parks it in the stand and releases.")
    print("   Success: pipette lifted, tip inside the sample cup, plunger pressed and released, pipette back in the stand, hands away.")
    print("\n2. dispense_in_rack")
    print("   Left hand takes the parked pipette, dispenses into the test tube standing in the rack, then parks it again and releases.")
    print("   Success: a separate second press with the tip inside the racked tube, plunger released, pipette back in the stand, hands away. The tube stays in the rack.")
    raise SystemExit(0)

import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from tasks.config.hands import register_task
from mani_skill.utils.structs import Actor, Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.wrappers.record import RecordEpisode
from tasks.robots.panda_dual_sharpa_hand_mobile_base import PandaDualSharpaHandAngledMobileBase
from tasks.robots.panda_dual_wuji_hand_mobile_base import PandaDualWujiHandMobileBase  # noqa: F401
from tasks.robots.panda_dual_xhand_hand_mobile_base import PandaDualXHandHandMobileBase  # noqa: F401
from tasks.config.layout_randomization import ARM_INIT_QPOS_NOISE_STD, arm_qpos_noise, episode_rng, offset_pose, uniform_offset, layout_rng
from tasks.config.guidance_markers import pop_guidance_marker_flag
from tasks.scene_builders import TableSceneBuilder
from tasks.config.sim_profiles import make_sim_config
from tasks.config.object_materials import object_material
from tasks.config.workspace_rig import (
    ARM_INIT_QPOS,
    PLATFORM_ROOT_X,
    TABLE_CENTER_X,
    set_work_table_pose,
)
from tasks.config.energy_fence import set_task_object_energy_fence
PIPETTE_URDF=ROOT/"tasks/assets/task_object_meshes/pipette/mobility.urdf"
PIPETTE_GRIP_ORIGINAL_LENGTH=0.120
PIPETTE_GRIP_LENGTH=0.160
PIPETTE_GRIP_EXTENSION=PIPETTE_GRIP_LENGTH-PIPETTE_GRIP_ORIGINAL_LENGTH
# Engineering proxy for a spring-return manual pipette button. At the 20 mm
# success threshold this produces 3.2 N restoring force; full travel is 4 N.
PIPETTE_PLUNGER_STIFFNESS=80.0 # N/m (was 160 -> 3.2 N at the 20 mm press; the Sharpa thumb saturates at ~3.3 N, so the press was at the hand's ceiling. 80 N/m x 15 mm = 1.2 N)
PIPETTE_PLUNGER_DAMPING=1.0  # N*s/m
PIPETTE_PLUNGER_FORCE_LIMIT=6.0  # N
PIPETTE_PLUNGER_FRICTION=0.05  # N
# stage, first_press, second_press, was_pressed, primed, dipped.
# `primed` and `dipped` were added: they are the two halves of the
# aspirate milestone, latched independently because a human pipettes dip-then-
# press as readily as press-then-dip, and they live on the evaluator object
# rather than the env -- which is why the state-restore gate could not see them
# until it learned to walk task-owned helper objects. Without them a restored
# mid-episode frame forgets that the tip was ever in the dish.
PIPETTE_TASK_STATE_LEGACY_DIM=4
PIPETTE_TASK_STATE_DIM=6
# both 5 mm windows widened to 8 mm. The tube mouth is 22 mm
# wide and the rack slot clears the tube by 3 mm, so 8 mm still means the
# tip is IN the tube / the tube is IN its slot; 5 mm was the DrawerPlug
# socket-class tolerance on a HAND-HELD target with no force feedback.
# (operator: "I tried both pipette subtasks for ages and success never
# fires"): 8 mm was an authored number, not the geometry. The tube's inner bore is
# tube_r = 11 mm, and the z window already requires the tip to be 30-180 mm above
# the tube's bottom -- at that height the glass wall physically blocks anything
# further out than the bore, so the honest tolerance IS the bore (+1 mm for the
# tip's own 3.5 mm radius reading its axis point).
TIP_IN_TUBE_XY_TOL=0.012
# operator feedback ("motions are fine, success never fires"):
# * "hand on the pipette / tube" was the distance to the URDF ROOT (pipette root =
#   top of the body, tube root = bottom): a hand gripping mid-body sat 0.10-0.15 m
#   from the root, so `pipette_lifted` never latched. Now measured to the body
#   SEGMENT (same rule as the spoon handle).
# * the plunger press counts at 50 % of travel (7.5 mm), not 80 %.
# the tube is never handled any more (the long task follows the S1
# subtask: it stays in the rack), so the hand-on-tube distance is a HUD channel only.
PIPETTE_BODY_SEGMENT_LOCAL=((0.0,0.0,-0.130),(0.0,0.0,0.070))   # gripped body, root frame
TUBE_BODY_SEGMENT_LOCAL=((0.0,0.0,0.0),(0.0,0.0,0.180))
HAND_ON_OBJECT_DIST=0.08
PLUNGER_PRESSED_FRAC=0.50   # 7.5 mm of the 15 mm travel; the HUD prints `plunger` live so the reachable depth is visible
# (operator: 'I can only put the pipette back a little tilted'): the return
# check accepts up to 35 deg of tilt and a lower root, instead of 10 deg / 0.16 m.
PIPETTE_RETURN_TILT_COS=0.819      # cos(35 deg)
PIPETTE_RETURN_Z_MIN=0.08
PIPETTE_RETURN_Z_MAX=0.32
PLUNGER_RELEASED_FRAC=0.20
# Task layer: explicit masses (kg), values in force read back.
TASK_OBJECT_MASS={"test_tube":0.021,"tube_rack":0.133,"sample_cup":0.052,"pipette_stand":0.212,"pipette":0.100}
ENV_ID="PipetteTransferLong-Sharpa-v1"
# Use the standard dual-arm angled variant. It inserts the physical two-piece
# adapter bracket between each Panda flange and Sharpa wrist.
ROBOT_UID="panda_dual_sharpa_hand_angled_mobile_base"
# (user decision): the aspirate subtask S0 is a one-hand job, so it
# runs on the single RIGHT-arm trolley rig, not the dual model. Same trolley x,
# same arm y (-0.220): the right arm of the dual build and the right arm of this
# build are the same arm in the same place, so the S0 geometry is unchanged.
RIGHT_ROBOT_UID="panda_sharpa_hand_right_angled_mobile_base"
RIGHT_ROBOT_UIDS=[RIGHT_ROBOT_UID,"panda_wuji_hand_right_angled_mobile_base","panda_xhand_hand_right_angled_mobile_base","panda_gripper_right_mobile_base"]
LEFT_ROBOT_UID="panda_sharpa_hand_left_angled_mobile_base"
LEFT_ROBOT_UIDS=[LEFT_ROBOT_UID,"panda_wuji_hand_left_angled_mobile_base","panda_xhand_hand_left_angled_mobile_base","panda_gripper_left_mobile_base"]
# Match the dual-arm workspace placement used by cabinet_pour.py.  The robot
# asset owns local kinematics; this task owns the world placement.
# Arm layer: the trolley x is the shared rig's, not a copied literal.
DUAL_ROBOT_BASE_POSE=sapien.Pose(p=[PLATFORM_ROOT_X,0.0,0.0])
TASK_PLAN=(("aspirate_pipette","Right hand grasps the vertically parked pipette, aspirates in the fixed sample cup, then returns and releases it in the long stand"),("dispense_in_rack","Left hand grasps the parked pipette again, dispenses into the test tube standing in the rack, then returns and releases the pipette in the stand; the tube stays in the rack"))
TASK_SUCCESS_CONDITIONS=("right fingertip point within 0.08 m of the pipette BODY; tip anywhere inside the cup; plunger pressed >= 50% (7.5 mm) while there, then released <= 20%; pipette back in the stand (xy < 0.14 m of the stand, root 0.08-0.32 m up, < 35 deg tilt, still) with both hands > 0.08 m away", "stage 1 complete; pipette tip inside the tube mouth (xy < 12 mm, 3-18 cm above the tube bottom; the tube stays in the rack) while the plunger is pressed >= 50%, plunger released, pipette back in the stand (xy < 0.14 m, root 0.08-0.32 m up, < 35 deg tilt, still) with both hands > 0.08 m away")

# Per-episode layout randomization: uniform half-ranges around the nominal
# lane, drawn from the episode RNG. The tube rides in the rack, the pipette in its stand.
# The stand's 110 mm base and the 110 mm dish keep a gap of at least PIPETTE_MIN_STAND_CUP_GAP
# (nominal 40 mm); a draw that closes it further is rejected and redrawn.
PIPETTE_CUP_RANDOM_XY=(0.020,0.020)
PIPETTE_RACK_RANDOM_XY=(0.020,0.020)
PIPETTE_STAND_RANDOM_XY=(0.020,0.020)
PIPETTE_STAND_RANDOM_YAW_DEG=10.0
PIPETTE_STAND_HALF_DEPTH_Y=0.055
PIPETTE_CUP_RADIUS=0.055
PIPETTE_MIN_STAND_CUP_GAP=0.020
PIPETTE_STAND_NOMINAL_Q=(.7071068,0,0,.7071068)

@dataclass
class PipetteConfig:
    # All objects share one X lane. From +Y toward -Y they are ordered:
    # Along +Y to -Y: chemical test-tube rack -> glass culture dish ->
    # glass test tube lane -> separate pipette stand.
    # Task layer: x is expressed against the shared rig's table centre so the
    # lane follows the rig. Measured with the arm roots at
    # x=-0.870, y=+/-0.220: at the old x=0.10 the tube sat 0.959 m, the stand
    # 0.925 m and the rack 0.902 m from their own arm, all past the 0.90 m
    # reach limit -- the two workstations were simply out of range.
    # Second pull-in (IK sweep): at x = TABLE_CENTER_X-0.16 the tube
    # and the pipette handle sat 0.87-0.89 m in front of their arm base -- inside
    # the surface reach band but with NO IK solution for a palm 10 cm above
    # them (Panda envelope ~0.85 m). Every workstation now sits ~0.75 m out,
    # where the other families' manipulated objects live (juice apple 0.79,
    # cabinet_pour bowl 0.72).
    # (operator: "S1 is no longer a strict subset of the long task -- either
    # the dish and the stand overlap, or S1's stand is somewhere else"). Correct. S1 is a
    # LEFT-hand job (the tube stays in the rack, and only the left arm reaches the rack),
    # while S0 and the long task pick the pipette with the RIGHT hand, so the ONE stand
    # position has to be reachable by BOTH arms. The whole lane therefore moves 0.05 m
    # closer to the arms and the stand moves from y=-0.27 to y=-0.17 (measured: stand
    # 0.733 m from the right arm root / 0.820 from the left, dish 0.800 right, rack 0.775
    # left). The stand's 120x110 mm base and the 110 mm dish keep a 40 mm gap. No env
    # overrides these numbers any more.
    rack_xy: tuple[float,float]=(TABLE_CENTER_X-0.35,0.24)
    cup_xy: tuple[float,float]=(TABLE_CENTER_X-0.35,-0.02)
    # The tube starts in the rack's central slot.
    tube_xy: tuple[float,float]=(TABLE_CENTER_X-0.35,0.24)
    pipette_xy: tuple[float,float]=(TABLE_CENTER_X-0.33,-0.17)
    pipette_stand_xy: tuple[float,float]=(TABLE_CENTER_X-0.33,-0.17)
    # Root height places the 225 mm evaluation tip at the stand's 14 mm cup.
    pipette_stand_z: float=0.239
    table_z: float=0.0
    cup_h: float=0.03
    tube_h: float=0.18
    tube_r: float=0.011
    rack_top: float=0.12
    rack_slot_z: float=0.115
    plunger_upper: float=0.015 # was 25 mm. A real manual pipette's first stop is ~10 mm, and the operator's thumb reached 9 mm of OUR 25 mm travel -- the 50 % gate (12.5 mm) was past the retargeted thumb's stroke. 50 % of 15 mm = 7.5 mm.
    pipette_return_radius: float=0.14 # a pipette leaning 30 deg in the deep well has its root ~0.12 m off the stand axis

def _mat(color,roughness=.5):
    rgba=list(color) if len(color)==4 else [*color,1.0]
    return sapien.render.RenderMaterial(base_color=rgba,roughness=roughness)

def _glass_mat(alpha=.22):
    """Create transparent borosilicate-like glass across SAPIEN versions."""
    # Keep the base color bright and use moderate alpha: very low alpha can
    # become black on SAPIEN's fallback renderer.
    # Saturated cyan-blue glass reads as glass in the fallback renderer;
    # nearly neutral low-alpha colors were appearing gray.
    mat=_mat([.68,.90,1.0,max(alpha,.94)],roughness=.08)
    for attr,value in (("transmission",.12),("ior",1.46),("specular",.92)):
        if hasattr(mat,attr): setattr(mat,attr,value)
    return mat

def _actor(env, idx, name, pose, build, dynamic=False, material=None):
    """Build one task body.

    ``material`` is the shared table's PhysxMaterial for this object's class.
    The geometry helpers below add collision shapes without one, so it is
    stamped on every record afterwards; without it the body silently lands on
    the 0.30 fallback, which is a friction value with no physical basis.
    """
    b=env.scene.create_actor_builder(); b.set_scene_idxs([idx]); build(b); b.initial_pose=pose
    if material is not None:
        for record in getattr(b,"collision_records",[]):
            record.material=material
    actor=b.build_dynamic(name) if dynamic else b.build_kinematic(name)
    if dynamic: set_task_object_energy_fence(actor)
    key=name.rsplit("_",1)[0]
    if key in TASK_OBJECT_MASS:
        import sapien.physx as _physx
        for raw in actor._objs:
            comp=raw.find_component_by_type(_physx.PhysxRigidDynamicComponent)
            if comp is not None: comp.mass=float(TASK_OBJECT_MASS[key])
    return actor
def _box(b, half, p, color, collision=True,q=None,material=None):
    pose=sapien.Pose(p=p,q=q or [1,0,0,0])
    if collision: b.add_box_collision(pose=pose,half_size=half)
    b.add_box_visual(pose=pose,half_size=half,material=material or _mat(color))
def _cyl(b, r, h, p, color, collision=True,material=None):
    # SAPIEN cylinders use the local X axis; rotate X to world Z so this
    # helper produces a horizontal disk/vertical vessel as intended.
    z_axis=sapien.Pose(p=p,q=[.7071068,0,.7071068,0])
    if collision: b.add_cylinder_collision(pose=z_axis,radius=r,half_length=h/2)
    b.add_cylinder_visual(pose=z_axis,radius=r,half_length=h/2,material=material or _mat(color))

# Collision bands for the thin glassware. The visual wall stays as
# thin as real glass; the COLLISION wall is a thicker radial band, because PhysX
# flips a convex body to the far side of a box once it has sunk past the box's
# mid-plane: with the 1.2 mm tube wall the pipette tip went straight through at
# 37 N of lateral push (measured with a world->pipette spring), and the 12 mm
# shoulder sank 3 mm into the rim when pressed down. Tube band 7.5-11.5 mm keeps
# the outer radius at the rim's (rack play unchanged, 2.5 mm a side), narrows the
# bore from 9.8 to 7.5 mm (tip radius 3.5 -> 4 mm of play) and holds ~60 N
# lateral / 60 N shoulder push without letting anything through. The dish keeps
# its inner face and grows 5 mm outward (nothing has to fit around it).
TUBE_COLLISION_BAND=(0.0075,0.0115)   # (inner, outer) radius of the tube's collision wall, m
DISH_COLLISION_WALL=0.005             # dish collision wall thickness, outward from the visual inner face, m

def _open_glass_wall(b,radius,height,z_center,wall=.0015,segments=24,collision_band=None):
    """Approximate a hollow glass cylinder without sealing its open mouth.

    ``collision_band=(r_in, r_out)`` puts the collision boxes on a different
    radial band than the visual glass (see TUBE_COLLISION_BAND); ``None`` keeps
    collision and visual identical.
    """
    glass=_glass_mat(.72)
    tangent_half=radius*np.sin(np.pi/segments)*1.08
    for theta in np.linspace(0,2*np.pi,segments,endpoint=False):
        phi=theta+np.pi/2
        q=[np.cos(phi/2),0,0,np.sin(phi/2)]
        _box(
            b,[tangent_half,wall/2,height/2],
            [(radius-wall/2)*np.cos(theta),(radius-wall/2)*np.sin(theta),z_center],
            [.72,.90,1.0],q=q,material=glass,collision=collision_band is None,
        )
        if collision_band is not None:
            r_in,r_out=collision_band; rc=.5*(r_in+r_out); half_t=.5*(r_out-r_in)
            b.add_box_collision(
                pose=sapien.Pose(p=[rc*np.cos(theta),rc*np.sin(theta),z_center],q=q),
                half_size=[tangent_half*(rc/(radius-wall/2)),half_t,height/2],
            )

def _ring(b,radius,height,z_center,color,wall=.003,segments=20):
    """Opaque plastic ring (same tangent-box construction as the glass walls)."""
    tangent_half=radius*np.sin(np.pi/segments)*1.08
    for theta in np.linspace(0,2*np.pi,segments,endpoint=False):
        phi=theta+np.pi/2
        _box(b,[tangent_half,wall/2,height/2],
             [(radius-wall/2)*np.cos(theta),(radius-wall/2)*np.sin(theta),z_center],
             color,q=[np.cos(phi/2),0,0,np.sin(phi/2)])

def _build_glass_beaker(b,cfg):
    """Build a shallow transparent glass culture/sample dish."""
    outer_r=.055; wall=.0018; bottom=.003
    dish_band=(outer_r-wall,outer_r-wall+DISH_COLLISION_WALL)
    _open_glass_wall(b,outer_r,cfg.cup_h,cfg.cup_h/2,wall=wall,segments=28,collision_band=dish_band)
    # Use a non-refractive, lightly tinted glass disk for the floor. This
    # prevents transparent depth sorting from making the bottom look tilted;
    # keep a 1 mm clearance above the table instead of merging both surfaces.
    floor_mat=_mat([.68,.88,1.0,1.0],roughness=.16)
    _cyl(b,outer_r,bottom,[0,0,bottom/2+.001],[.68,.88,1.0],material=floor_mat)
    # Thick glass rim; the shallow profile distinguishes a culture dish from
    # the former tall beaker while keeping the opening physically accessible.
    _open_glass_wall(b,outer_r+.0005,.004,cfg.cup_h-.002,wall=.0025,segments=28,collision_band=dish_band)

def _build_test_tube(b,cfg):
    r=cfg.tube_r; bottom_center=r
    _open_glass_wall(b,r,cfg.tube_h-r,bottom_center+(cfg.tube_h-r)/2,wall=.0012,segments=20,
                     collision_band=TUBE_COLLISION_BAND)
    # Slightly stronger cyan tint than the dish so the tube remains visible
    # against the pale tabletop while still reading as transparent glass.
    glass=_mat([.48,.80,1.0,.92],roughness=.10)
    for attr,value in (("transmission",.12),("ior",1.46),("specular",.92)):
        if hasattr(glass,attr): setattr(glass,attr,value)
    b.add_sphere_collision(pose=sapien.Pose(p=[0,0,bottom_center]),radius=r)
    b.add_sphere_visual(pose=sapien.Pose(p=[0,0,bottom_center]),radius=r,material=glass)
    _open_glass_wall(b,r+.0004,.003,cfg.tube_h-.0015,wall=.0018,segments=20,collision_band=TUBE_COLLISION_BAND)

def _build_test_tube_rack(b):
    """Build the original compact 3x3 two-level chemical test-tube rack."""
    plastic=[.10,.42,.62]; half_span=.0575; pitch=.035; bar=.007
    # Two perforated support grids. The center opening is the target slot.
    for z in (.038,.112):
        for offset in (-1.5*pitch,-.5*pitch,.5*pitch,1.5*pitch):
            _box(b,[half_span,bar/2,.003],[0,offset,z],plastic)
            _box(b,[bar/2,half_span,.003],[offset,0,z],plastic)
    for x in (-half_span,half_span):
        for y in (-half_span,half_span):
            _box(b,[.005,.005,.055],[x,y,.055],plastic)
            _box(b,[.011,.011,.003],[x,y,.003],plastic)

def _build_pipette_stand(b):
    """Pipette holder: base plate, tip cup, two guide rings on rear posts.

    Operator: "revert the deep well -- leaning the pipette against the
    stand is not hard in teleop". The 138 mm-tall well built before that is
    gone and the original open rack is back. The rings clear the 24 mm body by
    ~4 mm a side, and `returned` accepts up to 35 deg of lean with the root
    anywhere inside 0.14 m of the stand axis, so a pipette dropped tip-first into
    the cup and left leaning on a ring still counts as parked.
    """
    blue=[.16,.30,.42]
    _box(b,[.060,.055,.006],[0,0,.006],blue)                                       # base plate
    for x in (-.040,.040):
        _box(b,[.006,.006,.095],[x,-.045,.101],blue)                               # rear posts
    _box(b,[.046,.006,.006],[0,-.045,.196],blue)                                   # rear crossbar
    _cyl(b,.045,.006,[0,0,.009],[.06,.18,.25],material=_mat([.06,.18,.25],.75))    # tip cup (the tip's rest)
    # (operator: "the two rings are floating in mid air"): the rings sit on
    # the stand axis while the posts stand at y = -0.045, so nothing joined them. Each
    # ring now hangs off a rear bracket plate that spans BOTH posts, and the rings are
    # the stand's own plastic instead of glass.
    for z in (.085,.165):
        _ring(b,.019,.012,z,blue,wall=.003,segments=20)                            # guide ring, 16 mm bore
        _box(b,[.040,.013,.005],[0,-.032,z],blue)                                  # bracket: ring back -> both posts

def _quat_rotate(q,v):
    """Rotate local vectors by batched SAPIEN quaternions (w, x, y, z)."""
    w,x,y,z=q.unbind(-1)
    vx,vy,vz=v
    return torch.stack((
        (1-2*(y*y+z*z))*vx + 2*(x*y-z*w)*vy + 2*(x*z+y*w)*vz,
        2*(x*y+z*w)*vx + (1-2*(x*x+z*z))*vy + 2*(y*z-x*w)*vz,
        2*(x*z-y*w)*vx + 2*(y*z+x*w)*vy + (1-2*(x*x+y*y))*vz,
    ),dim=-1)

def _lengthen_pipette_grip(builder):
    """Extend only the upper hand-held body while keeping the tip fixed."""
    links={link.name:link for link in builder.link_builders}
    try:
        base=links["base"]; plunger=links["plunger"]
        record_groups=(base.visual_records,base.collision_records)
        if any(len(records)<2 for records in record_groups): raise KeyError("base grip records")
    except KeyError as exc:
        raise RuntimeError("Unexpected pipette URDF layout; cannot extend grip safely") from exc

    body_center_z=PIPETTE_GRIP_EXTENSION/2
    rear_grip_center_z=0.060+PIPETTE_GRIP_EXTENSION
    for records in record_groups:
        body,rear_grip=records[:2]
        if body.type!="cylinder" or rear_grip.type!="cylinder":
            raise RuntimeError("Unexpected pipette grip geometry")
        # SAPIEN builder records store cylinder half-length in ``length``.
        body.length=PIPETTE_GRIP_LENGTH/2
        body.pose=sapien.Pose(p=[0,0,body_center_z],q=list(body.pose.q))
        rear_grip.pose=sapien.Pose(p=[0,0,rear_grip_center_z],q=list(rear_grip.pose.q))

    joint=plunger.joint_record
    parent_p=list(joint.pose_in_parent.p)
    parent_p[2]+=PIPETTE_GRIP_EXTENSION
    joint.pose_in_parent=sapien.Pose(p=parent_p,q=list(joint.pose_in_parent.q))

def _load_pipette(env,idx,cfg):
    loader=env.scene.create_urdf_loader(); loader.fix_root_link=False; loader.scale=1.0
    # Vertical in the separate stand at the rack's Y-negative side; the local
    # -Z tip points down into the lower support cup.
    d=loader.parse(str(PIPETTE_URDF)); b=d["articulation_builders"][0]; _lengthen_pipette_grip(b); b.set_scene_idxs([idx]); b.initial_pose=sapien.Pose(p=[cfg.pipette_stand_xy[0],cfg.pipette_stand_xy[1],cfg.table_z+cfg.pipette_stand_z],q=[1,0,0,0])
    # Task layer: the pipette body is the object the hand grips, so its class
    # comes from the shared table instead of the 0.30 fallback.
    pipette_material=object_material("hard_plastic")
    for link_builder in b.link_builders:
        for record in getattr(link_builder,"collision_records",[]):
            record.material=pipette_material
    pipette=b.build(name=f"pipette_{idx}",fix_root_link=False)
    set_task_object_energy_fence(pipette)
    if len(pipette.active_joints)!=1 or not pipette.active_joints[0].name.endswith("plunger_joint"):
        names=[joint.name for joint in pipette.active_joints]
        raise RuntimeError(f"Expected one pipette plunger joint, found {names}")
    plunger=pipette.active_joints[0]
    plunger.set_drive_properties(
        stiffness=PIPETTE_PLUNGER_STIFFNESS,
        damping=PIPETTE_PLUNGER_DAMPING,
        force_limit=PIPETTE_PLUNGER_FORCE_LIMIT,
        mode="force",
    )
    plunger.set_drive_target(0.0)
    if hasattr(plunger,"set_friction"):
        plunger.set_friction(PIPETTE_PLUNGER_FRICTION)
    return pipette

def build_scene(env,cfg):
    table=TableSceneBuilder(env=env); table.build(); env.table_scene=table; set_work_table_pose(env); cups=[]; tubes=[]; racks=[]; pips=[]; stands=[]
    for i in range(env.num_envs):
        if env.LOAD_CUP:
            cups.append(_actor(env,i,f"sample_cup_{i}",sapien.Pose(p=[*cfg.cup_xy,cfg.table_z]),lambda b: _build_glass_beaker(b,cfg),material=object_material("glass",dominant="table")))
        if env.LOAD_RACK: racks.append(_actor(env,i,f"tube_rack_{i}",sapien.Pose(p=[*cfg.rack_xy,cfg.table_z]),_build_test_tube_rack,material=object_material("hard_plastic",dominant="table")))
        if env.LOAD_TUBE: tubes.append(_actor(env,i,f"test_tube_{i}",sapien.Pose(p=[*cfg.tube_xy,cfg.table_z+.003]),lambda b: _build_test_tube(b,cfg),True,material=object_material("glass")))
        if env.LOAD_PIPETTE_STAND:
            # Rotate the stand counter-clockwise 90 degrees about world Z;
            # its center and the vertical pipette pose remain unchanged.
            stands.append(_actor(
                env,i,f"pipette_stand_{i}",
                sapien.Pose(p=[*cfg.pipette_stand_xy,cfg.table_z],q=[.7071068,0,0,.7071068]),
                _build_pipette_stand,
                material=object_material("hard_plastic",dominant="table"),
            ))
        if env.LOAD_PIPETTE: pips.append(_load_pipette(env,i,cfg))
    return table,Actor.merge(cups,"sample_cup") if cups else None,Actor.merge(tubes,"test_tube") if tubes else None,Actor.merge(racks,"tube_rack") if racks else None,Actor.merge(stands,"pipette_stand") if stands else None,Articulation.merge(pips,"pipette") if pips else None

def _point_to_segment_dist(point,pose,seg_local):
    """Distance from world points (N,3) to a segment given in an actor's local frame."""
    a=pose.p+_quat_rotate(pose.q,torch.tensor(seg_local[0],device=point.device,dtype=point.dtype))
    b=pose.p+_quat_rotate(pose.q,torch.tensor(seg_local[1],device=point.device,dtype=point.dtype))
    ab=b-a; s=torch.clamp(torch.sum((point-a)*ab,dim=1)/torch.clamp(torch.sum(ab*ab,dim=1),min=1e-9),0.0,1.0)
    return torch.linalg.norm(point-(a+s.unsqueeze(1)*ab),dim=1)

def _hand_tcps(env):
    """Right / left palm positions. On the dual rig both exist; on the single
    right-arm rig (S0) the 'left hand' is parked at infinity so every left-hand
    predicate is simply false instead of raising."""
    ag=env.agent
    if hasattr(ag,"right_tcp_pos"): return ag.right_tcp_pos,ag.left_tcp_pos
    # Single right-arm rig: build the SAME point the dual agent calls right_tcp_pos
    # (0.7 * mean fingertip + 0.3 * palm). The single agents' own tcp_pos is the bare
    # palm origin, ~0.1 m behind the fingers, and the 0.08 m "hand at the pipette"
    # gate was written for the fingertip-weighted point.
    tips=torch.stack([t.pose.p for t in ag.tip_links],dim=0)
    rt=0.7*tips.mean(dim=0)+0.3*ag.palm_link.pose.p
    return rt,torch.full_like(rt,999.0)

class PipetteEvaluator:
    def __init__(self,env,cfg):
        self.env=env; self.cfg=cfg
        self.stage=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        self.first_press=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.second_press=torch.zeros_like(self.first_press); self.was_pressed=torch.zeros_like(self.first_press)
        # from an operator log of a full S0 run: the tip WAS in the dish
        # (t=580-600) and the plunger WAS worked (t=620-640, 9 mm) -- but not at the
        # same instant, because a human pipettes dip-then-press as readily as
        # press-then-dip. Requiring both in one frame made a correct run read as a
        # failure. Both halves are latched separately now, while the pipette is held.
        self.primed=torch.zeros_like(self.first_press)   # plunger worked while holding it
        self.dipped=torch.zeros_like(self.first_press)   # tip taken into the sample dish
    def reset(self,idx):
        self.stage[idx]=0; self.first_press[idx]=False; self.second_press[idx]=False
        self.was_pressed[idx]=False; self.primed[idx]=False; self.dipped[idx]=False
    def evaluate(self):
        e,c=self.env,self.cfg; rt,lt=_hand_tcps(e); dual=hasattr(e.agent,"right_tcp_pos"); zero=torch.zeros(e.num_envs,dtype=torch.bool,device=e.device); tip_cup=zero; tip_tube=zero; q=torch.zeros(e.num_envs,device=e.device); tcp_pip=torch.full_like(q,999.); tcp_tube=torch.full_like(q,999.)
        if e.pipette is not None:
            pip=e.pipette.pose.p; q=e.pipette.get_qpos()[:,0]
            tip=pip+_quat_rotate(e.pipette.pose.q,torch.tensor([0,0,-.225],device=e.device))
            pip_axis=_quat_rotate(e.pipette.pose.q,torch.tensor([0,0,1],device=e.device))
            pressed=q>=c.plunger_upper*PLUNGER_PRESSED_FRAC; released=q<=c.plunger_upper*PLUNGER_RELEASED_FRAC
            # The pipette hand is the subtask rig's: S0 (stage 0) is the RIGHT-arm subtask,
            # S1 (stage >= 1) the LEFT-arm one. A single rig has only one hand and
            # `_hand_tcps` puts it in `rt`; on the dual rig the stage picks the hand, and
            # "hand away" means BOTH hands away from the parked pipette.
            tcp_pip_r=_point_to_segment_dist(rt,e.pipette.pose,PIPETTE_BODY_SEGMENT_LOCAL)
            if dual:
                tcp_pip_l=_point_to_segment_dist(lt,e.pipette.pose,PIPETTE_BODY_SEGMENT_LOCAL)
                tcp_pip=torch.where(self.stage>=1,tcp_pip_l,tcp_pip_r)
                hands_away=(tcp_pip_r>HAND_ON_OBJECT_DIST)&(tcp_pip_l>HAND_ON_OBJECT_DIST)
            else:
                tcp_pip=tcp_pip_r; hands_away=tcp_pip>HAND_ON_OBJECT_DIST
            pipette_lifted=(pip[:,2]>c.table_z+.04)&(tcp_pip<HAND_ON_OBJECT_DIST)
            if e.sample_cup is not None:
                cup=e.sample_cup.pose.p; mouth_z=cup[:,2]+c.cup_h
                tip_cup=torch.linalg.norm((tip-cup)[:,:2],dim=1)<.040
                tip_cup &= (tip[:,2]>cup[:,2]-.01)&(tip[:,2]<mouth_z-.003)   # anywhere inside the cup below the rim
            if e.test_tube is not None:
                tube=e.test_tube.pose.p; tip_tube=torch.linalg.norm((tip-tube)[:,:2],dim=1)<TIP_IN_TUBE_XY_TOL; tip_tube &= (tip[:,2]>tube[:,2]+.03)&(tip[:,2]<tube[:,2]+c.tube_h)
            # Both halves of "aspirate" only count while the hand actually holds the
            # lifted pipette; their ORDER is free (see PipetteEvaluator.__init__).
            self.primed |= pipette_lifted&pressed
            self.dipped |= pipette_lifted&tip_cup
            self.first_press |= (self.stage==0)&self.primed&self.dipped
            self.second_press |= (self.stage>=1)&tip_tube&pressed
            stand_xy=e.pipette_stand.pose.p[:,:2] if e.pipette_stand is not None else torch.tensor(c.pipette_stand_xy,device=e.device)
            returned=torch.linalg.norm(pip[:,:2]-stand_xy,dim=1)<c.pipette_return_radius
            # The long stand stores the pipette vertically, so its URDF root
            # remains above the tabletop while the local -Z tip points down.
            returned &= (pip[:,2]>c.table_z+PIPETTE_RETURN_Z_MIN)&(pip[:,2]<c.table_z+PIPETTE_RETURN_Z_MAX)
            returned &= torch.linalg.norm(e.pipette.root_linear_velocity,dim=1)<.05
            returned &= torch.linalg.norm(e.pipette.root_angular_velocity,dim=1)<.25   # a leaning pipette wobbles ~0.1 rad/s against the well wall
            returned &= pip_axis[:,2] > PIPETTE_RETURN_TILT_COS
            returned &= hands_away
            self.stage=torch.where((self.stage==0)&self.first_press&released&returned,torch.ones_like(self.stage),self.stage)
            # S1 (subtask, the long task's own rule now): the tube
            # STAYS in the rack; the second press plus the parked pipette finish the task.
            # Stage 2 (the lifted tube of the retired dual-hand dispense) no longer exists,
            # the counter goes 1 -> 3 so the recorded S1 data keep their stage values.
            self.stage=torch.where((self.stage==1)&self.second_press&released&returned,torch.full_like(self.stage,3),self.stage)
        else:
            returned=zero
        if e.test_tube is not None:
            tcp_tube=_point_to_segment_dist(lt,e.test_tube.pose,TUBE_BODY_SEGMENT_LOCAL)   # HUD only: the tube is never handled
        return dict(success=self.stage>=3,stage=self.stage,first_press=self.first_press,plunger_worked=self.primed,tip_was_in_cup=self.dipped,second_press=self.second_press,pipette_returned=returned,tcp_pipette=tcp_pip,tcp_tube=tcp_tube,tip_in_cup=tip_cup,tip_in_tube=tip_tube,plunger=q)

@register_task(
    "PipetteTransferLong",
    side="both",
    max_episode_steps=1600,
    obs_mode="state",
    control_mode="pd_joint_pos",
    reward_mode="normalized_dense",
    render_mode="rgb_array",
)
class PipetteTransferLongEnv(BaseEnv):
    SUPPORTED_ROBOTS=[ROBOT_UID,"panda_dual_wuji_hand_angled_mobile_base","panda_dual_xhand_hand_angled_mobile_base","panda_dual_gripper_mobile_base"]; TARGET_STAGE=0
    LOAD_CUP=True; LOAD_PIPETTE=True; LOAD_TUBE=True; LOAD_RACK=True; LOAD_PIPETTE_STAND=True
    def __init__(self,*a,robot_init_qpos_noise=ARM_INIT_QPOS_NOISE_STD,**kw):
        # Gaussian arm-joint noise from the episode RNG (fleet std by default);
        # the dual-arm teleop entry point passes 0.0 on purpose.
        self.robot_init_qpos_noise=float(robot_init_qpos_noise)
        if self.robot_init_qpos_noise<0.0:
            raise ValueError("robot_init_qpos_noise must be non-negative")
        self.cfg=PipetteConfig()
        self._customize_config(self.cfg)   # subtasks may move workstations BEFORE the scene is built
        self.show_guidance_markers = pop_guidance_marker_flag(kw)
        super().__init__(
            *a,robot_uids=kw.pop("robot_uids",ROBOT_UID),**kw,
        )
    @property
    def _default_sim_config(self):
        # Environment layer: the locked fleet profile. This env used to return a
        # bare SimConfig(), i.e. ManiSkill's defaults -- 100 Hz, solver 15/1 and
        # a 20 mm contact offset, against 5 mm task tolerances.
        return make_sim_config()
    @property
    def _default_sensor_configs(self): return [CameraConfig("base_camera",sapien_utils.look_at([-.85,-1.15,.8],[0,0.0,.08]),640,480,1.1,.01,100)]
    @property
    def _default_human_render_camera_configs(self): return CameraConfig("render_camera",sapien_utils.look_at([-.85,-1.15,.8],[0,0.0,.08]),1280,720,1.1,.01,100)
    def _customize_config(self,cfg):
        """Hook for subtasks (e.g. the left-hand dispense stage moves the stand)."""
    def _load_agent(self,options): super()._load_agent(options,DUAL_ROBOT_BASE_POSE)
    def _set_work_table_pose(self,table_scene=None):
        """Environment layer: the work table belongs to the shared rig."""
        if table_scene is not None and getattr(self,"table_scene",None) is None:
            self.table_scene=table_scene
        set_work_table_pose(self)
    def _load_scene(self,options):
        self.table_scene,self.sample_cup,self.test_tube,self.tube_rack,self.pipette_stand,self.pipette=build_scene(self,self.cfg)
        for x in (self.sample_cup,self.test_tube,self.tube_rack,self.pipette_stand,self.pipette):
            if x is not None: self.add_to_state_dict_registry(x)
    def _after_reconfigure(self,options): self.eval=PipetteEvaluator(self,self.cfg)
    def _initialize_episode(self,env_idx,options):
        self.table_scene.initialize(env_idx)
        self._set_work_table_pose()
        # BaseEnv reset restores the table; explicitly restore the same robot
        # world pose so partial resets cannot move the trolley to the origin
        # (dual rig for the long task / S1, single right-arm rig for S0).
        self.agent.robot.set_root_pose(DUAL_ROBOT_BASE_POSE)
        lay=self._sample_layout(env_idx)   # first episode-RNG draw: layout (see layout_rng)
        self._reset_arms()
        self.eval.reset(env_idx); n=len(env_idx)
        T=lambda rows: torch.tensor(np.asarray(rows),dtype=torch.float32,device=self.device)
        ident=T([[1,0,0,0]]*n)
        if self.sample_cup is not None: self.sample_cup.set_pose(Pose.create_from_pq(T([[x,y,self.cfg.table_z] for x,y in lay['cup']]),ident))
        if self.tube_rack is not None: self.tube_rack.set_pose(Pose.create_from_pq(T([[x,y,self.cfg.table_z] for x,y in lay['rack']]),ident))
        if self.pipette_stand is not None: self.pipette_stand.set_pose(Pose.create_from_pq(T([[x,y,self.cfg.table_z] for x,y in lay['stand']]),T(lay['stand_q'])))
        if self.pipette is not None:
            self.pipette.set_qpos(torch.zeros_like(self.pipette.get_qpos())); self.pipette.set_qvel(torch.zeros_like(self.pipette.get_qvel()))
            pip_pose=Pose.create_from_pq(
                T([[x,y,self.cfg.table_z+self.cfg.pipette_stand_z] for x,y in lay['stand']]),
                ident,
            )
            self.pipette.set_pose(pip_pose)
            self.pipette.set_root_linear_velocity(torch.zeros((n,3),device=self.device))
            self.pipette.set_root_angular_velocity(torch.zeros((n,3),device=self.device))
        if self.test_tube is not None:
            self.test_tube.set_pose(Pose.create_from_pq(T([[x+(self.cfg.tube_xy[0]-self.cfg.rack_xy[0]),y+(self.cfg.tube_xy[1]-self.cfg.rack_xy[1]),self.cfg.table_z+.003] for x,y in lay['rack']]),ident)); self.test_tube.set_linear_velocity(torch.zeros((n,3),device=self.device)); self.test_tube.set_angular_velocity(torch.zeros((n,3),device=self.device))
    def _sample_layout(self,env_idx):
        """Per-episode workstation layout from the episode RNG (see the PIPETTE_*_RANDOM constants).

        Draw order per sub-environment: cup (dx,dy), rack (dx,dy), stand (dx,dy,yaw); the whole
        triple is redrawn while the stand base would come closer than PIPETTE_MIN_STAND_CUP_GAP to
        the dish. Returns world xy per object, plus the stand's quaternion."""
        c=self.cfg; out={"cup":[],"rack":[],"stand":[],"stand_q":[]}
        for i in env_idx.tolist():
            rng=layout_rng(self,i)
            for _ in range(64):
                cdx,cdy,_=uniform_offset(rng,PIPETTE_CUP_RANDOM_XY); rdx,rdy,_=uniform_offset(rng,PIPETTE_RACK_RANDOM_XY)
                sdx,sdy,syaw=uniform_offset(rng,PIPETTE_STAND_RANDOM_XY,PIPETTE_STAND_RANDOM_YAW_DEG)
                cup=(c.cup_xy[0]+cdx,c.cup_xy[1]+cdy); stand=(c.pipette_stand_xy[0]+sdx,c.pipette_stand_xy[1]+sdy)
                if (cup[1]-stand[1])-PIPETTE_STAND_HALF_DEPTH_Y-PIPETTE_CUP_RADIUS>=PIPETTE_MIN_STAND_CUP_GAP: break
            else:
                cup=tuple(c.cup_xy); stand=tuple(c.pipette_stand_xy); rdx=rdy=0.0; syaw=0.0
            _,q=offset_pose((0,0,0),PIPETTE_STAND_NOMINAL_Q,0.0,0.0,syaw)
            out["cup"].append(cup); out["rack"].append((c.rack_xy[0]+rdx,c.rack_xy[1]+rdy)); out["stand"].append(stand); out["stand_q"].append(q)
        return out
    def _reset_arms(self):
        """Arm layer: both arms start from the shared rig pose.

        Without this the arms fall through to the agent keyframe, which sits
        2.6 rad away at one joint -- the operator feels a different wrist
        attitude from the first frame than in every other task.
        """
        qpos=self.agent.robot.get_qpos().clone()
        active=[j.name for j in self.agent.robot.get_active_joints()]
        idx=[active.index(n) for n in getattr(self.agent,"arm_joint_names",[]) if n in active]
        shared=torch.tensor(ARM_INIT_QPOS,dtype=qpos.dtype,device=self.device)
        for start in range(0,len(idx),len(ARM_INIT_QPOS)):
            block=idx[start:start+len(ARM_INIT_QPOS)]
            if len(block)==len(ARM_INIT_QPOS):
                qpos[:,torch.tensor(block,device=self.device)]=shared
        for k in range(qpos.shape[0]):
            index,noise=arm_qpos_noise(episode_rng(self,k),self.agent,1,self.robot_init_qpos_noise,device=self.device)
            qpos[k,index]+=noise[0]
        self.agent.robot.set_qpos(qpos)
        self.agent.robot.set_qvel(torch.zeros_like(self.agent.robot.get_qvel()))
        self.agent.controller.reset()

    def evaluate(self): return self.eval.evaluate()
    def _get_obs_extra(self,info):
        out={"stage":info["stage"][:,None],"first_press":info["first_press"][:,None],"second_press":info["second_press"][:,None],"pipette_returned":info["pipette_returned"][:,None]}
        if self.pipette is not None: out["pipette_pose"]=self.pipette.pose.raw_pose
        if self.sample_cup is not None: out["cup_pose"]=self.sample_cup.pose.raw_pose
        if self.test_tube is not None: out["tube_pose"]=self.test_tube.pose.raw_pose
        if self.tube_rack is not None: out["rack_pose"]=self.tube_rack.pose.raw_pose
        if self.pipette_stand is not None: out["pipette_stand_pose"]=self.pipette_stand.pose.raw_pose
        return out
    def get_state_dict(self):
        state=super().get_state_dict()
        state["pipette_task_state"]=torch.stack((
            self.eval.stage.float(),
            self.eval.first_press.float(),
            self.eval.second_press.float(),
            self.eval.was_pressed.float(),
            self.eval.primed.float(),
            self.eval.dipped.float(),
        ),dim=-1)
        return state
    def set_state_dict(self,state,env_idx=None):
        state=dict(state)
        task_state=state.pop("pipette_task_state",None)
        if env_idx is None:
            restore_idx=torch.arange(self.num_envs,dtype=torch.long,device=self.device)
            super().set_state_dict(state)
        else:
            restore_idx=torch.as_tensor(
                env_idx,dtype=torch.long,device=self.device,
            ).flatten()
            super().set_state_dict(state,restore_idx)
        if isinstance(task_state,dict):
            task_state=task_state.get("value")
        if task_state is None:
            return
        task_state=torch.as_tensor(task_state,device=self.device).reshape(
            len(restore_idx),-1,
        )
        if task_state.shape[-1] not in (
            PIPETTE_TASK_STATE_LEGACY_DIM, PIPETTE_TASK_STATE_DIM
        ):
            raise ValueError(
                f"pipette_task_state must have {PIPETTE_TASK_STATE_LEGACY_DIM} or "
                f"{PIPETTE_TASK_STATE_DIM} fields, got {task_state.shape[-1]}"
            )
        self.eval.stage[restore_idx]=task_state[:,0].to(self.eval.stage.dtype)
        self.eval.first_press[restore_idx]=task_state[:,1].bool()
        self.eval.second_press[restore_idx]=task_state[:,2].bool()
        self.eval.was_pressed[restore_idx]=task_state[:,3].bool()
        if task_state.shape[-1]==PIPETTE_TASK_STATE_DIM:
            self.eval.primed[restore_idx]=task_state[:,4].bool()
            self.eval.dipped[restore_idx]=task_state[:,5].bool()
    def compute_dense_reward(self,obs,action,info): return info["stage"].float()
    def compute_normalized_dense_reward(self,obs,action,info): return self.compute_dense_reward(obs,action,info)/3

class PipetteSubtaskEnv(PipetteTransferLongEnv):
    TARGET_STAGE=0
    REQUIRE_FINAL_PLACEMENT=False
    def __init__(self,*a,target_stage=None,**kw): self.TARGET_STAGE=self.__class__.TARGET_STAGE if target_stage is None else target_stage; super().__init__(*a,**kw)
    def _initialize_episode(self,env_idx,options):
        super()._initialize_episode(env_idx,options)
        self.eval.stage[env_idx]=self.TARGET_STAGE
        self.eval.first_press[env_idx]=self.TARGET_STAGE>=1
        self.eval.second_press[env_idx]=self.TARGET_STAGE>=2
        # Standalone subtasks keep the exact corresponding full-task reset
        # poses. Only the predecessor stage flags and object masks differ.
    def evaluate(self):
        info=super().evaluate()
        info["success"]=(info["stage"]>=3) if self.REQUIRE_FINAL_PLACEMENT else (info["stage"]>self.TARGET_STAGE)
        return info
    def compute_normalized_dense_reward(self,obs,action,info): return (info["stage"].float()+info["success"].float())/4

def print_task_list():
    print(f"{ENV_ID} subtasks (2)")
    for i,((n,d),c) in enumerate(zip(TASK_PLAN,TASK_SUCCESS_CONDITIONS),1): print(f"\n{i}. {n}\n   task: {d}\n   success: {c}")
VIDEO_DIRS={
    ENV_ID:"videos/pipette_transfer",
    "PipetteTransferAspirateLiquid-Sharpa-v1":"videos/pipette_aspirate",
    "PipetteTransferDispenseLiquid-Sharpa-v1":"videos/pipette_dispense",
}

def run_cli(default_env_id=ENV_ID,default_output_dir=None):
    parser=argparse.ArgumentParser(description="Pipette transfer smoke runner (dual rig; S0 subtask on the right-arm rig)")
    parser.add_argument("--env-id",default=default_env_id,choices=(ENV_ID,"PipetteTransferAspirateLiquid-Sharpa-v1","PipetteTransferDispenseLiquid-Sharpa-v1"))
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--list",action="store_true")
    parser.add_argument("--record",action="store_true",help="record an MP4 video")
    parser.add_argument("--output-dir",default=default_output_dir,help="video output directory")
    parser.add_argument("--video-fps",type=int,default=25)
    args=parser.parse_args()
    if args.list:
        print_task_list(); return
    # Import wrappers so their Gym registrations are available when selected.
    import tasks.pipette_transfer.pipette_transfer_aspirate_liquid, tasks.pipette_transfer.pipette_transfer_dispense_liquid
    output_dir=args.output_dir or VIDEO_DIRS[args.env_id]
    env=gym.make(args.env_id,num_envs=1,obs_mode="state",reward_mode="normalized_dense",control_mode="pd_joint_pos",render_mode="rgb_array" if args.record else None,render_backend="gpu" if args.record else "none")
    if args.record:
        env=RecordEpisode(env,output_dir=output_dir,save_trajectory=False,save_video_trigger=lambda episode_idx: True,max_steps_per_video=args.steps,video_fps=args.video_fps)
        print(f"Video output directory: {Path(output_dir).resolve()}")
    obs,info=env.reset(seed=args.seed)
    for step in range(args.steps):
        obs,reward,terminated,truncated,info=env.step(env.action_space.sample())
        if args.record: env.render()
        if step % 25 == 0:
            print(f"step={step:04d} stage={int(info['stage'][0])} reward={float(reward[0]):.3f} success={bool(info['success'][0])}")
        if bool(terminated.any()) or bool(truncated.any()): break
    env.close()

if __name__=="__main__": run_cli()

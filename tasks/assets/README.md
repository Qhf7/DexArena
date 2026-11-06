# DexArena simulation assets

This directory contains all the geometry and kinematic descriptions used by the
simulation: the Franka Panda arm, every supported dexterous hand, the mobile
trolley base, and the combined URDFs that the tasks actually load.

## Directory layout

```
tasks/assets/
├── urdf/                     ← combined URDFs (arm + hand[+ adapter]); tasks load these
├── franka_description/       ← stock Franka Panda meshes (link0..7, visual + collision)
├── mobile_base_meshes/       ← PartNet-Mobility 22870 trolley (meshes + textures)
├── marvin_wrist_camera_meshes/ ← wrist camera housing
├── sharpa_hand_meshes/       ← Sharpa Wave (with built-in wrist link)
├── wuji_hand_meshes/         ← Wuji Hand (ships with a separate docking adapter — see its README)
└── xhand_hand_meshes/        ← RobotEra XHand
```

Each `<hand>_meshes/` directory may optionally contain its own `README.md`
documenting upstream provenance, license, and any hand-specific assembly
details. **Always read the hand-specific README before changing a URDF that
references it.**

## URDF inventory

The `urdf/` directory contains pre-assembled robots that combine the Panda arm
with each hand. Tasks reference these by filename — never load a raw upstream
URDF directly.

### Single-arm (right hand)

| File | Arm | Hand | Notes |
|------|-----|------|-------|
| `panda_sharpa_hand_right.urdf`   | Panda | Sharpa Wave | Uses `_with_wrist` variant from upstream |
| `panda_wuji_hand_right.urdf`     | Panda | Wuji        | Includes `hand_docking_link` adapter |
| `panda_xhand_hand_right.urdf`    | Panda | XHand       | |

### Single-arm on the DUAL trolley (derived from the dual assets)

These four were derived from `panda_dual_<hand>_hand_mobile_base.urdf` by pruning one arm
and de-prefixing the arm chain (the hand chain keeps its `left_` / `right_`). They are
committed assets with no generator: **if a dual mobile-base URDF changes, these four have
to be brought back in step by hand** — nothing checks it for you.

They keep the dual robot's **full-size** trolley and leave the shoulder at its dual mount
(`0 ±0.22 0`) rather than centring it, so at a given root pose each one's arm is
kinematically identical, joint for joint, to the corresponding arm of the dual robot
(verified: 0.0 m palm and fingertip FK error over random poses). A subtask solved with one
of these composes into the long-horizon dual task without re-deriving reach, and left/right
are exact mirrors of each other.

| File | Arm | Hand | Notes |
|------|-----|------|-------|
| `panda_sharpa_hand_left_mobile_base.urdf`  | Panda | Sharpa Wave (left meshes)  | arm at `y = +0.22` |
| `panda_sharpa_hand_right_mobile_base.urdf` | Panda | Sharpa Wave | arm at `y = -0.22` |
| `panda_wuji_hand_left_mobile_base.urdf`    | Panda | Wuji (left meshes)  | arm at `y = +0.22` |
| `panda_wuji_hand_right_mobile_base.urdf`   | Panda | Wuji | arm at `y = -0.22` |
| `panda_gripper_left_mobile_base.urdf`      | Panda | Franka gripper | arm at `y = +0.22`; `scripts/build_gripper_mobile_base_urdfs.py` |
| `panda_gripper_right_mobile_base.urdf`     | Panda | Franka gripper | arm at `y = -0.22`; same generator |

Naming follows the single-arm convention, not the dual one: the arm chain and (wuji) the
docking flange lose their side prefix, while the hand chain keeps `left_` / `right_`. That
rule is not a guess: applying it to the dual assets reproduced the previous hand-authored
right URDFs with an identical kinematic graph (52 links / 51 joints for sharpa, 45 / 44 for
wuji), for both hands.

`panda_sharpa_hand_{left,right}_angled_mobile_base.urdf` are derived from the two sharpa
assets at import time (each side's agent module generates its own, mirroring
`_generate_dual_angled_urdf`), so they inherit this trolley automatically. They are
gitignored build products, not committed.

**Layout impact.** The two right assets moved to this geometry on 2026-08-12 (they were
previously the compact 0.75 trolley with a centred arm). The trolley's front edge now sits
at `x = +0.4068` from the root instead of `+0.3051`, i.e. **10.2 cm further forward**, and
the shoulder moved 22 cm in `-y`. Any base pose picked for the compact trolley is now too
close to the work surface — it loses ~7 cm of the ~3 cm of table clearance it had, and
because fixed-root links against a static table generate no contact pairs this
interpenetrates **silently**; it does not raise.

`tasks/cabinet_pouring/cabinet_pouring_long.py`, `tasks/drawer_plug/drawer_plug_long.py`, and `tasks/nut_bolt/nut_bolt_long.py`
therefore pick their base pose by trolley footprint
(`COMPACT_TROLLEY_BASE_POSE = -0.76` vs `FULL_TROLLEY_BASE_POSE = -0.87`, keyed on the
agent's `mobile_base_horizontal_scale`), not by arm count; every registered agent declares
the full-size trolley.

### CabinetPour task-object collision assets

`tasks/assets/task_object_meshes/` is gitignored and is distributed with the
dataset repository. CabinetPour bottle 4403 loads
`cabinet_pour/bottle_4403/body_col_profiled/*.obj` (the particle-container
shell) and `body_col_hand/*.obj` (the solids the hand touches) in addition to
its visual, lid and metadata files. Both directories ship with the assets; they
were generated once from the visual mesh by a tool that is not part of this
repository. The earlier `body_col_proc` radial envelope (up to 4.85 mm of
invisible outer collision and overlapping contact cells) is not shipped.

### Dual-arm

| File | Hands | Notes |
|------|-------|-------|
| `panda_dual_gripper_mobile_base.urdf` | 2× Franka gripper | PartNet 22870 trolley, upright symmetric arms; caster joints fixed |
| `panda_dual_sharpa_hand_mobile_base.urdf` | 2× Sharpa Wave | PartNet 22870 trolley, upright symmetric arms; caster joints fixed |
| `panda_dual_wuji_hand_mobile_base.urdf` | 2× Wuji | PartNet 22870 trolley, upright symmetric arms; caster joints fixed |

## Conventions

### Naming

- Single-arm files: `panda_<hand>_hand_right.urdf`
- Dual-arm files:   `panda_dual_<hand>_<base>.urdf`
- Mesh dirs:        `<hand>_hand_meshes/` (or `<hand>_description/` for stock arms)

### Kinematic chain — single arm

```
panda_link0..7 → panda_link8 → [optional adapter] → <hand_root_link> → fingers
```

The hand root link varies per hand:

| Hand | Root link | Has built-in mount geometry? |
|------|-----------|------------------------------|
| Sharpa  | `right_hand_wrist` | yes (wrist link from `_with_wrist` variant) |
| Wuji    | `hand_docking_link` → `right_palm_link` | yes (adapter from `docking/` upstream subdir) |

When integrating a new hand, **always include any wrist/adapter geometry the
manufacturer ships**.

### Kinematic chain — dual arm

```
mobile_base (world root at the trolley top center)
├── mobile_base_origin → fixed PartNet-Mobility 22870 trolley geometry
├── mobile_base_to_left_panda  → left_panda_link0..8  → left hand
└── mobile_base_to_right_panda → right_panda_link0..8 → right hand
```

The three dual-arm variants use the same 22870 asset and uniform
`0.726998794452` scale. The required PartNet-Mobility 22870 meshes and textures
are vendored under `tasks/assets/mobile_base_meshes`. The
caster joints are fixed so they do not add uncontrolled DOFs to the
manipulator.

The single-arm variants are generated
from the dual URDF at full scale with the arm left at `y = ±0.22` (see above), so their
root assembly is:

```
mobile_base (world root at the trolley top center)
├── mobile_base_origin → fixed PartNet-Mobility 22870 trolley geometry (full scale)
└── mobile_base_to_panda → panda_link0..8 → <one hand chain>   # origin 0 ±0.22 0
```

Every base keeps the same `0.91964257 m` height and the same
top-center root frame, so tasks may use the same table/floor height and
`ROBOT_BASE_POSE.z` convention.

On the dual URDFs, all link and joint names on the left arm are prefixed `left_` and on
the right arm `right_`. This namespacing avoids collision with link names defined inside
each hand's upstream URDF. The generated single-arm URDFs keep that prefix on the hand
chain only, matching the older hand-authored single-arm assets.

### Mesh path convention

All mesh references in the combined URDFs use **relative paths from the URDF
file's location** (`tasks/assets/urdf/`):

```xml
<mesh filename="../<hand>_hand_meshes/visual/some_link.obj"/>
```

Never use `package://` paths — those require a ROS catkin environment that we
don't run.

## Adding a new hand

1. Clone the upstream repo into a scratch dir.
2. Copy meshes into `tasks/assets/<hand>_hand_meshes/`.
3. **If the manufacturer ships a wrist / forearm / adapter piece, include it.**
   The `_meshes/` README should explain why.
4. Generate `tasks/assets/urdf/panda_<hand>_hand_right.urdf` by combining the
   Panda arm prefix with the upstream hand URDF, rewriting mesh paths.
5. Validate with the XML + mesh-existence check (see the howto doc).
6. For dual-arm support, mirror the same hand into `left_*` / `right_*`
   prefixed links and attach to the mobile base.

## Upstream licenses

Each `<hand>_meshes/` directory inherits the license of its upstream
repository. See the per-hand README for the authoritative source. Mesh files
are redistributed in this repo for reproducibility; check upstream LICENSEs
before redistributing further.


### 2026-08-26 · Wuji / XHand angled (90° adapter) variants

`panda_{wuji,xhand}_hand_{right,left}_angled_mobile_base.urdf` and `panda_dual_{wuji,xhand}_hand_angled_mobile_base.urdf` are **generated** at import time by `tasks/robots/angled_mount.py` from the straight-mount URDFs in the table above (the fixed flange-to-hand joint is replaced by the same two-segment elbow bracket Sharpa uses plus each hand's measured mounting rotation; for Wuji/XHand the hand root is then extrapolated along the bracket's second segment by the hub depth). Do not hand-edit these generated files; change `ANGLED_MOUNTS` / `HUB_DEPTH` instead. Project rule: every task environment whose end effector is a hand uses the `*_angled_mobile_base` uid.

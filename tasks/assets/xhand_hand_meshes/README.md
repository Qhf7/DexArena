# XHand1 Hand Meshes

Mesh files for the RobotEra **XHand1** dexterous hand.

## Right hand (`right_hand_*.STL`) — vendor

Sourced from the official RobotEra XHand1 URDF package (v1.3), via the
standalone MIT-licensed mirror `wengmister/dex-urdf-plus`. Used by:

- `tasks/assets/urdf/panda_xhand_hand_right.urdf` (single arm, right hand)
- `tasks/assets/urdf/panda_xhand_hand_right_handonly.urdf` (dex-retargeting)

## Left hand (`left_hand_*.STL`) — generated mirror

RobotEra ships **right-hand assets only**. The left meshes here are generated,
not vendor files:

```bash
python -m data_gen.teleop.make_mirrored_hand --hand xhand
```

Each mesh is reflected `y -> -y` in its own frame with the triangle winding
reversed (outward normals preserved), matching the `y -> -y` mirror the tool
applies to `panda_xhand_hand_right_handonly.urdf` to produce
`panda_xhand_hand_left_handonly.urdf`. That reflection plane is the convention
measured off the vendor left hands we already ship (sharpa, wuji).

Do **not** drive a left hand with the right-hand URDF/retargeting config: the
thumb ends up on the wrong side of the palm and every retargeted grasp is
mirror-broken. `tests/test_hand_retargeting_chirality.py` pins this (chirality
sign flip + fingertips landing within 0.05 mm of the mirrored right solution).

Regenerate the left meshes after any change to the right ones.

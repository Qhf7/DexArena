# XHand1 Hand Meshes

Mesh files for the RobotEra **XHand1** dexterous hand.

## Right hand (`right_hand_*.STL`) — vendor

Sourced from the official RobotEra XHand1 URDF package (v1.3), via the
standalone MIT-licensed mirror `wengmister/dex-urdf-plus`. Used by the
right-hand and dual-arm XHand URDFs under `tasks/assets/urdf/`
(`panda_xhand_hand_right*.urdf`, `panda_dual_xhand_hand_*.urdf`).

## Left hand (`left_hand_*.STL`) — generated mirror

RobotEra ships **right-hand assets only**. The left meshes here are generated,
not vendor files: each mesh is reflected `y -> -y` in its own frame with the
triangle winding reversed (outward normals preserved). That reflection plane is
the convention measured off the vendor left hands we already ship (sharpa, wuji).
The left-hand URDFs (`panda_xhand_hand_left_*.urdf`, `panda_dual_xhand_hand_*.urdf`)
reference these meshes.

Do **not** drive a left hand with the right-hand URDF: the thumb ends up on the
wrong side of the palm and every grasp is mirror-broken.

Regenerate the left meshes with the same reflection after any change to the right ones.

"""Set a body's mass without inheriting an unphysical inertia.

Rescaling inertia in proportion to mass is only valid when the ORIGINAL inertia
was itself consistent with the original mass. That assumption fails for any
link whose inertia sat on SAPIEN's minimum-inertia floor (1e-6 kg*m^2) while
its mass came from a near-zero asset density: the ratio new_mass/old_mass is
then meaningless and the scaled inertia can be thousands of times too large.

So: scale proportionally, then check the result is realizable against the body's
own collision geometry, and fall back to a uniform-solid-box inertia when it is
not. Both branches produce a valid rigid body; the first keeps the mesh's shape
information, the second is a defensible approximation.
"""

from __future__ import annotations

import numpy as np


def _pose_matrix(pose):
    w, x, y, z = np.asarray(pose.q, dtype=float)
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return rotation, np.asarray(pose.p, dtype=float)


def _shape_points(shape):
    kind = type(shape).__name__.replace("PhysxCollisionShape", "")
    if kind == "Box":
        h = np.asarray(shape.half_size, dtype=float)
        return np.array([
            [sx * h[0], sy * h[1], sz * h[2]]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ])
    if kind == "Sphere":
        r = float(shape.radius)
        return np.array([[-r, -r, -r], [r, r, r]])
    if kind == "Capsule":
        r, half = float(shape.radius), float(shape.half_length)
        return np.array([[-(half + r), -r, -r], [half + r, r, r]])
    try:
        return (np.asarray(shape.vertices, dtype=float)
                * np.asarray(shape.scale, dtype=float))
    except Exception:
        return None


def body_extent(component) -> np.ndarray | None:
    """Axis-aligned size of the body's collision geometry, in the body frame."""
    lo = hi = None
    try:
        shapes = component.get_collision_shapes()
    except Exception:
        return None
    for shape in shapes:
        points = _shape_points(shape)
        if points is None or len(points) == 0:
            continue
        try:
            rotation, translation = _pose_matrix(shape.get_local_pose())
        except Exception:
            rotation, translation = np.eye(3), np.zeros(3)
        world = points @ rotation.T + translation
        lo = world.min(0) if lo is None else np.minimum(lo, world.min(0))
        hi = world.max(0) if hi is None else np.maximum(hi, world.max(0))
    if lo is None:
        return None
    return np.maximum(hi - lo, 1e-6)


def uniform_box_inertia(mass: float, extent) -> np.ndarray:
    """Principal inertia of a uniform solid box of this mass and size."""
    a, b, c = (float(v) for v in extent)
    k = float(mass) / 12.0
    return np.array([k * (b * b + c * c), k * (a * a + c * c), k * (a * a + b * b)])


def _is_realizable(inertia, mass: float, extent) -> bool:
    inertia = np.asarray(inertia, dtype=float)
    if not np.all(np.isfinite(inertia)) or np.any(inertia <= 0):
        return False
    # Triangle inequality: no principal moment may exceed the sum of the others.
    ix, iy, iz = inertia
    if ix + iy < iz or iy + iz < ix or iz + ix < iy:
        return False
    # Upper bound: all the mass at the far corner of the bounding box.
    return bool(inertia.max() <= float(mass) * float(np.max(extent)) ** 2 * 1.05)


def set_mass(component, mass: float) -> str:
    """Assign an explicit mass and a realizable inertia.

    Returns "scaled" or "box", naming which branch produced the inertia, so
    callers and audits can tell the two apart.
    """
    mass = float(mass)
    extent = body_extent(component)
    current = float(getattr(component, "mass", 0.0) or 0.0)
    scaled = None
    if current > 0:
        scaled = np.asarray(component.inertia, dtype=float) * (mass / current)
    component.mass = mass
    if extent is None:
        if scaled is not None:
            component.inertia = scaled
        return "scaled"
    if scaled is not None and _is_realizable(scaled, mass, extent):
        component.inertia = scaled
        return "scaled"
    component.inertia = uniform_box_inertia(mass, extent)
    return "box"

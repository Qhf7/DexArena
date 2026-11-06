"""Velocity ceilings for free task objects (never the robot).

A rigid task object that ends up interpenetrating a finger, a rack wall or the
table gets shot out by PhysX's depenetration impulse; with the engine default
(unbounded) that impulse is what turns a 3 mm overlap into a body flying off the
table at several m/s. The frozen families (drawer_plug, juice, package_opening,
meal_preparation, office) cap it at 0.5 m/s and the linear velocity at 3 m/s;
the same ceilings are applied here so the newer families (furniture, pipette,
table_clean) behave the same way. The values are ceilings, not dynamics: a
body that is not being crushed never touches them.
"""
from __future__ import annotations

import sapien

MAX_DEPENETRATION_VELOCITY = 0.5   # m/s
MAX_LINEAR_VELOCITY = 3.0          # m/s


def set_task_object_energy_fence(task_object):
    """Apply the ceilings to an Actor or Articulation (all links), CPU/GPU alike."""
    items = task_object.get_links() if hasattr(task_object, "get_links") else (task_object,)
    for item in items:
        for raw in getattr(item, "_objs", []):
            component = raw
            if not hasattr(component, "set_max_depenetration_velocity"):
                component = raw.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if component is None:
                continue
            component.set_max_depenetration_velocity(MAX_DEPENETRATION_VELOCITY)
            component.set_max_linear_velocity(MAX_LINEAR_VELOCITY)

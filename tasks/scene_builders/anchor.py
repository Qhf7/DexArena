"""Decide where at a scene's work surface the rig should stand, and where to
look at it from.

A work surface only fixes three of the four numbers the workstation frame needs:
the two that put it on the surface, and the height. The fourth — which side the
robot approaches from — is a property of the room, not of the surface.

Both questions are answered by raycasting the scene's collision geometry rather
than by reasoning over bounding boxes. Bounding boxes cannot answer them: a
dataset that ships a whole apartment as one mesh (ReplicaCAD's stage) collapses
every wall into a single box covering the floor plan, so box-based free space
reports the void *outside* the building as the roomiest spot in the scene. A ray
stops at the wall.

Rays hit whatever is in the scene, including the robot, so park it out of the way
before calling these.
"""
from __future__ import annotations

import numpy as np
import sapien

# AABBs wider than this in x or y are room shells (floors, stage meshes,
# ceilings) rather than furniture, and are skipped: one of them would otherwise
# report every position in the scene as blocked.
MAX_OBSTACLE_EXTENT = 8.0


def robot_entity_ids(env, scene_idx: int) -> set[int]:
    """Entity ids belonging to the agent in one sub-scene, to skip as obstacles.

    The agent is loaded before the scene builder runs, so it is already sitting
    in the sub-scene when anchors are computed.
    """
    ids: set[int] = set()
    agent = getattr(env, "agent", None)
    if agent is None:
        return ids
    robots = getattr(agent, "robots", None) or [agent.robot]
    for robot in robots:
        objs = robot._objs
        art = objs[scene_idx] if scene_idx < len(objs) else objs[0]
        for link in art.links:
            ids.add(id(link.entity))
    return ids


def entity_aabb(entity) -> np.ndarray | None:
    """Tight world AABB of one entity's render geometry, or None."""
    rbc = entity.find_component_by_type(sapien.render.RenderBodyComponent)
    if rbc is None or len(rbc.render_shapes) == 0:
        return None
    try:
        return np.asarray(rbc.compute_global_aabb_tight(), dtype=np.float64)
    except Exception:
        return None


def collect_obstacle_aabbs(
    sub_scene: sapien.Scene,
    exclude_ids: set[int] | None = None,
    height_band: tuple[float, float] = (0.05, 1.6),
) -> np.ndarray:
    """Boxes of everything that would block the rig, as (N, 2, 2) xy extents.

    Only geometry that intersects ``height_band`` above the floor counts, which
    drops floors and ceilings without needing to name them.
    """
    exclude_ids = exclude_ids or set()
    boxes = []
    for entity in sub_scene.entities:
        if id(entity) in exclude_ids:
            continue
        aabb = entity_aabb(entity)
        if aabb is None:
            continue
        lo, hi = aabb
        if hi[2] < height_band[0] or lo[2] > height_band[1]:
            continue
        if np.any(hi[:2] - lo[:2] > MAX_OBSTACLE_EXTENT):
            continue
        boxes.append([lo[:2], hi[:2]])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 2, 2)


def clearance(point: np.ndarray, boxes: np.ndarray) -> float:
    """Distance from an xy point to the nearest box, 0 inside one."""
    if len(boxes) == 0:
        return np.inf
    lo, hi = boxes[:, 0, :], boxes[:, 1, :]
    outside = np.maximum(np.maximum(lo - point, point - hi), 0.0)
    return float(np.linalg.norm(outside, axis=1).min())


def room_center(boxes: np.ndarray) -> np.ndarray | None:
    """Middle of the furniture, as a stand-in for "inside the room".

    Datasets that ship a room as one mesh (ReplicaCAD's stage) give no usable
    wall geometry, so free space alone cannot tell inside from outside — the
    emptiest floor in such a scene is the void beyond the walls. Staying near
    the furniture is what keeps a placement indoors.
    """
    if len(boxes) == 0:
        return None
    return 0.5 * (boxes[:, 0, :] + boxes[:, 1, :]).mean(axis=0)


def ray_distance(physx_system, origin: np.ndarray, direction: np.ndarray, limit: float) -> float:
    """How far a ray travels from ``origin`` before hitting anything."""
    hit = physx_system.raycast(
        np.asarray(origin, dtype=np.float32), np.asarray(direction, dtype=np.float32), limit
    )
    return limit if hit is None else float(hit.distance)


def open_distance(physx_system, origin: np.ndarray, bearing: float, limit: float) -> float:
    """``ray_distance`` for a horizontal ray at a compass bearing."""
    return ray_distance(physx_system, origin, [np.cos(bearing), np.sin(bearing), 0.0], limit)


SIDES = ((0.0, (1.0, 0.0)), (np.pi, (-1.0, 0.0)), (np.pi / 2, (0.0, 1.0)), (-np.pi / 2, (0.0, -1.0)))


def surface_clutter(
    sub_scene,
    exclude_ids: set[int] | None,
    top_z: float,
    sits_within: float = 0.12,
    max_extent: float = 1.2,
) -> np.ndarray:
    """Boxes of whatever is *standing on* a surface, from render geometry.

    Rays cannot answer this. These datasets' collision meshes are nothing like
    what they draw — an AI2THOR laptop collides as a 1.4 cm slab while its screen
    stands 29 cm tall, so a ray skims over it and the task spawns its object
    behind a screen that hides it completely.

    Selected by where an object's *underside* is: resting on the surface, within
    ``sits_within``. Anything that merely passes through the height band is not
    clutter — a chair tucked under a table rises through it from the floor, and
    counting those left half the library with nowhere to work.
    """
    exclude_ids = exclude_ids or set()
    boxes = []
    for entity in sub_scene.entities:
        if id(entity) in exclude_ids:
            continue
        aabb = entity_aabb(entity)
        if aabb is None:
            continue
        lo, hi = aabb
        if not top_z - 0.05 <= lo[2] <= top_z + sits_within:
            continue
        if hi[2] <= top_z + 0.02:  # as flat as the surface itself
            continue
        if np.any(hi[:2] - lo[:2] > max_extent):
            continue
        boxes.append([lo[:2], hi[:2]])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 2, 2)


def pick_surface_spot(
    aabb: np.ndarray,
    physx_system,
    reach: float,
    inset: float,
    cart_footprint: float,
    cart_overhang: float = 0.30,
    clutter: np.ndarray | None = None,
    spawn_radius: float = 0.14,
    headroom: float = 0.7,
    samples: int = 11,
    end_margin: float = 0.2,
    probe_height: float = 0.25,
    flatness_tol: float = 0.04,
) -> tuple[float, float, float, float] | None:
    """Workstation frame on a work surface, or None if it has no usable spot.

    A surface's bounding box says almost nothing about where on it you can work.
    An AI2THOR kitchen counter is one L-shaped actor whose box centre is in the
    empty corner; it has a sink cut into it and a stove standing on it, and its
    far side is against a wall. So instead of trusting the box, this samples
    points along each of the four sides and asks the geometry four questions:

    * rays dropped over the spawn disc must all hit the surface at exactly
      ``probe_height``. Long means a hole (a sink) or thin air (the L's missing
      corner); short means something tall is standing there.
    * rays fired sideways just above the surface must clear the whole disc, so
      nothing already sitting there is in the way. Dropped rays alone are not
      enough: one aimed into a cereal bowl hits the inside of its base, which is
      within tolerance of the table it stands on, and the task spawned its can
      inside the bowl. The disc, not just its centre, because the object lands
      anywhere within it.
    * ``clutter`` boxes, which catch what the rays cannot -- see
      :func:`surface_clutter`.
    * a ray fired straight up must clear ``headroom``, or the arm is working
      under a pendant lamp or a wall cabinet.
    * a ray fired outward *from the surface's near edge* must run the length of
      the trolley, or the rig cannot stand there. From the edge, not from the
      workspace: the robot is a fixed ``reach`` behind the workspace, so working
      deeper into a counter drives the cart into its base cabinet, and a ray cast
      from the workspace passes clean over that cabinet at worktop height.

    Returns the roomiest spot that passes, with yaw pointing from the robot into
    the surface.
    """
    lo, hi = np.asarray(aabb[0], dtype=np.float64), np.asarray(aabb[1], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    top_z = float(hi[2])
    disc = [(0.0, 0.0)] + [
        (spawn_radius * np.cos(a), spawn_radius * np.sin(a))
        for a in np.arange(4) * (np.pi / 2)
    ]

    # Try the nearest reach first and only give ground when nothing there works:
    # a dining table can be covered in laptops and statues near the edge and
    # still have a clear far end.
    # How deep in the workspace can sit before the trolley, a fixed `reach`
    # behind it, backs into the base of the surface.
    max_inset = max(reach - cart_overhang, inset)
    for extra in (0.0, 0.15, 0.30, 0.45):
        if inset + extra > max_inset:
            break
        best, best_score = None, -np.inf
        for yaw, inward in SIDES:
            inward = np.asarray(inward, dtype=np.float64)
            along = np.array([-inward[1], inward[0]])
            depth = float(abs(inward @ half[:2]))
            length = float(abs(along @ half[:2]))
            step = min(inset + extra, depth)
            if step > depth:
                continue
            base = center[:2] - inward * (depth - step)
            reachable = max(length - end_margin, 0.0)
            for offset in np.linspace(-reachable, reachable, samples):
                spot = base + along * offset
                if clutter is not None and clearance(spot, clutter) < spawn_radius:
                    continue
                if not _spawn_disc_is_clear(
                    physx_system, spot, top_z, disc, spawn_radius, probe_height, flatness_tol
                ):
                    continue
                over = np.array([spot[0], spot[1], top_z + 0.02])
                if ray_distance(physx_system, over, [0.0, 0.0, 1.0], headroom) < headroom:
                    continue
                edge = spot + inward * step
                beside = np.array([edge[0], edge[1], top_z + probe_height])
                floor = open_distance(physx_system, beside, yaw + np.pi, cart_footprint)
                if floor < cart_footprint:
                    continue
                if floor > best_score:
                    best, best_score = (spot[0], spot[1], top_z, yaw), floor
        if best is not None:
            break

    if best is None:
        return None
    return float(best[0]), float(best[1]), float(best[2]), float(best[3])


def _spawn_disc_is_clear(physx_system, spot, top_z, disc, radius, probe_height, tol) -> bool:
    """True when the whole spawn disc is bare surface at the expected height."""
    for dx, dy in disc:
        origin = np.array([spot[0] + dx, spot[1] + dy, top_z + probe_height])
        drop = ray_distance(physx_system, origin, [0.0, 0.0, -1.0], probe_height * 4)
        if abs(drop - probe_height) > tol:
            return False
    grazing = np.array([spot[0], spot[1], top_z + 0.03])
    for bearing in np.arange(8) * (np.pi / 4):
        if open_distance(physx_system, grazing, float(bearing), radius) < radius:
            return False
    return True


def pick_surface_side(
    aabb: np.ndarray,
    physx_system,
    reach: float,
    inset: float,
    probe_height: float = 0.25,
) -> tuple[float, float, float, float]:
    """Which side of a surface to work from, ignoring what is on top of it.

    Kept for surfaces known to be clear rectangles. Prefer
    :func:`pick_surface_spot`, which also checks the spot itself.
    """
    lo, hi = np.asarray(aabb[0], dtype=np.float64), np.asarray(aabb[1], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    limit = 2 * reach

    best = None
    for yaw, inward in SIDES:
        inward = np.asarray(inward, dtype=np.float64)
        depth = float(abs(inward @ half[:2]))
        step = min(inset, depth)
        origin = center[:2] - inward * (depth - step)
        probe = np.array([origin[0], origin[1], hi[2] + probe_height])
        open_floor = open_distance(physx_system, probe, yaw + np.pi, limit)
        if best is None or open_floor > best[0]:
            best = (open_floor, origin, yaw)

    _, origin, yaw = best
    return float(origin[0]), float(origin[1]), float(hi[2]), float(yaw)

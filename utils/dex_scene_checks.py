"""Native-table geometry checks used by the environment preview, never policies."""
from __future__ import annotations


def check_robot_table_clearance(env, surface):
    """Prove separation of the trolley and table using both visible and collision bounds.

    The supported layouts approach the table along +X. Positive separation on
    this axis rules out trolley intersections with the whole table, including
    its apron and legs, without relying on physics collision-filter settings.
    """
    import numpy as np
    import sapien

    def bounds(entity):
        boxes = []
        for cls in (sapien.render.RenderBodyComponent, sapien.physx.PhysxRigidBaseComponent):
            component = entity.find_component_by_type(cls)
            if component is None:
                continue
            if isinstance(component, sapien.physx.PhysxRigidBaseComponent) and not component.get_collision_shapes():
                continue
            boxes.append(np.asarray(component.compute_global_aabb_tight()))
        if not boxes or not np.isfinite(boxes).all():
            raise AssertionError(f"Missing or invalid geometry bounds: {entity.name}")
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)

    table_lo, table_hi = bounds(surface)
    trolley = []
    for link in env.unwrapped.agent.robot.get_links():
        if "mobile_base_link" not in link.name:
            continue
        for obj in link._objs:
            lo, hi = bounds(obj.entity)
            trolley.append({"link": link.name, "aabb": [lo.tolist(), hi.tolist()],
                            "separation_m": float(table_lo[0] - hi[0])})
    if not trolley:
        raise AssertionError("No mobile-base geometry found for table-clearance check")
    nearest = min(trolley, key=lambda row: row["separation_m"])
    gap = nearest["separation_m"]
    errors = []
    if gap < .005:
        errors.append(f"Robot trolley overlaps or is too close to native table: "
                      f"{nearest['link']}, X separation {gap:.4f} m (required >= 0.005 m)")
    return {"minimum_separation_m": gap, "required_separation_m": .005,
            "table_aabb": [table_lo.tolist(), table_hi.tolist()], "trolley_links": trolley,
            "errors": errors}


def inspect_native_scene(env):
    import numpy as np
    import sapien
    u = env.unwrapped
    px = sapien.physx
    errors = []
    for name in u._disabled_original_surfaces:
        for entity in u.scene.actors[name]._objs:
            render = entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if render is not None and render.is_enabled:
                errors.append(f"Original {name} is still rendered")
            body = entity.find_component_by_type(px.PhysxRigidBaseComponent)
            if body is not None and any(any(s.get_collision_groups()) for s in body.get_collision_shapes()):
                errors.append(f"Original {name} still has active collisions")
    candidates = []
    for entity in u._native_scene_entities:
        render = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        body = entity.find_component_by_type(px.PhysxRigidBaseComponent)
        if render is None or body is None or not body.get_collision_shapes():
            continue
        lo, hi = np.asarray(render.compute_global_aabb_tight())
        if (abs(hi[2]) < .015 and lo[2] < -.025 and
                np.all(lo[:2] < [.05, .05]) and np.all(hi[:2] > [-.05, -.05])):
            candidates.append((float(np.prod(hi[:2]-lo[:2])), entity, lo, hi))
    if not candidates:
        raise AssertionError("No native collidable tabletop at the task origin")
    _, surface, lo, hi = max(candidates, key=lambda v:v[0])
    clearance = check_robot_table_clearance(env, surface)
    errors.extend(clearance["errors"])
    edge_alignment = u._native_edge_alignment
    edge_error = float(lo[0] - edge_alignment["original_near_edge_m"])
    if abs(edge_error) > 1e-5:
        errors.append(f"Native table edge drifted from original eval: {edge_error} m")
    from utils.dex_scene_eval import HEIGHT_TOLERANCE_M
    height_error = float(hi[2] - edge_alignment["original_tabletop_z_m"])
    if abs(height_error) > HEIGHT_TOLERANCE_M:
        errors.append(f"Native tabletop height drifted from original eval: {height_error} m")
    # Support uses the actual collidable tabletop polygon, including rotations.
    from scipy.spatial import ConvexHull
    vertices = []
    body = surface.find_component_by_type(px.PhysxRigidBaseComponent)
    for shape in body.get_collision_shapes():
        if isinstance(shape, px.PhysxCollisionShapeBox):
            import itertools
            local = np.asarray(list(itertools.product((-1, 1), repeat=3))) * shape.half_size
        elif isinstance(shape, (px.PhysxCollisionShapeConvexMesh, px.PhysxCollisionShapeTriangleMesh)):
            local = np.asarray(shape.vertices) * shape.scale
        else:
            continue
        T = (surface.pose * shape.local_pose).to_transformation_matrix()
        vertices.extend(local @ T[:3, :3].T + T[:3, 3])
    vertices = np.asarray(vertices)
    top_vertices = vertices[vertices[:, 2] > vertices[:, 2].max()-.01, :2]
    hull = ConvexHull(top_vertices)
    supported = []
    task_aabbs = {}
    robot_ids = {link._objs[0].entity.global_id for link in u.agent.robot.get_links()}
    task_entities = []
    articulated_moving = set()
    for actor in u.scene.actors.values():
        task_entities.extend(actor._objs)
    for art in u.scene.articulations.values():
        for link in art.get_links():
            for item in link._objs:
                task_entities.append(item.entity)
                node = item
                while node.parent is not None:
                    if node.joint.type != "fixed":
                        articulated_moving.add(item.entity.global_id)
                        break
                    node = node.parent
    seen = set()
    for entity in task_entities:
        if entity.global_id in seen or entity.global_id in robot_ids:
            continue
        seen.add(entity.global_id)
        render = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        body = entity.find_component_by_type(px.PhysxRigidBaseComponent)
        if render is None or not render.is_enabled or body is None or not body.get_collision_shapes():
            continue
        if not any(any(s.get_collision_groups()[:2]) for s in body.get_collision_shapes()):
            continue
        bottom, top = np.asarray(render.compute_global_aabb_tight())
        if np.isfinite(bottom).all() and np.isfinite(top).all():
            task_aabbs[entity.name] = [bottom.tolist(), top.tolist()]
        else:
            errors.append(f"Non-finite task geometry: {entity.name}")
        if entity.global_id in articulated_moving:
            continue  # Hinged doors/drawers are supported by their fixture.
        # Only bodies resting on the tabletop; held objects and raised fixture
        # links (doors, lids) need not be supported by the tabletop footprint.
        if not (-.025 <= bottom[2] <= .025 and top[2] > .002 and np.linalg.norm((bottom[:2]+top[:2])/2)<2):
            continue
        corners = np.asarray([[x,y] for x in (bottom[0],top[0]) for y in (bottom[1],top[1])])
        margin = float(-(corners @ hull.equations[:, :2].T + hull.equations[:, 2]).max())
        supported.append({"entity":entity.name,"aabb":[bottom.tolist(),top.tolist()],"edge_margin":margin})
        if margin < -.02:
            errors.append(f"Task object beyond native tabletop: {entity.name} ({margin:.3f} m)")
    return {"surface":surface.name,"surface_aabb":[lo.tolist(),hi.tolist()],
            "surface_polygon":top_vertices[hull.vertices].tolist(),
            "native_entities":len(u._native_scene_entities),
            "original_surfaces_disabled":list(u._disabled_original_surfaces),
            "robot_table_clearance":clearance,
            "robot_root_pose":u.agent.robot.pose.raw_pose.cpu().tolist(),
            "robot_qpos":u.agent.robot.qpos.cpu().tolist(),
            "wrist_poses":{link.name:link.pose.raw_pose.cpu().tolist()
                           for link in u.agent.robot.get_links()
                           if link.name in ("left_hand_wrist", "right_hand_wrist")},
            "tabletop_height_error_m":height_error,
            "original_eval_edge_alignment":{**edge_alignment, "error_m":edge_error},
            "table_supported_objects":supported,
            "task_aabbs":task_aabbs,"errors":errors}

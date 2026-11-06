"""Opt-in native-surface evaluation, outside the frozen task import closure.

The original table is disabled. Tasks use the selected background table and
collision geometry (an authored kitchen surface or the prepared laboratory table). Task objects and logic remain in their existing tabletop coordinate
frame; the room is aligned to that frame without changing the robot reset pose.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path


@dataclass(frozen=True)
class BackgroundSpec:
    scene_id: str
    label: str
    # Measured with FullSceneBuilder; x, y, surface height, yaw, floor height.
    anchor: tuple[float, float, float, float, float]


BACKGROUND_VERSION = "dex_native_surface_v6"
WORKSPACE_HEIGHT_M = .9196429
HEIGHT_TOLERANCE_M = .001
BACKGROUNDS = (
    BackgroundSpec("robocasa_l3_s2", "Blue kitchen", (3.4499967, -3.2499874, .9200000, -3.1415890, 0.0)),
    BackgroundSpec("robocasa_l3_s1", "Light wood kitchen island", (3.4499967, -3.2499874, .9200000, -3.1415890, 0.0)),
    BackgroundSpec("robocasa_l3_s6", "Warm wood kitchen stone island", (3.4499967, -3.2499874, .9200000, -3.1415890, 0.0)),
)
BACKGROUND_IDS = tuple(s.scene_id for s in BACKGROUNDS)
# Extra scenes are opt-in: keep the published 30 x 3 benchmark matrix stable.
OPTIONAL_BACKGROUNDS = (
    BackgroundSpec("labutopia_lab003", "LabUtopia empty laboratory",
                   (0.0, 0.0, WORKSPACE_HEIGHT_M, 0.0, 0.0)),
)
AVAILABLE_BACKGROUNDS = BACKGROUNDS + OPTIONAL_BACKGROUNDS
AVAILABLE_BACKGROUND_IDS = tuple(s.scene_id for s in AVAILABLE_BACKGROUNDS)


def background_manifest(scene_id: str | None) -> dict | None:
    if scene_id is None:
        return None
    spec = next((s for s in BACKGROUNDS + OPTIONAL_BACKGROUNDS if s.scene_id == scene_id), None)
    if spec is None:
        raise ValueError(f"Unknown evaluation background {scene_id!r}; choose {AVAILABLE_BACKGROUND_IDS}")
    height_delta = spec.anchor[2] - spec.anchor[4] - WORKSPACE_HEIGHT_M
    if abs(height_delta) > HEIGHT_TOLERANCE_M:
        raise ValueError(f"Native surface height differs from the collection workspace: {height_delta} m")
    root = Path(__file__).resolve().parents[1]
    files = ("utils/dex_scene_eval.py", "tasks/scene_builders/backgrounds.py",
             "tasks/scene_builders/scene_library.py", "tasks/scene_builders/anchor.py",
             "utils/dex_cameras.py", "utils/dex_camera_rigs.py",
             "utils/camera_configs.py", "utils/dex_scene_checks.py",
             "utils/scene_matrix.py")
    laboratory = scene_id == "labutopia_lab003"
    if laboratory:
        files += ("tasks/scene_builders/labutopia.py", "tasks/scene_builders/labutopia_sources.json",
                  "tasks/scene_builders/convert_labutopia.py")
    return {"version": BACKGROUND_VERSION, **asdict(spec),
            "mode": "prepared_workspace" if laboratory else "native_surface",
            "physics": ("collection_table_support_and_static_room_triangle_mesh" if laboratory
                        else "authored_scene_collision_geometry"),
            "original_table": "render_and_collision_disabled",
            "robot_root_z_offset": 0.0, "robot_reset": "original_pose_and_qpos_unchanged",
            "native_height_delta_m": height_delta, "height_tolerance_m": HEIGHT_TOLERANCE_M,
            "scene_friction": .30, "camera": "original_evaluation_profile",
            "source_sha256": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}}


def _transform(spec):
    import sapien
    x, y, z, yaw, _ = spec.anchor
    if spec.scene_id == "labutopia_lab003":
        return sapien.Pose([0, 0, -z])
    c, sn = math.cos(-yaw), math.sin(-yaw)
    # Same island alignment as the referenced apartment task renders: native
    # tabletop at z=0, approximate edge at x=-.425. At reset, align to the
    # original task mesh edge rather than its nominal dimensions.
    return sapien.Pose([-.125 - c*x + sn*y, -sn*x - c*y, -z],
                       [math.cos(-yaw/2), 0, 0, math.sin(-yaw/2)])


def _clone_collision(shape):
    """Copy actual authored collision geometry, including its local pose."""
    import sapien
    px = sapien.physx
    material = px.PhysxMaterial(.30, .30, 0.0)
    cls = type(shape)
    if isinstance(shape, px.PhysxCollisionShapeBox):
        result = cls(shape.half_size, material)
    elif isinstance(shape, px.PhysxCollisionShapeConvexMesh):
        result = cls(shape.vertices, shape.scale, material)
    elif isinstance(shape, px.PhysxCollisionShapeTriangleMesh):
        result = cls(shape.vertices, shape.triangles, shape.scale, material)
    elif isinstance(shape, px.PhysxCollisionShapePlane):
        result = cls(material)
    elif isinstance(shape, px.PhysxCollisionShapeSphere):
        result = cls(shape.radius, material)
    elif isinstance(shape, (px.PhysxCollisionShapeCapsule, px.PhysxCollisionShapeCylinder)):
        result = cls(shape.radius, shape.half_length, material)
    else:
        raise TypeError(f"Unsupported native scene collision: {cls.__name__}")
    result.local_pose = shape.local_pose
    result.contact_offset = shape.contact_offset
    result.rest_offset = shape.rest_offset
    result.set_collision_groups([1, 1, 0, 0])
    return result


def _clone_entity(template):
    import sapien
    entity = sapien.Entity()
    entity.name = template.name
    entity.set_pose(template.pose)
    render = template.find_component_by_type(sapien.render.RenderBodyComponent)
    if render is not None:
        entity.add_component(render.clone())
    body = template.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
    if body is not None:
        copy = sapien.physx.PhysxRigidStaticComponent()
        for shape in body.get_collision_shapes():
            copy.attach(_clone_collision(shape))
        entity.add_component(copy)
    return entity


def _disable_original_surfaces(env):
    import sapien
    disabled = []
    for name, actor in env.scene.actors.items():
        if name != "table-workspace" and not (name.endswith("ground") and "scene" not in name):
            continue
        for entity in actor._objs:
            render = entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if render is not None:
                render.disable()
            body = entity.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
            if body is not None:
                for shape in body.get_collision_shapes():
                    shape.set_collision_groups([0, 0, 0, 0])
        disabled.append(name)
    if "table-workspace" not in disabled:
        raise RuntimeError("Original task table was not found; refusing double-table scene")
    return disabled


def _scene_spec(scene_id):
    """Resolve evaluation-only island styles without changing the collection library."""
    from tasks.scene_builders.scene_library import SceneSpec, get_scene_spec
    if scene_id in ("robocasa_l3_s1", "robocasa_l3_s6"):
        return SceneSpec(scene_id, "robocasa", 3 * 12 + int(scene_id.rsplit("s", 1)[1]))
    return get_scene_spec(scene_id)


def _align_native_table_edge(env):
    """Match the original task's visible working edge after its reset placement.

    The stock table's visible mesh is offset from its collision box. Read the
    real mesh instead of deriving the working edge from nominal table depth.
    Only authored room entities move; task objects and the robot stay put.
    """
    import numpy as np
    import sapien
    original = env.scene.actors["table-workspace"]._objs[0]
    original_box = np.asarray(original.find_component_by_type(
        sapien.render.RenderBodyComponent).compute_global_aabb_tight())
    candidates = []
    for entity in env._native_scene_entities:
        render = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        body = entity.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
        if render is None or body is None or not body.get_collision_shapes():
            continue
        lo, hi = np.asarray(render.compute_global_aabb_tight())
        if (abs(hi[2]) < .015 and lo[2] < -.025 and
                np.all(lo[:2] < [.05, .05]) and np.all(hi[:2] > [-.05, -.05])):
            candidates.append((float(np.prod(hi[:2]-lo[:2])), entity, lo))
    if not candidates:
        raise RuntimeError("No native tabletop to align to the original eval edge")
    _, surface, lower = max(candidates, key=lambda row: row[0])
    shift = float(original_box[0, 0] - lower[0])
    for entity in env._native_scene_entities:
        pose = entity.pose
        position = pose.p.copy()
        position[0] += shift
        entity.set_pose(sapien.Pose(position, pose.q))
    actual_box = np.asarray(surface.find_component_by_type(
        sapien.render.RenderBodyComponent).compute_global_aabb_tight())
    height_error = float(actual_box[1, 2] - original_box[1, 2])
    if abs(height_error) > HEIGHT_TOLERANCE_M:
        raise RuntimeError(f"Native tabletop height does not match original eval: {height_error} m")
    error = float(actual_box[0, 0] - original_box[0, 0])
    if abs(error) > 1e-5:
        raise RuntimeError(f"Native working edge does not match original eval: {error} m")
    env._native_scene_translation_x += shift
    env._native_edge_alignment = {
        "reference": "original task table-workspace visible mesh after reset",
        "axis": "X, robot-facing edge", "original_near_edge_m": float(original_box[0, 0]),
        "native_near_edge_m": float(actual_box[0, 0]), "error_m": error,
        "original_tabletop_z_m": float(original_box[1, 2]),
        "native_tabletop_z_m": float(actual_box[1, 2]),
        "tabletop_height_error_m": height_error,
        "room_x_adjustment_m": env._native_scene_translation_x,
        "original_table_aabb": original_box.tolist(),
    }
    env.eval_background_manifest["edge_alignment"] = env._native_edge_alignment
    env.eval_background_manifest["world_from_source_scene"]["position"][0] = (
        env._native_source_transform_x + env._native_scene_translation_x)


def _templates(scene_id, env_id):
    """Build in isolation, then freeze authored fixtures as collidable geometry."""
    import gymnasium as gym
    import numpy as np
    import random
    import sapien
    import torch
    import mani_skill.envs  # noqa: F401
    from tasks.scene_builders.backgrounds import ADAPTERS

    spec = next(s for s in BACKGROUNDS + OPTIONAL_BACKGROUNDS if s.scene_id == scene_id)
    scene_spec = _scene_spec(scene_id)
    rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    source = None
    try:
        source = gym.make("Empty-v1", robot_uids="panda", num_envs=1,
                          obs_mode="none", sim_backend="physx_cpu", render_mode="rgb_array")
        source.reset(seed=0)
        u = source.unwrapped
        ss = u.scene.sub_scenes[0]
        before = {e.global_id for e in ss.entities}
        adapter = ADAPTERS[scene_spec.source](u)
        adapter.build([scene_spec.config])
        transform = _transform(spec)
        templates, hidden = [], []
        for entity in list(ss.entities):
            if entity.global_id in before:
                continue
            render = entity.find_component_by_type(sapien.render.RenderBodyComponent)
            body = entity.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
            if render is None and body is None:
                continue
            copy = _clone_entity(entity)
            copy.name = "eval-scene/" + entity.name
            copy.set_pose(transform * entity.pose)
            templates.append(copy)
        kept = []
        for copy in templates:
            cloned_render = copy.find_component_by_type(sapien.render.RenderBodyComponent)
            if cloned_render is not None:
                lo, hi = np.asarray(cloned_render.compute_global_aabb_tight()).reshape(2, 3)
                prop = (-.05 <= lo[2] <= .45 and np.max(hi[:2]-lo[:2]) < 1.2 and
                        np.linalg.norm((lo[:2]+hi[:2])/2 - [.1, 0]) < 1.6)
                chair = (any(n in copy.name.lower() for n in ("chair", "stool")) and
                         hi[0] > -1.7 and lo[0] < -.35 and hi[1] > -.75 and lo[1] < .75)
                if prop or chair:
                    hidden.append(copy.name)
                    continue
            kept.append(copy)
        templates = kept
        if not templates:
            raise RuntimeError(f"No renderable background entities for {scene_id}")
        return templates, hidden, transform
    finally:
        if source is not None:
            source.close()
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.random.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def install_eval_background(env_id: str, scene_id: str | None) -> dict | None:
    """Install after the camera profile and before gym.make; rebuilt on reset.

    A separate process is recommended per tasks/background pair. Passing None
    explicitly removes a previously installed background from this registry id.
    """
    from mani_skill.utils.registration import REGISTERED_ENVS
    manifest = background_manifest(scene_id)
    entry = REGISTERED_ENVS[env_id]
    base_cls = entry.cls
    if "_dex_background_base" in base_cls.__dict__:
        base_cls = base_cls._dex_background_base
    if scene_id is None:
        entry.cls = base_cls
        return None
    templates, hidden, transform = _templates(scene_id, env_id)
    manifest["world_from_source_scene"] = {"position": transform.p.tolist(), "quaternion_wxyz": transform.q.tolist()}
    manifest["floor_fit"] = "native_height_matches_workspace_within_1mm_robot_unchanged"
    manifest["hidden_background_entities"] = hidden
    manifest["render_entity_count"] = len(templates)

    class BackgroundEnv(base_cls):
        _dex_background_base = base_cls

        def __init__(self, *args, **kwargs):
            if kwargs.get("num_envs", 1) != 1:
                raise ValueError("Background evaluation requires num_envs=1")
            if kwargs.get("sim_backend", "physx_cpu") != "physx_cpu":
                raise ValueError("Background evaluation requires sim_backend='physx_cpu'")
            self.eval_background_manifest = manifest
            super().__init__(*args, **kwargs)

        def _load_scene(self, options):
            super()._load_scene(options)
            ss = self.scene.sub_scenes[0]
            self._native_scene_entities = []
            self._native_scene_translation_x = 0.0
            self._native_source_transform_x = float(transform.p[0])
            for template in templates:
                entity = _clone_entity(template)
                ss.add_entity(entity)
                self._native_scene_entities.append(entity)
            self._disabled_original_surfaces = _disable_original_surfaces(self)

        def _initialize_episode(self, env_idx, options):
            super()._initialize_episode(env_idx, options)
            _align_native_table_edge(self)


    BackgroundEnv.__name__ = base_cls.__name__ + "EvalBackground"
    entry.cls = BackgroundEnv
    return manifest


def make_scene_eval_env(env_id: str, *, scene_background: str, **kwargs):
    """Algorithm-independent factory for the three kitchens and optional scenes.

    Use this in ACT/DP/RL/VLA environment construction in place of gym.make.
    obs_mode may be state or rgb. Controllers, geometry and camera setup are
    identical across algorithms; seeds are supplied to env.reset by the caller.
    """
    import gymnasium as gym
    import tasks  # noqa: F401
    from tasks.config.collection_targets import SUBTASKS, env_ids
    from utils.dex_cameras import install_camera_profile
    if env_id not in env_ids("Sharpa", SUBTASKS):
        raise ValueError(f"{env_id} is not one of the 30 canonical subtasks")
    spec = registered_task_settings(env_id)
    fixed = dict(num_envs=1, sim_backend="physx_cpu", robot_uids=spec["robot_uids"],
                 control_mode=spec["control_mode"], reconfiguration_freq=1,
                 max_episode_steps=spec["max_episode_steps"])
    for key, value in fixed.items():
        if key in kwargs and kwargs[key] != value:
            raise ValueError(f"Fixed scene evaluation requires {key}={value!r}, got {kwargs[key]!r}")
        kwargs[key] = value
    if kwargs.get("sensor_configs", {"width": 256, "height": 256}) != {"width": 256, "height": 256}:
        raise ValueError("Fixed evaluation cameras must be 256 x 256")
    kwargs["sensor_configs"] = {"width": 256, "height": 256}
    kwargs.setdefault("obs_mode", "rgb")
    kwargs.setdefault("render_mode", "rgb_array")
    if kwargs.get("render_backend") == "none":
        raise ValueError("A render backend is required to build scene backgrounds")
    from mani_skill.utils.registration import REGISTERED_ENVS
    original_class = REGISTERED_ENVS[env_id].cls
    try:
        camera = install_camera_profile(env_id, "robodex_fixed")
        manifest = install_eval_background(env_id, scene_background)
        env = gym.make(env_id, **kwargs)
    finally:
        # An ordinary gym.make in the same process must still construct the
        # original task. Live env instances retain their subclass for resets.
        REGISTERED_ENVS[env_id].cls = original_class
    from utils.dex_camera_rigs import describe, family_of
    env.unwrapped.eval_background_manifest = {
        **manifest, "camera_profile": camera,
        "camera_rig": describe(family_of(env_id), env.unwrapped),
        "fixed_environment": fixed,
    }
    return env


def registered_task_settings(env_id: str) -> dict:
    """Use the real task registry for robot/horizon; never a second task list."""
    import tasks  # noqa: F401
    from tasks.config.collection_targets import SUBTASKS, env_ids
    from mani_skill.utils.registration import REGISTERED_ENVS
    if env_id not in env_ids("Sharpa", SUBTASKS):
        raise ValueError(f"{env_id} is not a canonical subtask")
    entry = REGISTERED_ENVS[env_id]
    robot = entry.default_kwargs.get("robot_uids")
    if robot is None:
        # TableClean declares its default in the base constructor rather than
        # the registration decorator. Read that declaration without building it.
        import inspect
        for cls in entry.cls.__mro__:
            parameter = inspect.signature(cls.__init__).parameters.get("robot_uids")
            if parameter is not None and isinstance(parameter.default, str):
                robot = parameter.default
                break
    if not isinstance(robot, str) or not entry.max_episode_steps:
        raise ValueError(f"{env_id} has no unambiguous robot/horizon declaration")
    return {"robot_uids": robot,
            "control_mode": "pd_joint_pos", "max_episode_steps": entry.max_episode_steps}

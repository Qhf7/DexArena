"""Contracts for the opt-in common scene factory; GPU checks use scene_matrix."""
from types import SimpleNamespace

import pytest

from utils.scene_matrix import matrix, subtasks
from utils.dex_scene_eval import BACKGROUND_IDS, background_manifest, make_scene_eval_env, registered_task_settings


def test_exact_canonical_cartesian_product():
    rows = matrix()
    assert len(subtasks()) == 30
    assert len(rows) == len({r["key"] for r in rows}) == 90
    assert {(r["env_id"], r["scene_background"]) for r in rows} == {
        (env, scene) for env in subtasks() for scene in BACKGROUND_IDS}
    with pytest.raises(ValueError, match="Unknown evaluation background"):
        background_manifest("misspelled_scene")


def test_all_thirty_tasks_resolve_their_declared_robot_and_horizon():
    for env_id in subtasks():
        settings = registered_task_settings(env_id)
        assert "sharpa" in settings["robot_uids"]
        assert settings["max_episode_steps"] > 0
        assert settings["control_mode"] == "pd_joint_pos"


@pytest.mark.parametrize("kwargs", [
    {"num_envs": 2}, {"sim_backend": "physx_cuda"}, {"robot_uids": "panda"},
    {"control_mode": "pd_joint_delta_pos"}, {"reconfiguration_freq": 0},
    {"max_episode_steps": 1}, {"sensor_configs": {"width": 128, "height": 128}},
])
def test_protocol_drift_is_rejected_before_build(kwargs):
    with pytest.raises(ValueError, match="[Ff]ixed"):
        make_scene_eval_env("OfficeDeskClickMouse-Sharpa-v1", scene_background=BACKGROUND_IDS[0], **kwargs)


def test_factory_restores_registry_after_failure(monkeypatch):
    import tasks  # noqa: F401
    import utils.dex_scene_eval as module
    from mani_skill.utils.registration import REGISTERED_ENVS
    env_id = "OfficeDeskClickMouse-Sharpa-v1"
    before = REGISTERED_ENVS[env_id].cls

    def broken(*args):
        raise RuntimeError("missing scene asset")

    monkeypatch.setattr(module, "install_eval_background", broken)
    with pytest.raises(RuntimeError, match="missing scene asset"):
        make_scene_eval_env(env_id, scene_background=BACKGROUND_IDS[0])
    assert REGISTERED_ENVS[env_id].cls is before


def test_factory_preserves_state_mode_and_restores_registry(monkeypatch):
    import gymnasium as gym
    import tasks  # noqa: F401
    import utils.dex_scene_eval as module
    import utils.dex_camera_rigs as rigs
    from mani_skill.utils.registration import REGISTERED_ENVS
    env_id = "OfficeDeskClickMouse-Sharpa-v1"
    before = REGISTERED_ENVS[env_id].cls
    captured = {}
    fake = SimpleNamespace(unwrapped=SimpleNamespace())

    def make(env_id, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(gym, "make", make)
    monkeypatch.setattr(module, "install_eval_background", lambda *args: {})
    monkeypatch.setattr(rigs, "describe", lambda *args: {})
    assert make_scene_eval_env(env_id, scene_background=BACKGROUND_IDS[0], obs_mode="state") is fake
    assert captured["obs_mode"] == "state"
    assert captured["num_envs"] == 1
    assert captured["sensor_configs"] == {"width": 256, "height": 256}
    assert REGISTERED_ENVS[env_id].cls is before


def test_adapter_is_outside_all_collection_fingerprints():
    import tasks  # noqa: F401
    from tasks.config.collection_targets import COLLECTION_TARGETS
    from tasks.config.code_fingerprint import env_modules
    for env_id in COLLECTION_TARGETS:
        closure = env_modules(env_id)
        assert "utils.dex_scene_eval" not in closure
        assert "utils.dex_cameras" not in closure
        assert "utils.dex_camera_rigs" not in closure


def test_confirmed_native_tables_and_floor_alignment():
    from utils.dex_scene_eval import BACKGROUNDS, _transform
    import numpy as np
    import sapien
    assert BACKGROUND_IDS == ('robocasa_l3_s2', 'robocasa_l3_s1', 'robocasa_l3_s6')
    for spec in BACKGROUNDS:
        point = _transform(spec) * sapien.Pose(spec.anchor[:3])
        np.testing.assert_allclose(point.p, [-.125, 0, 0], atol=1e-6)
        manifest = background_manifest(spec.scene_id)
        assert manifest['robot_root_z_offset'] == 0
        assert abs(manifest['native_height_delta_m']) < .001
        assert manifest['mode'] == 'native_surface'
        assert manifest['original_table'] == 'render_and_collision_disabled'


def test_native_collision_support_replaces_disabled_original_table():
    import sapien
    from utils.dex_scene_eval import _clone_collision, _disable_original_surfaces
    px = sapien.physx
    physics = px.PhysxCpuSystem()
    scene = sapien.Scene([physics])
    scene.set_timestep(1/240)
    material = px.PhysxMaterial(.3, .3, 0)
    # The old table would stop the ball at z=.2 if its collision were retained.
    old = sapien.Entity()
    old_body = px.PhysxRigidStaticComponent()
    old_body.attach(px.PhysxCollisionShapeBox([.4, .7, .03], material))
    old.add_component(old_body)
    old.set_pose(sapien.Pose([0, 0, .17]))
    scene.add_entity(old)
    fake = SimpleNamespace(scene=SimpleNamespace(actors={'table-workspace':SimpleNamespace(_objs=[old])}))
    _disable_original_surfaces(fake)
    source_shape = px.PhysxCollisionShapeBox([.4, .7, .03], material)
    source_shape.local_pose = sapien.Pose([0, 0, -.03])
    native = sapien.Entity()
    native_body = px.PhysxRigidStaticComponent()
    native_body.attach(_clone_collision(source_shape))
    native.add_component(native_body)
    scene.add_entity(native)
    ball = sapien.Entity()
    ball_body = px.PhysxRigidDynamicComponent()
    ball_body.attach(px.PhysxCollisionShapeSphere(.015, material))
    ball.add_component(ball_body)
    ball.set_pose(sapien.Pose([0, 0, .3]))
    scene.add_entity(ball)
    for _ in range(240):
        scene.step()
    assert float(ball.pose.p[2]) == pytest.approx(.015, abs=.002)
    assert all(v == 0 for v in old_body.get_collision_shapes()[0].get_collision_groups())


def test_eval_island_style_does_not_mutate_collection_library():
    from utils.dex_scene_eval import _scene_spec
    from tasks.scene_builders.scene_library import SCENES_BY_NAME
    before = dict(SCENES_BY_NAME)
    scene = _scene_spec("robocasa_l3_s1")
    assert scene.source == "robocasa" and scene.config == 37
    extra = _scene_spec("robocasa_l3_s6")
    assert extra.source == "robocasa" and extra.config == 42
    assert SCENES_BY_NAME == before
    assert _scene_spec("architecthor_test_02") == SCENES_BY_NAME["architecthor_test_02"]


@pytest.mark.parametrize("table_near_x, passes", [(-.6169796, False), (-.425, True)])
def test_trolley_table_clearance_rejects_centered_dining_table(table_near_x, passes):
    """Regression: task-object support checks missed a 15 cm trolley overlap."""
    import sapien
    from utils.dex_scene_checks import check_robot_table_clearance
    px = sapien.physx
    scene = sapien.Scene([px.PhysxCpuSystem()])
    material = px.PhysxMaterial(.3, .3, 0)

    def box(name, lower, upper):
        import numpy as np
        lower, upper = np.asarray(lower), np.asarray(upper)
        entity = sapien.Entity()
        entity.name = name
        body = px.PhysxRigidStaticComponent()
        body.attach(px.PhysxCollisionShapeBox((upper-lower)/2, material))
        entity.add_component(body)
        entity.set_pose(sapien.Pose((upper+lower)/2))
        scene.add_entity(entity)
        return entity

    # Real PhysX bounds, without a renderer. Values measured on the scene assets.
    trolley = box("mobile_base_link_6", [-1.2919, -.4007, -.5448], [-.4632, .4078, .1565])
    table = box("dining_table", [table_near_x, -1.1067, -.7640], [table_near_x+1.2340, 1.1067, 0])
    link = SimpleNamespace(name=trolley.name, _objs=[SimpleNamespace(entity=trolley)])
    env = SimpleNamespace(unwrapped=SimpleNamespace(agent=SimpleNamespace(
        robot=SimpleNamespace(get_links=lambda: [link]))))
    result = check_robot_table_clearance(env, table)
    assert (not result["errors"]) == passes
    assert result["minimum_separation_m"] == pytest.approx(table_near_x+.4632, abs=1e-5)


def test_working_edge_follows_actual_original_mesh_on_every_reset():
    import numpy as np
    import sapien
    from utils.dex_scene_eval import _align_native_table_edge

    class Entity:
        def __init__(self, lower, upper):
            self.box = np.asarray([lower, upper])
            self.pose = sapien.Pose()

        def set_pose(self, pose):
            self.pose = pose

        def find_component_by_type(self, cls):
            if cls is sapien.render.RenderBodyComponent:
                return SimpleNamespace(compute_global_aabb_tight=lambda: self.box+self.pose.p)
            if cls is sapien.physx.PhysxRigidBaseComponent:
                return SimpleNamespace(get_collision_shapes=lambda: [object()])

    original = Entity([-.4402, -1.2, -.92], [.77, 1.2, 0])
    table = Entity([-.425, -1.1, -.77], [.81, 1.1, 0])
    wall = Entity([-2., -2., -.77], [-1.9, 2., 2.])
    task_object = Entity([0, 0, 0], [.1, .1, .1])
    env = SimpleNamespace(scene=SimpleNamespace(actors={
        'table-workspace': SimpleNamespace(_objs=[original]),
        'task-object': SimpleNamespace(_objs=[task_object])}),
        _native_scene_entities=[table, wall], _native_scene_translation_x=0.,
        _native_source_transform_x=1.,
        eval_background_manifest={'world_from_source_scene': {'position': [1., 0., 0.]}})
    for original_shift in (0., 0., -.02):
        original.set_pose(sapien.Pose([original_shift, 0, 0]))
        _align_native_table_edge(env)
        assert env._native_edge_alignment['error_m'] == pytest.approx(0, abs=1e-6)
        assert env._native_edge_alignment['tabletop_height_error_m'] == pytest.approx(0, abs=1e-6)
        expected_shift = -.0152+original_shift
        assert float(table.pose.p[0]) == pytest.approx(expected_shift, abs=1e-6)
        np.testing.assert_allclose(wall.pose.p, table.pose.p, atol=1e-6)
        np.testing.assert_allclose(task_object.pose.p, [0, 0, 0])
        assert env.eval_background_manifest['world_from_source_scene']['position'][0] == pytest.approx(1+expected_shift, abs=1e-6)

    table.box[1, 2] = .008
    with pytest.raises(RuntimeError, match="tabletop height"):
        _align_native_table_edge(env)


def test_low_native_surface_is_rejected_without_robot_compensation(monkeypatch):
    import utils.dex_scene_eval as module
    dining = module.BackgroundSpec("low_table", "Low table", (0, 0, .767, 0, 0))
    monkeypatch.setattr(module, "BACKGROUNDS", (dining,))
    with pytest.raises(ValueError, match="height differs"):
        module.background_manifest("low_table")

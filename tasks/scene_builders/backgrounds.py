"""Background-scene adapters.

Each adapter wraps one scene dataset behind the same calls the full-scene builder
needs: ``build``, ``initialize``, ``floor_z``, and ``work_surface_anchors`` —
where in this scene is there a counter or table to work at.

The adapters exist because the upstream ManiSkill scene builders assume they own
the robot: they place it, teleport it, and in the ReplicaCAD and AI2THOR cases
raise on any robot that is not a Fetch. Here the workstation owns the robot, so
the adapters strip that out.

Adding a dataset means adding one adapter and registering it in ``ADAPTERS``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import sapien
import torch

from tasks.scene_builders.anchor import (
    entity_aabb,
    pick_surface_spot,
    robot_entity_ids,
    surface_clutter,
)

GLB_SCENE_DIR = Path(
    os.environ.get(
        "ROBODEX_SCENE_GLB_DIR",
        Path(os.environ.get("MS_ASSET_DIR", Path.home() / ".maniskill"))
        / "data" / "scene_datasets" / "glb_scenes",
    )
)

# A surface is only worth working at if it is roughly waist-to-counter height.
# Below this a coffee table, above it a shelf.
WORK_HEIGHT_RANGE = (0.6, 1.15)

# Floor the trolley needs in front of a surface: the cart is ~1.07 m deep, plus
# room enough not to be wedged against whatever is behind it.
CART_FOOTPRINT = 1.35


class BackgroundAdapter:
    """Builds a static background scene, one config per sub-env."""

    builds_lighting = False

    def __init__(self, env):
        self.env = env

    def build(self, configs: list) -> None:
        raise NotImplementedError

    def initialize(self, env_idx: torch.Tensor) -> None:
        pass

    def work_surface_anchors(self, reach: float, inset: float):
        """``(x, y, surface_z, yaw)`` per sub-env, or None if the scene cannot
        say where its work surfaces are (in which case specs must be explicit)."""
        return None

    @property
    def floor_z(self) -> float:
        """Height of the walkable floor. Datasets so far all put it at 0."""
        return 0.0


def _appliance_types():
    """Fixture classes that sit on a counter top.

    A workspace point inside one of these is in the sink or on the burners.
    Imported lazily because robocasa's fixture modules pull in the whole
    dataset's asset paths.
    """
    from mani_skill.utils.scene_builder.robocasa.fixtures.accessories import (
        Accessory,
        CoffeeMachine,
        Toaster,
    )
    from mani_skill.utils.scene_builder.robocasa.fixtures.hood import Hood
    from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
    from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink
    from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Oven, Stove, Stovetop

    return (Sink, Stove, Stovetop, Oven, Microwave, Hood, CoffeeMachine, Toaster, Accessory)


class NamedSurfaceAdapter(BackgroundAdapter):
    """Base for datasets whose furniture is loose meshes with useful names.

    ReplicaCAD and AI2THOR both ship a room as unlabelled geometry plus asset
    names (``frl_apartment_table_01``, ``Dining_Table_1``). The name is the only
    semantics available, so surfaces are matched on it, filtered to working
    height, and the biggest top wins.
    """

    SURFACE_NAME_HINTS: tuple[str, ...] = ("table",)
    MIN_SURFACE_AREA = 0.25  # m^2 of footprint; smaller is a stool or a shelf

    def work_surface_anchors(self, reach: float, inset: float):
        out = []
        for i in range(self.env.num_envs):
            physx_system = self.env.scene.sub_scenes[i].physx_system
            # Largest first: the biggest surface is the most likely to have a
            # clear stretch, but a counter run can be entirely sink and stove, so
            # fall through to the next one when nothing on it works out.
            skip = robot_entity_ids(self.env, i)
            for surface in self._candidate_surfaces(i):
                spot = pick_surface_spot(
                    surface,
                    physx_system,
                    reach=reach,
                    inset=inset,
                    cart_footprint=CART_FOOTPRINT,
                    clutter=surface_clutter(
                        self.env.scene.sub_scenes[i], skip, float(surface[1][2])
                    ),
                )
                if spot is not None:
                    out.append(spot)
                    break
            else:
                raise ValueError(
                    f"no work surface with a clear spot the rig can reach in "
                    f"'{self._config_name(i)}'"
                )
        return out

    def _config_name(self, scene_idx: int) -> str:
        return str(self.builder.build_configs[self.builder.build_config_idxs[scene_idx]])

    def _candidate_surfaces(self, scene_idx: int) -> list[np.ndarray]:
        """Work surfaces in this sub-scene, largest footprint first."""
        found = []
        prefix = f"env-{scene_idx}_"
        for name, actor in self.builder.scene_objects.items():
            lowered = name.lower()
            if not name.startswith(prefix):
                continue
            if not any(hint in lowered for hint in self.SURFACE_NAME_HINTS):
                continue
            aabb = entity_aabb(actor._objs[0])
            if aabb is None:
                continue
            if not WORK_HEIGHT_RANGE[0] <= aabb[1][2] <= WORK_HEIGHT_RANGE[1]:
                continue
            area = float(np.prod(aabb[1][:2] - aabb[0][:2]))
            if area < self.MIN_SURFACE_AREA:
                continue
            found.append((area, aabb))
        return [aabb for _, aabb in sorted(found, key=lambda r: -r[0])]


def _robot_free(builder_cls):
    """Strip a scene builder's robot handling.

    ReplicaCAD and AI2THOR both teleport the robot around during ``initialize``
    and then hard-raise on anything that is not a Fetch. The workstation places
    our rig itself, so drop that half and keep the object settling.
    """

    class _RobotFree(builder_cls):
        def initialize(self, env_idx: torch.Tensor) -> None:
            from mani_skill.utils.structs import Articulation

            for obj, pose in self._default_object_poses:
                obj.set_pose(pose)
                if isinstance(obj, Articulation):
                    obj.set_qpos(obj.qpos[0] * 0)
                    obj.set_qvel(obj.qvel[0] * 0)
            if self.scene.gpu_sim_enabled and len(env_idx) == self.env.num_envs:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

    _RobotFree.__name__ = f"RobotFree{builder_cls.__name__}"
    return _RobotFree


class RoboCasaAdapter(BackgroundAdapter):
    """RoboCasa kitchens. Configs are build_config_idx = layout * 12 + style.

    Unlike the name-matched datasets this does not use the generic sampler.
    RoboCasa describes its counters properly, including which pieces of a top are
    free of the sink and the cooktop, and that beats sampling a bounding box: an
    L-shaped counter run's box is mostly not counter, and switching this to the
    sampler moved four of twelve kitchens somewhere worse.
    """

    builds_lighting = True

    def build(self, configs: list) -> None:
        from mani_skill.utils.scene_builder.robocasa.scene_builder import (
            RoboCasaSceneBuilder,
        )

        self.builder = RoboCasaSceneBuilder(self.env)
        self.builder.build(build_config_idxs=[int(c) for c in configs])

    def initialize(self, env_idx: torch.Tensor) -> None:
        # Only repositions fetch/unitree robots, so it is a no-op for our arms,
        # but call it so fixture state stays whatever upstream expects.
        self.builder.initialize(env_idx)

    def work_surface_anchors(self, reach: float, inset: float):
        """A clear stretch of counter, worked from whichever side has floor.

        RoboCasa's own robot placement is not usable for this: it parks a mobile
        robot in front of a randomly chosen appliance, which puts the workspace
        in the sink basin or on the burners. Instead this walks the counters'
        *reset regions* — the flat top pieces RoboCasa places objects on, already
        split around sink and cooktop openings — throws out any that still fall
        under an appliance, and keeps the largest one the rig can stand at.
        """
        return [
            self._counter_spot(data["fixtures"], scene_idx, inset)
            for scene_idx, data in enumerate(self.builder.scene_data)
        ]

    def _counter_spot(self, fixtures, scene_idx: int, inset: float):
        from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType
        from mani_skill.utils.scene_builder.robocasa.fixtures.fixture_utils import (
            fixture_is_type,
        )
        from mani_skill.utils.scene_builder.robocasa.utils import object_utils as OU

        physx_system = self.env.scene.sub_scenes[scene_idx].physx_system
        blockers = [f for f in fixtures.values() if isinstance(f, _appliance_types())]

        best, best_area = None, 0.0
        for counter in fixtures.values():
            if not fixture_is_type(counter, FixtureType.COUNTER_NON_CORNER):
                continue
            for region in counter.get_reset_regions(self.builder, fixtures).values():
                area = float(region["size"][0] * region["size"][1])
                if area <= best_area:
                    continue
                spot = self._region_spot(
                    counter, region, blockers, physx_system, inset, OU
                )
                if spot is not None:
                    best, best_area = spot, area
        if best is None:
            raise ValueError("no counter in this kitchen has a clear reachable spot")
        return best

    def _region_spot(self, counter, region, blockers, physx_system, inset, OU):
        """Workstation frame for one counter region, or None if the rig cannot use it.

        A counter is worked from a long side, and which side faces the room is
        not fixed — an island has floor on both — so each is tried and a ray
        decides.
        """
        from tasks.scene_builders.anchor import open_distance

        depth, height = float(counter.size[1]), float(counter.size[2])
        region_y, half_y = float(region["offset"][1]), float(region["size"][1]) / 2
        for side in (-1.0, 1.0):
            y = float(np.clip(side * (depth / 2 - inset), region_y - half_y, region_y + half_y))
            offset = np.array([float(region["offset"][0]), y, height / 2])
            pos = OU.get_pos_after_rel_offset(counter, offset)
            if any(OU.point_in_fixture(pos, b, only_2d=True) for b in blockers):
                continue
            # Facing into the counter is facing away from the side we stand on.
            yaw = float(counter.rot + (np.pi / 2 if side < 0 else -np.pi / 2))
            probe = np.array([pos[0], pos[1], pos[2] + 0.25])
            if open_distance(physx_system, probe, yaw + np.pi, CART_FOOTPRINT) < CART_FOOTPRINT:
                continue
            return float(pos[0]), float(pos[1]), float(pos[2]), yaw
        return None


class ReplicaCADAdapter(NamedSurfaceAdapter):
    """ReplicaCAD apartments. Configs are scene_instance json filenames."""

    builds_lighting = True
    SURFACE_NAME_HINTS = ("table",)

    def build(self, configs: list) -> None:
        from mani_skill.utils.scene_builder.replicacad.scene_builder import (
            ReplicaCADSceneBuilder,
        )

        self.builder = _robot_free(ReplicaCADSceneBuilder)(
            self.env, include_staging_scenes=True
        )
        name_to_idx = {name: i for i, name in enumerate(self.builder.build_configs)}
        self.builder.build([name_to_idx[str(c)] for c in configs])

    def initialize(self, env_idx: torch.Tensor) -> None:
        self.builder.initialize(env_idx)


class AI2THORAdapter(NamedSurfaceAdapter):
    """AI2THOR rooms and houses. Configs are ``"<dataset>:<scene json>"``.

    The four datasets are separate registries — iTHOR is single rooms, ProcTHOR
    is 12k procedurally furnished houses, ArchitecTHOR is hand-authored houses,
    RoboTHOR is navigation apartments — but they share a builder, so the dataset
    is carried in the config string.
    """

    builds_lighting = True
    # AI2THOR names its furniture by asset: kitchen worktops come through as
    # StandardCounterHeightWidth / StandardIslandHeight, the rest as Dining_Table,
    # Desk, Dresser and so on.
    SURFACE_NAME_HINTS = ("table", "counter", "island", "desk", "dresser")

    def build(self, configs: list) -> None:
        from mani_skill.utils.scene_builder.ai2thor.variants import (
            AI2THORBaseSceneBuilder,
        )

        datasets = {str(c).split(":", 1)[0] for c in configs}
        if len(datasets) > 1:
            raise ValueError(
                f"one AI2THOR dataset per vectorized env, got {sorted(datasets)}"
            )
        dataset = datasets.pop()

        base = type(f"{dataset}SceneBuilder", (AI2THORBaseSceneBuilder,), {"scene_dataset": dataset})
        self.builder = _robot_free(base)(self.env)
        name_to_idx = {
            cfg.config_file: i for i, cfg in enumerate(self.builder.build_configs)
        }
        self.builder.build([name_to_idx[str(c).split(":", 1)[1]] for c in configs])

    def initialize(self, env_idx: torch.Tensor) -> None:
        self.builder.initialize(env_idx)

    def _config_name(self, scene_idx: int) -> str:
        cfg = self.builder.build_configs[self.builder.build_config_idxs[scene_idx]]
        return f"{cfg.source}:{cfg.config_file}"


class GLBAdapter(BackgroundAdapter):
    """A single static mesh as the room (e.g. an Objaverse interior).

    Config is ``dict(file=..., scale=..., euler=..., offset=...)``. Objaverse
    exports are y-up and in arbitrary units, so both the rotation and the scale
    have to be set per model — there is no metadata to read them from. The room
    is one mesh, so its benches cannot be picked out either: these specs carry
    explicit anchors, measured with ``scene_library.sh surfaces``.
    """

    def build(self, configs: list) -> None:
        from transforms3d.euler import euler2quat

        self.actors = []
        for i, cfg in enumerate(configs):
            path = Path(cfg["file"])
            if not path.is_absolute():
                path = GLB_SCENE_DIR / path
            if not path.exists():
                raise FileNotFoundError(
                    f"background mesh {path} is missing; run scripts/scene_library.sh fetch"
                )
            scale = float(cfg.get("scale", 1.0))
            euler = cfg.get("euler", (np.pi / 2, 0.0, 0.0))  # y-up -> z-up
            offset = cfg.get("offset", (0.0, 0.0, 0.0))
            builder = self.env.scene.create_actor_builder()
            builder.set_scene_idxs([i])
            pose = sapien.Pose(q=euler2quat(*euler))
            builder.add_visual_from_file(filename=str(path), scale=[scale] * 3, pose=pose)
            builder.add_nonconvex_collision_from_file(
                filename=str(path), scale=[scale] * 3, pose=pose
            )
            builder.initial_pose = sapien.Pose(p=list(offset))
            self.actors.append(builder.build_static(name=f"background_mesh_{i}"))


# Imported here because the lab adapter shares BackgroundAdapter above.
from tasks.scene_builders.labutopia import LabUtopiaAdapter  # noqa: E402


ADAPTERS: dict[str, type[BackgroundAdapter]] = {
    "robocasa": RoboCasaAdapter,
    "replicacad": ReplicaCADAdapter,
    "ai2thor": AI2THORAdapter,
    "glb": GLBAdapter,
    "labutopia": LabUtopiaAdapter,
}

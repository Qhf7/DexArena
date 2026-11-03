"""Run a task on a scene's own work surface.

``RandomizedTableSceneBuilder`` randomizes a synthetic room: procedural textures
on a table top and a five-sided box around it. This builder takes the other
route — it puts the rig in real authored interiors (RoboCasa kitchens,
ReplicaCAD apartments, static meshes) and works on **the surface that scene
already has**: a kitchen counter, an island, a dining table. LabUtopia is a
prepared-workspace variant: its room includes an empty collection-size work table.

Tasks keep their geometry. Everything a task places — the object, the goal, the
cameras — is expressed in a *workstation frame* whose origin sits on the work
surface, which is where ``TableSceneBuilder`` puts its own origin too. What
changes per scene is where that frame lands and how high the surface is:

* ``anchor_pose`` — the frame in the world. Its +x points from the robot *into*
  the surface, so the open floor the robot stands on is at -x.
* ``robot_root_pose`` — the trolley on the floor beside the surface. The cart is
  ``CART_HEIGHT`` tall, so a lower surface means the arm mounts higher above it.

Usage:
    self.table_scene = FullSceneBuilder(env=self, scene_ids="robocasa_l0_s0",
                                        robot_reach=0.87)
    self.table_scene.build()
    ...
    self.obj.set_pose(self.table_scene.world_pose_from_pq(p=xyz, q=qs, env_idx=env_idx))
    self.agent.robot.set_root_pose(self.table_scene.robot_root_pose)

Limitation: all parallel sub-envs must draw from the same dataset. The upstream
RoboCasa/ReplicaCAD builders each build for every sub-env in one call, so two
datasets cannot coexist in one vectorized env. Rendering the library uses one
scene per process, so this only bites multi-scene training.
"""
from __future__ import annotations

import json
import os.path as osp
import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sapien
import torch
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from tasks.scene_builders.anchor import (
    collect_obstacle_aabbs,
    ray_distance,
    robot_entity_ids,
    room_center,
)
from tasks.scene_builders.backgrounds import ADAPTERS
from tasks.scene_builders.lighting import LightingRandomizer

# Height of the trolley's mounting deck above the floor. The carts were built to
# match ManiSkill's table, which is why a task written for the bare table drops
# onto a ~0.92 m kitchen counter unchanged.
CART_HEIGHT = 0.9196429

# Nominal camera offset from the workspace, as (horizontal, 0, vertical). Scaled
# down per scene by whatever the line of sight actually allows.
VIEW_OFFSET = np.array([2.6, 0.0, 1.9])
VIEW_MARGIN = 0.2   # stop short of whatever the ray hit
MIN_VIEW_DISTANCE = 1.2


@dataclass(frozen=True)
class SceneSpec:
    """One entry in the scene library."""

    name: str
    source: str  # key into backgrounds.ADAPTERS
    config: Any  # build_config_idx / scene json name / glb dict
    anchor: tuple[float, float, float, float] | str = "auto"
    """Workstation frame as ``(x, y, surface_z, yaw)``, or "auto" to ask the
    scene where its work surfaces are. ``yaw`` points from the robot into the
    surface."""
    nudge: tuple[float, float] = (0.0, 0.0)
    """``(lateral, depth)`` metres to slide the finished anchor along its own
    axes: lateral is sideways across the surface, depth is further into it.

    This is the knob for "work further down the counter, away from the sink" —
    it keeps everything the automatic placement worked out (which side, how high,
    how much floor the trolley has) and only slides the frame, taking the robot
    and the cameras with it."""
    note: str = ""


def _robocasa_specs() -> list[SceneSpec]:
    """RoboCasa kitchens: one style from each of 4 style families per layout,
    trimmed to 12 so the library is not three quarters kitchen."""
    return [
        SceneSpec(
            name=f"robocasa_l{layout}_s{style}",
            source="robocasa",
            config=layout * 12 + style,
            note=f"RoboCasa kitchen layout {layout}, style {style}",
        )
        for layout, style in zip(range(10), (0, 4, 8, 2, 6, 10, 1, 5, 9, 3))
    ] + [
        SceneSpec(
            name=f"robocasa_l{layout}_s{style}",
            source="robocasa",
            config=layout * 12 + style,
            note=f"RoboCasa kitchen layout {layout}, style {style}",
        )
        for layout, style in ((3, 4), (7, 8))
    ]


def _replicacad_specs() -> list[SceneSpec]:
    """ReplicaCAD: the furnished apartments plus a couple of staging variants.

    All 90 configs are the same apartment shell with the furniture moved, so a
    handful covers what this dataset has to offer.
    """
    names = [f"apt_{i}.scene_instance.json" for i in range(6)]
    names += [f"v3_sc0_staging_{i:02d}.scene_instance.json" for i in (0, 6)]
    return [
        SceneSpec(
            name="replicacad_" + osp.basename(n).split(".")[0],
            source="replicacad",
            config=n,
            note="ReplicaCAD apartment",
        )
        for n in names
    ]


def _ai2thor_scene_files(dataset: str) -> list[str]:
    """Scene filenames shipped with ManiSkill's AI2THOR metadata.

    Read rather than hardcoded: the lists are thousands long for ProcTHOR, and
    they belong to the installed ManiSkill rather than to this repo.
    """
    from mani_skill.utils.scene_builder.ai2thor import scene_builder as ai2thor

    with open(Path(ai2thor.DATASET_CONFIG_DIR) / f"{dataset}.json") as f:
        return sorted(json.load(f)["scenes"])


# Scenes taken out after review. ArchitecTHOR-Test-04's only work surface is a
# desk too cramped for the trolley to stand at properly.
DROPPED_SCENES = ("ArchitecTHOR-Test-04",)


def _dropped(config_file: str) -> bool:
    return any(name in config_file for name in DROPPED_SCENES)


def _ai2thor_specs() -> list[SceneSpec]:
    """AI2THOR, where most of the variety in this library comes from.

    iTHOR is single rooms by type (kitchens 1-30, living rooms 201-230, bedrooms
    301-330 — bathrooms are skipped, nothing there to work at), ArchitecTHOR is
    hand-authored houses, and ProcTHOR is 12k procedurally furnished ones, of
    which a deterministic spread is taken.
    """
    specs: list[SceneSpec] = []

    ithor = _ai2thor_scene_files("iTHOR")
    rooms = {"kitchen": range(1, 31), "living": range(201, 231), "bedroom": range(301, 331)}
    for kind, numbers in rooms.items():
        matching = [
            f for f in ithor
            if (m := re.search(r"FloorPlan(\d+)", f)) and int(m.group(1)) in numbers
        ]
        for f in matching[:: max(len(matching) // 4, 1)][:4]:
            number = re.search(r"FloorPlan(\d+)", f).group(1)
            specs.append(
                SceneSpec(
                    name=f"ithor_{kind}_{number}",
                    source="ai2thor",
                    config=f"iTHOR:{f}",
                    note=f"AI2THOR iTHOR {kind}",
                )
            )

    architec = [f for f in _ai2thor_scene_files("ArchitecTHOR") if not _dropped(f)]
    for f in architec[:6]:
        stem = Path(f).name.split(".")[0].replace("ArchitecTHOR-", "").lower().replace("-", "_")
        specs.append(
            SceneSpec(
                name=f"architecthor_{stem}",
                source="ai2thor",
                config=f"ArchitecTHOR:{f}",
                note="AI2THOR ArchitecTHOR house",
            )
        )

    procthor = _ai2thor_scene_files("ProcTHOR")
    for f in procthor[:: max(len(procthor) // 11, 1)][:11]:
        stem = Path(f).name.split(".")[0].replace("ProcTHOR-", "").lower().replace("-", "_")
        specs.append(
            SceneSpec(
                name=f"procthor_{stem}",
                source="ai2thor",
                config=f"ProcTHOR:{f}",
                note="AI2THOR ProcTHOR house",
            )
        )
    return specs


# Measured off the mesh with `scene_library.sh surfaces`: the workbench running
# along the far wall, top at 0.96 m, 0.65 m deep with its near edge at y=4.35.
# The rig works it from the room side, which is -y.
WORKSHOP_BENCH_ANCHOR = (-2.0, 4.65, 0.96, np.pi / 2)


def _glb_specs() -> list[SceneSpec]:
    """Static meshes. One mesh for the whole room, so nothing can be picked out
    of it automatically — scale, orientation and anchor are all hand-set."""
    return [
        SceneSpec(
            name="workshop_minimalistic",
            source="glb",
            config=dict(
                file="64d6f92fc5d44770843782de76e03e61.glb",
                # Already in meters (7.1 x 8.6 m floor, 4.8 m to the ceiling).
                # y-up -> z-up puts the floor at z=-0.168, so lift it to zero.
                scale=1.0,
                euler=(np.pi / 2, 0.0, 0.0),
                offset=(0.0, 0.0, 0.168),
            ),
            anchor=WORKSHOP_BENCH_ANCHOR,
            note="Objaverse 'Minimalistic Workshop Environment' (industrial)",
        )
    ]


# Hand corrections on top of the automatic placement, in metres of
# ``SceneSpec.nudge``. Every one of these came from looking at a render: the
# automatic spot was on the surface and reachable, but sat at the narrow end of a
# counter, or next to a burner, or on the wrong side of a laptop.
# Anchors pinned outright, where a nudge cannot express what is wanted.
SCENE_ANCHORS: dict[str, tuple[float, float, float, float]] = {}

SCENE_NUDGES: dict[str, tuple[float, float]] = {
    "architecthor_test_00": (0.9, 0.0),    # off the counter's narrow end
    "architecthor_test_01": (-0.55, 0.0),  # further down the counter, so the rig
                                           # is not jammed against the wall corner
    "workshop_minimalistic": (0.4, 0.0),  # clear of the tools already on the bench
}

SCENE_LIBRARY: list[SceneSpec] = [
    replace(
        spec,
        anchor=SCENE_ANCHORS.get(spec.name, spec.anchor),
        nudge=SCENE_NUDGES.get(spec.name, spec.nudge),
    )
    for spec in _robocasa_specs() + _replicacad_specs() + _ai2thor_specs() + _glb_specs() + [
        SceneSpec(
            name="labutopia_lab003",
            source="labutopia",
            config={"file": "labutopia_lab003/room.glb"},
            anchor=(0.0, 0.0, CART_HEIGHT, 0.0),
            note="LabUtopia lab_003 room with an empty grey work table; no task props",
        ),
    ]
]
SCENES_BY_NAME: dict[str, SceneSpec] = {s.name: s for s in SCENE_LIBRARY}


def get_scene_spec(name: str) -> SceneSpec:
    if name not in SCENES_BY_NAME:
        raise KeyError(
            f"unknown scene '{name}'; {len(SCENE_LIBRARY)} available, e.g. "
            + ", ".join(list(SCENES_BY_NAME)[:3])
        )
    return SCENES_BY_NAME[name]


class FullSceneBuilder(SceneBuilder):
    """Rig at a scene's own work surface, with the scene as the environment."""

    builds_lighting = True

    def __init__(
        self,
        env,
        scene_ids: str | Sequence[str],
        robot_init_qpos_noise: float = 0.02,
        random_lighting: bool = False,
        robot_reach: float = 0.87,
        surface_inset: float = 0.30,
        ambient_range: tuple[float, float] = (0.1, 0.6),
        temperature_range: tuple[float, float] = (2700.0, 6500.0),
        intensity_range: tuple[float, float] = (0.6, 1.4),
        light_down_bias: float = 0.3,
    ):
        super().__init__(env, robot_init_qpos_noise=robot_init_qpos_noise)
        self.specs = self._resolve_specs(scene_ids)
        sources = {s.source for s in self.specs}
        if len(sources) > 1:
            raise ValueError(
                "all sub-envs must use one dataset per vectorized env, got "
                f"{sorted(sources)} (see module docstring)"
            )
        self.source = sources.pop()
        self.background = ADAPTERS[self.source](env)
        # How far in front of the trolley the workspace sits, and how far in from
        # the surface's near edge to put it. Together these decide how much floor
        # is left between the cart and the counter.
        self.robot_reach = robot_reach
        self.surface_inset = surface_inset
        self.random_lighting = random_lighting
        self._lighting = LightingRandomizer(
            env,
            ambient_range=ambient_range,
            temperature_range=temperature_range,
            intensity_range=intensity_range,
            light_down_bias=light_down_bias,
        )
        # Tasks written against RandomizedTableSceneBuilder read this; nothing
        # jitters the surface height here, so it stays zero.
        self.table_top_offsets = torch.zeros(env.num_envs, device=env.device)
        self.anchor_pose: Pose | None = None
        self.surface_heights: np.ndarray | None = None
        self.anchor_actor = None
        self.view_azimuth: float = np.pi
        self.view_scale: float = 1.0

    def _resolve_specs(self, scene_ids) -> list[SceneSpec]:
        if isinstance(scene_ids, str):
            scene_ids = [scene_ids]
        scene_ids = list(scene_ids)
        if len(scene_ids) == 1:
            scene_ids = scene_ids * self.env.num_envs
        if len(scene_ids) != self.env.num_envs:
            raise ValueError(
                f"got {len(scene_ids)} scene ids for {self.env.num_envs} sub-envs"
            )
        return [get_scene_spec(n) for n in scene_ids]

    # -- build ---------------------------------------------------------------

    def build(self):
        self.background.build([s.config for s in self.specs])
        # Probing casts rays through the scene, and the robot is already in it
        # (agents load before scene builders). Park it out of range first.
        with self._robot_parked():
            self._resolve_anchors()
            self.view_azimuth, self.view_scale = self._pick_viewpoint()
        self._build_anchor_actor()
        # Mirror what the RoboCasa builder does for mobile robots: point the
        # articulation's initial pose at where the rig will actually stand, so
        # GPU sim does not start it inside a wall.
        if self.env.agent is not None:
            initial = self.env.agent.robot.initial_pose
            initial.raw_pose[:] = self.robot_root_pose.raw_pose.to(initial.device)

    @contextmanager
    def _robot_parked(self):
        agent = self.env.agent
        if agent is None:
            yield
            return
        previous = agent.robot.pose
        agent.robot.set_root_pose(sapien.Pose(p=[0.0, 0.0, -500.0]))
        try:
            yield
        finally:
            agent.robot.set_root_pose(previous)

    def _resolve_anchors(self) -> None:
        found = None
        if any(spec.anchor == "auto" for spec in self.specs):
            found = self.background.work_surface_anchors(
                reach=self.robot_reach, inset=self.surface_inset
            )
            if found is None:
                raise ValueError(
                    f"'{self.source}' scenes cannot report their work surfaces, so "
                    "every spec needs an explicit (x, y, surface_z, yaw) anchor"
                )
        anchors, heights = [], []
        for i, spec in enumerate(self.specs):
            x, y, z, yaw = spec.anchor if spec.anchor != "auto" else found[i]
            if not np.isfinite([x, y, z, yaw]).all():
                raise ValueError(f"scene '{spec.name}' produced a non-finite anchor")
            lateral, depth = spec.nudge
            # +x of the workstation points into the surface, +y across it.
            x += depth * np.cos(yaw) - lateral * np.sin(yaw)
            y += depth * np.sin(yaw) + lateral * np.cos(yaw)
            anchors.append([x, y, z, *euler2quat(0, 0, yaw)])
            heights.append(z - self.background.floor_z)
        self.anchor_pose = Pose.create(
            torch.tensor(np.asarray(anchors, dtype=np.float32), device=self.env.device)
        )
        self.surface_heights = np.asarray(heights, dtype=np.float32)

    def _pick_viewpoint(self, spread: float = np.pi * 5 / 9, samples: int = 21):
        """Bearing and standoff scale for the cameras, in the workstation frame.

        A fixed viewpoint works on an open table top and fails at a scene's own
        surface, where the wall behind a counter is right there. Two things are
        measured, both by casting the *actual line of sight* from the workspace
        out to where a camera would be:

        * the bearing, swept across the arc on the robot's side. Bearings with
          nothing to see are dropped, and among the rest the one that leaves the
          room's furniture behind the rig wins.
        * how much of the nominal standoff that bearing has. Overshooting is not
          harmless: ReplicaCAD's walls are culled from behind so a camera outside
          the flat still sees in, but AI2THOR's are solid, and every scene came
          back a flat wall of colour.

        Returns ``(bearing, scale)``, where the task multiplies its nominal eye
        offsets by ``scale``. One viewpoint is shared by every sub-env because
        camera configs are, so this reads sub-env 0.
        """
        sub_scene = self.env.scene.sub_scenes[0]
        origin = self.anchor_pose.p[0].cpu().numpy()
        forward = self.anchor_pose.to_transformation_matrix()[0, :2, 0].cpu().numpy()
        yaw = float(np.arctan2(forward[1], forward[0]))
        eye = origin + np.array([0.0, 0.0, 0.1])

        bearings = np.linspace(-spread, spread, samples) + np.pi
        nominal = np.linalg.norm(VIEW_OFFSET)
        rise = VIEW_OFFSET[2] / nominal
        run = np.sqrt(max(1.0 - rise * rise, 0.0))
        reach = np.array(
            [
                ray_distance(
                    sub_scene.physx_system,
                    eye,
                    [run * np.cos(yaw + b), run * np.sin(yaw + b), rise],
                    nominal,
                )
                for b in bearings
            ]
        )
        usable = reach >= MIN_VIEW_DISTANCE
        if not usable.any():
            usable = reach >= reach.max()

        centroid = room_center(
            collect_obstacle_aabbs(sub_scene, exclude_ids=robot_entity_ids(self.env, 0))
        )
        toward_room = None if centroid is None else centroid - origin[:2]
        if toward_room is not None and np.linalg.norm(toward_room) > 1e-6:
            toward_room = toward_room / np.linalg.norm(toward_room)
            directions = np.stack([np.cos(yaw + bearings), np.sin(yaw + bearings)], axis=1)
            score = -(directions @ toward_room)
        else:
            score = reach
        pick = int(np.argmax(np.where(usable, score, -np.inf)))
        clear = max(reach[pick] - VIEW_MARGIN, MIN_VIEW_DISTANCE * 0.5)
        return float(bearings[pick]), float(min(clear / nominal, 1.0))

    def _build_anchor_actor(self):
        """A pose with no geometry, for cameras to hang off.

        Camera configs are written in workstation coordinates, and a sub-env's
        workstation lands wherever its scene had room. Mounting on an actor that
        *is* the workstation frame keeps the configs identical across scenes.
        """
        builder = self.env.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.anchor_actor = builder.build_kinematic(name="workstation_anchor")
        self.anchor_actor.set_pose(self.anchor_pose)
        self.scene_objects = [self.anchor_actor]

    # -- initialize ----------------------------------------------------------

    def initialize(self, env_idx: torch.Tensor):
        self.background.initialize(env_idx)
        self.anchor_actor.set_pose(self.anchor_pose)
        if self.random_lighting:
            self._lighting.apply(env_idx)

    # -- workstation frame ---------------------------------------------------

    def world_pose(self, local: sapien.Pose) -> Pose:
        """Workstation-local pose -> batched world pose, one per sub-env."""
        return self.anchor_pose * Pose.create(local)

    def world_pose_from_pq(
        self,
        p: torch.Tensor,
        q: torch.Tensor | None = None,
        env_idx: torch.Tensor | None = None,
    ) -> Pose:
        """Batched workstation-local position/quaternion -> world pose.

        ``env_idx`` selects the anchors to apply, and must be passed whenever
        ``p`` covers only the sub-envs being reset rather than all of them.
        """
        anchor = self.anchor_pose if env_idx is None else self.anchor_pose[env_idx]
        return anchor * Pose.create_from_pq(p=p, q=q)

    @property
    def robot_root_pose(self) -> Pose:
        """Where the trolley stands: on the floor, ``robot_reach`` back from the
        workspace. The deck is a fixed height above the floor, so its offset from
        the work surface is whatever that surface's height leaves over."""
        local = np.zeros((self.env.num_envs, 3), dtype=np.float32)
        local[:, 0] = -self.robot_reach
        local[:, 2] = CART_HEIGHT - self.surface_heights
        return self.anchor_pose * Pose.create_from_pq(
            p=torch.tensor(local, device=self.env.device)
        )

    @property
    def mount_actor(self):
        """Actor to mount task cameras on so they follow the workstation."""
        return self.anchor_actor

    def mount_pose(self, workstation_pose):
        """Camera pose relative to ``mount_actor``.

        The mount actor is the workstation frame itself, so this is a no-op; it
        exists so tasks do not have to know that.
        """
        return workstation_pose

"""LabUtopia lab_003 room with the empty work table used in the approved preview.

The room contains fixed architecture and cabinets. Task props belong to the
task, never to this adapter. The floor is z=0 and the table top is z=0.9196429.
"""

from pathlib import Path

import numpy as np
import sapien

from tasks.scene_builders.backgrounds import GLB_SCENE_DIR, BackgroundAdapter

SCENE_ID = "labutopia_lab003"
ROOM_FILE = "labutopia_lab003/room.glb"
TABLE_HEIGHT = 0.9196429
# Same room transform as the approved pipette preview, lifted so the floor is 0.
ROOM_OFFSET = (-1.35, 0.8, 0.0834567)
TABLE_CENTER_X = 0.18


class LabUtopiaAdapter(BackgroundAdapter):
    """Static collidable room and one empty, grey collection-size work table."""

    def build(self, configs: list) -> None:
        from mani_skill.utils.scene_builder.table import scene_builder as stock_table

        self.actors = []
        table_mesh = Path(stock_table.__file__).parent / "assets/table.glb"
        quarter_turn = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        physical = sapien.physx.PhysxMaterial(0.30, 0.30, 0.0)
        # Give this scene its own finish; never mutate SAPIEN's cached table material.
        template = sapien.render.RenderShapeTriangleMesh(str(table_mesh))
        original = template.parts[0].material
        finish = sapien.render.RenderMaterial(
            base_color=[0.22, 0.25, 0.28, 1.0],
            roughness=0.62,
            metallic=original.metallic,
            specular=original.specular,
        )
        finish.normal_texture = original.normal_texture
        finish.roughness_texture = original.roughness_texture
        for i, config in enumerate(configs):
            room_file = GLB_SCENE_DIR / config["file"]
            if not room_file.is_file():
                raise FileNotFoundError(
                    f"LabUtopia room {room_file} is missing; run "
                    "scripts/scene_library.sh fetch-labutopia (see docs/howto_labutopia.md)"
                )
            room = self.env.scene.create_actor_builder()
            room.set_scene_idxs([i])
            room.add_visual_from_file(str(room_file))
            room.add_nonconvex_collision_from_file(str(room_file), material=physical)
            room.initial_pose = sapien.Pose(ROOM_OFFSET, [0, 0, 0, 1])
            self.actors.append(room.build_static(name=f"labutopia_room_{i}"))

            table = self.env.scene.create_actor_builder()
            table.set_scene_idxs([i])
            # Match ManiSkill's table mesh, scale, orientation and support box.
            table.add_visual_from_file(
                str(table_mesh),
                scale=[1.75] * 3,
                pose=sapien.Pose(q=quarter_turn),
                material=finish,
            )
            table.add_box_collision(
                pose=sapien.Pose([0, 0, TABLE_HEIGHT / 2]),
                half_size=[2.418 / 2, 1.209 / 2, TABLE_HEIGHT / 2],
                material=physical,
            )
            table.initial_pose = sapien.Pose([TABLE_CENTER_X, 0, 0], quarter_turn)
            self.actors.append(table.build_static(name=f"labutopia_worktable_{i}"))

    def work_surface_anchors(self, reach: float, inset: float):
        return [(0.0, 0.0, TABLE_HEIGHT, 0.0)] * self.env.num_envs

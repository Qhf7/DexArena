"""Randomized table scene builder.

Drop-in replacement for ``mani_skill.utils.scene_builder.table.TableSceneBuilder``
that adds opt-in domain randomization for:

* table height (jittered downward per reset)
* table top texture (per-sub-env, resampled on every ``initialize()`` call)
* background enclosure — four walls + a ceiling around the workstation
  (visual-only actor with per-sub-env texture, also resampled per reset)

Defaults are inert. With no flags enabled, behavior matches the parent class.

Known limitations:
    1. Height jitter shifts the table actor only. Tasks placing objects on the
       table must read ``self.table_top_offsets[env_idx]`` to keep them planted.
    2. Texture pool is procedurally generated (solid colors + gradients), not
       photographic. Swap ``texture_dir`` for a real-texture library when
       desired.

Usage:
    self.table_scene = RandomizedTableSceneBuilder(
        env=self,
        random_table_height=0.03,
        random_table_texture=True,
        random_background=True,
    )
    self.table_scene.build()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien
import torch
from tasks.scene_builders.table import TableSceneBuilder
from sapien.render import RenderTexture2D

from tasks.scene_builders.textures import _ensure_texture_pool


DEFAULT_TEXTURE_DIR = Path.home() / ".maniskill" / "data" / "scene_textures" / "v1"


class RandomizedTableSceneBuilder(TableSceneBuilder):
    def __init__(
        self,
        env,
        random_table_height: float = 0.0,
        random_table_texture: bool = False,
        random_background: bool = False,
        random_lighting: bool = False,
        room_center: tuple = (0.3, 0.0, 0.7),
        room_half_size: tuple = (1.7, 1.6, 1.0),
        wall_thickness: float = 0.02,
        texture_dir: Path | None = None,
        n_textures: int = 100,
        seed: int = 42,
        robot_init_qpos_noise: float = 0.02,
        ambient_range: tuple[float, float] = (0.1, 0.6),
        tint_range: tuple[float, float] = (0.8, 1.2),
        light_down_bias: float = 0.3,
    ):
        super().__init__(env=env, robot_init_qpos_noise=robot_init_qpos_noise)
        self.random_table_height = random_table_height
        self.random_table_texture = random_table_texture
        self.random_background = random_background
        self.random_lighting = random_lighting
        self.room_center = room_center
        self.room_half_size = room_half_size
        self.wall_thickness = wall_thickness
        self.texture_dir = Path(texture_dir) if texture_dir is not None else DEFAULT_TEXTURE_DIR
        self.n_textures = n_textures
        self.seed = seed
        self.ambient_range = ambient_range
        self.tint_range = tint_range
        self.light_down_bias = light_down_bias
        self.wall = None
        self.table_top_offsets = torch.zeros(env.num_envs, device=env.device)
        self._sampled_ambients: np.ndarray | None = None
        # path -> RenderTexture2D, reused across resets to avoid re-uploading.
        self._tex_cache: dict = {}
        # sub-env index -> directional light component, so resets replace rather
        # than accumulate lights.
        self._dir_lights: dict = {}

    def build(self):
        super().build()
        if not (self.random_table_texture or self.random_background):
            return
        # Load pool once; sampling deferred to initialize().
        self._tex_paths = _ensure_texture_pool(self.texture_dir, self.n_textures, self.seed)
        self._table_tex_idx = np.zeros(self.env.num_envs, dtype=np.int64)
        self._wall_tex_idx = np.zeros(self.env.num_envs, dtype=np.int64)
        if self.random_background:
            self._build_wall()  # wall actor itself is build-time; only its texture re-binds per reset

    def initialize(self, env_idx):
        super().initialize(env_idx)
        self._initialize_height(env_idx)
        if self.random_table_texture or self.random_background:
            self._resample_and_apply_textures(env_idx)
        self._apply_lighting(env_idx)

    def _initialize_height(self, env_idx):
        if self.random_table_height <= 0.0:
            return
        from mani_skill.utils.structs.pose import Pose

        rng = self.env._batched_episode_rng[env_idx]
        delta = rng.uniform(-self.random_table_height, 0.0).astype("float32")
        delta_t = torch.as_tensor(delta, device=self.table_top_offsets.device)
        self.table_top_offsets[env_idx] = delta_t
        cur_p = self.table.pose.p.clone()
        cur_q = self.table.pose.q.clone()
        cur_p[env_idx, 2] = cur_p[env_idx, 2] + delta_t.to(cur_p.device)
        self.table.set_pose(Pose.create_from_pq(p=cur_p, q=cur_q))

    def _resample_and_apply_textures(self, env_idx):
        # env_idx is a torch.LongTensor of sub-env indices being reset this call.
        idx_list = env_idx.tolist() if hasattr(env_idx, "tolist") else list(env_idx)
        rng = self.env._episode_rng
        if self.random_table_texture:
            new_table = rng.randint(0, self.n_textures, size=len(idx_list))
            self._table_tex_idx[idx_list] = new_table
            self._apply_table_textures(idx_list)
        if self.random_background:
            new_wall = rng.randint(0, self.n_textures, size=len(idx_list))
            self._wall_tex_idx[idx_list] = new_wall
            self._apply_wall_textures(idx_list)

    def _apply_lighting(self, env_idx):
        """Per-sub-env ambient + one directional light. Called from initialize()."""
        if not self.random_lighting:
            return
        idx_list = env_idx.tolist() if hasattr(env_idx, "tolist") else list(env_idx)
        rng = self.env._episode_rng
        if self._sampled_ambients is None:
            self._sampled_ambients = np.zeros((self.env.num_envs, 3), dtype=np.float32)
        for i in idx_list:
            amb = rng.uniform(self.ambient_range[0], self.ambient_range[1], size=3).astype(np.float32)
            self._sampled_ambients[i] = amb
            sub = self.scene.sub_scenes[i]
            sub.set_ambient_light(amb.tolist())
            # Drop the previous reset's directional light so they don't pile up
            # (add_directional_light appends; it does not replace).
            old = self._dir_lights.pop(i, None)
            if old is not None:
                try:
                    sub.remove_light(old)
                except Exception:
                    try:
                        sub.remove_entity(old.entity)
                    except Exception:
                        pass
            # One directional light with randomized direction + warm/cool tint
            direction = rng.uniform(-1.0, 1.0, size=3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            # Force the light to point somewhat downward
            direction[2] = -abs(direction[2]) - self.light_down_bias
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            tint = rng.uniform(self.tint_range[0], self.tint_range[1], size=3).astype(np.float32)
            self._dir_lights[i] = sub.add_directional_light(
                direction=direction.tolist(),
                color=tint.tolist(),
                shadow=False,
            )

    def _texture(self, path: str) -> RenderTexture2D:
        """Load a texture once and reuse it; textures are immutable by path."""
        tex = self._tex_cache.get(path)
        if tex is None:
            tex = RenderTexture2D(filename=path, mipmap_levels=4)
            self._tex_cache[path] = tex
        return tex

    def _retexture(self, part, tex_path: str):
        tex = self._texture(tex_path)
        rbc = part.find_component_by_type(sapien.render.RenderBodyComponent)
        for shape in rbc.render_shapes:
            for tri in shape.parts:
                tri.material.set_base_color_texture(tex)
                tri.material.set_normal_texture(None)
                tri.material.set_metallic_texture(None)
                tri.material.set_roughness_texture(None)

    def _apply_table_textures(self, idx_list):
        for i in idx_list:
            self._retexture(self.table._objs[i], str(self._tex_paths[self._table_tex_idx[i]]))

    def _apply_wall_textures(self, idx_list):
        for i in idx_list:
            self._retexture(self.wall._objs[i], str(self._tex_paths[self._wall_tex_idx[i]]))

    def _build_wall(self):
        """Enclose the workstation in four walls + a ceiling (visual only).

        The floor is provided by the ManiSkill ground plane, so only the four
        side walls and the top are added. Each surface is a thin box centered on
        ``room_center``; all five share one actor per sub-env so a single texture
        swap in :meth:`_apply_wall_textures` re-skins the whole room.
        """
        from mani_skill.utils.structs.actor import Actor
        from sapien.render import RenderMaterial

        hx, hy, hz = self.room_half_size
        t = self.wall_thickness
        # (local center offset, half extents) for each surface.
        surfaces = [
            ([hx, 0.0, 0.0], [t / 2, hy, hz]),   # +x wall
            ([-hx, 0.0, 0.0], [t / 2, hy, hz]),  # -x wall
            ([0.0, hy, 0.0], [hx, t / 2, hz]),   # +y wall
            ([0.0, -hy, 0.0], [hx, t / 2, hz]),  # -y wall
            ([0.0, 0.0, hz], [hx, hy, t / 2]),   # ceiling
        ]

        per_env_walls = []
        for i in range(self.env.num_envs):
            wb = self.scene.create_actor_builder()
            wb.set_scene_idxs([i])
            for offset, half in surfaces:
                wb.add_box_visual(
                    pose=sapien.Pose(p=offset),
                    half_size=half,
                    material=RenderMaterial(),
                )
            wb.initial_pose = sapien.Pose(p=list(self.room_center))
            per_env_walls.append(wb.build_kinematic(name=f"wall_{i}"))

        self.wall = Actor.merge(per_env_walls, name="wall")
        self.scene_objects.append(self.wall)

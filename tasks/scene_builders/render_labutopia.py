"""Render the empty laboratory, or an existing task placed in it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/labutopia"))
    parser.add_argument(
        "--env-id",
        help="Optional canonical subtask; omitted means a truly empty work table",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shader", choices=("default", "rt"), default="default")
    parser.add_argument(
        "--spp", type=int, default=64, help="Samples per pixel for --shader rt"
    )
    parser.add_argument("--denoiser", choices=("none", "oidn"), default="none")
    args = parser.parse_args(argv)

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import numpy as np
    import sapien
    from mani_skill.utils import sapien_utils
    from PIL import Image

    from tasks.scene_builders.labutopia import SCENE_ID, TABLE_HEIGHT
    from tasks.scene_builders.scene_library import FullSceneBuilder

    if args.env_id:
        from utils.dex_scene_eval import make_scene_eval_env

        env = make_scene_eval_env(
            args.env_id, scene_background=SCENE_ID, obs_mode="state"
        )
        height = 0.0
    else:
        env = gym.make(
            "Empty-v1",
            # Some ManiSkill versions snapshot a controller even for robot='none'.
            # Build the stock agent and park it outside the room for this preview.
            robot_uids="panda",
            obs_mode="none",
            sim_backend="physx_cpu",
            render_mode="rgb_array",
        )
        height = TABLE_HEIGHT
    try:
        env.reset(seed=args.seed)
        u = env.unwrapped
        if not args.env_id:
            builder = FullSceneBuilder(u, scene_ids=SCENE_ID)
            builder.build()
            u.agent.robot.set_root_pose(sapien.Pose([0, 0, -500]))
            for entity in u.ground._objs:
                render = entity.find_component_by_type(
                    sapien.render.RenderBodyComponent
                )
                if render is not None:
                    render.disable()
                body = entity.find_component_by_type(
                    sapien.physx.PhysxRigidBaseComponent
                )
                for shape in body.get_collision_shapes():
                    shape.set_collision_groups([0, 0, 0, 0])
        scene = u.scene.sub_scenes[0]
        scene.set_ambient_light([0.30, 0.31, 0.33])
        scene.add_directional_light([0.3, -0.6, -1], [0.7, 0.69, 0.67], shadow=False)
        sapien.render.set_camera_shader_dir(args.shader)
        if args.shader == "rt":
            sapien.render.set_ray_tracing_samples_per_pixel(args.spp)
            sapien.render.set_ray_tracing_path_depth(4)
            sapien.render.set_ray_tracing_denoiser(args.denoiser)
            scene.add_area_light_for_ray_tracing(
                sapien.Pose([-0.3, 0.4, height + 1.65], [0.7071068, 0, 0.7071068, 0]),
                [1.2, 1.2, 1.2],
                0.7,
                0.7,
            )
        args.out.mkdir(parents=True, exist_ok=True)
        views = []
        for name, eye, target, fov, width, height_px in (
            ("close", [-0.78, 0.46, 0.56], [-0.13, -0.05, 0.25], 1.06, 1200, 800),
            ("right_rear", [-2.15, -1.65, 1.55], [-0.10, 0, -0.20], 0.90, 1200, 1200),
        ):
            camera = scene.add_camera(name, width, height_px, fov, 0.01, 100)
            camera.set_local_pose(
                sapien_utils.look_at(
                    np.asarray(eye) + [0, 0, height],
                    np.asarray(target) + [0, 0, height],
                ).sp
            )
            u.scene.update_render()
            camera.take_picture()
            pixels = np.asarray(camera.get_picture("Color"))[..., :3]
            if pixels.dtype != np.uint8:
                pixels = np.clip(pixels * 255, 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(args.out / f"{name}.png")
            views.append(name)
        (args.out / "manifest.json").write_text(
            json.dumps(
                {
                    "scene_id": SCENE_ID,
                    "task": args.env_id,
                    "seed": args.seed,
                    "shader": args.shader,
                    "denoiser": args.denoiser if args.shader == "rt" else None,
                    "background_tabletop_props": [],
                    "views": views,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

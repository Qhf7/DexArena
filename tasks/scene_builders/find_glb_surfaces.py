"""Find the work surfaces inside a single-mesh scene.

A room that ships as one mesh (an Objaverse interior, say) has no actors to query
and no names to match, so the anchor for such a scene has to be measured. This
reports the horizontal, upward-facing patches at working height, biggest first,
already transformed by the spec's ``euler``/``offset`` so the numbers can be
pasted straight into a ``SceneSpec.anchor``.

    ./scripts/scene_library.sh surfaces workshop_minimalistic
    python -m tasks.scene_builders.find_glb_surfaces workshop_minimalistic

``yaw`` still has to be chosen by eye: it points from the robot into the surface,
so pick the side with floor behind it and confirm with a render.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tasks.scene_builders.backgrounds import GLB_SCENE_DIR, WORK_HEIGHT_RANGE
from tasks.scene_builders.scene_library import get_scene_spec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", help="name of a glb-source scene in the library")
    p.add_argument("--min-area", type=float, default=0.15, help="m^2")
    p.add_argument("--z-tol", type=float, default=0.02, help="height bucket size, m")
    args = p.parse_args(argv)

    import trimesh
    from transforms3d.euler import euler2mat

    spec = get_scene_spec(args.scene)
    if spec.source != "glb":
        raise SystemExit(f"'{args.scene}' is a {spec.source} scene, not a mesh")
    cfg = spec.config

    path = Path(cfg["file"])
    if not path.is_absolute():
        path = GLB_SCENE_DIR / path
    mesh = trimesh.load(path, force="mesh")

    rot = euler2mat(*cfg.get("euler", (np.pi / 2, 0.0, 0.0)))
    verts = mesh.vertices * float(cfg.get("scale", 1.0)) @ rot.T + np.asarray(
        cfg.get("offset", (0.0, 0.0, 0.0))
    )
    tris = verts[mesh.faces]

    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    with np.errstate(invalid="ignore"):
        up = normals[:, 2] / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
    centers = tris.mean(axis=1)

    # Upward-facing (within ~15 degrees of flat), at a height worth working at.
    flat = (up > 0.96) & (centers[:, 2] >= WORK_HEIGHT_RANGE[0]) & (centers[:, 2] <= WORK_HEIGHT_RANGE[1])
    if not flat.any():
        print(f"no horizontal surface in {WORK_HEIGHT_RANGE} m")
        return 1

    buckets: dict[int, list[int]] = {}
    for idx in np.flatnonzero(flat):
        buckets.setdefault(int(round(centers[idx, 2] / args.z_tol)), []).append(idx)

    rows = []
    for key, idxs in buckets.items():
        area = float(areas[idxs].sum())
        if area < args.min_area:
            continue
        pts = tris[idxs].reshape(-1, 3)
        rows.append((area, key * args.z_tol, pts.min(axis=0), pts.max(axis=0)))

    if not rows:
        print(f"nothing above --min-area {args.min_area} m^2")
        return 1
    for area, z, lo, hi in sorted(rows, reverse=True, key=lambda r: r[0]):
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        print(
            f"z={z:5.2f} m  area={area:6.2f} m^2  "
            f"x=[{lo[0]:6.2f},{hi[0]:6.2f}]  y=[{lo[1]:6.2f},{hi[1]:6.2f}]  "
            f"center=({cx:.2f}, {cy:.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

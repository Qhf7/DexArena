"""Convert only lab_003's room subtree; no loose task instruments are exported.

USD world transforms, face-varying UVs and material subsets are preserved.
MDL shaders are approximated as glTF PBR; large meshes are simplified for SAPIEN.
Dependencies are needed only when preparing assets, never at simulation runtime.
"""

from pathlib import Path


def convert(source: Path, destination: Path) -> dict:
    import json
    import re
    from collections import defaultdict

    import numpy as np
    import trimesh
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdShade

    SOURCE = Path(source)
    stage = Usd.Stage.OpenMasked(
        str(SOURCE), Usd.StagePopulationMask(["/World/lab_015"])
    )
    xcache = UsdGeom.XformCache()
    groups = defaultdict(list)
    materials = {}
    records = []

    def material(prim):
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        key = str(mat.GetPath()) if mat else "default"
        if key in materials:
            return key
        color = np.array([0.72, 0.74, 0.75, 1.0])
        metallic, roughness, texture = 0.0, 0.65, None
        attrs = {}
        mdl = None
        if mat:
            for child in Usd.PrimRange(mat.GetPrim()):
                for a in child.GetAttributes():
                    value = a.Get()
                    if value is not None:
                        attrs[a.GetName()] = value
                        if a.GetName() == "info:mdl:sourceAsset":
                            p = (
                                Path(value.resolvedPath)
                                if value.resolvedPath
                                else SOURCE.parent / value.path
                            )
                            if p.exists():
                                mdl = p
        if mdl:
            text = mdl.read_text(errors="replace")
            m = re.search(r"Diffuse\s*:\s*color\(([^)]+)\)", text)
            if m:
                color[:3] = [float(v.rstrip("f ")) for v in m.group(1).split(",")]
            m = re.search(r'texture_2d\("([^"]+)"', text)
            if m:
                candidate = (mdl.parent / m.group(1)).resolve()
                if candidate.exists():
                    texture = candidate
            m = re.search(r"reflection_glossiness\s*:\s*([\d.]+)", text)
            if m:
                roughness = max(0.12, 1.0 - float(m.group(1)))
            m = re.search(r"reflection_metalness\s*:\s*([\d.]+)", text)
            if m:
                metallic = float(m.group(1))
        for name in [
            "inputs:diffuse_color",
            "inputs:diffuse_color_constant",
            "inputs:base_color_factor",
            "inputs:diffuseColor",
        ]:
            if name in attrs:
                color[:3] = np.asarray(attrs[name])[:3]
        if "inputs:diffuse_tint" in attrs:
            color[:3] *= np.asarray(attrs["inputs:diffuse_tint"])
        for name in [
            "inputs:reflection_roughness_constant",
            "inputs:roughness_factor",
            "inputs:roughness",
        ]:
            if name in attrs:
                roughness = float(attrs[name])
        for name in ["inputs:metallic_factor", "inputs:metallic"]:
            if name in attrs:
                metallic = float(attrs[name])
        for name in ["inputs:diffuse_texture", "inputs:file", "inputs:texture"]:
            if name in attrs and hasattr(attrs[name], "path"):
                value = attrs[name]
                p = (
                    Path(value.resolvedPath)
                    if value.resolvedPath
                    else SOURCE.parent / value.path
                )
                if p.is_file():
                    texture = p
        if "glass" in key.lower():
            color = np.array([0.7, 0.86, 0.9, 0.20])
            roughness = 0.12
        im = None
        if texture:
            im = Image.open(texture).convert("RGB")
            im.thumbnail((1024, 1024))
            color[:3] = 1
        color[:3] = np.clip(color[:3], 0, 1)
        materials[key] = trimesh.visual.material.PBRMaterial(
            name=key.rsplit("/", 1)[-1],
            baseColorFactor=(color * 255).astype(np.uint8),
            baseColorTexture=im,
            metallicFactor=metallic,
            roughnessFactor=max(0.15, roughness),
            doubleSided=True,
            alphaMode="BLEND" if color[3] < 1 else "OPAQUE",
        )
        return key

    for prim in Usd.PrimRange(
        stage.GetPrimAtPath("/World/lab_015"), Usd.TraverseInstanceProxies()
    ):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not len(vertices) or not len(indices):
            continue
        transform = np.asarray(xcache.GetLocalToWorldTransform(prim)).T
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        corners, face_ids = [], []
        start = 0
        for face_id, count in enumerate(counts):
            for j in range(1, count - 1):
                corners.append([start, start + j, start + j + 1])
                face_ids.append(face_id)
            start += count
        corners = np.asarray(corners, dtype=np.int64)
        face_ids = np.asarray(face_ids, dtype=np.int64)
        faces = indices[corners]
        uvs = None
        pvapi = UsdGeom.PrimvarsAPI(prim)
        for uvname in ["st", "UVMap", "st0", "map1"]:
            pv = pvapi.GetPrimvar(uvname)
            if pv and pv.HasValue():
                if not len(pv.Get()):
                    continue
                values = np.asarray(pv.ComputeFlattened())
                if values.ndim != 2:
                    continue
                interp = pv.GetInterpolation()
                if interp == "faceVarying" and len(values) == len(indices):
                    uvs = values[corners]
                elif interp in ("vertex", "varying") and len(values) == len(vertices):
                    uvs = values[faces]
                break
        assignment = np.full(len(counts), material(prim), dtype=object)
        for child in prim.GetChildren():
            if child.IsA(UsdGeom.Subset):
                subset = UsdGeom.Subset(child)
                ii = subset.GetIndicesAttr().Get()
                if ii is not None and subset.GetElementTypeAttr().Get() == "face":
                    assignment[np.asarray(ii, dtype=np.int64)] = material(child)
        for key in sorted(set(assignment.tolist())):
            selected = assignment[face_ids] == key
            verts = vertices[faces[selected]].reshape(-1, 3)
            ff = np.arange(len(verts)).reshape(-1, 3)
            if (
                mesh.GetOrientationAttr().Get() == "leftHanded"
                or np.linalg.det(transform[:3, :3]) < 0
            ):
                ff = ff[:, ::-1]
            uv = (
                uvs[selected].reshape(-1, 2)
                if uvs is not None
                else np.zeros((len(verts), 2))
            )
            groups[key].append((verts.astype(np.float32), ff, uv.astype(np.float32)))
        records.append(
            {
                "path": str(prim.GetPath()),
                "bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                "triangles": len(faces),
            }
        )

    scene = trimesh.Scene()
    for key, chunks in groups.items():
        verts, faces, uvs, offset = [], [], [], 0
        for v, f, uv in chunks:
            verts.append(v)
            faces.append(f + offset)
            uvs.append(uv)
            offset += len(v)
        m = trimesh.Trimesh(
            vertices=np.concatenate(verts), faces=np.concatenate(faces), process=False
        )
        m.visual = trimesh.visual.TextureVisuals(
            uv=np.concatenate(uvs), material=materials[key]
        )
        m.merge_vertices()
        if len(m.faces) > 12000:
            from scipy.spatial import cKDTree

            smaller = m.simplify_quadric_decimation(
                face_count=max(6000, len(m.faces) // 8)
            )
            nearest = cKDTree(m.vertices).query(smaller.vertices)[1]
            smaller.visual = trimesh.visual.TextureVisuals(
                uv=m.visual.uv[nearest], material=materials[key]
            )
            m = smaller
        scene.add_geometry(m, geom_name=key.replace("/", "_"))
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output)
    (output.parent / "room_geometry.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    return {
        "meshes": len(records),
        "materials": len(groups),
        "triangles": sum(len(m.faces) for m in scene.geometry.values()),
        "bounds": scene.bounds.tolist(),
    }

"""Render metric DSLR depth maps from aligned ScanNet++ meshes.

The output layout matches ``ScanNetPPDataset`` when
``native_image_directory=resized_undistorted_images``. Existing non-empty PNGs
are skipped, making the preprocessing pass resumable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from numpy.typing import NDArray


def _scene_ids(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Scene list must be a JSON list: {path}")
        return [str(scene_id) for scene_id in payload]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _camera_value(frame: dict[str, Any], metadata: dict[str, Any], key: str) -> float:
    value = frame.get(key, metadata.get(key))
    if value is None:
        raise ValueError(f"Camera metadata has no {key!r}")
    return float(value)


# The 5 cm reconstruction mesh and metadata AABB use different source meshes;
# their extrema can differ by a few millimeters after decimation.
MESH_BOUNDS_TOLERANCE_METERS = 5e-3


def _mesh_as_trimesh(path: Path, trimesh_module: Any) -> Any:
    loaded = trimesh_module.load(path, process=False)
    if isinstance(loaded, trimesh_module.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {path}")
        return trimesh_module.util.concatenate(geometries)
    return loaded


def _mesh_in_nerfstudio_coordinates(mesh: Any, metadata: dict[str, Any]) -> Any:
    expected_bounds = metadata.get("aabb_range")
    if expected_bounds is None:
        return mesh
    expected = np.asarray(expected_bounds, dtype=np.float64)
    if np.allclose(mesh.bounds, expected, rtol=0.0, atol=MESH_BOUNDS_TOLERANCE_METERS):
        return mesh
    # ScanNet++ exports meshes as (x, y, z), while its Nerfstudio DSLR poses
    # and aabb_range use (y, x, -z).
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    aligned = mesh.copy()
    aligned.apply_transform(transform)
    if not np.allclose(aligned.bounds, expected, rtol=0.0, atol=MESH_BOUNDS_TOLERANCE_METERS):
        raise ValueError(
            "ScanNet++ mesh bounds do not match Nerfstudio aabb_range after "
            f"coordinate conversion: {aligned.bounds} versus {expected}"
        )
    return aligned


def _invalidate_anonymous_pixels(
    depth: NDArray[np.float32], mask_path: Path
) -> NDArray[np.float32]:
    if not mask_path.is_file():
        return depth
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        if mask.size != (depth.shape[1], depth.shape[0]):
            mask = mask.resize((depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST)
        anonymous = np.asarray(mask) < 255
    depth[anonymous] = 0.0
    return depth


def _is_valid_depth_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
            return image.format == "PNG" and image.mode in {"I", "I;16"}
    except OSError:
        return False


def render_scene(
    scene_root: Path,
    image_directory: str,
    depth_directory: str,
    overwrite: bool,
    max_frames: int | None,
    pyrender: Any,
    trimesh: Any,
) -> tuple[int, int]:
    dslr_root = scene_root / "dslr"
    metadata_path = dslr_root / "nerfstudio" / "transforms_undistorted.json"
    if not metadata_path.is_file():
        metadata_path = dslr_root / "nerfstudio" / "transforms.json"
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in {metadata_path}")

    image_root = dslr_root / image_directory
    depth_root = dslr_root / depth_directory
    depth_root.mkdir(parents=True, exist_ok=True)
    mask_root = dslr_root / "resized_undistorted_masks"
    mesh_path = scene_root / "scans" / "mesh_aligned_0.05.ply"
    scene_mesh = _mesh_as_trimesh(mesh_path, trimesh)
    scene_mesh = _mesh_in_nerfstudio_coordinates(scene_mesh, metadata)
    mesh = pyrender.Mesh.from_trimesh(scene_mesh, smooth=False)
    render_scene = pyrender.Scene()
    render_scene.add(mesh)
    renderer = pyrender.OffscreenRenderer(1, 1)
    rendered = 0
    skipped = 0
    try:
        for frame in frames:
            if max_frames is not None and rendered >= max_frames:
                break
            name = Path(str(frame["file_path"])).name
            image_path = image_root / name
            if not image_path.is_file():
                continue
            depth_path = depth_root / f"{image_path.stem}.png"
            if not overwrite and _is_valid_depth_image(depth_path):
                skipped += 1
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            camera_width = _camera_value(frame, metadata, "w")
            camera_height = _camera_value(frame, metadata, "h")
            scale_x = width / camera_width
            scale_y = height / camera_height
            camera = pyrender.IntrinsicsCamera(
                fx=_camera_value(frame, metadata, "fl_x") * scale_x,
                fy=_camera_value(frame, metadata, "fl_y") * scale_y,
                cx=_camera_value(frame, metadata, "cx") * scale_x,
                cy=_camera_value(frame, metadata, "cy") * scale_y,
                znear=0.05,
                zfar=20.0,
            )
            pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f"Invalid camera pose for {name}: {pose.shape}")
            camera_node = render_scene.add(camera, pose=pose)
            renderer.viewport_width = width
            renderer.viewport_height = height
            try:
                depth = renderer.render(render_scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            finally:
                render_scene.remove_node(camera_node)
            if not np.any(depth > 0.0):
                raise RuntimeError(
                    f"Rendered empty depth for {scene_root.name}/{name}; check pose convention"
                )
            depth = _invalidate_anonymous_pixels(depth, mask_root / f"{image_path.stem}.png")
            depth_mm = np.clip(np.rint(depth * 1000.0), 0, 65535).astype(np.uint16)
            temporary = depth_path.with_suffix(".tmp")
            Image.fromarray(depth_mm).save(temporary, format="PNG")
            temporary.replace(depth_path)
            rendered += 1
    finally:
        renderer.delete()
    return rendered, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--image-directory", default="resized_undistorted_images")
    parser.add_argument("--depth-directory", default="resized_undistorted_depths")
    parser.add_argument("--pyopengl-platform", default="egl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    os.environ["PYOPENGL_PLATFORM"] = args.pyopengl_platform
    try:
        pyrender = importlib.import_module("pyrender")
        trimesh = importlib.import_module("trimesh")
    except ImportError as error:
        raise RuntimeError("Install pyrender and trimesh to render ScanNet++ depths") from error

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    scenes = _scene_ids(args.scene_list)[args.shard_index :: args.num_shards]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    totals = {
        "scenes": 0,
        "rendered": 0,
        "skipped": 0,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    for index, scene_id in enumerate(scenes, start=1):
        rendered, skipped = render_scene(
            args.data_root / scene_id,
            args.image_directory,
            args.depth_directory,
            args.overwrite,
            args.max_frames,
            pyrender,
            trimesh,
        )
        totals["scenes"] += 1
        totals["rendered"] += rendered
        totals["skipped"] += skipped
        print(
            json.dumps(
                {
                    "scene": scene_id,
                    "index": index,
                    "total_scenes": len(scenes),
                    "rendered": rendered,
                    "skipped": skipped,
                }
            ),
            flush=True,
        )
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()

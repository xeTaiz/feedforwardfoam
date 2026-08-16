#!/usr/bin/env python3
"""Rank geometric camera triplets from a native ScanNet++ DSLR scene.

The utility only reads the scene. It writes ranked metadata and a contact sheet
under ``--output-dir``; it does not modify the ScanNet++ source tree.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

MAX_INTERPOLATION_OFFSET = 0.25
MAX_PERPENDICULAR_FRACTION = 0.25
MAX_ANGLE_DEGREES = 40.0


@dataclass(frozen=True)
class Camera:
    """Geometry and source-image information for one valid frame."""

    name: str
    image_path: Path
    center: np.ndarray
    forward: np.ndarray


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_cameras(scene_root: Path) -> list[Camera]:
    transforms_path = scene_root / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"ScanNet++ metadata not found: {transforms_path}")

    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = metadata.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Expected a 'frames' list in {transforms_path}")

    image_dir = scene_root / "dslr" / "resized_undistorted_images"
    cameras: list[Camera] = []
    for frame_index, frame in enumerate(frames):
        if frame.get("is_bad"):
            continue
        try:
            name = Path(frame["file_path"]).name
            transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid frame at index {frame_index}: {error}") from error
        if not name:
            raise ValueError(f"Frame at index {frame_index} has an empty file_path")
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"Frame {name!r} has a non-finite or non-4x4 transform")

        # Nerfstudio camera-to-world poses look down local -Z.
        forward = -transform[:3, 2]
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-12:
            raise ValueError(f"Frame {name!r} has a degenerate forward axis")
        cameras.append(
            Camera(
                name=name,
                image_path=image_dir / name,
                center=transform[:3, 3].copy(),
                forward=forward / forward_norm,
            )
        )

    cameras.sort(key=lambda camera: camera.name)
    if len({camera.name for camera in cameras}) != len(cameras):
        raise ValueError("Valid frames must have unique image basenames")
    if len(cameras) < 3:
        raise ValueError(f"Need at least 3 non-bad frames, found {len(cameras)}")
    return cameras


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _candidate(
    context0: Camera,
    context1: Camera,
    target: Camera,
    local_scale: float,
) -> dict[str, Any] | None:
    segment = context1.center - context0.center
    baseline = float(np.linalg.norm(segment))
    if baseline <= 1e-12:
        return None

    interpolation = float(np.dot(target.center - context0.center, segment) / baseline**2)
    if not 0.25 <= interpolation <= 0.75:
        return None
    projected = context0.center + interpolation * segment
    perpendicular_distance = float(np.linalg.norm(target.center - projected))
    perpendicular_fraction = perpendicular_distance / baseline
    if perpendicular_fraction > MAX_PERPENDICULAR_FRACTION:
        return None

    angle_contexts = _angle_degrees(context0.forward, context1.forward)
    angle_context0_target = _angle_degrees(context0.forward, target.forward)
    angle_target_context1 = _angle_degrees(target.forward, context1.forward)
    angles = (angle_contexts, angle_context0_target, angle_target_context1)
    max_angle = max(angles)
    if max_angle > MAX_ANGLE_DEGREES:
        return None

    interpolation_score = 1.0 - abs(interpolation - 0.5) / MAX_INTERPOLATION_OFFSET
    perpendicular_score = 1.0 - perpendicular_fraction / MAX_PERPENDICULAR_FRACTION
    # x * exp(1-x) is zero at a zero baseline and peaks at a moderate baseline
    # of twice the target's median N-neighbor distance.
    baseline_ratio = baseline / (2.0 * local_scale)
    baseline_score = baseline_ratio * math.exp(1.0 - baseline_ratio)
    angle_score = 1.0 - max_angle / MAX_ANGLE_DEGREES
    score = (
        0.35 * interpolation_score
        + 0.25 * perpendicular_score
        + 0.20 * baseline_score
        + 0.20 * angle_score
    )

    def vector(values: np.ndarray) -> list[float]:
        return [float(value) for value in values]

    return {
        "context0": context0.name,
        "target": target.name,
        "context1": context1.name,
        "score": score,
        "geometry": {
            "interpolation": interpolation,
            "perpendicular_distance": perpendicular_distance,
            "perpendicular_fraction": perpendicular_fraction,
            "context_baseline": baseline,
            "local_scale": local_scale,
            "baseline_ratio": baseline_ratio,
            "angles_degrees": {
                "context0_context1": angle_contexts,
                "context0_target": angle_context0_target,
                "target_context1": angle_target_context1,
                "maximum": max_angle,
            },
            "centers": {
                "context0": vector(context0.center),
                "target": vector(target.center),
                "context1": vector(context1.center),
            },
            "forward_axes": {
                "context0": vector(context0.forward),
                "target": vector(target.forward),
                "context1": vector(context1.forward),
            },
        },
    }


def _rank_candidates(cameras: list[Camera], neighbors: int) -> list[dict[str, Any]]:
    centers = np.stack([camera.center for camera in cameras])
    candidates: list[dict[str, Any]] = []
    for target_index, target in enumerate(cameras):
        distances = np.linalg.norm(centers - target.center, axis=1)
        neighbor_indices = [
            int(index)
            for index in np.argsort(distances, kind="stable")
            if int(index) != target_index
        ][:neighbors]
        positive_distances = [distances[index] for index in neighbor_indices if distances[index] > 0]
        local_scale = float(np.median(positive_distances)) if positive_distances else 1.0
        local_scale = max(local_scale, 1e-12)

        for first_index, second_index in itertools.combinations(neighbor_indices, 2):
            # Canonical name ordering makes endpoint roles independent of distance ties.
            context0, context1 = sorted(
                (cameras[first_index], cameras[second_index]), key=lambda camera: camera.name
            )
            candidate = _candidate(context0, context1, target, local_scale)
            if candidate is not None:
                candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["target"],
            item["context0"],
            item["context1"],
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    return candidates


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _thumbnail(path: Path, resolution: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Frame image not found: {path}")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (resolution, resolution), (20, 20, 20))
    canvas.paste(image, ((resolution - image.width) // 2, (resolution - image.height) // 2))
    return canvas


def _write_contact_sheet(
    output_path: Path,
    candidates: list[dict[str, Any]],
    cameras_by_name: dict[str, Camera],
    resolution: int,
    top_k: int,
) -> None:
    selected = candidates[:top_k]
    label_height = 42
    header_height = 28
    width = resolution * 3
    height = max(1, len(selected) * (header_height + label_height + resolution))
    sheet = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    font = _font(max(10, min(16, resolution // 14)))
    small_font = _font(max(9, min(13, resolution // 18)))

    for row, candidate in enumerate(selected):
        y = row * (header_height + label_height + resolution)
        geometry = candidate["geometry"]
        header = (
            f"#{candidate['rank']} score={candidate['score']:.3f}  "
            f"t={geometry['interpolation']:.3f}  perp={geometry['perpendicular_fraction']:.3f}  "
            f"baseline={geometry['context_baseline']:.3f}  "
            f"max-angle={geometry['angles_degrees']['maximum']:.1f} deg"
        )
        draw.text((5, y + 5), header, fill=(255, 220, 110), font=font)
        for column, role in enumerate(("context0", "target", "context1")):
            x = column * resolution
            name = candidate[role]
            label_y = y + header_height
            draw.text((x + 5, label_y + 2), role, fill=(130, 210, 255), font=small_font)
            draw.text((x + 5, label_y + 20), name, fill=(245, 245, 245), font=small_font)
            tile = _thumbnail(cameras_by_name[name].image_path, resolution)
            sheet.paste(tile, (x, label_y + label_height))
    sheet.save(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank ScanNet++ DSLR camera triplets whose target lies between two nearby "
            "context cameras, then write candidates.json and a labeled contact sheet."
        )
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        required=True,
        help="Scene directory containing dslr/nerfstudio/transforms_undistorted.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory (the source scene is never modified)",
    )
    parser.add_argument(
        "--image-resolution",
        type=_positive_int,
        default=256,
        help="Square contact-sheet thumbnail size in pixels (default: 256)",
    )
    parser.add_argument(
        "--neighbors",
        type=_positive_int,
        default=30,
        help="Nearest cameras considered as contexts for each target (default: 30)",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=20,
        help="Number of ranked triplets shown in the contact sheet (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cameras = _load_cameras(args.scene_root)
    candidates = _rank_candidates(cameras, args.neighbors)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "scene_root": str(args.scene_root),
        "valid_frame_count": len(cameras),
        "candidate_count": len(candidates),
        "parameters": {
            "neighbors": args.neighbors,
            "top_k": args.top_k,
            "image_resolution": args.image_resolution,
            "interpolation_range": [0.25, 0.75],
            "maximum_perpendicular_fraction": MAX_PERPENDICULAR_FRACTION,
            "maximum_pairwise_forward_angle_degrees": MAX_ANGLE_DEGREES,
        },
        "score": {
            "interpolation_weight": 0.35,
            "perpendicular_weight": 0.25,
            "baseline_weight": 0.20,
            "angle_weight": 0.20,
            "baseline_optimum_local_scale_multiple": 2.0,
        },
        "candidates": candidates,
    }
    candidates_path = args.output_dir / "candidates.json"
    candidates_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_contact_sheet(
        args.output_dir / "contact_sheet.png",
        candidates,
        {camera.name: camera for camera in cameras},
        args.image_resolution,
        args.top_k,
    )
    print(
        f"Wrote {len(candidates)} candidates to {candidates_path} and "
        f"top {min(args.top_k, len(candidates))} to {args.output_dir / 'contact_sheet.png'}"
    )


if __name__ == "__main__":
    main()

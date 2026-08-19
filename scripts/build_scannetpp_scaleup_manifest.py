#!/usr/bin/env python3
"""Build deterministic, scene-disjoint ScanNet++ training episodes.

The official ScanNet++ NVS train/validation split supplies scene separation.
Within each scene, camera triplets are selected by calibrated pose: the target
must lie between two contexts, the baseline must be useful but local, and large
view-angle changes are rejected. Remaining episodes are balanced over three
angle bins. Actual projected support is measured during validation and reported
per bin; it is deliberately not approximated here from poses.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from typing import Any

from select_scannetpp_triplet import (  # pyright: ignore[reportImplicitRelativeImport, reportPrivateUsage]
    _load_cameras,
    _rank_candidates,
)
from feedforwardfoam.data.scannetpp import validate_native_camera

IMAGE_PREFIX = "dslr/resized_undistorted_images"
ANGLE_BINS = (
    ("low_angle", 3.0, 8.0),
    ("mid_angle", 8.0, 16.0),
    ("high_angle", 16.0, 25.0),
)


def _bin_name(candidate: dict[str, Any]) -> str | None:
    geometry = candidate["geometry"]
    if not 0.5 <= float(geometry["baseline_ratio"]) <= 2.5:
        return None
    if float(geometry["perpendicular_fraction"]) > 0.20:
        return None
    angle = float(geometry["angles_degrees"]["maximum"])
    for name, lower, upper in ANGLE_BINS:
        if lower <= angle < upper:
            return name
    return None


def _episode(scene_id: str, candidate: dict[str, Any], bin_name: str) -> dict[str, Any]:
    def image(name: str) -> str:
        return f"{IMAGE_PREFIX}/{name}"

    return {
        "scene_id": scene_id,
        "context_names": [image(candidate["context0"]), image(candidate["context1"])],
        "target_names": [image(candidate["target"])],
        "bin": bin_name,
        "selector_score": float(candidate["score"]),
        "geometry": candidate["geometry"],
    }


def _select_scene_episodes(
    scene_root: Path, *, neighbors: int, episodes_per_scene: int
) -> list[dict[str, Any]]:
    transforms_path = scene_root / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    # The training loader rejects scenes whose native camera breaks the centered
    # square-pixel pinhole model, so reject them here instead of mid-run.
    validate_native_camera(json.loads(transforms_path.read_text()))
    candidates = _rank_candidates(_load_cameras(scene_root), neighbors)
    bins: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in ANGLE_BINS}
    for candidate in candidates:
        bin_name = _bin_name(candidate)
        if bin_name is not None:
            bins[bin_name].append(candidate)

    selected: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    positions = {name: 0 for name in bins}
    labels = [name for name, _, _ in ANGLE_BINS]
    while len(selected) < episodes_per_scene:
        progressed = False
        for label in labels:
            entries = bins[label]
            while positions[label] < len(entries):
                candidate = entries[positions[label]]
                positions[label] += 1
                if candidate["target"] in used_targets:
                    continue
                selected.append(_episode(scene_root.name, candidate, label))
                used_targets.add(candidate["target"])
                progressed = True
                break
            if len(selected) == episodes_per_scene:
                break
        if not progressed:
            break
    for episode in selected:
        for name in (*episode["context_names"], *episode["target_names"]):
            if not (scene_root / name).is_file():
                raise FileNotFoundError(f"Selected view is missing: {scene_root / name}")
    return selected


def _read_scene_ids(path: Path) -> list[str]:
    scene_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"Duplicate scene IDs in {path}")
    return scene_ids


def _select_scene_result(
    request: tuple[Path, int, int],
) -> tuple[str, list[dict[str, Any]], str | None]:
    scene_root, neighbors, episodes_per_scene = request
    try:
        selected = _select_scene_episodes(
            scene_root, neighbors=neighbors, episodes_per_scene=episodes_per_scene
        )
    except (FileNotFoundError, ValueError, KeyError) as error:
        return scene_root.name, [], str(error)
    if not selected:
        return scene_root.name, [], "no triplet passed pose constraints"
    return scene_root.name, selected, None


def _build_split(
    data_dir: Path,
    split_file: Path,
    *,
    neighbors: int,
    episodes_per_scene: int,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_ids = _read_scene_ids(split_file)
    requests = [(data_dir / scene_id, neighbors, episodes_per_scene) for scene_id in scene_ids]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_select_scene_result, requests)
        selected_by_scene = list(results)

    episodes: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for scene_id, selected, error in selected_by_scene:
        if error is not None:
            skipped[scene_id] = error
        else:
            episodes.extend(selected)
    counts = Counter(str(episode["bin"]) for episode in episodes)
    stats = {
        "requested_scenes": len(scene_ids),
        "selected_scenes": len({str(episode["scene_id"]) for episode in episodes}),
        "episodes": len(episodes),
        "bins": dict(sorted(counts.items())),
        "skipped_scenes": skipped,
    }
    return episodes, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scannetpp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--train-episodes-per-scene", type=int, default=6)
    parser.add_argument("--val-episodes-per-scene", type=int, default=3)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.neighbors < 2:
        parser.error("--neighbors must be at least 2")
    if args.train_episodes_per_scene <= 0 or args.val_episodes_per_scene <= 0:
        parser.error("episodes per scene must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    data_dir = args.scannetpp_root / "data"
    splits_dir = args.scannetpp_root / "splits"
    train, train_stats = _build_split(
        data_dir,
        splits_dir / "nvs_sem_train.txt",
        neighbors=args.neighbors,
        episodes_per_scene=args.train_episodes_per_scene,
        workers=args.workers,
    )
    validation, val_stats = _build_split(
        data_dir,
        splits_dir / "nvs_sem_val.txt",
        neighbors=args.neighbors,
        episodes_per_scene=args.val_episodes_per_scene,
        workers=args.workers,
    )
    manifest = {
        "version": 1,
        "selection": {
            "neighbors": args.neighbors,
            "baseline_ratio": [0.5, 2.5],
            "perpendicular_fraction_max": 0.20,
            "angle_bins_degrees": {name: [lower, upper] for name, lower, upper in ANGLE_BINS},
            "note": "Pose bins; projected target support is measured during validation.",
        },
        "train": train,
        "val": validation,
        "stats": {"train": train_stats, "val": val_stats},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["stats"], indent=2))


if __name__ == "__main__":
    main()

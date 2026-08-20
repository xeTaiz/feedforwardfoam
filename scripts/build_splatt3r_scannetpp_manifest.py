"""Build the exact Splatt3R ScanNet++ train/evaluation protocol manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from feedforwardfoam.data.scannetpp import ScanNetPPDataset

BINS = {
    "close": "0.9_0.9",
    "medium": "0.7_0.7",
    "wide": "0.5_0.5",
    "very_wide": "0.3_0.3",
}
BAD_TRAIN_SCENES = {"303745abc7"}


def _read_scene_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _protocol_frame_names(scene_root: Path, image_directory: str) -> list[str]:
    _ = ScanNetPPDataset(
        scene_root,
        split="train",
        context_views=2,
        target_views=1,
        native_image_directory=image_directory,
        image_resolution=256,
    )
    ordered_names = json.loads((scene_root / "dslr" / "train_test_lists.json").read_text())["train"]
    # The published fixed tuple files index the unfiltered train-list order.
    # Filtering is_bad here makes valid published indices exceed the list.
    return [str(Path("dslr") / image_directory / Path(name).name) for name in ordered_names]


def build_manifest(
    *,
    scene_root: Path,
    split_root: Path,
    coverage_root: Path,
    test_assets: Path,
    image_directory: str,
    evaluation_stride: int,
) -> dict[str, Any]:
    if evaluation_stride <= 0:
        raise ValueError("evaluation_stride must be positive")

    train_requested = _read_scene_ids(split_root / "nvs_sem_train.txt")
    train = [
        scene_id
        for scene_id in train_requested
        if scene_id not in BAD_TRAIN_SCENES
        and (scene_root / scene_id).is_dir()
        and (coverage_root / f"{scene_id}.json").is_file()
    ]
    if not train:
        raise ValueError("No training scenes have both ScanNet++ data and Splatt3R coverage")

    official_val = set(_read_scene_ids(split_root / "nvs_sem_val.txt"))
    frame_names: dict[str, list[str]] = {}
    validation: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for label, thresholds in BINS.items():
        samples = json.loads(
            (test_assets / f"splatt3r_scannetpp_test_{thresholds}.json").read_text()
        )
        selected = samples[::evaluation_stride]
        counts[label] = len(selected)
        for scene_id, context_0, context_1, target in selected:
            if scene_id not in official_val:
                raise ValueError(f"Splatt3R test tuple uses non-validation scene {scene_id}")
            if not (scene_root / scene_id).is_dir():
                raise FileNotFoundError(f"Missing Splatt3R evaluation scene: {scene_id}")
            names = frame_names.get(scene_id)
            if names is None:
                names = _protocol_frame_names(scene_root / scene_id, image_directory)
                frame_names[scene_id] = names
            indices = [int(context_0), int(context_1), int(target)]
            if any(index < 0 or index >= len(names) for index in indices):
                raise ValueError(f"Splatt3R tuple index is out of range for {scene_id}: {indices}")
            selected_names = [names[index] for index in indices]
            if any(not (scene_root / scene_id / name).is_file() for name in selected_names):
                raise FileNotFoundError(
                    f"Splatt3R tuple references a missing image in {scene_id}: {selected_names}"
                )
            validation.append(
                {
                    "scene_id": scene_id,
                    "context_names": selected_names[:2],
                    "target_names": selected_names[2:],
                    "bin": label,
                }
            )

    return {
        "version": 1,
        "protocol": "Splatt3R ScanNet++",
        "train": train,
        "val": validation,
        "stats": {
            "requested_train_scenes": len(train_requested),
            "train_scenes": len(train),
            "evaluation_stride": evaluation_stride,
            "evaluation_episodes": len(validation),
            "evaluation_bins": counts,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--test-assets", type=Path, default=Path("data/manifests"))
    parser.add_argument("--image-directory", default="resized_undistorted_images")
    parser.add_argument("--evaluation-stride", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        scene_root=args.scene_root,
        split_root=args.split_root,
        coverage_root=args.coverage_root,
        test_assets=args.test_assets,
        image_directory=args.image_directory,
        evaluation_stride=args.evaluation_stride,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["stats"], indent=2))


if __name__ == "__main__":
    main()

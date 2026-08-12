#!/usr/bin/env python3
"""Copy an audited ScanNet++ DSLR subset used by the P0 multi-view matrix."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.scene_manifest.read_text())
    scene_ids = sorted(set(manifest["train"] + manifest["val"]))
    for scene_id in scene_ids:
        source = args.source_root / scene_id / "dslr"
        destination = args.destination_root / scene_id / "dslr"
        destination.mkdir(parents=True, exist_ok=True)
        for relative in (
            Path("nerfstudio/transforms_undistorted.json"),
            Path("train_test_lists.json"),
        ):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        source_images = source / "resized_undistorted_images"
        target_images = destination / "resized_undistorted_images"
        if target_images.exists():
            shutil.rmtree(target_images)
        shutil.copytree(source_images, target_images)
        print(f"prepared {scene_id}", flush=True)


if __name__ == "__main__":
    main()

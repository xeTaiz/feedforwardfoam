import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from feedforwardfoam.data.multiscene import MultiSceneScanNetPP
from feedforwardfoam.data.scannetpp import ScanNetPPDataset

ROOT = Path(__file__).resolve().parents[1]


def _write_native_scene(root, scene_id: str, views: int = 12):
    scene = root / scene_id
    image_dir = scene / "dslr" / "resized_undistorted_images"
    metadata_dir = scene / "dslr" / "nerfstudio"
    image_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    frames = []
    for index in range(views):
        filename = f"frame_{index:03d}.JPG"
        image = np.zeros((6, 10, 3), dtype=np.uint8)
        image[..., 0] = index
        Image.fromarray(image).save(image_dir / filename)
        pose = np.eye(4)
        pose[0, 3] = index
        frames.append({"file_path": filename, "transform_matrix": pose.tolist()})
    metadata = {
        "camera_model": "PINHOLE",
        "fl_x": 8.0,
        "fl_y": 8.0,
        "cx": 5.0,
        "cy": 3.0,
        "w": 10,
        "h": 6,
        "frames": frames,
        "test_frames": frames[-4:],
    }
    (metadata_dir / "transforms_undistorted.json").write_text(json.dumps(metadata))
    (scene / "dslr" / "train_test_lists.json").write_text(
        json.dumps({"train": [frame["file_path"] for frame in frames], "test": []})
    )
    raw_metadata = {"frames": [dict(frame, is_bad=False) for frame in frames]}
    (metadata_dir / "transforms.json").write_text(json.dumps(raw_metadata))
    return scene


def test_native_scannetpp_center_crops_and_samples_multiple_targets(tmp_path):
    scene = _write_native_scene(tmp_path, "scene-a")
    dataset = ScanNetPPDataset(
        scene,
        split="train",
        context_views=1,
        target_views=4,
        image_resolution=8,
        seed=3,
    )
    episode = dataset[0]
    names = [episode.context[0].name, *(view.name for view in episode.target)]
    assert len(set(names)) == 5
    assert episode.context[0].image.shape == (8, 8, 3)
    assert episode.context[0].fov_x_radians == pytest.approx(2 * math.atan(3 / 8))


def test_native_scannetpp_fov_uses_camera_units_for_scaled_images(tmp_path):
    scene = _write_native_scene(tmp_path, "scaled")
    image_path = scene / "dslr" / "resized_undistorted_images" / "frame_000.JPG"
    Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(image_path)
    dataset = ScanNetPPDataset(
        scene,
        split="train",
        context_views=1,
        target_views=1,
        image_resolution=8,
    )
    assert dataset.episode_from_indices([0, 1]).context[0].fov_x_radians == pytest.approx(
        2 * math.atan(3 / 8)
    )


def test_splatt3r_overlap_sampler_uses_coverage_and_four_targets(tmp_path):
    scene = _write_native_scene(tmp_path, "overlap")
    coverage = np.full((12, 12), 0.8, dtype=np.float32)
    np.fill_diagonal(coverage, 1.0)
    coverage_path = tmp_path / "overlap.json"
    coverage_path.write_text(json.dumps({"overlap": coverage.tolist()}))
    dataset = ScanNetPPDataset(
        scene,
        split="train",
        context_views=2,
        target_views=4,
        image_resolution=8,
        overlap_path=coverage_path,
        context_overlap_threshold=0.5,
        target_overlap_threshold=0.6,
        seed=13,
    )
    state = dataset.generator.get_state()
    first = dataset.sample_episode()
    dataset.generator.set_state(state)
    repeated = dataset.sample_episode()
    assert len(first.context) == 2
    assert len(first.target) == 4
    assert len({view.name for view in first.context + first.target}) == 6
    assert [view.name for view in first.context + first.target] == [
        view.name for view in repeated.context + repeated.target
    ]


def test_multiscene_split_sampling_and_state_roundtrip(tmp_path):
    for scene_id in ("train-a", "train-b", "val-a"):
        _write_native_scene(tmp_path, scene_id)
    manifest = tmp_path / "scenes.json"
    manifest.write_text(json.dumps({"train": ["train-a", "train-b"], "val": ["val-a"]}))
    dataset = MultiSceneScanNetPP(
        tmp_path,
        manifest,
        split="train",
        context_views=1,
        target_views=2,
        image_resolution=8,
        target_pool_size=4,
        reserve_support_view=False,
        seed=4,
    )
    state = dataset.state_dict()
    first = dataset.sample_episode()
    dataset.load_state_dict(state)
    repeated = dataset.sample_episode()
    assert first.scene_id == repeated.scene_id
    assert [view.name for view in first.context + first.target] == [
        view.name for view in repeated.context + repeated.target
    ]
    validation = MultiSceneScanNetPP(
        tmp_path,
        manifest,
        split="val",
        context_views=1,
        target_views=2,
        image_resolution=8,
        target_pool_size=4,
        reserve_support_view=False,
        seed=5,
    )
    episodes = validation.fixed_episodes(3, seed=6)
    assert all(episode.scene_id == "val-a" for episode in episodes)
    assert len(episodes[0].target) == 2


def test_multiscene_explicit_episodes_preserve_triplets_and_balance_bins(tmp_path):
    for scene_id in ("train-a", "train-b", "val-a", "val-b"):
        _write_native_scene(tmp_path, scene_id)
    raw_metadata_path = tmp_path / "val-a" / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    raw_metadata = json.loads(raw_metadata_path.read_text())
    raw_metadata["frames"][0]["is_bad"] = True
    raw_metadata_path.write_text(json.dumps(raw_metadata))
    prefix = "dslr/resized_undistorted_images"
    filtered = ScanNetPPDataset(
        tmp_path / "val-a",
        split="train",
        context_views=2,
        target_views=1,
        image_resolution=8,
    )
    with pytest.raises(ValueError, match="frame_000.JPG"):
        filtered.episode_from_names(
            [f"{prefix}/frame_000.JPG", f"{prefix}/frame_002.JPG"],
            [f"{prefix}/frame_001.JPG"],
        )

    def entry(scene_id: str, label: str, offset: int) -> dict:
        return {
            "scene_id": scene_id,
            "context_names": [
                f"{prefix}/frame_{offset:03d}.JPG",
                f"{prefix}/frame_{offset + 2:03d}.JPG",
            ],
            "target_names": [f"{prefix}/frame_{offset + 1:03d}.JPG"],
            "bin": label,
        }

    manifest = tmp_path / "episodes.json"
    manifest.write_text(
        json.dumps(
            {
                "train": [
                    entry("train-a", "low_angle", 0),
                    entry("train-b", "mid_angle", 3),
                ],
                "val": [
                    entry("val-a", "low_angle", 0),
                    entry("val-b", "mid_angle", 3),
                    entry("val-a", "low_angle", 6),
                ],
            }
        )
    )
    dataset = MultiSceneScanNetPP(
        tmp_path,
        manifest,
        split="train",
        context_views=2,
        target_views=1,
        image_resolution=8,
        target_pool_size=None,
        reserve_support_view=False,
        seed=9,
    )
    state = dataset.state_dict()
    first = dataset.sample_episode()
    dataset.load_state_dict(state)
    repeated = dataset.sample_episode()
    assert [view.name for view in first.context + first.target] == [
        view.name for view in repeated.context + repeated.target
    ]

    validation = MultiSceneScanNetPP(
        tmp_path,
        manifest,
        split="val",
        context_views=2,
        target_views=1,
        image_resolution=8,
        target_pool_size=None,
        reserve_support_view=False,
        seed=10,
    )
    records = validation.fixed_episode_records(4, seed=11)
    assert [label for label, _ in records] == [
        "low_angle",
        "mid_angle",
        "low_angle",
        "mid_angle",
    ]
    assert all(len(episode.context) == 2 for _, episode in records)
    exhaustive = validation.all_episode_records()
    assert [label for label, _ in exhaustive] == [
        "low_angle",
        "mid_angle",
        "low_angle",
    ]
    assert exhaustive[0][1].context[0].name.endswith("frame_000.JPG")


def test_native_scannetpp_loads_explicit_named_triplet(tmp_path):
    scene = _write_native_scene(tmp_path, "named", views=12)
    dataset = ScanNetPPDataset(
        scene,
        split="train",
        context_views=2,
        target_views=1,
        image_resolution=8,
    )
    prefix = "dslr/resized_undistorted_images"
    episode = dataset.episode_from_names(
        [f"{prefix}/frame_002.JPG", f"{prefix}/frame_008.JPG"],
        [f"{prefix}/frame_005.JPG"],
    )
    assert [view.name for view in episode.context + episode.target] == [
        f"{prefix}/frame_002.JPG",
        f"{prefix}/frame_008.JPG",
        f"{prefix}/frame_005.JPG",
    ]


def test_native_scannetpp_samples_two_contexts_and_separate_target(tmp_path):
    scene = _write_native_scene(tmp_path, "two-context", views=12)
    dataset = ScanNetPPDataset(
        scene,
        split="train",
        context_views=2,
        target_views=1,
        image_resolution=8,
        target_pool_size=6,
        seed=9,
    )
    episode = dataset.sample_episode()
    assert len(episode.context) == 2
    assert len(episode.target) == 1
    assert len({view.name for view in episode.context + episode.target}) == 3


def test_one_context_control_can_reserve_the_two_context_support_view(tmp_path):
    scene = _write_native_scene(tmp_path, "reserved", views=12)
    seed = 11
    two_context = ScanNetPPDataset(
        scene,
        split="train",
        context_views=2,
        target_views=1,
        target_pool_size=6,
        seed=seed,
    ).sample_episode()
    control = ScanNetPPDataset(
        scene,
        split="train",
        context_views=1,
        target_views=1,
        target_pool_size=6,
        reserve_support_view=True,
        seed=seed,
    ).sample_episode()
    assert control.context[0].name == two_context.context[0].name
    assert control.target[0].name == two_context.target[0].name


def test_native_scannetpp_rejects_off_center_intrinsics(tmp_path):
    scene = _write_native_scene(tmp_path, "bad")
    path = scene / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    metadata = json.loads(path.read_text())
    metadata["cx"] = 4.0
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="principal point"):
        ScanNetPPDataset(
            scene,
            split="train",
            context_views=1,
            target_views=1,
            image_resolution=8,
        )


def test_manifest_builder_rejects_scenes_the_loader_cannot_load(tmp_path):
    scene = _write_native_scene(tmp_path, "anisotropic", views=12)
    path = scene / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    metadata = json.loads(path.read_text())
    metadata["fl_y"] = float(metadata["fl_x"]) * 1.5
    path.write_text(json.dumps(metadata))

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from build_scannetpp_scaleup_manifest import _select_scene_episodes
    finally:
        sys.path.pop(0)

    with pytest.raises(ValueError, match="square pixels"):
        _select_scene_episodes(scene, neighbors=4, episodes_per_scene=1)


def test_splatt3r_manifest_builder_preserves_every_fixed_bin_tuple(tmp_path):
    _write_native_scene(tmp_path, "train-a")
    _write_native_scene(tmp_path, "val-a")
    split_root = tmp_path / "splits"
    split_root.mkdir()
    (split_root / "nvs_sem_train.txt").write_text("train-a\n")
    (split_root / "nvs_sem_val.txt").write_text("val-a\n")
    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir()
    (coverage_root / "train-a.json").write_text("{}")
    assets = tmp_path / "assets"
    assets.mkdir()
    for thresholds in ("0.9_0.9", "0.7_0.7", "0.5_0.5", "0.3_0.3"):
        (assets / f"splatt3r_scannetpp_test_{thresholds}.json").write_text(
            json.dumps([["val-a", 0, 2, 1]])
        )

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from build_splatt3r_scannetpp_manifest import build_manifest
    finally:
        sys.path.pop(0)
    manifest = build_manifest(
        scene_root=tmp_path,
        split_root=split_root,
        coverage_root=coverage_root,
        test_assets=assets,
        image_directory="resized_undistorted_images",
        evaluation_stride=1,
    )
    assert manifest["train"] == ["train-a"]
    assert [entry["bin"] for entry in manifest["val"]] == [
        "close",
        "medium",
        "wide",
        "very_wide",
    ]
    assert manifest["stats"]["evaluation_episodes"] == 4

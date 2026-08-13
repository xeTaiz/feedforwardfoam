import json
import math

import numpy as np
import pytest
from PIL import Image

from feedforwardfoam.data.multiscene import MultiSceneScanNetPP
from feedforwardfoam.data.scannetpp import ScanNetPPDataset


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
        seed=5,
    )
    episodes = validation.fixed_episodes(3, seed=6)
    assert all(episode.scene_id == "val-a" for episode in episodes)
    assert len(episodes[0].target) == 2


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

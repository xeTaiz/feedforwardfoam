import json

import numpy as np
from PIL import Image

from feedforwardfoam.data import BlenderNvsDataset


def test_blender_loader_uses_disjoint_context_and_targets(tmp_path):
    frames = []
    for index in range(3):
        filename = f"r_{index}.png"
        Image.fromarray(np.full((8, 10, 3), index * 50, dtype=np.uint8)).save(tmp_path / filename)
        frames.append(
            {
                "file_path": f"./r_{index}",
                "transform_matrix": np.eye(4).tolist(),
            }
        )
    metadata = {"camera_angle_x": 0.7, "frames": frames}
    (tmp_path / "transforms_train.json").write_text(json.dumps(metadata))

    dataset = BlenderNvsDataset(tmp_path, context_views=1, target_views=1, seed=3)
    episode = dataset[0]

    assert episode.context[0].name != episode.target[0].name
    assert episode.context[0].image.shape == (8, 10, 3)
    assert episode.context[0].image.max() <= 1.0
    assert episode.context[0].alpha is not None
    assert episode.context[0].alpha.shape == (8, 10)


def test_blender_loader_preserves_rgba_foreground_mask(tmp_path):
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[:2, :, 3] = 255
    frames = []
    for index in range(2):
        Image.fromarray(rgba).save(tmp_path / f"r_{index}.png")
        frames.append({"file_path": f"./r_{index}", "transform_matrix": np.eye(4).tolist()})
    (tmp_path / "transforms_train.json").write_text(
        json.dumps({"camera_angle_x": 0.7, "frames": frames})
    )

    view = BlenderNvsDataset(tmp_path, context_views=1, target_views=1)[0].context[0]
    assert view.alpha is not None
    assert view.alpha[:2].mean() == 1
    assert view.alpha[2:].mean() == 0
    assert view.image[:2].mean() == 1
    assert view.image[2:].mean() == 0

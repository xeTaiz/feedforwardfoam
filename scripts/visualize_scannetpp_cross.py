#!/usr/bin/env python3
"""Visualize cross-view Power Foam renders on a native ScanNet++ validation scene.

Loads the same backbone/head that ``train.py`` instantiates, predicts a separate
Power Foam from each view, and renders the cross/swapped views through the
existing ``PowerFoamRendererBridge``. Output per pair: eight PNG components and
one row of a labeled contact sheet. Aggregate metrics are written to
``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

from feedforwardfoam.backbone import FrozenVGGTOmega
from feedforwardfoam.data.scannetpp import ScanNetPPDataset
from feedforwardfoam.data.types import View
from feedforwardfoam.head import CanonicalPowerFoamHead
from feedforwardfoam.renderer import (
    PowerFoamRendererBridge,
    camera_from_view,
    pinhole_ray_map_from_view,
    powerfoam_args,
)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _load_font(size: int) -> ImageFont.ImageFont:
    """Return a widely-available TrueType font, falling back to PIL default."""
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _colorize_depth(depth: torch.Tensor) -> Image.Image:
    """Robust per-image normalization. Magma if matplotlib is available, else grayscale."""
    flat = depth.detach().to(torch.float32).cpu()
    finite = torch.isfinite(flat)
    if finite.any():
        lo = torch.quantile(flat[finite], 0.02).item()
        hi = torch.quantile(flat[finite], 0.98).item()
    else:
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    normalized = ((flat - lo) / (hi - lo)).clamp(0, 1)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import numpy as np
        from matplotlib import cm

        array = (cm.magma(normalized.numpy())[..., :3] * 255).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")
    except ImportError:
        array = (normalized * 255).round().byte().numpy()
        return Image.fromarray(array, mode="L").convert("RGB")


def _to_pil_rgb(image: torch.Tensor) -> Image.Image:
    array = image.detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _labeled_tile(pil: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    """Draw a label band above the image tile. Label height is 22 pixels."""
    band_height = 22
    canvas = Image.new("RGB", (pil.width, pil.height + band_height), (0, 0, 0))
    canvas.paste(pil, (0, band_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill=(255, 255, 255), font=font)
    return canvas


def _per_render_metrics(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(rendered.clamp(0, 1), target).item()
    return {"mse": mse, "psnr": -10.0 * torch.log10(torch.tensor(mse + 1e-10)).item()}


def _build_head(
    config: dict[str, Any], register_dim: int, device: torch.device
) -> CanonicalPowerFoamHead:
    """Instantiate the head with exactly the kwargs used by ``train.py``."""
    head_cfg = config["head"]
    return CanonicalPowerFoamHead(
        register_dim=register_dim,
        hidden_dim=int(head_cfg["hidden_dim"]),
        max_cells=int(head_cfg["max_cells"]),
        num_texel_sites=int(head_cfg["num_texel_sites"]),
        spherical_voronoi_dof=int(head_cfg["spherical_voronoi_dof"]),
        radius_mode=str(head_cfg.get("radius_mode", "learned_absolute")),
        radius_scale_init=float(head_cfg.get("radius_scale_init", 1.5)),
        radius_residual_log_scale=float(head_cfg.get("radius_residual_log_scale", 0.25)),
        density_mode=str(head_cfg.get("density_mode", "learned")),
        fixed_density=float(head_cfg.get("fixed_density", 100.0)),
        initialize_rgb_from_image=bool(head_cfg.get("initialize_rgb_from_image", False)),
        initialize_normals_from_depth=bool(head_cfg.get("initialize_normals_from_depth", True)),
        base_depth_mode=str(head_cfg.get("base_depth_mode", "predicted")),
        constant_base_depth=float(head_cfg.get("constant_base_depth", 2.0)),
        point_residual_scale=float(head_cfg.get("point_residual_scale", 0.05)),
        normal_residual_radians=float(head_cfg.get("normal_residual_radians", 0.25)),
        rgb_residual_scale=float(head_cfg.get("rgb_residual_scale", 0.5)),
    ).to(device)


def _build_bridge(config: dict[str, Any], reference_camera) -> PowerFoamRendererBridge:
    head_cfg = config["head"]
    renderer_cfg = config["renderer"]
    return PowerFoamRendererBridge(
        powerfoam_args(
            num_texel_sites=int(head_cfg["num_texel_sites"]),
            sv_dof=int(head_cfg["spherical_voronoi_dof"]),
            bkgd_color=tuple(renderer_cfg["bkgd_color"]),
            is_pinhole=bool(renderer_cfg["is_pinhole"]),
        ),
        reference_camera,
    )


def _stack_image(image: torch.Tensor) -> torch.Tensor:
    """Inject a single-view image into the [1, 1, 3, H, W] backbone contract."""
    return image.permute(2, 0, 1)[None, None].to(torch.float32)


def _predict_foam(
    head: CanonicalPowerFoamHead,
    backbone: FrozenVGGTOmega,
    view: View,
    device: torch.device,
) -> tuple[Any, torch.Tensor]:
    """Run VGGT-Ω and decode one foam from one view."""
    inputs = _stack_image(view.image).to(device)
    features = backbone(inputs)
    ray_map = pinhole_ray_map_from_view(view, device)
    foam = head(inputs, features, ray_map, view.alpha)
    depth = features["depth"][0, 0, 0]
    height, width = view.image.shape[:2]
    if depth.shape[-2:] != (height, width):
        depth = F.interpolate(
            depth[None, None], size=(height, width), mode="bilinear", align_corners=False
        )[0, 0]
    return foam, depth.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="VGGT-Ω checkpoint path")
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-id", type=str, default="ff17657f71")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Visualization requires CUDA because Power Foam uses Warp kernels")
    device = torch.device("cuda")

    config = _load_config(args.config)
    resolution = int(config["backbone"]["image_resolution"])
    if resolution <= 0 or resolution % 16 != 0:
        raise ValueError(
            f"backbone.image_resolution must be positive and divisible by 16, got {resolution}"
        )

    scene_root = args.data_root / args.scene_id
    if not scene_root.exists():
        raise FileNotFoundError(f"ScanNet++ scene directory not found: {scene_root}")
    dataset = ScanNetPPDataset(
        scene_root,
        split="test",
        context_views=1,
        target_views=1,
        image_resolution=resolution,
        target_pool_size=32,
        seed=args.seed,
    )
    if len(dataset) < 2:
        raise ValueError(
            f"Scene {args.scene_id} has only {len(dataset)} val views, need at least 2"
        )

    backbone = FrozenVGGTOmega(args.checkpoint).to(device)
    backbone.eval()
    register_dim = backbone.register_dim

    head = _build_head(config, register_dim, device)
    head_state = torch.load(args.head_checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(head_state["head"])
    head.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(14)

    pair_indices = []
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(int(args.pairs)):
        indices = dataset._sample_indices(generator)
        if len(indices) < 2:
            raise ValueError("dataset._sample_indices must return at least two indices")
        pair_indices.append(tuple(indices[:2]))

    records: list[dict[str, Any]] = []
    self_squared_errors: list[float] = []
    self_pixel_counts: list[int] = []
    cross_squared_errors: list[float] = []
    cross_pixel_counts: list[int] = []
    per_render_psnrs: list[dict[str, float]] = []
    contact_rows: list[Image.Image] = []

    for pair_index, (source_index, target_index) in enumerate(pair_indices):
        view1 = dataset._load_view(dataset.frames[source_index])
        view2 = dataset._load_view(dataset.frames[target_index])
        target_image_1 = view1.image.to(device)
        target_image_2 = view2.image.to(device)
        camera_1 = camera_from_view(view1, device)
        camera_2 = camera_from_view(view2, device)

        with torch.no_grad():
            foam1, depth1 = _predict_foam(head, backbone, view1, device)
            foam2, depth2 = _predict_foam(head, backbone, view2, device)
            bridge = _build_bridge(config, camera_1)
            render_f1_c1 = bridge.render(foam1, camera_1)
            render_f1_c2 = bridge.render(foam1, camera_2)
            render_f2_c2 = bridge.render(foam2, camera_2)
            render_f2_c1 = bridge.render(foam2, camera_1)

        component_dir = args.output_dir / f"pair_{pair_index:02d}"
        component_dir.mkdir(parents=True, exist_ok=True)
        file_paths: dict[str, str] = {}

        gt1 = _to_pil_rgb(view1.image)
        gt2 = _to_pil_rgb(view2.image)
        depth1_pil = _colorize_depth(depth1)
        depth2_pil = _colorize_depth(depth2)
        f1c1 = _to_pil_rgb(render_f1_c1.rgb)
        f1c2 = _to_pil_rgb(render_f1_c2.rgb)
        f2c2 = _to_pil_rgb(render_f2_c2.rgb)
        f2c1 = _to_pil_rgb(render_f2_c1.rgb)

        for name, pil in (
            ("gt_view1", gt1),
            ("vggt_depth1", depth1_pil),
            ("foam1_to_cam1", f1c1),
            ("foam2_to_cam1", f2c1),
            ("gt_view2", gt2),
            ("vggt_depth2", depth2_pil),
            ("foam2_to_cam2", f2c2),
            ("foam1_to_cam2", f1c2),
        ):
            path = component_dir / f"{name}.png"
            pil.save(path)
            file_paths[name] = str(path.relative_to(args.output_dir))

        metrics_f1_c1 = _per_render_metrics(render_f1_c1.rgb, target_image_1)
        metrics_f2_c2 = _per_render_metrics(render_f2_c2.rgb, target_image_2)
        metrics_f1_c2 = _per_render_metrics(render_f1_c2.rgb, target_image_2)
        metrics_f2_c1 = _per_render_metrics(render_f2_c1.rgb, target_image_1)

        def _aggregate(errors: list[float], counts: list[int]) -> tuple[float, float]:
            total_se = sum(errors)
            total_pixels = sum(counts)
            aggregate_mse = total_se / max(total_pixels, 1)
            psnr = -10.0 * float(torch.log10(torch.tensor(aggregate_mse + 1e-10)).item())
            return float(aggregate_mse), psnr

        pixels_per_render = int(target_image_1.shape[0] * target_image_1.shape[1] * 3)

        self_pair_se = [
            metrics_f1_c1["mse"] * pixels_per_render,
            metrics_f2_c2["mse"] * pixels_per_render,
        ]
        self_pair_counts = [pixels_per_render, pixels_per_render]
        self_aggregate_mse, self_psnr_aggregate = _aggregate(self_pair_se, self_pair_counts)
        self_squared_errors.extend(self_pair_se)
        self_pixel_counts.extend(self_pair_counts)

        cross_pair_se = [
            metrics_f1_c2["mse"] * pixels_per_render,
            metrics_f2_c1["mse"] * pixels_per_render,
        ]
        cross_pair_counts = [pixels_per_render, pixels_per_render]
        cross_aggregate_mse, cross_psnr_aggregate = _aggregate(cross_pair_se, cross_pair_counts)
        cross_squared_errors.extend(cross_pair_se)
        cross_pixel_counts.extend(cross_pair_counts)

        per_render = {
            "f1_to_c1": metrics_f1_c1,
            "f2_to_c2": metrics_f2_c2,
            "f1_to_c2": metrics_f1_c2,
            "f2_to_c1": metrics_f2_c1,
        }
        per_render_psnrs.append(per_render)
        per_render_psnr_mean = sum(metric["psnr"] for metric in per_render.values()) / len(
            per_render
        )

        record = {
            "pair_index": pair_index,
            "source_index": source_index,
            "target_index": target_index,
            "view1_name": view1.name,
            "view2_name": view2.name,
            "files": file_paths,
            "per_render": per_render,
            "self_aggregate_mse": self_aggregate_mse,
            "self_psnr_aggregate": self_psnr_aggregate,
            "cross_aggregate_mse": cross_aggregate_mse,
            "cross_psnr_aggregate": cross_psnr_aggregate,
            "per_render_psnr_mean": per_render_psnr_mean,
        }
        records.append(record)

        def _label(stem: str, metric: dict[str, float]) -> str:
            return f"{stem} | PSNR {metric['psnr']:.2f}"

        display_size = (240, 240)
        shown = [
            gt1,
            depth1_pil,
            f1c1,
            f2c1,
            gt2,
            depth2_pil,
            f2c2,
            f1c2,
        ]
        labels = [
            f"GT view1 | {Path(view1.name).name}",
            "VGGT depth1",
            _label("foam1->cam1", metrics_f1_c1),
            _label("foam2->cam1 cross", metrics_f2_c1),
            f"GT view2 | {Path(view2.name).name}",
            "VGGT depth2",
            _label("foam2->cam2", metrics_f2_c2),
            _label("foam1->cam2 cross", metrics_f1_c2),
        ]
        row_tiles = [
            _labeled_tile(tile.resize(display_size, Image.Resampling.NEAREST), label, font)
            for tile, label in zip(shown, labels, strict=True)
        ]
        total_width = sum(tile.width for tile in row_tiles)
        row_height = row_tiles[0].height
        row = Image.new("RGB", (total_width, row_height), (0, 0, 0))
        offset = 0
        for tile in row_tiles:
            row.paste(tile, (offset, 0))
            offset += tile.width
        contact_rows.append(row)

    if contact_rows:
        sheet_height = sum(row.height for row in contact_rows)
        sheet_width = contact_rows[0].width
        sheet = Image.new("RGB", (sheet_width, sheet_height), (0, 0, 0))
        offset = 0
        for row in contact_rows:
            sheet.paste(row, (0, offset))
            offset += row.height
        sheet_path = args.output_dir / "contact_sheet.png"
        sheet.save(sheet_path)

    self_aggregate_mse, self_psnr_aggregate = _aggregate(self_squared_errors, self_pixel_counts)
    cross_aggregate_mse, cross_psnr_aggregate = _aggregate(cross_squared_errors, cross_pixel_counts)
    all_psnrs = [
        metric["psnr"] for per_render in per_render_psnrs for metric in per_render.values()
    ]
    mean_per_render_psnr = sum(all_psnrs) / max(len(all_psnrs), 1)

    summary = {
        "scene_id": args.scene_id,
        "config": str(args.config),
        "vggt_checkpoint": str(args.checkpoint),
        "head_checkpoint": str(args.head_checkpoint),
        "pairs": int(args.pairs),
        "seed": int(args.seed),
        "image_resolution": resolution,
        "per_pair": records,
        "aggregate": {
            "self_psnr": self_psnr_aggregate,
            "self_mse": self_aggregate_mse,
            "cross_psnr": cross_psnr_aggregate,
            "cross_mse": cross_aggregate_mse,
            "mean_per_render_psnr": mean_per_render_psnr,
            "render_count": len(all_psnrs),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()

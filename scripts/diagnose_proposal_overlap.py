#!/usr/bin/env python3
"""Report how far apart two context views' Foam proposals actually land.

Incremental containment merging can only remove a duplicate when the second
view's site falls inside a site already emitted by the first view. If the two
unprojected clouds are offset by more than a cell radius the merge is a no-op
no matter which criterion is used, and the real defect is cross-view depth
consistency rather than the merge rule. This script separates those cases by
reporting nearest-neighbour distances between the two proposal groups in units
of the local cell radius.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from feedforwardfoam.backbone import FrozenVGGTOmega
from feedforwardfoam.fusion import (
    align_depths_to_calibrated_cameras,
    build_canonical_support,
)
from feedforwardfoam.train import (
    _build_datasets,
    _configured_fixed_episode,
    _context_tensor,
    _predict,
    build_head,
)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=values.device)
    quantiles = torch.quantile(values, probabilities)
    return {
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }


def diagnose(config_path: Path, data_root: Path, checkpoint: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text())
    # Unreduced concatenation keeps both groups intact and equally sized.
    config["head"]["proposal_views"] = "all"
    config["head"]["proposal_reduction"] = "all"
    device = torch.device("cuda")
    torch.manual_seed(int(config["seed"]))

    train_dataset, _ = _build_datasets(config, data_root)
    episode = _configured_fixed_episode(train_dataset, config)
    if episode is None:
        raise ValueError("This diagnostic needs a fixed context/target episode")

    backbone = FrozenVGGTOmega(checkpoint).to(device)
    backbone.eval()
    head = build_head(config, backbone.register_dim, "foam", device)
    head.eval()

    with torch.no_grad():
        params, features = _predict(head, backbone, episode, "foam", device)

    total = params.points.shape[0]
    if total % 2 != 0:
        raise ValueError("Expected two equally sized proposal groups")
    half = total // 2
    first = params.points[:half]
    second = params.points[half:]
    # Upstream applies this softplus to raw radii; inverse_softplus inverts it.
    physical_radii = F.softplus(params.radii, beta=100.0).reshape(-1)

    distances = torch.cdist(second, first)
    nearest, nearest_index = distances.min(dim=1)
    incumbent_radius = physical_radii[:half][nearest_index]
    newcomer_radius = physical_radii[half:]

    ratio = nearest / incumbent_radius.clamp_min(1e-9)
    # Fraction of view-1 sites the exact power test would swallow.
    power_hits = (nearest.square() + newcomer_radius.square()) <= incumbent_radius.square()
    ball_hits = nearest <= incumbent_radius

    # Spacing inside one view sets the scale a duplicate must beat.
    own = torch.cdist(first, first)
    own.fill_diagonal_(float("inf"))
    own_nearest = own.min(dim=1).values

    # Restrict to canonical pixels the fusion stage marks as co-visible. If the
    # two depth clouds are registered, a co-visible view-0 site must have a
    # view-1 site within roughly the within-view spacing. If it does not, the
    # clouds are offset rather than describing disjoint surface.
    images = _context_tensor(episode, device)
    with torch.no_grad():
        raw_features = {name: value.clone() for name, value in backbone(images).items()}
        aligned_depths, _ = align_depths_to_calibrated_cameras(raw_features, episode.context)
        raw_features["depth"] = aligned_depths
        support = build_canonical_support(images, raw_features, episode.context, device)
    co_visible_ratio = None
    co_visible_fraction = 0.0
    if support is not None:
        support_map = support.maps[:, -1:]
        side = int(round(half**0.5))
        if support_map.shape[-2:] != (side, side):
            support_map = F.interpolate(
                support_map, size=(side, side), mode="bilinear", align_corners=True
            )
        co_visible = support_map.reshape(-1) > 0.5
        co_visible_fraction = float(co_visible.float().mean())
        forward = torch.cdist(first, second).min(dim=1).values
        forward_ratio = forward / physical_radii[:half].clamp_min(1e-9)
        if int(co_visible.sum()) > 0:
            co_visible_ratio = _quantiles(forward_ratio[co_visible])

    return {
        "config": str(config_path),
        "scene_id": episode.scene_id,
        "proposals_per_view": half,
        "depth_alignment_scale": float(features["depth_alignment_scale"].item()),
        "mean_physical_radius": float(physical_radii.mean()),
        "cross_view_nearest_distance": _quantiles(nearest),
        "cross_view_nearest_over_radius": _quantiles(ratio),
        "within_view_nearest_distance": _quantiles(own_nearest),
        "within_view_nearest_over_radius": _quantiles(
            own_nearest / physical_radii[:half].clamp_min(1e-9)
        ),
        "co_visible_fraction": co_visible_fraction,
        "co_visible_nearest_over_radius": co_visible_ratio,
        "power_containment_fraction": float(power_hits.float().mean()),
        "ball_containment_fraction": float(ball_hits.float().mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    report = diagnose(args.config, args.data_root, args.checkpoint)
    text = json.dumps(report, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

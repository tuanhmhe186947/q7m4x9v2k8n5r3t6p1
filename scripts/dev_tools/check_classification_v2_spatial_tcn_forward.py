from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.models.spatial_tcn import SpatialTCNClassifier, SpatialTCNConfig

DEFAULT_NPZ = Path("outputs/classification_v2/train_ready_windows/X_spatial_sequences.npz")
DEFAULT_Y = Path("outputs/classification_v2/train_ready_windows/y_behavior.csv")
MODEL_GROUPS = (
    "bbox_xywh_n",
    "bbox_shape_n",
    "motion_delta",
    "roi_class_relation",
    "social_relation",
    "quality_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check classification_v2 SpatialTCN forward pass.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--y-csv", type=Path, default=DEFAULT_Y)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.npz)
    labels = pd.read_csv(args.y_csv).iloc[:, 0].fillna("").astype(str)
    label_order = sorted(labels.unique().tolist())

    missing = [name for name in [*MODEL_GROUPS, "length_mask", "observed_mask"] if name not in data.files]
    if missing:
        raise SystemExit(f"missing arrays: {missing}")
    batch_size = min(args.batch_size, data["length_mask"].shape[0])
    idx = np.arange(batch_size)
    features = {name: torch.from_numpy(data[name][idx]).float() for name in MODEL_GROUPS}
    length_mask = torch.from_numpy(data["length_mask"][idx]).float()
    observed_mask = torch.from_numpy(data["observed_mask"][idx]).float()
    config = SpatialTCNConfig(
        input_dims={name: int(features[name].shape[-1]) for name in MODEL_GROUPS},
        num_classes=len(label_order),
        hidden_dim=args.hidden_dim,
        dropout=0.0,
    )
    model = SpatialTCNClassifier(config)
    model.eval()
    with torch.no_grad():
        logits = model(features, length_mask=length_mask, observed_mask=observed_mask)
        perturbed = {name: value.clone() for name, value in features.items()}
        padding = length_mask.eq(0).unsqueeze(-1)
        for name in MODEL_GROUPS:
            perturbed[name] = torch.where(padding, torch.full_like(perturbed[name], 9999.0), perturbed[name])
        perturbed_logits = model(perturbed, length_mask=length_mask, observed_mask=observed_mask)
    max_padding_delta = float((logits - perturbed_logits).abs().max().item())
    errors = []
    if tuple(logits.shape) != (batch_size, len(label_order)):
        errors.append(f"logit_shape={tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        errors.append("logits_nonfinite")
    if max_padding_delta > 1e-5:
        errors.append(f"padding_invariance_failed_delta={max_padding_delta}")

    result = {
        "npz": str(args.npz),
        "y_csv": str(args.y_csv),
        "batch_size": int(batch_size),
        "label_count": int(len(label_order)),
        "feature_groups": {name: list(features[name].shape) for name in MODEL_GROUPS},
        "logit_shape": list(logits.shape),
        "max_padding_delta": max_padding_delta,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pig_behavior.classification_v2.datasets.visual_interaction_loader import (
    VisualInteractionDatasetConfig,
    VisualInteractionWindowDataset,
    visual_interaction_collate,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check visual interaction loader and model branch.")
    parser.add_argument("--cache-manifest-csv", type=Path, required=True)
    parser.add_argument("--window-context-csv", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=16)
    parser.add_argument("--packed-cache", type=Path, default=None)
    parser.add_argument("--packed-cache-index", type=Path, default=None)
    parser.add_argument("--require-packed-cache", action="store_true")
    args = parser.parse_args()
    dataset = VisualInteractionWindowDataset(
        VisualInteractionDatasetConfig(
            cache_manifest_csv=args.cache_manifest_csv,
            window_context_csv=args.window_context_csv,
            max_windows=args.max_windows,
            packed_cache_npy=args.packed_cache,
            packed_cache_index_csv=args.packed_cache_index,
            require_packed_cache=args.require_packed_cache,
        )
    )
    batch = next(iter(DataLoader(dataset, batch_size=min(8, len(dataset)), collate_fn=visual_interaction_collate)))
    model = MultimodalFusionClassifier(
        MultimodalFusionConfig(
            spatial_input_dims={},
            num_classes=10,
            enable_image=False,
            enable_spatial=False,
            enable_interaction_context=False,
            enable_visual_context=True,
        )
    )
    logits = model(
        image=torch.empty(0),
        spatial_features={},
        length_mask=batch["visual_context_length_mask"],
        visual_context_image=batch["visual_context_image"],
        visual_context_length_mask=batch["visual_context_length_mask"],
        visual_context_observed_mask=batch["visual_context_observed_mask"],
    )
    observed_rows = int(batch["visual_context_observed_mask"].sum().item())
    errors = [error for values in batch["errors"] for error in values]
    if logits.shape != (len(batch["window_id"]), 10):
        errors.append(f"unexpected_logits_shape={tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        errors.append("nonfinite_logits")
    result = {
        "batch_rows": len(batch["window_id"]),
        "visual_context_shape": list(batch["visual_context_image"].shape),
        "observed_frame_slots": observed_rows,
        "logits_shape": list(logits.shape),
        "cache_load_audit": dataset.load_audit(),
        "errors": errors,
        "valid": not errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

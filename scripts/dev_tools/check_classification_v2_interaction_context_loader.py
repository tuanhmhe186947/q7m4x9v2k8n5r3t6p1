from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from pig_behavior.classification_v2.datasets.interaction_context_loader import (
    INTERACTION_CONTEXT_FEATURE_COLUMNS,
    InteractionContextDatasetConfig,
    InteractionContextWindowDataset,
    interaction_context_collate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check classification_v2 interaction context tensor loader.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/interaction_window_context_manifest.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/interaction_context_loader_audit.json"),
    )
    parser.add_argument("--sample-rows", type=int, default=32)
    args = parser.parse_args()

    dataset = InteractionContextWindowDataset(InteractionContextDatasetConfig(manifest_csv=args.manifest))
    indices = _sample_indices(dataset, args.sample_rows)
    loader = DataLoader(Subset(dataset, indices), batch_size=max(1, len(indices)), collate_fn=interaction_context_collate)
    batch = next(iter(loader)) if indices else None
    audit = _audit(args, dataset, indices, batch)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def _sample_indices(dataset: InteractionContextWindowDataset, sample_rows: int) -> list[int]:
    """Sample ready and not-ready windows so both mask states are exercised."""

    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive")
    manifest = dataset.manifest
    ready = manifest.index[manifest["scene_partner_context_ready"].astype(str).str.lower().isin({"true", "1"})]
    not_ready = manifest.index.difference(ready)
    half = max(1, sample_rows // 2)
    selected = list(ready[:half]) + list(not_ready[: max(0, sample_rows - half)])
    if not selected:
        selected = list(manifest.index[:sample_rows])
    return [int(idx) for idx in selected[:sample_rows]]


def _audit(
    args: argparse.Namespace,
    dataset: InteractionContextWindowDataset,
    indices: list[int],
    batch: dict[str, object] | None,
) -> dict[str, object]:
    errors: list[str] = []
    if batch is None:
        errors.append("empty_interaction_context_sample")
        feature_shape = None
        finite = False
        ready_mask_sum = 0.0
    else:
        features = batch["interaction_context_features"]
        ready_mask = batch["interaction_context_available_mask"]
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            errors.append("interaction_context_features_not_2d_tensor")
            feature_shape = None
            finite = False
        else:
            feature_shape = list(features.shape)
            finite = bool(torch.isfinite(features).all().item())
            if feature_shape[1] != len(INTERACTION_CONTEXT_FEATURE_COLUMNS):
                errors.append(f"unexpected_feature_dim={feature_shape[1]}")
        if not isinstance(ready_mask, torch.Tensor) or ready_mask.ndim != 1:
            errors.append("interaction_context_available_mask_not_1d_tensor")
            ready_mask_sum = 0.0
        else:
            ready_mask_sum = float(ready_mask.sum().item())
        if not finite:
            errors.append("interaction_context_features_nonfinite")

    return {
        "schema_version": "classification_v2_interaction_context_loader_audit_v1",
        "manifest": str(args.manifest),
        "dataset_rows": int(len(dataset)),
        "sample_indices": indices,
        "sample_rows": int(len(indices)),
        "feature_columns": list(INTERACTION_CONTEXT_FEATURE_COLUMNS),
        "feature_shape": feature_shape,
        "ready_mask_sum": ready_mask_sum,
        "status_counts_sample": dict(Counter(batch["context_status"])) if batch is not None else {},
        "forbidden_inputs_passed_to_model": [],
        "metadata_keys_not_model_inputs": ["window_id", "context_status"],
        "errors": errors,
    }


if __name__ == "__main__":
    main()

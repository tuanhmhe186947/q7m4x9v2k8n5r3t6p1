from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    image_sequence_collate,
)
from pig_behavior.classification_v2.evaluation.metrics import DEFAULT_LABEL_ORDER
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.training.spatial_tcn_smoke import MODEL_GROUPS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-check classification_v2 multimodal fusion forward pass."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/multimodal_forward_smoke_audit.json"),
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--sample-per-source", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arrays = {name: value for name, value in np.load(args.root / "X_spatial_sequences.npz").items()}
    labels = pd.read_csv(args.root / "y_behavior.csv").iloc[:, 0].fillna("").astype(str)
    windows = pd.read_csv(args.root / "image_window_context_manifest.csv", low_memory=False)
    _validate_arrays(arrays, labels, windows)

    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=args.root / "image_frame_context_manifest.csv",
            window_context_csv=args.root / "image_window_context_manifest.csv",
            image_size=args.image_size,
            require_complete=False,
        )
    )
    try:
        indices = _sample_indices(windows, args.sample_per_source)
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=max(1, len(indices)),
            shuffle=False,
            collate_fn=image_sequence_collate,
        )
        batch = next(iter(loader)) if indices else None
        audit = _run_forward_smoke(args, arrays, labels, windows, indices, batch)
    finally:
        dataset.close()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(1)


def _run_forward_smoke(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    labels: pd.Series,
    windows: pd.DataFrame,
    indices: list[int],
    batch: dict[str, Any] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if batch is None:
        errors.append("empty_multimodal_sample")
        return _audit(args, labels, windows, indices, None, None, None, errors, warnings)

    sample_errors = [err for item_errors in batch["errors"] for err in item_errors]
    if sample_errors:
        errors.extend(sample_errors[:20])

    expected_window_ids = windows.iloc[indices]["window_id"].astype(str).tolist()
    if batch["window_id"] != expected_window_ids:
        errors.append("image_batch_window_id_order_mismatch")

    spatial_features = {
        name: torch.from_numpy(arrays[name][indices]).float()
        for name in MODEL_GROUPS
    }
    length_mask = torch.from_numpy(arrays["length_mask"][indices]).float()
    observed_mask = torch.from_numpy(arrays["observed_mask"][indices]).float()
    if length_mask.shape == batch["length_mask"].shape and not torch.equal(
        length_mask,
        batch["length_mask"],
    ):
        warnings.append("same_shape_branch_length_masks_differ")
    if observed_mask.shape == batch["observed_mask"].shape and not torch.equal(
        observed_mask,
        batch["observed_mask"],
    ):
        warnings.append("same_shape_branch_observed_masks_differ")

    label_order = _label_order(labels)
    model = MultimodalFusionClassifier(
        MultimodalFusionConfig(
            spatial_input_dims={name: int(arrays[name].shape[-1]) for name in MODEL_GROUPS},
            num_classes=len(label_order),
            image_embedding_dim=args.hidden_dim,
            spatial_embedding_dim=args.hidden_dim,
            fusion_hidden_dim=args.hidden_dim,
            dropout=0.0,
        )
    )
    model.eval()
    with torch.no_grad():
        logits = model(
            image=batch["image"],
            spatial_features=spatial_features,
            length_mask=batch["length_mask"],
            observed_mask=batch["observed_mask"],
            spatial_length_mask=length_mask,
            spatial_observed_mask=observed_mask,
        )
        max_padding_delta = _masked_padding_delta(
            model,
            batch["image"],
            spatial_features,
            batch["length_mask"],
            batch["observed_mask"],
            length_mask,
            observed_mask,
        )

    if list(logits.shape) != [len(indices), len(label_order)]:
        errors.append(f"unexpected_logit_shape={list(logits.shape)}")
    if not torch.isfinite(logits).all():
        errors.append("logits_nonfinite")
    if max_padding_delta > 1e-5:
        errors.append(f"mask_invariance_failed_delta={max_padding_delta:.8f}")

    return _audit(
        args,
        labels,
        windows,
        indices,
        batch,
        logits,
        max_padding_delta,
        errors,
        warnings,
        spatial_mask_shape=list(length_mask.shape),
    )


def _masked_padding_delta(
    model: MultimodalFusionClassifier,
    image: torch.Tensor,
    spatial_features: dict[str, torch.Tensor],
    image_length_mask: torch.Tensor,
    image_observed_mask: torch.Tensor,
    spatial_length_mask: torch.Tensor,
    spatial_observed_mask: torch.Tensor,
) -> float:
    deltas: list[float] = []
    if image.shape[1] >= 2:
        masked_image_observed = image_observed_mask.clone()
        masked_image_observed[:, -1] = 0.0
        perturbed_image = image.clone()
        perturbed_image[:, -1] = 7.0
        baseline = model(
            image=image,
            spatial_features=spatial_features,
            length_mask=image_length_mask,
            observed_mask=masked_image_observed,
            spatial_length_mask=spatial_length_mask,
            spatial_observed_mask=spatial_observed_mask,
        )
        perturbed = model(
            image=perturbed_image,
            spatial_features=spatial_features,
            length_mask=image_length_mask,
            observed_mask=masked_image_observed,
            spatial_length_mask=spatial_length_mask,
            spatial_observed_mask=spatial_observed_mask,
        )
        deltas.append(float((baseline - perturbed).abs().max().item()))

    if spatial_length_mask.shape[1] < 2:
        return max(deltas) if deltas else 0.0
    masked_spatial_observed = spatial_observed_mask.clone()
    masked_spatial_observed[:, -1] = 0.0
    perturbed_image = image.clone()
    perturbed_spatial = {name: value.clone() for name, value in spatial_features.items()}
    for value in perturbed_spatial.values():
        value[:, -1] = 1234.0
    baseline = model(
        image=image,
        spatial_features=spatial_features,
        length_mask=image_length_mask,
        observed_mask=image_observed_mask,
        spatial_length_mask=spatial_length_mask,
        spatial_observed_mask=masked_spatial_observed,
    )
    perturbed = model(
        image=perturbed_image,
        spatial_features=perturbed_spatial,
        length_mask=image_length_mask,
        observed_mask=image_observed_mask,
        spatial_length_mask=spatial_length_mask,
        spatial_observed_mask=masked_spatial_observed,
    )
    deltas.append(float((baseline - perturbed).abs().max().item()))
    return max(deltas) if deltas else 0.0


def _audit(
    args: argparse.Namespace,
    labels: pd.Series,
    windows: pd.DataFrame,
    indices: list[int],
    batch: dict[str, Any] | None,
    logits: torch.Tensor | None,
    max_padding_delta: float | None,
    errors: list[str],
    warnings: list[str],
    spatial_mask_shape: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "root": str(args.root),
        "image_size": int(args.image_size),
        "model_groups": list(MODEL_GROUPS),
        "label_order": _label_order(labels),
        "num_classes": int(len(_label_order(labels))),
        "complete_window_rows": int(_to_bool(windows["window_image_context_complete"]).sum()),
        "sample_indices": [int(i) for i in indices],
        "sample_source_counts": (
            windows.iloc[indices]["source_type"].value_counts(dropna=False).to_dict()
            if indices
            else {}
        ),
        "sample_label_counts": (
            labels.iloc[indices].value_counts(dropna=False).to_dict()
            if indices
            else {}
        ),
        "batch_shape": list(batch["image"].shape) if batch is not None else None,
        "image_mask_shape": list(batch["length_mask"].shape) if batch is not None else None,
        "spatial_mask_shape": spatial_mask_shape,
        "logit_shape": list(logits.shape) if logits is not None else None,
        "max_masked_padding_delta": max_padding_delta,
        "forbidden_inputs_passed_to_model": [],
        "metadata_keys_not_model_inputs": [
            "window_id",
            "source_type",
            "video_key",
            "image_context_ids",
            "expected_frame_indices",
        ],
        "errors": errors,
        "warnings": warnings,
    }


def _validate_arrays(
    arrays: dict[str, np.ndarray],
    labels: pd.Series,
    windows: pd.DataFrame,
) -> None:
    missing = [
        name
        for name in [*MODEL_GROUPS, "length_mask", "observed_mask"]
        if name not in arrays
    ]
    if missing:
        raise ValueError(f"missing spatial arrays: {missing}")
    expected = int(len(labels))
    counts = {"labels": len(labels), "image_windows": len(windows)}
    counts.update({name: int(value.shape[0]) for name, value in arrays.items()})
    mismatched = {name: count for name, count in counts.items() if count != expected}
    if mismatched:
        raise ValueError(f"row count mismatch against labels={expected}: {mismatched}")


def _sample_indices(windows: pd.DataFrame, sample_per_source: int) -> list[int]:
    complete = windows[_to_bool(windows["window_image_context_complete"])].copy()
    indices: list[int] = []
    for _, group in complete.groupby("source_type", sort=True):
        group = group.sort_values(["video_key", "object_track_key", "window_start_frame"])
        if len(group) <= sample_per_source:
            indices.extend(group.index.tolist())
            continue
        positions = sorted(
            {
                round(i * (len(group) - 1) / max(1, sample_per_source - 1))
                for i in range(sample_per_source)
            }
        )
        indices.extend(group.iloc[positions].index.tolist())
    return indices


def _label_order(labels: pd.Series) -> list[str]:
    observed = set(labels.dropna().astype(str).tolist())
    ordered = [label for label in DEFAULT_LABEL_ORDER if label in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()

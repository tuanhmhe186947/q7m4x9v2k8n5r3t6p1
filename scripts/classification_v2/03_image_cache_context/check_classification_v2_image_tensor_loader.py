from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Subset

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    image_sequence_collate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check classification_v2 image tensor loader.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/image_tensor_loader_smoke_audit.json"),
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--sample-per-source", type=int, default=4)
    args = parser.parse_args()

    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=args.root / "image_frame_context_manifest.csv",
            window_context_csv=args.root / "image_window_context_manifest.csv",
            image_size=args.image_size,
            require_complete=True,
        )
    )
    try:
        indices = _sample_indices(dataset.windows, args.sample_per_source)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=max(1, len(indices)), shuffle=False, collate_fn=image_sequence_collate)
        batch = next(iter(loader)) if indices else None
        errors: list[str] = []
        if batch is None:
            errors.append("empty_image_loader_sample")
        else:
            sample_errors = [err for item_errors in batch["errors"] for err in item_errors]
            if sample_errors:
                errors.extend(sample_errors[:20])
            image_shape = list(batch["image"].shape)
            if len(image_shape) != 5 or image_shape[2:] != [3, args.image_size, args.image_size]:
                errors.append(f"unexpected_image_shape={image_shape}")
            if float(batch["observed_mask"].sum().item()) != float(batch["length_mask"].sum().item()):
                errors.append("observed_mask_not_complete_for_sample")
        audit = {
            "root": str(args.root),
            "image_size": int(args.image_size),
            "dataset_rows_complete": int(len(dataset)),
            "sample_indices": [int(i) for i in indices],
            "sample_source_counts": dataset.windows.iloc[indices]["source_type"].value_counts(dropna=False).to_dict()
            if indices
            else {},
            "batch_shape": list(batch["image"].shape) if batch is not None else None,
            "batch_sources": batch["source_type"] if batch is not None else [],
            "observed_slots": float(batch["observed_mask"].sum().item()) if batch is not None else 0.0,
            "length_slots": float(batch["length_mask"].sum().item()) if batch is not None else 0.0,
            "errors": errors,
        }
    finally:
        dataset.close()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def _sample_indices(windows: pd.DataFrame, sample_per_source: int) -> list[int]:
    indices: list[int] = []
    for _, group in windows.groupby("source_type", sort=True):
        group = group.sort_values(["video_key", "object_track_key", "window_start_frame"])
        if len(group) <= sample_per_source:
            indices.extend(group.index.tolist())
            continue
        positions = sorted(
            {round(i * (len(group) - 1) / max(1, sample_per_source - 1)) for i in range(sample_per_source)}
        )
        indices.extend(group.iloc[positions].index.tolist())
    return indices


if __name__ == "__main__":
    main()

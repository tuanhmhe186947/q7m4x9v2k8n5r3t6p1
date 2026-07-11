from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check packed image-cache lineage and pixel equivalence.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox"),
    )
    parser.add_argument("--sample-size", type=int, default=64)
    args = parser.parse_args()
    audit = check_packed_cache(args.root, args.sample_size)
    print(json.dumps(audit, indent=2))
    if not audit["valid"]:
        raise SystemExit(2)


def check_packed_cache(root: Path, sample_size: int) -> dict[str, Any]:
    """Compare deterministic packed rows with individual cache bytes and loader output."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    manifest = pd.read_csv(root / "manifest.csv", low_memory=False).sort_values(
        "image_context_id", kind="mergesort"
    )
    index = pd.read_csv(root / "packed_image_cache_index.csv", low_memory=False).sort_values(
        "image_context_id", kind="mergesort"
    )
    tensor = np.load(root / "packed_rgb_64_letterbox.npy", mmap_mode="r")
    errors: list[str] = []
    if len(manifest) != len(index) or len(index) != tensor.shape[0]:
        errors.append(f"row_count_mismatch=manifest:{len(manifest)} index:{len(index)} tensor:{tensor.shape[0]}")
    if manifest["image_context_id"].astype(str).tolist() != index["image_context_id"].astype(str).tolist():
        errors.append("ordered_image_context_id_mismatch")
    expected_rows = np.arange(len(index), dtype=np.int64)
    if not np.array_equal(pd.to_numeric(index["packed_row"]).to_numpy(dtype=np.int64), expected_rows):
        errors.append("packed_row_not_contiguous")
    positions = np.linspace(0, len(index) - 1, min(sample_size, len(index)), dtype=int)
    pixel_mismatches = 0
    for position in positions:
        path = Path(str(manifest.iloc[int(position)]["cache_path"]))
        if not path.is_absolute():
            path = root / path
        if not np.array_equal(np.load(path), np.asarray(tensor[int(position)])):
            pixel_mismatches += 1
    if pixel_mismatches:
        errors.append(f"pixel_mismatches={pixel_mismatches}")

    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            packed_image_cache_npy=root / "packed_rgb_64_letterbox.npy",
            packed_image_cache_index_csv=root / "packed_image_cache_index.csv",
            image_size=64,
            require_complete=False,
            require_cached_images=True,
            image_cache_size=0,
        )
    )
    loader_failures = 0
    try:
        for position in positions[: min(16, len(positions))]:
            context_id = str(index.iloc[int(position)]["image_context_id"])
            frame = dataset.frame_by_context_id.get(context_id)
            if frame is None or dataset._load_context_image(context_id, frame) is None:
                loader_failures += 1
        load_audit = dataset.image_load_audit()
    finally:
        dataset.close()
    if loader_failures:
        errors.append(f"packed_loader_failures={loader_failures}")
    return {
        "schema_version": "classification_v2_packed_image_cache_check_v1",
        "root": str(root),
        "manifest_rows": int(len(manifest)),
        "index_rows": int(len(index)),
        "tensor_shape": [int(value) for value in tensor.shape],
        "tensor_dtype": str(tensor.dtype),
        "sample_rows": int(len(positions)),
        "pixel_mismatches": int(pixel_mismatches),
        "packed_loader_failures": int(loader_failures),
        "image_load_audit": load_audit,
        "errors": errors,
        "valid": not errors,
    }


if __name__ == "__main__":
    main()

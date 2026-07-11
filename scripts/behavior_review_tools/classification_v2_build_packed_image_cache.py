from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack classification_v2 image cache files into one mmap tensor.")
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox/manifest.csv"),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument(
        "--available-column",
        default=None,
        help="Optional boolean manifest column; only available tensors are packed and masked rows stay in source manifest.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = build_packed_cache(
        cache_manifest=args.cache_manifest,
        image_size=args.image_size,
        output_dir=args.output_dir,
        max_contexts=args.max_contexts,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        available_column=args.available_column,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2))
    if not audit["valid"]:
        raise SystemExit(2)


def build_packed_cache(
    *,
    cache_manifest: Path,
    image_size: int,
    output_dir: Path | None,
    max_contexts: int | None,
    workers: int,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
    available_column: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic row-addressable tensor without changing source cache files."""

    if image_size <= 0 or workers <= 0 or checkpoint_every <= 0:
        raise ValueError("image_size, workers, and checkpoint_every must be positive")
    if max_contexts is not None and max_contexts <= 0:
        raise ValueError("max_contexts must be positive when provided")
    manifest = pd.read_csv(cache_manifest, low_memory=False)
    source_rows = len(manifest)
    required = {"image_context_id", "cache_path", "image_size", "resize_policy"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"cache manifest missing columns: {missing}")
    if manifest["image_context_id"].duplicated().any():
        raise ValueError("cache manifest contains duplicate image_context_id rows")
    if pd.to_numeric(manifest["image_size"], errors="coerce").ne(image_size).any():
        raise ValueError("cache manifest image_size mismatch")
    masked_unavailable_rows = 0
    if available_column is not None:
        if available_column not in manifest.columns:
            raise ValueError(f"available column missing from cache manifest: {available_column}")
        available = _to_bool(manifest[available_column])
        masked_unavailable_rows = int((~available).sum())
        manifest = manifest[available].copy()
    manifest = manifest.sort_values("image_context_id", kind="mergesort").reset_index(drop=True)
    if max_contexts is not None:
        manifest = manifest.head(int(max_contexts)).copy()
    if manifest.empty:
        raise ValueError("cache manifest selection is empty")
    output_dir = output_dir or cache_manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"packed_rgb_{image_size}_letterbox.npy"
    index_path = output_dir / "packed_image_cache_index.csv"
    partial_audit_path = output_dir / "packed_image_cache_audit.partial.json"
    audit_path = output_dir / "packed_image_cache_audit.json"
    shape = (len(manifest), image_size, image_size, 3)
    if overwrite:
        for path in [tensor_path, index_path, partial_audit_path, audit_path]:
            path.unlink(missing_ok=True)
    start_row = 0
    if resume:
        if not tensor_path.exists() or not partial_audit_path.exists():
            raise FileNotFoundError("--resume requires packed tensor and partial audit")
        partial = json.loads(partial_audit_path.read_text(encoding="utf-8"))
        if partial.get("shape") != list(shape) or partial.get("source_manifest_sha256") != _sha256(cache_manifest):
            raise ValueError("packed cache partial lineage mismatch")
        start_row = int(partial.get("completed_rows", 0))
        packed = np.lib.format.open_memmap(tensor_path, mode="r+", dtype=np.uint8, shape=shape)
    else:
        if tensor_path.exists():
            raise FileExistsError(f"packed tensor exists; use --resume or --overwrite: {tensor_path}")
        packed = np.lib.format.open_memmap(tensor_path, mode="w+", dtype=np.uint8, shape=shape)

    records = [
        (int(row_index), str(row.cache_path))
        for row_index, row in enumerate(manifest.itertuples(index=False))
        if row_index >= start_row
    ]
    completed_rows = start_row
    failed_rows = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="packed-cache") as executor:
        for row_index, image, error in executor.map(
            lambda item: _load_cache_row(cache_manifest.parent, image_size, *item),
            records,
            chunksize=64,
        ):
            if error:
                failed_rows += 1
                raise ValueError(f"packed cache source row {row_index} failed: {error}")
            packed[row_index] = image
            completed_rows = row_index + 1
            if completed_rows % checkpoint_every == 0:
                packed.flush()
                _write_partial_audit(
                    partial_audit_path,
                    tensor_path=tensor_path,
                    source_manifest=cache_manifest,
                    shape=shape,
                    completed_rows=completed_rows,
                    workers=workers,
                )
    packed.flush()
    index = pd.DataFrame(
        {
            "image_context_id": manifest["image_context_id"].astype(str),
            "packed_row": np.arange(len(manifest), dtype=np.int64),
        }
    )
    index.to_csv(index_path, index=False)
    tensor = np.load(tensor_path, mmap_mode="r")
    verification_count = min(32, len(manifest))
    verification_positions = np.linspace(0, len(manifest) - 1, verification_count, dtype=int)
    verification_mismatches = 0
    for position in verification_positions:
        source_path = Path(str(manifest.iloc[int(position)]["cache_path"]))
        if not source_path.is_absolute():
            source_path = cache_manifest.parent / source_path
        source_image = np.load(source_path)
        if not np.array_equal(np.asarray(tensor[int(position)]), source_image):
            verification_mismatches += 1
    audit = {
        "schema_version": "classification_v2_packed_image_cache_audit_v1",
        "source_cache_manifest": str(cache_manifest),
        "source_manifest_sha256": _sha256(cache_manifest),
        "packed_tensor_npy": str(tensor_path),
        "packed_index_csv": str(index_path),
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "source_rows": int(len(manifest)),
        "source_manifest_rows": int(source_rows),
        "selected_available_rows": int(len(manifest)),
        "masked_unavailable_rows": int(masked_unavailable_rows),
        "available_column": available_column,
        "packed_rows": int(tensor.shape[0]),
        "index_rows": int(len(index)),
        "start_row": int(start_row),
        "completed_rows": int(completed_rows),
        "failed_rows": int(failed_rows),
        "duplicate_index_ids": int(index["image_context_id"].duplicated().sum()),
        "tensor_size_bytes": int(tensor_path.stat().st_size),
        "verification_sample_rows": int(verification_count),
        "verification_mismatches": int(verification_mismatches),
        "resize_policy_values": sorted(manifest["resize_policy"].astype(str).unique().tolist()),
        "valid": bool(
            tensor.shape == shape
            and tensor.dtype == np.uint8
            and len(index) == len(manifest)
            and completed_rows == len(manifest)
            and failed_rows == 0
            and verification_mismatches == 0
            and not index["image_context_id"].duplicated().any()
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _load_cache_row(base: Path, image_size: int, row_index: int, cache_value: str):
    """Load and validate one source tensor for ordered threaded packing."""

    path = Path(cache_value)
    if not path.is_absolute():
        path = base / path
    try:
        image = np.load(path)
    except Exception as exc:
        return row_index, None, f"load_error:{exc}"
    expected_shape = (image_size, image_size, 3)
    if image.dtype != np.uint8 or image.shape != expected_shape:
        return row_index, None, f"contract_mismatch:dtype={image.dtype}:shape={image.shape}"
    return row_index, image, ""


def _write_partial_audit(
    path: Path,
    *,
    tensor_path: Path,
    source_manifest: Path,
    shape: tuple[int, int, int, int],
    completed_rows: int,
    workers: int,
) -> None:
    audit = {
        "schema_version": "classification_v2_packed_image_cache_partial_audit_v1",
        "packed_tensor_npy": str(tensor_path),
        "source_manifest_sha256": _sha256(source_manifest),
        "shape": list(shape),
        "completed_rows": int(completed_rows),
        "workers": int(workers),
        "complete": False,
    }
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()

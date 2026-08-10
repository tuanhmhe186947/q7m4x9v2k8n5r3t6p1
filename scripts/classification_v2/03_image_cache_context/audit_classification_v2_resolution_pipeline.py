"""Audit native actor-crop detail for the inner-only resolution pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pig_behavior.classification_v2.datasets.resolution_pipeline import (
    build_inner_resolution_binding,
    native_crop_pixel_audit,
    scan_legacy_jpeg_headers,
    split_image_context_sequence,
    storage_projection,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inner-only native crop and storage audit for future 64/160/224 "
            "runtime RGB experiments. It never materializes an RGB cache."
        )
    )
    parser.add_argument("--frame-context-csv", required=True, type=Path)
    parser.add_argument("--window-context-csv", required=True, type=Path)
    parser.add_argument("--inner-selection-csv", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--packed-64-npy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--legacy-header-workers", type=int, default=4)
    parser.add_argument("--legacy-header-checkpoint-every", type=int, default=1000)
    parser.add_argument("--expected-inner-windows", type=int, default=39454)
    parser.add_argument("--expected-inner-observations", type=int, default=201792)
    parser.add_argument("--expected-packed-rows", type=int, default=245680)
    args = parser.parse_args()

    binding = build_inner_resolution_binding(
        frame_context_csv=args.frame_context_csv,
        window_context_csv=args.window_context_csv,
        inner_selection_csv=args.inner_selection_csv,
        media_root=args.media_root,
        expected_window_count=args.expected_inner_windows,
        expected_observation_count=args.expected_inner_observations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    header_csv = args.output_dir / "legacy_jpeg_headers_inner.csv"
    checkpoint_json = args.output_dir / "legacy_jpeg_headers_inner.checkpoint.json"
    started = time.perf_counter()
    header_scan = scan_legacy_jpeg_headers(
        binding.frames,
        media_root=args.media_root,
        output_csv=header_csv,
        checkpoint_json=checkpoint_json,
        workers=args.legacy_header_workers,
        checkpoint_every=args.legacy_header_checkpoint_every,
    )
    header_scan["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    if not header_scan["complete"] or header_scan["failed"]:
        raise RuntimeError(f"legacy header scan incomplete: {header_scan}")

    native = native_crop_pixel_audit(binding, header_csv)
    packed = np.load(args.packed_64_npy, mmap_mode="r")
    expected_shape_tail = (64, 64, 3)
    if packed.dtype != np.uint8 or tuple(packed.shape[1:]) != expected_shape_tail:
        raise RuntimeError(
            "packed 64 cache contract mismatch: "
            f"dtype={packed.dtype} shape={packed.shape}"
        )
    if packed.shape[0] != args.expected_packed_rows:
        raise RuntimeError(
            "packed 64 row count mismatch: "
            f"expected={args.expected_packed_rows} observed={packed.shape[0]}"
        )
    selected_context_slots = sum(
        len(split_image_context_sequence(value))
        for value in binding.windows["image_context_id_sequence"].astype(str)
    )
    audit = {
        "schema_version": "classification_v2_resolution_pipeline_audit_v1",
        "inner_binding": {
            "window_count": binding.window_count,
            "observation_count": binding.observation_count,
            "source_counts": binding.frames["source_type"].value_counts().to_dict(),
            "scientific_identity_sha256": binding.identity_sha256,
            "inner_roles": sorted(
                binding.selection["primary_s1_role"].astype(str).unique().tolist()
            ),
            "selected_context_slots": int(selected_context_slots),
            "window_to_unique_observation_factor": round(
                selected_context_slots / binding.observation_count,
                9,
            ),
        },
        "high_resolution_source": {
            "cvat": "original_local_mp4_plus_authoritative_bbox",
            "legacy": "immutable_recovered_crop_jpeg",
            "depends_on_packed_64": False,
            "media_root": str(args.media_root),
        },
        "legacy_header_scan": header_scan,
        "native_crop_pixel_distribution": native,
        "current_packed_64": {
            "rows": int(packed.shape[0]),
            "shape": [int(value) for value in packed.shape],
            "dtype": str(packed.dtype),
            "payload_bytes": int(packed.nbytes),
        },
        "storage_audit": {
            "t6_inner": storage_projection(binding.observation_count),
            "full_packed_universe": storage_projection(int(packed.shape[0])),
            "inner_window_occurrences_if_not_deduplicated": storage_projection(
                int(selected_context_slots)
            ),
            "explanation": (
                "Large 150-200 GB projections arise from float32 storage, "
                "window-level overlap, multiple resolutions, or extra copies. "
                "This pipeline requires none of these persistent realizations."
            ),
        },
        "storage_policy": {
            "full_160_npy_materialization": "FORBIDDEN",
            "full_224_npy_materialization": "FORBIDDEN",
            "persistent_rgb_authority": "source_compressed_or_uint8_only",
        },
    }
    output_json = args.output_dir / "resolution_pipeline_audit.json"
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

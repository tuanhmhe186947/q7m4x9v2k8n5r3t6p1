"""Materialize only the current-authority H5-to-T6 spatial feature bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_SCHEMA_DIMENSION,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    load_current_spatial_tensor_bundle,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    CANONICAL_SOCIAL_IDENTITY_COLUMN,
    DERIVATION_COLUMNS,
    SPATIAL_FRAME_FEATURES,
    export_spatial_sequences,
)
from pig_behavior.classification_v2.temporal_views.h5_bundle import (
    build_h5_window_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_current_frame_projection(path: Path) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    needed = {
        "object_track_key",
        "frame_index",
        CANONICAL_SOCIAL_IDENTITY_COLUMN,
        "motion_schema_id",
        "motion_schema_version",
        "motion_schema_dimension",
        "motion_schema_feature_names",
        "motion_schema_hash",
    }
    needed.update(DERIVATION_COLUMNS)
    needed.update(
        feature for group in SPATIAL_FRAME_FEATURES.values() for feature in group
    )
    selected = [column for column in header if column in needed]
    if CANONICAL_SOCIAL_IDENTITY_COLUMN not in selected:
        raise ValueError(
            "current H5 frame authority lacks canonical social identity column"
        )
    return pd.read_csv(path, usecols=selected, low_memory=False), header


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".json",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        staged = Path(handle.name)
    os.replace(staged, path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".csv",
        delete=False,
        mode="w",
        encoding="utf-8",
        newline="",
    ) as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
        staged = Path(handle.name)
    os.replace(staged, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
    np.savez_compressed(staged, **arrays)
    os.replace(staged, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-cohort-csv", type=Path, required=True)
    parser.add_argument("--reviewed-frame-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewed-snapshot", required=True)
    parser.add_argument("--reviewed-snapshot-sha256", required=True)
    parser.add_argument("--split-hash", required=True)
    parser.add_argument("--temporal-contract-sha256", required=True)
    parser.add_argument("--producer-code-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"H5 output directory is not empty: {args.output_dir}"
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    cohort = pd.read_csv(args.h5_cohort_csv, low_memory=False)
    frames, source_header = _read_current_frame_projection(
        args.reviewed_frame_features
    )
    windows = build_h5_window_manifest(cohort, frames)
    export = export_spatial_sequences(windows, frames)
    if export.audit["errors"]:
        raise ValueError(f"H5 spatial export failed: {export.audit['errors']}")

    cohort_path = args.output_dir / "common_h5_matched_cohort.csv"
    window_path = args.output_dir / "h5_window_manifest.csv"
    npz_path = args.output_dir / "X_spatial_sequences.npz"
    audit_path = args.output_dir / "spatial_sequence_audit.json"
    manifest_path = args.output_dir / "h5_training_bundle_manifest.json"
    _write_csv_atomic(cohort_path, cohort)
    _write_csv_atomic(window_path, windows)
    _write_npz_atomic(npz_path, export.arrays)
    audit = {
        "schema_version": "classification_v2.h5_spatial_bundle_audit.v1",
        "bundle_role": "strictly_causal_H5_then_fixed_T6",
        "authority_bindings": {
            "reviewed_snapshot": args.reviewed_snapshot,
            "reviewed_snapshot_sha256": args.reviewed_snapshot_sha256,
            "split_hash": args.split_hash,
            "temporal_contract_sha256": args.temporal_contract_sha256,
        },
        "matched_cohort": {
            "path": str(cohort_path),
            "rows": int(len(cohort)),
            "sha256": _sha256(cohort_path),
        },
        "window_manifest": {
            "path": str(window_path),
            "rows": int(len(windows)),
            "sha256": _sha256(window_path),
            "view_type": "T6_TARGET_PLUS_H5",
            "history_length": 5,
            "target_length": 6,
            "history_precedes_target": True,
            "label_source": "T6_target_only",
        },
        "input_projection": {
            "frame_authority_path": str(args.reviewed_frame_features),
            "frame_authority_sha256": _sha256(args.reviewed_frame_features),
            "source_header_count": len(source_header),
            "selected_frame_columns": sorted(frames.columns.tolist()),
        },
        "producer": {
            "code_sha256": args.producer_code_sha256,
            "materialization": "current_H5_derived_feature_bundle_only",
        },
        "model_input_prohibition": {
            "history_behavior_labels_used_as_model_input": False,
            "target_behavior_labels_used_as_model_input": False,
            "source_labels_used_as_model_input": False,
        },
        **export.audit,
    }
    _write_json_atomic(audit_path, audit)
    arrays, loaded_audit = load_current_spatial_tensor_bundle(npz_path, audit_path)
    if arrays["motion_delta"].shape[-1] != MOTION_SCHEMA_DIMENSION:
        raise ValueError("H5 motion tensor does not match the canonical 12D schema")
    manifest = {
        "schema_version": "classification_v2.h5_training_bundle_manifest.v1",
        "status": "CURRENT_AUTHORITY_DERIVED_FEATURE_BUNDLE",
        "bundle": {
            "path": str(npz_path),
            "size_bytes": npz_path.stat().st_size,
            "sha256": _sha256(npz_path),
            "content_sha256": loaded_audit["spatial_tensor_content_hash"],
        },
        "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
        "authority_bindings": audit["authority_bindings"],
        "matched_cohort": audit["matched_cohort"],
        "window_manifest": audit["window_manifest"],
        "producer": audit["producer"],
        "feature_schema": {
            "sha256": loaded_audit["spatial_schema_hash"],
            "ordered_feature_names": loaded_audit["feature_names"],
            "mask_contract_version": loaded_audit["spatial_mask_contract_version"],
        },
        "model_input_prohibition": audit["model_input_prohibition"],
    }
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

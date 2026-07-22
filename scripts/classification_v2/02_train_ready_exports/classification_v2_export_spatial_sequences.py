from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    DERIVATION_COLUMNS,
    SPATIAL_FRAME_FEATURES,
    export_spatial_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export leakage-safe per-frame spatial arrays for reviewed "
            "classification_v2 windows."
        )
    )
    parser.add_argument(
        "--window-manifest-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/sequence_features_reviewed/"
            "sequence_window_manifest.csv"
        ),
    )
    parser.add_argument(
        "--frame-features-csv",
        type=Path,
        default=Path("outputs/classification_v2/review_policy/reviewed_frame_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows"),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived spatial export artifacts explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = args.output_dir / "X_spatial_sequences.npz"
    audit_path = args.output_dir / "spatial_sequence_audit.json"
    require_output_paths_available(
        [npz_path, audit_path],
        overwrite=args.overwrite,
    )
    if not args.window_manifest_csv.exists():
        raise FileNotFoundError(args.window_manifest_csv)
    if not args.frame_features_csv.exists():
        raise FileNotFoundError(args.frame_features_csv)

    windows = pd.read_csv(args.window_manifest_csv, low_memory=False)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be > 0")
        windows = windows.head(args.max_rows).copy()

    # Read only columns needed by the exporter plus all possible spatial columns.
    header = pd.read_csv(args.frame_features_csv, nrows=0).columns.tolist()
    needed = {
        "object_track_key",
        "frame_index",
        "nearest_pig_id",
        "nearest_track_id",
    }
    needed.update(DERIVATION_COLUMNS)
    needed.update(
        feature
        for group in SPATIAL_FRAME_FEATURES.values()
        for feature in group
    )
    usecols = [c for c in header if c in needed]
    frames = pd.read_csv(args.frame_features_csv, usecols=usecols, low_memory=False)

    export = export_spatial_sequences(windows, frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_fn = np.savez_compressed if args.compress else np.savez
    save_fn(npz_path, **export.arrays)

    audit = {
        "window_manifest_csv": str(args.window_manifest_csv),
        "frame_features_csv": str(args.frame_features_csv),
        "outputs": {
            "X_spatial_sequences_npz": str(npz_path),
            "audit_json": str(audit_path),
        },
        **export.audit,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {npz_path}")
    print(f"[OK] wrote {audit_path}")
    summary_keys = [
        "rows",
        "max_window_length",
        "array_shapes",
        "observed_ratio",
        "observed_within_length_ratio",
        "padding_slots",
        "missing_observed_slots_within_length",
        "errors",
        "warnings",
    ]
    print(json.dumps({k: audit[k] for k in summary_keys}, indent=2))
    if audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

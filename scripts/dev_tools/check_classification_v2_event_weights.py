from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check classification_v2 event-balanced weight artifact.")
    parser.add_argument(
        "--event-weight-csv",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/event_weight_manifest.csv"),
    )
    parser.add_argument(
        "--window-manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/check_event_weight_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.event_weight_csv.exists():
        raise FileNotFoundError(args.event_weight_csv)
    if not args.window_manifest_csv.exists():
        raise FileNotFoundError(args.window_manifest_csv)

    weights = pd.read_csv(args.event_weight_csv, low_memory=False)
    window_rows = sum(1 for _ in open(args.window_manifest_csv, encoding="utf-8")) - 1
    required = {
        "window_id",
        "event_overlap_cluster_id",
        "windows_per_event",
        "inverse_windows_per_event",
        "event_balanced_sample_weight",
        "window_valid_for_event_weight",
    }
    missing = sorted(required.difference(weights.columns))
    duplicate_window_id = int(weights["window_id"].duplicated().sum()) if "window_id" in weights else 0
    negative_weight = int((pd.to_numeric(weights["event_balanced_sample_weight"], errors="coerce") < 0).sum())
    invalid_nonzero = int(
        (
            ~_as_bool(weights["window_valid_for_event_weight"])
            & pd.to_numeric(weights["event_balanced_sample_weight"], errors="coerce").fillna(0.0).ne(0.0)
        ).sum()
    )

    errors = []
    if missing:
        errors.append(f"missing_columns={missing}")
    if len(weights) != window_rows:
        errors.append(f"row_count_mismatch weights={len(weights)} windows={window_rows}")
    if duplicate_window_id:
        errors.append(f"duplicate_window_id={duplicate_window_id}")
    if negative_weight:
        errors.append(f"negative_event_balanced_sample_weight={negative_weight}")
    if invalid_nonzero:
        errors.append(f"invalid_window_nonzero_weight={invalid_nonzero}")

    audit = {
        "event_weight_csv": str(args.event_weight_csv),
        "window_manifest_csv": str(args.window_manifest_csv),
        "rows": int(len(weights)),
        "event_overlap_cluster_count": int(weights["event_overlap_cluster_id"].nunique())
        if "event_overlap_cluster_id" in weights
        else 0,
        "max_windows_per_event": int(weights["windows_per_event"].max()) if "windows_per_event" in weights else 0,
        "event_balanced_sample_weight_sum": float(weights["event_balanced_sample_weight"].sum())
        if "event_balanced_sample_weight" in weights
        else 0.0,
        "duplicate_window_id": duplicate_window_id,
        "negative_event_balanced_sample_weight": negative_weight,
        "invalid_window_nonzero_weight": invalid_nonzero,
        "warnings": ["do not use overlapping windows as independent statistical units"],
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()

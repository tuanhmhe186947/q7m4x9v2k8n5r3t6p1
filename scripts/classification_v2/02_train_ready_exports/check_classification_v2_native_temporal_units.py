from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check classification_v2 native temporal-unit dataset.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units/native_temporal_unit_manifest.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units/check_native_temporal_units_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest_csv.exists():
        raise FileNotFoundError(args.manifest_csv)
    df = pd.read_csv(args.manifest_csv, low_memory=False)
    required = {
        "temporal_unit_key",
        "source_type",
        "behavior_label",
        "label_frame_count",
        "review_include_in_training",
        "native_unit_valid_for_main_eval",
        "native_unit_sample_weight",
    }
    missing = sorted(required.difference(df.columns))
    duplicate_temporal_units = int(df["temporal_unit_key"].duplicated().sum()) if "temporal_unit_key" in df else 0
    cvat_bad = int((df["source_type"].eq("cvat_tracking_xml") & df["label_frame_count"].ne(6)).sum())
    legacy_bad = int((df["source_type"].eq("legacy_recovered") & df["label_frame_count"].ne(16)).sum())
    negative_weight = int((pd.to_numeric(df["native_unit_sample_weight"], errors="coerce").fillna(0.0) < 0).sum())

    errors = []
    if missing:
        errors.append(f"missing_columns={missing}")
    if duplicate_temporal_units:
        errors.append(f"duplicate_temporal_unit_key={duplicate_temporal_units}")
    if cvat_bad:
        errors.append(f"cvat_non_6f_units={cvat_bad}")
    if legacy_bad:
        errors.append(f"legacy_non_16f_units={legacy_bad}")
    if negative_weight:
        errors.append(f"negative_native_unit_sample_weight={negative_weight}")

    audit = {
        "manifest_csv": str(args.manifest_csv),
        "rows": int(len(df)),
        "duplicate_temporal_unit_key": duplicate_temporal_units,
        "source_type_counts": df["source_type"].value_counts(dropna=False).to_dict() if "source_type" in df else {},
        "behavior_label_counts": df["behavior_label"].value_counts(dropna=False).to_dict()
        if "behavior_label" in df
        else {},
        "native_unit_valid_for_main_eval_counts": df["native_unit_valid_for_main_eval"]
        .value_counts(dropna=False)
        .to_dict()
        if "native_unit_valid_for_main_eval" in df
        else {},
        "cvat_non_6f_units": cvat_bad,
        "legacy_non_16f_units": legacy_bad,
        "negative_native_unit_sample_weight": negative_weight,
        "warnings": ["native temporal units are the primary publication prediction unit"],
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

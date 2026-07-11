"""Independently check persisted source-matched view invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 source matched views.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/split_manifest.csv"),
    )
    parser.add_argument(
        "--view-csv",
        type=Path,
        default=Path("outputs/classification_v2/source_matched_views/source_matched_view_manifest.csv"),
    )
    args = parser.parse_args()
    original = pd.read_csv(args.input_csv, usecols=["window_id", "behavior_window_label"])
    views = pd.read_csv(args.view_csv, low_memory=False)
    errors: list[str] = []
    if len(original) != len(views):
        errors.append(f"row_count_changed={len(original)}->{len(views)}")
    if views["window_id"].duplicated().any():
        errors.append(f"duplicate_window_id={int(views['window_id'].duplicated().sum())}")
    joined = original.merge(
        views[["window_id", "behavior_window_label"]],
        on="window_id",
        how="outer",
        suffixes=("_before", "_after"),
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        errors.append("window_id_set_changed")
    changed = joined["behavior_window_label_before"].astype(str).ne(
        joined["behavior_window_label_after"].astype(str)
    )
    if changed.any():
        errors.append(f"behavior_labels_changed={int(changed.sum())}")
    matched = views.loc[_to_bool(views["source_class_balance_keep"])]
    contingency = matched.groupby(["behavior_window_label", "source_type"])["window_id"].count()
    unequal = {
        str(label): counts.to_dict()
        for label, counts in contingency.groupby(level=0)
        if counts.nunique() != 1
    }
    if unequal:
        errors.append(f"source_class_quota_unequal={unequal}")
    result = {
        "schema_version": "classification_v2_source_matched_views_check_v1",
        "rows": int(len(views)),
        "matched_6frame_rows": int(_to_bool(views["view_matched_6frame"]).sum()),
        "source_class_balanced_rows": int(len(matched)),
        "labels_with_balanced_source_support": int(contingency.index.get_level_values(0).nunique()),
        "errors": errors,
        "valid": not errors,
    }
    output = args.view_csv.parent / "check_source_matched_view_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()

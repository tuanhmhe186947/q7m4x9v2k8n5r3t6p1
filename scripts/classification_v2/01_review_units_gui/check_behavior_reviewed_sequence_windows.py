"""Fail closed on final reviewed-window behavior coverage and lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.features.sequence_windows import (
    audit_sequence_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-manifest-csv", type=Path, required=True)
    parser.add_argument("--review-unit-manifest-csv", type=Path, required=True)
    parser.add_argument("--reviewed-frame-features-csv", type=Path, required=True)
    parser.add_argument("--sequence-build-audit-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    return parser.parse_args()


def audit_reviewed_windows(
    windows: pd.DataFrame,
    review_units: pd.DataFrame,
    build_audit: dict[str, object],
    reviewed_frame_path: Path,
) -> dict[str, object]:
    audit = audit_sequence_windows(windows)
    errors = list(audit["errors"])
    if "window_uid" in windows.columns:
        errors.append("forbidden_window_uid_column")
    if "window_id" in windows and windows["window_id"].duplicated().any():
        errors.append("duplicate_window_id")
    unit_ids = review_units.get("review_unit_id", pd.Series(dtype=str))
    if unit_ids.astype(str).duplicated().any():
        errors.append("duplicate_review_unit_id")
    if not unit_ids.astype(str).is_unique:
        errors.append("review_unit_manifest_not_one_to_one")
    parameters = build_audit.get("parameters", {})
    if not isinstance(parameters, dict) or parameters.get("build_strategy") != "full_rebuild":
        errors.append("final_sequence_build_not_full_rebuild")
    input_csv = str(build_audit.get("input_csv", ""))
    if Path(input_csv) != reviewed_frame_path:
        errors.append("sequence_input_not_explicit_reviewed_frame_artifact")
    audit.update(
        {
            "review_unit_manifest_rows": int(len(review_units)),
            "reviewed_frame_features_csv": str(reviewed_frame_path),
            "errors": sorted(set(errors)),
            "valid": not errors,
        }
    )
    return audit


def main() -> None:
    args = parse_args()
    for path in (
        args.window_manifest_csv,
        args.review_unit_manifest_csv,
        args.reviewed_frame_features_csv,
        args.sequence_build_audit_json,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    windows = pd.read_csv(args.window_manifest_csv, low_memory=False)
    review_units = pd.read_csv(args.review_unit_manifest_csv, low_memory=False)
    build_audit = json.loads(args.sequence_build_audit_json.read_text(encoding="utf-8"))
    audit = audit_reviewed_windows(
        windows,
        review_units,
        build_audit,
        args.reviewed_frame_features_csv,
    )
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if audit["errors"]:
        raise SystemExit(f"FAIL: {audit['errors']}")
    print(f"[PASS] reviewed sequence-window contract: {args.audit_json}")


if __name__ == "__main__":
    main()

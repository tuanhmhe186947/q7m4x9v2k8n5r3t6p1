"""Validate GUI native-unit inputs without opening Tk or writing decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.behavior_review_contract import (
    audit_review_unit_contract,
)


def validate_gui_contract(units: pd.DataFrame, frames: pd.DataFrame) -> dict[str, Any]:
    contract = audit_review_unit_contract(units)
    errors = list(contract["errors"])
    required_frames = {"temporal_unit_key", "source_type", "frame_index"}
    missing = sorted(required_frames - set(frames.columns))
    if missing:
        errors.append(f"missing_frame_columns={missing}")
    else:
        indexed = pd.DataFrame(
            {
                "temporal_unit_key": frames["temporal_unit_key"]
                .fillna("")
                .astype(str),
                "frame_index": pd.to_numeric(frames["frame_index"], errors="coerce"),
            }
        )
        frames_by_unit = {
            str(key): sorted(group["frame_index"].dropna().astype(int).tolist())
            for key, group in indexed.groupby("temporal_unit_key", sort=False)
        }
        for row in units.itertuples(index=False):
            start = int(row.unit_start_frame)
            end = int(row.unit_end_frame)
            observed = frames_by_unit.get(str(row.temporal_unit_key), [])
            if observed != list(range(start, end + 1)):
                errors.append(f"gui_frame_scope_mismatch={row.review_unit_id}")
    return {
        "review_units": int(len(units)),
        "frame_rows": int(len(frames)),
        "errors": errors,
        "warnings": list(contract["warnings"]),
        "valid": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-units-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()
    units = pd.read_csv(args.review_units_csv, low_memory=False)
    frames = pd.read_csv(args.frame_features_csv, low_memory=False)
    audit = validate_gui_contract(units, frames)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if audit["errors"]:
        raise SystemExit(f"FAIL: {audit['errors']}")
    print(f"[PASS] GUI contract validated without opening GUI: {args.audit_json}")


if __name__ == "__main__":
    main()

"""Audit one CVAT anchor through frame, interval, and review-unit layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _numeric(series: pd.Series) -> pd.Series:
    """Normalize frame-like columns for deterministic filtering."""
    return pd.to_numeric(series, errors="coerce")


def _target_rows(
    frame: pd.DataFrame,
    *,
    video_key: str,
    pig_id: str,
) -> pd.DataFrame:
    """Select one pig in one video without treating pig_id as global identity."""
    return frame[
        frame["video_key"].fillna("").astype(str).eq(video_key)
        & frame["pig_id"].fillna("").astype(str).eq(pig_id)
    ].copy()


def audit_anchor_case(
    enhanced: pd.DataFrame,
    intervals: pd.DataFrame,
    review_units: pd.DataFrame,
    *,
    video_key: str,
    pig_id: str,
    anchor: int,
    expected_behavior: str,
    expected_template: str,
) -> dict[str, Any]:
    """Validate the same anchor contract at all three derived-data layers."""
    errors: list[str] = []
    end = anchor + 5

    frame_rows = _target_rows(
        enhanced,
        video_key=video_key,
        pig_id=pig_id,
    )
    frame_rows["frame_index"] = _numeric(frame_rows["frame_index"])
    frame_rows = frame_rows[frame_rows["frame_index"].between(anchor, end)]
    frame_indices = [
        int(value)
        for value in sorted(
            frame_rows["frame_index"].dropna().astype(int).unique()
        )
    ]
    frame_behaviors = sorted(
        frame_rows["behavior"].fillna("").astype(str).unique()
    )
    anchor_behaviors = sorted(
        frame_rows.loc[frame_rows["frame_index"].eq(anchor), "behavior"]
        .fillna("")
        .astype(str)
        .unique()
    )
    if frame_indices != list(range(anchor, end + 1)):
        errors.append(f"enhanced_frame_indices={frame_indices}")
    if anchor_behaviors != [expected_behavior]:
        errors.append(f"enhanced_anchor_behaviors={anchor_behaviors}")

    interval_rows = _target_rows(
        intervals,
        video_key=video_key,
        pig_id=pig_id,
    )
    interval_rows["label_window_start"] = _numeric(
        interval_rows["label_window_start"]
    )
    interval_rows = interval_rows[
        interval_rows["label_window_start"].eq(anchor)
    ]
    interval_behaviors = sorted(
        interval_rows["behavior_temporal_final"]
        .fillna("")
        .astype(str)
        .unique()
    )
    if len(interval_rows) != 1:
        errors.append(f"temporal_interval_rows={len(interval_rows)}")
    if interval_behaviors != [expected_behavior]:
        errors.append(f"temporal_interval_behaviors={interval_behaviors}")

    unit_rows = _target_rows(
        review_units,
        video_key=video_key,
        pig_id=pig_id,
    )
    unit_rows["unit_start_frame"] = _numeric(unit_rows["unit_start_frame"])
    unit_rows = unit_rows[unit_rows["unit_start_frame"].eq(anchor)]
    unit_behaviors = sorted(
        unit_rows["behavior_label"].fillna("").astype(str).unique()
    )
    templates = sorted(
        unit_rows["review_template"].fillna("").astype(str).unique()
    )
    if len(unit_rows) != 1:
        errors.append(f"review_unit_rows={len(unit_rows)}")
    if unit_behaviors != [expected_behavior]:
        errors.append(f"review_unit_behaviors={unit_behaviors}")
    if templates != [expected_template]:
        errors.append(f"review_templates={templates}")

    return {
        "video_key": video_key,
        "pig_id": pig_id,
        "anchor": anchor,
        "interval_end": end,
        "expected_behavior": expected_behavior,
        "expected_template": expected_template,
        "enhanced_frame_rows": int(len(frame_rows)),
        "enhanced_frame_indices": frame_indices,
        "enhanced_behaviors": frame_behaviors,
        "enhanced_anchor_behaviors": anchor_behaviors,
        "temporal_interval_rows": int(len(interval_rows)),
        "temporal_interval_behaviors": interval_behaviors,
        "review_unit_rows": int(len(unit_rows)),
        "review_unit_behaviors": unit_behaviors,
        "review_templates": templates,
        "errors": errors,
        "valid": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enhanced-csv", type=Path, required=True)
    parser.add_argument("--intervals-csv", type=Path, required=True)
    parser.add_argument("--review-units-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--video-key", default="Pigs281119_000085_30fps")
    parser.add_argument("--pig-id", default="ID_4")
    parser.add_argument("--anchor", type=int, default=1020)
    parser.add_argument("--expected-behavior", default="social-nose")
    parser.add_argument("--expected-template", default="interaction")
    args = parser.parse_args()

    result = audit_anchor_case(
        pd.read_csv(args.enhanced_csv, low_memory=False),
        pd.read_csv(args.intervals_csv, low_memory=False),
        pd.read_csv(args.review_units_csv, low_memory=False),
        video_key=args.video_key,
        pig_id=args.pig_id,
        anchor=args.anchor,
        expected_behavior=args.expected_behavior,
        expected_template=args.expected_template,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

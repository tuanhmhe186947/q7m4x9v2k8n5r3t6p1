from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.datasets.image_context_index import (
    audit_image_context_identifier_contract,
    audit_mandatory_cvat_video_case,
)

REQUIRED_FRAME_COLUMNS = {
    "image_context_id",
    "frame_uid",
    "source_type",
    "video_key",
    "object_track_key",
    "frame_index",
    "image_context_source",
    "resolved_media_path",
    "resolved_media_exists",
    "bbox_context_valid",
    "full_frame_context_available",
    "partner_context_available",
    "image_context_loadable",
    "image_context_error",
}

REQUIRED_WINDOW_COLUMNS = {
    "window_id",
    "object_track_key",
    "window_start_frame",
    "window_end_frame",
    "expected_frame_indices",
    "frame_uid_sequence",
    "image_context_id_sequence",
    "observed_image_context_rows",
    "loadable_image_context_rows",
    "missing_image_context_slots",
    "window_image_context_complete",
}


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check classification_v2 image-context index artifacts."
    )
    parser.add_argument(
        "--frame-context-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/"
            "image_frame_context_manifest.csv"
        ),
    )
    parser.add_argument(
        "--window-context-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/"
            "image_window_context_manifest.csv"
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/"
            "image_context_index_audit.json"
        ),
    )
    args = parser.parse_args()

    frames = pd.read_csv(args.frame_context_csv, low_memory=False)
    windows = pd.read_csv(args.window_context_csv, low_memory=False)
    audit = (
        json.loads(args.audit_json.read_text(encoding="utf-8"))
        if args.audit_json.exists()
        else {}
    )

    errors: list[str] = []
    missing_frame_cols = sorted(REQUIRED_FRAME_COLUMNS.difference(frames.columns))
    missing_window_cols = sorted(REQUIRED_WINDOW_COLUMNS.difference(windows.columns))
    if missing_frame_cols:
        errors.append(f"missing_frame_columns={missing_frame_cols}")
    if missing_window_cols:
        errors.append(f"missing_window_columns={missing_window_cols}")
    if "image_context_id" in frames and frames["image_context_id"].duplicated().any():
        duplicate_context_ids = int(frames["image_context_id"].duplicated().sum())
        errors.append(f"duplicate_image_context_id={duplicate_context_ids}")
    if "window_id" in windows and windows["window_id"].duplicated().any():
        errors.append(f"duplicate_window_id={int(windows['window_id'].duplicated().sum())}")
    if "source_type" in frames:
        available_sources = set(frames["source_type"].astype(str))
        missing_sources = {"legacy_recovered", "cvat_tracking_xml"}.difference(
            available_sources
        )
        if missing_sources:
            errors.append(f"missing_source_types={sorted(missing_sources)}")

    loadable = (
        _to_bool(frames["image_context_loadable"])
        if "image_context_loadable" in frames
        else pd.Series([], dtype=bool)
    )
    cvat_case = audit_mandatory_cvat_video_case(frames)
    if not cvat_case["ok"]:
        errors.append("mandatory_cvat_gui_video_case_failed")
        errors.extend(
            f"mandatory_cvat_gui_video_case:{error}"
            for error in cvat_case["errors"]
        )
    identifier_contract = audit_image_context_identifier_contract(
        frames,
        windows,
    )
    errors.extend(
        f"identifier_contract:{error}"
        for error in identifier_contract["errors"]
    )

    result = {
        "frame_rows": int(len(frames)),
        "window_rows": int(len(windows)),
        "frame_loadable": int(loadable.sum()) if len(loadable) else 0,
        "frame_unloadable": int((~loadable).sum()) if len(loadable) else 0,
        "source_counts": (
            frames["source_type"].value_counts(dropna=False).to_dict()
            if "source_type" in frames
            else {}
        ),
        "window_complete": int(_to_bool(windows["window_image_context_complete"]).sum())
        if "window_image_context_complete" in windows
        else 0,
        "mandatory_cvat_gui_video_case_rows": cvat_case["rows"],
        "mandatory_cvat_gui_video_case_ok": cvat_case["ok"],
        "mandatory_cvat_gui_video_case": cvat_case,
        "identifier_contract": identifier_contract,
        "audit_errors": audit.get("errors", []),
        "audit_warnings": audit.get("warnings", []),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

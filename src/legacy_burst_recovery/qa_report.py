from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _value_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).to_dict().items()}


def build_qa_summary(
    raw_df: pd.DataFrame,
    accepted_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    path_report: pd.DataFrame,
    timestamp_audit: pd.DataFrame,
    depth_audit: pd.DataFrame,
    tracking_failures: pd.DataFrame,
) -> dict[str, object]:
    if dense_df.empty:
        successful_tracklets = 0
        low_confidence_tracklets = 0
        failed_tracklets = 0
        anchor_mod_distribution = {}
        dense_frame_count_distribution = {}
        gt_support_distribution = {}
    else:
        grouped_status = dense_df.groupby("tracklet_id")["tracking_status"]
        successful_tracklets = int(grouped_status.apply(lambda s: s.isin(["ok", "corrected_by_gt"]).all()).sum())
        low_confidence_tracklets = int(grouped_status.apply(lambda s: s.eq("low_confidence").any()).sum())
        failed_tracklets = int(grouped_status.apply(lambda s: s.eq("failed").any()).sum())
        anchor_mod_distribution = _value_counts(
            dense_df.drop_duplicates("tracklet_id")["legacy_anchor_frame_mod_6"]
        )
        dense_frame_count_distribution = _value_counts(dense_df.groupby("tracklet_id").size())
        gt_support_distribution = _value_counts(dense_df.groupby("tracklet_id")["is_gt_support_frame"].sum())

    tracking_failure_examples = []
    if not tracking_failures.empty:
        tracking_failure_examples = tracking_failures.head(20).to_dict(orient="records")

    timestamp_status_counts = {}
    missing_times_txt_count = 0
    parse_failed_times_txt_count = 0
    timestamp_mismatch_count = 0
    timestamp_ok_count = 0
    if "timestamp_status" in timestamp_audit:
        timestamp_status_counts = _value_counts(timestamp_audit["timestamp_status"])
        missing_times_txt_count = int(timestamp_audit["timestamp_status"].eq("missing").sum())
        parse_failed_times_txt_count = int(timestamp_audit["timestamp_status"].eq("parse_failed").sum())
        timestamp_mismatch_count = int(timestamp_audit["timestamp_status"].eq("mismatch").sum())
        timestamp_ok_count = int(timestamp_audit["timestamp_status"].eq("ok").sum())

    summary = {
        "total_csv_rows": int(len(raw_df)),
        "unique_group_id": int(raw_df["group_id"].nunique()) if "group_id" in raw_df else 0,
        "unique_group_id_pig_id": int(raw_df[["group_id", "pig_id"]].drop_duplicates().shape[0])
        if {"group_id", "pig_id"}.issubset(raw_df.columns)
        else 0,
        "unique_source_videos": int(raw_df["video_final"].nunique()) if "video_final" in raw_df else 0,
        "resolved_videos": int(path_report["exists"].sum()) if "exists" in path_report else 0,
        "missing_videos": int((~path_report["exists"]).sum()) if "exists" in path_report else 0,
        "resolved_times_txt": int(path_report["times_txt_exists"].sum()) if "times_txt_exists" in path_report else 0,
        "missing_times_txt_count": missing_times_txt_count,
        "parse_failed_times_txt_count": parse_failed_times_txt_count,
        "timestamp_mismatch_count": timestamp_mismatch_count,
        "timestamp_ok_count": timestamp_ok_count,
        "timestamp_status_distribution": timestamp_status_counts,
        "recovered_tracklets": int(dense_df["tracklet_id"].nunique()) if not dense_df.empty else 0,
        "successful_tracklets": successful_tracklets,
        "low_confidence_tracklets": low_confidence_tracklets,
        "failed_tracklets": failed_tracklets,
        "rejected_rows": int(len(rejected_df)),
        "behavior_distribution": _value_counts(accepted_df["behavior"]) if "behavior" in accepted_df else {},
        "frame_mismatch_count": int(raw_df["frame_mismatch"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        if "frame_mismatch" in raw_df
        else 0,
        "anchor_frame_mod_6_distribution": anchor_mod_distribution,
        "dense_frame_count_distribution": dense_frame_count_distribution,
        "gt_support_frame_count_distribution": gt_support_distribution,
        "interpolated_frame_count": int(dense_df["is_interpolated"].sum()) if "is_interpolated" in dense_df else 0,
        "tracking_failure_examples": tracking_failure_examples,
        "path_resolution_failures": path_report[~path_report["exists"]].head(20).to_dict(orient="records")
        if "exists" in path_report
        else [],
        "depth_provenance_completeness": _value_counts(depth_audit["depth_resources_complete"])
        if "depth_resources_complete" in depth_audit
        else {},
    }
    return summary


def write_qa_reports(output_root: Path, summary: dict[str, object]) -> None:
    (output_root / "qa_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Legacy Burst Recovery QA Report", ""]
    for key, value in summary.items():
        lines.append(f"## {key}")
        lines.append("")
        if isinstance(value, (dict, list)):
            lines.append(json.dumps(value, indent=2, ensure_ascii=False))
        else:
            lines.append(str(value))
        lines.append("")
    (output_root / "qa_report.md").write_text("\n".join(lines), encoding="utf-8")

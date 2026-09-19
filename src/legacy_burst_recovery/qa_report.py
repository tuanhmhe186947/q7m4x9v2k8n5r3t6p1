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
        successful_tracklets = int(
            grouped_status.apply(lambda s: s.isin(["ok", "ok_gt", "corrected_by_gt", "interpolated"]).all()).sum()
        )
        low_confidence_tracklets = int(grouped_status.apply(lambda s: s.eq("low_confidence").any()).sum())
        failed_tracklets = int(grouped_status.apply(lambda s: s.eq("failed").any()).sum())
        anchor_mod_distribution = (
            _value_counts(dense_df.drop_duplicates("tracklet_id")["legacy_anchor_frame_mod_6"])
            if "legacy_anchor_frame_mod_6" in dense_df
            else {}
        )
        dense_frame_count_distribution = _value_counts(dense_df.groupby("tracklet_id").size())
        gt_support_distribution = (
            _value_counts(dense_df.groupby("tracklet_id")["is_gt_support_frame"].sum())
            if "is_gt_support_frame" in dense_df
            else {}
        )

    tracklet_level = dense_df.drop_duplicates("tracklet_id") if not dense_df.empty else pd.DataFrame()
    clean_tracklets = (
        int(tracklet_level["training_tier"].eq("clean").sum()) if "training_tier" in tracklet_level else 0
    )
    review_tracklets = (
        int(tracklet_level["training_tier"].eq("review").sum()) if "training_tier" in tracklet_level else 0
    )
    rejected_tracklets = (
        int(tracklet_level["training_tier"].eq("rejected").sum()) if "training_tier" in tracklet_level else 0
    )
    long_occlusion_tracklets = (
        int(tracklet_level["tracking_status_summary"].eq("long_occlusion").sum())
        if "tracking_status_summary" in tracklet_level
        else 0
    )
    accepted_with_warning_tracklets = (
        int(tracklet_level["training_tier"].eq("warning").sum()) if "training_tier" in tracklet_level else 0
    )
    id_switch_rejected_tracklets = (
        int(
            (
                tracklet_level["training_tier"].eq("rejected")
                & tracklet_level["manual_reason"]
                .fillna("")
                .astype(str)
                .str.contains("id switch", case=False, regex=False)
            ).sum()
        )
        if {"training_tier", "manual_reason"}.issubset(tracklet_level.columns)
        else 0
    )
    if not dense_df.empty and "legacy_gt_mode" in dense_df:
        legacy_gt_mode_values = sorted(dense_df["legacy_gt_mode"].fillna("single_anchor").astype(str).unique().tolist())
        legacy_gt_mode = "multi_anchor" if "multi_anchor" in legacy_gt_mode_values else legacy_gt_mode_values[0]
    else:
        legacy_gt_mode = "single_anchor"
    legacy_gt_tracklets = (
        int(tracklet_level["legacy_gt_mode"].eq("multi_anchor").sum())
        if "legacy_gt_mode" in tracklet_level
        else 0
    )
    legacy_gt_complete_tracklets = (
        int(
            (
                tracklet_level["legacy_gt_mode"].eq("multi_anchor")
                & pd.to_numeric(tracklet_level["legacy_gt_support_count"], errors="coerce").ge(6)
            ).sum()
        )
        if {"legacy_gt_mode", "legacy_gt_support_count"}.issubset(tracklet_level.columns)
        else 0
    )
    legacy_gt_review_tracklets = (
        int(
            (
                tracklet_level["legacy_gt_mode"].eq("multi_anchor")
                & tracklet_level["training_tier"].eq("legacy_gt_review")
            ).sum()
        )
        if {"legacy_gt_mode", "training_tier"}.issubset(tracklet_level.columns)
        else 0
    )
    legacy_gt_missing_tracklets = (
        int(
            (
                tracklet_level["legacy_gt_mode"].eq("multi_anchor")
                & pd.to_numeric(tracklet_level["legacy_gt_support_count"], errors="coerce").lt(6)
            ).sum()
        )
        if {"legacy_gt_mode", "legacy_gt_support_count"}.issubset(tracklet_level.columns)
        else 0
    )
    legacy_gt_support_count_distribution = (
        _value_counts(pd.to_numeric(tracklet_level["legacy_gt_support_count"], errors="coerce"))
        if "legacy_gt_support_count" in tracklet_level
        else {}
    )
    bbox_source_distribution = _value_counts(dense_df["bbox_source"]) if "bbox_source" in dense_df else {}
    detector_disagrees_with_legacy_gt_count = (
        int(dense_df["detector_disagrees_with_legacy_gt"].fillna(False).astype(bool).sum())
        if "detector_disagrees_with_legacy_gt" in dense_df
        else 0
    )
    multi_gt_corrected_frames = (
        int(dense_df["tracking_status"].eq("corrected_by_gt").sum()) if "tracking_status" in dense_df else 0
    )
    interpolated_between_gt_frames = (
        int(dense_df["bbox_source"].eq("interpolated_between_gt").sum()) if "bbox_source" in dense_df else 0
    )
    id_switch_risk_frames = (
        int(pd.to_numeric(dense_df["id_switch_risk_score"], errors="coerce").fillna(0.0).ge(0.5).sum())
        if "id_switch_risk_score" in dense_df
        else 0
    )
    mask_filter_applied = False
    if "mask_filter_applied" in dense_df and not dense_df.empty:
        mask_filter_applied = bool(
            dense_df["mask_filter_applied"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"]).any()
        )
    scene_mask_path = ""
    if "scene_mask_path" in dense_df and not dense_df.empty:
        scene_mask_values = dense_df["scene_mask_path"].dropna().astype(str)
        scene_mask_values = scene_mask_values[scene_mask_values.ne("")]
        scene_mask_path = scene_mask_values.iloc[0] if not scene_mask_values.empty else ""
    total_raw_detections = (
        int(pd.to_numeric(dense_df["num_detections_raw"], errors="coerce").fillna(0).sum())
        if "num_detections_raw" in dense_df
        else 0
    )
    total_mask_kept_detections = (
        int(pd.to_numeric(dense_df["num_detections_after_mask"], errors="coerce").fillna(0).sum())
        if "num_detections_after_mask" in dense_df
        else 0
    )
    total_mask_rejected_detections = (
        int(pd.to_numeric(dense_df["num_detections_outside_mask"], errors="coerce").fillna(0).sum())
        if "num_detections_outside_mask" in dense_df
        else 0
    )
    frames_with_more_than_8_raw_detections = (
        int(pd.to_numeric(dense_df["num_detections_raw"], errors="coerce").fillna(0).gt(8).sum())
        if "num_detections_raw" in dense_df
        else 0
    )
    frames_with_more_than_8_masked_detections = (
        int(pd.to_numeric(dense_df["num_detections_after_mask"], errors="coerce").fillna(0).gt(8).sum())
        if "num_detections_after_mask" in dense_df
        else 0
    )
    selected_bbox_outside_mask_count = 0
    if "qa_notes" in dense_df:
        selected_bbox_outside_mask_count = int(
            dense_df["qa_notes"]
            .fillna("")
            .astype(str)
            .str.contains("selected_bbox_outside_scene_mask", regex=False)
            .sum()
        )
    elif {"selected_det_center_in_mask", "selected_det_bbox_mask_coverage"}.issubset(dense_df.columns):
        center_outside = dense_df["selected_det_center_in_mask"].eq(False)
        coverage = pd.to_numeric(dense_df["selected_det_bbox_mask_coverage"], errors="coerce")
        selected_bbox_outside_mask_count = int((center_outside | coverage.lt(0.5)).fillna(False).sum())

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
        "legacy_gt_mode": legacy_gt_mode,
        "legacy_gt_tracklets": legacy_gt_tracklets,
        "legacy_gt_complete_tracklets": legacy_gt_complete_tracklets,
        "legacy_gt_review_tracklets": legacy_gt_review_tracklets,
        "legacy_gt_missing_tracklets": legacy_gt_missing_tracklets,
        "legacy_gt_support_count_distribution": legacy_gt_support_count_distribution,
        "bbox_source_distribution": bbox_source_distribution,
        "detector_disagrees_with_legacy_gt_count": detector_disagrees_with_legacy_gt_count,
        "multi_gt_corrected_frames": multi_gt_corrected_frames,
        "interpolated_between_gt_frames": interpolated_between_gt_frames,
        "id_switch_risk_frames": id_switch_risk_frames,
        "mask_filter_applied": mask_filter_applied,
        "scene_mask_path": scene_mask_path,
        "total_raw_detections": total_raw_detections,
        "total_mask_kept_detections": total_mask_kept_detections,
        "total_mask_rejected_detections": total_mask_rejected_detections,
        "frames_with_more_than_8_raw_detections": frames_with_more_than_8_raw_detections,
        "frames_with_more_than_8_masked_detections": frames_with_more_than_8_masked_detections,
        "selected_bbox_outside_mask_count": selected_bbox_outside_mask_count,
        "clean_tracklets": clean_tracklets,
        "review_tracklets": review_tracklets,
        "rejected_tracklets": rejected_tracklets,
        "long_occlusion_tracklets": long_occlusion_tracklets,
        "id_switch_rejected_tracklets": id_switch_rejected_tracklets,
        "accepted_with_warning_tracklets": accepted_with_warning_tracklets,
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

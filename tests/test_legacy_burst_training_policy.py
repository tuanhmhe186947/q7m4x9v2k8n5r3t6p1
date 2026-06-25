from __future__ import annotations

from pathlib import Path

import pandas as pd

from legacy_burst_recovery.qa_report import build_qa_summary
from legacy_burst_recovery.sequence_view_builder import build_sequence_views
from legacy_burst_recovery.training_policy import (
    MANUAL_REVIEW_TEMPLATE_COLUMNS,
    apply_training_policy,
    build_manual_review_template,
)


def _dense_rows() -> pd.DataFrame:
    rows = []
    for frame_index in [0, 6, 12]:
        rows.append(
            {
                "tracklet_id": "tracklet_clean",
                "group_id": "g1",
                "sample_id": "s1",
                "pig_id": "p1",
                "behavior": "eat",
                "hidden": False,
                "legacy_anchor_frame": 0,
                "legacy_interval_frame_list": "0|6|12",
                "frame_index": frame_index,
                "timestamp_sec": float(frame_index),
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "qa_status": "ok",
                "qa_notes": "",
                "tracking_status": "ok",
                "track_confidence": 0.9,
                "det_confidence": 0.95,
                "is_interpolated": False,
                "is_gt_support_frame": frame_index == 0,
                "source_video_resolved": "video_a.mp4",
                "source_folder": "folder_a",
                "timestamp_file_resolved": "times_a.txt",
                "depth_video_path": "depth_a.mp4",
                "background_depth_path": "bgd_a.png",
                "depth_scale_path": "scale_a.npy",
                "inverse_intrinsic_path": "inv_a.npy",
                "rot_path": "rot_a.npy",
            }
        )

    for frame_index, interpolated in zip([0, 3, 6, 9, 12], [False, True, True, True, False], strict=True):
        rows.append(
            {
                "tracklet_id": "tracklet_occlusion",
                "group_id": "g2",
                "sample_id": "s2",
                "pig_id": "p2",
                "behavior": "move",
                "hidden": False,
                "legacy_anchor_frame": 0,
                "legacy_interval_frame_list": "0|6|12",
                "frame_index": frame_index,
                "timestamp_sec": float(frame_index),
                "x1": 1.0,
                "y1": 1.0,
                "x2": 11.0,
                "y2": 11.0,
                "qa_status": "review" if interpolated else "ok",
                "qa_notes": "missing_or_unreliable_detection_short_gap" if interpolated else "",
                "tracking_status": "interpolated" if interpolated else "ok",
                "track_confidence": 0.2 if interpolated else 0.9,
                "det_confidence": None if interpolated else 0.92,
                "is_interpolated": interpolated,
                "is_gt_support_frame": frame_index == 0,
                "source_video_resolved": "video_b.mp4",
                "source_folder": "folder_b",
                "timestamp_file_resolved": "times_b.txt",
                "depth_video_path": "depth_b.mp4",
                "background_depth_path": "bgd_b.png",
                "depth_scale_path": "scale_b.npy",
                "inverse_intrinsic_path": "inv_b.npy",
                "rot_path": "rot_b.npy",
            }
        )

    for frame_index in [0, 6, 12]:
        rows.append(
            {
                "tracklet_id": "tracklet_warning",
                "group_id": "g3",
                "sample_id": "s3",
                "pig_id": "p3",
                "behavior": "stand",
                "hidden": False,
                "legacy_anchor_frame": 0,
                "legacy_interval_frame_list": "0|6|12",
                "frame_index": frame_index,
                "timestamp_sec": float(frame_index),
                "x1": 2.0,
                "y1": 2.0,
                "x2": 12.0,
                "y2": 12.0,
                "qa_status": "ok",
                "qa_notes": "",
                "tracking_status": "ok",
                "track_confidence": 0.88,
                "det_confidence": 0.93,
                "is_interpolated": False,
                "is_gt_support_frame": frame_index == 0,
                "source_video_resolved": "video_c.mp4",
                "source_folder": "folder_c",
                "timestamp_file_resolved": "times_c.txt",
                "depth_video_path": "depth_c.mp4",
                "background_depth_path": "bgd_c.png",
                "depth_scale_path": "scale_c.npy",
                "inverse_intrinsic_path": "inv_c.npy",
                "rot_path": "rot_c.npy",
            }
        )

    return pd.DataFrame(rows)


def test_training_policy_and_manual_review_overrides(tmp_path: Path) -> None:
    dense_df = _dense_rows()
    manual_review_df = pd.DataFrame(
        [
            {
                "tracklet_id": "tracklet_warning",
                "manual_decision": "accept_with_note",
                "manual_reason": "minor partial bbox at last frame",
                "include_in_training": True,
            }
        ]
    )

    enriched = apply_training_policy(dense_df, manual_review_df)
    tracklets = enriched.drop_duplicates("tracklet_id").set_index("tracklet_id")

    assert bool(tracklets.loc["tracklet_clean", "include_in_training"]) is True
    assert tracklets.loc["tracklet_clean", "training_tier"] == "clean"

    assert bool(tracklets.loc["tracklet_occlusion", "include_in_training"]) is False
    assert tracklets.loc["tracklet_occlusion", "training_tier"] == "hard_occlusion"
    assert tracklets.loc["tracklet_occlusion", "tracking_status_summary"] == "long_occlusion"
    assert enriched[enriched["tracklet_id"].eq("tracklet_occlusion")]["qa_status"].eq("review").all()

    assert bool(tracklets.loc["tracklet_warning", "include_in_training"]) is True
    assert tracklets.loc["tracklet_warning", "training_tier"] == "warning"
    assert tracklets.loc["tracklet_warning", "manual_decision"] == "accept_with_note"

    template_df = build_manual_review_template(enriched, tmp_path)
    assert list(template_df.columns) == MANUAL_REVIEW_TEMPLATE_COLUMNS
    assert set(template_df["tracklet_id"]) == {"tracklet_clean", "tracklet_occlusion", "tracklet_warning"}

    sequence_df = build_sequence_views(enriched, ["sparse_3_0_6_12"])
    assert set(sequence_df["tracklet_id"]) == {"tracklet_clean", "tracklet_warning"}
    assert set(sequence_df["training_tier"]) == {"clean", "warning"}


def test_manual_review_prefers_sample_id_over_stale_tracklet_id() -> None:
    dense_df = _dense_rows()
    manual_review_df = pd.DataFrame(
        [
            {
                "tracklet_id": "tracklet_clean",
                "sample_id": "s3",
                "manual_decision": "reject",
                "manual_reason": "stale tracklet id from pre-filter review",
                "include_in_training": False,
            }
        ]
    )

    enriched, audit_df = apply_training_policy(
        dense_df,
        manual_review_df,
        return_manual_review_audit=True,
    )
    tracklets = enriched.drop_duplicates("tracklet_id").set_index("tracklet_id")

    assert tracklets.loc["tracklet_warning", "training_tier"] == "rejected"
    assert tracklets.loc["tracklet_warning", "manual_decision"] == "reject"
    assert tracklets.loc["tracklet_clean", "training_tier"] == "clean"
    assert tracklets.loc["tracklet_clean", "manual_decision"] == ""

    audit_row = audit_df.iloc[0]
    assert audit_row["match_key_used"] == "sample_id"
    assert audit_row["matched_tracklet_id"] == "tracklet_warning"
    assert audit_row["matched_sample_id"] == "s3"
    assert bool(audit_row["applied"]) is True
    assert audit_row["reason"] == "applied"


def test_manual_review_tracklet_fallback_rejects_stable_identifier_conflict() -> None:
    dense_df = _dense_rows()
    manual_review_df = pd.DataFrame(
        [
            {
                "tracklet_id": "tracklet_clean",
                "group_id": "wrong_group",
                "manual_decision": "reject",
                "manual_reason": "stale tracklet id with conflicting stable id",
                "include_in_training": False,
            }
        ]
    )

    enriched, audit_df = apply_training_policy(
        dense_df,
        manual_review_df,
        return_manual_review_audit=True,
    )
    tracklets = enriched.drop_duplicates("tracklet_id").set_index("tracklet_id")

    assert tracklets.loc["tracklet_clean", "training_tier"] == "clean"
    assert tracklets.loc["tracklet_clean", "manual_decision"] == ""

    audit_row = audit_df.iloc[0]
    assert audit_row["match_key_used"] == "tracklet_id"
    assert audit_row["matched_tracklet_id"] == "tracklet_clean"
    assert audit_row["matched_group_id"] == "g1"
    assert bool(audit_row["applied"]) is False
    assert audit_row["reason"] == "stable_identifier_conflict:group_id"


def test_qa_summary_counts_training_review_buckets() -> None:
    dense_df = _dense_rows()
    manual_review_df = pd.DataFrame(
        [
            {
                "tracklet_id": "tracklet_warning",
                "manual_decision": "accept_with_note",
                "manual_reason": "minor partial bbox at last frame",
                "include_in_training": True,
            },
            {
                "tracklet_id": "tracklet_occlusion",
                "manual_decision": "reject",
                "manual_reason": "real ID switch for 2-3 frames",
                "include_in_training": False,
            },
        ]
    )
    enriched = apply_training_policy(dense_df, manual_review_df)

    summary = build_qa_summary(
        raw_df=pd.DataFrame({"group_id": ["g1"], "pig_id": ["p1"], "video_final": ["video_a.mp4"]}),
        accepted_df=pd.DataFrame({"behavior": ["eat"], "frame_mismatch": [False]}),
        rejected_df=pd.DataFrame(),
        dense_df=enriched,
        path_report=pd.DataFrame(),
        timestamp_audit=pd.DataFrame(),
        depth_audit=pd.DataFrame(),
        tracking_failures=pd.DataFrame(),
    )

    assert summary["clean_tracklets"] == 1
    assert summary["accepted_with_warning_tracklets"] == 1
    assert summary["rejected_tracklets"] == 1
    assert summary["id_switch_rejected_tracklets"] == 1

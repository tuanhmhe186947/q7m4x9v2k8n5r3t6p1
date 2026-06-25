from __future__ import annotations

import pandas as pd

from legacy_burst_recovery.detector import Detection
from legacy_burst_recovery.legacy_gt_loader import load_legacy_gt_bboxes
from legacy_burst_recovery.tracker import track_dense_range
from legacy_burst_recovery.training_policy import apply_training_policy


def test_load_legacy_gt_bboxes_builds_map_and_review_audit(tmp_path) -> None:
    csv_path = tmp_path / "legacy_gt.csv"
    pd.DataFrame(
        [
            {
                "group_id": "g1",
                "sample_id": "s1",
                "pig_id": "p1",
                "frame_index": 3,
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 10,
                "legacy_order": 0,
                "img_name": "a.jpg",
            },
            {
                "group_id": "g1",
                "sample_id": "s1",
                "pig_id": "p1",
                "frame_index": 6,
                "x1": 1,
                "y1": 1,
                "x2": 11,
                "y2": 11,
                "legacy_order": 1,
                "img_name": "b.jpg",
            },
            {
                "group_id": "g2",
                "sample_id": "s2",
                "pig_id": "p2",
                "frame_index": 3,
                "x1": 5,
                "y1": 5,
                "x2": 4,
                "y2": 8,
                "legacy_order": 0,
                "img_name": "bad.jpg",
            },
            {
                "group_id": "g2",
                "sample_id": "s2",
                "pig_id": "p2",
                "frame_index": 3,
                "x1": 5,
                "y1": 5,
                "x2": 9,
                "y2": 8,
                "legacy_order": 0,
                "img_name": "dup.jpg",
            },
        ]
    ).to_csv(csv_path, index=False)

    legacy_gt_map, audit_df = load_legacy_gt_bboxes(csv_path)

    assert legacy_gt_map[("g1", "p1")][3]["bbox"] == (0.0, 0.0, 10.0, 10.0)
    assert legacy_gt_map[("g1", "p1")][3]["legacy_order"] == 0
    g1_audit = audit_df[audit_df["group_id"].eq("g1") & audit_df["pig_id"].eq("p1")].iloc[0]
    assert g1_audit["loaded_gt_frames"] == 2
    assert g1_audit["qa_status"] == "review"
    assert "missing_legacy_gt_frames" in g1_audit["qa_notes"]

    g2_audit = audit_df[audit_df["group_id"].eq("g2") & audit_df["pig_id"].eq("p2")].iloc[0]
    assert g2_audit["bbox_invalid_rows"] == 1
    assert g2_audit["duplicate_gt_frames"] == "3"
    assert g2_audit["qa_status"] == "review"


def test_multi_anchor_tracker_gt_wins_and_interpolates_between_gt() -> None:
    tracked = track_dense_range(
        [3, 4, 5, 6],
        (0.0, 0.0, 10.0, 10.0),
        {
            3: [Detection(3, 100, 100, 120, 120, 0.99)],
            4: [],
            5: [Detection(5, 100, 100, 120, 120, 0.99)],
            6: [Detection(6, 2, 2, 12, 12, 0.95)],
        },
        [3, 6],
        legacy_gt_by_frame={
            3: {"bbox": (0.0, 0.0, 10.0, 10.0)},
            6: {"bbox": (3.0, 3.0, 13.0, 13.0)},
        },
        legacy_gt_mode="multi_anchor",
    )

    by_frame = {box.frame_index: box for box in tracked}
    assert by_frame[3].bbox_source == "gt_legacy"
    assert by_frame[3].bbox == (0.0, 0.0, 10.0, 10.0)
    assert by_frame[3].detector_disagrees_with_legacy_gt is True
    assert by_frame[4].bbox_source == "interpolated_between_gt"
    assert by_frame[5].tracking_status in {"low_confidence", "needs_review"}
    assert by_frame[6].bbox_source == "gt_legacy"
    assert by_frame[6].track_confidence == 1.0


def test_multi_anchor_training_policy_requires_gt_support() -> None:
    dense_df = pd.DataFrame(
        [
            {
                "tracklet_id": "complete",
                "frame_index": frame,
                "tracking_status": "ok",
                "qa_status": "ok",
                "qa_notes": "",
                "is_interpolated": False,
                "track_confidence": 1.0,
                "det_confidence": 0.9,
                "legacy_gt_mode": "multi_anchor",
                "legacy_gt_support_count": 3,
                "legacy_gt_bbox_available": frame in {3, 9, 15},
            }
            for frame in [3, 6, 9, 12, 15]
        ]
        + [
            {
                "tracklet_id": "missing",
                "frame_index": frame,
                "tracking_status": "ok",
                "qa_status": "ok",
                "qa_notes": "",
                "is_interpolated": False,
                "track_confidence": 1.0,
                "det_confidence": 0.9,
                "legacy_gt_mode": "multi_anchor",
                "legacy_gt_support_count": 2,
                "legacy_gt_bbox_available": frame in {3, 9},
            }
            for frame in [3, 6, 9, 12, 15]
        ],
    )

    enriched = apply_training_policy(dense_df)
    tracklets = enriched.drop_duplicates("tracklet_id").set_index("tracklet_id")

    assert bool(tracklets.loc["complete", "include_in_training"]) is True
    assert tracklets.loc["complete", "training_tier"] == "clean"
    assert bool(tracklets.loc["missing", "include_in_training"]) is False
    assert tracklets.loc["missing", "training_tier"] == "legacy_gt_review"
    assert tracklets.loc["missing", "auto_qa_status"] == "review"

from __future__ import annotations

import pandas as pd
import pytest

from legacy_burst_recovery.detector import Detection
from legacy_burst_recovery.legacy_gt_loader import load_legacy_gt_bboxes
from legacy_burst_recovery.main import legacy_gt_preflight_errors
from legacy_burst_recovery.tracker import track_dense_range
from legacy_burst_recovery.training_policy import apply_training_policy


def test_load_legacy_gt_bboxes_builds_map_and_review_audit(tmp_path) -> None:
    csv_path = tmp_path / "legacy_gt.csv"
    frames = [3, 6, 9, 12, 15, 18]
    rows = [
        {
            "group_id": "g1",
            "sample_id": "s1",
            "pig_id": "p1",
            "frame_index": frame,
            "frames": "3|6|9|12|15|18",
            "behavior": "stand",
            "hidden": "Yes" if slot == 2 else "No",
            "x1": float(slot),
            "y1": 0.0,
            "x2": 10.0 + slot,
            "y2": 10.0,
            "legacy_order": slot,
            "img_name": f"g1_{slot}.jpg",
        }
        for slot, frame in enumerate(frames)
    ]
    duplicate_rows = [dict(row, group_id="g2", sample_id="s2", pig_id="p2") for row in rows]
    duplicate_rows.append(dict(duplicate_rows[0], x1=99.0, x2=109.0))
    pd.DataFrame(rows + duplicate_rows).to_csv(csv_path, index=False)

    legacy_gt_map, audit_df = load_legacy_gt_bboxes(csv_path)

    assert legacy_gt_map[("g1", "p1")][3]["bbox"] == (0.0, 0.0, 10.0, 10.0)
    assert legacy_gt_map[("g1", "p1")][3]["legacy_order"] == 0
    assert legacy_gt_map[("g1", "p1")][9]["hidden"] == "Yes"
    g1_audit = audit_df[audit_df["group_id"].eq("g1") & audit_df["pig_id"].eq("p1")].iloc[0]
    assert g1_audit["loaded_gt_frames"] == 6
    assert g1_audit["qa_status"] == "ok"

    g2_audit = audit_df[audit_df["group_id"].eq("g2") & audit_df["pig_id"].eq("p2")].iloc[0]
    assert ("g2", "p2") not in legacy_gt_map
    assert g2_audit["duplicate_gt_frames"] == "3"
    assert g2_audit["qa_status"] == "error"


@pytest.mark.parametrize("anchor_count", [3, 4, 5])
def test_load_legacy_gt_bboxes_excludes_incomplete_actor(
    tmp_path,
    anchor_count: int,
) -> None:
    csv_path = tmp_path / f"legacy_gt_{anchor_count}.csv"
    frames = [0, 3, 6, 9, 12, 15]
    rows = [
        {
            "group_id": "g1",
            "pig_id": "p1",
            "frame_index": frame,
            "legacy_order": slot,
            "frames": "0|3|6|9|12|15",
            "behavior": "stand",
            "hidden": "No",
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 10.0,
        }
        for slot, frame in enumerate(frames[:anchor_count])
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    legacy_gt_map, audit = load_legacy_gt_bboxes(csv_path)

    assert ("g1", "p1") not in legacy_gt_map
    assert audit.iloc[0]["qa_status"] == "error"
    assert "anchor_row_count_not_six" in audit.iloc[0]["qa_notes"]


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
    assert by_frame[3].is_anchor_frame is True
    assert by_frame[3].bbox == (0.0, 0.0, 10.0, 10.0)
    assert by_frame[3].detector_disagrees_with_legacy_gt is True
    assert by_frame[4].bbox_source == "interpolated_between_gt"
    assert by_frame[4].is_anchor_frame is False
    assert by_frame[5].tracking_status in {"low_confidence", "needs_review"}
    assert by_frame[6].bbox_source == "gt_legacy"
    assert by_frame[6].is_anchor_frame is True
    assert by_frame[6].track_confidence == 1.0


def test_legacy_gt_preflight_stops_before_video_on_invalid_actor() -> None:
    accepted = pd.DataFrame(
        [{"group_id": "g1", "pig_id": "p1", "behavior": "stand"}]
    )
    audit = pd.DataFrame(
        [{"group_id": "g1", "pig_id": "p1", "qa_status": "error"}]
    )

    errors = legacy_gt_preflight_errors(accepted, {}, audit)

    assert "invalid_legacy_gt_actor_keys=1" in errors
    assert "center_actor_keys_missing_six_anchor_gt=1" in errors


def test_legacy_gt_preflight_accepts_matching_six_anchor_actor() -> None:
    accepted = pd.DataFrame(
        [{"group_id": "g1", "pig_id": "p1", "behavior": "stand"}]
    )
    legacy_gt_map = {
        ("g1", "p1"): {
            frame: {"behavior": "stand"}
            for frame in [0, 3, 6, 9, 12, 15]
        }
    }
    audit = pd.DataFrame(
        [{"group_id": "g1", "pig_id": "p1", "qa_status": "ok"}]
    )

    assert legacy_gt_preflight_errors(accepted, legacy_gt_map, audit) == []


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
                "legacy_gt_support_count": 6,
                "legacy_gt_bbox_available": True,
            }
            for frame in [3, 6, 9, 12, 15, 18]
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
                "legacy_gt_support_count": 5,
                "legacy_gt_bbox_available": True,
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

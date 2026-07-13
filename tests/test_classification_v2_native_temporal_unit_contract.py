import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.native_temporal_units import (
    build_native_temporal_units,
)


def _intervals() -> pd.DataFrame:
    """Create one complete CVAT six-frame native interval."""

    return pd.DataFrame(
        {
            "temporal_unit_key": ["unit-0"],
            "source_type": ["cvat_tracking_xml"],
            "dataset_id": ["dataset-0"],
            "video_key": ["video-0"],
            "object_track_key": ["track-0"],
            "pig_id": ["ID_1"],
            "track_id": [1],
            "temporal_label_mode": ["cvat_anchor_6f_interval"],
            "label_anchor_frame_index": [0],
            "label_window_start": [0],
            "label_window_end": [5],
            "label_frame_count": [6],
            "observed_frame_count": [6],
            "expected_observed_frame_count": [6],
            "temporal_interval_complete": [True],
            "behavior_temporal_final": ["stand"],
            "temporal_consistency_status": ["stable"],
            "timestamp_start_sec": [0.0],
            "timestamp_end_sec": [5.0 / 30.0],
            "bbox_valid_ratio_interval": [1.0],
            "hidden_ratio_interval": [0.0],
            "visible_ratio_interval": [1.0],
            "spatiotemporal_feature_valid_ratio_interval": [1.0],
            "interaction_annotation_policy": ["actor_only"],
            "interaction_role_policy": ["actor"],
            "label_propagation_policy": ["none"],
            "allow_label_propagation": [False],
            "requires_partner_context": [False],
            "social_nose_actor_only": [True],
            "fight_group_label": [False],
        }
    )


def _reviewed_frames() -> pd.DataFrame:
    """Create six aligned, unchanged reviewed frame rows."""

    return pd.DataFrame(
        {
            "temporal_unit_key": ["unit-0"] * 6,
            "frame_index": list(range(6)),
            "behavior_before_review": ["stand"] * 6,
            "behavior_after_review": ["stand"] * 6,
            "review_decision_applied": [False] * 6,
            "review_manual_decision": [""] * 6,
            "review_corrected_behavior": [""] * 6,
            "review_training_action": ["include"] * 6,
            "review_sample_weight": [1.0] * 6,
            "review_include_in_training": [True] * 6,
        }
    )


def test_native_temporal_unit_uses_reviewed_behavior_contract() -> None:
    tables = build_native_temporal_units(_intervals(), _reviewed_frames())

    assert tables.audit["errors"] == []
    assert tables.manifest.loc[0, "behavior_label"] == "stand"
    assert tables.manifest.loc[0, "reviewed_frame_count"] == 6


def test_native_temporal_unit_blocks_unapplied_correction_payload() -> None:
    frames = _reviewed_frames()
    frames["review_corrected_behavior"] = "eat"

    tables = build_native_temporal_units(_intervals(), frames)

    assert tables.manifest.loc[0, "behavior_label"] == "stand"
    assert "corrected_without_applied_decision=1" in tables.audit["errors"]


def test_native_temporal_unit_accepts_complete_applied_relabel() -> None:
    frames = _reviewed_frames()
    frames["behavior_after_review"] = "eat"
    frames["review_decision_applied"] = True
    frames["review_manual_decision"] = "correct_label"
    frames["review_corrected_behavior"] = "eat"

    tables = build_native_temporal_units(_intervals(), frames)

    assert tables.audit["errors"] == []
    assert tables.manifest.loc[0, "behavior_label"] == "eat"
    assert bool(tables.manifest.loc[0, "behavior_changed_by_review"])


def test_native_temporal_unit_rejects_partial_decision_scope() -> None:
    frames = _reviewed_frames()
    frames.loc[:2, "review_decision_applied"] = True

    tables = build_native_temporal_units(_intervals(), frames)

    assert "partial_review_decision_scope=1" in tables.audit["errors"]


def test_native_temporal_unit_rejects_duplicate_unit_frame() -> None:
    frames = _reviewed_frames()
    frames = pd.concat([frames, frames.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate_reviewed_unit_frame_rows=2"):
        build_native_temporal_units(_intervals(), frames)


def test_native_temporal_unit_reports_incomplete_frame_coverage() -> None:
    frames = _reviewed_frames().iloc[:-1].copy()

    tables = build_native_temporal_units(_intervals(), frames)

    assert "reviewed_row_count_mismatch=1" in tables.audit["errors"]
    assert "reviewed_end_frame_mismatch=1" in tables.audit["errors"]


def test_native_temporal_unit_audits_legacy_blank_weight_default() -> None:
    frames = _reviewed_frames()
    frames["review_sample_weight"] = pd.NA

    tables = build_native_temporal_units(_intervals(), frames)

    assert tables.audit["errors"] == []
    assert tables.audit["input_alignment"][
        "defaulted_review_sample_weight_rows"
    ] == 6
    assert tables.manifest.loc[0, "review_sample_weight_mean"] == 1.0

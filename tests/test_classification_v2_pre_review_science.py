from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.features.context_policy import (
    apply_context_policy,
    audit_context_policy,
)
from pig_behavior.classification_v2.features.sequence_windows import (
    audit_sequence_windows,
    build_sequence_windows,
)
from pig_behavior.classification_v2.features.temporal_harmonization import (
    audit_temporal_harmonization,
    build_temporal_label_intervals,
    harmonize_temporal_labels,
)
from pig_behavior.classification_v2.review.behavior_review_contract import (
    BEHAVIOR_REVIEW_TEMPLATE,
    audit_review_unit_contract,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    SPATIAL_FRAME_FEATURES,
)
from pig_behavior.classification_v2.train_ready_features import (
    select_window_feature_columns,
)


def _frame_rows(
    source_type: str,
    frame_indices: list[int],
    behaviors: list[str],
) -> pd.DataFrame:
    """Build the smallest valid frame/object input for temporal tests."""
    return pd.DataFrame(
        {
            "source_type": [source_type] * len(frame_indices),
            "dataset_id": ["dataset-a"] * len(frame_indices),
            "video_key": ["video-a"] * len(frame_indices),
            "frame_uid": [f"video-a::f{index:06d}" for index in frame_indices],
            "frame_index": frame_indices,
            "relative_frame_index": list(range(len(frame_indices))),
            "pig_id": ["ID_1"] * len(frame_indices),
            "track_id": ["1"] * len(frame_indices),
            "behavior": behaviors,
            "bbox_valid": [True] * len(frame_indices),
            "hidden": ["No"] * len(frame_indices),
        }
    )


def test_cvat_anchor_label_covers_all_six_training_frames() -> None:
    frames = _frame_rows(
        "cvat_tracking_xml",
        list(range(1020, 1026)),
        ["social-nose", "stand", "stand", "stand", "stand", "stand"],
    )
    before = frames.copy(deep=True)

    harmonized = harmonize_temporal_labels(frames)
    intervals = build_temporal_label_intervals(harmonized)

    assert len(harmonized) == len(before) == 6
    pd.testing.assert_series_equal(frames["behavior"], before["behavior"])
    assert harmonized["behavior_temporal_final"].eq("social-nose").all()
    assert harmonized["label_window_start"].eq(1020).all()
    assert harmonized["label_window_end"].eq(1025).all()
    assert len(intervals) == 1
    assert intervals.iloc[0]["behavior_temporal_final"] == "social-nose"
    assert intervals.iloc[0]["temporal_consistency_status"] == "stable"


def test_legacy_native_unit_is_exact_constant_sixteen_frame_burst() -> None:
    frames = _frame_rows(
        "legacy_recovered",
        list(range(100, 116)),
        ["lying"] * 16,
    )
    harmonized = harmonize_temporal_labels(frames)
    intervals = build_temporal_label_intervals(harmonized)

    assert harmonized["temporal_unit_key"].nunique() == 1
    assert len(intervals) == 1
    interval = intervals.iloc[0]
    assert interval["observed_frame_count"] == 16
    assert interval["label_window_start"] == 100
    assert interval["label_window_end"] == 115
    assert interval["behavior_temporal_final"] == "lying"
    assert bool(interval["temporal_interval_complete"]) is True


def test_lineage_claims_reach_frame_interval_window_and_audits() -> None:
    scope = "legacy-only-unreviewed-development"
    frames = _frame_rows(
        "legacy_recovered",
        list(range(16)),
        ["stand"] * 16,
    )
    frames["lineage_scope"] = scope
    frames["human_review_complete"] = False

    context = apply_context_policy(frames)
    harmonized, intervals, windows = build_sequence_windows(
        context,
        window_lengths=[6, 8, 12, 16],
    )

    for table in (context, harmonized, intervals, windows):
        assert table["lineage_scope"].eq(scope).all()
        assert table["human_review_complete"].eq(False).all()

    audits = (
        audit_context_policy(context),
        audit_temporal_harmonization(harmonized, intervals),
        audit_sequence_windows(windows, intervals),
    )
    for audit in audits:
        assert audit["errors"] == []
        assert audit["lineage_scope"] == scope
        assert audit["human_review_complete"] is False


def test_untrusted_cvat_hidden_is_audited_but_not_used_as_effective_hidden() -> None:
    frames = _frame_rows(
        "cvat_tracking_xml",
        list(range(0, 6)),
        ["stand"] * 6,
    )
    frames["hidden"] = "Yes"
    frames["hidden_is_trusted"] = False

    harmonized = harmonize_temporal_labels(frames)
    interval = build_temporal_label_intervals(harmonized).iloc[0]

    assert interval["hidden_ratio_raw_interval"] == 1.0
    assert interval["hidden_ratio_trusted_interval"] == 0.0
    assert interval["hidden_metadata_untrusted_ratio_interval"] == 1.0


def test_hidden_review_does_not_change_training_weight_by_itself() -> None:
    rows = _frame_rows(
        "cvat_tracking_xml",
        [0, 1],
        ["stand", "stand"],
    )
    rows["frame_uid"] = ["video-a::f000000", "video-a::f000001"]
    rows["hidden"] = ["Yes", "No"]
    rows["hidden_review_status"] = ["reviewed", "reviewed"]

    reviewed = apply_context_policy(rows)

    assert reviewed["include_in_training"].all()
    assert reviewed["hidden_is_trusted"].all()
    assert reviewed.loc[0, "sample_weight"] == reviewed.loc[1, "sample_weight"]
    assert reviewed.loc[0, "training_tier"] == reviewed.loc[1, "training_tier"]


def test_context_policy_groups_object_uids_by_explicit_scene_key() -> None:
    rows = _frame_rows(
        "cvat_tracking_xml",
        [0, 0],
        ["fight", "fight"],
    )
    rows["track_id"] = ["1", "2"]
    rows["pig_id"] = ["ID_1", "ID_2"]

    reviewed = apply_context_policy(rows)

    assert reviewed["scene_frame_uid"].nunique() == 1
    assert reviewed["frame_uid"].nunique() == 2
    assert reviewed["global_context_pig_count"].eq(2).all()
    assert reviewed["interaction_partner_count"].eq(1).all()
    assert set(reviewed["interaction_partner_ids"]) == {"ID_1", "ID_2"}


def test_high_trusted_hidden_ratio_is_not_an_automatic_window_exclusion() -> None:
    frames = _frame_rows(
        "cvat_tracking_xml",
        list(range(0, 6)),
        ["stand"] * 6,
    )
    frames["hidden"] = "Yes"
    frames["hidden_is_trusted"] = True
    frames["hidden_review_status"] = "reviewed"
    frames["spatiotemporal_feature_valid"] = True

    _, _, windows = build_sequence_windows(
        frames,
        window_lengths=[6],
        exclude_high_hidden_from_main=False,
    )

    assert len(windows) == 1
    assert bool(windows.iloc[0]["high_hidden_ratio_window"]) is True
    assert bool(windows.iloc[0]["hidden_exclusion_policy_enabled"]) is False
    assert bool(windows.iloc[0]["window_valid_for_main_train"]) is True


def test_cvat_eight_frame_window_does_not_use_frames_beyond_its_span() -> None:
    frames = _frame_rows(
        "cvat_tracking_xml",
        list(range(0, 12)),
        ["stand"] * 12,
    )
    frames["timestamp_sec"] = frames["frame_index"] / 30.0
    frames["cx_n"] = frames["frame_index"].astype(float)
    frames["cy_n"] = 0.0
    frames["displacement_n"] = 1_000.0
    frames.loc[frames["frame_index"] >= 8, "cx_n"] = 10_000.0

    _, _, windows = build_sequence_windows(frames, window_lengths=[8])

    first = windows.sort_values("window_start_frame").iloc[0]
    assert first["window_start_frame"] == 0
    assert first["window_end_frame"] == 7
    assert first["observed_frame_count_window"] == 8
    assert first["observed_row_count_window"] == 8
    assert first["adjacent_motion_pair_count_window"] == 7
    assert first["path_length_n_window"] == 7.0


@pytest.mark.parametrize("bad_index", [None, "bad", 0.5, -1])
def test_sequence_builder_rejects_invalid_frame_index_without_row_loss(
    bad_index: object,
) -> None:
    frames = _frame_rows(
        "legacy_recovered",
        list(range(0, 16)),
        ["stand"] * 16,
    )
    frames["frame_index"] = frames["frame_index"].astype(object)
    frames.loc[3, "frame_index"] = bad_index

    with pytest.raises(ValueError, match="Temporal identity contract failed"):
        build_sequence_windows(frames, window_lengths=[6])


def test_temporal_harmonization_fills_partial_missing_object_track_keys() -> None:
    frames = _frame_rows(
        "legacy_recovered",
        [0, 1],
        ["stand", "stand"],
    )
    frames["track_id"] = ["1", "2"]
    frames["pig_id"] = ["ID_1", "ID_2"]
    frames["object_track_key"] = ["preserved-key", ""]

    harmonized = harmonize_temporal_labels(frames)

    assert harmonized.loc[0, "object_track_key"] == "preserved-key"
    assert harmonized.loc[1, "object_track_key"] != ""
    assert "track=2" in harmonized.loc[1, "object_track_key"]


def test_temporal_harmonization_rejects_duplicate_track_frame_rows() -> None:
    frames = _frame_rows(
        "legacy_recovered",
        [0, 0],
        ["stand", "stand"],
    )

    with pytest.raises(ValueError, match="duplicate_track_frame_rows=2"):
        harmonize_temporal_labels(frames)


def test_all_ten_behaviors_route_to_the_settled_review_groups() -> None:
    units = []
    for index, (behavior, template) in enumerate(BEHAVIOR_REVIEW_TEMPLATE.items()):
        start = index * 6
        units.append(
            {
                "review_unit_id": f"unit-{index}",
                "review_unit_type": "cvat_interval_6",
                "temporal_unit_key": f"unit-{index}",
                "source_type": "cvat_tracking_xml",
                "unit_start_frame": start,
                "unit_end_frame": start + 5,
                "unit_frame_count": 6,
                "display_frame_indices": ",".join(
                    str(frame) for frame in range(start, start + 6)
                ),
                "behavior_label": behavior,
                "review_template": template,
                "apply_scope": "cvat_interval_6f",
            }
        )

    audit = audit_review_unit_contract(pd.DataFrame(units))
    assert audit["errors"] == []


def test_hidden_metadata_is_audit_only_and_never_selected_for_model_x() -> None:
    windows = pd.DataFrame(
        {
            "speed_mean_window": [0.2],
            "bbox_valid_ratio_window": [1.0],
            "hidden_ratio_window": [0.5],
            "visible_ratio_window": [0.5],
            "behavior_window_label": ["stand"],
            "window_valid_for_main_train": [True],
        }
    )

    selected = select_window_feature_columns(windows)

    assert "speed_mean_window" in selected
    assert "bbox_valid_ratio_window" in selected
    assert "hidden_ratio_window" not in selected
    assert "visible_ratio_window" not in selected
    assert "quality_mask" not in SPATIAL_FRAME_FEATURES


def test_explicit_feature_whitelist_preserves_contract_order() -> None:
    windows = pd.DataFrame(
        {
            "speed_mean_window": [0.2],
            "bbox_valid_ratio_window": [1.0],
            "unused_numeric": [7.0],
        }
    )

    selected = select_window_feature_columns(
        windows,
        feature_whitelist=[
            "bbox_valid_ratio_window",
            "speed_mean_window",
        ],
    )

    assert selected == ["bbox_valid_ratio_window", "speed_mean_window"]


def test_explicit_feature_whitelist_fails_closed_on_contract_drift() -> None:
    windows = pd.DataFrame({"speed_mean_window": [0.2]})

    with pytest.raises(ValueError, match="Missing whitelisted feature columns"):
        select_window_feature_columns(
            windows,
            feature_whitelist=["speed_mean_window", "motion_active_ratio_window"],
        )

    with pytest.raises(ValueError, match="forbidden feature columns"):
        select_window_feature_columns(
            windows.assign(review_score=0.5),
            feature_whitelist=["speed_mean_window", "review_score"],
        )

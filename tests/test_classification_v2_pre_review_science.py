from __future__ import annotations

import pandas as pd

from pig_behavior.classification_v2.features.context_policy import apply_context_policy
from pig_behavior.classification_v2.features.sequence_windows import build_sequence_windows
from pig_behavior.classification_v2.features.temporal_harmonization import (
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
    assert "hidden" not in SPATIAL_FRAME_FEATURES["quality_mask"]
    assert "roi_feature_valid" not in SPATIAL_FRAME_FEATURES["quality_mask"]

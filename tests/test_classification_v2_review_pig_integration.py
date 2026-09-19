from __future__ import annotations

from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.pig_strenet_review_evidence import (
    PIG_REVIEW_EVIDENCE_COLUMNS,
)
from pig_behavior.classification_v2.review.review_unit_builder import (
    ReviewUnitConfig,
    build_review_units,
)
from pig_behavior.classification_v2.train_ready_features import (
    select_window_feature_columns,
)


def _interval() -> dict[str, object]:
    return {
        "temporal_unit_key": "unit-1",
        "behavior_temporal_final": "eat",
        "temporal_observation_ratio_unit": 1.0,
        "temporal_pair_coverage_ratio_unit": 1.0,
        "bbox_valid_ratio_interval": 1.0,
        "motion_active_ratio_per_second_unit": 0.0,
        "motion_stationary_ratio_per_second_unit": 1.0,
        "motion_speed_n_per_second_p90_unit": 0.0,
        "trajectory_straightness_unit": 0.0,
        "bbox_shape_change_p90_unit": 0.0,
        "roi_feeder_near_ratio_unit": 1.0,
        "roi_feeder_availability_ratio_unit": 1.0,
        "roi_feeder_contact_ratio_unit": 1.0,
        "roi_feeder_contact_longest_run_ratio_unit": 1.0,
        "roi_drinker_near_ratio_unit": 0.0,
        "roi_drinker_availability_ratio_unit": 1.0,
        "roi_drinker_contact_ratio_unit": 0.0,
        "roi_drinker_contact_longest_run_ratio_unit": 0.0,
        "roi_toy_near_ratio_unit": 0.0,
        "roi_toy_availability_ratio_unit": 1.0,
        "roi_toy_contact_ratio_unit": 0.0,
        "roi_toy_contact_longest_run_ratio_unit": 0.0,
        "social_pair_contact_ratio_unit": 0.0,
        "social_neighbor_availability_ratio_unit": 0.0,
        "social_partner_persistence_ratio_unit": 0.0,
        "social_nearest_dist_p50_unit": 1.0,
        "social_aggression_proxy_n_per_second_p90_unit": 0.0,
        "source_type": "legacy_recovered",
        "dataset_id": "fixture",
        "video_key": "video",
        "object_track_key": "fixture|video|track=1",
        "pig_id": "ID_1",
        "track_id": "1",
        "label_window_start": 0,
        "label_window_end": 15,
        "temporal_label_mode": "legacy_native_burst_16f",
        "label_anchor_frame_index": 0,
        "temporal_consistency_status": "stable",
        "behavior_consistency_in_interval": True,
        "temporal_interval_complete": True,
    }


def _window() -> dict[str, object]:
    return {
        "window_id": "window-1",
        "source_type": "legacy_recovered",
        "dataset_id": "fixture",
        "video_key": "video",
        "object_track_key": "fixture|video|track=1",
        "pig_id": "ID_1",
        "window_length_frames": 6,
        "window_start_frame": 0,
        "window_end_frame": 5,
        "behavior_window_label": "eat",
        "sequence_label_status": "stable",
        "window_valid_for_main_train": True,
    }


def _write_pig_artifacts(root: Path) -> None:
    pd.DataFrame(
        [
            {
                "pair_id": "pair-1",
                "temporal_unit_key": "unit-1",
                "source_type": "legacy_recovered",
                "video_key": "video",
                "history_expected_frame_count": 6,
                "history_available_ratio": 1.0,
                "history_complete": True,
                "target_complete": True,
                "history_window_start_frame": 0,
                "history_window_end_frame": 5,
                "history_duration_sec": 5 / 30,
                "target_duration_sec": 6 / 30,
            }
        ]
    ).to_csv(root / "pair_manifest.csv", index=False)
    pd.DataFrame([{"pair_id": "pair-1"}]).to_csv(
        root / "history_features.csv",
        index=False,
    )


def test_review_builder_attaches_pig_evidence_without_model_x_leakage(
    tmp_path: Path,
) -> None:
    intervals = pd.DataFrame([_interval()])
    windows = pd.DataFrame([_window()])
    intervals_csv = tmp_path / "intervals.csv"
    windows_csv = tmp_path / "windows.csv"
    artifact_dir = tmp_path / "pig_artifacts"
    output_dir = tmp_path / "review"
    artifact_dir.mkdir()
    intervals.to_csv(intervals_csv, index=False)
    windows.to_csv(windows_csv, index=False)
    _write_pig_artifacts(artifact_dir)

    audit = build_review_units(
        ReviewUnitConfig(
            intervals_csv=intervals_csv,
            sequence_window_manifest_csv=windows_csv,
            output_dir=output_dir,
            pig_strenet_artifact_dir=artifact_dir,
        )
    )

    review = pd.read_csv(output_dir / "review_unit_manifest.csv")
    assert audit["pig_strenet_review_evidence"]["configured"] is True
    assert len(review) == len(intervals)
    assert review["behavior_label"].tolist() == ["eat"]
    assert set(PIG_REVIEW_EVIDENCE_COLUMNS).issubset(review.columns)
    assert review["review_pig_pair_count"].tolist() == [1]
    assert review["review_pig_evidence_available"].tolist() == [True]

    model_columns = select_window_feature_columns(review)
    assert not any(column.startswith("review_pig_") for column in model_columns)

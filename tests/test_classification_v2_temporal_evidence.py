from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from pig_behavior.classification_v2.features.temporal_evidence import (
    TEMPORAL_EVIDENCE_BASE_COLUMNS,
    UNIT_TEMPORAL_EVIDENCE_COLUMNS,
    add_unit_temporal_evidence,
    summarize_temporal_evidence,
)


def _six_frame_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": ["cvat|track=4|anchor=0"] * 6,
            "label_window_start": [0] * 6,
            "label_window_end": [5] * 6,
            "frame_index": list(range(6)),
            "timestamp_sec": np.arange(6) / 30.0,
            "cx_n": np.arange(6) * 0.01,
            "cy_n": np.zeros(6),
            "area_n": [0.12, 0.12, 0.13, 0.13, 0.12, 0.12],
            "aspect_ratio": [2.0, 2.0, 2.1, 2.1, 2.0, 2.0],
            "shape_change_score": [0.0, 0.01, 0.02, 0.01, 0.01, 0.0],
            "roi_feeder_min_dist_n": [0.2, 0.1, 0.0, 0.0, 0.1, 0.2],
            "roi_feeder_max_overlap_ratio": [0, 0, 0.3, 0.4, 0, 0],
            "roi_feeder_near": [False, True, True, True, True, False],
            "roi_feeder_contact": [False, False, True, True, False, False],
            "nearest_pig_id": ["ID_2", "ID_2", "ID_2", "ID_3", "ID_3", "ID_3"],
            "nearest_dist_n": [0.2, 0.1, 0.05, 0.04, 0.06, 0.08],
            "pair_contact_with_nearest": [False, False, True, True, True, False],
            "approach_speed_n_per_frame": [0, 0.1, 0.1, 0.01, 0, 0],
            "aggression_score_proxy": [0, 0, 0.2, 0.4, 0.1, 0],
            "behavior": ["move"] * 6,
        }
    )


def test_temporal_evidence_is_label_independent() -> None:
    frames = _six_frame_fixture()
    first = summarize_temporal_evidence(
        frames,
        expected_start=0,
        expected_end=5,
    )
    frames["behavior"] = ["lying", "fight", "eat", "stand", "move", "drink"]
    frames["manual_corrected_behavior"] = ["eat"] * 6
    frames["review_decision"] = ["accept"] * 6
    frames["hidden"] = [True] * 6
    second = summarize_temporal_evidence(
        frames,
        expected_start=0,
        expected_end=5,
    )

    assert set(first) == set(TEMPORAL_EVIDENCE_BASE_COLUMNS)
    assert first == second
    assert not any(
        token in column
        for column in first
        for token in ("behavior", "manual", "review", "hidden", "target_roi")
    )


def test_straight_motion_and_persistence_metrics_are_correct() -> None:
    evidence = summarize_temporal_evidence(
        _six_frame_fixture(),
        expected_start=0,
        expected_end=5,
    )

    assert evidence["temporal_observation_ratio"] == 1.0
    assert evidence["temporal_pair_coverage_ratio"] == 1.0
    assert evidence["motion_active_ratio"] == 1.0
    assert evidence["motion_longest_active_run_ratio"] == 1.0
    assert evidence["trajectory_straightness"] == 1.0
    assert evidence["turning_direction_concentration"] == 1.0
    assert evidence["roi_feeder_contact_ratio"] == 2 / 6
    assert evidence["roi_feeder_contact_longest_run_ratio"] == 2 / 6
    assert evidence["roi_feeder_contact_episode_count"] == 1
    assert evidence["social_partner_persistence_ratio"] == 0.5
    assert evidence["social_partner_turnover_rate"] == 0.2
    assert evidence["social_pair_contact_ratio"] == 0.5


def test_gaps_and_duplicates_are_audited_without_row_loss() -> None:
    frames = _six_frame_fixture().iloc[[0, 1, 3, 3, 5]].copy()
    frames.iloc[3, frames.columns.get_loc("frame_index")] = 3
    evidence = summarize_temporal_evidence(
        frames,
        expected_start=0,
        expected_end=5,
    )

    assert evidence["temporal_observation_ratio"] == 4 / 6
    assert evidence["temporal_duplicate_frame_ratio"] == 1 / 5
    assert evidence["temporal_max_gap_frames"] == 2.0
    assert evidence["temporal_contiguous_pair_ratio"] == 0.2


def test_unit_attachment_preserves_rows_keys_and_labels() -> None:
    frames = _six_frame_fixture()
    before = frames.copy(deep=True)

    enriched = add_unit_temporal_evidence(frames)

    assert len(enriched) == len(before)
    pdt.assert_series_equal(enriched["behavior"], before["behavior"])
    pdt.assert_series_equal(
        enriched["temporal_unit_key"],
        before["temporal_unit_key"],
    )
    assert set(UNIT_TEMPORAL_EVIDENCE_COLUMNS).issubset(enriched.columns)
    assert enriched["motion_active_ratio_unit"].nunique() == 1

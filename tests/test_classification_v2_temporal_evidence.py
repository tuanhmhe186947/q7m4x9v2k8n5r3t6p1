from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from pig_behavior.classification_v2.contracts.feature_semantics import (
    _assign_tabular_families,
)
from pig_behavior.classification_v2.contracts.temporal_evidence import (
    audit_temporal_evidence_lineage,
)
from pig_behavior.classification_v2.features.sequence_windows import (
    build_sequence_windows,
)
from pig_behavior.classification_v2.features.temporal_evidence import (
    TEMPORAL_EVIDENCE_BASE_COLUMNS,
    UNIT_TEMPORAL_EVIDENCE_COLUMNS,
    WINDOW_TEMPORAL_EVIDENCE_COLUMNS,
    add_unit_temporal_evidence,
    summarize_temporal_evidence,
)
from pig_behavior.classification_v2.features.temporal_harmonization import (
    build_temporal_label_intervals,
)
from pig_behavior.classification_v2.review.review_unit_builder import (
    _base_units_from_intervals,
    _finalize_unit_review_fields,
)


def _six_frame_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * 6,
            "dataset_id": ["fixture"] * 6,
            "video_key": ["video"] * 6,
            "object_track_key": ["fixture|video|track=4"] * 6,
            "pig_id": ["ID_4"] * 6,
            "track_id": ["4"] * 6,
            "temporal_label_mode": ["cvat_anchor_6f_interval"] * 6,
            "label_anchor_frame_index": [0] * 6,
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
            "roi_feeder_available": [True] * 6,
            "roi_feeder_max_overlap_ratio": [0, 0, 0.3, 0.4, 0, 0],
            "roi_feeder_near": [False, True, True, True, True, False],
            "roi_feeder_contact": [False, False, True, True, False, False],
            "nearest_pig_id": ["ID_2", "ID_2", "ID_2", "ID_3", "ID_3", "ID_3"],
            "nearest_track_id": ["2", "2", "2", "3", "3", "3"],
            "nearest_dist_n": [0.2, 0.1, 0.05, 0.04, 0.06, 0.08],
            "pair_contact_with_nearest": [False, False, True, True, True, False],
            "approach_speed_n_per_frame": [0, 0.1, 0.1, 0.01, 0, 0],
            "aggression_score_proxy": [0, 0, 0.2, 0.4, 0.1, 0],
            "behavior": ["move"] * 6,
            "bbox_valid": [True] * 6,
            "hidden": [False] * 6,
            "hidden_is_trusted": [False] * 6,
            "spatiotemporal_feature_valid": [True] * 6,
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
    assert evidence["trajectory_tortuosity_log1p"] == 0.0
    assert evidence["turning_direction_concentration"] == 1.0
    assert 0.0 <= evidence["turning_direction_concentration"] <= 1.0
    assert evidence["roi_feeder_contact_ratio"] == 2 / 6
    assert evidence["roi_feeder_contact_longest_run_ratio"] == 2 / 6
    assert evidence["roi_feeder_contact_episode_count"] == 1
    assert evidence["social_partner_persistence_ratio"] == 0.5
    assert evidence["social_partner_turnover_rate"] == 0.2
    assert evidence["social_pair_contact_ratio"] == 0.5


def test_explicit_roi_availability_and_track_partner_fallback_are_respected() -> None:
    frames = _six_frame_fixture()
    frames["roi_feeder_available"] = False
    frames["nearest_pig_id"] = ""

    evidence = summarize_temporal_evidence(
        frames,
        expected_start=0,
        expected_end=5,
    )

    assert evidence["roi_feeder_availability_ratio"] == 0.0
    assert evidence["roi_feeder_contact_ratio"] == 0.0
    assert evidence["social_partner_persistence_ratio"] == 0.5


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


def test_temporal_intervals_receive_one_constant_unit_evidence_vector() -> None:
    enriched = add_unit_temporal_evidence(_six_frame_fixture())

    intervals = build_temporal_label_intervals(enriched)

    assert len(intervals) == 1
    assert intervals.iloc[0]["behavior_temporal_final"] == "move"
    assert intervals.iloc[0]["motion_active_ratio_unit"] == 1.0
    assert intervals.iloc[0]["roi_feeder_contact_ratio_unit"] == 2 / 6


def test_stale_cvat_hidden_trust_is_rejected_without_review_provenance() -> None:
    frames = _six_frame_fixture()
    frames["hidden"] = True
    frames["hidden_is_trusted"] = True

    intervals = build_temporal_label_intervals(frames)

    assert intervals.iloc[0]["hidden_ratio_raw_interval"] == 1.0
    assert intervals.iloc[0]["hidden_ratio_trusted_interval"] == 0.0
    assert intervals.iloc[0]["hidden_metadata_untrusted_ratio_interval"] == 1.0


def test_sequence_window_recomputes_evidence_inside_requested_span() -> None:
    enriched = add_unit_temporal_evidence(_six_frame_fixture())

    _, _, windows = build_sequence_windows(
        enriched,
        window_lengths=(6,),
        legacy_window_stride=1,
    )

    assert len(windows) == 1
    assert windows.iloc[0]["temporal_observation_ratio_window"] == 1.0
    assert windows.iloc[0]["motion_active_ratio_window"] == 1.0
    assert windows.iloc[0]["roi_feeder_contact_ratio_window"] == 2 / 6


def test_trainer_whitelist_and_semantics_cover_every_new_window_feature() -> None:
    trainer = json.loads(
        Path("configs/classification_v2/trainer_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    semantics = json.loads(
        Path("configs/classification_v2/feature_semantics_v1.json").read_text(
            encoding="utf-8"
        )
    )
    whitelist = trainer["tabular_feature_whitelist"]
    assignments = _assign_tabular_families(
        list(WINDOW_TEMPORAL_EVIDENCE_COLUMNS),
        semantics["tabular_families"],
    )

    assert len(whitelist) == len(set(whitelist))
    assert set(WINDOW_TEMPORAL_EVIDENCE_COLUMNS).issubset(whitelist)
    assert all(assignments.values())


def test_cross_artifact_temporal_evidence_audit_passes_one_unit_lineage() -> None:
    enriched = add_unit_temporal_evidence(_six_frame_fixture())
    harmonized, intervals, windows = build_sequence_windows(
        enriched,
        window_lengths=(6,),
    )
    review_units = _base_units_from_intervals(intervals)
    review_units["window_review_hit_count"] = 0
    review_units["review_templates_hit"] = ""
    review_units["review_reasons_window"] = ""
    review_units["review_priority_window_max"] = 0.0
    review_units = _finalize_unit_review_fields(review_units)
    trainer = json.loads(
        Path("configs/classification_v2/trainer_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )

    audit = audit_temporal_evidence_lineage(
        harmonized,
        intervals,
        windows,
        review_units,
        trainer,
    )

    assert audit["valid"] is True
    assert audit["keys"]["duplicate_temporal_unit_key"] == 0
    assert audit["keys"]["duplicate_window_id"] == 0
    assert audit["review_units"]["evidence_available_rows"] == 1

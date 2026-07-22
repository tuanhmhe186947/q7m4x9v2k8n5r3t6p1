from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from pig_behavior.classification_v2.review.behavior_evidence import (
    REVIEW_EVIDENCE_COLUMNS,
    add_behavior_review_evidence,
)
from pig_behavior.classification_v2.review.review_unit_builder import (
    ReviewUnitConfig,
    _base_units_from_intervals,
    _finalize_unit_review_fields,
    build_review_units,
)
from pig_behavior.classification_v2.train_ready_features import (
    select_window_feature_columns,
)


def _unit(behavior: str, **overrides: float) -> dict[str, object]:
    row: dict[str, object] = {
        "temporal_unit_key": "unit-1",
        "behavior_temporal_final": behavior,
        "temporal_observation_ratio_unit": 1.0,
        "temporal_pair_coverage_ratio_unit": 1.0,
        "bbox_valid_ratio_interval": 1.0,
        "motion_active_ratio_per_second_unit": 0.0,
        "motion_stationary_ratio_per_second_unit": 1.0,
        "motion_speed_n_per_second_p90_unit": 0.0,
        "trajectory_straightness_unit": 0.0,
        "bbox_shape_change_p90_unit": 0.0,
        "roi_feeder_near_ratio_unit": 0.0,
        "roi_feeder_availability_ratio_unit": 1.0,
        "roi_feeder_contact_ratio_unit": 0.0,
        "roi_feeder_contact_longest_run_ratio_unit": 0.0,
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
    }
    row.update(overrides)
    return row


def test_review_evidence_flags_stationary_move_without_changing_label() -> None:
    units = pd.DataFrame([_unit("move")])
    labels_before = units["behavior_temporal_final"].copy()

    scored = add_behavior_review_evidence(units)

    pdt.assert_series_equal(scored["behavior_temporal_final"], labels_before)
    assert len(scored) == len(units)
    assert scored.iloc[0]["review_evidence_conflict_score"] > 0.9
    assert "move_with_weak_motion_evidence" in scored.iloc[0][
        "review_evidence_reason_auto"
    ]


def test_target_roi_persistence_supports_eat_but_not_drink() -> None:
    feeder_support = {
        "roi_feeder_near_ratio_unit": 1.0,
        "roi_feeder_contact_ratio_unit": 1.0,
        "roi_feeder_contact_longest_run_ratio_unit": 1.0,
    }
    units = pd.DataFrame(
        [
            _unit("eat", **feeder_support),
            _unit("drink", **feeder_support),
        ]
    )

    scored = add_behavior_review_evidence(units)

    assert scored.iloc[0]["review_evidence_conflict_score"] == 0.0
    assert scored.iloc[1]["review_evidence_conflict_score"] == 1.0
    assert "different_roi_has_stronger_support" in scored.iloc[1][
        "review_evidence_reason_auto"
    ]


def test_fight_without_partner_evidence_is_high_priority() -> None:
    scored = add_behavior_review_evidence(pd.DataFrame([_unit("fight")]))

    assert scored.iloc[0]["review_evidence_conflict_score"] == 0.0
    assert scored.iloc[0]["review_evidence_insufficiency_score"] == 1.0
    assert not bool(scored.iloc[0]["review_social_evidence_available"])
    assert scored.iloc[0]["review_evidence_priority_auto"] >= 60.0
    assert "social_evidence_unavailable" in scored.iloc[0][
        "review_evidence_reason_auto"
    ]
    assert "fight_vs_social-nose_stand_move" in scored.iloc[0][
        "review_confusion_pairs_auto"
    ]


def test_available_social_context_can_contradict_fight_label() -> None:
    scored = add_behavior_review_evidence(
        pd.DataFrame(
            [
                _unit(
                    "fight",
                    social_neighbor_availability_ratio_unit=1.0,
                )
            ]
        )
    )

    assert scored.iloc[0]["review_evidence_insufficiency_score"] == 0.0
    assert scored.iloc[0]["review_evidence_conflict_score"] == 1.0
    assert "fight_without_persistent_contact_or_aggression" in scored.iloc[0][
        "review_evidence_reason_auto"
    ]


def test_pig_strenet_motion_support_is_validity_masked() -> None:
    base = _unit("move")
    base.update(
        {
            "review_pig_evidence_available": True,
            "review_pig_diff_valid_ratio": 1.0,
            "review_pig_diff_active_pixel_ratio": 1.0,
            "review_pig_history_transition_available": False,
            "review_pig_stationary_to_motion_score": 1.0,
        }
    )
    valid_diff = add_behavior_review_evidence(pd.DataFrame([base])).iloc[0]
    no_diff = dict(base)
    no_diff["review_pig_diff_valid_ratio"] = 0.0
    masked = add_behavior_review_evidence(pd.DataFrame([no_diff])).iloc[0]

    assert valid_diff["review_evidence_conflict_score"] == 0.0
    assert masked["review_evidence_conflict_score"] > 0.9
    assert masked["review_temporal_phase_support_score"] == 0.0


def test_explore_roi_contact_requires_stationary_pattern_for_conflict() -> None:
    roi_contact = {
        "roi_feeder_near_ratio_unit": 1.0,
        "roi_feeder_contact_ratio_unit": 1.0,
        "roi_feeder_contact_longest_run_ratio_unit": 1.0,
    }
    moving = add_behavior_review_evidence(
        pd.DataFrame(
            [
                _unit(
                    "explore",
                    motion_active_ratio_per_second_unit=1.0,
                    motion_stationary_ratio_per_second_unit=0.0,
                    motion_speed_n_per_second_p90_unit=0.60,
                    **roi_contact,
                )
            ]
        )
    ).iloc[0]
    stationary = add_behavior_review_evidence(
        pd.DataFrame([_unit("explore", **roi_contact)])
    ).iloc[0]

    assert moving["review_evidence_conflict_score"] < 0.45
    assert stationary["review_evidence_conflict_score"] > 0.65
    assert "stationary_persistent_roi_contact" in stationary[
        "review_evidence_reason_auto"
    ]


def test_missing_target_roi_is_insufficient_not_contradictory() -> None:
    scored = add_behavior_review_evidence(
        pd.DataFrame(
            [
                _unit(
                    "eat",
                    roi_feeder_availability_ratio_unit=0.0,
                )
            ]
        )
    )

    assert scored.iloc[0]["review_evidence_conflict_score"] == 0.0
    assert scored.iloc[0]["review_evidence_insufficiency_score"] == 1.0
    assert "target_roi_evidence_unavailable" in scored.iloc[0][
        "review_evidence_reason_auto"
    ]


def test_review_scores_are_never_selected_for_model_x() -> None:
    scored = add_behavior_review_evidence(pd.DataFrame([_unit("move")]))
    scored["speed_n_per_second_mean_window"] = 0.0
    scored["motion_active_ratio_per_second_window"] = 0.5
    scored["roi_feeder_contact_ratio_window"] = 0.25
    scored["social_partner_persistence_ratio_window"] = 0.75

    selected = select_window_feature_columns(scored)

    assert "speed_n_per_second_mean_window" in selected
    assert "motion_active_ratio_per_second_window" in selected
    assert "roi_feeder_contact_ratio_window" in selected
    assert "social_partner_persistence_ratio_window" in selected
    assert not set(REVIEW_EVIDENCE_COLUMNS).intersection(selected)


def test_review_unit_builder_routes_evidence_conflict_without_relabeling() -> None:
    interval = _unit("move")
    interval.update(
        {
            "source_type": "cvat_tracking_xml",
            "dataset_id": "fixture",
            "video_key": "video",
            "object_track_key": "fixture|video|track=4",
            "pig_id": "ID_4",
            "track_id": "4",
            "label_window_start": 0,
            "label_window_end": 5,
            "temporal_label_mode": "cvat_anchor_6f_interval",
            "label_anchor_frame_index": 0,
            "temporal_consistency_status": "stable",
            "behavior_consistency_in_interval": True,
            "temporal_interval_complete": True,
            "bbox_valid_ratio_interval": 1.0,
        }
    )
    units = _base_units_from_intervals(pd.DataFrame([interval]))
    units["window_review_hit_count"] = 0
    units["review_templates_hit"] = ""
    units["review_reasons_window"] = ""
    units["review_priority_window_max"] = 0.0

    finalized = _finalize_unit_review_fields(units)

    assert finalized.iloc[0]["behavior_label"] == "move"
    assert finalized.iloc[0]["review_template"] == "motion"
    assert bool(finalized.iloc[0]["include_in_review"])
    assert "move_with_weak_motion_evidence" in finalized.iloc[0]["review_reason"]
    assert finalized.iloc[0]["apply_scope"] == "cvat_interval_6f"


def test_complete_legacy_review_includes_stable_native_unit() -> None:
    interval = _unit(
        "eat",
        roi_feeder_near_ratio_unit=1.0,
        roi_feeder_contact_ratio_unit=1.0,
        roi_feeder_contact_longest_run_ratio_unit=1.0,
    )
    interval.update(
        {
            "source_type": "legacy_recovered",
            "dataset_id": "fixture",
            "video_key": "video",
            "object_track_key": "fixture|video|track=4",
            "pig_id": "ID_4",
            "track_id": "4",
            "label_window_start": 0,
            "label_window_end": 15,
            "temporal_label_mode": "legacy_native_burst_16f",
            "label_anchor_frame_index": 0,
            "temporal_consistency_status": "stable",
            "behavior_consistency_in_interval": True,
            "temporal_interval_complete": True,
            "bbox_valid_ratio_interval": 1.0,
        }
    )
    units = _base_units_from_intervals(pd.DataFrame([interval]))
    units["window_review_hit_count"] = 0
    units["review_templates_hit"] = ""
    units["review_reasons_window"] = ""
    units["review_priority_window_max"] = 0.0

    selected = _finalize_unit_review_fields(units)
    complete = _finalize_unit_review_fields(
        units,
        include_all_retained_legacy_units=True,
    )

    assert not bool(selected.iloc[0]["include_in_review"])
    assert bool(complete.iloc[0]["include_in_review"])
    assert "full_legacy_native_unit_review" in complete.iloc[0]["review_reason"]
    assert complete.iloc[0]["behavior_review_cohort"] == (
        "behavior_mandatory_census"
    )
    assert complete.iloc[0]["review_template"] == "roi"


def test_complete_legacy_review_audit_matches_full_manifest(
    tmp_path: Path,
) -> None:
    interval = _unit(
        "eat",
        roi_feeder_near_ratio_unit=1.0,
        roi_feeder_contact_ratio_unit=1.0,
        roi_feeder_contact_longest_run_ratio_unit=1.0,
    )
    interval.update(
        {
            "source_type": "legacy_recovered",
            "dataset_id": "fixture",
            "video_key": "video",
            "object_track_key": "fixture|video|track=4",
            "pig_id": "ID_4",
            "track_id": "4",
            "label_window_start": 0,
            "label_window_end": 15,
            "temporal_label_mode": "legacy_native_burst_16f",
            "label_anchor_frame_index": 0,
            "temporal_consistency_status": "stable",
            "behavior_consistency_in_interval": True,
            "temporal_interval_complete": True,
            "bbox_valid_ratio_interval": 1.0,
        }
    )
    window = {
        "window_id": "window-1",
        "source_type": "legacy_recovered",
        "dataset_id": "fixture",
        "video_key": "video",
        "object_track_key": "fixture|video|track=4",
        "pig_id": "ID_4",
        "window_length_frames": 6,
        "window_start_frame": 0,
        "window_end_frame": 5,
        "behavior_window_label": "eat",
        "sequence_label_status": "stable",
        "window_valid_for_main_train": True,
    }
    intervals_csv = tmp_path / "intervals.csv"
    windows_csv = tmp_path / "windows.csv"
    output_dir = tmp_path / "review"
    pd.DataFrame([interval]).to_csv(intervals_csv, index=False)
    pd.DataFrame([window]).to_csv(windows_csv, index=False)

    audit = build_review_units(
        ReviewUnitConfig(
            intervals_csv=intervals_csv,
            sequence_window_manifest_csv=windows_csv,
            output_dir=output_dir,
            include_all_retained_legacy_units=True,
        )
    )

    assert audit["review_scope"]["expected_legacy_native_units"] == 1
    assert audit["review_scope"]["reviewed_legacy_native_units"] == 1
    assert audit["review_scope"]["missing_legacy_native_units"] == 0
    full_review = pd.read_csv(output_dir / "full_review_unit_manifest.csv")
    assert full_review["review_unit_id"].tolist() == ["unit-1"]

    checker = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "check_review_unit_template_coverage.py"
    )
    command = [
        sys.executable,
        str(checker),
        "--review-unit-dir",
        str(output_dir),
        "--allow-incomplete-label-coverage",
        "--require-complete-legacy",
    ]
    passed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert passed.returncode == 0, passed.stdout + passed.stderr

    full_review.iloc[0:0].to_csv(
        output_dir / "full_review_unit_manifest.csv",
        index=False,
    )
    failed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "missing_complete_legacy_review_units=1" in failed.stdout

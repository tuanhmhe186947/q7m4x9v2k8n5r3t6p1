from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.review.interaction_calibration_sampling import (
    InteractionCalibrationSamplingConfig,
    InteractionCalibrationSamplingError,
    build_interaction_calibration_sample,
    calibration_sample_size_options,
    wilson_half_width,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []
    sources = ("cvat_tracking_xml", "legacy_recovered")
    for index in range(120):
        source = sources[index % 2]
        video = f"video_{index % 12:02d}"
        review_id = f"review_{index:04d}"
        behavior = "fight" if index % 3 else "social-nose"
        universe_records.append(
            {
                "review_unit_id": review_id,
                "temporal_unit_key": f"unit_{index:04d}",
                "behavior_label": behavior,
                "source_type": source,
                "recording_date": f"2019-12-{1 + index % 4:02d}",
                "video_key": video,
                "dataset_id": f"dataset_{source}",
                "object_track_key": f"actor_{index % 8}",
                "pig_id": f"ID_{index % 8}",
                "track_id": index % 8,
                "unit_start_frame": index * 6,
                "unit_end_frame": index * 6 + 5,
                "display_frame_indices": ",".join(
                    str(index * 6 + offset) for offset in range(6)
                ),
                "review_pig_history_display_frame_indices": "",
                "review_pig_history_available_ratio": 0.0,
                "include_in_review": index % 4 == 0,
                "review_predicate_media_or_actor_authority_risk": (
                    index % 17 == 0
                ),
                "review_social_evidence_available": index % 13 != 0,
                "evidence_quality_stratum": (
                    "target_complete_partial_history"
                    if index % 11 == 0
                    else "target_and_history_complete"
                ),
            }
        )
        diagnostic_records.append(
            {
                "review_unit_id": review_id,
                "old_include_in_review": index % 4 == 0,
                "new_include_in_review": index % 19 == 0,
                "neighborhood_evidence_available": index % 23 != 0,
                "frames_with_valid_neighbors": 6,
                "neighbor_valid_ratio": 1.0 if index % 23 else 0.0,
                "any_contact_proxy_ratio": 0.0 if index % 9 == 0 else 0.8,
                "max_concurrent_contact_proxy_count": index % 4,
                "min_edge_distance_over_unit": 0.01 * (index % 5),
                "median_min_edge_distance": 0.02 * (index % 5),
                "overlap_present_ratio": 0.0 if index % 9 == 0 else 0.5,
                "crowding_ratio": 1.0 if index % 3 else 0.5,
                "center_edge_top1_agreement_ratio": (
                    0.5 if index % 5 == 0 else 1.0
                ),
                "previous_selector_evidence_invalid": index % 29 == 0,
            }
        )
    return pd.DataFrame(universe_records), pd.DataFrame(diagnostic_records)


def _build() -> object:
    universe, diagnostic = _inputs()
    return build_interaction_calibration_sample(
        universe,
        diagnostic,
        producer_sha="a" * 40,
        input_hashes={"universe": "b" * 64, "diagnostic": "c" * 64},
        presentation_version="blind.v1",
        presentation_semantic_hash="d" * 64,
        config=InteractionCalibrationSamplingConfig(
            development_count=30,
            confirmation_count=18,
            seed=42,
        ),
    )


def test_grouped_sample_is_deterministic_and_disjoint() -> None:
    first = _build()
    second = _build()
    pd.testing.assert_frame_equal(
        first.blinded_manifest,
        second.blinded_manifest,
    )
    assert first.audit == second.audit
    assert first.audit["checker"]["group_leakage_count"] == 0
    internal = first.internal_trace
    development = set(
        internal.loc[
            internal["frozen_subset"].eq("CALIBRATION_DEVELOPMENT_SET"),
            "review_unit_id",
        ]
    )
    confirmation = set(
        internal.loc[
            internal["frozen_subset"].eq("BLINDED_CONFIRMATION_SET"),
            "review_unit_id",
        ]
    )
    assert development.isdisjoint(confirmation)


def test_public_manifest_hides_machine_hypotheses() -> None:
    result = _build()
    columns = "|".join(result.blinded_manifest.columns).casefold()
    for token in (
        "behavior",
        "candidate",
        "contact",
        "crowd",
        "date",
        "label",
        "partner",
        "reason",
        "score",
        "selector",
        "source",
        "stratum",
        "video",
    ):
        assert token not in columns


def test_required_major_strata_are_represented() -> None:
    result = _build()
    assert result.audit["valid"]
    assert not result.audit["checker"]["missing_required_strata"]


def test_confirmation_contains_no_human_decision_columns() -> None:
    result = _build()
    forbidden = {
        "reviewed_behavior",
        "review_decision",
        "manual_review_decision",
        "human_decision",
    }
    assert forbidden.isdisjoint(result.blinded_manifest.columns)
    assert forbidden.isdisjoint(result.internal_trace.columns)


def test_group_assignment_prevents_near_scene_leakage() -> None:
    result = _build()
    split = result.group_split
    assert not split["calibration_group_key"].duplicated().any()
    assert split.groupby("calibration_group_key")["frozen_subset"].nunique().max() == 1


def test_wrong_input_schema_fails_closed() -> None:
    universe, diagnostic = _inputs()
    with pytest.raises(
        InteractionCalibrationSamplingError,
        match="missing columns",
    ):
        build_interaction_calibration_sample(
            universe.drop(columns=["video_key"]),
            diagnostic,
            producer_sha="a" * 40,
            input_hashes={"universe": "b" * 64},
            presentation_version="blind.v1",
            presentation_semantic_hash="d" * 64,
        )


def test_wilson_options_are_honest() -> None:
    assert wilson_half_width(96) < 0.10
    options = calibration_sample_size_options().set_index("OPTION")
    assert (
        options.loc[
            "MINIMUM_PILOT",
            "EXPECTED_95_PERCENT_CI_HALF_WIDTH_WORST_CASE",
        ]
        > 0.10
    )
    assert (
        options.loc[
            "RECOMMENDED_CALIBRATION",
            "EXPECTED_95_PERCENT_CI_HALF_WIDTH_WORST_CASE",
        ]
        < 0.10
    )

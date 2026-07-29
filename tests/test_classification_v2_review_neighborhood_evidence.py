from __future__ import annotations

import math

import pandas as pd
import pytest

from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_SCHEMA_HASH,
    SPATIAL_SCHEMA_TOTAL_DIMENSION,
)
from pig_behavior.classification_v2.review.neighborhood_evidence import (
    REVIEW_NEIGHBORHOOD_FRAME_FIELDS,
    REVIEW_NEIGHBORHOOD_SCHEMA_HASH,
    REVIEW_NEIGHBORHOOD_UNIT_COLUMNS,
    ReviewNeighborhoodEvidenceError,
    build_review_neighborhood_evidence,
    canonical_review_neighborhood_schema_payload,
    require_review_neighborhood_evidence,
    review_neighborhood_schema_hash,
)

PRODUCER_SHA = "a" * 40
INPUT_HASHES = {"frame_features": "b" * 64}
CANONICAL_SPATIAL_HASH = (
    "18377d825ba84974e49305e46561ada81353f9ffd0f2d2526471af1c199daad4"
)


def row(
    actor: str,
    box: tuple[float, float, float, float],
    *,
    unit: str = "unit-a",
    scene: str = "scene-0",
    frame: int = 0,
    bbox_valid: bool = True,
) -> dict[str, object]:
    return {
        "temporal_unit_key": unit,
        "scene_frame_uid": scene,
        "frame_index": frame,
        "object_track_key": actor,
        "bbox_valid": bbox_valid,
        "x1": box[0],
        "y1": box[1],
        "x2": box[2],
        "y2": box[3],
        "image_width": 100.0,
        "image_height": 100.0,
    }


def build(rows: list[dict[str, object]]):
    return build_review_neighborhood_evidence(
        pd.DataFrame(rows),
        producer_sha=PRODUCER_SHA,
        input_hashes=INPUT_HASHES,
    )


@pytest.mark.parametrize(
    ("neighbor_box", "expected_edge", "expected_center"),
    [
        ((20.0, 0.0, 30.0, 10.0), 0.10, 0.20),
        (
            (20.0, 20.0, 30.0, 30.0),
            math.sqrt(0.10**2 + 0.10**2),
            math.sqrt(0.20**2 + 0.20**2),
        ),
        ((10.0, 0.0, 20.0, 10.0), 0.0, 0.10),
        (
            (5.0, 20.0, 15.0, 30.0),
            0.10,
            math.sqrt(0.05**2 + 0.20**2),
        ),
        ((5.0, 5.0, 15.0, 15.0), 0.0, math.sqrt(0.05**2 * 2)),
        ((0.0, 0.0, 10.0, 10.0), 0.0, 0.0),
        ((2.0, 2.0, 8.0, 8.0), 0.0, 0.0),
    ],
)
def test_independent_edge_and_center_golden_cases(
    neighbor_box: tuple[float, float, float, float],
    expected_edge: float,
    expected_center: float,
) -> None:
    result = build(
        [
            row("actor", (0.0, 0.0, 10.0, 10.0)),
            row("neighbor", neighbor_box, unit="unit-b"),
        ]
    )
    actor = result.frame_evidence[
        result.frame_evidence["actor_key_audit_only"].eq("actor")
    ].iloc[0]
    assert actor["min_edge_distance_n"] == pytest.approx(expected_edge)
    assert actor["min_center_distance_n"] == pytest.approx(expected_center)


def test_overlap_metrics_and_proxy_are_independently_verified() -> None:
    result = build(
        [
            row("actor", (0.0, 0.0, 10.0, 10.0)),
            row("neighbor", (5.0, 5.0, 15.0, 15.0), unit="unit-b"),
        ]
    )
    actor = result.frame_evidence[
        result.frame_evidence["actor_key_audit_only"].eq("actor")
    ].iloc[0]
    assert actor["max_pair_iou"] == pytest.approx(25.0 / 175.0)
    assert actor["max_pair_overlap_min_area"] == pytest.approx(0.25)
    assert bool(actor["any_contact_proxy"])
    assert int(actor["contact_proxy_count"]) == 1


def test_touching_edge_is_not_called_physical_contact() -> None:
    result = build(
        [
            row("actor", (0.0, 0.0, 10.0, 10.0)),
            row("neighbor", (10.0, 0.0, 20.0, 10.0), unit="unit-b"),
        ]
    )
    actor = result.frame_evidence[
        result.frame_evidence["actor_key_audit_only"].eq("actor")
    ].iloc[0]
    assert actor["min_edge_distance_n"] == 0.0
    assert not bool(actor["any_contact_proxy"])
    payload = canonical_review_neighborhood_schema_payload()
    assert payload["contact_proxy"]["physical_contact_claimed"] is False
    assert "proxy" in payload["contact_proxy"]["name"]


def test_actor_is_never_its_own_neighbor_and_missing_is_not_far() -> None:
    result = build([row("actor", (0.0, 0.0, 10.0, 10.0))])
    frame = result.frame_evidence.iloc[0]
    unit = result.unit_evidence.iloc[0]
    assert frame["valid_neighbor_count"] == 0
    assert not bool(frame["neighborhood_evidence_available"])
    assert math.isnan(float(frame["min_edge_distance_n"]))
    assert unit["frames_with_valid_neighbors"] == 0
    assert unit["neighbor_valid_ratio"] == 0.0
    assert unit["any_contact_proxy_ratio"] == 0.0
    assert math.isnan(float(unit["median_min_edge_distance"]))
    assert not bool(unit["neighborhood_evidence_available"])


def test_invalid_geometry_is_unavailable_not_observed_zero() -> None:
    result = build(
        [
            row("actor", (0.0, 0.0, 10.0, 10.0)),
            row(
                "invalid",
                (10.0, 10.0, 5.0, 5.0),
                unit="unit-b",
                bbox_valid=False,
            ),
        ]
    )
    actor = result.frame_evidence[
        result.frame_evidence["actor_key_audit_only"].eq("actor")
    ].iloc[0]
    assert not bool(actor["neighborhood_evidence_available"])
    assert math.isnan(float(actor["min_center_distance_n"]))


def test_row_order_permutation_is_deterministic() -> None:
    rows = [
        row("actor", (0.0, 0.0, 10.0, 10.0)),
        row("neighbor-b", (5.0, 5.0, 15.0, 15.0), unit="unit-b"),
        row("neighbor-a", (20.0, 0.0, 30.0, 10.0), unit="unit-c"),
    ]
    forward = build(rows)
    reverse = build(list(reversed(rows)))
    pd.testing.assert_frame_equal(
        forward.frame_evidence,
        reverse.frame_evidence,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        forward.unit_evidence,
        reverse.unit_evidence,
        check_exact=True,
    )
    assert forward.metadata == reverse.metadata


def test_behavior_label_and_review_decision_do_not_affect_evidence() -> None:
    rows = [
        row("actor", (0.0, 0.0, 10.0, 10.0)),
        row("neighbor", (20.0, 0.0, 30.0, 10.0), unit="unit-b"),
    ]
    first = pd.DataFrame(rows)
    first["behavior_label"] = ["fight", "social-nose"]
    first["manual_review_decision"] = ["accept", "correct"]
    second = first.copy()
    second["behavior_label"] = ["lying", "move"]
    second["manual_review_decision"] = ["pending", "pending"]
    result_a = build_review_neighborhood_evidence(
        first,
        producer_sha=PRODUCER_SHA,
        input_hashes=INPUT_HASHES,
    )
    result_b = build_review_neighborhood_evidence(
        second,
        producer_sha=PRODUCER_SHA,
        input_hashes=INPUT_HASHES,
    )
    pd.testing.assert_frame_equal(
        result_a.unit_evidence,
        result_b.unit_evidence,
        check_exact=True,
    )
    assert "behavior_label" not in result_a.unit_evidence
    assert "manual_review_decision" not in result_a.unit_evidence


def test_absolute_identity_is_metadata_only() -> None:
    result = build(
        [
            row("actor-999", (0.0, 0.0, 10.0, 10.0)),
            row("neighbor-123", (20.0, 0.0, 30.0, 10.0), unit="unit-b"),
        ]
    )
    numeric = result.frame_evidence[list(REVIEW_NEIGHBORHOOD_FRAME_FIELDS)]
    assert not any(
        token in column
        for column in numeric.columns
        for token in ("identity", "track", "actor_key", "partner")
    )
    assert "actor_key_audit_only" in result.frame_evidence
    assert "actor_key_audit_only" not in result.unit_evidence


def test_same_actor_keys_across_temporal_units_never_cross_units() -> None:
    rows = [
        row(
            "actor",
            (0.0, 0.0, 10.0, 10.0),
            unit="unit-a",
            scene="scene-a",
            frame=0,
        ),
        row(
            "neighbor",
            (20.0, 0.0, 30.0, 10.0),
            unit="unit-b",
            scene="scene-a",
            frame=0,
        ),
        row(
            "actor",
            (0.0, 0.0, 10.0, 10.0),
            unit="unit-c",
            scene="scene-b",
            frame=1,
        ),
    ]
    result = build(rows)
    units = result.unit_evidence.set_index("temporal_unit_key")
    assert bool(units.loc["unit-a", "neighborhood_evidence_available"])
    assert not bool(units.loc["unit-c", "neighborhood_evidence_available"])
    assert math.isnan(float(units.loc["unit-c", "min_edge_distance_over_unit"]))
    assert (
        result.metadata["temporal_pair_dynamics"]
        == "none_in_v1"
    )


def test_more_than_seven_neighbors_fails_current_authority() -> None:
    rows = [row("actor", (0.0, 0.0, 10.0, 10.0))]
    for index in range(8):
        rows.append(
            row(
                f"neighbor-{index}",
                (20.0 + index, 0.0, 30.0 + index, 10.0),
                unit=f"unit-{index}",
            )
        )
    with pytest.raises(
        ReviewNeighborhoodEvidenceError,
        match="more than seven",
    ):
        build(rows)


def test_checker_fails_reordered_or_shrunk_schema() -> None:
    result = build(
        [
            row("actor", (0.0, 0.0, 10.0, 10.0)),
            row("neighbor", (20.0, 0.0, 30.0, 10.0), unit="unit-b"),
        ]
    )
    reordered = result.unit_evidence[
        list(reversed(REVIEW_NEIGHBORHOOD_UNIT_COLUMNS))
    ]
    with pytest.raises(
        ReviewNeighborhoodEvidenceError,
        match="ordered_unit_columns_mismatch",
    ):
        require_review_neighborhood_evidence(reordered, result.metadata)
    shrunk = result.unit_evidence.drop(columns=["crowding_ratio"])
    with pytest.raises(ReviewNeighborhoodEvidenceError):
        require_review_neighborhood_evidence(shrunk, result.metadata)


def test_checker_rejects_missingness_as_observed_zero_distance() -> None:
    result = build([row("actor", (0.0, 0.0, 10.0, 10.0))])
    invalid = result.unit_evidence.copy()
    invalid["min_edge_distance_over_unit"] = 0.0
    with pytest.raises(
        ReviewNeighborhoodEvidenceError,
        match="unavailable_distance_not_nan",
    ):
        require_review_neighborhood_evidence(invalid, result.metadata)


def test_identical_inputs_have_identical_schema_and_metadata_hashes() -> None:
    rows = [
        row("actor", (0.0, 0.0, 10.0, 10.0)),
        row("neighbor", (20.0, 0.0, 30.0, 10.0), unit="unit-b"),
    ]
    first = build(rows)
    second = build(rows)
    assert REVIEW_NEIGHBORHOOD_SCHEMA_HASH == review_neighborhood_schema_hash()
    assert first.metadata == second.metadata
    pd.testing.assert_frame_equal(
        first.unit_evidence,
        second.unit_evidence,
        check_exact=True,
    )


def test_review_only_contract_cannot_enter_model_x_or_select_candidates() -> None:
    payload = canonical_review_neighborhood_schema_payload()
    assert payload["model_x_usage"] == "forbidden"
    assert payload["candidate_selection_binding"] == "none_in_v1"
    assert payload["behavior_label_dependency"] == "forbidden"
    assert payload["review_decision_dependency"] == "forbidden"


def test_historical_spatial_46d_authority_is_unchanged() -> None:
    assert SPATIAL_SCHEMA_TOTAL_DIMENSION == 46
    assert SPATIAL_SCHEMA_HASH == CANONICAL_SPATIAL_HASH
    assert review_neighborhood_schema_hash() != SPATIAL_SCHEMA_HASH


def test_invalid_hash_bindings_fail_closed() -> None:
    frame = pd.DataFrame([row("actor", (0.0, 0.0, 10.0, 10.0))])
    with pytest.raises(
        ReviewNeighborhoodEvidenceError,
        match="producer_sha",
    ):
        build_review_neighborhood_evidence(
            frame,
            producer_sha="short",
            input_hashes=INPUT_HASHES,
        )
    with pytest.raises(
        ReviewNeighborhoodEvidenceError,
        match="input hash",
    ):
        build_review_neighborhood_evidence(
            frame,
            producer_sha=PRODUCER_SHA,
            input_hashes={"frame_features": "bad"},
        )

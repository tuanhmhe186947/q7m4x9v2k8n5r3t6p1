"""Synthetic contract tests for the opt-in H1-r2 implementation."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pig_behavior.tracking.association import (
    apply_h1_r2_hidden_owner_preference,
    match_and_update_tracks,
)
from pig_behavior.tracking.config import TrackingConfig, validate_config
from pig_behavior.tracking.constants import H1_R2_TELEMETRY_KEYS
from pig_behavior.tracking.owner_preference import (
    OWNER_PREFERENCE_THRESHOLD,
    OWNER_PREFERENCE_WEIGHTS,
    OwnerPreferenceFeatures,
    appearance_similarity,
    build_owner_preference_features,
    decide_owner_preference,
    normalized_center_similarity,
    overlap_similarity,
    owner_preference_score,
    scale_similarity,
    track_freshness,
)
from pig_behavior.tracking.profiles.realtime import (
    PRESENTATION_PROFILES,
    REALTIME_FAST_CONFIG,
    REALTIME_FAST_H1_R2_CONFIG,
)
from pig_behavior.tracking.schemas import (
    Detection,
    FixedTrack,
    TrackingRuntimeState,
)
from pig_behavior.tracking.telemetry import (
    resolve_output_timing_contract,
    summarize_tracking_telemetry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = (
    PROJECT_ROOT / "docs/tracking/h1_r2/H1_R2_GOLDEN_CASES.yaml"
)


def _features(values: list[float]) -> OwnerPreferenceFeatures:
    return OwnerPreferenceFeatures(*map(float, values))


def _eligible_features(
    *,
    overlap: float = 0.8,
    center: float = 0.8,
    appearance: float = 0.8,
    freshness: float = 0.8,
) -> OwnerPreferenceFeatures:
    return OwnerPreferenceFeatures(
        overlap_similarity=overlap,
        normalized_center_similarity=center,
        scale_similarity=0.8,
        appearance_similarity=appearance,
        motion_consistency=0.8,
        track_freshness=freshness,
        appearance_available=1.0,
        motion_available=1.0,
    )


def _track(
    fixed_id: int,
    box: list[float],
    hist: list[float],
    *,
    missed: int,
) -> FixedTrack:
    track = FixedTrack(
        fixed_id=fixed_id,
        last_box=np.asarray(box, dtype=np.float32),
    )
    track.hist_bank.append(np.asarray(hist, dtype=np.float32))
    track.ever_detected = True
    track.hits = 2
    track.missed = missed
    track.last_source = "predicted" if missed else "detected"
    return track


def _strong_owner_fixture() -> tuple[
    np.ndarray,
    list[FixedTrack],
    list[int],
    list[Detection],
    list[FixedTrack],
    TrackingConfig,
]:
    hidden = _track(1, [0, 0, 20, 20], [1.0, 0.0], missed=1)
    visible = _track(2, [30, 0, 50, 20], [0.0, 1.0], missed=0)
    detection = Detection(
        box=np.asarray([0, 0, 20, 20], dtype=np.float32),
        score=0.9,
        raw_id=None,
        class_id=0,
        hist=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    cfg = TrackingConfig(mode="realtime", **REALTIME_FAST_H1_R2_CONFIG)
    return (
        np.asarray([[0.1]], dtype=np.float32),
        [visible],
        [0],
        [detection],
        [hidden],
        cfg,
    )


def test_all_ten_frozen_golden_cases() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert len(golden["cases"]) == 10
    assert golden["weights"] == OWNER_PREFERENCE_WEIGHTS
    for case in golden["cases"]:
        hidden = _features(case["hidden"])
        visible = _features(case["visible"])

        assert owner_preference_score(hidden, visible) == pytest.approx(
            case["expected_score"],
            abs=golden["tolerance"],
        )


def test_feature_formulas_and_uniform_scale_invariance() -> None:
    small_track = np.asarray([10, 10, 30, 30], dtype=np.float32)
    small_detection = np.asarray([12, 10, 32, 30], dtype=np.float32)
    large_track = small_track * 10.0
    large_detection = small_detection * 10.0

    assert overlap_similarity(small_track, small_detection) == pytest.approx(
        overlap_similarity(large_track, large_detection)
    )
    assert normalized_center_similarity(
        small_track,
        small_detection,
    ) == pytest.approx(
        normalized_center_similarity(large_track, large_detection)
    )
    assert scale_similarity(small_track, small_detection) == pytest.approx(
        scale_similarity(large_track, large_detection)
    )


def test_appearance_missingness_is_neutral_with_zero_mask() -> None:
    quality, available = appearance_similarity(None, np.asarray([1.0, 0.0]))

    assert quality == 0.5
    assert available == 0.0


def test_hidden_visible_swap_is_complementary() -> None:
    hidden = _eligible_features(overlap=0.9, center=0.9)
    visible = _eligible_features(overlap=0.3, center=0.4)

    forward = owner_preference_score(hidden, visible)
    reverse = owner_preference_score(visible, hidden)

    assert forward + reverse == pytest.approx(1.0)


def test_identical_evidence_is_neutral_and_abstains() -> None:
    evidence = _eligible_features()

    decision = decide_owner_preference(
        evidence,
        evidence,
        detection_confidence=0.9,
        hidden_detection_opportunities=1,
        visible_detection_opportunities=0,
    )

    assert decision.owner_preference_score == 0.5
    assert decision.apply is False
    assert decision.reason == "tie_or_margin"


@pytest.mark.parametrize(
    ("field", "weaker", "stronger"),
    [
        ("overlap_similarity", 0.5, 0.9),
        ("normalized_center_similarity", 0.5, 0.9),
        ("appearance_similarity", 0.5, 0.9),
    ],
)
def test_stronger_hidden_evidence_never_lowers_preference(
    field: str,
    weaker: float,
    stronger: float,
) -> None:
    visible = _eligible_features(overlap=0.5, center=0.5, appearance=0.5)
    hidden = _eligible_features(overlap=0.5, center=0.5, appearance=0.5)

    weak_score = owner_preference_score(
        replace(hidden, **{field: weaker}),
        visible,
    )
    strong_score = owner_preference_score(
        replace(hidden, **{field: stronger}),
        visible,
    )

    assert strong_score >= weak_score


def test_hidden_age_cannot_improve_preference() -> None:
    visible = _eligible_features(freshness=1.0)
    recent = _eligible_features(freshness=0.8)
    stale = replace(recent, track_freshness=0.2)

    assert owner_preference_score(stale, visible) <= owner_preference_score(
        recent,
        visible,
    )
    assert track_freshness(4) < track_freshness(1)


def test_missing_evidence_never_becomes_strong_positive() -> None:
    hidden = replace(
        _eligible_features(),
        appearance_similarity=0.5,
        appearance_available=0.0,
        motion_consistency=0.5,
        motion_available=0.0,
    )
    visible = _eligible_features()

    decision = decide_owner_preference(
        hidden,
        visible,
        detection_confidence=0.9,
        hidden_detection_opportunities=1,
        visible_detection_opportunities=0,
    )

    assert decision.apply is False
    assert decision.reason == "missing_evidence"


def test_exact_threshold_applies_deterministically() -> None:
    hidden = _eligible_features(overlap=0.9)
    visible = _eligible_features(overlap=0.1)

    decision = decide_owner_preference(
        hidden,
        visible,
        detection_confidence=0.9,
        hidden_detection_opportunities=1,
        visible_detection_opportunities=0,
    )

    assert decision.owner_preference_score == pytest.approx(
        OWNER_PREFERENCE_THRESHOLD
    )
    assert decision.apply is True


def test_first_observation_has_unavailable_motion() -> None:
    box = np.asarray([0, 0, 20, 20], dtype=np.float32)
    features = build_owner_preference_features(
        reference_box=box,
        detection_box=box,
        track_descriptor=None,
        detection_descriptor=np.asarray([1.0, 0.0]),
        predicted_box=None,
        motion_is_available=False,
        detection_opportunities_since_confirmed=0,
    )

    assert features.motion_consistency == 0.5
    assert features.motion_available == 0.0
    assert features.appearance_similarity == 0.5
    assert features.appearance_available == 0.0


def test_all_scores_are_finite_and_bounded() -> None:
    values = np.linspace(0.0, 1.0, 8)
    for offset in range(8):
        hidden = OwnerPreferenceFeatures(*np.roll(values, offset))
        visible = OwnerPreferenceFeatures(*np.roll(values, -offset))
        score = owner_preference_score(hidden, visible)

        assert math.isfinite(score)
        assert 0.0 <= score <= 1.0


def test_algorithm_is_independent_of_telemetry_and_runtime_none() -> None:
    fixture = _strong_owner_fixture()
    costs_without = fixture[0].copy()
    costs_with = fixture[0].copy()
    reserved_without: dict[int, int] = {}
    reserved_with: dict[int, int] = {}
    runtime = TrackingRuntimeState()

    apply_h1_r2_hidden_owner_preference(
        costs_without,
        deepcopy(fixture[1]),
        fixture[2],
        deepcopy(fixture[3]),
        deepcopy(fixture[4]),
        set(),
        100,
        100,
        fixture[5],
        None,
        1,
        "visible_high_conf",
        reserved_without,
    )
    apply_h1_r2_hidden_owner_preference(
        costs_with,
        deepcopy(fixture[1]),
        fixture[2],
        deepcopy(fixture[3]),
        deepcopy(fixture[4]),
        set(),
        100,
        100,
        fixture[5],
        runtime,
        1,
        "visible_high_conf",
        reserved_with,
    )

    np.testing.assert_array_equal(costs_with, costs_without)
    assert reserved_with == reserved_without == {0: 1}
    summary = summarize_tracking_telemetry(runtime)
    assert summary["h1_r2_owner_preference_applied"] == 1
    assert set(H1_R2_TELEMETRY_KEYS).issubset(summary)


def test_reserved_column_does_not_displace_other_visible_assignment() -> None:
    fixture = _strong_owner_fixture()
    second_visible = _track(
        3,
        [60, 0, 80, 20],
        [0.0, 1.0],
        missed=0,
    )
    second_detection = Detection(
        box=np.asarray([60, 0, 80, 20], dtype=np.float32),
        score=0.9,
        raw_id=None,
        class_id=0,
        hist=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    costs = np.asarray(
        [
            [0.1, 0.8],
            [0.8, 0.1],
        ],
        dtype=np.float32,
    )
    reserved: dict[int, int] = {}

    rows, cols = apply_h1_r2_hidden_owner_preference(
        costs,
        [fixture[1][0], second_visible],
        [0, 1],
        [fixture[3][0], second_detection],
        fixture[4],
        set(),
        100,
        100,
        fixture[5],
        None,
        1,
        "visible_high_conf",
        reserved,
    )

    assert reserved == {0: 1}
    assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == [(1, 1)]


def test_synthetic_association_reacquires_only_the_reserved_hidden_owner() -> None:
    cfg = TrackingConfig(mode="realtime", **REALTIME_FAST_H1_R2_CONFIG)
    hidden = _track(1, [0, 0, 20, 20], [1.0, 0.0], missed=1)
    visible = _track(2, [5, 0, 25, 20], [0.0, 1.0], missed=0)
    tracks = {
        1: hidden,
        2: visible,
        **{
            fixed_id: FixedTrack(
                fixed_id=fixed_id,
                last_box=np.asarray(
                    [fixed_id * 10, 40, fixed_id * 10 + 8, 48],
                    dtype=np.float32,
                ),
            )
            for fixed_id in range(3, 9)
        },
    }
    detection = Detection(
        box=np.asarray([0, 0, 20, 20], dtype=np.float32),
        score=0.9,
        raw_id=None,
        class_id=0,
        hist=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    runtime = TrackingRuntimeState()

    match_and_update_tracks(
        tracks,
        [detection],
        frame,
        None,
        cfg,
        runtime,
        frame_index=2,
    )

    assert tracks[1].last_source == "detected"
    assert tracks[1].missed == 0
    assert tracks[2].last_source != "detected"
    assert runtime.telemetry["h1_r2_owner_preference_applied"] == 1
    assert runtime.telemetry["h1_r2_reacquisition_observed"] == 1


def test_realtime_fast_path_is_unchanged_when_h1_r2_is_disabled() -> None:
    fixture = _strong_owner_fixture()
    costs = fixture[0].copy()
    reserved: dict[int, int] = {}
    cfg = TrackingConfig(mode="realtime", **REALTIME_FAST_CONFIG)

    apply_h1_r2_hidden_owner_preference(
        costs,
        fixture[1],
        fixture[2],
        fixture[3],
        fixture[4],
        set(),
        100,
        100,
        cfg,
        TrackingRuntimeState(),
        1,
        "visible_high_conf",
        reserved,
    )

    np.testing.assert_array_equal(costs, fixture[0])
    assert reserved == {}


def test_profile_is_opt_in_causal_and_preserves_realtime_fast() -> None:
    candidate_only = {
        key: value
        for key, value in REALTIME_FAST_H1_R2_CONFIG.items()
        if key not in REALTIME_FAST_CONFIG
    }
    inherited = {
        key: REALTIME_FAST_H1_R2_CONFIG[key] for key in REALTIME_FAST_CONFIG
    }
    cfg = TrackingConfig(mode="realtime", **REALTIME_FAST_H1_R2_CONFIG)

    assert inherited == REALTIME_FAST_CONFIG
    assert candidate_only == {"h1_r2_owner_preference": True}
    assert REALTIME_FAST_CONFIG.get("h1_r2_owner_preference") is None
    assert PRESENTATION_PROFILES["realtime"]["eval_config"] == "realtime_fast"
    assert resolve_output_timing_contract(cfg) == ("causal_framewise", 0)


def test_invalid_h1_r2_combinations_fail_closed() -> None:
    invalid = TrackingConfig(
        mode="realtime",
        **{
            **REALTIME_FAST_H1_R2_CONFIG,
            "causal_hidden_detection_reservation": True,
        },
    )

    with pytest.raises(ValueError, match="H1-r1"):
        validate_config(invalid)


def test_prefix_result_cannot_depend_on_unprovided_future_state() -> None:
    fixture = _strong_owner_fixture()

    def decide(frame_index: int) -> tuple[np.ndarray, dict[int, int]]:
        costs = fixture[0].copy()
        reserved: dict[int, int] = {}
        apply_h1_r2_hidden_owner_preference(
            costs,
            deepcopy(fixture[1]),
            fixture[2],
            deepcopy(fixture[3]),
            deepcopy(fixture[4]),
            set(),
            100,
            100,
            fixture[5],
            None,
            frame_index,
            "visible_high_conf",
            reserved,
        )
        return costs, reserved

    prefix_costs, prefix_reserved = decide(1)
    repeated_costs, repeated_reserved = decide(1)

    np.testing.assert_array_equal(prefix_costs, repeated_costs)
    assert prefix_reserved == repeated_reserved

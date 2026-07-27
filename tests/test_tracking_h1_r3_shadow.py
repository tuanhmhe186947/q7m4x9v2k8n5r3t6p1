"""Focused tests for the reservation-disabled H1-r3 shadow observer."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from pig_behavior.tracking.association import observe_h1_r3_shadow_pairs
from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.constants import H1_R3_SHADOW_TELEMETRY_KEYS
from pig_behavior.tracking.h1_r3_shadow import (
    H1_R3_SCORE_THRESHOLD,
    H1_R3_SUPPORT_MARGIN,
    H1R3Evidence,
    decide_h1_r3_shadow,
)
from pig_behavior.tracking.profiles.realtime import REALTIME_FAST_CONFIG
from pig_behavior.tracking.schemas import (
    Detection,
    FixedTrack,
    TrackingRuntimeState,
)
from pig_behavior.tracking.telemetry import summarize_tracking_telemetry


def _evidence(
    *,
    overlap: float,
    freshness: float,
    appearance: float | None = 0.8,
    motion: float | None = 0.8,
) -> H1R3Evidence:
    return H1R3Evidence(
        overlap_similarity=overlap,
        normalized_center_similarity=overlap,
        scale_similarity=1.0,
        appearance_similarity=appearance,
        motion_consistency=motion,
        track_freshness=freshness,
        appearance_available=int(appearance is not None),
        motion_available=int(motion is not None),
        overlap_valid=1,
        freshness_valid=1,
        appearance_quality=1.0 if appearance is not None else 0.0,
        motion_quality=0.5 if motion is not None else 0.0,
        reference_source="causal_prediction",
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
    track.hits = 3
    track.missed = missed
    track.last_source = "predicted" if missed else "detected"
    return track


def _detection() -> Detection:
    return Detection(
        box=np.asarray([0, 0, 20, 20], dtype=np.float32),
        score=0.9,
        raw_id=None,
        class_id=0,
        hist=np.asarray([1.0, 0.0], dtype=np.float32),
    )


def test_frozen_threshold_and_margin_are_one_gate() -> None:
    assert H1_R3_SCORE_THRESHOLD == 0.625
    assert H1_R3_SUPPORT_MARGIN == 0.25
    assert H1_R3_SCORE_THRESHOLD == 0.5 + 0.5 * H1_R3_SUPPORT_MARGIN


def test_strong_nonperfect_hidden_evidence_would_activate() -> None:
    decision = decide_h1_r3_shadow(
        _evidence(overlap=0.85, freshness=0.875),
        _evidence(overlap=0.30, freshness=1.0),
        detection_confidence=0.9,
    )

    assert decision.core_eligible
    assert decision.would_activate
    assert decision.owner_preference_lower_bound is not None
    assert decision.owner_preference_lower_bound >= H1_R3_SCORE_THRESHOLD


def test_identical_evidence_is_neutral_and_abstains() -> None:
    evidence = _evidence(overlap=0.7, freshness=0.75)
    decision = decide_h1_r3_shadow(
        evidence,
        evidence,
        detection_confidence=0.9,
    )

    assert decision.relative_owner_support_lower == pytest.approx(0.0)
    assert decision.owner_preference_lower_bound == pytest.approx(0.5)
    assert not decision.would_activate


def test_swapping_sides_reverses_observed_support() -> None:
    hidden = _evidence(overlap=0.8, freshness=0.75)
    visible = _evidence(overlap=0.3, freshness=1.0)
    forward = decide_h1_r3_shadow(
        hidden,
        visible,
        detection_confidence=0.9,
    )
    reverse = decide_h1_r3_shadow(
        visible,
        hidden,
        detection_confidence=0.9,
    )

    assert forward.relative_owner_support_lower == pytest.approx(
        -reverse.relative_owner_support_upper
    )
    assert forward.relative_owner_support_upper == pytest.approx(
        -reverse.relative_owner_support_lower
    )


def test_missing_optional_evidence_uses_conservative_bounds() -> None:
    observed = decide_h1_r3_shadow(
        _evidence(overlap=0.8, freshness=1.0),
        _evidence(overlap=0.2, freshness=1.0),
        detection_confidence=0.9,
    )
    missing = decide_h1_r3_shadow(
        _evidence(
            overlap=0.8,
            freshness=1.0,
            appearance=None,
            motion=None,
        ),
        _evidence(overlap=0.2, freshness=1.0),
        detection_confidence=0.9,
    )

    assert missing.appearance_lower == -0.15
    assert missing.motion_lower == -0.10
    assert (
        missing.owner_preference_lower_bound
        <= observed.owner_preference_lower_bound
    )


def test_hidden_age_alone_cannot_improve_support() -> None:
    visible = _evidence(overlap=0.4, freshness=1.0)
    recent = decide_h1_r3_shadow(
        _evidence(overlap=0.8, freshness=0.875),
        visible,
        detection_confidence=0.9,
    )
    stale = decide_h1_r3_shadow(
        _evidence(overlap=0.8, freshness=0.125),
        visible,
        detection_confidence=0.9,
    )

    assert (
        stale.owner_preference_lower_bound
        <= recent.owner_preference_lower_bound
    )


def test_observer_returns_no_command_and_mutates_no_association_state() -> None:
    hidden = _track(1, [0, 0, 20, 20], [1.0, 0.0], missed=1)
    visible = _track(2, [30, 0, 50, 20], [0.0, 1.0], missed=0)
    detection = _detection()
    costs = np.asarray([[0.1]], dtype=np.float32)
    original_costs = costs.copy()
    tracks_before = deepcopy([visible, hidden])
    runtime = TrackingRuntimeState(h1_r3_shadow_enabled=True)
    cfg = TrackingConfig(mode="realtime", **REALTIME_FAST_CONFIG)

    result = observe_h1_r3_shadow_pairs(
        costs,
        [visible],
        [0],
        [detection],
        [hidden],
        set(),
        100,
        100,
        cfg,
        runtime,
        2,
        np.asarray([0]),
        np.asarray([0]),
    )

    assert result is None
    np.testing.assert_array_equal(costs, original_costs)
    for before, after in zip(
        tracks_before,
        [visible, hidden],
        strict=True,
    ):
        np.testing.assert_array_equal(before.last_box, after.last_box)
        assert before.missed == after.missed
        assert before.last_source == after.last_source
        assert len(before.hist_bank) == len(after.hist_bank)
        for before_hist, after_hist in zip(
            before.hist_bank,
            after.hist_bank,
            strict=True,
        ):
            np.testing.assert_array_equal(before_hist, after_hist)
    assert len(runtime.h1_r3_shadow_candidate_rows) == 1


def test_disabled_runtime_and_runtime_none_are_no_ops() -> None:
    hidden = _track(1, [0, 0, 20, 20], [1.0, 0.0], missed=1)
    visible = _track(2, [30, 0, 50, 20], [0.0, 1.0], missed=0)
    cfg = TrackingConfig(mode="realtime", **REALTIME_FAST_CONFIG)
    arguments = (
        np.asarray([[0.1]], dtype=np.float32),
        [visible],
        [0],
        [_detection()],
        [hidden],
        set(),
        100,
        100,
        cfg,
    )
    disabled = TrackingRuntimeState()

    observe_h1_r3_shadow_pairs(
        *arguments,
        disabled,
        2,
        np.asarray([0]),
        np.asarray([0]),
    )
    observe_h1_r3_shadow_pairs(
        *arguments,
        None,
        2,
        np.asarray([0]),
        np.asarray([0]),
    )

    assert disabled.h1_r3_shadow_candidate_rows == []
    assert all(
        int(disabled.telemetry.get(key, 0)) == 0
        for key in H1_R3_SHADOW_TELEMETRY_KEYS
    )


def test_shadow_counters_survive_canonical_summarization() -> None:
    runtime = TrackingRuntimeState()
    for index, key in enumerate(H1_R3_SHADOW_TELEMETRY_KEYS, start=1):
        runtime.telemetry[key] = index

    summary = summarize_tracking_telemetry(runtime)

    for index, key in enumerate(H1_R3_SHADOW_TELEMETRY_KEYS, start=1):
        assert summary[key] == index

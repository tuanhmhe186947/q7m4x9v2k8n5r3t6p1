from __future__ import annotations

import copy
import json
import pickle
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pig_behavior.tracking.h2_cdsp_shadow import (
    H2_SHADOW_COUNTERS,
    evaluate_h2_cdsp_formula,
    observe_h2_cdsp_shadow,
)
from pig_behavior.tracking.schemas import FixedTrack, TrackingRuntimeState
from pig_behavior.tracking.telemetry import summarize_tracking_telemetry

GOLDEN_PATH = (
    Path(__file__).parents[1]
    / "docs"
    / "tracking"
    / "h2_cdsp"
    / "H2_CDSP_GOLDEN_CASES.yaml"
)


def _formula_for_case(case: dict[str, object]):
    values = case["input"]
    assert isinstance(values, dict)
    initial_confidence = values.get("initial_confidence", 1.0)
    if initial_confidence == "NaN":
        initial_confidence = float("nan")
    return evaluate_h2_cdsp_formula(
        age=int(values["age"]),
        initial_confidence=float(initial_confidence),
        initial_uncertainty=float(values["initial_uncertainty"]),
        appearance_available=bool(values["appearance_available"]),
        appearance_quality=float(values["appearance_quality"]),
        motion_available=bool(values["motion_available"]),
        motion_quality=float(values["motion_quality"]),
        occlusion_support=bool(values["occlusion_support"]),
        boundary_seen=bool(values.get("boundary_seen", False)),
        trusted_match=bool(values.get("trusted_match", False)),
        sequence_changed=bool(values.get("sequence_changed", False)),
        baseline_terminated=bool(values.get("baseline_terminated", False)),
    )


@pytest.mark.parametrize(
    "case",
    json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"],
    ids=lambda case: case["case_id"],
)
def test_all_frozen_golden_cases(case: dict[str, object]) -> None:
    result = _formula_for_case(case)
    expected = case["expected"]
    assert isinstance(expected, dict)
    assert result.state == expected["state"]
    assert result.usable is expected["usable"]
    assert result.direct_assignment is False
    assert result.reserves_detection is False
    for name in (
        "confidence",
        "uncertainty",
        "motion_reliability",
        "appearance_reliability",
    ):
        assert getattr(result, name) == pytest.approx(expected[name], abs=1e-9)


def _trusted_track() -> FixedTrack:
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([10.0, 20.0, 30.0, 50.0], dtype=np.float32),
    )
    track.state = "VISIBLE"
    track.last_source = "detected"
    track.last_ambiguous = False
    track.last_score = 0.9
    track.hits = 4
    track.ever_detected = True
    track.hist_bank.append(np.array([0.2, 0.8], dtype=np.float32))
    track.reliable_center_history.append(np.array([19.0, 35.0]))
    track.reliable_center_history.append(np.array([20.0, 35.0]))
    track.reliable_velocity_xy = np.array([1.0, 0.0], dtype=np.float32)
    return track


def _track_signature(track: FixedTrack) -> dict[str, object]:
    values = {item.name: getattr(track, item.name) for item in fields(track)}
    signature = copy.deepcopy(
        {
            key: value
            for key, value in values.items()
            if not isinstance(value, np.ndarray)
        }
    )
    signature["arrays"] = {
        key: value.copy()
        for key, value in values.items()
        if isinstance(value, np.ndarray)
    }
    return signature


def test_observer_is_disabled_by_default_and_does_not_mutate_track() -> None:
    track = _trusted_track()
    runtime = TrackingRuntimeState()
    before = _track_signature(track)
    before_all_fields = pickle.dumps(track)
    result = observe_h2_cdsp_shadow(
        {1: track},
        frame_index=10,
        cfg=SimpleNamespace(track_high_conf=0.5),
        runtime=runtime,
        sequence_token=object(),
    )
    assert result is None
    assert runtime.h2_shadow_transition_rows == []
    after = _track_signature(track)
    assert before.keys() == after.keys()
    assert before["arrays"].keys() == after["arrays"].keys()
    for key, value in before["arrays"].items():
        np.testing.assert_array_equal(value, after["arrays"][key])
    assert pickle.dumps(track) == before_all_fields


def test_observer_returns_diagnostics_only_and_preserves_track_fields() -> None:
    track = _trusted_track()
    runtime = TrackingRuntimeState(h2_shadow_enabled=True)
    token = object()
    before_box = track.last_box.copy()
    before_all_fields = pickle.dumps(track)
    observe_h2_cdsp_shadow(
        {1: track},
        frame_index=10,
        cfg=SimpleNamespace(track_high_conf=0.5),
        runtime=runtime,
        sequence_token=token,
    )
    assert pickle.dumps(track) == before_all_fields
    track.state = "MISSING"
    track.last_source = "predicted"
    track.missed = 1
    before_dropout = track.last_box.copy()
    observe_h2_cdsp_shadow(
        {1: track},
        frame_index=11,
        cfg=SimpleNamespace(track_high_conf=0.5),
        runtime=runtime,
        sequence_token=token,
    )
    np.testing.assert_array_equal(before_box, before_dropout)
    np.testing.assert_array_equal(track.last_box, before_dropout)
    row = runtime.h2_shadow_transition_rows[-1]
    assert row["preservation_state"] == "DROPOUT_GRACE"
    assert row["direct_assignment"] is False
    assert row["reserves_detection"] is False
    assert runtime.telemetry["h2_shadow_dropout_entries"] == 1


def test_scheduled_skip_does_not_refresh_trusted_snapshot() -> None:
    track = _trusted_track()
    runtime = TrackingRuntimeState(h2_shadow_enabled=True)
    token = object()
    observe_h2_cdsp_shadow(
        {1: track},
        frame_index=10,
        cfg=SimpleNamespace(track_high_conf=0.5),
        runtime=runtime,
        sequence_token=token,
        detector_frame=True,
    )
    observe_h2_cdsp_shadow(
        {1: track},
        frame_index=11,
        cfg=SimpleNamespace(track_high_conf=0.5),
        runtime=runtime,
        sequence_token=token,
        detector_frame=False,
    )
    state = runtime.h2_shadow_track_states[1]
    assert state.snapshot.last_trusted_frame_index == 10
    assert state.state == "DROPOUT_GRACE"
    assert runtime.telemetry["h2_shadow_visible_confirmed_tracks"] == 1


def test_causal_reentry_records_survival_without_assignment_command() -> None:
    track = _trusted_track()
    runtime = TrackingRuntimeState(h2_shadow_enabled=True)
    token = object()
    observe_h2_cdsp_shadow(
        {1: track},
        frame_index=10,
        cfg=SimpleNamespace(track_high_conf=0.5),
        runtime=runtime,
        sequence_token=token,
        detector_frame=True,
    )
    observe_h2_cdsp_shadow(
        {1: track},
        frame_index=11,
        cfg=SimpleNamespace(track_high_conf=0.5),
        runtime=runtime,
        sequence_token=token,
        detector_frame=False,
    )
    observe_h2_cdsp_shadow(
        {1: track},
        frame_index=12,
        cfg=SimpleNamespace(track_high_conf=0.5),
        runtime=runtime,
        sequence_token=token,
        detector_frame=True,
    )
    row = runtime.h2_shadow_transition_rows[-1]
    assert row["reentry_frame"] == 12
    assert row["preserved_state_available_at_reentry"] is True
    assert row["extra_usable_evidence_relative_to_baseline"] is False
    assert row["direct_assignment"] is False
    assert row["reserves_detection"] is False
    assert runtime.telemetry["h2_shadow_reentry_opportunities"] == 1
    assert runtime.telemetry["h2_shadow_states_surviving_to_reentry"] == 1
    assert runtime.telemetry["h2_shadow_extra_usable_state_at_reentry"] == 0


def test_h2_telemetry_is_canonical_and_fails_if_missing() -> None:
    runtime = TrackingRuntimeState(h2_shadow_enabled=True)
    summary = summarize_tracking_telemetry(runtime)
    assert all(key in summary for key in H2_SHADOW_COUNTERS)
    del runtime.telemetry["h2_shadow_stage_calls"]
    with pytest.raises(KeyError, match="missing canonical H2 shadow telemetry"):
        summarize_tracking_telemetry(runtime)


def test_missing_optional_evidence_never_increases_confidence() -> None:
    present = evaluate_h2_cdsp_formula(
        age=3,
        appearance_available=True,
        appearance_quality=0.8,
        motion_available=True,
        motion_quality=0.8,
        occlusion_support=True,
    )
    absent = evaluate_h2_cdsp_formula(
        age=3,
        appearance_available=False,
        appearance_quality=0.0,
        motion_available=False,
        motion_quality=0.0,
        occlusion_support=True,
    )
    assert absent.confidence == present.confidence
    assert absent.uncertainty >= present.uncertainty

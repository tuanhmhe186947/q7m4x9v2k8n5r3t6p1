"""Canonical temporal-view semantics, validated against fixtures only."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pig_behavior.classification_v2.temporal_views.builder_contract import (
    TemporalWindowSample,
    WindowContractError,
    builder_contract_summary,
    deterministic_window_id,
    expected_frame_indices,
    require_window,
    validate_window,
)
from pig_behavior.classification_v2.temporal_views.registry import (
    CAUSAL_HISTORY_VIEWS,
    HISTORICAL_C6_BURST_OFFSETS,
    S6_AT_16_BURST_OFFSETS,
    S6_AT_16_PAIR_DELTAS,
    TARGET_CONTIGUOUS_VIEWS,
    TEMPORAL_VIEW_NAMES,
    temporal_view_registry_contract,
    temporal_view_spec,
    views_are_distinct,
)

FPS = 30.0


def _window(
    view_name: str,
    *,
    endpoint: int = 100,
    burst_start: int | None = None,
    target_labels: tuple[str, ...] | None = None,
    history_labels: tuple[str, ...] | None = None,
) -> TemporalWindowSample:
    spec = temporal_view_spec(view_name)
    target, history = expected_frame_indices(spec, endpoint)
    labels = target_labels or tuple(["move"] * len(target))
    return TemporalWindowSample(
        view_name=view_name,
        actor_authority="cvat|dataset|video|track7",
        split_group_id="date=291119",
        endpoint_frame_index=endpoint,
        target_frame_indices=target,
        target_timestamps_sec=tuple(index / FPS for index in target),
        target_labels=labels,
        history_frame_indices=history,
        history_timestamps_sec=tuple(index / FPS for index in history),
        history_labels=history_labels or tuple(["lying"] * len(history)),
        target_actor_authority=tuple(["cvat|dataset|video|track7"] * len(target)),
        target_split_group_ids=tuple(["date=291119"] * len(target)),
        burst_start_frame_index=burst_start,
        available_history_frames=len(history),
        model_input_fields=("cx_n", "cy_n", "speed_n_per_second"),
    )


def test_registry_exposes_the_exact_canonical_names() -> None:
    assert TARGET_CONTIGUOUS_VIEWS == (
        "T6_TARGET_CONTIGUOUS",
        "T8_TARGET_CONTIGUOUS",
        "T12_TARGET_CONTIGUOUS",
        "T16_TARGET_CONTIGUOUS",
    )
    assert CAUSAL_HISTORY_VIEWS == (
        "T6_TARGET_PLUS_H6",
        "T6_TARGET_PLUS_H12",
        "T6_TARGET_PLUS_H24",
    )
    assert "S6_AT_16_SPARSE" in TEMPORAL_VIEW_NAMES
    assert "HISTORICAL_C6_SCREEN" in TEMPORAL_VIEW_NAMES
    contract = temporal_view_registry_contract()
    assert contract["ambiguous_6c_name_used"] is False
    assert contract["future_frame_dependence"] == 0
    assert contract["cross_length_families_separated"] is True


@pytest.mark.parametrize("view_name", TARGET_CONTIGUOUS_VIEWS)
def test_contiguous_targets_are_trailing_and_causal(view_name: str) -> None:
    spec = temporal_view_spec(view_name)
    offsets = spec.target_offsets_from_endpoint
    assert offsets[-1] == 0
    assert offsets == tuple(range(-(spec.target_length - 1), 1))
    assert spec.pair_deltas == tuple([1] * (spec.target_length - 1))
    assert max(offsets) == 0
    report = validate_window(_window(view_name))
    assert report["valid"], report["errors"]
    assert report["future_frame_dependence"] == 0


def test_exact_frame_offsets_and_timestamp_ordering() -> None:
    spec = temporal_view_spec("T8_TARGET_CONTIGUOUS")
    target, history = expected_frame_indices(spec, 200)
    assert target == (193, 194, 195, 196, 197, 198, 199, 200)
    assert history == ()

    window = _window("T8_TARGET_CONTIGUOUS", endpoint=200)
    assert validate_window(window)["valid"]

    shifted = replace(
        window,
        target_frame_indices=(192, *target[1:]),
    )
    assert not validate_window(shifted)["valid"]


def test_future_frames_are_rejected() -> None:
    window = _window("T6_TARGET_CONTIGUOUS", endpoint=50)
    future = replace(
        window,
        target_frame_indices=(*window.target_frame_indices[:-1], 51),
    )
    report = validate_window(future)
    assert not report["valid"]
    assert any("future" in error for error in report["errors"])


def test_cross_label_target_windows_are_rejected() -> None:
    window = _window(
        "T6_TARGET_CONTIGUOUS",
        target_labels=("move", "move", "move", "eat", "eat", "eat"),
    )
    report = validate_window(window)
    assert not report["valid"]
    assert any("cross-label" in error for error in report["errors"])
    with pytest.raises(WindowContractError):
        require_window(window)


@pytest.mark.parametrize("view_name", CAUSAL_HISTORY_VIEWS)
def test_history_precedes_target_and_may_carry_a_different_label(
    view_name: str,
) -> None:
    spec = temporal_view_spec(view_name)
    window = _window(view_name)
    assert spec.target_length == 6
    assert len(window.history_frame_indices) == spec.history_length
    assert max(window.history_frame_indices) < min(window.target_frame_indices)
    assert set(window.history_labels) != set(window.target_labels)
    report = validate_window(window)
    assert report["valid"], report["errors"]
    assert report["history_labels_in_x"] is False


def test_history_after_the_prediction_endpoint_is_rejected() -> None:
    window = _window("T6_TARGET_PLUS_H6")
    overlapping = replace(
        window,
        history_frame_indices=tuple(
            index + 12 for index in window.history_frame_indices
        ),
        history_timestamps_sec=tuple(
            (index + 12) / FPS for index in window.history_frame_indices
        ),
    )
    report = validate_window(overlapping)
    assert not report["valid"]
    assert any("strictly before" in error for error in report["errors"])


def test_history_labels_never_enter_model_x() -> None:
    window = _window("T6_TARGET_PLUS_H12")
    leaking = replace(
        window,
        model_input_fields=("cx_n", "history_behavior_label"),
    )
    report = validate_window(leaking)
    assert not report["valid"]
    assert any("never enter model X" in error for error in report["errors"])


def test_s6_at_16_sparse_uses_exact_offsets_and_pair_deltas() -> None:
    spec = temporal_view_spec("S6_AT_16_SPARSE")
    assert spec.legacy_burst_offsets == S6_AT_16_BURST_OFFSETS == (0, 3, 6, 9, 12, 15)
    assert spec.pair_deltas == S6_AT_16_PAIR_DELTAS == (3, 3, 3, 3, 3)
    assert spec.contiguous_target is False
    assert spec.legacy_only is True
    assert spec.primary_cross_source_eligible is False
    assert spec.uses_real_elapsed_seconds is True

    window = _window("S6_AT_16_SPARSE", endpoint=115, burst_start=100)
    assert window.target_frame_indices == (100, 103, 106, 109, 112, 115)
    assert validate_window(window)["valid"]


def test_historical_c6_screen_is_distinct_and_non_transferable() -> None:
    spec = temporal_view_spec("HISTORICAL_C6_SCREEN")
    assert spec.legacy_burst_offsets == HISTORICAL_C6_BURST_OFFSETS == (5, 6, 7, 8, 9, 10)
    assert spec.metrics_transferable is False
    assert views_are_distinct("HISTORICAL_C6_SCREEN", "S6_AT_16_SPARSE")
    assert views_are_distinct("HISTORICAL_C6_SCREEN", "T6_TARGET_CONTIGUOUS")
    assert (
        temporal_view_spec("S6_AT_16_SPARSE").legacy_burst_offsets
        != spec.legacy_burst_offsets
    )
    window = _window("HISTORICAL_C6_SCREEN", endpoint=110, burst_start=100)
    assert window.target_frame_indices == (105, 106, 107, 108, 109, 110)
    assert validate_window(window)["valid"]


def test_actor_authority_and_split_group_must_be_stable() -> None:
    window = _window("T6_TARGET_CONTIGUOUS")
    mixed_actor = replace(
        window,
        target_actor_authority=(
            "cvat|dataset|video|track7",
            *["cvat|dataset|video|track8"] * 5,
        ),
    )
    assert not validate_window(mixed_actor)["valid"]

    mixed_group = replace(
        window,
        target_split_group_ids=("date=291119", *["date=301119"] * 5),
    )
    assert not validate_window(mixed_group)["valid"]


def test_insufficient_history_is_masked_not_padded() -> None:
    window = _window("T6_TARGET_PLUS_H24")
    short = replace(window, available_history_frames=9)
    report = validate_window(short)
    assert report["valid"]
    assert any("insufficient causal history" in text for text in report["warnings"])


def test_window_ids_are_deterministic_and_view_specific() -> None:
    first = deterministic_window_id(_window("T6_TARGET_CONTIGUOUS"))
    repeat = deterministic_window_id(_window("T6_TARGET_CONTIGUOUS"))
    other_view = deterministic_window_id(_window("T8_TARGET_CONTIGUOUS"))
    other_endpoint = deterministic_window_id(
        _window("T6_TARGET_CONTIGUOUS", endpoint=101)
    )
    assert first == repeat
    assert first != other_view
    assert first != other_endpoint


def test_pair_endpoints_stay_inside_the_selected_view() -> None:
    for view_name in (*TARGET_CONTIGUOUS_VIEWS, "S6_AT_16_SPARSE"):
        spec = temporal_view_spec(view_name)
        offsets = spec.target_offsets_from_endpoint
        deltas = tuple(
            later - earlier
            for earlier, later in zip(offsets, offsets[1:], strict=False)
        )
        assert deltas == spec.pair_deltas
        assert min(offsets) >= -(max(offsets) - min(offsets))


def test_builder_contract_declares_no_production_dataset_build() -> None:
    summary = builder_contract_summary()
    assert summary["production_datasets_built"] is False
    assert len(summary["views"]) == len(TEMPORAL_VIEW_NAMES)

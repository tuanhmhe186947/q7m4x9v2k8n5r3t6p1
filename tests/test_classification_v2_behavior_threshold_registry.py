from __future__ import annotations

from dataclasses import replace

import pytest

from pig_behavior.classification_v2.review.behavior_threshold_registry import (
    ThresholdRegistryError,
    canonical_threshold_registry,
    derive_threshold_variant,
    resolve_threshold,
    threshold_by_name,
    threshold_registry_hash,
    threshold_registry_snapshot,
    validate_threshold_registry,
)

EXPECTED_PRODUCTION_THRESHOLD_NAMES = {
    "low_motion_support",
    "strong_motion_support",
    "low_roi_support",
    "strong_roi_support",
    "low_social_support",
    "strong_social_support",
    "conflict_review_threshold",
    "aggression_reference_n_per_second",
    "shape_transition_reference",
    "motion_speed_reference_n_per_second",
    "social_proximity_reference_n",
    "pig_diff_inner_reference",
    "roi_near_distance_diagonal_n",
    "roi_contact_distance_diagonal_n",
    "social_near_distance_n",
    "high_hidden_ratio_interval",
    "bbox_valid_ratio_authority_floor",
    "temporal_interval_complete_floor",
    "temporal_stability_indicator_floor",
    "partner_context_availability_floor",
    "relevant_evidence_availability_floor",
    "evidence_valid_ratio_positive",
}


def test_every_production_threshold_has_one_active_registry_entry() -> None:
    entries = canonical_threshold_registry()
    active = [entry for entry in entries if not entry.deprecated]
    assert {entry.threshold_name for entry in active} == (
        EXPECTED_PRODUCTION_THRESHOLD_NAMES
    )
    assert len({entry.threshold_id for entry in active}) == len(active)
    assert validate_threshold_registry(entries)["valid"] is True


def test_every_threshold_has_exact_metric_and_authority_binding() -> None:
    for entry in canonical_threshold_registry():
        resolved = resolve_threshold(
            entry.threshold_id,
            metric_id=entry.metric_id,
            metric_version=entry.metric_version,
            metric_units=entry.metric_units,
        )
        assert resolved == entry
        assert resolved.authority_hash
        assert resolved.semantic_hash


def test_missing_authority_hash_fails_closed() -> None:
    entries = list(canonical_threshold_registry())
    entries[0] = replace(entries[0], authority_hash="")
    with pytest.raises(ThresholdRegistryError, match="missing_authority_hash"):
        validate_threshold_registry(entries)


def test_missing_metric_version_fails_closed() -> None:
    entries = list(canonical_threshold_registry())
    entries[0] = replace(entries[0], metric_version="")
    with pytest.raises(ThresholdRegistryError, match="missing_metric_version"):
        validate_threshold_registry(entries)


def test_mismatched_metric_version_fails_closed() -> None:
    entry = canonical_threshold_registry()[0]
    with pytest.raises(ThresholdRegistryError, match="metric_version"):
        resolve_threshold(
            entry.threshold_id,
            metric_id=entry.metric_id,
            metric_version="wrong.metric.version",
            metric_units=entry.metric_units,
        )


def test_non_finite_threshold_fails_closed() -> None:
    entries = list(canonical_threshold_registry())
    entries[0] = replace(entries[0], threshold_value=float("nan"))
    with pytest.raises(
        ThresholdRegistryError,
        match="non_finite_threshold_value",
    ):
        validate_threshold_registry(entries)


def test_duplicate_active_threshold_ids_fail_closed() -> None:
    entries = list(canonical_threshold_registry())
    entries.append(entries[0])
    with pytest.raises(
        ThresholdRegistryError,
        match="duplicate_active_threshold_ids",
    ):
        validate_threshold_registry(entries)


def test_code_default_cannot_override_registry() -> None:
    entry = threshold_by_name("low_motion_support")
    with pytest.raises(ThresholdRegistryError, match="code_default"):
        resolve_threshold(
            entry.threshold_id,
            metric_id=entry.metric_id,
            metric_version=entry.metric_version,
            metric_units=entry.metric_units,
            code_default=entry.threshold_value + 0.01,
        )


def test_registry_hash_changes_with_threshold_value() -> None:
    entries = list(canonical_threshold_registry())
    base_hash = threshold_registry_hash(entries)
    entries[0] = derive_threshold_variant(
        entries[0],
        threshold_value=entries[0].threshold_value + 0.01,
    )
    assert threshold_registry_hash(entries) != base_hash


def test_registry_hash_changes_with_metric_version() -> None:
    entries = list(canonical_threshold_registry())
    base_hash = threshold_registry_hash(entries)
    entries[0] = derive_threshold_variant(
        entries[0],
        metric_version="classification_v2.mutated_metric.v2",
    )
    assert threshold_registry_hash(entries) != base_hash


def test_manifest_hash_mismatch_fails_closed() -> None:
    entry = canonical_threshold_registry()[0]
    with pytest.raises(
        ThresholdRegistryError,
        match="manifest_registry_hash",
    ):
        resolve_threshold(
            entry.threshold_id,
            metric_id=entry.metric_id,
            metric_version=entry.metric_version,
            metric_units=entry.metric_units,
            manifest_registry_hash="0" * 64,
        )


def test_snapshot_hash_matches_runtime_registry() -> None:
    snapshot = threshold_registry_snapshot()
    assert snapshot["registry_hash"] == threshold_registry_hash()
    assert snapshot["active_threshold_count"] == len(
        EXPECTED_PRODUCTION_THRESHOLD_NAMES
    )

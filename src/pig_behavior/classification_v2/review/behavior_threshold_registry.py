"""Canonical fail-closed threshold authority for Behavior Review selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Any

from pig_behavior.classification_v2.features.spatial_semantics import (
    AXIS_DISTANCE_METRIC_ID,
    AXIS_DISTANCE_METRIC_VERSION,
    DIAGONAL_DISTANCE_METRIC_ID,
    DIAGONAL_DISTANCE_METRIC_VERSION,
    ROI_CONTACT_THRESHOLD_VALUE,
    ROI_NEAR_THRESHOLD_VALUE,
    SOCIAL_NEAR_THRESHOLD_VALUE,
)

THRESHOLD_REGISTRY_ID = "classification_v2.behavior_threshold_registry"
THRESHOLD_REGISTRY_VERSION = "classification_v2.behavior_threshold_registry.v1"
THRESHOLD_AUTHORITY_ID = "classification_v2.finding_c.freeze.d563fc9"
THRESHOLD_AUTHORITY_VERSION = "v1"
SELECTION_METRIC_VERSION = "classification_v2.behavior_review_metrics.v1"

ALLOWED_AUTHORITY_TYPES = frozenset(
    {
        "PHYSICALLY_INTERPRETABLE_RULE",
        "FROZEN_PROJECT_SCREENING_HEURISTIC",
        "DISTRIBUTION_DERIVED_SCREENING_THRESHOLD",
        "EMPIRICALLY_CALIBRATED_THRESHOLD",
    }
)
ALLOWED_OPERATORS = frozenset(
    {
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
        "SCALE_BY_REFERENCE",
    }
)
NOT_CALIBRATED_POPULATION_ID = "none_not_empirically_calibrated"
NOT_CALIBRATED_POPULATION_HASH = hashlib.sha256(
    NOT_CALIBRATED_POPULATION_ID.encode("utf-8")
).hexdigest()


class ThresholdRegistryError(ValueError):
    """Raised when Behavior threshold authority cannot be resolved exactly."""


@dataclass(frozen=True, slots=True)
class ThresholdAuthority:
    """One immutable threshold-to-metric and scientific-authority binding."""

    threshold_id: str
    threshold_version: str
    threshold_name: str
    predicate_scope: str
    behavior_scope: str
    feature_name: str
    metric_id: str
    metric_version: str
    metric_units: str
    threshold_value: float
    comparison_operator: str
    authority_type: str
    authority_id: str
    authority_version: str
    authority_hash: str
    calibration_population_id: str
    calibration_population_hash: str
    calibration_method: str
    missing_value_policy: str
    availability_requirement: str
    semantic_hash: str
    deprecated: bool = False
    replacement_threshold_id: str = ""

    def payload_without_hashes(self) -> dict[str, Any]:
        """Return the semantic payload used to bind authority and registry."""

        payload = asdict(self)
        payload.pop("authority_hash")
        payload.pop("semantic_hash")
        return payload


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _entry(
    *,
    threshold_id: str,
    threshold_name: str,
    predicate_scope: str,
    behavior_scope: str,
    feature_name: str,
    metric_id: str,
    metric_version: str,
    metric_units: str,
    threshold_value: float,
    comparison_operator: str,
    availability_requirement: str,
    missing_value_policy: str = "MISSING_IS_UNAVAILABLE_NOT_NUMERIC_ZERO",
    authority_type: str = "FROZEN_PROJECT_SCREENING_HEURISTIC",
) -> ThresholdAuthority:
    provisional = ThresholdAuthority(
        threshold_id=threshold_id,
        threshold_version="v1",
        threshold_name=threshold_name,
        predicate_scope=predicate_scope,
        behavior_scope=behavior_scope,
        feature_name=feature_name,
        metric_id=metric_id,
        metric_version=metric_version,
        metric_units=metric_units,
        threshold_value=float(threshold_value),
        comparison_operator=comparison_operator,
        authority_type=authority_type,
        authority_id=THRESHOLD_AUTHORITY_ID,
        authority_version=THRESHOLD_AUTHORITY_VERSION,
        authority_hash="",
        calibration_population_id=NOT_CALIBRATED_POPULATION_ID,
        calibration_population_hash=NOT_CALIBRATED_POPULATION_HASH,
        calibration_method=(
            "Frozen current project screening value at base SHA d563fc9; "
            "not fitted to Behavior decisions and not empirically calibrated"
        ),
        missing_value_policy=missing_value_policy,
        availability_requirement=availability_requirement,
        semantic_hash="",
    )
    authority_hash = _stable_hash(
        {
            "authority_type": provisional.authority_type,
            "authority_id": provisional.authority_id,
            "authority_version": provisional.authority_version,
            "calibration_population_id": provisional.calibration_population_id,
            "calibration_population_hash": (
                provisional.calibration_population_hash
            ),
            "calibration_method": provisional.calibration_method,
        }
    )
    with_authority = replace(provisional, authority_hash=authority_hash)
    return replace(
        with_authority,
        semantic_hash=_stable_hash(with_authority.payload_without_hashes()),
    )


def canonical_threshold_registry() -> tuple[ThresholdAuthority, ...]:
    """Return the complete active Behavior Review threshold registry."""

    ratio = "normalized_support_ratio"
    metric = SELECTION_METRIC_VERSION
    entries = (
        _entry(
            threshold_id="behavior.motion.low_support.v1",
            threshold_name="low_motion_support",
            predicate_scope="MOTION_CONTRADICTION",
            behavior_scope="move",
            feature_name="review_motion_support_score",
            metric_id="behavior_motion_support",
            metric_version=metric,
            metric_units=ratio,
            threshold_value=0.25,
            comparison_operator="<",
            availability_requirement="review_motion_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.motion.strong_support.v1",
            threshold_name="strong_motion_support",
            predicate_scope=(
                "MOTION_CONTRADICTION|ROI_POSSIBLE_FALSE_NEGATIVE"
            ),
            behavior_scope="stand|explore",
            feature_name="review_motion_support_score",
            metric_id="behavior_motion_support",
            metric_version=metric,
            metric_units=ratio,
            threshold_value=0.60,
            comparison_operator=">",
            availability_requirement="review_motion_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.roi.low_support.v1",
            threshold_name="low_roi_support",
            predicate_scope="ROI_CONTRADICTION",
            behavior_scope="eat|drink|playwithtoy",
            feature_name="review_roi_support_score",
            metric_id="behavior_roi_support",
            metric_version=metric,
            metric_units=ratio,
            threshold_value=0.25,
            comparison_operator="<",
            availability_requirement="review_roi_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.roi.strong_competing_support.v1",
            threshold_name="strong_roi_support",
            predicate_scope="ROI_CONTRADICTION",
            behavior_scope="eat|drink|playwithtoy",
            feature_name="review_competing_roi_support_score",
            metric_id="behavior_roi_support",
            metric_version=metric,
            metric_units=ratio,
            threshold_value=0.65,
            comparison_operator=">",
            availability_requirement="review_roi_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.social.low_support.v1",
            threshold_name="low_social_support",
            predicate_scope="INTERACTION_CONTRADICTION",
            behavior_scope="fight|social-nose",
            feature_name="review_social_or_fight_support_score",
            metric_id="behavior_social_support",
            metric_version=metric,
            metric_units=ratio,
            threshold_value=0.25,
            comparison_operator="<",
            availability_requirement="review_social_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.social.strong_fight_like.v1",
            threshold_name="strong_social_support",
            predicate_scope="INTERACTION_CONTRADICTION",
            behavior_scope="social-nose",
            feature_name="review_fight_like_score",
            metric_id="behavior_fight_like_support",
            metric_version=metric,
            metric_units=ratio,
            threshold_value=0.60,
            comparison_operator=">",
            availability_requirement="review_social_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.conflict.review_gate.v1",
            threshold_name="conflict_review_threshold",
            predicate_scope=(
                "ROI_CONTRADICTION|ROI_POSSIBLE_FALSE_NEGATIVE|"
                "INTERACTION_CONTRADICTION|MOTION_CONTRADICTION|"
                "POSTURE_CONTRADICTION"
            ),
            behavior_scope="behavior_specific",
            feature_name="review_evidence_conflict_score",
            metric_id="behavior_conflict_score",
            metric_version=metric,
            metric_units=ratio,
            threshold_value=0.45,
            comparison_operator=">=",
            availability_requirement=(
                "review_relevant_evidence_available=true"
            ),
        ),
        _entry(
            threshold_id="behavior.social.aggression_reference.v1",
            threshold_name="aggression_reference_n_per_second",
            predicate_scope="INTERACTION_CONTRADICTION",
            behavior_scope="fight|social-nose",
            feature_name=(
                "social_aggression_proxy_n_per_second_p90_unit"
            ),
            metric_id="social_aggression_proxy",
            metric_version=metric,
            metric_units="normalized_distance_per_second2_proxy",
            threshold_value=0.60,
            comparison_operator="SCALE_BY_REFERENCE",
            availability_requirement="review_social_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.posture.shape_transition_reference.v1",
            threshold_name="shape_transition_reference",
            predicate_scope="POSTURE_CONTRADICTION",
            behavior_scope="lying|sitting",
            feature_name="bbox_shape_change_p90_unit",
            metric_id="bbox_shape_transition",
            metric_version=metric,
            metric_units="normalized_shape_delta",
            threshold_value=0.20,
            comparison_operator="SCALE_BY_REFERENCE",
            availability_requirement="review_posture_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.motion.speed_reference.v1",
            threshold_name="motion_speed_reference_n_per_second",
            predicate_scope="MOTION_CONTRADICTION",
            behavior_scope="move|explore|stand|fight",
            feature_name="motion_speed_n_per_second_p90_unit",
            metric_id="axis_normalized_speed",
            metric_version=metric,
            metric_units="axis_normalized_distance_per_second",
            threshold_value=0.36,
            comparison_operator="SCALE_BY_REFERENCE",
            availability_requirement="review_motion_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.social.proximity_reference.v1",
            threshold_name="social_proximity_reference_n",
            predicate_scope="INTERACTION_CONTRADICTION",
            behavior_scope="fight|social-nose",
            feature_name="social_nearest_dist_p50_unit",
            metric_id=AXIS_DISTANCE_METRIC_ID,
            metric_version=AXIS_DISTANCE_METRIC_VERSION,
            metric_units="axis_normalized_image_distance",
            threshold_value=0.12,
            comparison_operator="SCALE_BY_REFERENCE",
            availability_requirement="review_social_evidence_available=true",
        ),
        _entry(
            threshold_id="behavior.pig.diff_inner_reference.v1",
            threshold_name="pig_diff_inner_reference",
            predicate_scope=(
                "MOTION_CONTRADICTION|POSTURE_CONTRADICTION|"
                "INTERACTION_CONTRADICTION"
            ),
            behavior_scope="move|explore|stand|fight|social-nose|lying|sitting",
            feature_name="review_pig_diff_inner_mean",
            metric_id="pig_strenet_inner_difference_mean",
            metric_version="classification_v2.pig_strenet_review.v3",
            metric_units="normalized_absolute_pixel_difference",
            threshold_value=0.20,
            comparison_operator="SCALE_BY_REFERENCE",
            availability_requirement="review_pig_diff_valid_ratio>0",
        ),
        _entry(
            threshold_id="behavior.native.roi_near_distance.v1",
            threshold_name="roi_near_distance_diagonal_n",
            predicate_scope=(
                "ROI_CONTRADICTION|ROI_POSSIBLE_FALSE_NEGATIVE"
            ),
            behavior_scope="eat|drink|playwithtoy|explore",
            feature_name="roi_*_near_ratio_unit",
            metric_id=DIAGONAL_DISTANCE_METRIC_ID,
            metric_version=DIAGONAL_DISTANCE_METRIC_VERSION,
            metric_units="diagonal_normalized_image_distance",
            threshold_value=ROI_NEAR_THRESHOLD_VALUE,
            comparison_operator="<=",
            availability_requirement="roi_*_availability_ratio_unit>0",
        ),
        _entry(
            threshold_id="behavior.native.roi_contact_distance.v1",
            threshold_name="roi_contact_distance_diagonal_n",
            predicate_scope=(
                "ROI_CONTRADICTION|ROI_POSSIBLE_FALSE_NEGATIVE"
            ),
            behavior_scope="eat|drink|playwithtoy|explore",
            feature_name="roi_*_contact_ratio_unit",
            metric_id=DIAGONAL_DISTANCE_METRIC_ID,
            metric_version=DIAGONAL_DISTANCE_METRIC_VERSION,
            metric_units="diagonal_normalized_image_distance",
            threshold_value=ROI_CONTACT_THRESHOLD_VALUE,
            comparison_operator="<=",
            availability_requirement="roi_*_availability_ratio_unit>0",
        ),
        _entry(
            threshold_id="behavior.native.social_near_distance.v1",
            threshold_name="social_near_distance_n",
            predicate_scope="INTERACTION_CONTRADICTION",
            behavior_scope="fight|social-nose",
            feature_name="social_nearest_dist_p50_unit",
            metric_id=AXIS_DISTANCE_METRIC_ID,
            metric_version=AXIS_DISTANCE_METRIC_VERSION,
            metric_units="axis_normalized_image_distance",
            threshold_value=SOCIAL_NEAR_THRESHOLD_VALUE,
            comparison_operator="<=",
            availability_requirement=(
                "social_neighbor_availability_ratio_unit>0"
            ),
        ),
        _entry(
            threshold_id="behavior.authority.hidden_ratio.v1",
            threshold_name="high_hidden_ratio_interval",
            predicate_scope="MEDIA_OR_ACTOR_AUTHORITY_RISK",
            behavior_scope="all",
            feature_name="hidden_ratio_interval",
            metric_id="interval_hidden_ratio",
            metric_version=metric,
            metric_units="fraction_of_interval_frames",
            threshold_value=0.50,
            comparison_operator=">",
            availability_requirement="hidden_ratio_interval is finite",
        ),
        _entry(
            threshold_id="behavior.authority.bbox_validity.v1",
            threshold_name="bbox_valid_ratio_authority_floor",
            predicate_scope="MEDIA_OR_ACTOR_AUTHORITY_RISK",
            behavior_scope="all",
            feature_name="bbox_valid_ratio_interval",
            metric_id="interval_bbox_valid_ratio",
            metric_version=metric,
            metric_units="fraction_of_interval_frames",
            threshold_value=0.0,
            comparison_operator="<=",
            availability_requirement="bbox_valid_ratio_interval is finite",
        ),
        _entry(
            threshold_id="behavior.authority.interval_complete.v1",
            threshold_name="temporal_interval_complete_floor",
            predicate_scope="MEDIA_OR_ACTOR_AUTHORITY_RISK",
            behavior_scope="all",
            feature_name="temporal_interval_complete",
            metric_id="boolean_authority_indicator",
            metric_version=metric,
            metric_units="indicator",
            threshold_value=1.0,
            comparison_operator="<",
            availability_requirement="field must be present or explicit true",
        ),
        _entry(
            threshold_id="behavior.temporal.stability.v1",
            threshold_name="temporal_stability_indicator_floor",
            predicate_scope="TEMPORAL_CONTRADICTION",
            behavior_scope="all",
            feature_name="temporal_consistency_status",
            metric_id="temporal_stability_indicator",
            metric_version=metric,
            metric_units="indicator",
            threshold_value=1.0,
            comparison_operator="<",
            availability_requirement="temporal_consistency_status is present",
        ),
        _entry(
            threshold_id="behavior.social.partner_context_available.v1",
            threshold_name="partner_context_availability_floor",
            predicate_scope="PARTNER_CONTEXT_INSUFFICIENT",
            behavior_scope="fight|social-nose",
            feature_name="review_social_evidence_available",
            metric_id="evidence_availability_indicator",
            metric_version=metric,
            metric_units="indicator",
            threshold_value=1.0,
            comparison_operator="<",
            availability_requirement="interaction behavior",
        ),
        _entry(
            threshold_id="behavior.evidence.sufficiency.v1",
            threshold_name="relevant_evidence_availability_floor",
            predicate_scope="EVIDENCE_INSUFFICIENCY",
            behavior_scope="behavior_specific",
            feature_name="review_relevant_evidence_available",
            metric_id="evidence_availability_indicator",
            metric_version=metric,
            metric_units="indicator",
            threshold_value=1.0,
            comparison_operator="<",
            availability_requirement="behavior has a declared evidence branch",
        ),
        _entry(
            threshold_id="behavior.evidence.valid_ratio_positive.v1",
            threshold_name="evidence_valid_ratio_positive",
            predicate_scope=(
                "ROI_CONTRADICTION|INTERACTION_CONTRADICTION|"
                "MOTION_CONTRADICTION|POSTURE_CONTRADICTION|"
                "EVIDENCE_INSUFFICIENCY"
            ),
            behavior_scope="behavior_specific",
            feature_name="declared_evidence_valid_ratio",
            metric_id="evidence_valid_ratio",
            metric_version=metric,
            metric_units="fraction",
            threshold_value=0.0,
            comparison_operator=">",
            availability_requirement="modality-specific validity field exists",
        ),
    )
    validate_threshold_registry(entries)
    return entries


def validate_threshold_registry(
    entries: Iterable[ThresholdAuthority],
) -> dict[str, Any]:
    """Validate completeness, uniqueness, hashes, metrics, and replacements."""

    rows = tuple(entries)
    errors: list[str] = []
    active = [entry for entry in rows if not entry.deprecated]
    active_ids = [entry.threshold_id for entry in active]
    duplicate_ids = sorted(
        {
            threshold_id
            for threshold_id in active_ids
            if active_ids.count(threshold_id) > 1
        }
    )
    if duplicate_ids:
        errors.append(f"duplicate_active_threshold_ids={duplicate_ids}")
    all_ids = {entry.threshold_id for entry in rows}
    for entry in rows:
        required_text = {
            "threshold_id": entry.threshold_id,
            "threshold_version": entry.threshold_version,
            "threshold_name": entry.threshold_name,
            "predicate_scope": entry.predicate_scope,
            "behavior_scope": entry.behavior_scope,
            "feature_name": entry.feature_name,
            "metric_id": entry.metric_id,
            "metric_version": entry.metric_version,
            "metric_units": entry.metric_units,
            "authority_type": entry.authority_type,
            "authority_id": entry.authority_id,
            "authority_version": entry.authority_version,
            "authority_hash": entry.authority_hash,
            "calibration_population_id": entry.calibration_population_id,
            "calibration_population_hash": (
                entry.calibration_population_hash
            ),
            "calibration_method": entry.calibration_method,
            "missing_value_policy": entry.missing_value_policy,
            "availability_requirement": entry.availability_requirement,
            "semantic_hash": entry.semantic_hash,
        }
        for field, value in required_text.items():
            if not str(value).strip():
                errors.append(f"{entry.threshold_id}:missing_{field}")
        if (
            not isinstance(entry.threshold_value, (int, float))
            or not math.isfinite(float(entry.threshold_value))
        ):
            errors.append(f"{entry.threshold_id}:non_finite_threshold_value")
        if entry.authority_type not in ALLOWED_AUTHORITY_TYPES:
            errors.append(f"{entry.threshold_id}:invalid_authority_type")
        if entry.comparison_operator not in ALLOWED_OPERATORS:
            errors.append(f"{entry.threshold_id}:invalid_comparison_operator")
        expected_authority_hash = _stable_hash(
            {
                "authority_type": entry.authority_type,
                "authority_id": entry.authority_id,
                "authority_version": entry.authority_version,
                "calibration_population_id": entry.calibration_population_id,
                "calibration_population_hash": (
                    entry.calibration_population_hash
                ),
                "calibration_method": entry.calibration_method,
            }
        )
        if entry.authority_hash != expected_authority_hash:
            errors.append(f"{entry.threshold_id}:authority_hash_mismatch")
        if entry.semantic_hash != _stable_hash(
            entry.payload_without_hashes()
        ):
            errors.append(f"{entry.threshold_id}:semantic_hash_mismatch")
        if entry.deprecated and (
            not entry.replacement_threshold_id
            or entry.replacement_threshold_id not in all_ids
        ):
            errors.append(
                f"{entry.threshold_id}:deprecated_without_valid_replacement"
            )
    audit = {
        "registry_id": THRESHOLD_REGISTRY_ID,
        "registry_version": THRESHOLD_REGISTRY_VERSION,
        "active_threshold_count": len(active),
        "duplicate_active_threshold_ids": duplicate_ids,
        "errors": errors,
        "valid": not errors,
    }
    if errors:
        raise ThresholdRegistryError("; ".join(errors))
    return audit


def threshold_registry_hash(
    entries: Iterable[ThresholdAuthority] | None = None,
) -> str:
    """Hash the complete ordered active registry payload."""

    registry = tuple(entries or canonical_threshold_registry())
    validate_threshold_registry(registry)
    payload = [
        asdict(entry)
        for entry in sorted(
            (item for item in registry if not item.deprecated),
            key=lambda item: item.threshold_id,
        )
    ]
    return _stable_hash(
        {
            "registry_id": THRESHOLD_REGISTRY_ID,
            "registry_version": THRESHOLD_REGISTRY_VERSION,
            "active_thresholds": payload,
        }
    )


def derive_threshold_variant(
    entry: ThresholdAuthority,
    **changes: Any,
) -> ThresholdAuthority:
    """Create an isolated hash-consistent variant for tests or sensitivity."""

    unsupported = set(changes).difference(
        {"threshold_value", "metric_id", "metric_version", "metric_units"}
    )
    if unsupported:
        raise ThresholdRegistryError(
            f"unsupported_threshold_variant_fields={sorted(unsupported)}"
        )
    variant = replace(
        entry,
        **changes,
        semantic_hash="",
    )
    return replace(
        variant,
        semantic_hash=_stable_hash(variant.payload_without_hashes()),
    )


def resolve_threshold(
    threshold_id: str,
    *,
    metric_id: str,
    metric_version: str,
    metric_units: str,
    entries: Iterable[ThresholdAuthority] | None = None,
    code_default: float | None = None,
    manifest_registry_hash: str | None = None,
) -> ThresholdAuthority:
    """Resolve one active threshold and reject every authority mismatch."""

    registry = tuple(entries or canonical_threshold_registry())
    validate_threshold_registry(registry)
    matches = [
        entry
        for entry in registry
        if entry.threshold_id == threshold_id and not entry.deprecated
    ]
    if len(matches) != 1:
        raise ThresholdRegistryError(
            f"active_threshold_resolution_count={threshold_id}:{len(matches)}"
        )
    entry = matches[0]
    mismatches = []
    if entry.metric_id != metric_id:
        mismatches.append(f"metric_id={metric_id}:{entry.metric_id}")
    if entry.metric_version != metric_version:
        mismatches.append(
            f"metric_version={metric_version}:{entry.metric_version}"
        )
    if entry.metric_units != metric_units:
        mismatches.append(f"metric_units={metric_units}:{entry.metric_units}")
    if code_default is not None and not math.isclose(
        float(code_default),
        entry.threshold_value,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        mismatches.append(
            f"code_default={code_default}:{entry.threshold_value}"
        )
    current_hash = threshold_registry_hash(registry)
    if (
        manifest_registry_hash is not None
        and manifest_registry_hash != current_hash
    ):
        mismatches.append(
            f"manifest_registry_hash={manifest_registry_hash}:{current_hash}"
        )
    if mismatches:
        raise ThresholdRegistryError(
            f"{threshold_id} binding mismatch: " + "; ".join(mismatches)
        )
    return entry


def threshold_by_name(
    threshold_name: str,
    *,
    entries: Iterable[ThresholdAuthority] | None = None,
) -> ThresholdAuthority:
    """Resolve a unique active threshold by its code-facing name."""

    registry = tuple(entries or canonical_threshold_registry())
    validate_threshold_registry(registry)
    matches = [
        entry
        for entry in registry
        if entry.threshold_name == threshold_name and not entry.deprecated
    ]
    if len(matches) != 1:
        raise ThresholdRegistryError(
            f"active_threshold_name_resolution={threshold_name}:{len(matches)}"
        )
    entry = matches[0]
    return resolve_threshold(
        entry.threshold_id,
        metric_id=entry.metric_id,
        metric_version=entry.metric_version,
        metric_units=entry.metric_units,
        entries=registry,
    )


def threshold_registry_snapshot() -> dict[str, Any]:
    """Return the immutable publication payload for the active registry."""

    entries = canonical_threshold_registry()
    audit = validate_threshold_registry(entries)
    return {
        "schema_version": THRESHOLD_REGISTRY_VERSION,
        "registry_id": THRESHOLD_REGISTRY_ID,
        "registry_version": THRESHOLD_REGISTRY_VERSION,
        "registry_hash": threshold_registry_hash(entries),
        "active_threshold_count": audit["active_threshold_count"],
        "thresholds": [asdict(entry) for entry in entries],
    }


__all__ = [
    "ALLOWED_AUTHORITY_TYPES",
    "SELECTION_METRIC_VERSION",
    "THRESHOLD_REGISTRY_ID",
    "THRESHOLD_REGISTRY_VERSION",
    "ThresholdAuthority",
    "ThresholdRegistryError",
    "canonical_threshold_registry",
    "derive_threshold_variant",
    "resolve_threshold",
    "threshold_by_name",
    "threshold_registry_hash",
    "threshold_registry_snapshot",
    "validate_threshold_registry",
]

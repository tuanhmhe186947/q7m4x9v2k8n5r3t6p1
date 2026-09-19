"""Authoritative fail-closed Pig-STRENet motion tensor schema."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

MOTION_SCHEMA_ID = "schema.pig_strenet_motion_v2"
MOTION_SCHEMA_VERSION = "classification_v2.motion_tensor.v2"
MOTION_SCHEMA_DTYPE = "float32"
ACCELERATION_SEMANTICS_VERSION = (
    "classification_v2.acceleration_semantics.v3"
)
GENERIC_ACCELERATION_ALIAS = "acceleration_n_per_second2"
LEGACY_ACCELERATION_AUDIT_ALIAS = (
    "legacy_acceleration_alias_tangential_only"
)
_AMBIGUOUS_ACCELERATION_PATTERN = re.compile(
    r"(?<!tangential_)acceleration_n_per_second2"
)
MOTION_FEATURE_NAMES: tuple[str, ...] = (
    "vx_n_per_second",
    "vy_n_per_second",
    "bw_rate_n_per_second",
    "bh_rate_n_per_second",
    "area_rate_n_per_second",
    "aspect_ratio_rate_per_second",
    "speed_n_per_second",
    "direction_change_rad",
    "tangential_acceleration_n_per_second2",
    "ax_n_per_second2",
    "ay_n_per_second2",
    "acceleration_vector_magnitude_n_per_second2",
)
MOTION_SCHEMA_DIMENSION = len(MOTION_FEATURE_NAMES)
MOTION_REQUIRED_MASKS: tuple[str, ...] = (
    "valid_motion_pair",
    "velocity_valid",
    "bbox_rate_valid",
    "direction_valid",
    "direction_change_valid",
    "tangential_acceleration_valid",
    "vector_acceleration_valid",
    "motion_feature_available",
)
MOTION_AGGREGATION_OUTPUTS: tuple[str, ...] = (
    "observed_frame_count",
    "possible_pair_count",
    "valid_pair_count",
    "valid_pair_ratio",
    "motion_feature_coverage",
    "velocity_possible_count",
    "velocity_valid_count",
    "velocity_coverage",
    "direction_change_possible_count",
    "direction_change_valid_count",
    "direction_change_coverage",
    "acceleration_possible_count",
    "acceleration_valid_count",
    "acceleration_coverage",
)


class MotionSchemaError(ValueError):
    """Raised when a producer/exporter motion contract fails closed."""


def acceleration_compatibility_registry() -> dict[str, dict[str, Any]]:
    """Return the explicit non-predictive boundary for historical aliases."""

    return {
        LEGACY_ACCELERATION_AUDIT_ALIAS: {
            "predictive": False,
            "deprecated": True,
            "semantic_target": (
                "tangential_acceleration_n_per_second2"
            ),
            "allowed_in_current_export": False,
            "allowed_in_model_x": False,
            "replaces_ambiguous_name": GENERIC_ACCELERATION_ALIAS,
        }
    }


def ambiguous_acceleration_names(
    names: Sequence[str],
) -> list[str]:
    """Return names that imply generic acceleration for d(speed)/dt."""

    return [
        str(name)
        for name in names
        if _AMBIGUOUS_ACCELERATION_PATTERN.search(str(name))
    ]


def require_unambiguous_acceleration_names(
    names: Sequence[str],
    *,
    context: str,
) -> None:
    """Reject generic acceleration semantics at predictive boundaries."""

    ambiguous = ambiguous_acceleration_names(names)
    if ambiguous:
        raise MotionSchemaError(
            f"{context} contains ambiguous acceleration names: {ambiguous}"
        )


def canonical_motion_schema_payload() -> dict[str, Any]:
    """Return the exact payload used by the scientific-contract hash."""

    return {
        "schema_id": MOTION_SCHEMA_ID,
        "schema_version": MOTION_SCHEMA_VERSION,
        "dtype": MOTION_SCHEMA_DTYPE,
        "ordered_feature_names": list(MOTION_FEATURE_NAMES),
        "validity_masks": list(MOTION_REQUIRED_MASKS),
        "aggregation_outputs": list(MOTION_AGGREGATION_OUTPUTS),
    }


def motion_schema_hash() -> str:
    """Return deterministic SHA-256 over tensor-defining semantics."""

    encoded = json.dumps(
        canonical_motion_schema_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


MOTION_SCHEMA_HASH = motion_schema_hash()


def motion_schema_metadata() -> dict[str, Any]:
    """Return the authoritative manifest fields for audits and artifacts."""

    return {
        **canonical_motion_schema_payload(),
        "dimension": MOTION_SCHEMA_DIMENSION,
        "schema_hash": MOTION_SCHEMA_HASH,
        "acceleration_semantics_version": (
            ACCELERATION_SEMANTICS_VERSION
        ),
        "acceleration_compatibility_registry": (
            acceleration_compatibility_registry()
        ),
    }


def motion_schema_preflight(
    *,
    source_columns: Sequence[str],
    actual_feature_names: Sequence[str],
    actual_masks: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a declared tensor group and producer surface to the authority."""

    actual_names = [str(name) for name in actual_feature_names]
    masks = [str(name) for name in actual_masks]
    source = {str(name) for name in source_columns}
    expected_names = list(MOTION_FEATURE_NAMES)
    expected_masks = list(MOTION_REQUIRED_MASKS)
    observed_metadata = dict(metadata or motion_schema_metadata())

    duplicate_features = sorted(
        {name for name in actual_names if actual_names.count(name) > 1}
    )
    blank_features = [
        index for index, name in enumerate(actual_names) if not name.strip()
    ]
    missing_features = [name for name in expected_names if name not in source]
    unexpected_features = [
        name for name in actual_names if name not in expected_names
    ]
    order_mismatches = [
        {
            "index": index,
            "expected": expected,
            "actual": actual,
        }
        for index, (expected, actual) in enumerate(
            zip(expected_names, actual_names, strict=False)
        )
        if expected != actual
    ]
    if len(actual_names) != len(expected_names):
        order_mismatches.append(
            {
                "index": "dimension",
                "expected": len(expected_names),
                "actual": len(actual_names),
            }
        )
    required_masks_missing = [
        name for name in expected_masks if name not in source or name not in masks
    ]

    errors: list[str] = []
    if blank_features:
        errors.append(f"blank_motion_feature_names={blank_features}")
    if duplicate_features:
        errors.append(f"duplicate_motion_features={duplicate_features}")
    if missing_features:
        errors.append(f"missing_required_motion_features={missing_features}")
    if unexpected_features:
        errors.append(f"unexpected_motion_features={unexpected_features}")
    if order_mismatches:
        errors.append(f"motion_feature_order_mismatch={order_mismatches}")
    if required_masks_missing:
        errors.append(f"required_motion_masks_missing={required_masks_missing}")

    expected_metadata = motion_schema_metadata()
    metadata_fields = (
        "schema_id",
        "schema_version",
        "dtype",
        "dimension",
        "schema_hash",
    )
    for field in metadata_fields:
        if observed_metadata.get(field) != expected_metadata[field]:
            errors.append(
                f"motion_schema_{field}_mismatch="
                f"{observed_metadata.get(field)!r}:{expected_metadata[field]!r}"
            )

    return {
        "schema_id": MOTION_SCHEMA_ID,
        "schema_version": MOTION_SCHEMA_VERSION,
        "expected_dimension": MOTION_SCHEMA_DIMENSION,
        "actual_dimension": len(actual_names),
        "expected_feature_names": expected_names,
        "actual_feature_names": actual_names,
        "expected_schema_hash": MOTION_SCHEMA_HASH,
        "actual_schema_hash": observed_metadata.get("schema_hash"),
        "missing_features": missing_features,
        "duplicate_features": duplicate_features,
        "unexpected_features": unexpected_features,
        "blank_feature_indices": blank_features,
        "order_mismatches": order_mismatches,
        "required_masks_missing": required_masks_missing,
        "errors": errors,
    }


def require_motion_schema(
    *,
    source_columns: Sequence[str],
    actual_feature_names: Sequence[str],
    actual_masks: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a clean preflight or raise one precise transactional error."""

    result = motion_schema_preflight(
        source_columns=source_columns,
        actual_feature_names=actual_feature_names,
        actual_masks=actual_masks,
        metadata=metadata,
    )
    if result["errors"]:
        raise MotionSchemaError(
            "Motion schema preflight failed: " + "; ".join(result["errors"])
        )
    return result


__all__ = [
    "MOTION_AGGREGATION_OUTPUTS",
    "ACCELERATION_SEMANTICS_VERSION",
    "GENERIC_ACCELERATION_ALIAS",
    "LEGACY_ACCELERATION_AUDIT_ALIAS",
    "MOTION_FEATURE_NAMES",
    "MOTION_REQUIRED_MASKS",
    "MOTION_SCHEMA_DIMENSION",
    "MOTION_SCHEMA_DTYPE",
    "MOTION_SCHEMA_HASH",
    "MOTION_SCHEMA_ID",
    "MOTION_SCHEMA_VERSION",
    "MotionSchemaError",
    "acceleration_compatibility_registry",
    "ambiguous_acceleration_names",
    "canonical_motion_schema_payload",
    "motion_schema_hash",
    "motion_schema_metadata",
    "motion_schema_preflight",
    "require_unambiguous_acceleration_names",
    "require_motion_schema",
]

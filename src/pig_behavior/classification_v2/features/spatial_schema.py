"""Canonical fail-closed spatial predictive tensor schema."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
)

SPATIAL_SCHEMA_ID = "schema.classification_v2_spatial_predictive_v1"
SPATIAL_SCHEMA_VERSION = "classification_v2.spatial_predictive_tensor.v1"
SPATIAL_SCHEMA_DTYPE = "float32"
SPATIAL_SCHEMA_POLICY = "POLICY_CURRENT_ONLY_FAIL_CLOSED"
EXPECTED_CURRENT_SPATIAL_DIMENSION = 46

SPATIAL_PREDICTIVE_GROUP_NAMES: tuple[str, ...] = (
    "bbox_xywh_n",
    "bbox_shape_n",
    "motion_delta",
    "roi_class_relation",
    "social_relation",
)

SPATIAL_PREDICTIVE_FEATURES: dict[str, tuple[str, ...]] = {
    "bbox_xywh_n": ("cx_n", "cy_n", "bw_n", "bh_n"),
    "bbox_shape_n": ("area_n", "aspect_ratio"),
    "motion_delta": tuple(MOTION_FEATURE_NAMES),
    "roi_class_relation": (
        "roi_feeder_min_dist_n",
        "roi_feeder_max_overlap_ratio",
        "roi_feeder_max_iou",
        "roi_feeder_center_inside",
        "roi_feeder_near",
        "roi_feeder_contact",
        "roi_drinker_min_dist_n",
        "roi_drinker_max_overlap_ratio",
        "roi_drinker_max_iou",
        "roi_drinker_center_inside",
        "roi_drinker_near",
        "roi_drinker_contact",
        "roi_toy_min_dist_n",
        "roi_toy_max_overlap_ratio",
        "roi_toy_max_iou",
        "roi_toy_center_inside",
        "roi_toy_near",
        "roi_toy_contact",
    ),
    "social_relation": (
        "nearest_dist_n",
        "nearest_pair_iou",
        "nearest_pair_overlap_ratio",
        "social_density_near_count",
        "social_contact_count",
        "partner_distance_delta_n",
        "approach_speed_n_per_second",
        "retreat_speed_n_per_second",
        "pair_contact_with_nearest",
        "aggression_score_proxy_per_second",
    ),
}

SPATIAL_GROUP_CONTRACTS: dict[str, dict[str, Any]] = {
    "bbox_xywh_n": {
        "group_optional": False,
        "fixed_group_dimension": 4,
        "availability_mask": "spatial_quality_mask",
        "missing_fill_policy": "ZERO_PLACEHOLDER_MASKED_NOT_OBSERVED",
    },
    "bbox_shape_n": {
        "group_optional": False,
        "fixed_group_dimension": 2,
        "availability_mask": "spatial_quality_mask",
        "missing_fill_policy": "ZERO_PLACEHOLDER_MASKED_NOT_OBSERVED",
    },
    "motion_delta": {
        "group_optional": False,
        "fixed_group_dimension": len(MOTION_FEATURE_NAMES),
        "availability_mask": "motion_feature_available_mask",
        "missing_fill_policy": "ZERO_PLACEHOLDER_MASKED_NOT_OBSERVED",
    },
    "roi_class_relation": {
        "group_optional": False,
        "fixed_group_dimension": 18,
        "availability_mask": "roi_validity_mask",
        "missing_fill_policy": "ZERO_PLACEHOLDER_MASKED_NOT_OBSERVED",
    },
    "social_relation": {
        "group_optional": False,
        "fixed_group_dimension": 10,
        "availability_mask": "social_validity_mask",
        "missing_fill_policy": "ZERO_PLACEHOLDER_MASKED_NOT_OBSERVED",
    },
}

SPATIAL_GROUP_SCHEMA_VERSIONS: dict[str, str] = {
    group: f"classification_v2.spatial_group.{group}.v1"
    for group in SPATIAL_PREDICTIVE_GROUP_NAMES
}


def _spatial_group_schema_hash(group: str) -> str:
    payload = {
        "group_name": group,
        "group_schema_version": SPATIAL_GROUP_SCHEMA_VERSIONS[group],
        "ordered_feature_names": list(SPATIAL_PREDICTIVE_FEATURES[group]),
        **SPATIAL_GROUP_CONTRACTS[group],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SPATIAL_GROUP_SCHEMA_HASHES: dict[str, str] = {
    group: _spatial_group_schema_hash(group)
    for group in SPATIAL_PREDICTIVE_GROUP_NAMES
}

SPATIAL_SCHEMA_TOTAL_DIMENSION = sum(
    len(SPATIAL_PREDICTIVE_FEATURES[group])
    for group in SPATIAL_PREDICTIVE_GROUP_NAMES
)
if SPATIAL_SCHEMA_TOTAL_DIMENSION != EXPECTED_CURRENT_SPATIAL_DIMENSION:
    raise RuntimeError(
        "Canonical spatial dimension changed without an explicit schema "
        f"migration: {SPATIAL_SCHEMA_TOTAL_DIMENSION} != "
        f"{EXPECTED_CURRENT_SPATIAL_DIMENSION}"
    )


class SpatialSchemaError(ValueError):
    """Raised when any spatial producer/exporter/loader contract diverges."""


def canonical_spatial_schema_payload() -> dict[str, Any]:
    """Return the exact hash-defining current spatial tensor authority."""

    return {
        "schema_id": SPATIAL_SCHEMA_ID,
        "schema_version": SPATIAL_SCHEMA_VERSION,
        "dtype": SPATIAL_SCHEMA_DTYPE,
        "policy": SPATIAL_SCHEMA_POLICY,
        "ordered_group_names": list(SPATIAL_PREDICTIVE_GROUP_NAMES),
        "groups": [
            {
                "group_name": group,
                "group_schema_version": SPATIAL_GROUP_SCHEMA_VERSIONS[group],
                "group_schema_hash": SPATIAL_GROUP_SCHEMA_HASHES[group],
                "ordered_feature_names": list(
                    SPATIAL_PREDICTIVE_FEATURES[group]
                ),
                **SPATIAL_GROUP_CONTRACTS[group],
            }
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        ],
        "total_dimension": SPATIAL_SCHEMA_TOTAL_DIMENSION,
        "individual_feature_optionality": "FORBIDDEN",
        "structural_group_optionality": "FORBIDDEN",
    }


def spatial_schema_hash() -> str:
    """Return SHA-256 over all tensor-defining spatial semantics."""

    encoded = json.dumps(
        canonical_spatial_schema_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SPATIAL_SCHEMA_HASH = spatial_schema_hash()


def spatial_schema_metadata() -> dict[str, Any]:
    """Return exact schema metadata for sidecars and model contracts."""

    return {
        **canonical_spatial_schema_payload(),
        "schema_hash": SPATIAL_SCHEMA_HASH,
        "group_dimensions": {
            group: len(SPATIAL_PREDICTIVE_FEATURES[group])
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        },
        "group_feature_names": {
            group: list(SPATIAL_PREDICTIVE_FEATURES[group])
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        },
        "group_schema_versions": dict(SPATIAL_GROUP_SCHEMA_VERSIONS),
        "group_schema_hashes": dict(SPATIAL_GROUP_SCHEMA_HASHES),
    }


def canonical_spatial_feature_groups() -> dict[str, list[str]]:
    """Return a mutable copy without weakening the immutable authority."""

    return {
        group: list(SPATIAL_PREDICTIVE_FEATURES[group])
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
    }


def spatial_tensor_content_hash(arrays: Mapping[str, Any]) -> str:
    """Hash tensor content deterministically, independent of NPZ timestamps."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_current_spatial_tensor_bundle(
    npz_path: Path,
    audit_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one current bundle only after schema and content-hash validation."""

    if not npz_path.exists():
        raise FileNotFoundError(f"spatial tensor bundle not found: {npz_path}")
    if not audit_path.exists():
        raise FileNotFoundError(
            f"spatial tensor audit sidecar not found: {audit_path}"
        )
    with np.load(npz_path, allow_pickle=False) as bundle:
        arrays = {name: value.copy() for name, value in bundle.items()}
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpatialSchemaError("spatial tensor audit must be a JSON object")
    feature_names = payload.get("feature_names")
    metadata = payload.get("spatial_schema")
    if not isinstance(feature_names, dict) or not isinstance(metadata, dict):
        raise SpatialSchemaError(
            "spatial tensor audit lacks current feature_names or "
            "spatial_schema metadata"
        )
    require_spatial_tensor_bundle(
        arrays=arrays,
        feature_names=feature_names,
        metadata=metadata,
    )
    observed_hash = payload.get("spatial_tensor_content_hash")
    actual_hash = spatial_tensor_content_hash(arrays)
    if observed_hash != actual_hash:
        raise SpatialSchemaError(
            "spatial tensor content hash mismatch: "
            f"{observed_hash!r}:{actual_hash!r}"
        )
    return arrays, payload


def spatial_schema_preflight(
    *,
    source_columns: Sequence[Any],
    actual_feature_groups: Mapping[Any, Sequence[Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare producer columns and declared order to the complete authority."""

    expected_groups = list(SPATIAL_PREDICTIVE_GROUP_NAMES)
    actual_groups = list(actual_feature_groups)
    source = set(source_columns)
    errors: list[str] = []
    group_results: dict[str, dict[str, Any]] = {}

    if actual_groups != expected_groups:
        errors.append(
            "spatial_group_order_mismatch="
            f"{actual_groups!r}:{expected_groups!r}"
        )

    for raw_group in actual_groups:
        if not isinstance(raw_group, str) or not raw_group:
            errors.append(f"invalid_spatial_group_name={raw_group!r}")
    unexpected_groups = [
        group for group in actual_groups if group not in expected_groups
    ]
    missing_groups = [
        group for group in expected_groups if group not in actual_groups
    ]
    if unexpected_groups:
        errors.append(f"unexpected_spatial_groups={unexpected_groups!r}")
    if missing_groups:
        errors.append(f"missing_spatial_groups={missing_groups!r}")

    for group in expected_groups:
        expected_names = list(SPATIAL_PREDICTIVE_FEATURES[group])
        raw_names = list(actual_feature_groups.get(group, ()))
        non_string = [
            index
            for index, name in enumerate(raw_names)
            if not isinstance(name, str)
        ]
        blank = [
            index
            for index, name in enumerate(raw_names)
            if isinstance(name, str) and not name
        ]
        whitespace_names = [
            index
            for index, name in enumerate(raw_names)
            if isinstance(name, str) and name != name.strip()
        ]
        names = [
            name if isinstance(name, str) else f"<NON_STRING:{name!r}>"
            for name in raw_names
        ]
        duplicates = sorted(
            {
                name
                for name in names
                if names.count(name) > 1
            }
        )
        missing_source = [
            name for name in expected_names if name not in source
        ]
        unexpected = [
            name for name in names if name not in expected_names
        ]
        exact_order = names == expected_names
        group_errors: list[str] = []
        if non_string:
            group_errors.append(f"null_or_non_string_names={non_string}")
        if blank:
            group_errors.append(f"blank_names={blank}")
        if whitespace_names:
            group_errors.append(
                f"whitespace_normalized_names={whitespace_names}"
            )
        if duplicates:
            group_errors.append(f"duplicate_names={duplicates!r}")
        if missing_source:
            group_errors.append(
                f"missing_required_source_columns={missing_source!r}"
            )
        if unexpected:
            group_errors.append(f"unexpected_names={unexpected!r}")
        if not exact_order:
            group_errors.append(
                f"ordered_names_mismatch={names!r}:{expected_names!r}"
            )
        expected_dimension = len(expected_names)
        if len(names) != expected_dimension:
            group_errors.append(
                f"dimension_mismatch={len(names)}:{expected_dimension}"
            )
        if group_errors:
            errors.extend(f"{group}:{item}" for item in group_errors)
        group_results[group] = {
            "expected_feature_names": expected_names,
            "actual_feature_names": names,
            "expected_dimension": expected_dimension,
            "actual_dimension": len(names),
            "errors": group_errors,
        }

    observed_metadata = dict(metadata or spatial_schema_metadata())
    expected_metadata = spatial_schema_metadata()
    for field in (
        "schema_id",
        "schema_version",
        "dtype",
        "policy",
        "total_dimension",
        "schema_hash",
        "ordered_group_names",
        "group_dimensions",
        "group_feature_names",
        "groups",
        "group_schema_versions",
        "group_schema_hashes",
    ):
        if observed_metadata.get(field) != expected_metadata[field]:
            errors.append(
                f"spatial_schema_{field}_mismatch="
                f"{observed_metadata.get(field)!r}:"
                f"{expected_metadata[field]!r}"
            )

    return {
        "schema_id": SPATIAL_SCHEMA_ID,
        "schema_version": SPATIAL_SCHEMA_VERSION,
        "schema_hash": SPATIAL_SCHEMA_HASH,
        "expected_total_dimension": SPATIAL_SCHEMA_TOTAL_DIMENSION,
        "actual_total_dimension": sum(
            result["actual_dimension"] for result in group_results.values()
        ),
        "expected_group_names": expected_groups,
        "actual_group_names": actual_groups,
        "groups": group_results,
        "errors": errors,
    }


def require_spatial_schema(
    *,
    source_columns: Sequence[Any],
    actual_feature_groups: Mapping[Any, Sequence[Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a clean all-group preflight or fail before tensor creation."""

    result = spatial_schema_preflight(
        source_columns=source_columns,
        actual_feature_groups=actual_feature_groups,
        metadata=metadata,
    )
    if result["errors"]:
        raise SpatialSchemaError(
            "Spatial schema preflight failed: "
            + "; ".join(result["errors"])
        )
    return result


def require_spatial_tensor_bundle(
    *,
    arrays: Mapping[str, Any],
    feature_names: Mapping[Any, Sequence[Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind sidecar order, tensor widths/dtypes, and current schema metadata."""

    ordered_feature_names = {
        group: feature_names[group]
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        if group in feature_names
    }
    ordered_feature_names.update(
        {
            group: names
            for group, names in feature_names.items()
            if group not in SPATIAL_PREDICTIVE_GROUP_NAMES
        }
    )
    result = require_spatial_schema(
        source_columns=[
            feature
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
            for feature in SPATIAL_PREDICTIVE_FEATURES[group]
        ],
        actual_feature_groups=ordered_feature_names,
        metadata=metadata,
    )
    errors: list[str] = []
    tensor_shapes: dict[str, list[int]] = {}
    tensor_dtypes: dict[str, str] = {}
    mask_shapes: dict[str, list[int]] = {}
    first_two_dimensions: tuple[int, int] | None = None
    for group in SPATIAL_PREDICTIVE_GROUP_NAMES:
        if group not in arrays:
            errors.append(f"missing_spatial_tensor_group={group}")
            continue
        array = arrays[group]
        shape = [int(value) for value in getattr(array, "shape", ())]
        dtype = str(getattr(array, "dtype", ""))
        tensor_shapes[group] = shape
        tensor_dtypes[group] = dtype
        if len(shape) != 3:
            errors.append(f"spatial_tensor_rank_mismatch={group}:{shape}")
        elif shape[-1] != len(SPATIAL_PREDICTIVE_FEATURES[group]):
            errors.append(
                "spatial_tensor_dimension_mismatch="
                f"{group}:{shape[-1]}:"
                f"{len(SPATIAL_PREDICTIVE_FEATURES[group])}"
            )
        if dtype != SPATIAL_SCHEMA_DTYPE:
            errors.append(
                f"spatial_tensor_dtype_mismatch={group}:{dtype}:"
                f"{SPATIAL_SCHEMA_DTYPE}"
            )
        if len(shape) >= 2:
            observed_first_two = (shape[0], shape[1])
            if first_two_dimensions is None:
                first_two_dimensions = observed_first_two
            elif observed_first_two != first_two_dimensions:
                errors.append(
                    "spatial_tensor_sequence_shape_mismatch="
                    f"{group}:{observed_first_two}:{first_two_dimensions}"
                )

        mask_name = SPATIAL_GROUP_CONTRACTS[group]["availability_mask"]
        if mask_name not in arrays:
            errors.append(
                f"missing_spatial_availability_mask={group}:{mask_name}"
            )
            continue
        mask = np.asarray(arrays[mask_name])
        mask_shape = [int(value) for value in mask.shape]
        mask_shapes[str(mask_name)] = mask_shape
        if first_two_dimensions is not None and tuple(mask.shape[:2]) != (
            first_two_dimensions
        ):
            errors.append(
                "spatial_availability_mask_shape_mismatch="
                f"{group}:{mask_shape}:{first_two_dimensions}"
            )
        if str(mask.dtype) != SPATIAL_SCHEMA_DTYPE:
            errors.append(
                "spatial_availability_mask_dtype_mismatch="
                f"{group}:{mask.dtype}:{SPATIAL_SCHEMA_DTYPE}"
            )
        if mask.size and not np.isin(mask, (0.0, 1.0)).all():
            errors.append(
                f"spatial_availability_mask_not_binary={group}:{mask_name}"
            )
        values = np.asarray(array)
        if group == "roi_class_relation" and mask.ndim == 3:
            if mask.shape[-1] != 3:
                errors.append(
                    f"roi_availability_class_dimension={mask.shape[-1]}:3"
                )
            else:
                for class_index in range(3):
                    feature_slice = slice(
                        class_index * 6,
                        (class_index + 1) * 6,
                    )
                    unavailable = mask[:, :, class_index] == 0.0
                    if np.any(values[:, :, feature_slice][unavailable] != 0.0):
                        errors.append(
                            "roi_placeholder_nonzero_when_unavailable="
                            f"class_index:{class_index}"
                        )
        elif mask.ndim == 2:
            unavailable = mask == 0.0
            if np.any(values[unavailable] != 0.0):
                errors.append(
                    f"placeholder_nonzero_when_unavailable={group}"
                )
    if errors:
        raise SpatialSchemaError(
            "Spatial tensor bundle preflight failed: " + "; ".join(errors)
        )
    return {
        **result,
        "tensor_shapes": tensor_shapes,
        "tensor_dtypes": tensor_dtypes,
        "availability_mask_shapes": mask_shapes,
    }


__all__ = [
    "EXPECTED_CURRENT_SPATIAL_DIMENSION",
    "SPATIAL_GROUP_CONTRACTS",
    "SPATIAL_GROUP_SCHEMA_HASHES",
    "SPATIAL_GROUP_SCHEMA_VERSIONS",
    "SPATIAL_PREDICTIVE_FEATURES",
    "SPATIAL_PREDICTIVE_GROUP_NAMES",
    "SPATIAL_SCHEMA_DTYPE",
    "SPATIAL_SCHEMA_HASH",
    "SPATIAL_SCHEMA_ID",
    "SPATIAL_SCHEMA_POLICY",
    "SPATIAL_SCHEMA_TOTAL_DIMENSION",
    "SPATIAL_SCHEMA_VERSION",
    "SpatialSchemaError",
    "canonical_spatial_feature_groups",
    "canonical_spatial_schema_payload",
    "load_current_spatial_tensor_bundle",
    "require_spatial_schema",
    "require_spatial_tensor_bundle",
    "spatial_schema_hash",
    "spatial_schema_metadata",
    "spatial_schema_preflight",
    "spatial_tensor_content_hash",
]

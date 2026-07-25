"""Fail-closed validator for the Classification V2 scientific contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

STATUS_VOCABULARY = {
    "IMPLEMENTED_AND_TESTED",
    "IMPLEMENTED_PARTIALLY_TESTED",
    "IMPLEMENTED_NOT_TESTED",
    "DECLARED_NOT_IMPLEMENTED",
    "IMPLEMENTATION_DIFFERS_FROM_CONTRACT",
    "IMPLEMENTED_WITHOUT_CONTRACT",
    "UNKNOWN_REQUIRES_REVIEW",
    "DEPRECATED",
    "REVIEW_ONLY",
    "MODEL_FORBIDDEN",
}

REQUIRED_SECTIONS = {
    "contract_metadata",
    "status_vocabulary",
    "dataset_instance_evidence",
    "object_track_key_contract",
    "artifacts",
    "stage_defaults",
    "stages",
    "feature_defaults",
    "features",
    "invariant_defaults",
    "invariants",
    "model_schemas",
    "missingness_categories",
    "assumptions",
    "golden_case_defaults",
    "golden_cases",
    "implementation_inventory",
    "known_gaps",
    "change_impact_questions",
    "artifact_invalidation_rules",
    "independent_review",
    "remediation_phases",
}

FEATURE_FIELDS = {
    "feature_id",
    "feature_name",
    "feature_family",
    "producer_stage",
    "formula_latex",
    "formula_plain",
    "required_inputs",
    "output_dtype",
    "units",
    "coordinate_system",
    "normalization",
    "computation_grain",
    "grouping_key",
    "pair_reset_key",
    "validity_mask",
    "availability_mask",
    "missing_value_semantics",
    "zero_value_semantics",
    "denominator",
    "aggregation_rule",
    "minimum_valid_observations",
    "no_valid_observation_behavior",
    "physical_interpretation",
    "is_physical_measurement",
    "review_eligible",
    "model_eligible",
    "model_group",
    "leakage_risk",
    "threshold_dependencies",
    "schema_version",
    "code_locations",
    "test_locations",
    "implementation_status",
    "known_limitations",
}

STAGE_FIELDS = {
    "stage_id",
    "stage_name",
    "purpose",
    "input_artifacts",
    "output_artifacts",
    "input_grain",
    "output_grain",
    "grouping_keys",
    "canonical_identity_keys",
    "identity_contract_ids",
    "pair_reset_key",
    "temporal_support",
    "required_columns",
    "produced_columns",
    "forbidden_columns",
    "schema_version",
    "evidence_semantics_version",
    "model_eligibility",
    "review_eligibility",
    "deterministic_ordering",
    "missingness_policy",
    "failure_policy",
    "checker",
    "tests",
    "audit_artifact",
    "code_locations",
    "current_status",
}

INVARIANT_FIELDS = {
    "invariant_id",
    "stage_id",
    "invariant_description",
    "scientific_reason",
    "severity",
    "fatal",
    "input_scope",
    "expected_condition",
    "failure_condition",
    "checker",
    "audit_field",
    "unit_test",
    "integration_test",
    "golden_case_ids",
    "code_locations",
    "implementation_status",
    "known_gap",
}

GOLDEN_FIELDS = {
    "case_id",
    "scientific_purpose",
    "input_rows",
    "image_dimensions",
    "timestamps",
    "identities",
    "temporal_unit_key",
    "geometry_validity",
    "roi_availability",
    "expected_pair_masks",
    "expected_formulas",
    "expected_numerical_values",
    "expected_aggregate_values",
    "expected_selected_neighbor",
    "expected_audit_coverage",
    "tolerance",
    "related_invariants",
}

PAIR_FAMILIES = {"motion", "social_temporal", "roi_temporal"}
AGGREGATE_FAMILIES = {"motion_aggregate", "social_aggregate", "roi_aggregate"}


def load_contract(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML without a non-standard dependency."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON-compatible YAML: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("contract root must be a mapping")
    return payload


def expand_entities(
    contract: dict[str, Any],
    section: str,
    defaults_section: str,
) -> list[dict[str, Any]]:
    """Apply declared defaults before registry validation or rendering."""

    defaults = contract.get(defaults_section, {})
    return [{**defaults, **item} for item in contract.get(section, [])]


def canonical_schema_payload(schema: dict[str, Any]) -> dict[str, Any]:
    """Return fields that define tensor meaning and deterministic hash."""

    return {
        "schema_id": schema["schema_id"],
        "schema_version": schema["schema_version"],
        "dtype": schema["dtype"],
        "ordered_feature_names": schema["ordered_feature_names"],
        "validity_masks": schema["validity_masks"],
        "aggregation_outputs": schema["aggregation_outputs"],
    }


def schema_hash(schema: dict[str, Any]) -> str:
    encoded = json.dumps(
        canonical_schema_payload(schema),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _duplicate_ids(items: list[dict[str, Any]], field: str) -> list[str]:
    values = [str(item.get(field, "")).strip() for item in items]
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _require_fields(
    items: list[dict[str, Any]],
    fields: set[str],
    id_field: str,
    errors: list[str],
) -> None:
    for item in items:
        item_id = str(item.get(id_field, "<missing>"))
        missing = sorted(fields.difference(item))
        if missing:
            errors.append(f"{item_id}:missing_fields={missing}")


def _validate_status(
    items: list[dict[str, Any]],
    field: str,
    errors: list[str],
) -> None:
    for item in items:
        value = item.get(field)
        if value not in STATUS_VOCABULARY:
            errors.append(
                f"{item.get(next(iter(item)), '<unknown>')}:invalid_status={value}"
            )


def _object_track_key_contract_errors(
    contract: dict[str, Any],
    stages: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    identity = contract["object_track_key_contract"]
    required_fields = {
        "schema_id",
        "schema_version",
        "identity_scope_components",
        "existing_key_field",
        "identity_fallback_order",
        "identity_discriminators",
        "component_order",
        "component_names",
        "component_delimiter",
        "name_value_delimiter",
        "serialization_templates",
        "escaping_policy",
        "blank_policy",
        "integer_string_policy",
        "unicode_policy",
        "pig_id_authoritative",
        "row_order_authoritative",
        "absolute_path_authoritative",
        "random_value_authoritative",
    }
    if not isinstance(identity, dict):
        return ["object_track_key_contract:not_mapping"]
    missing = sorted(required_fields.difference(identity))
    if missing:
        errors.append(f"object_track_key_contract:missing_fields={missing}")
        return errors
    expected_values = {
        "schema_id": "schema.classification_v2.object_track_key",
        "schema_version": "classification_v2.object_track_key.v1",
        "identity_scope_components": [
            "source_type",
            "dataset_id",
            "video_key",
        ],
        "existing_key_field": "object_track_key",
        "identity_fallback_order": [
            "track_id",
            "object_id_in_image",
            "object_id",
        ],
        "identity_discriminators": {
            "track_id": "track_id",
            "object_id_in_image": "object_id",
            "object_id": "object_id",
        },
        "component_order": ["source", "dataset", "video", "identity"],
        "component_names": {
            "source_type": "source",
            "dataset_id": "dataset",
            "video_key": "video",
        },
        "component_delimiter": "|",
        "name_value_delimiter": "=",
        "serialization_templates": {
            "track_id": (
                "source={source}|dataset={dataset}|video={video}|"
                "track_id={value}"
            ),
            "object_id": (
                "source={source}|dataset={dataset}|video={video}|"
                "object_id={value}"
            ),
        },
    }
    for field, expected in expected_values.items():
        if identity[field] != expected:
            errors.append(
                f"object_track_key_contract:{field}_mismatch"
            )
    escaping = identity["escaping_policy"]
    if escaping != {
        "algorithm": "RFC3986_PERCENT_ENCODING",
        "encoding": "UTF-8",
        "safe_characters": "-_.~",
        "hex_case": "UPPER",
    }:
        errors.append("object_track_key_contract:escaping_policy_mismatch")
    blank = identity["blank_policy"]
    if blank != {
        "normalization": "STRINGIFY_TRIM_UNICODE_WHITESPACE",
        "null_tokens": ["", "nan", "None", "<NA>"],
        "scope_components": "FAIL_CLOSED",
        "identity_components": "FALL_THROUGH_THEN_FAIL_CLOSED",
        "existing_key": "PRESERVE_AND_VALIDATE_WHEN_DERIVABLE",
    }:
        errors.append("object_track_key_contract:blank_policy_mismatch")
    for field in (
        "pig_id_authoritative",
        "row_order_authoritative",
        "absolute_path_authoritative",
        "random_value_authoritative",
    ):
        if identity[field] is not False:
            errors.append(f"object_track_key_contract:{field}_must_be_false")
    schema_id = str(identity["schema_id"])
    for stage in stages:
        unknown = set(stage["identity_contract_ids"]) - {schema_id}
        if unknown:
            errors.append(
                f"{stage['stage_id']}:unknown_identity_contracts="
                f"{sorted(unknown)}"
            )
    required_stages = {
        "stage.legacy_cvat_source_merge",
        "stage.frame_local_primitives",
    }
    declared_stages = {
        stage["stage_id"]
        for stage in stages
        if schema_id in stage["identity_contract_ids"]
    }
    if not required_stages.issubset(declared_stages):
        errors.append(
            "object_track_key_contract:missing_stage_bindings="
            f"{sorted(required_stages - declared_stages)}"
        )
    return errors


def _reference_errors(
    contract: dict[str, Any],
    stages: list[dict[str, Any]],
    features: list[dict[str, Any]],
    invariants: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    stage_ids = {item["stage_id"] for item in stages}
    feature_ids = {item["feature_id"] for item in features}
    invariant_ids = {item["invariant_id"] for item in invariants}
    artifact_ids = {item["artifact_id"] for item in contract["artifacts"]}
    case_ids = {item["case_id"] for item in contract["golden_cases"]}

    for stage in stages:
        for artifact in [
            *stage["input_artifacts"],
            *stage["output_artifacts"],
        ]:
            if artifact not in artifact_ids:
                errors.append(
                    f"{stage['stage_id']}:unresolved_artifact={artifact}"
                )
    for feature in features:
        if feature["producer_stage"] not in stage_ids:
            errors.append(
                f"{feature['feature_id']}:unresolved_stage="
                f"{feature['producer_stage']}"
            )
    for invariant in invariants:
        referenced_stages = invariant["stage_id"]
        if isinstance(referenced_stages, str):
            referenced_stages = [referenced_stages]
        for stage_id in referenced_stages:
            if stage_id not in stage_ids:
                errors.append(
                    f"{invariant['invariant_id']}:unresolved_stage={stage_id}"
                )
        for case_id in invariant["golden_case_ids"]:
            if case_id not in case_ids:
                errors.append(
                    f"{invariant['invariant_id']}:unresolved_case={case_id}"
                )
    for case in contract["golden_cases"]:
        for invariant_id in case["related_invariants"]:
            if invariant_id not in invariant_ids:
                errors.append(
                    f"{case['case_id']}:unresolved_invariant={invariant_id}"
                )
    for gap in contract["known_gaps"]:
        if gap["affected_stage"] not in stage_ids:
            errors.append(
                f"{gap['gap_id']}:unresolved_stage={gap['affected_stage']}"
            )
        for feature_id in gap["affected_features"]:
            if feature_id not in feature_ids:
                errors.append(
                    f"{gap['gap_id']}:unresolved_feature={feature_id}"
                )
    all_contract_ids = stage_ids | feature_ids | invariant_ids
    for item in contract["implementation_inventory"]:
        for contract_id in item["contract_ids"]:
            if contract_id not in all_contract_ids:
                errors.append(
                    f"{item['implementation_id']}:unresolved_contract="
                    f"{contract_id}"
                )
    return errors


def _semantic_errors(features: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for feature in features:
        feature_id = feature["feature_id"]
        if feature["model_eligible"]:
            for field in (
                "formula_plain",
                "units",
                "validity_mask",
                "missing_value_semantics",
            ):
                if feature[field] in ("", None, []):
                    errors.append(
                        f"{feature_id}:model_feature_missing_{field}"
                    )
        if feature["feature_family"] in PAIR_FAMILIES:
            if not str(feature["pair_reset_key"]).strip():
                errors.append(f"{feature_id}:pair_feature_without_reset_key")
        if feature["feature_family"] in AGGREGATE_FAMILIES:
            if not str(feature["denominator"]).strip():
                errors.append(f"{feature_id}:aggregate_without_denominator")
        coordinate = str(feature["coordinate_system"]).casefold()
        if "image" in coordinate and not feature["is_physical_measurement"]:
            interpretation = str(feature["physical_interpretation"]).casefold()
            if not any(
                marker in interpretation
                for marker in ("not physical", "image-coordinate", "pixel")
            ):
                errors.append(
                    f"{feature_id}:image_coordinate_physical_limit_undeclared"
                )
        if feature["review_eligible"] and feature["model_eligible"]:
            if feature["implementation_status"] in {
                "REVIEW_ONLY",
                "MODEL_FORBIDDEN",
            }:
                errors.append(
                    f"{feature_id}:review_model_eligibility_conflict"
                )
        if feature["feature_name"].startswith(("target_roi_", "roi_target_")):
            if feature["model_eligible"]:
                errors.append(f"{feature_id}:target_roi_leakage")
    return errors


def _schema_errors(
    contract: dict[str, Any],
    features: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    feature_by_name = {
        feature["feature_name"]: feature
        for feature in features
    }
    for schema in contract["model_schemas"]:
        schema_id = schema.get("schema_id", "")
        if schema_id in ids:
            errors.append(f"duplicate_schema_id={schema_id}")
        ids.add(schema_id)
        names = schema.get("ordered_feature_names", [])
        if not names:
            errors.append(f"{schema_id}:empty_ordered_feature_names")
        if len(names) != len(set(names)):
            errors.append(f"{schema_id}:duplicate_feature_names")
        if schema.get("dimension") != len(names):
            errors.append(
                f"{schema_id}:dimension_mismatch="
                f"{schema.get('dimension')}:{len(names)}"
            )
        actual_hash = schema_hash(schema)
        if schema.get("schema_hash") != actual_hash:
            errors.append(
                f"{schema_id}:schema_hash_mismatch="
                f"{schema.get('schema_hash')}:{actual_hash}"
            )
        if schema.get("missing_feature_policy") != "FAIL_CLOSED":
            errors.append(f"{schema_id}:missing_feature_policy_not_fail_closed")
        violations = set(schema.get("schema_violations", []))
        required = {
            "MISSING",
            "REORDERED",
            "DUPLICATED",
            "UNEXPECTED",
        }
        if not required.issubset(violations):
            errors.append(f"{schema_id}:incomplete_schema_violations")
        referenced_names = [
            *names,
            *schema.get("validity_masks", []),
            *schema.get("aggregation_outputs", []),
        ]
        unresolved = sorted(
            name for name in referenced_names if name not in feature_by_name
        )
        if unresolved:
            errors.append(f"{schema_id}:unresolved_features={unresolved}")
        ineligible = sorted(
            name
            for name in names
            if name in feature_by_name
            and not feature_by_name[name]["model_eligible"]
        )
        if ineligible:
            errors.append(f"{schema_id}:model_ineligible_features={ineligible}")
    return errors


def _implemented_test_errors(
    stages: list[dict[str, Any]],
    features: list[dict[str, Any]],
    invariants: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for stage in stages:
        if (
            stage["current_status"] == "IMPLEMENTED_AND_TESTED"
            and not stage["tests"]
        ):
            errors.append(f"{stage['stage_id']}:tested_status_without_tests")
    for feature in features:
        if (
            feature["implementation_status"] == "IMPLEMENTED_AND_TESTED"
            and not feature["test_locations"]
        ):
            errors.append(f"{feature['feature_id']}:tested_status_without_tests")
    for invariant in invariants:
        if invariant["implementation_status"] == "IMPLEMENTED_AND_TESTED":
            if not invariant["unit_test"] and not invariant["integration_test"]:
                errors.append(
                    f"{invariant['invariant_id']}:tested_status_without_tests"
                )
    return errors


def _motion_values(case: dict[str, Any]) -> dict[str, Any]:
    rows = case["input_rows"]
    timestamps = [float(row["timestamp_sec"]) for row in rows]
    xs = [float(row["cx_n"]) for row in rows]
    ys = [float(row["cy_n"]) for row in rows]
    units = [str(row["temporal_unit_key"]) for row in rows]
    identities = [str(row["object_track_key"]) for row in rows]
    valid_geometry = [bool(row.get("geometry_valid", True)) for row in rows]
    masks = [False]
    vx: list[float | None] = [None]
    vy: list[float | None] = [None]
    speed: list[float | None] = [None]
    direction: list[float | None] = [None]
    for index in range(1, len(rows)):
        delta_t = timestamps[index] - timestamps[index - 1]
        valid = (
            delta_t > 0
            and units[index] == units[index - 1]
            and identities[index] == identities[index - 1]
            and valid_geometry[index]
            and valid_geometry[index - 1]
        )
        masks.append(valid)
        if not valid:
            vx.append(None)
            vy.append(None)
            speed.append(None)
            direction.append(None)
            continue
        vx_value = (xs[index] - xs[index - 1]) / delta_t
        vy_value = (ys[index] - ys[index - 1]) / delta_t
        vx.append(vx_value)
        vy.append(vy_value)
        speed.append(math.hypot(vx_value, vy_value))
        direction.append(math.atan2(vy_value, vx_value))
    initial = [None] * min(2, len(rows))
    direction_change: list[float | None] = list(initial)
    tangential: list[float | None] = list(initial)
    ax: list[float | None] = list(initial)
    ay: list[float | None] = list(initial)
    vector_magnitude: list[float | None] = list(initial)
    for index in range(2, len(rows)):
        delta_t = timestamps[index] - timestamps[index - 1]
        valid = masks[index] and masks[index - 1] and delta_t > 0
        if not valid:
            direction_change.append(None)
            tangential.append(None)
            ax.append(None)
            ay.append(None)
            vector_magnitude.append(None)
            continue
        raw_direction = float(direction[index]) - float(direction[index - 1])
        wrapped = (raw_direction + math.pi) % (2.0 * math.pi) - math.pi
        ax_value = (float(vx[index]) - float(vx[index - 1])) / delta_t
        ay_value = (float(vy[index]) - float(vy[index - 1])) / delta_t
        direction_change.append(wrapped)
        tangential.append(
            (float(speed[index]) - float(speed[index - 1])) / delta_t
        )
        ax.append(ax_value)
        ay.append(ay_value)
        vector_magnitude.append(math.hypot(ax_value, ay_value))
    valid_speeds = [
        value
        for value, valid in zip(speed, masks, strict=True)
        if valid and value is not None
    ]
    return {
        "valid_motion_pair": masks,
        "vx_n_per_second": vx,
        "vy_n_per_second": vy,
        "speed_n_per_second": speed,
        "direction_rad": direction,
        "direction_change_rad": direction_change,
        "tangential_acceleration_n_per_second2": tangential,
        "ax_n_per_second2": ax,
        "ay_n_per_second2": ay,
        "acceleration_vector_magnitude_n_per_second2": vector_magnitude,
        "valid_pair_count": len(valid_speeds),
        "possible_pair_count": max(0, len(rows) - 1),
        "valid_pair_ratio": (
            len(valid_speeds) / max(1, len(rows) - 1)
        ),
        "speed_mean_valid_pairs": (
            sum(valid_speeds) / len(valid_speeds) if valid_speeds else 0.0
        ),
        "motion_available": bool(valid_speeds),
    }


def _compare_number(
    expected: Any,
    actual: Any,
    tolerance: float,
    path: str,
    errors: list[str],
) -> None:
    if expected is None or actual is None:
        if expected is not actual:
            errors.append(f"{path}:expected={expected}:actual={actual}")
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if bool(expected) != bool(actual):
            errors.append(f"{path}:expected={expected}:actual={actual}")
        return
    if not math.isclose(
        float(expected),
        float(actual),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        errors.append(f"{path}:expected={expected}:actual={actual}")


def _compare_nested(
    expected: Any,
    actual: Any,
    tolerance: float,
    path: str,
    errors: list[str],
) -> None:
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            errors.append(f"{path}:shape_mismatch")
            return
        for index, (left, right) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _compare_nested(
                left,
                right,
                tolerance,
                f"{path}[{index}]",
                errors,
            )
        return
    _compare_number(expected, actual, tolerance, path, errors)


def _golden_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cases = expand_entities(
        contract,
        "golden_cases",
        "golden_case_defaults",
    )
    _require_fields(cases, GOLDEN_FIELDS, "case_id", errors)
    for case in cases:
        case_id = case["case_id"]
        tolerance = float(case["tolerance"])
        calculation = case.get("calculation", {})
        kind = calculation.get("kind")
        if kind == "motion":
            actual = _motion_values(case)
            expected = {
                **case["expected_pair_masks"],
                **case["expected_numerical_values"],
                **case["expected_aggregate_values"],
                **case["expected_audit_coverage"],
            }
            for key, value in expected.items():
                if key in actual:
                    _compare_nested(
                        value,
                        actual[key],
                        tolerance,
                        f"{case_id}:{key}",
                        errors,
                    )
        elif kind == "distance":
            width = float(case["image_dimensions"]["width_px"])
            height = float(case["image_dimensions"]["height_px"])
            dx = float(calculation["dx_px"])
            dy = float(calculation["dy_px"])
            actual = {
                "axis_normalized": math.hypot(dx / width, dy / height),
                "diagonal_normalized": math.hypot(dx, dy)
                / math.hypot(width, height),
            }
            for key, value in case["expected_numerical_values"].items():
                _compare_number(
                    value,
                    actual[key],
                    tolerance,
                    f"{case_id}:{key}",
                    errors,
                )
        elif kind == "roi":
            available = [bool(row["roi_available"]) for row in case["input_rows"]]
            contact = [bool(row["roi_contact"]) for row in case["input_rows"]]
            available_count = sum(available)
            contact_count = sum(
                is_available and is_contact
                for is_available, is_contact in zip(
                    available,
                    contact,
                    strict=True,
                )
            )
            actual = {
                "target_roi_available_frame_count": available_count,
                "target_roi_availability_ratio_unit": (
                    available_count / len(available) if available else 0.0
                ),
                "target_roi_contact_ratio_unit": (
                    contact_count / available_count
                    if available_count
                    else 0.0
                ),
                "target_roi_contact_available": bool(available_count),
            }
            for key, value in {
                **case["expected_numerical_values"],
                **case["expected_aggregate_values"],
                **case["expected_audit_coverage"],
            }.items():
                if key in actual:
                    _compare_number(
                        value,
                        actual[key],
                        tolerance,
                        f"{case_id}:{key}",
                        errors,
                    )
        elif kind == "neighbor_tie":
            candidates = calculation["candidates"]
            selected = min(
                candidates,
                key=lambda item: (
                    float(item["distance"]),
                    str(item["object_track_key"]),
                    str(item["track_id"]),
                    str(item["object_id"]),
                ),
            )
            if selected["object_track_key"] != case["expected_selected_neighbor"]:
                errors.append(f"{case_id}:neighbor_tie_mismatch")
    return errors


def _generated_sync_errors(
    contract_path: Path,
    contract: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    root = contract_path.parent
    required_files = contract["contract_metadata"]["required_files"]
    for name in required_files:
        if not (root / name).exists():
            errors.append(f"missing_required_file={name}")
    diagram_path = root / "01_pipeline_dataflow.mmd"
    if diagram_path.exists():
        text = diagram_path.read_text(encoding="utf-8")
        declared = set(
            re.findall(r"stage\.[a-z0-9_]+", text)
        )
        expected = {stage["stage_id"] for stage in contract["stages"]}
        if declared != expected:
            errors.append(
                "diagram_stage_ids_mismatch="
                f"missing={sorted(expected - declared)}:"
                f"extra={sorted(declared - expected)}"
            )
    review_path = root / "13_independent_review_report.md"
    if review_path.exists():
        report = review_path.read_text(encoding="utf-8")
        blocking = {
            gap["gap_id"]
            for gap in contract["known_gaps"]
            if gap["severity"] in {"CRITICAL", "HIGH"}
            and gap["status"] != "RESOLVED"
        }
        hidden = sorted(gap_id for gap_id in blocking if gap_id not in report)
        if hidden:
            errors.append(f"blocking_gaps_missing_from_final_report={hidden}")
    return errors


def validate_contract(
    contract_path: Path,
    *,
    check_generated: bool = True,
) -> dict[str, Any]:
    """Validate structure, references, semantics, schemas and golden numbers."""

    contract = load_contract(contract_path)
    errors: list[str] = []
    missing_sections = sorted(REQUIRED_SECTIONS.difference(contract))
    if missing_sections:
        errors.append(f"missing_sections={missing_sections}")
        return {"valid": False, "errors": errors}
    if set(contract["status_vocabulary"]) != STATUS_VOCABULARY:
        errors.append("status_vocabulary_mismatch")

    stages = expand_entities(contract, "stages", "stage_defaults")
    features = expand_entities(contract, "features", "feature_defaults")
    invariants = expand_entities(
        contract,
        "invariants",
        "invariant_defaults",
    )
    _require_fields(stages, STAGE_FIELDS, "stage_id", errors)
    _require_fields(features, FEATURE_FIELDS, "feature_id", errors)
    _require_fields(invariants, INVARIANT_FIELDS, "invariant_id", errors)
    for items, field in (
        (stages, "stage_id"),
        (features, "feature_id"),
        (invariants, "invariant_id"),
        (contract["golden_cases"], "case_id"),
        (contract["known_gaps"], "gap_id"),
    ):
        duplicates = _duplicate_ids(items, field)
        if duplicates:
            errors.append(f"duplicate_{field}={duplicates}")
        blanks = [item for item in items if not str(item.get(field, "")).strip()]
        if blanks:
            errors.append(f"blank_{field}={len(blanks)}")
    _validate_status(stages, "current_status", errors)
    _validate_status(features, "implementation_status", errors)
    _validate_status(invariants, "implementation_status", errors)
    errors.extend(_reference_errors(contract, stages, features, invariants))
    errors.extend(_object_track_key_contract_errors(contract, stages))
    errors.extend(_semantic_errors(features))
    errors.extend(_schema_errors(contract, features))
    errors.extend(_implemented_test_errors(stages, features, invariants))
    errors.extend(_golden_errors(contract))
    if check_generated:
        errors.extend(_generated_sync_errors(contract_path, contract))

    status_counts = Counter(
        [stage["current_status"] for stage in stages]
        + [feature["implementation_status"] for feature in features]
        + [invariant["implementation_status"] for invariant in invariants]
    )
    gap_counts = Counter(gap["severity"] for gap in contract["known_gaps"])
    return {
        "valid": not errors,
        "errors": errors,
        "counts": {
            "stages": len(stages),
            "features": len(features),
            "invariants": len(invariants),
            "golden_cases": len(contract["golden_cases"]),
            "model_schemas": len(contract["model_schemas"]),
            "statuses": dict(sorted(status_counts.items())),
            "gaps": dict(sorted(gap_counts.items())),
        },
        "schema_hashes": {
            schema["schema_id"]: schema_hash(schema)
            for schema in contract["model_schemas"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "docs/classification_v2/scientific_contract_v1/"
            "00_pipeline_contract.yaml"
        ),
    )
    parser.add_argument("--skip-generated", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = validate_contract(
        args.contract,
        check_generated=not args.skip_generated,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["valid"]:
        print("PASS scientific contract")
    else:
        print("FAIL scientific contract")
        for error in result["errors"]:
            print(f"- {error}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())

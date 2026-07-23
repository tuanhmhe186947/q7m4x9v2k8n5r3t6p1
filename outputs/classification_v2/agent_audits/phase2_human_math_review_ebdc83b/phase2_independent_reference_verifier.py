"""Independent elementary-arithmetic verifier for Phase 2 motion evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-12
SCHEMA_ID = "schema.pig_strenet_motion_v2"
SCHEMA_VERSION = "classification_v2.motion_tensor.v2"
FEATURES = [
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
]
MASKS = [
    "valid_motion_pair",
    "velocity_valid",
    "bbox_rate_valid",
    "direction_valid",
    "direction_change_valid",
    "tangential_acceleration_valid",
    "vector_acceleration_valid",
    "motion_feature_available",
]
AGGREGATES = [
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
]
EXPECTED_HASH = (
    "ec0c511b5f5198240492be49c0492e543c9e38eb4a4ff446259b958c2a59963b"
)


def main() -> None:
    checks: list[dict[str, Any]] = []
    rows = _read_csv(ROOT / "phase2_golden_rows.csv")
    units = _read_csv(ROOT / "phase2_golden_unit_aggregates.csv")
    keyed = {(row["case_id"], int(row["row_index"])): row for row in rows}

    expected_values: dict[tuple[str, int], dict[str, Any]] = {
        ("stationary", 1): {
            "velocity_valid": True,
            "vx_n_per_second": 0.0,
            "vy_n_per_second": 0.0,
            "speed_n_per_second": 0.0,
            "direction_valid": False,
        },
        ("stationary", 2): {
            "tangential_acceleration_valid": True,
            "tangential_acceleration_n_per_second2": 0.0,
            "vector_acceleration_valid": True,
            "ax_n_per_second2": 0.0,
            "ay_n_per_second2": 0.0,
            "acceleration_vector_magnitude_n_per_second2": 0.0,
        },
        ("horizontal", 1): {
            "vx_n_per_second": 1.0,
            "vy_n_per_second": 0.0,
            "speed_n_per_second": 1.0,
        },
        ("horizontal", 2): {
            "direction_change_valid": True,
            "direction_change_rad": 0.0,
            "tangential_acceleration_n_per_second2": 0.0,
            "acceleration_vector_magnitude_n_per_second2": 0.0,
        },
        ("diagonal", 1): {
            "vx_n_per_second": 3.0,
            "vy_n_per_second": 4.0,
            "speed_n_per_second": 5.0,
        },
        ("irregular", 1): {
            "velocity_sample_time_sec": 0.5,
            "vx_n_per_second": 2.0,
        },
        ("irregular", 2): {
            "velocity_sample_time_sec": 2.0,
            "acceleration_delta_t_sec": 1.5,
            "tangential_acceleration_n_per_second2": 0.0,
            "ax_n_per_second2": 0.0,
        },
        ("width_only", 1): {
            "bw_rate_n_per_second": 0.1,
            "bh_rate_n_per_second": 0.0,
            "area_rate_n_per_second": 0.01,
        },
        ("height_only", 1): {
            "bw_rate_n_per_second": 0.0,
            "bh_rate_n_per_second": 0.05,
            "area_rate_n_per_second": 0.01,
        },
        ("area_change", 1): {
            "area_rate_n_per_second": 0.01,
        },
        ("aspect_change", 1): {
            "aspect_ratio_rate_per_second": 1.0,
        },
        ("invalid_pair", 1): {
            "valid_motion_pair": False,
            "velocity_valid": False,
            "bbox_rate_valid": False,
            "vector_acceleration_valid": False,
        },
        ("first_valid_velocity", 1): {
            "velocity_valid": True,
            "vector_acceleration_valid": False,
        },
        ("constant_velocity", 2): {
            "tangential_acceleration_valid": True,
            "tangential_acceleration_n_per_second2": 0.0,
            "vector_acceleration_valid": True,
            "acceleration_vector_magnitude_n_per_second2": 0.0,
        },
        ("speed_increase", 2): {
            "tangential_acceleration_n_per_second2": 1.0,
            "ax_n_per_second2": 1.0,
            "ay_n_per_second2": 0.0,
            "acceleration_vector_magnitude_n_per_second2": 1.0,
            "direction_change_rad": 0.0,
        },
        ("speed_decrease", 2): {
            "tangential_acceleration_n_per_second2": -1.0,
            "ax_n_per_second2": -1.0,
            "acceleration_vector_magnitude_n_per_second2": 1.0,
        },
        ("direction_change", 2): {
            "tangential_acceleration_n_per_second2": 0.0,
            "ax_n_per_second2": -1.0,
            "ay_n_per_second2": 1.0,
            "acceleration_vector_magnitude_n_per_second2": math.sqrt(2.0),
            "direction_change_rad": math.pi / 2.0,
        },
        ("angle_wrap", 2): {
            "direction_change_rad": math.radians(2.0),
        },
        ("zero_speed_direction", 1): {
            "velocity_valid": True,
            "speed_n_per_second": 0.0,
            "direction_valid": False,
            "direction_change_valid": False,
        },
        ("invalid_middle_velocity", 2): {
            "velocity_valid": False,
            "vector_acceleration_valid": False,
        },
        ("invalid_middle_velocity", 3): {
            "velocity_valid": False,
            "vector_acceleration_valid": False,
        },
        ("cross_unit", 2): {
            "valid_motion_pair": False,
            "velocity_valid": False,
            "vector_acceleration_valid": False,
        },
        ("actor_discontinuity", 1): {
            "valid_motion_pair": False,
            "velocity_valid": False,
        },
        ("all_velocity_invalid", 2): {
            "velocity_valid": False,
            "vector_acceleration_valid": False,
        },
        ("one_velocity_no_acceleration", 1): {
            "velocity_valid": True,
            "vector_acceleration_valid": False,
        },
        ("one_acceleration", 2): {
            "vector_acceleration_valid": True,
        },
    }
    for key, expected in expected_values.items():
        actual = keyed[key]
        for field, value in expected.items():
            _check(
                checks,
                f"golden_{key[0]}_{key[1]}_{field}",
                _equal(_value(actual[field]), value),
                {"actual": _value(actual[field]), "expected": value},
            )

    _verify_unit_denominators(checks, units)
    _verify_schema(checks)
    _verify_negative_schemas(checks)
    _verify_production_trace(checks)
    _verify_tensor_preflight(checks)

    errors = [
        check["check_id"] for check in checks if not check["passed"]
    ]
    result = {
        "schema_version": "classification_v2.phase2_independent_reference.v1",
        "implementation_commit": (
            "ebdc83bc942ba34dd4f820a6aba46f37233a04d6"
        ),
        "reference_method": (
            "Python standard-library booleans, subtraction, division, "
            "square root, atan2 and modular angle wrapping only"
        ),
        "imports_production_motion_functions": False,
        "imports_production_aggregate_functions": False,
        "calls_exporter_for_expected_values": False,
        "tolerance": TOLERANCE,
        "schema_hash": EXPECTED_HASH,
        "formulas": {
            "velocity_sample_time": "(t_previous+t_current)/2",
            "velocity": "(position_current-position_previous)/delta_t",
            "speed": "sqrt(vx^2+vy^2)",
            "acceleration_delta_t": (
                "current_velocity_midpoint-previous_velocity_midpoint"
            ),
            "tangential_acceleration": (
                "(speed_current-speed_previous)/acceleration_delta_t"
            ),
            "ax": "(vx_current-vx_previous)/acceleration_delta_t",
            "ay": "(vy_current-vy_previous)/acceleration_delta_t",
            "vector_magnitude": "sqrt(ax^2+ay^2)",
            "direction_change": (
                "(direction_current-direction_previous+pi)%(2*pi)-pi"
            ),
        },
        "worked_numerators_and_denominators": {
            "constant_velocity": {
                "vx": "(1-0)/(1-0)=1",
                "speed": "sqrt(1^2+0^2)=1",
            },
            "speed_increase": {
                "midpoints": "[0.5,1.5]",
                "acceleration_delta_t": "1.5-0.5=1",
                "tangential": "(2-1)/1=1",
                "ax": "(2-1)/1=1",
                "vector_magnitude": "sqrt(1^2+0^2)=1",
            },
            "direction_change": {
                "midpoints": "[0.5,1.5]",
                "acceleration_delta_t": "1.5-0.5=1",
                "tangential": "(1-1)/1=0",
                "ax": "(0-1)/1=-1",
                "ay": "(1-0)/1=1",
                "vector_magnitude": "sqrt((-1)^2+1^2)=sqrt(2)",
            },
            "irregular_timing": {
                "midpoints": "[(0+1)/2,(1+3)/2]=[0.5,2.0]",
                "acceleration_delta_t": "2.0-0.5=1.5",
            },
        },
        "checks": checks,
        "errors": errors,
        "golden_arithmetic_pass": not [
            error for error in errors if error.startswith("golden_")
        ],
        "negative_schema_tests_pass": not [
            error for error in errors if error.startswith("negative_schema_")
        ],
        "production_trace_audited": not [
            error for error in errors if error.startswith("production_")
        ],
        "status": "PASS" if not errors else "FAIL",
    }
    (ROOT / "phase2_reference_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "status": result["status"],
            "checks": len(checks),
            "errors": errors,
        },
        indent=2,
    ))
    if errors:
        raise SystemExit(1)


def _verify_unit_denominators(
    checks: list[dict[str, Any]],
    units: list[dict[str, str]],
) -> None:
    for row in units:
        case_id = row["case_id"]
        observed = int(float(row["observed_frame_count"]))
        possible = max(observed - 1, 0)
        higher_possible = max(observed - 2, 0)
        valid = int(float(row["valid_pair_count"]))
        velocity_valid = int(float(row["velocity_valid_count"]))
        direction_valid = int(float(row["direction_change_valid_count"]))
        acceleration_valid = int(float(row["acceleration_valid_count"]))
        expected = {
            "possible_pair_count": possible,
            "velocity_possible_count": possible,
            "direction_change_possible_count": higher_possible,
            "acceleration_possible_count": higher_possible,
            "valid_pair_ratio": valid / possible if possible else 0.0,
            "velocity_coverage": (
                velocity_valid / possible if possible else 0.0
            ),
            "direction_change_coverage": (
                direction_valid / higher_possible
                if higher_possible
                else 0.0
            ),
            "acceleration_coverage": (
                acceleration_valid / higher_possible
                if higher_possible
                else 0.0
            ),
        }
        for field, value in expected.items():
            actual = float(row[field])
            _check(
                checks,
                f"unit_denominator_{case_id}_{field}",
                _equal(actual, value),
                {
                    "actual": actual,
                    "expected": value,
                    "numerator": {
                        "valid_pair_ratio": valid,
                        "velocity_coverage": velocity_valid,
                        "direction_change_coverage": direction_valid,
                        "acceleration_coverage": acceleration_valid,
                    }.get(field),
                    "denominator": {
                        "valid_pair_ratio": possible,
                        "velocity_coverage": possible,
                        "direction_change_coverage": higher_possible,
                        "acceleration_coverage": higher_possible,
                    }.get(field),
                },
            )


def _verify_schema(checks: list[dict[str, Any]]) -> None:
    payload = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "dtype": "float32",
        "ordered_feature_names": FEATURES,
        "validity_masks": MASKS,
        "aggregation_outputs": AGGREGATES,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_hash = hashlib.sha256(encoded).hexdigest()
    _check(
        checks,
        "schema_dimension_12",
        len(FEATURES) == 12,
        {"actual": len(FEATURES), "expected": 12},
    )
    _check(
        checks,
        "schema_hash_reproducible",
        actual_hash == EXPECTED_HASH,
        {"actual": actual_hash, "expected": EXPECTED_HASH},
    )


def _verify_negative_schemas(checks: list[dict[str, Any]]) -> None:
    mutations = {
        "missing_feature": FEATURES[:-1],
        "reordered_feature": [FEATURES[1], FEATURES[0], *FEATURES[2:]],
        "duplicate_feature": [*FEATURES[:-1], FEATURES[0]],
        "wrong_dimension": FEATURES[:-1],
        "unexpected_feature": [*FEATURES, "unexpected"],
    }
    for name, actual in mutations.items():
        reasons = []
        if len(actual) != 12:
            reasons.append("dimension")
        if len(set(actual)) != len(actual):
            reasons.append("duplicate")
        if actual != FEATURES:
            reasons.append("order_or_membership")
        _check(
            checks,
            f"negative_schema_{name}",
            bool(reasons),
            {"actual_names": actual, "violation_reasons": reasons},
        )
    _check(
        checks,
        "negative_schema_wrong_hash",
        ("0" * 64) != EXPECTED_HASH,
        {"actual": "0" * 64, "expected": EXPECTED_HASH},
    )
    _check(
        checks,
        "negative_schema_wrong_version",
        "pre_v2" != SCHEMA_VERSION,
        {"actual": "pre_v2", "expected": SCHEMA_VERSION},
    )


def _verify_production_trace(checks: list[dict[str, Any]]) -> None:
    trace = _read_csv(ROOT / "phase2_production_motion_trace.csv")
    summary = _read_csv(ROOT / "phase2_production_unit_summary.csv")
    units_by_source: dict[str, set[str]] = {}
    for row in summary:
        units_by_source.setdefault(row["source_type"], set()).add(
            row["temporal_unit_key"]
        )
    for source_type, expected in (
        ("cvat_tracking_xml", 10),
        ("legacy_recovered", 10),
    ):
        actual = len(units_by_source.get(source_type, set()))
        _check(
            checks,
            f"production_{source_type}_unit_count",
            actual == expected,
            {"actual": actual, "expected": expected},
        )
    first_by_unit: dict[str, dict[str, str]] = {}
    stationary_valid = 0
    nonzero_valid = 0
    valid_acceleration = 0
    for row in trace:
        first_by_unit.setdefault(row["temporal_unit_key"], row)
        velocity_valid = _bool(row["velocity_valid"])
        speed = _value(row["speed_n_per_second"])
        if velocity_valid and speed == 0.0:
            stationary_valid += 1
        if velocity_valid and isinstance(speed, float) and speed > 0:
            nonzero_valid += 1
        valid_acceleration += int(_bool(row["vector_acceleration_valid"]))
    _check(
        checks,
        "production_first_row_each_unit_invalid",
        all(not _bool(row["valid_motion_pair"]) for row in first_by_unit.values()),
        {"units_checked": len(first_by_unit)},
    )
    _check(
        checks,
        "production_stationary_valid_present",
        stationary_valid > 0,
        {"stationary_valid_count": stationary_valid},
    )
    _check(
        checks,
        "production_nonzero_velocity_present",
        nonzero_valid > 0,
        {"nonzero_velocity_count": nonzero_valid},
    )
    _check(
        checks,
        "production_valid_acceleration_present",
        valid_acceleration > 0,
        {"valid_acceleration_count": valid_acceleration},
    )


def _verify_tensor_preflight(checks: list[dict[str, Any]]) -> None:
    preflight = json.loads(
        (ROOT / "phase2_tensor_schema_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    expected_empty = (
        "missing_features",
        "duplicate_features",
        "order_mismatches",
        "required_masks_missing",
        "errors",
    )
    _check(
        checks,
        "producer_exporter_dimension_12",
        preflight["expected_dimension"] == 12
        and preflight["actual_dimension"] == 12,
        {
            "expected": preflight["expected_dimension"],
            "actual": preflight["actual_dimension"],
        },
    )
    for field in expected_empty:
        _check(
            checks,
            f"producer_exporter_{field}_empty",
            preflight[field] == [],
            {"actual": preflight[field], "expected": []},
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _value(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return None
    if text.casefold() in {"true", "false"}:
        return _bool(text)
    try:
        return float(text)
    except ValueError:
        return text


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if expected is None:
        return actual is None
    return isinstance(actual, float) and math.isclose(
        actual,
        float(expected),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    )


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    evidence: dict[str, Any],
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "passed": bool(passed),
            "evidence": evidence,
        }
    )


if __name__ == "__main__":
    main()

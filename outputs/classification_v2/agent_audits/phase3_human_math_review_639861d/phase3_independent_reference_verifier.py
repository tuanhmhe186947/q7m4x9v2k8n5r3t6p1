"""Independent Phase 3 verifier using only standard-library arithmetic."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-12
IMPLEMENTATION_COMMIT = "639861d9d41112a5bfdddbb96c1ce15471e07acb"

checks: list[dict[str, Any]] = []
errors: list[str] = []


def check(
    check_id: str,
    actual: Any,
    expected: Any,
    *,
    numerator: str | None = None,
    denominator: str | None = None,
) -> None:
    if isinstance(expected, float):
        passed = math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        )
    else:
        passed = actual == expected
    evidence = {
        "actual": actual,
        "expected": expected,
        "numerator": numerator,
        "denominator": denominator,
    }
    checks.append(
        {
            "check_id": check_id,
            "passed": passed,
            "evidence": evidence,
        }
    )
    if not passed:
        errors.append(check_id)


def as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def as_float(value: Any) -> float:
    text = str(value).strip()
    return math.nan if text in {"", "nan", "NaN"} else float(text)


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def verify_distance() -> None:
    rows = {
        row["case_id"]: row
        for row in read_csv("phase3_distance_golden_cases.csv")
    }
    diagonal_non_square = 100.0 / math.sqrt(1000.0**2 + 500.0**2)
    diagonal_square = 100.0 / math.sqrt(1000.0**2 + 1000.0**2)
    expected = {
        "distance_square_horizontal": (True, 0.1, diagonal_square),
        "distance_square_vertical": (True, 0.1, diagonal_square),
        "distance_non_square_horizontal": (
            True,
            0.1,
            diagonal_non_square,
        ),
        "distance_non_square_vertical": (
            True,
            0.2,
            diagonal_non_square,
        ),
        "distance_diagonal_equal_horizontal": (
            True,
            0.1,
            diagonal_non_square,
        ),
        "distance_diagonal_equal_vertical": (
            True,
            0.2,
            diagonal_non_square,
        ),
        "distance_coincident_centers": (True, 0.0, 0.0),
        "distance_invalid_width": (False, math.nan, math.nan),
        "distance_invalid_height": (False, math.nan, math.nan),
        "distance_invalid_actor_geometry": (
            False,
            math.nan,
            math.nan,
        ),
        "distance_invalid_partner_geometry": (
            False,
            math.nan,
            math.nan,
        ),
    }
    for case_id, (available, axis, diagonal) in expected.items():
        row = rows[case_id]
        check(
            f"{case_id}_availability",
            as_bool(row["distance_available"]),
            available,
        )
        actual_axis = as_float(row["axis_normalized_distance"])
        actual_diagonal = as_float(
            row["diagonal_normalized_distance"]
        )
        if available:
            check(
                f"{case_id}_axis",
                actual_axis,
                axis,
                numerator="sqrt((dx_px/W)^2+(dy_px/H)^2)",
                denominator="W and H separately",
            )
            check(
                f"{case_id}_diagonal",
                actual_diagonal,
                diagonal,
                numerator="sqrt(dx_px^2+dy_px^2)",
                denominator="sqrt(W^2+H^2)",
            )
        else:
            check(
                f"{case_id}_axis_unavailable",
                math.isnan(actual_axis),
                True,
            )
            check(
                f"{case_id}_diagonal_unavailable",
                math.isnan(actual_diagonal),
                True,
            )
        check(
            f"{case_id}_not_physical",
            as_bool(row["is_physical_measurement"]),
            False,
        )
    check(
        "non_square_axis_anisotropy",
        as_float(
            rows["distance_non_square_vertical"][
                "axis_normalized_distance"
            ]
        ),
        2.0
        * as_float(
            rows["distance_non_square_horizontal"][
                "axis_normalized_distance"
            ]
        ),
    )
    check(
        "non_square_diagonal_isotropy",
        as_float(
            rows["distance_diagonal_equal_horizontal"][
                "diagonal_normalized_distance"
            ]
        ),
        as_float(
            rows["distance_diagonal_equal_vertical"][
                "diagonal_normalized_distance"
            ]
        ),
    )


def verify_social() -> None:
    rows = {
        row["case_id"]: row
        for row in read_csv("phase3_social_golden_cases.csv")
    }
    expected_partner = {
        "social_one_valid_neighbor": ("video-a|track=B", 1, True),
        "social_equal_distance_neighbors": (
            "video-a|track=B",
            2,
            True,
        ),
        "social_equal_distance_permutation": (
            "video-a|track=B",
            2,
            True,
        ),
        "social_actor_self_row_present": (
            "video-a|track=B",
            1,
            True,
        ),
        "social_blank_pig_id": ("video-a|track=B", 1, True),
        "social_duplicate_pig_id": ("video-a|track=B", 2, True),
        "social_no_valid_neighbor": ("", 0, False),
        "social_same_pig_id_cross_video": ("", 0, False),
    }
    for case_id, (partner, tie_count, available) in expected_partner.items():
        row = rows[case_id]
        check(
            f"{case_id}_partner",
            row["nearest_partner_key"],
            partner,
        )
        check(
            f"{case_id}_tie_count",
            int(row["nearest_tie_count"]),
            tie_count,
        )
        check(
            f"{case_id}_available",
            as_bool(row["nearest_neighbor_available"]),
            available,
        )
        check(
            f"{case_id}_not_self",
            row["nearest_partner_key"] != row["actor_key"],
            True,
        )
    check(
        "equal_distance_permutation_same_partner",
        rows["social_equal_distance_neighbors"]["nearest_partner_key"],
        rows["social_equal_distance_permutation"]["nearest_partner_key"],
    )
    check(
        "equal_distance_permutation_same_axis_distance",
        as_float(
            rows["social_equal_distance_neighbors"][
                "nearest_distance_axis"
            ]
        ),
        as_float(
            rows["social_equal_distance_permutation"][
                "nearest_distance_axis"
            ]
        ),
    )
    continuity_expected = {
        "social_partner_b_remains_b": (True, True, False, False),
        "social_partner_b_changes_c": (True, False, True, False),
        "social_partner_b_unavailable": (False, False, False, True),
        "social_temporal_unit_reset": (False, False, False, False),
        "social_actor_identity_reset": (False, False, False, False),
    }
    for case_id, (
        valid,
        same,
        switch,
        no_neighbor,
    ) in continuity_expected.items():
        row = rows[case_id]
        check(
            f"{case_id}_continuity_valid",
            as_bool(row["partner_continuity_valid"]),
            valid,
        )
        check(
            f"{case_id}_same_partner",
            as_bool(row["same_partner_as_previous"]),
            same,
        )
        check(
            f"{case_id}_switch",
            as_bool(row["partner_switch"]),
            switch,
        )
        check(
            f"{case_id}_no_neighbor",
            as_bool(row["no_neighbor"]),
            no_neighbor,
        )


def verify_roi() -> None:
    rows = {
        row["case_id"]: row
        for row in read_csv("phase3_roi_golden_cases.csv")
    }
    expected = {
        "roi_three_available_two_contact": (5, 3, 2, 0.6, 2.0 / 3.0, True),
        "roi_all_unavailable": (5, 0, 0, 0.0, 0.0, False),
        "roi_all_available_none_contact": (5, 5, 0, 1.0, 0.0, True),
        "roi_all_available_all_contact": (5, 5, 5, 1.0, 1.0, True),
        "roi_mixed_invalid_geometry": (5, 2, 1, 0.4, 0.5, True),
        "roi_availability_ratio_denominator": (5, 3, 0, 0.6, 0.0, True),
        "roi_contact_ratio_denominator": (
            5,
            3,
            2,
            0.6,
            2.0 / 3.0,
            True,
        ),
        "roi_zero_placeholder_unavailable": (
            3,
            0,
            0,
            0.0,
            0.0,
            False,
        ),
    }
    for case_id, (
        observed,
        available,
        contact,
        availability_ratio,
        contact_ratio,
        unit_available,
    ) in expected.items():
        row = rows[case_id]
        check(
            f"{case_id}_observed",
            int(float(row["observed_frame_count"])),
            observed,
        )
        check(
            f"{case_id}_available_count",
            int(float(row["roi_available_frame_count"])),
            available,
        )
        check(
            f"{case_id}_contact_count",
            int(float(row["roi_contact_frame_count"])),
            contact,
        )
        check(
            f"{case_id}_availability_ratio",
            as_float(row["roi_availability_ratio"]),
            availability_ratio,
            numerator=str(available),
            denominator=str(observed),
        )
        check(
            f"{case_id}_contact_ratio",
            as_float(row["roi_contact_ratio"]),
            contact_ratio,
            numerator=str(contact),
            denominator=str(available),
        )
        check(
            f"{case_id}_unit_available",
            as_bool(row["target_roi_unit_available"]),
            unit_available,
        )
    check(
        "reject_incorrect_two_over_five",
        as_float(
            rows["roi_three_available_two_contact"]["roi_contact_ratio"]
        )
        != 0.4,
        True,
        numerator="2",
        denominator="3 ROI-available frames, not 5 observed frames",
    )
    check(
        "target_roi_model_forbidden",
        as_bool(
            rows["roi_review_only_requested_by_model"][
                "model_export_allowed"
            ]
        ),
        False,
    )
    check(
        "label_independent_roi_allowed",
        as_bool(
            rows["roi_label_independent_allowed"]["model_export_allowed"]
        ),
        True,
    )


def verify_registry_and_preflights() -> None:
    registry = read_json("phase3_distance_metric_registry.json")
    check(
        "registry_physical_distance_claimed",
        registry["physical_distance_claimed"],
        False,
    )
    check(
        "registry_homography_unavailable",
        registry["homography_world_plane_available"],
        False,
    )
    threshold = registry["threshold_bindings"][0]
    check(
        "social_threshold_value_unchanged",
        float(threshold["threshold_value"]),
        0.08,
    )
    check(
        "social_threshold_axis_bound",
        threshold["metric_id"],
        "image_axis_normalized_distance",
    )
    check(
        "social_threshold_not_recalibrated",
        threshold["recalibrated_in_phase3"],
        False,
    )
    permutation = read_json("phase3_row_permutation_results.json")
    check("production_permutation_pass", permutation["pass"], True)
    check(
        "production_permutation_digest_equal",
        permutation["first_digest"],
        permutation["permuted_digest"],
    )
    leakage = read_json("phase3_model_leakage_preflight.json")
    check("leakage_preflight_pass", leakage["pass"], True)
    check(
        "four_forbidden_features_failed",
        len(leakage["forbidden_failures"]),
        4,
    )
    check(
        "label_independent_preflight_allowed",
        leakage["label_independent_allowed"],
        True,
    )


def verify_production_trace() -> None:
    audit = read_json("phase3_bounded_regression_audit.json")
    trace = read_csv("phase3_production_social_trace.csv")
    roi = read_csv("phase3_production_roi_summary.csv")
    check(
        "production_code_authority_sha",
        audit["code_authority_sha"],
        IMPLEMENTATION_COMMIT,
    )
    check("production_errors_empty", audit["errors"], [])
    check(
        "production_independent_checker_valid",
        audit["independent_checker"]["valid"],
        True,
    )
    check(
        "production_cvat_units",
        int(audit["source_unit_counts"]["cvat_tracking_xml"]),
        10,
    )
    check(
        "production_legacy_units",
        int(audit["source_unit_counts"]["legacy_recovered"]),
        10,
    )
    self_neighbors = sum(
        bool(row["nearest_partner_key"])
        and row["nearest_partner_key"] == row["canonical_actor_key"]
        for row in trace
    )
    check("production_self_neighbors", self_neighbors, 0)
    first_by_actor: dict[tuple[str, str], dict[str, str]] = {}
    for row in trace:
        key = (row["temporal_unit_key"], row["canonical_actor_key"])
        first_by_actor.setdefault(key, row)
    check(
        "production_first_social_rows_invalid_continuity",
        all(
            not as_bool(row["partner_continuity_valid"])
            for row in first_by_actor.values()
        ),
        True,
    )
    check(
        "production_no_neighbor_not_switch",
        all(
            not as_bool(row["partner_switch"])
            for row in trace
            if not as_bool(row["nearest_neighbor_available"])
        ),
        True,
    )
    check(
        "production_distance_availability_consistent",
        all(
            as_bool(row["distance_available"])
            == as_bool(row["nearest_neighbor_available"])
            for row in trace
        ),
        True,
    )
    roi_consistent = True
    zero_unavailable = True
    for row in roi:
        observed = int(row["observed_frame_count"])
        available = int(row["target_roi_available_frame_count"])
        contact = int(row["target_roi_contact_frame_count"])
        availability_ratio = float(
            row["target_roi_availability_ratio_unit"]
        )
        contact_ratio = float(row["target_roi_contact_ratio_unit"])
        expected_availability = available / observed if observed else 0.0
        expected_contact = contact / available if available else 0.0
        roi_consistent &= contact <= available
        roi_consistent &= math.isclose(
            availability_ratio,
            expected_availability,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        )
        roi_consistent &= math.isclose(
            contact_ratio,
            expected_contact,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        )
        if available == 0:
            zero_unavailable &= not as_bool(
                row["target_roi_unit_available"]
            )
    check("production_roi_denominators", roi_consistent, True)
    check("production_zero_roi_unavailable", zero_unavailable, True)


def main() -> int:
    verify_distance()
    verify_social()
    verify_roi()
    verify_registry_and_preflights()
    verify_production_trace()
    payload = {
        "schema_version": (
            "classification_v2.phase3_independent_reference.v1"
        ),
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "imports_production_distance_functions": False,
        "imports_production_social_functions": False,
        "imports_production_roi_functions": False,
        "expected_values_are_hard_coded": True,
        "tolerance": TOLERANCE,
        "checks": checks,
        "errors": errors,
        "golden_arithmetic_pass": not errors,
        "row_permutation_invariance_pass": not any(
            "permutation" in error for error in errors
        ),
        "roi_denominator_audit_pass": not any(
            "roi_" in error for error in errors
        ),
        "model_leakage_preflight_pass": not any(
            "leakage" in error or "model_forbidden" in error
            for error in errors
        ),
        "production_trace_audited": not any(
            error.startswith("production_") for error in errors
        ),
        "status": "PASS" if not errors else "FAIL",
    }
    (ROOT / "phase3_reference_verification.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checks": len(checks),
                "errors": errors,
            },
            separators=(",", ":"),
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

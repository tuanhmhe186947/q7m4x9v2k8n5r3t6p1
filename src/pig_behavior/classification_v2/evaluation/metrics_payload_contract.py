"""Paper-facing metrics payload contract for classification_v2.

Native temporal metrics prove the right statistical unit; this contract adds the
second layer needed for a Q2 claim: uncertainty and a predeclared smallest effect
size of interest. It deliberately checks payload structure, not model quality,
because quality claims require comparing a proposed model against a baseline.
"""

from __future__ import annotations

import math
from typing import Any

REQUIRED_NATIVE_METRICS = (
    "rows",
    "accuracy",
    "macro_f1_supported",
    "macro_recall_supported",
    "per_class",
    "focus_pair_confusions",
)

REQUIRED_CI_METRICS = (
    "accuracy",
    "macro_f1_supported",
    "macro_recall_supported",
)


def check_paper_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate metrics fields required before a paper-facing model claim."""

    errors: list[str] = []
    warnings: list[str] = []
    native_metrics = payload.get("native_temporal_metrics", {})
    if not isinstance(native_metrics, dict) or not native_metrics:
        errors.append("missing_native_temporal_metrics")
    else:
        missing_metrics = [name for name in REQUIRED_NATIVE_METRICS if name not in native_metrics]
        if missing_metrics:
            errors.append(f"missing_required_native_metrics={missing_metrics}")

    ci_payload = payload.get("confidence_intervals", {})
    if not isinstance(ci_payload, dict) or not ci_payload:
        errors.append("missing_confidence_intervals")
    else:
        _check_confidence_intervals(ci_payload, errors)

    sesoi = payload.get("sesoi", {})
    if not isinstance(sesoi, dict) or not sesoi:
        errors.append("missing_sesoi")
    else:
        _check_sesoi(sesoi, warnings, errors)

    statistical_unit = payload.get("statistical_unit")
    if statistical_unit != "native_temporal_unit":
        errors.append(f"statistical_unit_must_be_native_temporal_unit={statistical_unit}")

    return {
        "schema_version": "classification_v2_paper_metrics_payload_contract_v1",
        "required_native_metrics": list(REQUIRED_NATIVE_METRICS),
        "required_ci_metrics": list(REQUIRED_CI_METRICS),
        "primary_metric": sesoi.get("primary_metric") if isinstance(sesoi, dict) else None,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _check_confidence_intervals(ci_payload: dict[str, Any], errors: list[str]) -> None:
    """Ensure CI entries exist and have finite, ordered bounds."""

    missing_ci = [name for name in REQUIRED_CI_METRICS if name not in ci_payload]
    if missing_ci:
        errors.append(f"missing_confidence_interval_metrics={missing_ci}")
    for metric_name in REQUIRED_CI_METRICS:
        entry = ci_payload.get(metric_name)
        if not isinstance(entry, dict):
            continue
        for key in ("estimate", "ci_low", "ci_high", "method"):
            if key not in entry:
                errors.append(f"missing_confidence_interval_field={metric_name}.{key}")
        try:
            estimate = float(entry.get("estimate"))
            ci_low = float(entry.get("ci_low"))
            ci_high = float(entry.get("ci_high"))
        except (TypeError, ValueError):
            errors.append(f"non_numeric_confidence_interval={metric_name}")
            continue
        if not all(math.isfinite(value) for value in (estimate, ci_low, ci_high)):
            errors.append(f"non_finite_confidence_interval={metric_name}")
            continue
        if not ci_low <= ci_high:
            errors.append(f"confidence_interval_not_ordered={metric_name}")
        if not all(0.0 <= value <= 1.0 for value in (estimate, ci_low, ci_high)):
            errors.append(f"confidence_interval_out_of_metric_bounds={metric_name}")


def _check_sesoi(sesoi: dict[str, Any], warnings: list[str], errors: list[str]) -> None:
    """Validate the predeclared smallest-effect contract."""

    primary_metric = sesoi.get("primary_metric")
    if primary_metric not in REQUIRED_CI_METRICS:
        errors.append(f"unsupported_sesoi_primary_metric={primary_metric}")
    try:
        minimum_effect = float(sesoi.get("minimum_effect_size"))
    except (TypeError, ValueError):
        errors.append("missing_or_non_numeric_sesoi_minimum_effect_size")
        return
    if minimum_effect <= 0:
        errors.append("sesoi_minimum_effect_size_must_be_positive")
    if sesoi.get("comparison_required") is not True:
        warnings.append("sesoi_comparison_not_marked_required")

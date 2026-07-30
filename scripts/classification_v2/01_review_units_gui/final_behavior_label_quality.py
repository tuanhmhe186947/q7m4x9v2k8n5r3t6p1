"""Label-quality sidecar contract for final Classification V2 review."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

QUALITY_SIDECAR_NAME = "behavior_label_quality_review.csv"
QUALITY_COLUMNS = [
    "review_unit_id",
    "original_behavior",
    "reviewed_behavior",
    "label_status",
    "source_label_error_confirmed",
    "error_pattern",
    "review_confidence",
    "scope_component",
    "selection_assessment",
]
MODEL_X_FORBIDDEN_COLUMNS = tuple(QUALITY_COLUMNS)

SUPPORTED = "SUPPORTED"
SOURCE_LABEL_ERROR_CONFIRMED = "SOURCE_LABEL_ERROR_CONFIRMED"
TECHNICAL_DEFECT = "TECHNICAL_DEFECT"

ERROR_PATTERNS = (
    "ROI_PROXIMITY_ONLY_FALSE_POSITIVE",
    "ROI_CONTACT_ABSENT_FALSE_POSITIVE",
    "INTERACTION_PHASE_OR_TEMPORAL_WINDOW_ERROR",
    "OTHER_CLEAR_SOURCE_LABEL_ERROR",
)
TECHNICAL_ERROR_PATTERN = "TECHNICAL_MEDIA_OR_PRESENTATION_DEFECT"

ROI_SCOPE_COMPONENT = "ROI_DIRECTION_CORRECTED_NONINTERACTION"
INTERACTION_SCOPE_COMPONENT = "POST_CALIBRATION_FULL_INTERACTION_CENSUS"


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def review_confidence(label_strength: Any) -> str:
    """Map the existing decision strength onto sidecar provenance."""

    strength = _text(label_strength).casefold()
    mapping = {
        "strong": "HIGH",
        "medium": "MEDIUM",
        "weak": "LOW",
        "boundary": "LOW",
    }
    if strength not in mapping:
        raise ValueError(f"invalid manual_label_strength={strength!r}")
    return mapping[strength]


def selection_assessment(scope_component: str, decision: str) -> str:
    """Separate selector audit outcomes from full-census outcomes."""

    if decision == "exclude":
        return "TECHNICAL_REVIEW_DEFECT"
    if scope_component == ROI_SCOPE_COMPONENT:
        if decision == "accept":
            return "SELECTOR_FLAGGED_BUT_SOURCE_LABEL_SUPPORTED"
        return "SELECTOR_FLAG_CONFIRMED_SOURCE_LABEL_ERROR"
    if scope_component == INTERACTION_SCOPE_COMPONENT:
        if decision == "accept":
            return "CENSUS_SOURCE_LABEL_SUPPORTED"
        return "CENSUS_SOURCE_LABEL_ERROR_FOUND"
    if decision == "accept":
        return "SOURCE_LABEL_SUPPORTED"
    return "SOURCE_LABEL_ERROR_FOUND"


def build_quality_record(
    unit: Mapping[str, Any],
    decision_record: Mapping[str, Any],
    *,
    error_pattern: str = "",
) -> dict[str, str] | None:
    """Build one terminal quality record; pending is workflow-only."""

    decision = _text(decision_record.get("manual_review_decision")).casefold()
    if decision == "pending" or not decision:
        return None
    if decision not in {"accept", "corrected", "exclude"}:
        raise ValueError(f"invalid manual_review_decision={decision!r}")

    original = _text(unit.get("behavior_label"))
    scope_component = _text(unit.get("final_scope_component"))
    corrected = _text(decision_record.get("manual_corrected_behavior"))
    confidence = review_confidence(
        decision_record.get("manual_label_strength")
    )

    if decision == "accept":
        reviewed = original
        label_status = SUPPORTED
        source_error = "NO"
        pattern = "NONE"
    elif decision == "corrected":
        if not corrected or corrected == original:
            raise ValueError("corrected decision requires a different behavior")
        if error_pattern not in ERROR_PATTERNS:
            raise ValueError(
                "corrected decision requires a declared clear-error pattern"
            )
        reviewed = corrected
        label_status = SOURCE_LABEL_ERROR_CONFIRMED
        source_error = "YES"
        pattern = error_pattern
    else:
        reviewed = ""
        label_status = TECHNICAL_DEFECT
        source_error = "NOT_APPLICABLE"
        pattern = TECHNICAL_ERROR_PATTERN

    return {
        "review_unit_id": _text(unit.get("review_unit_id")),
        "original_behavior": original,
        "reviewed_behavior": reviewed,
        "label_status": label_status,
        "source_label_error_confirmed": source_error,
        "error_pattern": pattern,
        "review_confidence": confidence,
        "scope_component": scope_component,
        "selection_assessment": selection_assessment(
            scope_component,
            decision,
        ),
    }


def validate_quality_records(
    quality: pd.DataFrame,
    units: pd.DataFrame,
    decisions: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Validate the sidecar without requiring unfinished rows."""

    missing = [column for column in QUALITY_COLUMNS if column not in quality]
    if missing:
        return [f"quality_sidecar_missing_columns={','.join(missing)}"]
    ids = quality["review_unit_id"].fillna("").astype(str).str.strip()
    errors: list[str] = []
    if ids.eq("").any():
        errors.append("quality_sidecar_blank_review_unit_id")
    if ids.duplicated().any():
        errors.append("quality_sidecar_duplicate_review_unit_id")

    unit_rows = {
        _text(row.get("review_unit_id")): row
        for _, row in units.iterrows()
    }
    unexpected = sorted(set(ids) - set(unit_rows))
    if unexpected:
        errors.append(
            f"quality_sidecar_unexpected_review_unit_ids={len(unexpected)}"
        )

    for raw in quality.to_dict(orient="records"):
        review_id = _text(raw.get("review_unit_id"))
        unit = unit_rows.get(review_id)
        decision = decisions.get(review_id)
        if unit is None or decision is None:
            errors.append(f"quality_sidecar_without_decision={review_id}")
            continue
        pattern = _text(raw.get("error_pattern"))
        try:
            expected = build_quality_record(
                unit,
                decision,
                error_pattern=pattern,
            )
        except ValueError as exc:
            errors.append(f"quality_sidecar_invalid={review_id}:{exc}")
            continue
        if expected is None:
            errors.append(f"quality_sidecar_for_pending_decision={review_id}")
            continue
        for column in QUALITY_COLUMNS:
            if _text(raw.get(column)) != expected[column]:
                errors.append(
                    f"quality_sidecar_mismatch={review_id}:{column}"
                )
    return sorted(set(errors))

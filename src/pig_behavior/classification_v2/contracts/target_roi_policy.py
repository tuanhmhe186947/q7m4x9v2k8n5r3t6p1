"""Shared target-selected ROI prohibition for every model-input path."""

from __future__ import annotations

from typing import Any

TARGET_ROI_SHARED_POLICY_ID = (
    "policy.classification_v2.target_selected_roi_model_forbidden"
)
ROI_TARGET_MODEL_POLICY_VERSION = (
    "classification_v2.target_roi_model_forbidden.v2"
)
TARGET_ROI_MODEL_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "target_roi_",
    "roi_target_",
    "label_selected_target_roi",
    "label_conditioned_roi",
    "behavior_selected_roi",
)
TARGET_ROI_MODEL_FORBIDDEN_EXACT: frozenset[str] = frozenset(
    {
        "label_selected_roi_class_indicator",
        "label_selected_target_roi_class",
        "label_conditioned_roi_identity",
        "behavior_selected_roi_class",
        "target_roi_contact",
        "target_roi_distance",
        "target_roi_contact_ratio_unit",
    }
)
TARGET_ROI_MODEL_SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "label_selected_roi_class_indicator": (
        "label_selected_target_roi_class",
        "behavior_selected_roi_class",
        "target_roi_class",
        "roi_target_class",
    ),
    "label_conditioned_roi_identity": (
        "target_roi_identity",
        "roi_target_identity",
    ),
    "target_roi_contact": (
        "roi_target_contact",
        "behavior_selected_roi_contact",
    ),
    "target_roi_distance": (
        "roi_target_distance",
        "behavior_selected_roi_distance",
    ),
    "target_roi_contact_ratio_unit": (
        "roi_target_contact_ratio_unit",
        "behavior_selected_roi_contact_ratio_unit",
    ),
}
TARGET_ROI_MODEL_FORBIDDEN_REASON = (
    "behavior-selected target ROI is label-derived review evidence"
)
TARGET_ROI_MODEL_APPLICABLE_CHECKERS: tuple[str, ...] = (
    "train_ready_feature_selection",
    "spatial_tensor_export",
    "model_input_column_preflight",
    "model_input_manifest_validation",
)


def is_target_roi_model_forbidden(column: str) -> bool:
    """Return whether target-selected ROI semantics are forbidden from model X."""

    normalized = str(column).strip().lower()
    aliases = {
        alias
        for values in TARGET_ROI_MODEL_SEMANTIC_ALIASES.values()
        for alias in values
    }
    return (
        normalized in TARGET_ROI_MODEL_FORBIDDEN_EXACT
        or normalized in aliases
        or normalized.startswith(TARGET_ROI_MODEL_FORBIDDEN_PREFIXES)
    )


def target_roi_model_policy_registry() -> dict[str, Any]:
    """Return the single target-selected ROI model-prohibition authority."""

    return {
        "policy_id": TARGET_ROI_SHARED_POLICY_ID,
        "policy_version": ROI_TARGET_MODEL_POLICY_VERSION,
        "canonical_forbidden_names": sorted(
            TARGET_ROI_MODEL_FORBIDDEN_EXACT
        ),
        "semantic_alias_mapping": {
            key: list(values)
            for key, values in sorted(
                TARGET_ROI_MODEL_SEMANTIC_ALIASES.items()
            )
        },
        "forbidden_prefixes": list(TARGET_ROI_MODEL_FORBIDDEN_PREFIXES),
        "reason": TARGET_ROI_MODEL_FORBIDDEN_REASON,
        "applicable_exporters_and_checkers": list(
            TARGET_ROI_MODEL_APPLICABLE_CHECKERS
        ),
        "review_only_evidence_allowed": True,
        "model_input_allowed": False,
    }


def target_roi_policy_metadata(column: str) -> dict[str, Any]:
    """Return explicit review/model/leakage policy for one ROI feature."""

    forbidden = is_target_roi_model_forbidden(column)
    return {
        "feature_name": column,
        "review_eligible": True,
        "model_eligible": not forbidden,
        "model_forbidden_reason": (
            TARGET_ROI_MODEL_FORBIDDEN_REASON if forbidden else ""
        ),
        "leakage_risk": "CRITICAL_LABEL_SELECTED" if forbidden else "LOW",
        "semantics_version": ROI_TARGET_MODEL_POLICY_VERSION,
    }


__all__ = [
    "ROI_TARGET_MODEL_POLICY_VERSION",
    "TARGET_ROI_MODEL_APPLICABLE_CHECKERS",
    "TARGET_ROI_MODEL_FORBIDDEN_EXACT",
    "TARGET_ROI_MODEL_FORBIDDEN_PREFIXES",
    "TARGET_ROI_MODEL_FORBIDDEN_REASON",
    "TARGET_ROI_MODEL_SEMANTIC_ALIASES",
    "TARGET_ROI_SHARED_POLICY_ID",
    "is_target_roi_model_forbidden",
    "target_roi_model_policy_registry",
    "target_roi_policy_metadata",
]

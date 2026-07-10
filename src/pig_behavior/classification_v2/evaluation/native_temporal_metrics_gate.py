"""Native temporal metrics gate for paper-facing classification_v2 records.

This module keeps the Q2 claim boundary explicit. Window-level training samples
are useful for optimization, but paper-facing behavior recognition results must
be auditable at the native temporal/review-unit level so metrics do not inflate
by repeatedly scoring overlapping windows from the same annotation unit.
"""

from __future__ import annotations

from typing import Any

PAPER_SAFE_SPLIT_POLICIES = {
    "recording_group_oof",
    "video_session_oof",
    "video_session_safe",
    "session_video_safe",
}

MODEL_RESULT_KINDS = {
    "model_evaluation",
    "baseline_evaluation",
    "ablation_evaluation",
}

PROTOCOL_RESULT_KINDS = {
    "protocol_gate",
    "data_gate",
    "review_gate",
    "engineering_smoke",
}


def default_evaluation_contract() -> dict[str, Any]:
    """Return the default paper-safe evaluation contract for new records."""

    return {
        "schema_version": "classification_v2_native_temporal_metrics_gate_v1",
        "paper_claim_level": "Q2_strong",
        "result_kind": "protocol_gate",
        "primary_metric_unit": "native_temporal_unit",
        "split_policy": "recording_group_oof",
        "source_domain_control_required": True,
        "native_temporal_metrics_required": True,
        "window_metrics_are_secondary": True,
        "feature_leakage_guard_required": True,
        "review_unit_decisions_required": True,
        "pig_identity_scope": "annotation_local_not_cross_video",
        "interaction_context": "full_frame_partner_context",
        "external_generalization_claim": False,
    }


def check_native_temporal_metrics_gate(
    *,
    evaluation_contract: dict[str, Any] | None,
    metrics_payload: dict[str, Any] | None,
    paper_facing: bool,
    experiment_stage: str,
) -> dict[str, Any]:
    """Validate that a record cannot support a Q2 model claim with weak metrics.

    Protocol/data-gate records are allowed to be paper-facing without model
    metrics, but model/baseline/ablation records must include native temporal
    prediction evidence. This prevents a later report from treating overlapping
    sequence-window scores as the primary scientific metric.
    """

    errors: list[str] = []
    warnings: list[str] = []
    contract = evaluation_contract or {}
    requires_contract = paper_facing or experiment_stage == "paper_facing_candidate"
    if requires_contract and not contract:
        errors.append("missing_evaluation_contract")
        contract = {}

    _check_contract_fields(contract, errors)
    result_kind = str(contract.get("result_kind", "engineering_smoke"))
    if result_kind in MODEL_RESULT_KINDS:
        _check_model_metrics_payload(metrics_payload, errors, warnings)
    elif result_kind not in PROTOCOL_RESULT_KINDS:
        errors.append(f"unknown_result_kind={result_kind}")
    elif (
        metrics_payload
        and _payload_has_window_metrics(metrics_payload)
        and not _payload_has_native_metrics(metrics_payload)
    ):
        warnings.append("protocol_record_has_window_metrics_without_native_metrics")

    return {
        "schema_version": contract.get("schema_version"),
        "paper_claim_level": contract.get("paper_claim_level"),
        "result_kind": result_kind,
        "primary_metric_unit": contract.get("primary_metric_unit"),
        "split_policy": contract.get("split_policy"),
        "source_domain_control_required": contract.get("source_domain_control_required"),
        "native_temporal_metrics_required": contract.get("native_temporal_metrics_required"),
        "external_generalization_claim": contract.get("external_generalization_claim"),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _check_contract_fields(contract: dict[str, Any], errors: list[str]) -> None:
    """Check the claim-boundary fields that define paper-safe evaluation."""

    if not contract:
        return
    if contract.get("paper_claim_level") != "Q2_strong":
        errors.append("paper_claim_level_must_be_Q2_strong")
    if contract.get("primary_metric_unit") != "native_temporal_unit":
        errors.append("primary_metric_unit_must_be_native_temporal_unit")
    if contract.get("split_policy") not in PAPER_SAFE_SPLIT_POLICIES:
        errors.append(f"split_policy_not_video_session_safe={contract.get('split_policy')}")
    if contract.get("source_domain_control_required") is not True:
        errors.append("source_domain_control_required_must_be_true")
    if contract.get("native_temporal_metrics_required") is not True:
        errors.append("native_temporal_metrics_required_must_be_true")
    if contract.get("window_metrics_are_secondary") is not True:
        errors.append("window_metrics_are_secondary_must_be_true")
    if contract.get("feature_leakage_guard_required") is not True:
        errors.append("feature_leakage_guard_required_must_be_true")
    if contract.get("review_unit_decisions_required") is not True:
        errors.append("review_unit_decisions_required_must_be_true")
    if contract.get("pig_identity_scope") != "annotation_local_not_cross_video":
        errors.append("pig_identity_scope_must_be_annotation_local_not_cross_video")
    if contract.get("interaction_context") != "full_frame_partner_context":
        errors.append("interaction_context_must_be_full_frame_partner_context")
    if contract.get("external_generalization_claim") is not False:
        errors.append("external_generalization_claim_must_be_false_without_external_cohort")


def _check_model_metrics_payload(
    metrics_payload: dict[str, Any] | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Require native temporal evidence for actual model-evaluation records."""

    if not metrics_payload:
        errors.append("missing_metrics_payload_for_model_result")
        return
    if not _payload_has_native_metrics(metrics_payload):
        errors.append("missing_native_temporal_metrics_or_prediction_audit")
    if _payload_has_window_metrics(metrics_payload) and not _payload_has_native_metrics(metrics_payload):
        errors.append("window_metrics_present_without_native_temporal_metrics")
    if _payload_has_window_metrics(metrics_payload):
        warnings.append("window_metrics_treated_as_secondary")
    payload_unit = metrics_payload.get("primary_metric_unit")
    if payload_unit is not None and payload_unit != "native_temporal_unit":
        errors.append(f"metrics_payload_primary_unit_not_native_temporal_unit={payload_unit}")


def _payload_has_native_metrics(metrics_payload: dict[str, Any]) -> bool:
    """Detect acceptable native temporal prediction evidence in metric payloads."""

    return any(
        key in metrics_payload
        for key in (
            "native_temporal_metrics",
            "native_temporal_prediction_audit",
            "review_unit_metrics",
            "temporal_unit_metrics",
        )
    )


def _payload_has_window_metrics(metrics_payload: dict[str, Any]) -> bool:
    """Detect window-level metrics so they can be demoted from primary evidence."""

    return any(key in metrics_payload for key in ("window_metrics", "sequence_window_metrics", "window_level_metrics"))

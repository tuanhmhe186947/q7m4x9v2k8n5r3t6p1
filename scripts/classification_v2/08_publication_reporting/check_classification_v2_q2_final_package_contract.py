from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_MODEL_FAMILIES = [
    "B2_spatial_only",
    "B3_actor_image_only",
    "B4_actor_spatial",
    "B5_actor_spatial_partner_context",
    "B6_actor_spatial_partner_multitask",
    "B7_full_candidate_domain_controls",
]


def main() -> None:
    """Validate the S9 final package contract without running final evaluation."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 Q2 final calibration and paper-package contract."
    )
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/q2_final_calibration_paper_package_contract_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_final_package_contract_audit.json"),
    )
    args = parser.parse_args()
    audit = check_contract(args.contract_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_contract(contract_json: Path) -> dict[str, Any]:
    """Return a deterministic audit for final calibration and reporting safety."""

    errors: list[str] = []
    contract = _read_json(contract_json, errors)
    _check_claim_boundary(contract.get("claim_boundary", {}), errors)
    _check_execution_policy(contract.get("execution_policy", {}), errors)
    _check_model_selection(contract.get("model_selection_policy", {}), errors)
    _check_calibration(contract.get("calibration_policy", {}), errors)
    _check_required_items(
        "required_metric_tables",
        contract.get("required_metric_tables", []),
        [
            "pooled_native_unit_metrics",
            "outer_fold_metrics",
            "source_type_slice_metrics",
            "recording_group_slice_metrics",
            "behavior_label_slice_metrics",
            "matched_6_frame_source_class_balanced_metrics",
            "per_class_source_slice_metrics",
            "probability_calibration_metrics",
            "confusion_pair_metrics",
            "ablation_delta_metrics",
            "cluster_bootstrap_confidence_intervals",
            "holm_corrected_pairwise_tests",
        ],
        errors,
    )
    _check_required_items(
        "required_figures",
        contract.get("required_figures", []),
        [
            "macro_f1_ablation_bar_with_ci",
            "per_behavior_f1_heatmap",
            "confusion_matrix_native_units",
            "calibration_reliability_diagram",
            "source_slice_performance_plot",
            "confusion_pair_error_breakdown",
        ],
        errors,
    )
    _check_artifact_paths("required_package_artifacts", contract.get("required_package_artifacts", {}), errors)
    _check_required_items(
        "required_reproducibility_fields",
        contract.get("required_reproducibility_fields", []),
        [
            "git_commit",
            "git_dirty",
            "python_executable",
            "python_version",
            "torch_version",
            "cuda_available",
            "cuda_version",
            "device_name",
            "seed_values",
            "deterministic_flags",
            "data_snapshot_id",
            "feature_whitelist_sha256",
            "split_manifest_sha256",
            "training_config_sha256",
            "prediction_file_sha256",
        ],
        errors,
    )
    _check_required_items(
        "required_safety_checks",
        contract.get("required_safety_checks", []),
        [
            "model_X_excludes_manual_review_audit_identifier_path_and_label_columns",
            "pig_id_annotation_local_not_identity_split_key",
            "recording_date_or_video_group_safe_split",
            "reviewed_rows_equal_enhanced_rows_after_apply",
            "window_validity_respects_review_training_action",
            "native_temporal_unit_is_primary_metric_unit",
            "outer_test_never_used_for_selection_threshold_or_calibration",
            "hard_negative_review_does_not_auto_change_labels",
        ],
        errors,
    )
    _check_completion_gate(contract.get("completion_gate", {}), errors)

    return {
        "schema_version": "classification_v2_q2_final_package_contract_audit_v1",
        "contract_json": str(contract_json),
        "contract_version": contract.get("version"),
        "target_strength": contract.get("claim_boundary", {}).get("target_strength"),
        "external_generalization_claim": contract.get("claim_boundary", {}).get("external_generalization_claim"),
        "full_oof_predictions_required_before_final_claim": contract.get("execution_policy", {}).get(
            "full_oof_predictions_required_before_final_claim"
        ),
        "outer_test_used_for_model_selection": contract.get("execution_policy", {}).get(
            "outer_test_used_for_model_selection"
        ),
        "outer_test_used_for_threshold_tuning": contract.get("execution_policy", {}).get(
            "outer_test_used_for_threshold_tuning"
        ),
        "outer_test_used_for_calibration_fit": contract.get("execution_policy", {}).get(
            "outer_test_used_for_calibration_fit"
        ),
        "calibration_fit_scope": contract.get("execution_policy", {}).get("calibration_fit_scope"),
        "final_test_is_single_touch": contract.get("execution_policy", {}).get("final_test_is_single_touch"),
        "primary_metric": contract.get("model_selection_policy", {}).get("primary_metric"),
        "model_family_count": len(contract.get("model_selection_policy", {}).get("model_family_labels", [])),
        "calibration_method_count": len(contract.get("calibration_policy", {}).get("allowed_methods", [])),
        "metric_table_count": len(contract.get("required_metric_tables", [])),
        "figure_count": len(contract.get("required_figures", [])),
        "package_artifact_count": len(contract.get("required_package_artifacts", {})),
        "reproducibility_field_count": len(contract.get("required_reproducibility_fields", [])),
        "safety_check_count": len(contract.get("required_safety_checks", [])),
        "can_claim_q2_result": contract.get("completion_gate", {}).get("can_claim_q2_result"),
        "errors": errors,
        "valid": not errors,
    }


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_contract_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check_claim_boundary(boundary: dict[str, Any], errors: list[str]) -> None:
    if boundary.get("target_strength") != "Q2_strong":
        errors.append("target_strength_must_be_Q2_strong")
    primary = str(boundary.get("primary_claim", "")).lower()
    if "recording" not in primary and "video-safe" not in primary:
        errors.append("primary_claim_must_reference_recording_or_video_safe_validation")
    if boundary.get("external_generalization_claim") is not False:
        errors.append("external_generalization_claim_must_be_false")
    prohibited = " ".join(str(item).lower() for item in boundary.get("prohibited_claims", []))
    for token in ["external farm", "external camera", "external cohort", "unseen biological pig", "state-of-the-art"]:
        if token not in prohibited:
            errors.append(f"missing_prohibited_claim={token}")


def _check_execution_policy(policy: dict[str, Any], errors: list[str]) -> None:
    expected = {
        "full_oof_predictions_required_before_final_claim": True,
        "outer_test_used_for_model_selection": False,
        "outer_test_used_for_threshold_tuning": False,
        "outer_test_used_for_calibration_fit": False,
        "calibration_fit_scope": "inner_validation_or_oof_train_folds_only",
        "final_test_is_single_touch": True,
        "requires_explicit_user_authorization_for_full_oof_or_final_test": True,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            errors.append(f"execution_policy_{key}_must_be_{value}")


def _check_model_selection(policy: dict[str, Any], errors: list[str]) -> None:
    if policy.get("selection_unit") != "native_temporal_unit":
        errors.append("selection_unit_must_be_native_temporal_unit")
    if policy.get("primary_metric") != "macro_f1":
        errors.append("primary_metric_must_be_macro_f1")
    _check_required_items(
        "tie_break_metrics",
        policy.get("tie_break_metrics", []),
        ["balanced_accuracy", "multiclass_mcc", "negative_log_likelihood", "top_label_ece"],
        errors,
    )
    if policy.get("candidate_set_is_predeclared") is not True:
        errors.append("candidate_set_is_predeclared_must_be_true")
    _check_required_items("model_family_labels", policy.get("model_family_labels", []), REQUIRED_MODEL_FAMILIES, errors)
    sesoi = policy.get("minimum_effect_size_of_interest", {})
    if float(sesoi.get("macro_f1_absolute_delta", 0.0)) <= 0.0:
        errors.append("sesoi_macro_f1_absolute_delta_must_be_positive")
    if float(sesoi.get("balanced_accuracy_absolute_delta", 0.0)) <= 0.0:
        errors.append("sesoi_balanced_accuracy_absolute_delta_must_be_positive")


def _check_calibration(policy: dict[str, Any], errors: list[str]) -> None:
    _check_required_items(
        "allowed_methods",
        policy.get("allowed_methods", []),
        ["temperature_scaling", "vector_temperature_scaling"],
        errors,
    )
    if policy.get("default_method") != "temperature_scaling":
        errors.append("default_calibration_method_must_be_temperature_scaling")
    _check_required_items(
        "fit_inputs",
        policy.get("fit_inputs", []),
        ["oof_train_fold_logits", "inner_validation_logits"],
        errors,
    )
    forbidden = policy.get("forbidden_fit_inputs", [])
    for item in ["outer_test_logits", "manual_review_columns", "identifier_columns", "path_columns"]:
        if item not in forbidden:
            errors.append(f"forbidden_fit_inputs_missing={item}")
    _check_artifact_paths("calibration_required_outputs", policy.get("required_outputs", {}), errors)


def _check_artifact_paths(name: str, artifacts: dict[str, str], errors: list[str]) -> None:
    if not artifacts:
        errors.append(f"{name}_missing")
        return
    empty = [key for key, value in artifacts.items() if not str(value).strip()]
    if empty:
        errors.append(f"{name}_empty_paths={empty}")


def _check_completion_gate(gate: dict[str, Any], errors: list[str]) -> None:
    if gate.get("can_claim_q2_result") is not False:
        errors.append("contract_must_not_claim_q2_result_without_execution")
    if not str(gate.get("reason", "")).strip():
        errors.append("completion_gate_reason_missing")


def _check_required_items(name: str, values: list[str], required: list[str], errors: list[str]) -> None:
    missing = [item for item in required if item not in values]
    if missing:
        errors.append(f"{name}_missing={missing}")


if __name__ == "__main__":
    main()

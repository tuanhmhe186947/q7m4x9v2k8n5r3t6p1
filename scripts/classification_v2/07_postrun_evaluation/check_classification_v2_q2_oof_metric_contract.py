from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def main() -> None:
    """Validate the Q2 OOF metric contract without running full OOF training."""

    parser = argparse.ArgumentParser(description="Check the classification_v2 Q2 native-unit OOF metric contract.")
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/q2_oof_metric_contract_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_oof_metric_contract_audit.json"),
    )
    args = parser.parse_args()
    audit = check_contract(args.contract_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_contract(contract_json: Path) -> dict[str, Any]:
    """Return a deterministic audit for the paper-facing OOF metric contract."""

    errors: list[str] = []
    warnings: list[str] = []
    contract = _read_json(contract_json, errors)
    _check_claim_boundary(contract.get("claim_boundary", {}), errors)
    _check_execution_policy(contract.get("execution_policy", {}), errors)
    _check_label_order(contract.get("primary_label_order", []), errors)
    _check_prediction_schema(contract.get("required_prediction_columns", []), errors)
    _check_metrics(contract, errors)
    _check_confusion_pairs(contract.get("required_confusion_pairs", []), errors)
    _check_native_collapse(contract.get("required_native_collapse", {}), errors)
    _check_required_items(
        "required_leakage_controls",
        contract.get("required_leakage_controls", []),
        [
            "recording_group_safe_outer_folds",
            "fold-local_standardization",
            "no_manual_review_columns_in_X",
            "no_identifier_or_path_columns_in_X",
            "no_behavior_or_original_behavior_columns_in_X",
            "pig_id_not_used_as_cross_video_identity",
            "outer_test_not_used_for_threshold_tuning",
            "native_temporal_unit_is_primary_metric_unit",
        ],
        errors,
    )
    _check_required_items(
        "required_source_domain_controls",
        contract.get("required_source_domain_controls", []),
        [
            "overall_source_type_slice",
            "per_behavior_source_type_slice",
            "matched_6_frame_legacy_vs_cvat_subset",
            "source_shortcut_probe_or_domain_control_ablation",
        ],
        errors,
    )
    _check_required_items(
        "required_baseline_comparisons",
        contract.get("required_baseline_comparisons", []),
        [
            "B2_spatial_only",
            "B3_actor_image_only",
            "B4_actor_spatial",
            "B5_actor_spatial_partner_context",
            "B6_actor_spatial_partner_multitask",
            "B7_full_candidate_domain_controls",
        ],
        errors,
    )
    outputs = contract.get("required_native_outputs", {})
    if not outputs.get("native_unit_predictions_csv") or not outputs.get("q2_oof_metrics_json"):
        errors.append("required_native_outputs_missing_prediction_or_metric_path")
    if contract.get("statistical_unit") != "native_temporal_unit":
        errors.append("statistical_unit_must_be_native_temporal_unit")
    if "group" not in str(contract.get("primary_split_policy", "")).lower():
        errors.append("primary_split_policy_must_be_group_safe")

    return {
        "schema_version": "classification_v2_q2_oof_metric_contract_audit_v1",
        "contract_json": str(contract_json),
        "contract_version": contract.get("version"),
        "claim_boundary": contract.get("claim_boundary", {}),
        "full_oof_execution_allowed_by_contract": contract.get("execution_policy", {}).get(
            "full_oof_execution_allowed_by_contract"
        ),
        "outer_test_used_for_threshold_tuning": contract.get("execution_policy", {}).get(
            "outer_test_used_for_threshold_tuning"
        ),
        "statistical_unit": contract.get("statistical_unit"),
        "primary_split_policy": contract.get("primary_split_policy"),
        "label_count": len(contract.get("primary_label_order", [])),
        "prediction_column_count": len(contract.get("required_prediction_columns", [])),
        "required_metric_group_count": len(contract.get("required_metric_groups", [])),
        "required_scalar_metric_count": len(contract.get("required_scalar_metrics", [])),
        "confusion_pair_count": len(contract.get("required_confusion_pairs", [])),
        "leakage_control_count": len(contract.get("required_leakage_controls", [])),
        "source_domain_control_count": len(contract.get("required_source_domain_controls", [])),
        "baseline_comparison_count": len(contract.get("required_baseline_comparisons", [])),
        "warnings": warnings,
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
    if "video-safe" not in primary and "recording" not in primary:
        errors.append("primary_claim_must_reference_video_or_recording_safe_validation")
    if boundary.get("external_generalization_claim") is not False:
        errors.append("external_generalization_claim_must_be_false")
    prohibited = " ".join(str(item).lower() for item in boundary.get("prohibited_claims", []))
    for token in ["external farm", "external camera", "unseen biological pig", "sota"]:
        if token not in prohibited:
            errors.append(f"missing_prohibited_claim={token}")


def _check_execution_policy(policy: dict[str, Any], errors: list[str]) -> None:
    if policy.get("full_oof_execution_allowed_by_contract") is not False:
        errors.append("contract_must_not_authorize_full_oof_execution")
    if policy.get("outer_test_used_for_threshold_tuning") is not False:
        errors.append("outer_test_used_for_threshold_tuning_must_be_false")
    if policy.get("model_selection_uses_outer_test") is not False:
        errors.append("model_selection_uses_outer_test_must_be_false")
    if policy.get("requires_explicit_user_authorization_for_full_oof") is not True:
        errors.append("full_oof_must_require_explicit_user_authorization")


def _check_label_order(labels: list[str], errors: list[str]) -> None:
    if labels != list(VALID_BEHAVIORS):
        errors.append(f"primary_label_order_mismatch={labels}")


def _check_prediction_schema(columns: list[str], errors: list[str]) -> None:
    required = ["window_id", "temporal_unit_key", "oof_fold_id", "true_label", "predicted_label"]
    missing = [column for column in required if column not in columns]
    missing.extend(f"prob_{label}" for label in VALID_BEHAVIORS if f"prob_{label}" not in columns)
    if missing:
        errors.append(f"required_prediction_columns_missing={missing}")


def _check_metrics(contract: dict[str, Any], errors: list[str]) -> None:
    _check_required_items(
        "required_metric_groups",
        contract.get("required_metric_groups", []),
        [
            "pooled_native_unit_metrics",
            "outer_fold_metrics",
            "source_type_slice_metrics",
            "recording_group_slice_metrics",
            "behavior_label_slice_metrics",
            "matched_6_frame_source_class_balanced_metrics",
            "per_class_source_slice_metrics",
            "probability_calibration_metrics",
            "paired_cluster_bootstrap_ci",
            "holm_corrected_ablation_tests",
            "confusion_pair_metrics",
        ],
        errors,
    )
    _check_required_items(
        "required_scalar_metrics",
        contract.get("required_scalar_metrics", []),
        [
            "macro_f1",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "multiclass_mcc",
            "negative_log_likelihood",
            "multiclass_brier",
            "top_label_ece",
        ],
        errors,
    )


def _check_confusion_pairs(pairs: list[list[str]], errors: list[str]) -> None:
    observed = {tuple(pair) for pair in pairs}
    required = {
        ("fight", "social-nose"),
        ("fight", "stand"),
        ("fight", "move"),
        ("eat", "stand"),
        ("eat", "explore"),
        ("drink", "stand"),
        ("drink", "explore"),
        ("playwithtoy", "explore"),
        ("playwithtoy", "stand"),
        ("playwithtoy", "move"),
        ("lying", "sitting"),
        ("move", "explore"),
        ("move", "stand"),
    }
    missing = sorted(required.difference(observed))
    invalid = sorted(pair for pair in observed if any(label not in VALID_BEHAVIORS for label in pair))
    if missing:
        errors.append(f"required_confusion_pairs_missing={missing}")
    if invalid:
        errors.append(f"invalid_confusion_pairs={invalid}")


def _check_native_collapse(collapse: dict[str, Any], errors: list[str]) -> None:
    if collapse.get("collapse_from") != "overlapping_sequence_windows":
        errors.append("native_collapse_from_must_be_overlapping_sequence_windows")
    if collapse.get("collapse_to") != "temporal_unit_key":
        errors.append("native_collapse_to_must_be_temporal_unit_key")
    if collapse.get("one_prediction_per_native_unit") is not True:
        errors.append("one_prediction_per_native_unit_must_be_true")
    if collapse.get("window_only_metrics_are_not_primary") is not True:
        errors.append("window_only_metrics_are_not_primary_must_be_true")


def _check_required_items(name: str, values: list[str], required: list[str], errors: list[str]) -> None:
    missing = [item for item in required if item not in values]
    if missing:
        errors.append(f"{name}_missing={missing}")


if __name__ == "__main__":
    main()

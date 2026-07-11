from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


REQUIRED_CONFUSION_PAIRS = {
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


def main() -> None:
    """Validate the Q2 hard-negative mining contract without selecting samples."""

    parser = argparse.ArgumentParser(description="Check classification_v2 Q2 hard-negative mining contract.")
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/q2_hard_negative_mining_contract_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_hard_negative_contract_audit.json"),
    )
    args = parser.parse_args()
    audit = check_contract(args.contract_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_contract(contract_json: Path) -> dict[str, Any]:
    """Return a strict audit of the hard-negative mining policy contract."""

    errors: list[str] = []
    contract = _read_json(contract_json, errors)
    _check_claim_boundary(contract.get("claim_boundary", {}), errors)
    _check_execution_policy(contract.get("execution_policy", {}), errors)
    _check_input_contract(contract.get("input_contract", {}), errors)
    _check_columns(contract, errors)
    _check_confusion_pairs(contract.get("predeclared_confusion_pairs", []), errors)
    _check_selection_policy(contract.get("selection_policy", {}), errors)
    _check_required_items(
        "required_shortlist_columns",
        contract.get("required_shortlist_columns", []),
        [
            "review_shortlist_id",
            "temporal_unit_key",
            "review_unit_id",
            "oof_fold_id",
            "source_type",
            "video_key",
            "true_label",
            "predicted_label",
            "predicted_confidence",
            "true_label_probability",
            "prediction_margin",
            "confusion_pair",
            "selection_reason",
            "priority_score",
            "review_template",
            "manual_review_decision",
        ],
        errors,
    )
    _check_required_items(
        "required_audit_counts",
        contract.get("required_audit_counts", []),
        [
            "input_native_unit_rows",
            "eligible_oof_rows",
            "confusion_error_rows",
            "uncertain_correct_rows",
            "shortlist_rows",
            "duplicate_temporal_unit_key",
            "duplicate_review_unit_id",
            "missing_review_unit_count",
            "per_fold_counts",
            "per_source_counts",
            "per_true_label_counts",
            "per_confusion_pair_counts",
        ],
        errors,
    )
    _check_required_items(
        "leakage_guards",
        contract.get("leakage_guards", []),
        [
            "selection_uses_oof_predictions_only",
            "selection_thresholds_predeclared_or_inner_train_only",
            "outer_test_never_used_to_choose_thresholds",
            "review_shortlist_does_not_change_training_labels",
            "manual_decisions_apply_only_through_review_unit_decision_pipeline",
            "pig_id_annotation_local_not_identity_split_key",
        ],
        errors,
    )
    _check_required_items(
        "review_safety_rules",
        contract.get("review_safety_rules", []),
        [
            "pending_decisions_do_not_apply_corrected_behavior",
            "exclude_sets_training_mask_or_weight_only_without_dropping_rows",
            "corrected_labels_require_review_unit_id_and_apply_scope",
            "interaction_shortlist_requires_full_frame_partner_context_when_available",
        ],
        errors,
    )
    outputs = contract.get("required_output_artifacts", {})
    if not outputs.get("shortlist_csv") or not outputs.get("audit_json"):
        errors.append("required_output_artifacts_missing_shortlist_or_audit")

    return {
        "schema_version": "classification_v2_q2_hard_negative_contract_audit_v1",
        "contract_json": str(contract_json),
        "contract_version": contract.get("version"),
        "target_strength": contract.get("claim_boundary", {}).get("target_strength"),
        "external_generalization_claim": contract.get("claim_boundary", {}).get(
            "external_generalization_claim"
        ),
        "requires_oof_native_predictions": contract.get("execution_policy", {}).get(
            "requires_oof_native_predictions"
        ),
        "outer_test_used_for_threshold_tuning": contract.get("execution_policy", {}).get(
            "outer_test_used_for_threshold_tuning"
        ),
        "automatic_label_change_allowed": contract.get("execution_policy", {}).get(
            "automatic_label_change_allowed"
        ),
        "predeclared_confusion_pair_count": len(contract.get("predeclared_confusion_pairs", [])),
        "required_shortlist_column_count": len(contract.get("required_shortlist_columns", [])),
        "required_audit_count_count": len(contract.get("required_audit_counts", [])),
        "leakage_guard_count": len(contract.get("leakage_guards", [])),
        "review_safety_rule_count": len(contract.get("review_safety_rules", [])),
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
    if boundary.get("external_generalization_claim") is not False:
        errors.append("external_generalization_claim_must_be_false")
    primary = str(boundary.get("primary_claim", "")).lower()
    if "recording" not in primary and "video-safe" not in primary:
        errors.append("primary_claim_must_reference_recording_or_video_safe_validation")


def _check_execution_policy(policy: dict[str, Any], errors: list[str]) -> None:
    expected = {
        "requires_oof_native_predictions": True,
        "requires_fold_assignment_lineage": True,
        "may_run_without_full_oof_predictions": False,
        "outer_test_used_for_threshold_tuning": False,
        "automatic_label_change_allowed": False,
        "automatic_row_drop_allowed": False,
        "review_unit_decision_required_before_training_label_change": True,
    }
    for key, value in expected.items():
        if policy.get(key) is not value:
            errors.append(f"execution_policy_{key}_must_be_{value}")


def _check_input_contract(inputs: dict[str, str], errors: list[str]) -> None:
    required = [
        "native_unit_predictions_csv",
        "q2_oof_metrics_json",
        "review_unit_manifest_csv",
        "fold_assignments_csv",
    ]
    missing = [name for name in required if not inputs.get(name)]
    if missing:
        errors.append(f"input_contract_missing={missing}")


def _check_columns(contract: dict[str, Any], errors: list[str]) -> None:
    _check_required_items(
        "required_input_columns",
        contract.get("required_input_columns", []),
        [
            "temporal_unit_key",
            "oof_fold_id",
            "true_label",
            "native_predicted_behavior",
            "native_metric_include",
            "source_type",
            "video_key",
            "behavior_label",
        ],
        errors,
    )
    missing_probability = [
        f"prob_{label}"
        for label in VALID_BEHAVIORS
        if f"prob_{label}" not in contract.get("required_probability_columns", [])
    ]
    if missing_probability:
        errors.append(f"required_probability_columns_missing={missing_probability}")


def _check_confusion_pairs(pairs: list[list[str]], errors: list[str]) -> None:
    observed = {tuple(pair) for pair in pairs}
    missing = sorted(REQUIRED_CONFUSION_PAIRS.difference(observed))
    invalid = sorted(pair for pair in observed if any(label not in VALID_BEHAVIORS for label in pair))
    if missing:
        errors.append(f"predeclared_confusion_pairs_missing={missing}")
    if invalid:
        errors.append(f"invalid_confusion_pairs={invalid}")


def _check_selection_policy(policy: dict[str, Any], errors: list[str]) -> None:
    expected_true = [
        "include_only_oof_mistakes",
        "include_predeclared_confusion_pairs_only",
        "allow_high_uncertainty_correct_predictions",
        "per_fold_class_source_cap_required",
    ]
    for key in expected_true:
        if policy.get(key) is not True:
            errors.append(f"selection_policy_{key}_must_be_true")
    if policy.get("uncertainty_source") != "native OOF probabilities only":
        errors.append("uncertainty_source_must_be_native_oof_probabilities_only")
    confidence = float(policy.get("default_high_confidence_error_threshold", -1.0))
    margin = float(policy.get("default_low_margin_threshold", -1.0))
    if not 0.5 <= confidence <= 1.0:
        errors.append("default_high_confidence_error_threshold_out_of_range")
    if not 0.0 < margin < 0.5:
        errors.append("default_low_margin_threshold_out_of_range")
    sort_keys = policy.get("deterministic_sort_keys", [])
    if not sort_keys or sort_keys[0] != "priority_score_desc":
        errors.append("deterministic_sort_keys_must_start_with_priority_score_desc")


def _check_required_items(name: str, values: list[str], required: list[str], errors: list[str]) -> None:
    missing = [item for item in required if item not in values]
    if missing:
        errors.append(f"{name}_missing={missing}")


if __name__ == "__main__":
    main()

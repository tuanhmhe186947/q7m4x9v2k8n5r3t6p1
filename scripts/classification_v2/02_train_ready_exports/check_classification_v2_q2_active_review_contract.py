from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_DECISION_COLUMNS = [
    "review_item_id",
    "review_unit_id",
    "review_unit_type",
    "temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "pig_id",
    "track_id",
    "object_track_key",
    "unit_start_frame",
    "unit_end_frame",
    "display_frame_indices",
    "review_template",
    "behavior_label",
    "original_behavior",
    "review_reason",
    "apply_scope",
    "manual_review_decision",
    "manual_corrected_behavior",
    "manual_label_strength",
    "manual_training_action",
    "manual_sample_weight",
    "manual_note",
]


def main() -> None:
    """Validate the active-review loop policy without applying decisions."""

    parser = argparse.ArgumentParser(description="Check classification_v2 Q2 active-review loop contract.")
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/q2_active_review_loop_contract_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_active_review_contract_audit.json"),
    )
    args = parser.parse_args()
    audit = check_contract(args.contract_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_contract(contract_json: Path) -> dict[str, Any]:
    """Return a deterministic audit for active-review loop safety."""

    errors: list[str] = []
    contract = _read_json(contract_json, errors)
    _check_claim_boundary(contract.get("claim_boundary", {}), errors)
    _check_execution_policy(contract.get("execution_policy", {}), errors)
    _check_artifact_paths("input_artifacts", contract.get("input_artifacts", {}), errors)
    _check_artifact_paths("review_gui_outputs", contract.get("review_gui_outputs", {}), errors)
    _check_artifact_paths("apply_outputs", contract.get("apply_outputs", {}), errors)
    _check_required_items(
        "required_decision_columns",
        contract.get("required_decision_columns", []),
        REQUIRED_DECISION_COLUMNS,
        errors,
    )
    _check_review_groups(contract.get("review_group_policy", {}), errors)
    _check_required_items(
        "allowed_manual_review_decisions",
        contract.get("allowed_manual_review_decisions", []),
        ["pending", "accept", "corrected", "exclude"],
        errors,
    )
    _check_required_items(
        "required_gui_context",
        contract.get("required_gui_context", []),
        [
            "legacy_crop_loading",
            "cvat_video_bbox_loading",
            "video_key_to_30fps_alias_resolution",
            "uppercase_lowercase_mp4_resolution",
            "recursive_data_videos_search",
            "full_frame_partner_context_for_interaction_when_available",
            "roi_overlay_for_roi_intent_when_available",
            "bbox_sequence_contact_sheet",
        ],
        errors,
    )
    _check_required_items(
        "apply_safety_rules",
        contract.get("apply_safety_rules", []),
        [
            "pending_units_are_ignored",
            "accept_keeps_behavior_label",
            "corrected_applies_to_entire_review_unit",
            "exclude_sets_include_flag_false_weight_zero_without_row_drop",
            "duplicate_review_unit_decisions_fail_or_are_deterministically_reported",
            "behavior_before_review_is_preserved",
            "enhanced_frame_features_are_not_overwritten",
        ],
        errors,
    )
    _check_required_items(
        "required_audit_counts",
        contract.get("required_audit_counts", []),
        [
            "decisions_loaded",
            "pending_ignored",
            "accepted_units",
            "corrected_units",
            "excluded_units",
            "affected_frames",
            "changed_behavior_frames",
            "excluded_frames",
            "duplicate_decision_count",
            "missing_review_unit_count",
            "rows_input",
            "rows_output",
        ],
        errors,
    )
    _check_required_items(
        "train_ready_rebuild_rules",
        contract.get("train_ready_rebuild_rules", []),
        [
            "rebuild_sequence_windows_from_reviewed_frame_features",
            "window_validity_respects_review_include_in_training",
            "window_validity_respects_review_training_action",
            "mixed_transition_windows_are_flagged_not_silently_dropped",
            "model_X_excludes_manual_review_audit_identifier_path_and_label_columns",
        ],
        errors,
    )

    return {
        "schema_version": "classification_v2_q2_active_review_contract_audit_v1",
        "contract_json": str(contract_json),
        "contract_version": contract.get("version"),
        "target_strength": contract.get("claim_boundary", {}).get("target_strength"),
        "external_generalization_claim": contract.get("claim_boundary", {}).get("external_generalization_claim"),
        "active_review_can_apply_without_human_decision": contract.get("execution_policy", {}).get(
            "active_review_can_apply_without_human_decision"
        ),
        "pending_decisions_apply": contract.get("execution_policy", {}).get("pending_decisions_apply"),
        "exclude_drops_rows": contract.get("execution_policy", {}).get("exclude_drops_rows"),
        "decision_key": contract.get("execution_policy", {}).get("decision_key"),
        "decision_column_count": len(contract.get("required_decision_columns", [])),
        "gui_context_count": len(contract.get("required_gui_context", [])),
        "apply_safety_rule_count": len(contract.get("apply_safety_rules", [])),
        "required_audit_count_count": len(contract.get("required_audit_counts", [])),
        "train_ready_rebuild_rule_count": len(contract.get("train_ready_rebuild_rules", [])),
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
    if "video-safe" not in str(boundary.get("primary_claim", "")).lower():
        errors.append("primary_claim_must_reference_video_safe_validation")


def _check_execution_policy(policy: dict[str, Any], errors: list[str]) -> None:
    expected = {
        "active_review_can_select_items": True,
        "active_review_can_apply_without_human_decision": False,
        "pending_decisions_apply": False,
        "corrected_requires_manual_corrected_behavior": True,
        "exclude_drops_rows": False,
        "decision_key": "review_unit_id",
        "full_oof_execution_required_before_model_claim_update": True,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            errors.append(f"execution_policy_{key}_must_be_{value}")


def _check_artifact_paths(name: str, artifacts: dict[str, str], errors: list[str]) -> None:
    if not artifacts:
        errors.append(f"{name}_missing")
        return
    empty = [key for key, value in artifacts.items() if not str(value).strip()]
    if empty:
        errors.append(f"{name}_empty_paths={empty}")


def _check_review_groups(groups: dict[str, list[str]], errors: list[str]) -> None:
    expected = {
        "interaction": ["fight", "social-nose"],
        "ROI-intent": ["eat", "drink", "playwithtoy"],
        "motion/context": ["move", "explore", "stand"],
        "posture": ["lying", "sitting"],
    }
    for name, labels in expected.items():
        if groups.get(name) != labels:
            errors.append(f"review_group_{name}_mismatch={groups.get(name)}")
    if "stand" in groups.get("posture", []):
        errors.append("stand_must_not_be_posture")
    if "fight" in groups.get("motion/context", []):
        errors.append("fight_must_not_be_motion_context")
    if "playwithtoy" not in groups.get("ROI-intent", []):
        errors.append("playwithtoy_must_be_roi_intent")


def _check_required_items(name: str, values: list[str], required: list[str], errors: list[str]) -> None:
    missing = [item for item in required if item not in values]
    if missing:
        errors.append(f"{name}_missing={missing}")


if __name__ == "__main__":
    main()

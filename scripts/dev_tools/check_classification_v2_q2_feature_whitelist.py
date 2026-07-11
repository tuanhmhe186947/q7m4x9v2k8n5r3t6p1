from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any


REQUIRED_BRANCHES = {
    "actor_letterbox_image_sequence",
    "window_tabular_context",
    "spatial_temporal_sequence",
    "partner_full_frame_context",
}

REQUIRED_FORBIDDEN_PATTERNS = [
    "manual_*",
    "review_*",
    "*review*",
    "*decision*",
    "*corrected*",
    "behavior_before_review",
    "original_behavior",
    "behavior_label",
    "behavior_temporal_final",
    "*label*",
    "*policy*",
    "*reason*",
    "review_unit_id",
    "window_id",
    "window_uid",
    "temporal_unit_key",
    "frame_uid",
    "image_context_id",
    "video_key",
    "dataset_id",
    "pig_id",
    "track_id",
    "object_track_key",
    "*path*",
    "*file*",
    "target_roi_*",
    "roi_target_*",
]

PROBE_FORBIDDEN_COLUMNS = [
    "manual_review_decision",
    "review_sample_weight",
    "behavior_before_review",
    "original_behavior",
    "behavior_label",
    "review_reason",
    "review_unit_id",
    "window_id",
    "window_uid",
    "temporal_unit_key",
    "frame_uid",
    "image_context_id",
    "video_key",
    "dataset_id",
    "pig_id",
    "track_id",
    "object_track_key",
    "crop_path",
    "source_file",
    "target_roi_contact",
]


def main() -> None:
    """Validate the paper-facing Q2 model-input whitelist contract."""

    parser = argparse.ArgumentParser(description="Check classification_v2 Q2 feature whitelist.")
    parser.add_argument(
        "--whitelist-json",
        type=Path,
        default=Path("configs/classification_v2/q2_feature_whitelist_v1.json"),
    )
    parser.add_argument(
        "--trainer-contract-v1",
        type=Path,
        default=Path("configs/classification_v2/trainer_contract_v1.json"),
    )
    parser.add_argument(
        "--trainer-contract-v2",
        type=Path,
        default=Path("configs/classification_v2/trainer_contract_v2.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_feature_whitelist_audit.json"),
    )
    args = parser.parse_args()
    audit = check_whitelist(
        whitelist_json=args.whitelist_json,
        trainer_contract_v1=args.trainer_contract_v1,
        trainer_contract_v2=args.trainer_contract_v2,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_whitelist(
    *,
    whitelist_json: Path,
    trainer_contract_v1: Path,
    trainer_contract_v2: Path,
) -> dict[str, Any]:
    """Return deterministic leakage and branch-coverage checks for Q2 inputs."""

    errors: list[str] = []
    whitelist = _read_json(whitelist_json, errors)
    trainer_v1 = _read_json(trainer_contract_v1, errors)
    trainer_v2 = _read_json(trainer_contract_v2, errors)
    policy = whitelist.get("selection_policy", {})
    if policy.get("never_use_all_numeric_columns") is not True:
        errors.append("never_use_all_numeric_columns_must_be_true")
    if policy.get("fail_closed_on_unknown_columns") is not True:
        errors.append("fail_closed_on_unknown_columns_must_be_true")
    for key in [
        "label_columns_allowed_in_X",
        "manual_review_columns_allowed_in_X",
        "identifier_columns_allowed_in_X",
        "path_columns_allowed_in_X",
        "outer_test_dependent_columns_allowed_in_X",
    ]:
        if policy.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    _check_claim_boundary(whitelist.get("claim_boundary", {}), errors)
    branches = whitelist.get("input_branches", {})
    missing_branches = sorted(REQUIRED_BRANCHES.difference(branches))
    if missing_branches:
        errors.append(f"missing_input_branches={missing_branches}")
    _check_branch_contracts(branches, errors)
    forbidden_patterns = whitelist.get("forbidden_model_input_patterns", [])
    missing_forbidden = [pattern for pattern in REQUIRED_FORBIDDEN_PATTERNS if pattern not in forbidden_patterns]
    if missing_forbidden:
        errors.append(f"missing_forbidden_patterns={missing_forbidden}")
    leaked_probe_columns = [
        column for column in PROBE_FORBIDDEN_COLUMNS if not _is_forbidden(column, forbidden_patterns)
    ]
    if leaked_probe_columns:
        errors.append(f"forbidden_probe_columns_not_blocked={leaked_probe_columns}")
    target = whitelist.get("target_y", {})
    if len(target.get("allowed_labels", [])) != 10:
        errors.append("allowed_labels_must_have_10_behaviors")
    _check_required_items(
        "mask_and_weight_inputs",
        whitelist.get("mask_and_weight_inputs", []),
        ["window_valid_for_main_train", "review_sample_weight", "event_balanced_sample_weight"],
        errors,
    )
    tabular_count = len(trainer_v1.get("tabular_feature_whitelist", []))
    spatial_count = len(trainer_v2.get("spatial_sequence_feature_whitelist", []))
    if tabular_count <= 0:
        errors.append("trainer_contract_v1_tabular_whitelist_empty")
    if spatial_count <= 0:
        errors.append("trainer_contract_v2_spatial_whitelist_empty")

    return {
        "schema_version": "classification_v2_q2_feature_whitelist_audit_v1",
        "whitelist_json": str(whitelist_json),
        "contract_version": whitelist.get("version"),
        "target_strength": whitelist.get("claim_boundary", {}).get("target_strength"),
        "external_generalization_claim": whitelist.get("claim_boundary", {}).get(
            "external_generalization_claim"
        ),
        "never_use_all_numeric_columns": policy.get("never_use_all_numeric_columns"),
        "fail_closed_on_unknown_columns": policy.get("fail_closed_on_unknown_columns"),
        "input_branch_count": len(branches),
        "forbidden_pattern_count": len(forbidden_patterns),
        "forbidden_probe_column_count": len(PROBE_FORBIDDEN_COLUMNS),
        "forbidden_probe_columns_not_blocked": leaked_probe_columns,
        "allowed_label_count": len(target.get("allowed_labels", [])),
        "tabular_trainer_whitelist_count": int(tabular_count),
        "spatial_trainer_whitelist_count": int(spatial_count),
        "errors": errors,
        "valid": not errors,
    }


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check_claim_boundary(boundary: dict[str, Any], errors: list[str]) -> None:
    if boundary.get("target_strength") != "Q2_strong":
        errors.append("target_strength_must_be_Q2_strong")
    if boundary.get("external_generalization_claim") is not False:
        errors.append("external_generalization_claim_must_be_false")
    identity_scope = str(boundary.get("pig_id_identity_scope", "")).lower()
    if "annotation-local" not in identity_scope or "never" not in identity_scope:
        errors.append("pig_id_identity_scope_must_be_annotation_local")


def _check_branch_contracts(branches: dict[str, Any], errors: list[str]) -> None:
    actor = branches.get("actor_letterbox_image_sequence", {})
    if actor.get("enabled") is not True:
        errors.append("actor_letterbox_image_sequence_must_be_enabled")
    if actor.get("cache_policy") != "letterbox_preserve_aspect_rgb_pad_black_v1":
        errors.append("actor_cache_policy_must_be_letterbox_preserve_aspect")
    tabular = branches.get("window_tabular_context", {})
    if "trainer_contract_v1.json#tabular_feature_whitelist" not in str(tabular.get("source_contract", "")):
        errors.append("tabular_branch_must_reference_trainer_contract_v1_whitelist")
    spatial = branches.get("spatial_temporal_sequence", {})
    if "trainer_contract_v2.json#spatial_sequence_feature_whitelist" not in str(spatial.get("source_contract", "")):
        errors.append("spatial_branch_must_reference_trainer_contract_v2_whitelist")
    partner = branches.get("partner_full_frame_context", {})
    if partner.get("enabled_for_interaction_models") is not True:
        errors.append("partner_context_must_be_enabled_for_interaction_models")


def _is_forbidden(column: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(column, pattern) for pattern in patterns)


def _check_required_items(name: str, values: list[str], required: list[str], errors: list[str]) -> None:
    missing = [item for item in required if item not in values]
    if missing:
        errors.append(f"{name}_missing={missing}")


if __name__ == "__main__":
    main()

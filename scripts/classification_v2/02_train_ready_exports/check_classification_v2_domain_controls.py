"""Check source-aware weighting policies remain train-only and non-duplicating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.training.samplers import WEIGHT_POLICIES, build_training_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check classification_v2 domain control contracts."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/source_matched_views/check_domain_controls.json"),
    )
    parser.add_argument(
        "--spatial-source-probe-audit",
        type=Path,
        default=Path("outputs/classification_v2/domain_controls/grouped_spatial_source_probe.json"),
    )
    parser.add_argument(
        "--feature-shift-audit",
        type=Path,
        default=Path("outputs/classification_v2/domain_controls/domain_feature_shift_audit.json"),
    )
    parser.add_argument(
        "--source-probe-audit",
        type=Path,
        default=Path("outputs/classification_v2/domain_controls/grouped_source_probe_audit.json"),
    )
    parser.add_argument(
        "--availability-probe-audit",
        type=Path,
        default=Path(
            "outputs/classification_v2/domain_controls/"
            "grouped_availability_behavior_probe_audit.json"
        ),
    )
    args = parser.parse_args()
    training = pd.DataFrame(
        {
            "window_id": [f"w{index}" for index in range(8)],
            "behavior_true": ["fight", "fight", "stand", "stand", "stand", "eat", "eat", "eat"],
            "source_type": ["cvat_tracking_xml"] * 5 + ["legacy_recovered"] * 3,
            "temporal_unit_key": ["u0", "u0", "u1", "u2", "u2", "u3", "u3", "u3"],
        }
    )
    audits = {}
    errors: list[str] = []
    for policy in WEIGHT_POLICIES:
        weights, audit = build_training_weights(training, policy=policy)
        audits[policy] = audit
        if len(weights) != len(training) or not weights.index.equals(training.index):
            errors.append(f"weight_alignment_failed={policy}")
        if abs(float(weights.mean()) - 1.0) > 1e-12:
            errors.append(f"weight_mean_not_one={policy}")
        if audit["row_duplication_used"]:
            errors.append(f"row_duplication_used={policy}")
    source_probe = json.loads(args.source_probe_audit.read_text(encoding="utf-8"))
    availability_probe = json.loads(
        args.availability_probe_audit.read_text(encoding="utf-8")
    )
    spatial_probe = json.loads(args.spatial_source_probe_audit.read_text(encoding="utf-8"))
    feature_shift = json.loads(args.feature_shift_audit.read_text(encoding="utf-8"))
    if source_probe.get("schema_version") != "classification_v2_grouped_source_probe_v2":
        errors.append(f"source_probe_schema={source_probe.get('schema_version')}")
    if source_probe.get("statistical_unit") != "native_temporal_unit":
        errors.append("source_probe_not_native_unit")
    if source_probe.get("valid") is not True:
        errors.append("source_probe_invalid")
    if source_probe.get("oof_fold_count", 0) < 2:
        errors.append("source_probe_insufficient_folds")
    if source_probe.get("oof_prediction_rows") != source_probe.get(
        "eligible_native_unit_rows"
    ):
        errors.append("source_probe_native_oof_count_mismatch")
    if source_probe.get("eligible_native_unit_to_oof_row_loss") != 0:
        errors.append("source_probe_native_row_loss")
    if source_probe.get("every_eligible_native_unit_tested_once") is not True:
        errors.append("source_probe_oof_coverage")
    if source_probe.get("source_identifier_in_features") is not False:
        errors.append("source_identifier_entered_probe_features")
    if not source_probe.get("feature_whitelist"):
        errors.append("source_probe_missing_feature_whitelist")
    if source_probe.get("input_contract", {}).get("ordered_window_lineage_match") is not True:
        errors.append("source_probe_window_lineage_unbound")
    if source_probe.get("role_contract", {}).get("valid") is not True:
        errors.append("source_probe_role_contract_invalid")
    if not all(
        fold.get("validation_and_test_excluded_from_fit")
        for fold in source_probe.get("folds", [])
    ):
        errors.append("source_probe_fit_leakage_flag")

    if availability_probe.get("schema_version") != (
        "classification_v2_availability_behavior_probe_v1"
    ):
        errors.append(
            f"availability_probe_schema={availability_probe.get('schema_version')}"
        )
    if availability_probe.get("statistical_unit") != "native_temporal_unit":
        errors.append("availability_probe_not_native_unit")
    if availability_probe.get("valid") is not True:
        errors.append("availability_probe_invalid")
    if availability_probe.get("oof_fold_count", 0) < 2:
        errors.append("availability_probe_insufficient_folds")
    if availability_probe.get("availability_features_enter_classifier_x") is not False:
        errors.append("availability_probe_entered_classifier_x")
    if availability_probe.get("target_derived_availability_columns"):
        errors.append("availability_probe_target_derived_fields")
    if not availability_probe.get("availability_columns"):
        errors.append("availability_probe_missing_explicit_columns")
    if availability_probe.get("input_contract", {}).get(
        "ordered_window_lineage_match"
    ) is not True:
        errors.append("availability_probe_window_lineage_unbound")
    if availability_probe.get("role_contract", {}).get("valid") is not True:
        errors.append("availability_probe_role_contract_invalid")
    if availability_probe.get("oof_prediction_rows") != availability_probe.get(
        "eligible_native_unit_rows"
    ):
        errors.append("availability_probe_native_oof_count_mismatch")
    if availability_probe.get("eligible_native_unit_to_oof_row_loss") != 0:
        errors.append("availability_probe_native_row_loss")
    if not all(
        fold.get("validation_and_test_excluded_from_fit")
        for fold in availability_probe.get("folds", [])
    ):
        errors.append("availability_probe_fit_leakage_flag")

    if (
        spatial_probe.get("fold_count", 0) <= 0
        or spatial_probe.get("source_type_in_model_x") is not False
    ):
        errors.append("grouped_spatial_source_probe_contract")
    if set(spatial_probe.get("pooled_controls", {})) != {
        "real_sequence",
        "repeat_first_frame",
        "mean_only",
    }:
        errors.append("grouped_spatial_source_probe_controls")
    if feature_shift.get("schema_version") != "classification_v2_domain_feature_shift_v2":
        errors.append(f"domain_feature_shift_schema={feature_shift.get('schema_version')}")
    if feature_shift.get("eligible_rows") != source_probe.get("eligible_window_rows"):
        errors.append("domain_feature_shift_eligible_scope")
    if feature_shift.get("feature_count") != source_probe.get("feature_count"):
        errors.append("domain_feature_shift_scope")
    if feature_shift.get("feature_whitelist_sha256") != source_probe.get(
        "feature_whitelist_sha256"
    ):
        errors.append("domain_feature_shift_whitelist_mismatch")
    if feature_shift.get("ordered_window_id_sha256_observed") != source_probe.get(
        "input_contract", {}
    ).get("ordered_window_id_sha256_observed"):
        errors.append("domain_feature_shift_window_lineage_mismatch")
    if feature_shift.get("camera_safe_claim_allowed") is not False:
        errors.append("camera_safe_claim_not_blocked_without_metadata")
    result = {
        "schema_version": "classification_v2_domain_controls_check_v2",
        "policies": audits,
        "all_policies_training_fold_only": all(
            audit["fit_scope"] == "training_fold_only"
            for audit in audits.values()
        ),
        "grouped_source_probe": source_probe,
        "grouped_availability_behavior_probe": availability_probe,
        "grouped_spatial_source_probe": spatial_probe,
        "domain_feature_shift": feature_shift,
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

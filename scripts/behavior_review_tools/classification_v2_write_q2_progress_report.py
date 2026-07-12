from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a compact Q2 classification_v2 progress report."
    )
    parser.add_argument(
        "--snapshot-check-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_snapshot_check_audit.json"),
    )
    parser.add_argument(
        "--baseline-config-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_baseline_config_audit.json"),
    )
    parser.add_argument(
        "--baseline-smoke-check-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_baseline_smoke_check_audit.json"),
    )
    parser.add_argument(
        "--reproducibility-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/training_reproducibility_cuda_post_s0a/"
            "reproducibility_audit.json"
        ),
    )
    parser.add_argument(
        "--b4-seed-variance-check-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/b4_seed_variance_check_audit.json"),
    )
    parser.add_argument(
        "--q2-oof-metric-contract-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_oof_metric_contract_audit.json"),
    )
    parser.add_argument(
        "--q2-hard-negative-contract-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_hard_negative_contract_audit.json"),
    )
    parser.add_argument(
        "--q2-active-review-contract-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_active_review_contract_audit.json"),
    )
    parser.add_argument(
        "--q2-final-package-contract-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_final_package_contract_audit.json"),
    )
    parser.add_argument(
        "--q2-feature-whitelist-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_feature_whitelist_audit.json"),
    )
    parser.add_argument(
        "--q2-final-package-stub-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_final_package_stub_audit.json"),
    )
    parser.add_argument(
        "--full-oof-preflight-policy-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_oof_preflight_policy_audit.json"),
    )
    parser.add_argument(
        "--full-oof-authorization-policy-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization_policy_audit.json"
        ),
    )
    parser.add_argument(
        "--full-oof-authorization-template-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization_template_audit.json"
        ),
    )
    parser.add_argument(
        "--full-oof-preflight-freshness-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_preflight_freshness_audit.json"
        ),
    )
    parser.add_argument(
        "--full-oof-run-plan-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_multimodal_oof_run_plan.json"
        ),
    )
    parser.add_argument(
        "--full-runner-default-config-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_runner_default_config_audit.json"
        ),
    )
    parser.add_argument(
        "--full-oof-execution-gate-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_execution_gate_audit.json"
        ),
    )
    parser.add_argument(
        "--full-oof-launch-packet-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_launch_packet_audit.json"
        ),
    )
    parser.add_argument(
        "--full-oof-completion-gate-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_completion_gate_audit.json"
        ),
    )
    parser.add_argument(
        "--full-oof-postrun-registration-packet-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_postrun_registration_packet_audit.json"
        ),
    )
    parser.add_argument(
        "--image-cache-inventory-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/image_cache_inventory_audit.json"
        ),
    )
    parser.add_argument(
        "--image-cache-letterbox-policy-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "image_cache_letterbox_policy_audit.json"
        ),
    )
    parser.add_argument(
        "--visual-interaction-cache-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "visual_interaction_cache_audit.json"
        ),
    )
    parser.add_argument(
        "--pig-id-locality-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/pig_id_locality_contract_audit.json"
        ),
    )
    parser.add_argument(
        "--split-group-leakage-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/split_group_leakage_audit.json"
        ),
    )
    parser.add_argument(
        "--interaction-context-index-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "interaction_context_index_audit.json"
        ),
    )
    parser.add_argument(
        "--model-architecture-contract-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "model_architecture_contract_audit.json"
        ),
    )
    parser.add_argument(
        "--ablation-shortcut-contract-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "ablation_shortcut_contract_audit.json"
        ),
    )
    parser.add_argument(
        "--ablation-reporting-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "ablation_reporting_audit.json"
        ),
    )
    parser.add_argument(
        "--full-learned-oof-contract-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_learned_oof_contract_audit.json"
        ),
    )
    parser.add_argument(
        "--data-module-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/data_module_audit.json"),
    )
    parser.add_argument(
        "--training-config-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/training_config_audit.json"
        ),
    )
    parser.add_argument(
        "--q2-ablation-matrix-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/q2_ablation_matrix_audit.json"
        ),
    )
    parser.add_argument(
        "--visual-context-ablation-smoke-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "visual_context_ablation_smoke_audit.json"
        ),
    )
    parser.add_argument(
        "--source-domain-control-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/source_domain_controls/"
            "source_domain_control_audit.json"
        ),
    )
    parser.add_argument(
        "--loader-input-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/loader_input_audit.json"
        ),
    )
    parser.add_argument(
        "--spatial-sequence-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/spatial_sequence_audit.json"
        ),
    )
    parser.add_argument(
        "--feature-semantics-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/feature_semantics_audit.json"
        ),
    )
    parser.add_argument(
        "--event-weight-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/event_weight_audit.json"),
    )
    parser.add_argument(
        "--image-sequence-loader-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/"
            "image_sequence_loader_smoke_audit.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_progress_report.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_progress_report.md"),
    )
    args = parser.parse_args()

    snapshot = _load_optional_json(args.snapshot_check_json)
    baseline_configs = _load_optional_json(args.baseline_config_audit_json)
    baseline_smokes = _load_optional_json(args.baseline_smoke_check_json)
    reproducibility = _load_optional_json(args.reproducibility_audit_json)
    b4_seed_variance = _load_optional_json(args.b4_seed_variance_check_json)
    q2_oof_metric_contract = _load_optional_json(args.q2_oof_metric_contract_json)
    q2_hard_negative_contract = _load_optional_json(args.q2_hard_negative_contract_json)
    q2_active_review_contract = _load_optional_json(args.q2_active_review_contract_json)
    q2_final_package_contract = _load_optional_json(args.q2_final_package_contract_json)
    q2_feature_whitelist = _load_optional_json(args.q2_feature_whitelist_json)
    q2_final_package_stub = _load_optional_json(args.q2_final_package_stub_json)
    full_oof_preflight_policy = _load_optional_json(args.full_oof_preflight_policy_json)
    full_oof_authorization_policy = _load_optional_json(
        args.full_oof_authorization_policy_json
    )
    full_oof_authorization_template = _load_optional_json(
        args.full_oof_authorization_template_json
    )
    full_oof_preflight_freshness = _load_optional_json(
        args.full_oof_preflight_freshness_json
    )
    full_oof_run_plan = _load_optional_json(args.full_oof_run_plan_json)
    full_runner_default_config = _load_optional_json(
        args.full_runner_default_config_json
    )
    full_oof_execution_gate = _load_optional_json(
        args.full_oof_execution_gate_json
    )
    full_oof_launch_packet = _load_optional_json(args.full_oof_launch_packet_json)
    full_oof_completion_gate = _load_optional_json(
        args.full_oof_completion_gate_json
    )
    full_oof_postrun_registration_packet = _load_optional_json(
        args.full_oof_postrun_registration_packet_json
    )
    image_cache_inventory = _load_optional_json(args.image_cache_inventory_json)
    image_cache_letterbox_policy = _load_optional_json(
        args.image_cache_letterbox_policy_json
    )
    visual_interaction_cache = _load_optional_json(args.visual_interaction_cache_json)
    pig_id_locality = _load_optional_json(args.pig_id_locality_json)
    split_group_leakage = _load_optional_json(args.split_group_leakage_json)
    interaction_context_index = _load_optional_json(
        args.interaction_context_index_json
    )
    model_architecture_contract = _load_optional_json(
        args.model_architecture_contract_json
    )
    ablation_shortcut_contract = _load_optional_json(
        args.ablation_shortcut_contract_json
    )
    ablation_reporting = _load_optional_json(args.ablation_reporting_json)
    full_learned_oof_contract = _load_optional_json(
        args.full_learned_oof_contract_json
    )
    data_module_audit = _load_optional_json(args.data_module_audit_json)
    training_config_audit = _load_optional_json(args.training_config_audit_json)
    q2_ablation_matrix = _load_optional_json(args.q2_ablation_matrix_json)
    visual_context_ablation_smoke = _load_optional_json(
        args.visual_context_ablation_smoke_json
    )
    source_domain_control = _load_optional_json(args.source_domain_control_json)
    loader_input_audit = _load_optional_json(args.loader_input_audit_json)
    spatial_sequence_audit = _load_optional_json(args.spatial_sequence_audit_json)
    feature_semantics_audit = _load_optional_json(args.feature_semantics_audit_json)
    event_weight_audit = _load_optional_json(args.event_weight_audit_json)
    image_sequence_loader_audit = _load_optional_json(
        args.image_sequence_loader_audit_json
    )

    full_oof_run_plan_errors = _full_oof_run_plan_errors(full_oof_run_plan)
    gates = [
        _gate("S0A snapshot/data contract", snapshot.get("valid") is True, snapshot.get("errors")),
        _gate(
            "B2-B7 config matrix",
            baseline_configs.get("valid") is True,
            baseline_configs.get("errors"),
        ),
        _gate(
            "B2-B7 CUDA smoke",
            baseline_smokes.get("valid") is True,
            baseline_smokes.get("errors"),
        ),
        _gate(
            "B4 inner-validation seed variance",
            b4_seed_variance.get("valid") is True,
            b4_seed_variance.get("errors"),
        ),
        _gate(
            "S5 Q2 native-OOF metric contract",
            q2_oof_metric_contract.get("valid") is True
            and q2_oof_metric_contract.get("full_oof_execution_allowed_by_contract") is False
            and q2_oof_metric_contract.get("outer_test_used_for_threshold_tuning") is False,
            q2_oof_metric_contract.get("errors"),
        ),
        _gate(
            "S7 hard-negative review contract",
            q2_hard_negative_contract.get("valid") is True
            and q2_hard_negative_contract.get("outer_test_used_for_threshold_tuning") is False
            and q2_hard_negative_contract.get("automatic_label_change_allowed") is False,
            q2_hard_negative_contract.get("errors"),
        ),
        _gate(
            "S8 active-review loop contract",
            q2_active_review_contract.get("valid") is True
            and q2_active_review_contract.get(
                "active_review_can_apply_without_human_decision"
            )
            is False
            and q2_active_review_contract.get("pending_decisions_apply") is False
            and q2_active_review_contract.get("exclude_drops_rows") is False
            and q2_active_review_contract.get("decision_key") == "review_unit_id",
            q2_active_review_contract.get("errors"),
        ),
        _gate(
            "S9 final calibration and paper-package contract",
            q2_final_package_contract.get("valid") is True
            and q2_final_package_contract.get("outer_test_used_for_model_selection") is False
            and q2_final_package_contract.get("outer_test_used_for_threshold_tuning") is False
            and q2_final_package_contract.get("outer_test_used_for_calibration_fit") is False
            and q2_final_package_contract.get("can_claim_q2_result") is False,
            q2_final_package_contract.get("errors"),
        ),
        _gate(
            "Q2 feature whitelist leakage guard",
            q2_feature_whitelist.get("valid") is True
            and q2_feature_whitelist.get("never_use_all_numeric_columns") is True
            and q2_feature_whitelist.get("fail_closed_on_unknown_columns") is True
            and q2_feature_whitelist.get("forbidden_probe_columns_not_blocked") == [],
            q2_feature_whitelist.get("errors"),
        ),
        _gate(
            "Q2 pig_id annotation-local leakage guard",
            pig_id_locality.get("valid") is True
            and pig_id_locality.get("contracts_with_pig_id")
            == pig_id_locality.get("contracts_with_scope_hint")
            and pig_id_locality.get("forbidden_identity_allowance_count") == 0,
            pig_id_locality.get("errors"),
        ),
        _gate(
            "Q2 split group leakage artifact guard",
            split_group_leakage.get("valid") is True
            and split_group_leakage.get("group_split_leakage_count") == 0
            and split_group_leakage.get("video_split_leakage_count") == 0
            and split_group_leakage.get("split_group_key_uses_pig_id_only_count") == 0,
            split_group_leakage.get("errors"),
        ),
        _gate(
            "Q2 final package skeleton no-claim gate",
            q2_final_package_stub.get("valid") is True
            and q2_final_package_stub.get("status") == "BLOCKED_PENDING_FULL_OOF"
            and q2_final_package_stub.get("can_claim_q2_result") is False
            and q2_final_package_stub.get("paper_facing_metrics_available") is False,
            q2_final_package_stub.get("errors"),
        ),
        _gate(
            "Image cache canonical inventory",
            image_cache_inventory.get("valid") is True
            and image_cache_inventory.get("canonical_cache_dir_exists") is True
            and image_cache_inventory.get(
                "ad_hoc_active_training_reference_count",
                0,
            )
            == 0,
            image_cache_inventory.get("errors"),
        ),
        _gate(
            "Image cache letterbox aspect policy",
            image_cache_letterbox_policy.get("valid") is True
            and image_cache_letterbox_policy.get("resize_policies")
            == [image_cache_letterbox_policy.get("expected_resize_policy")]
            and image_cache_letterbox_policy.get("letterbox_geometry_invalid_rows") == 0,
            image_cache_letterbox_policy.get("errors"),
        ),
        _gate(
            "Visual interaction partner context cache",
            visual_interaction_cache.get("valid") is True
            and visual_interaction_cache.get("cvat_ready_rows", 0) > 0
            and visual_interaction_cache.get("legacy_ready_rows") == 0
            and visual_interaction_cache.get("label_gated") is False
            and visual_interaction_cache.get("rows_dropped_for_missing_context") == 0,
            visual_interaction_cache.get("errors"),
        ),
        _gate(
            "Interaction numeric context leakage guard",
            interaction_context_index.get("valid") is True
            and interaction_context_index.get("interaction_ready_rows", 0) > 0
            and interaction_context_index.get(
                "non_interaction_scene_partner_ready_rows",
                0,
            )
            > 0
            and interaction_context_index.get("duplicate_window_id") == 0
            and interaction_context_index.get(
                "scene_partner_context_not_evaluated_count"
            )
            == 0
            and interaction_context_index.get("window_uid_present") is False
            and interaction_context_index.get(
                "forbidden_model_input_columns_present"
            )
            == [],
            interaction_context_index.get("errors"),
        ),
        _gate(
            "Q2 multimodal architecture contract",
            model_architecture_contract.get("valid") is True
            and model_architecture_contract.get("paper_candidate_ready") is False
            and model_architecture_contract.get("missing_required_paper_branches")
            == [],
            model_architecture_contract.get("errors"),
        ),
        _gate(
            "Q2 ablation shortcut no-claim contract",
            ablation_shortcut_contract.get("valid") is True
            and ablation_shortcut_contract.get("paper_candidate_ready") is False
            and ablation_shortcut_contract.get(
                "planned_required_ablations_not_recorded"
            )
            == [],
            ablation_shortcut_contract.get("errors"),
        ),
        _gate(
            "Q2 ablation reporting guard",
            ablation_reporting.get("valid") is True
            and ablation_reporting.get("paper_claim_level") == "Q2_strong"
            and ablation_reporting.get("external_generalization_claim") is False
            and set(ablation_reporting.get("native_oof_comparable_ids", []))
            >= {"B0", "B1", "B2"},
            ablation_reporting.get("errors"),
        ),
        _gate(
            "Full learned OOF contract no-claim gate",
            full_learned_oof_contract.get("valid") is True
            and full_learned_oof_contract.get("paper_claim_level") == "Q2_strong"
            and full_learned_oof_contract.get("external_generalization_claim")
            is False
            and full_learned_oof_contract.get("paper_ready") is False
            and (full_learned_oof_contract.get("required_record") or {}).get(
                "exists"
            )
            is False,
            full_learned_oof_contract.get("errors"),
        ),
        _gate(
            "Strict train-ready data module boundary",
            data_module_audit.get("valid") is True
            and data_module_audit.get("duplicate_window_id") == 0
            and data_module_audit.get("auxiliary_targets_not_model_inputs") is True
            and (data_module_audit.get("actor_image_load_audit") or {}).get(
                "source_image_loads"
            )
            == 0
            and (data_module_audit.get("visual_context_load_audit") or {}).get(
                "individual_cache_loads"
            )
            == 0,
            data_module_audit.get("errors"),
        ),
        _gate(
            "Strict multimodal training config audit",
            training_config_audit.get("valid") is True
            and training_config_audit.get("snapshot_valid") is True
            and training_config_audit.get("missing_paths") == {},
            training_config_audit.get("errors"),
        ),
        _gate(
            "Q2 ablation matrix outer-test lock",
            q2_ablation_matrix.get("valid") is True
            and q2_ablation_matrix.get("outer_test_execution_allowed") is False
            and q2_ablation_matrix.get("threshold_freeze_status")
            == "frozen_from_B4_inner_validation_seed_variance"
            and q2_ablation_matrix.get("outer_fold_count", 0) >= 5
            and q2_ablation_matrix.get("confirmatory_seed_count", 0) >= 3,
            q2_ablation_matrix.get("errors"),
        ),
        _gate(
            "Visual context ablation smoke wiring",
            visual_context_ablation_smoke.get("valid") is True
            and visual_context_ablation_smoke.get("full_visual_packed_hits", 0)
            > 0
            and visual_context_ablation_smoke.get(
                "metric_interpretation"
            )
            == "wiring_smoke_only_not_statistical_ablation_evidence",
            visual_context_ablation_smoke.get("errors"),
        ),
        _gate(
            "Source-domain matched control contract",
            source_domain_control.get("valid") is True
            and source_domain_control.get("duplicate_window_id") == 0
            and source_domain_control.get("kept_rows", 0) > 0
            and source_domain_control.get("imbalanced_strata_after_count") == 0
            and source_domain_control.get("forbidden_x_columns") == []
            and len(source_domain_control.get("source_labels", [])) >= 2,
            source_domain_control.get("errors"),
        ),
        _gate(
            "Loader input source-domain leakage guard",
            loader_input_audit.get("valid") is True
            and loader_input_audit.get("forbidden_x_columns") == []
            and loader_input_audit.get("whitelist_missing_in_tabular_x") == []
            and loader_input_audit.get("tabular_x_columns_not_in_whitelist")
            == []
            and loader_input_audit.get("source_domain_kept_rows", 0) > 0
            and loader_input_audit.get(
                "source_domain_imbalanced_strata_after_count"
            )
            == 0,
            loader_input_audit.get("errors"),
        ),
        _gate(
            "Spatial sequence train-ready tensor contract",
            spatial_sequence_audit.get("errors") == []
            and spatial_sequence_audit.get("rows") == 160740
            and spatial_sequence_audit.get("truncated_windows") == 0
            and spatial_sequence_audit.get("forbidden_selected") == []
            and spatial_sequence_audit.get("observed_within_length_ratio", 0.0)
            > 0.99,
            spatial_sequence_audit.get("errors"),
        ),
        _gate(
            "Feature semantics leakage annotation contract",
            feature_semantics_audit.get("valid") is True
            and feature_semantics_audit.get("errors") == []
            and feature_semantics_audit.get("forbidden_tabular_features") == []
            and feature_semantics_audit.get("tabular_feature_count") == 39,
            feature_semantics_audit.get("errors"),
        ),
        _gate(
            "Event-balanced sample weight contract",
            event_weight_audit.get("errors") == []
            and event_weight_audit.get("rows") == 160740
            and event_weight_audit.get("duplicate_window_id") == 0
            and event_weight_audit.get("missing_event_key_rows") == 0
            and event_weight_audit.get("invalid_weight_zero_count", 0) > 0,
            event_weight_audit.get("errors"),
        ),
        _gate(
            "Image sequence loader source coverage smoke",
            image_sequence_loader_audit.get("errors") == []
            and image_sequence_loader_audit.get("checked_windows", 0) >= 24
            and image_sequence_loader_audit.get("error_windows") == 0
            and image_sequence_loader_audit.get("missing_frames") == 0
            and {"cvat_tracking_xml", "legacy_recovered"}.issubset(
                set((image_sequence_loader_audit.get("checked_by_source") or {}).keys())
            ),
            image_sequence_loader_audit.get("errors"),
        ),
        _gate(
            "Full OOF preflight canonical path policy",
            full_oof_preflight_policy.get("valid") is True
            and full_oof_preflight_policy.get("canonical_config_errors") == []
            and full_oof_preflight_policy.get("missing_bad_tokens") == [],
            full_oof_preflight_policy.get("errors"),
        ),
        _gate(
            "Full OOF authorization artifact policy",
            full_oof_authorization_policy.get("valid") is True
            and full_oof_authorization_policy.get("requires_authorization_json") is True
            and full_oof_authorization_policy.get("missing_invalid_tokens") == [],
            full_oof_authorization_policy.get("errors"),
        ),
        _gate(
            "Full OOF authorization template fail-closed policy",
            full_oof_authorization_template.get("valid") is True
            and full_oof_authorization_template.get("template_authorized_default") is False
            and full_oof_authorization_template.get(
                "template_acknowledges_long_run_default"
            )
            is False
            and full_oof_authorization_template.get("template_acknowledges_no_claim_default")
            is False,
            full_oof_authorization_template.get("errors"),
        ),
        _gate(
            "Full OOF fresh preflight authorization-ready guard",
            full_oof_preflight_freshness.get("valid") is True
            and full_oof_preflight_freshness.get("preflight_valid") is True
            and full_oof_preflight_freshness.get("preflight_fresh") is True
            and full_oof_preflight_freshness.get("full_oof_execution_allowed")
            is False
            and full_oof_preflight_freshness.get(
                "preflight_authorization_ready"
            )
            is True
            and full_oof_preflight_freshness.get(
                "authorization_must_refresh_preflight"
            )
            is False,
            full_oof_preflight_freshness.get("errors"),
        ),
        _gate(
            "Full OOF workload plan output isolation",
            not full_oof_run_plan_errors,
            full_oof_run_plan_errors,
        ),
        _gate(
            "Full runner default config matches workload plan",
            full_runner_default_config.get("valid") is True
            and full_runner_default_config.get("errors") == []
            and full_runner_default_config.get("runner_config_sha256")
            == full_oof_run_plan.get("config_sha256"),
            full_runner_default_config.get("errors"),
        ),
        _gate(
            "Full OOF execution gate rejects unauthorized runs",
            full_oof_execution_gate.get("valid") is True
            and full_oof_execution_gate.get("full_training_invoked") is False
            and full_oof_execution_gate.get("case_count") == 3,
            full_oof_execution_gate.get("errors"),
        ),
        _gate(
            "Full OOF launch packet ready for human authorization",
            full_oof_launch_packet.get("valid") is True
            and full_oof_launch_packet.get("packet_status")
            == "READY_FOR_HUMAN_AUTHORIZATION"
            and full_oof_launch_packet.get("packet_matches_current_inputs")
            is True
            and full_oof_launch_packet.get("full_oof_execution_allowed")
            is False
            and full_oof_launch_packet.get("authorization_required") is True
            and full_oof_launch_packet.get("authorization_template_authorized")
            is False
            and _launch_packet_ready(full_oof_launch_packet),
            full_oof_launch_packet.get("errors"),
        ),
        _gate(
            "Full OOF completion gate is fail-closed",
            full_oof_completion_gate.get("valid") is True
            and full_oof_completion_gate.get("fail_closed") is True
            and (
                full_oof_completion_gate.get("completion_ready")
                is full_oof_completion_gate.get("q2_claim_allowed")
            ),
            full_oof_completion_gate.get("errors"),
        ),
        _gate(
            "Full OOF post-run registration packet is reviewable",
            full_oof_postrun_registration_packet.get("valid") is True
            and full_oof_postrun_registration_packet.get("packet_status")
            == "READY_FOR_POST_FULL_OOF_REGISTRATION"
            and full_oof_postrun_registration_packet.get(
                "packet_matches_current_inputs"
            )
            is True
            and full_oof_postrun_registration_packet.get("runs_training")
            is False
            and full_oof_postrun_registration_packet.get("runs_registration")
            is False
            and full_oof_postrun_registration_packet.get(
                "q2_claim_allowed_by_packet"
            )
            is False
            and _optional_true(
                full_oof_postrun_registration_packet,
                "register_cmd_command_ready",
            )
            and _optional_true(
                full_oof_postrun_registration_packet,
                "completion_gate_cmd_command_ready",
            ),
            full_oof_postrun_registration_packet.get("errors"),
        ),
        _gate(
            "Strict trainer reproducibility",
            reproducibility.get("errors") == []
            and reproducibility.get("forbidden_model_input_rejected") is True,
            reproducibility.get("errors"),
        ),
    ]
    remaining = [
        "Full OOF remains blocked until explicit authorization and matching clean preflight.",
        (
            "S5 metric contract is defined; real per-source/matched metrics "
            "still need explicitly authorized full OOF predictions."
        ),
        (
            "B4 seed variance is estimated from bounded validation-only smoke; "
            "full inner-validation variance still needs non-smoke OOF authorization."
        ),
        (
            "S7 hard-negative contract is defined; actual shortlist generation "
            "needs explicitly authorized native OOF predictions."
        ),
        (
            "S8 active-review loop contract is defined; actual decisions "
            "require human GUI review before apply."
        ),
        (
            "S9 final package contract is defined; final paper metrics still "
            "need explicitly authorized full OOF/final-test execution."
        ),
        (
            "Q2 feature whitelist is defined; every new trainer must consume "
            "it or an equivalent checked contract."
        ),
        (
            "Q2 final package skeleton exists only as a blocked/no-claim "
            "artifact until full OOF is authorized and complete."
        ),
        (
            "Full OOF preflight rejects ad hoc smoke/resume cache roots and "
            "requires canonical packed cache paths."
        ),
        (
            "Full OOF runner requires authorization JSON bound to preflight "
            "config hash and Git commit."
        ),
        (
            "Full OOF authorization template is intentionally non-authorized "
            "until a reviewer edits the approval fields."
        ),
        (
            "Image cache inventory records non-canonical smoke/resume cache "
            "roots without deleting derived outputs automatically."
        ),
        (
            "Actor image cache policy is audited as letterbox aspect-preserving "
            "padding, not square stretch resizing."
        ),
        (
            "Visual interaction cache is audited as CVAT actor-nearest-partner "
            "union context, with legacy crop-only rows masked rather than dropped."
        ),
        (
            "pig_id remains annotation-local in Q2 contracts and must not be "
            "used as a biological identity or cross-video split key."
        ),
        (
            "Current split artifacts are audited so split_group_key and "
            "video_key do not cross train/val/test boundaries."
        ),
        (
            "Full OOF launch packet is review-ready but still requires an "
            "explicit authorization JSON before training can start."
        ),
        (
            "Full OOF completion gate is fail-closed: missing or invalid full "
            "prediction artifacts keep q2_claim_allowed=false."
        ),
        (
            "Full OOF post-run registration packet defines the registry and "
            "completion-check commands, but it does not execute registration."
        ),
    ]
    result = {
        "schema_version": "classification_v2_q2_progress_report_v1",
        "overall_status": (
            "PASS_PARTIAL_ROADMAP" if all(gate["passed"] for gate in gates) else "FAIL"
        ),
        "claim_boundary": (
            "Q2 internal recording-date/video-safe improvement only; "
            "no external farm/camera/cohort claim."
        ),
        "gates": gates,
        "remaining_work": remaining,
        "evidence": {
            "snapshot": _evidence_snapshot(snapshot),
            "baseline_configs": _evidence_baseline_configs(baseline_configs),
            "baseline_smokes": _evidence_baseline_smokes(baseline_smokes),
            "b4_seed_variance": _evidence_b4_seed_variance(b4_seed_variance),
            "q2_oof_metric_contract": _evidence_q2_oof_metric_contract(q2_oof_metric_contract),
            "q2_hard_negative_contract": _evidence_q2_hard_negative_contract(
                q2_hard_negative_contract
            ),
            "q2_active_review_contract": _evidence_q2_active_review_contract(
                q2_active_review_contract
            ),
            "q2_final_package_contract": _evidence_q2_final_package_contract(
                q2_final_package_contract
            ),
            "q2_feature_whitelist": _evidence_q2_feature_whitelist(q2_feature_whitelist),
            "pig_id_locality": _evidence_pig_id_locality(pig_id_locality),
            "split_group_leakage": _evidence_split_group_leakage(split_group_leakage),
            "q2_final_package_stub": _evidence_q2_final_package_stub(q2_final_package_stub),
            "image_cache_inventory": _evidence_image_cache_inventory(image_cache_inventory),
            "image_cache_letterbox_policy": _evidence_image_cache_letterbox_policy(
                image_cache_letterbox_policy
            ),
            "visual_interaction_cache": _evidence_visual_interaction_cache(
                visual_interaction_cache
            ),
            "interaction_context_index": _evidence_interaction_context_index(
                interaction_context_index
            ),
            "model_architecture_contract": _evidence_model_architecture_contract(
                model_architecture_contract
            ),
            "ablation_shortcut_contract": _evidence_ablation_shortcut_contract(
                ablation_shortcut_contract
            ),
            "ablation_reporting": _evidence_ablation_reporting(ablation_reporting),
            "full_learned_oof_contract": _evidence_full_learned_oof_contract(
                full_learned_oof_contract
            ),
            "data_module": _evidence_data_module(data_module_audit),
            "training_config": _evidence_training_config(training_config_audit),
            "q2_ablation_matrix": _evidence_q2_ablation_matrix(
                q2_ablation_matrix
            ),
            "visual_context_ablation_smoke": _evidence_visual_context_ablation_smoke(
                visual_context_ablation_smoke
            ),
            "source_domain_control": _evidence_source_domain_control(
                source_domain_control
            ),
            "loader_input_audit": _evidence_loader_input_audit(
                loader_input_audit
            ),
            "spatial_sequence": _evidence_spatial_sequence(spatial_sequence_audit),
            "feature_semantics": _evidence_feature_semantics(
                feature_semantics_audit
            ),
            "event_weight": _evidence_event_weight(event_weight_audit),
            "image_sequence_loader": _evidence_image_sequence_loader(
                image_sequence_loader_audit
            ),
            "full_oof_preflight_policy": _evidence_full_oof_preflight_policy(
                full_oof_preflight_policy
            ),
            "full_oof_authorization_policy": _evidence_full_oof_authorization_policy(
                full_oof_authorization_policy
            ),
            "full_oof_authorization_template": _evidence_full_oof_authorization_template(
                full_oof_authorization_template
            ),
            "full_oof_preflight_freshness": _evidence_full_oof_preflight_freshness(
                full_oof_preflight_freshness
            ),
            "full_oof_run_plan": _evidence_full_oof_run_plan(full_oof_run_plan),
            "full_runner_default_config": _evidence_full_runner_default_config(
                full_runner_default_config
            ),
            "full_oof_execution_gate": _evidence_full_oof_execution_gate(
                full_oof_execution_gate
            ),
            "full_oof_launch_packet": _evidence_full_oof_launch_packet(
                full_oof_launch_packet
            ),
            "full_oof_completion_gate": _evidence_full_oof_completion_gate(
                full_oof_completion_gate
            ),
            "full_oof_postrun_registration_packet": (
                _evidence_full_oof_postrun_registration_packet(
                    full_oof_postrun_registration_packet
                )
            ),
            "reproducibility": _evidence_reproducibility(reproducibility),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["overall_status"], "gates": gates}, indent=2))
    if result["overall_status"] == "FAIL":
        raise SystemExit(1)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path), "errors": [f"missing:{path}"], "valid": False}
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(name: str, passed: bool, errors: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "errors": errors or []}


def _optional_true(audit: dict[str, Any], key: str) -> bool:
    """Accept old audit files without a key, but require True when present."""

    return key not in audit or audit.get(key) is True


def _evidence_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.get("expected_snapshot_id"),
        "current_snapshot_id": snapshot.get("current_snapshot_id"),
        "valid": snapshot.get("valid"),
    }


def _evidence_baseline_configs(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_count": audit.get("baseline_count"),
        "valid": audit.get("valid"),
        "snapshot_ids": sorted({row.get("snapshot_id") for row in audit.get("baselines", [])}),
    }


def _evidence_baseline_smokes(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_count": audit.get("baseline_count"),
        "require_device": audit.get("require_device"),
        "runtime_python_executable": audit.get("runtime_python_executable"),
        "devices": sorted({row.get("device") for row in audit.get("baselines", [])}),
        "git_dirty_values": sorted(
            {str(row.get("git_dirty")) for row in audit.get("baselines", [])}
        ),
    }


def _evidence_reproducibility(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "errors": audit.get("errors"),
        "forbidden_model_input_rejected": audit.get("forbidden_model_input_rejected"),
        "prediction_sha256": audit.get("prediction_sha256"),
        "test_prediction_sha256": audit.get("test_prediction_sha256"),
    }


def _evidence_b4_seed_variance(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "seed_count": audit.get("seed_count"),
        "summary": audit.get("summary"),
    }


def _evidence_q2_oof_metric_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "contract_version": audit.get("contract_version"),
        "statistical_unit": audit.get("statistical_unit"),
        "primary_split_policy": audit.get("primary_split_policy"),
        "label_count": audit.get("label_count"),
        "required_metric_group_count": audit.get("required_metric_group_count"),
        "confusion_pair_count": audit.get("confusion_pair_count"),
        "full_oof_execution_allowed_by_contract": audit.get(
            "full_oof_execution_allowed_by_contract"
        ),
        "outer_test_used_for_threshold_tuning": audit.get("outer_test_used_for_threshold_tuning"),
    }


def _evidence_q2_hard_negative_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "contract_version": audit.get("contract_version"),
        "requires_oof_native_predictions": audit.get("requires_oof_native_predictions"),
        "outer_test_used_for_threshold_tuning": audit.get("outer_test_used_for_threshold_tuning"),
        "automatic_label_change_allowed": audit.get("automatic_label_change_allowed"),
        "predeclared_confusion_pair_count": audit.get("predeclared_confusion_pair_count"),
        "required_shortlist_column_count": audit.get("required_shortlist_column_count"),
        "leakage_guard_count": audit.get("leakage_guard_count"),
    }


def _evidence_q2_active_review_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "contract_version": audit.get("contract_version"),
        "active_review_can_apply_without_human_decision": audit.get(
            "active_review_can_apply_without_human_decision"
        ),
        "pending_decisions_apply": audit.get("pending_decisions_apply"),
        "exclude_drops_rows": audit.get("exclude_drops_rows"),
        "decision_key": audit.get("decision_key"),
        "decision_column_count": audit.get("decision_column_count"),
        "gui_context_count": audit.get("gui_context_count"),
        "apply_safety_rule_count": audit.get("apply_safety_rule_count"),
    }


def _evidence_q2_final_package_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "contract_version": audit.get("contract_version"),
        "full_oof_predictions_required_before_final_claim": audit.get(
            "full_oof_predictions_required_before_final_claim"
        ),
        "outer_test_used_for_model_selection": audit.get("outer_test_used_for_model_selection"),
        "outer_test_used_for_threshold_tuning": audit.get("outer_test_used_for_threshold_tuning"),
        "outer_test_used_for_calibration_fit": audit.get("outer_test_used_for_calibration_fit"),
        "calibration_fit_scope": audit.get("calibration_fit_scope"),
        "final_test_is_single_touch": audit.get("final_test_is_single_touch"),
        "primary_metric": audit.get("primary_metric"),
        "model_family_count": audit.get("model_family_count"),
        "metric_table_count": audit.get("metric_table_count"),
        "figure_count": audit.get("figure_count"),
        "package_artifact_count": audit.get("package_artifact_count"),
        "can_claim_q2_result": audit.get("can_claim_q2_result"),
    }


def _evidence_q2_feature_whitelist(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "contract_version": audit.get("contract_version"),
        "never_use_all_numeric_columns": audit.get("never_use_all_numeric_columns"),
        "fail_closed_on_unknown_columns": audit.get("fail_closed_on_unknown_columns"),
        "input_branch_count": audit.get("input_branch_count"),
        "forbidden_pattern_count": audit.get("forbidden_pattern_count"),
        "forbidden_probe_columns_not_blocked": audit.get("forbidden_probe_columns_not_blocked"),
        "tabular_trainer_whitelist_count": audit.get("tabular_trainer_whitelist_count"),
        "spatial_trainer_whitelist_count": audit.get("spatial_trainer_whitelist_count"),
    }


def _evidence_pig_id_locality(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "contract_count": audit.get("contract_count"),
        "contracts_with_pig_id": audit.get("contracts_with_pig_id"),
        "contracts_with_scope_hint": audit.get("contracts_with_scope_hint"),
        "forbidden_identity_allowance_count": audit.get(
            "forbidden_identity_allowance_count"
        ),
    }


def _evidence_split_group_leakage(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "rows": audit.get("rows"),
        "group_count": audit.get("group_count"),
        "video_count": audit.get("video_count"),
        "split_counts": audit.get("split_counts"),
        "group_split_leakage_count": audit.get("group_split_leakage_count"),
        "video_split_leakage_count": audit.get("video_split_leakage_count"),
        "split_group_key_uses_pig_id_only_count": audit.get(
            "split_group_key_uses_pig_id_only_count"
        ),
        "builder_leakage_group_count": audit.get("builder_leakage_group_count"),
    }


def _evidence_q2_final_package_stub(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "status": audit.get("status"),
        "can_claim_q2_result": audit.get("can_claim_q2_result"),
        "paper_facing_metrics_available": audit.get("paper_facing_metrics_available"),
        "missing_required_package_artifact_count": audit.get(
            "missing_required_package_artifact_count"
        ),
        "feature_whitelist_valid": audit.get("feature_whitelist_valid"),
    }


def _evidence_image_cache_inventory(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "schema_version": audit.get("schema_version"),
        "canonical_cache_dir": audit.get("canonical_cache_dir"),
        "canonical_cache_dir_exists": audit.get("canonical_cache_dir_exists"),
        "cache_dir_count": audit.get("cache_dir_count"),
        "ad_hoc_cache_dir_count": audit.get("ad_hoc_cache_dir_count"),
        "ad_hoc_cache_dirs": audit.get("ad_hoc_cache_dirs"),
        "ad_hoc_active_training_reference_count": audit.get(
            "ad_hoc_active_training_reference_count"
        ),
        "ad_hoc_cache_policy": audit.get("ad_hoc_cache_policy"),
        "warnings": audit.get("warnings"),
    }


def _evidence_image_cache_letterbox_policy(audit: dict[str, Any]) -> dict[str, Any]:
    summary = audit.get("letterbox_geometry_summary") or {}
    return {
        "valid": audit.get("valid"),
        "cache_manifest": audit.get("cache_manifest"),
        "manifest_rows": audit.get("manifest_rows"),
        "resize_policies": audit.get("resize_policies"),
        "expected_resize_policy": audit.get("expected_resize_policy"),
        "non_square_source_crop_rows": summary.get("non_square_source_crop_rows"),
        "padded_canvas_rows": summary.get("padded_canvas_rows"),
        "letterbox_geometry_invalid_rows": audit.get("letterbox_geometry_invalid_rows"),
        "source_equivalence_checked": audit.get("source_equivalence_checked"),
        "source_equivalence_mismatches": audit.get("source_equivalence_mismatches"),
    }


def _evidence_visual_interaction_cache(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "rows": audit.get("rows"),
        "cvat_ready_rows": audit.get("cvat_ready_rows"),
        "legacy_ready_rows": audit.get("legacy_ready_rows"),
        "available_rows": audit.get("available_rows"),
        "unavailable_rows": audit.get("unavailable_rows"),
        "context_kinds": audit.get("context_kinds"),
        "resize_policies": audit.get("resize_policies"),
        "packed_tensor_shape": audit.get("packed_tensor_shape"),
        "packed_index_rows": audit.get("packed_index_rows"),
        "label_gated": audit.get("label_gated"),
        "rows_dropped_for_missing_context": audit.get(
            "rows_dropped_for_missing_context"
        ),
    }


def _evidence_interaction_context_index(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "rows": audit.get("rows"),
        "duplicate_window_id": audit.get("duplicate_window_id"),
        "window_uid_present": audit.get("window_uid_present"),
        "interaction_ready_rows": audit.get("interaction_ready_rows"),
        "non_interaction_scene_partner_ready_rows": audit.get(
            "non_interaction_scene_partner_ready_rows"
        ),
        "scene_partner_context_not_evaluated_count": audit.get(
            "scene_partner_context_not_evaluated_count"
        ),
        "model_input_feature_columns": audit.get("model_input_feature_columns"),
        "forbidden_model_input_columns_present": audit.get(
            "forbidden_model_input_columns_present"
        ),
    }


def _evidence_model_architecture_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "contract_version": audit.get("contract_version"),
        "paper_candidate_ready": audit.get("paper_candidate_ready"),
        "missing_required_paper_branches": audit.get(
            "missing_required_paper_branches"
        ),
        "paper_candidate_blockers": audit.get("paper_candidate_blockers"),
    }


def _evidence_ablation_shortcut_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "contract_version": audit.get("contract_version"),
        "paper_candidate_ready": audit.get("paper_candidate_ready"),
        "planned_required_ablations_not_recorded": audit.get(
            "planned_required_ablations_not_recorded"
        ),
        "paper_candidate_blockers": audit.get("paper_candidate_blockers"),
    }


def _evidence_ablation_reporting(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "paper_claim_level": audit.get("paper_claim_level"),
        "external_generalization_claim": audit.get(
            "external_generalization_claim"
        ),
        "native_oof_comparable_ids": audit.get("native_oof_comparable_ids"),
        "smoke_only_ids": audit.get("smoke_only_ids"),
        "primary_metric": audit.get("primary_metric"),
        "sesoi": audit.get("sesoi"),
    }


def _evidence_full_learned_oof_contract(audit: dict[str, Any]) -> dict[str, Any]:
    required_record = audit.get("required_record") or {}
    alignment = audit.get("alignment_report") or {}
    return {
        "valid": audit.get("valid"),
        "paper_claim_level": audit.get("paper_claim_level"),
        "external_generalization_claim": audit.get(
            "external_generalization_claim"
        ),
        "paper_ready": audit.get("paper_ready"),
        "required_record_exists": required_record.get("exists"),
        "train_ready_rows": alignment.get("train_ready_rows"),
        "native_oof_fold_count": alignment.get("native_oof_fold_count"),
        "row_count_mismatches": alignment.get("row_count_mismatches"),
    }


def _evidence_data_module(audit: dict[str, Any]) -> dict[str, Any]:
    actor_load = audit.get("actor_image_load_audit") or {}
    visual_load = audit.get("visual_context_load_audit") or {}
    return {
        "valid": audit.get("valid"),
        "rows": audit.get("rows"),
        "eligible_rows": audit.get("eligible_rows"),
        "duplicate_window_id": audit.get("duplicate_window_id"),
        "auxiliary_targets_not_model_inputs": audit.get(
            "auxiliary_targets_not_model_inputs"
        ),
        "actor_source_image_loads": actor_load.get("source_image_loads"),
        "actor_packed_cache_hits": actor_load.get("packed_image_cache_hits"),
        "visual_individual_cache_loads": visual_load.get(
            "individual_cache_loads"
        ),
        "visual_packed_cache_hits": visual_load.get("packed_cache_hits"),
    }


def _evidence_training_config(audit: dict[str, Any]) -> dict[str, Any]:
    config = audit.get("config") or {}
    model = config.get("model") or {}
    execution = config.get("execution") or {}
    return {
        "valid": audit.get("valid"),
        "snapshot_id": audit.get("snapshot_id"),
        "snapshot_valid": audit.get("snapshot_valid"),
        "missing_paths": audit.get("missing_paths"),
        "architecture_version": model.get("architecture_version"),
        "execution_mode": execution.get("mode"),
        "fold_id": execution.get("fold_id"),
    }


def _evidence_q2_ablation_matrix(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "baseline_count": audit.get("baseline_count"),
        "ablation_count": audit.get("ablation_count"),
        "outer_fold_count": audit.get("outer_fold_count"),
        "confirmatory_seed_count": audit.get("confirmatory_seed_count"),
        "outer_test_execution_allowed": audit.get(
            "outer_test_execution_allowed"
        ),
        "threshold_freeze_status": audit.get("threshold_freeze_status"),
    }


def _evidence_visual_context_ablation_smoke(
    audit: dict[str, Any]
) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "full_visual_packed_hits": audit.get("full_visual_packed_hits"),
        "full_trainable_parameter_count": audit.get(
            "full_trainable_parameter_count"
        ),
        "no_visual_trainable_parameter_count": audit.get(
            "no_visual_trainable_parameter_count"
        ),
        "metric_interpretation": audit.get("metric_interpretation"),
    }


def _evidence_source_domain_control(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "rows": audit.get("rows"),
        "eligible_rows": audit.get("eligible_rows"),
        "kept_rows": audit.get("kept_rows"),
        "excluded_rows": audit.get("excluded_rows"),
        "source_labels": audit.get("source_labels"),
        "duplicate_window_id": audit.get("duplicate_window_id"),
        "forbidden_x_columns": audit.get("forbidden_x_columns"),
        "balanced_strata_after_count": audit.get(
            "balanced_strata_after_count"
        ),
        "imbalanced_strata_after_count": audit.get(
            "imbalanced_strata_after_count"
        ),
    }


def _evidence_loader_input_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "trainer_contract_json": audit.get("trainer_contract_json"),
        "tabular_x_column_count": audit.get("tabular_x_column_count"),
        "tabular_feature_whitelist_count": audit.get(
            "tabular_feature_whitelist_count"
        ),
        "forbidden_x_columns": audit.get("forbidden_x_columns"),
        "tabular_x_columns_not_in_whitelist": audit.get(
            "tabular_x_columns_not_in_whitelist"
        ),
        "source_domain_kept_rows": audit.get("source_domain_kept_rows"),
        "source_domain_imbalanced_strata_after_count": audit.get(
            "source_domain_imbalanced_strata_after_count"
        ),
    }


def _evidence_spatial_sequence(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": audit.get("rows"),
        "max_window_length": audit.get("max_window_length"),
        "truncated_windows": audit.get("truncated_windows"),
        "observed_within_length_ratio": audit.get("observed_within_length_ratio"),
        "forbidden_selected": audit.get("forbidden_selected"),
        "array_shapes": audit.get("array_shapes"),
        "warnings": audit.get("warnings"),
    }


def _evidence_feature_semantics(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "tabular_feature_count": audit.get("tabular_feature_count"),
        "tabular_family_counts": audit.get("tabular_family_counts"),
        "forbidden_tabular_features": audit.get("forbidden_tabular_features"),
        "spatial_assignment_count": len(audit.get("spatial_assignments", {}) or {}),
        "warnings": audit.get("warnings"),
    }


def _evidence_event_weight(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": audit.get("rows"),
        "unique_window_ids": audit.get("unique_window_ids"),
        "duplicate_window_id": audit.get("duplicate_window_id"),
        "invalid_weight_zero_count": audit.get("invalid_weight_zero_count"),
        "event_overlap_cluster_count": audit.get("event_overlap_cluster_count"),
        "event_balanced_weight_sum": audit.get("event_balanced_weight_sum"),
        "warnings": audit.get("warnings"),
    }


def _evidence_image_sequence_loader(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "checked_windows": audit.get("checked_windows"),
        "ok_windows": audit.get("ok_windows"),
        "error_windows": audit.get("error_windows"),
        "loaded_frames": audit.get("loaded_frames"),
        "missing_frames": audit.get("missing_frames"),
        "checked_by_source": audit.get("checked_by_source"),
        "image_size": audit.get("image_size"),
    }


def _full_oof_run_plan_errors(plan: dict[str, Any]) -> list[str]:
    """Require full workload plans to use full-output roots, not smoke roots."""

    errors: list[str] = []
    config = plan.get("config") or {}
    load_audit = plan.get("load_audit") or {}
    output_dir = str(config.get("output_dir") or "")
    output_parts = _normalized_path_parts(output_dir)
    forbidden_parts = {"model_smoke", "smoke", "pilot", "resume_smoke"}
    if plan.get("valid") is not True or plan.get("errors"):
        errors.append(f"invalid_full_oof_run_plan={plan.get('errors')}")
    if plan.get("run_mode") != "full":
        errors.append(f"full_oof_plan_run_mode_not_full={plan.get('run_mode')}")
    if plan.get("paper_facing_candidate_plan") is not True:
        errors.append("full_oof_plan_not_paper_facing_candidate")
    if int(plan.get("selected_fold_count") or 0) != int(
        plan.get("available_fold_count") or -1
    ):
        errors.append(
            "full_oof_plan_selected_fold_count_mismatch="
            f"{plan.get('selected_fold_count')}!={plan.get('available_fold_count')}"
        )
    if int(plan.get("total_eval_rows") or 0) != int(
        load_audit.get("eligible_rows") or -1
    ):
        errors.append(
            "full_oof_plan_total_eval_rows_mismatch="
            f"{plan.get('total_eval_rows')}!={load_audit.get('eligible_rows')}"
        )
    if output_parts.intersection(forbidden_parts):
        errors.append(f"full_oof_plan_output_dir_forbidden={output_dir}")
    if "model_full" not in output_parts:
        errors.append(f"full_oof_plan_output_dir_not_model_full={output_dir}")
    if config.get("require_cached_images") is not True:
        errors.append("full_oof_plan_must_require_actor_cached_images")
    if config.get("packed_image_cache_npy") in (None, ""):
        errors.append("full_oof_plan_missing_packed_actor_cache")
    if config.get("packed_image_cache_index_csv") in (None, ""):
        errors.append("full_oof_plan_missing_packed_actor_cache_index")
    if config.get("require_packed_visual_context") is not True:
        errors.append("full_oof_plan_must_require_packed_visual_context")
    if config.get("visual_context_cache_manifest_csv") in (None, ""):
        errors.append("full_oof_plan_missing_visual_context_manifest")
    if config.get("visual_context_packed_cache_npy") in (None, ""):
        errors.append("full_oof_plan_missing_packed_visual_context")
    if config.get("visual_context_packed_cache_index_csv") in (None, ""):
        errors.append("full_oof_plan_missing_packed_visual_context_index")
    for key in (
        "packed_image_cache_npy",
        "packed_image_cache_index_csv",
        "visual_context_cache_manifest_csv",
        "visual_context_packed_cache_npy",
        "visual_context_packed_cache_index_csv",
    ):
        path = config.get(key)
        if path and not Path(str(path)).exists():
            errors.append(f"full_oof_plan_missing_file={key}:{path}")
    return errors


def _normalized_path_parts(path: str) -> set[str]:
    return {part for part in path.replace("\\", "/").lower().split("/") if part}


def _evidence_full_oof_run_plan(plan: dict[str, Any]) -> dict[str, Any]:
    config = plan.get("config") or {}
    load_audit = plan.get("load_audit") or {}
    return {
        "valid": plan.get("valid"),
        "run_mode": plan.get("run_mode"),
        "paper_facing_candidate_plan": plan.get("paper_facing_candidate_plan"),
        "output_dir": config.get("output_dir"),
        "require_cached_images": config.get("require_cached_images"),
        "packed_image_cache_npy": config.get("packed_image_cache_npy"),
        "packed_image_cache_index_csv": config.get("packed_image_cache_index_csv"),
        "require_packed_visual_context": config.get("require_packed_visual_context"),
        "visual_context_packed_cache_npy": config.get(
            "visual_context_packed_cache_npy"
        ),
        "visual_context_packed_cache_index_csv": config.get(
            "visual_context_packed_cache_index_csv"
        ),
        "available_fold_count": plan.get("available_fold_count"),
        "selected_fold_count": plan.get("selected_fold_count"),
        "total_eval_rows": plan.get("total_eval_rows"),
        "eligible_rows": load_audit.get("eligible_rows"),
        "total_train_steps": plan.get("total_train_steps"),
        "config_sha256": plan.get("config_sha256"),
    }


def _evidence_full_runner_default_config(audit: dict[str, Any]) -> dict[str, Any]:
    runner_config = audit.get("runner_config") or {}
    return {
        "valid": audit.get("valid"),
        "runner_config_sha256": audit.get("runner_config_sha256"),
        "plan_config_sha256": audit.get("plan_config_sha256"),
        "output_dir": runner_config.get("output_dir"),
        "require_cached_images": runner_config.get("require_cached_images"),
        "require_packed_visual_context": runner_config.get(
            "require_packed_visual_context"
        ),
        "image_size": runner_config.get("image_size"),
        "hidden_dim": runner_config.get("hidden_dim"),
        "precision": runner_config.get("precision"),
        "checked_existing_path_keys": audit.get("checked_existing_path_keys"),
    }


def _evidence_full_oof_execution_gate(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "case_count": audit.get("case_count"),
        "full_training_invoked": audit.get("full_training_invoked"),
        "case_names": [
            case.get("name")
            for case in audit.get("cases", []) or []
        ],
        "errors": audit.get("errors"),
    }


def _launch_packet_cmd_ready(audit: dict[str, Any]) -> bool:
    """Require full-run launch packets to bind CMD setup and Python executable."""

    command = audit.get("launch_command") or []
    cmd_command = audit.get("cmd_launch_command") or []
    python_executable = str(audit.get("python_executable") or "")
    return bool(
        python_executable
        and command
        and command[0] == python_executable
        and cmd_command
        and "PYTHONPATH=%CD%\\src" in cmd_command
        and "&&" in cmd_command
    )


def _launch_packet_ready(audit: dict[str, Any]) -> bool:
    """Use checker readiness booleans when present, with old-audit fallback."""

    if "launch_command_python_ready" in audit or "cmd_launch_command_ready" in audit:
        return bool(
            audit.get("launch_command_python_ready") is True
            and audit.get("cmd_launch_command_ready") is True
        )
    return _launch_packet_cmd_ready(audit)


def _evidence_full_oof_launch_packet(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "packet_status": audit.get("packet_status"),
        "packet_matches_current_inputs": audit.get("packet_matches_current_inputs"),
        "full_oof_execution_allowed": audit.get("full_oof_execution_allowed"),
        "authorization_required": audit.get("authorization_required"),
        "authorization_template_authorized": audit.get(
            "authorization_template_authorized"
        ),
        "python_executable": audit.get("python_executable"),
        "launch_command_python_ready": (
            audit.get("launch_command_python_ready")
            if "launch_command_python_ready" in audit
            else _launch_packet_cmd_ready(audit)
        ),
        "cmd_launch_command_ready": (
            audit.get("cmd_launch_command_ready")
            if "cmd_launch_command_ready" in audit
            else _launch_packet_cmd_ready(audit)
        ),
        "estimated_training_seconds_excluding_eval": audit.get(
            "estimated_training_seconds_excluding_eval"
        ),
        "estimated_training_minutes_excluding_eval": audit.get(
            "estimated_training_minutes_excluding_eval"
        ),
        "review_checklist_count": audit.get("review_checklist_count"),
        "preflight_config_sha256": audit.get("preflight_config_sha256"),
        "git_commit": audit.get("git_commit"),
    }


def _evidence_full_oof_completion_gate(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "completion_ready": audit.get("completion_ready"),
        "q2_claim_allowed": audit.get("q2_claim_allowed"),
        "fail_closed": audit.get("fail_closed"),
        "missing_artifact_count": audit.get("missing_artifact_count"),
        "blocking_reasons": audit.get("blocking_reasons"),
        "output_dir": audit.get("output_dir"),
        "registry_record_json": audit.get("registry_record_json"),
        "preflight_config_sha256": audit.get("preflight_config_sha256"),
        "preflight_git_commit": audit.get("preflight_git_commit"),
    }


def _evidence_full_oof_postrun_registration_packet(
    audit: dict[str, Any]
) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "packet_status": audit.get("packet_status"),
        "packet_matches_current_inputs": audit.get("packet_matches_current_inputs"),
        "runs_training": audit.get("runs_training"),
        "runs_registration": audit.get("runs_registration"),
        "q2_claim_allowed_by_packet": audit.get("q2_claim_allowed_by_packet"),
        "python_executable": audit.get("python_executable"),
        "register_cmd_command_ready": audit.get("register_cmd_command_ready"),
        "completion_gate_cmd_command_ready": audit.get(
            "completion_gate_cmd_command_ready"
        ),
        "required_artifact_count": audit.get("required_artifact_count"),
    }


def _evidence_full_oof_preflight_policy(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "canonical_config_errors": audit.get("canonical_config_errors"),
        "required_bad_token_count": audit.get("required_bad_token_count"),
        "missing_bad_tokens": audit.get("missing_bad_tokens"),
        "ad_hoc_config_error_count": len(audit.get("ad_hoc_config_errors", []) or []),
    }


def _evidence_full_oof_authorization_policy(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "requires_authorization_json": audit.get("requires_authorization_json"),
        "authorization_purpose": audit.get("authorization_purpose"),
        "valid_authorization_errors": audit.get("valid_authorization_errors"),
        "required_invalid_token_count": audit.get("required_invalid_token_count"),
        "missing_invalid_tokens": audit.get("missing_invalid_tokens"),
        "invalid_authorization_error_count": len(
            audit.get("invalid_authorization_errors", []) or []
        ),
    }


def _evidence_full_oof_authorization_template(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "template_authorized_default": audit.get("template_authorized_default"),
        "template_acknowledges_long_run_default": audit.get(
            "template_acknowledges_long_run_default"
        ),
        "template_acknowledges_no_claim_default": audit.get(
            "template_acknowledges_no_claim_default"
        ),
        "template_binds_preflight_config_sha256": audit.get(
            "template_binds_preflight_config_sha256"
        ),
        "template_binds_git_commit": audit.get("template_binds_git_commit"),
    }


def _evidence_full_oof_preflight_freshness(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "preflight_valid": audit.get("preflight_valid"),
        "preflight_fresh": audit.get("preflight_fresh"),
        "preflight_authorization_ready": audit.get(
            "preflight_authorization_ready"
        ),
        "full_oof_execution_allowed": audit.get("full_oof_execution_allowed"),
        "execution_blocked_reason": audit.get("execution_blocked_reason"),
        "authorization_must_refresh_preflight": audit.get(
            "authorization_must_refresh_preflight"
        ),
        "git_commit_matches": audit.get("git_commit_matches"),
        "snapshot_matches": audit.get("snapshot_matches"),
        "preflight_git_commit": audit.get("preflight_git_commit"),
        "current_git_commit": audit.get("current_git_commit"),
        "preflight_snapshot_id": audit.get("preflight_snapshot_id"),
        "current_snapshot_id": audit.get("current_snapshot_id"),
        "warnings": audit.get("warnings"),
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# classification_v2 Q2 Progress Report",
        "",
        f"Status: **{result['overall_status']}**",
        "",
        f"Claim boundary: {result['claim_boundary']}",
        "",
        "## Gates",
    ]
    for gate in result["gates"]:
        marker = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"- {marker}: {gate['name']}")
        if gate["errors"]:
            lines.append(f"  - errors: `{gate['errors']}`")
    lines.extend(["", "## Remaining Work"])
    lines.extend(f"- {item}" for item in result["remaining_work"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

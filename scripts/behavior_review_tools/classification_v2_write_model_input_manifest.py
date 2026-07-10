from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path("outputs/classification_v2/train_ready_windows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write classification_v2 model input contract manifest.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--classification-root",
        type=Path,
        default=Path("outputs/classification_v2"),
        help="Root containing native temporal units, publication splits, and model smoke artifacts.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    classification_root = args.classification_root
    manifest = {
        "version": "classification_v2_train_ready_contract_v2",
        "root": str(root),
        "artifacts": {
            "tabular_X": str(root / "X_window_features.csv"),
            "spatial_sequence_X": str(root / "X_spatial_sequences.npz"),
            "spatial_sequence_audit": str(root / "spatial_sequence_audit.json"),
            "image_sequence_loader_audit": str(root / "image_sequence_loader_smoke_audit.json"),
            "image_frame_context_manifest": str(root / "image_frame_context_manifest.csv"),
            "image_window_context_manifest": str(root / "image_window_context_manifest.csv"),
            "image_context_index_audit": str(root / "image_context_index_audit.json"),
            "image_tensor_loader_smoke_audit": str(root / "image_tensor_loader_smoke_audit.json"),
            "interaction_window_context_manifest": str(root / "interaction_window_context_manifest.csv"),
            "interaction_context_audit": str(root / "interaction_context_audit.json"),
            "y": str(root / "y_behavior.csv"),
            "y_auxiliary_targets": str(root / "y_auxiliary_targets.csv"),
            "auxiliary_targets_audit": str(root / "auxiliary_targets_audit.json"),
            "multitask_loss_audit": str(root / "multitask_loss_audit.json"),
            "train_mask": str(root / "train_mask.csv"),
            "sample_weight": str(root / "sample_weight.csv"),
            "event_weight_manifest": str(root / "event_weight_manifest.csv"),
            "event_weight_audit": str(root / "event_weight_audit.json"),
            "split_manifest": str(root / "split_manifest.csv"),
            "class_weight_policy": str(root / "class_weight_policy.json"),
            "native_temporal_unit_manifest": str(
                classification_root / "native_temporal_units" / "native_temporal_unit_manifest.csv"
            ),
            "native_temporal_unit_audit": str(
                classification_root / "native_temporal_units" / "native_temporal_unit_audit.json"
            ),
            "window_publication_split_manifest": str(
                classification_root / "publication_splits" / "publication_split_manifest.csv"
            ),
            "window_publication_split_audit": str(
                classification_root / "publication_splits" / "publication_split_audit.json"
            ),
            "native_publication_split_manifest": str(
                classification_root / "native_temporal_units_publication_splits" / "publication_split_manifest.csv"
            ),
            "native_publication_split_audit": str(
                classification_root / "native_temporal_units_publication_splits" / "publication_split_audit.json"
            ),
            "native_oof_fold_manifest": str(
                classification_root / "native_temporal_units_oof_folds" / "native_oof_fold_manifest.csv"
            ),
            "native_oof_fold_audit": str(
                classification_root / "native_temporal_units_oof_folds" / "native_oof_fold_audit.json"
            ),
            "spatial_tcn_forward_smoke_script": "scripts/dev_tools/check_classification_v2_spatial_tcn_forward.py",
            "spatial_tcn_overfit_smoke_audit": str(
                classification_root / "model_smoke" / "spatial_tcn_overfit_smoke.json"
            ),
            "spatial_tcn_overfit_smoke_checkpoint": str(
                classification_root / "model_smoke" / "spatial_tcn_overfit_smoke.pt"
            ),
            "spatial_tcn_smoke_train_audit": str(
                classification_root / "model_smoke" / "spatial_tcn_smoke_train" / "spatial_tcn_smoke_train_audit.json"
            ),
            "spatial_tcn_smoke_train_predictions": str(
                classification_root / "model_smoke" / "spatial_tcn_smoke_train" / "spatial_tcn_smoke_predictions.csv"
            ),
            "spatial_tcn_smoke_train_checkpoint": str(
                classification_root / "model_smoke" / "spatial_tcn_smoke_train" / "spatial_tcn_smoke_train.pt"
            ),
            "multimodal_forward_smoke_script": "scripts/dev_tools/check_classification_v2_multimodal_forward.py",
            "multimodal_forward_smoke_audit": str(
                classification_root / "model_smoke" / "multimodal_forward_smoke_audit.json"
            ),
            "multimodal_smoke_train_audit": str(
                classification_root / "model_smoke" / "multimodal_smoke_train" / "multimodal_smoke_train_audit.json"
            ),
            "multimodal_smoke_train_predictions": str(
                classification_root / "model_smoke" / "multimodal_smoke_train" / "multimodal_smoke_predictions.csv"
            ),
            "multimodal_smoke_train_checkpoint": str(
                classification_root / "model_smoke" / "multimodal_smoke_train" / "multimodal_smoke_train.pt"
            ),
            "experiment_registry_ledger": str(classification_root / "experiment_registry" / "experiment_ledger.jsonl"),
            "spatial_tcn_smoke_train_experiment_record": str(
                classification_root / "experiment_registry" / "spatial_tcn_smoke_train_record.json"
            ),
            "spatial_control_shortcut_audit": str(
                classification_root / "model_smoke" / "spatial_control_shortcut_audit.json"
            ),
            "native_temporal_prediction_audit": str(
                classification_root
                / "model_smoke"
                / "native_temporal_predictions"
                / "native_temporal_prediction_audit.json"
            ),
            "native_temporal_predictions": str(
                classification_root
                / "model_smoke"
                / "native_temporal_predictions"
                / "native_temporal_predictions.csv"
            ),
            "model_readiness_report_md": str(root / "model_readiness_report.md"),
            "model_readiness_report_json": str(root / "model_readiness_report.json"),
            "feature_semantics_contract": "configs/classification_v2/feature_semantics_v1.json",
            "feature_semantics_audit": str(root / "feature_semantics_audit.json"),
            "trainer_contract": "configs/classification_v2/trainer_contract_v1.json",
            "trainer_contract_audit": str(root / "trainer_contract_audit.json"),
        },
        "model_input_branches": {
            "tabular_context_branch": {
                "source": "tabular_X",
                "description": (
                    "Window-level temporal, quality, motion, geometry, and social "
                    "numeric features. Scientific feature-family semantics are "
                    "declared in feature_semantics_contract."
                ),
            },
            "spatial_temporal_branch": {
                "source": "spatial_sequence_X",
                "description": (
                    "Per-frame normalized bbox, motion deltas, feeder/drinker/toy "
                    "class-specific ROI relation, social relation, and quality masks."
                ),
            },
            "image_sequence_branch": {
                "source": "image_frame_context_manifest + image_window_context_manifest",
                "description": (
                    "Actor crop sequence index keyed by image_context_id. Legacy rows load "
                    "pre-cropped images; CVAT rows load video frames and crop bbox, with "
                    "full-frame/partner context availability audited separately."
                ),
            },
            "interaction_context_branch": {
                "source": "interaction_window_context_manifest",
                "description": (
                    "Audit-only scene/full-frame and partner-context readiness for every window. "
                    "scene_partner_context_ready is label-independent; interaction_context_ready "
                    "is the fight/social-nose subset for review compatibility. Use these columns "
                    "for asset/model gating, not as numeric model input X."
                ),
            },
        },
        "training_contract": {
            "split": "Use split_manifest.csv; never random-split frames/windows.",
            "publication_split": (
                "Use native_publication_split_manifest for confirmatory event-level evaluation; "
                "window_publication_split_manifest is for engineering/window-level sensitivity only."
            ),
            "mask": (
                "Use train_mask.csv/window_valid_for_main_train to exclude "
                "invalid/incomplete/review-excluded windows."
            ),
            "sample_weight": "Use sample_weight.csv and class_weight_policy.json together.",
            "event_weight": (
                "Use event_weight_manifest.csv/event_balanced_sample_weight for overlapping-window "
                "training augmentation; do not treat window count as independent test sample size."
            ),
            "prediction_schema": (
                "Smoke/baseline prediction CSVs must include row_index, window_id, source_split, "
                "y_true, y_pred, confidence, and prediction_split."
            ),
            "label": "Use y_behavior.csv only as target y.",
            "auxiliary_labels": "Use y_auxiliary_targets.csv only as auxiliary y/mask heads, never as input X.",
            "auxiliary_loss": (
                "Use masked auxiliary losses for posture, motion/context, ROI intent, and interaction heads. "
                "Rows with false has_*_aux_target masks must contribute zero for that auxiliary head."
            ),
            "trainer_input": (
                "Use trainer_contract_v1 tabular_feature_whitelist exactly. "
                "Never build X from all numeric columns in a manifest."
            ),
            "oof_evaluation": (
                "Use native_oof_fold_manifest.csv for grouped out-of-fold native temporal-unit evaluation. "
                "Each recording_group_id is held out exactly once."
            ),
            "primary_prediction_unit": "native temporal unit / review unit, not overlapping sequence window",
        },
        "forbidden_model_inputs": [
            "manual_*",
            "review_*",
            "behavior_before_review",
            "original_behavior",
            "review_unit_id",
            "window_id",
            "temporal_unit_key",
            "video_key",
            "dataset_id",
            "pig_id",
            "track_id",
            "path columns",
            "label/policy text columns",
            "target_roi_*",
            "roi_target_*",
        ],
        "upgrade_path": [
            "smoke_test_tabular_baseline",
            "add_spatial_temporal_branch",
            "add_image_sequence_branch",
            "add_multimodal_forward_smoke",
            "add_multimodal_tiny_overfit_smoke",
            "add_interaction_full_frame_partner_context_audit",
            "add_auxiliary_multitask_targets",
            "add_masked_auxiliary_multitask_loss",
            "multi_task_heads_posture_motion_roi_interaction",
            "graph_social_interaction_branch",
            "hard_negative_mining_from_confusion_focus",
            "active_learning_uncertain_windows",
        ],
    }
    missing = [name for name, path in manifest["artifacts"].items() if not Path(path).exists()]
    manifest["missing_artifacts"] = missing

    output_json = args.output_json or (root / "model_input_contract.json")
    output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

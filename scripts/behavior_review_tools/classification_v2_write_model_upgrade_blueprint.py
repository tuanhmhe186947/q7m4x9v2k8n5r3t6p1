from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/classification_v2/train_ready_windows")
CLASSIFICATION_ROOT = Path("outputs/classification_v2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write classification_v2 model-upgrade blueprint.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--classification-root", type=Path, default=CLASSIFICATION_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    classification_root = args.classification_root
    contract = _load_json(root / "model_input_contract.json")
    spatial = _load_json(root / "spatial_sequence_audit.json")
    class_weights = _load_json(root / "class_weight_policy.json")
    auxiliary_targets = _load_json(root / "auxiliary_targets_audit.json")
    image_context = _load_json(root / "image_context_index_audit.json")
    image_tensor_loader = _load_json(root / "image_tensor_loader_smoke_audit.json")
    interaction_context = _load_json(root / "interaction_context_audit.json")
    source_shortcut = _load_json(classification_root / "model_smoke" / "source_shortcut_audit.json")
    spatial_controls = _load_json(classification_root / "model_smoke" / "spatial_control_shortcut_audit.json")
    smoke_train = _load_json(
        classification_root / "model_smoke" / "spatial_tcn_smoke_train" / "spatial_tcn_smoke_train_audit.json"
    )
    multimodal_forward = _load_json(classification_root / "model_smoke" / "multimodal_forward_smoke_audit.json")
    multimodal_smoke_train = _load_json(
        classification_root / "model_smoke" / "multimodal_smoke_train" / "multimodal_smoke_train_audit.json"
    )
    native_predictions = _load_json(
        classification_root / "model_smoke" / "native_temporal_predictions" / "native_temporal_prediction_audit.json"
    )
    experiment_record = _load_json(
        classification_root / "experiment_registry" / "spatial_tcn_smoke_train_record.json"
    )

    blueprint = {
        "version": "classification_v2_model_upgrade_blueprint_v2",
        "scope": (
            "Framework roadmap and smoke-ready implementation surface only. Full model training and "
            "publication claims require separate controlled experiments."
        ),
        "publication_claim_boundary": {
            "target_claim": "Q2-strong: improved pig behavior recognition under session/video-safe validation.",
            "not_claimed": "Q1 generalization across external farm/camera/cohort.",
            "reason": "No external farm/camera/cohort validation is available in the current artifacts.",
            "identity_scope": "pig_id is an annotation/track ID within a video/session, not biological identity across videos.",
            "primary_prediction_unit": "native temporal unit / review unit; overlapping windows are training augmentation.",
        },
        "pass_fail_snapshot": _pass_fail_snapshot(contract, image_context, smoke_train, spatial_controls),
        "artifact_contract": contract.get("artifacts", {}),
        "data_contract": {
            "X": {
                "tabular_context": contract["artifacts"].get("tabular_X"),
                "spatial_temporal": contract["artifacts"].get("spatial_sequence_X"),
                "image_context_index": contract["artifacts"].get("image_frame_context_manifest"),
                "forbidden_inputs": contract.get("forbidden_model_inputs", []),
            },
            "y": contract["artifacts"].get("y"),
            "mask_weight": {
                "train_mask": contract["artifacts"].get("train_mask"),
                "sample_weight": contract["artifacts"].get("sample_weight"),
                "event_weight": contract["artifacts"].get("event_weight_manifest"),
            },
            "validation": {
                "split_manifest": contract["artifacts"].get("split_manifest"),
                "native_publication_split": contract["artifacts"].get("native_publication_split_manifest"),
                "rule": "No random frame/window split; report native temporal-unit metrics for confirmatory evaluation.",
            },
        },
        "module_script_design": {
            "image_context_index": {
                "module": "src/pig_behavior/classification_v2/datasets/image_context_index.py",
                "builder": "scripts/behavior_review_tools/classification_v2_build_image_context_index.py",
                "checker": "scripts/dev_tools/check_classification_v2_image_context_index.py",
                "purpose": "Frame-level actor crop/video+bbox index keyed by image_context_id plus window references.",
            },
            "image_sequence_tensor_loader": {
                "module": "src/pig_behavior/classification_v2/datasets/image_sequence_dataset.py",
                "checker": "scripts/dev_tools/check_classification_v2_image_tensor_loader.py",
                "purpose": "Load legacy crop and CVAT video+bbox sequences into [B,T,3,H,W] tensors with masks.",
            },
            "spatial_tcn_smoke_trainer": {
                "module": "src/pig_behavior/classification_v2/training/spatial_tcn_smoke.py",
                "runner": "scripts/behavior_review_tools/classification_v2_spatial_tcn_smoke_train.py",
                "checker": "scripts/dev_tools/check_classification_v2_spatial_tcn_smoke_train.py",
                "purpose": "Reusable split-safe smoke training, prediction CSV schema, checkpoint, and metrics audit.",
            },
            "multimodal_forward_smoke": {
                "module": "src/pig_behavior/classification_v2/models/multimodal_fusion.py",
                "checker": "scripts/dev_tools/check_classification_v2_multimodal_forward.py",
                "purpose": (
                    "Forward-pass smoke for late fusion of image crop sequences and whitelisted "
                    "spatial-temporal bbox/ROI/social feature groups."
                ),
            },
            "multimodal_smoke_trainer": {
                "module": "src/pig_behavior/classification_v2/training/multimodal_smoke.py",
                "runner": "scripts/behavior_review_tools/classification_v2_multimodal_smoke_train.py",
                "checker": "scripts/dev_tools/check_classification_v2_multimodal_smoke_train.py",
                "purpose": "Tiny split-safe image+spatial overfit smoke; not full training.",
            },
            "interaction_context_index": {
                "module": "src/pig_behavior/classification_v2/datasets/interaction_context_index.py",
                "builder": "scripts/behavior_review_tools/classification_v2_build_interaction_context_index.py",
                "checker": "scripts/dev_tools/check_classification_v2_interaction_context_index.py",
                "purpose": "Audit full-frame and partner-context readiness for fight/social-nose windows.",
            },
            "auxiliary_targets": {
                "builder": "scripts/behavior_review_tools/classification_v2_build_auxiliary_targets.py",
                "checker": "scripts/dev_tools/check_classification_v2_auxiliary_targets.py",
                "purpose": "Build y-only posture/motion/ROI/interaction auxiliary targets and masks.",
            },
            "experiment_registry": {
                "module": "src/pig_behavior/classification_v2/experiments/registry.py",
                "runner": "scripts/behavior_review_tools/classification_v2_register_experiment.py",
                "checker": "scripts/dev_tools/check_classification_v2_experiment_registry.py",
                "purpose": "File-based provenance record with artifact hashes, git commit, config, and metrics.",
            },
            "shortcut_controls": {
                "checker": "scripts/dev_tools/check_classification_v2_spatial_control_shortcuts.py",
                "purpose": "Quantify source shortcut under real, repeat-first-frame, and mean-only spatial controls.",
            },
            "native_temporal_prediction_collapse": {
                "module": "src/pig_behavior/classification_v2/evaluation/native_temporal_collapse.py",
                "runner": "scripts/behavior_review_tools/classification_v2_collapse_window_predictions_to_native_units.py",
                "checker": "scripts/dev_tools/check_classification_v2_native_temporal_predictions.py",
                "purpose": "Collapse overlapping window predictions to native temporal/review units for claim-safe metrics.",
            },
        },
        "training_phases": _training_phases(
            contract,
            spatial,
            image_context,
            image_tensor_loader,
            class_weights,
            smoke_train,
            native_predictions,
            multimodal_forward,
            multimodal_smoke_train,
            interaction_context,
            auxiliary_targets,
        ),
        "known_risks": [
            {
                "risk": "source/domain shortcut is strong",
                "evidence": {
                    "tabular_source_balanced_accuracy": source_shortcut.get("balanced_accuracy"),
                    "spatial_control_balanced_accuracy": {
                        name: values.get("balanced_accuracy")
                        for name, values in spatial_controls.get("controls", {}).items()
                    },
                },
                "required_control": "Report source-balanced, video/session-safe, and native temporal-unit metrics.",
            },
            {
                "risk": "overlapping windows inflate apparent sample size",
                "required_control": "Use event-balanced weights for training and native temporal-unit evaluation for claims.",
            },
            {
                "risk": "interaction labels need full-frame/partner context",
                "required_control": "Use CVAT video full-frame rendering and partner overlays for interaction review/training audits.",
            },
        ],
        "next_recommended_commands": [
            "python scripts/behavior_review_tools/classification_v2_build_image_context_index.py",
            "python scripts/dev_tools/check_classification_v2_image_context_index.py",
            "python scripts/behavior_review_tools/classification_v2_spatial_tcn_smoke_train.py --steps 8 --per-class-train 4 --per-class-eval 2 --hidden-dim 64",
            "python scripts/dev_tools/check_classification_v2_spatial_tcn_smoke_train.py",
            "python scripts/dev_tools/check_classification_v2_spatial_control_shortcuts.py --max-rows-per-split 5000",
            "python scripts/behavior_review_tools/classification_v2_register_experiment.py --name spatial_tcn_smoke_train --metrics-json outputs/classification_v2/model_smoke/spatial_tcn_smoke_train/spatial_tcn_smoke_train_audit.json --artifact outputs/classification_v2/model_smoke/spatial_tcn_smoke_train/spatial_tcn_smoke_train_audit.json --artifact outputs/classification_v2/model_smoke/spatial_tcn_smoke_train/spatial_tcn_smoke_predictions.csv --artifact outputs/classification_v2/model_smoke/spatial_tcn_smoke_train/spatial_tcn_smoke_train.pt --artifact outputs/classification_v2/train_ready_windows/model_input_contract.json --notes split_safe_smoke_subset_not_full_training",
        ],
    }
    output_json = args.output_json or (root / "model_upgrade_blueprint.json")
    output_json.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "version": blueprint["version"],
                "phase_count": len(blueprint["training_phases"]),
                "pass_fail": blueprint["pass_fail_snapshot"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _training_phases(
    contract: dict[str, Any],
    spatial: dict[str, Any],
    image_context: dict[str, Any],
    image_tensor_loader: dict[str, Any],
    class_weights: dict[str, Any],
    smoke_train: dict[str, Any],
    native_predictions: dict[str, Any],
    multimodal_forward: dict[str, Any],
    multimodal_smoke_train: dict[str, Any],
    interaction_context: dict[str, Any],
    auxiliary_targets: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "phase": "P0_data_contract_and_leakage_gates",
            "status": "implemented",
            "evidence": {
                "model_input_contract": contract.get("version"),
                "missing_artifacts": contract.get("missing_artifacts", []),
                "forbidden_inputs": contract.get("forbidden_model_inputs", []),
            },
        },
        {
            "phase": "P1_spatial_temporal_bbox_roi_social",
            "status": "smoke_trainer_implemented",
            "inputs": ["X_spatial_sequences.npz", "train_mask.csv", "sample_weight.csv", "event_weight_manifest.csv"],
            "model": "Mask-aware SpatialTCN baseline over bbox, motion, ROI, social, and quality groups.",
            "evidence": {
                "array_shapes": spatial.get("array_shapes", {}),
                "smoke_train_rows": smoke_train.get("train_rows"),
                "smoke_eval_rows": smoke_train.get("eval_rows"),
                "smoke_loss_reduction": smoke_train.get("loss_reduction"),
            },
        },
        {
            "phase": "P2_image_sequence_branch",
            "status": "image_context_index_ready",
            "inputs": ["image_frame_context_manifest.csv", "image_window_context_manifest.csv"],
            "model": "Actor crop encoder plus optional full-frame/partner context for interaction samples.",
            "evidence": {
                "frame_rows": image_context.get("frame_rows"),
                "frame_loadable_count": image_context.get("frame_loadable_count"),
                "frame_unloadable_count": image_context.get("frame_unloadable_count"),
                "duplicate_image_context_id": image_context.get("duplicate_image_context_id"),
                "tensor_loader_batch_shape": image_tensor_loader.get("batch_shape"),
                "tensor_loader_observed_slots": image_tensor_loader.get("observed_slots"),
                "tensor_loader_length_slots": image_tensor_loader.get("length_slots"),
            },
        },
        {
            "phase": "P3_multimodal_fusion",
            "status": "forward_smoke_implemented",
            "model": "Late fusion of spatial-temporal embedding, image-sequence embedding, tabular context, and masks.",
            "evidence": {
                "batch_shape": multimodal_forward.get("batch_shape"),
                "logit_shape": multimodal_forward.get("logit_shape"),
                "max_masked_padding_delta": multimodal_forward.get("max_masked_padding_delta"),
                "smoke_train_rows": multimodal_smoke_train.get("train_rows"),
                "smoke_eval_rows": multimodal_smoke_train.get("eval_rows"),
                "smoke_loss_reduction": multimodal_smoke_train.get("loss_reduction"),
                "errors": multimodal_forward.get("errors", []),
            },
            "required_before_full_training": [
                "source-balanced validation report",
                "native temporal-unit prediction collapse",
                "interaction full-frame/partner branch for interaction claims",
            ],
        },
        {
            "phase": "P3b_native_temporal_evaluation",
            "status": "collapse_schema_implemented",
            "model": "Confidence-weighted vote from window predictions to temporal_unit_key.",
            "evidence": {
                "window_prediction_rows": native_predictions.get("window_prediction_rows"),
                "native_units_predicted": native_predictions.get("native_units_predicted"),
                "native_units_unpredicted": native_predictions.get("native_units_unpredicted"),
            },
        },
        {
            "phase": "P4_multitask_heads",
            "status": "auxiliary_targets_implemented",
            "heads": {
                "behavior": ["drink", "eat", "fight", "social-nose", "explore", "lying", "stand", "move", "sitting", "playwithtoy"],
                "posture": ["lying", "sitting", "standing_or_other"],
                "motion_context": ["move", "explore", "stand", "other"],
                "roi_intent": ["eat", "drink", "playwithtoy", "none"],
                "interaction": ["fight", "social-nose", "none"],
            },
            "loss": "Weighted behavior loss plus auxiliary heads; use class/event/sample weights.",
            "evidence": {
                "auxiliary_target_rows": auxiliary_targets.get("rows"),
                "positive_counts": auxiliary_targets.get("aux_target_positive_counts", {}),
                "errors": auxiliary_targets.get("errors", []),
            },
        },
        {
            "phase": "P4b_interaction_full_frame_partner_context",
            "status": "audit_index_implemented",
            "inputs": ["interaction_window_context_manifest.csv", "interaction_context_audit.json"],
            "evidence": {
                "interaction_window_rows": interaction_context.get("interaction_window_rows"),
                "interaction_ready_rows": interaction_context.get("interaction_ready_rows"),
                "interaction_status_counts": interaction_context.get("interaction_status_counts", {}),
                "interaction_label_counts": interaction_context.get("interaction_label_counts", {}),
            },
            "claim_gate": (
                "fight/social-nose model claims should report context-ready subset metrics or explicitly "
                "state when crop-only rows lack full-frame partner context."
            ),
        },
        {
            "phase": "P5_graph_social_branch",
            "status": "design_ready",
            "inputs": ["nearest/pair relation features", "partner bbox/crops", "full-frame context for CVAT"],
            "target_confusions": ["fight_vs_social-nose", "fight_vs_move", "social-nose_actor_only"],
        },
        {
            "phase": "P6_hard_negative_and_active_learning",
            "status": "evaluation_ready",
            "source": class_weights.get("confusion_focus_pairs", []),
            "policy": "Select uncertain/focus-pair mistakes into review_unit shortlist; never overwrite labels automatically.",
        },
    ]


def _pass_fail_snapshot(
    contract: dict[str, Any],
    image_context: dict[str, Any],
    smoke_train: dict[str, Any],
    spatial_controls: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    if contract.get("missing_artifacts"):
        failures.append("contract_missing_artifacts")
    if image_context.get("frame_unloadable_count") not in {0, None}:
        failures.append("image_context_unloadable_frames")
    if image_context.get("duplicate_image_context_id") not in {0, None}:
        failures.append("duplicate_image_context_id")
    if smoke_train.get("errors"):
        failures.append("spatial_tcn_smoke_train_errors")
    if spatial_controls.get("errors"):
        failures.append("spatial_control_shortcut_errors")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "caveats": [
            "PASS means smoke/data-contract readiness, not full training readiness.",
            "Strong source shortcut remains a scientific risk and must be controlled in evaluation.",
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/classification_v2/train_ready_windows")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a concrete model-upgrade blueprint for classification_v2 training."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    contract = _load_json(root / "model_input_contract.json")
    spatial = _load_json(root / "spatial_sequence_audit.json")
    class_weights = _load_json(root / "class_weight_policy.json")
    smoke_metrics_path = root / "smoke_tabular_baseline" / "smoke_metrics.json"
    confusion_path = root / "smoke_tabular_baseline" / "confusion_focus_eval.json"
    smoke_metrics = _load_json(smoke_metrics_path) if smoke_metrics_path.exists() else {}
    confusion = _load_json(confusion_path) if confusion_path.exists() else {}

    blueprint = {
        "version": "classification_v2_model_upgrade_blueprint_v1",
        "scope": "Groundwork only: architecture/evaluation plan and artifact contract; no full training performed.",
        "root": str(root),
        "required_artifacts": contract["artifacts"],
        "data_contract": {
            "split": "Use split_manifest.csv; group-safe by source_type|dataset_id|video_key.",
            "label": "Use y_behavior.csv as the only y source.",
            "mask": "Use train_mask.csv to exclude invalid/incomplete/review-excluded windows.",
            "sample_weight": "Use sample_weight.csv and class_weight_policy.json.",
            "tabular_features": {
                "source": contract["artifacts"]["tabular_X"],
                "feature_count": smoke_metrics.get("feature_count"),
            },
            "spatial_sequences": {
                "source": contract["artifacts"]["spatial_sequence_X"],
                "array_shapes": spatial.get("array_shapes", {}),
                "feature_names": spatial.get("feature_names", {}),
                "observed_mask": "Use observed_mask to mask padded or missing frame slots.",
            },
            "image_sequences": {
                "source": "runtime loader from reviewed_frame_features.csv + split_manifest.csv",
                "smoke_audit": contract["artifacts"]["image_sequence_loader_audit"],
            },
        },
        "training_phases": [
            {
                "phase": "P0_tabular_smoke_baseline",
                "status": "implemented_smoke_only",
                "inputs": ["tabular_features"],
                "model": "standardized linear/logistic baseline or shallow MLP",
                "purpose": "Verify split/mask/sample_weight/class_weight contract and produce confusion-focused predictions.",
                "current_smoke_outputs": {
                    "metrics_json": str(smoke_metrics_path),
                    "predictions_csv": str(root / "smoke_tabular_baseline" / "smoke_predictions.csv"),
                    "confusion_focus_json": str(confusion_path),
                    "val_accuracy": smoke_metrics.get("metrics", {}).get("val", {}).get("accuracy"),
                    "test_accuracy": smoke_metrics.get("metrics", {}).get("test", {}).get("accuracy"),
                    "test_macro_f1": smoke_metrics.get("metrics", {}).get("test", {}).get("macro_f1"),
                },
            },
            {
                "phase": "P1_spatial_temporal_branch",
                "status": "ready_for_smoke_training",
                "inputs": ["spatial_sequences", "tabular_features"],
                "model": "TCN/GRU/Transformer encoder over per-frame spatial arrays plus tabular fusion.",
                "must_use": ["observed_mask", "train_mask", "sample_weight", "class_weights"],
                "expected_gain": "Better separation of move/explore/stand, lying/sitting transitions, and interaction proximity cues.",
            },
            {
                "phase": "P2_image_sequence_branch",
                "status": "loader_smoke_passed",
                "inputs": ["image_sequences", "spatial_sequences", "tabular_features"],
                "model": "Small CNN/ViT frame encoder with temporal pooling; keep sequence-level split.",
                "must_verify_before_training": [
                    "Batch loader tensor shape for train/val/test",
                    "GPU/CPU fallback",
                    "No path/id/review columns enter model tensor",
                ],
            },
            {
                "phase": "P3_multi_task_heads",
                "status": "design_ready",
                "heads": {
                    "behavior": ["drink", "eat", "fight", "social-nose", "explore", "lying", "stand", "move", "sitting", "playwithtoy"],
                    "posture": ["lying", "sitting", "standing_or_other"],
                    "motion_context": ["move", "explore", "stand", "other"],
                    "roi_intent": ["eat", "drink", "playwithtoy", "none"],
                    "interaction": ["fight", "social-nose", "none"],
                },
                "loss": "Weighted sum; behavior remains primary, auxiliary heads regularize confusion groups.",
            },
            {
                "phase": "P4_graph_social_branch",
                "status": "design_ready",
                "inputs": ["social_relation", "nearest/pair overlap features", "optional full-frame partner crops later"],
                "model": "Per-frame pig graph encoder over actor-nearest/partner features; fuse with temporal branch.",
                "target_confusions": ["fight_vs_social-nose", "fight_vs_move", "social-nose_actor_only"],
            },
            {
                "phase": "P5_hard_negative_mining",
                "status": "evaluation_ready",
                "source": str(confusion_path),
                "pairs": class_weights.get("confusion_focus_pairs", []),
                "policy": "After each smoke/full run, sample high-confidence mistakes from focus pairs into review/active-learning queue.",
            },
            {
                "phase": "P6_active_learning_loop",
                "status": "design_ready",
                "selection": [
                    "low confidence windows",
                    "high entropy between focus-pair classes",
                    "rare class candidates: playwithtoy/drink/social-nose/fight",
                    "source-domain disagreements legacy vs cvat",
                ],
                "output": "review_unit shortlist, not direct label overwrite.",
            },
        ],
        "current_confusion_focus_top": _top_focus_confusions(confusion),
        "class_weight_policy": {
            "source": str(root / "class_weight_policy.json"),
            "class_counts": class_weights.get("class_counts", {}),
            "class_weights": class_weights.get("class_weights", {}),
        },
        "hard_constraints": contract.get("forbidden_model_inputs", []),
        "next_recommended_commands": [
            "python scripts/behavior_review_tools/classification_v2_train_smoke_tabular.py",
            "python scripts/dev_tools/evaluate_classification_v2_confusion_focus.py --predictions-csv outputs/classification_v2/train_ready_windows/smoke_tabular_baseline/smoke_predictions.csv --output-json outputs/classification_v2/train_ready_windows/smoke_tabular_baseline/confusion_focus_eval.json",
            "Implement P1 spatial-temporal branch smoke trainer before image backbone full training.",
        ],
    }

    output_json = args.output_json or (root / "model_upgrade_blueprint.json")
    output_json.write_text(json.dumps(blueprint, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "phases": len(blueprint["training_phases"])}, indent=2))


def _top_focus_confusions(confusion: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    pairs = confusion.get("focus_pairs", {})
    rows = []
    for pair, values in pairs.items():
        rows.append({"pair": pair, **values})
    return sorted(rows, key=lambda x: int(x.get("total_pair_confusions", 0)), reverse=True)[:limit]


if __name__ == "__main__":
    main()

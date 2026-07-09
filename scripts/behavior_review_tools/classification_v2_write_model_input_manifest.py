from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path("outputs/classification_v2/train_ready_windows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write classification_v2 model input contract manifest.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    manifest = {
        "version": "classification_v2_train_ready_contract_v1",
        "root": str(root),
        "artifacts": {
            "tabular_X": str(root / "X_window_features.csv"),
            "spatial_sequence_X": str(root / "X_spatial_sequences.npz"),
            "image_sequence_loader_audit": str(root / "image_sequence_loader_smoke_audit.json"),
            "y": str(root / "y_behavior.csv"),
            "train_mask": str(root / "train_mask.csv"),
            "sample_weight": str(root / "sample_weight.csv"),
            "event_weight_manifest": str(root / "event_weight_manifest.csv"),
            "split_manifest": str(root / "split_manifest.csv"),
            "class_weight_policy": str(root / "class_weight_policy.json"),
        },
        "model_input_branches": {
            "tabular_context_branch": {
                "source": "tabular_X",
                "description": (
                    "Window-level geometry, motion, non-target ROI relation, social, "
                    "and quality numeric features."
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
                "source": "runtime loader from reviewed_frame_features + split_manifest",
                "description": (
                    "Legacy crop sequence or CVAT video+bbox crop sequence, checked by "
                    "image_sequence_loader_smoke_audit."
                ),
            },
        },
        "training_contract": {
            "split": "Use split_manifest.csv; never random-split frames/windows.",
            "mask": (
                "Use train_mask.csv/window_valid_for_main_train to exclude "
                "invalid/incomplete/review-excluded windows."
            ),
            "sample_weight": "Use sample_weight.csv and class_weight_policy.json together.",
            "event_weight": (
                "Use event_weight_manifest.csv/event_balanced_sample_weight for overlapping-window "
                "training augmentation; do not treat window count as independent test sample size."
            ),
            "label": "Use y_behavior.csv only as target y.",
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

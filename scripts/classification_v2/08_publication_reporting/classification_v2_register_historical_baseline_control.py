"""Register invalid historical classifier artifacts as engineering evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.experiments.historical_baseline import (
    HistoricalFullOOFConfig,
    LegacySequenceCheckpointConfig,
    build_historical_baseline_reconciliation,
    write_historical_baseline_reconciliation,
)

ROOT = Path("outputs/classification_v2/train_ready_windows")
RUN_ROOT = Path("outputs/classification_v2/model_full/full_multimodal_oof")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register historical classifier artifacts without promoting their "
            "invalid performance metrics."
        )
    )
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "split_manifest.csv")
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=ROOT / "image_window_context_manifest.csv",
    )
    parser.add_argument(
        "--interaction-manifest",
        type=Path,
        default=ROOT / "interaction_window_context_manifest.csv",
    )
    parser.add_argument(
        "--run-audit",
        type=Path,
        default=RUN_ROOT / "full_multimodal_oof_audit.json",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=RUN_ROOT / "full_multimodal_oof_metrics.json",
    )
    parser.add_argument(
        "--prediction-schema-json",
        type=Path,
        default=RUN_ROOT / "full_multimodal_oof_prediction_schema_audit.json",
    )
    parser.add_argument(
        "--window-predictions",
        type=Path,
        default=RUN_ROOT / "full_multimodal_oof_predictions.csv",
    )
    parser.add_argument(
        "--native-predictions",
        type=Path,
        default=RUN_ROOT / "full_multimodal_oof_unit_predictions.csv",
    )
    parser.add_argument(
        "--fold-artifact-dir",
        type=Path,
        default=RUN_ROOT / "fold_artifacts",
    )
    parser.add_argument(
        "--origin-git-commit",
        default="18d6692705e9f77d137ddc152963e4156d782745",
    )
    parser.add_argument(
        "--alignment-fix-commit",
        default="bfdf9131ce9fe6746bc662c403db3cc702c76a7e",
    )
    parser.add_argument("--expected-manifest-rows", type=int, default=160_740)
    parser.add_argument("--expected-mismatch-rows", type=int, default=151_440)
    parser.add_argument(
        "--legacy-checkpoint",
        type=Path,
        default=Path("models/behavior/pig_behavior_sequence.pt"),
    )
    parser.add_argument(
        "--legacy-checkpoint-sha256",
        default="e85ca2e30a1962ebc6d2b053152933116f4526a80453e384174f17b574f4fe11",
    )
    parser.add_argument("--without-legacy-checkpoint", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/experiment_registry/historical_controls/"
            "historical_baseline_reconciliation_18d6692.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    full_config = HistoricalFullOOFConfig(
        split_manifest_csv=args.split_manifest,
        image_manifest_csv=args.image_manifest,
        interaction_manifest_csv=args.interaction_manifest,
        run_audit_json=args.run_audit,
        metrics_json=args.metrics_json,
        prediction_schema_json=args.prediction_schema_json,
        window_predictions_csv=args.window_predictions,
        native_predictions_csv=args.native_predictions,
        fold_artifact_dir=args.fold_artifact_dir,
        origin_git_commit=args.origin_git_commit,
        alignment_fix_commit=args.alignment_fix_commit,
        expected_manifest_rows=args.expected_manifest_rows,
        expected_positional_mismatch_rows=args.expected_mismatch_rows,
    )
    legacy_config = None
    if not args.without_legacy_checkpoint:
        legacy_config = LegacySequenceCheckpointConfig(
            checkpoint_path=args.legacy_checkpoint,
            expected_sha256=args.legacy_checkpoint_sha256,
        )
    payload = build_historical_baseline_reconciliation(
        full_config,
        legacy_config,
    )
    write_historical_baseline_reconciliation(
        payload,
        args.output_json,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "historical_full_oof_status": payload[
                    "historical_full_oof"
                ]["status"],
                "legacy_sequence_checkpoint_status": (
                    payload.get("legacy_sequence_checkpoint") or {}
                ).get("status"),
                "performance_claim_allowed": False,
                "errors": payload["errors"],
                "valid": payload["valid"],
            },
            indent=2,
        )
    )
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

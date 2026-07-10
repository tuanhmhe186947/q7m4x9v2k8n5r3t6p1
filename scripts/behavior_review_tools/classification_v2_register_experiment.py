from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.experiments.registry import ExperimentRecordConfig, write_experiment_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a classification_v2 experiment record.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_v2/experiment_registry"))
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--experiment-stage", default="engineering_smoke")
    parser.add_argument("--paper-facing", action="store_true")
    parser.add_argument(
        "--result-kind",
        default="protocol_gate",
        choices=["protocol_gate", "data_gate", "review_gate", "engineering_smoke", "model_evaluation", "baseline_evaluation", "ablation_evaluation"],
        help="Scientific role of this record; model-like results must include native temporal metrics.",
    )
    parser.add_argument("--primary-metric-unit", default="native_temporal_unit")
    parser.add_argument("--split-policy", default="recording_group_oof")
    parser.add_argument(
        "--external-generalization-claim",
        action="store_true",
        help="Only use when an external farm/camera/cohort test set exists.",
    )
    parser.add_argument("--max-hash-bytes", type=int, default=100_000_000)
    parser.add_argument(
        "--dataset-snapshot-json",
        type=Path,
        default=Path("outputs/classification_v2/training_snapshots/c2v2_eb531fc8c09011b3.json"),
    )
    parser.add_argument(
        "--paper-protocol-json",
        type=Path,
        default=Path("configs/classification_v2/paper_grade_protocol_v1.json"),
    )
    parser.add_argument(
        "--paper-protocol-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/paper_grade_protocol/paper_grade_protocol_audit.json"),
    )
    parser.add_argument(
        "--source-domain-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/source_domain_controls/source_domain_control_audit.json"),
    )
    parser.add_argument(
        "--native-oof-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units_oof_folds/native_oof_fold_audit.json"),
    )
    parser.add_argument(
        "--trainer-contract-json",
        type=Path,
        default=Path("configs/classification_v2/trainer_contract_v1.json"),
    )
    parser.add_argument(
        "--loader-input-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/loader_input_audit.json"),
    )
    args = parser.parse_args()

    record = write_experiment_record(
        ExperimentRecordConfig(
            name=args.name,
            output_dir=args.output_dir,
            metrics_json=args.metrics_json,
            artifacts=tuple(args.artifact),
            notes=args.notes,
            experiment_stage=args.experiment_stage,
            paper_facing=args.paper_facing,
            dataset_snapshot_json=args.dataset_snapshot_json,
            paper_protocol_json=args.paper_protocol_json,
            paper_protocol_audit_json=args.paper_protocol_audit_json,
            source_domain_audit_json=args.source_domain_audit_json,
            native_oof_audit_json=args.native_oof_audit_json,
            trainer_contract_json=args.trainer_contract_json,
            loader_input_audit_json=args.loader_input_audit_json,
            result_kind=args.result_kind,
            primary_metric_unit=args.primary_metric_unit,
            split_policy=args.split_policy,
            external_generalization_claim=args.external_generalization_claim,
            max_hash_bytes=args.max_hash_bytes,
        )
    )
    print(json.dumps({"record_path": record["record_path"], "ledger_path": record["ledger_path"]}, indent=2))


if __name__ == "__main__":
    main()

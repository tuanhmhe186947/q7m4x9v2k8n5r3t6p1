from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.training_snapshot import check_training_snapshot
from pig_behavior.classification_v2.training.config import (
    load_training_config,
    resolve_temporal_view_manifest,
    training_config_to_jsonable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check strict classification_v2 training config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_v2/multimodal_context_multitask.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/training_config_audit.json"),
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    errors: list[str] = []
    required_paths = {
        name: getattr(config.dataset, name)
        for name in [
            "snapshot_json",
            "trainer_contract_json",
            "train_ready_root",
            "actor_packed_cache",
            "actor_packed_index",
            "visual_cache_manifest",
            "visual_packed_cache",
            "visual_packed_index",
            "native_oof_fold_manifest",
            "grouped_fold_roles",
            "temporal_view_selection_manifest",
            "fold_event_weight_manifest",
            "auxiliary_targets_csv",
        ]
    }
    required_paths["temporal_view_manifest"] = resolve_temporal_view_manifest(
        config
    )
    missing_paths = {name: str(path) for name, path in required_paths.items() if not path.exists()}
    if missing_paths:
        errors.append(f"missing_paths={missing_paths}")
    snapshot = check_training_snapshot(config.dataset.snapshot_json)
    if snapshot.get("valid") is not True:
        errors.append(f"snapshot_invalid={snapshot.get('errors')}")
    trainer_contract = json.loads(config.dataset.trainer_contract_json.read_text(encoding="utf-8"))
    forbidden = set(trainer_contract.get("forbidden_x_patterns", []))
    if not {"manual_*", "review_*", "*behavior*", "*label*", "source_type"}.issubset(forbidden):
        errors.append("trainer_contract_forbidden_patterns_incomplete")
    result = {
        "schema_version": "classification_v2_training_config_audit_v1",
        "config_json": str(args.config),
        "config": training_config_to_jsonable(config),
        "snapshot_id": snapshot.get("expected_snapshot_id"),
        "snapshot_valid": snapshot.get("valid"),
        "missing_paths": missing_paths,
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

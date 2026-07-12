"""Validate Q2 B2-B7 baseline configs against the predeclared ablation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.contracts.training_snapshot import check_training_snapshot
from pig_behavior.classification_v2.training.config import load_training_config

BASELINE_CONFIGS = {
    "B2": Path("configs/classification_v2/baseline_spatial_tcn.json"),
    "B3": Path("configs/classification_v2/baseline_actor_image.json"),
    "B4": Path("configs/classification_v2/baseline_actor_spatial.json"),
    "B5": Path("configs/classification_v2/baseline_actor_spatial_partner_context.json"),
    "B6": Path("configs/classification_v2/baseline_actor_spatial_partner_multitask.json"),
    "B7": Path("configs/classification_v2/full_candidate_domain_controls.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 Q2 baseline training configs.")
    parser.add_argument(
        "--matrix-json",
        type=Path,
        default=Path("configs/classification_v2/q2_ablation_matrix_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_baseline_config_audit.json"),
    )
    args = parser.parse_args()

    matrix = json.loads(args.matrix_json.read_text(encoding="utf-8"))
    expected_by_id = {row["id"]: row for row in matrix.get("baselines", [])}
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    snapshot_cache: dict[Path, dict[str, Any]] = {}

    for baseline_id, path in BASELINE_CONFIGS.items():
        expected = expected_by_id.get(baseline_id)
        if expected is None:
            errors.append(f"missing_matrix_baseline={baseline_id}")
            continue
        if not path.exists():
            errors.append(f"missing_config={baseline_id}:{path}")
            continue
        config = load_training_config(path)
        observed = {
            "enable_image": config.model.enable_image,
            "enable_spatial": config.model.enable_spatial,
            "enable_interaction_context": config.model.enable_interaction_context,
            "enable_visual_context": config.model.enable_visual_context,
            "enable_multitask": config.model.enable_multitask,
            "spatial_groups": list(config.model.spatial_feature_groups),
        }
        snapshot_path = config.dataset.snapshot_json
        if snapshot_path not in snapshot_cache:
            # Snapshot profiling hashes large train-ready artifacts, so reuse
            # the result across B2-B7 configs that intentionally share lineage.
            snapshot_cache[snapshot_path] = check_training_snapshot(snapshot_path)
        snapshot = snapshot_cache[snapshot_path]
        mismatches = {
            key: {"expected": expected.get(key), "observed": observed.get(key)}
            for key in observed
            if expected.get(key) != observed.get(key)
        }
        if mismatches:
            errors.append(f"baseline_config_mismatch:{baseline_id}={mismatches}")
        if snapshot.get("valid") is not True:
            errors.append(f"snapshot_invalid:{baseline_id}={snapshot.get('errors')}")
        rows.append(
            {
                "baseline_id": baseline_id,
                "config_json": str(path),
                "snapshot_id": snapshot.get("expected_snapshot_id"),
                "snapshot_valid": snapshot.get("valid"),
                "matrix_name": expected.get("name"),
                "observed": observed,
                "matrix_domain_controls": {
                    key: expected.get(key)
                    for key in ["weight_policy", "matched_6frame_report_required"]
                    if key in expected
                },
            }
        )

    result = {
        "schema_version": "classification_v2_q2_baseline_config_audit_v1",
        "matrix_json": str(args.matrix_json),
        "baseline_count": len(rows),
        "errors": errors,
        "valid": not errors,
        "baselines": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Validate the predeclared S6 baseline and ablation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 Q2 ablation matrix.")
    parser.add_argument(
        "--matrix-json",
        type=Path,
        default=Path("configs/classification_v2/q2_ablation_matrix_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_ablation_matrix_audit.json"),
    )
    parser.add_argument(
        "--b4-seed-variance-check-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/b4_seed_variance_check_audit.json"),
    )
    args = parser.parse_args()
    matrix = json.loads(args.matrix_json.read_text(encoding="utf-8"))
    b4_seed_variance = _load_optional_json(args.b4_seed_variance_check_json)
    errors: list[str] = []
    baseline_ids = [row["id"] for row in matrix.get("baselines", [])]
    ablation_ids = [row["id"] for row in matrix.get("ablations", [])]
    if baseline_ids != ["B2", "B3", "B4", "B5", "B6", "B7"]:
        errors.append(f"baseline_ids={baseline_ids}")
    if len(ablation_ids) != len(set(ablation_ids)) or len(ablation_ids) < 10:
        errors.append(f"ablation_ids={ablation_ids}")
    if len(matrix.get("outer_folds", [])) != 5:
        errors.append("outer_fold_count_not_five")
    if len(matrix.get("confirmatory_seeds", [])) < 3:
        errors.append("confirmatory_seed_count_below_three")
    if matrix.get("test_predictions_used_for_model_selection") is not False:
        errors.append("outer_test_used_for_selection")
    if matrix.get("promotion_rules", {}).get("outer_test_threshold_tuning_allowed") is not False:
        errors.append("outer_test_threshold_tuning_allowed")
    required_removals = {
        "bbox_xywh_n",
        "motion_delta",
        "roi_class_relation",
        "social_relation",
        "actor_image",
        "visual_context",
        "interaction_context",
        "auxiliary_heads",
        "source_class_balanced_weighting",
    }
    observed_removals = {item for row in matrix.get("ablations", []) for item in row.get("remove", [])}
    missing_removals = sorted(required_removals.difference(observed_removals))
    if missing_removals:
        errors.append(f"missing_ablation_factors={missing_removals}")
    threshold_freeze_status = _threshold_freeze_status(
        b4_seed_variance,
        expected_seed_count=len(matrix.get("confirmatory_seeds", [])),
    )
    if threshold_freeze_status.startswith("pending"):
        errors.append(f"threshold_freeze_status={threshold_freeze_status}")
    result = {
        "schema_version": "classification_v2_q2_ablation_matrix_audit_v1",
        "baseline_count": len(baseline_ids),
        "ablation_count": len(ablation_ids),
        "outer_fold_count": len(matrix.get("outer_folds", [])),
        "confirmatory_seed_count": len(matrix.get("confirmatory_seeds", [])),
        "threshold_freeze_status": threshold_freeze_status,
        "b4_seed_variance_check_json": str(args.b4_seed_variance_check_json),
        "b4_seed_count": b4_seed_variance.get("seed_count"),
        "b4_seed_variance_valid": b4_seed_variance.get("valid"),
        "outer_test_execution_allowed": False,
        "claim_boundary": matrix.get("claim_boundary"),
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _load_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"valid": False, "errors": [f"missing_json={path}"]}
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_freeze_status(
    b4_seed_variance: dict[str, object],
    *,
    expected_seed_count: int,
) -> str:
    """Derive threshold-freeze readiness from the B4 seed variance audit."""

    if b4_seed_variance.get("valid") is not True:
        return "pending_B4_inner_validation_seed_variance"
    seed_count = int(b4_seed_variance.get("seed_count") or 0)
    if seed_count < expected_seed_count:
        return "pending_B4_inner_validation_seed_count"
    return "frozen_from_B4_inner_validation_seed_variance"


if __name__ == "__main__":
    main()

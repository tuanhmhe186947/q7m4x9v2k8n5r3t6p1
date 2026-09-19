"""Evaluate combined C6 fusion against its same-run actor-only reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.training.legacy_development_c6_modality_matrix import (
    COMBINED_ALL7_FAMILY,
    LINEAGE_SCOPE,
    _paired_prediction_comparison,
    load_c6_matrix_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

SCHEMA = "classification_v2.legacy_c6_actor_reference_comparison.v1"
CANDIDATE_MODE = "combined_all7__real"
BASELINE_MODE = "actor_only"


def evaluate_actor_reference(
    config_path: Path,
    *,
    repeat_id: str,
) -> dict[str, Any]:
    """Build one hash-audited, explicitly unmatched-width comparison."""

    config = load_c6_matrix_config(config_path)
    family = config.payload["experiment_contract"][
        "changed_scientific_family"
    ]
    if family != COMBINED_ALL7_FAMILY:
        raise ValueError(f"actor reference requires combined family={family}")
    runs = config.output_root / "runs" / repeat_id
    candidate_run_path = runs / CANDIDATE_MODE / "run.json"
    baseline_run_path = runs / BASELINE_MODE / "run.json"
    candidate = _read_json(candidate_run_path)
    baseline = _read_json(baseline_run_path)
    errors = _run_pair_errors(candidate, baseline, config.sha256, repeat_id)
    candidate_predictions = runs / CANDIDATE_MODE / "native_predictions.csv"
    baseline_predictions = runs / BASELINE_MODE / "native_predictions.csv"
    if not candidate_predictions.is_file():
        errors.append("missing_candidate_predictions")
    if not baseline_predictions.is_file():
        errors.append("missing_baseline_predictions")
    if errors:
        return _failure_payload(config.sha256, repeat_id, errors)
    comparison = _paired_prediction_comparison(
        pd.read_csv(candidate_predictions),
        pd.read_csv(baseline_predictions),
        iterations=int(config.payload["evaluation"]["bootstrap_draws"]),
        seed=int(config.payload["evaluation"]["bootstrap_seed"]),
    )
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "lineage_scope": LINEAGE_SCOPE,
        "config_sha256": config.sha256,
        "repeat_id": repeat_id,
        "candidate_mode": CANDIDATE_MODE,
        "baseline_mode": BASELINE_MODE,
        "candidate_parameter_count": int(candidate["parameter_count"]),
        "baseline_parameter_count": int(baseline["parameter_count"]),
        "parameter_matched": False,
        "interpretation": "same_run_unmatched_width_actor_reference",
        "selection_sha256": candidate["selection_sha256"],
        "cache_manifest_sha256": candidate["cache_manifest_sha256"],
        "candidate_run_sha256": file_sha256(candidate_run_path),
        "baseline_run_sha256": file_sha256(baseline_run_path),
        "candidate_predictions_sha256": file_sha256(candidate_predictions),
        "baseline_predictions_sha256": file_sha256(baseline_predictions),
        "comparison": comparison,
        "outer_predictions_used_for_model_selection": False,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "errors": [],
        "valid": True,
    }


def _run_pair_errors(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    config_sha256: str,
    repeat_id: str,
) -> list[str]:
    errors: list[str] = []
    expected_common = {
        "config_sha256": config_sha256,
        "repeat_id": repeat_id,
        "lineage_scope": LINEAGE_SCOPE,
        "status": "completed",
        "valid": True,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
    }
    for name, packet in (("candidate", candidate), ("baseline", baseline)):
        for field, expected in expected_common.items():
            if packet.get(field) != expected:
                errors.append(f"{name}_{field}_drift")
    if candidate.get("mode_id") != CANDIDATE_MODE:
        errors.append("candidate_mode_drift")
    if baseline.get("mode_id") != BASELINE_MODE:
        errors.append("baseline_mode_drift")
    for field in ("selection_sha256", "cache_manifest_sha256"):
        if candidate.get(field) != baseline.get(field):
            errors.append(f"pair_{field}_drift")
    if candidate.get("parameter_count") == baseline.get("parameter_count"):
        errors.append("actor_reference_unexpectedly_parameter_matched")
    return errors


def _failure_payload(
    config_sha256: str,
    repeat_id: str,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "FAIL",
        "lineage_scope": LINEAGE_SCOPE,
        "config_sha256": config_sha256,
        "repeat_id": repeat_id,
        "errors": errors,
        "valid": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required={path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repeat-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_actor_reference(
        args.config,
        repeat_id=args.repeat_id,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a grouped, weighted, development-only post-review selector."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from pig_behavior.classification_v2.review.post_review_learning import (
    assert_not_active_behavior_ledger_path,
    sha256_file,
    validate_review_close_authority,
    write_json,
)
from pig_behavior.classification_v2.review.post_review_selector_candidate import (
    SELECTOR_SCHEMA_VERSION,
    SELECTOR_STATUS,
    SelectorCandidateConfig,
    aggregate_masked_selector_features,
    build_selector_outcomes,
    run_post_review_selector_candidate,
    selector_feature_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one fixed post-review evidence-disagreement selector. "
            "Outputs are development diagnostics only."
        )
    )
    parser.add_argument("--review-close-authority-json", type=Path, required=True)
    parser.add_argument("--primary-scope-csv", type=Path, required=True)
    parser.add_argument("--primary-quality-csv", type=Path, required=True)
    parser.add_argument("--control-scope-csv", type=Path, required=True)
    parser.add_argument("--control-quality-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = {
        "review_close_authority": args.review_close_authority_json,
        "primary_scope": args.primary_scope_csv,
        "primary_quality": args.primary_quality_csv,
        "control_scope": args.control_scope_csv,
        "control_quality": args.control_quality_csv,
        "frame_features": args.frame_features_csv,
    }
    for path in [*input_paths.values(), args.output_dir]:
        assert_not_active_behavior_ledger_path(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("selector_output_dir_not_empty")

    authority = json.loads(
        args.review_close_authority_json.read_text(encoding="utf-8")
    )
    validate_review_close_authority(authority)
    _validate_frozen_review_hashes(authority, input_paths)
    primary_scope = pd.read_csv(args.primary_scope_csv, low_memory=False)
    primary_quality = pd.read_csv(args.primary_quality_csv, low_memory=False)
    control_scope = pd.read_csv(args.control_scope_csv, low_memory=False)
    control_quality = pd.read_csv(args.control_quality_csv, low_memory=False)
    frame_features = pd.read_csv(args.frame_features_csv, low_memory=False)

    outcomes, outcome_audit = build_selector_outcomes(
        review_close_authority=authority,
        primary_scope=primary_scope,
        primary_quality=primary_quality,
        control_scope=control_scope,
        control_quality=control_quality,
    )
    aggregates, aggregate_audit = aggregate_masked_selector_features(
        frame_features,
        temporal_unit_keys=outcomes["temporal_unit_key"].tolist(),
    )
    config = SelectorCandidateConfig(
        seed=args.seed,
        fold_count=args.fold_count,
        regularization_c=args.regularization_c,
    )
    result = run_post_review_selector_candidate(
        outcomes=outcomes,
        aggregates=aggregates,
        config=config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "feature_contract": args.output_dir / "selector_feature_contract.json",
        "outcome_audit": args.output_dir / "selector_outcome_audit.json",
        "aggregate_audit": (
            args.output_dir / "selector_feature_aggregation_audit.json"
        ),
        "outcomes": args.output_dir / "selector_development_outcomes.csv",
        "model_x": args.output_dir / "selector_aggregate_model_x.csv",
        "fold_manifest": args.output_dir / "selector_fold_manifest.csv",
        "oof_predictions": args.output_dir / "selector_oof_predictions.csv",
        "metrics": args.output_dir / "selector_metrics.json",
        "coefficients": args.output_dir / "selector_coefficients.csv",
        "formula": args.output_dir / "selector_formula.json",
        "leakage_audit": args.output_dir / "selector_leakage_audit.json",
        "manifest": args.output_dir / "selector_candidate_manifest.json",
        "exact_command": args.output_dir / "exact_command.txt",
    }
    write_json(output_paths["feature_contract"], selector_feature_contract())
    write_json(output_paths["outcome_audit"], outcome_audit)
    write_json(output_paths["aggregate_audit"], aggregate_audit)
    outcomes.to_csv(output_paths["outcomes"], index=False)
    aggregates.reset_index().to_csv(output_paths["model_x"], index=False)
    result["fold_manifest"].to_csv(output_paths["fold_manifest"], index=False)
    result["predictions"].to_csv(output_paths["oof_predictions"], index=False)
    write_json(output_paths["metrics"], result["metrics"])
    result["coefficients"].to_csv(output_paths["coefficients"], index=False)
    write_json(output_paths["formula"], result["formula"])
    write_json(output_paths["leakage_audit"], result["leakage_audit"])
    output_paths["exact_command"].write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]),
        encoding="utf-8",
    )

    code_state = _git_state()
    manifest = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "status": SELECTOR_STATUS,
        "scientific_authority": "NONE",
        "code": code_state,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "config": {
            "seed": config.seed,
            "fold_count": config.fold_count,
            "regularization_c": config.regularization_c,
            "max_iter": config.max_iter,
            "group_boundary": "recording_date_with_nested_video_audit",
            "model_family": "weighted_multinomial_logistic_evidence_model",
            "selector_formula": "1-P(reviewed=original|masked_spatiotemporal_X)",
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "feature_contract_hash": selector_feature_contract()["contract_hash"],
        "fold_audits": result["fold_audits"],
        "control_consumed_for_development": True,
        "control_validation_authority_for_candidate": False,
        "fresh_probability_holdout_required": True,
        "active_review_manifest_changed": False,
        "automatic_label_change_authorized": False,
        "automatic_review_selection_change_authorized": False,
        "model_x_contains_review_fields": False,
        "model_x_contains_source_label": False,
        "model_x_contains_source_provenance": False,
    }
    write_json(output_paths["manifest"], manifest)
    inventory_entries = []
    for name, path in sorted(output_paths.items()):
        inventory_entries.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    write_json(
        args.output_dir / "artifact_inventory.json",
        {
            "schema_version": (
                "classification_v2.post_review_selector_inventory.v1"
            ),
            "status": SELECTOR_STATUS,
            "artifacts": inventory_entries,
        },
    )
    print("PASS: post-review selector candidate built")
    print(args.output_dir)


def _validate_frozen_review_hashes(
    authority: dict[str, object],
    input_paths: dict[str, Path],
) -> None:
    artifacts = authority["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("review_close_artifacts_invalid")
    for name in ("primary_scope", "primary_quality", "control_scope", "control_quality"):
        binding = artifacts.get(name)
        if not isinstance(binding, dict):
            raise ValueError(f"review_close_binding_missing={name}")
        if sha256_file(input_paths[name]) != str(binding.get("sha256", "")):
            raise ValueError(f"review_close_input_hash_drift={name}")


def _git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"head_sha": head, "tree_hash": tree, "worktree_dirty": bool(status)}


if __name__ == "__main__":
    main()

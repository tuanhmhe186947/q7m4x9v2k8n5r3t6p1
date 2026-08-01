"""Analyse frozen primary/control review outcomes and named features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.post_review_learning import (
    analyze_post_review_learning,
    assert_not_active_behavior_ledger_path,
    bindings_from_paths,
    validate_review_close_authority,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-close-authority-json", type=Path, required=True)
    parser.add_argument("--primary-scope-csv", type=Path, required=True)
    parser.add_argument("--primary-quality-csv", type=Path, required=True)
    parser.add_argument("--control-scope-csv", type=Path, required=True)
    parser.add_argument("--control-quality-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument(
        "--feature-column",
        action="append",
        required=True,
        dest="feature_columns",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.review_close_authority_json,
        args.primary_scope_csv,
        args.primary_quality_csv,
        args.control_scope_csv,
        args.control_quality_csv,
        args.frame_features_csv,
        args.output_dir,
    ):
        assert_not_active_behavior_ledger_path(path)
    authority = json.loads(
        args.review_close_authority_json.read_text(encoding="utf-8")
    )
    validate_review_close_authority(authority)

    input_bindings = bindings_from_paths(
        {
            "primary_scope": args.primary_scope_csv,
            "primary_quality": args.primary_quality_csv,
            "control_scope": args.control_scope_csv,
            "control_quality": args.control_quality_csv,
            "frame_features": args.frame_features_csv,
        }
    )
    expected = {
        name: input_bindings[name]
        for name in (
            "primary_scope",
            "primary_quality",
            "control_scope",
            "control_quality",
        )
    }
    validate_review_close_authority(authority, expected_bindings=expected)
    result = analyze_post_review_learning(
        review_close_authority=authority,
        primary_scope=pd.read_csv(args.primary_scope_csv, low_memory=False),
        primary_quality=pd.read_csv(args.primary_quality_csv, low_memory=False),
        control_scope=pd.read_csv(args.control_scope_csv, low_memory=False),
        control_quality=pd.read_csv(args.control_quality_csv, low_memory=False),
        frame_features=pd.read_csv(args.frame_features_csv, low_memory=False),
        feature_columns=args.feature_columns,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "outcomes": args.output_dir / "post_review_outcomes.csv",
        "transition_matrix": (
            args.output_dir / "original_to_reviewed_transition_matrix.csv"
        ),
        "stratified_outcomes": (
            args.output_dir / "post_review_stratified_outcomes.csv"
        ),
        "feature_contrasts": (
            args.output_dir / "post_review_feature_contrasts.csv"
        ),
    }
    for name, path in output_paths.items():
        result[name].to_csv(path, index=False)
    summary = dict(result["summary"])
    summary["review_close_authority"] = {
        "path": str(args.review_close_authority_json.resolve()),
        "sha256": bindings_from_paths(
            {"authority": args.review_close_authority_json}
        )["authority"]["sha256"],
    }
    write_json(args.output_dir / "post_review_learning_summary.json", summary)
    output_bindings = bindings_from_paths(output_paths)
    write_json(
        args.output_dir / "post_review_learning_artifact_inventory.json",
        {
            "schema_version": (
                "classification_v2.post_review_learning_inventory.v1"
            ),
            "inputs": input_bindings,
            "outputs": output_bindings,
            "review_fields_entering_model_x": 0,
            "automatic_training_or_apply_performed": False,
        },
    )
    print("PASS: post-review diagnostic artifacts written")
    print(args.output_dir)


if __name__ == "__main__":
    main()

"""Run grouped source-shortcut diagnostics on whitelisted clean features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.domain_controls import (
    audit_domain_feature_shift,
    grouped_source_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grouped classification_v2 domain controls.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument(
        "--grouped-roles",
        type=Path,
        default=Path("outputs/classification_v2/q2_grouped_folds/q2_outer_inner_roles.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/domain_controls"),
    )
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()
    features = pd.read_csv(args.root / "X_window_features.csv", low_memory=False)
    metadata = pd.read_csv(args.root / "split_manifest.csv", low_memory=False)
    predictions, audit = grouped_source_probe(
        features,
        metadata,
        pd.read_csv(args.root / "event_weight_manifest.csv", low_memory=False),
        pd.read_csv(args.grouped_roles, low_memory=False),
        max_iter=args.max_iter,
    )
    shift_audit = audit_domain_feature_shift(features, metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "grouped_source_probe_predictions.csv", index=False)
    payload = {
        "prediction_csv": str(args.output_dir / "grouped_source_probe_predictions.csv"),
        **audit,
    }
    (args.output_dir / "grouped_source_probe_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "domain_feature_shift_audit.json").write_text(
        json.dumps(shift_audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

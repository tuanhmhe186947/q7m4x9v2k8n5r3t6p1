"""Check source-aware weighting policies remain train-only and non-duplicating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.training.samplers import WEIGHT_POLICIES, build_training_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 domain control contracts.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/source_matched_views/check_domain_controls.json"),
    )
    args = parser.parse_args()
    training = pd.DataFrame(
        {
            "window_id": [f"w{index}" for index in range(8)],
            "behavior_true": ["fight", "fight", "stand", "stand", "stand", "eat", "eat", "eat"],
            "source_type": ["cvat_tracking_xml"] * 5 + ["legacy_recovered"] * 3,
            "temporal_unit_key": ["u0", "u0", "u1", "u2", "u2", "u3", "u3", "u3"],
        }
    )
    audits = {}
    errors: list[str] = []
    for policy in WEIGHT_POLICIES:
        weights, audit = build_training_weights(training, policy=policy)
        audits[policy] = audit
        if len(weights) != len(training) or not weights.index.equals(training.index):
            errors.append(f"weight_alignment_failed={policy}")
        if abs(float(weights.mean()) - 1.0) > 1e-12:
            errors.append(f"weight_mean_not_one={policy}")
        if audit["row_duplication_used"]:
            errors.append(f"row_duplication_used={policy}")
    result = {
        "schema_version": "classification_v2_domain_controls_check_v1",
        "policies": audits,
        "all_policies_training_fold_only": all(
            audit["fit_scope"] == "training_fold_only" for audit in audits.values()
        ),
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

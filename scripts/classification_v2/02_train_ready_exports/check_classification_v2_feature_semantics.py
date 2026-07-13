from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.feature_semantics import (
    audit_feature_semantics,
)
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit classification_v2 feature semantics and leakage status."
        )
    )
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/feature_semantics_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/"
            "feature_semantics_audit.json"
        ),
    )
    parser.add_argument(
        "--tabular-x-csv",
        type=Path,
        default=None,
        help="Optional bounded/smoke tabular X artifact override.",
    )
    parser.add_argument(
        "--spatial-npz",
        type=Path,
        default=None,
        help="Optional bounded/smoke spatial X artifact override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing derived feature-semantics audit explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_output_paths_available(
        [args.output_json],
        overwrite=args.overwrite,
    )
    result = audit_feature_semantics(
        args.contract_json,
        tabular_x_csv=args.tabular_x_csv,
        spatial_npz=args.spatial_npz,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_keys = [
        "valid",
        "tabular_feature_count",
        "tabular_expected_feature_count",
        "tabular_contract_match",
        "tabular_features_missing_from_x",
        "unexpected_tabular_x_features",
        "tabular_family_counts",
        "declared_spatial_array_count",
        "undeclared_spatial_arrays",
        "spatial_model_input_array_count",
        "spatial_non_model_arrays",
        "spatial_model_input_role_errors",
        "roi_context",
        "errors",
        "warnings",
    ]
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

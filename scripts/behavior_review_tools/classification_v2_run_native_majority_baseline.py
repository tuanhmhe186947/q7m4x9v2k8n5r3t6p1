from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.experiments.native_majority_baseline import (
    NativeMajorityBaselineConfig,
    run_native_majority_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a no-training native temporal majority baseline for classification_v2."
    )
    parser.add_argument(
        "--native-manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units/native_temporal_unit_manifest.csv"),
    )
    parser.add_argument(
        "--native-oof-fold-manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units_oof_folds/native_oof_fold_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/native_majority_baseline"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    args = parser.parse_args()

    result = run_native_majority_baseline(
        NativeMajorityBaselineConfig(
            native_manifest_csv=args.native_manifest_csv,
            native_oof_fold_manifest_csv=args.native_oof_fold_manifest_csv,
            output_dir=args.output_dir,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    )
    print(json.dumps({key: value for key, value in result.items() if key != "audit"}, indent=2))
    if result["audit"]["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

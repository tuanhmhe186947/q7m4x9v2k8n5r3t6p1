from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.ablation_reporting import (
    AblationReportingConfig,
    build_ablation_reporting_audit,
)


def main() -> None:
    """Build and validate the Q2 ablation and shortcut-aware reporting audit."""

    parser = argparse.ArgumentParser(description="Check classification_v2 ablation reporting evidence.")
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/ablation_shortcut_contract_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/ablation_reporting_audit.json"),
    )
    args = parser.parse_args()
    audit = build_ablation_reporting_audit(
        AblationReportingConfig(contract_json=args.contract_json, output_json=args.output_json)
    )
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

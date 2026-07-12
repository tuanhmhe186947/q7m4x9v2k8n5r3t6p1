from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.ablation_shortcut_contract import (
    check_ablation_shortcut_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 ablation and shortcut-control contract.")
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/ablation_shortcut_contract_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/ablation_shortcut_contract_audit.json"),
    )
    args = parser.parse_args()

    result = check_ablation_shortcut_contract(args.contract_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

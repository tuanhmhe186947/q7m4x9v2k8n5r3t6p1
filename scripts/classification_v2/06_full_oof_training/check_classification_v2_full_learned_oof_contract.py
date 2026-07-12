from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.full_learned_oof_contract import (
    FullLearnedOofContractConfig,
    check_full_learned_oof_contract,
)


def main() -> None:
    """Validate readiness for full learned native-OOF multimodal evaluation."""

    parser = argparse.ArgumentParser(description="Check classification_v2 full learned OOF contract readiness.")
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/full_learned_oof_contract_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_learned_oof_contract_audit.json"),
    )
    args = parser.parse_args()
    audit = check_full_learned_oof_contract(
        FullLearnedOofContractConfig(contract_json=args.contract_json, output_json=args.output_json)
    )
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

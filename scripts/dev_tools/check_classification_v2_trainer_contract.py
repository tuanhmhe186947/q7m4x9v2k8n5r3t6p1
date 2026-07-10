from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.trainer_contract import check_trainer_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 trainer input contract.")
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/trainer_contract_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/trainer_contract_audit.json"),
    )
    args = parser.parse_args()

    audit = check_trainer_contract(args.contract_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

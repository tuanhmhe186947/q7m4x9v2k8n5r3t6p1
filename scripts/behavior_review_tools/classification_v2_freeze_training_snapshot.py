from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.training_snapshot import freeze_training_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze an immutable classification_v2 train-ready artifact snapshot."
    )
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/data_contract_v1.json"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = freeze_training_snapshot(args.contract_json, output_path=args.output_json)
    print(json.dumps({k: snapshot[k] for k in ["snapshot_id", "snapshot_path", "errors"]}, indent=2))
    if snapshot["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

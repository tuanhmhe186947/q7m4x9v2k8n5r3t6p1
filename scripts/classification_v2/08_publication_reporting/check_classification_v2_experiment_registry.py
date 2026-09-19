from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.experiments.record_contract import check_experiment_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 experiment registry record.")
    parser.add_argument(
        "--record-json",
        type=Path,
        default=Path("outputs/classification_v2/experiment_registry/spatial_tcn_smoke_train_record.json"),
    )
    args = parser.parse_args()
    result = check_experiment_record(args.record_json)
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

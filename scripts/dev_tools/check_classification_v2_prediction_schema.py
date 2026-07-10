from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.prediction_schema_contract import check_prediction_schema_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 model/baseline prediction schema.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/prediction_schema_audit.json"),
    )
    args = parser.parse_args()

    result = check_prediction_schema_csv(args.predictions_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

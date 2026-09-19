"""Check Pig-STRENet review artifacts before behavior-review selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.pig_strenet_artifact_run import (
    audit_pig_strenet_artifact_run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-run-scope",
        choices=("smoke", "full"),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allow-missing-roi-visual", action="store_true")
    args = parser.parse_args()

    audit = audit_pig_strenet_artifact_run(
        args.artifact_dir,
        input_csv=args.input_csv,
        expected_run_scope=args.expected_run_scope,
        require_roi_visual=not args.allow_missing_roi_visual,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit("FAIL: Pig-STRENet artifact contract")
    print("PASS: Pig-STRENet artifact contract")


if __name__ == "__main__":
    main()

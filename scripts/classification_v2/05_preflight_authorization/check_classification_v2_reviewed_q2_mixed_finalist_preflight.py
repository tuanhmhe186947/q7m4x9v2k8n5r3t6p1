"""Run the read-only mixed-reviewed SF128/A128 short-gate preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.reviewed_q2_mixed_finalist_preflight import (
    build_reviewed_q2_mixed_finalist_preflight,
    write_reviewed_q2_mixed_finalist_preflight,
)


def main() -> None:
    """Build one paired-finalist audit; never start training or full OOF."""

    parser = argparse.ArgumentParser(
        description=(
            "Check the reviewed mixed-data SF128/A128 contract, source "
            "support, shortcut audit, and inference inputs."
        )
    )
    parser.add_argument("--data-contract-json", type=Path, required=True)
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--comparison-contract-json", type=Path, required=True)
    parser.add_argument("--handoff-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing audit only when explicitly requested.",
    )
    args = parser.parse_args()

    result = build_reviewed_q2_mixed_finalist_preflight(
        args.data_contract_json,
        args.snapshot_json,
        args.comparison_contract_json,
        args.handoff_json,
        project_root=args.project_root,
        output_json=args.output_json,
    )
    persisted = write_reviewed_q2_mixed_finalist_preflight(
        result,
        data_contract_json=args.data_contract_json,
        output_json=args.output_json,
        project_root=args.project_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(persisted, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

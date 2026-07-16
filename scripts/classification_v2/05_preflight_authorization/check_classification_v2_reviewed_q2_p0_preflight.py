"""Run the read-only reviewed-Q2 P0 preflight in an explicit agent root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.reviewed_q2_p0_preflight import (
    build_reviewed_q2_p0_preflight,
    write_reviewed_q2_p0_preflight,
)


def main() -> None:
    """Build and persist one P0 result; never start training or review GUI."""

    parser = argparse.ArgumentParser(
        description=(
            "Check reviewed-Q2 contract, review evidence, snapshot, and "
            "leakage-safe model inputs without training."
        )
    )
    parser.add_argument("--data-contract-json", type=Path, required=True)
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing P0 audit only when explicitly requested.",
    )
    args = parser.parse_args()

    result = build_reviewed_q2_p0_preflight(
        args.data_contract_json,
        args.snapshot_json,
        project_root=args.project_root,
        output_json=args.output_json,
    )
    persisted = write_reviewed_q2_p0_preflight(
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

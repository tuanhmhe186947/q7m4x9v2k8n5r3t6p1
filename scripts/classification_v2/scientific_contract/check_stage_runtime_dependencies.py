"""Validate complete runtime production-code authority for all stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.runtime_dependencies import (
    audit_all_stage_runtime_dependencies,
)
from pig_behavior.classification_v2.contracts.semantic_lineage import (
    load_code_contract_mapping,
    load_scientific_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_scientific_contract(args.contract)
    stage_ids = [str(stage["stage_id"]) for stage in contract["stages"]]
    mapping_rows = load_code_contract_mapping(args.mapping)
    audit = audit_all_stage_runtime_dependencies(
        args.project_root.resolve(),
        stage_ids,
        mapping_rows,
    )
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

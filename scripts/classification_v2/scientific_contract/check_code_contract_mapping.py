"""Check two-way coverage between contract entities and current source code."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any

from validate_scientific_contract import expand_entities, load_contract


def _symbols(path: Path) -> set[str]:
    if not path.exists() or path.suffix != ".py":
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    found.add(target.id)
    return found


def check_mapping(
    contract_path: Path,
    mapping_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    stages = expand_entities(contract, "stages", "stage_defaults")
    features = expand_entities(contract, "features", "feature_defaults")
    invariants = expand_entities(
        contract,
        "invariants",
        "invariant_defaults",
    )
    authoritative_ids = {
        *(stage["stage_id"] for stage in stages),
        *(feature["feature_id"] for feature in features),
        *(invariant["invariant_id"] for invariant in invariants),
    }
    with mapping_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapped_ids = {
        row["contract_item_id"]
        for row in rows
        if row.get("contract_item_id")
    }
    errors: list[str] = []
    missing_contract_items = sorted(authoritative_ids - mapped_ids)
    unexpected_contract_items = sorted(mapped_ids - authoritative_ids)
    if missing_contract_items:
        errors.append(f"contract_items_without_mapping={missing_contract_items}")
    if unexpected_contract_items:
        errors.append(
            f"mapping_items_without_contract={unexpected_contract_items}"
        )

    symbol_cache: dict[Path, set[str]] = {}
    for row in rows:
        source = row.get("source_file", "")
        symbol = row.get("symbol", "")
        status = row.get("current_implementation_status", "")
        if not source:
            if status not in {
                "DECLARED_NOT_IMPLEMENTED",
                "UNKNOWN_REQUIRES_REVIEW",
                "REVIEW_ONLY",
                "MODEL_FORBIDDEN",
            }:
                errors.append(
                    f"{row['contract_item_id']}:implemented_without_source"
                )
            continue
        path = project_root / source
        if not path.exists():
            errors.append(f"{row['contract_item_id']}:missing_source={source}")
            continue
        if symbol:
            symbols = symbol_cache.setdefault(path, _symbols(path))
            if symbol not in symbols:
                errors.append(
                    f"{row['contract_item_id']}:missing_symbol="
                    f"{source}#{symbol}"
                )
        test_file = row.get("test_file", "")
        if test_file and not (project_root / test_file).exists():
            errors.append(
                f"{row['contract_item_id']}:missing_test_file={test_file}"
            )

    inventory_errors: list[str] = []
    for item in contract["implementation_inventory"]:
        path = project_root / item["source_file"]
        if not path.exists():
            inventory_errors.append(
                f"{item['implementation_id']}:missing_source"
            )
            continue
        symbol = item["symbol"]
        if symbol and symbol not in symbol_cache.setdefault(path, _symbols(path)):
            inventory_errors.append(
                f"{item['implementation_id']}:missing_symbol={symbol}"
            )
        if not item["contract_ids"]:
            inventory_errors.append(
                f"{item['implementation_id']}:implementation_without_contract"
            )
    errors.extend(inventory_errors)
    return {
        "valid": not errors,
        "errors": errors,
        "contract_item_count": len(authoritative_ids),
        "mapped_contract_item_count": len(mapped_ids & authoritative_ids),
        "implementation_inventory_count": len(
            contract["implementation_inventory"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "docs/classification_v2/scientific_contract_v1/"
            "00_pipeline_contract.yaml"
        ),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(
            "docs/classification_v2/scientific_contract_v1/"
            "10_code_contract_mapping.csv"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check_mapping(
        args.contract,
        args.mapping,
        args.project_root.resolve(),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["valid"]:
        print("PASS code-contract mapping")
    else:
        print("FAIL code-contract mapping")
        for error in result["errors"]:
            print(f"- {error}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Audit loader inputs from one hash-bound model-input manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.loader_input_audit import (
    audit_loader_input_contract,
    write_loader_input_audit,
)


def main() -> None:
    """Audit only paths bound by one generated model-input manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-input-contract-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        audit = audit_loader_input_contract(
            model_input_contract_json=args.model_input_contract_json,
            project_root=args.project_root,
        )
        result = write_loader_input_audit(
            audit,
            output_path=args.output_json,
            project_root=args.project_root,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "valid": False,
                    "errors": [str(exc)],
                    "artifact_written": False,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

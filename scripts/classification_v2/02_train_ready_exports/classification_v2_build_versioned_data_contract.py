"""Build a run-bound classification_v2 contract from explicit artifact paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    VersionedDataContractError,
    build_versioned_data_contract,
    write_versioned_data_contract,
)


def parse_args() -> argparse.Namespace:
    """Require all lineage inputs and outputs explicitly on the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a versioned classification_v2 data contract without "
            "canonical output fallbacks."
        )
    )
    parser.add_argument("--template-json", type=Path, required=True)
    parser.add_argument("--artifact-map-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate in memory first and write only an explicit, valid destination."""

    args = parse_args()
    try:
        build = build_versioned_data_contract(
            args.template_json,
            args.artifact_map_json,
            output_path=args.output_json,
            project_root=args.project_root,
        )
        audit = write_versioned_data_contract(
            build,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, VersionedDataContractError) as exc:
        errors = list(getattr(exc, "errors", (str(exc),)))
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "valid": False,
                    "errors": errors,
                    "artifact_written": False,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        raise SystemExit(2) from exc
    print(json.dumps(audit, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()

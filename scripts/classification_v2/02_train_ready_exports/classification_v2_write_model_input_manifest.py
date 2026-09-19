"""Write a model-input manifest from one generated versioned data contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.model_input_manifest import (
    ModelInputManifestError,
    build_model_input_manifest,
    write_model_input_manifest,
)


def parse_args() -> argparse.Namespace:
    """Require explicit contract and output paths; no canonical defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-contract-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate all bindings before writing the agent-owned manifest."""

    args = parse_args()
    try:
        build = build_model_input_manifest(
            args.data_contract_json,
            output_path=args.output_json,
            project_root=args.project_root,
        )
        audit = write_model_input_manifest(
            build,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, ModelInputManifestError) as exc:
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
    print(json.dumps(audit, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

"""Audit one classification_v2 run packet without modifying its artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.training.run_lineage_audit import (
    audit_run_lineage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check run manifests, hashes, and prediction linkage."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--registry-csv",
        type=Path,
        help="Override the run's declared registry after remote fold merge.",
    )
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument(
        "--deep-input-hashes",
        action="store_true",
        help="Rehash large frozen inputs instead of checking path/size only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_dir.is_dir():
        raise FileNotFoundError(args.run_dir)
    if not args.dry_run:
        require_output_paths_available(
            [args.output_json],
            overwrite=args.overwrite,
        )
    audit = {
        **audit_run_lineage(
            args.run_dir,
            deep_input_hashes=args.deep_input_hashes,
            registry_csv=args.registry_csv,
        ),
        "dry_run": bool(args.dry_run),
        "require_success": bool(args.require_success),
        "output_json": str(args.output_json),
    }
    if not args.dry_run:
        _write_json_atomic(args.output_json, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    failed = not audit["integrity_valid"] or (
        args.require_success and not audit["run_succeeded"]
    )
    if failed:
        raise SystemExit(2)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()

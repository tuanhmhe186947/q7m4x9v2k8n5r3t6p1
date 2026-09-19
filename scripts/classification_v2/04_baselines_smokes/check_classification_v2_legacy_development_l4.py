"""Run bounded L4 correctness gates for legacy-only unreviewed development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l4 import (
    load_legacy_l4_config,
    run_legacy_l4_fold_epoch,
    run_legacy_l4_short,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the committed legacy L4 model-correctness ladder."
    )
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--mode", choices=("short", "fold_epoch"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--short-audit-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists; pass --overwrite explicitly: {args.output_json}"
        )
    config = load_legacy_l4_config(args.config_json)
    if args.mode == "short":
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required for short mode")
        if args.checkpoint_path.exists() and not args.overwrite:
            raise FileExistsError(
                "checkpoint exists; pass --overwrite explicitly: "
                f"{args.checkpoint_path}"
            )
        result = run_legacy_l4_short(
            config,
            checkpoint_path=args.checkpoint_path,
        )
    else:
        if args.short_audit_json is None:
            raise ValueError("--short-audit-json is required for fold_epoch mode")
        result = run_legacy_l4_fold_epoch(
            config,
            short_audit_path=args.short_audit_json,
        )
    _write_json_atomic(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
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

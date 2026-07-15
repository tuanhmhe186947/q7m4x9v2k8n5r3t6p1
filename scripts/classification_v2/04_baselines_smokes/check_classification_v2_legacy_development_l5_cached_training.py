"""Preflight, run and repeat-gate the crash-bounded legacy L5 short head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l5_cached_training import (
    audit_legacy_l5_cached_training_repeat_gate,
    load_legacy_l5_cached_training_config,
    preflight_legacy_l5_cached_short_training,
    run_legacy_l5_cached_short_training,
    write_legacy_l5_cached_training_repeat_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the legacy L5 cached-feature short-training gate."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight",
        help="Validate real parents and CPU batch bounds without CUDA.",
    )
    _add_config_argument(preflight)
    run = subparsers.add_parser(
        "run",
        help="Execute exactly one fresh-process CUDA short run.",
    )
    _add_config_argument(run)
    run.add_argument("--run-id", required=True)
    repeat = subparsers.add_parser(
        "repeat-gate",
        help="Compare two separate completed runs and write the short gate.",
    )
    _add_config_argument(repeat)
    repeat.add_argument("--primary-result-json", type=Path, required=True)
    repeat.add_argument("--repeat-result-json", type=Path, required=True)
    args = parser.parse_args()

    config = load_legacy_l5_cached_training_config(args.config_json)
    if args.command == "preflight":
        result = preflight_legacy_l5_cached_short_training(config)
    elif args.command == "run":
        result = run_legacy_l5_cached_short_training(
            config,
            run_id=args.run_id,
        )
    else:
        audit = audit_legacy_l5_cached_training_repeat_gate(
            config,
            primary_result_path=args.primary_result_json,
            repeat_result_path=args.repeat_result_json,
        )
        if not audit["valid"]:
            print(json.dumps(audit, indent=2, ensure_ascii=True))
            raise SystemExit(1)
        output_path, result = write_legacy_l5_cached_training_repeat_gate(
            config,
            primary_result_path=args.primary_result_json,
            repeat_result_path=args.repeat_result_json,
        )
        result = {**result, "output_path": str(output_path.resolve())}
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-json", type=Path, required=True)


if __name__ == "__main__":
    main()

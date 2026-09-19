"""Preflight, run, and audit the immutable legacy L5 temporal ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    CANONICAL_VIEWS,
    load_temporal_ladder_config,
    preflight_temporal_ladder_view,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder_runtime import (
    audit_temporal_ladder_run,
    run_temporal_ladder_view,
    write_temporal_ladder_repeat_gate,
    write_temporal_ladder_short_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run gated legacy L5 V1 temporal-length controls."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_config(preflight)
    _add_view(preflight)
    run = subparsers.add_parser("run")
    _add_config(run)
    _add_view(run)
    run.add_argument("--run-id", required=True)
    audit = subparsers.add_parser("audit-run")
    _add_config(audit)
    audit.add_argument("--result-json", type=Path, required=True)
    repeat = subparsers.add_parser("repeat-gate")
    _add_config(repeat)
    _add_view(repeat)
    repeat.add_argument("--primary-result-json", type=Path, required=True)
    repeat.add_argument("--repeat-result-json", type=Path, required=True)
    matrix = subparsers.add_parser("matrix-gate")
    _add_config(matrix)
    args = parser.parse_args()

    config = load_temporal_ladder_config(args.config_json)
    if args.command == "preflight":
        result = preflight_temporal_ladder_view(config, args.view_id)
    elif args.command == "run":
        result = run_temporal_ladder_view(
            config,
            view_id=args.view_id,
            run_id=args.run_id,
        )
    elif args.command == "audit-run":
        result = audit_temporal_ladder_run(
            config,
            result_path=args.result_json,
        )
    elif args.command == "repeat-gate":
        path, result = write_temporal_ladder_repeat_gate(
            config,
            view_id=args.view_id,
            primary_result_path=args.primary_result_json,
            repeat_result_path=args.repeat_result_json,
        )
        result = {**result, "output_path": str(path.resolve())}
    else:
        path, result = write_temporal_ladder_short_matrix(config)
        result = {**result, "output_path": str(path.resolve())}
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-json", type=Path, required=True)


def _add_view(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--view-id",
        choices=tuple(CANONICAL_VIEWS),
        required=True,
    )


if __name__ == "__main__":
    main()

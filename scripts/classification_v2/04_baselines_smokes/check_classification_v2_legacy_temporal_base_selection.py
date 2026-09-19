"""Run the controlled legacy Stage A temporal-base screening matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_temporal_base_selection import (
    audit_temporal_base_short_matrix,
    execute_temporal_base_run,
    load_temporal_base_selection_config,
    preflight_temporal_base_selection,
    validate_full_launch_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run-mode")
    actions.add_argument("--audit-short-matrix", action="store_true")
    parser.add_argument("--repeat-id")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_temporal_base_selection_config(args.config)
    full_gate = validate_full_launch_gate(config)
    if args.preflight:
        payload = preflight_temporal_base_selection(config)
        if full_gate is not None:
            payload["full_launch_gate"] = full_gate
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["valid"] else 1
    if args.audit_short_matrix:
        output, payload = audit_temporal_base_short_matrix(config)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": payload["status"],
                    "full_confirmation_authorized": payload[
                        "full_confirmation_authorized"
                    ],
                    "errors": payload["errors"],
                    "valid": payload["valid"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["valid"] else 1
    if not args.repeat_id:
        raise ValueError("--repeat-id is required with --run-mode")
    output, payload = execute_temporal_base_run(
        config,
        str(args.run_mode),
        str(args.repeat_id),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": payload["status"],
                "mode_id": payload["mode_id"],
                "repeat_id": payload["repeat_id"],
                "parameter_count": payload["parameter_count"],
                "optimizer_steps": payload["optimizer_steps"],
                "metrics": payload["metrics"],
                "errors": payload["errors"],
                "valid": payload["valid"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

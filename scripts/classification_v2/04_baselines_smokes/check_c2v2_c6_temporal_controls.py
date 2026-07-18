"""Check or run the fail-closed C6 temporal-control matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_c6_temporal_controls import (
    audit_c6_temporal_short_gate,
    data_c6_temporal_control_preflight,
    execute_c6_temporal_control_run,
    load_c6_temporal_control_config,
    static_c6_temporal_control_preflight,
    synthetic_c6_temporal_control_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--static-preflight", action="store_true")
    actions.add_argument("--synthetic-preflight", action="store_true")
    actions.add_argument("--data-preflight", action="store_true")
    actions.add_argument("--audit-short-gate", action="store_true")
    actions.add_argument("--run-mode")
    parser.add_argument("--repeat-id")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_c6_temporal_control_config(args.config)
    if args.static_preflight:
        payload = static_c6_temporal_control_preflight(config)
    elif args.synthetic_preflight:
        payload = synthetic_c6_temporal_control_preflight(config)
    elif args.data_preflight:
        _, payload = data_c6_temporal_control_preflight(config)
    elif args.audit_short_gate:
        output, payload = audit_c6_temporal_short_gate(config)
        payload = {**payload, "output": str(output)}
    else:
        if not args.repeat_id:
            raise ValueError("--repeat-id is required with --run-mode")
        output, payload = execute_c6_temporal_control_run(
            config,
            str(args.run_mode),
            str(args.repeat_id),
        )
        payload = {**payload, "output": str(output)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

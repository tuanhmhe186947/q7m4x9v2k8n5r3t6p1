"""Preflight, run, or audit one legacy L6 ROI-relation control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation import (
    MODES,
    load_roi_relation_training_config,
    preflight_roi_relation_mode,
)
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation_runtime import (
    audit_roi_relation_run,
    run_roi_relation_mode,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--mode", choices=MODES, required=True)
        if name == "run":
            child.add_argument("--run-id", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--result", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_roi_relation_training_config(args.config)
    if args.command == "preflight":
        payload = preflight_roi_relation_mode(config, args.mode)
    elif args.command == "run":
        payload = run_roi_relation_mode(
            config,
            mode=args.mode,
            run_id=args.run_id,
        )
    else:
        payload = audit_roi_relation_run(config, result_path=args.result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

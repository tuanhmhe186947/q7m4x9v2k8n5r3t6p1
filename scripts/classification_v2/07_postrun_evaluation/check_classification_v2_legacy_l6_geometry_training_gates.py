"""Write deterministic repeat or complete short-matrix L6 geometry gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l6_geometry import (
    MODES,
    load_geometry_training_config,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry_runtime import (
    write_geometry_repeat_gate,
    write_geometry_short_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    repeat = subparsers.add_parser("repeat")
    repeat.add_argument("--config", type=Path, required=True)
    repeat.add_argument("--mode", choices=MODES, required=True)
    repeat.add_argument("--primary-result", type=Path, required=True)
    repeat.add_argument("--repeat-result", type=Path, required=True)
    repeat.add_argument("--output", type=Path, required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--config", type=Path, required=True)
    for mode in MODES:
        matrix.add_argument(
            f"--{mode.replace('_', '-')}-gate",
            dest=f"{mode}_gate",
            type=Path,
            required=True,
        )
    matrix.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_geometry_training_config(args.config)
    if args.command == "repeat":
        payload = write_geometry_repeat_gate(
            config,
            mode=args.mode,
            primary_result_path=args.primary_result,
            repeat_result_path=args.repeat_result,
            output_path=args.output,
        )
    else:
        gates = {
            mode: getattr(args, f"{mode}_gate")
            for mode in MODES
        }
        payload = write_geometry_short_matrix(
            config,
            repeat_gate_paths=gates,
            output_path=args.output,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

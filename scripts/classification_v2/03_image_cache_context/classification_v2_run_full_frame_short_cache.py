"""Build or compare one immutable legacy L6 full-frame cache replica."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l6_full_frame_cache import (
    build_full_frame_cache,
    load_full_frame_cache_config,
    write_full_frame_cache_repeat_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument(
        "--replica",
        choices=("primary", "repeat"),
        required=True,
    )
    repeat = subparsers.add_parser("repeat")
    repeat.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_full_frame_cache_config(args.config)
    if args.command == "build":
        payload = build_full_frame_cache(config, replica=args.replica)
        output_path = config.output_root(args.replica)
    else:
        output_path, payload = write_full_frame_cache_repeat_gate(config)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "status": payload["status"],
                "valid": payload["valid"],
                "errors": payload["errors"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

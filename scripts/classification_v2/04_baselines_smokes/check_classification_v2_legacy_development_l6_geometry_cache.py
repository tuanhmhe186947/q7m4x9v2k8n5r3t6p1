"""Preflight, build, and audit the immutable legacy L6 geometry cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l6_geometry_cache import (
    audit_geometry_cache,
    build_geometry_cache,
    load_geometry_cache_config,
    preflight_geometry_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the legacy_16f T6 geometry cache without media reads."
    )
    parser.add_argument(
        "command",
        choices=("preflight", "build-cache", "audit-cache"),
    )
    parser.add_argument("--config-json", type=Path, required=True)
    args = parser.parse_args()

    config = load_geometry_cache_config(args.config_json)
    if args.command == "preflight":
        result = preflight_geometry_cache(config)
    elif args.command == "build-cache":
        path, result = build_geometry_cache(config)
        result = {**result, "manifest_path": str(path.resolve())}
    else:
        result = audit_geometry_cache(config)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

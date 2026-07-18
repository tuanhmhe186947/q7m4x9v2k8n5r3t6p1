"""Build or audit the legacy T6 pen-context cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l6_pen_context_cache import (
    audit_pen_context_cache,
    audit_pen_context_cache_repeat,
    build_pen_context_cache,
    load_pen_context_cache_config,
    preflight_pen_context_cache,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_json", type=Path)
    parser.add_argument(
        "--action",
        choices=("preflight", "build", "audit", "repeat"),
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=("primary", "repeat"),
        default="primary",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_pen_context_cache_config(args.config_json)
    if args.action == "preflight":
        payload = preflight_pen_context_cache(config, variant=args.variant)
    elif args.action == "build":
        path, payload = build_pen_context_cache(config, variant=args.variant)
        payload = {**payload, "manifest_path": str(path)}
    elif args.action == "audit":
        payload = audit_pen_context_cache(
            config,
            cache_root=config.cache_root(args.variant),
        )
    else:
        payload = audit_pen_context_cache_repeat(config)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    payload,
                    handle,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

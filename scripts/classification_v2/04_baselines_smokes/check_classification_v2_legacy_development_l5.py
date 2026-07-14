"""Run frozen L5 readiness and 224px cache gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l5 import (
    audit_legacy_l5_cache,
    audit_legacy_l5_readiness,
    load_legacy_l5_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run legacy-only L5 readiness and cache gates."
    )
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("readiness", "cache_short", "cache_full"),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--readiness-audit-json", type=Path)
    parser.add_argument("--short-cache-audit-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists; pass --overwrite explicitly: {args.output_json}"
        )
    config = load_legacy_l5_config(args.config_json)
    if args.mode == "readiness":
        result = audit_legacy_l5_readiness(config)
    else:
        if args.cache_root is None or args.readiness_audit_json is None:
            raise ValueError(
                "cache modes require --cache-root and "
                "--readiness-audit-json"
            )
        mode = "short" if args.mode == "cache_short" else "full"
        result = audit_legacy_l5_cache(
            config,
            cache_root=args.cache_root,
            mode=mode,
            readiness_audit_path=args.readiness_audit_json,
            short_cache_audit_path=args.short_cache_audit_json,
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

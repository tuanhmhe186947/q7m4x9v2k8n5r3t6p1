from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l6_social_relation_cache import (
    audit_social_relation_cache,
    build_social_relation_cache,
    load_social_relation_cache_config,
    preflight_social_relation_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_json", type=Path)
    parser.add_argument(
        "--action",
        choices=("preflight", "build", "audit"),
        required=True,
    )
    args = parser.parse_args()
    config = load_social_relation_cache_config(args.config_json)
    if args.action == "preflight":
        result = preflight_social_relation_cache(config)
    elif args.action == "build":
        path, result = build_social_relation_cache(config)
        result = {**result, "manifest_path": str(path)}
    else:
        result = audit_social_relation_cache(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

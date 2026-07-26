"""Create one single-use, run-local Classification V2 stage authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_preflight import EXPECTED_STAGE_IDS, validate_config

from pig_behavior.classification_v2.lineage_authorization import (
    create_stage_authorization,
)
from pig_behavior.classification_v2.lineage_config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=EXPECTED_STAGE_IDS, required=True)
    parser.add_argument("--ttl-hours", type=int, default=24)
    args = parser.parse_args()
    root, config = load_config(args.config)
    errors = validate_config(root, config)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}))
        return 1
    try:
        path, payload = create_stage_authorization(
            root=root,
            config_path=args.config.resolve(),
            config=config,
            stage_id=args.stage,
            ttl_hours=args.ttl_hours,
        )
    except (FileExistsError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}))
        return 2
    print(
        json.dumps(
            {
                "status": "AUTHORIZED",
                "stage": args.stage,
                "authorization_path": str(path),
                "authorization_id": payload["authorization_id"],
                "expires_at_utc": payload["expires_at_utc"],
                "single_use": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

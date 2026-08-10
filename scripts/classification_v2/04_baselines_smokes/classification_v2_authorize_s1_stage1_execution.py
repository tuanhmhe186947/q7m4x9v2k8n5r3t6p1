"""Issue four single-use permits for the authorized Stage-1 initial screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.stage1_execution_authorization import (
    Stage1ExecutionAuthorizationError,
    create_stage1_execution_permits,
)


def parse_args() -> argparse.Namespace:
    """Parse only execution-path inputs; scientific controls are immutable."""

    parser = argparse.ArgumentParser(
        description="Issue per-arm Stage-1 L4 execution permits."
    )
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--binding-bundle", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--ttl-hours", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    """Create exact per-view permits, or report a fail-closed reason."""

    args = parse_args()
    try:
        permits = create_stage1_execution_permits(
            repository_root=args.repository_root,
            outputs_root=args.outputs_root,
            authority_path=args.authority,
            binding_bundle_path=args.binding_bundle,
            ttl_hours=args.ttl_hours,
        )
    except Stage1ExecutionAuthorizationError as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": "AUTHORIZED",
                "single_use": True,
                "permits": {
                    view: {
                        "path": str(permit.path),
                        "sha256": permit.sha256,
                        "trial_id": permit.payload["trial_id"],
                    }
                    for view, permit in permits.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

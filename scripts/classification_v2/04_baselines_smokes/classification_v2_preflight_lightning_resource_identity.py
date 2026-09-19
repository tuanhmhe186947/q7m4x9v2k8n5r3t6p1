"""Materialize one fail-closed Lightning resource identity preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.lightning_resource_identity import (
    LightningResourceIdentityError,
    write_lightning_resource_preflight,
)


def parse_args() -> argparse.Namespace:
    """Accept one exact control-plane observation and one immutable contract."""

    parser = argparse.ArgumentParser(
        description="Validate exact Lightning teamspace, Studio, SSH, and L4 identity."
    )
    parser.add_argument("--resource-contract", type=Path, required=True)
    parser.add_argument("--observed-resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Write PASS only when the active resource is the registered resource."""

    args = parse_args()
    try:
        report = write_lightning_resource_preflight(
            contract_path=args.resource_contract,
            observed_resource_path=args.observed_resource,
            output_path=args.output,
        )
    except LightningResourceIdentityError as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

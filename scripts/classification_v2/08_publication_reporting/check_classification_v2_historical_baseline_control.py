"""Recheck historical classifier controls and artifact hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.experiments.historical_baseline import (
    check_historical_baseline_reconciliation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a historical baseline reconciliation audit."
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/experiment_registry/historical_controls/"
            "historical_baseline_reconciliation_18d6692.json"
        ),
    )
    args = parser.parse_args()
    result = check_historical_baseline_reconciliation(args.audit_json)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

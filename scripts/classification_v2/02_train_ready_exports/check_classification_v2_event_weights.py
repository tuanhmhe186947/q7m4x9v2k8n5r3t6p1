from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.datasets.event_weights import (
    audit_event_weight_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check classification_v2 event-balanced weight artifact."
    )
    parser.add_argument(
        "--event-weight-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/"
            "event_weight_manifest.csv"
        ),
    )
    parser.add_argument(
        "--window-manifest-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/sequence_features_reviewed/"
            "sequence_window_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/train_ready_windows/"
            "check_event_weight_audit.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.event_weight_csv.exists():
        raise FileNotFoundError(args.event_weight_csv)
    if not args.window_manifest_csv.exists():
        raise FileNotFoundError(args.window_manifest_csv)
    if not args.dry_run:
        require_output_paths_available(
            [args.output_json],
            overwrite=args.overwrite,
        )

    weights = pd.read_csv(args.event_weight_csv, low_memory=False)
    windows = pd.read_csv(args.window_manifest_csv, low_memory=False)
    audit = {
        **audit_event_weight_manifest(weights, windows),
        "event_weight_csv": str(args.event_weight_csv),
        "window_manifest_csv": str(args.window_manifest_csv),
        "dry_run": bool(args.dry_run),
        "audit_written": not args.dry_run,
    }
    if not args.dry_run:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(args.output_json)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

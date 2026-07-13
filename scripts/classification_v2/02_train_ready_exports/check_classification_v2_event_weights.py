from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.event_weight_csv.exists():
        raise FileNotFoundError(args.event_weight_csv)
    if not args.window_manifest_csv.exists():
        raise FileNotFoundError(args.window_manifest_csv)

    weights = pd.read_csv(args.event_weight_csv, low_memory=False)
    windows = pd.read_csv(args.window_manifest_csv, low_memory=False)
    audit = {
        **audit_event_weight_manifest(weights, windows),
        "event_weight_csv": str(args.event_weight_csv),
        "window_manifest_csv": str(args.window_manifest_csv),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

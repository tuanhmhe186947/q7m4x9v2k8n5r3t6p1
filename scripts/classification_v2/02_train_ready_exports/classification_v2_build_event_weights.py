from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.datasets.event_weights import (
    build_event_weight_manifest,
    json_default,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event-balanced sample weights for classification_v2 windows.")
    parser.add_argument(
        "--window-manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.window_manifest_csv.exists():
        raise FileNotFoundError(args.window_manifest_csv)

    windows = pd.read_csv(args.window_manifest_csv, low_memory=False)
    tables = build_event_weight_manifest(windows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "event_weight_manifest.csv"
    audit_path = args.output_dir / "event_weight_audit.json"
    tables.weights.to_csv(weights_path, index=False)
    audit = {
        **tables.audit,
        "window_manifest_csv": str(args.window_manifest_csv),
        "event_weight_manifest_csv": str(weights_path),
        "event_weight_audit_json": str(audit_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=json_default))
    if audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

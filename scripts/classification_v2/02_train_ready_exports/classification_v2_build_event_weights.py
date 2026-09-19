from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.datasets.event_weights import (
    build_event_weight_manifest,
    json_default,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build event-balanced sample weights for classification_v2 windows."
        )
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
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows"),
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.window_manifest_csv.exists():
        raise FileNotFoundError(args.window_manifest_csv)

    weights_path = args.output_csv or (
        args.output_dir / "event_weight_manifest.csv"
    )
    audit_path = args.audit_json or (
        args.output_dir / "event_weight_audit.json"
    )
    if not args.dry_run:
        require_output_paths_available(
            [weights_path, audit_path],
            overwrite=args.overwrite,
        )
    windows = pd.read_csv(args.window_manifest_csv, low_memory=False)
    tables = build_event_weight_manifest(windows)
    audit = {
        **tables.audit,
        "window_manifest_csv": str(args.window_manifest_csv),
        "event_weight_manifest_csv": str(weights_path),
        "event_weight_audit_json": str(audit_path),
        "dry_run": bool(args.dry_run),
    }
    audit["event_weight_manifest_written"] = (
        not audit["errors"] and not args.dry_run
    )
    if not audit["errors"] and not args.dry_run:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_temp = weights_path.with_suffix(weights_path.suffix + ".tmp")
        tables.weights.to_csv(weights_temp, index=False)
        weights_temp.replace(weights_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_temp = audit_path.with_suffix(audit_path.suffix + ".tmp")
        audit_temp.write_text(
            json.dumps(
                audit,
                indent=2,
                ensure_ascii=False,
                default=json_default,
            ),
            encoding="utf-8",
        )
        audit_temp.replace(audit_path)
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=json_default))
    if audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

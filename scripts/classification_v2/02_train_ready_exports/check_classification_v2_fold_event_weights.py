"""Rebuild and verify one fold-local event-weight artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.datasets.fold_event_weights import (
    audit_fold_event_weight_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check fold-local native-event and class weights."
    )
    parser.add_argument("--fold-event-weight-csv", type=Path, required=True)
    parser.add_argument("--window-manifest-csv", type=Path, required=True)
    parser.add_argument("--fold-role-csv", type=Path, required=True)
    parser.add_argument("--selection-manifest-csv", type=Path, required=True)
    parser.add_argument("--selection-col", default="fixed6_keep")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--class-weight-max", type=float, default=5.0)
    parser.add_argument("--sample-weight-max", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [
        args.fold_event_weight_csv,
        args.window_manifest_csv,
        args.fold_role_csv,
        args.selection_manifest_csv,
    ]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing fold event-weight check inputs={missing}")
    if not args.dry_run:
        require_output_paths_available(
            [args.output_json],
            overwrite=args.overwrite,
        )
    audit = audit_fold_event_weight_manifest(
        pd.read_csv(args.fold_event_weight_csv, low_memory=False),
        pd.read_csv(args.window_manifest_csv, low_memory=False),
        pd.read_csv(args.fold_role_csv, low_memory=False),
        selection=pd.read_csv(args.selection_manifest_csv, low_memory=False),
        selection_col=args.selection_col,
        class_weight_power=args.class_weight_power,
        class_weight_max=args.class_weight_max,
        sample_weight_max=args.sample_weight_max,
    )
    audit.update(
        {
            "dry_run": bool(args.dry_run),
            "fold_event_weight_csv": str(args.fold_event_weight_csv),
            "window_manifest_csv": str(args.window_manifest_csv),
            "fold_role_csv": str(args.fold_role_csv),
            "selection_manifest_csv": str(args.selection_manifest_csv),
            "output_json": str(args.output_json),
            "audit_written": not args.dry_run,
        }
    )
    if not args.dry_run:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(audit, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        temporary.replace(args.output_json)
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    if not audit["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

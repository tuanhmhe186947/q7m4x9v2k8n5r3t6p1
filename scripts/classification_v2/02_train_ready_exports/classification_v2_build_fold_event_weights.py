"""Build fold-local event/class weights from reviewed temporal windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.datasets.fold_event_weights import (
    build_fold_event_weight_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train-fold-only native-event and class weights."
    )
    parser.add_argument("--window-manifest-csv", type=Path, required=True)
    parser.add_argument("--fold-role-csv", type=Path, required=True)
    parser.add_argument("--selection-manifest-csv", type=Path, required=True)
    parser.add_argument("--selection-col", default="fixed6_keep")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--class-summary-csv", type=Path, required=True)
    parser.add_argument("--event-summary-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--class-weight-max", type=float, default=5.0)
    parser.add_argument("--sample-weight-max", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_inputs(
        args.window_manifest_csv,
        args.fold_role_csv,
        args.selection_manifest_csv,
    )
    outputs = [
        args.output_csv,
        args.class_summary_csv,
        args.event_summary_csv,
        args.audit_json,
    ]
    if not args.dry_run:
        require_output_paths_available(outputs, overwrite=args.overwrite)
    tables = build_fold_event_weight_manifest(
        pd.read_csv(args.window_manifest_csv, low_memory=False),
        pd.read_csv(args.fold_role_csv, low_memory=False),
        selection=pd.read_csv(args.selection_manifest_csv, low_memory=False),
        selection_col=args.selection_col,
        class_weight_power=args.class_weight_power,
        class_weight_max=args.class_weight_max,
        sample_weight_max=args.sample_weight_max,
    )
    audit = {
        **tables.audit,
        "dry_run": bool(args.dry_run),
        "inputs": {
            "window_manifest_csv": _artifact(args.window_manifest_csv),
            "fold_role_csv": _artifact(args.fold_role_csv),
            "selection_manifest_csv": _artifact(args.selection_manifest_csv),
        },
        "outputs": {
            "fold_event_weight_csv": str(args.output_csv),
            "class_summary_csv": str(args.class_summary_csv),
            "event_summary_csv": str(args.event_summary_csv),
            "audit_json": str(args.audit_json),
        },
        "artifacts_written": not args.dry_run,
    }
    if not args.dry_run:
        _write_csv_atomic(args.output_csv, tables.weights)
        _write_csv_atomic(args.class_summary_csv, tables.class_summary)
        _write_csv_atomic(args.event_summary_csv, tables.event_summary)
        _write_json_atomic(args.audit_json, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=True))


def _require_inputs(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing fold event-weight inputs={missing}")


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()

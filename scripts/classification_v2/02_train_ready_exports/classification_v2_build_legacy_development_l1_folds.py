"""Build recording-date-safe folds for the bounded legacy L1 packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.legacy_development_l1 import (
    build_legacy_development_folds,
)
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-units-csv", type=Path, required=True)
    parser.add_argument("--temporal-selection-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this derived fold packet explicitly.",
    )
    return parser.parse_args()


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "recording_groups": output_dir / "recording_group_manifest.csv",
        "native_folds": output_dir / "native_oof_fold_manifest.csv",
        "window_folds": output_dir / "window_oof_fold_manifest.csv",
        "class_support": output_dir / "class_by_fold_support.csv",
        "source_support": output_dir / "source_by_fold_support.csv",
        "audit": output_dir / "legacy_development_l1_fold_audit.json",
    }


def _assert_derived_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    immutable_data = (Path.cwd() / "data").resolve()
    if resolved == immutable_data or resolved.is_relative_to(immutable_data):
        raise ValueError(f"derived output must not be under data/: {resolved}")


def main() -> None:
    args = parse_args()
    _assert_derived_output(args.output_dir)
    paths = _output_paths(args.output_dir)
    require_output_paths_available(paths.values(), overwrite=args.overwrite)
    native_units = pd.read_csv(args.native_units_csv, low_memory=False)
    temporal_selection = pd.read_csv(
        args.temporal_selection_csv,
        low_memory=False,
    )
    tables = build_legacy_development_folds(
        native_units,
        temporal_selection,
        group_level="recording_date",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables.recording_groups.to_csv(paths["recording_groups"], index=False)
    tables.native_folds.to_csv(paths["native_folds"], index=False)
    tables.window_folds.to_csv(paths["window_folds"], index=False)
    tables.class_by_fold_support.to_csv(paths["class_support"], index=False)
    tables.source_by_fold_support.to_csv(paths["source_support"], index=False)

    output_paths = {
        name: path for name, path in paths.items() if name != "audit"
    }
    audit = {
        **tables.audit,
        "status": "PASS_LEGACY_DEVELOPMENT_L1_FOLDS",
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "model_training_authorized": False,
        "input_artifacts": {
            "native_units": {
                "path": str(args.native_units_csv),
                "sha256": file_sha256(args.native_units_csv),
            },
            "temporal_selection": {
                "path": str(args.temporal_selection_csv),
                "sha256": file_sha256(args.temporal_selection_csv),
            },
        },
        "output_artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in output_paths.items()
        },
    }
    paths["audit"].write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

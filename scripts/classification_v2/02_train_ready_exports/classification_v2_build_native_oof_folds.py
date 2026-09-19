from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.native_oof_folds import build_native_oof_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Build classification_v2 native temporal-unit OOF folds.")
    parser.add_argument(
        "--native-split-manifest",
        type=Path,
        default=Path(
            "outputs/classification_v2/native_temporal_units_publication_splits/publication_split_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units_oof_folds"),
    )
    args = parser.parse_args()

    result = build_native_oof_folds(pd.read_csv(args.native_split_manifest, low_memory=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "native_oof_fold_manifest.csv"
    audit_path = args.output_dir / "native_oof_fold_audit.json"
    result.manifest.to_csv(manifest_path, index=False)
    audit = {
        "native_split_manifest": str(args.native_split_manifest),
        "native_oof_fold_manifest": str(manifest_path),
        "native_oof_fold_audit": str(audit_path),
        **result.audit,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

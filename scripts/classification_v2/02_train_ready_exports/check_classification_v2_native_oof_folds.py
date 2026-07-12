from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.native_oof_folds import audit_native_oof_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 native temporal-unit OOF folds.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units_oof_folds/native_oof_fold_manifest.csv"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units_oof_folds/native_oof_fold_audit.json"),
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.manifest.exists():
        errors.append(f"missing_manifest={args.manifest}")
        manifest = pd.DataFrame()
    else:
        manifest = pd.read_csv(args.manifest, low_memory=False)
        check_audit = audit_native_oof_folds(manifest)
        errors.extend(check_audit["errors"])
    if not args.audit_json.exists():
        errors.append(f"missing_audit={args.audit_json}")
        stored_audit = {}
    else:
        stored_audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        errors.extend(stored_audit.get("errors", []))
    result = {
        "manifest": str(args.manifest),
        "audit_json": str(args.audit_json),
        "rows": int(len(manifest)),
        "fold_count": stored_audit.get("fold_count"),
        "recording_group_count": stored_audit.get("recording_group_count"),
        "duplicate_temporal_unit_key": stored_audit.get("duplicate_temporal_unit_key"),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

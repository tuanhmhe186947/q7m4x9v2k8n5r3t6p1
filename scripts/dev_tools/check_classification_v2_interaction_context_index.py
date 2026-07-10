from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.datasets.interaction_context_index import INTERACTION_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 interaction context index artifacts.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/interaction_window_context_manifest.csv"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/interaction_context_audit.json"),
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.manifest.exists():
        errors.append(f"missing_manifest={args.manifest}")
        manifest = pd.DataFrame()
    else:
        manifest = pd.read_csv(args.manifest, low_memory=False)
    if not args.audit_json.exists():
        errors.append(f"missing_audit={args.audit_json}")
        audit = {}
    else:
        audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        errors.extend(audit.get("errors", []))

    required_cols = {
        "window_id",
        "behavior_window_label",
        "is_interaction_window",
        "interaction_context_required",
        "expected_frame_slots",
        "available_frame_context_rows",
        "full_frame_context_available_count",
        "partner_context_available_count",
        "interaction_context_ready",
        "interaction_context_status",
    }
    missing_cols = sorted(required_cols.difference(manifest.columns))
    if missing_cols:
        errors.append(f"missing_cols={missing_cols}")
    if "window_id" in manifest and int(manifest["window_id"].duplicated().sum()):
        errors.append("duplicate_window_id")
    if "behavior_window_label" in manifest:
        interaction_labels = set(
            manifest.loc[
                manifest["behavior_window_label"].astype(str).isin(INTERACTION_LABELS),
                "behavior_window_label",
            ].astype(str)
        )
        missing_labels = sorted(INTERACTION_LABELS.difference(interaction_labels))
        if missing_labels:
            errors.append(f"missing_interaction_labels={missing_labels}")

    result = {
        "manifest": str(args.manifest),
        "audit_json": str(args.audit_json),
        "window_rows": int(len(manifest)),
        "interaction_window_rows": audit.get("interaction_window_rows"),
        "interaction_ready_rows": audit.get("interaction_ready_rows"),
        "interaction_status_counts": audit.get("interaction_status_counts"),
        "interaction_label_counts": audit.get("interaction_label_counts"),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

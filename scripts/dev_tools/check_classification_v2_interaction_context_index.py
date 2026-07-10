from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.datasets.interaction_context_index import INTERACTION_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 scene/interaction context index artifacts.")
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
        "scene_context_required",
        "scene_partner_context_required",
        "scene_context_ready",
        "scene_partner_context_ready",
        "scene_partner_context_status",
        "scene_partner_context_policy",
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
    if {"is_interaction_window", "scene_partner_context_ready"}.issubset(manifest.columns):
        is_interaction = _to_bool(manifest["is_interaction_window"])
        non_interaction_ready = int(_to_bool(manifest.loc[~is_interaction, "scene_partner_context_ready"]).sum())
        if non_interaction_ready == 0:
            errors.append("scene_partner_context_appears_label_gated")
    if "scene_partner_context_status" in manifest:
        not_evaluated = int(manifest["scene_partner_context_status"].astype(str).eq("not_evaluated").sum())
        if not_evaluated:
            errors.append(f"scene_partner_context_not_evaluated={not_evaluated}")

    result = {
        "manifest": str(args.manifest),
        "audit_json": str(args.audit_json),
        "interaction_window_rows": audit.get("interaction_window_rows"),
        "interaction_ready_rows": audit.get("interaction_ready_rows"),
        "scene_partner_context_ready_rows": audit.get("scene_partner_context_ready_rows"),
        "non_interaction_scene_partner_ready_rows": audit.get("non_interaction_scene_partner_ready_rows"),
        "scene_partner_status_counts": audit.get("scene_partner_status_counts"),
        "interaction_status_counts": audit.get("interaction_status_counts"),
        "interaction_label_counts": audit.get("interaction_label_counts"),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()

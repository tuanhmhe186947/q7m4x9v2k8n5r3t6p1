from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("outputs/classification_v2")


DECISION_FILES = {
    "roi": ROOT / "review_policy/roi_review_unit_gui_pilot/behavior_unit_review_decisions.csv",
    "motion": ROOT / "review_policy/motion_review_unit_gui_pilot/behavior_unit_review_decisions.csv",
    "posture": ROOT / "review_policy/posture_review_unit_gui_pilot/behavior_unit_review_decisions.csv",
    "interaction": ROOT / "review_policy/interaction_review_unit_gui_pilot/behavior_unit_review_decisions.csv",
}

REQUIRED_DECISION_COLUMNS = [
    "review_item_id",
    "review_unit_id",
    "review_unit_type",
    "temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "pig_id",
    "track_id",
    "object_track_key",
    "unit_start_frame",
    "unit_end_frame",
    "display_frame_indices",
    "review_template",
    "behavior_label",
    "original_behavior",
    "review_reason",
    "apply_scope",
    "manual_review_decision",
    "manual_corrected_behavior",
    "manual_label_strength",
    "manual_training_action",
    "manual_sample_weight",
    "manual_note",
]


def counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[column].fillna("<NA>").astype(str).value_counts().items()}


def read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def decision_schema_status() -> dict[str, object]:
    out: dict[str, object] = {}
    for name, path in DECISION_FILES.items():
        df = read(path)
        out[name] = {
            "rows": int(len(df)),
            "missing_columns": [c for c in REQUIRED_DECISION_COLUMNS if c not in df.columns],
            "has_window_uid": "window_uid" in df.columns,
            "duplicate_review_unit_id": int(df["review_unit_id"].astype(str).duplicated().sum())
            if "review_unit_id" in df.columns
            else None,
            "manual_review_decision": counts(df, "manual_review_decision"),
        }
    return out


def main() -> None:
    enhanced = read(ROOT / "frame_features/spatiotemporal_frame_features_enhanced.csv", usecols=["behavior"])
    reviewed = read(
        ROOT / "review_policy/reviewed_frame_features.csv",
        usecols=[
            "behavior",
            "behavior_before_review",
            "review_include_in_training",
            "source_type",
            "review_decision_applied",
        ],
    )
    intervals = read(
        ROOT / "sequence_features_reviewed/temporal_label_intervals.csv",
        usecols=["temporal_unit_key", "source_type", "behavior_temporal_final", "temporal_consistency_status"],
    )
    windows = read(
        ROOT / "sequence_features_reviewed/sequence_window_manifest.csv",
        usecols=[
            "window_id",
            "source_type",
            "behavior_window_label",
            "sequence_label_status",
            "window_valid_for_main_train",
            "window_exclusion_reason",
            "review_excluded_frame_count_window",
        ],
    )
    units = read(ROOT / "review_units/review_unit_manifest.csv", usecols=["review_unit_id", "behavior_label"])
    audit = json.loads((ROOT / "review_policy/apply_review_unit_decisions_audit.json").read_text(encoding="utf-8"))
    seq_audit = json.loads((ROOT / "sequence_features_reviewed/sequence_window_audit.json").read_text(encoding="utf-8"))

    template_counts = {}
    for name in ["interaction", "roi", "motion", "posture"]:
        df = read(ROOT / f"review_units/{name}_review_unit_template.csv", usecols=["behavior_label"])
        template_counts[name] = counts(df, "behavior_label")

    summary = {
        "enhanced_rows": int(len(enhanced)),
        "reviewed_rows": int(len(reviewed)),
        "temporal_interval_rows": int(len(intervals)),
        "reviewed_sequence_window_rows": int(len(windows)),
        "duplicate_temporal_unit_key_in_intervals": int(intervals["temporal_unit_key"].astype(str).duplicated().sum()),
        "duplicate_review_unit_id": int(units["review_unit_id"].astype(str).duplicated().sum()),
        "decision_csv_schema_status": decision_schema_status(),
        "review_template_label_counts": template_counts,
        "applied_decision_counts": audit.get("apply_audit", {}),
        "label_distribution_before_review": counts(reviewed, "behavior_before_review"),
        "label_distribution_after_review": counts(reviewed, "behavior"),
        "source_distribution_reviewed": counts(reviewed, "source_type"),
        "source_distribution_windows": counts(windows, "source_type"),
        "temporal_consistency_status": counts(intervals, "temporal_consistency_status"),
        "reviewed_sequence_label_status": counts(windows, "sequence_label_status"),
        "window_valid_for_main_train": counts(windows, "window_valid_for_main_train"),
        "review_excluded_frame_count_window": counts(windows, "review_excluded_frame_count_window"),
        "sequence_build_strategy": seq_audit.get("parameters", {}).get("build_strategy"),
        "sequence_audit_errors": seq_audit.get("errors", []),
        "sequence_audit_warnings": seq_audit.get("warnings", []),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

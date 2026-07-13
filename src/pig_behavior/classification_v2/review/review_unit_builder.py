"""Build canonical review units for classification_v2.

Design rule:
- Review unit is not the same as a training window.
- legacy_recovered is reviewed as a full 16-frame burst.
- cvat_tracking_xml is reviewed as a 6-frame label interval.
- Training windows 6/8/12/16 are rebuilt downstream from reviewed units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.review.behavior_review_contract import (
    audit_review_unit_contract,
)

LEGACY_SOURCE = "legacy_recovered"
CVAT_SOURCE = "cvat_tracking_xml"
INTERACTION_BEHAVIORS = {"fight", "social-nose"}
ROI_BEHAVIORS = {"eat", "drink", "playwithtoy"}
MOTION_BEHAVIORS = {"move", "explore", "stand"}
# stand is intentionally not in posture review. In this dataset, stand is
# a context/motion candidate more often than a lying/sitting posture ambiguity.
POSTURE_BEHAVIORS = {"lying", "sitting"}
# Always surface rare/high-risk labels even when no window-level review template
# has flagged them. eat/drink stay gated by ROI/window review signals because
# reviewing all feeding units would be too large; playwithtoy is rare enough
# and ROI-dominant enough to review all units.
ALWAYS_REVIEW_BEHAVIORS = INTERACTION_BEHAVIORS | {"playwithtoy"}
ALWAYS_REVIEW_REASON_BY_BEHAVIOR = {
    "fight": "interaction_unit_candidate",
    "social-nose": "interaction_unit_candidate",
    "playwithtoy": "rare_roi_behavior_candidate",
}


@dataclass(slots=True)
class ReviewUnitConfig:
    intervals_csv: Path
    sequence_window_manifest_csv: Path
    output_dir: Path
    window_review_manifest_csv: Path | None = None
    max_units_per_template: int = 0


def build_review_units(config: ReviewUnitConfig) -> dict[str, Any]:
    intervals = pd.read_csv(config.intervals_csv, low_memory=False)
    windows = pd.read_csv(config.sequence_window_manifest_csv, low_memory=False)

    _validate_columns(
        intervals,
        [
            "temporal_unit_key",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "pig_id",
            "track_id",
            "label_window_start",
            "label_window_end",
            "behavior_temporal_final",
            "temporal_consistency_status",
        ],
        "intervals_csv",
    )
    _validate_columns(
        windows,
        [
            "window_id",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "pig_id",
            "window_length_frames",
            "window_start_frame",
            "window_end_frame",
            "behavior_window_label",
            "sequence_label_status",
            "window_valid_for_main_train",
        ],
        "sequence_window_manifest_csv",
    )

    units = _base_units_from_intervals(intervals)
    units = _add_window_coverage(units, windows)

    review_manifest = None
    if config.window_review_manifest_csv and config.window_review_manifest_csv.exists():
        review_manifest = pd.read_csv(config.window_review_manifest_csv, low_memory=False)
        if "window_id" not in review_manifest.columns:
            raise ValueError(
                f"{config.window_review_manifest_csv} must contain window_id. "
                "Rebuild review templates with the window_id standard package."
            )
        units = _add_window_review_signals(units, windows, review_manifest)
    else:
        units["window_review_hit_count"] = 0
        units["review_templates_hit"] = ""
        units["review_reasons_window"] = ""
        units["review_priority_window_max"] = 0.0

    units = _finalize_unit_review_fields(units)

    contract_audit = audit_review_unit_contract(units)
    input_errors = _input_contract_errors(intervals, windows, units)
    capacity_errors = _template_capacity_errors(
        units,
        config.max_units_per_template,
    )
    errors = input_errors + list(contract_audit["errors"]) + capacity_errors
    warnings = list(contract_audit["warnings"])

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if errors:
        failed_audit = {
            "errors": errors,
            "warnings": warnings,
            "rows": {
                "intervals": int(len(intervals)),
                "windows": int(len(windows)),
                "review_units": int(len(units)),
            },
            "review_unit_contract": contract_audit,
        }
        audit_path = config.output_dir / "review_unit_audit.json"
        audit_path.write_text(
            json.dumps(failed_audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise ValueError("review unit contract failed: " + "; ".join(errors))

    unit_path = config.output_dir / "review_unit_manifest.csv"
    units.to_csv(unit_path, index=False)

    outputs: dict[str, Path] = {"review_unit_manifest": unit_path}
    templates = _write_template_unit_files(units, config.output_dir, config.max_units_per_template)
    outputs.update(templates)

    template_audit = _audit_template_partition(units, templates)
    errors.extend(template_audit["errors"])
    warnings.extend(template_audit["warnings"])

    audit = {
        "errors": errors,
        "warnings": warnings,
        "inputs": {
            "intervals_csv": str(config.intervals_csv),
            "sequence_window_manifest_csv": str(config.sequence_window_manifest_csv),
            "window_review_manifest_csv": str(config.window_review_manifest_csv)
            if config.window_review_manifest_csv
            else None,
        },
        "rows": {
            "intervals": int(len(intervals)),
            "windows": int(len(windows)),
            "review_units": int(len(units)),
        },
        "unit_distribution": {
            "source_type": _counts(units, "source_type"),
            "review_unit_type": _counts(units, "review_unit_type"),
            "behavior_label": _counts(units, "behavior_label"),
            "temporal_consistency_status": _counts(units, "temporal_consistency_status"),
            "include_in_review": _counts(units, "include_in_review"),
        },
        "templates": {
            name: {
                "path": str(path),
                "rows": int(pd.read_csv(path, low_memory=False).shape[0]),
            }
            for name, path in outputs.items()
        },
        "review_unit_contract": contract_audit,
        "template_partition": template_audit,
        "review_reason_counts": _counts(
            units[units["include_in_review"].astype(bool)],
            "review_reason",
        ),
    }

    audit_path = config.output_dir / "review_unit_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise ValueError("review template partition failed: " + "; ".join(errors))
    return audit


def _base_units_from_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    out = intervals.copy()
    out["review_unit_id"] = out["temporal_unit_key"].astype(str)
    out["review_unit_type"] = np.where(
        out["source_type"].astype(str).eq(LEGACY_SOURCE),
        "legacy_burst_16",
        np.where(
            out["source_type"].astype(str).eq(CVAT_SOURCE),
            "cvat_interval_6",
            "temporal_interval",
        ),
    )
    out["unit_start_frame"] = pd.to_numeric(
        out["label_window_start"],
        errors="coerce",
    ).astype("Int64")
    out["unit_end_frame"] = pd.to_numeric(
        out["label_window_end"],
        errors="coerce",
    ).astype("Int64")
    out["unit_frame_count"] = (out["unit_end_frame"] - out["unit_start_frame"] + 1).astype("Int64")
    frame_bounds = zip(
        out["unit_start_frame"],
        out["unit_end_frame"],
        strict=False,
    )
    out["display_frame_indices"] = [
        _frame_list(start, end) for start, end in frame_bounds
    ]
    out["display_frame_count"] = (
        out["display_frame_indices"].astype(str).map(lambda s: 0 if not s else len(s.split(",")))
    )
    out["behavior_label"] = out["behavior_temporal_final"].fillna("").astype(str)

    keep = [
        "review_unit_id",
        "review_unit_type",
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "unit_start_frame",
        "unit_end_frame",
        "unit_frame_count",
        "display_frame_indices",
        "display_frame_count",
        "behavior_label",
        "temporal_label_mode",
        "label_anchor_frame_index",
        "temporal_consistency_status",
        "behavior_consistency_in_interval",
        "temporal_interval_complete",
        "bbox_valid_ratio_interval",
        "hidden_ratio_interval",
        "visible_ratio_interval",
        "spatiotemporal_feature_valid_ratio_interval",
        "interval_review_reason",
        "interaction_annotation_policy",
        "interaction_role_policy",
        "label_propagation_policy",
        "allow_label_propagation",
        "requires_partner_context",
        "social_nose_actor_only",
        "fight_group_label",
        # ROI/context columns are optional in temporal intervals, but keeping them
        # here makes ROI review GUIs self-contained when the upstream builder
        # provides aggregated ROI evidence. Missing columns are filled with NA.
        "roi_feature_required",
        "roi_target_class",
        "roi_target_available",
        "roi_target_near",
        "roi_target_contact",
        "roi_context_quality",
        "use_for_roi_training",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    return out[keep].copy()


def _add_window_coverage(units: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_type", "dataset_id", "video_key", "object_track_key", "pig_id"]
    w = windows[
        [
            "window_id",
            *keys,
            "window_length_frames",
            "window_start_frame",
            "window_end_frame",
            "behavior_window_label",
            "sequence_label_status",
            "window_valid_for_main_train",
        ]
    ].copy()
    u = units[["review_unit_id", *keys, "unit_start_frame", "unit_end_frame"]].copy()

    for col in ["unit_start_frame", "unit_end_frame"]:
        u[col] = pd.to_numeric(u[col], errors="coerce")
    for col in ["window_start_frame", "window_end_frame", "window_length_frames"]:
        w[col] = pd.to_numeric(w[col], errors="coerce")

    merged = u.merge(w, on=keys, how="left")
    overlaps = merged[
        (merged["window_end_frame"] >= merged["unit_start_frame"])
        & (merged["window_start_frame"] <= merged["unit_end_frame"])
    ].copy()

    if overlaps.empty:
        coverage = pd.DataFrame({"review_unit_id": units["review_unit_id"].astype(str)})
    else:
        coverage = (
            overlaps.groupby("review_unit_id", dropna=False)
            .agg(
                affected_window_count=("window_id", "nunique"),
                affected_window_lengths=("window_length_frames", lambda s: _join_unique_ints(s)),
                affected_main_train_windows=(
                    "window_valid_for_main_train",
                    lambda s: int(_to_bool(s).sum()),
                ),
                affected_stable_windows=(
                    "sequence_label_status",
                    lambda s: int(s.astype(str).eq("stable").sum()),
                ),
                affected_uncertain_or_transition_windows=(
                    "sequence_label_status",
                    lambda s: int(
                        s.astype(str)
                        .isin(["uncertain", "transition", "incomplete"])
                        .sum()
                    ),
                ),
                affected_behavior_labels=(
                    "behavior_window_label",
                    lambda s: _join_unique_strings(s),
                ),
            )
            .reset_index()
        )

    out = units.merge(coverage, on="review_unit_id", how="left")
    defaults = {
        "affected_window_count": 0,
        "affected_window_lengths": "",
        "affected_main_train_windows": 0,
        "affected_stable_windows": 0,
        "affected_uncertain_or_transition_windows": 0,
        "affected_behavior_labels": "",
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
        out[col] = out[col].fillna(value)
    return out


def _add_window_review_signals(
    units: pd.DataFrame, windows: pd.DataFrame, review_manifest: pd.DataFrame
) -> pd.DataFrame:
    keys = ["source_type", "dataset_id", "video_key", "object_track_key", "pig_id"]
    cols = ["window_id", *keys, "window_length_frames", "window_start_frame", "window_end_frame"]
    w = windows[cols].drop_duplicates("window_id").copy()
    candidate_columns = [
        "window_id",
        "review_template",
        "review_reason",
        "review_priority",
    ]
    rcols = [c for c in candidate_columns if c in review_manifest.columns]
    r = review_manifest[rcols].copy()
    wr = r.merge(w, on="window_id", how="left")

    u = units[["review_unit_id", *keys, "unit_start_frame", "unit_end_frame"]].copy()
    for col in ["unit_start_frame", "unit_end_frame"]:
        u[col] = pd.to_numeric(u[col], errors="coerce")
    for col in ["window_start_frame", "window_end_frame"]:
        wr[col] = pd.to_numeric(wr[col], errors="coerce")

    merged = u.merge(wr, on=keys, how="left")
    hit = merged[
        (merged["window_end_frame"] >= merged["unit_start_frame"])
        & (merged["window_start_frame"] <= merged["unit_end_frame"])
    ].copy()

    if hit.empty:
        signals = pd.DataFrame({"review_unit_id": units["review_unit_id"].astype(str)})
    else:
        signals = (
            hit.groupby("review_unit_id", dropna=False)
            .agg(
                window_review_hit_count=("window_id", "nunique"),
                review_templates_hit=("review_template", lambda s: _join_unique_strings(s)),
                review_reasons_window=("review_reason", lambda s: _join_unique_strings(s)),
                review_priority_window_max=(
                    "review_priority",
                    lambda s: float(
                        pd.to_numeric(s, errors="coerce").max(skipna=True)
                        if pd.to_numeric(s, errors="coerce").notna().any()
                        else 0.0
                    ),
                ),
            )
            .reset_index()
        )

    out = units.merge(signals, on="review_unit_id", how="left")
    defaults = {
        "window_review_hit_count": 0,
        "review_templates_hit": "",
        "review_reasons_window": "",
        "review_priority_window_max": 0.0,
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
        out[col] = out[col].fillna(value)
    return out


def _finalize_unit_review_fields(units: pd.DataFrame) -> pd.DataFrame:
    out = units.copy()
    sort_columns = [
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "unit_start_frame",
        "temporal_unit_key",
    ]
    out = out.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    behavior = out["behavior_label"].fillna("").astype(str)
    status = out["temporal_consistency_status"].fillna("").astype(str)
    interval_reason = out["interval_review_reason"].fillna("").astype(str)
    window_reasons = out["review_reasons_window"].fillna("").astype(str)
    review_hit = pd.to_numeric(out["window_review_hit_count"], errors="coerce").fillna(0).gt(0)

    reason = np.where(
        review_hit,
        window_reasons,
        np.where(interval_reason.ne("") & interval_reason.ne("nan"), interval_reason, ""),
    )
    reason = pd.Series(reason, index=out.index).replace("nan", "")
    temporal_bad = ~status.eq("stable")
    reason = reason.mask(reason.eq("") & temporal_bad, "temporal_unit_not_stable")

    # Force-review high-risk/rare behaviors even when there is no inherited
    # window-level review signal. This keeps interaction units and all
    # playwithtoy units visible to human review without expanding eat/drink
    # to every stable ROI interval.
    for label, label_reason in ALWAYS_REVIEW_REASON_BY_BEHAVIOR.items():
        reason = reason.mask(reason.eq("") & behavior.eq(label), label_reason)

    out["review_reason"] = reason.fillna("")
    out["include_in_review"] = out["review_reason"].astype(str).ne("")

    priority = pd.to_numeric(out["review_priority_window_max"], errors="coerce").fillna(0.0)
    priority = priority + 30 * behavior.isin(INTERACTION_BEHAVIORS).astype(int)
    priority = priority + 25 * behavior.eq("playwithtoy").astype(int)
    priority = priority + 20 * temporal_bad.astype(int)
    priority = priority + 10 * out["review_unit_type"].astype(str).eq("legacy_burst_16").astype(int)
    out["review_priority"] = priority.astype(float)

    out["review_template"] = np.select(
        [
            behavior.isin(INTERACTION_BEHAVIORS),
            behavior.isin(ROI_BEHAVIORS),
            behavior.isin(MOTION_BEHAVIORS),
            behavior.isin(POSTURE_BEHAVIORS),
            temporal_bad,
        ],
        ["interaction", "roi", "motion", "posture", "temporal_consistency"],
        default="general",
    )
    out["recommended_gui"] = "review_temporal_unit_gui"
    out["apply_scope"] = np.where(
        out["review_unit_type"].astype(str).eq("legacy_burst_16"),
        "whole_legacy_burst_16f",
        "cvat_interval_6f",
    )
    out["review_item_id"] = [f"unit_review_{idx:08d}" for idx in range(len(out))]

    # Manual decision columns, empty by default.
    for col in [
        "manual_review_decision",
        "manual_corrected_behavior",
        "manual_label_strength",
        "manual_training_action",
        "manual_sample_weight",
        "manual_note",
    ]:
        if col not in out.columns:
            out[col] = ""

    first_cols = [
        "review_item_id",
        "review_unit_id",
        "review_unit_type",
        "review_template",
        "review_reason",
        "review_priority",
        "recommended_gui",
        "apply_scope",
    ]
    rest = [c for c in out.columns if c not in first_cols]
    return out[first_cols + rest]


def _write_template_unit_files(units: pd.DataFrame, output_dir: Path, cap: int) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    review_units = units[units["include_in_review"].astype(bool)].copy()
    for name in ["roi", "motion", "interaction", "posture", "temporal_consistency"]:
        part = review_units[review_units["review_template"].astype(str).eq(name)].copy()
        if cap > 0 and len(part) > cap:
            raise ValueError(
                f"canonical template {name} has {len(part)} rows, exceeding cap={cap}; "
                "use a separate pilot/shortlist builder instead of truncating it"
            )
        path = output_dir / f"{name}_review_unit_template.csv"
        part.to_csv(path, index=False)
        paths[f"{name}_review_unit_template"] = path
    full_path = output_dir / "full_review_unit_manifest.csv"
    review_units.sort_values("review_priority", ascending=False).to_csv(full_path, index=False)
    paths["full_review_unit_manifest"] = full_path
    return paths


def _template_capacity_errors(units: pd.DataFrame, cap: int) -> list[str]:
    """Reject canonical truncation before any manifest or template is written."""
    if cap <= 0:
        return []
    review_units = units[units["include_in_review"].astype(bool)]
    errors = []
    for name in ["roi", "motion", "interaction", "posture", "temporal_consistency"]:
        count = int(review_units["review_template"].astype(str).eq(name).sum())
        if count > cap:
            errors.append(f"canonical_template_cap_exceeded={name}:{count}:{cap}")
    return errors


def _input_contract_errors(
    intervals: pd.DataFrame,
    windows: pd.DataFrame,
    units: pd.DataFrame,
) -> list[str]:
    """Check one-to-one lineage before any canonical review file is written."""
    errors: list[str] = []
    interval_keys = intervals["temporal_unit_key"].fillna("").astype(str).str.strip()
    if interval_keys.eq("").any():
        errors.append(f"blank_interval_temporal_unit_key={int(interval_keys.eq('').sum())}")
    if interval_keys.duplicated(keep=False).any():
        count = int(interval_keys.duplicated(keep=False).sum())
        errors.append(f"duplicate_interval_temporal_unit_key={count}")
    if len(intervals) != len(units):
        errors.append(f"interval_review_unit_row_mismatch={len(intervals)}:{len(units)}")
    if set(interval_keys) != set(units["temporal_unit_key"].astype(str)):
        errors.append("interval_review_unit_key_set_mismatch")

    window_ids = windows["window_id"].fillna("").astype(str).str.strip()
    if window_ids.eq("").any():
        errors.append(f"blank_window_id={int(window_ids.eq('').sum())}")
    if window_ids.duplicated(keep=False).any():
        errors.append(f"duplicate_window_id={int(window_ids.duplicated(keep=False).sum())}")
    if "window_uid" in intervals.columns or "window_uid" in windows.columns:
        errors.append("forbidden_window_uid_column")
    return errors


def _audit_template_partition(
    units: pd.DataFrame,
    templates: dict[str, Path],
) -> dict[str, Any]:
    """Prove that policy templates partition the full review manifest."""
    errors: list[str] = []
    warnings: list[str] = []
    review_ids = set(
        units.loc[units["include_in_review"].astype(bool), "review_unit_id"].astype(str)
    )
    union_ids: set[str] = set()
    for name, path in templates.items():
        if name == "full_review_unit_manifest":
            continue
        part = pd.read_csv(path, low_memory=False)
        ids = part["review_unit_id"].fillna("").astype(str)
        if ids.duplicated(keep=False).any():
            errors.append(f"duplicate_template_review_unit_id={name}")
        overlap = union_ids.intersection(ids)
        if overlap:
            errors.append(f"review_unit_in_multiple_templates={name}:count={len(overlap)}")
        union_ids.update(ids)
    if union_ids != review_ids:
        missing = len(review_ids - union_ids)
        unexpected = len(union_ids - review_ids)
        errors.append(
            f"template_union_mismatch=missing:{missing}:unexpected:{unexpected}"
        )
    return {
        "expected_review_units": int(len(review_ids)),
        "template_union_units": int(len(union_ids)),
        "errors": errors,
        "warnings": warnings,
    }


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _frame_list(start: Any, end: Any) -> str:
    if pd.isna(start) or pd.isna(end):
        return ""
    try:
        a, b = int(start), int(end)
    except Exception:
        return ""
    if b < a:
        return ""
    if b - a > 200:
        return ""
    return ",".join(str(x) for x in range(a, b + 1))


def _join_unique_strings(s: pd.Series) -> str:
    vals = []
    for value in s.dropna().astype(str):
        if not value or value == "nan":
            continue
        for token in str(value).split(";"):
            token = token.strip()
            if token and token not in vals:
                vals.append(token)
    return ";".join(vals)


def _join_unique_ints(s: pd.Series) -> str:
    vals = sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()))
    return ",".join(str(v) for v in vals)


def _counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).to_dict().items()}

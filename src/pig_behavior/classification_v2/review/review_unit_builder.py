"""Build canonical review units for classification_v2.

Design rule:
- Review unit is not the same as a training window.
- legacy_recovered is reviewed as a full 16-frame burst.
- cvat_tracking_xml is reviewed as a 6-frame label interval.
- Training windows 6/8/12/16 are rebuilt downstream from reviewed units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.review.behavior_evidence import (
    REVIEW_EVIDENCE_COLUMNS,
    add_behavior_review_evidence,
    audit_behavior_review_evidence,
)
from pig_behavior.classification_v2.review.behavior_review_contract import (
    audit_review_unit_contract,
)
from pig_behavior.classification_v2.review.behavior_review_selection import (
    BehaviorReviewSelectionConfig,
    assign_behavior_review_cohorts,
    audit_behavior_review_selection,
)
from pig_behavior.classification_v2.review.pig_strenet_review_evidence import (
    PIG_REVIEW_EVIDENCE_COLUMNS,
    attach_pig_strenet_review_evidence,
    load_pig_strenet_review_evidence,
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
    output_dir: Path
    sequence_window_manifest_csv: Path | None = None
    window_review_manifest_csv: Path | None = None
    max_units_per_template: int = 0
    include_all_retained_legacy_units: bool = False
    pig_strenet_artifact_dir: Path | None = None
    include_all_retained_native_units: bool = False
    behavior_selection: BehaviorReviewSelectionConfig = field(
        default_factory=BehaviorReviewSelectionConfig
    )


def build_review_units(config: ReviewUnitConfig) -> dict[str, Any]:
    intervals = pd.read_csv(config.intervals_csv, low_memory=False)
    windows = None
    if config.sequence_window_manifest_csv is not None:
        windows = pd.read_csv(config.sequence_window_manifest_csv, low_memory=False)
    pig_strenet_audit: dict[str, Any] = {
        "configured": False,
        "valid": True,
        "errors": [],
        "warnings": ["pig_strenet_review_evidence_not_configured"],
    }
    if config.pig_strenet_artifact_dir is not None:
        pig_evidence, pig_strenet_audit = load_pig_strenet_review_evidence(
            config.pig_strenet_artifact_dir
        )
        interval_keys = set(intervals["temporal_unit_key"].astype(str))
        evidence_keys = set(pig_evidence["temporal_unit_key"].astype(str))
        missing_evidence = sorted(interval_keys.difference(evidence_keys))
        unused_evidence = sorted(evidence_keys.difference(interval_keys))
        if missing_evidence or unused_evidence:
            raise ValueError(
                "Pig-STRENet review evidence key mismatch: "
                f"missing={len(missing_evidence)} unused={len(unused_evidence)}"
            )
        intervals = attach_pig_strenet_review_evidence(
            intervals,
            pig_evidence,
        )
        pig_strenet_audit["configured"] = True
        pig_strenet_audit["matched_temporal_units"] = int(len(intervals))

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
    if windows is not None:
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
    if windows is None:
        units = _add_native_only_window_fields(units)
    else:
        units = _add_window_coverage(units, windows)

    review_manifest = None
    if config.window_review_manifest_csv and windows is None:
        raise ValueError(
            "window-review overlay requires sequence_window_manifest_csv; "
            "use native-only mode without --window-review-manifest-csv"
        )
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

    units = _finalize_unit_review_fields(
        units,
        include_all_retained_legacy_units=config.include_all_retained_legacy_units,
        include_all_retained_native_units=config.include_all_retained_native_units,
        behavior_selection=config.behavior_selection,
    )
    behavior_evidence_audit = audit_behavior_review_evidence(units)
    behavior_selection_audit = audit_behavior_review_selection(
        units,
        config.behavior_selection,
    )

    legacy_units = units[units["review_unit_type"].astype(str).eq("legacy_burst_16")]
    reviewed_legacy_units = legacy_units[
        legacy_units["include_in_review"].astype(bool)
    ]
    expected_legacy_ids = set(legacy_units["review_unit_id"].astype(str))
    reviewed_legacy_ids = set(reviewed_legacy_units["review_unit_id"].astype(str))
    missing_legacy_ids = sorted(expected_legacy_ids - reviewed_legacy_ids)
    review_scope = {
        "include_all_retained_legacy_units": bool(
            config.include_all_retained_legacy_units
        ),
        "expected_legacy_native_units": int(len(expected_legacy_ids)),
        "reviewed_legacy_native_units": int(len(reviewed_legacy_ids)),
        "missing_legacy_native_units": int(len(missing_legacy_ids)),
        "missing_legacy_native_unit_sample": missing_legacy_ids[:20],
    }
    errors: list[str] = []
    if config.include_all_retained_legacy_units:
        if not expected_legacy_ids:
            errors.append("required_complete_legacy_review_but_no_legacy_units")
        elif missing_legacy_ids:
            errors.append(
                "missing_complete_legacy_review_units="
                f"{len(missing_legacy_ids)}"
            )

    contract_audit = audit_review_unit_contract(units)
    input_errors = _input_contract_errors(intervals, windows, units)
    capacity_errors = _template_capacity_errors(
        units,
        config.max_units_per_template,
    )
    errors.extend(input_errors + list(contract_audit["errors"]) + capacity_errors)
    errors.extend(behavior_evidence_audit["errors"])
    errors.extend(behavior_selection_audit["errors"])
    errors.extend(pig_strenet_audit.get("errors", []))
    warnings = list(contract_audit["warnings"])
    warnings.extend(behavior_evidence_audit["warnings"])
    warnings.extend(behavior_selection_audit["warnings"])
    warnings.extend(pig_strenet_audit.get("warnings", []))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if errors:
        failed_audit = {
            "errors": errors,
            "warnings": warnings,
            "rows": {
                "intervals": int(len(intervals)),
                "windows": int(len(windows)) if windows is not None else None,
                "review_units": int(len(units)),
            },
            "review_unit_contract": contract_audit,
            "behavior_evidence": behavior_evidence_audit,
            "behavior_selection": behavior_selection_audit,
            "pig_strenet_review_evidence": pig_strenet_audit,
            "review_scope": review_scope,
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
            "sequence_window_manifest_csv": (
                str(config.sequence_window_manifest_csv)
                if config.sequence_window_manifest_csv
                else None
            ),
            "window_review_manifest_csv": str(config.window_review_manifest_csv)
            if config.window_review_manifest_csv
            else None,
            "pig_strenet_artifact_dir": str(config.pig_strenet_artifact_dir)
            if config.pig_strenet_artifact_dir
            else None,
        },
        "rows": {
            "intervals": int(len(intervals)),
        "windows": int(len(windows)) if windows is not None else None,
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
        "behavior_evidence": behavior_evidence_audit,
        "behavior_selection": behavior_selection_audit,
        "pig_strenet_review_evidence": pig_strenet_audit,
        "review_scope": review_scope,
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
    out = add_behavior_review_evidence(
        intervals,
        behavior_col="behavior_temporal_final",
    )
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
        *REVIEW_EVIDENCE_COLUMNS,
        *PIG_REVIEW_EVIDENCE_COLUMNS,
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    return out[keep].copy()


def _add_native_only_window_fields(units: pd.DataFrame) -> pd.DataFrame:
    """Mark window coverage as intentionally unavailable in native-only mode."""

    out = units.copy()
    out["window_dependency_mode"] = "native_only"
    out["window_coverage_computed"] = False
    out["review_population_source"] = "native_intervals"
    for column in [
        "affected_window_count",
        "affected_window_lengths",
        "affected_main_train_windows",
        "affected_stable_windows",
        "affected_uncertain_or_transition_windows",
        "affected_behavior_labels",
    ]:
        out[column] = pd.NA
    return out


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
    out["window_dependency_mode"] = "manifest"
    out["window_coverage_computed"] = True
    out["review_population_source"] = "native_intervals_plus_windows"
    return out


def _add_window_review_signals(
    units: pd.DataFrame, windows: pd.DataFrame, review_manifest: pd.DataFrame
) -> pd.DataFrame:
    keys = ["source_type", "dataset_id", "video_key", "object_track_key", "pig_id"]
    cols = ["window_id", *keys, "window_length_frames", "window_start_frame", "window_end_frame"]
    required_review_columns = [
        "window_id",
        "review_template",
        "review_reason",
        "review_priority",
    ]
    missing_review_columns = [
        column
        for column in required_review_columns
        if column not in review_manifest.columns
    ]
    if missing_review_columns:
        raise ValueError(
            f"window review manifest missing columns: {missing_review_columns}"
        )
    w = windows[cols].copy()
    r = review_manifest[required_review_columns].copy()
    _validate_window_review_overlay_keys(w, r)
    wr = r.merge(
        w,
        on="window_id",
        how="left",
        validate="one_to_one",
    )

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


def _validate_window_review_overlay_keys(
    windows: pd.DataFrame,
    review_manifest: pd.DataFrame,
) -> None:
    """Reject duplicate, blank, or unknown window-review overlay keys."""

    window_ids = windows["window_id"].fillna("").astype(str).str.strip()
    review_ids = (
        review_manifest["window_id"].fillna("").astype(str).str.strip()
    )
    counts = {
        "blank_window_id": int(window_ids.eq("").sum()),
        "duplicate_window_id_rows": int(
            window_ids.duplicated(keep=False).sum()
        ),
        "blank_window_review_id": int(review_ids.eq("").sum()),
        "duplicate_window_review_id_rows": int(
            review_ids.duplicated(keep=False).sum()
        ),
        "unknown_window_review_id_rows": int(
            (~review_ids.isin(set(window_ids))).sum()
        ),
    }
    errors = [f"{name}={count}" for name, count in counts.items() if count]
    if errors:
        raise ValueError("window review overlay key contract failed: " + "; ".join(errors))


def _finalize_unit_review_fields(
    units: pd.DataFrame,
    *,
    include_all_retained_legacy_units: bool = False,
    include_all_retained_native_units: bool = False,
    behavior_selection: BehaviorReviewSelectionConfig | None = None,
) -> pd.DataFrame:
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
    evidence_reason = out.get(
        "review_evidence_reason_auto",
        pd.Series("", index=out.index),
    ).fillna("").astype(str)
    reason = pd.Series(
        [
            _combine_reason_tokens(base_reason, auto_reason)
            for base_reason, auto_reason in zip(
                reason,
                evidence_reason,
                strict=True,
            )
        ],
        index=out.index,
    )
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

    if include_all_retained_legacy_units:
        legacy_mask = out["review_unit_type"].astype(str).eq("legacy_burst_16")
        out.loc[
            legacy_mask & out["review_reason"].astype(str).eq(""),
            "review_reason",
        ] = "full_legacy_native_unit_review"
        out.loc[legacy_mask, "include_in_review"] = True

    priority = pd.to_numeric(out["review_priority_window_max"], errors="coerce").fillna(0.0)
    evidence_priority = pd.to_numeric(
        out.get("review_evidence_priority_auto", 0.0),
        errors="coerce",
    )
    if isinstance(evidence_priority, pd.Series):
        evidence_priority = evidence_priority.fillna(0.0)
    priority = priority + evidence_priority
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

    out, _ = assign_behavior_review_cohorts(
        out,
        config=behavior_selection,
        include_all_retained_legacy_units=include_all_retained_legacy_units,
        include_all_retained_native_units=include_all_retained_native_units,
    )

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
    windows: pd.DataFrame | None,
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
    if windows is not None:
        uncovered_units = int(
            pd.to_numeric(units["affected_window_count"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        if uncovered_units:
            errors.append(f"review_units_without_window_coverage={uncovered_units}")

    if windows is not None:
        window_ids = windows["window_id"].fillna("").astype(str).str.strip()
        if window_ids.eq("").any():
            errors.append(f"blank_window_id={int(window_ids.eq('').sum())}")
        if window_ids.duplicated(keep=False).any():
            errors.append(
                f"duplicate_window_id={int(window_ids.duplicated(keep=False).sum())}"
            )
    if "window_uid" in intervals.columns or (
        windows is not None and "window_uid" in windows.columns
    ):
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


def _combine_reason_tokens(*values: Any) -> str:
    """Join semicolon-delimited review reasons without duplicate tokens."""

    return _join_unique_strings(pd.Series(list(values), dtype="object"))


def _counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).to_dict().items()}

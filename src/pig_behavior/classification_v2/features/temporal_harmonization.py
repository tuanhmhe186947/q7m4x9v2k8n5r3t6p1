"""Temporal label harmonization for classification_v2.

This step sits after enhanced frame-level spatio-temporal features and before
sequence/window manifest construction.

It makes the source-specific label contract explicit:
- legacy_recovered: one recovered legacy burst/tracklet is treated as a
  16-frame constant-behavior temporal unit.
- cvat_tracking_xml / cvat_selected_native: labels are anchor labels with a
  fixed stride, usually 6 frames. Anchor k covers the interval k..k+stride-1.

The function does not correct labels and does not drop rows. It only adds
harmonized metadata, interval-level status, and interaction label policy columns
so later template builders and GUIs do not need to guess source semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.identifiers import scene_frame_key
from pig_behavior.classification_v2.contracts.lineage_claims import (
    add_optional_lineage_claims_to_audit,
    attach_optional_lineage_claims,
    require_lineage_claims_preserved,
    resolve_optional_lineage_claims,
)
from pig_behavior.classification_v2.features.context_policy import (
    normalize_hidden_provenance,
)
from pig_behavior.classification_v2.features.temporal_evidence import (
    attach_unit_evidence_to_intervals,
)

VALID_BEHAVIORS: set[str] = {
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
}

INTERACTION_BEHAVIORS: set[str] = {"fight", "social-nose"}
CVAT_SOURCE_TYPES: set[str] = {"cvat_tracking_xml", "cvat_selected_native"}
LEGACY_SOURCE_TYPE = "legacy_recovered"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_index",
    "pig_id",
    "track_id",
    "behavior",
    "bbox_valid",
)


@dataclass(slots=True)
class TemporalHarmonizationConfig:
    """Configuration for temporal label harmonization."""

    cvat_label_stride: int = 6
    legacy_expected_sequence_length: int = 16
    legacy_min_complete_ratio: float = 1.0

    def validate(self) -> None:
        if self.cvat_label_stride <= 0:
            raise ValueError("cvat_label_stride must be > 0")
        if self.legacy_expected_sequence_length <= 0:
            raise ValueError("legacy_expected_sequence_length must be > 0")
        if not (0 < self.legacy_min_complete_ratio <= 1):
            raise ValueError("legacy_min_complete_ratio must be in (0, 1]")


def harmonize_temporal_labels(
    frame_features: pd.DataFrame,
    *,
    cvat_label_stride: int = 6,
    legacy_expected_sequence_length: int = 16,
    legacy_min_complete_ratio: float = 1.0,
) -> pd.DataFrame:
    """Return frame rows with explicit temporal label metadata.

    The output keeps all input rows and appends harmonization columns. It never
    mutates the original ``behavior`` label. Downstream review/apply steps can
    later replace labels by writing ``behavior_train`` or similar columns.
    """
    resolve_optional_lineage_claims(
        frame_features,
        artifact_name="temporal harmonization input",
    )
    config = TemporalHarmonizationConfig(
        cvat_label_stride=cvat_label_stride,
        legacy_expected_sequence_length=legacy_expected_sequence_length,
        legacy_min_complete_ratio=legacy_min_complete_ratio,
    )
    config.validate()

    missing = [c for c in REQUIRED_COLUMNS if c not in frame_features.columns]
    if missing:
        raise ValueError(f"Missing temporal harmonization input columns: {missing}")

    out = frame_features.copy().reset_index(drop=True)
    out = _normalize_columns(out)
    out = _ensure_object_track_key(out)
    _validate_temporal_identity_contract(out)
    out = _assign_temporal_units(out, config)
    intervals = build_temporal_label_intervals(out, config=config)
    out = _map_interval_columns_to_frames(out, intervals)
    out = _add_interaction_policy_columns(out)
    out = _add_harmonization_quality_columns(out)
    require_lineage_claims_preserved(
        frame_features,
        out,
        source_name="temporal harmonization input",
        derived_name="temporal harmonization output",
    )
    return out


def build_temporal_label_intervals(
    harmonized_or_frame_features: pd.DataFrame,
    *,
    config: TemporalHarmonizationConfig | None = None,
    cvat_label_stride: int = 6,
    legacy_expected_sequence_length: int = 16,
) -> pd.DataFrame:
    """Build one row per temporal label interval/unit.

    This function accepts either raw/enhanced frame features or the output of
    :func:`harmonize_temporal_labels`. If temporal unit columns are missing, they
    are created first.
    """
    claims = resolve_optional_lineage_claims(
        harmonized_or_frame_features,
        artifact_name="temporal interval input",
    )
    if config is None:
        config = TemporalHarmonizationConfig(
            cvat_label_stride=cvat_label_stride,
            legacy_expected_sequence_length=legacy_expected_sequence_length,
        )
    config.validate()

    normalized = _normalize_columns(harmonized_or_frame_features.copy())
    if "temporal_unit_key" not in normalized.columns:
        df = _ensure_object_track_key(normalized)
        _validate_temporal_identity_contract(df)
        df = _assign_temporal_units(df, config)
    else:
        df = _ensure_object_track_key(normalized)
        _validate_temporal_identity_contract(df)

    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce")
    if "timestamp_sec" in df.columns:
        df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
    else:
        df["timestamp_sec"] = np.nan

    bool_cols = [
        "bbox_valid",
        "hidden",
        "hidden_is_trusted",
        "spatiotemporal_feature_valid",
    ]
    for col in bool_cols:
        if col not in df.columns:
            if col == "hidden":
                df[col] = False
            elif col == "hidden_is_trusted":
                df[col] = _default_hidden_trust(df)
            else:
                df[col] = True
        df[col] = _to_bool_series(df[col])

    rows: list[dict[str, Any]] = []
    group_cols = ["temporal_unit_key"]
    for unit_key, g in df.groupby(group_cols, dropna=False, sort=False):
        unit_key = str(unit_key[0] if isinstance(unit_key, tuple) else unit_key)
        first = g.iloc[0]
        behavior_values = [b for b in g["behavior"].dropna().astype(str).tolist() if b]
        unique_behaviors = sorted(set(behavior_values))
        behavior_counts = g["behavior"].dropna().astype(str).value_counts()
        dominant_behavior = str(behavior_counts.idxmax()) if not behavior_counts.empty else ""
        num_behaviors = int(len(unique_behaviors))

        source_type = str(first.get("source_type", ""))
        is_cvat = source_type in CVAT_SOURCE_TYPES
        is_legacy = source_type == LEGACY_SOURCE_TYPE

        label_start = _first_valid_numeric(g.get("label_window_start"), default=np.nan)
        label_end = _first_valid_numeric(g.get("label_window_end"), default=np.nan)
        anchor = _first_valid_numeric(g.get("label_anchor_frame_index"), default=np.nan)

        if not np.isfinite(label_start):
            label_start = float(g["frame_index"].min())
        if not np.isfinite(label_end):
            label_end = float(g["frame_index"].max())
        if not np.isfinite(anchor):
            anchor = label_start

        label_frame_count = (
            int(label_end - label_start + 1)
            if np.isfinite(label_start) and np.isfinite(label_end)
            else 0
        )
        observed_frame_count = int(g["frame_index"].nunique(dropna=True))
        expected_count = (
            config.cvat_label_stride
            if is_cvat
            else config.legacy_expected_sequence_length
            if is_legacy
            else label_frame_count
        )
        # CVAT may only have anchor rows. Completeness means interval assignment is valid;
        # dense bbox observation is tracked separately by observed_frame_count.
        if is_cvat:
            interval_complete = label_frame_count == config.cvat_label_stride
        elif is_legacy:
            interval_complete = observed_frame_count >= int(
                np.ceil(config.legacy_expected_sequence_length * config.legacy_min_complete_ratio)
            )
        else:
            interval_complete = observed_frame_count >= max(1, expected_count)

        anchor_behavior = ""
        if is_cvat:
            anchor_behavior = _anchor_behavior_for_interval(g, anchor)
            final_behavior = anchor_behavior if anchor_behavior in VALID_BEHAVIORS else ""
            # CVAT labels are anchor labels: anchor k applies to frames k..k+stride-1.
            # Raw frame labels inside the interval are retained for audit, but they must
            # not turn a valid anchor-labeled interval into "mixed".
            if not final_behavior:
                status = "uncertain"
            elif interval_complete:
                status = "stable"
            else:
                status = "incomplete"
            behavior_consistent = bool(final_behavior) and status == "stable"
        else:
            anchor_behavior = str(first.get("behavior", "")) if is_legacy else ""
            final_behavior = dominant_behavior if num_behaviors == 1 else ""
            if num_behaviors == 0:
                status = "uncertain"
            elif num_behaviors == 1 and interval_complete:
                status = "stable"
            elif num_behaviors == 1 and not interval_complete:
                status = "incomplete"
            else:
                status = "mixed"
            behavior_consistent = bool(num_behaviors == 1)

        hidden_raw = _to_bool_series(g["hidden"])
        hidden_trust = _to_bool_series(g["hidden_is_trusted"])
        hidden_effective = hidden_raw & hidden_trust
        hidden_ratio_raw = float(hidden_raw.mean()) if len(g) else 0.0
        hidden_ratio = float(hidden_effective.mean()) if len(g) else 0.0
        hidden_untrusted_ratio = float((hidden_raw & ~hidden_trust).mean()) if len(g) else 0.0
        hidden_review_coverage = float(hidden_trust.mean()) if len(g) else 0.0
        bbox_valid_ratio = float(_to_bool_series(g["bbox_valid"]).mean()) if len(g) else 0.0
        spatio_valid_ratio = (
            float(_to_bool_series(g["spatiotemporal_feature_valid"]).mean()) if len(g) else 0.0
        )

        rows.append(
            {
                "temporal_unit_key": unit_key,
                "source_type": source_type,
                "dataset_id": str(first.get("dataset_id", "")),
                "video_key": str(first.get("video_key", "")),
                "object_track_key": str(first.get("object_track_key", "")),
                "pig_id": str(first.get("pig_id", "")),
                "track_id": str(first.get("track_id", "")),
                "temporal_label_mode": str(first.get("temporal_label_mode", "unknown_temporal")),
                "label_anchor_frame_index": _nullable_int(anchor),
                "label_window_start": _nullable_int(label_start),
                "label_window_end": _nullable_int(label_end),
                "label_frame_count": label_frame_count,
                "observed_frame_start": _nullable_int(float(g["frame_index"].min())),
                "observed_frame_end": _nullable_int(float(g["frame_index"].max())),
                "observed_frame_count": observed_frame_count,
                "expected_observed_frame_count": int(expected_count) if expected_count else 0,
                "temporal_interval_complete": bool(interval_complete),
                "num_behaviors_in_interval": num_behaviors,
                "unique_behaviors_in_interval": "|".join(unique_behaviors),
                "dominant_behavior_in_interval": final_behavior or dominant_behavior,
                "anchor_behavior_in_interval": anchor_behavior,
                "raw_num_behaviors_in_interval": num_behaviors,
                "raw_unique_behaviors_in_interval": "|".join(unique_behaviors),
                "raw_dominant_behavior_in_interval": dominant_behavior,
                "raw_behavior_consistency_in_interval": bool(num_behaviors == 1),
                "behavior_temporal_final": final_behavior,
                "temporal_consistency_status": status,
                "behavior_consistency_in_interval": behavior_consistent,
                "timestamp_start_sec": _first_valid_numeric(
                    g.get("timestamp_sec"), default=np.nan, how="min"
                ),
                "timestamp_end_sec": _first_valid_numeric(
                    g.get("timestamp_sec"), default=np.nan, how="max"
                ),
                "bbox_valid_ratio_interval": bbox_valid_ratio,
                "hidden_ratio_interval": hidden_ratio,
                "visible_ratio_interval": 1.0 - hidden_ratio,
                "hidden_ratio_raw_interval": hidden_ratio_raw,
                "hidden_ratio_trusted_interval": hidden_ratio,
                "hidden_metadata_untrusted_ratio_interval": (hidden_untrusted_ratio),
                "hidden_review_coverage_ratio_interval": hidden_review_coverage,
                "spatiotemporal_feature_valid_ratio_interval": spatio_valid_ratio,
                "interval_review_reason": _interval_review_reason(
                    status,
                    final_behavior,
                    bbox_valid_ratio,
                    hidden_ratio,
                    hidden_untrusted_ratio,
                    spatio_valid_ratio,
                ),
            }
        )

    intervals = pd.DataFrame(rows)
    if intervals.empty:
        return attach_optional_lineage_claims(intervals, claims)

    intervals = attach_unit_evidence_to_intervals(intervals, df)
    intervals = _add_interaction_policy_columns(
        intervals, behavior_col="dominant_behavior_in_interval"
    )
    intervals = intervals.sort_values(
        [
            c
            for c in [
                "source_type",
                "dataset_id",
                "video_key",
                "object_track_key",
                "label_window_start",
            ]
            if c in intervals.columns
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    intervals = attach_optional_lineage_claims(intervals, claims)
    require_lineage_claims_preserved(
        harmonized_or_frame_features,
        intervals,
        source_name="temporal interval input",
        derived_name="temporal interval output",
    )
    return intervals


def audit_temporal_harmonization(
    df: pd.DataFrame, intervals: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Return a compact audit summary."""
    errors: list[str] = []
    warnings: list[str] = []

    required_new = [
        "object_track_key",
        "temporal_unit_key",
        "temporal_label_mode",
        "label_window_start",
        "label_window_end",
        "behavior_original_frame",
        "behavior_temporal_final",
        "temporal_consistency_status",
        "label_propagation_policy",
    ]
    missing_new = [c for c in required_new if c not in df.columns]
    if missing_new:
        errors.append(f"missing_harmonized_columns={missing_new}")

    if "behavior" in df.columns:
        invalid = sorted(set(df["behavior"].dropna().astype(str)).difference(VALID_BEHAVIORS))
        if invalid:
            warnings.append(f"invalid_or_unknown_behaviors={invalid}")

    if {"label_window_start", "label_window_end"}.issubset(df.columns):
        invalid_window = int(
            (
                pd.to_numeric(df["label_window_end"], errors="coerce")
                < pd.to_numeric(df["label_window_start"], errors="coerce")
            ).sum()
        )
        if invalid_window:
            errors.append(f"invalid_temporal_window_count={invalid_window}")

    if intervals is not None and not intervals.empty:
        interval_status = _value_counts_dict(intervals, "temporal_consistency_status")
        incomplete_or_mixed = int(
            interval_status.get("mixed", 0)
            + interval_status.get("incomplete", 0)
            + interval_status.get("uncertain", 0)
        )
        if incomplete_or_mixed:
            warnings.append(f"intervals_need_review_or_exclusion={incomplete_or_mixed}")
    else:
        interval_status = {}

    audit = {
        "rows": int(len(df)),
        "frames": int(scene_frame_key(df).nunique(dropna=True)),
        "frame_objects": int(df["frame_uid"].nunique(dropna=True))
        if "frame_uid" in df.columns
        else 0,
        "temporal_intervals": int(len(intervals)) if intervals is not None else None,
        "sources": _value_counts_dict(df, "source_type"),
        "behaviors": _value_counts_dict(df, "behavior"),
        "temporal_label_mode": _value_counts_dict(df, "temporal_label_mode"),
        "temporal_consistency_status_frame_rows": _value_counts_dict(
            df, "temporal_consistency_status"
        ),
        "temporal_consistency_status_intervals": interval_status,
        "label_propagation_policy": _value_counts_dict(df, "label_propagation_policy"),
        "allow_label_propagation": _value_counts_dict(df, "allow_label_propagation"),
        "hidden_review_status": _value_counts_dict(df, "hidden_review_status"),
        "hidden_trust_status": _value_counts_dict(df, "hidden_trust_status"),
        "hidden_is_trusted": _value_counts_dict(df, "hidden_is_trusted"),
        "errors": errors,
        "warnings": warnings,
    }
    audit = add_optional_lineage_claims_to_audit(
        audit,
        df,
        artifact_name="temporal harmonization audit frame table",
    )
    if intervals is not None:
        try:
            require_lineage_claims_preserved(
                df,
                intervals,
                source_name="temporal harmonization audit frames",
                derived_name="temporal harmonization audit intervals",
            )
        except ValueError as exc:
            audit["errors"].append(f"lineage_claim_contract={exc}")
    return audit


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "frame_index",
        "relative_frame_index",
        "timestamp_sec",
        "label_window_start",
        "label_window_end",
        "label_anchor_frame_index",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in [
        "source_type",
        "dataset_id",
        "video_key",
        "scene_frame_uid",
        "frame_uid",
        "pig_id",
        "track_id",
        "behavior",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    if "bbox_valid" not in out.columns:
        out["bbox_valid"] = True
    out["bbox_valid"] = _to_bool_series(out["bbox_valid"])
    if "hidden" in out.columns:
        out["hidden"] = _to_bool_series(out["hidden"])
    else:
        out["hidden"] = False
    return normalize_hidden_provenance(out)


def _ensure_object_track_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    track_for_key = (
        out["track_id"].replace("", pd.NA).fillna(out["pig_id"]).astype(str)
    )
    pig_for_key = (
        out["pig_id"].replace("", pd.NA).fillna(out["track_id"]).astype(str)
    )
    derived_key = (
        out["source_type"].astype(str)
        + "|"
        + out["dataset_id"].astype(str)
        + "|"
        + out["video_key"].astype(str)
        + "|track="
        + track_for_key
        + "|pig="
        + pig_for_key
    )
    if "object_track_key" not in out.columns:
        out["object_track_key"] = derived_key
        return out

    current_key = out["object_track_key"].fillna("").astype(str).str.strip()
    missing_key = current_key.eq("")
    out["object_track_key"] = current_key.where(~missing_key, derived_key)
    return out


def _validate_temporal_identity_contract(df: pd.DataFrame) -> None:
    """Reject rows that cannot form unique source-local temporal trajectories."""
    key = df["object_track_key"].fillna("").astype(str).str.strip()
    frame_index = pd.to_numeric(df["frame_index"], errors="coerce")
    track = df["track_id"].fillna("").astype(str).str.strip()
    pig = df["pig_id"].fillna("").astype(str).str.strip()
    invalid = (
        key.eq("")
        | (track.eq("") & pig.eq(""))
        | frame_index.isna()
        | frame_index.mod(1).ne(0)
        | frame_index.lt(0)
    )
    duplicate = pd.DataFrame(
        {
            "object_track_key": key,
            "frame_index": frame_index,
        }
    ).duplicated(keep=False)
    duplicate &= ~invalid
    if invalid.any() or duplicate.any():
        affected = invalid | duplicate
        sample = [str(value) for value in df.index[affected].tolist()[:10]]
        raise ValueError(
            "Temporal identity contract failed: "
            f"invalid_rows={int(invalid.sum())}, "
            f"duplicate_track_frame_rows={int(duplicate.sum())}, "
            f"sample_source_indices={sample}"
        )


def _assign_temporal_units(df: pd.DataFrame, config: TemporalHarmonizationConfig) -> pd.DataFrame:
    out = df.copy()
    frame_idx = pd.to_numeric(out["frame_index"], errors="coerce")
    source = out["source_type"].astype(str)
    cvat_mask = source.isin(CVAT_SOURCE_TYPES)
    legacy_mask = source.eq(LEGACY_SOURCE_TYPE)

    # Preserve existing columns when present, but recompute when missing/empty.
    if "temporal_label_mode" not in out.columns:
        out["temporal_label_mode"] = "unknown_temporal"
    out["temporal_label_mode"] = out["temporal_label_mode"].fillna("").astype(str)
    out.loc[cvat_mask, "temporal_label_mode"] = f"cvat_anchor_{config.cvat_label_stride}f_interval"
    out.loc[legacy_mask, "temporal_label_mode"] = (
        f"legacy_{config.legacy_expected_sequence_length}f_constant"
    )

    cvat_anchor = np.floor(frame_idx / config.cvat_label_stride) * config.cvat_label_stride

    if "label_anchor_frame_index" not in out.columns:
        out["label_anchor_frame_index"] = np.nan
    if "label_window_start" not in out.columns:
        out["label_window_start"] = np.nan
    if "label_window_end" not in out.columns:
        out["label_window_end"] = np.nan

    out.loc[cvat_mask, "label_anchor_frame_index"] = cvat_anchor.loc[cvat_mask]
    out.loc[cvat_mask, "label_window_start"] = cvat_anchor.loc[cvat_mask]
    out.loc[cvat_mask, "label_window_end"] = (
        cvat_anchor.loc[cvat_mask] + config.cvat_label_stride - 1
    )

    if legacy_mask.any():
        if "relative_frame_index" in out.columns:
            rel = pd.to_numeric(out["relative_frame_index"], errors="coerce")
            legacy_anchor = frame_idx - rel
        else:
            legacy_anchor = frame_idx.groupby(out["object_track_key"]).transform("min")
        out.loc[legacy_mask, "label_anchor_frame_index"] = legacy_anchor.loc[legacy_mask]

    if "temporal_unit_key" not in out.columns:
        out["temporal_unit_key"] = ""
    out["temporal_unit_key"] = out["temporal_unit_key"].fillna("").astype(str)

    out.loc[cvat_mask, "temporal_unit_key"] = (
        out.loc[cvat_mask, "object_track_key"].astype(str)
        + "|anchor="
        + out.loc[cvat_mask, "label_anchor_frame_index"].round().astype("Int64").astype(str)
    )
    out.loc[legacy_mask, "temporal_unit_key"] = (
        out.loc[legacy_mask, "object_track_key"].astype(str) + "|legacy_sequence"
    )
    other_mask = ~(cvat_mask | legacy_mask)
    out.loc[other_mask & out["temporal_unit_key"].eq(""), "temporal_unit_key"] = (
        out.loc[other_mask & out["temporal_unit_key"].eq(""), "object_track_key"].astype(str)
        + "|frame="
        + frame_idx.loc[other_mask & out["temporal_unit_key"].eq("")]
        .round()
        .astype("Int64")
        .astype(str)
    )

    # Legacy window start/end from actual sequence span.
    unit_min = out.groupby("temporal_unit_key", dropna=False)["frame_index"].transform("min")
    unit_max = out.groupby("temporal_unit_key", dropna=False)["frame_index"].transform("max")
    missing_start = out["label_window_start"].isna()
    missing_end = out["label_window_end"].isna()
    out.loc[missing_start, "label_window_start"] = unit_min.loc[missing_start]
    out.loc[missing_end, "label_window_end"] = unit_max.loc[missing_end]

    out["label_anchor_frame_index"] = (
        pd.to_numeric(out["label_anchor_frame_index"], errors="coerce").round().astype("Int64")
    )
    out["label_window_start"] = (
        pd.to_numeric(out["label_window_start"], errors="coerce").round().astype("Int64")
    )
    out["label_window_end"] = (
        pd.to_numeric(out["label_window_end"], errors="coerce").round().astype("Int64")
    )
    return out


def _map_interval_columns_to_frames(df: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if intervals.empty:
        out["behavior_original_frame"] = out["behavior"]
        out["behavior_temporal_final"] = ""
        out["temporal_consistency_status"] = "uncertain"
        return out

    map_cols = [
        "behavior_temporal_final",
        "temporal_consistency_status",
        "behavior_consistency_in_interval",
        "num_behaviors_in_interval",
        "unique_behaviors_in_interval",
        "dominant_behavior_in_interval",
        "anchor_behavior_in_interval",
        "raw_num_behaviors_in_interval",
        "raw_unique_behaviors_in_interval",
        "raw_dominant_behavior_in_interval",
        "raw_behavior_consistency_in_interval",
        "temporal_interval_complete",
        "observed_frame_count",
        "label_frame_count",
        "bbox_valid_ratio_interval",
        "hidden_ratio_interval",
        "visible_ratio_interval",
        "hidden_ratio_raw_interval",
        "hidden_ratio_trusted_interval",
        "hidden_metadata_untrusted_ratio_interval",
        "hidden_review_coverage_ratio_interval",
        "spatiotemporal_feature_valid_ratio_interval",
        "interval_review_reason",
    ]
    idx = intervals.set_index("temporal_unit_key")
    for col in map_cols:
        if col in idx.columns:
            out[col] = out["temporal_unit_key"].map(idx[col])
    out["behavior_original_frame"] = out["behavior"]
    out["temporal_harmonization_valid"] = (
        out["temporal_unit_key"].astype(str).ne("") & out["behavior_temporal_final"].notna()
    )
    return out


def _anchor_behavior_for_interval(g: pd.DataFrame, anchor: float | int | None) -> str:
    """Return the canonical behavior on the CVAT anchor frame.

    CVAT behavior labels in this project are sparse anchor labels. The label on
    frame k applies to the whole interval k..k+stride-1. Raw labels on non-anchor
    rows may contain propagated/default/old values and are kept only for audit.
    """
    if anchor is None:
        return ""
    try:
        anchor_value = int(round(float(anchor)))
    except Exception:
        return ""

    frame_idx = pd.to_numeric(g.get("frame_index"), errors="coerce")
    anchor_rows = g[frame_idx.eq(anchor_value)]
    if anchor_rows.empty:
        return ""

    values = anchor_rows.get("behavior", pd.Series([], dtype=object)).dropna().astype(str)
    values = [v.strip() for v in values.tolist() if v.strip()]
    for value in values:
        if value in VALID_BEHAVIORS:
            return value
    return values[0] if values else ""


def _add_interaction_policy_columns(
    df: pd.DataFrame, behavior_col: str = "behavior"
) -> pd.DataFrame:
    out = df.copy()
    behavior = (
        out.get(behavior_col, out.get("behavior", pd.Series("", index=out.index)))
        .fillna("")
        .astype(str)
    )

    out["interaction_annotation_policy"] = "not_interaction"
    out["interaction_role_policy"] = "none"
    out["label_propagation_policy"] = "none"
    out["allow_label_propagation"] = False
    out["requires_partner_context"] = False
    out["social_nose_actor_only"] = False
    out["fight_group_label"] = False

    fight = behavior.eq("fight")
    social = behavior.eq("social-nose")

    out.loc[fight, "interaction_annotation_policy"] = "fight_directly_involved_group"
    out.loc[fight, "interaction_role_policy"] = "attacker_or_target_reacting_or_directly_involved"
    out.loc[fight, "label_propagation_policy"] = "directly_involved_pigs"
    out.loc[fight, "allow_label_propagation"] = True
    out.loc[fight, "requires_partner_context"] = True
    out.loc[fight, "fight_group_label"] = True

    out.loc[social, "interaction_annotation_policy"] = "social_nose_active_actor_only"
    out.loc[social, "interaction_role_policy"] = "active_snout_actor_only"
    out.loc[social, "label_propagation_policy"] = "actor_only"
    out.loc[social, "allow_label_propagation"] = False
    out.loc[social, "requires_partner_context"] = True
    out.loc[social, "social_nose_actor_only"] = True
    return out


def _add_harmonization_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    status = (
        out.get("temporal_consistency_status", pd.Series("uncertain", index=out.index))
        .fillna("uncertain")
        .astype(str)
    )
    out["temporal_unit_stable_for_training"] = status.eq("stable")
    out["temporal_unit_needs_review"] = status.isin({"mixed", "incomplete", "uncertain"})
    out["sequence_label_status_unit"] = np.select(
        [status.eq("stable"), status.eq("mixed"), status.eq("incomplete"), status.eq("uncertain")],
        ["stable", "mixed", "incomplete", "uncertain"],
        default="uncertain",
    )
    return out


def _interval_review_reason(
    status: str,
    behavior: str,
    bbox_ratio: float,
    hidden_ratio: float,
    hidden_untrusted_ratio: float,
    spatio_ratio: float,
) -> str:
    reasons: list[str] = []
    if status != "stable":
        reasons.append(f"temporal_{status}")
    if bbox_ratio < 1.0:
        reasons.append("bbox_invalid_in_interval")
    if hidden_ratio > 0.5:
        reasons.append("high_hidden_ratio_interval")
    if hidden_untrusted_ratio > 0:
        reasons.append("untrusted_hidden_metadata_interval")
    if spatio_ratio < 1.0:
        reasons.append("spatiotemporal_feature_invalid_in_interval")
    if behavior in INTERACTION_BEHAVIORS:
        reasons.append("interaction_requires_partner_context")
    return ";".join(reasons)


def _default_hidden_trust(df: pd.DataFrame) -> pd.Series:
    """Infer backward-compatible trust without trusting CVAT tracking output."""
    source = (
        df.get(
            "source_type",
            pd.Series("", index=df.index),
        )
        .fillna("")
        .astype(str)
    )
    return source.eq(LEGACY_SOURCE_TYPE)


def _first_valid_numeric(s: Any, default: float = np.nan, how: str = "first") -> float:
    if s is None:
        return default
    if not isinstance(s, pd.Series):
        try:
            return float(s)
        except Exception:
            return default
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return default
    if how == "min":
        return float(vals.min())
    if how == "max":
        return float(vals.max())
    return float(vals.iloc[0])


def _nullable_int(x: float | int | None) -> Any:
    if x is None:
        return pd.NA
    try:
        if not np.isfinite(float(x)):
            return pd.NA
        return int(round(float(x)))
    except Exception:
        return pd.NA


def _to_bool_series(s: pd.Series | Iterable[Any]) -> pd.Series:
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("<NA>").astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}

"""Enhanced spatio-temporal feature builder for classification_v2.

This module is intentionally placed after geometry/context/ROI feature construction.
It does not drop rows and it does not correct labels. Its job is to enrich the
frame-object CSV with review/training features that describe temporal motion,
shape dynamics, continuous ROI relations, and local social context.

Typical input:
    outputs/classification_v2/frame_features/spatiotemporal_frame_features_roi.csv

Typical output:
    outputs/classification_v2/frame_features/spatiotemporal_frame_features_enhanced.csv

Design rules:
- Preserve every input row.
- Do not change ``behavior``.
- Add deterministic temporal unit metadata for legacy 16f and CVAT 6f intervals.
- Add row-level and temporal-unit aggregate features useful for review templates.
- Keep features explicit so later review/template/sequence steps do not need to
  infer source-specific timing rules again.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.context_policy import (
    normalize_hidden_provenance,
)
from pig_behavior.classification_v2.features.temporal_evidence import (
    TemporalEvidenceConfig,
    add_unit_temporal_evidence,
)

ROI_BEHAVIORS: set[str] = {"eat", "drink", "playwithtoy"}
INTERACTION_BEHAVIORS: set[str] = {"fight", "social-nose"}
MOTION_BEHAVIORS: set[str] = {"move", "explore", "stand"}
SHAPE_BEHAVIORS: set[str] = {"lying", "sitting"}
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

ROI_CLASSES: tuple[str, ...] = ("feeder", "drinker", "toy")
BEHAVIOR_TO_TARGET_ROI: dict[str, str] = {
    "eat": "feeder",
    "drink": "drinker",
    "playwithtoy": "toy",
}

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_uid",
    "frame_index",
    "pig_id",
    "track_id",
    "behavior",
    "bbox_valid",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "frame_index",
    "relative_frame_index",
    "timestamp_sec",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "cx",
    "cy",
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
    "box_diag_n",
    "roi_target_min_dist_n",
    "roi_target_min_dist_px",
    "roi_target_max_overlap_ratio",
    "roi_target_max_iou",
)

BOOL_COLUMNS: tuple[str, ...] = (
    "bbox_valid",
    "roi_feature_required",
    "roi_target_available",
    "roi_target_near",
    "roi_target_contact",
    "roi_target_center_inside",
    "sequence_complete",
    "sequence_range_valid",
)


@dataclass(slots=True)
class EnhancedFeatureConfig:
    """Configuration for enhanced spatio-temporal feature extraction."""

    cvat_label_stride: int = 6
    legacy_expected_sequence_length: int = 16
    social_near_distance_n: float = 0.08
    social_contact_iou_threshold: float = 0.01
    social_contact_overlap_threshold: float = 0.05
    max_frame_group_size_for_social: int = 64
    stationary_speed_threshold: float = 0.002
    active_speed_threshold: float = 0.006
    turning_angle_threshold_rad: float = float(np.pi / 6.0)

    def validate(self) -> None:
        if self.cvat_label_stride <= 0:
            raise ValueError("cvat_label_stride must be > 0")
        if self.legacy_expected_sequence_length <= 0:
            raise ValueError("legacy_expected_sequence_length must be > 0")
        if self.social_near_distance_n <= 0:
            raise ValueError("social_near_distance_n must be > 0")
        if self.social_contact_iou_threshold < 0:
            raise ValueError("social_contact_iou_threshold must be >= 0")
        if self.social_contact_overlap_threshold < 0:
            raise ValueError("social_contact_overlap_threshold must be >= 0")
        if self.max_frame_group_size_for_social <= 1:
            raise ValueError("max_frame_group_size_for_social must be > 1")
        self.temporal_evidence_config().validate()

    def temporal_evidence_config(self) -> TemporalEvidenceConfig:
        """Return the shared unit/window evidence threshold contract."""

        return TemporalEvidenceConfig(
            stationary_speed_threshold=self.stationary_speed_threshold,
            active_speed_threshold=self.active_speed_threshold,
            turning_angle_threshold_rad=self.turning_angle_threshold_rad,
        )


def build_enhanced_spatiotemporal_features(
    frame_features: pd.DataFrame,
    *,
    cvat_label_stride: int = 6,
    legacy_expected_sequence_length: int = 16,
    social_near_distance_n: float = 0.08,
    social_contact_iou_threshold: float = 0.01,
    social_contact_overlap_threshold: float = 0.05,
    stationary_speed_threshold: float = 0.002,
    active_speed_threshold: float = 0.006,
    turning_angle_threshold_rad: float = float(np.pi / 6.0),
) -> pd.DataFrame:
    """Add enhanced spatio-temporal, ROI-duration, and social-context features.

    The returned dataframe has the same number of rows and keeps all original
    columns. New columns are appended.
    """
    config = EnhancedFeatureConfig(
        cvat_label_stride=cvat_label_stride,
        legacy_expected_sequence_length=legacy_expected_sequence_length,
        social_near_distance_n=social_near_distance_n,
        social_contact_iou_threshold=social_contact_iou_threshold,
        social_contact_overlap_threshold=social_contact_overlap_threshold,
        stationary_speed_threshold=stationary_speed_threshold,
        active_speed_threshold=active_speed_threshold,
        turning_angle_threshold_rad=turning_angle_threshold_rad,
    )
    config.validate()

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame_features.columns]
    if missing:
        raise ValueError(f"Missing enhanced spatiotemporal input columns: {missing}")

    out = frame_features.copy().reset_index(drop=True)
    out = _normalize_basic_columns(out)
    out = _add_behavior_group_columns(out)
    out = _add_temporal_unit_columns(out, config)
    out = _add_temporal_deltas(out)
    out = _add_roi_temporal_columns(out)
    out = _add_social_context_columns(out, config)
    out = _add_temporal_unit_aggregates(out)
    out = add_unit_temporal_evidence(
        out,
        config=config.temporal_evidence_config(),
    )
    out = _add_review_helper_columns(out)

    return out


def audit_enhanced_spatiotemporal_features(df: pd.DataFrame) -> dict[str, Any]:
    """Return audit summary for enhanced spatio-temporal features."""
    errors: list[str] = []
    warnings: list[str] = []

    missing_required = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing_required:
        errors.append(f"missing_required_input_columns={missing_required}")

    required_new = [
        "object_track_key",
        "temporal_unit_key",
        "temporal_label_mode",
        "label_window_start",
        "label_window_end",
        "source_sequence_length",
        "behavior_consistency_in_unit",
        "speed_n_per_frame",
        "motion_energy_unit",
        "target_roi_contact_ratio_unit",
        "nearest_pig_id",
        "nearest_dist_n",
        "social_density_near_count",
        "motion_active_ratio_unit",
        "roi_feeder_contact_ratio_unit",
        "social_partner_persistence_ratio_unit",
        "spatiotemporal_feature_valid",
    ]
    missing_new = [c for c in required_new if c not in df.columns]
    if missing_new:
        errors.append(f"missing_enhanced_columns={missing_new}")

    if "behavior" in df.columns:
        invalid_behaviors = sorted(
            set(df["behavior"].dropna().astype(str)).difference(
                VALID_BEHAVIORS
            )
        )
        if invalid_behaviors:
            warnings.append(f"invalid_or_unknown_behaviors={invalid_behaviors}")

    invalid_temporal = 0
    if {"label_window_start", "label_window_end"}.issubset(df.columns):
        invalid_temporal = int(
            (
                pd.to_numeric(df["label_window_end"], errors="coerce")
                < pd.to_numeric(df["label_window_start"], errors="coerce")
            ).sum()
        )
        if invalid_temporal:
            errors.append(f"invalid_temporal_window_count={invalid_temporal}")

    feature_valid_count = _value_counts_dict(df, "spatiotemporal_feature_valid")
    if feature_valid_count.get("False", 0) > 0:
        warnings.append(
            "some_rows_have_spatiotemporal_feature_valid_false="
            f"{feature_valid_count.get('False', 0)}"
        )

    return {
        "rows": int(len(df)),
        "frames": int(df["frame_uid"].nunique(dropna=True)) if "frame_uid" in df.columns else 0,
        "sources": _value_counts_dict(df, "source_type"),
        "datasets": _value_counts_dict(df, "dataset_id"),
        "behaviors": _value_counts_dict(df, "behavior"),
        "behavior_review_group": _value_counts_dict(df, "behavior_review_group"),
        "temporal_label_mode": _value_counts_dict(df, "temporal_label_mode"),
        "preferred_review_gui_auto": _value_counts_dict(df, "preferred_review_gui_auto"),
        "behavior_consistency_in_unit": _value_counts_dict(df, "behavior_consistency_in_unit"),
        "source_sequence_complete_auto": _value_counts_dict(df, "source_sequence_complete_auto"),
        "spatiotemporal_feature_valid": feature_valid_count,
        "roi_target_contact_ratio_unit": _numeric_summary(df, "target_roi_contact_ratio_unit"),
        "speed_n_per_frame": _numeric_summary(df, "speed_n_per_frame"),
        "nearest_dist_n": _numeric_summary(df, "nearest_dist_n"),
        "motion_energy_unit": _numeric_summary(df, "motion_energy_unit"),
        "social_density_near_count": _numeric_summary(df, "social_density_near_count"),
        "motion_active_ratio_unit": _numeric_summary(
            df,
            "motion_active_ratio_unit",
        ),
        "roi_feeder_contact_ratio_unit": _numeric_summary(
            df,
            "roi_feeder_contact_ratio_unit",
        ),
        "social_partner_persistence_ratio_unit": _numeric_summary(
            df,
            "social_partner_persistence_ratio_unit",
        ),
        "new_feature_columns": [
            c
            for c in df.columns
            if c.startswith(
                (
                    "temporal_",
                    "label_",
                    "source_sequence_",
                    "behavior_consistency",
                    "num_behaviors",
                    "dominant_behavior",
                    "delta_",
                    "speed_",
                    "accel_",
                    "direction_",
                    "motion_",
                    "path_",
                    "displacement_",
                    "target_roi_",
                    "roi_",
                    "nearest_",
                    "social_",
                    "pair_",
                    "approach_",
                    "separation_",
                    "bbox_stability",
                    "shape_transition",
                    "preferred_review_",
                    "review_feature_",
                    "spatiotemporal_",
                )
            )
        ],
        "errors": errors,
        "warnings": warnings,
    }


def _normalize_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"source_type", "hidden"}.issubset(out.columns):
        out = normalize_hidden_provenance(out)

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in BOOL_COLUMNS:
        if col in out.columns:
            out[col] = _to_bool_series(out[col])

    for col in [
        "source_type",
        "dataset_id",
        "video_key",
        "frame_uid",
        "pig_id",
        "track_id",
        "behavior",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    # Avoid empty track/pig identifiers breaking grouping. These are not used to
    # overwrite the original identity columns, only to build stable review keys.
    out["_track_for_key"] = out["track_id"].replace("", pd.NA).fillna(out["pig_id"])
    out["_pig_for_key"] = out["pig_id"].replace("", pd.NA).fillna(out["track_id"])

    out["object_track_key"] = (
        out["source_type"].astype(str)
        + "|"
        + out["dataset_id"].astype(str)
        + "|"
        + out["video_key"].astype(str)
        + "|track="
        + out["_track_for_key"].astype(str)
        + "|pig="
        + out["_pig_for_key"].astype(str)
    )
    return out


def _add_behavior_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    behavior = out["behavior"].astype(str)

    groups: list[str] = []
    for b in behavior:
        labels = []
        if b in ROI_BEHAVIORS:
            labels.append("roi")
        if b in MOTION_BEHAVIORS:
            labels.append("motion")
        if b in INTERACTION_BEHAVIORS:
            labels.append("interaction")
        if b in SHAPE_BEHAVIORS:
            labels.append("posture")
        groups.append("+".join(labels) if labels else "other")

    out["behavior_review_group"] = groups
    out["is_roi_behavior"] = behavior.isin(ROI_BEHAVIORS)
    out["is_motion_behavior"] = behavior.isin(MOTION_BEHAVIORS)
    out["is_interaction_behavior"] = behavior.isin(INTERACTION_BEHAVIORS)
    out["is_shape_behavior"] = behavior.isin(SHAPE_BEHAVIORS)
    return out


def _add_temporal_unit_columns(df: pd.DataFrame, config: EnhancedFeatureConfig) -> pd.DataFrame:
    out = df.copy()

    frame_idx = pd.to_numeric(out["frame_index"], errors="coerce")
    source_type = out["source_type"].astype(str)

    cvat_mask = source_type.eq("cvat_tracking_xml") | source_type.eq("cvat_selected_native")
    legacy_mask = source_type.eq("legacy_recovered")

    anchor = pd.Series(np.nan, index=out.index, dtype="float64")
    window_start = pd.Series(np.nan, index=out.index, dtype="float64")
    window_end = pd.Series(np.nan, index=out.index, dtype="float64")
    label_mode = pd.Series("unknown_temporal", index=out.index, dtype="object")

    cvat_anchor = np.floor(frame_idx / config.cvat_label_stride) * config.cvat_label_stride
    anchor.loc[cvat_mask] = cvat_anchor.loc[cvat_mask]
    window_start.loc[cvat_mask] = cvat_anchor.loc[cvat_mask]
    window_end.loc[cvat_mask] = cvat_anchor.loc[cvat_mask] + config.cvat_label_stride - 1
    label_mode.loc[cvat_mask] = f"cvat_anchor_{config.cvat_label_stride}f_interval"

    # Legacy: each recovered tracklet is intended to be a constant behavior unit.
    # We derive the actual window from rows available for the object_track_key.
    if legacy_mask.any():
        rel = pd.to_numeric(
            out.get(
                "relative_frame_index",
                pd.Series(np.nan, index=out.index),
            ),
            errors="coerce",
        )
        # Prefer actual relative-frame metadata when available.
        legacy_anchor = frame_idx - rel
        anchor.loc[legacy_mask] = legacy_anchor.loc[legacy_mask]
        label_mode.loc[legacy_mask] = f"legacy_{config.legacy_expected_sequence_length}f_constant"

    out["temporal_label_mode"] = label_mode
    out["label_anchor_frame_index"] = anchor.round().astype("Int64")
    out["label_window_start"] = window_start.round().astype("Int64")
    out["label_window_end"] = window_end.round().astype("Int64")

    out["temporal_unit_key"] = ""
    out.loc[cvat_mask, "temporal_unit_key"] = (
        out.loc[cvat_mask, "object_track_key"].astype(str)
        + "|anchor="
        + out.loc[cvat_mask, "label_anchor_frame_index"].astype(str)
    )
    out.loc[legacy_mask, "temporal_unit_key"] = (
        out.loc[legacy_mask, "object_track_key"].astype(str) + "|legacy_sequence"
    )
    other_mask = ~(cvat_mask | legacy_mask)
    out.loc[other_mask, "temporal_unit_key"] = (
        out.loc[other_mask, "object_track_key"].astype(str)
        + "|frame="
        + frame_idx.loc[other_mask].round().astype("Int64").astype(str)
    )

    # Legacy window start/end come from the actual frame span in each unit.
    unit_frame_stats = out.groupby(
        "temporal_unit_key",
        dropna=False,
    )["frame_index"].agg(["min", "max", "nunique"])
    out["source_sequence_length"] = (
        out["temporal_unit_key"]
        .map(unit_frame_stats["nunique"])
        .astype("Int64")
    )

    missing_start = out["label_window_start"].isna()
    out.loc[missing_start, "label_window_start"] = (
        out.loc[missing_start, "temporal_unit_key"]
        .map(unit_frame_stats["min"])
        .round()
        .astype("Int64")
    )
    missing_end = out["label_window_end"].isna()
    out.loc[missing_end, "label_window_end"] = (
        out.loc[missing_end, "temporal_unit_key"]
        .map(unit_frame_stats["max"])
        .round()
        .astype("Int64")
    )

    legacy_unit = legacy_mask
    out["source_sequence_expected_length"] = np.where(
        legacy_unit,
        config.legacy_expected_sequence_length,
        np.where(
            cvat_mask,
            config.cvat_label_stride,
            out["source_sequence_length"].astype("float64"),
        ),
    )
    out["source_sequence_complete_auto"] = pd.to_numeric(
        out["source_sequence_length"], errors="coerce"
    ) >= pd.to_numeric(out["source_sequence_expected_length"], errors="coerce")

    behavior_stats = out.groupby("temporal_unit_key", dropna=False)["behavior"].agg(
        num_behaviors_in_unit=lambda s: int(s.dropna().astype(str).nunique()),
        unique_behaviors_in_unit=lambda s: "|".join(sorted(set(s.dropna().astype(str)))),
    )
    dominant = (
        out.groupby("temporal_unit_key", dropna=False)["behavior"]
        .agg(lambda s: s.dropna().astype(str).value_counts().idxmax() if len(s.dropna()) else "")
        .rename("dominant_behavior_in_unit")
    )
    behavior_stats = behavior_stats.join(dominant)
    out["num_behaviors_in_unit"] = (
        out["temporal_unit_key"]
        .map(behavior_stats["num_behaviors_in_unit"])
        .astype("Int64")
    )
    out["unique_behaviors_in_unit"] = (
        out["temporal_unit_key"].map(behavior_stats["unique_behaviors_in_unit"]).fillna("")
    )
    out["dominant_behavior_in_unit"] = (
        out["temporal_unit_key"].map(behavior_stats["dominant_behavior_in_unit"]).fillna("")
    )
    out["behavior_consistency_in_unit"] = out["num_behaviors_in_unit"].fillna(0).le(1)

    return out


def _add_temporal_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["object_track_key", "frame_index"], kind="mergesort")

    g = out.groupby("object_track_key", dropna=False, sort=False)

    out["prev_frame_index"] = g["frame_index"].shift(1)
    out["next_frame_index"] = g["frame_index"].shift(-1)
    out["delta_frame_prev"] = out["frame_index"] - out["prev_frame_index"]
    out["delta_frame_next"] = out["next_frame_index"] - out["frame_index"]

    if "timestamp_sec" in out.columns:
        out["prev_timestamp_sec"] = g["timestamp_sec"].shift(1)
        out["delta_time_prev_sec"] = out["timestamp_sec"] - out["prev_timestamp_sec"]
    else:
        out["prev_timestamp_sec"] = np.nan
        out["delta_time_prev_sec"] = np.nan

    denom_frame = out["delta_frame_prev"].replace(0, np.nan)
    denom_time = out["delta_time_prev_sec"].replace(0, np.nan)

    for col in ["cx_n", "cy_n", "bw_n", "bh_n", "area_n", "aspect_ratio", "box_diag_n"]:
        if col not in out.columns:
            out[col] = np.nan

    out["delta_cx_n"] = g["cx_n"].diff()
    out["delta_cy_n"] = g["cy_n"].diff()
    out["delta_bw_n"] = g["bw_n"].diff()
    out["delta_bh_n"] = g["bh_n"].diff()
    out["delta_area_n"] = g["area_n"].diff()
    out["delta_aspect_ratio"] = g["aspect_ratio"].diff()
    out["delta_box_diag_n"] = g["box_diag_n"].diff()

    out["displacement_n"] = np.sqrt(out["delta_cx_n"] ** 2 + out["delta_cy_n"] ** 2)
    out["vx_n_per_frame"] = out["delta_cx_n"] / denom_frame
    out["vy_n_per_frame"] = out["delta_cy_n"] / denom_frame
    out["speed_n_per_frame"] = out["displacement_n"] / denom_frame
    out["speed_n_per_sec"] = out["displacement_n"] / denom_time

    out["prev_speed_n_per_frame"] = g["speed_n_per_frame"].shift(1)
    out["accel_n_per_frame2"] = (
        out["speed_n_per_frame"] - out["prev_speed_n_per_frame"]
    ) / denom_frame
    out["abs_accel_n_per_frame2"] = out["accel_n_per_frame2"].abs()

    out["direction_rad"] = np.arctan2(out["delta_cy_n"], out["delta_cx_n"])
    out["prev_direction_rad"] = g["direction_rad"].shift(1)
    out["direction_change_rad"] = _angle_diff(out["direction_rad"], out["prev_direction_rad"])
    out["abs_direction_change_rad"] = out["direction_change_rad"].abs()

    out["shape_change_score"] = np.sqrt(
        out["delta_bw_n"].fillna(0) ** 2
        + out["delta_bh_n"].fillna(0) ** 2
        + out["delta_area_n"].fillna(0) ** 2
        + (out["delta_aspect_ratio"].fillna(0) / 10.0) ** 2
    )

    # The first track row has no previous movement, so use zero magnitudes.
    fill_zero_cols = [
        "delta_cx_n",
        "delta_cy_n",
        "delta_bw_n",
        "delta_bh_n",
        "delta_area_n",
        "delta_aspect_ratio",
        "delta_box_diag_n",
        "displacement_n",
        "vx_n_per_frame",
        "vy_n_per_frame",
        "speed_n_per_frame",
        "accel_n_per_frame2",
        "abs_accel_n_per_frame2",
        "direction_change_rad",
        "abs_direction_change_rad",
        "shape_change_score",
    ]
    for col in fill_zero_cols:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out.sort_index(kind="mergesort")


def _add_roi_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    target_class = (
        out.get("roi_target_class", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
    )
    behavior = out["behavior"].astype(str)
    out["roi_target_class_inferred"] = target_class.where(
        target_class.ne(""), behavior.map(BEHAVIOR_TO_TARGET_ROI).fillna("")
    )

    for col in [
        "roi_target_near",
        "roi_target_contact",
        "roi_target_center_inside",
        "roi_target_available",
    ]:
        if col in out.columns:
            out[col] = _to_bool_series(out[col])
        else:
            out[col] = False

    for col in ["roi_target_min_dist_n", "roi_target_max_overlap_ratio", "roi_target_max_iou"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["object_track_key", "frame_index"], kind="mergesort")
    g = out.groupby("object_track_key", dropna=False, sort=False)

    out["prev_roi_target_contact"] = (
        g["roi_target_contact"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    out["prev_roi_target_near"] = (
        g["roi_target_near"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    out["roi_target_entry_event"] = out["roi_target_contact"] & ~out["prev_roi_target_contact"]
    out["roi_target_exit_event"] = ~out["roi_target_contact"] & out["prev_roi_target_contact"]
    out["roi_target_near_entry_event"] = out["roi_target_near"] & ~out["prev_roi_target_near"]
    out["roi_target_near_exit_event"] = ~out["roi_target_near"] & out["prev_roi_target_near"]
    out["roi_motion_inside_score"] = np.where(
        out["roi_target_contact"] | out["roi_target_near"],
        out.get("speed_n_per_frame", 0.0),
        0.0,
    )

    # Per-unit aggregate values are joined later in _add_temporal_unit_aggregates.
    return out.sort_index(kind="mergesort")


def _add_social_context_columns(df: pd.DataFrame, config: EnhancedFeatureConfig) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    n = len(out)

    nearest_pig_id_arr = np.full(n, "", dtype=object)
    nearest_track_id_arr = np.full(n, "", dtype=object)
    nearest_dist_arr = np.full(n, np.nan, dtype="float64")
    nearest_iou_arr = np.zeros(n, dtype="float64")
    nearest_overlap_arr = np.zeros(n, dtype="float64")
    near_count_arr = np.zeros(n, dtype="int64")
    contact_count_arr = np.zeros(n, dtype="int64")
    frame_size_arr = np.zeros(n, dtype="int64")

    required = ["cx_n", "cy_n", "x1", "y1", "x2", "y2", "bbox_area"]
    for col in required:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["_social_frame_group_key"] = _social_frame_group_key(out)
    frame_key_cols = ["_social_frame_group_key"]

    valid_bbox = _to_bool_series(
        out.get("bbox_valid", pd.Series(True, index=out.index))
    ).to_numpy(dtype=bool)
    pig_ids_all = out["pig_id"].fillna("").astype(str).to_numpy(dtype=object)
    track_ids_all = out["track_id"].fillna("").astype(str).to_numpy(dtype=object)
    centers_all = out[["cx_n", "cy_n"]].to_numpy(dtype="float64")
    boxes_all = out[["x1", "y1", "x2", "y2"]].to_numpy(dtype="float64")
    areas_all = out["bbox_area"].to_numpy(dtype="float64")

    # groupby.indices returns integer positions because index has been reset.
    for idx_arr in out.groupby(frame_key_cols, dropna=False, sort=False).indices.values():
        idx_arr = np.asarray(idx_arr, dtype="int64")
        m = idx_arr.size
        if m == 0:
            continue
        frame_size_arr[idx_arr] = int(m)
        if m == 1 or m > config.max_frame_group_size_for_social:
            continue

        centers = centers_all[idx_arr]
        finite_geometry = (
            np.isfinite(centers).all(axis=1)
            & np.isfinite(boxes_all[idx_arr]).all(axis=1)
            & np.isfinite(areas_all[idx_arr])
            & (areas_all[idx_arr] > 0)
            & valid_bbox[idx_arr]
        )
        if not finite_geometry.any():
            continue

        dx = centers[:, [0]] - centers[:, 0][None, :]
        dy = centers[:, [1]] - centers[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        dist[~finite_geometry, :] = np.inf
        dist[:, ~finite_geometry] = np.inf

        nearest_pos = np.argmin(dist, axis=1)
        nearest_dist = dist[np.arange(m), nearest_pos]
        has_neighbor = np.isfinite(nearest_dist)
        near_count = (dist <= config.social_near_distance_n).sum(axis=1)

        pair_iou_mat, pair_overlap_mat = _pairwise_box_overlap(
            boxes_all[idx_arr],
            areas_all[idx_arr],
        )
        pair_iou_mat[~finite_geometry, :] = 0.0
        pair_iou_mat[:, ~finite_geometry] = 0.0
        pair_overlap_mat[~finite_geometry, :] = 0.0
        pair_overlap_mat[:, ~finite_geometry] = 0.0
        contact_mat = (pair_iou_mat >= config.social_contact_iou_threshold) | (
            pair_overlap_mat >= config.social_contact_overlap_threshold
        )
        np.fill_diagonal(contact_mat, False)
        contact_count = contact_mat.sum(axis=1)

        valid_rows = has_neighbor & finite_geometry
        source_pos = np.where(valid_rows)[0]
        target_positions = idx_arr[source_pos]
        neighbor_positions = nearest_pos[source_pos]

        nearest_pig_id_arr[target_positions] = pig_ids_all[idx_arr[neighbor_positions]]
        nearest_track_id_arr[target_positions] = track_ids_all[idx_arr[neighbor_positions]]
        nearest_dist_arr[target_positions] = nearest_dist[source_pos]

        valid_global = idx_arr[finite_geometry]
        near_count_arr[valid_global] = near_count[finite_geometry].astype("int64")
        contact_count_arr[valid_global] = contact_count[finite_geometry].astype(
            "int64"
        )

        # Nearest pair overlap values.
        for local_pos in source_pos:
            global_pos = idx_arr[local_pos]
            neighbor_pos = nearest_pos[local_pos]
            nearest_iou_arr[global_pos] = pair_iou_mat[local_pos, neighbor_pos]
            nearest_overlap_arr[global_pos] = pair_overlap_mat[
                local_pos,
                neighbor_pos,
            ]

    out["nearest_pig_id"] = nearest_pig_id_arr
    out["nearest_track_id"] = nearest_track_id_arr
    out["nearest_dist_n"] = nearest_dist_arr
    out["nearest_pair_iou"] = nearest_iou_arr
    out["nearest_pair_overlap_ratio"] = nearest_overlap_arr
    out["social_density_near_count"] = near_count_arr
    out["social_contact_count"] = contact_count_arr
    out["social_context_frame_size"] = frame_size_arr

    out = out.sort_values(["object_track_key", "frame_index"], kind="mergesort")
    g = out.groupby("object_track_key", dropna=False, sort=False)
    denom_frame = (
        out["delta_frame_prev"].replace(0, np.nan)
        if "delta_frame_prev" in out.columns
        else np.nan
    )
    out["prev_nearest_pig_id"] = g["nearest_pig_id"].shift(1).fillna("")
    out["prev_nearest_dist_n"] = g["nearest_dist_n"].shift(1)
    same_neighbor = (
        out["nearest_pig_id"]
        .astype(str)
        .eq(out["prev_nearest_pig_id"].astype(str))
        & out["nearest_pig_id"].astype(str).ne("")
    )
    out["nearest_dist_delta"] = np.where(
        same_neighbor,
        out["nearest_dist_n"] - out["prev_nearest_dist_n"],
        np.nan,
    )
    out["approach_speed_n_per_frame"] = np.where(
        same_neighbor,
        -out["nearest_dist_delta"] / denom_frame,
        0.0,
    )
    out["separation_speed_n_per_frame"] = np.where(
        same_neighbor,
        out["nearest_dist_delta"] / denom_frame,
        0.0,
    )
    out["approach_speed_n_per_frame"] = (
        pd.to_numeric(out["approach_speed_n_per_frame"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    out["separation_speed_n_per_frame"] = (
        pd.to_numeric(out["separation_speed_n_per_frame"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    out["pair_contact_with_nearest"] = (
        out["nearest_pair_iou"] >= config.social_contact_iou_threshold
    ) | (
        out["nearest_pair_overlap_ratio"]
        >= config.social_contact_overlap_threshold
    )
    out["aggression_score_proxy"] = (
        out["pair_contact_with_nearest"].astype(float)
        * (
            out.get("speed_n_per_frame", 0.0).fillna(0.0)
            + out["approach_speed_n_per_frame"].fillna(0.0)
        )
        * (1.0 + out["social_density_near_count"].fillna(0.0))
    )

    out = out.drop(columns=["_social_frame_group_key"])
    return out.sort_index(kind="mergesort")


def _social_frame_group_key(rows: pd.DataFrame) -> pd.Series:
    """Build a row-wise frame key without merging partially missing UIDs."""
    frame_uid = rows.get(
        "frame_uid",
        pd.Series("", index=rows.index),
    ).fillna("").astype(str).str.strip()
    frame_index = pd.to_numeric(rows["frame_index"], errors="coerce")
    invalid_fallback = (
        frame_uid.eq("")
        & (
            frame_index.isna()
            | frame_index.mod(1).ne(0)
            | frame_index.lt(0)
        )
    )
    if invalid_fallback.any():
        sample = [str(value) for value in rows.index[invalid_fallback].tolist()[:10]]
        raise ValueError(
            "Social frame grouping contract failed: "
            f"invalid_fallback_rows={int(invalid_fallback.sum())}, "
            f"sample_source_indices={sample}"
        )

    fallback = "frame=" + frame_index.round().astype("Int64").astype(str)
    local_frame = ("uid=" + frame_uid).where(frame_uid.ne(""), fallback)
    return (
        rows["source_type"].astype(str)
        + "|"
        + rows["dataset_id"].astype(str)
        + "|"
        + rows["video_key"].astype(str)
        + "|"
        + local_frame
    )


def _add_temporal_unit_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("temporal_unit_key", dropna=False, sort=False)

    agg_spec: dict[str, tuple[str, str | Any]] = {
        "speed_mean_unit": ("speed_n_per_frame", "mean"),
        "speed_max_unit": ("speed_n_per_frame", "max"),
        "speed_std_unit": ("speed_n_per_frame", "std"),
        "accel_abs_mean_unit": ("abs_accel_n_per_frame2", "mean"),
        "accel_abs_max_unit": ("abs_accel_n_per_frame2", "max"),
        "direction_change_abs_mean_unit": ("abs_direction_change_rad", "mean"),
        "direction_change_abs_max_unit": ("abs_direction_change_rad", "max"),
        "path_length_n_unit": ("displacement_n", "sum"),
        "motion_energy_unit": (
            "speed_n_per_frame",
            lambda s: float(
                np.nansum(np.asarray(s, dtype="float64") ** 2)
            ),
        ),
        "shape_transition_score_unit": ("shape_change_score", "max"),
        "area_n_std_unit": ("area_n", "std"),
        "aspect_ratio_std_unit": ("aspect_ratio", "std"),
        "target_roi_contact_frame_count_unit": ("roi_target_contact", "sum"),
        "target_roi_near_frame_count_unit": ("roi_target_near", "sum"),
        "target_roi_entry_count_unit": ("roi_target_entry_event", "sum"),
        "target_roi_exit_count_unit": ("roi_target_exit_event", "sum"),
        "target_roi_min_dist_n_mean_unit": ("roi_target_min_dist_n", "mean"),
        "target_roi_min_dist_n_min_unit": ("roi_target_min_dist_n", "min"),
        "target_roi_overlap_mean_unit": ("roi_target_max_overlap_ratio", "mean"),
        "target_roi_overlap_max_unit": ("roi_target_max_overlap_ratio", "max"),
        "nearest_dist_mean_unit": ("nearest_dist_n", "mean"),
        "nearest_dist_min_unit": ("nearest_dist_n", "min"),
        "social_density_mean_unit": ("social_density_near_count", "mean"),
        "social_density_max_unit": ("social_density_near_count", "max"),
        "pair_contact_frame_count_unit": ("pair_contact_with_nearest", "sum"),
        "approach_speed_max_unit": ("approach_speed_n_per_frame", "max"),
        "aggression_score_proxy_max_unit": ("aggression_score_proxy", "max"),
        "aggression_score_proxy_mean_unit": ("aggression_score_proxy", "mean"),
    }

    available_agg = {
        out_col: pd.NamedAgg(column=in_col, aggfunc=func)
        for out_col, (in_col, func) in agg_spec.items()
        if in_col in out.columns
    }
    unit_agg = g.agg(**available_agg)

    # Displacement from first to last position in the temporal unit.
    first_last = g[["cx_n", "cy_n"]].agg(["first", "last"])
    unit_agg["displacement_n_unit"] = np.sqrt(
        (first_last[("cx_n", "last")] - first_last[("cx_n", "first")]) ** 2
        + (first_last[("cy_n", "last")] - first_last[("cy_n", "first")]) ** 2
    )
    unit_agg["displacement_ratio_unit"] = unit_agg["displacement_n_unit"] / unit_agg.get(
        "path_length_n_unit", np.nan
    ).replace(0, np.nan)
    unit_agg["motion_burstiness_unit"] = unit_agg.get("speed_std_unit", np.nan) / (
        unit_agg.get("speed_mean_unit", np.nan) + 1e-9
    )
    unit_agg["bbox_stability_unit"] = 1.0 / (
        1.0
        + unit_agg.get("area_n_std_unit", np.nan).fillna(0.0)
        + unit_agg.get("aspect_ratio_std_unit", np.nan).fillna(0.0)
    )

    frame_count = g.size().rename("frame_count_unit")
    unit_agg = unit_agg.join(frame_count)
    unit_agg["target_roi_contact_ratio_unit"] = unit_agg.get(
        "target_roi_contact_frame_count_unit",
        0,
    ) / unit_agg["frame_count_unit"].replace(0, np.nan)
    unit_agg["target_roi_near_ratio_unit"] = unit_agg.get(
        "target_roi_near_frame_count_unit",
        0,
    ) / unit_agg["frame_count_unit"].replace(0, np.nan)
    unit_agg["pair_contact_ratio_unit"] = unit_agg.get(
        "pair_contact_frame_count_unit",
        0,
    ) / unit_agg["frame_count_unit"].replace(0, np.nan)

    # Prefix columns are already descriptive; map back to each row.
    for col in unit_agg.columns:
        out[col] = out["temporal_unit_key"].map(unit_agg[col])

    numeric_fill = [
        "speed_mean_unit",
        "speed_max_unit",
        "speed_std_unit",
        "accel_abs_mean_unit",
        "accel_abs_max_unit",
        "direction_change_abs_mean_unit",
        "direction_change_abs_max_unit",
        "path_length_n_unit",
        "motion_energy_unit",
        "shape_transition_score_unit",
        "target_roi_contact_ratio_unit",
        "target_roi_near_ratio_unit",
        "pair_contact_ratio_unit",
        "motion_burstiness_unit",
        "bbox_stability_unit",
    ]
    for col in numeric_fill:
        if col in out.columns:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    return out


def _add_review_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Preferred GUI is not a decision; it is a routing hint for later template builders.
    preferred = np.full(len(out), "spatial", dtype=object)
    preferred[out["is_motion_behavior"].to_numpy(dtype=bool)] = "temporal"
    preferred[out["is_roi_behavior"].to_numpy(dtype=bool)] = "temporal"
    preferred[out["is_interaction_behavior"].to_numpy(dtype=bool)] = "spatial_context"
    preferred[out["is_shape_behavior"].to_numpy(dtype=bool)] = "spatial"
    inconsistent = ~out["behavior_consistency_in_unit"].fillna(False).astype(bool)
    preferred[inconsistent.to_numpy(dtype=bool)] = "temporal"
    out["preferred_review_gui_auto"] = preferred

    reason_parts: list[list[str]] = [[] for _ in range(len(out))]

    def add_reason(mask: pd.Series, reason: str) -> None:
        arr = mask.fillna(False).to_numpy(dtype=bool)
        for i, flag in enumerate(arr):
            if flag:
                reason_parts[i].append(reason)

    add_reason(
        out["is_roi_behavior"] & ~out.get("roi_target_contact", False),
        "roi_label_without_target_contact",
    )
    add_reason(
        out["is_motion_behavior"]
        & out.get("speed_mean_unit", 0).fillna(0).lt(0.002),
        "motion_label_low_motion",
    )
    add_reason(
        out["behavior"].eq("stand")
        & out.get("speed_mean_unit", 0).fillna(0).gt(0.01),
        "stand_label_high_motion",
    )
    add_reason(
        out["is_interaction_behavior"]
        & out.get("nearest_dist_min_unit", np.inf)
        .fillna(np.inf)
        .gt(0.12),
        "interaction_label_no_close_neighbor",
    )
    add_reason(
        out["is_shape_behavior"] & out.get("shape_transition_score_unit", 0).fillna(0).gt(0.20),
        "posture_label_high_shape_change",
    )
    add_reason(
        ~out["behavior_consistency_in_unit"].fillna(True).astype(bool),
        "multiple_behaviors_in_temporal_unit",
    )
    add_reason(
        ~out["source_sequence_complete_auto"].fillna(True).astype(bool),
        "incomplete_temporal_unit",
    )

    out["review_feature_reason_auto"] = [";".join(parts) if parts else "" for parts in reason_parts]
    out["review_feature_priority_auto"] = np.select(
        [
            out["review_feature_reason_auto"]
            .astype(str)
            .str.contains("multiple_behaviors|interaction_label", regex=True),
            out["review_feature_reason_auto"].astype(str).ne(""),
        ],
        ["high", "medium"],
        default="low",
    )

    # A feature is valid for review if geometry is valid and temporal unit assignment worked.
    bbox_valid = _to_bool_series(out.get("bbox_valid", pd.Series(True, index=out.index)))
    out["spatiotemporal_feature_valid"] = (
        bbox_valid
        & out["temporal_unit_key"].astype(str).ne("")
        & out["label_window_start"].notna()
        & out["label_window_end"].notna()
    )

    # Remove temporary helper columns that should not become data contract.
    out = out.drop(columns=[c for c in ["_track_for_key", "_pig_for_key"] if c in out.columns])
    return out


def _pairwise_box_overlap(boxes: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(boxes)
    iou = np.zeros((n, n), dtype="float64")
    overlap_min = np.zeros((n, n), dtype="float64")
    if n == 0:
        return iou, overlap_min

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    xx1 = np.maximum(x1[:, None], x1[None, :])
    yy1 = np.maximum(y1[:, None], y1[None, :])
    xx2 = np.minimum(x2[:, None], x2[None, :])
    yy2 = np.minimum(y2[:, None], y2[None, :])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area_i = areas[:, None]
    area_j = areas[None, :]
    union = area_i + area_j - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
        minimum_area = np.minimum(area_i, area_j)
        overlap_min = np.where(
            minimum_area > 0,
            inter / minimum_area,
            0.0,
        )
    iou[~np.isfinite(iou)] = 0.0
    overlap_min[~np.isfinite(overlap_min)] = 0.0
    return iou, overlap_min


def _angle_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    diff = a - b
    return (diff + math.pi) % (2 * math.pi) - math.pi


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


def _numeric_summary(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    if column not in df.columns:
        return {}
    s = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }

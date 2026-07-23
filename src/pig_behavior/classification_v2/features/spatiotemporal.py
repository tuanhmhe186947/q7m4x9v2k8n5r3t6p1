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

from pig_behavior.classification_v2.contracts.identifiers import scene_frame_key
from pig_behavior.classification_v2.contracts.lineage_claims import (
    add_optional_lineage_claims_to_audit,
    require_lineage_claims_preserved,
    resolve_optional_lineage_claims,
)
from pig_behavior.classification_v2.features.context_policy import (
    normalize_hidden_provenance,
)
from pig_behavior.classification_v2.features.native_evidence_contract import (
    NATIVE_EVIDENCE_SEMANTICS_VERSION,
    NATIVE_FEATURE_COMPUTATION_GRAIN,
    NATIVE_MOTION_SCHEMA_VERSION,
    NATIVE_PAIR_SCOPE_KEY,
    PAIR_COVERAGE_COLUMNS,
)
from pig_behavior.classification_v2.features.social import (
    build_static_social_context_features,
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

MOTION_GRAIN_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "temporal_unit_key",
)

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "temporal_unit_key",
    "frame_uid",
    "frame_index",
    "timestamp_sec",
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
    stationary_speed_threshold_per_second: float = 0.06
    active_speed_threshold_per_second: float = 0.18
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
            stationary_speed_threshold_per_second=(
                self.stationary_speed_threshold_per_second
            ),
            active_speed_threshold_per_second=(
                self.active_speed_threshold_per_second
            ),
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
    stationary_speed_threshold_per_second: float = 0.06,
    active_speed_threshold_per_second: float = 0.18,
    turning_angle_threshold_rad: float = float(np.pi / 6.0),
) -> pd.DataFrame:
    """Add enhanced spatio-temporal, ROI-duration, and social-context features.

    The returned dataframe has the same number of rows and keeps all original
    columns. New columns are appended.
    """
    resolve_optional_lineage_claims(
        frame_features,
        artifact_name="enhanced spatiotemporal input",
    )
    config = EnhancedFeatureConfig(
        cvat_label_stride=cvat_label_stride,
        legacy_expected_sequence_length=legacy_expected_sequence_length,
        social_near_distance_n=social_near_distance_n,
        social_contact_iou_threshold=social_contact_iou_threshold,
        social_contact_overlap_threshold=social_contact_overlap_threshold,
        stationary_speed_threshold=stationary_speed_threshold,
        active_speed_threshold=active_speed_threshold,
        stationary_speed_threshold_per_second=(
            stationary_speed_threshold_per_second
        ),
        active_speed_threshold_per_second=active_speed_threshold_per_second,
        turning_angle_threshold_rad=turning_angle_threshold_rad,
    )
    config.validate()

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame_features.columns]
    if missing:
        raise ValueError(f"Missing enhanced spatiotemporal input columns: {missing}")
    if "feature_computation_grain" in frame_features.columns:
        input_grains = set(
            frame_features["feature_computation_grain"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        if input_grains != {"FRAME_LOCAL_PRIMITIVES"}:
            raise ValueError(
                "enhanced spatiotemporal input must be FRAME_LOCAL_PRIMITIVES: "
                f"found={sorted(input_grains)}"
            )

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
    out["feature_computation_grain"] = NATIVE_FEATURE_COMPUTATION_GRAIN
    out["pair_scope_key"] = out["temporal_unit_key"].astype(str)
    out["evidence_semantics_version"] = (
        NATIVE_EVIDENCE_SEMANTICS_VERSION
    )
    out["motion_schema_version"] = NATIVE_MOTION_SCHEMA_VERSION
    out["pair_recomputed_for_view"] = False
    out["aggregate_recomputed_for_view"] = False

    require_lineage_claims_preserved(
        frame_features,
        out,
        source_name="enhanced spatiotemporal input",
        derived_name="enhanced spatiotemporal output",
    )
    return out


def audit_enhanced_spatiotemporal_features(
    df: pd.DataFrame,
    *,
    input_rows: int | None = None,
    code_sha: str = "",
    input_sha256: str = "",
    contract_manifest_sha256: str = "",
) -> dict[str, Any]:
    """Return audit summary for enhanced spatio-temporal features."""
    errors: list[str] = []
    warnings: list[str] = []
    if input_rows is not None and int(input_rows) != len(df):
        errors.append(
            f"population_row_count_mismatch={int(input_rows)}:{len(df)}"
        )
    lineage_fields = {
        "code_sha": code_sha,
        "input_sha256": input_sha256,
        "contract_manifest_sha256": contract_manifest_sha256,
    }
    if any(str(value).strip() for value in lineage_fields.values()):
        for field, value in lineage_fields.items():
            if not str(value).strip():
                errors.append(f"missing_{field}")

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
        "speed_n_per_second",
        "motion_energy_n_per_second2_unit",
        "target_roi_contact_ratio_unit",
        "nearest_pig_id",
        "nearest_dist_n",
        "social_density_near_count",
        "motion_active_ratio_per_second_unit",
        "roi_feeder_contact_ratio_unit",
        "social_partner_persistence_ratio_unit",
        "spatiotemporal_feature_valid",
        "feature_computation_grain",
        "pair_scope_key",
        "evidence_semantics_version",
        "motion_schema_version",
        "previous_observation_available",
        "valid_delta_time",
        "current_geometry_valid",
        "previous_geometry_valid",
        "same_temporal_unit_pair",
        "same_actor_trajectory_pair",
        "valid_motion_pair",
        *PAIR_COVERAGE_COLUMNS,
        "pair_recomputed_for_view",
        "aggregate_recomputed_for_view",
    ]
    missing_new = [c for c in required_new if c not in df.columns]
    if missing_new:
        errors.append(f"missing_enhanced_columns={missing_new}")

    grain_mismatch = 0
    pair_scope_mismatch = 0
    invalid_view_recompute_claim = 0
    if "feature_computation_grain" in df.columns:
        grain_mismatch = int(
            df["feature_computation_grain"]
            .fillna("")
            .astype(str)
            .ne(NATIVE_FEATURE_COMPUTATION_GRAIN)
            .sum()
        )
        if grain_mismatch:
            errors.append(f"invalid_feature_computation_grain={grain_mismatch}")
    if {"pair_scope_key", "temporal_unit_key"}.issubset(df.columns):
        pair_scope_mismatch = int(
            df["pair_scope_key"]
            .fillna("")
            .astype(str)
            .ne(df["temporal_unit_key"].fillna("").astype(str))
            .sum()
        )
        if pair_scope_mismatch:
            errors.append(f"native_pair_scope_mismatch={pair_scope_mismatch}")
    provenance_expected = {
        "evidence_semantics_version": NATIVE_EVIDENCE_SEMANTICS_VERSION,
        "motion_schema_version": NATIVE_MOTION_SCHEMA_VERSION,
    }
    for column, expected in provenance_expected.items():
        if column not in df:
            continue
        mismatch = int(
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne(expected)
            .sum()
        )
        if mismatch:
            errors.append(f"native_provenance_mismatch={column}:{mismatch}")
    for column in (
        "pair_recomputed_for_view",
        "aggregate_recomputed_for_view",
    ):
        if column in df.columns:
            invalid_view_recompute_claim += int(_to_bool_series(df[column]).sum())
    if invalid_view_recompute_claim:
        errors.append(
            "native_evidence_claims_final_view_recompute="
            f"{invalid_view_recompute_claim}"
        )

    cross_unit_pair_count = 0
    native_pair_reset_errors = 0
    invalid_pair_aggregate_contributions = 0
    pair_coverage_complete = not set(PAIR_COVERAGE_COLUMNS).difference(
        df.columns
    )
    if {
        *MOTION_GRAIN_COLUMNS,
        "frame_index",
        "speed_n_per_second",
        "valid_motion_pair",
        "previous_observation_available",
        "previous_temporal_unit_key",
        "temporal_unit_key",
    }.issubset(df.columns):
        valid_pair = _to_bool_series(df["valid_motion_pair"])
        cross_unit_pair_count = int(
            (
                valid_pair
                & df["previous_temporal_unit_key"]
                .fillna("")
                .astype(str)
                .ne("")
                & df["previous_temporal_unit_key"]
                .fillna("")
                .astype(str)
                .ne(df["temporal_unit_key"].astype(str))
            ).sum()
        )
        if cross_unit_pair_count:
            errors.append(
                f"cross_unit_pair_count={cross_unit_pair_count}"
            )
        starts = (
            df.sort_values(
                [*MOTION_GRAIN_COLUMNS, "frame_index"],
                kind="mergesort",
            )
            .groupby(list(MOTION_GRAIN_COLUMNS), sort=False)
            .head(1)
        )
        native_pair_reset_errors = int(
            _to_bool_series(
                starts["previous_observation_available"]
            ).sum()
            + _to_bool_series(starts["valid_motion_pair"]).sum()
        )
        if native_pair_reset_errors:
            errors.append(
                f"native_pair_reset_errors={native_pair_reset_errors}"
            )
        invalid_speed = pd.to_numeric(
            df.loc[~valid_pair, "speed_n_per_second"],
            errors="coerce",
        ).notna()
        invalid_pair_aggregate_contributions = int(invalid_speed.sum())
        if invalid_pair_aggregate_contributions:
            errors.append(
                "invalid_pair_aggregate_contributions="
                f"{invalid_pair_aggregate_contributions}"
            )
    if not pair_coverage_complete:
        errors.append("pair_coverage_incomplete")

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

    audit = {
        "feature_computation_grain": NATIVE_FEATURE_COMPUTATION_GRAIN,
        "pair_scope_key": NATIVE_PAIR_SCOPE_KEY,
        "evidence_semantics_version": NATIVE_EVIDENCE_SEMANTICS_VERSION,
        "motion_schema_version": NATIVE_MOTION_SCHEMA_VERSION,
        "code_sha": str(code_sha).lower(),
        "input_sha256": str(input_sha256).lower(),
        "contract_manifest_sha256": str(
            contract_manifest_sha256
        ).lower(),
        "input_rows": int(input_rows) if input_rows is not None else None,
        "rows": int(len(df)),
        "frames": int(scene_frame_key(df).nunique(dropna=True)),
        "frame_objects": int(df["frame_uid"].nunique(dropna=True))
        if "frame_uid" in df.columns
        else 0,
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
        "speed_n_per_second": _numeric_summary(df, "speed_n_per_second"),
        "nearest_dist_n": _numeric_summary(df, "nearest_dist_n"),
        "motion_energy_n_per_second2_unit": _numeric_summary(
            df,
            "motion_energy_n_per_second2_unit",
        ),
        "social_density_near_count": _numeric_summary(df, "social_density_near_count"),
        "motion_active_ratio_per_second_unit": _numeric_summary(
            df,
            "motion_active_ratio_per_second_unit",
        ),
        "roi_feeder_contact_ratio_unit": _numeric_summary(
            df,
            "roi_feeder_contact_ratio_unit",
        ),
        "social_partner_persistence_ratio_unit": _numeric_summary(
            df,
            "social_partner_persistence_ratio_unit",
        ),
        "feature_computation_grain_values": _value_counts_dict(
            df,
            "feature_computation_grain",
        ),
        "pair_scope_mismatch": pair_scope_mismatch,
        "cross_unit_pair_count": cross_unit_pair_count,
        "native_pair_reset_errors": native_pair_reset_errors,
        "invalid_pair_aggregate_contributions": (
            invalid_pair_aggregate_contributions
        ),
        "pair_coverage_complete": pair_coverage_complete,
        "invalid_view_recompute_claim": invalid_view_recompute_claim,
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
    return add_optional_lineage_claims_to_audit(
        audit,
        df,
        artifact_name="enhanced spatiotemporal audit frame table",
    )


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
        "scene_frame_uid",
        "frame_uid",
        "pig_id",
        "track_id",
        "behavior",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    for identity_column in ("object_track_key", "temporal_unit_key"):
        identity = out[identity_column].fillna("").astype(str).str.strip()
        blank = int(identity.eq("").sum())
        if blank:
            raise ValueError(
                f"{identity_column} is required and nonblank: rows={blank}"
            )
        out[identity_column] = identity
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
    input_temporal_unit_key = (
        out["temporal_unit_key"].fillna("").astype(str).copy()
    )

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

    if not out["temporal_unit_key"].astype(str).equals(
        input_temporal_unit_key
    ):
        raise RuntimeError("temporal_unit_key changed during native evidence")
    return out


def _add_temporal_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_pair_scope_columns(out, operation="temporal motion")
    grain = list(MOTION_GRAIN_COLUMNS)
    duplicate = out.duplicated([*grain, "frame_index"], keep=False)
    if duplicate.any():
        raise ValueError(
            "temporal motion requires unique actor/unit/frame rows: "
            f"duplicates={int(duplicate.sum())}"
        )
    out["_native_pair_input_order"] = np.arange(len(out), dtype="int64")
    sort_columns = [*grain, "frame_index"]
    if "frame_uid" in out.columns:
        sort_columns.append("frame_uid")
    out = out.sort_values(sort_columns, kind="mergesort")
    g = out.groupby(grain, dropna=False, sort=False)

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

    geometry_columns = [
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "area_n",
        "aspect_ratio",
    ]
    for col in [*geometry_columns, "box_diag_n"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    observed = _to_bool_series(
        out.get("observed_mask", pd.Series(True, index=out.index))
    )
    previous_observed = observed.groupby(
        [out[column] for column in grain],
        dropna=False,
        sort=False,
    ).shift(1).fillna(False)
    out["previous_observation_available"] = (
        out["prev_frame_index"].notna()
        & observed
        & previous_observed
    )
    out["previous_temporal_unit_key"] = (
        g["temporal_unit_key"].shift(1).fillna("").astype(str)
    )
    out["previous_object_track_key"] = (
        g["object_track_key"].shift(1).fillna("").astype(str)
    )
    out["same_temporal_unit_pair"] = (
        out["previous_observation_available"]
        & out["previous_temporal_unit_key"].eq(
            out["temporal_unit_key"].astype(str)
        )
    )
    out["same_actor_trajectory_pair"] = (
        out["previous_observation_available"]
        & out["previous_object_track_key"].eq(
            out["object_track_key"].astype(str)
        )
    )

    bbox_valid = _to_bool_series(
        out.get("bbox_valid", pd.Series(True, index=out.index))
    )
    finite_geometry = (
        out[geometry_columns]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    out["current_geometry_valid"] = bbox_valid & finite_geometry
    out["previous_geometry_valid"] = out[
        "current_geometry_valid"
    ].groupby(
        [out[column] for column in grain],
        dropna=False,
        sort=False,
    ).shift(1).fillna(False)
    finite_delta_time = np.isfinite(out["delta_time_prev_sec"])
    out["valid_delta_time"] = (
        out["previous_observation_available"]
        & finite_delta_time
        & out["delta_time_prev_sec"].gt(0)
    )
    out["timestamp_monotonic_pair"] = out["valid_delta_time"]
    valid_delta_frame = (
        np.isfinite(out["delta_frame_prev"])
        & out["delta_frame_prev"].gt(0)
    )
    out["valid_motion_pair"] = (
        out["previous_observation_available"]
        & out["same_temporal_unit_pair"]
        & out["same_actor_trajectory_pair"]
        & out["current_geometry_valid"]
        & out["previous_geometry_valid"]
        & out["valid_delta_time"]
        & valid_delta_frame
    )
    out["motion_delta_frames"] = out["delta_frame_prev"]
    out["motion_delta_seconds"] = out["delta_time_prev_sec"]
    out["adjacent_motion_pair_valid"] = (
        out["valid_motion_pair"] & out["delta_frame_prev"].eq(1)
    )
    out["sparse_velocity_pair_valid"] = (
        out["valid_motion_pair"] & out["delta_frame_prev"].gt(1)
    )
    out["motion_velocity_pair_valid"] = out["valid_motion_pair"]
    out["motion_pair_invalid_nonpositive_frame"] = (
        out["previous_observation_available"] & ~valid_delta_frame
    )
    out["motion_pair_invalid_nonpositive_time"] = (
        out["previous_observation_available"] & ~out["valid_delta_time"]
    )
    out["motion_pair_invalid_geometry"] = (
        out["previous_observation_available"]
        & ~(
            out["current_geometry_valid"]
            & out["previous_geometry_valid"]
        )
    )

    raw_delta: dict[str, pd.Series] = {}
    for source_column, delta_column in [
        ("cx_n", "delta_cx_n"),
        ("cy_n", "delta_cy_n"),
        ("bw_n", "delta_bw_n"),
        ("bh_n", "delta_bh_n"),
        ("area_n", "delta_area_n"),
        ("aspect_ratio", "delta_aspect_ratio"),
        ("box_diag_n", "delta_box_diag_n"),
    ]:
        raw_delta[source_column] = g[source_column].diff()
        out[delta_column] = raw_delta[source_column].where(
            out["adjacent_motion_pair_valid"]
        )

    raw_displacement = np.hypot(
        raw_delta["cx_n"],
        raw_delta["cy_n"],
    )
    out["displacement_n"] = raw_displacement.where(
        out["adjacent_motion_pair_valid"]
    )
    out["sparse_displacement_n"] = raw_displacement.where(
        out["sparse_velocity_pair_valid"]
    )
    denom_frame = out["delta_frame_prev"].where(
        out["adjacent_motion_pair_valid"]
    )
    denom_time = out["delta_time_prev_sec"].where(
        out["motion_velocity_pair_valid"]
    )
    out["vx_n_per_frame"] = out["delta_cx_n"] / denom_frame
    out["vy_n_per_frame"] = out["delta_cy_n"] / denom_frame
    out["speed_n_per_frame"] = out["displacement_n"] / denom_frame
    out["speed_n_per_second"] = raw_displacement / denom_time
    out["speed_n_per_sec"] = out["speed_n_per_second"]
    out["vx_n_per_second"] = raw_delta["cx_n"] / denom_time
    out["vy_n_per_second"] = raw_delta["cy_n"] / denom_time
    out["bw_rate_n_per_second"] = raw_delta["bw_n"] / denom_time
    out["bh_rate_n_per_second"] = raw_delta["bh_n"] / denom_time
    out["area_rate_n_per_second"] = raw_delta["area_n"] / denom_time
    out["aspect_ratio_rate_per_second"] = (
        raw_delta["aspect_ratio"] / denom_time
    )

    out["prev_speed_n_per_frame"] = g["speed_n_per_frame"].shift(1)
    out["accel_n_per_frame2"] = (
        out["speed_n_per_frame"] - out["prev_speed_n_per_frame"]
    ).where(out["adjacent_motion_pair_valid"])
    out["abs_accel_n_per_frame2"] = out["accel_n_per_frame2"].abs()

    previous_speed_per_second = g["speed_n_per_second"].shift(1)
    previous_velocity_valid = g["motion_velocity_pair_valid"].shift(1)
    previous_delta_seconds = g["motion_delta_seconds"].shift(1)
    acceleration_delta_seconds = (
        out["motion_delta_seconds"] + previous_delta_seconds
    ) / 2.0
    acceleration_valid = (
        out["motion_velocity_pair_valid"]
        & _to_bool_series(previous_velocity_valid)
        & acceleration_delta_seconds.gt(0)
    )
    out["acceleration_pair_valid"] = acceleration_valid
    out["acceleration_delta_seconds"] = acceleration_delta_seconds.where(
        acceleration_valid
    )
    out["acceleration_n_per_second2"] = (
        out["speed_n_per_second"] - previous_speed_per_second
    ).div(acceleration_delta_seconds).where(acceleration_valid)
    out["abs_acceleration_n_per_second2"] = out[
        "acceleration_n_per_second2"
    ].abs()

    out["direction_rad"] = np.arctan2(
        out["delta_cy_n"],
        out["delta_cx_n"],
    ).where(out["adjacent_motion_pair_valid"])
    out["prev_direction_rad"] = g["direction_rad"].shift(1)
    previous_adjacent = g["adjacent_motion_pair_valid"].shift(1)
    heading_pair_valid = (
        out["adjacent_motion_pair_valid"]
        & _to_bool_series(previous_adjacent)
    )
    out["direction_change_rad"] = _angle_diff(
        out["direction_rad"],
        out["prev_direction_rad"],
    ).where(heading_pair_valid)
    out["direction_change_pair_valid"] = heading_pair_valid
    out["abs_direction_change_rad"] = out["direction_change_rad"].abs()

    out["shape_change_score"] = np.sqrt(
        out["delta_bw_n"] ** 2
        + out["delta_bh_n"] ** 2
        + out["delta_area_n"] ** 2
        + (out["delta_aspect_ratio"] / 10.0) ** 2
    ).where(out["adjacent_motion_pair_valid"])

    pair_numeric_columns = [
        "delta_cx_n",
        "delta_cy_n",
        "delta_bw_n",
        "delta_bh_n",
        "delta_area_n",
        "delta_aspect_ratio",
        "delta_box_diag_n",
        "displacement_n",
        "sparse_displacement_n",
        "vx_n_per_frame",
        "vy_n_per_frame",
        "speed_n_per_frame",
        "speed_n_per_second",
        "speed_n_per_sec",
        "vx_n_per_second",
        "vy_n_per_second",
        "bw_rate_n_per_second",
        "bh_rate_n_per_second",
        "area_rate_n_per_second",
        "aspect_ratio_rate_per_second",
        "accel_n_per_frame2",
        "abs_accel_n_per_frame2",
        "direction_change_rad",
        "abs_direction_change_rad",
        "shape_change_score",
        "acceleration_n_per_second2",
        "abs_acceleration_n_per_second2",
    ]
    for col in pair_numeric_columns:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    return (
        out.sort_values("_native_pair_input_order", kind="mergesort")
        .drop(columns=["_native_pair_input_order"])
    )


def _add_roi_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_pair_scope_columns(out, operation="ROI temporal transitions")

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

    grain = list(MOTION_GRAIN_COLUMNS)
    out = out.sort_values([*grain, "frame_index"], kind="mergesort")
    g = out.groupby(grain, dropna=False, sort=False)

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
    previous_roi_available = (
        g["roi_target_available"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    adjacent_pair_valid = _to_bool_series(
        out.get("adjacent_motion_pair_valid", pd.Series(False, index=out.index))
    )
    out["roi_transition_pair_valid"] = (
        adjacent_pair_valid
        & out["roi_target_available"]
        & previous_roi_available
    )
    out["roi_target_entry_event"] = (
        out["roi_transition_pair_valid"]
        & out["roi_target_contact"]
        & ~out["prev_roi_target_contact"]
    )
    out["roi_target_exit_event"] = (
        out["roi_transition_pair_valid"]
        & ~out["roi_target_contact"]
        & out["prev_roi_target_contact"]
    )
    out["roi_target_near_entry_event"] = (
        out["roi_transition_pair_valid"]
        & out["roi_target_near"]
        & ~out["prev_roi_target_near"]
    )
    out["roi_target_near_exit_event"] = (
        out["roi_transition_pair_valid"]
        & ~out["roi_target_near"]
        & out["prev_roi_target_near"]
    )
    out["roi_motion_inside_score"] = np.where(
        out["roi_target_contact"] | out["roi_target_near"],
        out.get("speed_n_per_frame", 0.0),
        0.0,
    )
    out["roi_motion_inside_score_per_second"] = np.where(
        out["roi_target_contact"] | out["roi_target_near"],
        out.get("speed_n_per_second", 0.0),
        0.0,
    )

    # Per-unit aggregate values are joined later in _add_temporal_unit_aggregates.
    return out.sort_index(kind="mergesort")


def _add_social_context_columns(df: pd.DataFrame, config: EnhancedFeatureConfig) -> pd.DataFrame:
    _require_pair_scope_columns(df, operation="social temporal context")
    out = build_static_social_context_features(
        df.reset_index(drop=True),
        near_distance_n=config.social_near_distance_n,
        contact_iou_threshold=config.social_contact_iou_threshold,
        contact_overlap_threshold=config.social_contact_overlap_threshold,
        max_frame_group_size=config.max_frame_group_size_for_social,
    )

    grain = list(MOTION_GRAIN_COLUMNS)
    out = out.sort_values([*grain, "frame_index"], kind="mergesort")
    g = out.groupby(grain, dropna=False, sort=False)
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
    finite_partner_distance = (
        out["nearest_dist_n"].notna() & out["prev_nearest_dist_n"].notna()
    )
    adjacent_pair_valid = _to_bool_series(
        out.get("adjacent_motion_pair_valid", pd.Series(False, index=out.index))
    )
    velocity_pair_valid = _to_bool_series(
        out.get("motion_velocity_pair_valid", pd.Series(False, index=out.index))
    )
    out["social_adjacent_pair_valid"] = (
        same_neighbor & finite_partner_distance & adjacent_pair_valid
    )
    out["social_velocity_pair_valid"] = (
        same_neighbor & finite_partner_distance & velocity_pair_valid
    )
    raw_partner_delta = out["nearest_dist_n"] - out["prev_nearest_dist_n"]
    out["nearest_dist_delta"] = np.where(
        out["social_adjacent_pair_valid"],
        raw_partner_delta,
        0.0,
    )
    out["nearest_dist_delta_sparse"] = np.where(
        out["social_velocity_pair_valid"] & ~out["social_adjacent_pair_valid"],
        raw_partner_delta,
        0.0,
    )
    out["partner_distance_delta_n"] = np.where(
        out["social_velocity_pair_valid"],
        raw_partner_delta,
        0.0,
    )
    out["approach_speed_n_per_frame"] = np.where(
        out["social_adjacent_pair_valid"],
        -out["nearest_dist_delta"] / denom_frame,
        0.0,
    )
    out["separation_speed_n_per_frame"] = np.where(
        out["social_adjacent_pair_valid"],
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
    delta_seconds = pd.to_numeric(
        out.get("motion_delta_seconds", np.nan),
        errors="coerce",
    )
    signed_partner_velocity = pd.Series(
        np.where(
            out["social_velocity_pair_valid"],
            raw_partner_delta / delta_seconds,
            0.0,
        ),
        index=out.index,
        dtype="float64",
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["approach_speed_n_per_second"] = (-signed_partner_velocity).clip(
        lower=0.0
    )
    out["retreat_speed_n_per_second"] = signed_partner_velocity.clip(lower=0.0)
    out["separation_speed_n_per_second"] = out[
        "retreat_speed_n_per_second"
    ]

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
    out["aggression_score_proxy_per_second"] = (
        out["pair_contact_with_nearest"].astype(float)
        * (
            out.get(
                "speed_n_per_second",
                pd.Series(0.0, index=out.index),
            ).fillna(0.0)
            + out["approach_speed_n_per_second"].fillna(0.0)
        )
        * (1.0 + out["social_density_near_count"].fillna(0.0))
    )

    return out.sort_index(kind="mergesort")


def _add_temporal_unit_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_pair_scope_columns(out, operation="temporal aggregation")
    internal_columns = [
        "_speed_n_per_frame_valid_pair",
        "_speed_n_per_second_valid_pair",
        "_acceleration_n_per_second2_valid_pair",
        "_accel_n_per_frame2_valid_pair",
        "_direction_change_valid_pair",
    ]
    out[internal_columns[0]] = pd.to_numeric(
        out["speed_n_per_frame"],
        errors="coerce",
    ).where(_to_bool_series(out["adjacent_motion_pair_valid"]))
    out[internal_columns[1]] = pd.to_numeric(
        out["speed_n_per_second"],
        errors="coerce",
    ).where(_to_bool_series(out["motion_velocity_pair_valid"]))
    out[internal_columns[2]] = pd.to_numeric(
        out["abs_acceleration_n_per_second2"],
        errors="coerce",
    ).where(_to_bool_series(out["acceleration_pair_valid"]))
    out[internal_columns[3]] = pd.to_numeric(
        out["abs_accel_n_per_frame2"],
        errors="coerce",
    ).where(_to_bool_series(out["acceleration_pair_valid"]))
    out[internal_columns[4]] = pd.to_numeric(
        out["abs_direction_change_rad"],
        errors="coerce",
    ).where(_to_bool_series(out["direction_change_pair_valid"]))
    g = out.groupby("temporal_unit_key", dropna=False, sort=False)

    agg_spec: dict[str, tuple[str, str | Any]] = {
        "speed_mean_unit": (internal_columns[0], "mean"),
        "speed_max_unit": (internal_columns[0], "max"),
        "speed_std_unit": (internal_columns[0], "std"),
        "accel_abs_mean_unit": (internal_columns[3], "mean"),
        "accel_abs_max_unit": (internal_columns[3], "max"),
        "direction_change_abs_mean_unit": (internal_columns[4], "mean"),
        "direction_change_abs_max_unit": (internal_columns[4], "max"),
        "path_length_n_unit": ("displacement_n", "sum"),
        "motion_energy_unit": (
            internal_columns[0],
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
    physical_agg_spec: dict[str, tuple[str, str | Any]] = {
        "speed_n_per_second_mean_unit": (internal_columns[1], "mean"),
        "speed_n_per_second_max_unit": (internal_columns[1], "max"),
        "speed_n_per_second_std_unit": (internal_columns[1], "std"),
        "motion_energy_n_per_second2_unit": (
            internal_columns[1],
            lambda s: float(
                np.nansum(np.asarray(s, dtype="float64") ** 2)
            ),
        ),
        "acceleration_n_per_second2_abs_mean_unit": (
            internal_columns[2],
            "mean",
        ),
        "acceleration_n_per_second2_abs_max_unit": (
            internal_columns[2],
            "max",
        ),
        "approach_speed_n_per_second_max_unit": (
            "approach_speed_n_per_second",
            "max",
        ),
        "retreat_speed_n_per_second_max_unit": (
            "retreat_speed_n_per_second",
            "max",
        ),
        "aggression_score_proxy_per_second_max_unit": (
            "aggression_score_proxy_per_second",
            "max",
        ),
        "aggression_score_proxy_per_second_mean_unit": (
            "aggression_score_proxy_per_second",
            "mean",
        ),
    }
    agg_spec.update(physical_agg_spec)

    available_agg = {
        out_col: pd.NamedAgg(column=in_col, aggfunc=func)
        for out_col, (in_col, func) in agg_spec.items()
        if in_col in out.columns
    }
    unit_agg = g.agg(**available_agg)
    observed = _to_bool_series(
        out.get("observed_mask", pd.Series(True, index=out.index))
    )
    coverage = pd.DataFrame(
        {
            "temporal_unit_key": out["temporal_unit_key"].astype(str),
            "observed_frame_count": observed.astype("int64"),
            "valid_pair_count": _to_bool_series(
                out["valid_motion_pair"]
            ).astype("int64"),
        }
    ).groupby("temporal_unit_key", sort=False).sum()
    coverage["possible_pair_count"] = np.maximum(
        coverage["observed_frame_count"] - 1,
        0,
    )
    possible_denominator = coverage["possible_pair_count"].replace(
        0,
        np.nan,
    )
    coverage["valid_pair_ratio"] = (
        coverage["valid_pair_count"]
        .div(possible_denominator)
        .fillna(0.0)
    )
    coverage["motion_feature_coverage"] = coverage["valid_pair_ratio"]
    coverage["motion_feature_available"] = coverage[
        "valid_pair_count"
    ].gt(0)
    coverage["motion_feature_coverage_available"] = coverage[
        "possible_pair_count"
    ].gt(0)
    unit_agg = unit_agg.join(coverage)

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
    unit_agg["motion_burstiness_n_per_second_unit"] = unit_agg.get(
        "speed_n_per_second_std_unit",
        np.nan,
    ) / (unit_agg.get("speed_n_per_second_mean_unit", np.nan) + 1e-9)
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
        "motion_energy_n_per_second2_unit",
        "shape_transition_score_unit",
        "target_roi_contact_ratio_unit",
        "target_roi_near_ratio_unit",
        "pair_contact_ratio_unit",
        "motion_burstiness_unit",
        "motion_burstiness_n_per_second_unit",
        "bbox_stability_unit",
        "speed_n_per_second_mean_unit",
        "speed_n_per_second_max_unit",
        "speed_n_per_second_std_unit",
        "acceleration_n_per_second2_abs_mean_unit",
        "acceleration_n_per_second2_abs_max_unit",
    ]
    for col in numeric_fill:
        if col in out.columns:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    return out.drop(columns=internal_columns)


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
        & out.get("speed_n_per_second_mean_unit", 0)
        .fillna(0)
        .lt(0.06),
        "motion_label_low_motion",
    )
    add_reason(
        out["behavior"].eq("stand")
        & out.get("speed_n_per_second_mean_unit", 0)
        .fillna(0)
        .gt(0.30),
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


def _angle_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    diff = a - b
    return (diff + math.pi) % (2 * math.pi) - math.pi


def _require_pair_scope_columns(
    frame: pd.DataFrame,
    *,
    operation: str,
) -> None:
    missing = sorted(set(MOTION_GRAIN_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(
            f"{operation} requires authoritative pair-scope columns: "
            f"{missing}"
        )
    for column in ("object_track_key", "temporal_unit_key"):
        blank = int(
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
        if blank:
            raise ValueError(
                f"{operation} requires nonblank {column}: rows={blank}"
            )


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

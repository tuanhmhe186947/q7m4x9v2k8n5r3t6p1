"""Audit T6/T8/T12/T16/S6@16 availability and source shortcuts."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "source_type",
    "view_type",
    "sampling_pattern",
    "physical_span_seconds",
    "expected_slot_count",
    "observed_slot_count",
    "pair_delta_frames",
    "pair_delta_seconds",
    "primary_cross_source_eligible",
}
SOURCE_SIGNATURE_COLUMNS = (
    "view_type",
    "sampling_pattern",
    "physical_span_seconds",
    "expected_slot_count",
    "observed_slot_count",
    "pair_delta_frames",
    "pair_delta_seconds",
)
CONTIGUOUS_VIEW_LENGTHS = (6, 8, 12, 16)


def audit_pre_review_structural_view_availability(
    frames: pd.DataFrame,
    *,
    source_fps: float,
    legacy_window_stride: int = 3,
) -> dict[str, Any]:
    """Count exact source-frame views without claiming review eligibility."""
    required = {
        "source_type",
        "object_track_key",
        "temporal_unit_key",
        "frame_index",
    }
    missing = sorted(required.difference(frames.columns))
    if missing:
        raise ValueError(f"structural view audit missing columns: {missing}")
    fps = float(source_fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("source_fps must be finite and > 0")
    if legacy_window_stride <= 0:
        raise ValueError("legacy_window_stride must be > 0")

    work = frames.loc[:, sorted(required)].drop_duplicates().copy()
    work["frame_index"] = pd.to_numeric(
        work["frame_index"],
        errors="raise",
    ).astype(int)
    records: list[dict[str, Any]] = []
    for source, source_rows in work.groupby("source_type", sort=True):
        if source == "legacy_recovered":
            records.extend(
                _legacy_structural_views(
                    source_rows,
                    fps=fps,
                    stride=legacy_window_stride,
                )
            )
        elif source == "cvat_tracking_xml":
            records.extend(_cvat_structural_views(source_rows, fps=fps))
    view_rows = pd.DataFrame.from_records(records)
    audit = audit_final_view_contract(view_rows)
    audit.update(
        {
            "scope": "PRE_REVIEW_STRUCTURAL_ONLY",
            "review_eligibility_applied": False,
            "behavior_labels_consumed": False,
            "not_behavior_review_authority": True,
            "not_train_ready_authority": True,
            "canonical_source_fps": fps,
            "view_dependent_features_must_be_recomputed_after_apply": [
                "displacement_speed_acceleration",
                "heading_path_and_transitions",
                "pen_normal_parallel_motion",
                "social_motion_trends",
                "hidden_ratio_and_run",
                "duration_fps_masks_weights_eligibility",
                "all_temporal_aggregates",
            ],
        }
    )
    return audit


def audit_final_view_contract(
    windows: pd.DataFrame,
    *,
    direct_accuracy_threshold: float = 0.95,
    minimum_uplift: float = 0.10,
) -> dict[str, Any]:
    """Report view schemas, timing, availability, and source predictability."""
    missing = sorted(REQUIRED_COLUMNS.difference(windows.columns))
    if missing:
        return {
            "schema_version": "classification_v2.final_view_audit.v1",
            "rows": int(len(windows)),
            "errors": [f"missing_columns={missing}"],
            "primary_view_recommendation": None,
        }

    work = windows.copy()
    work["source_type"] = work["source_type"].fillna("").astype(str)
    work["view_type"] = work["view_type"].fillna("").astype(str)
    work["sampling_pattern"] = (
        work["sampling_pattern"].fillna("").astype(str)
    )
    eligible = _to_bool_series(work["primary_cross_source_eligible"])
    errors: list[str] = []
    invalid_sparse_primary = work["view_type"].eq("S6@16") & eligible
    if invalid_sparse_primary.any():
        errors.append(
            "S6@16_marked_primary_cross_source_eligible="
            f"{int(invalid_sparse_primary.sum())}"
        )

    availability = _source_view_availability(work)
    predictability: dict[str, Any] = {
        "all_views": _source_signature_report(
            work,
            SOURCE_SIGNATURE_COLUMNS,
            direct_accuracy_threshold=direct_accuracy_threshold,
            minimum_uplift=minimum_uplift,
        ),
        "primary_eligible_views": _source_signature_report(
            work.loc[eligible],
            SOURCE_SIGNATURE_COLUMNS,
            direct_accuracy_threshold=direct_accuracy_threshold,
            minimum_uplift=minimum_uplift,
        ),
        "by_view": {},
    }
    candidate_views: list[str] = []
    for view, group in work.groupby("view_type", sort=True):
        report = _source_signature_report(
            group,
            tuple(
                column
                for column in SOURCE_SIGNATURE_COLUMNS
                if column not in {"view_type", "sampling_pattern"}
            ),
            direct_accuracy_threshold=direct_accuracy_threshold,
            minimum_uplift=minimum_uplift,
        )
        predictability["by_view"][str(view)] = report
        source_count = int(group["source_type"].nunique())
        is_contiguous = group["sampling_pattern"].eq("contiguous").all()
        has_primary_rows = _to_bool_series(
            group["primary_cross_source_eligible"]
        ).all()
        if (
            source_count >= 2
            and is_contiguous
            and has_primary_rows
            and not report["near_direct_source_signature"]
        ):
            candidate_views.append(str(view))

    candidate_views.sort(key=_view_sort_key)
    primary = candidate_views[0] if candidate_views else None
    if primary is None:
        errors.append("no_cross_source_primary_view_without_metadata_shortcut")

    ablations: list[dict[str, str]] = []
    for view in sorted(work["view_type"].unique(), key=_view_sort_key):
        if view == primary:
            continue
        group = work.loc[work["view_type"].eq(view)]
        if view == "S6@16":
            reason = "legacy_only_sparse_ablation_not_primary"
        elif group["source_type"].nunique() < 2:
            reason = "single_source_ablation"
        else:
            reason = "cross_length_ablation"
        ablations.append({"view_type": str(view), "reason": reason})

    return {
        "schema_version": "classification_v2.final_view_audit.v1",
        "rows": int(len(work)),
        "per_view_feature_schema": _per_view_schema(work),
        "per_view_physical_span_distribution": _span_distribution(work),
        "source_by_view_availability": availability,
        "source_predictability_from_view_metadata": predictability,
        "primary_view_recommendation": primary,
        "ablation_view_recommendations": ablations,
        "primary_model_must_select_exactly_one_view_type": True,
        "errors": errors,
    }


def _legacy_structural_views(
    frame: pd.DataFrame,
    *,
    fps: float,
    stride: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, unit in frame.groupby("temporal_unit_key", sort=False):
        source_frames = sorted(unit["frame_index"].unique().tolist())
        if len(source_frames) != 16:
            continue
        start = source_frames[0]
        if source_frames != list(range(start, start + 16)):
            continue
        for length in CONTIGUOUS_VIEW_LENGTHS:
            for _ in range(0, 16 - length + 1, stride):
                records.append(
                    _structural_view_record(
                        source_type="legacy_recovered",
                        view_type=f"T{length}_contiguous",
                        sampling_pattern="contiguous",
                        expected_slot_count=length,
                        pair_delta_frames=[1] * (length - 1),
                        fps=fps,
                    )
                )
        records.append(
            _structural_view_record(
                source_type="legacy_recovered",
                view_type="S6@16",
                sampling_pattern=(
                    "uniform_sparse_offsets_0_3_6_9_12_15"
                ),
                expected_slot_count=6,
                pair_delta_frames=[3] * 5,
                fps=fps,
                primary_eligible=False,
            )
        )
    return records


def _cvat_structural_views(
    frame: pd.DataFrame,
    *,
    fps: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, track in frame.groupby("object_track_key", sort=False):
        frame_set = set(track["frame_index"].tolist())
        starts = (
            track.groupby("temporal_unit_key", sort=False)["frame_index"]
            .min()
            .sort_values()
            .tolist()
        )
        for start in starts:
            for length in CONTIGUOUS_VIEW_LENGTHS:
                expected = set(range(int(start), int(start) + length))
                if not expected.issubset(frame_set):
                    continue
                records.append(
                    _structural_view_record(
                        source_type="cvat_tracking_xml",
                        view_type=f"T{length}_contiguous",
                        sampling_pattern="contiguous",
                        expected_slot_count=length,
                        pair_delta_frames=[1] * (length - 1),
                        fps=fps,
                    )
                )
    return records


def _structural_view_record(
    *,
    source_type: str,
    view_type: str,
    sampling_pattern: str,
    expected_slot_count: int,
    pair_delta_frames: list[int],
    fps: float,
    primary_eligible: bool = True,
) -> dict[str, Any]:
    pair_delta_seconds = [value / fps for value in pair_delta_frames]
    return {
        "source_type": source_type,
        "view_type": view_type,
        "sampling_pattern": sampling_pattern,
        "physical_span_seconds": sum(pair_delta_seconds),
        "expected_slot_count": expected_slot_count,
        "observed_slot_count": expected_slot_count,
        "pair_delta_frames": json.dumps(pair_delta_frames),
        "pair_delta_seconds": json.dumps(pair_delta_seconds),
        "primary_cross_source_eligible": primary_eligible,
    }


def _per_view_schema(frame: pd.DataFrame) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for view, group in frame.groupby("view_type", sort=True):
        output[str(view)] = sorted(
            column for column in frame.columns if group[column].notna().any()
        )
    return output


def _span_distribution(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (source, view), group in frame.groupby(
        ["source_type", "view_type"],
        sort=True,
    ):
        values = pd.to_numeric(
            group["physical_span_seconds"],
            errors="coerce",
        ).dropna()
        records.append(
            {
                "source_type": str(source),
                "view_type": str(view),
                "count": int(len(values)),
                "p10": _quantile(values, 0.10),
                "p50": _quantile(values, 0.50),
                "p90": _quantile(values, 0.90),
            }
        )
    return records


def _source_view_availability(frame: pd.DataFrame) -> dict[str, Any]:
    table = pd.crosstab(frame["source_type"], frame["view_type"])
    return {
        str(source): {str(view): int(value) for view, value in row.items()}
        for source, row in table.iterrows()
    }


def _source_signature_report(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    direct_accuracy_threshold: float,
    minimum_uplift: float,
) -> dict[str, Any]:
    if frame.empty:
        return _empty_signature_report(columns)
    sources = frame["source_type"].fillna("").astype(str)
    signatures = frame.loc[:, list(columns)].apply(
        lambda row: json.dumps(
            [_canonical_value(value) for value in row.tolist()],
            separators=(",", ":"),
        ),
        axis=1,
    )
    table = pd.crosstab(signatures, sources)
    rows = int(table.to_numpy().sum())
    baseline = float(sources.value_counts().max() / rows)
    accuracy = float(table.max(axis=1).sum() / rows)
    pure = table.gt(0).sum(axis=1).eq(1)
    pure_rows = int(table.loc[pure].to_numpy().sum())
    uplift = accuracy - baseline
    near_direct = bool(
        sources.nunique() >= 2
        and accuracy >= direct_accuracy_threshold
        and uplift >= minimum_uplift
    )
    return {
        "rows": rows,
        "source_count": int(sources.nunique()),
        "columns": list(columns),
        "unique_signatures": int(len(table)),
        "majority_source_baseline": baseline,
        "signature_mapping_accuracy": accuracy,
        "uplift_over_majority": uplift,
        "source_pure_signature_row_ratio": float(pure_rows / rows),
        "near_direct_source_signature": near_direct,
    }


def _empty_signature_report(columns: tuple[str, ...]) -> dict[str, Any]:
    return {
        "rows": 0,
        "source_count": 0,
        "columns": list(columns),
        "unique_signatures": 0,
        "majority_source_baseline": 0.0,
        "signature_mapping_accuracy": 0.0,
        "uplift_over_majority": 0.0,
        "source_pure_signature_row_ratio": 0.0,
        "near_direct_source_signature": False,
    }


def _canonical_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _quantile(values: pd.Series, quantile: float) -> float | None:
    return float(values.quantile(quantile)) if len(values) else None


def _view_sort_key(view: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"T(\d+)_contiguous", str(view))
    if match:
        return (0, int(match.group(1)), str(view))
    if view == "S6@16":
        return (2, 6, str(view))
    return (1, 0, str(view))


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )

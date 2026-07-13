"""Label-independent temporal evidence for behavior classification.

The functions in this module only read signals available at inference time.
They never inspect behavior labels, review decisions, Hidden annotations, or
target-selected ROI fields. The same summarizer is used for native temporal
units and arbitrary training windows so their feature semantics stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ROI_CLASSES: tuple[str, ...] = ("feeder", "drinker", "toy")


@dataclass(frozen=True, slots=True)
class TemporalEvidenceConfig:
    """Fixed, label-independent thresholds for descriptive evidence."""

    stationary_speed_threshold: float = 0.002
    active_speed_threshold: float = 0.006
    turning_angle_threshold_rad: float = float(np.pi / 6.0)

    def validate(self) -> None:
        """Reject ambiguous or scientifically inconsistent thresholds."""

        if self.stationary_speed_threshold < 0:
            raise ValueError("stationary_speed_threshold must be >= 0")
        if self.active_speed_threshold <= self.stationary_speed_threshold:
            raise ValueError(
                "active_speed_threshold must exceed stationary_speed_threshold"
            )
        if not 0 < self.turning_angle_threshold_rad <= np.pi:
            raise ValueError("turning_angle_threshold_rad must be in (0, pi]")


MOTION_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "temporal_observation_ratio",
    "temporal_pair_coverage_ratio",
    "temporal_contiguous_pair_ratio",
    "temporal_max_gap_frames",
    "temporal_duplicate_frame_ratio",
    "temporal_timing_valid_ratio",
    "motion_speed_p10",
    "motion_speed_p50",
    "motion_speed_p90",
    "motion_speed_p95",
    "motion_active_ratio",
    "motion_stationary_ratio",
    "motion_intermediate_ratio",
    "motion_longest_active_run_ratio",
    "motion_longest_stationary_run_ratio",
    "motion_active_episode_count",
    "motion_stationary_episode_count",
    "motion_speed_trend",
    "motion_speed_trend_valid",
    "motion_jerk_abs_mean",
    "motion_jerk_abs_p90",
    "turning_abs_mean",
    "turning_rate",
    "turning_direction_concentration",
    "turning_pair_coverage_ratio",
    "trajectory_path_length_n",
    "trajectory_displacement_n",
    "trajectory_straightness",
    "trajectory_tortuosity_log1p",
)

BBOX_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "bbox_area_p10",
    "bbox_area_p50",
    "bbox_area_p90",
    "bbox_area_robust_spread",
    "bbox_aspect_p10",
    "bbox_aspect_p50",
    "bbox_aspect_p90",
    "bbox_aspect_robust_spread",
    "bbox_shape_change_p90",
)

ROI_EVIDENCE_METRICS: tuple[str, ...] = (
    "availability_ratio",
    "near_ratio",
    "contact_ratio",
    "contact_longest_run_ratio",
    "contact_episode_count",
    "min_dist_p10",
    "min_dist_p50",
    "overlap_p90",
)

ROI_EVIDENCE_COLUMNS: tuple[str, ...] = tuple(
    f"roi_{roi_class}_{metric}"
    for roi_class in ROI_CLASSES
    for metric in ROI_EVIDENCE_METRICS
)

SOCIAL_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "social_neighbor_availability_ratio",
    "social_partner_persistence_ratio",
    "social_partner_turnover_rate",
    "social_pair_contact_ratio",
    "social_contact_longest_run_ratio",
    "social_contact_episode_count",
    "social_approach_ratio",
    "social_nearest_dist_p10",
    "social_nearest_dist_p50",
    "social_nearest_dist_p90",
    "social_aggression_proxy_p90",
)

TEMPORAL_EVIDENCE_BASE_COLUMNS: tuple[str, ...] = (
    *MOTION_EVIDENCE_COLUMNS,
    *BBOX_EVIDENCE_COLUMNS,
    *ROI_EVIDENCE_COLUMNS,
    *SOCIAL_EVIDENCE_COLUMNS,
)

UNIT_TEMPORAL_EVIDENCE_COLUMNS: tuple[str, ...] = tuple(
    f"{column}_unit" for column in TEMPORAL_EVIDENCE_BASE_COLUMNS
)

WINDOW_TEMPORAL_EVIDENCE_COLUMNS: tuple[str, ...] = tuple(
    f"{column}_window" for column in TEMPORAL_EVIDENCE_BASE_COLUMNS
)


def summarize_temporal_evidence(
    frame_rows: pd.DataFrame,
    *,
    expected_start: int | float | None = None,
    expected_end: int | float | None = None,
    suffix: str = "",
    config: TemporalEvidenceConfig | None = None,
) -> dict[str, float | int | bool]:
    """Summarize inference-safe evidence without reading a behavior label.

    Motion is recomputed inside the supplied unit/window. This prevents the
    first row from inheriting a displacement from a previous review unit.
    Invalid or duplicate frame pairs remain represented through coverage
    metrics instead of being silently removed.
    """

    config = config or TemporalEvidenceConfig()
    config.validate()
    work = _ordered_rows(frame_rows)
    frames = _numeric_array(work, "frame_index")
    expected_count = _expected_frame_count(frames, expected_start, expected_end)
    row_valid = _row_quality_mask(work)

    summary: dict[str, float | int | bool] = {}
    summary.update(_temporal_quality_summary(work, frames, expected_count))
    summary.update(
        _motion_summary(
            work,
            frames,
            expected_count,
            row_valid,
            config,
        )
    )
    summary.update(_bbox_summary(work, frames, row_valid))
    summary.update(_roi_summary(work, frames, row_valid))
    summary.update(_social_summary(work, frames, row_valid))

    missing = sorted(set(TEMPORAL_EVIDENCE_BASE_COLUMNS).difference(summary))
    if missing:
        raise RuntimeError(f"temporal evidence implementation missing columns: {missing}")
    return {f"{name}{suffix}": summary[name] for name in TEMPORAL_EVIDENCE_BASE_COLUMNS}


def add_unit_temporal_evidence(
    frame_features: pd.DataFrame,
    *,
    config: TemporalEvidenceConfig | None = None,
) -> pd.DataFrame:
    """Attach one repeated evidence vector per native temporal unit.

    The function preserves row order, row count, existing labels, and keys.
    Repeated unit values make downstream interval construction deterministic.
    """

    if "temporal_unit_key" not in frame_features.columns:
        raise ValueError("temporal_unit_key is required for unit evidence")
    out = frame_features.copy()
    row_count = len(out)
    unit_summaries: list[dict[str, Any]] = []
    for unit_key, group in out.groupby(
        "temporal_unit_key",
        dropna=False,
        sort=False,
    ):
        first = group.iloc[0]
        evidence = summarize_temporal_evidence(
            group,
            expected_start=_optional_number(first.get("label_window_start")),
            expected_end=_optional_number(first.get("label_window_end")),
            suffix="_unit",
            config=config,
        )
        unit_summaries.append({"temporal_unit_key": str(unit_key), **evidence})

    summary_table = pd.DataFrame(unit_summaries)
    if summary_table["temporal_unit_key"].duplicated().any():
        raise ValueError("duplicate temporal_unit_key in evidence summary")
    summary_table = summary_table.set_index("temporal_unit_key")
    keys = out["temporal_unit_key"].astype(str)
    for column in UNIT_TEMPORAL_EVIDENCE_COLUMNS:
        out[column] = keys.map(summary_table[column])
    if len(out) != row_count:
        raise RuntimeError("unit temporal evidence changed row count")
    return out


def attach_unit_evidence_to_intervals(
    intervals: pd.DataFrame,
    frame_features: pd.DataFrame,
) -> pd.DataFrame:
    """Copy validated constant unit evidence onto one-row-per-unit intervals."""

    if intervals.empty:
        return intervals.copy()
    available = [
        column
        for column in UNIT_TEMPORAL_EVIDENCE_COLUMNS
        if column in frame_features.columns
    ]
    if not available:
        return intervals.copy()
    if "temporal_unit_key" not in frame_features.columns:
        raise ValueError("frame features lack temporal_unit_key")

    grouped = frame_features.groupby("temporal_unit_key", dropna=False, sort=False)
    inconsistent = {
        column: int(grouped[column].nunique(dropna=False).gt(1).sum())
        for column in available
    }
    inconsistent = {key: value for key, value in inconsistent.items() if value}
    if inconsistent:
        raise ValueError(f"nonconstant unit temporal evidence: {inconsistent}")

    evidence = grouped[available].first().reset_index()
    out = intervals.merge(
        evidence,
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    if len(out) != len(intervals):
        raise RuntimeError("attaching unit evidence changed interval row count")
    return out


def _ordered_rows(frame_rows: pd.DataFrame) -> pd.DataFrame:
    """Return stable temporal order while retaining duplicate observations."""

    work = frame_rows.copy()
    if "frame_index" not in work.columns:
        work["frame_index"] = np.nan
    work["frame_index"] = pd.to_numeric(work["frame_index"], errors="coerce")
    work["_evidence_order"] = np.arange(len(work), dtype="int64")
    return work.sort_values(
        ["frame_index", "_evidence_order"],
        kind="mergesort",
    ).reset_index(drop=True)


def _temporal_quality_summary(
    work: pd.DataFrame,
    frames: np.ndarray,
    expected_count: int,
) -> dict[str, float | int]:
    """Quantify temporal completeness without rejecting incomplete samples."""

    finite_frames = frames[np.isfinite(frames)]
    unique_count = int(np.unique(finite_frames).size)
    row_count = int(len(work))
    deltas = np.diff(frames)
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    valid_pair_count = int(len(positive))
    expected_pairs = max(0, expected_count - 1)
    contiguous_count = int(np.isclose(positive, 1.0).sum())

    timestamps = _numeric_array(work, "timestamp_sec")
    time_deltas = np.diff(timestamps)
    valid_time = np.isfinite(time_deltas) & (time_deltas > 0)
    return {
        "temporal_observation_ratio": _bounded_ratio(unique_count, expected_count),
        "temporal_pair_coverage_ratio": _bounded_ratio(
            valid_pair_count,
            expected_pairs,
        ),
        "temporal_contiguous_pair_ratio": _bounded_ratio(
            contiguous_count,
            expected_pairs,
        ),
        "temporal_max_gap_frames": float(positive.max()) if positive.size else 0.0,
        "temporal_duplicate_frame_ratio": _bounded_ratio(
            max(0, row_count - unique_count),
            row_count,
        ),
        "temporal_timing_valid_ratio": _bounded_ratio(
            int(valid_time.sum()),
            expected_pairs,
        ),
    }


def _motion_summary(
    work: pd.DataFrame,
    frames: np.ndarray,
    expected_count: int,
    row_valid: np.ndarray,
    config: TemporalEvidenceConfig,
) -> dict[str, float | int | bool]:
    """Describe speed, runs, jerk, turning, and trajectory shape."""

    cx = _numeric_array(work, "cx_n")
    cy = _numeric_array(work, "cy_n")
    delta_frame = np.diff(frames)
    dx = np.diff(cx)
    dy = np.diff(cy)
    pair_valid = (
        np.isfinite(delta_frame)
        & (delta_frame > 0)
        & np.isfinite(dx)
        & np.isfinite(dy)
        & row_valid[:-1]
        & row_valid[1:]
    )
    contiguous = pair_valid & np.isclose(delta_frame, 1.0)
    distance = np.hypot(dx, dy)
    speed = np.full(distance.shape, np.nan, dtype="float64")
    speed[pair_valid] = distance[pair_valid] / delta_frame[pair_valid]
    valid_speed = speed[np.isfinite(speed)]

    active = pair_valid & (speed >= config.active_speed_threshold)
    stationary = pair_valid & (speed <= config.stationary_speed_threshold)
    intermediate = pair_valid & ~(active | stationary)
    valid_pair_count = int(pair_valid.sum())
    active_run = _pair_run_stats(active, pair_valid, contiguous)
    stationary_run = _pair_run_stats(stationary, pair_valid, contiguous)

    speed_trend, speed_trend_valid = _linear_trend(
        frames[1:][pair_valid],
        speed[pair_valid],
    )
    jerk = _jerk_values(speed, pair_valid, contiguous)
    turning = _turning_summary(speed, dx, dy, pair_valid, contiguous, config)

    path_length = float(np.nansum(distance[pair_valid])) if pair_valid.any() else 0.0
    displacement = _connected_displacement(
        cx,
        cy,
        pair_valid,
    )
    straightness = (
        float(np.clip(displacement / path_length, 0.0, 1.0))
        if path_length > 0
        else 0.0
    )
    tortuosity_excess = max(
        0.0,
        path_length / max(displacement, 1e-9) - 1.0,
    )
    tortuosity = float(np.log1p(tortuosity_excess)) if path_length > 0 else 0.0
    return {
        "motion_speed_p10": _quantile(valid_speed, 0.10),
        "motion_speed_p50": _quantile(valid_speed, 0.50),
        "motion_speed_p90": _quantile(valid_speed, 0.90),
        "motion_speed_p95": _quantile(valid_speed, 0.95),
        "motion_active_ratio": _bounded_ratio(int(active.sum()), valid_pair_count),
        "motion_stationary_ratio": _bounded_ratio(
            int(stationary.sum()),
            valid_pair_count,
        ),
        "motion_intermediate_ratio": _bounded_ratio(
            int(intermediate.sum()),
            valid_pair_count,
        ),
        "motion_longest_active_run_ratio": _bounded_ratio(
            active_run["longest"],
            valid_pair_count,
        ),
        "motion_longest_stationary_run_ratio": _bounded_ratio(
            stationary_run["longest"],
            valid_pair_count,
        ),
        "motion_active_episode_count": int(active_run["episodes"]),
        "motion_stationary_episode_count": int(stationary_run["episodes"]),
        "motion_speed_trend": speed_trend,
        "motion_speed_trend_valid": speed_trend_valid,
        "motion_jerk_abs_mean": _mean_abs(jerk),
        "motion_jerk_abs_p90": _quantile(np.abs(jerk), 0.90),
        **turning,
        "trajectory_path_length_n": path_length,
        "trajectory_displacement_n": displacement,
        "trajectory_straightness": straightness,
        "trajectory_tortuosity_log1p": tortuosity,
    }


def _turning_summary(
    speed: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    pair_valid: np.ndarray,
    contiguous: np.ndarray,
    config: TemporalEvidenceConfig,
) -> dict[str, float]:
    """Measure direction stability only while the actor is visibly moving."""

    moving = pair_valid & (speed > config.stationary_speed_threshold)
    direction = np.full(speed.shape, np.nan, dtype="float64")
    direction[moving] = np.arctan2(dy[moving], dx[moving])
    turn_valid = moving[:-1] & moving[1:] & contiguous[:-1] & contiguous[1:]
    changes = _angle_difference(direction[1:], direction[:-1])[turn_valid]
    moving_direction = direction[np.isfinite(direction)]
    concentration = 0.0
    if moving_direction.size:
        resultant = np.mean(np.exp(1j * moving_direction))
        concentration = float(np.clip(np.abs(resultant), 0.0, 1.0))
    return {
        "turning_abs_mean": _mean_abs(changes),
        "turning_rate": _bounded_ratio(
            int((np.abs(changes) >= config.turning_angle_threshold_rad).sum()),
            int(changes.size),
        ),
        "turning_direction_concentration": concentration,
        "turning_pair_coverage_ratio": _bounded_ratio(
            int(changes.size),
            max(0, len(speed) - 1),
        ),
    }


def _bbox_summary(
    work: pd.DataFrame,
    frames: np.ndarray,
    row_valid: np.ndarray,
) -> dict[str, float]:
    """Use robust bbox quantiles as weak posture/shape descriptors."""

    area = _finite(_numeric_array(work, "area_n")[row_valid])
    aspect = _finite(_numeric_array(work, "aspect_ratio")[row_valid])
    shape_valid = np.zeros(len(work), dtype=bool)
    if len(work) > 1:
        contiguous = np.isclose(np.diff(frames), 1.0)
        shape_valid[1:] = row_valid[:-1] & row_valid[1:] & contiguous
    shape_change = _finite(
        _numeric_array(work, "shape_change_score")[shape_valid]
    )
    area_q = [_quantile(area, q) for q in (0.10, 0.50, 0.90)]
    aspect_q = [_quantile(aspect, q) for q in (0.10, 0.50, 0.90)]
    return {
        "bbox_area_p10": area_q[0],
        "bbox_area_p50": area_q[1],
        "bbox_area_p90": area_q[2],
        "bbox_area_robust_spread": _robust_spread(area_q),
        "bbox_aspect_p10": aspect_q[0],
        "bbox_aspect_p50": aspect_q[1],
        "bbox_aspect_p90": aspect_q[2],
        "bbox_aspect_robust_spread": _robust_spread(aspect_q),
        "bbox_shape_change_p90": _quantile(shape_change, 0.90),
    }


def _roi_summary(
    work: pd.DataFrame,
    frames: np.ndarray,
    row_valid: np.ndarray,
) -> dict[str, float | int]:
    """Summarize each physical ROI independently of the behavior label."""

    out: dict[str, float | int] = {}
    for roi_class in ROI_CLASSES:
        prefix = f"roi_{roi_class}"
        distance = _numeric_array(work, f"{prefix}_min_dist_n")
        overlap = _numeric_array(work, f"{prefix}_max_overlap_ratio")
        near = _bool_array(work, f"{prefix}_near")
        contact = _bool_array(work, f"{prefix}_contact")
        distance_available = np.isfinite(distance)
        if f"{prefix}_available" in work.columns:
            available = _bool_array(work, f"{prefix}_available")
            available = available & distance_available & row_valid
        else:
            available = distance_available & row_valid
        frame_runs = _frame_run_stats(contact, available, frames)
        available_count = int(available.sum())
        out[f"{prefix}_availability_ratio"] = _bounded_ratio(
            available_count,
            len(work),
        )
        out[f"{prefix}_near_ratio"] = _bounded_ratio(
            int((near & available).sum()),
            available_count,
        )
        out[f"{prefix}_contact_ratio"] = _bounded_ratio(
            int((contact & available).sum()),
            available_count,
        )
        out[f"{prefix}_contact_longest_run_ratio"] = _bounded_ratio(
            frame_runs["longest"],
            available_count,
        )
        out[f"{prefix}_contact_episode_count"] = int(frame_runs["episodes"])
        out[f"{prefix}_min_dist_p10"] = _quantile(distance[available], 0.10)
        out[f"{prefix}_min_dist_p50"] = _quantile(distance[available], 0.50)
        out[f"{prefix}_overlap_p90"] = _quantile(overlap[available], 0.90)
    return out


def _social_summary(
    work: pd.DataFrame,
    frames: np.ndarray,
    row_valid: np.ndarray,
) -> dict[str, float | int]:
    """Measure partner/contact persistence without exporting partner identity."""

    if "nearest_pig_id" in work.columns:
        partner = work["nearest_pig_id"].fillna("").astype(str).to_numpy()
    else:
        partner = np.full(len(work), "", dtype=object)
    if "nearest_track_id" in work.columns:
        track_partner = work["nearest_track_id"].fillna("").astype(str).to_numpy()
        has_pig_partner = np.char.str_len(partner.astype(str)) > 0
        partner = np.where(has_pig_partner, partner, track_partner)
    neighbor_available = (
        np.char.str_len(partner.astype(str)) > 0
    ) & row_valid
    available_count = int(neighbor_available.sum())
    persistence = 0.0
    if available_count:
        _, counts = np.unique(partner[neighbor_available], return_counts=True)
        persistence = float(counts.max() / available_count)

    contiguous = np.zeros(len(work), dtype=bool)
    if len(work) > 1:
        contiguous[1:] = np.isclose(np.diff(frames), 1.0)
    comparable = (
        neighbor_available[1:]
        & neighbor_available[:-1]
        & contiguous[1:]
    )
    turnover = int((partner[1:] != partner[:-1])[comparable].sum())

    contact = _bool_array(work, "pair_contact_with_nearest")
    contact_runs = _frame_run_stats(contact, neighbor_available, frames)
    approach = _numeric_array(work, "approach_speed_n_per_frame")
    nearest_dist = _finite(
        _numeric_array(work, "nearest_dist_n")[neighbor_available]
    )
    aggression = _finite(
        _numeric_array(work, "aggression_score_proxy")[row_valid]
    )
    return {
        "social_neighbor_availability_ratio": _bounded_ratio(
            available_count,
            len(work),
        ),
        "social_partner_persistence_ratio": persistence,
        "social_partner_turnover_rate": _bounded_ratio(
            turnover,
            int(comparable.sum()),
        ),
        "social_pair_contact_ratio": _bounded_ratio(
            int((contact & neighbor_available).sum()),
            available_count,
        ),
        "social_contact_longest_run_ratio": _bounded_ratio(
            contact_runs["longest"],
            available_count,
        ),
        "social_contact_episode_count": int(contact_runs["episodes"]),
        "social_approach_ratio": _bounded_ratio(
            int(((approach > 0) & neighbor_available).sum()),
            available_count,
        ),
        "social_nearest_dist_p10": _quantile(nearest_dist, 0.10),
        "social_nearest_dist_p50": _quantile(nearest_dist, 0.50),
        "social_nearest_dist_p90": _quantile(nearest_dist, 0.90),
        "social_aggression_proxy_p90": _quantile(aggression, 0.90),
    }


def _pair_run_stats(
    active: np.ndarray,
    valid: np.ndarray,
    contiguous: np.ndarray,
) -> dict[str, int]:
    """Count episodes in pair-aligned evidence while respecting frame gaps."""

    longest = 0
    current = 0
    episodes = 0
    for index, flag in enumerate(active):
        continues = index == 0 or bool(contiguous[index] and contiguous[index - 1])
        if bool(flag and valid[index]):
            if current == 0 or not continues:
                episodes += 1
                current = 1
            else:
                current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {"longest": longest, "episodes": episodes}


def _frame_run_stats(
    active: np.ndarray,
    valid: np.ndarray,
    frames: np.ndarray,
) -> dict[str, int]:
    """Count frame-aligned persistence episodes and break on missing frames."""

    longest = 0
    current = 0
    episodes = 0
    for index, flag in enumerate(active):
        continues = (
            index > 0
            and np.isfinite(frames[index])
            and np.isfinite(frames[index - 1])
            and np.isclose(frames[index] - frames[index - 1], 1.0)
        )
        if bool(flag and valid[index]):
            if current == 0 or not continues:
                episodes += 1
                current = 1
            else:
                current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {"longest": longest, "episodes": episodes}


def _jerk_values(
    speed: np.ndarray,
    pair_valid: np.ndarray,
    contiguous: np.ndarray,
) -> np.ndarray:
    """Return jerk only where three consecutive motion pairs are observed."""

    if speed.size < 3:
        return np.array([], dtype="float64")
    accel = np.full(max(0, speed.size - 1), np.nan, dtype="float64")
    accel_valid = (
        pair_valid[:-1]
        & pair_valid[1:]
        & contiguous[:-1]
        & contiguous[1:]
    )
    accel[accel_valid] = speed[1:][accel_valid] - speed[:-1][accel_valid]
    jerk_valid = np.isfinite(accel[:-1]) & np.isfinite(accel[1:])
    return (accel[1:] - accel[:-1])[jerk_valid]


def _linear_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """Fit a simple within-unit slope when at least three points exist."""

    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or float(np.ptp(x[valid])) <= 0:
        return 0.0, False
    centered_x = x[valid] - float(np.mean(x[valid]))
    centered_y = y[valid] - float(np.mean(y[valid]))
    denominator = float(np.sum(centered_x**2))
    if denominator <= 0:
        return 0.0, False
    return float(np.sum(centered_x * centered_y) / denominator), True


def _connected_displacement(
    cx: np.ndarray,
    cy: np.ndarray,
    pair_valid: np.ndarray,
) -> float:
    """Sum endpoint displacement within connected valid trajectory segments."""

    usable = pair_valid
    total = 0.0
    start: int | None = None
    for pair_index, valid in enumerate(usable):
        if valid and start is None:
            start = pair_index
        is_last_pair = pair_index == len(usable) - 1
        if start is not None and (not valid or is_last_pair):
            end = pair_index + 1 if valid and is_last_pair else pair_index
            total += float(
                np.hypot(
                    cx[end] - cx[start],
                    cy[end] - cy[start],
                )
            )
            start = None
    return total


def _row_quality_mask(work: pd.DataFrame) -> np.ndarray:
    """Combine only inference-time geometry quality fields that are present."""

    valid = np.ones(len(work), dtype=bool)
    for column in [
        "bbox_valid",
        "actor_bbox_valid",
        "geometry_feature_valid",
        "spatiotemporal_feature_valid",
    ]:
        if column in work.columns:
            valid &= _bool_array(work, column)
    return valid


def _expected_frame_count(
    frames: np.ndarray,
    expected_start: int | float | None,
    expected_end: int | float | None,
) -> int:
    """Use declared unit/window bounds, falling back to the observed span."""

    start = _optional_number(expected_start)
    end = _optional_number(expected_end)
    if start is not None and end is not None and end >= start:
        return int(round(end - start + 1))
    finite = frames[np.isfinite(frames)]
    if finite.size:
        return max(1, int(round(float(finite.max() - finite.min() + 1))))
    return 0


def _numeric_array(work: pd.DataFrame, column: str) -> np.ndarray:
    """Read a numeric inference-time column or return all-NaN values."""

    if column not in work.columns:
        return np.full(len(work), np.nan, dtype="float64")
    return pd.to_numeric(work[column], errors="coerce").to_numpy(dtype="float64")


def _bool_array(work: pd.DataFrame, column: str) -> np.ndarray:
    """Read a tolerant boolean inference-time relation column."""

    if column not in work.columns:
        return np.zeros(len(work), dtype=bool)
    series = work[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
        .to_numpy(dtype=bool)
    )


def _quantile(values: np.ndarray, q: float) -> float:
    """Return a finite quantile or zero when evidence is unavailable."""

    finite = _finite(values)
    return float(np.quantile(finite, q)) if finite.size else 0.0


def _mean_abs(values: np.ndarray) -> float:
    """Return mean absolute finite magnitude or zero."""

    finite = _finite(values)
    return float(np.mean(np.abs(finite))) if finite.size else 0.0


def _finite(values: np.ndarray) -> np.ndarray:
    """Return finite float values without changing source rows."""

    array = np.asarray(values, dtype="float64")
    return array[np.isfinite(array)]


def _bounded_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a stable ratio in [0, 1] for count-like evidence."""

    if denominator is None or float(denominator) <= 0:
        return 0.0
    return float(np.clip(float(numerator) / float(denominator), 0.0, 1.0))


def _robust_spread(quantiles: list[float]) -> float:
    """Scale the p90-p10 span by the median to reduce camera-scale effects."""

    p10, p50, p90 = quantiles
    return float((p90 - p10) / (abs(p50) + 1e-9))


def _angle_difference(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Return wrapped signed angle differences in [-pi, pi]."""

    return (current - previous + np.pi) % (2.0 * np.pi) - np.pi


def _optional_number(value: Any) -> float | None:
    """Convert nullable scalar metadata without accepting nonfinite values."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None

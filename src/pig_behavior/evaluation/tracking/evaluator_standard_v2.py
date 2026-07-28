"""Scientifically versioned tracking evaluation under Standard V2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

import pandas as pd

from .contracts import (
    EVALUATOR_CONTRACT_ID,
    HOTA_ALPHAS,
    IDENTITY_EPISODE_CONTRACT_ID,
    IDSW_POLICY,
    MATCHING_CONTRACT_ID,
    REFERENCE_PARITY_PASS,
    SEQUENCE_BOUNDARY_POLICY,
    build_metric_metadata,
    resolve_evaluator_code_sha,
)
from .cvat_io import TrackingObject
from .hota_standard_v2 import (
    HOTAStandardV2Result,
    combine_hota_sequences,
    evaluate_hota_sequence,
    hota_at_alpha,
)
from .identity_episodes_v2 import (
    IdentityEpisodeResult,
    MatchedIdentityRow,
    build_identity_episode_result,
)
from .identity_standard_v2 import (
    IdentityStandardV2Metrics,
    aggregate_identity_standard_v2,
    evaluate_identity_standard_v2,
)
from .matching_standard_v2 import (
    match_frame_eligible_with_ambiguity,
)

FrameObjects = Mapping[int, Sequence[TrackingObject]]


@dataclass(frozen=True, slots=True)
class TrackingMetricsStandardV2:
    """Flat, versioned metric row for one sequence or the dataset aggregate."""

    video_stem: str
    sequence_count: int
    evaluator_contract_id: str
    identity_episode_contract_id: str
    matching_contract_id: str
    hota_threshold_set: tuple[float, ...]
    include_hidden: bool
    sequence_boundary_policy: str
    idsw_policy: str
    identity_authority_policy: str
    reference_parity_status: str
    evaluator_code_sha: str
    metric_config_sha256: str
    frames_per_second: float | None
    gt_detections: int
    pred_detections: int
    tp: int
    fp: int
    fn: int
    detection_precision: float
    detection_recall: float
    hota: float
    deta: float
    assa: float
    loca: float
    hota_at_alpha_050_diagnostic: float
    hota_by_alpha: tuple[float, ...]
    deta_by_alpha: tuple[float, ...]
    assa_by_alpha: tuple[float, ...]
    loca_by_alpha: tuple[float, ...]
    hota_tp_by_alpha: tuple[int, ...]
    hota_fp_by_alpha: tuple[int, ...]
    hota_fn_by_alpha: tuple[int, ...]
    idf1: float
    id_precision: float
    id_recall: float
    idtp: int
    idfp: int
    idfn: int
    idsw_standard: int
    fragments: int
    identity_error_episode_count: int
    recovered_identity_error_episode_count: int
    terminal_identity_error_episode_count: int
    censored_identity_error_episode_count: int
    persistent_pairwise_identity_swap_count: int
    wrong_id_matched_frames: int
    wrong_id_matched_seconds: float | None
    median_identity_error_episode_seconds: float | None
    p95_identity_error_episode_seconds: float | None
    max_identity_error_episode_seconds: float | None
    median_recovery_latency_seconds: float | None
    p95_recovery_latency_seconds: float | None
    ambiguous_identity_rows: int
    authoritative_matched_gt_frames: int
    idsw_standard_per_1000_authoritative_matched_gt_frames: float
    wrong_id_matched_frames_per_1000_authoritative_matched_gt_frames: float
    gt_trajectories_with_identity_error_count: int
    authoritative_gt_trajectories: int
    gt_trajectories_with_identity_error_pct: float
    videos_with_terminal_episode_count: int
    videos_with_terminal_episode_pct: float
    videos_with_persistent_pairwise_swap_count: int
    videos_with_persistent_pairwise_swap_pct: float
    hidden_gt_rows: int
    visible_gt_rows: int
    hidden_prediction_rows: int
    evaluated_hidden_rows: int
    evaluated_visible_rows: int
    evaluated_frames: int
    gt_ids: int
    pred_ids: int


@dataclass(frozen=True, slots=True)
class StandardV2Evaluation:
    """Metric row plus auditable sufficient statistics and episode tables."""

    metrics: TrackingMetricsStandardV2
    hota_result: HOTAStandardV2Result
    identity_result: IdentityStandardV2Metrics
    episode_result: IdentityEpisodeResult
    matched_identity_rows: tuple[MatchedIdentityRow, ...]


def _validate_fps(frames_per_second: float | None) -> float | None:
    if frames_per_second is None:
        return None
    fps = float(frames_per_second)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("frames_per_second must be finite and positive")
    return fps


def _validate_and_filter(
    by_frame: FrameObjects,
    *,
    include_hidden: bool,
    role: str,
) -> tuple[dict[int, list[TrackingObject]], int, int]:
    filtered: dict[int, list[TrackingObject]] = {}
    hidden_rows = 0
    visible_rows = 0
    for frame in sorted(by_frame):
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError(f"{role} frame keys must be integers")
        seen_ids: set[str] = set()
        kept: list[TrackingObject] = []
        for obj in by_frame[frame]:
            if obj.frame != frame:
                raise ValueError(f"{role} object frame differs from frame key")
            if obj.obj_id in seen_ids:
                raise ValueError(
                    f"Duplicate {role} identity {obj.obj_id!r} at frame {frame}"
                )
            seen_ids.add(obj.obj_id)
            bbox = tuple(float(value) for value in obj.bbox)
            if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
                raise ValueError(f"{role} bbox must contain four finite values")
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"{role} bbox must have positive area")
            if obj.hidden:
                hidden_rows += 1
            else:
                visible_rows += 1
            if include_hidden or not obj.hidden:
                kept.append(obj)
        if kept:
            filtered[frame] = kept
    return filtered, hidden_rows, visible_rows


def _detection_matches(
    gt_by_frame: FrameObjects,
    pred_by_frame: FrameObjects,
    *,
    sequence_key: str,
    iou_threshold: float,
) -> tuple[int, int, int, tuple[MatchedIdentityRow, ...]]:
    tp = 0
    fp = 0
    fn = 0
    rows: list[MatchedIdentityRow] = []
    for frame in sorted(set(gt_by_frame).union(pred_by_frame)):
        gt_objects = gt_by_frame.get(frame, ())
        pred_objects = pred_by_frame.get(frame, ())
        matches, ambiguous_matches = match_frame_eligible_with_ambiguity(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold,
        )
        tp += len(matches)
        fn += len(gt_objects) - len(matches)
        fp += len(pred_objects) - len(matches)
        for gt_index, pred_index, _iou in matches:
            rows.append(
                MatchedIdentityRow(
                    sequence_key=sequence_key,
                    frame=frame,
                    gt_id=gt_objects[gt_index].obj_id,
                    pred_id=pred_objects[pred_index].obj_id,
                    authority_ambiguous=(
                        (gt_index, pred_index) in ambiguous_matches
                    ),
                )
            )
    return tp, fp, fn, tuple(rows)


def _nearest_rank_p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _episode_seconds(
    episodes: IdentityEpisodeResult,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    durations = [
        episode.duration_seconds
        for episode in episodes.episodes
        if episode.duration_seconds is not None
    ]
    recoveries = [
        episode.recovery_latency_seconds
        for episode in episodes.episodes
        if episode.recovery_latency_seconds is not None
    ]
    if len(durations) != len(episodes.episodes):
        duration_values: list[float] = []
    else:
        duration_values = [float(value) for value in durations]
    if not duration_values:
        median_duration = None
        p95_duration = None
        max_duration = None
    else:
        median_duration = float(median(duration_values))
        p95_duration = _nearest_rank_p95(duration_values)
        max_duration = max(duration_values)
    recovery_values = [float(value) for value in recoveries]
    median_recovery = (
        float(median(recovery_values)) if recovery_values else None
    )
    return (
        median_duration,
        p95_duration,
        max_duration,
        median_recovery,
        _nearest_rank_p95(recovery_values),
    )


def _safe_rate(numerator: int, denominator: int, scale: float = 1.0) -> float:
    return float(scale * numerator / denominator) if denominator else 0.0


def _build_metric_row(
    *,
    video_stem: str,
    sequence_count: int,
    include_hidden: bool,
    evaluator_code_sha: str,
    frames_per_second: float | None,
    hota_result: HOTAStandardV2Result,
    identity_result: IdentityStandardV2Metrics,
    episode_result: IdentityEpisodeResult,
    tp: int,
    fp: int,
    fn: int,
    hidden_gt_rows: int,
    visible_gt_rows: int,
    hidden_prediction_rows: int,
    evaluated_hidden_rows: int,
    evaluated_visible_rows: int,
    evaluated_frames: int,
    gt_ids: int,
    pred_ids: int,
) -> TrackingMetricsStandardV2:
    metadata = build_metric_metadata(
        include_hidden=include_hidden,
        evaluator_code_sha=evaluator_code_sha,
        identity_authority_policy=episode_result.authority_policy,
    )
    duration_stats = _episode_seconds(episode_result)
    affected_gt = {
        (episode.sequence_key, episode.gt_id)
        for episode in episode_result.episodes
    }
    authority_gt = {
        (authority.sequence_key, authority.gt_id)
        for authority in episode_result.authorities
    }
    terminal_videos = {
        episode.sequence_key
        for episode in episode_result.episodes
        if episode.status == "terminal"
    }
    persistent_videos = {
        event.sequence_key
        for event in episode_result.pairwise_events
        if event.persistent
    }
    video_denominator = sequence_count
    return TrackingMetricsStandardV2(
        video_stem=video_stem,
        sequence_count=sequence_count,
        evaluator_contract_id=EVALUATOR_CONTRACT_ID,
        identity_episode_contract_id=IDENTITY_EPISODE_CONTRACT_ID,
        matching_contract_id=MATCHING_CONTRACT_ID,
        hota_threshold_set=HOTA_ALPHAS,
        include_hidden=include_hidden,
        sequence_boundary_policy=SEQUENCE_BOUNDARY_POLICY,
        idsw_policy=IDSW_POLICY,
        identity_authority_policy=episode_result.authority_policy,
        reference_parity_status=REFERENCE_PARITY_PASS,
        evaluator_code_sha=evaluator_code_sha,
        metric_config_sha256=str(metadata["metric_config_sha256"]),
        frames_per_second=frames_per_second,
        gt_detections=tp + fn,
        pred_detections=tp + fp,
        tp=tp,
        fp=fp,
        fn=fn,
        detection_precision=_safe_rate(tp, tp + fp),
        detection_recall=_safe_rate(tp, tp + fn),
        hota=hota_result.hota_mean,
        deta=hota_result.deta_mean,
        assa=hota_result.assa_mean,
        loca=hota_result.loca_mean,
        hota_at_alpha_050_diagnostic=hota_at_alpha(hota_result, 0.5)["hota"],
        hota_by_alpha=hota_result.hota,
        deta_by_alpha=hota_result.deta,
        assa_by_alpha=hota_result.assa,
        loca_by_alpha=hota_result.loca,
        hota_tp_by_alpha=hota_result.tp,
        hota_fp_by_alpha=hota_result.fp,
        hota_fn_by_alpha=hota_result.fn,
        idf1=identity_result.idf1,
        id_precision=identity_result.idp,
        id_recall=identity_result.idr,
        idtp=identity_result.idtp,
        idfp=identity_result.idfp,
        idfn=identity_result.idfn,
        idsw_standard=identity_result.idsw_standard,
        fragments=identity_result.fragments,
        identity_error_episode_count=(
            episode_result.identity_error_episode_count
        ),
        recovered_identity_error_episode_count=(
            episode_result.recovered_identity_error_episode_count
        ),
        terminal_identity_error_episode_count=(
            episode_result.terminal_identity_error_episode_count
        ),
        censored_identity_error_episode_count=(
            episode_result.censored_identity_error_episode_count
        ),
        persistent_pairwise_identity_swap_count=(
            episode_result.persistent_pairwise_identity_swap_count
        ),
        wrong_id_matched_frames=episode_result.wrong_id_matched_frames,
        wrong_id_matched_seconds=episode_result.wrong_id_matched_seconds,
        median_identity_error_episode_seconds=duration_stats[0],
        p95_identity_error_episode_seconds=duration_stats[1],
        max_identity_error_episode_seconds=duration_stats[2],
        median_recovery_latency_seconds=duration_stats[3],
        p95_recovery_latency_seconds=duration_stats[4],
        ambiguous_identity_rows=len(episode_result.ambiguous_rows),
        authoritative_matched_gt_frames=tp,
        idsw_standard_per_1000_authoritative_matched_gt_frames=_safe_rate(
            identity_result.idsw_standard,
            tp,
            1000.0,
        ),
        wrong_id_matched_frames_per_1000_authoritative_matched_gt_frames=(
            _safe_rate(episode_result.wrong_id_matched_frames, tp, 1000.0)
        ),
        gt_trajectories_with_identity_error_count=len(affected_gt),
        authoritative_gt_trajectories=len(authority_gt),
        gt_trajectories_with_identity_error_pct=_safe_rate(
            len(affected_gt),
            len(authority_gt),
            100.0,
        ),
        videos_with_terminal_episode_count=len(terminal_videos),
        videos_with_terminal_episode_pct=_safe_rate(
            len(terminal_videos),
            video_denominator,
            100.0,
        ),
        videos_with_persistent_pairwise_swap_count=len(persistent_videos),
        videos_with_persistent_pairwise_swap_pct=_safe_rate(
            len(persistent_videos),
            video_denominator,
            100.0,
        ),
        hidden_gt_rows=hidden_gt_rows,
        visible_gt_rows=visible_gt_rows,
        hidden_prediction_rows=hidden_prediction_rows,
        evaluated_hidden_rows=evaluated_hidden_rows,
        evaluated_visible_rows=evaluated_visible_rows,
        evaluated_frames=evaluated_frames,
        gt_ids=gt_ids,
        pred_ids=pred_ids,
    )


def evaluate_tracking_standard_v2(
    gt_by_frame: FrameObjects,
    pred_by_frame: FrameObjects,
    *,
    video_stem: str,
    include_hidden: bool = True,
    detection_iou_threshold: float = 0.5,
    frames_per_second: float | None = None,
    evaluator_code_sha: str | None = None,
    explicit_identity_authority: Mapping[tuple[str, str], str] | None = None,
) -> StandardV2Evaluation:
    """Evaluate one sequence without changing either input representation."""
    if not math.isclose(float(detection_iou_threshold), 0.5):
        raise ValueError("Standard V2 detection and identity threshold is fixed at 0.5")
    fps = _validate_fps(frames_per_second)
    gt, hidden_gt, visible_gt = _validate_and_filter(
        gt_by_frame,
        include_hidden=include_hidden,
        role="GT",
    )
    pred, hidden_pred, _visible_pred = _validate_and_filter(
        pred_by_frame,
        include_hidden=include_hidden,
        role="prediction",
    )
    hota_result = evaluate_hota_sequence(gt, pred, sequence_key=video_stem)
    identity_result = evaluate_identity_standard_v2(
        gt,
        pred,
        iou_threshold=detection_iou_threshold,
        sequence_id=video_stem,
    )
    tp, fp, fn, matched_rows = _detection_matches(
        gt,
        pred,
        sequence_key=video_stem,
        iou_threshold=detection_iou_threshold,
    )
    fps_map = {video_stem: fps} if fps is not None else None
    episode_result = build_identity_episode_result(
        matched_rows,
        explicit_authority=explicit_identity_authority,
        fps_by_sequence=fps_map,
    )
    evaluated_objects = [obj for objects in gt.values() for obj in objects]
    metrics = _build_metric_row(
        video_stem=video_stem,
        sequence_count=1,
        include_hidden=include_hidden,
        evaluator_code_sha=evaluator_code_sha or resolve_evaluator_code_sha(),
        frames_per_second=fps,
        hota_result=hota_result,
        identity_result=identity_result,
        episode_result=episode_result,
        tp=tp,
        fp=fp,
        fn=fn,
        hidden_gt_rows=hidden_gt,
        visible_gt_rows=visible_gt,
        hidden_prediction_rows=hidden_pred,
        evaluated_hidden_rows=sum(obj.hidden for obj in evaluated_objects),
        evaluated_visible_rows=sum(not obj.hidden for obj in evaluated_objects),
        evaluated_frames=len(set(gt).union(pred)),
        gt_ids=len({obj.obj_id for obj in evaluated_objects}),
        pred_ids=len(
            {obj.obj_id for objects in pred.values() for obj in objects}
        ),
    )
    return StandardV2Evaluation(
        metrics=metrics,
        hota_result=hota_result,
        identity_result=identity_result,
        episode_result=episode_result,
        matched_identity_rows=matched_rows,
    )


def aggregate_tracking_standard_v2(
    evaluations: Sequence[StandardV2Evaluation],
) -> StandardV2Evaluation:
    """Aggregate sequence sufficient statistics without crossing boundaries."""
    if not evaluations:
        raise ValueError("At least one sequence evaluation is required")
    include_hidden_values = {
        evaluation.metrics.include_hidden for evaluation in evaluations
    }
    code_shas = {evaluation.metrics.evaluator_code_sha for evaluation in evaluations}
    if len(include_hidden_values) != 1 or len(code_shas) != 1:
        raise ValueError("V2 aggregate requires identical evaluator metadata")
    include_hidden = next(iter(include_hidden_values))
    code_sha = next(iter(code_shas))
    hota_result = combine_hota_sequences(
        [evaluation.hota_result for evaluation in evaluations],
    )
    identity_result = aggregate_identity_standard_v2(
        [evaluation.identity_result for evaluation in evaluations],
    )
    matched_rows = tuple(
        row
        for evaluation in evaluations
        for row in evaluation.matched_identity_rows
    )
    fps_map = {
        evaluation.metrics.video_stem: evaluation.metrics.frames_per_second
        for evaluation in evaluations
        if evaluation.metrics.frames_per_second is not None
    }
    episode_result = build_identity_episode_result(
        matched_rows,
        fps_by_sequence=fps_map or None,
        explicit_authority=(
            {
                (authority.sequence_key, authority.gt_id): authority.pred_id
                for evaluation in evaluations
                for authority in evaluation.episode_result.authorities
            }
            if {
                evaluation.episode_result.authority_policy
                for evaluation in evaluations
            }
            == {"IDENTITY_AUTHORITY_EXPLICIT_V2"}
            else None
        ),
    )
    metrics_rows = [evaluation.metrics for evaluation in evaluations]
    tp = sum(row.tp for row in metrics_rows)
    fp = sum(row.fp for row in metrics_rows)
    fn = sum(row.fn for row in metrics_rows)
    metrics = _build_metric_row(
        video_stem="ALL",
        sequence_count=len(evaluations),
        include_hidden=include_hidden,
        evaluator_code_sha=code_sha,
        frames_per_second=None,
        hota_result=hota_result,
        identity_result=identity_result,
        episode_result=episode_result,
        tp=tp,
        fp=fp,
        fn=fn,
        hidden_gt_rows=sum(row.hidden_gt_rows for row in metrics_rows),
        visible_gt_rows=sum(row.visible_gt_rows for row in metrics_rows),
        hidden_prediction_rows=sum(
            row.hidden_prediction_rows for row in metrics_rows
        ),
        evaluated_hidden_rows=sum(
            row.evaluated_hidden_rows for row in metrics_rows
        ),
        evaluated_visible_rows=sum(
            row.evaluated_visible_rows for row in metrics_rows
        ),
        evaluated_frames=sum(row.evaluated_frames for row in metrics_rows),
        gt_ids=sum(row.gt_ids for row in metrics_rows),
        pred_ids=sum(row.pred_ids for row in metrics_rows),
    )
    return StandardV2Evaluation(
        metrics=metrics,
        hota_result=hota_result,
        identity_result=identity_result,
        episode_result=episode_result,
        matched_identity_rows=matched_rows,
    )


def metrics_to_dataframe_standard_v2(
    evaluations: Sequence[StandardV2Evaluation],
) -> pd.DataFrame:
    """Convert V2 metric bundles into a stable tabular representation."""
    rows: list[dict[str, Any]] = [
        asdict(evaluation.metrics) for evaluation in evaluations
    ]
    dataframe = pd.DataFrame(rows)
    ratio_columns = (
        "detection_precision",
        "detection_recall",
        "hota",
        "deta",
        "assa",
        "loca",
        "idf1",
        "id_precision",
        "id_recall",
    )
    for column in ratio_columns:
        if column in dataframe:
            dataframe[f"{column}_pct"] = (dataframe[column] * 100).round(6)
    return dataframe

"""High-level tracking evaluation logic and report generation."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .assets import (
    DETECTOR_WEIGHTS,
    EVAL_OUTPUT_ROOT,
    PREDICTION_ROOT,
    PROJECT_ROOT,
    TRACKING_GT_DIR,
    VIDEO_DIR,
    TrackingPair,
    list_tracking_pairs,
    resolve_mask_path,
    video_metadata,
)
from .cvat_io import TrackingObject, parse_cvat_video_xml, read_cvat_task_size
from .diagnostics import (
    continuity_gaps_for_pair,
    continuity_gaps_to_dataframe,
    identity_events_for_pair,
    identity_events_to_dataframe,
    identity_mapping_for_pair,
    identity_mapping_to_dataframe,
)
from .matching import match_frame
from .metrics import (
    TrackingMetrics,
    aggregate_metrics,
    attach_remapped_metrics,
    compute_association_accuracy,
    compute_id_metrics,
    continuity_stats_from_matches,
    remap_prediction_ids,
)


def evaluate_tracking(
    gt_by_frame: dict[int, list[TrackingObject]],
    pred_by_frame: dict[int, list[TrackingObject]],
    *,
    iou_threshold: float = 0.5,
    video_stem: str = "",
    gap_tolerance_frames: int = 15,
) -> TrackingMetrics:
    """Compute CLEAR MOT, IDF1, and HOTA-style summary metrics."""
    frames = sorted(set(gt_by_frame).union(pred_by_frame))
    total_gt = sum(len(items) for items in gt_by_frame.values())
    total_pred = sum(len(items) for items in pred_by_frame.values())

    matches_count = 0
    fp = 0
    fn = 0
    idsw = 0
    fragments = 0
    iou_sum = 0.0

    last_match_for_gt: dict[str, str] = {}
    last_tracked_state: dict[str, bool] = defaultdict(bool)
    active_tracklet_lengths: dict[str, int] = defaultdict(int)
    tracklet_lengths: list[int] = []
    ever_tracked: set[str] = set()
    pair_counts: Counter[tuple[str, str]] = Counter()
    gt_id_counts: Counter[str] = Counter()
    pred_id_counts: Counter[str] = Counter()
    matched_frames_by_gt: dict[str, list[int]] = defaultdict(list)

    for frame in frames:
        gt_objects = gt_by_frame.get(frame, [])
        pred_objects = pred_by_frame.get(frame, [])
        current_gt_ids = {obj.obj_id for obj in gt_objects}
        for gt_id, is_tracked in list(last_tracked_state.items()):
            if is_tracked and gt_id not in current_gt_ids:
                tracklet_lengths.append(active_tracklet_lengths[gt_id])
                active_tracklet_lengths[gt_id] = 0
                last_tracked_state[gt_id] = False

        gt_id_counts.update(obj.obj_id for obj in gt_objects)
        pred_id_counts.update(obj.obj_id for obj in pred_objects)

        matches = match_frame(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold,
        )
        matched_gt = {gt_idx for gt_idx, _pred_idx, _iou in matches}
        matched_pred = {pred_idx for _gt_idx, pred_idx, _iou in matches}

        matches_count += len(matches)
        fp += len(pred_objects) - len(matched_pred)
        fn += len(gt_objects) - len(matched_gt)
        iou_sum += sum(iou for _gt_idx, _pred_idx, iou in matches)

        matched_gt_ids_this_frame: set[str] = set()
        for gt_idx, pred_idx, _iou in matches:
            gt_id = gt_objects[gt_idx].obj_id
            pred_id = pred_objects[pred_idx].obj_id
            pair_counts[(gt_id, pred_id)] += 1
            matched_gt_ids_this_frame.add(gt_id)
            matched_frames_by_gt[gt_id].append(frame)

            previous_pred_id = last_match_for_gt.get(gt_id)
            if previous_pred_id is not None and previous_pred_id != pred_id:
                idsw += 1
            last_match_for_gt[gt_id] = pred_id

            if gt_id in ever_tracked and not last_tracked_state[gt_id]:
                fragments += 1
            ever_tracked.add(gt_id)
            last_tracked_state[gt_id] = True

        for gt in gt_objects:
            if gt.obj_id not in matched_gt_ids_this_frame:
                if last_tracked_state[gt.obj_id]:
                    tracklet_lengths.append(active_tracklet_lengths[gt.obj_id])
                    active_tracklet_lengths[gt.obj_id] = 0
                last_tracked_state[gt.obj_id] = False

        for gt_id in matched_gt_ids_this_frame:
            active_tracklet_lengths[gt_id] += 1

    for gt_id, is_tracked in last_tracked_state.items():
        if is_tracked:
            tracklet_lengths.append(active_tracklet_lengths[gt_id])

    mota = 1.0 - ((fn + fp + idsw) / total_gt) if total_gt else 0.0
    motp = iou_sum / matches_count if matches_count else 0.0
    precision = matches_count / total_pred if total_pred else 0.0
    recall = matches_count / total_gt if total_gt else 0.0

    idtp, idfp, idfn, idf1 = compute_id_metrics(
        pair_counts,
        gt_id_counts,
        pred_id_counts,
    )
    deta = matches_count / (matches_count + fp + fn) if matches_count + fp + fn else 0
    assa = compute_association_accuracy(pair_counts, gt_id_counts, pred_id_counts)
    hota = math.sqrt(deta * assa) if deta > 0 and assa > 0 else 0.0
    tracklets = len(tracklet_lengths)
    avg_tracklet_length = (
        float(sum(tracklet_lengths) / tracklets) if tracklets else 0.0
    )
    (
        gap_tolerant_fragments,
        gap_tolerant_tracklets,
        gap_tolerant_avg_tracklet_length,
        gap_tolerant_suppressed_fragments,
    ) = continuity_stats_from_matches(
        matched_frames_by_gt,
        gap_tolerance_frames=gap_tolerance_frames,
    )

    return TrackingMetrics(
        video_stem=video_stem,
        gt_detections=total_gt,
        pred_detections=total_pred,
        matches=matches_count,
        fp=fp,
        fn=fn,
        idsw=idsw,
        fragments=fragments,
        tracklets=tracklets,
        avg_tracklet_length_frames=avg_tracklet_length,
        gap_tolerance_frames=max(0, int(gap_tolerance_frames)),
        gap_tolerant_fragments=gap_tolerant_fragments,
        gap_tolerant_tracklets=gap_tolerant_tracklets,
        gap_tolerant_avg_tracklet_length_frames=gap_tolerant_avg_tracklet_length,
        gap_tolerant_suppressed_fragments=gap_tolerant_suppressed_fragments,
        mota=mota,
        motp_iou=motp,
        precision=precision,
        recall=recall,
        idf1=idf1,
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        deta=deta,
        assa=assa,
        hota=hota,
        evaluated_frames=len(frames),
        gt_ids=len(gt_id_counts),
        pred_ids=len(pred_id_counts),
    )


def evaluate_pair(
    pair: TrackingPair,
    *,
    iou_threshold: float = 0.5,
    include_hidden: bool = False,
    gap_tolerance_frames: int = 15,
) -> TrackingMetrics | None:
    """Evaluate one pair when a prediction XML is available."""
    if pair.pred_xml is None or not pair.pred_xml.exists():
        return None

    gt = parse_cvat_video_xml(pair.gt_xml, include_hidden=include_hidden)
    pred = parse_cvat_video_xml(pair.pred_xml, include_hidden=include_hidden)
    metrics = evaluate_tracking(
        gt,
        pred,
        iou_threshold=iou_threshold,
        video_stem=pair.video_stem,
        gap_tolerance_frames=gap_tolerance_frames,
    )
    remapped_pred, _mapping, mapped_matches, coverage = remap_prediction_ids(
        gt,
        pred,
        iou_threshold=iou_threshold,
    )
    remapped_metrics = evaluate_tracking(
        gt,
        remapped_pred,
        iou_threshold=iou_threshold,
        video_stem=pair.video_stem,
        gap_tolerance_frames=gap_tolerance_frames,
    )
    attach_remapped_metrics(
        metrics,
        remapped_metrics,
        mapped_matches=mapped_matches,
        coverage=coverage,
    )
    metrics.gt_xml = str(pair.gt_xml)
    metrics.pred_xml = str(pair.pred_xml)
    metrics.video_path = str(pair.video_path)
    return metrics


def metrics_to_dataframe(metrics: list[TrackingMetrics]) -> pd.DataFrame:
    """Convert metrics to a dataframe with percent columns."""
    rows = [asdict(metric) for metric in metrics]
    df = pd.DataFrame(rows)
    percent_cols = [
        "mota",
        "motp_iou",
        "precision",
        "recall",
        "idf1",
        "deta",
        "assa",
        "hota",
        "remapped_mota",
        "remapped_idf1",
        "remapped_assa",
        "remapped_hota",
        "idmap_coverage",
    ]
    for col in percent_cols:
        if col in df.columns:
            df[f"{col}_pct"] = (df[col] * 100).round(2)
    return df


def pairs_to_dataframe(pairs: list[TrackingPair]) -> pd.DataFrame:
    """Convert matched assets to a dataframe."""
    rows = []
    for pair in pairs:
        gt_size = read_cvat_task_size(pair.gt_xml)
        metadata = {
            "video_frame_count": None,
            "video_fps": None,
            "video_width": None,
            "video_height": None,
            **video_metadata(pair.video_path),
        }
        rows.append(
            {
                "video_stem": pair.video_stem,
                "video_path": str(pair.video_path),
                "gt_xml": str(pair.gt_xml),
                "pred_xml": str(pair.pred_xml) if pair.pred_xml else "",
                "has_prediction": pair.pred_xml is not None and pair.pred_xml.exists(),
                "gt_task_size": gt_size,
                **metadata,
            }
        )
    return pd.DataFrame(rows)


def evaluate_dataset(
    *,
    iou_threshold: float = 0.5,
    include_hidden: bool = False,
    gap_tolerance_frames: int = 15,
    tracking_gt_dir: Path = TRACKING_GT_DIR,
    video_dir: Path = VIDEO_DIR,
    prediction_root: Path = PREDICTION_ROOT,
    output_root: Path = EVAL_OUTPUT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Evaluate all matched GT/prediction pairs and save reports."""
    pairs = list_tracking_pairs(
        tracking_gt_dir=tracking_gt_dir,
        video_dir=video_dir,
        prediction_root=prediction_root,
    )
    asset_df = pairs_to_dataframe(pairs)

    metrics = []
    identity_events = []
    remapped_identity_events = []
    identity_mapping_rows = []
    continuity_gap_rows = []
    for pair in pairs:
        result = evaluate_pair(
            pair,
            iou_threshold=iou_threshold,
            include_hidden=include_hidden,
            gap_tolerance_frames=gap_tolerance_frames,
        )
        if result is not None:
            metrics.append(result)
            identity_events.extend(
                identity_events_for_pair(
                    pair,
                    iou_threshold=iou_threshold,
                    include_hidden=include_hidden,
                )
            )
            remapped_identity_events.extend(
                identity_events_for_pair(
                    pair,
                    iou_threshold=iou_threshold,
                    include_hidden=include_hidden,
                    remap_ids=True,
                )
            )
            identity_mapping_rows.extend(
                identity_mapping_for_pair(
                    pair,
                    iou_threshold=iou_threshold,
                    include_hidden=include_hidden,
                )
            )
            continuity_gap_rows.extend(
                continuity_gaps_for_pair(
                    pair,
                    iou_threshold=iou_threshold,
                    include_hidden=include_hidden,
                    gap_tolerance_frames=gap_tolerance_frames,
                    remap_ids=True,
                )
            )

    all_rows = metrics + ([aggregate_metrics(metrics)] if metrics else [])
    metrics_df = metrics_to_dataframe(all_rows)

    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    asset_df.to_csv(run_dir / "tracking_eval_assets.csv", index=False)
    metrics_df.to_csv(run_dir / "tracking_metrics.csv", index=False)
    identity_events_to_dataframe(identity_events).to_csv(
        run_dir / "tracking_identity_events.csv",
        index=False,
    )
    identity_events_to_dataframe(remapped_identity_events).to_csv(
        run_dir / "tracking_remapped_identity_events.csv",
        index=False,
    )
    identity_mapping_to_dataframe(identity_mapping_rows).to_csv(
        run_dir / "tracking_id_mapping.csv",
        index=False,
    )
    continuity_gaps_to_dataframe(continuity_gap_rows).to_csv(
        run_dir / "tracking_continuity_gaps.csv",
        index=False,
    )
    with (run_dir / "tracking_eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "iou_threshold": iou_threshold,
                "include_hidden": include_hidden,
                "gap_tolerance_frames": gap_tolerance_frames,
                "tracking_gt_dir": str(tracking_gt_dir),
                "video_dir": str(video_dir),
                "prediction_root": str(prediction_root),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return asset_df, metrics_df, run_dir


def run_tracker_for_pair(
    pair: TrackingPair,
    *,
    weights_path: Path = DETECTOR_WEIGHTS,
    mask_path: Path | None = None,
    output_root: Path = (
        PROJECT_ROOT / "outputs" / "evaluation" / "tracking_predictions"
    ),
    max_frames: int | None = None,
    tracking_overrides: dict[str, Any] | None = None,
) -> Path:
    """Run the project tracker for one pair and return generated prediction XML."""
    from pig_behavior.data_preparation.tracking_annotation import (
        TrackingConfig,
        run_tracking,
    )

    mask_path = mask_path or resolve_mask_path()
    output_dir = output_root / pair.video_stem
    cfg = TrackingConfig(
        video_path=pair.video_path,
        weights_path=weights_path,
        mask_path=mask_path,
        output_dir=output_dir,
        max_frames=max_frames,
        display_inline=False,
        show=False,
        **(tracking_overrides or {}),
    )
    summary = run_tracking(cfg)
    return Path(summary.cvat_video_xml)

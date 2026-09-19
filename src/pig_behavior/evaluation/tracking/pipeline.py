"""Pipeline orchestration for tracking prediction and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .assets import (
    TrackingPair,
    find_prediction_xml,
    list_tracking_pairs,
    normalize_key,
)
from .config import TrackingEvaluationPipelineConfig
from .cvat_io import read_task_name
from .evaluator import (
    pairs_to_dataframe,
    run_tracker_for_pair,
)
from .reporting import build_markdown_report


def find_gt_xml_for_video(video_path: Path, gt_dir: Path) -> Path | None:
    """Find a GT CVAT XML that contains the video stem in its file/task name."""
    video_key = normalize_key(video_path.stem)
    for xml_path in sorted(gt_dir.glob("*.xml")):
        if video_key in normalize_key(xml_path.stem):
            return xml_path
        if video_key in normalize_key(read_task_name(xml_path)):
            return xml_path
    return None


def build_pairs(config: TrackingEvaluationPipelineConfig) -> list[TrackingPair]:
    """Build evaluation pairs from explicit paths or dataset directories."""
    if config.video_paths:
        pairs = []
        for video_path in config.video_paths:
            resolved_video = video_path.resolve()
            gt_xml = find_gt_xml_for_video(resolved_video, config.gt_dir)
            if gt_xml is None:
                print(
                    f"Warning: Skipping video '{resolved_video.name}' because "
                    "no matching GT XML was found."
                )
                continue
            pred_xml = find_prediction_xml(
                resolved_video.stem,
                config.prediction_root,
                preferred_mode=config.tracking_mode,
            )
            pairs.append(
                TrackingPair(
                    video_stem=resolved_video.stem,
                    video_path=resolved_video,
                    gt_xml=gt_xml.resolve(),
                    pred_xml=pred_xml,
                )
            )
        return pairs

    if config.video_path is None and config.gt_xml is None:
        return list_tracking_pairs(
            tracking_gt_dir=config.gt_dir,
            video_dir=config.video_dir,
            prediction_root=config.prediction_root,
            preferred_mode=config.tracking_mode,
        )

    if config.video_path is None:
        raise ValueError("--video is required when --gt-xml is provided.")

    video_path = config.video_path.resolve()
    gt_xml = config.gt_xml or find_gt_xml_for_video(video_path, config.gt_dir)
    if gt_xml is None:
        raise FileNotFoundError(f"No GT XML found for video: {video_path}")

    pred_xml = find_prediction_xml(
        video_path.stem,
        config.prediction_root,
        preferred_mode=config.tracking_mode,
    )
    return [
        TrackingPair(
            video_stem=video_path.stem,
            video_path=video_path,
            gt_xml=gt_xml.resolve(),
            pred_xml=pred_xml,
        )
    ]


def ensure_predictions(
    pairs: list[TrackingPair],
    config: TrackingEvaluationPipelineConfig,
) -> list[TrackingPair]:
    """Run the tracker where requested and return pairs with prediction paths."""
    resolved_pairs = []
    for pair in pairs:
        should_track = config.force_track or (
            config.run_missing_tracker and pair.pred_xml is None
        )
        if should_track:
            pred_xml = run_tracker_for_pair(
                pair,
                weights_path=config.weights_path,
                mask_path=config.mask_path,
                output_root=config.prediction_root,
                max_frames=config.max_frames,
                tracking_overrides=tracking_rule_overrides(config),
            )
            pair = TrackingPair(
                video_stem=pair.video_stem,
                video_path=pair.video_path,
                gt_xml=pair.gt_xml,
                pred_xml=pred_xml,
            )
        resolved_pairs.append(pair)
    return resolved_pairs


def tracking_rule_overrides(
    config: TrackingEvaluationPipelineConfig,
) -> dict[str, object]:
    """Return rule flags passed through to the tracking engine."""
    overrides = {
        "USE_IOU_FALLBACK": config.USE_IOU_FALLBACK,
        "USE_AREA_OCCLUSION_FREEZE": config.USE_AREA_OCCLUSION_FREEZE,
        "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": (
            config.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE
        ),
        "USE_MERGED_BOX_SPLIT": config.USE_MERGED_BOX_SPLIT,
        "device": config.device,
        "half": config.half,
        "mode": config.tracking_mode,
    }
    if config.profile_overrides:
        overrides.update(config.profile_overrides)
    overrides["overrides"] = set(overrides)
    return overrides


def runtime_telemetry_to_dataframe(
    pairs: list[TrackingPair],
) -> pd.DataFrame:
    """Collect per-video tracker telemetry beside each prediction XML."""
    rows: list[dict[str, object]] = []
    for pair in pairs:
        telemetry_path = (
            pair.pred_xml.with_name("tracking_quality_report.json")
            if pair.pred_xml is not None
            else None
        )
        row: dict[str, object] = {
            "video_stem": pair.video_stem,
            "telemetry_available": False,
            "telemetry_path": str(telemetry_path) if telemetry_path else "",
        }
        if telemetry_path is not None and telemetry_path.is_file():
            payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
            telemetry = payload.get("telemetry")
            if isinstance(telemetry, dict):
                row.update(telemetry)
                row["telemetry_available"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def save_pipeline_report(
    pairs: list[TrackingPair],
    metrics_df: pd.DataFrame,
    config: TrackingEvaluationPipelineConfig,
    identity_events_df: pd.DataFrame | None = None,
    remapped_identity_events_df: pd.DataFrame | None = None,
    identity_mapping_df: pd.DataFrame | None = None,
    continuity_gaps_df: pd.DataFrame | None = None,
    runtime_telemetry_df: pd.DataFrame | None = None,
) -> Path:
    """Save assets, metrics, and config for one evaluation run."""
    run_dir = config.output_root
    run_dir.mkdir(parents=True, exist_ok=True)

    asset_df = pairs_to_dataframe(pairs)
    asset_df.to_csv(run_dir / "tracking_eval_assets.csv", index=False)
    metrics_df.to_csv(run_dir / "tracking_metrics.csv", index=False)
    if identity_events_df is not None:
        identity_events_df.to_csv(run_dir / "tracking_identity_events.csv", index=False)
    if remapped_identity_events_df is not None:
        remapped_identity_events_df.to_csv(
            run_dir / "tracking_remapped_identity_events.csv",
            index=False,
        )
    if identity_mapping_df is not None:
        identity_mapping_df.to_csv(run_dir / "tracking_id_mapping.csv", index=False)
    if continuity_gaps_df is not None:
        continuity_gaps_df.to_csv(
            run_dir / "tracking_continuity_gaps.csv",
            index=False,
        )
    if runtime_telemetry_df is not None:
        runtime_telemetry_df.to_csv(
            run_dir / "tracking_runtime_telemetry.csv",
            index=False,
        )
    (run_dir / "tracking_report.md").write_text(
        build_markdown_report(
            asset_df,
            metrics_df,
            config,
            identity_mapping_df=identity_mapping_df,
            remapped_identity_events_df=remapped_identity_events_df,
            continuity_gaps_df=continuity_gaps_df,
        ),
        encoding="utf-8",
    )
    payload = {
        **asdict(config),
        "video_path": str(config.video_path) if config.video_path else None,
        "video_paths": (
            [str(path) for path in config.video_paths]
            if config.video_paths
            else None
        ),
        "gt_xml": str(config.gt_xml) if config.gt_xml else None,
        "gt_dir": str(config.gt_dir),
        "video_dir": str(config.video_dir),
        "prediction_root": str(config.prediction_root),
        "output_root": str(config.output_root),
        "weights_path": str(config.weights_path),
        "mask_path": str(config.mask_path) if config.mask_path else None,
    }
    with (run_dir / "tracking_eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return run_dir


def run_pipeline(
    config: TrackingEvaluationPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run the only active report generator, Tracking Evaluator Standard V2."""
    from .pipeline_standard_v2 import run_pipeline_standard_v2

    return run_pipeline_standard_v2(config)

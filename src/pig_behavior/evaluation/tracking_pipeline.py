"""Run tracking predictions and evaluate them against CVAT video XML labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from pig_behavior.evaluation.tracking_metrics import (
    DETECTOR_WEIGHTS,
    EVAL_OUTPUT_ROOT,
    PREDICTION_ROOT,
    TRACKING_GT_DIR,
    VIDEO_DIR,
    TrackingPair,
    aggregate_metrics,
    continuity_gaps_for_pair,
    continuity_gaps_to_dataframe,
    evaluate_pair,
    find_prediction_xml,
    identity_events_for_pair,
    identity_events_to_dataframe,
    identity_mapping_for_pair,
    identity_mapping_to_dataframe,
    list_tracking_pairs,
    metrics_to_dataframe,
    normalize_key,
    pairs_to_dataframe,
    read_task_name,
    run_tracker_for_pair,
)
from pig_behavior.tracking_path_config import (
    DEFAULT_TRACKING_PATH_CONFIG,
    load_tracking_path_profile,
    profile_path,
    profile_video_path,
    profile_video_paths,
)


@dataclass(slots=True)
class TrackingEvaluationPipelineConfig:
    """Configuration for the tracking -> evaluation pipeline."""

    video_path: Path | None = None
    video_paths: list[Path] | None = None
    gt_xml: Path | None = None
    gt_dir: Path = TRACKING_GT_DIR
    video_dir: Path = VIDEO_DIR
    prediction_root: Path = PREDICTION_ROOT
    output_root: Path = EVAL_OUTPUT_ROOT
    weights_path: Path = DETECTOR_WEIGHTS
    mask_path: Path | None = None
    iou_threshold: float = 0.5
    include_hidden: bool = False
    gap_tolerance_frames: int = 15
    # Bật tắt việc chạy tracker cho các video chưa có prediction XML
    run_missing_tracker: bool = True
    force_track: bool = False
    max_frames: int | None = None


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
                raise FileNotFoundError(f"No GT XML found for video: {resolved_video}")
            pred_xml = find_prediction_xml(resolved_video.stem, config.prediction_root)
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
        )

    if config.video_path is None:
        raise ValueError("--video is required when --gt-xml is provided.")

    video_path = config.video_path.resolve()
    gt_xml = config.gt_xml or find_gt_xml_for_video(video_path, config.gt_dir)
    if gt_xml is None:
        raise FileNotFoundError(f"No GT XML found for video: {video_path}")

    pred_xml = find_prediction_xml(video_path.stem, config.prediction_root)
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
            )
            pair = TrackingPair(
                video_stem=pair.video_stem,
                video_path=pair.video_path,
                gt_xml=pair.gt_xml,
                pred_xml=pred_xml,
            )
        resolved_pairs.append(pair)
    return resolved_pairs


def _format_metric_value(value: object) -> str:
    """Format dataframe values for concise Markdown tables."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) >= 1 else f"{value:.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected dataframe columns as a GitHub Markdown table."""
    if df.empty:
        return "_No rows._"

    table = df.loc[:, columns].copy()
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        rows.append(
            "| "
            + " | ".join(_format_metric_value(row[column]) for column in columns)
            + " |"
        )
    return "\n".join(rows)


def build_markdown_report(
    asset_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config: TrackingEvaluationPipelineConfig,
    identity_mapping_df: pd.DataFrame | None = None,
    remapped_identity_events_df: pd.DataFrame | None = None,
    continuity_gaps_df: pd.DataFrame | None = None,
) -> str:
    """Build a readable Markdown summary for one evaluation run."""
    evaluated_df = metrics_df[metrics_df["video_stem"] != "ALL"].copy()
    aggregate_df = metrics_df[metrics_df["video_stem"] == "ALL"].copy()
    missing_df = asset_df[~asset_df["has_prediction"]].copy()

    paper_metric_columns = [
        "video_stem",
        "remapped_mota_pct",
        "remapped_idf1_pct",
        "remapped_hota_pct",
        "precision_pct",
        "recall_pct",
        "remapped_idsw",
        "remapped_fragments",
        "remapped_tracklets",
        "remapped_avg_tracklet_length_frames",
        "remapped_gap_tolerant_fragments",
        "remapped_gap_tolerant_tracklets",
        "remapped_gap_tolerant_avg_tracklet_length_frames",
        "remapped_gap_tolerant_suppressed_fragments",
        "idmap_coverage_pct",
        "fp",
        "fn",
    ]
    raw_metric_columns = [
        "video_stem",
        "mota_pct",
        "idf1_pct",
        "hota_pct",
        "precision_pct",
        "recall_pct",
        "idsw",
        "fragments",
        "tracklets",
        "avg_tracklet_length_frames",
        "gap_tolerant_fragments",
        "gap_tolerant_tracklets",
        "gap_tolerant_avg_tracklet_length_frames",
        "gap_tolerant_suppressed_fragments",
        "fp",
        "fn",
    ]
    asset_columns = [
        "video_stem",
        "has_prediction",
        "gt_task_size",
        "video_frame_count",
        "video_fps",
        "video_width",
        "video_height",
    ]

    lines = [
        "# Tracking Evaluation Report",
        "",
        "## Run Config",
        "",
        f"- IoU threshold: `{config.iou_threshold}`",
        f"- Include hidden boxes: `{config.include_hidden}`",
        f"- Gap tolerance frames: `{config.gap_tolerance_frames}`",
        f"- Run missing tracker: `{config.run_missing_tracker}`",
        f"- Force track: `{config.force_track}`",
        f"- Ground-truth directory: `{config.gt_dir}`",
        f"- Prediction root: `{config.prediction_root}`",
        "",
        "## Summary",
        "",
        f"- Ground-truth videos found: `{len(asset_df)}`",
        f"- Videos evaluated: `{len(evaluated_df)}`",
        f"- Videos missing predictions: `{len(missing_df)}`",
        "",
    ]

    if not aggregate_df.empty:
        lines.extend(
            [
                "## Aggregate Metrics For Paper",
                "",
                _markdown_table(aggregate_df, paper_metric_columns),
                "",
            ]
        )

    lines.extend(
        [
            "## Per-Video Metrics For Paper",
            "",
            _markdown_table(evaluated_df, paper_metric_columns),
            "",
            "## Raw Absolute-ID Metrics For Audit",
            "",
            (
                "These strict metrics compare literal ID names before global "
                "remapping. Use them to audit CVAT/tracker naming issues, not as "
                "the main paper conclusion when initial ID numbering is arbitrary."
            ),
            "",
            _markdown_table(metrics_df, raw_metric_columns),
            "",
            "## Asset Coverage",
            "",
            _markdown_table(asset_df, asset_columns),
            "",
        ]
    )

    if not missing_df.empty:
        lines.extend(
            [
                "## Missing Predictions",
                "",
                (
                    "These videos had ground-truth XML but no prediction XML, "
                    "so they were not scored."
                ),
                "",
                _markdown_table(missing_df, ["video_stem", "video_path", "gt_xml"]),
                "",
            ]
        )

    if identity_mapping_df is not None:
        lines.extend(
            [
                "## Fixed ID Mapping",
                "",
                (
                    "`tracking_id_mapping.csv` stores the fixed prediction ID_N -> "
                    "GT ID_N mapping selected from whole-video matched overlap."
                ),
                "",
                _markdown_table(
                    identity_mapping_df.head(20),
                    [
                        "video_stem",
                        "pred_id",
                        "mapped_gt_id",
                        "matched_frames",
                        "total_matched_frames",
                        "mapping_coverage",
                    ],
                ),
                "",
            ]
        )

    if remapped_identity_events_df is not None:
        switch_count = int(
            (
                remapped_identity_events_df["event"]
                .str.contains("switch", na=False)
                .sum()
            )
            if "event" in remapped_identity_events_df.columns
            else 0
        )
        lines.extend(
            [
                "## Remapped Identity Diagnostics",
                "",
                (
                    "- `tracking_remapped_identity_events.csv`: identity events "
                    "after fixed ID remapping; these are continuity errors that "
                    "remain after removing arbitrary initial ID numbering."
                ),
                f"- Remapped identity event rows: `{len(remapped_identity_events_df)}`",
                f"- Remapped ID switch rows: `{switch_count}`",
                "",
            ]
        )

    if continuity_gaps_df is not None:
        tolerated_count = int(
            continuity_gaps_df["tolerated"].fillna(False).sum()
            if "tolerated" in continuity_gaps_df.columns
            else 0
        )
        remaining_count = len(continuity_gaps_df) - tolerated_count
        top_gaps_df = continuity_gaps_df.sort_values(
            ["gap_frames", "video_stem", "gt_id"],
            ascending=[False, True, True],
        ).head(20)
        lines.extend(
            [
                "## Continuity Gap Diagnostics",
                "",
                (
                    "- `tracking_continuity_gaps.csv`: matched-track gaps after "
                    "fixed ID remapping. Gaps shorter than or equal to the configured "
                    "tolerance are not counted as gap-tolerant fragments."
                ),
                f"- Total matched-track gaps: `{len(continuity_gaps_df)}`",
                f"- Tolerated gaps: `{tolerated_count}`",
                f"- Remaining fragment gaps: `{remaining_count}`",
                "",
                _markdown_table(
                    top_gaps_df,
                    [
                        "video_stem",
                        "gt_id",
                        "previous_matched_frame",
                        "next_matched_frame",
                        "gap_frames",
                        "tolerated",
                        "previous_pred_id",
                        "next_pred_id",
                        "id_changed",
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Metric Guide",
            "",
            (
                "- `MOTA`: overall tracking accuracy. Penalizes missed objects, "
                "false positives, and ID switches."
            ),
            (
                "- `IDF1`: identity consistency score. Higher means the same pig "
                "ID is preserved better."
            ),
            (
                "- `Remapped *`: metrics after one-to-one global ID mapping. "
                "Use these for paper reporting when initial tracker ID numbering "
                "is arbitrary."
            ),
            "- `HOTA`: combined detection and association score.",
            "- `MOTP IoU`: average box overlap quality for matched objects.",
            "- `FP`: predicted boxes that did not match ground truth.",
            "- `FN`: ground-truth boxes missed by prediction.",
            "- `IDSW`: ID switches.",
            "- `Fragments`: tracks that were interrupted and later recovered.",
            (
                "- `Tracklets`: continuous matched identity segments. "
                "`Avg. tracklet length` is their mean length in frames."
            ),
            (
                "- `Gap-tolerant *`: continuity metrics after merging matched "
                "segments separated by short gaps up to `gap_tolerance_frames`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def save_pipeline_report(
    pairs: list[TrackingPair],
    metrics_df: pd.DataFrame,
    config: TrackingEvaluationPipelineConfig,
    identity_events_df: pd.DataFrame | None = None,
    remapped_identity_events_df: pd.DataFrame | None = None,
    identity_mapping_df: pd.DataFrame | None = None,
    continuity_gaps_df: pd.DataFrame | None = None,
) -> Path:
    """Save assets, metrics, and config for one evaluation run."""
    run_dir = config.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
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
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return run_dir


def run_pipeline(
    config: TrackingEvaluationPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run prediction generation when needed, then evaluate available predictions."""
    pairs = build_pairs(config)
    pairs = ensure_predictions(pairs, config)
    asset_df = pairs_to_dataframe(pairs)

    metrics = []
    identity_events = []
    remapped_identity_events = []
    identity_mapping_rows = []
    continuity_gap_rows = []
    for pair in pairs:
        if pair.pred_xml is None:
            continue
        result = evaluate_pair(
            pair,
            iou_threshold=config.iou_threshold,
            include_hidden=config.include_hidden,
            gap_tolerance_frames=config.gap_tolerance_frames,
        )
        if result is not None:
            metrics.append(result)
            identity_events.extend(
                identity_events_for_pair(
                    pair,
                    iou_threshold=config.iou_threshold,
                    include_hidden=config.include_hidden,
                )
            )
            remapped_identity_events.extend(
                identity_events_for_pair(
                    pair,
                    iou_threshold=config.iou_threshold,
                    include_hidden=config.include_hidden,
                    remap_ids=True,
                )
            )
            identity_mapping_rows.extend(
                identity_mapping_for_pair(
                    pair,
                    iou_threshold=config.iou_threshold,
                    include_hidden=config.include_hidden,
                )
            )
            continuity_gap_rows.extend(
                continuity_gaps_for_pair(
                    pair,
                    iou_threshold=config.iou_threshold,
                    include_hidden=config.include_hidden,
                    gap_tolerance_frames=config.gap_tolerance_frames,
                    remap_ids=True,
                )
            )

    metric_rows = metrics + ([aggregate_metrics(metrics)] if metrics else [])
    metrics_df = metrics_to_dataframe(metric_rows)
    identity_events_df = identity_events_to_dataframe(identity_events)
    remapped_identity_events_df = identity_events_to_dataframe(
        remapped_identity_events
    )
    identity_mapping_df = identity_mapping_to_dataframe(identity_mapping_rows)
    continuity_gaps_df = continuity_gaps_to_dataframe(continuity_gap_rows)
    run_dir = save_pipeline_report(
        pairs,
        metrics_df,
        config,
        identity_events_df=identity_events_df,
        remapped_identity_events_df=remapped_identity_events_df,
        identity_mapping_df=identity_mapping_df,
        continuity_gaps_df=continuity_gaps_df,
    )
    return asset_df, metrics_df, run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Run pig tracking prediction and evaluate against CVAT XML labels.",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-key", type=str, default=None)
    parser.add_argument(
        "--all-config-videos",
        action="store_true",
        help="Evaluate every video listed in the selected tracking path profile.",
    )
    parser.add_argument(
        "--path-config",
        type=Path,
        default=DEFAULT_TRACKING_PATH_CONFIG,
        help="JSON path profile file for fast video/dir switching.",
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--gt-xml", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument(
        "--gap-tolerance-frames",
        type=int,
        default=15,
        help=(
            "Merge continuity gaps up to this many frames for gap-tolerant "
            "fragment/tracklet metrics. Use 0 for strict frame-by-frame scoring."
        ),
    )
    parser.add_argument(
        "--run-missing-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the package tracking annotator for videos without prediction XML. "
            "Use --no-run-missing-tracker to evaluate only existing predictions."
        ),
    )
    parser.add_argument(
        "--force-track",
        action="store_true",
        help="Always rerun tracker before evaluation, even if prediction XML exists.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrackingEvaluationPipelineConfig:
    """Build config from parsed args."""
    profile = load_tracking_path_profile(args.path_config, args.profile)
    video_path = args.video
    video_paths = None
    if args.all_config_videos:
        video_paths = profile_video_paths(profile)
    elif video_path is None and args.video_key:
        video_path = profile_video_path(profile, args.video_key)

    return TrackingEvaluationPipelineConfig(
        video_path=video_path,
        video_paths=video_paths,
        gt_xml=args.gt_xml,
        gt_dir=args.gt_dir
        or profile_path(profile, "gt_dir", TRACKING_GT_DIR)
        or TRACKING_GT_DIR,
        video_dir=args.video_dir
        or profile_path(profile, "video_dir", VIDEO_DIR)
        or VIDEO_DIR,
        prediction_root=args.prediction_root
        or profile_path(profile, "prediction_root", PREDICTION_ROOT)
        or PREDICTION_ROOT,
        output_root=args.output_root
        or profile_path(profile, "evaluation_output_root", EVAL_OUTPUT_ROOT)
        or EVAL_OUTPUT_ROOT,
        weights_path=args.weights
        or profile_path(profile, "weights", DETECTOR_WEIGHTS)
        or DETECTOR_WEIGHTS,
        mask_path=args.mask or profile_path(profile, "mask", None),
        iou_threshold=args.iou_threshold,
        include_hidden=args.include_hidden,
        gap_tolerance_frames=args.gap_tolerance_frames,
        run_missing_tracker=args.run_missing_tracker,
        force_track=args.force_track,
        max_frames=args.max_frames,
    )


def main() -> int:
    """CLI entry point."""
    config = config_from_args(parse_args())
    asset_df, metrics_df, output_dir = run_pipeline(config)
    print("[assets]")
    print(asset_df.to_string(index=False))
    print("[metrics]")
    print(metrics_df.to_string(index=False))
    print("[output]", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

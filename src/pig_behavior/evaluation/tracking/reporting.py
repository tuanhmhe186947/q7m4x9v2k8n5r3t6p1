"""Markdown report generation for tracking evaluation."""

from __future__ import annotations

import pandas as pd

from .config import TrackingEvaluationPipelineConfig


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

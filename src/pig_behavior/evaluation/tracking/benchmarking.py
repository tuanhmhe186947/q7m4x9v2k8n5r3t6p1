"""Tracking rule and detector benchmarking."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import TrackingEvaluationPipelineConfig
from .pipeline import run_pipeline
from .reporting import _markdown_table

TRACKING_RULE_FLAG_NAMES = (
    "USE_IOU_FALLBACK",
    "USE_AREA_OCCLUSION_FREEZE",
    "USE_MERGED_BOX_SPLIT",
)


def tracking_rule_combo_name(flags: dict[str, bool]) -> str:
    """Stable folder-friendly name for a rule flag combination."""
    return "_".join(
        [
            f"iou{int(flags['USE_IOU_FALLBACK'])}",
            f"area{int(flags['USE_AREA_OCCLUSION_FREEZE'])}",
            f"merge{int(flags['USE_MERGED_BOX_SPLIT'])}",
        ]
    )


def iter_tracking_rule_flag_combinations() -> list[dict[str, bool]]:
    """Return every True/False combination for the tracking rule flags."""
    return [
        dict(zip(TRACKING_RULE_FLAG_NAMES, values, strict=True))
        for values in itertools.product(
            [False, True],
            repeat=len(TRACKING_RULE_FLAG_NAMES),
        )
    ]


def aggregate_metrics_dict(metrics_df: pd.DataFrame) -> dict[str, object]:
    """Return the aggregate ALL row, or an empty row when no metrics exist."""
    if metrics_df.empty:
        return {}
    aggregate_df = metrics_df[metrics_df["video_stem"] == "ALL"]
    row = aggregate_df.iloc[0] if not aggregate_df.empty else metrics_df.iloc[-1]
    return row.to_dict()


def build_rule_benchmark_report(
    summary_df: pd.DataFrame,
    detailed_metrics_df: pd.DataFrame,
    benchmark_root: Path,
    title: str = "Tracking Rule Flag Benchmark",
) -> str:
    """Build a compact Markdown report for the 8-combo benchmark."""
    preferred_columns = [
        "detector",
        "weights_path",
        "combo",
        "USE_IOU_FALLBACK",
        "USE_AREA_OCCLUSION_FREEZE",
        "USE_MERGED_BOX_SPLIT",
        "elapsed_sec",
        "fps_evaluated_frames",
        "remapped_mota_pct",
        "remapped_idf1_pct",
        "remapped_hota_pct",
        "remapped_idsw",
        "remapped_fragments",
        "remapped_gap_tolerant_fragments",
        "fp",
        "fn",
        "run_dir",
    ]
    columns = [column for column in preferred_columns if column in summary_df.columns]
    lines = [
        f"# {title}",
        "",
        f"- Output folder: `{benchmark_root}`",
        f"- Flag combinations: `{len(summary_df)}`",
        f"- Detailed metric rows: `{len(detailed_metrics_df)}`",
        "",
        "## Aggregate Metrics",
        "",
        _markdown_table(summary_df, columns) if columns else "_No rows._",
        "",
        "## Files",
        "",
        "- `tracking_rule_benchmark_summary.csv`: one aggregate row per combo.",
        "- `tracking_rule_benchmark_detailed_metrics.csv`: all per-video and ALL rows.",
        "- Each combo folder contains the normal `tracking_report.md` and diagnostics.",
        "",
    ]
    return "\n".join(lines)


def run_tracking_rule_benchmark(
    config: TrackingEvaluationPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run all rule flag combinations and write combined benchmark outputs."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_root = config.output_root / "tracking_rule_benchmark" / run_id
    benchmark_prediction_root = (
        config.prediction_root / "tracking_rule_benchmark" / run_id
    )
    benchmark_root.mkdir(parents=True, exist_ok=True)
    benchmark_prediction_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    detailed_metrics: list[pd.DataFrame] = []
    asset_tables: list[pd.DataFrame] = []

    for flags in iter_tracking_rule_flag_combinations():
        combo = tracking_rule_combo_name(flags)
        combo_config = replace(
            config,
            prediction_root=benchmark_prediction_root / combo,
            output_root=benchmark_root / combo,
            run_missing_tracker=True,
            force_track=True,
            **flags,
        )

        started = time.perf_counter()
        asset_df, metrics_df, run_dir = run_pipeline(combo_config)
        elapsed_sec = time.perf_counter() - started
        aggregate_metrics = aggregate_metrics_dict(metrics_df)
        evaluated_frames = float(aggregate_metrics.get("evaluated_frames", 0) or 0)
        fps = evaluated_frames / elapsed_sec if elapsed_sec > 0 else 0.0

        summary_rows.append(
            {
                "detector": config.detector_name,
                "weights_path": str(config.weights_path),
                "combo": combo,
                **flags,
                "elapsed_sec": round(elapsed_sec, 4),
                "fps_evaluated_frames": round(fps, 4),
                "asset_rows": int(len(asset_df)),
                "run_dir": str(run_dir),
                "prediction_root": str(combo_config.prediction_root),
                **aggregate_metrics,
            }
        )

        metrics_with_combo = metrics_df.copy()
        metrics_with_combo.insert(0, "combo", combo)
        metrics_with_combo.insert(0, "detector", config.detector_name)
        for flag_name, flag_value in reversed(tuple(flags.items())):
            metrics_with_combo.insert(2, flag_name, flag_value)
        detailed_metrics.append(metrics_with_combo)

        assets_with_combo = asset_df.copy()
        assets_with_combo.insert(0, "combo", combo)
        assets_with_combo.insert(0, "detector", config.detector_name)
        asset_tables.append(assets_with_combo)

        print(
            f"[benchmark] {config.detector_name}/{combo}: elapsed={elapsed_sec:.2f}s "
            f"fps={fps:.2f} output={run_dir}"
        )

    summary_df = pd.DataFrame(summary_rows)
    detailed_metrics_df = (
        pd.concat(detailed_metrics, ignore_index=True)
        if detailed_metrics
        else pd.DataFrame()
    )
    assets_df = (
        pd.concat(asset_tables, ignore_index=True) if asset_tables else pd.DataFrame()
    )

    summary_df.to_csv(
        benchmark_root / "tracking_rule_benchmark_summary.csv",
        index=False,
    )
    detailed_metrics_df.to_csv(
        benchmark_root / "tracking_rule_benchmark_detailed_metrics.csv",
        index=False,
    )
    assets_df.to_csv(
        benchmark_root / "tracking_rule_benchmark_assets.csv",
        index=False,
    )
    (benchmark_root / "tracking_rule_benchmark_report.md").write_text(
        build_rule_benchmark_report(
            summary_df,
            detailed_metrics_df,
            benchmark_root,
        ),
        encoding="utf-8",
    )
    with (benchmark_root / "tracking_rule_benchmark_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "mode": "rule_flag_benchmark",
                "detector": config.detector_name,
                "weights_path": str(config.weights_path),
                "benchmark_root": str(benchmark_root),
                "benchmark_prediction_root": str(benchmark_prediction_root),
                "rule_flags": list(TRACKING_RULE_FLAG_NAMES),
                "base_config": asdict(config),
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return summary_df, detailed_metrics_df, benchmark_root


def iter_detector_benchmark_configs(
    config: TrackingEvaluationPipelineConfig,
    benchmark_root: Path,
    benchmark_prediction_root: Path,
) -> list[TrackingEvaluationPipelineConfig]:
    """Return isolated v8/v26 configs for detector comparison."""
    detectors = [
        ("yolov8", config.weights_path),
        ("yolov26", config.weights_v26_path),
    ]
    configs: list[TrackingEvaluationPipelineConfig] = []
    for detector_name, weights_path in detectors:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"{detector_name} weights not found: {weights_path}"
            )
        configs.append(
            replace(
                config,
                detector_name=detector_name,
                weights_path=weights_path,
                prediction_root=benchmark_prediction_root / detector_name,
                output_root=benchmark_root / detector_name,
                run_missing_tracker=True,
                force_track=True,
            )
        )
    return configs


def run_tracking_detector_benchmark(
    config: TrackingEvaluationPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run the full rule benchmark independently for YOLOv8 and YOLOv26."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_root = config.output_root / "tracking_detector_benchmark" / run_id
    benchmark_prediction_root = (
        config.prediction_root / "tracking_detector_benchmark" / run_id
    )
    benchmark_root.mkdir(parents=True, exist_ok=True)
    benchmark_prediction_root.mkdir(parents=True, exist_ok=True)

    summary_tables: list[pd.DataFrame] = []
    detailed_tables: list[pd.DataFrame] = []
    detector_configs = iter_detector_benchmark_configs(
        config,
        benchmark_root,
        benchmark_prediction_root,
    )
    for detector_config in detector_configs:
        summary_df, detailed_metrics_df, detector_output = run_tracking_rule_benchmark(
            detector_config,
        )
        summary_tables.append(summary_df)
        detailed_tables.append(detailed_metrics_df)
        print(
            f"[detector-benchmark] {detector_config.detector_name}: "
            f"output={detector_output}"
        )

    summary_df = (
        pd.concat(summary_tables, ignore_index=True)
        if summary_tables
        else pd.DataFrame()
    )
    detailed_metrics_df = (
        pd.concat(detailed_tables, ignore_index=True)
        if detailed_tables
        else pd.DataFrame()
    )
    summary_df.to_csv(
        benchmark_root / "tracking_detector_benchmark_summary.csv",
        index=False,
    )
    detailed_metrics_df.to_csv(
        benchmark_root / "tracking_detector_benchmark_detailed_metrics.csv",
        index=False,
    )
    (benchmark_root / "tracking_detector_benchmark_report.md").write_text(
        build_rule_benchmark_report(
            summary_df,
            detailed_metrics_df,
            benchmark_root,
            title="Tracking Detector Benchmark",
        ),
        encoding="utf-8",
    )
    with (benchmark_root / "tracking_detector_benchmark_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "mode": "detector_benchmark",
                "benchmark_root": str(benchmark_root),
                "benchmark_prediction_root": str(benchmark_prediction_root),
                "detectors": [
                    {
                        "name": detector_config.detector_name,
                        "weights_path": str(detector_config.weights_path),
                    }
                    for detector_config in detector_configs
                ],
                "base_config": asdict(config),
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return summary_df, detailed_metrics_df, benchmark_root

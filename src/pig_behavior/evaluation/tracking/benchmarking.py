"""Tracking rule and detector benchmarking."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .config import TrackingEvaluationPipelineConfig
from .pipeline import run_pipeline
from .reporting import _markdown_table

TRACKING_RULE_FLAG_NAMES = (
    "USE_IOU_FALLBACK",
    "USE_AREA_OCCLUSION_FREEZE",
    "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE",
    "USE_MERGED_BOX_SPLIT",
)


def tracking_rule_combo_name(flags: dict[str, bool]) -> str:
    """Stable folder-friendly name for a rule flag combination."""
    return "_".join(
        [
            f"iou{int(flags['USE_IOU_FALLBACK'])}",
            f"area{int(flags['USE_AREA_OCCLUSION_FREEZE'])}",
            f"condarea{int(flags['USE_CONDITIONAL_AREA_OCCLUSION_FREEZE'])}",
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
        "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE",
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
    
    # Filter for table display in report (only ALL row_type)
    if "row_type" in summary_df.columns:
        table_df = summary_df[summary_df["row_type"] == "ALL"]
    else:
        table_df = summary_df

    lines = [
        f"# {title}",
        "",
        f"- Output folder: `{benchmark_root}`",
        f"- Flag combinations: `{len(table_df)}`",
        f"- Detailed metric rows: `{len(detailed_metrics_df)}`",
        "",
        "## Aggregate Metrics",
        "",
        _markdown_table(table_df, columns) if columns else "_No rows._",
        "",
        "## Files",
        "",
        "- `*_summary.csv`: enriched summary, containing both `ALL` and `PER_VIDEO` rows.",
        "- `*_summary_all_only.csv`: aggregate-only summary used for quick ranking.",
        "- `*_detailed_metrics.csv`: all per-video and ALL rows.",
        "- Each combo folder contains the normal `tracking_report.md` and diagnostics (like ID mapping, event counts, and continuity gaps).",
        "- Note: The diagnostic summary columns in the summary CSV files are only high-level counts/worst-case statistics. To view detailed information, open the respective files in `run_dir`.",
        "",
    ]
    return "\n".join(lines)


def collect_run_diagnostics(run_dir: Path, video_stem: str | None = None) -> dict[str, object]:
    """Read diagnostic files if they exist and return safe counts/worst-case stats, optionally filtered by video_stem."""
    diagnostics = {
        "id_mapping_rows": 0,
        "remapped_identity_event_rows": 0,
        "remapped_id_switch_rows": 0,
        "identity_event_rows": 0,
        "identity_id_switch_rows": 0,
        "continuity_gap_rows": 0,
        "tolerated_gap_rows": 0,
        "remaining_fragment_gap_rows": 0,
        "id_changed_gap_rows": 0,
        "worst_gap_frames": None,
        "worst_gap_gt_id": None,
        "worst_gap_prev_pred_id": None,
        "worst_gap_next_pred_id": None,
    }

    # Helper function to read and filter DataFrame by video_stem
    def read_and_filter(file_name: str) -> pd.DataFrame | None:
        path = run_dir / file_name
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            if video_stem is not None and str(video_stem).upper() != "ALL" and "video_stem" in df.columns:
                df = df[df["video_stem"] == video_stem]
            return df
        except Exception:
            return None

    # 1. Read tracking_id_mapping.csv
    df_map = read_and_filter("tracking_id_mapping.csv")
    if df_map is not None:
        diagnostics["id_mapping_rows"] = int(len(df_map))

    # 2. Read tracking_remapped_identity_events.csv
    df_remap = read_and_filter("tracking_remapped_identity_events.csv")
    if df_remap is not None:
        diagnostics["remapped_identity_event_rows"] = int(len(df_remap))
        if "event" in df_remap.columns:
            diagnostics["remapped_id_switch_rows"] = int(df_remap["event"].str.contains("id_switch", na=False).sum())

    # 3. Read tracking_identity_events.csv
    df_events = read_and_filter("tracking_identity_events.csv")
    if df_events is not None:
        diagnostics["identity_event_rows"] = int(len(df_events))
        if "event" in df_events.columns:
            diagnostics["identity_id_switch_rows"] = int(df_events["event"].str.contains("id_switch", na=False).sum())

    # 4. Read tracking_continuity_gaps.csv
    df_gaps = read_and_filter("tracking_continuity_gaps.csv")
    if df_gaps is not None:
        diagnostics["continuity_gap_rows"] = int(len(df_gaps))
        if "tolerated" in df_gaps.columns:
            tolerated_series = df_gaps["tolerated"].astype(str).str.lower() == "true"
            diagnostics["tolerated_gap_rows"] = int(tolerated_series.sum())
            diagnostics["remaining_fragment_gap_rows"] = int((~tolerated_series).sum())
        if "id_changed" in df_gaps.columns:
            id_changed_series = df_gaps["id_changed"].astype(str).str.lower() == "true"
            diagnostics["id_changed_gap_rows"] = int(id_changed_series.sum())
        if not df_gaps.empty and "gap_frames" in df_gaps.columns:
            try:
                idx_max = df_gaps["gap_frames"].idxmax()
                if pd.notna(idx_max):
                    max_row = df_gaps.loc[idx_max]
                    diagnostics["worst_gap_frames"] = int(max_row["gap_frames"])
                    diagnostics["worst_gap_gt_id"] = str(max_row.get("gt_id", ""))
                    diagnostics["worst_gap_prev_pred_id"] = str(max_row.get("previous_pred_id", ""))
                    diagnostics["worst_gap_next_pred_id"] = str(max_row.get("next_pred_id", ""))
            except Exception:
                pass

    return diagnostics


def build_enriched_summary_rows(
    metrics_df: pd.DataFrame,
    config: TrackingEvaluationPipelineConfig,
    combo: str,
    flags: dict[str, bool],
    run_dir: Path,
    asset_df: pd.DataFrame | None,
    elapsed_sec: float,
    fps: float,
) -> list[dict[str, object]]:
    """Build enriched summary rows including both ALL and PER_VIDEO rows with metrics and diagnostics."""
    rows = []
    for _, row in metrics_df.iterrows():
        video_stem = row.get("video_stem", "")
        row_type = "ALL" if str(video_stem).upper() == "ALL" else "PER_VIDEO"
        
        diagnostics = collect_run_diagnostics(run_dir, video_stem)
        
        entry = {
            "row_type": row_type,
            "video_stem": video_stem,
            "detector": config.detector_name,
            "weights_path": str(config.weights_path),
            "combo": combo,
            **flags,
            "elapsed_sec": round(elapsed_sec, 4),
            "fps_evaluated_frames": round(fps, 4),
            "asset_rows": int(len(asset_df)) if asset_df is not None else 0,
            "prediction_root": str(config.prediction_root),
            "run_dir": str(run_dir),
            **diagnostics,
        }
        # Merge metrics and ensure metadata overrides
        entry.update(row.to_dict())
        
        # Enforce metadata values
        entry["row_type"] = row_type
        entry["video_stem"] = video_stem
        entry["detector"] = config.detector_name
        entry["weights_path"] = str(config.weights_path)
        entry["combo"] = combo
        entry["run_dir"] = str(run_dir)
        for flag_name, flag_val in flags.items():
            entry[flag_name] = flag_val
            
        rows.append(entry)
    return rows


def run_tracking_rule_benchmark(
    config: TrackingEvaluationPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run all rule flag combinations and write combined benchmark outputs."""
    benchmark_root = config.output_root
    benchmark_prediction_root = config.prediction_root
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
            run_missing_tracker=config.run_missing_tracker,
            force_track=config.force_track,
            **flags,
        )

        started = time.perf_counter()
        asset_df, metrics_df, run_dir = run_pipeline(combo_config)
        elapsed_sec = time.perf_counter() - started
        aggregate_metrics = aggregate_metrics_dict(metrics_df)
        evaluated_frames = float(aggregate_metrics.get("evaluated_frames", 0) or 0)
        fps = evaluated_frames / elapsed_sec if elapsed_sec > 0 else 0.0

        # Build enriched summary rows (containing both ALL and PER_VIDEO)
        enriched_rows = build_enriched_summary_rows(
            metrics_df=metrics_df,
            config=combo_config,
            combo=combo,
            flags=flags,
            run_dir=run_dir,
            asset_df=asset_df,
            elapsed_sec=elapsed_sec,
            fps=fps,
        )
        summary_rows.extend(enriched_rows)

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

    # Save summary.csv (containing ALL + PER_VIDEO)
    summary_df.to_csv(
        benchmark_root / "tracking_rule_benchmark_summary.csv",
        index=False,
    )
    # Save summary_all_only.csv (containing only ALL)
    summary_all_only_df = summary_df[summary_df["row_type"] == "ALL"]
    summary_all_only_df.to_csv(
        benchmark_root / "tracking_rule_benchmark_summary_all_only.csv",
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
                run_missing_tracker=config.run_missing_tracker,
                force_track=config.force_track,
            )
        )
    return configs


def run_tracking_detector_benchmark(
    config: TrackingEvaluationPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run the full rule benchmark independently for YOLOv8 and YOLOv26."""
    benchmark_root = config.output_root
    benchmark_prediction_root = config.prediction_root
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
    # Save aggregate-only detector summary
    summary_all_only_df = summary_df[summary_df["row_type"] == "ALL"]
    summary_all_only_df.to_csv(
        benchmark_root / "tracking_detector_benchmark_summary_all_only.csv",
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

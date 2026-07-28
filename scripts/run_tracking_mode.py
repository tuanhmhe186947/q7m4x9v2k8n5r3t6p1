#!/usr/bin/env python3
"""Presentation-friendly runner for tracking mode comparisons."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.tracking.profiles import (  # noqa: E402
    PRESENTATION_PROFILES,
    RetiredTrackingProfileError,
    get_eval_config,
    get_presentation_profile,
)

DEFAULT_RULE_COMBO = "iou0_area0_condarea0_merge0"
DEFAULT_COMPARE_MODES = "bytetrack_raw,realtime_fast,hybrid_bytetrack"
SUMMARY_COLUMNS = [
    "presentation_mode",
    "baseline_role",
    "causality_level",
    "uses_offline_smoothing",
    "uses_identity_repair",
    "uses_delayed_repair",
    "detect_every_n_frames",
    "latency_window_frames",
    "output_timing_contract",
    "declared_delay_frames",
    "tracking_mode",
    "eval_config",
    "rule_output",
    "video_stem",
    "gt_detections",
    "pred_detections",
    "matches",
    "precision_pct",
    "recall_pct",
    "mota_pct",
    "motp_iou_pct",
    "idf1_pct",
    "hota_pct",
    "remapped_idsw",
    "idsw",
    "remapped_mota_pct",
    "remapped_hota_pct",
    "remapped_idf1_pct",
    "remapped_assa_pct",
    "idmap_coverage_pct",
    "fp",
    "fn",
    "fragments",
    "remapped_fragments",
    "gap_tolerant_fragments",
    "remapped_gap_tolerant_fragments",
    "tracklets",
    "remapped_tracklets",
    "evaluated_frames",
    "video_frame_count",
    "video_fps",
    "video_duration_sec",
    "compare_elapsed_sec",
    "compare_evaluated_fps",
    "compare_realtime_factor",
]
RUNTIME_COLUMNS = [
    "presentation_mode",
    "baseline_role",
    "causality_level",
    "tracking_mode",
    "eval_config",
    "status",
    "return_code",
    "compare_elapsed_sec",
    "evaluated_frames",
    "compare_evaluated_fps",
    "video_frame_count",
    "video_duration_sec",
    "compare_realtime_factor",
    "output_timing_contract",
    "declared_delay_frames",
]
SCIENTIFIC_COLUMNS = [
    "presentation_mode",
    "baseline_role",
    "causality_level",
    "uses_offline_smoothing",
    "uses_identity_repair",
    "uses_delayed_repair",
    "detect_every_n_frames",
    "latency_window_frames",
    "output_timing_contract",
    "declared_delay_frames",
    "evaluated_video_count",
    "evaluated_frames",
    "video_frame_count",
    "video_duration_sec",
    "remapped_idsw_total",
    "remapped_idsw_mean",
    "remapped_idsw_std",
    "remapped_idsw_median",
    "remapped_hota_pct_mean",
    "remapped_hota_pct_std",
    "remapped_hota_pct_median",
    "remapped_idf1_pct_mean",
    "remapped_idf1_pct_std",
    "remapped_idf1_pct_median",
    "fp_total",
    "fn_total",
    "remapped_fragments_total",
    "compare_elapsed_sec",
    "compare_evaluated_fps",
    "compare_realtime_factor",
]


def _active_profile_arg(value: str) -> str:
    try:
        get_presentation_profile(value)
    except RetiredTrackingProfileError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    except KeyError as exc:
        choices = ", ".join(sorted(PRESENTATION_PROFILES))
        raise argparse.ArgumentTypeError(
            f"Unknown profile '{value}'. Active profiles: {choices}."
        ) from exc
    return value


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run tracking/evaluation using a named presentation mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/run_tracking_mode.py --mode hybrid_bytetrack --task eval -a
  python scripts/run_tracking_mode.py --mode realtime_fast --task eval -v Pigs291119_000263_30fps
  python scripts/run_tracking_mode.py --mode bytetrack_raw --task eval -a
  python scripts/run_tracking_mode.py --task compare -v Pigs291119_000263_30fps

Default eval rule combo: {DEFAULT_RULE_COMBO}
Use --all-rule-combos to run the full rule benchmark matrix instead.
""",
    )
    parser.add_argument(
        "--mode",
        type=_active_profile_arg,
        default="hybrid_bytetrack",
        metavar="PROFILE",
        help=(
            "Active presentation profile to run: "
            f"{', '.join(sorted(PRESENTATION_PROFILES))}."
        ),
    )
    parser.add_argument(
        "--compare-modes",
        nargs="?",
        const=DEFAULT_COMPARE_MODES,
        default=None,
        metavar="MODES",
        help=(
            "Mode list for --task compare. Omit MODES to compare "
            f"{DEFAULT_COMPARE_MODES}, or pass a comma-separated "
            "list of active profiles."
        ),
    )
    parser.add_argument(
        "--task",
        choices=["track", "eval", "compare"],
        default="eval",
        help=(
            "track=generate prediction/XML only; eval=run evaluation and let "
            "evaluate_tracking.py track missing predictions unless --eval-existing is set; "
            "compare=run eval for multiple modes and write CSV/Markdown summaries."
        ),
    )
    parser.add_argument(
        "--eval-existing",
        action="store_true",
        help=(
            "With --task eval, evaluate existing predictions only and do not "
            "run missing tracking."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-v",
        "--video",
        type=str,
        help="Comma-separated names, paths, keys, or aliases.",
    )
    group.add_argument(
        "-a",
        "--all-videos",
        action="store_true",
        help="Run on all configured videos.",
    )
    parser.add_argument(
        "--path-profile",
        type=str,
        default=None,
        help="Path profile forwarded to track_videos.py/evaluate_tracking.py as --profile.",
    )
    parser.add_argument("--path-config", type=str, default=None)
    parser.add_argument(
        "--rule-combo",
        action="append",
        default=None,
        help="Evaluation rule combo. Defaults to iou0_area0_condarea0_merge0.",
    )
    parser.add_argument(
        "--all-rule-combos",
        action="store_true",
        help="Do not force the default rule combo; let evaluation run all rule combos.",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List presentation modes and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    parser.add_argument(
        "--compare-output-root",
        type=str,
        default=None,
        help="Output root for --task compare summaries and per-mode eval outputs.",
    )
    parser.add_argument(
        "--compare-prediction-root",
        type=str,
        default=None,
        help="Prediction root for --task compare per-mode predictions.",
    )
    args, extra_args = parser.parse_known_args(argv)
    if args.compare_modes is not None and args.task != "compare":
        parser.error("--compare-modes is only valid with --task compare")
    if not args.list_modes and not args.video and not args.all_videos:
        parser.error("one of the arguments -v/--video -a/--all-videos is required")
    return args, extra_args


def _append_video_selection(cmd: list[str], args: argparse.Namespace) -> None:
    if args.all_videos:
        cmd.append("-a")
    else:
        cmd.extend(["-v", args.video])


def _append_path_selection(cmd: list[str], args: argparse.Namespace) -> None:
    if args.path_profile:
        cmd.extend(["--profile", args.path_profile])
    if args.path_config:
        cmd.extend(["--path-config", args.path_config])


def _rule_combos(args: argparse.Namespace) -> list[str]:
    if args.all_rule_combos:
        return []
    raw_combos = args.rule_combo or [DEFAULT_RULE_COMBO]
    combos: list[str] = []
    for raw_combo in raw_combos:
        combos.extend(part.strip() for part in raw_combo.split(",") if part.strip())
    return combos


def _tracking_command(args: argparse.Namespace, mode: str, eval_config: str) -> list[str]:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "track_videos.py")]
    _append_video_selection(cmd, args)
    _append_path_selection(cmd, args)
    cmd.extend(["--eval-config", eval_config, "--mode", mode])
    return cmd


def _evaluation_command(
    args: argparse.Namespace,
    mode: str,
    eval_config: str,
    *,
    output_root: Path | None = None,
    prediction_root: Path | None = None,
) -> list[str]:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_tracking.py")]
    _append_video_selection(cmd, args)
    _append_path_selection(cmd, args)
    cmd.extend(["--mode", mode, "--eval-config", eval_config])
    for combo in _rule_combos(args):
        cmd.extend(["--rule-combo", combo])
    if output_root is not None:
        cmd.extend(["--output-root", str(output_root)])
    if prediction_root is not None:
        cmd.extend(["--prediction-root", str(prediction_root)])
    if args.eval_existing:
        cmd.append("--no-run-missing-tracker")
    return cmd


def _selected_modes(args: argparse.Namespace) -> list[str]:
    if args.task != "compare":
        return [args.mode]
    raw_modes = args.compare_modes or DEFAULT_COMPARE_MODES
    modes = [part.strip() for part in raw_modes.split(",") if part.strip()]
    for name in modes:
        try:
            get_presentation_profile(name)
        except RetiredTrackingProfileError as exc:
            raise ValueError(str(exc)) from exc
        except KeyError as exc:
            choices = ", ".join(sorted(PRESENTATION_PROFILES))
            raise ValueError(
                f"Unknown profile '{name}'. Active profiles: {choices}."
            ) from exc
    if not modes:
        choices = ", ".join(sorted(PRESENTATION_PROFILES))
        raise ValueError(f"No profiles selected. Active profiles: {choices}.")
    return modes


def _compare_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.compare_output_root)
        if args.compare_output_root
        else PROJECT_ROOT / "outputs" / "eval" / "mode_compare" / run_name
    )
    prediction_root = (
        Path(args.compare_prediction_root)
        if args.compare_prediction_root
        else PROJECT_ROOT / "outputs" / "pred" / "mode_compare" / run_name
    )
    return output_root, prediction_root


def _read_metrics_rows(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_sum(values: list[object]) -> float | None:
    numeric_values = [_safe_float(value) for value in values]
    present_values = [value for value in numeric_values if value is not None]
    if not present_values:
        return None
    return float(sum(present_values))


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _mode_science_metadata(
    presentation_mode: str,
    tracking_mode: str,
    eval_config: str,
    overrides: dict[str, object],
) -> dict[str, str]:
    offline_smoothing = _truthy(overrides.get("enable_offline_smoothing", False))
    identity_repair_keys = [
        "identity_swap_guard",
        "hidden_owner_guard",
        "hidden_owner_guard_hold_assignment",
        "reentry_unowned_raw_mismatch_episode_reject",
        "occlusion_reid_prefer_gap_over_bad_match",
        "overlap_small_box_suppression",
        "hidden_suffix_id_swap_repair",
        "suffix_pair_swap_repair",
        "realtime_visible_close_competitor_guard",
        "realtime_visible_better_competitor_reject",
        "realtime_visible_better_competitor_prefer",
        "realtime_low_conf_recovery_guard",
    ]
    uses_identity_repair = any(
        _truthy(overrides.get(key, False)) for key in identity_repair_keys
    )
    uses_motion_pair_stabilizer = (
        tracking_mode == "realtime"
        and _truthy(overrides.get("realtime_motion_pair_stabilizer", False))
    )
    fixed_lag_frames = int(
        overrides.get("realtime_motion_pair_fixed_lag_frames", 0) or 0
    )
    uses_fixed_lag = uses_motion_pair_stabilizer and fixed_lag_frames > 0
    uses_global_graph = uses_motion_pair_stabilizer and not uses_fixed_lag
    uses_delayed_repair = uses_motion_pair_stabilizer or (
        offline_smoothing
        and _truthy(overrides.get("local_pair_swap_repair", False))
    )
    detect_every_n_frames = str(overrides.get("detect_every_n_frames", ""))
    latency_window = ""
    if uses_fixed_lag:
        latency_window = str(fixed_lag_frames)
    elif offline_smoothing and _truthy(
        overrides.get("local_pair_swap_repair", False)
    ):
        latency_window = str(overrides.get("local_pair_swap_window_frames", ""))

    if presentation_mode == "bytetrack_raw" or eval_config == "bytetrack_raw":
        baseline_role = "raw_bytetrack_baseline_same_detector_pipeline"
        causality_level = "online_raw"
        output_timing_contract = "causal_framewise"
        declared_delay_frames = 0
    elif uses_fixed_lag:
        baseline_role = "realtime_quality_fixed_lag_candidate"
        causality_level = "fixed_lag_realtime"
        output_timing_contract = "fixed_lag_framewise"
        declared_delay_frames = fixed_lag_frames
    elif uses_global_graph:
        baseline_role = "post_video_global_graph_candidate"
        causality_level = "post_video_global_graph"
        output_timing_contract = "post_video_global_graph"
        declared_delay_frames = -1
    elif tracking_mode == "realtime":
        baseline_role = "causal_realtime_candidate"
        causality_level = "online_realtime"
        output_timing_contract = "causal_framewise"
        declared_delay_frames = 0
    elif offline_smoothing:
        baseline_role = "offline_quality_upper_bound"
        causality_level = "offline_postprocessed"
        output_timing_contract = "post_video_offline"
        declared_delay_frames = -1
    else:
        baseline_role = "tracking_candidate"
        causality_level = "online_or_near_online"
        output_timing_contract = "causal_framewise"
        declared_delay_frames = 0

    return {
        "baseline_role": baseline_role,
        "causality_level": causality_level,
        "uses_offline_smoothing": str(offline_smoothing).lower(),
        "uses_identity_repair": str(uses_identity_repair).lower(),
        "uses_delayed_repair": str(uses_delayed_repair).lower(),
        "detect_every_n_frames": detect_every_n_frames,
        "latency_window_frames": latency_window,
        "output_timing_contract": output_timing_contract,
        "declared_delay_frames": str(declared_delay_frames),
    }


def _asset_rows_by_video(metrics_csv: Path) -> dict[str, dict[str, str]]:
    assets_csv = metrics_csv.parent / "tracking_eval_assets.csv"
    if not assets_csv.exists():
        return {}
    with assets_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assets = {row.get("video_stem", ""): row for row in rows if row.get("video_stem")}
    total_frames = _safe_sum([row.get("video_frame_count") for row in rows])
    duration_values = []
    for row in rows:
        frame_count = _safe_float(row.get("video_frame_count"))
        fps = _safe_float(row.get("video_fps"))
        if frame_count is not None and fps and fps > 0:
            duration_values.append(frame_count / fps)
    if total_frames is not None or duration_values:
        assets["ALL"] = {
            "video_stem": "ALL",
            "video_frame_count": _format_float(total_frames, digits=0),
            "video_fps": "",
            "video_duration_sec": (
                _format_float(sum(duration_values), digits=4)
                if duration_values
                else ""
            ),
        }
    return assets


def _video_context(
    row: dict[str, str],
    assets_by_video: dict[str, dict[str, str]],
) -> dict[str, str]:
    asset = assets_by_video.get(row.get("video_stem", ""), {})
    video_frame_count = asset.get("video_frame_count", "")
    video_fps = asset.get("video_fps", "")
    video_duration_sec = asset.get("video_duration_sec", "")
    if not video_duration_sec:
        frame_count = _safe_float(video_frame_count)
        fps = _safe_float(video_fps)
        if frame_count is not None and fps and fps > 0:
            video_duration_sec = _format_float(frame_count / fps)
    return {
        "video_frame_count": video_frame_count,
        "video_fps": video_fps,
        "video_duration_sec": video_duration_sec,
    }


def _runtime_context(
    row: dict[str, str],
    runtime_metadata: dict[str, dict[str, str]],
    presentation_mode: str,
    video_duration_sec: str,
) -> dict[str, str]:
    metadata = runtime_metadata.get(presentation_mode, {})
    elapsed = _safe_float(metadata.get("compare_elapsed_sec"))
    evaluated_frames = _safe_float(row.get("evaluated_frames"))
    duration = _safe_float(video_duration_sec)
    evaluated_fps = (
        evaluated_frames / elapsed
        if evaluated_frames is not None and elapsed and elapsed > 0
        else None
    )
    realtime_factor = (
        duration / elapsed
        if duration is not None and elapsed and elapsed > 0
        else None
    )
    return {
        "compare_elapsed_sec": metadata.get("compare_elapsed_sec", ""),
        "compare_evaluated_fps": _format_float(evaluated_fps),
        "compare_realtime_factor": _format_float(realtime_factor),
    }


def _write_compare_summary(
    compare_output_root: Path,
    mode_metadata: dict[str, dict[str, str]],
    runtime_metadata: dict[str, dict[str, str]] | None = None,
) -> tuple[Path, Path]:
    runtime_metadata = runtime_metadata or {}
    summary_rows: list[dict[str, str]] = []
    for metrics_csv in sorted(compare_output_root.rglob("tracking_metrics.csv")):
        try:
            relative = metrics_csv.relative_to(compare_output_root)
        except ValueError:
            relative = metrics_csv
        presentation_mode = relative.parts[0] if relative.parts else ""
        metadata = mode_metadata.get(presentation_mode, {})
        science_metadata = {
            key: metadata.get(key, "")
            for key in (
                "baseline_role",
                "causality_level",
                "uses_offline_smoothing",
                "uses_identity_repair",
                "uses_delayed_repair",
                "detect_every_n_frames",
                "latency_window_frames",
            )
        }
        rule_output = str(relative.parent)
        assets_by_video = _asset_rows_by_video(metrics_csv)
        for row in _read_metrics_rows(metrics_csv):
            video_context = _video_context(row, assets_by_video)
            runtime_context = _runtime_context(
                row,
                runtime_metadata,
                presentation_mode,
                video_context["video_duration_sec"],
            )
            summary_rows.append(
                {
                    "presentation_mode": presentation_mode,
                    **science_metadata,
                    "tracking_mode": metadata.get("tracking_mode", ""),
                    "eval_config": metadata.get("eval_config", ""),
                    "rule_output": rule_output,
                    "video_stem": row.get("video_stem", ""),
                    "gt_detections": row.get("gt_detections", ""),
                    "pred_detections": row.get("pred_detections", ""),
                    "matches": row.get("matches", ""),
                    "precision_pct": row.get("precision_pct", ""),
                    "recall_pct": row.get("recall_pct", ""),
                    "mota_pct": row.get("mota_pct", ""),
                    "motp_iou_pct": row.get("motp_iou_pct", ""),
                    "idf1_pct": row.get("idf1_pct", ""),
                    "hota_pct": row.get("hota_pct", ""),
                    "remapped_idsw": row.get("remapped_idsw", ""),
                    "idsw": row.get("idsw", ""),
                    "remapped_mota_pct": row.get("remapped_mota_pct", ""),
                    "remapped_hota_pct": row.get("remapped_hota_pct", ""),
                    "remapped_idf1_pct": row.get("remapped_idf1_pct", ""),
                    "remapped_assa_pct": row.get("remapped_assa_pct", ""),
                    "idmap_coverage_pct": row.get("idmap_coverage_pct", ""),
                    "fp": row.get("fp", ""),
                    "fn": row.get("fn", ""),
                    "fragments": row.get("fragments", ""),
                    "remapped_fragments": row.get("remapped_fragments", ""),
                    "gap_tolerant_fragments": row.get("gap_tolerant_fragments", ""),
                    "remapped_gap_tolerant_fragments": row.get(
                        "remapped_gap_tolerant_fragments",
                        "",
                    ),
                    "tracklets": row.get("tracklets", ""),
                    "remapped_tracklets": row.get("remapped_tracklets", ""),
                    "evaluated_frames": row.get("evaluated_frames", ""),
                    **video_context,
                    **runtime_context,
                }
            )

    compare_output_root.mkdir(parents=True, exist_ok=True)
    csv_path = compare_output_root / "mode_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = compare_output_root / "mode_comparison_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Mode Comparison Summary\n\n")
        handle.write(f"- Rows: `{len(summary_rows)}`\n")
        handle.write(f"- Output root: `{compare_output_root}`\n\n")
        handle.write("| " + " | ".join(SUMMARY_COLUMNS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |\n")
        for row in summary_rows:
            values = (str(row.get(column, "")) for column in SUMMARY_COLUMNS)
            handle.write("| " + " | ".join(values) + " |\n")
    return csv_path, md_path


def _write_runtime_summary(
    compare_output_root: Path,
    mode_metadata: dict[str, dict[str, str]],
    runtime_metadata: dict[str, dict[str, str]],
) -> tuple[Path, Path]:
    rows: list[dict[str, str]] = []
    summary_csv = compare_output_root / "mode_comparison_summary.csv"
    summary_rows = _read_metrics_rows(summary_csv) if summary_csv.exists() else []
    all_rows = {
        row.get("presentation_mode", ""): row
        for row in summary_rows
        if row.get("video_stem") == "ALL"
    }
    for presentation_mode, metadata in mode_metadata.items():
        runtime = runtime_metadata.get(presentation_mode, {})
        all_row = all_rows.get(presentation_mode, {})
        rows.append(
            {
                "presentation_mode": presentation_mode,
                "baseline_role": metadata.get("baseline_role", ""),
                "causality_level": metadata.get("causality_level", ""),
                "tracking_mode": metadata.get("tracking_mode", ""),
                "eval_config": metadata.get("eval_config", ""),
                "status": runtime.get("status", ""),
                "return_code": runtime.get("return_code", ""),
                "compare_elapsed_sec": runtime.get("compare_elapsed_sec", ""),
                "evaluated_frames": all_row.get("evaluated_frames", ""),
                "compare_evaluated_fps": all_row.get("compare_evaluated_fps", ""),
                "video_frame_count": all_row.get("video_frame_count", ""),
                "video_duration_sec": all_row.get("video_duration_sec", ""),
                "compare_realtime_factor": all_row.get("compare_realtime_factor", ""),
            }
        )

    csv_path = compare_output_root / "mode_runtime_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = compare_output_root / "mode_runtime_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Mode Runtime Summary\n\n")
        handle.write(
            "`compare_elapsed_sec` measures the full compare subprocess for each mode, "
            "including tracking if predictions were missing and evaluation/report writing.\n\n"
        )
        handle.write("| " + " | ".join(RUNTIME_COLUMNS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in RUNTIME_COLUMNS) + " |\n")
        for row in rows:
            values = (str(row.get(column, "")) for column in RUNTIME_COLUMNS)
            handle.write("| " + " | ".join(values) + " |\n")
    return csv_path, md_path


def _numeric_series(rows: list[dict[str, str]], column: str) -> list[float]:
    return [value for value in (_safe_float(row.get(column)) for row in rows) if value is not None]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _scientific_stat(
    rows: list[dict[str, str]],
    column: str,
    suffix: str,
) -> dict[str, str]:
    values = _numeric_series(rows, column)
    return {
        f"{column}_{suffix}_mean": _format_float(_mean(values)),
        f"{column}_{suffix}_std": _format_float(_std(values)),
        f"{column}_{suffix}_median": _format_float(_median(values)),
    }


def _write_scientific_summary(compare_output_root: Path) -> tuple[Path, Path]:
    summary_csv = compare_output_root / "mode_comparison_summary.csv"
    summary_rows = _read_metrics_rows(summary_csv) if summary_csv.exists() else []
    modes = sorted({row.get("presentation_mode", "") for row in summary_rows})
    rows: list[dict[str, str]] = []
    for presentation_mode in modes:
        if not presentation_mode:
            continue
        mode_rows = [
            row
            for row in summary_rows
            if row.get("presentation_mode") == presentation_mode
        ]
        all_row = next((row for row in mode_rows if row.get("video_stem") == "ALL"), {})
        per_video_rows = [row for row in mode_rows if row.get("video_stem") != "ALL"]
        hota_stats = _scientific_stat(per_video_rows, "remapped_hota_pct", "per_video")
        idf1_stats = _scientific_stat(per_video_rows, "remapped_idf1_pct", "per_video")
        idsw_stats = _scientific_stat(per_video_rows, "remapped_idsw", "per_video")
        row = {
            "presentation_mode": presentation_mode,
            "baseline_role": all_row.get("baseline_role", ""),
            "causality_level": all_row.get("causality_level", ""),
            "uses_offline_smoothing": all_row.get("uses_offline_smoothing", ""),
            "uses_identity_repair": all_row.get("uses_identity_repair", ""),
            "uses_delayed_repair": all_row.get("uses_delayed_repair", ""),
            "detect_every_n_frames": all_row.get("detect_every_n_frames", ""),
            "latency_window_frames": all_row.get("latency_window_frames", ""),
            "evaluated_video_count": str(len(per_video_rows)),
            "evaluated_frames": all_row.get("evaluated_frames", ""),
            "video_frame_count": all_row.get("video_frame_count", ""),
            "video_duration_sec": all_row.get("video_duration_sec", ""),
            "remapped_idsw_total": all_row.get("remapped_idsw", ""),
            "remapped_idsw_mean": idsw_stats["remapped_idsw_per_video_mean"],
            "remapped_idsw_std": idsw_stats["remapped_idsw_per_video_std"],
            "remapped_idsw_median": idsw_stats["remapped_idsw_per_video_median"],
            "remapped_hota_pct_mean": hota_stats["remapped_hota_pct_per_video_mean"],
            "remapped_hota_pct_std": hota_stats["remapped_hota_pct_per_video_std"],
            "remapped_hota_pct_median": hota_stats["remapped_hota_pct_per_video_median"],
            "remapped_idf1_pct_mean": idf1_stats["remapped_idf1_pct_per_video_mean"],
            "remapped_idf1_pct_std": idf1_stats["remapped_idf1_pct_per_video_std"],
            "remapped_idf1_pct_median": idf1_stats["remapped_idf1_pct_per_video_median"],
            "fp_total": all_row.get("fp", ""),
            "fn_total": all_row.get("fn", ""),
            "remapped_fragments_total": all_row.get("remapped_fragments", ""),
            "compare_elapsed_sec": all_row.get("compare_elapsed_sec", ""),
            "compare_evaluated_fps": all_row.get("compare_evaluated_fps", ""),
            "compare_realtime_factor": all_row.get("compare_realtime_factor", ""),
        }
        rows.append(row)

    csv_path = compare_output_root / "mode_scientific_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCIENTIFIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = compare_output_root / "mode_scientific_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Mode Scientific Summary\n\n")
        handle.write(
            "Per-video mean/std/median columns exclude the aggregate `ALL` "
            "row; total columns use the `ALL` row.\n\n"
        )
        handle.write("| " + " | ".join(SCIENTIFIC_COLUMNS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in SCIENTIFIC_COLUMNS) + " |\n")
        for row in rows:
            values = (
                str(row.get(column, "")) for column in SCIENTIFIC_COLUMNS
            )
            handle.write("| " + " | ".join(values) + " |\n")
    return csv_path, md_path


def _print_modes() -> None:
    print("Available presentation modes:")
    for name, metadata in PRESENTATION_PROFILES.items():
        print(f" - {name}: tracking_mode={metadata['mode']}, eval_config={metadata['eval_config']}")
        print(f"   {metadata['description']}")


def main(argv: list[str] | None = None) -> int:
    args, extra_args = parse_args(argv)
    if args.list_modes:
        _print_modes()
        return 0

    commands: list[tuple[str, list[str]]] = []
    mode_metadata: dict[str, dict[str, str]] = {}
    try:
        selected_modes = _selected_modes(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    compare_output_root: Path | None = None
    compare_prediction_root: Path | None = None
    if args.task == "compare":
        compare_output_root, compare_prediction_root = _compare_roots(args)
    for mode_name in selected_modes:
        profile = get_presentation_profile(mode_name)
        tracking_mode = str(profile["mode"])
        eval_config = str(profile["eval_config"])
        overrides = get_eval_config(eval_config)
        mode_metadata[mode_name] = {
            "tracking_mode": tracking_mode,
            "eval_config": eval_config,
            **_mode_science_metadata(
                mode_name,
                tracking_mode,
                eval_config,
                overrides,
            ),
        }
        if args.task == "track":
            commands.append((mode_name, _tracking_command(args, tracking_mode, eval_config)))
        elif args.task == "compare":
            assert compare_output_root is not None
            assert compare_prediction_root is not None
            commands.append(
                (
                    mode_name,
                    _evaluation_command(
                        args,
                        tracking_mode,
                        eval_config,
                        output_root=compare_output_root / mode_name,
                        prediction_root=compare_prediction_root / mode_name,
                    ),
                )
            )
        else:
            commands.append((mode_name, _evaluation_command(args, tracking_mode, eval_config)))

    runtime_metadata: dict[str, dict[str, str]] = {}
    for mode_name, cmd in commands:
        cmd.extend(extra_args)
        print(f"Command: {' '.join(cmd)}")
        if args.dry_run:
            continue
        started = time.perf_counter()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        elapsed_sec = time.perf_counter() - started
        runtime_metadata[mode_name] = {
            "status": "ok" if result.returncode == 0 else "failed",
            "return_code": str(result.returncode),
            "compare_elapsed_sec": _format_float(elapsed_sec),
        }
        if result.returncode != 0:
            return result.returncode
    if args.task == "compare" and not args.dry_run:
        assert compare_output_root is not None
        csv_path, md_path = _write_compare_summary(
            compare_output_root,
            mode_metadata,
            runtime_metadata,
        )
        runtime_csv_path, runtime_md_path = _write_runtime_summary(
            compare_output_root,
            mode_metadata,
            runtime_metadata,
        )
        scientific_csv_path, scientific_md_path = _write_scientific_summary(compare_output_root)
        print(f"[compare-summary-csv] {csv_path}")
        print(f"[compare-summary-md] {md_path}")
        print(f"[compare-runtime-csv] {runtime_csv_path}")
        print(f"[compare-runtime-md] {runtime_md_path}")
        print(f"[compare-scientific-csv] {scientific_csv_path}")
        print(f"[compare-scientific-md] {scientific_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

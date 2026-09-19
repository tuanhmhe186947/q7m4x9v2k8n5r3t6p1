#!/usr/bin/env python3
"""Automated multi-objective search for stable hybrid_bytetrack metrics."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

@dataclass(frozen=True)
class SearchPreset:
    name: str
    description: str
    overrides: dict[str, object]
    scopes: tuple[str, ...] = ("quick", "balanced", "full")
    family: str = "general"


@dataclass(frozen=True)
class RuleCombo:
    name: str
    flags: dict[str, bool]


@dataclass(frozen=True)
class SmoothMode:
    name: str
    overrides: dict[str, bool]


DEFAULT_TARGET_VIDEO_STEMS = (
    "Pigs291119_000263_30fps",
    "Pigs291119_000226_30fps",
    "Pigs301119_000327_30fps",
    "Pigs301119_000328_30fps",
)
DEFAULT_TARGET_VIDEO_SOURCE = (
    PROJECT_ROOT
    / "outputs"
    / "eval"
    / "hybrid_bytetrack"
    / "Tracking mới tắt smooth"
    / "yolov8"
    / "iou0_area0_condarea0_merge0"
    / "tracking_metrics.csv"
)

PRESETS: tuple[SearchPreset, ...] = (
    SearchPreset("base", "Current hybrid_bytetrack defaults.", {}, family="baseline"),
    SearchPreset(
        "det023",
        "Slightly lower detector threshold.",
        {"det_conf": 0.23},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "det022",
        "Lower detector threshold to 0.22.",
        {"det_conf": 0.22},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "det020",
        "Lower detector threshold to 0.20.",
        {"det_conf": 0.20},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "det018",
        "Aggressive detector recall at 0.18.",
        {"det_conf": 0.18},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "det015",
        "Very aggressive detector recall at 0.15.",
        {"det_conf": 0.15},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "det022_raw64",
        "det_conf 0.22 with more raw detections.",
        {"det_conf": 0.22, "max_raw_detections": 64},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "det020_raw64",
        "det_conf 0.20 with more raw detections.",
        {"det_conf": 0.20, "max_raw_detections": 64},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "det020_raw96",
        "det_conf 0.20 with wide raw detection budget.",
        {"det_conf": 0.20, "max_raw_detections": 96},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "track045",
        "Lower ByteTrack high-confidence and new-track gates.",
        {
            "track_high_conf": 0.45,
            "initial_track_conf": 0.45,
            "motion_gate_confidence": 0.45,
        },
        family="association",
    ),
    SearchPreset(
        "det022_track045",
        "Lower high-confidence tracking gate to 0.45.",
        {
            "det_conf": 0.22,
            "track_high_conf": 0.45,
            "initial_track_conf": 0.45,
            "motion_gate_confidence": 0.45,
        },
        ("balanced", "full"),
        family="association",
    ),
    SearchPreset(
        "track055",
        "Raise ByteTrack high-confidence and new-track gates.",
        {
            "track_high_conf": 0.55,
            "initial_track_conf": 0.55,
            "motion_gate_confidence": 0.55,
        },
        ("balanced", "full"),
        family="association",
    ),
    SearchPreset(
        "det022_track055",
        "Raise high-confidence tracking gate to 0.55.",
        {
            "det_conf": 0.22,
            "track_high_conf": 0.55,
            "initial_track_conf": 0.55,
            "motion_gate_confidence": 0.55,
        },
        ("full",),
        family="association",
    ),
    SearchPreset(
        "match075",
        "Loosen ByteTrack association match threshold toward constants.py default.",
        {"track_match_iou": 0.75},
        family="association",
    ),
    SearchPreset(
        "match085",
        "Tighten ByteTrack association match threshold.",
        {"track_match_iou": 0.85},
        ("balanced", "full"),
        family="association",
    ),
    SearchPreset(
        "det020_loose_motion",
        "Allow low-confidence matches to move farther.",
        {
            "det_conf": 0.20,
            "low_conf_max_center_jump": 0.10,
            "low_conf_max_box_jump_scale": 2.00,
            "max_raw_detections": 64,
        },
        family="association",
    ),
    SearchPreset(
        "low_conf_tight_motion",
        "Restrict low-confidence motion rescue.",
        {
            "low_conf_max_center_jump": 0.06,
            "low_conf_max_box_jump_scale": 1.50,
            "low_conf_min_iou": 0.03,
        },
        ("balanced", "full"),
        family="association",
    ),
    SearchPreset(
        "det020_loose_reid",
        "Loosen lost-track re-identification appearance threshold.",
        {
            "det_conf": 0.20,
            "lost_track_reid_appearance_threshold": 0.32,
            "max_raw_detections": 64,
        },
        ("full",),
        family="association",
    ),
    SearchPreset(
        "reid_strict",
        "Tighten lost-track re-identification appearance threshold.",
        {"lost_track_reid_appearance_threshold": 0.18},
        ("balanced", "full"),
        family="association",
    ),
    SearchPreset(
        "maxlost60",
        "Shorter lost-track buffer than hybrid_bytetrack default.",
        {"max_missing_frames": 60, "max_lost_frames": 60},
        ("balanced", "full"),
        family="lifecycle",
    ),
    SearchPreset(
        "maxlost120",
        "Longer lost-track buffer for recovery after long occlusion gaps.",
        {"max_missing_frames": 120, "max_lost_frames": 120},
        ("balanced", "full"),
        family="lifecycle",
    ),
    SearchPreset(
        "nms085",
        "Keep more overlapping detector boxes before association.",
        {"nms_iou": 0.85},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "det022_nms085",
        "Moderate recall plus looser detector NMS.",
        {"det_conf": 0.22, "nms_iou": 0.85, "max_raw_detections": 64},
        ("detector_probe",),
        family="detection",
    ),
    SearchPreset(
        "identity_guard_less_sensitive",
        "Require stronger gain before identity swap guard rewrites IDs.",
        {"identity_swap_min_gain": 0.030, "identity_swap_iom_threshold": 0.12},
        ("balanced", "full"),
        family="occlusion_identity",
    ),
    SearchPreset(
        "identity_guard_more_sensitive",
        "Allow identity swap guard to act on smaller gains.",
        {"identity_swap_min_gain": 0.005, "identity_swap_iom_threshold": 0.08},
        ("full",),
        family="occlusion_identity",
    ),
    SearchPreset(
        "no_identity_guard",
        "Ablate offline identity swap guard while keeping smoothing mode controls.",
        {"identity_swap_guard": False},
        ("balanced", "full"),
        family="occlusion_identity",
    ),
    SearchPreset(
        "occlusion_lower_penalty",
        "Reduce owner-preservation pressure during occlusion matching.",
        {"occlusion_switch_penalty": 0.30, "occlusion_appearance_penalty": 0.20},
        ("balanced", "full"),
        family="occlusion_identity",
    ),
    SearchPreset(
        "occlusion_higher_penalty",
        "Increase owner-preservation pressure during occlusion matching.",
        {"occlusion_switch_penalty": 0.60, "occlusion_appearance_penalty": 0.40},
        ("balanced", "full"),
        family="occlusion_identity",
    ),
    SearchPreset(
        "hold_occlusion_longer",
        "Hold occluded boxes longer before treating the track as lost.",
        {"occlusion_hold_max_frames": 45, "occlusion_hold_hidden_frames": 3},
        ("full",),
        family="occlusion_identity",
    ),
    SearchPreset(
        "hidden_motion_looser",
        "Allow hidden motion prediction to move farther after occlusion.",
        {"hidden_max_motion_step_box_scale": 2.00, "hidden_moving_displacement": 0.050},
        ("balanced", "full"),
        family="occlusion_identity",
    ),
    SearchPreset(
        "hidden_motion_tighter",
        "Constrain hidden motion prediction after occlusion.",
        {"hidden_max_motion_step_box_scale": 1.20, "hidden_moving_displacement": 0.025},
        ("balanced", "full"),
        family="occlusion_identity",
    ),
    SearchPreset(
        "smooth_conservative",
        "Increase temporal smoothing inertia for stable tracks.",
        {
            "high_conf_smooth_alpha": 0.85,
            "mid_conf_smooth_alpha": 0.65,
            "low_conf_smooth_alpha": 0.45,
        },
        ("balanced", "full"),
        family="smoothing",
    ),
    SearchPreset(
        "smooth_responsive",
        "Make box smoothing respond faster to new detections.",
        {
            "high_conf_smooth_alpha": 0.65,
            "mid_conf_smooth_alpha": 0.45,
            "low_conf_smooth_alpha": 0.25,
        },
        ("balanced", "full"),
        family="smoothing",
    ),
    SearchPreset(
        "refine_wider_gap",
        "Allow post-refinement to bridge longer short gaps.",
        {"refine_max_gap_frames": 24, "refine_size_jump_threshold": 0.55},
        ("full",),
        family="smoothing",
    ),
)

RULE_COMBOS: dict[str, RuleCombo] = {
    "iou0_area0_condarea0_merge0": RuleCombo(
        "iou0_area0_condarea0_merge0",
        {
            "USE_IOU_FALLBACK": False,
            "USE_AREA_OCCLUSION_FREEZE": False,
            "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": False,
            "USE_MERGED_BOX_SPLIT": False,
        },
    ),
    "iou1_area0_condarea0_merge0": RuleCombo(
        "iou1_area0_condarea0_merge0",
        {
            "USE_IOU_FALLBACK": True,
            "USE_AREA_OCCLUSION_FREEZE": False,
            "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": False,
            "USE_MERGED_BOX_SPLIT": False,
        },
    ),
}

SMOOTH_MODES: dict[str, SmoothMode] = {
    "nosmooth": SmoothMode(
        "nosmooth",
        {
            "enable_offline_smoothing": False,
            "identity_swap_guard": False,
            "smooth_boxes": False,
            "refine_boxes": False,
        },
    ),
    "smooth": SmoothMode(
        "smooth",
        {
            "enable_offline_smoothing": True,
            "identity_swap_guard": True,
            "smooth_boxes": True,
            "refine_boxes": True,
        },
    ),
}

RANK_COLUMNS: dict[str, tuple[str, ...]] = {
    "balanced": (
        "selection_score",
        "is_pareto_optimal",
        "target_total_idsw",
        "target_min_hota_pct",
        "remapped_hota_pct",
        "remapped_idf1_pct",
        "worst_video_hota_pct",
        "remapped_mota_pct",
        "remapped_idsw",
        "fn",
        "fp",
        "fragments",
    ),
    "identity": (
        "is_pareto_optimal",
        "target_total_idsw",
        "remapped_idsw",
        "max_video_idsw",
        "target_min_hota_pct",
        "remapped_idf1_pct",
        "remapped_hota_pct",
        "fragments",
        "fn",
        "fp",
    ),
    "recall": (
        "is_pareto_optimal",
        "fn",
        "max_video_fn",
        "target_total_idsw",
        "remapped_hota_pct",
        "remapped_idf1_pct",
        "remapped_idsw",
        "fp",
        "fragments",
    ),
}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/optimize_tracking_metrics.py -a\n"
            "  python scripts/optimize_tracking_metrics.py -a --scope full\n"
            "  python scripts/optimize_tracking_metrics.py -v Pigs291119_000263_30fps --scope quick\n"
            "  python scripts/optimize_tracking_metrics.py -a --rule-scope iou --scope full\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-v", "--video", type=str, default=None, help="Comma-separated names, paths, keys, or aliases.")
    group.add_argument("-a", "--all-videos", action="store_true", help="Run search on all configured videos.")
    parser.add_argument("-p", "--profile", type=str, default=None, help="Path profile name.")
    parser.add_argument("--path-config", type=str, default=None, help="Custom tracking_paths.json path.")
    parser.add_argument(
        "--tracking-mode",
        "--mode",
        dest="tracking_mode",
        choices=["realtime", "bytetrack_raw", "hybrid_bytetrack"],
        default="hybrid_bytetrack",
        help="Tracking mode for generated predictions.",
    )
    parser.add_argument(
        "--scope",
        choices=["quick", "balanced", "full", "detector_probe"],
        default="balanced",
        help="Preset breadth. Use full when leaving the machine overnight.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        default=None,
        help="Run only selected preset name. Can be repeated.",
    )
    parser.add_argument(
        "--rule-scope",
        choices=["baseline", "iou"],
        default="baseline",
        help="Rule combo to test: baseline=iou0_area0_condarea0_merge0, iou=iou1_area0_condarea0_merge0.",
    )
    parser.add_argument(
        "--smooth-mode",
        choices=["nosmooth", "smooth", "both"],
        default="both",
        help="Smoothing variants to test.",
    )
    smooth_group = parser.add_mutually_exclusive_group()
    smooth_group.add_argument("--smooth", dest="smooth_override", action="store_true", default=None)
    smooth_group.add_argument("--no-smooth", dest="smooth_override", action="store_false", default=None)
    parser.add_argument(
        "--rank-by",
        choices=sorted(RANK_COLUMNS),
        default="balanced",
        help="Final ranking priority.",
    )
    parser.add_argument(
        "--target-video",
        action="append",
        default=None,
        help=(
            "Video stem(s) to surface in ranking diagnostics. Can be repeated "
            "or comma-separated. Defaults to the known focus videos."
        ),
    )
    parser.add_argument("--run-name", type=str, default=None, help="Stable output folder name for resume.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without tracking.")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_known_args(argv)


def contains_arg(args: list[str], *names: str) -> bool:
    for arg in args:
        for name in names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def run_pipeline_worker(config):
    from pig_behavior.evaluation.tracking.pipeline import run_pipeline  # noqa: PLC0415

    _, metrics_df, _ = run_pipeline(config)
    return metrics_df


def run_pipeline_isolated(config):
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        return pool.apply(run_pipeline_worker, (config,))


def gt_backed_profile_videos(args: argparse.Namespace) -> str:
    from pig_behavior.evaluation.tracking.pipeline import find_gt_xml_for_video  # noqa: PLC0415
    from pig_behavior.tracking_path_config import (  # noqa: PLC0415
        load_tracking_path_profile,
        profile_path,
        profile_video_paths,
    )

    profile = load_tracking_path_profile(
        Path(args.path_config) if args.path_config else None,
        args.profile,
    )
    gt_dir = profile_path(
        profile,
        "gt_dir",
        PROJECT_ROOT / "data" / "annotations" / "tracking",
    ) or PROJECT_ROOT / "data" / "annotations" / "tracking"
    valid_videos = [
        video_path
        for video_path in profile_video_paths(profile)
        if find_gt_xml_for_video(video_path, gt_dir) is not None
    ]
    if not valid_videos:
        raise SystemExit("No configured videos have matching GT XML.")
    return ",".join(str(video_path) for video_path in valid_videos)


def build_pipeline_argv(args: argparse.Namespace, extras: list[str], run_name: str) -> list[str]:
    pipeline_argv: list[str] = []
    if args.all_videos:
        pipeline_argv.extend(["--video", gt_backed_profile_videos(args)])
    elif args.video:
        pipeline_argv.extend(["--video", args.video])
    else:
        pipeline_argv.extend(["--video", gt_backed_profile_videos(args)])

    if args.profile:
        pipeline_argv.extend(["--profile", args.profile])
    if args.path_config:
        pipeline_argv.extend(["--path-config", args.path_config])
    pipeline_argv.extend(["--tracking-mode", args.tracking_mode])
    pipeline_argv.extend(["--force-track", "--no-benchmark-rules"])

    if not contains_arg(extras, "--prediction-root"):
        default_prediction_root = PROJECT_ROOT / "outputs" / "pred" / args.tracking_mode / run_name / "optimizer"
        pipeline_argv.extend(["--prediction-root", str(default_prediction_root)])
    if not contains_arg(extras, "--output-root"):
        default_output_root = PROJECT_ROOT / "outputs" / "eval" / args.tracking_mode / run_name / "optimizer"
        pipeline_argv.extend(["--output-root", str(default_output_root)])

    pipeline_argv.extend(extras)
    return pipeline_argv


def selected_presets(scope: str, names: list[str] | None) -> list[SearchPreset]:
    available = {preset.name: preset for preset in PRESETS}
    if names:
        missing = sorted(set(names) - set(available))
        if missing:
            raise SystemExit(f"Unknown preset(s): {', '.join(missing)}")
        return [available[name] for name in names]
    return [preset for preset in PRESETS if scope in preset.scopes]


def selected_rule_combos(rule_scope: str) -> list[RuleCombo]:
    if rule_scope == "baseline":
        return [RULE_COMBOS["iou0_area0_condarea0_merge0"]]
    if rule_scope == "iou":
        return [RULE_COMBOS["iou1_area0_condarea0_merge0"]]
    raise ValueError(f"Unknown rule scope: {rule_scope}")


def selected_smooth_modes(args: argparse.Namespace) -> list[SmoothMode]:
    if args.smooth_override is True:
        return [SMOOTH_MODES["smooth"]]
    if args.smooth_override is False:
        return [SMOOTH_MODES["nosmooth"]]
    if args.smooth_mode == "both":
        return [SMOOTH_MODES["nosmooth"], SMOOTH_MODES["smooth"]]
    return [SMOOTH_MODES[args.smooth_mode]]


def discover_default_target_video_stems(limit: int = 4) -> tuple[str, ...]:
    metrics_path = DEFAULT_TARGET_VIDEO_SOURCE
    if not metrics_path.exists():
        return DEFAULT_TARGET_VIDEO_STEMS
    import csv

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("video_stem") != "ALL"]
    if not rows:
        return DEFAULT_TARGET_VIDEO_STEMS

    def sort_key(row: dict[str, str]) -> tuple[float, float, float, float, float]:
        return (
            float(row.get("remapped_hota_pct", 0.0)),
            float(row.get("remapped_idf1_pct", 0.0)),
            -float(row.get("remapped_idsw", 0.0)),
            -float(row.get("fn", 0.0)),
            -float(row.get("fp", 0.0)),
        )

    weakest = sorted(rows, key=sort_key)[: max(1, limit)]
    discovered = tuple(row["video_stem"] for row in weakest if row.get("video_stem"))
    return discovered or DEFAULT_TARGET_VIDEO_STEMS


def selected_target_video_stems(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.target_video:
        return discover_default_target_video_stems()
    stems: list[str] = []
    for value in args.target_video:
        stems.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(stems))


def candidate_name(rule: RuleCombo, smooth: SmoothMode, preset: SearchPreset) -> str:
    return f"{rule.name}__{smooth.name}__{preset.name}"


def aggregate_row(metrics_df: Any) -> Any:
    if metrics_df.empty:
        raise RuntimeError("tracking_metrics.csv is empty")
    aggregate = metrics_df[metrics_df["video_stem"] == "ALL"]
    if not aggregate.empty:
        return aggregate.iloc[0]
    return metrics_df.iloc[-1]


def read_existing_metrics(run_dir: Path) -> Any | None:
    metrics_path = run_dir / "tracking_metrics.csv"
    if not metrics_path.exists():
        return None
    import pandas as pd

    metrics = pd.read_csv(metrics_path)
    if "evaluator_contract_id" in metrics.columns:
        raise RuntimeError(
            "Optimizer objectives are Legacy V1 historical fields and cannot "
            "consume Standard V2 without a separately authorized migration."
        )
    return metrics


def metric_value(row: Any, name: str, default: float = 0.0) -> float:
    if name not in row:
        return default
    value = row[name]
    try:
        if value != value:
            return default
    except TypeError:
        return default
    return float(value)


def short_video_key(video_stem: str) -> str:
    match = re.search(r"_(\d{6})_", video_stem)
    if match:
        return match.group(1)
    return re.sub(r"[^0-9A-Za-z]+", "_", video_stem).strip("_").lower()


def per_video_stability(
    metrics_df: Any,
    *,
    target_video_stems: tuple[str, ...] = (),
) -> dict[str, object]:
    """Summarize worst-case behavior across videos, excluding the aggregate row."""
    if "video_stem" not in metrics_df.columns:
        return {}
    per_video = metrics_df[metrics_df["video_stem"] != "ALL"].copy()
    if per_video.empty:
        return {}

    numeric_columns = [
        "remapped_hota_pct",
        "remapped_idf1_pct",
        "remapped_idsw",
        "fn",
        "fp",
        "fragments",
        "remapped_fragments",
    ]
    for column in numeric_columns:
        if column in per_video.columns:
            per_video[column] = per_video[column].astype(float)

    worst_hota_idx = (
        per_video["remapped_hota_pct"].idxmin()
        if "remapped_hota_pct" in per_video.columns
        else per_video.index[0]
    )
    worst_hota_row = per_video.loc[worst_hota_idx]
    summary: dict[str, object] = {
        "video_count": int(len(per_video)),
        "worst_video_stem": str(worst_hota_row.get("video_stem", "")),
        "worst_video_hota_pct": metric_value(worst_hota_row, "remapped_hota_pct"),
        "worst_video_idf1_pct": metric_value(worst_hota_row, "remapped_idf1_pct"),
        "max_video_idsw": int(per_video["remapped_idsw"].max()) if "remapped_idsw" in per_video.columns else 0,
        "max_video_fn": int(per_video["fn"].max()) if "fn" in per_video.columns else 0,
        "max_video_fp": int(per_video["fp"].max()) if "fp" in per_video.columns else 0,
        "max_video_fragments": (
            int(per_video["fragments"].max()) if "fragments" in per_video.columns else 0
        ),
        "hota_std": (
            round(float(per_video["remapped_hota_pct"].std(ddof=0)), 4)
            if "remapped_hota_pct" in per_video.columns
            else 0.0
        ),
        "idf1_std": (
            round(float(per_video["remapped_idf1_pct"].std(ddof=0)), 4)
            if "remapped_idf1_pct" in per_video.columns
            else 0.0
        ),
        "videos_with_idsw": (
            int((per_video["remapped_idsw"] > 0).sum())
            if "remapped_idsw" in per_video.columns
            else 0
        ),
    }
    target_rows = per_video[per_video["video_stem"].isin(target_video_stems)]
    if not target_rows.empty:
        summary["target_video_count"] = int(len(target_rows))
        summary["target_total_idsw"] = int(target_rows["remapped_idsw"].sum()) if "remapped_idsw" in target_rows else 0
        summary["target_max_idsw"] = int(target_rows["remapped_idsw"].max()) if "remapped_idsw" in target_rows else 0
        summary["target_min_hota_pct"] = (
            metric_value(target_rows.loc[target_rows["remapped_hota_pct"].idxmin()], "remapped_hota_pct")
            if "remapped_hota_pct" in target_rows
            else 0.0
        )
        summary["target_min_idf1_pct"] = (
            metric_value(target_rows.loc[target_rows["remapped_idf1_pct"].idxmin()], "remapped_idf1_pct")
            if "remapped_idf1_pct" in target_rows
            else 0.0
        )
    else:
        summary["target_video_count"] = 0
        summary["target_total_idsw"] = 0
        summary["target_max_idsw"] = 0
        summary["target_min_hota_pct"] = 0.0
        summary["target_min_idf1_pct"] = 0.0

    for stem in target_video_stems:
        rows = per_video[per_video["video_stem"] == stem]
        key = short_video_key(stem)
        if rows.empty:
            summary[f"target_{key}_present"] = False
            continue
        row = rows.iloc[0]
        summary[f"target_{key}_present"] = True
        summary[f"target_{key}_idsw"] = int(metric_value(row, "remapped_idsw"))
        summary[f"target_{key}_hota_pct"] = metric_value(row, "remapped_hota_pct")
        summary[f"target_{key}_idf1_pct"] = metric_value(row, "remapped_idf1_pct")
        summary[f"target_{key}_fn"] = int(metric_value(row, "fn"))
        summary[f"target_{key}_fp"] = int(metric_value(row, "fp"))
        summary[f"target_{key}_fragments"] = int(metric_value(row, "fragments"))
    return summary


def summary_from_metrics(
    metrics_df: Any,
    *,
    name: str,
    rule: RuleCombo,
    smooth: SmoothMode,
    preset: SearchPreset,
    run_dir: Path,
    elapsed_sec: float,
    status: str,
    target_video_stems: tuple[str, ...],
    error: str = "",
) -> dict[str, object]:
    row = aggregate_row(metrics_df)
    evaluated_frames = metric_value(row, "evaluated_frames")
    fps = evaluated_frames / elapsed_sec if elapsed_sec > 0 else 0.0
    return {
        "status": status,
        "candidate": name,
        "rule_combo": rule.name,
        "smooth_mode": smooth.name,
        "preset": preset.name,
        "preset_family": preset.family,
        "description": preset.description,
        "elapsed_sec": round(elapsed_sec, 4),
        "fps_evaluated_frames": round(fps, 4),
        "run_dir": str(run_dir),
        "error": error,
        "USE_IOU_FALLBACK": rule.flags["USE_IOU_FALLBACK"],
        "USE_AREA_OCCLUSION_FREEZE": rule.flags["USE_AREA_OCCLUSION_FREEZE"],
        "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": rule.flags["USE_CONDITIONAL_AREA_OCCLUSION_FREEZE"],
        "USE_MERGED_BOX_SPLIT": rule.flags["USE_MERGED_BOX_SPLIT"],
        "overrides_json": json.dumps({**smooth.overrides, **preset.overrides}, sort_keys=True),
        "fp": int(metric_value(row, "fp")),
        "fn": int(metric_value(row, "fn")),
        "remapped_idsw": int(metric_value(row, "remapped_idsw")),
        "fragments": int(metric_value(row, "fragments")),
        "remapped_fragments": int(metric_value(row, "remapped_fragments", metric_value(row, "fragments"))),
        "remapped_gap_tolerant_fragments": int(metric_value(row, "remapped_gap_tolerant_fragments")),
        "remapped_mota_pct": metric_value(row, "remapped_mota_pct"),
        "remapped_idf1_pct": metric_value(row, "remapped_idf1_pct"),
        "remapped_hota_pct": metric_value(row, "remapped_hota_pct"),
        **per_video_stability(metrics_df, target_video_stems=target_video_stems),
    }


def failure_summary(
    *,
    name: str,
    rule: RuleCombo,
    smooth: SmoothMode,
    preset: SearchPreset,
    run_dir: Path,
    elapsed_sec: float,
    error: str,
    target_video_stems: tuple[str, ...],
) -> dict[str, object]:
    row = {
        "status": "failed",
        "candidate": name,
        "rule_combo": rule.name,
        "smooth_mode": smooth.name,
        "preset": preset.name,
        "preset_family": preset.family,
        "description": preset.description,
        "elapsed_sec": round(elapsed_sec, 4),
        "fps_evaluated_frames": 0.0,
        "run_dir": str(run_dir),
        "error": error,
        "USE_IOU_FALLBACK": rule.flags["USE_IOU_FALLBACK"],
        "USE_AREA_OCCLUSION_FREEZE": rule.flags["USE_AREA_OCCLUSION_FREEZE"],
        "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": rule.flags["USE_CONDITIONAL_AREA_OCCLUSION_FREEZE"],
        "USE_MERGED_BOX_SPLIT": rule.flags["USE_MERGED_BOX_SPLIT"],
        "overrides_json": json.dumps({**smooth.overrides, **preset.overrides}, sort_keys=True),
        "target_video_count": 0,
        "target_total_idsw": None,
        "target_max_idsw": None,
        "target_min_hota_pct": None,
        "target_min_idf1_pct": None,
    }
    for key in (
        "fp",
        "fn",
        "remapped_idsw",
        "fragments",
        "remapped_fragments",
        "remapped_gap_tolerant_fragments",
        "remapped_mota_pct",
        "remapped_idf1_pct",
        "remapped_hota_pct",
    ):
        row[key] = None
    for stem in target_video_stems:
        key = short_video_key(stem)
        row[f"target_{key}_present"] = False
        row[f"target_{key}_idsw"] = None
        row[f"target_{key}_hota_pct"] = None
        row[f"target_{key}_idf1_pct"] = None
        row[f"target_{key}_fn"] = None
        row[f"target_{key}_fp"] = None
        row[f"target_{key}_fragments"] = None
    return row


def rank_summary(summary_df: Any, rank_by: str) -> Any:
    ranked = summary_df[summary_df["status"].isin(["completed", "resumed"])].copy()
    if ranked.empty:
        return ranked
    columns = [column for column in RANK_COLUMNS[rank_by] if column in ranked.columns]
    lower_is_better = {
        "remapped_idsw",
        "fn",
        "fp",
        "fragments",
        "remapped_fragments",
        "remapped_gap_tolerant_fragments",
        "target_total_idsw",
    }
    ascending = [column in lower_is_better for column in columns]
    return ranked.sort_values(columns, ascending=ascending, kind="stable").reset_index(drop=True)


def add_baseline_deltas(summary_df: Any) -> Any:
    df = summary_df.copy()
    successful = df["status"].isin(["completed", "resumed"])
    metric_columns = [
        "fp",
        "fn",
        "remapped_idsw",
        "fragments",
        "remapped_fragments",
        "remapped_gap_tolerant_fragments",
        "remapped_mota_pct",
        "remapped_idf1_pct",
        "remapped_hota_pct",
        "worst_video_hota_pct",
        "max_video_idsw",
        "max_video_fn",
        "hota_std",
        "target_total_idsw",
        "target_max_idsw",
        "target_min_hota_pct",
        "target_min_idf1_pct",
    ]
    for column in metric_columns:
        if column in df.columns:
            df[column] = df[column].astype(float)
            df[f"delta_{column}"] = None

    baselines = df[successful & (df["preset"] == "base")]
    for _, baseline in baselines.iterrows():
        mask = (
            successful
            & (df["rule_combo"] == baseline["rule_combo"])
            & (df["smooth_mode"] == baseline["smooth_mode"])
        )
        for column in metric_columns:
            if column in df.columns:
                df.loc[mask, f"delta_{column}"] = df.loc[mask, column] - float(baseline[column])
    return df


def add_pareto_and_score(summary_df: Any) -> Any:
    df = summary_df.copy()
    successful = df["status"].isin(["completed", "resumed"])
    df["is_pareto_optimal"] = False
    df["selection_score"] = None
    if not successful.any():
        return df

    objective_columns = [
        "remapped_hota_pct",
        "remapped_idf1_pct",
        "worst_video_hota_pct",
        "remapped_idsw",
        "fn",
        "fp",
        "max_video_idsw",
        "hota_std",
        "target_total_idsw",
        "target_min_hota_pct",
    ]
    for column in objective_columns:
        if column in df.columns:
            df[column] = df[column].astype(float)

    subset = df[successful].copy()
    maximize = {
        "remapped_hota_pct",
        "remapped_idf1_pct",
        "worst_video_hota_pct",
        "target_min_hota_pct",
    }
    pareto_flags: dict[int, bool] = {}
    for idx, row in subset.iterrows():
        dominated = False
        for other_idx, other in subset.iterrows():
            if idx == other_idx:
                continue
            no_worse = True
            strictly_better = False
            for column in objective_columns:
                if column not in subset.columns:
                    continue
                row_value = float(row[column])
                other_value = float(other[column])
                if column in maximize:
                    no_worse = no_worse and other_value >= row_value
                    strictly_better = strictly_better or other_value > row_value
                else:
                    no_worse = no_worse and other_value <= row_value
                    strictly_better = strictly_better or other_value < row_value
            if no_worse and strictly_better:
                dominated = True
                break
        pareto_flags[idx] = not dominated
    for idx, is_pareto in pareto_flags.items():
        df.loc[idx, "is_pareto_optimal"] = is_pareto

    def normalized(column: str, higher_is_better: bool) -> Any:
        values = subset[column].astype(float)
        low = values.min()
        high = values.max()
        if high == low:
            return values * 0.0 + 1.0
        scaled = (values - low) / (high - low)
        return scaled if higher_is_better else 1.0 - scaled

    score = (
        0.22 * normalized("remapped_hota_pct", True)
        + 0.18 * normalized("remapped_idf1_pct", True)
        + 0.12 * normalized("worst_video_hota_pct", True)
        + 0.12 * normalized("remapped_idsw", False)
        + 0.12 * normalized("target_total_idsw", False)
        + 0.08 * normalized("target_min_hota_pct", True)
        + 0.08 * normalized("fn", False)
        + 0.04 * normalized("fp", False)
        + 0.02 * normalized("max_video_idsw", False)
        + 0.02 * normalized("hota_std", False)
    )
    df.loc[subset.index, "selection_score"] = score.round(6)
    return df


def enrich_summary(summary_df: Any) -> Any:
    return add_pareto_and_score(add_baseline_deltas(summary_df))


def simple_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [str(row.get(column, "")).replace("\n", " ") for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def write_outputs(
    *,
    output_root: Path,
    summary_rows: list[dict[str, object]],
    detailed_frames: list[Any],
    rank_by: str,
    top_k: int,
    manifest: dict[str, object],
) -> tuple[Path, Path, Path]:
    import pandas as pd

    output_root.mkdir(parents=True, exist_ok=True)
    summary_df = enrich_summary(pd.DataFrame(summary_rows))
    ranked_df = rank_summary(summary_df, rank_by)
    detailed_df = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()
    successful_df = summary_df[summary_df["status"].isin(["completed", "resumed"])].copy()

    summary_path = output_root / "tracking_optimizer_summary.csv"
    ranked_path = output_root / "tracking_optimizer_ranked.csv"
    detailed_path = output_root / "tracking_optimizer_detailed_metrics.csv"
    report_path = output_root / "tracking_optimizer_report.md"
    manifest_path = output_root / "tracking_optimizer_manifest.json"

    summary_df.to_csv(summary_path, index=False)
    ranked_df.to_csv(ranked_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    top_rows = ranked_df.head(top_k).to_dict("records") if not ranked_df.empty else []
    baseline_rows = (
        rank_summary(successful_df[successful_df["preset"] == "base"], rank_by).to_dict("records")
        if not successful_df.empty
        else []
    )
    family_rows = []
    if not successful_df.empty and "preset_family" in successful_df.columns:
        for family, family_df in successful_df.groupby("preset_family", sort=True):
            ranked_family = rank_summary(family_df, rank_by)
            if ranked_family.empty:
                continue
            best = ranked_family.iloc[0].to_dict()
            best["preset_family"] = family
            family_rows.append(best)

    report = [
        "# Tracking Optimizer Report",
        "",
        f"- Output folder: `{output_root}`",
        f"- Ranking mode: `{rank_by}`",
        f"- Completed/resumed runs: `{len(ranked_df)}`",
        f"- Failed runs: `{int((summary_df['status'] == 'failed').sum()) if not summary_df.empty else 0}`",
        "",
        "## Baseline Diagnostics",
        "",
        simple_markdown_table(
            baseline_rows,
            [
                "candidate",
                "smooth_mode",
                "fn",
                "fp",
                "remapped_idsw",
                "target_total_idsw",
                "target_min_hota_pct",
                "remapped_idf1_pct",
                "remapped_hota_pct",
                "worst_video_stem",
                "worst_video_hota_pct",
            ],
        ),
        "",
        "## Best By Preset Family",
        "",
        simple_markdown_table(
            family_rows,
            [
                "preset_family",
                "candidate",
                "fn",
                "fp",
                "remapped_idsw",
                "target_total_idsw",
                "target_min_hota_pct",
                "selection_score",
                "run_dir",
            ],
        ),
        "",
        "## Top Results",
        "",
        simple_markdown_table(
            top_rows,
            [
                "candidate",
                "preset_family",
                "fn",
                "remapped_idsw",
                "target_total_idsw",
                "fp",
                "fragments",
                "remapped_idf1_pct",
                "remapped_hota_pct",
                "worst_video_hota_pct",
                "target_min_hota_pct",
                "max_video_idsw",
                "selection_score",
                "is_pareto_optimal",
                "run_dir",
            ],
        ),
        "",
        "## Files",
        "",
        "- `tracking_optimizer_summary.csv`: all candidate aggregate, stability, baseline-delta, and Pareto metrics.",
        "- `tracking_optimizer_ranked.csv`: successful candidates sorted by the selected scientific rank.",
        "- `tracking_optimizer_detailed_metrics.csv`: per-video and ALL rows for every run.",
        "- `tracking_optimizer_manifest.json`: full search plan and base pipeline config.",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    return summary_path, ranked_path, report_path


def print_plan(rules: list[RuleCombo], smooth_modes: list[SmoothMode], presets: list[SearchPreset]) -> None:
    total = len(rules) * len(smooth_modes) * len(presets)
    print(f"[plan] rule_combos={len(rules)} smooth_modes={len(smooth_modes)} presets={len(presets)} total_runs={total}")
    for rule in rules:
        for smooth in smooth_modes:
            for preset in presets:
                print(f" - {candidate_name(rule, smooth, preset)} [{preset.family}]")


def main(argv: list[str] | None = None) -> int:
    args, extras = parse_args(argv)
    if args.list_presets:
        for preset in PRESETS:
            print(f"{preset.name}: {preset.description} scopes={','.join(preset.scopes)}")
        return 0

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    rules = selected_rule_combos(args.rule_scope)
    smooth_modes = selected_smooth_modes(args)
    presets = selected_presets(args.scope, args.preset)
    target_video_stems = selected_target_video_stems(args)
    if args.dry_run:
        print_plan(rules, smooth_modes, presets)
        return 0

    from pig_behavior.evaluation.tracking.cli import (  # noqa: PLC0415
        config_from_args,
    )
    from pig_behavior.evaluation.tracking.cli import (
        parse_args as parse_pipeline_args,
    )

    pipeline_argv = build_pipeline_argv(args, extras, run_name)
    base_config = config_from_args(parse_pipeline_args(pipeline_argv))
    optimizer_output_root = base_config.output_root
    optimizer_prediction_root = base_config.prediction_root

    summary_rows: list[dict[str, object]] = []
    detailed_frames: list[Any] = []
    total_runs = len(rules) * len(smooth_modes) * len(presets)
    manifest = {
        "run_name": run_name,
        "pipeline_argv": pipeline_argv,
        "base_config": asdict(base_config),
        "scope": args.scope,
        "rule_scope": args.rule_scope,
        "smooth_mode": args.smooth_mode,
        "target_video_stems": target_video_stems,
        "rank_by": args.rank_by,
        "resume": args.resume,
        "planned_runs": total_runs,
        "rules": [asdict(rule) for rule in rules],
        "smooth_modes": [asdict(mode) for mode in smooth_modes],
        "presets": [asdict(preset) for preset in presets],
    }

    print_plan(rules, smooth_modes, presets)
    run_index = 0
    for rule in rules:
        for smooth in smooth_modes:
            for preset in presets:
                run_index += 1
                name = candidate_name(rule, smooth, preset)
                run_dir = optimizer_output_root / name
                pred_dir = optimizer_prediction_root / name
                print(f"[{run_index}/{total_runs}] {name}")
                started = time.perf_counter()
                try:
                    existing_df = read_existing_metrics(run_dir) if args.resume else None
                    if existing_df is not None:
                        metrics_df = existing_df
                        status = "resumed"
                        elapsed = 0.0
                    else:
                        profile_overrides = dict(base_config.profile_overrides or {})
                        profile_overrides.update(smooth.overrides)
                        profile_overrides.update(preset.overrides)
                        run_config = replace(
                            base_config,
                            prediction_root=pred_dir,
                            output_root=run_dir,
                            run_missing_tracker=True,
                            force_track=True,
                            profile_overrides=profile_overrides,
                            **rule.flags,
                        )
                        metrics_df = run_pipeline_isolated(run_config)
                        elapsed = time.perf_counter() - started
                        status = "completed"

                    summary_rows.append(
                        summary_from_metrics(
                            metrics_df,
                            name=name,
                            rule=rule,
                            smooth=smooth,
                            preset=preset,
                            run_dir=run_dir,
                            elapsed_sec=elapsed,
                            status=status,
                            target_video_stems=target_video_stems,
                        )
                    )
                    tagged_metrics = metrics_df.copy()
                    tagged_metrics.insert(0, "candidate", name)
                    tagged_metrics.insert(1, "rule_combo", rule.name)
                    tagged_metrics.insert(2, "smooth_mode", smooth.name)
                    tagged_metrics.insert(3, "preset", preset.name)
                    detailed_frames.append(tagged_metrics)
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.perf_counter() - started
                    summary_rows.append(
                        failure_summary(
                            name=name,
                            rule=rule,
                            smooth=smooth,
                            preset=preset,
                            run_dir=run_dir,
                            elapsed_sec=elapsed,
                            error=f"{type(exc).__name__}: {exc}",
                            target_video_stems=target_video_stems,
                        )
                    )
                    print(f"[failed] {name}: {type(exc).__name__}: {exc}")
                    if args.fail_fast:
                        raise
                finally:
                    write_outputs(
                        output_root=optimizer_output_root,
                        summary_rows=summary_rows,
                        detailed_frames=detailed_frames,
                        rank_by=args.rank_by,
                        top_k=args.top_k,
                        manifest=manifest,
                    )

    _, ranked_path, report_path = write_outputs(
        output_root=optimizer_output_root,
        summary_rows=summary_rows,
        detailed_frames=detailed_frames,
        rank_by=args.rank_by,
        top_k=args.top_k,
        manifest=manifest,
    )
    print(f"\n[done] optimizer output: {optimizer_output_root}")
    print(f"[done] ranked csv: {ranked_path}")
    print(f"[done] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

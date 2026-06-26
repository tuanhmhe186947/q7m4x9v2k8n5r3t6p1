#!/usr/bin/env python3
"""Benchmark detector weights by end-to-end tracking quality on GT videos."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pig_behavior.evaluation.tracking.config import TrackingEvaluationPipelineConfig  # noqa: E402
from pig_behavior.evaluation.tracking.pipeline import run_pipeline  # noqa: E402


DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "models/detector/pig_detector_yolov8_roboflow_2.pt",
    PROJECT_ROOT / "models/detector/pig_detector_yolov8_roboflow.pt",
    PROJECT_ROOT / "models/detector/pig_detector_yolov8.pt",
)
DEFAULT_GT_DIR = PROJECT_ROOT / "data/annotations/tracking"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data/videos"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/evaluation/weight_tracking_gt_benchmark"
DEFAULT_PREDICTION_ROOT = PROJECT_ROOT / "outputs/id_tracking/weight_tracking_gt_benchmark"
SUPPORTED_GT_SUFFIXES = (".xml",)
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
TRACKING_MODES = ("realtime", "bytetrack_raw", "hybrid_bytetrack", "bytetrack", "gt_export")
RANKING_COLUMNS = (
    "hota",
    "idf1",
    "mota",
    "assa",
    "precision",
    "recall",
    "idsw",
    "fragments",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current tracking pipeline for multiple YOLO detector weights "
            "and rank them using GT tracking metrics."
        )
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help=f"Directory containing GT tracking annotations (default: {DEFAULT_GT_DIR})",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help=f"Directory containing videos (default: {DEFAULT_VIDEO_DIR})",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        default=list(DEFAULT_WEIGHTS),
        help="Detector weights to benchmark.",
    )
    parser.add_argument(
        "--mode",
        choices=TRACKING_MODES,
        default="hybrid_bytetrack",
        help="Tracking mode to use for every weight/video run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for evaluation reports.",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=DEFAULT_PREDICTION_ROOT,
        help="Root directory for generated tracking predictions.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit forwarded to tracking/evaluation.",
    )
    parser.add_argument("--device", default="auto", help="Inference device forwarded to tracking.")
    parser.add_argument("--half", action="store_true", help="Use half precision where supported.")
    parser.add_argument(
        "--force-track",
        action="store_true",
        help="Regenerate tracking predictions even if outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered GT/video pairs and benchmark matrix.",
    )
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def normalized_gt_stem(gt_path: Path) -> str:
    stem = gt_path.stem
    for prefix in ("Tracking_annotation_", "tracking_annotation_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"


def discover_gt_files(gt_dir: Path) -> list[Path]:
    gt_files: list[Path] = []
    for suffix in SUPPORTED_GT_SUFFIXES:
        gt_files.extend(gt_dir.glob(f"*{suffix}"))
    return sorted(set(gt_files), key=lambda p: p.name.lower())


def match_videos(gt_files: list[Path], video_dir: Path) -> tuple[list[Path], list[Path]]:
    matched_videos: list[Path] = []
    skipped_gt: list[Path] = []

    for gt_file in gt_files:
        gt_stem = normalized_gt_stem(gt_file)
        video_path = next(
            (video_dir / f"{gt_stem}{suffix}" for suffix in VIDEO_SUFFIXES if (video_dir / f"{gt_stem}{suffix}").exists()),
            None,
        )
        if video_path is None:
            skipped_gt.append(gt_file)
            print(f"[WARN] No matching video for GT: {gt_file}", file=sys.stderr)
            continue
        matched_videos.append(video_path)

    return matched_videos, skipped_gt


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def load_aggregate_row(metrics_csv: Path) -> dict[str, str]:
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No metrics rows found in {metrics_csv}")
    for row in rows:
        if row.get("video_stem") == "ALL":
            return row
    return rows[-1]


def score_row(row: dict[str, str]) -> float:
    """Tracking-focused score: identity/association metrics first, detections second."""
    hota = as_float(row, "hota", 0.0)
    idf1 = as_float(row, "idf1", 0.0)
    assa = as_float(row, "assa", 0.0)
    mota = as_float(row, "mota", 0.0)
    recall = as_float(row, "recall", 0.0)
    precision = as_float(row, "precision", 0.0)
    idsw = as_int(row, "idsw")
    fragments = as_int(row, "fragments")
    gt_detections = max(as_int(row, "gt_detections", 1), 1)
    identity_penalty = (idsw + 0.25 * fragments) / gt_detections
    return (
        0.35 * hota
        + 0.25 * idf1
        + 0.15 * assa
        + 0.15 * mota
        + 0.05 * recall
        + 0.05 * precision
        - identity_penalty
    )


def config_kwargs(**kwargs: Any) -> dict[str, Any]:
    valid_names = {field.name for field in fields(TrackingEvaluationPipelineConfig)}
    return {key: value for key, value in kwargs.items() if key in valid_names}


def run_weight_benchmark(
    *,
    weight_path: Path,
    video_paths: list[Path],
    gt_dir: Path,
    video_dir: Path,
    output_root: Path,
    prediction_root: Path,
    mode: str,
    max_frames: int | None,
    device: str,
    half: bool,
    force_track: bool,
) -> tuple[dict[str, str], Path]:
    weight_name = safe_name(weight_path.stem)
    weight_output_root = output_root / weight_name
    weight_prediction_root = prediction_root / weight_name

    config = TrackingEvaluationPipelineConfig(
        **config_kwargs(
            video_paths=video_paths,
            gt_dir=gt_dir,
            video_dir=video_dir,
            prediction_root=weight_prediction_root,
            output_root=weight_output_root,
            weights_path=weight_path,
            detector_name=weight_name,
            tracking_mode=mode,
            run_missing_tracker=True,
            force_track=force_track,
            max_frames=max_frames,
            device=device,
            half=half,
        )
    )
    _assets_df, _metrics_df, run_dir = run_pipeline(config)
    aggregate = load_aggregate_row(run_dir / "tracking_metrics.csv")
    return aggregate, run_dir


def write_summary(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "weight",
        "score",
        *RANKING_COLUMNS,
        "gt_detections",
        "pred_detections",
        "matches",
        "fp",
        "fn",
        "evaluated_frames",
        "metrics_csv",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> int:
    args = parse_args()
    gt_dir = repo_path(args.gt_dir)
    video_dir = repo_path(args.video_dir)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = repo_path(args.output_root) / run_name / args.mode
    prediction_root = repo_path(args.prediction_root) / run_name / args.mode
    weights = [repo_path(path) for path in args.weights]

    gt_files = discover_gt_files(gt_dir)
    video_paths, skipped_gt = match_videos(gt_files, video_dir)

    print(f"Found {len(gt_files)} GT files")
    print(f"Matched {len(video_paths)} videos")
    print(f"Skipped {len(skipped_gt)} GT files without videos")
    print(f"Mode: {args.mode}")
    print("Weights:")
    for weight_path in weights:
        print(f"  - {weight_path}")
    print("Videos:")
    for video_path in video_paths:
        print(f"  - {video_path}")

    missing_weights = [path for path in weights if not path.exists()]
    if missing_weights:
        for weight_path in missing_weights:
            print(f"[ERROR] Missing weight file: {weight_path}", file=sys.stderr)
        return 2
    if not video_paths:
        print("[ERROR] No matched GT/video pairs to benchmark.", file=sys.stderr)
        return 2
    if args.dry_run:
        print("[OK] Dry run complete.")
        return 0

    summary_rows: list[dict[str, Any]] = []
    for index, weight_path in enumerate(weights, start=1):
        print(f"\n[{index}/{len(weights)}] Benchmarking {weight_path.name}")
        aggregate, run_dir = run_weight_benchmark(
            weight_path=weight_path,
            video_paths=video_paths,
            gt_dir=gt_dir,
            video_dir=video_dir,
            output_root=output_root,
            prediction_root=prediction_root,
            mode=args.mode,
            max_frames=args.max_frames,
            device=args.device,
            half=args.half,
            force_track=args.force_track,
        )
        metrics_csv = run_dir / "tracking_metrics.csv"
        row: dict[str, Any] = {
            "rank": 0,
            "weight": str(weight_path),
            "score": score_row(aggregate),
            "metrics_csv": str(metrics_csv),
        }
        for column in RANKING_COLUMNS:
            row[column] = aggregate.get(column, "")
        for column in ("gt_detections", "pred_detections", "matches", "fp", "fn", "evaluated_frames"):
            row[column] = aggregate.get(column, "")
        summary_rows.append(row)

    summary_rows.sort(
        key=lambda row: (
            -float(row["score"]),
            as_int({k: str(v) for k, v in row.items()}, "idsw"),
            as_int({k: str(v) for k, v in row.items()}, "fragments"),
        )
    )
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank
        row["score"] = f"{float(row['score']):.6f}"

    summary_csv = output_root / "weight_benchmark_summary.csv"
    write_summary(summary_rows, summary_csv)

    print("\nWeight benchmark ranking:")
    for row in summary_rows:
        print(
            "  #{rank}: {weight_name} score={score} hota={hota} idf1={idf1} "
            "mota={mota} idsw={idsw}".format(
                rank=row["rank"],
                weight_name=Path(str(row["weight"])).name,
                score=row["score"],
                hota=row.get("hota", ""),
                idf1=row.get("idf1", ""),
                mota=row.get("mota", ""),
                idsw=row.get("idsw", ""),
            )
        )
    print(f"\n[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Prediction root: {prediction_root}")
    print(f"[OK] Evaluation root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

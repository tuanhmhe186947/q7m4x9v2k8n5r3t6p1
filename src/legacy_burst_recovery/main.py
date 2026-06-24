from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import RecoveryConfig, ensure_output_dirs
from .csv_loader import validate_legacy_dataframe
from .dense_tracklet_builder import build_dense_tracklets
from .depth_provenance import build_depth_provenance_audit
from .manifest_writer import write_manifests
from .path_utils import collect_path_resolution, write_path_resolution_report
from .qa_report import build_qa_summary, write_qa_reports
from .runtime import RuntimeReporter, write_progress_state
from .sequence_view_builder import build_sequence_views
from .timestamp_utils import (
    build_timestamp_audit,
    build_timestamp_parse_debug,
    diagnostic_times_preview,
    parse_times_txt,
)
from .video_utils import count_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover legacy burst CSV rows into dense training-ready tracklets.")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--drive-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--detector-weights", type=Path, default=None)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--extract-crops", action="store_true")
    parser.add_argument("--extract-full-frames", action="store_true")
    parser.add_argument("--track-end-mode", choices=["sample_0_6_12", "full_legacy_burst"], default="sample_0_6_12")
    parser.add_argument("--save-debug-visuals", action="store_true")
    parser.add_argument("--no-detect-manifest-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--filter-group-id", default=None)
    parser.add_argument("--filter-video", default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--debug-times", type=int, default=0)
    parser.add_argument(
        "--sequence-views",
        nargs="*",
        default=["sparse_3_0_6_12"],
        choices=[
            "sparse_3_0_6_12",
            "legacy_old_pattern_6",
            "dense_6_same_span",
            "dense_12_same_span",
            "full_dense_0_to_12",
        ],
    )
    return parser.parse_args()


def apply_input_filters(raw_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = raw_df
    if args.filter_group_id is not None:
        df = df[df["group_id"].astype(str).eq(args.filter_group_id)]
    if args.filter_video is not None:
        df = df[df["video_final"].astype(str).str.contains(args.filter_video, case=False, regex=False, na=False)]
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    return df.copy()


def write_partial_qa(
    config: RecoveryConfig,
    reporter: RuntimeReporter,
    raw_df: pd.DataFrame,
    accepted_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    path_report: pd.DataFrame,
    timestamp_audit: pd.DataFrame,
    depth_audit: pd.DataFrame,
    tracking_failures: pd.DataFrame,
) -> None:
    dense_path = config.output_root / "legacy_dense_tracklet_map.csv"
    dense_df = pd.read_csv(dense_path) if dense_path.exists() else pd.DataFrame()
    summary = build_qa_summary(
        raw_df=raw_df,
        accepted_df=accepted_df,
        rejected_df=rejected_df,
        dense_df=dense_df,
        path_report=path_report,
        timestamp_audit=timestamp_audit,
        depth_audit=depth_audit,
        tracking_failures=tracking_failures,
    )
    summary["partial_exit"] = True
    summary["timing_report"] = dict(reporter.stage_timings)
    summary["timing_report"]["total_sec"] = reporter.total_sec()
    write_qa_reports(config.output_root, summary)
    reporter.write_timing_report()
    reporter.log("WROTE partial QA before exit")


def main() -> None:
    args = parse_args()
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be a positive integer")
    config = RecoveryConfig(
        input_csv=args.input_csv,
        drive_root=args.drive_root,
        output_root=args.output_root,
        detector_weights=args.detector_weights,
        manifest_only=args.manifest_only,
        extract_crops=args.extract_crops,
        extract_full_frames=args.extract_full_frames,
        track_end_mode=args.track_end_mode,
        save_debug_visuals=args.save_debug_visuals,
        no_detect_manifest_only=args.no_detect_manifest_only,
        max_rows=args.max_rows,
        max_videos=args.max_videos,
        filter_group_id=args.filter_group_id,
        filter_video=args.filter_video,
        log_file=args.log_file,
        flush_every=args.flush_every,
        resume=args.resume,
        progress=args.progress,
    )
    ensure_output_dirs(config.output_root)
    reporter = RuntimeReporter(config.output_root, config.log_file)

    raw_df = pd.DataFrame()
    accepted_df = pd.DataFrame()
    rejected_df = pd.DataFrame()
    path_report = pd.DataFrame()
    timestamp_audit = pd.DataFrame()
    timestamp_parse_debug = pd.DataFrame()
    depth_audit = pd.DataFrame()
    tracking_failures = pd.DataFrame()

    try:
        with reporter.stage("csv_load"):
            raw_loaded_df = pd.read_csv(config.input_csv)
            reporter.log(f"loaded CSV shape={raw_loaded_df.shape}")
            reporter.log(f"loaded CSV columns={list(raw_loaded_df.columns)}")
            raw_df = apply_input_filters(raw_loaded_df, args)
            reporter.log(f"filtered CSV shape={raw_df.shape}")
            accepted_df, rejected_df = validate_legacy_dataframe(raw_df)
            reporter.log(f"valid rows={len(accepted_df)}")
            reporter.log(f"rejected rows={len(rejected_df)}")
            reporter.log(f"unique source videos={accepted_df['video_final'].nunique() if not accepted_df.empty else 0}")

        with reporter.stage("path_resolution"):
            resources_by_video, path_report = collect_path_resolution(
                accepted_df,
                config.drive_root,
                show_progress=config.progress,
                max_videos=config.max_videos,
            )
            if config.max_videos is not None:
                allowed_videos = set(resources_by_video)
                accepted_df = accepted_df[accepted_df["video_final"].astype(str).isin(allowed_videos)].copy()
                reporter.log(f"rows after --max-videos={len(accepted_df)}")
            write_path_resolution_report(path_report, config.output_root)
            reporter.log(f"WROTE {config.output_root / 'path_resolution_report.csv'}")

        with reporter.stage("timestamp_audit"):
            timestamps_by_video: dict[str, list[float]] = {}
            source_rows: list[dict[str, object]] = []
            iterator = tqdm(
                resources_by_video.items(),
                desc="Auditing timestamps/videos",
                disable=not config.progress,
            )
            for original, resources in iterator:
                parse_result = parse_times_txt(resources.times_txt_path)
                timestamps = parse_result.timestamps
                timestamps_by_video[original] = timestamps
                source_rows.append(
                    {
                        "source_video_original": original,
                        "source_video_resolved": resources.source_video_resolved,
                        "source_folder": resources.source_folder,
                        "times_txt_path": resources.times_txt_path,
                        "times_txt_exists": parse_result.exists,
                        "times_txt_parse_error": parse_result.parse_error,
                        "times_txt_first_lines_preview": parse_result.first_lines_preview,
                        "times_txt_file_size_bytes": parse_result.file_size_bytes,
                        "times_txt_detected_format": parse_result.detected_format,
                        "times_txt_first_datetime": parse_result.first_datetime,
                        "times_txt_last_datetime": parse_result.last_datetime,
                        "timestamps": timestamps,
                        "num_color_frames": count_video_frames(resources.source_video_resolved),
                    }
                )
            if args.debug_times > 0:
                reporter.log(f"Debugging times.txt parsing for first {args.debug_times} source videos")
                for idx, (_original, resources) in enumerate(resources_by_video.items()):
                    if idx >= args.debug_times:
                        break
                    diagnostic = diagnostic_times_preview(resources.times_txt_path)
                    reporter.log(f"TIMES DEBUG path={diagnostic['path']}")
                    reporter.log(f"TIMES DEBUG exists={diagnostic['exists']} num={diagnostic['num_timestamps']}")
                    reporter.log(f"TIMES DEBUG detected_format={diagnostic['detected_format']}")
                    reporter.log(f"TIMES DEBUG parse_error={diagnostic['parse_error']}")
                    reporter.log(f"TIMES DEBUG first_datetime={diagnostic['first_datetime']}")
                    reporter.log(f"TIMES DEBUG last_datetime={diagnostic['last_datetime']}")
                    reporter.log(f"TIMES DEBUG raw_first_lines={diagnostic['first_lines']}")
                    reporter.log(f"TIMES DEBUG parsed_examples={diagnostic['parsed_examples']}")
            timestamp_audit = build_timestamp_audit(source_rows)
            timestamp_parse_debug = build_timestamp_parse_debug(source_rows)

        with reporter.stage("dense_manifest"):
            dense_df, _crop_paths, _full_paths, failure_rows = build_dense_tracklets(
                accepted_df,
                resources_by_video,
                timestamps_by_video,
                config,
                sequence_views=args.sequence_views,
                reporter=reporter,
            )

        with reporter.stage("sequence_manifest"):
            sequence_df = build_sequence_views(dense_df, args.sequence_views, show_progress=config.progress)

        with reporter.stage("depth_provenance"):
            depth_audit = build_depth_provenance_audit(resources_by_video, timestamp_audit)

        tracking_failures = pd.DataFrame(failure_rows)
        if not dense_df.empty:
            review_rows = dense_df[dense_df["qa_status"].ne("ok")].head(50)
            tracking_failures = pd.concat([tracking_failures, review_rows], ignore_index=True, sort=False)

        mismatch_mask = accepted_df["frame_mismatch"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame_mismatch_audit = accepted_df[mismatch_mask].copy()
        source_video_index = timestamp_audit.copy()

        with reporter.stage("write_outputs"):
            write_manifests(
                config.output_root,
                {
                    "legacy_dense_tracklet_map.csv": dense_df,
                    "legacy_training_sequence_manifest.csv": sequence_df,
                    "timestamp_audit.csv": timestamp_audit,
                    "timestamp_parse_debug.csv": timestamp_parse_debug,
                    "source_video_index.csv": source_video_index,
                    "frame_mismatch_audit.csv": frame_mismatch_audit,
                    "depth_provenance_audit.csv": depth_audit,
                    "rejected_rows.csv": rejected_df,
                    "tracking_failure_examples.csv": tracking_failures,
                },
            )
            for filename in [
                "legacy_dense_tracklet_map.csv",
                "legacy_training_sequence_manifest.csv",
                "timestamp_audit.csv",
                "timestamp_parse_debug.csv",
                "source_video_index.csv",
                "frame_mismatch_audit.csv",
                "depth_provenance_audit.csv",
                "rejected_rows.csv",
                "tracking_failure_examples.csv",
            ]:
                reporter.log(f"WROTE {config.output_root / filename}")

        summary = build_qa_summary(
            raw_df=raw_df,
            accepted_df=accepted_df,
            rejected_df=rejected_df,
            dense_df=dense_df,
            path_report=path_report,
            timestamp_audit=timestamp_audit,
            depth_audit=depth_audit,
            tracking_failures=tracking_failures,
        )
        summary["timing_report"] = dict(reporter.stage_timings)
        summary["timing_report"]["total_sec"] = reporter.total_sec()
        write_qa_reports(config.output_root, summary)
        reporter.log(f"WROTE {config.output_root / 'qa_summary.json'}")
        reporter.log(f"WROTE {config.output_root / 'qa_report.md'}")
        reporter.write_timing_report()
        write_progress_state(
            config.output_root,
            dense_df["tracklet_id"].nunique() if not dense_df.empty else 0,
            "",
            "",
            "complete",
            reporter.total_sec(),
        )
        reporter.log(f"WROTE {config.output_root / 'progress_state.json'}")
    except KeyboardInterrupt:
        reporter.log("INTERRUPTED received Ctrl+C; writing partial QA if possible")
        write_partial_qa(
            config,
            reporter,
            raw_df,
            accepted_df,
            rejected_df,
            path_report,
            timestamp_audit,
            depth_audit,
            tracking_failures,
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

"""CLI wrapper for temporal harmonization and sequence-window features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.features.sequence_windows import (
    audit_sequence_windows,
    build_sequence_windows,
)
from pig_behavior.classification_v2.features.temporal_harmonization import (
    TemporalHarmonizationConfig,
    audit_temporal_harmonization,
    build_temporal_label_intervals,
    harmonize_temporal_labels,
)


def _parse_window_lengths(value: str) -> list[int]:
    parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("window length list must not be empty")
    out = []
    for p in parts:
        try:
            n = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid window length: {p}") from exc
        if n <= 0:
            raise argparse.ArgumentTypeError("window lengths must be > 0")
        out.append(n)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build temporal label intervals and long-format 6/8/12/16 sequence-window "
            "features from enhanced frame features."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--harmonized-frame-csv", type=Path, default=None)
    parser.add_argument("--temporal-intervals-csv", type=Path, default=None)
    parser.add_argument("--sequence-window-manifest-csv", type=Path, default=None)
    parser.add_argument("--sequence-window-features-csv", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--window-lengths", type=_parse_window_lengths, default=[6, 8, 12, 16])
    parser.add_argument("--legacy-window-stride", type=int, default=3)
    parser.add_argument("--cvat-window-stride-intervals", type=int, default=1)
    parser.add_argument("--cvat-label-stride", type=int, default=6)
    parser.add_argument("--legacy-expected-sequence-length", type=int, default=16)
    parser.add_argument("--default-fps", type=float, default=None)
    parser.add_argument("--min-bbox-valid-ratio", type=float, default=1.0)
    parser.add_argument("--max-hidden-ratio-main", type=float, default=0.5)
    parser.add_argument("--min-spatiotemporal-valid-ratio", type=float, default=1.0)
    parser.add_argument("--exclude-mixed-windows", action="store_true")
    parser.add_argument(
        "--disable-fast-reuse",
        action="store_true",
        help=(
            "Force a full rebuild from the provided frame CSV instead of "
            "reusing canonical unreviewed window artifacts."
        ),
    )
    parser.add_argument("--max-windows-per-track", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def _to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _can_reuse_window_structure(df: pd.DataFrame, args: argparse.Namespace) -> bool:
    if args.disable_fast_reuse:
        return False
    if args.max_rows is not None:
        return False
    if "review_include_in_training" not in df.columns:
        return False
    if {"behavior_before_review", "behavior_after_review"}.issubset(df.columns):
        changed = (
            df["behavior_before_review"]
            .fillna("")
            .astype(str)
            .ne(df["behavior_after_review"].fillna("").astype(str))
        )
        if bool(changed.any()):
            return False
    base_manifest = Path("outputs/classification_v2/sequence_features/sequence_window_manifest.csv")
    return base_manifest.exists()


def _apply_review_overlay_to_windows(windows: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    out = windows.copy()
    if out.empty:
        return out

    required = {"object_track_key", "window_start_frame", "window_end_frame"}
    if not required.issubset(out.columns):
        missing = sorted(required.difference(out.columns))
        raise ValueError(f"Cannot overlay review masks; window manifest missing {missing}")

    f = frames.copy()
    f["frame_index"] = pd.to_numeric(f["frame_index"], errors="coerce")
    f = f.dropna(subset=["frame_index"]).sort_values(
        ["object_track_key", "frame_index"],
        kind="mergesort",
    )
    f["frame_index"] = f["frame_index"].astype(int)

    if "review_include_in_training" in f.columns:
        f["_review_include"] = _to_bool_series(f["review_include_in_training"])
    else:
        f["_review_include"] = True

    if "review_training_action" in f.columns:
        f["_review_action"] = f["review_training_action"].fillna("").astype(str).str.strip()
        excluded_action = f["_review_action"].str.lower().isin({"exclude", "reject"})
        f["_review_include"] = f["_review_include"] & ~excluded_action
    else:
        f["_review_action"] = ""

    if "review_sample_weight" in f.columns:
        f["_review_weight"] = (
            pd.to_numeric(f["review_sample_weight"], errors="coerce")
            .fillna(1.0)
            .clip(0.0, 1.0)
        )
    else:
        f["_review_weight"] = 1.0

    out["review_include_ratio_window"] = 1.0
    out["review_excluded_frame_count_window"] = 0
    out["review_training_actions_window"] = ""
    out["review_sample_weight_mean_window"] = 1.0
    out["window_sample_weight"] = 1.0

    frame_groups = {
        str(key): group.reset_index(drop=True)
        for key, group in f.groupby(
            "object_track_key",
            dropna=False,
            sort=False,
        )
    }

    window_groups = out.groupby(
        "object_track_key",
        dropna=False,
        sort=False,
    ).groups
    for object_key, win_idx in window_groups.items():
        fg = frame_groups.get(str(object_key))
        if fg is None or fg.empty:
            continue
        frame_idx = fg["frame_index"].to_numpy()
        include = fg["_review_include"].to_numpy(dtype=bool)
        weights = fg["_review_weight"].to_numpy(dtype=float)
        actions = fg["_review_action"].to_numpy(dtype=object)

        for row_idx in win_idx:
            start = int(pd.to_numeric(out.at[row_idx, "window_start_frame"], errors="coerce"))
            end = int(pd.to_numeric(out.at[row_idx, "window_end_frame"], errors="coerce"))
            left = int(frame_idx.searchsorted(start, side="left"))
            right = int(frame_idx.searchsorted(end, side="right"))
            if right <= left:
                continue

            include_slice = include[left:right]
            weight_slice = weights[left:right]
            action_values = sorted(
                {
                    str(value).strip()
                    for value in actions[left:right]
                    if str(value).strip()
                    and str(value).strip().lower() != "nan"
                }
            )
            excluded_count = int((~include_slice).sum())
            out.at[row_idx, "review_include_ratio_window"] = float(include_slice.mean())
            out.at[row_idx, "review_excluded_frame_count_window"] = excluded_count
            out.at[row_idx, "review_training_actions_window"] = "|".join(action_values)
            out.at[row_idx, "review_sample_weight_mean_window"] = float(weight_slice.mean())
            out.at[row_idx, "window_sample_weight"] = (
                0.0 if excluded_count else float(weight_slice.mean())
            )

            if excluded_count:
                out.at[row_idx, "window_valid_for_main_train"] = False
                reason = str(out.at[row_idx, "window_exclusion_reason"] or "").strip()
                token = "review_excluded_rows_in_window"
                if token not in reason.split(";"):
                    out.at[row_idx, "window_exclusion_reason"] = (
                        f"{reason};{token}" if reason else token
                    )
                out.at[row_idx, "window_training_tier_recommendation"] = "exclude"

    return out


def _try_fast_reviewed_rebuild(args: argparse.Namespace) -> bool:
    """Fast reviewed rebuild when human decisions only exclude/downweight rows.

    Corrected labels can change target ROI/label-derived columns, so those still
    use the full frame-to-window rebuild path. Pure review masks can safely reuse
    the already-built structural window manifest and overlay mask/weight fields
    from the reviewed frame CSV.
    """
    if args.disable_fast_reuse or args.max_rows is not None:
        return False

    base_dir = Path("outputs/classification_v2/sequence_features")
    base_manifest = base_dir / "sequence_window_manifest.csv"
    base_intervals = base_dir / "temporal_label_intervals.csv"
    if not base_manifest.exists() or not base_intervals.exists():
        return False

    header = pd.read_csv(args.input_csv, nrows=0)
    columns = set(header.columns)
    if "review_include_in_training" not in columns:
        return False

    if {"behavior_before_review", "behavior_after_review"}.issubset(columns):
        label_check = pd.read_csv(
            args.input_csv,
            usecols=["behavior_before_review", "behavior_after_review"],
            low_memory=False,
        )
        changed = (
            label_check["behavior_before_review"]
            .fillna("")
            .astype(str)
            .ne(label_check["behavior_after_review"].fillna("").astype(str))
        )
        if bool(changed.any()):
            return False

    output_dir = args.output_dir
    intervals_csv = args.temporal_intervals_csv or output_dir / "temporal_label_intervals.csv"
    manifest_csv = args.sequence_window_manifest_csv or output_dir / "sequence_window_manifest.csv"
    features_csv = args.sequence_window_features_csv or output_dir / "sequence_window_features.csv"
    audit_json = args.audit_json or output_dir / "sequence_window_audit.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    intervals_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)

    overlay_cols = [
        c
        for c in [
            "object_track_key",
            "frame_index",
            "review_include_in_training",
            "review_training_action",
            "review_sample_weight",
        ]
        if c in columns
    ]
    frames_overlay = pd.read_csv(args.input_csv, usecols=overlay_cols, low_memory=False)
    intervals = pd.read_csv(base_intervals, low_memory=False)
    windows = pd.read_csv(base_manifest, low_memory=False)
    windows = _apply_review_overlay_to_windows(windows, frames_overlay)

    intervals.to_csv(intervals_csv, index=False)
    windows.to_csv(manifest_csv, index=False)
    windows.to_csv(features_csv, index=False)

    temporal_audit = {
        "rows": None,
        "temporal_intervals": int(len(intervals)),
        "errors": [],
        "warnings": ["fast_path_reused_unreviewed_temporal_intervals_no_corrected_labels_detected"],
    }
    window_audit = audit_sequence_windows(windows, intervals)
    audit = {
        "input_csv": str(args.input_csv),
        "outputs": {
            "temporal_intervals_csv": str(intervals_csv),
            "sequence_window_manifest_csv": str(manifest_csv),
            "sequence_window_features_csv": str(features_csv),
            "audit_json": str(audit_json),
        },
        "parameters": {
            "build_strategy": "reuse_unreviewed_window_structure_with_review_overlay",
            "window_lengths": args.window_lengths,
            "legacy_window_stride": args.legacy_window_stride,
            "cvat_window_stride_intervals": args.cvat_window_stride_intervals,
            "cvat_label_stride": args.cvat_label_stride,
            "legacy_expected_sequence_length": args.legacy_expected_sequence_length,
            "default_fps": args.default_fps,
            "min_bbox_valid_ratio": args.min_bbox_valid_ratio,
            "max_hidden_ratio_main": args.max_hidden_ratio_main,
            "min_spatiotemporal_valid_ratio": args.min_spatiotemporal_valid_ratio,
            "include_mixed_windows": not args.exclude_mixed_windows,
            "disable_fast_reuse": args.disable_fast_reuse,
            "max_windows_per_track": args.max_windows_per_track,
            "max_rows": args.max_rows,
        },
        "temporal_harmonization": temporal_audit,
        "sequence_windows": window_audit,
        "errors": temporal_audit.get("errors", []) + window_audit.get("errors", []),
        "warnings": temporal_audit.get("warnings", []) + window_audit.get("warnings", []),
    }
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] fast reviewed rebuild wrote {intervals_csv} rows={len(intervals)}")
    print(f"[OK] fast reviewed rebuild wrote {manifest_csv} rows={len(windows)}")
    print(f"[OK] fast reviewed rebuild wrote {features_csv} rows={len(windows)}")
    print(f"[OK] fast reviewed rebuild wrote {audit_json}")
    if audit["errors"]:
        print(f"[ERRORS] {audit['errors']}")
    if audit["warnings"]:
        print(f"[WARNINGS] {audit['warnings']}")
    return True


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be > 0")

    if _try_fast_reviewed_rebuild(args):
        return

    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    build_strategy = "full_rebuild"
    if _can_reuse_window_structure(df, args):
        build_strategy = "reuse_unreviewed_window_structure_with_review_overlay"
        harmonized = harmonize_temporal_labels(
            df,
            cvat_label_stride=args.cvat_label_stride,
            legacy_expected_sequence_length=args.legacy_expected_sequence_length,
        )
        interval_config = TemporalHarmonizationConfig(
            cvat_label_stride=args.cvat_label_stride,
            legacy_expected_sequence_length=args.legacy_expected_sequence_length,
        )
        intervals = build_temporal_label_intervals(harmonized, config=interval_config)
        base_manifest = Path(
            "outputs/classification_v2/sequence_features/"
            "sequence_window_manifest.csv"
        )
        windows = pd.read_csv(base_manifest, low_memory=False)
        windows = _apply_review_overlay_to_windows(windows, harmonized)
    else:
        harmonized, intervals, windows = build_sequence_windows(
            df,
            window_lengths=args.window_lengths,
            legacy_window_stride=args.legacy_window_stride,
            cvat_window_stride_intervals=args.cvat_window_stride_intervals,
            cvat_label_stride=args.cvat_label_stride,
            legacy_expected_sequence_length=args.legacy_expected_sequence_length,
            default_fps=args.default_fps,
            min_bbox_valid_ratio=args.min_bbox_valid_ratio,
            max_hidden_ratio_main=args.max_hidden_ratio_main,
            min_spatiotemporal_valid_ratio=args.min_spatiotemporal_valid_ratio,
            include_mixed_windows=not args.exclude_mixed_windows,
            max_windows_per_track=args.max_windows_per_track,
        )

    output_dir = args.output_dir
    harmonized_csv = args.harmonized_frame_csv or (
        output_dir / "training_ready_frame_features_harmonized_preview.csv"
    )
    intervals_csv = args.temporal_intervals_csv or output_dir / "temporal_label_intervals.csv"
    manifest_csv = args.sequence_window_manifest_csv or output_dir / "sequence_window_manifest.csv"
    features_csv = args.sequence_window_features_csv or output_dir / "sequence_window_features.csv"
    audit_json = args.audit_json or output_dir / "sequence_window_audit.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    harmonized_csv.parent.mkdir(parents=True, exist_ok=True)
    intervals_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)

    harmonized.to_csv(harmonized_csv, index=False)
    intervals.to_csv(intervals_csv, index=False)
    windows.to_csv(manifest_csv, index=False)
    # At this stage manifest and feature table intentionally share rows. Keeping
    # a separate file path preserves the future contract if visual-path columns
    # or train-only columns are split later.
    windows.to_csv(features_csv, index=False)

    temporal_audit = audit_temporal_harmonization(harmonized, intervals)
    window_audit = audit_sequence_windows(windows, intervals)
    audit = {
        "input_csv": str(args.input_csv),
        "harmonized_frame_csv": str(harmonized_csv),
        "temporal_intervals_csv": str(intervals_csv),
        "sequence_window_manifest_csv": str(manifest_csv),
        "sequence_window_features_csv": str(features_csv),
        "parameters": {
            "build_strategy": build_strategy,
            "window_lengths": args.window_lengths,
            "legacy_window_stride": args.legacy_window_stride,
            "cvat_window_stride_intervals": args.cvat_window_stride_intervals,
            "cvat_label_stride": args.cvat_label_stride,
            "legacy_expected_sequence_length": args.legacy_expected_sequence_length,
            "default_fps": args.default_fps,
            "min_bbox_valid_ratio": args.min_bbox_valid_ratio,
            "max_hidden_ratio_main": args.max_hidden_ratio_main,
            "min_spatiotemporal_valid_ratio": args.min_spatiotemporal_valid_ratio,
            "include_mixed_windows": not args.exclude_mixed_windows,
            "disable_fast_reuse": args.disable_fast_reuse,
            "max_windows_per_track": args.max_windows_per_track,
            "max_rows": args.max_rows,
        },
        "temporal_harmonization": temporal_audit,
        "sequence_windows": window_audit,
        "errors": temporal_audit.get("errors", []) + window_audit.get("errors", []),
        "warnings": temporal_audit.get("warnings", []) + window_audit.get("warnings", []),
    }
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {harmonized_csv} rows={len(harmonized)} cols={len(harmonized.columns)}")
    print(f"[OK] wrote {intervals_csv} rows={len(intervals)}")
    print(f"[OK] wrote {manifest_csv} rows={len(windows)}")
    print(f"[OK] wrote {features_csv} rows={len(windows)}")
    print(f"[OK] wrote {audit_json}")
    if audit["errors"]:
        print(f"[ERRORS] {audit['errors']}")
    if audit["warnings"]:
        print(f"[WARNINGS] {audit['warnings']}")


if __name__ == "__main__":
    main()

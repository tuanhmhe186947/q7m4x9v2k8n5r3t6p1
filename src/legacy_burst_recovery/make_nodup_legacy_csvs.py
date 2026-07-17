"""Filter legacy metadata tables using an explicit source-video policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .check_duplicate_videos import normalize_source_video_key

REQUIRED_COLUMNS = {"video_final", "group_id", "pig_id"}


def _prepare_source_keys(
    frame: pd.DataFrame,
    *,
    label: str,
    allow_unresolved: bool,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    out = frame.copy()
    out["source_video_key"] = out["video_final"].map(
        normalize_source_video_key
    )
    unresolved = out["source_video_key"].eq("")
    if unresolved.any() and not allow_unresolved:
        sample = out.loc[unresolved, "video_final"].head(5).tolist()
        raise ValueError(
            f"{label} has unresolved source keys={int(unresolved.sum())}; "
            f"sample={sample}"
        )
    return out


def build_nodup_tables(
    center: pd.DataFrame,
    bbox: pd.DataFrame,
    exclusions: pd.DataFrame,
    *,
    allow_unresolved: bool = False,
) -> dict[str, Any]:
    """Return filtered/quarantined tables without writing any files."""
    if "source_video_key" not in exclusions.columns:
        raise ValueError(
            "exclusion CSV must contain canonical column source_video_key"
        )
    exclusion_keys = set(
        exclusions["source_video_key"].map(normalize_source_video_key)
    )
    if "" in exclusion_keys:
        raise ValueError("exclusion CSV contains blank or invalid source keys")

    center = _prepare_source_keys(
        center,
        label="center CSV",
        allow_unresolved=allow_unresolved,
    )
    bbox = _prepare_source_keys(
        bbox,
        label="bbox CSV",
        allow_unresolved=allow_unresolved,
    )
    center["duplicate_video"] = center["source_video_key"].isin(
        exclusion_keys
    )
    bbox["duplicate_video"] = bbox["source_video_key"].isin(exclusion_keys)

    duplicate_pairs = set(
        map(
            tuple,
            center.loc[
                center["duplicate_video"],
                ["group_id", "pig_id"],
            ]
            .astype(str)
            .values,
        )
    )
    bbox["duplicate_video_by_pair"] = [
        (str(group_id), str(pig_id)) in duplicate_pairs
        for group_id, pig_id in zip(
            bbox["group_id"],
            bbox["pig_id"],
            strict=False,
        )
    ]
    bbox["duplicate_video"] |= bbox["duplicate_video_by_pair"]

    center_dup = center.loc[center["duplicate_video"]].copy()
    center_keep = center.loc[~center["duplicate_video"]].copy()
    bbox_dup = bbox.loc[bbox["duplicate_video"]].copy()
    bbox_keep = bbox.loc[~bbox["duplicate_video"]].copy()
    audit = (
        center_dup.groupby("source_video_key")
        .agg(
            duplicate_center_rows=("sample_id", "count")
            if "sample_id" in center_dup.columns
            else ("group_id", "count"),
            duplicate_group_pig=("pig_id", "count"),
        )
        .reset_index()
        .sort_values("source_video_key")
    )
    summary = {
        "center_input_rows": int(len(center)),
        "center_keep_rows": int(len(center_keep)),
        "center_duplicate_rows": int(len(center_dup)),
        "bbox_input_rows": int(len(bbox)),
        "bbox_keep_rows": int(len(bbox_keep)),
        "bbox_duplicate_rows": int(len(bbox_dup)),
        "duplicate_source_keys": int(center_dup["source_video_key"].nunique()),
        "duplicate_group_pig": int(
            center_dup[["group_id", "pig_id"]].drop_duplicates().shape[0]
        ),
        "unresolved_center_rows": int(center["source_video_key"].eq("").sum()),
        "unresolved_bbox_rows": int(bbox["source_video_key"].eq("").sum()),
    }
    return {
        "center_keep": center_keep,
        "bbox_keep": bbox_keep,
        "center_duplicate": center_dup,
        "bbox_duplicate": bbox_dup,
        "audit": audit,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-csv", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-unresolved", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    center_path = args.center_csv.expanduser().resolve()
    bbox_path = args.bbox_csv.expanduser().resolve()
    exclude_path = args.exclude_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_paths = {
        "center_keep": output_dir / "old_burst_center_keyframes_nodup_videos.csv",
        "bbox_keep": output_dir / "old_burst_all_keyframe_bboxes_nodup_videos.csv",
        "center_duplicate": output_dir / "duplicate_video_quarantine_center.csv",
        "bbox_duplicate": output_dir / "duplicate_video_quarantine_all_bboxes.csv",
        "audit": output_dir / "duplicate_video_filter_audit.csv",
    }
    if any(path in {center_path, bbox_path, exclude_path} for path in output_paths.values()):
        raise ValueError("an output cannot overwrite an input CSV")
    if not args.dry_run and not args.overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "output exists; use a fresh directory or --overwrite: "
                + ", ".join(existing)
            )

    result = build_nodup_tables(
        pd.read_csv(center_path, low_memory=False),
        pd.read_csv(bbox_path, low_memory=False),
        pd.read_csv(exclude_path, low_memory=False),
        allow_unresolved=args.allow_unresolved,
    )
    print(result["summary"])
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    result["center_keep"].to_csv(output_paths["center_keep"], index=False)
    result["bbox_keep"].to_csv(output_paths["bbox_keep"], index=False)
    result["center_duplicate"].to_csv(
        output_paths["center_duplicate"],
        index=False,
    )
    result["bbox_duplicate"].to_csv(output_paths["bbox_duplicate"], index=False)
    result["audit"].to_csv(output_paths["audit"], index=False)
    for path in output_paths.values():
        print(f"saved={path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

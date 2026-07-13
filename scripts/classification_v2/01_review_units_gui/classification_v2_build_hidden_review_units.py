"""Build two-sided Hidden review cohorts from enhanced frame features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_builder import (
    HiddenReviewConfig,
    balanced_hidden_smoke_scope,
    build_hidden_review_manifest,
)

TEMPLATE_FILENAMES = {
    "hidden_yes_confirmation": "hidden_yes_review_template.csv",
    "hidden_no_high_risk": "hidden_no_risk_review_template.csv",
    "hidden_no_random_audit": "hidden_no_random_audit_template.csv",
    "hidden_no_clean_control": "hidden_no_clean_control_template.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic Hidden=Yes confirmation and Hidden=No "
            "false-negative audit cohorts."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--random-no-per-stratum", type=int, default=3)
    parser.add_argument("--clean-control-per-stratum", type=int, default=1)
    parser.add_argument("--max-high-risk-per-stratum", type=int, default=None)
    parser.add_argument("--high-risk-threshold", type=float, default=0.35)
    parser.add_argument("--clean-control-max-risk", type=float, default=0.10)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Smoke-only input row cap. Do not use for the full review build.",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="Smoke-only cap that retains rows from both legacy and CVAT.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in output-dir for the same lineage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be > 0")
    if args.max_rows_per_source is not None and args.max_rows_per_source <= 0:
        raise ValueError("--max-rows-per-source must be > 0")
    if args.max_rows is not None and args.max_rows_per_source is not None:
        raise ValueError("Use only one smoke row cap")

    output_paths = [
        args.output_dir / "hidden_review_unit_manifest.csv",
        args.output_dir / "hidden_review_frame_context.csv",
        args.output_dir / "hidden_review_template_audit.json",
        *(args.output_dir / name for name in TEMPLATE_FILENAMES.values()),
    ]
    _guard_outputs(output_paths, overwrite=args.overwrite)

    frames = pd.read_csv(args.input_csv, low_memory=False)
    if args.max_rows is not None:
        frames = frames.head(args.max_rows).copy()
    elif args.max_rows_per_source is not None:
        frames = balanced_hidden_smoke_scope(
            frames,
            args.max_rows_per_source,
        )
    config = HiddenReviewConfig(
        random_seed=args.random_seed,
        random_no_per_stratum=args.random_no_per_stratum,
        clean_control_per_stratum=args.clean_control_per_stratum,
        max_high_risk_per_stratum=args.max_high_risk_per_stratum,
        high_risk_threshold=args.high_risk_threshold,
        clean_control_max_risk=args.clean_control_max_risk,
    )
    manifest, templates, audit = build_hidden_review_manifest(
        frames,
        config=config,
    )
    audit["input_csv"] = str(args.input_csv)
    audit["output_dir"] = str(args.output_dir)
    audit["max_rows"] = args.max_rows
    audit["max_rows_per_source"] = args.max_rows_per_source
    frame_context = _selected_frame_context(frames, manifest)
    audit["frame_context_rows"] = int(len(frame_context))
    audit["frame_context_frames"] = int(frame_context["frame_uid"].nunique(dropna=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_paths[0], index=False)
    frame_context.to_csv(output_paths[1], index=False)
    for cohort, filename in TEMPLATE_FILENAMES.items():
        templates[cohort].to_csv(args.output_dir / filename, index=False)
    output_paths[2].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[PASS] Hidden review manifest rows={len(manifest)} cohorts={audit['cohort_counts']}")
    print(f"[PASS] frame context rows={len(frame_context)}")
    print(f"[PASS] audit={output_paths[2]}")


def _selected_frame_context(
    frames: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Keep all annotated objects on selected frames for GUI overlays."""
    keys = ["source_type", "dataset_id", "video_key", "frame_uid"]
    missing = [column for column in keys if column not in frames.columns]
    if missing:
        raise ValueError(f"Cannot build frame context; missing columns: {missing}")
    selected = manifest[keys].drop_duplicates()
    context = frames.merge(selected, on=keys, how="inner", validate="many_to_one")
    found = context[keys].drop_duplicates()
    if len(found) != len(selected):
        raise ValueError(
            f"Frame context lost selected frames: expected={len(selected)} found={len(found)}"
        )
    sort_columns = [
        "source_type",
        "video_key",
        "frame_index",
        "frame_uid",
        "pig_id",
    ]
    return context.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _guard_outputs(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        display = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {display}. Use --overwrite explicitly.")


if __name__ == "__main__":
    main()

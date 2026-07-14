"""Build audited native and T6/T8/T12/T16 legacy development manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    DEFAULT_TEMPORAL_TIERS,
    LEGACY_DEVELOPMENT_SCOPE,
    build_legacy_unreviewed_development_manifests,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

SOURCE_COLUMNS = [
    "source_type",
    "dataset_id",
    "video_key",
    "clip_id",
    "track_id",
    "pig_id",
    "relative_frame_index",
    "behavior",
    "bbox_valid",
    "include_in_training",
    "use_for_main_eval",
    "hidden",
]

HARMONIZED_COLUMNS = [
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "track_id",
    "pig_id",
    "frame_index",
    "temporal_unit_key",
    "behavior_temporal_final",
    "bbox_valid",
    "spatiotemporal_feature_valid",
    "include_in_training",
    "use_for_main_eval",
]

INTERVAL_COLUMNS = [
    "temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "pig_id",
    "track_id",
    "label_window_start",
    "label_window_end",
    "label_frame_count",
    "observed_frame_count",
    "expected_observed_frame_count",
    "temporal_interval_complete",
    "behavior_temporal_final",
    "behavior_consistency_in_interval",
    "bbox_valid_ratio_interval",
    "hidden_ratio_interval",
    "spatiotemporal_feature_valid_ratio_interval",
]

WINDOW_COLUMNS = [
    "window_id",
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "window_length_frames",
    "window_start_frame",
    "window_end_frame",
    "temporal_unit_keys_json",
    "num_temporal_units_window",
    "behavior_window_label",
    "window_valid_for_main_train",
]


def _parse_tiers(value: str) -> tuple[int, ...]:
    """Parse and lock the controlled temporal-length comparison set."""

    try:
        tiers = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("temporal tiers must be integers") from exc
    if tiers != DEFAULT_TEMPORAL_TIERS:
        raise argparse.ArgumentTypeError(
            f"temporal tiers must equal {DEFAULT_TEMPORAL_TIERS}"
        )
    return tiers


def parse_args() -> argparse.Namespace:
    """Parse immutable inputs and one versioned derived-output directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference-csv", type=Path, required=True)
    parser.add_argument("--harmonized-frame-csv", type=Path, required=True)
    parser.add_argument("--intervals-csv", type=Path, required=True)
    parser.add_argument("--window-manifest-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--temporal-tiers",
        type=_parse_tiers,
        default=DEFAULT_TEMPORAL_TIERS,
    )
    parser.add_argument("--legacy-window-stride", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this derived manifest set explicitly.",
    )
    return parser.parse_args()


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "source_units": output_dir / "source_unit_manifest.csv",
        "native_units": output_dir / "native_temporal_unit_manifest.csv",
        "all_sliding": output_dir / "temporal_tier_all_sliding_manifest.csv",
        "matched": output_dir / "temporal_tier_matched_manifest.csv",
        "audit": output_dir / "legacy_unreviewed_development_audit.json",
    }


def _assert_derived_output(output_dir: Path) -> None:
    """Prevent an operator typo from writing derived evidence under data/."""

    resolved = output_dir.resolve()
    immutable_data = (Path.cwd() / "data").resolve()
    if resolved == immutable_data or resolved.is_relative_to(immutable_data):
        raise ValueError(f"derived output must not be under data/: {resolved}")


def _read_columns(path: Path, columns: list[str], name: str) -> pd.DataFrame:
    """Read only the declared contract columns and reject missing inputs."""

    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(columns).difference(header.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    return pd.read_csv(path, usecols=columns, low_memory=False)


def main() -> None:
    """Write hash-bound manifests after all in-memory contracts pass."""

    args = parse_args()
    _assert_derived_output(args.output_dir)
    paths = _output_paths(args.output_dir)
    require_output_paths_available(
        list(paths.values()),
        overwrite=args.overwrite,
    )

    source = _read_columns(
        args.source_reference_csv,
        SOURCE_COLUMNS,
        "source_reference",
    )
    harmonized = _read_columns(
        args.harmonized_frame_csv,
        HARMONIZED_COLUMNS,
        "harmonized_frames",
    )
    intervals = _read_columns(
        args.intervals_csv,
        INTERVAL_COLUMNS,
        "intervals",
    )
    windows = _read_columns(
        args.window_manifest_csv,
        WINDOW_COLUMNS,
        "window_manifest",
    )
    tables = build_legacy_unreviewed_development_manifests(
        source,
        harmonized,
        intervals,
        windows,
        temporal_tiers=args.temporal_tiers,
        legacy_window_stride=args.legacy_window_stride,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables.source_units.to_csv(paths["source_units"], index=False)
    tables.native_units.to_csv(paths["native_units"], index=False)
    tables.all_sliding_windows.to_csv(paths["all_sliding"], index=False)
    tables.matched_windows.to_csv(paths["matched"], index=False)

    input_paths = {
        "source_reference_csv": args.source_reference_csv,
        "harmonized_frame_csv": args.harmonized_frame_csv,
        "intervals_csv": args.intervals_csv,
        "window_manifest_csv": args.window_manifest_csv,
    }
    output_paths = {
        name: path for name, path in paths.items() if name != "audit"
    }
    audit = {
        **tables.audit,
        "status": "PASS_LEGACY_UNREVIEWED_BOUNDED_DEVELOPMENT",
        "input_artifacts": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in input_paths.items()
        },
        "output_artifacts": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in output_paths.items()
        },
        "parameters": {
            "temporal_tiers": list(args.temporal_tiers),
            "legacy_window_stride": args.legacy_window_stride,
            "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
            "overwrite": args.overwrite,
        },
    }
    paths["audit"].write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "source_rows": audit["source_audit"]["rows"],
                "native_units": audit["native_unit_audit"]["rows"],
                "all_sliding_rows": audit["temporal_tier_audit"][
                    "all_sliding_rows"
                ],
                "matched_rows": audit["temporal_tier_audit"]["matched_rows"],
                "human_review_complete": False,
                "q2_claim_allowed": False,
                "audit_json": str(paths["audit"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

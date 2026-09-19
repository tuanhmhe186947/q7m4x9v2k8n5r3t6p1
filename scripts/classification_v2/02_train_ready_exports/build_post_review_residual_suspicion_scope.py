"""Build a review-only residual suspicion scope after composite review."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.post_review_residual_discovery import (
    build_review_informed_temporal_residuals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-universe-csv", type=Path, required=True)
    parser.add_argument("--composite-decisions-csv", type=Path, required=True)
    parser.add_argument("--expected-composite-decisions-sha256", required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-review-output-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--roi-coco-json", type=Path, required=True)
    parser.add_argument("--maximum-gap-run-units", type=int, default=2)
    parser.add_argument(
        "--included-severity",
        action="append",
        choices=("HIGH", "MEDIUM"),
        dest="included_severities",
        help=(
            "Repeat to select exact residual severities. "
            "Defaults to HIGH and MEDIUM."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    actual_hash = sha256_file(args.composite_decisions_csv)
    if actual_hash.casefold() != args.expected_composite_decisions_sha256.casefold():
        raise ValueError(
            "composite decision hash mismatch: "
            f"expected={args.expected_composite_decisions_sha256} "
            f"actual={actual_hash}"
        )

    universe = pd.read_csv(
        args.source_universe_csv,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    decisions = pd.read_csv(
        args.composite_decisions_csv,
        dtype=str,
        keep_default_na=False,
    )
    result = build_review_informed_temporal_residuals(
        universe,
        decisions,
        maximum_gap_run_units=args.maximum_gap_run_units,
        included_severities=tuple(
            args.included_severities or ("HIGH", "MEDIUM")
        ),
    )
    args.output_dir.mkdir(parents=True)
    output_paths = {
        "findings": args.output_dir / "all_post_review_temporal_gap_findings.csv",
        "selected_findings": (
            args.output_dir / "selected_review_informed_findings.csv"
        ),
        "selected_scope": (
            args.output_dir / "post_review_residual_suspicion_scope.csv"
        ),
        "control_population": (
            args.output_dir / "post_review_control_population.csv"
        ),
        "control_exclusion_scope": (
            args.output_dir / "post_review_control_exclusion_scope.csv"
        ),
    }
    for name, path in output_paths.items():
        result[name].to_csv(path, index=False)
    audit_path = args.output_dir / "post_review_residual_suspicion_audit.json"
    write_json(audit_path, result["audit"])

    command = review_command(
        scope_path=output_paths["selected_scope"],
        frame_features_csv=args.frame_features_csv,
        output_dir=args.future_review_output_dir,
        video_root=args.video_root,
        raw_root=args.raw_root,
        roi_coco_json=args.roi_coco_json,
    )
    command_path = args.output_dir / "exact_targeted_review_command.txt"
    command_path.write_text(command + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "classification_v2.post_review_residual_scope.v1",
        "status": "READY_FOR_TARGETED_HUMAN_REVIEW",
        "code_sha": git_head(),
        "selection_semantics": (
            "Short unreviewed temporal gaps bounded by the same effective "
            "label, with at least one reviewed source-label correction in "
            "the three-run neighborhood."
        ),
        "included_severities": list(
            args.included_severities or ("HIGH", "MEDIUM")
        ),
        "inputs": {
            "source_universe": path_record(args.source_universe_csv),
            "composite_decisions": path_record(args.composite_decisions_csv),
            "frame_features": path_record(args.frame_features_csv),
            "adjusted_roi": path_record(args.roi_coco_json),
        },
        "outputs": {
            name: path_record(path) for name, path in output_paths.items()
        },
        "audit": path_record(audit_path),
        "automatic_relabeling": False,
        "review_metadata_entering_model_x": False,
        "independent_control_cohort_included": False,
        "independent_control_next_step": (
            "Sample at least 120 controls from the explicit universe after "
            "subtracting post_review_control_exclusion_scope.csv."
        ),
    }
    manifest_path = args.output_dir / "post_review_residual_scope_manifest.json"
    write_json(manifest_path, manifest)
    inventory = {
        "schema_version": "classification_v2.artifact_inventory.v1",
        "artifacts": [
            path_record(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "artifact_inventory.json"
        ],
    }
    write_json(args.output_dir / "artifact_inventory.json", inventory)
    print("PASS: post-review residual suspicion scope written")
    print(args.output_dir.resolve())


def review_command(
    *,
    scope_path: Path,
    frame_features_csv: Path,
    output_dir: Path,
    video_root: Path,
    raw_root: Path,
    roi_coco_json: Path,
) -> str:
    executable = Path(sys.executable).resolve()
    script = Path(
        "scripts/classification_v2/01_review_units_gui/"
        "review_final_behavior_gui_v1.py"
    ).resolve()
    arguments = [
        f'cd /d "{Path.cwd().resolve()}"',
        f'"{executable}" "{script}"',
        f'--review-units-csv "{scope_path.resolve()}"',
        f'--frame-features-csv "{frame_features_csv.resolve()}"',
        f'--output-dir "{output_dir.resolve()}"',
        f'--video-root "{video_root.resolve()}"',
        f'--raw-root "{raw_root.resolve()}"',
        f'--roi-coco-json "{roi_coco_json.resolve()}"',
    ]
    return arguments[0] + " && " + " ".join(arguments[1:])


def path_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


if __name__ == "__main__":
    main()

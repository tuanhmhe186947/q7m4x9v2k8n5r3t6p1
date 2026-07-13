"""Audit identifier-v2 frame lineage and positional window alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.source_to_window_lineage import (
    audit_source_to_window_lineage,
)

DEFAULT_ROOT = Path(
    "outputs/classification_v2/rebuilds/"
    "scientific_smoke_identifier_v2_20260713"
)


def parse_args() -> argparse.Namespace:
    """Parse a versioned bounded root and explicit audit destination."""

    parser = argparse.ArgumentParser(
        description=(
            "Prove frame-object identifier and ordered-window lineage for a "
            "bounded classification_v2 rebuild. This never authorizes training."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repeat-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing derived lineage audit explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    """Load bounded artifacts, run the pure contract, and persist evidence."""

    args = parse_args()
    output_path = args.output_json or (
        args.root / "audits" / "identifier_v2_source_to_window_audit.json"
    )
    require_output_paths_available([output_path], overwrite=args.overwrite)
    paths = _artifact_paths(args.root)
    preload_errors: list[str] = []

    frame_stages = {
        name: _read_csv(path, name, preload_errors)
        for name, path in paths["frame_stages"].items()
    }
    sequence_manifest = _read_csv(
        paths["tables"]["sequence_manifest"],
        "sequence_manifest",
        preload_errors,
    )
    sequence_features = _read_csv(
        paths["tables"]["sequence_features"],
        "sequence_features",
        preload_errors,
    )
    image_frame_manifest = _read_csv(
        paths["tables"]["image_frame_manifest"],
        "image_frame_manifest",
        preload_errors,
    )
    image_window_manifest = _read_csv(
        paths["tables"]["image_window_manifest"],
        "image_window_manifest",
        preload_errors,
    )
    train_ready_tables = {
        name: _read_csv(path, name, preload_errors)
        for name, path in paths["train_ready_tables"].items()
    }
    artifact_audits = {
        name: _read_json(path, name, preload_errors)
        for name, path in paths["audits"].items()
    }
    spatial_array_rows = _read_npz_rows(
        paths["spatial_npz"],
        preload_errors,
    )

    x_table = train_ready_tables.get("X_window_features", pd.DataFrame())
    result = audit_source_to_window_lineage(
        frame_stages=frame_stages,
        sequence_manifest=sequence_manifest,
        sequence_features=sequence_features,
        image_frame_manifest=image_frame_manifest,
        image_window_manifest=image_window_manifest,
        x_columns=[str(column) for column in x_table.columns],
        artifact_audits=artifact_audits,
        artifact_row_counts={
            name: int(len(rows))
            for name, rows in train_ready_tables.items()
        },
        spatial_array_rows=spatial_array_rows,
        preload_errors=preload_errors,
    )
    all_paths = _flatten_paths(paths)
    repeatability = _repeatability_audit(args.root, args.repeat_root)
    result["repeatability"] = repeatability
    if repeatability["required"] and not repeatability["all_match"]:
        result["errors"].append("source_to_window_repeatability_mismatch")
        result["technical_pass"] = False
        result["status"] = "FAIL_IDENTIFIER_V2_LINEAGE"
    for name, path in repeatability["input_paths"].items():
        all_paths[f"repeatability.{name}"] = Path(path)
    result["inputs"] = {name: str(path) for name, path in all_paths.items()}
    result["artifact_sha256"] = {
        name: _sha256(path)
        for name, path in all_paths.items()
        if path.exists() and path.is_file()
    }
    result["code_state"] = _git_state()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "technical_pass": result["technical_pass"],
                "frame_rows": result["frame_lineage"]["reference"].get("rows"),
                "window_rows": result["row_lineage"]["expected_window_rows"],
                "ordered_window_id_sha256": result["window_lineage"].get(
                    "ordered_window_id_sha256"
                ),
                "authorization": result["authorization"],
                "repeatability": result["repeatability"],
                "human_review_blockers": result["human_review_blockers"],
                "errors": result["errors"],
                "warnings": result["warnings"],
                "output_json": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["technical_pass"]:
        raise SystemExit(2)


def _artifact_paths(root: Path) -> dict[str, Any]:
    """Return the fixed identifier-v2 bounded layout."""

    sequence_dir = root / "05_sequence"
    train_ready_dir = root / "07_train_ready"
    return {
        "frame_stages": {
            "context": root / "01_context" / "frame_objects_context_policy.csv",
            "geometry": (
                root
                / "02_geometry"
                / "spatiotemporal_frame_features_geometry.csv"
            ),
            "roi": root / "03_roi" / "spatiotemporal_frame_features_roi.csv",
            "enhanced": (
                root
                / "04_enhanced"
                / "spatiotemporal_frame_features_enhanced.csv"
            ),
            "harmonized": (
                sequence_dir
                / "training_ready_frame_features_harmonized_preview.csv"
            ),
        },
        "tables": {
            "sequence_manifest": sequence_dir / "sequence_window_manifest.csv",
            "sequence_features": sequence_dir / "sequence_window_features.csv",
            "image_frame_manifest": (
                train_ready_dir / "image_frame_context_manifest.csv"
            ),
            "image_window_manifest": (
                train_ready_dir / "image_window_context_manifest.csv"
            ),
        },
        "train_ready_tables": {
            "X_window_features": train_ready_dir / "X_window_features.csv",
            "y_behavior": train_ready_dir / "y_behavior.csv",
            "train_mask": train_ready_dir / "train_mask.csv",
            "sample_weight": train_ready_dir / "sample_weight.csv",
        },
        "audits": {
            "train_ready": train_ready_dir / "train_ready_audit.json",
            "spatial": train_ready_dir / "spatial_sequence_audit.json",
            "image_context": train_ready_dir / "image_context_index_audit.json",
        },
        "spatial_npz": train_ready_dir / "X_spatial_sequences.npz",
    }


def _read_csv(
    path: Path,
    name: str,
    errors: list[str],
) -> pd.DataFrame:
    """Read one bounded CSV and retain failures as audit errors."""

    if not path.exists():
        errors.append(f"missing_csv={name}:{path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        errors.append(f"invalid_csv={name}:{path}:{exc}")
        return pd.DataFrame()


def _read_json(
    path: Path,
    name: str,
    errors: list[str],
) -> dict[str, Any]:
    """Read one exporter audit and reject missing or non-object JSON."""

    if not path.exists():
        errors.append(f"missing_json={name}:{path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_json={name}:{path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"json_not_object={name}:{path}")
        return {}
    return payload


def _read_npz_rows(path: Path, errors: list[str]) -> dict[str, int]:
    """Read only array shapes; identifiers never enter this model-input file."""

    if not path.exists():
        errors.append(f"missing_npz={path}")
        return {}
    rows: dict[str, int] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = archive[name]
                if array.ndim == 0:
                    errors.append(f"scalar_spatial_array={name}")
                    continue
                rows[str(name)] = int(array.shape[0])
    except (OSError, ValueError) as exc:
        errors.append(f"invalid_npz={path}:{exc}")
    return rows


def _flatten_paths(paths: dict[str, Any]) -> dict[str, Path]:
    """Flatten grouped paths for reproducibility hashes."""

    flattened: dict[str, Path] = {}
    for group, value in paths.items():
        if isinstance(value, Path):
            flattened[group] = value
            continue
        for name, path in value.items():
            flattened[f"{group}.{name}"] = path
    return flattened


def _repeatability_audit(
    root: Path,
    repeat_root: Path | None,
) -> dict[str, Any]:
    """Compare deterministic stage CSV bytes when an independent root is given."""

    if repeat_root is None:
        return {
            "required": False,
            "pair_count": 0,
            "matching_pair_count": 0,
            "all_match": None,
            "comparisons": [],
            "input_paths": {},
        }
    relative_paths = [
        Path("01_context/frame_objects_context_policy.csv"),
        Path("02_geometry/spatiotemporal_frame_features_geometry.csv"),
        Path("03_roi/spatiotemporal_frame_features_roi.csv"),
        Path("04_enhanced/spatiotemporal_frame_features_enhanced.csv"),
        Path(
            "05_sequence/"
            "training_ready_frame_features_harmonized_preview.csv"
        ),
        Path("05_sequence/temporal_label_intervals.csv"),
        Path("05_sequence/sequence_window_manifest.csv"),
        Path("05_sequence/sequence_window_features.csv"),
    ]
    comparisons: list[dict[str, Any]] = []
    input_paths: dict[str, str] = {}
    for index, relative in enumerate(relative_paths):
        original = root / relative
        repeated = repeat_root / relative
        original_hash = _sha256(original) if original.exists() else None
        repeated_hash = _sha256(repeated) if repeated.exists() else None
        match = (
            original_hash is not None
            and repeated_hash is not None
            and original_hash == repeated_hash
        )
        comparisons.append(
            {
                "relative_path": relative.as_posix(),
                "original_sha256": original_hash,
                "repeated_sha256": repeated_hash,
                "match": match,
            }
        )
        input_paths[f"original_{index}"] = str(original)
        input_paths[f"repeated_{index}"] = str(repeated)
    matching = sum(bool(item["match"]) for item in comparisons)
    return {
        "required": True,
        "repeat_root": str(repeat_root),
        "pair_count": int(len(comparisons)),
        "matching_pair_count": int(matching),
        "all_match": bool(comparisons) and matching == len(comparisons),
        "comparisons": comparisons,
        "input_paths": input_paths,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    """Record code SHA and dirty paths without changing the worktree."""

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty_paths = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "git_sha": sha.stdout.strip() if sha.returncode == 0 else None,
        "dirty_worktree": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


if __name__ == "__main__":
    main()

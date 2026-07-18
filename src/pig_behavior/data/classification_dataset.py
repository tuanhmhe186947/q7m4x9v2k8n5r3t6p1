"""End-to-end classification dataset orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .classification_features import (
    AUTHORITY_POLICY,
    add_training_features,
    clean_merged_annotations,
)
from .cvat_native import (
    load_all_cvat_tasks,
    load_behaviors_from_project,
    select_cvat_annotation_source,
)
from .validation import validate_project_outputs


def find_project_root(start: Path | None = None) -> Path:
    """Return nearest parent containing project metadata."""
    current = Path.cwd() if start is None else Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DATA_EXPORT_ROOT = PROJECT_ROOT / "data" / "data"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images_clean"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLASSIFICATION_PROCESSED_DIR = PROCESSED_DIR / "classification"
ROI_COCO_JSON = ANNOTATION_DIR / "roi" / "ROI_annotations.coco.json"
OUT_CLEAN_NAME = "behavior_clean_merged.csv"
OUT_FEATS_NAME = "behavior_with_feats_rectROI.csv"
OUT_LINEAGE_NAME = "classification_source_lineage.json"
EXPECTED_TASK_NAMES = ("task_0", "task_1", "task_2", "task_3")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a fresh, explicit dataset-build lineage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cvat-export-root",
        type=Path,
        default=DATA_EXPORT_ROOT,
    )
    parser.add_argument(
        "--roi-coco-json",
        type=Path,
        default=ROI_COCO_JSON,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh versioned output directory; never the CVAT input root.",
    )
    parser.add_argument(
        "--copy-images-to",
        type=Path,
        default=None,
        help="Optional compatibility image copy. Omit for provenance rebuilds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all loading, authority, feature, and schema checks without writes.",
    )
    parser.add_argument(
        "--exclude-actor-key-csv",
        type=Path,
        default=None,
        help=(
            "Explicit policy CSV with group_id,pig_id,reason for actor keys "
            "excluded before authority validation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace this versioned lineage's derived outputs.",
    )
    return parser.parse_args(argv)


def copy_annotated_images(df: pd.DataFrame, dst_root: Path) -> int:
    """Copy source images referenced by annotations to the training image root."""
    copied = 0
    missing = []
    for row in df[["img_name", "image_path"]].drop_duplicates().itertuples(
        index=False
    ):
        src = Path(row.image_path)
        dst = dst_root / row.img_name
        if not src.exists():
            missing.append(str(src))
            continue
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)
            copied += 1
    if missing:
        raise FileNotFoundError(
            f"missing_source_images={len(missing)}; sample={missing[:10]}"
        )
    return copied


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV without exposing a partial final artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def validate_build_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Validate input authority and fresh derived-output paths."""
    cvat_root = args.cvat_export_root.expanduser().resolve()
    roi_path = args.roi_coco_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    immutable_data_root = (PROJECT_ROOT / "data").resolve()
    if not cvat_root.is_dir():
        raise FileNotFoundError(f"CVAT export folder not found: {cvat_root}")
    if not roi_path.is_file():
        raise FileNotFoundError(f"ROI COCO file not found: {roi_path}")
    if _is_within(output_dir, cvat_root):
        raise ValueError("output directory cannot be inside the CVAT source root")
    if _is_within(output_dir, immutable_data_root):
        raise ValueError("output directory cannot be inside immutable project data")
    if args.copy_images_to is not None:
        copy_root = args.copy_images_to.expanduser().resolve()
        if _is_within(copy_root, cvat_root):
            raise ValueError(
                "image copy directory cannot be inside the CVAT source root"
            )
        if _is_within(copy_root, immutable_data_root):
            raise ValueError(
                "image copy directory cannot be inside immutable project data"
            )

    output_paths = {
        "clean_csv": output_dir / OUT_CLEAN_NAME,
        "feature_csv": output_dir / OUT_FEATS_NAME,
        "lineage_json": output_dir / OUT_LINEAGE_NAME,
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing dataset artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    return output_paths


def collect_source_records(cvat_root: Path, roi_path: Path) -> list[dict[str, Any]]:
    """Freeze exactly one annotation authority and manifest per task."""
    task_dirs = sorted(path for path in cvat_root.glob("task_*") if path.is_dir())
    actual_names = tuple(path.name for path in task_dirs)
    if actual_names != EXPECTED_TASK_NAMES:
        raise ValueError(
            f"unexpected_cvat_task_set={actual_names}; "
            f"expected={EXPECTED_TASK_NAMES}"
        )

    source_paths: list[tuple[str, str, Path]] = []
    for task_dir in task_dirs:
        annotation_format, annotation_path = select_cvat_annotation_source(task_dir)
        source_paths.extend(
            [
                (task_dir.name, f"annotation_{annotation_format}", annotation_path),
                (task_dir.name, "task", task_dir / "task.json"),
                (task_dir.name, "manifest", task_dir / "data" / "manifest.jsonl"),
            ]
        )
    project_path = cvat_root / "project.json"
    if project_path.exists():
        source_paths.append(("project", "behavior_schema", project_path))
    source_paths.append(("scene", "roi_coco", roi_path))

    missing = [str(path) for _, _, path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing_dataset_source_files={len(missing)}; sample={missing[:10]}"
        )
    return [
        {
            "scope": scope,
            "role": role,
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for scope, role, path in source_paths
    ]


def validate_source_images(frame: pd.DataFrame) -> None:
    """Fail if the generated CSV points at missing task images."""
    image_paths = frame["image_path"].drop_duplicates().map(Path)
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing_source_images={len(missing)}; sample={missing[:10]}"
        )


def load_actor_exclusion_policy(path: Path | None) -> tuple[list[dict[str, str]], str | None]:
    """Load explicit actor exclusions and reject ambiguous policy rows."""
    if path is None:
        return [], None
    policy_path = path.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"actor_exclusion_policy_not_found={policy_path}")
    policy = pd.read_csv(policy_path, dtype=str, keep_default_na=False)
    required = {"group_id", "pig_id", "reason"}
    missing = sorted(required.difference(policy.columns))
    if missing:
        raise ValueError(f"actor_exclusion_policy_missing_columns={missing}")
    policy = policy[["group_id", "pig_id", "reason"]].copy()
    for column in policy.columns:
        policy[column] = policy[column].astype(str).str.strip()
    if policy.eq("").any(axis=None):
        raise ValueError("actor_exclusion_policy_contains_blank_values")
    if policy.duplicated(["group_id", "pig_id"]).any():
        raise ValueError("actor_exclusion_policy_contains_duplicate_keys")
    return policy.to_dict("records"), str(policy_path)


def main(argv: list[str] | None = None) -> None:
    """Build a strict six-anchor provenance CSV from current native CVAT."""
    args = parse_args(argv)
    output_paths = validate_build_paths(args)
    cvat_root = args.cvat_export_root.expanduser().resolve()
    roi_path = args.roi_coco_json.expanduser().resolve()
    source_records = collect_source_records(cvat_root, roi_path)
    behaviors = load_behaviors_from_project(cvat_root / "project.json")

    print("PROJECT_ROOT    :", PROJECT_ROOT)
    print("CVAT_EXPORT_ROOT:", cvat_root)
    print("OUTPUT_DIR      :", output_paths["clean_csv"].parent)
    print("ROI_COCO_JSON   :", roi_path)
    print("Behavior policy :", AUTHORITY_POLICY)
    print("Behaviors       :", behaviors)

    df_raw = load_all_cvat_tasks(cvat_root)
    exclusions, exclusion_policy_path = load_actor_exclusion_policy(
        args.exclude_actor_key_csv
    )
    exclusion_keys = {(row["group_id"], row["pig_id"]) for row in exclusions}
    source_keys = set(
        zip(
            df_raw["group_id"].astype(str),
            df_raw["pig_id"].astype(str),
            strict=True,
        )
    )
    unknown_keys = sorted(exclusion_keys.difference(source_keys))
    if unknown_keys:
        raise ValueError(f"actor_exclusion_policy_unknown_keys={unknown_keys}")
    exclusion_mask = pd.Series(False, index=df_raw.index)
    for group_id, pig_id in exclusion_keys:
        exclusion_mask |= df_raw["group_id"].eq(group_id) & df_raw["pig_id"].eq(
            pig_id
        )
    df_retained = df_raw.loc[~exclusion_mask].copy()
    print(
        "Raw rows:",
        len(df_raw),
        "Excluded rows:",
        int(exclusion_mask.sum()),
        "Retained rows:",
        len(df_retained),
    )

    df_clean = clean_merged_annotations(
        df_retained,
        behaviors,
        drop_hidden=False,
    )
    if len(df_clean) != len(df_retained):
        raise ValueError("cleaning_changed_row_count")
    validate_source_images(df_clean)

    df_feats = add_training_features(df_clean, roi_path)
    if len(df_feats) != len(df_clean):
        raise ValueError("feature_generation_changed_row_count")
    print("Rows:", len(df_feats), "Images:", df_feats["img_name"].nunique())
    print("Fine behavior distribution:")
    print(df_feats["behavior"].value_counts())
    print("Coarse behavior distribution:")
    print(df_feats["behavior_coarse"].value_counts())
    print("ROI flag sums:")
    print(df_feats[["in_feeder", "in_drinker", "in_toy"]].sum())

    copied = 0
    if args.copy_images_to is not None and not args.dry_run:
        copy_root = args.copy_images_to.expanduser().resolve()
        copy_root.mkdir(parents=True, exist_ok=True)
        copied = copy_annotated_images(df_clean, copy_root)
        validate_project_outputs(df_feats, copy_root)
        print(f"[COPY] copied/updated {copied} images into {copy_root}")

    lineage = {
        "schema_version": 1,
        "role": "legacy_six_anchor_provenance_input",
        "behavior_authority_policy": AUTHORITY_POLICY,
        "bbox_authority": "native_cvat_per_anchor",
        "hidden_authority": "native_cvat_seed_untrusted",
        "source_files": source_records,
        "row_counts": {
            "raw": int(len(df_raw)),
            "excluded": int(exclusion_mask.sum()),
            "retained": int(len(df_retained)),
            "validated": int(len(df_clean)),
            "feature": int(len(df_feats)),
        },
        "key_counts": {
            "images": int(df_feats["img_name"].nunique()),
            "groups": int(df_feats["group_id"].nunique()),
            "actor_keys": int(
                df_feats.groupby(["group_id", "pig_id"], dropna=False).ngroups
            ),
        },
        "quality_counts": {
            "behavior_disagreement_rows": int(
                df_feats["behavior_disagrees_with_authority"].sum()
            ),
            "bbox_outside_image_rows": int(df_feats["bbox_outside_image"].sum()),
            "hidden_attribute_missing_rows": int(
                (~df_feats["hidden_attribute_present"].astype(bool)).sum()
            ),
        },
        "annotation_format_rows": {
            str(key): int(value)
            for key, value in df_feats["annotation_format"].value_counts().items()
        },
        "copied_images": copied,
        "dry_run": bool(args.dry_run),
        "actor_exclusion_policy": {
            "path": exclusion_policy_path,
            "keys": exclusions,
            "key_count": len(exclusions),
        },
    }
    if args.dry_run:
        print("[DRY RUN] validations passed; no outputs or images were written.")
        print(json.dumps(lineage, indent=2, ensure_ascii=False))
        return

    write_csv_atomic(df_clean, output_paths["clean_csv"])
    write_csv_atomic(df_feats, output_paths["feature_csv"])
    lineage["output_files"] = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in output_paths.items()
        if name != "lineage_json"
    }
    write_json_atomic(lineage, output_paths["lineage_json"])
    print("[SAVE] clean CSV:", output_paths["clean_csv"])
    print("[SAVE] feature CSV:", output_paths["feature_csv"])
    print("[SAVE] lineage JSON:", output_paths["lineage_json"])


if __name__ == "__main__":
    main()

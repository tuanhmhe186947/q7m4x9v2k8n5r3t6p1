"""End-to-end classification dataset orchestration."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from .classification_features import add_training_features, clean_merged_annotations
from .cvat_native import load_all_cvat_tasks, load_behaviors_from_project
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
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = CLASSIFICATION_PROCESSED_DIR / RUN_ID

OUT_CLEAN_CSV = RUN_OUTPUT_DIR / "behavior_clean_merged.csv"
OUT_FEATS_CSV = RUN_OUTPUT_DIR / "behavior_with_feats_rectROI.csv"
ROI_COCO_JSON = ANNOTATION_DIR / "roi" / "ROI_annotations.coco.json"

DROP_HIDDEN = False
BEHAVIORS = load_behaviors_from_project(DATA_EXPORT_ROOT / "project.json")


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
        print(f"[WARN] missing source images: {len(missing)}")
        print("First missing examples:", missing[:5])
    return copied


def main() -> None:
    """Run the full conversion pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    RAW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT    :", PROJECT_ROOT)
    print("DATA_EXPORT_ROOT:", DATA_EXPORT_ROOT)
    print("RAW_IMAGE_DIR   :", RAW_IMAGE_DIR)
    print("RUN_ID          :", RUN_ID)
    print("RUN_OUTPUT_DIR  :", RUN_OUTPUT_DIR)
    print("ROI_COCO_JSON   :", ROI_COCO_JSON)
    print("OUT_CLEAN_CSV   :", OUT_CLEAN_CSV)
    print("OUT_FEATS_CSV   :", OUT_FEATS_CSV)
    print("Behaviors       :", BEHAVIORS)

    if not DATA_EXPORT_ROOT.exists():
        raise FileNotFoundError(f"CVAT export folder not found: {DATA_EXPORT_ROOT}")

    df_raw = load_all_cvat_tasks(DATA_EXPORT_ROOT)
    print("Raw rows:", len(df_raw))

    df_clean = clean_merged_annotations(df_raw, BEHAVIORS, DROP_HIDDEN)
    df_clean.to_csv(OUT_CLEAN_CSV, index=False, encoding="utf-8")
    print("[SAVE] clean CSV:", OUT_CLEAN_CSV)

    copied = copy_annotated_images(df_clean, RAW_IMAGE_DIR)
    print(f"[COPY] copied/updated {copied} images into {RAW_IMAGE_DIR}")

    df_feats = add_training_features(df_clean, ROI_COCO_JSON)
    df_feats.to_csv(OUT_FEATS_CSV, index=False, encoding="utf-8")
    print("[SAVE] training feature CSV:", OUT_FEATS_CSV)
    print("Rows:", len(df_feats), "Images:", df_feats["img_name"].nunique())
    print("Fine behavior distribution:")
    print(df_feats["behavior"].value_counts())
    print("Coarse behavior distribution:")
    print(df_feats["behavior_coarse"].value_counts())
    print("ROI flag sums:")
    print(df_feats[["in_feeder", "in_drinker", "in_toy"]].sum())

    validate_project_outputs(df_feats, RAW_IMAGE_DIR)
    print(
        "[OK] CSV schema and copied images match project training loader expectations."
    )


if __name__ == "__main__":
    main()

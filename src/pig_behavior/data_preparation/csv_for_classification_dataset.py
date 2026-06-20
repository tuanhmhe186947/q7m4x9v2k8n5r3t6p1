"""Build classification training CSVs from CVAT native task exports.

The maintained input flow is:

``data/data/task_*`` + ``data/annotations/roi/ROI_annotations.coco.json``
-> ``data/raw/images_clean``
-> ``data/processed/classification/<run_id>/*.csv``.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BEHAVIORS = [
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
]

BEHAVIOR_TO_COARSE = {
    "lying": "resting",
    "sitting": "resting",
    "eat": "feeding",
    "drink": "feeding",
    "move": "locomotion",
    "stand": "locomotion",
    "explore": "locomotion",
    "playwithtoy": "locomotion",
    "social-nose": "social",
    "fight": "social",
}

PROJECT_REQUIRED_COLUMNS = {
    "img_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "behavior",
    "behavior_coarse",
    "hidden",
    "group_id",
    "in_feeder",
    "in_drinker",
    "in_toy",
    "speed_feat",
    "min_dist_other",
    "num_close_other",
}


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
PIG_LABEL_PREFIX = "pig"


def load_behaviors_from_project(project_json: Path) -> list[str]:
    """Read allowed Behavior values from CVAT project.json when available."""
    if not project_json.exists():
        return DEFAULT_BEHAVIORS
    with project_json.open("r", encoding="utf-8") as f:
        project = json.load(f)
    for label in project.get("labels", []):
        for attr in label.get("attributes", []):
            if attr.get("name") == "Behavior" and attr.get("values"):
                return list(attr["values"])
    return DEFAULT_BEHAVIORS


BEHAVIORS = load_behaviors_from_project(DATA_EXPORT_ROOT / "project.json")
BEHAVIOR_SET = set(BEHAVIORS)


def parse_attrs(attrs: Any) -> dict[str, str | None]:
    """Normalize CVAT attribute list/dict to ID, Behavior, Hidden fields."""
    parsed: dict[str, str | None] = {"ID": None, "Behavior": None, "Hidden": "No"}
    if isinstance(attrs, dict):
        for key in parsed:
            parsed[key] = attrs.get(key, parsed[key])
    elif isinstance(attrs, list):
        for attr in attrs:
            name = attr.get("name")
            if name in parsed:
                parsed[name] = attr.get("value")
    return parsed


def parse_burst_from_filename(img_name: str) -> tuple[str, int]:
    """Parse burst group and sequence order from frame file names."""
    stem = Path(img_name).stem
    parts = stem.split("_")
    order = 0
    if parts and parts[-1].startswith("k"):
        try:
            order = int(parts[-1][1:])
        except ValueError:
            order = 0
    if len(parts) >= 2 and parts[-2].startswith("f"):
        group_id = "_".join(parts[:-2])
    else:
        group_id = stem
    return group_id, order


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load CVAT imageset manifest entries that map frame index to image name."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    frames = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "name" in item and "extension" in item:
                frames.append(item)
    return frames


def frame_file_name(frame_info: dict[str, Any]) -> str:
    """Return image file name from a CVAT manifest row."""
    name = frame_info["name"]
    ext = frame_info.get("extension", "")
    return name if name.lower().endswith(ext.lower()) else f"{name}{ext}"


def load_cvat_task(task_dir: Path) -> pd.DataFrame:
    """Load one CVAT native task folder into a flat dataframe."""
    task_json_path = task_dir / "task.json"
    annotations_path = task_dir / "annotations.json"
    manifest_path = task_dir / "data" / "manifest.jsonl"

    if not task_json_path.exists() or not annotations_path.exists():
        print(f"[WARN] skip incomplete task: {task_dir}")
        return pd.DataFrame()

    with task_json_path.open("r", encoding="utf-8") as f:
        task_json = json.load(f)
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    manifest = load_manifest(manifest_path)
    subset = task_json.get("subset") or task_dir.name
    rows: list[dict[str, Any]] = []

    for annotation_obj in annotations:
        for shape in annotation_obj.get("shapes", []):
            if shape.get("type") != "rectangle":
                continue
            if shape.get("outside") is True:
                continue
            label = str(shape.get("label", ""))
            if PIG_LABEL_PREFIX and not label.lower().startswith(PIG_LABEL_PREFIX):
                continue

            frame_idx = int(shape.get("frame", -1))
            if frame_idx < 0 or frame_idx >= len(manifest):
                print(f"[WARN] skip invalid frame {frame_idx} in {task_dir.name}")
                continue

            points = shape.get("points", [])
            if len(points) != 4:
                continue
            x1, y1, x2, y2 = map(float, points)
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            frame_info = manifest[frame_idx]
            img_name = frame_file_name(frame_info)
            group_id, order = parse_burst_from_filename(img_name)
            attrs = parse_attrs(shape.get("attributes", []))

            rows.append(
                {
                    "task": task_dir.name,
                    "subset": subset,
                    "frame": frame_idx,
                    "img_name": img_name,
                    "image_path": str(task_dir / "data" / img_name),
                    "width": frame_info.get("width"),
                    "height": frame_info.get("height"),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "pig_id": attrs["ID"],
                    "behavior": attrs["Behavior"],
                    "hidden": attrs["Hidden"] or "No",
                    "group_id": group_id,
                    "order": order,
                    "category_name": label,
                    "source": shape.get("source"),
                }
            )

    df = pd.DataFrame(rows)
    print(f"[LOAD] {task_dir.name}: {len(df)} boxes | subset={subset!r}")
    return df


def load_all_cvat_tasks(export_root: Path) -> pd.DataFrame:
    """Load all task_* folders under the CVAT export root."""
    task_dirs = sorted(p for p in export_root.glob("task_*") if p.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No task_* folders found under {export_root}")

    frames = [load_cvat_task(task_dir) for task_dir in task_dirs]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        raise ValueError("No CVAT rectangle annotations were loaded.")
    return df


def majority_fix_in_burst(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize behavior per (burst group, pig ID) by majority vote."""
    df = df.copy()
    priority = {behavior: idx for idx, behavior in enumerate(BEHAVIORS)}

    def majority_behavior(sub: pd.DataFrame) -> str | None:
        counts = sub["behavior"].dropna().value_counts()
        if counts.empty:
            return None
        max_count = counts.max()
        candidates = counts[counts == max_count].index.tolist()
        return sorted(candidates, key=lambda b: priority.get(b, 999))[0]

    majority_map: dict[tuple[Any, Any], str | None] = {}
    mixed = 0
    for key, sub in df.groupby(["group_id", "pig_id"], dropna=False):
        unique_behaviors = sub["behavior"].dropna().unique()
        if len(unique_behaviors) > 1:
            mixed += 1
        majority_map[key] = majority_behavior(sub)

    print(f"[MAJ] groups with mixed behaviors: {mixed}/{len(majority_map)}")
    df["behavior"] = [
        majority_map.get((row.group_id, row.pig_id), row.behavior)
        for row in df.itertuples(index=False)
    ]
    return df


def clean_merged_annotations(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filter and normalize raw CVAT rows."""
    df = df_raw.copy()
    before = len(df)
    df = df[df["behavior"].isin(BEHAVIOR_SET)].copy()
    print(f"[FILTER] invalid behavior: {before - len(df)} boxes")

    before = len(df)
    df = df.dropna(
        subset=["img_name", "pig_id", "behavior", "x1", "y1", "x2", "y2"]
    )
    print(f"[FILTER] missing required values: {before - len(df)} boxes")

    if DROP_HIDDEN:
        before = len(df)
        df = df[df["hidden"].astype(str).str.lower() != "yes"].copy()
        print(f"[FILTER] Hidden == Yes: {before - len(df)} boxes")

    invalid = (df["x2"] <= df["x1"]) | (df["y2"] <= df["y1"])
    if invalid.any():
        print(f"[FILTER] invalid bbox before clamp: {int(invalid.sum())} boxes")
        df = df[~invalid].copy()

    for col in ["width", "height"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["x1"] = df.apply(
        lambda r: max(0.0, min(float(r.x1), float(r.width)))
        if pd.notna(r.width)
        else r.x1,
        axis=1,
    )
    df["x2"] = df.apply(
        lambda r: max(0.0, min(float(r.x2), float(r.width)))
        if pd.notna(r.width)
        else r.x2,
        axis=1,
    )
    df["y1"] = df.apply(
        lambda r: max(0.0, min(float(r.y1), float(r.height)))
        if pd.notna(r.height)
        else r.y1,
        axis=1,
    )
    df["y2"] = df.apply(
        lambda r: max(0.0, min(float(r.y2), float(r.height)))
        if pd.notna(r.height)
        else r.y2,
        axis=1,
    )

    invalid_after = (df["x2"] <= df["x1"]) | (df["y2"] <= df["y1"])
    if invalid_after.any():
        print(f"[FILTER] invalid bbox after clamp: {int(invalid_after.sum())} boxes")
        df = df[~invalid_after].copy()

    df = majority_fix_in_burst(df)
    df = df.sort_values(["group_id", "order", "pig_id", "task", "frame"])
    df = df.reset_index(drop=True)

    print("===== SUMMARY CLEAN =====")
    print("Rows:", len(df))
    print("Images:", df["img_name"].nunique())
    print("Groups:", df["group_id"].nunique())
    print("Pig IDs:", sorted(df["pig_id"].dropna().unique().tolist()))
    print("Behavior distribution:")
    print(df["behavior"].value_counts())
    return df


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


def load_roi_boxes(
    coco_path: Path,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Load feeder/drinker/toy rectangular ROI boxes from COCO annotations."""
    if not coco_path.exists():
        print(f"[WARN] ROI COCO not found, ROI flags will be zero: {coco_path}")
        return {"feeder": [], "drinker": [], "toy": []}

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    categories = {cat["id"]: cat.get("name", "") for cat in coco.get("categories", [])}
    roi_boxes = {"feeder": [], "drinker": [], "toy": []}

    for ann in coco.get("annotations", []):
        name = categories.get(ann.get("category_id"), "").lower()
        target = None
        if "feeder" in name:
            target = "feeder"
        elif "drinker" in name or "drink" in name:
            target = "drinker"
        elif "toy" in name:
            target = "toy"
        if target is None:
            continue

        if "bbox" in ann and len(ann["bbox"]) == 4:
            x, y, w, h = map(float, ann["bbox"])
            roi_boxes[target].append((x, y, x + w, y + h))
        elif "segmentation" in ann and ann["segmentation"]:
            pts = np.array(ann["segmentation"][0], dtype=float).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            roi_boxes[target].append((float(x1), float(y1), float(x2), float(y2)))

    print("[ROI] boxes:", {k: len(v) for k, v in roi_boxes.items()})
    return roi_boxes


def rect_intersect_area(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Return rectangle intersection area."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def add_training_features(df_clean: pd.DataFrame, roi_coco_path: Path) -> pd.DataFrame:
    """Add columns required by pig_behavior.data_loader."""
    df = df_clean.copy()
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    df["bw"] = df["x2"] - df["x1"]
    df["bh"] = df["y2"] - df["y1"]

    df["cx_n"] = df["cx"] / df["width"].replace(0, np.nan)
    df["cy_n"] = df["cy"] / df["height"].replace(0, np.nan)
    df["bw_n"] = df["bw"] / df["width"].replace(0, np.nan)
    df["bh_n"] = df["bh"] / df["height"].replace(0, np.nan)

    roi_boxes = load_roi_boxes(roi_coco_path)
    for col in ["in_feeder", "in_drinker", "in_toy"]:
        df[col] = 0

    for idx, row in df.iterrows():
        pig_box = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
        df.at[idx, "in_feeder"] = int(
            any(rect_intersect_area(pig_box, box) > 0 for box in roi_boxes["feeder"])
        )
        df.at[idx, "in_drinker"] = int(
            any(rect_intersect_area(pig_box, box) > 0 for box in roi_boxes["drinker"])
        )
        df.at[idx, "in_toy"] = int(
            any(rect_intersect_area(pig_box, box) > 0 for box in roi_boxes["toy"])
        )

    diag = np.sqrt(df["width"].astype(float) ** 2 + df["height"].astype(float) ** 2)
    diag = diag.replace(0, np.nan)
    df["speed_feat"] = 0.0
    for (_gid, _pid), sub in df.groupby(["group_id", "pig_id"], dropna=False):
        sub = sub.sort_values("order")
        prev = None
        for idx, row in sub.iterrows():
            cur = np.array([float(row.cx), float(row.cy)])
            if prev is not None:
                df.at[idx, "speed_feat"] = float(
                    np.linalg.norm(cur - prev) / diag.loc[idx]
                )
            prev = cur

    df["min_dist_other"] = 1.0
    df["num_close_other"] = 0.0
    for (_img_name, _group_order), sub in df.groupby(
        ["img_name", "order"],
        dropna=False,
    ):
        centers = sub[["cx_n", "cy_n"]].to_numpy(dtype=float)
        idxs = sub.index.to_list()
        for pos, idx in enumerate(idxs):
            if len(idxs) <= 1:
                continue
            dists = np.linalg.norm(centers - centers[pos], axis=1)
            dists = dists[dists > 0]
            if len(dists) == 0:
                continue
            df.at[idx, "min_dist_other"] = float(dists.min())
            df.at[idx, "num_close_other"] = float((dists < 0.15).sum())

    df["behavior_coarse"] = df["behavior"].map(BEHAVIOR_TO_COARSE)
    before = len(df)
    df = df[df["behavior_coarse"].notna()].copy()
    print(f"[COARSE] kept {len(df)}/{before} rows")

    required_numeric = [
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "speed_feat",
        "min_dist_other",
        "num_close_other",
        "in_feeder",
        "in_drinker",
        "in_toy",
    ]
    df[required_numeric] = df[required_numeric].fillna(0.0)
    return df.reset_index(drop=True)


def validate_project_outputs(df_feats: pd.DataFrame) -> None:
    """Validate CSV schema and copied image files before finishing."""
    missing = sorted(PROJECT_REQUIRED_COLUMNS.difference(df_feats.columns))
    if missing:
        raise ValueError(f"Missing project training columns: {missing}")

    missing_images = [
        name
        for name in df_feats["img_name"].drop_duplicates()
        if not (RAW_IMAGE_DIR / name).exists()
    ]
    if missing_images:
        raise FileNotFoundError(
            "CSV references missing images under "
            f"{RAW_IMAGE_DIR}: {missing_images[:10]}"
        )


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

    df_clean = clean_merged_annotations(df_raw)
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

    validate_project_outputs(df_feats)
    print(
        "[OK] CSV schema and copied images match project training loader expectations."
    )


if __name__ == "__main__":
    main()

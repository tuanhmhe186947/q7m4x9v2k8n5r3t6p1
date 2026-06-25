"""Behavior annotation cleaning and spatial feature extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def majority_fix_in_burst(df: pd.DataFrame, behaviors: list[str]) -> pd.DataFrame:
    """Normalize behavior per (burst group, pig ID) by majority vote."""
    df = df.copy()
    priority = {behavior: idx for idx, behavior in enumerate(behaviors)}

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


def clean_merged_annotations(
    df_raw: pd.DataFrame,
    behaviors: list[str],
    drop_hidden: bool = False,
) -> pd.DataFrame:
    """Filter and normalize raw CVAT rows."""
    df = df_raw.copy()
    behavior_set = set(behaviors)

    before = len(df)
    df = df[df["behavior"].isin(behavior_set)].copy()
    print(f"[FILTER] invalid behavior: {before - len(df)} boxes")

    before = len(df)
    df = df.dropna(
        subset=["img_name", "pig_id", "behavior", "x1", "y1", "x2", "y2"]
    )
    print(f"[FILTER] missing required values: {before - len(df)} boxes")

    if drop_hidden:
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

    df = majority_fix_in_burst(df, behaviors)
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

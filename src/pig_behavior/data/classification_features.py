"""Behavior annotation cleaning and spatial feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

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

AUTHORITY_POLICY = "first_task_frame_per_group_pig"
ACTOR_KEY = ["group_id", "pig_id"]
ACTOR_SLOT_KEY = ["group_id", "pig_id", "order"]
EXPECTED_ORDERS = tuple(range(6))

def apply_first_task_frame_behavior_authority(
    df: pd.DataFrame,
    behaviors: list[str],
) -> pd.DataFrame:
    """Map each actor's first displayed-frame behavior to all six anchors."""
    required = {
        "group_id",
        "pig_id",
        "order",
        "frame",
        "behavior",
        "is_burst_first_task_frame",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing behavior-authority columns: {missing}")

    out = df.copy()
    out["order"] = pd.to_numeric(out["order"], errors="coerce")
    invalid_order = out["order"].isna() | ~out["order"].isin(EXPECTED_ORDERS)
    if invalid_order.any():
        sample = out.loc[
            invalid_order,
            ["group_id", "pig_id", "frame", "order"],
        ].head(10)
        raise ValueError(
            "invalid_legacy_anchor_order="
            f"{int(invalid_order.sum())}; sample={sample.to_dict(orient='records')}"
        )
    out["order"] = out["order"].astype(int)

    duplicate_slots = out.duplicated(ACTOR_SLOT_KEY, keep=False)
    if duplicate_slots.any():
        sample = out.loc[
            duplicate_slots,
            [*ACTOR_SLOT_KEY, "task", "frame", "behavior"],
        ].head(10)
        raise ValueError(
            "duplicate_actor_slot="
            f"{int(duplicate_slots.sum())}; sample={sample.to_dict(orient='records')}"
        )

    coverage = out.groupby(ACTOR_KEY, dropna=False)["order"].agg(
        lambda values: tuple(sorted(set(int(value) for value in values)))
    )
    incomplete = coverage[coverage.map(lambda orders: orders != EXPECTED_ORDERS)]
    if not incomplete.empty:
        sample = [
            {
                "group_id": str(group_id),
                "pig_id": str(pig_id),
                "orders": list(orders),
            }
            for (group_id, pig_id), orders in incomplete.head(10).items()
        ]
        raise ValueError(
            f"incomplete_actor_anchor_sets={len(incomplete)}; sample={sample}"
        )

    authority = out.loc[out["is_burst_first_task_frame"].eq(True)].copy()
    authority_counts = authority.groupby(ACTOR_KEY, dropna=False).size()
    all_actor_keys = pd.MultiIndex.from_frame(out[ACTOR_KEY].drop_duplicates())
    missing_authority = all_actor_keys.difference(authority_counts.index)
    duplicate_authority = authority_counts[authority_counts.ne(1)]
    if len(missing_authority) or not duplicate_authority.empty:
        raise ValueError(
            "invalid_behavior_authority_coverage="
            f"missing={len(missing_authority)},"
            f"non_unique={len(duplicate_authority)},"
            f"missing_sample={list(missing_authority[:10])},"
            f"non_unique_sample={list(duplicate_authority.index[:10])}"
        )

    valid_behaviors = set(behaviors)
    authority["behavior"] = authority["behavior"].astype("string").str.strip()
    invalid_authority = ~authority["behavior"].isin(valid_behaviors)
    if invalid_authority.any():
        sample = authority.loc[
            invalid_authority,
            [*ACTOR_KEY, "frame", "order", "behavior"],
        ].head(10)
        raise ValueError(
            "invalid_first_frame_behavior="
            f"{int(invalid_authority.sum())}; "
            f"sample={sample.to_dict(orient='records')}"
        )

    authority_table = authority[
        [*ACTOR_KEY, "behavior", "frame", "order"]
    ].rename(
        columns={
            "behavior": "behavior_authority_value",
            "frame": "behavior_authority_task_frame",
            "order": "behavior_authority_slot",
        }
    )
    out["behavior_before_authority"] = out["behavior"].astype("string").str.strip()
    before_rows = len(out)
    out = out.merge(
        authority_table,
        on=ACTOR_KEY,
        how="left",
        validate="many_to_one",
    )
    if len(out) != before_rows:
        raise ValueError("behavior_authority_merge_changed_row_count")

    out["behavior_disagrees_with_authority"] = out[
        "behavior_before_authority"
    ].ne(out["behavior_authority_value"])
    out["behavior"] = out["behavior_authority_value"]
    out["behavior_authority_policy"] = AUTHORITY_POLICY
    return out.drop(columns=["behavior_authority_value"])


def clean_merged_annotations(
    df_raw: pd.DataFrame,
    behaviors: list[str],
    drop_hidden: bool = False,
) -> pd.DataFrame:
    """Validate CVAT rows and apply first-task-frame behavior authority."""
    df = df_raw.copy()
    behavior_set = set(behaviors)

    if drop_hidden:
        raise ValueError(
            "drop_hidden is forbidden for canonical six-anchor provenance"
        )

    required_values = [
        "img_name",
        "group_id",
        "pig_id",
        "behavior",
        "hidden",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]
    missing_values = df[required_values].isna().any(axis=1)
    if missing_values.any():
        sample = df.loc[
            missing_values,
            ["task", "frame", "img_name", "group_id", "pig_id"],
        ].head(10)
        raise ValueError(
            "missing_required_annotation_values="
            f"{int(missing_values.sum())}; "
            f"sample={sample.to_dict(orient='records')}"
        )

    df["behavior"] = df["behavior"].astype("string").str.strip()
    invalid_behavior = ~df["behavior"].isin(behavior_set)
    if invalid_behavior.any():
        values = sorted(df.loc[invalid_behavior, "behavior"].astype(str).unique())
        raise ValueError(
            f"invalid_behavior_rows={int(invalid_behavior.sum())}; values={values}"
        )

    df["hidden"] = df["hidden"].astype("string").str.strip()
    invalid_hidden = ~df["hidden"].isin({"Yes", "No"})
    if invalid_hidden.any():
        values = sorted(df.loc[invalid_hidden, "hidden"].astype(str).unique())
        raise ValueError(
            f"invalid_hidden_rows={int(invalid_hidden.sum())}; values={values}"
        )

    for col in ["width", "height", "x1", "y1", "x2", "y2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_numeric = df[
        ["width", "height", "x1", "y1", "x2", "y2"]
    ].isna().any(axis=1)
    invalid_bbox = (
        invalid_numeric
        | df["width"].le(0)
        | df["height"].le(0)
        | df["x2"].le(df["x1"])
        | df["y2"].le(df["y1"])
    )
    if invalid_bbox.any():
        sample = df.loc[
            invalid_bbox,
            ["task", "frame", "img_name", "pig_id", "x1", "y1", "x2", "y2"],
        ].head(10)
        raise ValueError(
            f"invalid_bbox_rows={int(invalid_bbox.sum())}; "
            f"sample={sample.to_dict(orient='records')}"
        )

    df["bbox_outside_image"] = (
        df["x1"].lt(0)
        | df["y1"].lt(0)
        | df["x2"].gt(df["width"])
        | df["y2"].gt(df["height"])
    )
    df = apply_first_task_frame_behavior_authority(df, behaviors)
    df = df.sort_values(["group_id", "order", "pig_id", "task", "frame"])
    df = df.reset_index(drop=True)

    print("===== SUMMARY VALIDATED =====")
    print("Rows:", len(df))
    print("Images:", df["img_name"].nunique())
    print("Groups:", df["group_id"].nunique())
    print("Pig IDs:", sorted(df["pig_id"].dropna().unique().tolist()))
    print("Behavior authority:", AUTHORITY_POLICY)
    print(
        "Rows disagreeing with first-frame authority:",
        int(df["behavior_disagrees_with_authority"].sum()),
    )
    print("Bboxes outside image bounds:", int(df["bbox_outside_image"].sum()))
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

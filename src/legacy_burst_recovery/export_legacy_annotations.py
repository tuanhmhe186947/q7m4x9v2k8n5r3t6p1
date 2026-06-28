import argparse
import json
import re
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import pandas as pd

BEHAVIOR_CLASSES = [
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


def parse_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def normalize_source_video_key(path_value: str) -> str:
    p = str(path_value).replace("\\", "/").lower()
    m = re.search(r"(pigs\d{6})/pigs\d{6}/(\d+)/color\.mp4", p)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    m = re.search(r"(pigs\d{6})/(\d+)/color\.mp4", p)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    return ""


def safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def build_frame_object_csv(
    dense_df: pd.DataFrame,
    image_width: int,
    image_height: int,
    training_only: bool,
) -> pd.DataFrame:
    df = dense_df.copy()

    if training_only and "include_in_training" in df.columns:
        df = df[parse_bool_series(df["include_in_training"])].copy()

    required = ["group_id", "pig_id", "frame_index", "x1", "y1", "x2", "y2", "behavior"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dense CSV: {missing}")

    source_col = None
    for c in ["source_video_resolved", "color_video_path", "source_video_original", "video_final"]:
        if c in df.columns:
            source_col = c
            break

    if source_col is None:
        df["source_video_key"] = ""
    else:
        df["source_video_key"] = df[source_col].map(normalize_source_video_key)

    df["image_key"] = (
        df["source_video_key"].astype(str)
        + "::"
        + df["group_id"].astype(str)
        + "::f"
        + df["frame_index"].astype(int).astype(str).str.zfill(6)
    )

    df["image_name"] = (
        df["source_video_key"].astype(str).str.replace("/", "_", regex=False)
        + "__"
        + df["group_id"].astype(str)
        + "__f"
        + df["frame_index"].astype(int).astype(str).str.zfill(6)
        + ".jpg"
    )

    df["image_width"] = image_width
    df["image_height"] = image_height

    # Clamp bbox to image bounds
    df["x1"] = df["x1"].astype(float).clip(0, image_width - 1)
    df["x2"] = df["x2"].astype(float).clip(0, image_width - 1)
    df["y1"] = df["y1"].astype(float).clip(0, image_height - 1)
    df["y2"] = df["y2"].astype(float).clip(0, image_height - 1)

    df["bbox_w"] = (df["x2"] - df["x1"]).clip(lower=0)
    df["bbox_h"] = (df["y2"] - df["y1"]).clip(lower=0)
    df["bbox_area"] = df["bbox_w"] * df["bbox_h"]

    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    df["cx_n"] = df["cx"] / image_width
    df["cy_n"] = df["cy"] / image_height
    df["bw_n"] = df["bbox_w"] / image_width
    df["bh_n"] = df["bbox_h"] / image_height
    df["area_n"] = df["bbox_area"] / float(image_width * image_height)

    # Một ảnh có nhiều pig_id
    df["object_id_in_image"] = (
        df.groupby("image_key")
        .cumcount()
        .astype(int)
    )

    preferred_cols = [
        "image_key",
        "image_name",
        "image_width",
        "image_height",
        "source_video_key",
        "group_id",
        "sample_id",
        "tracklet_id",
        "pig_id",
        "behavior",
        "hidden",
        "frame_index",
        "timestamp_sec",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "cx",
        "cy",
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "area_n",
        "bbox_source",
        "tracking_status",
        "qa_status",
        "include_in_training",
        "training_tier",
        "crop_path",
        "source_video_resolved",
        "color_video_path",
        "depth_video_path",
        "times_txt_path",
        "legacy_gt_mode",
        "legacy_gt_support_frames",
        "mask_filter_applied",
    ]
    cols = [c for c in preferred_cols if c in df.columns]

    df = df.sort_values(["source_video_key", "group_id", "frame_index", "pig_id"]).copy()
    return df[cols]


def export_coco(frame_df: pd.DataFrame, out_json: Path) -> None:
    categories = [
        {"id": 1, "name": "pig", "supercategory": "animal"},
    ]

    behavior_to_id = {b: i + 1 for i, b in enumerate(BEHAVIOR_CLASSES)}

    images = []
    annotations = []

    image_id_map = {}
    ann_id = 1

    unique_images = (
        frame_df[
            [
                "image_key",
                "image_name",
                "image_width",
                "image_height",
                "source_video_key",
                "group_id",
                "frame_index",
            ]
        ]
        .drop_duplicates()
        .sort_values(["source_video_key", "group_id", "frame_index"])
    )

    for image_id, row in enumerate(unique_images.itertuples(index=False), start=1):
        image_id_map[row.image_key] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": row.image_name,
                "width": int(row.image_width),
                "height": int(row.image_height),
                "source_video_key": row.source_video_key,
                "group_id": row.group_id,
                "frame_index": int(row.frame_index),
            }
        )

    for row in frame_df.itertuples(index=False):
        x1 = safe_float(row.x1)
        y1 = safe_float(row.y1)
        w = safe_float(row.bbox_w)
        h = safe_float(row.bbox_h)

        if w <= 0 or h <= 0:
            continue

        behavior = str(getattr(row, "behavior", ""))
        pig_id = str(getattr(row, "pig_id", ""))

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id_map[row.image_key],
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "attributes": {
                    "pig_id": pig_id,
                    "behavior": behavior,
                    "behavior_id": behavior_to_id.get(behavior, -1),
                    "hidden": str(getattr(row, "hidden", "")),
                    "tracklet_id": str(getattr(row, "tracklet_id", "")),
                    "sample_id": str(getattr(row, "sample_id", "")),
                    "bbox_source": str(getattr(row, "bbox_source", "")),
                    "tracking_status": str(getattr(row, "tracking_status", "")),
                    "qa_status": str(getattr(row, "qa_status", "")),
                    "include_in_training": str(getattr(row, "include_in_training", "")),
                    "training_tier": str(getattr(row, "training_tier", "")),
                },
            }
        )
        ann_id += 1

    coco = {
        "info": {
            "description": "Legacy recovered pig behavior annotations",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_json.write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")


def export_cvat_1_1(frame_df: pd.DataFrame, out_xml: Path) -> None:
    root = Element("annotations")
    SubElement(root, "version").text = "1.1"

    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    SubElement(task, "name").text = "legacy_recovered_pig_behavior"
    labels = SubElement(task, "labels")

    label = SubElement(labels, "label")
    SubElement(label, "name").text = "Pig"

    attrs = SubElement(label, "attributes")

    attr_id = SubElement(attrs, "attribute")
    SubElement(attr_id, "name").text = "ID"
    SubElement(attr_id, "mutable").text = "False"
    SubElement(attr_id, "input_type").text = "select"
    SubElement(attr_id, "default_value").text = "ID_1"
    SubElement(attr_id, "values").text = "\n".join([f"ID_{i}" for i in range(1, 9)])

    attr_behavior = SubElement(attrs, "attribute")
    SubElement(attr_behavior, "name").text = "Behavior"
    SubElement(attr_behavior, "mutable").text = "False"
    SubElement(attr_behavior, "input_type").text = "select"
    SubElement(attr_behavior, "default_value").text = BEHAVIOR_CLASSES[0]
    SubElement(attr_behavior, "values").text = "\n".join(BEHAVIOR_CLASSES)

    attr_hidden = SubElement(attrs, "attribute")
    SubElement(attr_hidden, "name").text = "Hidden"
    SubElement(attr_hidden, "mutable").text = "False"
    SubElement(attr_hidden, "input_type").text = "select"
    SubElement(attr_hidden, "default_value").text = "No"
    SubElement(attr_hidden, "values").text = "Yes\nNo"

    image_rows = (
        frame_df[
            [
                "image_key",
                "image_name",
                "image_width",
                "image_height",
                "source_video_key",
                "group_id",
                "frame_index",
            ]
        ]
        .drop_duplicates()
        .sort_values(["source_video_key", "group_id", "frame_index"])
        .reset_index(drop=True)
    )

    image_id_map = {}
    for image_id, row in image_rows.iterrows():
        image_id_map[row["image_key"]] = image_id
        image_el = SubElement(
            root,
            "image",
            {
                "id": str(image_id),
                "name": str(row["image_name"]),
                "width": str(int(row["image_width"])),
                "height": str(int(row["image_height"])),
            },
        )

        g = frame_df[frame_df["image_key"].eq(row["image_key"])].sort_values("pig_id")

        for obj in g.itertuples(index=False):
            box = SubElement(
                image_el,
                "box",
                {
                    "label": "Pig",
                    "source": "manual",
                    "occluded": "0",
                    "xtl": f"{safe_float(obj.x1):.2f}",
                    "ytl": f"{safe_float(obj.y1):.2f}",
                    "xbr": f"{safe_float(obj.x2):.2f}",
                    "ybr": f"{safe_float(obj.y2):.2f}",
                    "z_order": "0",
                },
            )

            attr = SubElement(box, "attribute", {"name": "ID"})
            attr.text = str(getattr(obj, "pig_id", ""))

            attr = SubElement(box, "attribute", {"name": "Behavior"})
            attr.text = str(getattr(obj, "behavior", ""))

            hidden = str(getattr(obj, "hidden", "No"))
            if hidden.lower() in ["nan", "none", ""]:
                hidden = "No"
            attr = SubElement(box, "attribute", {"name": "Hidden"})
            attr.text = hidden

    tree = ElementTree(root)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense-csv",
        default=r"C:\Users\ironh\Downloads\PIG_Behavior_Project\data\raw\legacy_full_multigt_masked_nodup\legacy_dense_tracklet_map.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup\exports",
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Export all dense rows, not only include_in_training=True.",
    )
    args = parser.parse_args()

    dense_csv = Path(args.dense_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dense = pd.read_csv(dense_csv, low_memory=False)

    frame_df = build_frame_object_csv(
        dense_df=dense,
        image_width=args.image_width,
        image_height=args.image_height,
        training_only=not args.include_all,
    )

    out_frame_csv = out_dir / "legacy_frame_object_annotations.csv"
    out_coco = out_dir / "legacy_annotations_coco.json"
    out_cvat = out_dir / "legacy_annotations_cvat_1_1.xml"

    frame_df.to_csv(out_frame_csv, index=False)
    export_coco(frame_df, out_coco)
    export_cvat_1_1(frame_df, out_cvat)

    print("saved:", out_frame_csv)
    print("saved:", out_coco)
    print("saved:", out_cvat)
    print("object rows=", len(frame_df))
    print("images=", frame_df["image_key"].nunique())
    print("tracklets=", frame_df["tracklet_id"].nunique() if "tracklet_id" in frame_df.columns else "NO")
    print("pig boxes per image:")
    print(frame_df.groupby("image_key")["pig_id"].nunique().value_counts().sort_index().to_string())
    print("\nbehavior distribution:")
    print(frame_df.drop_duplicates("tracklet_id")["behavior"].value_counts().to_string())


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_LEGACY_GT_COLUMNS = {"group_id", "pig_id", "frame_index", "x1", "y1", "x2", "y2"}

LegacyGtMap = dict[tuple[str, str], dict[int, dict[str, Any]]]


def _parse_frame_list(value: object) -> list[int]:
    if value is None or pd.isna(value):
        return []
    frames: list[int] = []
    for part in str(value).replace(",", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            frames.append(int(float(part)))
        except ValueError:
            return []
    return frames


def _join_ints(values: list[int]) -> str:
    return "|".join(str(int(v)) for v in values)


def load_legacy_gt_bboxes(path: Path | None) -> tuple[LegacyGtMap, pd.DataFrame]:
    if path is None:
        return {}, pd.DataFrame(
            columns=[
                "group_id",
                "pig_id",
                "sample_id",
                "expected_gt_frames",
                "loaded_gt_frames",
                "legacy_gt_support_frames",
                "missing_gt_frames",
                "duplicate_gt_frames",
                "bbox_valid_rows",
                "bbox_invalid_rows",
                "qa_status",
                "qa_notes",
            ]
        )

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_LEGACY_GT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"--legacy-burst-bbox-csv is missing required columns: {missing}")

    df = df.copy()
    for column in ["frame_index", "x1", "y1", "x2", "y2"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_bbox = (
        df["frame_index"].isna()
        | df["x1"].isna()
        | df["y1"].isna()
        | df["x2"].isna()
        | df["y2"].isna()
        | df["x2"].le(df["x1"])
        | df["y2"].le(df["y1"])
    )
    df["_bbox_valid"] = ~invalid_bbox
    df["_group_key"] = list(zip(df["group_id"].astype(str), df["pig_id"].astype(str), strict=False))

    legacy_gt_map: LegacyGtMap = {}
    audit_rows: list[dict[str, object]] = []

    for (group_id_key, pig_id_key), group in df.groupby("_group_key", sort=False, dropna=False):
        valid = group[group["_bbox_valid"]].copy()
        duplicate_frame_values = (
            group.loc[group["frame_index"].notna() & group.duplicated("frame_index", keep=False), "frame_index"]
            .astype(int)
            .unique()
            .tolist()
        )
        duplicate_frame_values = sorted(duplicate_frame_values)
        duplicate_order_values: list[int] = []
        if "legacy_order" in group.columns:
            legacy_order = pd.to_numeric(group["legacy_order"], errors="coerce")
            duplicate_order_values = sorted(
                legacy_order[legacy_order.duplicated(keep=False)].dropna().astype(int).unique().tolist()
            )

        per_frame: dict[int, dict[str, Any]] = {}
        for _, row in valid.drop_duplicates("frame_index", keep="last").iterrows():
            frame_index = int(row["frame_index"])
            legacy_order = (
                None
                if "legacy_order" not in row or pd.isna(row.get("legacy_order"))
                else int(row["legacy_order"])
            )
            img_name = "" if "img_name" not in row or pd.isna(row.get("img_name")) else str(row["img_name"])
            per_frame[frame_index] = {
                "bbox": (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                "legacy_order": legacy_order,
                "img_name": img_name,
            }
        if per_frame:
            legacy_gt_map[(str(group_id_key), str(pig_id_key))] = per_frame

        support_frames = sorted(per_frame)
        expected_frames: list[int] = []
        if "frames" in group.columns:
            for value in group["frames"]:
                expected_frames = _parse_frame_list(value)
                if expected_frames:
                    break
        missing_frames = sorted(set(expected_frames) - set(support_frames)) if expected_frames else []

        notes: list[str] = []
        if len(support_frames) < 6:
            notes.append("missing_legacy_gt_frames")
        if duplicate_frame_values:
            notes.append("duplicate_frame_index_rows")
        if duplicate_order_values:
            notes.append("duplicate_legacy_order_rows")
        if int((~group["_bbox_valid"]).sum()) > 0:
            notes.append("invalid_bbox_rows")
        if expected_frames and missing_frames:
            notes.append("expected_frames_missing")

        audit_rows.append(
            {
                "group_id": group["group_id"].iloc[0],
                "pig_id": group["pig_id"].iloc[0],
                "sample_id": group["sample_id"].iloc[0] if "sample_id" in group.columns else "",
                "expected_gt_frames": _join_ints(expected_frames) if expected_frames else 6,
                "loaded_gt_frames": int(len(support_frames)),
                "legacy_gt_support_frames": _join_ints(support_frames),
                "missing_gt_frames": _join_ints(missing_frames),
                "duplicate_gt_frames": _join_ints(duplicate_frame_values),
                "bbox_valid_rows": int(group["_bbox_valid"].sum()),
                "bbox_invalid_rows": int((~group["_bbox_valid"]).sum()),
                "qa_status": "review" if notes else "ok",
                "qa_notes": ";".join(notes),
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    return legacy_gt_map, audit_df

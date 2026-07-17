from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_LEGACY_GT_SLOTS = frozenset(range(6))
REQUIRED_LEGACY_GT_COLUMNS = {
    "group_id",
    "pig_id",
    "frame_index",
    "legacy_order",
    "frames",
    "behavior",
    "hidden",
    "x1",
    "y1",
    "x2",
    "y2",
}

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


def _normalize_hidden_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "y"}:
        return "Yes"
    if text in {"no", "false", "0", "n"}:
        return "No"
    return None


def _truthy(values: pd.Series) -> pd.Series:
    return values.fillna(False).map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
    )


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
                "legacy_gt_anchor_slots",
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
    for column in ["frame_index", "legacy_order", "x1", "y1", "x2", "y2"]:
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
    df["_hidden_normalized"] = df["hidden"].map(_normalize_hidden_or_none)
    df["_group_key"] = list(zip(df["group_id"].astype(str), df["pig_id"].astype(str), strict=False))

    legacy_gt_map: LegacyGtMap = {}
    audit_rows: list[dict[str, object]] = []

    for (group_id_key, pig_id_key), group in df.groupby("_group_key", sort=False, dropna=False):
        duplicate_frame_values = (
            group.loc[group["frame_index"].notna() & group.duplicated("frame_index", keep=False), "frame_index"]
            .astype(int)
            .unique()
            .tolist()
        )
        duplicate_frame_values = sorted(duplicate_frame_values)
        duplicate_order_values = sorted(
            group.loc[
                group["legacy_order"].notna()
                & group.duplicated("legacy_order", keep=False),
                "legacy_order",
            ]
            .astype(int)
            .unique()
            .tolist()
        )
        expected_frame_variants = {
            tuple(parsed)
            for parsed in group["frames"].map(_parse_frame_list)
            if parsed
        }
        expected_frames = (
            list(next(iter(expected_frame_variants)))
            if len(expected_frame_variants) == 1
            else []
        )
        observed_frames = sorted(
            group["frame_index"].dropna().astype(int).unique().tolist()
        )
        observed_slots = sorted(
            group["legacy_order"].dropna().astype(int).unique().tolist()
        )
        support_frames = observed_frames
        missing_frames = sorted(set(expected_frames) - set(support_frames)) if expected_frames else []

        notes: list[str] = []
        if len(group) != len(EXPECTED_LEGACY_GT_SLOTS):
            notes.append("anchor_row_count_not_six")
        if set(observed_slots) != EXPECTED_LEGACY_GT_SLOTS:
            notes.append("anchor_slots_not_0_to_5")
        if len(support_frames) != len(EXPECTED_LEGACY_GT_SLOTS):
            notes.append("missing_legacy_gt_frames")
        if len(expected_frame_variants) != 1:
            notes.append("conflicting_or_missing_expected_frames")
        elif len(expected_frames) != len(EXPECTED_LEGACY_GT_SLOTS):
            notes.append("expected_frame_count_not_six")
        if duplicate_frame_values:
            notes.append("duplicate_frame_index_rows")
        if duplicate_order_values:
            notes.append("duplicate_legacy_order_rows")
        if int((~group["_bbox_valid"]).sum()) > 0:
            notes.append("invalid_bbox_rows")
        if group["_hidden_normalized"].isna().any():
            notes.append("invalid_hidden_rows")
        if expected_frames and missing_frames:
            notes.append("expected_frames_missing")
        if expected_frames and observed_frames != expected_frames:
            notes.append("anchor_frame_order_mismatch")
        if "hidden_is_trusted" in group.columns and _truthy(
            group["hidden_is_trusted"]
        ).any():
            notes.append("cvat_hidden_seed_must_be_untrusted")

        per_frame: dict[int, dict[str, Any]] = {}
        if not notes:
            for _, row in group.sort_values("legacy_order").iterrows():
                frame_index = int(row["frame_index"])
                img_name = (
                    ""
                    if "img_name" not in row or pd.isna(row.get("img_name"))
                    else str(row["img_name"])
                )
                per_frame[frame_index] = {
                    "bbox": (
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                    ),
                    "legacy_order": int(row["legacy_order"]),
                    "img_name": img_name,
                    "behavior": str(row["behavior"]),
                    "hidden": str(row["_hidden_normalized"]),
                    "hidden_source": "cvat_native_anchor",
                    "hidden_is_trusted": False,
                    "hidden_review_status": "seed_unreviewed",
                    "hidden_trust_status": "untrusted_cvat_seed",
                    "visibility_quality": "cvat_anchor_seed_unreviewed",
                }
            legacy_gt_map[(str(group_id_key), str(pig_id_key))] = per_frame

        audit_rows.append(
            {
                "group_id": group["group_id"].iloc[0],
                "pig_id": group["pig_id"].iloc[0],
                "sample_id": group["sample_id"].iloc[0] if "sample_id" in group.columns else "",
                "expected_gt_frames": _join_ints(expected_frames) if expected_frames else 6,
                "loaded_gt_frames": int(len(per_frame)),
                "legacy_gt_support_frames": _join_ints(support_frames),
                "legacy_gt_anchor_slots": _join_ints(observed_slots),
                "missing_gt_frames": _join_ints(missing_frames),
                "duplicate_gt_frames": _join_ints(duplicate_frame_values),
                "bbox_valid_rows": int(group["_bbox_valid"].sum()),
                "bbox_invalid_rows": int((~group["_bbox_valid"]).sum()),
                "qa_status": "error" if notes else "ok",
                "qa_notes": ";".join(notes),
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    return legacy_gt_map, audit_df


def hidden_seed_for_frame(
    frame_index: int,
    legacy_gt_by_frame: dict[int, dict[str, Any]],
    *,
    fallback_hidden: object,
) -> dict[str, object]:
    """Return a frame-level Hidden seed without claiming human trust."""
    if not legacy_gt_by_frame:
        hidden = _normalize_hidden_or_none(fallback_hidden)
        if hidden is None:
            raise ValueError(f"Unsupported legacy Hidden value: {fallback_hidden!r}")
        return {
            "hidden": hidden,
            "hidden_source": "legacy_prior_review",
            "hidden_is_trusted": True,
            "hidden_review_status": "prior_review_trusted",
            "hidden_trust_status": "trusted_prior_review",
            "visibility_quality": (
                "hidden_prior_review" if hidden == "Yes" else "visible_prior_review"
            ),
            "hidden_seed_method": "legacy_center_prior_review",
        }

    anchors = sorted(int(value) for value in legacy_gt_by_frame)
    if frame_index in legacy_gt_by_frame:
        hidden = _record_hidden(legacy_gt_by_frame[frame_index])
        method = "cvat_anchor_exact"
        source = "cvat_native_anchor"
        quality = "cvat_anchor_seed_unreviewed"
    else:
        left = [value for value in anchors if value < frame_index]
        right = [value for value in anchors if value > frame_index]
        if not left or not right:
            raise ValueError(
                "Dense frame is outside the bracketing CVAT anchor range: "
                f"frame={frame_index}, anchors={anchors}"
            )
        left_hidden = _record_hidden(legacy_gt_by_frame[max(left)])
        right_hidden = _record_hidden(legacy_gt_by_frame[min(right)])
        if left_hidden == right_hidden:
            hidden = left_hidden
            method = "cvat_anchor_pair_agreement"
            source = "cvat_anchor_pair_agreement"
            quality = "cvat_pair_seed_unreviewed"
        else:
            hidden = "Yes"
            method = "cvat_anchor_transition_conservative"
            source = "cvat_anchor_transition_conservative"
            quality = "cvat_transition_conservative_unreviewed"

    return {
        "hidden": hidden,
        "hidden_source": source,
        "hidden_is_trusted": False,
        "hidden_review_status": "seed_unreviewed",
        "hidden_trust_status": "untrusted_cvat_seed",
        "visibility_quality": quality,
        "hidden_seed_method": method,
    }


def _record_hidden(record: dict[str, Any]) -> str:
    hidden = _normalize_hidden_or_none(record.get("hidden"))
    if hidden is None:
        raise ValueError(f"Invalid CVAT anchor Hidden payload: {record.get('hidden')!r}")
    return hidden

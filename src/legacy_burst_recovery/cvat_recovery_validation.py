"""Validate recovered legacy dense rows against CVAT-derived inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from legacy_burst_recovery.csv_loader import parse_frames
from legacy_burst_recovery.legacy_gt_loader import hidden_seed_for_frame

ACTOR_KEY = ["group_id", "pig_id"]
FRAME_KEY = [*ACTOR_KEY, "frame_index"]
BBOX_COLUMNS = ["x1", "y1", "x2", "y2"]


def validate_cvat_recovered_dense(
    center: pd.DataFrame,
    anchors: pd.DataFrame,
    dense: pd.DataFrame,
    *,
    bbox_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Return a fail-closed audit of behavior, frame, and anchor contracts."""
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "center": {*ACTOR_KEY, "behavior", "frames", "behavior_authority_slot"},
        "anchors": {
            *FRAME_KEY,
            "legacy_order",
            "behavior",
            "hidden",
            *BBOX_COLUMNS,
        },
        "dense": {
            *FRAME_KEY,
            "behavior",
            "hidden",
            "hidden_source",
            "hidden_is_trusted",
            "hidden_review_status",
            "hidden_trust_status",
            "visibility_quality",
            "hidden_seed_method",
            "bbox_source",
            "legacy_gt_bbox_available",
            *BBOX_COLUMNS,
        },
    }
    tables = {"center": center, "anchors": anchors, "dense": dense}
    for name, columns in required.items():
        missing = sorted(columns.difference(tables[name].columns))
        if missing:
            errors.append(f"{name}_missing_columns={missing}")
    if errors:
        return _audit(center, anchors, dense, errors, warnings, {})

    center = _normalize(center)
    anchors = _normalize(anchors)
    dense = _normalize(dense)
    duplicates = {
        "center_actor_keys": int(center.duplicated(ACTOR_KEY).sum()),
        "anchor_frame_keys": int(anchors.duplicated(FRAME_KEY).sum()),
        "dense_frame_keys": int(dense.duplicated(FRAME_KEY).sum()),
    }
    for name, count in duplicates.items():
        if count:
            errors.append(f"duplicate_{name}={count}")

    authority_slot = pd.to_numeric(
        center["behavior_authority_slot"], errors="coerce"
    )
    invalid_authority = int(authority_slot.ne(0).sum())
    if invalid_authority:
        errors.append(f"non_k0_behavior_authority_rows={invalid_authority}")

    center_keys = _key_set(center)
    anchor_keys = _key_set(anchors)
    dense_keys = _key_set(dense)
    if center_keys != anchor_keys:
        errors.append(_key_difference("center_anchor_actor_keys", center_keys, anchor_keys))
    if center_keys != dense_keys:
        errors.append(_key_difference("center_dense_actor_keys", center_keys, dense_keys))

    expected_dense_rows = 0
    frame_set_mismatches = 0
    center_anchor_contract_mismatches = 0
    dense_row_count_mismatches = 0
    for row in center.drop_duplicates(ACTOR_KEY).itertuples(index=False):
        expected_anchors = parse_frames(row.frames)
        if (
            len(expected_anchors) != 6
            or len(set(expected_anchors)) != 6
            or expected_anchors[-1] - expected_anchors[0] != 15
        ):
            frame_set_mismatches += 1
            continue
        actor_mask = (
            anchors["group_id"].eq(row.group_id)
            & anchors["pig_id"].eq(row.pig_id)
        )
        actor_anchors = anchors.loc[actor_mask].sort_values("legacy_order")
        anchor_frames = actor_anchors["frame_index"].dropna().astype(int).tolist()
        anchor_slots = actor_anchors["legacy_order"].dropna().astype(int).tolist()
        if anchor_frames != expected_anchors or anchor_slots != list(range(6)):
            center_anchor_contract_mismatches += 1
        expected_frames = set(range(expected_anchors[0], expected_anchors[-1] + 1))
        dense_mask = (
            dense["group_id"].eq(row.group_id)
            & dense["pig_id"].eq(row.pig_id)
        )
        actor_dense = dense.loc[dense_mask]
        actual_frames = set(actor_dense["frame_index"].dropna().astype(int))
        expected_dense_rows += len(expected_frames)
        if len(actor_dense) != 16:
            dense_row_count_mismatches += 1
        if actual_frames != expected_frames:
            frame_set_mismatches += 1
    if frame_set_mismatches:
        errors.append(f"dense_actor_frame_set_mismatches={frame_set_mismatches}")
    if center_anchor_contract_mismatches:
        errors.append(
            "center_anchor_six_slot_mismatches="
            f"{center_anchor_contract_mismatches}"
        )
    if dense_row_count_mismatches:
        errors.append(f"dense_actor_row_count_mismatches={dense_row_count_mismatches}")

    behavior_counts = _behavior_mismatch_counts(center, anchors, dense)
    for name, count in behavior_counts.items():
        if count:
            errors.append(f"{name}={count}")

    anchor_checks = _anchor_checks(
        anchors,
        dense,
        bbox_tolerance=bbox_tolerance,
    )
    for name, count in anchor_checks.items():
        if count:
            errors.append(f"{name}={count}")

    hidden_checks = _hidden_checks(anchors, dense)
    for name, count in hidden_checks.items():
        if count:
            errors.append(f"{name}={count}")

    center_checks = _center_k0_checks(
        center,
        anchors,
        bbox_tolerance=bbox_tolerance,
    )
    for name, count in center_checks.items():
        if count:
            errors.append(f"{name}={count}")

    details = {
        "expected_dense_rows": expected_dense_rows,
        "frame_set_mismatches": frame_set_mismatches,
        "duplicates": duplicates,
        "behavior_checks": behavior_counts,
        "anchor_checks": anchor_checks,
        "hidden_checks": hidden_checks,
        "center_checks": center_checks,
    }
    return _audit(center, anchors, dense, errors, warnings, details)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ACTOR_KEY:
        out[column] = out[column].astype(str)
    if "frame_index" in out:
        out["frame_index"] = pd.to_numeric(out["frame_index"], errors="coerce")
    return out


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(frame[ACTOR_KEY].itertuples(index=False, name=None))


def _actor_key_count(frame: pd.DataFrame) -> int:
    if frame.empty or not set(ACTOR_KEY).issubset(frame.columns):
        return 0
    return len(_key_set(frame))


def _key_difference(
    name: str,
    expected: set[tuple[str, str]],
    actual: set[tuple[str, str]],
) -> str:
    return (
        f"{name}:missing={len(expected - actual)},"
        f"unexpected={len(actual - expected)}"
    )


def _behavior_mismatch_counts(
    center: pd.DataFrame,
    anchors: pd.DataFrame,
    dense: pd.DataFrame,
) -> dict[str, int]:
    authority = center[[*ACTOR_KEY, "behavior"]].rename(
        columns={"behavior": "k0_behavior"}
    )
    anchor_join = anchors.merge(authority, on=ACTOR_KEY, how="left")
    dense_join = dense.merge(authority, on=ACTOR_KEY, how="left")
    return {
        "anchor_behavior_not_mapped_from_k0": int(
            anchor_join["behavior"].ne(anchor_join["k0_behavior"]).sum()
        ),
        "dense_behavior_not_mapped_from_k0": int(
            dense_join["behavior"].ne(dense_join["k0_behavior"]).sum()
        ),
    }


def _anchor_checks(
    anchors: pd.DataFrame,
    dense: pd.DataFrame,
    *,
    bbox_tolerance: float,
) -> dict[str, int]:
    joined = anchors[FRAME_KEY + BBOX_COLUMNS].merge(
        dense[
            FRAME_KEY
            + BBOX_COLUMNS
            + ["bbox_source", "legacy_gt_bbox_available"]
        ],
        on=FRAME_KEY,
        how="left",
        suffixes=("_cvat", "_dense"),
        indicator=True,
    )
    missing = joined["_merge"].ne("both")
    mismatch = pd.Series(False, index=joined.index)
    for column in BBOX_COLUMNS:
        left = pd.to_numeric(joined[f"{column}_cvat"], errors="coerce")
        right = pd.to_numeric(joined[f"{column}_dense"], errors="coerce")
        mismatch |= left.sub(right).abs().gt(bbox_tolerance) | left.isna() | right.isna()
    gt_available = (
        joined["legacy_gt_bbox_available"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    gt_source = joined["bbox_source"].fillna("").astype(str).eq("gt_legacy")
    return {
        "cvat_anchor_missing_from_dense": int(missing.sum()),
        "cvat_anchor_bbox_mismatches": int((~missing & mismatch).sum()),
        "anchor_not_marked_gt_available": int((~missing & ~gt_available).sum()),
        "anchor_bbox_source_not_gt_legacy": int((~missing & ~gt_source).sum()),
    }


def _center_k0_checks(
    center: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    bbox_tolerance: float,
) -> dict[str, int]:
    required = {
        "center_frame_from_img",
        "center_frame_final",
        "frame_mismatch",
        "hidden",
        "bbox_anchor_slot",
        *BBOX_COLUMNS,
    }
    missing = required.difference(center.columns)
    if missing:
        return {"center_missing_k0_contract_columns": 1}

    k0 = anchors.loc[
        anchors["legacy_order"].eq(0),
        [*ACTOR_KEY, "frame_index", "hidden", *BBOX_COLUMNS],
    ]
    if (
        center.duplicated(ACTOR_KEY).any()
        or k0.duplicated(ACTOR_KEY).any()
    ):
        return {"center_k0_duplicate_join_keys": 1}
    joined = center.merge(
        k0,
        on=ACTOR_KEY,
        how="left",
        suffixes=("_center", "_k0"),
        indicator=True,
        validate="one_to_one",
    )
    missing_k0 = joined["_merge"].ne("both")
    frame_mismatch = pd.Series(False, index=joined.index)
    for column in ["center_frame_from_img", "center_frame_final"]:
        frame_mismatch |= pd.to_numeric(
            joined[column],
            errors="coerce",
        ).ne(pd.to_numeric(joined["frame_index"], errors="coerce"))
    frame_mismatch |= joined["frame_mismatch"].map(_as_bool)
    frame_mismatch |= pd.to_numeric(
        joined["bbox_anchor_slot"],
        errors="coerce",
    ).ne(0)
    bbox_mismatch = pd.Series(False, index=joined.index)
    for column in BBOX_COLUMNS:
        bbox_mismatch |= pd.to_numeric(
            joined[f"{column}_center"],
            errors="coerce",
        ).sub(pd.to_numeric(joined[f"{column}_k0"], errors="coerce")).abs().gt(
            bbox_tolerance
        )
    hidden_mismatch = joined["hidden_center"].map(_normalize_hidden).ne(
        joined["hidden_k0"].map(_normalize_hidden)
    )
    return {
        "center_k0_missing": int(missing_k0.sum()),
        "center_k0_frame_contract_mismatches": int(
            (~missing_k0 & frame_mismatch).sum()
        ),
        "center_k0_bbox_mismatches": int((~missing_k0 & bbox_mismatch).sum()),
        "center_k0_hidden_mismatches": int((~missing_k0 & hidden_mismatch).sum()),
    }


def _hidden_checks(
    anchors: pd.DataFrame,
    dense: pd.DataFrame,
) -> dict[str, int]:
    counts = {
        "hidden_seed_unverifiable_actor_keys": 0,
        "dense_hidden_seed_mismatches": 0,
        "dense_hidden_source_mismatches": 0,
        "dense_hidden_method_mismatches": 0,
        "dense_hidden_review_status_mismatches": 0,
        "dense_hidden_trust_status_mismatches": 0,
        "dense_hidden_visibility_quality_mismatches": 0,
        "dense_hidden_incorrectly_trusted": 0,
    }
    for key, actor_dense in dense.groupby(ACTOR_KEY, sort=False):
        group_id, pig_id = key
        actor_anchors = anchors.loc[
            anchors["group_id"].eq(group_id)
            & anchors["pig_id"].eq(pig_id)
        ]
        if (
            len(actor_anchors) != 6
            or actor_anchors["frame_index"].duplicated().any()
            or actor_anchors["legacy_order"].duplicated().any()
        ):
            counts["hidden_seed_unverifiable_actor_keys"] += 1
            continue
        anchor_map = {
            int(row.frame_index): {"hidden": row.hidden}
            for row in actor_anchors.itertuples(index=False)
        }
        for row in actor_dense.itertuples(index=False):
            expected = hidden_seed_for_frame(
                int(row.frame_index),
                anchor_map,
                fallback_hidden="No",
            )
            if _normalize_hidden(row.hidden) != expected["hidden"]:
                counts["dense_hidden_seed_mismatches"] += 1
            if str(row.hidden_source) != expected["hidden_source"]:
                counts["dense_hidden_source_mismatches"] += 1
            if str(row.hidden_seed_method) != expected["hidden_seed_method"]:
                counts["dense_hidden_method_mismatches"] += 1
            if str(row.hidden_review_status) != expected["hidden_review_status"]:
                counts["dense_hidden_review_status_mismatches"] += 1
            if str(row.hidden_trust_status) != expected["hidden_trust_status"]:
                counts["dense_hidden_trust_status_mismatches"] += 1
            if str(row.visibility_quality) != expected["visibility_quality"]:
                counts["dense_hidden_visibility_quality_mismatches"] += 1
            if _as_bool(row.hidden_is_trusted):
                counts["dense_hidden_incorrectly_trusted"] += 1
    return counts


def _normalize_hidden(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "y"}:
        return "Yes"
    if text in {"no", "false", "0", "n"}:
        return "No"
    return "INVALID"


def _as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _audit(
    center: pd.DataFrame,
    anchors: pd.DataFrame,
    dense: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FAIL" if errors else "PASS",
        "policy": {
            "behavior_authority": "per_actor_k0_to_k1_k5_and_all_dense_frames",
            "bbox_authority": "independent_native_cvat_bbox_at_each_k0_to_k5",
            "hidden_authority": "exact_cvat_anchor_plus_conservative_pair_seed",
            "hidden_trust": "untrusted_until_frame_object_human_review",
        },
        "counts": {
            "center_rows": int(len(center)),
            "anchor_rows": int(len(anchors)),
            "dense_rows": int(len(dense)),
            "center_actor_keys": _actor_key_count(center),
            "anchor_actor_keys": _actor_key_count(anchors),
            "dense_actor_keys": _actor_key_count(dense),
        },
        **details,
        "errors": errors,
        "warnings": warnings,
    }

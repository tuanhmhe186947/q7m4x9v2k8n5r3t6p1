"""Build legacy recovery inputs from current native CVAT six-frame anchors."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from legacy_burst_recovery.csv_loader import REQUIRED_COLUMNS, parse_frames
from legacy_burst_recovery.cvat_behavior_overlay import (
    LABEL_SOURCE,
    PROPAGATION_POLICY,
    load_cvat_legacy_rows,
)
from legacy_burst_recovery.export_legacy_annotations import (
    normalize_source_video_key,
)

EXPECTED_SLOTS = frozenset(range(6))
EXPECTED_ANCHOR_COUNT = len(EXPECTED_SLOTS)
VALID_PIG_ID = re.compile(r"ID_[1-8]")
VALID_SOURCE_VIDEO_KEY = re.compile(r"pigs\d{6}[a-z]?/\d{6}")
GROUP_VIDEO_HASH = re.compile(
    r"^burst_[^_]+_(?P<video_hash>[0-9a-fA-F]{8})_\d+$"
)
_GROUP_METADATA_COLUMNS = (
    "day_final",
    "video_final",
    "frames",
    "source_video_key",
)


def build_legacy_recovery_inputs(
    *,
    cvat_export_root: str | Path,
    metadata_scaffold_csv: str | Path,
    behavior_authority_slot: int = 0,
    min_anchor_count: int = EXPECTED_ANCHOR_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Return rebuilt center/scaffold rows, six-anchor rows, audit, and issues."""
    if behavior_authority_slot != 0:
        raise ValueError("legacy behavior authority must remain k0 (slot 0)")
    if min_anchor_count != EXPECTED_ANCHOR_COUNT:
        raise ValueError(
            "Canonical legacy recovery requires exactly six CVAT anchors; "
            "min_anchor_count must be 6"
        )

    scaffold_path = Path(metadata_scaffold_csv)
    scaffold = pd.read_csv(scaffold_path, low_memory=False)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(scaffold.columns))
    if missing_columns:
        raise ValueError(f"Metadata scaffold is missing columns: {missing_columns}")

    prepared, source_files = load_cvat_legacy_rows(cvat_export_root)
    scaffold = scaffold.copy()
    scaffold["group_id"] = scaffold["group_id"].astype(str)
    scaffold["pig_id"] = scaffold["pig_id"].astype(str)
    valid_groups = set(scaffold["group_id"])
    current = prepared.loc[prepared["group_id"].isin(valid_groups)].copy()
    issues: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[str] = []

    source_key_audit = _resolve_source_video_keys(
        scaffold,
        issues,
        errors,
        warnings,
    )
    source_hash_audit = _audit_group_video_hashes(
        scaffold,
        issues,
        errors,
    )
    _audit_group_metadata(scaffold, issues, errors)
    scaffold_duplicate_actor = scaffold.duplicated(
        ["group_id", "pig_id"],
        keep=False,
    )
    if scaffold_duplicate_actor.any():
        warnings.append(
            "duplicate scaffold actor rows were ignored as non-authoritative metadata"
        )
        for row in scaffold.loc[
            scaffold_duplicate_actor,
            ["group_id", "pig_id"],
        ].drop_duplicates().itertuples(index=False):
            issues.append(
                _issue(
                    "info",
                    "duplicate_scaffold_actor_metadata_ignored",
                    row.group_id,
                    row.pig_id,
                    "",
                    "CVAT native rows remain actor/bbox/behavior authority",
                )
            )
    _audit_anchor_rows(current, issues, errors)
    _audit_group_frame_maps(scaffold, current, issues, errors)

    authority = current.loc[
        current["selected_slot"].eq(behavior_authority_slot)
    ].copy()
    duplicate_authority = authority.duplicated(
        ["group_id", "pig_id"],
        keep=False,
    )
    if duplicate_authority.any():
        count = int(duplicate_authority.sum())
        errors.append(f"duplicate_k0_authority_rows={count}")

    old_keys = scaffold[["group_id", "pig_id"]].drop_duplicates()
    authority_keys = authority[["group_id", "pig_id"]].drop_duplicates()
    key_join = old_keys.merge(
        authority_keys,
        on=["group_id", "pig_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    _append_key_join_issues(key_join, issues)

    coverage = _anchor_coverage(current, authority_keys)
    for row in coverage.itertuples(index=False):
        if not bool(row.complete_anchor_set):
            issues.append(
                _issue(
                    "excluded",
                    "incomplete_anchor_set",
                    row.group_id,
                    row.pig_id,
                    "",
                    f"slots={row.anchor_slots}; count={row.anchor_count}",
                )
            )

    behavior_disagreement_by_slot = _behavior_disagreement_counts(
        current,
        authority,
    )
    behavior_disagreement_rows = sum(behavior_disagreement_by_slot.values())
    if behavior_disagreement_rows:
        warnings.append(
            "k1..k5 behavior disagreements were mapped to k0 and retained in audit"
        )

    recoverable_keys = coverage.loc[
        coverage["complete_anchor_set"],
        ["group_id", "pig_id"],
    ]
    excluded_for_anchor_count = int(
        (~coverage["complete_anchor_set"]).sum()
    )
    if excluded_for_anchor_count:
        warnings.append(
            "k0 keys without exactly six anchors were excluded from recovery inputs"
        )
    if key_join["_merge"].ne("both").any():
        warnings.append("old/new actor-key differences are recorded in issues")

    if errors:
        center = pd.DataFrame()
        anchors = pd.DataFrame()
    else:
        authority = authority.merge(
            recoverable_keys,
            on=["group_id", "pig_id"],
            how="inner",
            validate="one_to_one",
        )
        anchors = _build_anchor_table(current, authority, scaffold)
        center = _build_center_table(anchors, authority, scaffold)

    status = _status(errors, warnings, excluded_for_anchor_count)
    source_files = [
        *source_files,
        {
            "task": "metadata_scaffold",
            "path": str(scaffold_path),
            "sha256": _sha256(scaffold_path),
        },
    ]
    audit = {
        "schema_version": 1,
        "status": status,
        "policy": {
            "valid_group_universe": "metadata_scaffold_group_id",
            "behavior_authority_slot": behavior_authority_slot,
            "behavior_propagation_policy": PROPAGATION_POLICY,
            "bbox_authority": "each_native_cvat_k0_to_k5_shape",
            "min_anchor_count": min_anchor_count,
            "expected_slots": sorted(EXPECTED_SLOTS),
            "duplicate_anchor_policy": "fail_closed",
            "source_video_key_policy": (
                "validate_supplied_key_or_derive_from_video_final"
            ),
        },
        "counts": {
            "scaffold_rows": int(len(scaffold)),
            "scaffold_groups": int(scaffold["group_id"].nunique()),
            "scaffold_actor_keys": int(len(old_keys)),
            "scaffold_duplicate_actor_rows": int(scaffold_duplicate_actor.sum()),
            **source_key_audit,
            **source_hash_audit,
            "cvat_rows_all_groups": int(len(prepared)),
            "cvat_rows_valid_groups": int(len(current)),
            "cvat_groups_outside_scaffold": int(
                prepared.loc[
                    ~prepared["group_id"].isin(valid_groups),
                    "group_id",
                ].nunique()
            ),
            "k0_authority_keys": int(len(authority_keys)),
            "old_keys_missing_k0": int(key_join["_merge"].eq("left_only").sum()),
            "new_k0_keys": int(key_join["_merge"].eq("right_only").sum()),
            "complete_six_anchor_keys": int(coverage["complete_anchor_set"].sum()),
            "incomplete_k0_keys": int(
                (~coverage["complete_anchor_set"]).sum()
            ),
            "excluded_below_min_anchor_count": excluded_for_anchor_count,
            "recoverable_actor_keys": int(len(recoverable_keys)),
            "behavior_disagreement_rows_mapped_to_k0": int(
                behavior_disagreement_rows
            ),
            "center_rows_output": int(len(center)),
            "anchor_rows_output": int(len(anchors)),
        },
        "behavior_disagreement_by_slot_mapped_to_k0": {
            str(slot): int(behavior_disagreement_by_slot.get(slot, 0))
            for slot in sorted(EXPECTED_SLOTS)
        },
        "source_files": source_files,
        "errors": errors,
        "warnings": warnings,
    }
    issue_df = pd.DataFrame(
        issues,
        columns=["severity", "code", "group_id", "pig_id", "slot", "details"],
    )
    return center, anchors, audit, issue_df


def _audit_group_metadata(
    scaffold: pd.DataFrame,
    issues: list[dict[str, object]],
    errors: list[str],
) -> None:
    for column in _group_metadata_columns(scaffold):
        missing = scaffold[column].isna() | scaffold[column].astype(str).str.strip().eq("")
        for group_id in scaffold.loc[missing, "group_id"].drop_duplicates():
            issues.append(
                _issue(
                    "error",
                    "missing_group_metadata",
                    group_id,
                    "",
                    "",
                    f"column={column}",
                )
            )
            errors.append(f"missing_group_metadata:{group_id}:{column}")
        counts = scaffold.groupby("group_id", dropna=False)[column].nunique(
            dropna=False
        )
        for group_id in counts[counts.gt(1)].index:
            issues.append(
                _issue(
                    "error",
                    "conflicting_group_metadata",
                    group_id,
                    "",
                    "",
                    f"column={column}",
                )
            )
            errors.append(f"conflicting_group_metadata:{group_id}:{column}")


def _resolve_source_video_keys(
    scaffold: pd.DataFrame,
    issues: list[dict[str, object]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, int]:
    scaffold["source_video_key"] = scaffold["source_video_key"].astype(object)
    supplied_raw = scaffold["source_video_key"]
    supplied_missing = supplied_raw.isna() | supplied_raw.astype(str).str.strip().eq("")
    supplied = supplied_raw.map(normalize_source_video_key)
    derived = scaffold["video_final"].map(normalize_source_video_key)
    supplied_valid = supplied.astype(str).str.fullmatch(VALID_SOURCE_VIDEO_KEY)
    derived_valid = derived.astype(str).str.fullmatch(VALID_SOURCE_VIDEO_KEY)

    invalid_supplied = ~supplied_missing & ~supplied_valid
    unresolvable_derived = supplied_missing & ~derived_valid
    mismatch = (
        ~supplied_missing
        & supplied_valid
        & derived_valid
        & supplied.astype(str).ne(derived.astype(str))
    )
    day_mismatch = derived_valid & derived.astype(str).str.split("/").str[0].ne(
        scaffold["day_final"].astype(str).str.strip().str.lower()
    )

    _append_source_key_issues(
        scaffold,
        invalid_supplied,
        "invalid_supplied_source_video_key",
        issues,
        errors,
    )
    _append_source_key_issues(
        scaffold,
        unresolvable_derived,
        "cannot_derive_source_video_key",
        issues,
        errors,
    )
    _append_source_key_issues(
        scaffold,
        mismatch,
        "source_video_key_video_path_mismatch",
        issues,
        errors,
    )
    _append_source_key_issues(
        scaffold,
        day_mismatch,
        "source_video_key_day_mismatch",
        issues,
        errors,
    )

    derive_mask = supplied_missing & derived_valid
    scaffold.loc[derive_mask, "source_video_key"] = derived.loc[derive_mask]
    normalize_mask = ~supplied_missing & supplied_valid
    scaffold.loc[normalize_mask, "source_video_key"] = supplied.loc[normalize_mask]
    derived_groups = int(scaffold.loc[derive_mask, "group_id"].nunique())
    if derived_groups:
        warnings.append(
            "blank source_video_key values were deterministically derived from "
            "video_final and recorded in the audit"
        )
        for group_id in scaffold.loc[derive_mask, "group_id"].drop_duplicates():
            issues.append(
                _issue(
                    "info",
                    "source_video_key_derived",
                    group_id,
                    "",
                    "",
                    "derived deterministically from video_final",
                )
            )

    return {
        "source_video_key_groups_derived": derived_groups,
        "invalid_supplied_source_video_key_groups": int(
            scaffold.loc[invalid_supplied, "group_id"].nunique()
        ),
        "unresolvable_source_video_key_groups": int(
            scaffold.loc[unresolvable_derived, "group_id"].nunique()
        ),
        "source_video_key_path_mismatch_groups": int(
            scaffold.loc[mismatch, "group_id"].nunique()
        ),
        "source_video_key_day_mismatch_groups": int(
            scaffold.loc[day_mismatch, "group_id"].nunique()
        ),
    }


def _append_source_key_issues(
    scaffold: pd.DataFrame,
    mask: pd.Series,
    code: str,
    issues: list[dict[str, object]],
    errors: list[str],
) -> None:
    for row in scaffold.loc[
        mask,
        ["group_id", "source_video_key", "day_final", "video_final"],
    ].drop_duplicates("group_id").itertuples(index=False):
        issues.append(
            _issue(
                "error",
                code,
                row.group_id,
                "",
                "",
                (
                    f"source_video_key={row.source_video_key!r}; "
                    f"day_final={row.day_final!r}; video_final={row.video_final!r}"
                ),
            )
        )
        errors.append(f"{code}:{row.group_id}")


def _audit_group_video_hashes(
    scaffold: pd.DataFrame,
    issues: list[dict[str, object]],
    errors: list[str],
) -> dict[str, int]:
    checked = 0
    mismatches = 0
    groups = scaffold[["group_id", "video_final"]].drop_duplicates("group_id")
    for row in groups.itertuples(index=False):
        match = GROUP_VIDEO_HASH.fullmatch(str(row.group_id))
        normalized_key = normalize_source_video_key(row.video_final)
        if match is None or not VALID_SOURCE_VIDEO_KEY.fullmatch(normalized_key):
            continue
        checked += 1
        actual_hash = hashlib.md5(
            str(row.video_final).encode("utf-8")
        ).hexdigest()[:8]
        expected_hash = match.group("video_hash").lower()
        if actual_hash.lower() == expected_hash:
            continue
        mismatches += 1
        details = (
            f"expected={expected_hash}; actual={actual_hash}; "
            f"video_final={row.video_final!r}"
        )
        issues.append(
            _issue(
                "error",
                "group_video_hash_mismatch",
                row.group_id,
                "",
                "",
                details,
            )
        )
        errors.append(f"group_video_hash_mismatch:{row.group_id}")
    return {
        "group_video_hashes_checked": checked,
        "group_video_hash_mismatch_groups": mismatches,
    }


def _audit_anchor_rows(
    current: pd.DataFrame,
    issues: list[dict[str, object]],
    errors: list[str],
) -> None:
    invalid_pig = ~current["pig_id"].fillna("").astype(str).str.fullmatch(
        VALID_PIG_ID
    )
    invalid_behavior = current.loc[current["selected_slot"].eq(0), "behavior"].isna()
    hidden_text = current["hidden"].fillna("").astype(str).str.strip().str.lower()
    invalid_hidden = ~hidden_text.isin({"yes", "no", "true", "false", "1", "0"})
    if "hidden_attribute_present" in current.columns:
        hidden_present = current["hidden_attribute_present"].fillna(False).map(
            lambda value: value
            if isinstance(value, bool)
            else str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
        )
        invalid_hidden |= ~hidden_present
    invalid_bbox = (
        current[["x1", "y1", "x2", "y2"]].isna().any(axis=1)
        | current["x2"].le(current["x1"])
        | current["y2"].le(current["y1"])
    )
    if invalid_pig.any():
        errors.append(f"invalid_pig_id_rows={int(invalid_pig.sum())}")
        _append_anchor_field_issues(
            current,
            invalid_pig,
            "invalid_pig_id",
            issues,
        )
    if invalid_behavior.any():
        errors.append(f"invalid_k0_behavior_rows={int(invalid_behavior.sum())}")
        _append_anchor_field_issues(
            current,
            invalid_behavior,
            "invalid_k0_behavior",
            issues,
        )
    if invalid_hidden.any():
        errors.append(f"invalid_hidden_rows={int(invalid_hidden.sum())}")
        _append_anchor_field_issues(
            current,
            invalid_hidden,
            "invalid_hidden_attribute",
            issues,
        )
    if invalid_bbox.any():
        errors.append(f"invalid_bbox_rows={int(invalid_bbox.sum())}")
        _append_anchor_field_issues(
            current,
            invalid_bbox,
            "invalid_anchor_bbox",
            issues,
        )

    duplicate = current.duplicated(
        ["group_id", "selected_slot", "pig_id"],
        keep=False,
    )
    duplicate_rows = current.loc[duplicate].sort_values(
        ["group_id", "selected_slot", "pig_id"]
    )
    for key, group in duplicate_rows.groupby(
        ["group_id", "selected_slot", "pig_id"],
        sort=True,
    ):
        group_id, slot, pig_id = key
        boxes = group[["x1", "y1", "x2", "y2"]].round(3).values.tolist()
        labels = sorted(set(group["behavior"].dropna().astype(str)))
        issues.append(
            _issue(
                "error",
                "duplicate_anchor_identity",
                group_id,
                pig_id,
                slot,
                f"rows={len(group)}; boxes={boxes}; labels={labels}",
            )
        )
    if not duplicate_rows.empty:
        errors.append(
            "duplicate_anchor_identity_rows="
            f"{int(len(duplicate_rows))}"
        )


def _append_anchor_field_issues(
    current: pd.DataFrame,
    mask: pd.Series,
    code: str,
    issues: list[dict[str, object]],
) -> None:
    columns = ["group_id", "pig_id", "selected_slot"]
    for row in current.loc[mask, columns].drop_duplicates().itertuples(
        index=False
    ):
        issues.append(
            _issue(
                "error",
                code,
                row.group_id,
                row.pig_id,
                row.selected_slot,
                "native CVAT anchor field failed validation",
            )
        )


def _audit_group_frame_maps(
    scaffold: pd.DataFrame,
    current: pd.DataFrame,
    issues: list[dict[str, object]],
    errors: list[str],
) -> None:
    templates = scaffold.drop_duplicates("group_id", keep="first").set_index(
        "group_id"
    )
    observed = current.groupby(["group_id", "selected_slot"], as_index=False)[
        "selected_source_frame"
    ].agg(lambda values: sorted(set(int(value) for value in values.dropna())))
    for group_id, group in observed.groupby("group_id", sort=True):
        expected = parse_frames(templates.at[group_id, "frames"])
        observed_map = {
            int(row.selected_slot): row.selected_source_frame
            for row in group.itertuples(index=False)
        }
        invalid = any(len(values) != 1 for values in observed_map.values())
        actual = [
            values[0] if len(values) == 1 else None
            for _, values in sorted(observed_map.items())
        ]
        if invalid or actual != expected:
            issues.append(
                _issue(
                    "error",
                    "group_slot_frame_map_mismatch",
                    group_id,
                    "",
                    "",
                    f"expected={expected}; observed={actual}",
                )
            )
            errors.append(f"group_slot_frame_map_mismatch:{group_id}")


def _append_key_join_issues(
    key_join: pd.DataFrame,
    issues: list[dict[str, object]],
) -> None:
    for row in key_join.loc[key_join["_merge"].eq("left_only")].itertuples():
        issues.append(
            _issue(
                "excluded",
                "scaffold_actor_missing_k0",
                row.group_id,
                row.pig_id,
                0,
                "No behavior authority; dense fallback is forbidden",
            )
        )
    for row in key_join.loc[key_join["_merge"].eq("right_only")].itertuples():
        issues.append(
            _issue(
                "info",
                "new_k0_actor_key",
                row.group_id,
                row.pig_id,
                0,
                "New actor key is eligible only if anchor coverage passes",
            )
        )


def _anchor_coverage(
    current: pd.DataFrame,
    authority_keys: pd.DataFrame,
) -> pd.DataFrame:
    selected = current.merge(
        authority_keys,
        on=["group_id", "pig_id"],
        how="inner",
        validate="many_to_one",
    )
    coverage = selected.groupby(["group_id", "pig_id"], as_index=False).agg(
        anchor_count=("selected_slot", "nunique"),
        anchor_slots=(
            "selected_slot",
            lambda values: "|".join(map(str, sorted(set(values)))),
        ),
    )
    expected_slots = "|".join(map(str, sorted(EXPECTED_SLOTS)))
    coverage["complete_anchor_set"] = (
        coverage["anchor_count"].eq(EXPECTED_ANCHOR_COUNT)
        & coverage["anchor_slots"].eq(expected_slots)
    )
    return coverage


def _behavior_disagreement_counts(
    current: pd.DataFrame,
    authority: pd.DataFrame,
) -> dict[int, int]:
    mapped = current.merge(
        authority[["group_id", "pig_id", "behavior"]].rename(
            columns={"behavior": "k0_behavior"}
        ),
        on=["group_id", "pig_id"],
        how="inner",
        validate="many_to_one",
    )
    mismatch = mapped.loc[mapped["behavior"].ne(mapped["k0_behavior"])]
    return {
        int(slot): int(count)
        for slot, count in mismatch["selected_slot"].value_counts().items()
    }


def _build_anchor_table(
    current: pd.DataFrame,
    authority: pd.DataFrame,
    scaffold: pd.DataFrame,
) -> pd.DataFrame:
    authority_fields = authority[
        ["group_id", "pig_id", "behavior", "task", "task_frame", "img_name"]
    ].rename(
        columns={
            "behavior": "k0_behavior",
            "task": "behavior_authority_task",
            "task_frame": "behavior_authority_task_frame",
            "img_name": "behavior_authority_image_name",
        }
    )
    out = current.merge(
        authority_fields,
        on=["group_id", "pig_id"],
        how="inner",
        validate="many_to_one",
    )
    out["behavior_before_k0_mapping"] = out["behavior"]
    out["behavior"] = out["k0_behavior"]
    out["hidden"] = out["hidden"].map(_normalize_hidden_strict)
    out["hidden_source"] = "cvat_native_anchor"
    out["hidden_is_trusted"] = False
    out["hidden_review_status"] = "seed_unreviewed"
    out["hidden_trust_status"] = "untrusted_cvat_seed"
    out["visibility_quality"] = "cvat_anchor_seed_unreviewed"
    out["sample_id"] = out["group_id"] + "_" + out["pig_id"]
    out["legacy_order"] = out["selected_slot"].astype(int)
    out["order"] = out["legacy_order"]
    out["k"] = out["legacy_order"]
    out["frame_index"] = out["selected_source_frame"].astype(int)
    out["legacy_frame_index"] = out["frame_index"]
    out["cvat_frame_index"] = out["task_frame"].astype(int)
    out["label_source"] = LABEL_SOURCE
    out["behavior_authority_slot"] = 0
    out["behavior_propagation_policy"] = PROPAGATION_POLICY
    out["bbox_source"] = "cvat_native_six_anchor"

    group_meta = scaffold.drop_duplicates("group_id", keep="first")
    meta_columns = [
        column
        for column in _group_metadata_columns(group_meta)
        if column not in out.columns
    ]
    out = out.merge(
        group_meta[["group_id", *meta_columns]],
        on="group_id",
        how="left",
        validate="many_to_one",
    )
    columns = [
        "sample_id",
        "day_final",
        "video_final",
        "group_id",
        "pig_id",
        "behavior",
        "behavior_before_k0_mapping",
        "hidden",
        "img_name",
        "image_path",
        "task",
        "legacy_order",
        "order",
        "k",
        "frame_index",
        "legacy_frame_index",
        "cvat_frame_index",
        "frames",
        "width",
        "height",
        "x1",
        "y1",
        "x2",
        "y2",
        "label_source",
        "behavior_authority_slot",
        "behavior_authority_task",
        "behavior_authority_task_frame",
        "behavior_authority_image_name",
        "behavior_propagation_policy",
        "bbox_source",
        "source_video_key",
    ]
    columns.extend(column for column in out.columns if column not in columns)
    return out[columns].sort_values(
        ["group_id", "pig_id", "legacy_order"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_center_table(
    anchors: pd.DataFrame,
    authority: pd.DataFrame,
    scaffold: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = _group_metadata_columns(scaffold)
    templates = scaffold[["group_id", *group_columns]].drop_duplicates(
        "group_id",
        keep="first",
    ).set_index(
        "group_id",
        drop=False,
    )
    rows: list[dict[str, object]] = []
    for authority_row in authority.itertuples(index=False):
        group_id = str(authority_row.group_id)
        pig_id = str(authority_row.pig_id)
        actor_anchors = anchors.loc[
            anchors["group_id"].eq(group_id) & anchors["pig_id"].eq(pig_id)
        ].copy()
        template = templates.loc[group_id].to_dict()
        k0_rows = actor_anchors.loc[actor_anchors["legacy_order"].eq(0)]
        if len(k0_rows) != 1:
            raise ValueError(
                "Expected exactly one k0 anchor for "
                f"group_id={group_id}, pig_id={pig_id}; found={len(k0_rows)}"
            )
        chosen = k0_rows.iloc[0]
        center_frame = int(chosen["frame_index"])
        template.update(
            {
                "sample_id": f"{group_id}_{pig_id}",
                "match_source": "cvat_native_rebuilt_six_anchor",
                "group_id": group_id,
                "pig_id": pig_id,
                "behavior": authority_row.behavior,
                "hidden": _normalize_hidden_strict(chosen["hidden"]),
                "hidden_source": "cvat_native_k0_anchor",
                "hidden_is_trusted": False,
                "hidden_review_status": "seed_unreviewed",
                "hidden_trust_status": "untrusted_cvat_seed",
                "visibility_quality": "cvat_anchor_seed_unreviewed",
                "img_name": chosen["img_name"],
                "center_frame_from_img": center_frame,
                "center_frame_final": center_frame,
                "frame_mismatch": False,
                "x1": float(chosen["x1"]),
                "y1": float(chosen["y1"]),
                "x2": float(chosen["x2"]),
                "y2": float(chosen["y2"]),
                "label_source": LABEL_SOURCE,
                "behavior_authority_slot": 0,
                "behavior_authority_task": authority_row.task,
                "behavior_authority_task_frame": int(authority_row.task_frame),
                "behavior_authority_image_name": authority_row.img_name,
                "behavior_propagation_policy": PROPAGATION_POLICY,
                "bbox_anchor_slot": int(chosen["legacy_order"]),
                "anchor_count": int(actor_anchors["legacy_order"].nunique()),
                "anchor_slots": "|".join(
                    map(str, sorted(set(actor_anchors["legacy_order"])))
                ),
            }
        )
        rows.append(template)
    out = pd.DataFrame(rows)
    return out.sort_values(["group_id", "pig_id"], kind="mergesort").reset_index(
        drop=True
    )


def _issue(
    severity: str,
    code: str,
    group_id: object,
    pig_id: object,
    slot: object,
    details: str,
) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "group_id": str(group_id),
        "pig_id": str(pig_id),
        "slot": slot,
        "details": details,
    }


def _group_metadata_columns(scaffold: pd.DataFrame) -> list[str]:
    missing = [
        column for column in _GROUP_METADATA_COLUMNS if column not in scaffold.columns
    ]
    if missing:
        raise ValueError(
            "Metadata scaffold is missing resolver columns: "
            + ", ".join(missing)
        )
    return list(_GROUP_METADATA_COLUMNS)


def _status(
    errors: list[str],
    warnings: list[str],
    excluded: int,
) -> str:
    if errors:
        return "FAIL"
    if excluded:
        return "PASS_WITH_DECLARED_EXCLUSIONS"
    if warnings:
        return "PASS_WITH_WARNINGS"
    return "PASS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_hidden_strict(value: object) -> str:
    if pd.isna(value):
        raise ValueError("CVAT Hidden must not be missing")
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return "Yes"
    if text in {"no", "false", "0"}:
        return "No"
    raise ValueError(f"Unsupported CVAT Hidden value: {value!r}")

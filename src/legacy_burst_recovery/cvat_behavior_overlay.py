"""Apply first-task-frame behavior authority to recovered legacy bursts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import (
    normalize_behavior,
    normalize_pig_id,
)
from pig_behavior.data.cvat_native import (
    load_all_cvat_tasks,
    select_cvat_annotation_source,
)

AUTHORITY_POLICY = "first_task_frame_per_group"
PROPAGATION_POLICY = (
    "cvat_first_task_frame_per_burst_pig_to_all_legacy_dense_frames"
)
LABEL_SOURCE = "cvat_native_first_task_frame_behavior_authority"
MISSING_LABEL_SOURCE = "missing_cvat_first_frame_actor_no_training_label"
_WITH_SOURCE_FRAME = re.compile(
    r"^(?P<group>.+)_f(?P<source_frame>-?\d+)_k(?P<slot>\d+)(?:\.[^.]+)?$"
)
_WITHOUT_SOURCE_FRAME = re.compile(
    r"^(?P<group>.+)_k(?P<slot>\d+)(?:\.[^.]+)?$"
)


def parse_legacy_task_image_name(image_name: object) -> dict[str, object]:
    """Parse burst, source-frame, and selected-slot fields from a task image."""
    text = str(image_name).strip().replace("\\", "/").rsplit("/", 1)[-1]
    match = _WITH_SOURCE_FRAME.fullmatch(text)
    if match is not None:
        return {
            "group_id": match.group("group"),
            "source_frame": int(match.group("source_frame")),
            "slot": int(match.group("slot")),
        }

    match = _WITHOUT_SOURCE_FRAME.fullmatch(text)
    if match is not None:
        return {
            "group_id": match.group("group"),
            "source_frame": pd.NA,
            "slot": int(match.group("slot")),
        }

    raise ValueError(f"Invalid legacy CVAT task image name: {image_name!r}")


def apply_cvat_first_frame_behavior_authority(
    dense_df: pd.DataFrame,
    cvat_export_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Overlay the first displayed frame label onto each actor's dense rows.

    CVAT supplies behavior authority only. Dense rows remain authoritative for
    recovered boxes, Hidden, timestamps, paths, and tracking metadata.
    """
    required = {"group_id", "pig_id", "behavior"}
    missing = sorted(required.difference(dense_df.columns))
    if missing:
        raise ValueError(f"Dense legacy dataframe is missing columns: {missing}")
    if "legacy_behavior_authority_status" in dense_df.columns:
        raise ValueError("CVAT behavior authority has already been applied")

    source_prepared, source_files = load_cvat_legacy_rows(cvat_export_root)
    dense_actor_keys = set(
        zip(
            dense_df["group_id"].astype(str),
            dense_df["pig_id"].map(normalize_pig_id),
            strict=True,
        )
    )
    retained_mask = [
        (str(row.group_id), normalize_pig_id(row.pig_id)) in dense_actor_keys
        for row in source_prepared.itertuples(index=False)
    ]
    prepared = source_prepared.loc[retained_mask].copy()
    authority = select_first_task_frame_authority(prepared)
    errors = _authority_errors(authority)
    if errors:
        raise ValueError(
            "Invalid CVAT first-frame behavior authority: " + "; ".join(errors)
        )

    authority = _add_sampled_consistency(authority, prepared)
    out, join_audit, discrepancies = _overlay_dense(dense_df, authority)
    audit = _build_audit(
        dense_df=dense_df,
        prepared=prepared,
        source_prepared=source_prepared,
        authority=authority,
        join_audit=join_audit,
        source_files=source_files,
    )
    return out, audit, discrepancies


def apply_cvat_k0_behavior_authority(
    dense_df: pd.DataFrame,
    cvat_export_root: str | Path,
    *,
    authority_slot: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Compatibility wrapper; behavior authority is no longer selected by k0."""
    if authority_slot != 0:
        raise ValueError(
            "authority_slot is deprecated; first_task_frame_per_group is fixed"
        )
    return apply_cvat_first_frame_behavior_authority(
        dense_df,
        cvat_export_root,
    )


def select_first_task_frame_authority(prepared: pd.DataFrame) -> pd.DataFrame:
    """Select all actor boxes from the lowest CVAT frame ID in each burst."""
    invalid_task_frame = prepared["task_frame"].isna()
    if invalid_task_frame.any():
        raise ValueError(
            f"invalid_task_frame_rows={int(invalid_task_frame.sum())}"
        )
    tasks_per_group = prepared.groupby("group_id", dropna=False)["task"].nunique()
    conflicting_groups = int(tasks_per_group.gt(1).sum())
    if conflicting_groups:
        raise ValueError(f"groups_in_multiple_tasks={conflicting_groups}")

    if "burst_first_task_frame" not in prepared.columns:
        raise ValueError("missing_burst_first_task_frame")
    first_frame_counts = prepared.groupby(
        "group_id",
        dropna=False,
    )["burst_first_task_frame"].nunique(dropna=False)
    conflicting_first_frames = int(first_frame_counts.ne(1).sum())
    if conflicting_first_frames:
        raise ValueError(
            "conflicting_burst_first_task_frames="
            f"{conflicting_first_frames}"
        )
    first_frames = prepared[
        ["group_id", "burst_first_task_frame"]
    ].drop_duplicates("group_id").rename(
        columns={
            "burst_first_task_frame": "behavior_authority_task_frame"
        }
    )
    selected = prepared.merge(
        first_frames,
        on="group_id",
        how="inner",
        validate="many_to_one",
    )
    return selected.loc[
        selected["task_frame"].eq(selected["behavior_authority_task_frame"])
    ].copy()


def load_cvat_legacy_rows(
    cvat_export_root: str | Path,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Load all native CVAT tasks and parse legacy burst image semantics."""
    export_root = Path(cvat_export_root)
    return _prepare_cvat_rows(load_all_cvat_tasks(export_root), export_root)


def _prepare_cvat_rows(
    cvat: pd.DataFrame,
    export_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    required = {"task", "frame", "img_name", "pig_id", "behavior"}
    missing = sorted(required.difference(cvat.columns))
    if missing:
        raise ValueError(f"CVAT dataframe is missing columns: {missing}")

    parsed = cvat["img_name"].map(parse_legacy_task_image_name)
    out = cvat.copy()
    out["group_id"] = parsed.map(lambda item: item["group_id"])
    out["selected_slot"] = parsed.map(lambda item: item["slot"]).astype(int)
    out["selected_source_frame"] = parsed.map(lambda item: item["source_frame"])
    out["pig_id"] = out["pig_id"].map(normalize_pig_id)
    out["behavior"] = out["behavior"].map(normalize_behavior)
    out["task_frame"] = pd.to_numeric(out["frame"], errors="coerce")

    source_files: list[dict[str, str]] = []
    for task_dir in sorted(export_root.glob("task_*")):
        _, annotation_path = select_cvat_annotation_source(task_dir)
        for path in [annotation_path, task_dir / "data/manifest.jsonl"]:
            if path.exists():
                source_files.append(
                    {
                        "task": task_dir.name,
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                )
    return out, source_files


def _authority_errors(authority: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    invalid_identity = (
        authority["group_id"].astype(str).str.strip().eq("")
        | authority["pig_id"].isna()
        | authority["pig_id"].astype(str).str.strip().eq("")
    )
    if invalid_identity.any():
        errors.append(f"invalid_authority_identity_rows={int(invalid_identity.sum())}")

    invalid_behavior = authority["behavior"].isna()
    if invalid_behavior.any():
        errors.append(f"invalid_authority_behavior_rows={int(invalid_behavior.sum())}")

    duplicate_keys = authority.duplicated(["group_id", "pig_id"], keep=False)
    if duplicate_keys.any():
        errors.append(f"duplicate_authority_keys={int(duplicate_keys.sum())}")

    tasks_per_group = authority.groupby("group_id", dropna=False)["task"].nunique()
    conflicting_groups = int(tasks_per_group.gt(1).sum())
    if conflicting_groups:
        errors.append(f"groups_in_multiple_tasks={conflicting_groups}")
    return errors


def _add_sampled_consistency(
    authority: pd.DataFrame,
    prepared: pd.DataFrame,
) -> pd.DataFrame:
    sampled = prepared.merge(
        authority[["group_id", "pig_id", "behavior"]].rename(
            columns={"behavior": "authority_behavior"}
        ),
        on=["group_id", "pig_id"],
        how="inner",
        validate="many_to_one",
    )
    sampled["disagrees_with_authority"] = sampled["behavior"].ne(
        sampled["authority_behavior"]
    )
    summary = sampled.groupby(["group_id", "pig_id"], as_index=False).agg(
        legacy_sampled_behavior_disagreement_count=(
            "disagrees_with_authority",
            "sum",
        ),
        legacy_sampled_behavior_observation_count=("behavior", "size"),
        legacy_sampled_behavior_labels=(
            "behavior",
            lambda values: "|".join(sorted(set(values.dropna().astype(str)))),
        ),
    )
    summary["legacy_sampled_behavior_consistent"] = summary[
        "legacy_sampled_behavior_disagreement_count"
    ].eq(0)
    return authority.merge(
        summary,
        on=["group_id", "pig_id"],
        how="left",
        validate="one_to_one",
    )


def _overlay_dense(
    dense_df: pd.DataFrame,
    authority: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    out = dense_df.copy()
    out["_authority_group_id"] = out["group_id"].astype(str)
    out["_authority_pig_id"] = out["pig_id"].map(normalize_pig_id)
    out["legacy_behavior_before_cvat_authority"] = out["behavior"]

    dense_key_counts = out.groupby(
        ["_authority_group_id", "_authority_pig_id"],
        dropna=False,
    ).size()
    authority_fields = authority[
        [
            "task",
            "task_frame",
            "img_name",
            "group_id",
            "pig_id",
            "behavior",
            "selected_slot",
            "selected_source_frame",
            "legacy_sampled_behavior_disagreement_count",
            "legacy_sampled_behavior_observation_count",
            "legacy_sampled_behavior_labels",
            "legacy_sampled_behavior_consistent",
        ]
    ].rename(
        columns={
            "task": "legacy_behavior_authority_task",
            "task_frame": "legacy_behavior_authority_task_frame",
            "img_name": "legacy_behavior_authority_image_name",
            "group_id": "_authority_group_id",
            "pig_id": "_authority_pig_id",
            "behavior": "legacy_behavior_authority_label",
            "selected_slot": "legacy_behavior_authority_slot",
            "selected_source_frame": "legacy_behavior_authority_source_frame",
        }
    )
    out = out.merge(
        authority_fields,
        on=["_authority_group_id", "_authority_pig_id"],
        how="left",
        validate="many_to_one",
    )

    has_authority = out["legacy_behavior_authority_label"].notna()
    original_normalized = out["behavior"].map(normalize_behavior)
    out["legacy_behavior_changed_by_cvat_authority"] = (
        has_authority
        & original_normalized.ne(out["legacy_behavior_authority_label"])
    )
    out.loc[has_authority, "behavior"] = out.loc[
        has_authority,
        "legacy_behavior_authority_label",
    ]
    out.loc[~has_authority, "behavior"] = pd.NA
    out["legacy_behavior_authority_status"] = "authoritative_first_task_frame"
    out.loc[~has_authority, "legacy_behavior_authority_status"] = (
        "missing_first_frame_actor_excluded"
    )
    out["legacy_behavior_propagation_policy"] = PROPAGATION_POLICY

    if "label_source" not in out.columns:
        out["label_source"] = "legacy_dense_behavior"
    out.loc[has_authority, "label_source"] = LABEL_SOURCE
    out.loc[~has_authority, "label_source"] = MISSING_LABEL_SOURCE

    if "include_in_training" not in out.columns:
        out["include_in_training"] = True
    include = out["include_in_training"].map(_as_bool)
    out["include_in_training"] = include & has_authority
    if "training_tier" not in out.columns:
        out["training_tier"] = "legacy_recovered"
    if "qa_status" not in out.columns:
        out["qa_status"] = "ok"
    out.loc[~has_authority, "training_tier"] = "rejected"
    out.loc[~has_authority, "qa_status"] = "rejected"

    authority_keys = authority[["group_id", "pig_id"]].rename(
        columns={
            "group_id": "_authority_group_id",
            "pig_id": "_authority_pig_id",
        }
    )
    dense_keys = dense_key_counts.rename("dense_rows").reset_index()
    key_join = dense_keys.merge(
        authority_keys,
        on=["_authority_group_id", "_authority_pig_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    discrepancies = key_join.loc[key_join["_merge"].ne("both")].copy()
    discrepancies = discrepancies.rename(
        columns={
            "_authority_group_id": "group_id",
            "_authority_pig_id": "pig_id",
            "_merge": "join_status",
        }
    )
    discrepancies["join_status"] = discrepancies["join_status"].map(
        {
            "left_only": "dense_missing_first_frame_actor",
            "right_only": "first_frame_actor_missing_dense",
        }
    )

    join_audit = {
        "dense_groups": int(dense_keys["_authority_group_id"].nunique()),
        "authority_groups": int(authority_keys["_authority_group_id"].nunique()),
        "authority_groups_missing_dense": int(
            authority_keys.loc[
                ~authority_keys["_authority_group_id"].isin(
                    dense_keys["_authority_group_id"]
                ),
                "_authority_group_id",
            ].nunique()
        ),
        "dense_behavior_keys": int(len(dense_keys)),
        "authority_behavior_keys": int(len(authority_keys)),
        "matched_behavior_keys": int(key_join["_merge"].eq("both").sum()),
        "dense_keys_missing_authority": int(
            key_join["_merge"].eq("left_only").sum()
        ),
        "authority_keys_missing_dense": int(
            key_join["_merge"].eq("right_only").sum()
        ),
        "authority_keys_extra_within_dense_groups": int(
            key_join.loc[
                key_join["_merge"].eq("right_only")
                & key_join["_authority_group_id"].isin(
                    dense_keys["_authority_group_id"]
                )
            ].shape[0]
        ),
        "rows_with_authority": int(has_authority.sum()),
        "rows_excluded_missing_authority": int((~has_authority).sum()),
        "rows_changed_by_authority": int(
            out["legacy_behavior_changed_by_cvat_authority"].sum()
        ),
        "behavior_keys_changed_by_authority": int(
            out.loc[
                out["legacy_behavior_changed_by_cvat_authority"],
                ["_authority_group_id", "_authority_pig_id"],
            ].drop_duplicates().shape[0]
        ),
        "sampled_behavior_disagreement_rows_matched_dense": int(
            out.loc[
                has_authority,
                [
                    "_authority_group_id",
                    "_authority_pig_id",
                    "legacy_sampled_behavior_disagreement_count",
                ],
            ]
            .drop_duplicates(["_authority_group_id", "_authority_pig_id"])[
                "legacy_sampled_behavior_disagreement_count"
            ]
            .fillna(0)
            .sum()
        ),
    }
    return out.drop(columns=["_authority_group_id", "_authority_pig_id"]), join_audit, discrepancies


def _build_audit(
    *,
    dense_df: pd.DataFrame,
    prepared: pd.DataFrame,
    source_prepared: pd.DataFrame,
    authority: pd.DataFrame,
    join_audit: dict[str, Any],
    source_files: list[dict[str, str]],
) -> dict[str, Any]:
    sampled_duplicate_rows = int(
        prepared.duplicated(
            ["group_id", "selected_slot", "pig_id"],
            keep=False,
        ).sum()
    )
    anchor_coverage = prepared.groupby(
        ["group_id", "pig_id"],
        as_index=False,
    ).agg(
        anchor_count=("selected_slot", "nunique"),
        anchor_slots=(
            "selected_slot",
            lambda values: "|".join(map(str, sorted(set(values)))),
        ),
    )
    complete_anchor = (
        anchor_coverage["anchor_count"].eq(6)
        & anchor_coverage["anchor_slots"].eq("0|1|2|3|4|5")
    )
    incomplete_anchor_keys = int((~complete_anchor).sum())
    dense_actor_keys = set(
        zip(
            dense_df["group_id"].astype(str),
            dense_df["pig_id"].map(normalize_pig_id),
            strict=True,
        )
    )
    anchor_key_is_retained = pd.Series(
        [
            (str(row.group_id), normalize_pig_id(row.pig_id))
            in dense_actor_keys
            for row in anchor_coverage.itertuples(index=False)
        ],
        index=anchor_coverage.index,
    )
    incomplete_anchor_keys_in_dense = int(
        ((~complete_anchor) & anchor_key_is_retained).sum()
    )
    source_anchor_coverage = source_prepared.groupby(
        ["group_id", "pig_id"],
        as_index=False,
    ).agg(
        anchor_count=("selected_slot", "nunique"),
        anchor_slots=(
            "selected_slot",
            lambda values: "|".join(map(str, sorted(set(values)))),
        ),
    )
    source_complete_anchor = (
        source_anchor_coverage["anchor_count"].eq(6)
        & source_anchor_coverage["anchor_slots"].eq("0|1|2|3|4|5")
    )
    source_incomplete_anchor_keys = int((~source_complete_anchor).sum())
    disagreements = int(
        authority["legacy_sampled_behavior_disagreement_count"].fillna(0).sum()
    )
    groups_without_authority = sorted(
        set(prepared["group_id"].astype(str)).difference(
            authority["group_id"].astype(str)
        )
    )
    warnings: list[str] = []
    informational_findings: list[str] = []
    if join_audit["dense_keys_missing_authority"]:
        warnings.append(
            "dense actors absent from the first task frame were retained and excluded"
        )
    if join_audit["authority_keys_missing_dense"]:
        warnings.append(
            "first-frame actors without recovered dense tracklets were not fabricated"
        )
    if disagreements:
        informational_findings.append(
            "other sampled labels disagree with first-frame authority and are audit-only"
        )
    if sampled_duplicate_rows:
        warnings.append("duplicate non-authority sampled object keys were detected")
    if incomplete_anchor_keys_in_dense:
        warnings.append(
            "incomplete retained six-anchor actor keys were detected"
        )
    if groups_without_authority:
        warnings.append("CVAT groups without a first task frame were ignored")

    if join_audit["dense_keys_missing_authority"]:
        status = "PASS_WITH_DECLARED_EXCLUSIONS"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    return {
        "schema_version": 1,
        "status": status,
        "authority": {
            "field": "behavior",
            "policy": AUTHORITY_POLICY,
            "ordering_field": "task_frame",
            "key": ["group_id", "pig_id"],
            "scope": "legacy_native_16_frame_burst",
            "propagation_policy": PROPAGATION_POLICY,
            "bbox_authority": "legacy_dense_tracklet_map",
            "hidden_authority": "legacy_dense_tracklet_map",
        },
        "counts": {
            "dense_rows_input": int(len(dense_df)),
            "dense_rows_output": int(len(dense_df)),
            "source_cvat_object_rows": int(len(source_prepared)),
            "source_cvat_groups": int(source_prepared["group_id"].nunique()),
            "source_sampled_incomplete_anchor_keys": (
                source_incomplete_anchor_keys
            ),
            "retained_cvat_object_rows": int(len(prepared)),
            "retained_cvat_groups": int(prepared["group_id"].nunique()),
            "cvat_object_rows_filtered_before_authority_audit": int(
                len(source_prepared) - len(prepared)
            ),
            "cvat_groups_without_first_task_frame": int(
                len(groups_without_authority)
            ),
            "authority_rows": int(len(authority)),
            "sampled_behavior_disagreement_rows": disagreements,
            "sampled_duplicate_rows": sampled_duplicate_rows,
            "sampled_incomplete_anchor_keys": incomplete_anchor_keys,
            "sampled_incomplete_anchor_keys_in_dense": (
                incomplete_anchor_keys_in_dense
            ),
            **join_audit,
        },
        "cvat_groups_without_first_task_frame": groups_without_authority,
        "source_files": source_files,
        "errors": [],
        "warnings": warnings,
        "informational_findings": informational_findings,
    }


def _as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Audited migration of Hidden review manifests and human decisions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.identifiers import (
    ensure_frame_object_identifiers,
)
from pig_behavior.classification_v2.review.hidden_review_builder import (
    DECISION_COLUMNS,
)
from pig_behavior.classification_v2.review.hidden_review_identifiers import (
    HIDDEN_REVIEW_KEY_VERSION,
    attach_hidden_review_identifiers,
    audit_hidden_review_identifiers,
    build_hidden_review_subject_keys,
)

HIDDEN_REVIEW_MIGRATION_VERSION = (
    "classification_v2.hidden_review_identifier_migration.v1"
)
HUMAN_PAYLOAD_COLUMNS = (
    "hidden_before_review",
    "hidden_after_review",
    "hidden_review_status",
    "hidden_review_confidence",
    "hidden_review_reason",
    "hidden_reviewer",
    "hidden_reviewed_at",
)
CONTEXT_INVARIANT_COLUMNS = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_index",
    "object_track_key",
    "track_id",
    "pig_id",
    "object_id_in_image",
    "hidden_before_review",
    "behavior",
    "hidden_review_cohort",
)


def upgrade_hidden_review_manifest_identifiers(
    legacy_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Create a row-preserving identifier-v2 copy of a legacy manifest."""

    _require_columns(
        legacy_manifest,
        ["hidden_review_item_id", "hidden_before_review"],
        "legacy_manifest",
    )
    legacy_ids = _clean(legacy_manifest["hidden_review_item_id"])
    _require_nonempty_unique(legacy_ids, "legacy_hidden_review_item_id")
    upgraded = ensure_frame_object_identifiers(
        legacy_manifest,
        source_name="hidden_review_manifest_migration",
    )
    upgraded["legacy_hidden_review_item_id"] = legacy_ids.to_numpy()
    upgraded = attach_hidden_review_identifiers(upgraded)
    if len(upgraded) != len(legacy_manifest):
        raise AssertionError("Hidden manifest migration changed row count")
    return upgraded


def migrate_hidden_review_decisions(
    legacy_manifest: pd.DataFrame,
    upgraded_manifest: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Map every workload row and decision onto identifier v2.

    Mapping uses the label-independent subject key. The function keeps all
    human payload fields byte-for-byte equivalent as strings and reports every
    missing, extra, duplicate, stale, or ambiguous target instead of dropping
    rows.
    """

    _require_columns(
        legacy_manifest,
        ["hidden_review_item_id", "hidden_before_review"],
        "legacy_manifest",
    )
    _require_columns(
        upgraded_manifest,
        [
            "hidden_review_item_id",
            "hidden_review_subject_key",
            "hidden_review_key_version",
            "hidden_before_review",
        ],
        "upgraded_manifest",
    )
    _require_columns(decisions, DECISION_COLUMNS, "decisions")

    old = legacy_manifest.copy().reset_index(drop=True)
    new = upgraded_manifest.copy().reset_index(drop=True)
    old_ids = _clean(old["hidden_review_item_id"])
    new_ids = _clean(new["hidden_review_item_id"])
    decision_ids = _clean(decisions["hidden_review_item_id"])
    old_subject = build_hidden_review_subject_keys(old)
    new_subject = build_hidden_review_subject_keys(new)
    errors: list[str] = []
    warnings: list[str] = []

    _append_key_errors(errors, old_ids, "legacy_item_id")
    _append_key_errors(errors, new_ids, "upgraded_item_id")
    _append_key_errors(errors, decision_ids, "decision_item_id")
    _append_key_errors(errors, old_subject, "legacy_subject")
    _append_key_errors(errors, new_subject, "upgraded_subject")

    identifier_audit = audit_hidden_review_identifiers(new)
    errors.extend(
        f"upgraded_identifier_contract:{error}"
        for error in identifier_audit["errors"]
    )
    stored_subject = _raw_text(new["hidden_review_subject_key"])
    subject_drift = stored_subject.ne(new_subject)
    if subject_drift.any():
        errors.append(f"upgraded_subject_key_drift={int(subject_drift.sum())}")
    version = _clean(new["hidden_review_key_version"])
    invalid_version = version.ne(HIDDEN_REVIEW_KEY_VERSION.lower())
    if invalid_version.any():
        errors.append(
            f"invalid_upgraded_key_version={int(invalid_version.sum())}"
        )

    old_subject_set = set(old_subject)
    new_subject_set = set(new_subject)
    missing_subjects = old_subject_set.difference(new_subject_set)
    extra_subjects = new_subject_set.difference(old_subject_set)
    if missing_subjects:
        errors.append(f"legacy_subjects_missing_in_upgrade={len(missing_subjects)}")
    if extra_subjects:
        errors.append(f"unexpected_upgraded_subjects={len(extra_subjects)}")

    unknown_decisions = set(decision_ids).difference(set(old_ids))
    if unknown_decisions:
        errors.append(f"decision_ids_missing_from_legacy={len(unknown_decisions)}")

    mapping = _build_mapping(old, new, old_ids, new_ids, old_subject, new_subject)
    errors.extend(_context_invariant_errors(mapping))
    mapped_old_ids = set(mapping["legacy_hidden_review_item_id"])
    missing_mapping = set(old_ids).difference(mapped_old_ids)
    if missing_mapping:
        errors.append(f"legacy_manifest_rows_without_mapping={len(missing_mapping)}")

    migrated = _migrate_decision_rows(decisions, mapping, new)
    if len(migrated) != len(decisions):
        errors.append(
            "decision_row_count_changed="
            f"{len(decisions)}->{len(migrated)}"
        )
    unmapped_decisions = int(
        _clean(migrated["hidden_review_item_id"]).eq("").sum()
    )
    if unmapped_decisions:
        errors.append(f"unmapped_decision_rows={unmapped_decisions}")
    payload_changes = _human_payload_change_count(decisions, migrated)
    if payload_changes:
        errors.append(f"human_payload_changed_rows={payload_changes}")

    mapping["has_human_decision"] = mapping[
        "legacy_hidden_review_item_id"
    ].isin(set(decision_ids))
    audit = {
        "schema_version": HIDDEN_REVIEW_MIGRATION_VERSION,
        "legacy_manifest_rows": int(len(old)),
        "upgraded_manifest_rows": int(len(new)),
        "mapping_rows": int(len(mapping)),
        "decision_rows_before": int(len(decisions)),
        "decision_rows_after": int(len(migrated)),
        "mapped_decision_rows": int(mapping["has_human_decision"].sum()),
        "resolved_decision_rows": int(
            _clean(decisions["hidden_review_status"]).eq("reviewed").sum()
        ),
        "legacy_subject_count": int(old_subject.nunique()),
        "upgraded_subject_count": int(new_subject.nunique()),
        "missing_subject_count": int(len(missing_subjects)),
        "extra_subject_count": int(len(extra_subjects)),
        "unmapped_decision_rows": unmapped_decisions,
        "human_payload_changed_rows": payload_changes,
        "upgraded_identifier_audit": identifier_audit,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    return mapping, migrated, audit


def _build_mapping(
    old: pd.DataFrame,
    new: pd.DataFrame,
    old_ids: pd.Series,
    new_ids: pd.Series,
    old_subject: pd.Series,
    new_subject: pd.Series,
) -> pd.DataFrame:
    old_map = old.copy()
    old_map["legacy_hidden_review_item_id"] = old_ids
    old_map["hidden_review_subject_key"] = old_subject
    new_map = new.copy()
    new_map["hidden_review_item_id"] = new_ids
    new_map["hidden_review_subject_key"] = new_subject
    old_columns = [
        "legacy_hidden_review_item_id",
        "hidden_review_subject_key",
        *[col for col in CONTEXT_INVARIANT_COLUMNS if col in old_map.columns],
    ]
    new_columns = [
        "hidden_review_item_id",
        "hidden_review_subject_key",
        *[col for col in CONTEXT_INVARIANT_COLUMNS if col in new_map.columns],
    ]
    return old_map[old_columns].merge(
        new_map[new_columns],
        on="hidden_review_subject_key",
        how="inner",
        validate="one_to_one",
        suffixes=("_legacy", "_upgraded"),
    )


def _context_invariant_errors(mapping: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in CONTEXT_INVARIANT_COLUMNS:
        legacy = f"{column}_legacy"
        upgraded = f"{column}_upgraded"
        if legacy not in mapping.columns or upgraded not in mapping.columns:
            continue
        if column == "frame_index":
            left = pd.to_numeric(mapping[legacy], errors="coerce")
            right = pd.to_numeric(mapping[upgraded], errors="coerce")
        else:
            left = _clean(mapping[legacy])
            right = _clean(mapping[upgraded])
        changed = left.ne(right)
        if changed.any():
            errors.append(f"context_invariant_changed:{column}={int(changed.sum())}")
    return errors


def _migrate_decision_rows(
    decisions: pd.DataFrame,
    mapping: pd.DataFrame,
    upgraded_manifest: pd.DataFrame,
) -> pd.DataFrame:
    migrated = decisions.copy().reset_index(drop=True)
    old_ids = _clean(migrated["hidden_review_item_id"])
    id_map = mapping.set_index("legacy_hidden_review_item_id")[
        "hidden_review_item_id"
    ]
    subject_map = mapping.set_index("legacy_hidden_review_item_id")[
        "hidden_review_subject_key"
    ]
    migrated.insert(0, "legacy_hidden_review_item_id", old_ids)
    migrated["hidden_review_item_id"] = old_ids.map(id_map).fillna("")
    migrated["hidden_review_key_version"] = HIDDEN_REVIEW_KEY_VERSION
    migrated["hidden_review_subject_key"] = old_ids.map(subject_map).fillna("")

    new_by_id = upgraded_manifest.set_index("hidden_review_item_id")
    for column in ["identifier_schema_version", "scene_frame_uid", "frame_uid"]:
        if column not in new_by_id.columns:
            continue
        migrated[column] = migrated["hidden_review_item_id"].map(
            new_by_id[column]
        )
    migrated["hidden_review_migration_version"] = HIDDEN_REVIEW_MIGRATION_VERSION
    return migrated


def _human_payload_change_count(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> int:
    before_rows = before.reset_index(drop=True)
    after_rows = after.reset_index(drop=True)
    changed = pd.Series(False, index=before_rows.index)
    for column in HUMAN_PAYLOAD_COLUMNS:
        changed |= before_rows[column].fillna("").astype(str).ne(
            after_rows[column].fillna("").astype(str)
        )
    return int(changed.sum())


def _append_key_errors(
    errors: list[str],
    values: pd.Series,
    name: str,
) -> None:
    blank = values.eq("")
    duplicate = values.ne("") & values.duplicated(keep=False)
    if blank.any():
        errors.append(f"blank_{name}={int(blank.sum())}")
    if duplicate.any():
        errors.append(f"duplicate_{name}={int(duplicate.sum())}")


def _require_nonempty_unique(values: pd.Series, name: str) -> None:
    errors: list[str] = []
    _append_key_errors(errors, values, name)
    if errors:
        raise ValueError(f"Invalid {name}: {errors}")


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _clean(values: pd.Series) -> pd.Series:
    cleaned = values.fillna("").astype(str).str.strip()
    return cleaned.mask(cleaned.isin({"nan", "None", "<NA>"}), "").str.lower()


def _raw_text(values: pd.Series) -> pd.Series:
    cleaned = values.fillna("").astype(str).str.strip()
    return cleaned.mask(cleaned.isin({"nan", "None", "<NA>"}), "")

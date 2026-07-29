"""Fail-closed view of current Behavior candidates unaffected by interaction logic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SAFE_VIEW_SCHEMA_VERSION = (
    "classification_v2.safe_non_interaction_review_view.v1"
)
SAFE_VIEW_SEMANTIC_STATUS = "PRE_REVIEW_CALIBRATION_INFRASTRUCTURE"
ROI_DIRECTION_CORRECTED_VIEW_SCHEMA_VERSION = (
    "classification_v2.roi_direction_corrected_noninteraction_view.v1"
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "y"})
_INTERACTION_BEHAVIORS = frozenset({"fight", "social-nose"})
_ROI_LABELED_BEHAVIORS = frozenset({"eat", "drink", "playwithtoy"})
_SAFE_TEMPLATES = frozenset({"motion", "posture", "roi"})
_EXPLORE_ROI_FALSE_NEGATIVE_PREDICATES = frozenset(
    {"roi_possible_false_negative", "risk_triggered"}
)
_AFFECTED_BOOLEAN_FIELDS = (
    "review_predicate_interaction_contradiction",
    "review_predicate_partner_context_insufficient",
)
_AFFECTED_TOKEN_FIELDS = (
    "review_reason_codes",
    "review_reason",
    "review_selection_predicates",
    "review_evidence_reason_auto",
    "interval_review_reason",
)
_AFFECTED_TOKENS = (
    "interaction_contradiction",
    "interaction_requires_partner_context",
    "partner_context_insufficient",
    "persistent_partner_contact",
    "persistent_contact_or_aggression",
    "social_nose_with_fight_like_motion",
    "social_evidence_unavailable",
)
_REQUIRED_COLUMNS = frozenset(
    {
        "review_unit_id",
        "behavior_label",
        "review_template",
        "candidate_tier",
        "include_in_review",
        "review_reason_codes",
        "review_selection_predicates",
        "selection_predicate_version",
        "selection_config_hash",
    }
)


class SafeNonInteractionViewError(ValueError):
    """Raised when a safe view cannot be established without guessing."""


@dataclass(frozen=True)
class SafeNonInteractionViewResult:
    """Filtered view plus row-level dependency classification and audit."""

    view: pd.DataFrame
    dependency: pd.DataFrame
    audit: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Hash an immutable input or generated view."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in _TRUE_VALUES


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def _semicolon_tokens(value: object) -> set[str]:
    return {
        token.strip().casefold()
        for token in _text(value).split(";")
        if token.strip()
    }


def _dependency_reasons(row: pd.Series) -> tuple[list[str], bool]:
    reasons: list[str] = []
    unknown = False
    behavior = _text(row.get("behavior_label"))
    template = _text(row.get("review_template"))

    if behavior in _INTERACTION_BEHAVIORS:
        reasons.append(f"interaction_behavior={behavior}")
    if template == "interaction":
        reasons.append("interaction_review_template")
    elif template not in _SAFE_TEMPLATES:
        reasons.append(f"unknown_review_template={template or 'blank'}")
        unknown = True

    for field in _AFFECTED_BOOLEAN_FIELDS:
        if field not in row.index:
            reasons.append(f"missing_dependency_field={field}")
            unknown = True
        elif _truth(row.get(field)):
            reasons.append(f"active_dependency_field={field}")

    token_text = "|".join(_text(row.get(field)) for field in _AFFECTED_TOKEN_FIELDS)
    matched = sorted(token for token in _AFFECTED_TOKENS if token in token_text)
    reasons.extend(f"interaction_token={token}" for token in matched)

    if behavior not in _INTERACTION_BEHAVIORS and template == "interaction":
        unknown = True
    if behavior in _INTERACTION_BEHAVIORS and template != "interaction":
        unknown = True
        reasons.append("behavior_template_mismatch")
    return sorted(set(reasons)), unknown


def classify_candidate_dependencies(candidates: pd.DataFrame) -> pd.DataFrame:
    """Classify every current candidate without changing any source value."""

    missing = sorted(_REQUIRED_COLUMNS.difference(candidates.columns))
    if missing:
        raise SafeNonInteractionViewError(
            f"candidate manifest missing required columns: {missing}"
        )
    ids = candidates["review_unit_id"].fillna("").astype(str).str.strip()
    if ids.eq("").any():
        raise SafeNonInteractionViewError("candidate manifest has blank review keys")
    if ids.duplicated().any():
        raise SafeNonInteractionViewError(
            "candidate manifest has duplicate review keys"
        )
    included = candidates["include_in_review"].map(_truth)
    if not included.all():
        raise SafeNonInteractionViewError(
            f"candidate manifest contains non-candidates: {int((~included).sum())}"
        )
    auto = (
        candidates["candidate_tier"]
        .fillna("")
        .astype(str)
        .eq("AUTO_CARRY_LOW_RISK")
    )
    if auto.any():
        raise SafeNonInteractionViewError(
            f"candidate manifest contains auto-carry rows: {int(auto.sum())}"
        )

    records: list[dict[str, object]] = []
    for index, row in candidates.iterrows():
        reasons, unknown = _dependency_reasons(row)
        affected = bool(reasons)
        records.append(
            {
                "_source_index": int(index),
                "review_unit_id": str(row["review_unit_id"]).strip(),
                "interaction_affected": affected,
                "unknown_dependency": bool(unknown),
                "declared_exclusion_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame.from_records(records)


def build_safe_non_interaction_view(
    candidates: pd.DataFrame,
    *,
    producer_sha: str,
    input_sha256: str,
) -> SafeNonInteractionViewResult:
    """Return an exact-row subset of the immutable candidate publication."""

    if len(producer_sha) != 40:
        raise SafeNonInteractionViewError("producer_sha must be a full Git SHA")
    if len(input_sha256) != 64:
        raise SafeNonInteractionViewError("input_sha256 must be SHA-256")

    dependency = classify_candidate_dependencies(candidates)
    safe_mask = ~dependency["interaction_affected"]
    safe_indices = dependency.loc[safe_mask, "_source_index"].tolist()
    view = candidates.loc[safe_indices].copy().reset_index(drop=True)
    safe_ids = set(view["review_unit_id"].astype(str))
    candidate_ids = set(candidates["review_unit_id"].astype(str))
    if not safe_ids.issubset(candidate_ids):
        raise SafeNonInteractionViewError("safe view added unknown review keys")

    excluded = dependency.loc[~safe_mask]
    reason_counts: dict[str, int] = {}
    for value in excluded["declared_exclusion_reasons"].astype(str):
        for reason in filter(None, value.split(";")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    audit = {
        "schema_version": SAFE_VIEW_SCHEMA_VERSION,
        "semantic_status": SAFE_VIEW_SEMANTIC_STATUS,
        "authority_role": "VIEW_ONLY_NOT_CANDIDATE_PUBLICATION",
        "producer_sha": producer_sha,
        "input_hashes": {"current_candidate_manifest_sha256": input_sha256},
        "total_current_candidates": int(len(candidates)),
        "safe_non_interaction_view_count": int(len(view)),
        "excluded_interaction_affected_count": int((~safe_mask).sum()),
        "unknown_dependency_count": int(
            dependency["unknown_dependency"].sum()
        ),
        "view_keys_subset_of_current_candidates": True,
        "new_keys_added": 0,
        "current_candidate_publication_changed": False,
        "current_auto_carry_changed": False,
        "current_universe_changed": False,
        "same_workspace_subset_resume_safe": False,
        "separate_non_interaction_view_workspace_required": True,
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "excluded_review_key_set_sha256": hashlib.sha256(
            "\n".join(sorted(excluded["review_unit_id"].astype(str))).encode()
        ).hexdigest(),
        "safe_review_key_set_sha256": hashlib.sha256(
            "\n".join(sorted(safe_ids)).encode()
        ).hexdigest(),
    }
    return SafeNonInteractionViewResult(
        view=view,
        dependency=dependency,
        audit=audit,
    )


def build_roi_direction_corrected_noninteraction_view(
    safe_view: pd.DataFrame,
    *,
    preserve_review_keys: set[str] | None = None,
) -> SafeNonInteractionViewResult:
    """Remove only the inverted explore-near-ROI hard-review trigger.

    The input must already be the frozen non-interaction subset. Existing
    reviewed keys remain in the view so a running workspace can resume without
    losing decisions. Rows carrying another predicate, ROI-labeled behaviors,
    and the stratified audit remain reviewable.
    """

    required = {
        "review_unit_id",
        "behavior_label",
        "review_selection_predicates",
    }
    missing = sorted(required.difference(safe_view.columns))
    if missing:
        raise SafeNonInteractionViewError(
            f"safe view missing ROI correction columns: {missing}"
        )
    ids = safe_view["review_unit_id"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise SafeNonInteractionViewError(
            "safe view requires unique nonblank review keys"
        )

    preserved = set(preserve_review_keys or set())
    unknown_preserved = sorted(preserved.difference(ids))
    if unknown_preserved:
        raise SafeNonInteractionViewError(
            f"preserved review keys absent from safe view: {len(unknown_preserved)}"
        )

    behavior = safe_view["behavior_label"].fillna("").astype(str)
    predicate_sets = (
        safe_view["review_selection_predicates"]
        .fillna("")
        .astype(str)
        .map(_semicolon_tokens)
    )
    roi_only_explore = behavior.eq("explore") & predicate_sets.map(
        lambda values: (
            "roi_possible_false_negative" in values
            and values.issubset(_EXPLORE_ROI_FALSE_NEGATIVE_PREDICATES)
        )
    )
    preserved_mask = ids.isin(preserved)
    suppress = roi_only_explore & ~preserved_mask
    view = safe_view.loc[~suppress].copy().reset_index(drop=True)

    records = pd.DataFrame(
        {
            "review_unit_id": ids,
            "roi_only_explore_trigger": roi_only_explore,
            "preserved_existing_review": preserved_mask,
            "included_in_corrected_view": ~suppress,
            "correction_reason": [
                (
                    "PRESERVE_EXISTING_REVIEW"
                    if preserve
                    else (
                        "SUPPRESS_INVERTED_EXPLORE_NEAR_ROI_TRIGGER"
                        if inverted
                        else "UNCHANGED_REVIEW_REASON"
                    )
                )
                for inverted, preserve in zip(
                    roi_only_explore,
                    preserved_mask,
                    strict=True,
                )
            ],
        }
    )
    audit = {
        "schema_version": ROI_DIRECTION_CORRECTED_VIEW_SCHEMA_VERSION,
        "authority_role": "VIEW_ONLY_NOT_CANDIDATE_PUBLICATION",
        "input_safe_view_count": int(len(safe_view)),
        "corrected_view_count": int(len(view)),
        "suppressed_roi_only_explore_count": int(suppress.sum()),
        "preserved_existing_review_count": int(
            (roi_only_explore & preserved_mask).sum()
        ),
        "roi_labeled_behavior_review_count": int(
            view["behavior_label"].isin(_ROI_LABELED_BEHAVIORS).sum()
        ),
        "stratified_audit_review_count": int(
            view["candidate_tier"].eq("TIER_3_STRATIFIED_AUDIT").sum()
            if "candidate_tier" in view
            else 0
        ),
        "new_keys_added": 0,
        "source_rows_changed": False,
        "candidate_publication_changed": False,
        "auto_carry_publication_changed": False,
        "review_key_set_sha256": hashlib.sha256(
            "\n".join(view["review_unit_id"].astype(str)).encode()
        ).hexdigest(),
    }
    return SafeNonInteractionViewResult(
        view=view,
        dependency=records,
        audit=audit,
    )


def audit_safe_non_interaction_view(
    candidates: pd.DataFrame,
    view: pd.DataFrame,
    *,
    expected_candidate_sha256: str,
    actual_candidate_sha256: str,
) -> dict[str, Any]:
    """Independently check subset, exact rows, and declared exclusions."""

    errors: list[str] = []
    if expected_candidate_sha256 != actual_candidate_sha256:
        errors.append("candidate_manifest_hash_mismatch")
    try:
        dependency = classify_candidate_dependencies(candidates)
    except SafeNonInteractionViewError as exc:
        return {"valid": False, "errors": [str(exc)]}

    candidate_ids = candidates["review_unit_id"].astype(str)
    view_ids = view.get("review_unit_id", pd.Series(dtype="object")).astype(str)
    if view_ids.duplicated().any():
        errors.append("duplicate_view_review_keys")
    extra = sorted(set(view_ids).difference(candidate_ids))
    if extra:
        errors.append(f"new_view_keys={len(extra)}")

    expected = dependency.loc[
        ~dependency["interaction_affected"], "review_unit_id"
    ].tolist()
    if view_ids.tolist() != expected:
        errors.append("safe_view_order_or_membership_mismatch")

    indexed_candidates = candidates.set_index("review_unit_id", drop=False)
    indexed_view = view.set_index("review_unit_id", drop=False)
    shared = indexed_view.index.intersection(indexed_candidates.index)
    if list(view.columns) != list(candidates.columns):
        errors.append("safe_view_columns_changed")
    else:
        source_rows = indexed_candidates.loc[shared]
        view_rows = indexed_view.loc[shared]
        changed_columns: list[str] = []
        for column in candidates.columns:
            source_values = source_rows[column]
            view_values = view_rows[column]
            if pd.api.types.is_bool_dtype(source_values):
                equal = source_values.fillna(False).eq(
                    view_values.fillna(False).astype(bool)
                )
            elif pd.api.types.is_numeric_dtype(source_values):
                source_numeric = pd.to_numeric(source_values, errors="coerce")
                view_numeric = pd.to_numeric(view_values, errors="coerce")
                equal = (
                    source_numeric.eq(view_numeric)
                    | (source_numeric.isna() & view_numeric.isna())
                    | (
                        (source_numeric - view_numeric).abs()
                        <= 1e-12
                        * pd.concat(
                            [
                                source_numeric.abs(),
                                view_numeric.abs(),
                                pd.Series(1.0, index=source_numeric.index),
                            ],
                            axis=1,
                        ).max(axis=1)
                    )
                )
            else:
                equal = source_values.fillna("<NA>").astype(str).eq(
                    view_values.fillna("<NA>").astype(str)
                )
            if not equal.all():
                changed_columns.append(column)
        if changed_columns:
            errors.append(
                "safe_view_source_rows_changed="
                + ",".join(changed_columns[:20])
            )

    unexplained = dependency.loc[
        dependency["interaction_affected"]
        & dependency["declared_exclusion_reasons"].eq("")
    ]
    if not unexplained.empty:
        errors.append(f"unexplained_exclusions={len(unexplained)}")
    unknown = int(dependency["unknown_dependency"].sum())
    if unknown:
        errors.append(f"unknown_dependency_count={unknown}")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_count": int(len(candidates)),
        "view_count": int(len(view)),
        "excluded_count": int(len(candidates) - len(view)),
        "unknown_dependency_count": unknown,
        "new_keys_added": int(len(extra)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON for a diagnostic artifact."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ROI_DIRECTION_CORRECTED_VIEW_SCHEMA_VERSION",
    "SAFE_VIEW_SCHEMA_VERSION",
    "SAFE_VIEW_SEMANTIC_STATUS",
    "SafeNonInteractionViewError",
    "SafeNonInteractionViewResult",
    "audit_safe_non_interaction_view",
    "build_roi_direction_corrected_noninteraction_view",
    "build_safe_non_interaction_view",
    "classify_candidate_dependencies",
    "sha256_file",
    "write_json",
]

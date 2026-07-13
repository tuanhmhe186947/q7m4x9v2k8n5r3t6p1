"""Build and apply two-sided Hidden review for classification_v2.

Hidden is a frame/object visibility attribute. It is not a behavior target and
must not be propagated to an entire temporal interval without an explicit span
decision. CVAT Hidden values are treated as tracking-derived and untrusted until
reviewed; legacy values retain their prior-review provenance.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

HIDDEN_REVIEW_COHORTS: tuple[str, ...] = (
    "hidden_yes_confirmation",
    "hidden_no_high_risk",
    "hidden_no_random_audit",
    "hidden_no_clean_control",
)

DECISION_COLUMNS: tuple[str, ...] = (
    "hidden_review_item_id",
    "hidden_before_review",
    "hidden_after_review",
    "hidden_review_status",
    "hidden_review_confidence",
    "hidden_review_reason",
    "hidden_reviewer",
    "hidden_reviewed_at",
)

REQUIRED_FRAME_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_uid",
    "frame_index",
    "pig_id",
    "behavior",
    "hidden",
)


@dataclass(slots=True)
class HiddenReviewConfig:
    """Deterministic sampling and risk thresholds for Hidden review."""

    random_seed: int = 20260713
    random_no_per_stratum: int = 3
    clean_control_per_stratum: int = 1
    max_high_risk_per_stratum: int | None = None
    high_risk_threshold: float = 0.35
    clean_control_max_risk: float = 0.10
    pair_iou_threshold: float = 0.01
    pair_overlap_threshold: float = 0.05
    nearest_distance_threshold: float = 0.08
    shape_change_threshold: float = 0.25
    area_delta_threshold: float = 0.20
    stratum_columns: tuple[str, ...] = (
        "source_type",
        "video_key",
        "behavior",
    )

    def validate(self) -> None:
        """Reject sampling settings that would create ambiguous coverage."""
        if self.random_no_per_stratum < 0:
            raise ValueError("random_no_per_stratum must be >= 0")
        if self.clean_control_per_stratum < 0:
            raise ValueError("clean_control_per_stratum must be >= 0")
        if self.max_high_risk_per_stratum is not None:
            if self.max_high_risk_per_stratum <= 0:
                raise ValueError("max_high_risk_per_stratum must be > 0")
        if not 0 <= self.clean_control_max_risk <= self.high_risk_threshold <= 1:
            raise ValueError("risk thresholds must satisfy 0 <= clean <= high <= 1")


def balanced_hidden_smoke_scope(
    frame_features: pd.DataFrame,
    max_rows_per_source: int,
) -> pd.DataFrame:
    """Select a deterministic smoke scope containing Hidden Yes and No rows."""
    if max_rows_per_source <= 0:
        raise ValueError("max_rows_per_source must be > 0")
    _require_columns(frame_features, ["source_type", "hidden"], "frame_features")
    parts: list[pd.DataFrame] = []
    for _, group in frame_features.groupby("source_type", dropna=False, sort=True):
        yes = group.loc[group["hidden"].map(_normalize_hidden).eq("Yes")]
        no = group.loc[group["hidden"].map(_normalize_hidden).eq("No")]
        yes_quota = min(len(yes), max(1, max_rows_per_source // 4))
        no_quota = min(len(no), max_rows_per_source - yes_quota)
        selected = pd.concat(
            [yes.head(yes_quota), no.head(no_quota)],
            ignore_index=False,
        )
        remaining = max_rows_per_source - len(selected)
        if remaining > 0:
            used = set(selected.index)
            selected = pd.concat(
                [selected, group.loc[~group.index.isin(used)].head(remaining)],
                ignore_index=False,
            )
        parts.append(selected)
    if not parts:
        return frame_features.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=False).copy()


def build_hidden_review_manifest(
    frame_features: pd.DataFrame,
    *,
    config: HiddenReviewConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Build deterministic two-sided review cohorts without dropping source rows."""
    cfg = config or HiddenReviewConfig()
    cfg.validate()
    _require_columns(frame_features, REQUIRED_FRAME_COLUMNS, "frame_features")
    if frame_features.empty:
        raise ValueError("frame_features must not be empty")

    work = frame_features.copy().reset_index(drop=False)
    work = work.rename(columns={"index": "source_row_index"})
    work["hidden_before_review"] = work["hidden"].map(_normalize_hidden)
    work["hidden_review_item_id"] = _build_review_item_ids(work)
    _require_unique(work, "hidden_review_item_id", "frame_features")

    work["hidden_source"] = _initial_hidden_source(work)
    work["hidden_is_trusted_before_review"] = _initial_hidden_trust(work)
    work["hidden_trust_status_before_review"] = _initial_trust_status(work)
    risk_score, risk_reasons = _hidden_false_negative_risk(work, cfg)
    work["hidden_false_negative_risk_score"] = risk_score
    work["hidden_false_negative_risk_reasons"] = risk_reasons

    selected_parts: list[pd.DataFrame] = []
    hidden_yes = work["hidden_before_review"].eq("Yes")
    hidden_yes_rows = _add_sampling_metadata(
        work.loc[hidden_yes],
        work.loc[hidden_yes],
        cfg.stratum_columns,
        design="census_hidden_yes",
    )
    selected_parts.append(_assign_cohort(hidden_yes_rows, "hidden_yes_confirmation", 100))

    hidden_no = work.loc[~hidden_yes].copy()
    random_no = _sample_each_stratum(
        hidden_no,
        cfg.stratum_columns,
        cfg.random_no_per_stratum,
        cfg.random_seed,
        "random_audit",
    )
    random_no = _add_sampling_metadata(
        hidden_no,
        random_no,
        cfg.stratum_columns,
        design="stratified_random_hidden_no",
    )
    selected_parts.append(_assign_cohort(random_no, "hidden_no_random_audit", 40))

    random_ids = set(random_no["hidden_review_item_id"].astype(str))
    high_risk_pool = hidden_no.loc[
        hidden_no["hidden_false_negative_risk_score"].ge(cfg.high_risk_threshold)
        & ~hidden_no["hidden_review_item_id"].astype(str).isin(random_ids)
    ]
    high_risk = _cap_each_stratum(
        high_risk_pool,
        cfg.stratum_columns,
        cfg.max_high_risk_per_stratum,
        cfg.random_seed,
        "high_risk",
    )
    high_risk = _add_sampling_metadata(
        high_risk_pool,
        high_risk,
        cfg.stratum_columns,
        design="risk_enriched_hidden_no",
    )
    selected_parts.append(_assign_cohort(high_risk, "hidden_no_high_risk", 80))

    used_ids = random_ids | set(high_risk["hidden_review_item_id"].astype(str))
    remaining_no = hidden_no.loc[~hidden_no["hidden_review_item_id"].astype(str).isin(used_ids)]
    clean_pool = remaining_no.loc[
        remaining_no["hidden_false_negative_risk_score"].le(cfg.clean_control_max_risk)
    ]
    clean_no = _sample_each_stratum(
        clean_pool,
        cfg.stratum_columns,
        cfg.clean_control_per_stratum,
        cfg.random_seed,
        "clean_control",
    )
    clean_no = _add_sampling_metadata(
        clean_pool,
        clean_no,
        cfg.stratum_columns,
        design="clean_negative_control",
    )
    selected_parts.append(_assign_cohort(clean_no, "hidden_no_clean_control", 20))

    manifest = pd.concat(selected_parts, ignore_index=True, sort=False)
    manifest = _finalize_manifest(manifest)
    templates = {
        cohort: manifest.loc[manifest["hidden_review_cohort"].eq(cohort)].copy()
        for cohort in HIDDEN_REVIEW_COHORTS
    }
    audit = audit_hidden_review_manifest(
        frame_features,
        manifest,
        cfg,
        prepared_frame_features=work,
    )
    if audit["errors"]:
        raise ValueError(f"Hidden review manifest audit failed: {audit['errors']}")
    return manifest, templates, audit


def audit_hidden_review_manifest(
    frame_features: pd.DataFrame,
    manifest: pd.DataFrame,
    config: HiddenReviewConfig,
    *,
    prepared_frame_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return coverage, trust, and cohort evidence for a review manifest."""
    errors: list[str] = []
    warnings: list[str] = []
    if len(frame_features) <= 0:
        errors.append("empty_frame_features")
    if manifest.empty:
        errors.append("empty_hidden_review_manifest")
    if "hidden_review_item_id" not in manifest.columns:
        errors.append("missing_hidden_review_item_id")
    elif manifest["hidden_review_item_id"].duplicated().any():
        errors.append("duplicate_hidden_review_item_id")

    invalid_cohorts: list[str] = []
    if "hidden_review_cohort" in manifest.columns:
        invalid_cohorts = sorted(
            set(manifest["hidden_review_cohort"].dropna().astype(str)).difference(
                HIDDEN_REVIEW_COHORTS
            )
        )
    if invalid_cohorts:
        errors.append(f"invalid_hidden_review_cohorts={invalid_cohorts}")

    source = (
        prepared_frame_features.copy().reset_index(drop=True)
        if prepared_frame_features is not None
        else frame_features.copy().reset_index(drop=True)
    )
    if "hidden_before_review" not in source.columns:
        source["hidden_before_review"] = source["hidden"].map(_normalize_hidden)
    if "hidden_review_item_id" not in source.columns:
        source["hidden_review_item_id"] = _build_review_item_ids(source)
    _require_unique(source, "hidden_review_item_id", "frame_features")
    source_ids = set(source["hidden_review_item_id"].astype(str))
    manifest_ids = set(manifest.get("hidden_review_item_id", pd.Series(dtype=str)).astype(str))
    unknown_manifest_ids = manifest_ids.difference(source_ids)
    if unknown_manifest_ids:
        errors.append(f"manifest_items_outside_input_scope={len(unknown_manifest_ids)}")

    source_yes_ids = set(
        source.loc[
            source["hidden"].map(_normalize_hidden).eq("Yes"),
            "hidden_review_item_id",
        ].astype(str)
    )
    manifest_yes_ids = set(
        manifest.loc[
            manifest.get(
                "hidden_review_cohort",
                pd.Series("", index=manifest.index),
            ).eq("hidden_yes_confirmation"),
            "hidden_review_item_id",
        ].astype(str)
    )
    missing_yes_ids = source_yes_ids.difference(manifest_yes_ids)
    extra_yes_ids = manifest_yes_ids.difference(source_yes_ids)
    if missing_yes_ids:
        errors.append(f"hidden_yes_items_missing_from_manifest={len(missing_yes_ids)}")
    if extra_yes_ids:
        errors.append(f"invalid_hidden_yes_confirmation_items={len(extra_yes_ids)}")

    if "hidden_false_negative_risk_score" in source.columns:
        source_risk = pd.to_numeric(
            source["hidden_false_negative_risk_score"],
            errors="coerce",
        ).fillna(0.0)
    else:
        source_risk, _ = _hidden_false_negative_risk(source, config)
    source_high_risk_ids = set(
        source.loc[
            source["hidden_before_review"].eq("No") & source_risk.ge(config.high_risk_threshold),
            "hidden_review_item_id",
        ].astype(str)
    )
    selected_high_risk_ids = source_high_risk_ids.intersection(manifest_ids)

    before = manifest.get(
        "hidden_before_review",
        pd.Series("", index=manifest.index),
    ).map(_normalize_hidden)
    cohort = manifest.get(
        "hidden_review_cohort",
        pd.Series("", index=manifest.index),
    ).astype(str)
    invalid_yes_cohort = cohort.eq("hidden_yes_confirmation") & before.ne("Yes")
    invalid_no_cohort = ~cohort.eq("hidden_yes_confirmation") & before.ne("No")
    if invalid_yes_cohort.any() or invalid_no_cohort.any():
        errors.append("hidden_review_cohort_before_value_mismatch")

    source_no = frame_features.loc[frame_features["hidden"].map(_normalize_hidden).eq("No")]
    selected_no = manifest["hidden_before_review"].eq("No")
    if source_no.empty:
        warnings.append("no_hidden_no_rows_available_for_false_negative_audit")
    elif not selected_no.any():
        errors.append("no_hidden_no_rows_selected_for_false_negative_audit")

    return {
        "input_rows": int(len(frame_features)),
        "input_hidden": _counts(frame_features, "hidden"),
        "input_sources": _counts(frame_features, "source_type"),
        "manifest_rows": int(len(manifest)),
        "manifest_unique_items": int(
            manifest.get("hidden_review_item_id", pd.Series(dtype=str)).nunique()
        ),
        "input_hidden_yes_items": int(len(source_yes_ids)),
        "manifest_hidden_yes_items": int(len(manifest_yes_ids)),
        "missing_hidden_yes_items": int(len(missing_yes_ids)),
        "input_high_risk_hidden_no_items": int(len(source_high_risk_ids)),
        "selected_high_risk_hidden_no_items": int(len(selected_high_risk_ids)),
        "unselected_high_risk_hidden_no_items": int(
            len(source_high_risk_ids.difference(manifest_ids))
        ),
        "cohort_counts": _counts(manifest, "hidden_review_cohort"),
        "source_counts": _counts(manifest, "source_type"),
        "behavior_counts": _counts(manifest, "behavior"),
        "trust_before_counts": _counts(
            manifest,
            "hidden_trust_status_before_review",
        ),
        "selected_hidden_no_rows": int(selected_no.sum()),
        "unselected_hidden_no_rows": int(len(source_no) - selected_no.sum()),
        "random_seed": config.random_seed,
        "config": _config_payload(config),
        "errors": errors,
        "warnings": warnings,
    }


def audit_hidden_decision_coverage(
    manifest: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    require_resolved: bool = True,
) -> dict[str, Any]:
    """Validate one resolved decision for every selected review item."""
    _require_columns(
        manifest,
        ["hidden_review_item_id", "hidden_before_review"],
        "manifest",
    )
    _require_columns(decisions, DECISION_COLUMNS, "decisions")
    errors: list[str] = []
    warnings: list[str] = []

    duplicate_manifest = int(manifest["hidden_review_item_id"].duplicated().sum())
    duplicate_decisions = int(decisions["hidden_review_item_id"].astype(str).duplicated().sum())
    if duplicate_manifest:
        errors.append(f"duplicate_manifest_items={duplicate_manifest}")
    if duplicate_decisions:
        errors.append(f"duplicate_decision_items={duplicate_decisions}")

    manifest_ids = set(manifest["hidden_review_item_id"].astype(str))
    decision_ids = set(decisions["hidden_review_item_id"].astype(str))
    missing_ids = manifest_ids.difference(decision_ids)
    extra_ids = decision_ids.difference(manifest_ids)
    if missing_ids:
        errors.append(f"missing_decision_items={len(missing_ids)}")
    if extra_ids:
        errors.append(f"unknown_decision_items={len(extra_ids)}")

    normalized = _normalize_decisions(decisions)
    resolved = normalized["hidden_review_status"].eq("reviewed")
    unclear = normalized["hidden_review_status"].eq("unclear")
    pending = ~resolved & ~unclear
    invalid_after = resolved & ~normalized["hidden_after_review"].isin(["Yes", "No"])
    if invalid_after.any():
        errors.append(f"resolved_without_hidden_value={int(invalid_after.sum())}")
    if require_resolved and unclear.any():
        errors.append(f"unclear_decision_items={int(unclear.sum())}")
    if require_resolved and pending.any():
        errors.append(f"pending_decision_items={int(pending.sum())}")

    joined = manifest[["hidden_review_item_id", "hidden_before_review"]].merge(
        normalized[["hidden_review_item_id", "hidden_before_review"]],
        on="hidden_review_item_id",
        how="inner",
        suffixes=("_manifest", "_decision"),
    )
    stale = (
        joined["hidden_before_review_manifest"]
        .map(_normalize_hidden)
        .ne(joined["hidden_before_review_decision"].map(_normalize_hidden))
    )
    if stale.any():
        errors.append(f"stale_hidden_before_review={int(stale.sum())}")

    return {
        "manifest_items": int(len(manifest)),
        "decision_rows": int(len(decisions)),
        "covered_items": int(len(manifest_ids.intersection(decision_ids))),
        "missing_items": int(len(missing_ids)),
        "unknown_items": int(len(extra_ids)),
        "resolved_items": int(resolved.sum()),
        "unclear_items": int(unclear.sum()),
        "pending_items": int(pending.sum()),
        "decision_status_counts": _counts(normalized, "hidden_review_status"),
        "errors": errors,
        "warnings": warnings,
    }


def apply_hidden_review_decisions(
    frame_features: pd.DataFrame,
    manifest: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    require_resolved: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Apply frame/object decisions while preserving every input row."""
    _require_columns(frame_features, REQUIRED_FRAME_COLUMNS, "frame_features")
    coverage = audit_hidden_decision_coverage(
        manifest,
        decisions,
        require_resolved=require_resolved,
    )
    if coverage["errors"]:
        raise ValueError(f"Hidden decision coverage failed: {coverage['errors']}")

    out = frame_features.copy().reset_index(drop=True)
    out["hidden_review_item_id"] = _build_review_item_ids(out)
    _require_unique(out, "hidden_review_item_id", "frame_features")
    out = _initialize_hidden_provenance(out)

    normalized = _normalize_decisions(decisions)
    decision_map = normalized.set_index("hidden_review_item_id")
    manifest_map = manifest.set_index("hidden_review_item_id")
    matched_count = 0
    for row_index, item_id in out["hidden_review_item_id"].items():
        item_id = str(item_id)
        if item_id not in decision_map.index:
            continue
        decision = decision_map.loc[item_id]
        expected_before = _normalize_hidden(manifest_map.loc[item_id, "hidden_before_review"])
        actual_before = _normalize_hidden(out.at[row_index, "hidden_before_review"])
        if expected_before != actual_before:
            raise ValueError(
                "Hidden decision does not match source snapshot for "
                f"{item_id}: expected={expected_before} actual={actual_before}"
            )

        status = str(decision["hidden_review_status"])
        if status == "reviewed":
            _apply_resolved_decision(out, row_index, decision)
        elif status == "unclear":
            _apply_unclear_decision(out, row_index, decision)
        matched_count += 1

    out["hidden_effective_for_policy"] = out["hidden"].map(_normalize_hidden).eq("Yes") & _to_bool(
        out["hidden_is_trusted"]
    )
    out["hidden_review_available_mask"] = _to_bool(out["hidden_is_trusted"])

    if len(out) != len(frame_features):
        raise AssertionError("Hidden apply changed row count")
    audit = _audit_applied_hidden(
        frame_features,
        out,
        manifest,
        normalized,
        coverage,
        matched_count,
    )
    confusion = _hidden_confusion_audit(manifest, normalized)
    if audit["errors"]:
        raise ValueError(f"Hidden apply audit failed: {audit['errors']}")
    return out, audit, confusion


def _hidden_false_negative_risk(
    frame_features: pd.DataFrame,
    config: HiddenReviewConfig,
) -> tuple[pd.Series, pd.Series]:
    """Compute label-independent visibility risk used only for review sampling."""
    score = pd.Series(0.0, index=frame_features.index)
    reasons: dict[int, list[str]] = {int(idx): [] for idx in frame_features.index}

    def add(mask: pd.Series, weight: float, reason: str) -> None:
        active = _to_bool(mask).reindex(frame_features.index, fill_value=False)
        score.loc[active] = score.loc[active] + weight
        for idx in frame_features.index[active]:
            reasons[int(idx)].append(reason)

    add(_bool_column(frame_features, "bbox_was_clipped"), 0.15, "bbox_clipped")
    add(
        _numeric(frame_features, "nearest_pair_iou").ge(config.pair_iou_threshold),
        0.25,
        "pair_iou",
    )
    add(
        _numeric(frame_features, "nearest_pair_overlap_ratio").ge(config.pair_overlap_threshold),
        0.20,
        "pair_overlap",
    )
    nearest_distance = _numeric(frame_features, "nearest_dist_n")
    add(
        nearest_distance.notna() & nearest_distance.le(config.nearest_distance_threshold),
        0.15,
        "close_partner",
    )
    add(
        _bool_column(frame_features, "pair_contact_with_nearest"),
        0.20,
        "pair_contact",
    )
    add(
        _numeric(frame_features, "shape_change_score").ge(config.shape_change_threshold),
        0.15,
        "abrupt_shape_change",
    )
    add(
        _numeric(frame_features, "delta_area_n").abs().ge(config.area_delta_threshold),
        0.10,
        "abrupt_area_change",
    )
    add(
        frame_features["behavior"].astype(str).isin(["fight", "social-nose"]),
        0.10,
        "interaction_scene",
    )

    hidden_yes = frame_features["hidden_before_review"].eq("Yes")
    track_key = _review_track_key(frame_features)
    ordered = pd.DataFrame(
        {
            "track_key": track_key,
            "frame_index": _numeric(frame_features, "frame_index"),
            "hidden_yes": hidden_yes,
        },
        index=frame_features.index,
    ).sort_values(["track_key", "frame_index"], kind="mergesort")
    near_hidden = ordered.groupby("track_key", dropna=False)["hidden_yes"].shift(1).fillna(
        False
    ) | ordered.groupby("track_key", dropna=False)["hidden_yes"].shift(-1).fillna(False)
    add(near_hidden.reindex(frame_features.index), 0.25, "adjacent_hidden")

    score = score.clip(lower=0.0, upper=1.0)
    reason_series = pd.Series(
        {idx: ";".join(values) for idx, values in reasons.items()},
        index=frame_features.index,
        dtype="object",
    )
    return score, reason_series


def _build_review_item_ids(df: pd.DataFrame) -> pd.Series:
    """Build deterministic frame/object keys independent of mutable labels."""
    fields = [
        "source_type",
        "dataset_id",
        "video_key",
        "frame_uid",
        "frame_index",
        "track_id",
        "pig_id",
        "object_id_in_image",
    ]

    def make_id(row: pd.Series) -> str:
        raw = "|".join(_stable_text(row.get(field, "")) for field in fields)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"hidden_item_{digest}"

    return df.apply(make_id, axis=1)


def _initialize_hidden_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize source-aware trust without accepting CVAT tracking metadata."""
    out = df.copy()
    current_hidden = out["hidden"].map(_normalize_hidden)
    if "hidden_before_review" in out.columns:
        before = out["hidden_before_review"].fillna("").astype(str).str.strip()
        out["hidden_before_review"] = before.where(before.ne(""), current_hidden)
        out["hidden_before_review"] = out["hidden_before_review"].map(_normalize_hidden)
    else:
        out["hidden_before_review"] = current_hidden
    out["hidden"] = current_hidden
    out["hidden_after_review"] = ""
    out["hidden_source"] = _initial_hidden_source(out)
    out["hidden_is_trusted"] = _initial_hidden_trust(out)
    out["hidden_review_status"] = _initial_review_status(out)
    out["hidden_trust_status"] = _initial_trust_status(out)
    out["hidden_review_confidence"] = ""
    out["hidden_review_reason"] = ""
    out["hidden_reviewer"] = ""
    out["hidden_reviewed_at"] = ""
    out["visibility_quality"] = "unreviewed_tracking_derived"
    legacy = out["source_type"].astype(str).eq("legacy_recovered")
    out.loc[legacy & current_hidden.eq("Yes"), "visibility_quality"] = "hidden_prior_review"
    out.loc[legacy & current_hidden.eq("No"), "visibility_quality"] = "visible_prior_review"
    out.loc[out["hidden_review_status"].eq("unclear"), "visibility_quality"] = "unclear"
    return out


def _apply_resolved_decision(
    out: pd.DataFrame,
    row_index: int,
    decision: pd.Series,
) -> None:
    """Apply one resolved review decision and its complete provenance."""
    after = _normalize_hidden(decision["hidden_after_review"])
    out.at[row_index, "hidden"] = after
    out.at[row_index, "hidden_after_review"] = after
    out.at[row_index, "hidden_review_status"] = "reviewed"
    out.at[row_index, "hidden_review_confidence"] = str(decision["hidden_review_confidence"])
    out.at[row_index, "hidden_review_reason"] = str(decision["hidden_review_reason"])
    out.at[row_index, "hidden_reviewer"] = str(decision["hidden_reviewer"])
    out.at[row_index, "hidden_reviewed_at"] = str(decision["hidden_reviewed_at"])
    out.at[row_index, "hidden_is_trusted"] = True
    out.at[row_index, "hidden_trust_status"] = "trusted_current_review"
    out.at[row_index, "hidden_source"] = "current_human_review"
    out.at[row_index, "visibility_quality"] = (
        "hidden_reviewed" if after == "Yes" else "visible_reviewed"
    )


def _apply_unclear_decision(
    out: pd.DataFrame,
    row_index: int,
    decision: pd.Series,
) -> None:
    """Keep the original value but explicitly mark visibility as unresolved."""
    out.at[row_index, "hidden_after_review"] = ""
    out.at[row_index, "hidden_review_status"] = "unclear"
    out.at[row_index, "hidden_review_confidence"] = str(decision["hidden_review_confidence"])
    out.at[row_index, "hidden_review_reason"] = str(decision["hidden_review_reason"])
    out.at[row_index, "hidden_reviewer"] = str(decision["hidden_reviewer"])
    out.at[row_index, "hidden_reviewed_at"] = str(decision["hidden_reviewed_at"])
    out.at[row_index, "hidden_is_trusted"] = False
    out.at[row_index, "hidden_trust_status"] = "unclear_current_review"
    out.at[row_index, "hidden_source"] = "current_human_review_unclear"
    out.at[row_index, "visibility_quality"] = "unclear"


def _audit_applied_hidden(
    before: pd.DataFrame,
    after: pd.DataFrame,
    manifest: pd.DataFrame,
    decisions: pd.DataFrame,
    coverage: dict[str, Any],
    matched_count: int,
) -> dict[str, Any]:
    """Prove row preservation and summarize visibility corrections."""
    errors: list[str] = []
    if len(before) != len(after):
        errors.append("row_count_changed")
    if after["hidden_review_item_id"].duplicated().any():
        errors.append("duplicate_hidden_review_item_id_after_apply")
    changed = (
        after["hidden_before_review"]
        .map(_normalize_hidden)
        .ne(after["hidden"].map(_normalize_hidden))
    )
    return {
        "input_rows": int(len(before)),
        "output_rows": int(len(after)),
        "manifest_rows": int(len(manifest)),
        "decision_rows": int(len(decisions)),
        "matched_decisions": int(matched_count),
        "corrected_hidden_rows": int(changed.sum()),
        "yes_to_no_rows": int(
            (after["hidden_before_review"].eq("Yes") & after["hidden"].eq("No")).sum()
        ),
        "no_to_yes_rows": int(
            (after["hidden_before_review"].eq("No") & after["hidden"].eq("Yes")).sum()
        ),
        "hidden_before_counts": _counts(after, "hidden_before_review"),
        "hidden_after_counts": _counts(after, "hidden"),
        "review_status_counts": _counts(after, "hidden_review_status"),
        "trust_status_counts": _counts(after, "hidden_trust_status"),
        "visibility_quality_counts": _counts(after, "visibility_quality"),
        "source_counts": _counts(after, "source_type"),
        "coverage": coverage,
        "errors": errors,
        "warnings": [],
    }


def _hidden_confusion_audit(
    manifest: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    """Estimate Hidden false negatives from the random negative audit cohort."""
    joined = manifest.merge(
        decisions,
        on="hidden_review_item_id",
        how="left",
        suffixes=("_manifest", "_decision"),
    )
    before_col = "hidden_before_review_manifest"
    after_col = "hidden_after_review_decision"
    resolved = joined.get(
        "hidden_review_status_decision",
        pd.Series("", index=joined.index),
    ).eq("reviewed")
    random_no = (
        joined["hidden_review_cohort"].eq("hidden_no_random_audit")
        & joined[before_col].map(_normalize_hidden).eq("No")
        & resolved
    )
    after_normalized = pd.Series("", index=joined.index, dtype="object")
    after_normalized.loc[resolved] = joined.loc[resolved, after_col].map(_normalize_hidden)
    audited = int(random_no.sum())
    false_negative = int((random_no & after_normalized.eq("Yes")).sum())
    rate = false_negative / audited if audited else None
    low, high = _wilson_interval(false_negative, audited)
    sampling_weight = pd.to_numeric(
        joined.get(
            "hidden_sampling_weight",
            pd.Series(1.0, index=joined.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    weighted_denominator = float(sampling_weight.loc[random_no].sum())
    weighted_numerator = float(sampling_weight.loc[random_no & after_normalized.eq("Yes")].sum())
    weighted_rate = weighted_numerator / weighted_denominator if weighted_denominator > 0 else None
    high_risk = (
        joined["hidden_review_cohort"].eq("hidden_no_high_risk")
        & joined[before_col].map(_normalize_hidden).eq("No")
        & resolved
    )
    high_risk_reviewed = int(high_risk.sum())
    high_risk_corrected = int((high_risk & after_normalized.eq("Yes")).sum())
    transitions = (
        joined.loc[resolved]
        .assign(
            hidden_transition=lambda x: (
                x[before_col].map(_normalize_hidden) + "->" + x[after_col].map(_normalize_hidden)
            )
        )["hidden_transition"]
        .value_counts(dropna=False)
        .sort_index()
    )
    return {
        "reviewed_transition_counts": {str(key): int(value) for key, value in transitions.items()},
        "random_hidden_no_audited": audited,
        "random_hidden_no_false_negatives": false_negative,
        "estimated_false_negative_rate": rate,
        "poststratified_false_negative_rate": weighted_rate,
        "estimated_false_negative_rate_wilson95": [low, high],
        "high_risk_hidden_no_reviewed": high_risk_reviewed,
        "high_risk_hidden_no_corrected_to_yes": high_risk_corrected,
        "high_risk_correction_yield": (
            high_risk_corrected / high_risk_reviewed if high_risk_reviewed else None
        ),
        "interpretation": (
            "The post-stratified estimate uses only the random Hidden=No "
            "audit and its inverse inclusion weights. High-risk correction "
            "yield is enrichment evidence, not population prevalence."
        ),
    }


def _normalize_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Normalize GUI decision aliases into one strict decision contract."""
    out = decisions.copy()
    out["hidden_review_item_id"] = out["hidden_review_item_id"].astype(str)
    status = out["hidden_review_status"].fillna("").astype(str).str.strip().str.lower()
    status = status.replace(
        {
            "complete": "reviewed",
            "resolved": "reviewed",
            "skip": "pending",
            "": "pending",
        }
    )
    out["hidden_review_status"] = status
    out["hidden_before_review"] = out["hidden_before_review"].map(_normalize_hidden)
    after = out["hidden_after_review"].fillna("").astype(str).str.strip()
    resolved = status.eq("reviewed")
    out.loc[resolved, "hidden_after_review"] = after.loc[resolved].map(_normalize_hidden)
    out.loc[~resolved, "hidden_after_review"] = ""
    for column in DECISION_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out


def _finalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add blank decision fields and deterministic review order."""
    if manifest.empty:
        return manifest
    manifest = manifest.copy()
    manifest["hidden_review_status"] = "pending"
    manifest["hidden_after_review"] = ""
    manifest["hidden_review_confidence"] = ""
    manifest["hidden_review_reason"] = ""
    manifest["hidden_reviewer"] = ""
    manifest["hidden_reviewed_at"] = ""
    manifest["review_required"] = True
    sort_columns = [
        "hidden_review_priority",
        "source_type",
        "video_key",
        "behavior",
        "frame_index",
        "pig_id",
        "hidden_review_item_id",
    ]
    return manifest.sort_values(
        sort_columns,
        ascending=[False, True, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _assign_cohort(
    rows: pd.DataFrame,
    cohort: str,
    base_priority: int,
) -> pd.DataFrame:
    out = rows.copy()
    out["hidden_review_cohort"] = cohort
    out["hidden_review_priority"] = base_priority + (
        out["hidden_false_negative_risk_score"] * 10
    ).round().astype(int)
    return out


def _sample_each_stratum(
    rows: pd.DataFrame,
    columns: Iterable[str],
    count: int,
    seed: int,
    salt: str,
) -> pd.DataFrame:
    if count <= 0 or rows.empty:
        return rows.iloc[0:0].copy()
    available = [column for column in columns if column in rows.columns]
    if not available:
        return _stable_sample(rows, count, seed, salt)
    parts = []
    for key, group in rows.groupby(available, dropna=False, sort=True):
        parts.append(_stable_sample(group, count, seed, f"{salt}|{key}"))
    return pd.concat(parts, ignore_index=False) if parts else rows.iloc[0:0].copy()


def _add_sampling_metadata(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    columns: Iterable[str],
    *,
    design: str,
) -> pd.DataFrame:
    """Record stratum size and selection weight for reproducible review audit."""
    out = selected.copy()
    if out.empty:
        out["hidden_sampling_stratum"] = pd.Series(dtype="object")
        out["hidden_stratum_population"] = pd.Series(dtype="int64")
        out["hidden_stratum_selected"] = pd.Series(dtype="int64")
        out["hidden_sampling_probability"] = pd.Series(dtype="float64")
        out["hidden_sampling_weight"] = pd.Series(dtype="float64")
        out["hidden_sampling_design"] = pd.Series(dtype="object")
        return out

    population_strata = _sampling_strata(population, columns)
    selected_strata = _sampling_strata(out, columns)
    population_counts = population_strata.value_counts(dropna=False)
    selected_counts = selected_strata.value_counts(dropna=False)
    out["hidden_sampling_stratum"] = selected_strata
    out["hidden_stratum_population"] = selected_strata.map(population_counts).astype(int)
    out["hidden_stratum_selected"] = selected_strata.map(selected_counts).astype(int)
    out["hidden_sampling_probability"] = (
        out["hidden_stratum_selected"] / out["hidden_stratum_population"]
    )
    out["hidden_sampling_weight"] = (
        out["hidden_stratum_population"] / out["hidden_stratum_selected"]
    )
    out["hidden_sampling_design"] = design
    return out


def _sampling_strata(
    rows: pd.DataFrame,
    columns: Iterable[str],
) -> pd.Series:
    available = [column for column in columns if column in rows.columns]
    if not available:
        return pd.Series("all", index=rows.index, dtype="object")
    values = rows[available].fillna("<NA>").astype(str)
    return values.agg("|".join, axis=1)


def _cap_each_stratum(
    rows: pd.DataFrame,
    columns: Iterable[str],
    count: int | None,
    seed: int,
    salt: str,
) -> pd.DataFrame:
    if count is None:
        return rows.copy()
    return _sample_each_stratum(rows, columns, count, seed, salt)


def _stable_sample(
    rows: pd.DataFrame,
    count: int,
    seed: int,
    salt: str,
) -> pd.DataFrame:
    if len(rows) <= count:
        return rows.copy()
    ranked = rows.copy()
    ranked["_sample_rank"] = ranked["hidden_review_item_id"].map(
        lambda item_id: hashlib.sha256(f"{seed}|{salt}|{item_id}".encode()).hexdigest()
    )
    ranked = ranked.sort_values(
        ["_sample_rank", "hidden_review_item_id"],
        kind="mergesort",
    )
    return ranked.head(count).drop(columns=["_sample_rank"])


def _initial_hidden_source(df: pd.DataFrame) -> pd.Series:
    source = df["source_type"].astype(str)
    values = pd.Series("unknown_unreviewed", index=df.index, dtype="object")
    values.loc[source.eq("legacy_recovered")] = "legacy_prior_review"
    values.loc[source.eq("cvat_tracking_xml")] = "cvat_tracking_derived"
    reviewed = _reviewed_status_mask(df)
    unresolved = _unresolved_status_mask(df)
    values.loc[reviewed] = "current_human_review"
    values.loc[unresolved] = "current_human_review_unclear"
    return values


def _initial_hidden_trust(df: pd.DataFrame) -> pd.Series:
    legacy = df["source_type"].astype(str).eq("legacy_recovered")
    unresolved = _unresolved_status_mask(df)
    return ((legacy & ~unresolved) | _reviewed_status_mask(df)).astype(bool)


def _initial_review_status(df: pd.DataFrame) -> pd.Series:
    status = pd.Series(
        "tracking_derived_unreviewed",
        index=df.index,
        dtype="object",
    )
    legacy = df["source_type"].astype(str).eq("legacy_recovered")
    status.loc[legacy] = "prior_review_trusted"
    status.loc[_reviewed_status_mask(df)] = "reviewed"
    status.loc[_unresolved_status_mask(df)] = "unclear"
    return status


def _initial_trust_status(df: pd.DataFrame) -> pd.Series:
    status = pd.Series(
        "untrusted_tracking_derived",
        index=df.index,
        dtype="object",
    )
    legacy = df["source_type"].astype(str).eq("legacy_recovered")
    status.loc[legacy] = "trusted_prior_review"
    status.loc[_reviewed_status_mask(df)] = "trusted_current_review"
    status.loc[_unresolved_status_mask(df)] = "unclear_current_review"
    return status


def _reviewed_status_mask(df: pd.DataFrame) -> pd.Series:
    if "hidden_review_status" not in df.columns:
        return pd.Series(False, index=df.index)
    status = df["hidden_review_status"].fillna("").astype(str).str.lower()
    return status.isin(["reviewed", "resolved", "complete"])


def _unresolved_status_mask(df: pd.DataFrame) -> pd.Series:
    if "hidden_review_status" not in df.columns:
        return pd.Series(False, index=df.index)
    status = df["hidden_review_status"].fillna("").astype(str).str.lower()
    return status.isin(["unclear", "ambiguous"])


def _review_track_key(df: pd.DataFrame) -> pd.Series:
    if "object_track_key" in df.columns:
        key = df["object_track_key"].fillna("").astype(str)
        if key.str.strip().ne("").any():
            return key
    track = df.get("track_id", pd.Series("", index=df.index)).fillna("")
    return (
        df["source_type"].astype(str)
        + "|"
        + df["video_key"].astype(str)
        + "|"
        + track.astype(str)
        + "|"
        + df["pig_id"].astype(str)
    )


def _normalize_hidden(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Hidden value must not be missing")
    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "y"}:
        return "Yes"
    if text in {"no", "false", "0", "n"}:
        return "No"
    raise ValueError(f"Unsupported Hidden value: {value!r}")


def _to_bool(values: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "t"}
    return (
        values.fillna(False)
        .map(
            lambda value: value if isinstance(value, bool) else str(value).strip().lower() in truthy
        )
        .astype(bool)
    )


def _bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return _to_bool(df[column])


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _stable_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/").lower()


def _require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    name: str,
) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _require_unique(df: pd.DataFrame, column: str, name: str) -> None:
    duplicate = df[column].astype(str).duplicated(keep=False)
    if duplicate.any():
        sample = df.loc[duplicate, column].astype(str).head(5).tolist()
        raise ValueError(f"{name} has duplicate {column}: {sample}")


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + (z * z / total)
    center = proportion + (z * z / (2 * total))
    radius = z * math.sqrt((proportion * (1 - proportion) / total) + (z * z / (4 * total * total)))
    return (center - radius) / denominator, (center + radius) / denominator


def _config_payload(config: HiddenReviewConfig) -> dict[str, Any]:
    return {
        "random_seed": config.random_seed,
        "random_no_per_stratum": config.random_no_per_stratum,
        "clean_control_per_stratum": config.clean_control_per_stratum,
        "max_high_risk_per_stratum": config.max_high_risk_per_stratum,
        "high_risk_threshold": config.high_risk_threshold,
        "clean_control_max_risk": config.clean_control_max_risk,
        "pair_iou_threshold": config.pair_iou_threshold,
        "pair_overlap_threshold": config.pair_overlap_threshold,
        "nearest_distance_threshold": config.nearest_distance_threshold,
        "shape_change_threshold": config.shape_change_threshold,
        "area_delta_threshold": config.area_delta_threshold,
        "stratum_columns": list(config.stratum_columns),
    }

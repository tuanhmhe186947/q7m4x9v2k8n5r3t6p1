"""Fail-closed contracts for Classification V2 post-review learning.

This module deliberately separates three stages:

1. predeclare a random control audit from rows outside the primary scope;
2. freeze completed human-review artifacts before reading their outcomes;
3. analyse label changes and prepare, but never execute, final integration.

Review outcomes and audit fields are never eligible model inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONTROL_SCOPE_SCHEMA_VERSION = (
    "classification_v2.post_review_control_scope.v1"
)
REVIEW_CLOSE_SCHEMA_VERSION = "classification_v2.review_close_authority.v1"
LEARNING_SCHEMA_VERSION = "classification_v2.post_review_learning.v1"
INTEGRATION_SCHEMA_VERSION = (
    "classification_v2.final_review_integration_preflight.v1"
)
CORRECTED_SOURCE_SCHEMA_VERSION = (
    "classification_v2.corrected_source_authority.v1"
)

DEFAULT_CONTROL_SEED = 20260801
MINIMUM_CONTROL_ITEMS = 120
REQUIRED_WINDOW_LENGTHS = (6, 8, 12, 16)
ADJUSTED_ROI_SUFFIX = (
    "data/annotations/roi/ROI_annotations.toy_adjusted.coco.json"
)

RESOLVED_DECISIONS = frozenset({"accept", "corrected", "exclude"})
TECHNICAL_LABEL_STATUSES = frozenset({"TECHNICAL_DEFECT"})

OUTCOME_COLUMNS = frozenset(
    {
        "manual_review_decision",
        "manual_corrected_behavior",
        "manual_label_strength",
        "reviewed_behavior",
        "label_status",
        "source_label_error_confirmed",
        "error_pattern",
        "review_confidence",
        "selection_assessment",
    }
)

MODEL_X_FORBIDDEN_COLUMNS = frozenset(
    {
        *OUTCOME_COLUMNS,
        "post_review_control_selected",
        "post_review_control_stratum",
        "post_review_control_sampling_probability",
        "post_review_control_sampling_weight",
        "post_review_control_seed",
        "post_review_control_scope_version",
        "review_reason",
        "review_reason_codes",
        "review_selection_predicates",
        "behavior_label",
        "source_type",
        "video_key",
        "recording_date",
        "review_unit_id",
        "temporal_unit_key",
        "pig_id",
        "track_id",
        "crop_path",
        "candidate_rank",
        "risk_score",
        "risk_components",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PostReviewContractError(ValueError):
    """Raised when a post-review contract cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class ControlSelectionConfig:
    """Predeclared deterministic residual-control sampling configuration."""

    target_count: int = MINIMUM_CONTROL_ITEMS
    seed: int = DEFAULT_CONTROL_SEED
    stratum_columns: tuple[str, ...] = (
        "behavior_label",
        "source_type",
        "review_unit_type",
    )

    def validate(self) -> None:
        if self.target_count < MINIMUM_CONTROL_ITEMS:
            raise PostReviewContractError(
                f"control_target_below_minimum={self.target_count}"
            )
        if not self.stratum_columns:
            raise PostReviewContractError("control_stratum_columns_empty")
        if len(set(self.stratum_columns)) != len(self.stratum_columns):
            raise PostReviewContractError("duplicate_control_stratum_columns")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_not_active_behavior_ledger_path(path: Path) -> None:
    """Reject active Behavior decision-ledger paths before any file read."""

    normalized = str(path).replace("/", "\\").casefold()
    workspace = "human_review_workspace\\classification_v2\\"
    decisions = "\\human_decisions\\behavior"
    if workspace in normalized and decisions in normalized:
        raise PostReviewContractError(
            "active_behavior_decision_ledger_path_forbidden"
        )


def build_post_review_control_scope(
    population: pd.DataFrame,
    primary_scope: pd.DataFrame,
    *,
    config: ControlSelectionConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a deterministic, weighted control audit outside primary scope.

    The input population is explicit. This function does not assume whether it
    is the 6,061 candidate population or another frozen parent population.
    """

    cfg = config or ControlSelectionConfig()
    cfg.validate()
    _require_clean_sampling_input(population, "population")
    _require_clean_sampling_input(primary_scope, "primary_scope")
    _validate_key_frame(population, "population")
    _validate_key_frame(primary_scope, "primary_scope")

    population_keys = set(_normalized(population["review_unit_id"]))
    primary_keys = set(_normalized(primary_scope["review_unit_id"]))
    missing_primary = primary_keys - population_keys
    if missing_primary:
        raise PostReviewContractError(
            f"primary_scope_not_in_population={len(missing_primary)}"
        )
    if len(primary_keys) >= len(population_keys):
        raise PostReviewContractError("primary_scope_not_strict_subset")

    normalized_keys = _normalized(population["review_unit_id"])
    temporal_overlap_count = 0
    if "temporal_unit_key" in population and "temporal_unit_key" in primary_scope:
        population_temporal = _normalized(population["temporal_unit_key"])
        primary_temporal_values = _normalized(primary_scope["temporal_unit_key"])
        primary_temporal = set(primary_temporal_values)
        if "" in primary_temporal or population_temporal.eq("").any():
            raise PostReviewContractError("blank_temporal_unit_key")
        if population_temporal.duplicated().any():
            raise PostReviewContractError(
                "population_duplicate_temporal_unit_key"
            )
        if primary_temporal_values.duplicated().any():
            raise PostReviewContractError(
                "primary_scope_duplicate_temporal_unit_key"
            )
        pool_temporal = set(
            population_temporal.loc[~normalized_keys.isin(primary_keys)]
        )
        temporal_overlap_count = len(primary_temporal & pool_temporal)
        if temporal_overlap_count:
            raise PostReviewContractError(
                f"primary_control_temporal_overlap={temporal_overlap_count}"
            )

    pool = population.loc[~normalized_keys.isin(primary_keys)].copy()
    if len(pool) < cfg.target_count:
        raise PostReviewContractError(
            f"control_pool_too_small={len(pool)}<{cfg.target_count}"
        )

    pool = _with_sampling_strata(pool, cfg.stratum_columns)
    stratum_counts = pool.groupby(
        "_control_stratum", sort=True, dropna=False
    ).size()
    quotas = _allocate_stratum_quotas(stratum_counts, cfg.target_count)
    pool["_control_priority"] = pool["review_unit_id"].map(
        lambda key: _deterministic_priority(cfg.seed, str(key))
    )

    selected_parts: list[pd.DataFrame] = []
    for stratum, quota in quotas.items():
        group = pool.loc[pool["_control_stratum"].eq(stratum)].copy()
        group = group.sort_values(
            ["_control_priority", "review_unit_id"], kind="mergesort"
        )
        chosen = group.head(int(quota)).copy()
        chosen["post_review_control_sampling_probability"] = (
            float(quota) / float(len(group))
        )
        selected_parts.append(chosen)

    selected = pd.concat(selected_parts, ignore_index=False)
    selected = selected.sort_values(
        ["_control_priority", "review_unit_id"], kind="mergesort"
    ).copy()
    if len(selected) != cfg.target_count:
        raise PostReviewContractError(
            f"control_selection_count_mismatch={len(selected)}"
        )

    original_columns = list(population.columns)
    selected_source_rows = selected[original_columns].reset_index(drop=True)
    selected["post_review_control_selected"] = True
    selected["post_review_control_stratum"] = selected["_control_stratum"]
    selected["post_review_control_sampling_weight"] = 1.0 / selected[
        "post_review_control_sampling_probability"
    ]
    selected["post_review_control_seed"] = cfg.seed
    selected["post_review_control_scope_version"] = (
        CONTROL_SCOPE_SCHEMA_VERSION
    )
    selected = selected.drop(
        columns=["_control_stratum", "_control_priority"]
    ).reset_index(drop=True)
    if not selected[original_columns].equals(selected_source_rows):
        raise PostReviewContractError("selected_source_rows_changed")

    selected_keys = set(_normalized(selected["review_unit_id"]))
    overlap = selected_keys & primary_keys
    if overlap:
        raise PostReviewContractError(
            f"primary_control_overlap={len(overlap)}"
        )

    audit = {
        "schema_version": CONTROL_SCOPE_SCHEMA_VERSION,
        "status": "PREDECLARED_CONTROL_SCOPE",
        "population_rows": int(len(population)),
        "primary_scope_rows": int(len(primary_scope)),
        "residual_pool_rows": int(len(pool)),
        "selected_control_rows": int(len(selected)),
        "target_count": cfg.target_count,
        "seed": cfg.seed,
        "stratum_columns": list(cfg.stratum_columns),
        "stratum_pool_counts": {
            str(key): int(value) for key, value in stratum_counts.items()
        },
        "stratum_selected_counts": {
            str(key): int(value) for key, value in quotas.items()
        },
        "primary_control_overlap": 0,
        "primary_control_temporal_overlap": temporal_overlap_count,
        "population_key_hash": _key_hash(population_keys),
        "primary_scope_key_hash": _key_hash(primary_keys),
        "residual_pool_key_hash": _key_hash(
            set(_normalized(pool["review_unit_id"]))
        ),
        "selected_control_key_hash": _key_hash(selected_keys),
        "sampling_outcomes_used": False,
        "estimand": (
            "source_label_error_rate_in_explicit_residual_control_pool"
        ),
        "interpretation_limit": (
            "Controls estimate only the declared residual population and do "
            "not certify all excluded or auto-carried samples as correct."
        ),
    }
    return selected, audit


def build_review_close_authority(
    *,
    primary_scope: pd.DataFrame,
    primary_decisions: pd.DataFrame,
    primary_quality: pd.DataFrame,
    control_scope: pd.DataFrame,
    control_decisions: pd.DataFrame,
    control_quality: pd.DataFrame,
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    expected_primary_count: int = 2729,
    minimum_control_count: int = MINIMUM_CONTROL_ITEMS,
) -> dict[str, Any]:
    """Validate complete frozen copies and construct their close authority."""

    if len(primary_scope) != expected_primary_count:
        raise PostReviewContractError(
            "primary_scope_count_mismatch="
            f"{len(primary_scope)}!={expected_primary_count}"
        )
    if len(control_scope) < minimum_control_count:
        raise PostReviewContractError(
            "control_scope_below_minimum="
            f"{len(control_scope)}<{minimum_control_count}"
        )
    primary_stats = _complete_review_stats(
        primary_scope,
        primary_decisions,
        primary_quality,
        cohort="primary",
    )
    control_stats = _complete_review_stats(
        control_scope,
        control_decisions,
        control_quality,
        cohort="control",
    )
    required = {
        "primary_scope",
        "primary_decisions",
        "primary_quality",
        "control_scope",
        "control_decisions",
        "control_quality",
    }
    normalized = _validate_artifact_bindings(artifact_bindings, required)
    return {
        "schema_version": REVIEW_CLOSE_SCHEMA_VERSION,
        "status": "FROZEN",
        "human_review_complete": True,
        "primary_review": primary_stats,
        "control_review": control_stats,
        "expected_primary_count": expected_primary_count,
        "minimum_control_count": minimum_control_count,
        "artifacts": normalized,
        "active_ledger_touched": "NO",
        "model_input_fields_added": [],
        "scientific_scope": (
            "post-review selector and feature-semantics audit only"
        ),
    }


def validate_review_close_authority(
    authority: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Fail unless a completed, immutable post-review authority is bound."""

    if authority.get("schema_version") != REVIEW_CLOSE_SCHEMA_VERSION:
        raise PostReviewContractError("invalid_review_close_schema")
    if authority.get("status") != "FROZEN":
        raise PostReviewContractError("review_close_authority_not_frozen")
    if authority.get("human_review_complete") is not True:
        raise PostReviewContractError("human_review_not_complete")
    if authority.get("active_ledger_touched") != "NO":
        raise PostReviewContractError("active_ledger_noninterference_missing")
    expected_primary = int(authority.get("expected_primary_count", -1))
    minimum_control = int(authority.get("minimum_control_count", -1))
    if expected_primary <= 0:
        raise PostReviewContractError("review_close_expected_primary_invalid")
    if minimum_control < MINIMUM_CONTROL_ITEMS:
        raise PostReviewContractError("review_close_control_minimum_invalid")
    primary_review = authority.get("primary_review", {})
    control_review = authority.get("control_review", {})
    if int(primary_review.get("scope_rows", -1)) != expected_primary:
        raise PostReviewContractError("review_close_primary_count_invalid")
    if int(control_review.get("scope_rows", -1)) < minimum_control:
        raise PostReviewContractError("review_close_control_count_invalid")
    artifacts = authority.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PostReviewContractError("review_close_artifacts_missing")
    required = {
        "primary_scope",
        "primary_decisions",
        "primary_quality",
        "control_scope",
        "control_decisions",
        "control_quality",
    }
    _validate_artifact_bindings(artifacts, required)
    if expected_bindings is not None:
        expected = _validate_artifact_bindings(
            expected_bindings,
            set(expected_bindings),
        )
        for name, binding in expected.items():
            if artifacts.get(name) != binding:
                raise PostReviewContractError(
                    f"review_close_artifact_drift={name}"
                )


def analyze_post_review_learning(
    *,
    review_close_authority: Mapping[str, Any],
    primary_scope: pd.DataFrame,
    primary_quality: pd.DataFrame,
    control_scope: pd.DataFrame,
    control_quality: pd.DataFrame,
    frame_features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    """Analyse reviewed corrections without turning them into model inputs."""

    validate_review_close_authority(review_close_authority)
    primary_expected = int(
        review_close_authority["primary_review"]["scope_rows"]
    )
    control_expected = int(
        review_close_authority["control_review"]["scope_rows"]
    )
    if len(primary_scope) != primary_expected:
        raise PostReviewContractError("analysis_primary_scope_count_drift")
    if len(control_scope) != control_expected:
        raise PostReviewContractError("analysis_control_scope_count_drift")
    _validate_feature_whitelist(feature_columns)
    outcomes = pd.concat(
        [
            _review_outcomes(primary_scope, primary_quality, "SELECTED"),
            _review_outcomes(control_scope, control_quality, "CONTROL"),
        ],
        ignore_index=True,
    )
    if outcomes["review_unit_id"].duplicated().any():
        raise PostReviewContractError("primary_control_outcome_overlap")

    analyzable = outcomes.loc[~outcomes["technical_exclusion"]].copy()
    transition = (
        analyzable.groupby(
            ["original_behavior", "reviewed_behavior"], dropna=False
        )
        .size()
        .rename("count")
        .reset_index()
    )
    transition["changed"] = transition["original_behavior"].ne(
        transition["reviewed_behavior"]
    )

    selector = _selector_diagnostics(analyzable)
    strata_columns = [
        "selection_group",
        "source_type",
        "original_behavior",
        "label_status",
        "error_pattern",
    ]
    if "review_unit_type" in analyzable:
        strata_columns.append("review_unit_type")
    strata = (
        analyzable.groupby(
            strata_columns,
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    feature_contrasts = _feature_contrasts(
        analyzable,
        frame_features,
        feature_columns,
    )
    summary = {
        "schema_version": LEARNING_SCHEMA_VERSION,
        "status": "POST_REVIEW_DIAGNOSTIC_ONLY",
        "primary_items": int(
            outcomes["selection_group"].eq("SELECTED").sum()
        ),
        "control_items": int(
            outcomes["selection_group"].eq("CONTROL").sum()
        ),
        "changed_labels": int(analyzable["label_changed"].sum()),
        "unchanged_labels": int((~analyzable["label_changed"]).sum()),
        "technical_exclusions": int(outcomes["technical_exclusion"].sum()),
        "selector_diagnostics": selector,
        "feature_columns": list(feature_columns),
        "review_fields_entering_model_x": 0,
        "automatic_selector_change_authorized": False,
        "automatic_feature_change_authorized": False,
        "interpretation_limits": [
            "Changed and unchanged labels are audit outcomes, not features.",
            "Observed contrasts generate hypotheses requiring validation.",
            "Control estimates apply only to the frozen residual pool.",
            "Human corrections may include prior annotator mistakes.",
        ],
    }
    return {
        "summary": summary,
        "outcomes": outcomes,
        "transition_matrix": transition,
        "stratified_outcomes": strata,
        "feature_contrasts": feature_contrasts,
    }


def build_corrected_source_authority(
    *,
    identity_apply_manifests: Sequence[Mapping[str, Any]],
    observed_target_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Freeze a valid sequential mini-CVAT source-correction chain."""

    if not identity_apply_manifests:
        raise PostReviewContractError("identity_apply_manifests_empty")
    observed = {
        _path_key(path): str(digest).casefold()
        for path, digest in observed_target_hashes.items()
    }
    latest_hashes: dict[str, str] = {}
    canonical_paths: dict[str, str] = {}
    chains: dict[str, list[dict[str, str]]] = {}
    manifest_bindings: list[dict[str, str]] = []
    for manifest in identity_apply_manifests:
        manifest_hash = str(manifest.get("manifest_sha256", "")).casefold()
        if manifest.get("status") != "APPLIED":
            raise PostReviewContractError("identity_manifest_not_applied")
        if not _valid_sha256(manifest_hash):
            raise PostReviewContractError("identity_manifest_hash_missing")
        manifest_binding = {"sha256": manifest_hash}
        manifest_path = str(manifest.get("manifest_path", "")).strip()
        if manifest_path:
            manifest_binding["path"] = manifest_path
        manifest_bindings.append(manifest_binding)
        targets = manifest.get("targets", [])
        if not isinstance(targets, list) or not targets:
            raise PostReviewContractError("identity_manifest_targets_missing")
        for target in targets:
            path = str(target.get("path", "")).strip()
            before_hash = str(target.get("before_sha256", "")).casefold()
            after_hash = str(target.get("after_sha256", "")).casefold()
            if not path:
                raise PostReviewContractError("identity_target_path_missing")
            if not _valid_sha256(before_hash) or not _valid_sha256(after_hash):
                raise PostReviewContractError(
                    f"identity_target_hash_invalid={path}"
                )
            key = _path_key(path)
            if key in latest_hashes and latest_hashes[key] != before_hash:
                raise PostReviewContractError(
                    f"identity_target_chain_break={path}"
                )
            latest_hashes[key] = after_hash
            canonical_paths[key] = path
            chains.setdefault(key, []).append(
                {
                    "manifest_sha256": manifest_hash,
                    "before_sha256": before_hash,
                    "after_sha256": after_hash,
                }
            )

    missing_observed = sorted(set(latest_hashes) - set(observed))
    if missing_observed:
        raise PostReviewContractError(
            f"corrected_source_targets_not_observed={len(missing_observed)}"
        )
    for key, expected_hash in latest_hashes.items():
        if observed[key] != expected_hash:
            raise PostReviewContractError(
                f"corrected_source_final_hash_mismatch={canonical_paths[key]}"
            )
    return {
        "schema_version": CORRECTED_SOURCE_SCHEMA_VERSION,
        "status": "FROZEN",
        "identity_apply_manifests": manifest_bindings,
        "target_after_hashes": {
            canonical_paths[key]: latest_hashes[key]
            for key in sorted(latest_hashes)
        },
        "target_chains": {
            canonical_paths[key]: chains[key] for key in sorted(chains)
        },
        "source_annotations_changed": "YES",
        "behavior_decision_ledger_touched": "NO",
    }


def build_final_review_integration_preflight(
    *,
    review_close_authority: Mapping[str, Any],
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    identity_apply_manifests: Sequence[Mapping[str, Any]],
    corrected_source_authority: Mapping[str, Any] | None = None,
    conflict_resolutions: Sequence[Mapping[str, Any]] = (),
    window_lengths: Sequence[int] = REQUIRED_WINDOW_LENGTHS,
) -> dict[str, Any]:
    """Build a non-executing preflight for corrected-data reconstruction."""

    validate_review_close_authority(review_close_authority)
    required = {
        "primary_scope",
        "primary_decisions",
        "primary_quality",
        "control_scope",
        "control_decisions",
        "control_quality",
        "adjusted_roi",
        "corrected_source_authority",
        "rebuilt_frame_features",
    }
    bindings = _validate_artifact_bindings(artifact_bindings, required)
    close_bindings = review_close_authority["artifacts"]
    blockers: list[str] = []
    for name in required & set(close_bindings):
        if close_bindings[name] != bindings[name]:
            blockers.append(f"review_close_binding_drift={name}")

    roi_path = bindings["adjusted_roi"]["path"].replace("\\", "/")
    if not roi_path.endswith(ADJUSTED_ROI_SUFFIX):
        blockers.append("adjusted_roi_authority_mismatch")
    if tuple(window_lengths) != REQUIRED_WINDOW_LENGTHS:
        blockers.append("window_rebuild_must_be_exact_t6_t8_t12_t16")

    corrected_hashes: dict[str, Any] = {}
    if corrected_source_authority is None:
        blockers.append("corrected_source_authority_missing")
    else:
        if corrected_source_authority.get("schema_version") != (
            CORRECTED_SOURCE_SCHEMA_VERSION
        ):
            blockers.append("corrected_source_authority_schema_invalid")
        if corrected_source_authority.get("status") != "FROZEN":
            blockers.append("corrected_source_authority_not_frozen")
        candidate = corrected_source_authority.get("target_after_hashes", {})
        if isinstance(candidate, Mapping):
            corrected_hashes = {
                _path_key(path): str(digest).casefold()
                for path, digest in candidate.items()
            }
        else:
            blockers.append("corrected_source_target_hashes_invalid")
    if identity_apply_manifests:
        if not corrected_hashes:
            blockers.append("corrected_source_target_hashes_missing")
        else:
            try:
                build_corrected_source_authority(
                    identity_apply_manifests=identity_apply_manifests,
                    observed_target_hashes=corrected_hashes,
                )
            except PostReviewContractError as exc:
                blockers.append(f"corrected_source_chain_invalid={exc}")

    identity_rows: list[dict[str, Any]] = []
    declared_resolutions = {
        (str(row.get("manifest_sha256", "")), str(row.get("field", ""))): str(
            row.get("resolution", "")
        )
        for row in conflict_resolutions
    }
    for manifest in identity_apply_manifests:
        manifest_hash = str(manifest.get("manifest_sha256", ""))
        if manifest.get("status") != "APPLIED":
            blockers.append("identity_manifest_not_applied")
        if not _valid_sha256(manifest_hash):
            blockers.append("identity_manifest_hash_missing")
        for target in manifest.get("targets", []):
            behavior_updates = int(target.get("behavior_updates", 0))
            hidden_updates = int(target.get("hidden_updates", 0))
            row = {
                "manifest_sha256": manifest_hash,
                "target_path": str(target.get("path", "")),
                "target_after_sha256": str(target.get("after_sha256", "")),
                "bbox_updates": int(target.get("bbox_updates", 0)),
                "identity_updates": int(target.get("identity_updates", 0)),
                "behavior_updates": behavior_updates,
                "hidden_updates": hidden_updates,
            }
            identity_rows.append(row)
            target_path = row["target_path"]
            target_hash = row["target_after_sha256"]
            if not _valid_sha256(target_hash):
                blockers.append(
                    f"identity_target_after_hash_invalid={target_path}"
                )
            for field, count in (
                ("behavior", behavior_updates),
                ("hidden", hidden_updates),
            ):
                if count <= 0:
                    continue
                resolution = declared_resolutions.get((manifest_hash, field))
                if resolution not in {
                    "BEHAVIOR_REVIEW_WINS",
                    "IDENTITY_ADJUDICATION_WINS",
                    "EXPLICIT_MANUAL_RECONCILIATION",
                }:
                    blockers.append(
                        f"identity_{field}_conflict_unresolved={manifest_hash}"
                    )

    status = (
        "READY_FOR_REVIEWED_WINDOW_REBUILD" if not blockers else "BLOCKED"
    )
    return {
        "schema_version": INTEGRATION_SCHEMA_VERSION,
        "status": status,
        "blockers": sorted(set(blockers)),
        "artifacts": bindings,
        "identity_source_apply_targets": identity_rows,
        "conflict_resolutions": list(conflict_resolutions),
        "window_lengths": list(REQUIRED_WINDOW_LENGTHS),
        "window_structure_reuse_allowed": False,
        "full_frame_feature_recompute_required": True,
        "roi_authority": ADJUSTED_ROI_SUFFIX,
        "behavior_authority_precedence": (
            "FROZEN_BEHAVIOR_REVIEW_AFTER_EXPLICIT_CONFLICT_RESOLUTION"
        ),
        "hidden_authority_precedence": (
            "EXPLICIT_RECONCILIATION_REQUIRED_FOR_IDENTITY_EDITS"
        ),
        "automatic_apply_performed": False,
        "training_snapshot_authorized": False,
        "next_steps": [
            "apply frozen Behavior authority to rebuilt frame features",
            "full-recompute unified T6/T8/T12/T16 windows",
            "run leakage, mask, label, and lineage audits",
            "freeze reviewed-Q2 snapshot only after all gates pass",
        ],
    }


def _require_clean_sampling_input(frame: pd.DataFrame, label: str) -> None:
    contaminated = sorted(set(frame.columns) & OUTCOME_COLUMNS)
    if contaminated:
        raise PostReviewContractError(
            f"{label}_contains_review_outcomes={','.join(contaminated)}"
        )


def _validate_key_frame(frame: pd.DataFrame, label: str) -> None:
    _require_columns(frame, ["review_unit_id"], label)
    keys = _normalized(frame["review_unit_id"])
    if keys.eq("").any():
        raise PostReviewContractError(f"{label}_blank_review_unit_id")
    if keys.duplicated().any():
        raise PostReviewContractError(f"{label}_duplicate_review_unit_id")


def _with_sampling_strata(
    frame: pd.DataFrame,
    requested_columns: Sequence[str],
) -> pd.DataFrame:
    out = frame.copy()
    sampling_values: dict[str, pd.Series] = {}
    missing: list[str] = []
    for column in requested_columns:
        if column in out:
            sampling_values[column] = out[column]
        elif column == "recording_date" and "video_key" in out:
            sampling_values[column] = out["video_key"].map(_recording_date)
        else:
            missing.append(column)
    if missing:
        raise PostReviewContractError(
            f"control_stratum_columns_missing={','.join(missing)}"
        )
    values = pd.DataFrame(sampling_values, index=out.index)
    for column in values:
        values[column] = _normalized(values[column]).replace("", "<MISSING>")
    out["_control_stratum"] = values.agg("|".join, axis=1)
    return out


def _allocate_stratum_quotas(
    counts: pd.Series,
    target_count: int,
) -> dict[str, int]:
    if counts.empty:
        raise PostReviewContractError("control_pool_empty")
    counts = counts.sort_index().astype(int)
    total = int(counts.sum())
    ideal = counts.astype(float) * float(target_count) / float(total)
    quotas = np.floor(ideal).astype(int)

    if target_count >= len(counts):
        quotas = quotas.clip(lower=1)
    quotas = pd.Series(
        np.minimum(quotas.to_numpy(), counts.to_numpy()),
        index=counts.index,
        dtype=int,
    )
    while int(quotas.sum()) < target_count:
        eligible = counts.index[quotas < counts]
        if len(eligible) == 0:
            raise PostReviewContractError("control_quota_allocation_exhausted")
        score = ideal.loc[eligible] - quotas.loc[eligible]
        chosen = sorted(
            eligible,
            key=lambda key: (-float(score.loc[key]), str(key)),
        )[0]
        quotas.loc[chosen] += 1
    while int(quotas.sum()) > target_count:
        minimum = 1 if target_count >= len(counts) else 0
        eligible = counts.index[quotas > minimum]
        if len(eligible) == 0:
            raise PostReviewContractError("control_quota_reduction_exhausted")
        score = ideal.loc[eligible] - quotas.loc[eligible]
        chosen = sorted(
            eligible,
            key=lambda key: (float(score.loc[key]), str(key)),
        )[0]
        quotas.loc[chosen] -= 1
    return {str(key): int(value) for key, value in quotas.items()}


def _deterministic_priority(seed: int, key: str) -> str:
    payload = f"{seed}\0{key}".encode()
    return hashlib.sha256(payload).hexdigest()


def _recording_date(value: Any) -> str:
    match = re.search(r"(\d{8})", _text(value))
    return match.group(1) if match else "UNKNOWN"


def _complete_review_stats(
    scope: pd.DataFrame,
    decisions: pd.DataFrame,
    quality: pd.DataFrame,
    *,
    cohort: str,
) -> dict[str, Any]:
    _validate_key_frame(scope, f"{cohort}_scope")
    _validate_key_frame(decisions, f"{cohort}_decisions")
    _validate_key_frame(quality, f"{cohort}_quality")
    _require_columns(decisions, ["manual_review_decision"], "decisions")
    _require_columns(
        quality,
        [
            "label_status",
            "original_behavior",
            "reviewed_behavior",
            "source_label_error_confirmed",
        ],
        "quality",
    )
    scope_keys = set(_normalized(scope["review_unit_id"]))
    decision_keys = set(_normalized(decisions["review_unit_id"]))
    quality_keys = set(_normalized(quality["review_unit_id"]))
    if decision_keys != scope_keys:
        raise PostReviewContractError(
            f"{cohort}_decision_coverage_mismatch"
        )
    if quality_keys != scope_keys:
        raise PostReviewContractError(f"{cohort}_quality_coverage_mismatch")
    decisions_normalized = _normalized(decisions["manual_review_decision"])
    invalid = sorted(set(decisions_normalized.str.casefold()) - RESOLVED_DECISIONS)
    if invalid:
        raise PostReviewContractError(
            f"{cohort}_unresolved_or_invalid_decisions={invalid}"
        )
    outcomes = _review_outcomes(scope, quality, cohort.upper())
    decision_outcomes = decisions[[
        "review_unit_id",
        "manual_review_decision",
    ]].merge(
        outcomes[["review_unit_id", "label_changed", "technical_exclusion"]],
        on="review_unit_id",
        how="inner",
        validate="one_to_one",
    )
    normalized_decisions = _normalized(
        decision_outcomes["manual_review_decision"]
    ).str.casefold()
    consistent = (
        normalized_decisions.eq("accept")
        & ~decision_outcomes["label_changed"]
        & ~decision_outcomes["technical_exclusion"]
    ) | (
        normalized_decisions.eq("corrected")
        & decision_outcomes["label_changed"]
        & ~decision_outcomes["technical_exclusion"]
    ) | (
        normalized_decisions.eq("exclude")
        & decision_outcomes["technical_exclusion"]
    )
    if not consistent.all():
        raise PostReviewContractError(
            f"{cohort}_decision_quality_semantics_mismatch="
            f"{int((~consistent).sum())}"
        )
    return {
        "scope_rows": int(len(scope)),
        "decision_rows": int(len(decisions)),
        "quality_rows": int(len(quality)),
        "complete": True,
        "decision_counts": {
            str(key): int(value)
            for key, value in decisions_normalized.str.casefold()
            .value_counts()
            .sort_index()
            .items()
        },
    }


def _review_outcomes(
    scope: pd.DataFrame,
    quality: pd.DataFrame,
    selection_group: str,
) -> pd.DataFrame:
    _validate_key_frame(scope, f"{selection_group}_scope")
    _validate_key_frame(quality, f"{selection_group}_quality")
    _require_columns(
        scope,
        ["temporal_unit_key", "source_type", "behavior_label"],
        f"{selection_group}_scope",
    )
    required_quality = [
        "review_unit_id",
        "original_behavior",
        "reviewed_behavior",
        "label_status",
        "source_label_error_confirmed",
        "error_pattern",
    ]
    _require_columns(quality, required_quality, "quality")
    source_columns = ["review_unit_id"]
    for column in (
        "temporal_unit_key",
        "source_type",
        "video_key",
        "review_unit_type",
        "post_review_control_sampling_weight",
    ):
        if column in scope:
            source_columns.append(column)
    if "behavior_label" in scope:
        source_columns.append("behavior_label")
    joined = scope[source_columns].merge(
        quality[required_quality],
        on="review_unit_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise PostReviewContractError(
            f"{selection_group}_quality_coverage_incomplete"
        )
    joined = joined.drop(columns="_merge")
    if "behavior_label" in joined:
        source_behavior = _normalized(joined["behavior_label"])
        quality_behavior = _normalized(joined["original_behavior"])
        if not source_behavior.eq(quality_behavior).all():
            raise PostReviewContractError(
                f"{selection_group}_quality_original_label_mismatch"
            )
        joined = joined.drop(columns="behavior_label")
    joined["selection_group"] = selection_group
    joined["technical_exclusion"] = joined["label_status"].isin(
        TECHNICAL_LABEL_STATUSES
    )
    joined["label_changed"] = joined["original_behavior"].ne(
        joined["reviewed_behavior"]
    )
    invalid_changed = joined["technical_exclusion"] & joined["label_changed"]
    if invalid_changed.any():
        raise PostReviewContractError(
            "technical_exclusion_must_not_count_as_label_change"
        )
    expected_error = joined["label_changed"] & ~joined["technical_exclusion"]
    actual_error = joined["source_label_error_confirmed"].eq("YES")
    if not expected_error.eq(actual_error).all():
        raise PostReviewContractError(
            f"{selection_group}_label_change_error_flag_mismatch"
        )
    return joined


def _selector_diagnostics(outcomes: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for group_name, rows in outcomes.groupby("selection_group", sort=True):
        source_error = rows["source_label_error_confirmed"].eq("YES")
        weights = pd.Series(1.0, index=rows.index)
        if group_name == "CONTROL":
            if "post_review_control_sampling_weight" not in rows:
                raise PostReviewContractError("control_sampling_weight_missing")
            weights = pd.to_numeric(
                rows["post_review_control_sampling_weight"], errors="coerce"
            )
            if weights.isna().any() or (weights <= 0).any():
                raise PostReviewContractError("invalid_control_sampling_weight")
        weighted_rate = float(np.average(source_error.astype(float), weights=weights))
        effective_n = float(weights.sum() ** 2 / (weights.pow(2).sum()))
        weighted_errors = float((weights * source_error.astype(float)).sum())
        diagnostics[group_name] = {
            "items": int(len(rows)),
            "source_label_errors": int(source_error.sum()),
            "unweighted_rate": float(source_error.mean()),
            "weighted_rate": weighted_rate,
            "weighted_rate_wilson_95": _wilson_interval(
                weighted_rate,
                effective_n,
            ),
            "effective_sample_size": effective_n,
            "estimated_population_items": float(weights.sum()),
            "estimated_source_label_errors": weighted_errors,
            "is_full_candidate_population_prevalence": False,
            "is_declared_residual_pool_estimate": group_name == "CONTROL",
        }
    if "SELECTED" in diagnostics and "CONTROL" in diagnostics:
        diagnostics["selector_enrichment_ratio"] = _safe_ratio(
            diagnostics["SELECTED"]["weighted_rate"],
            diagnostics["CONTROL"]["weighted_rate"],
        )
        true_positives = diagnostics["SELECTED"][
            "estimated_source_label_errors"
        ]
        false_negatives = diagnostics["CONTROL"][
            "estimated_source_label_errors"
        ]
        diagnostics["estimated_selector_recall_within_explicit_population"] = (
            _safe_ratio(true_positives, true_positives + false_negatives)
        )
        diagnostics["estimated_false_negative_count_in_residual_pool"] = (
            false_negatives
        )
        diagnostics["selector_precision_on_primary_census"] = diagnostics[
            "SELECTED"
        ]["weighted_rate"]
    return diagnostics


def _feature_contrasts(
    outcomes: pd.DataFrame,
    frame_features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    _require_columns(
        frame_features,
        ["temporal_unit_key", *feature_columns],
        "frame_features",
    )
    if "temporal_unit_key" not in outcomes:
        raise PostReviewContractError("outcomes_temporal_unit_key_missing")
    if outcomes["temporal_unit_key"].duplicated().any():
        raise PostReviewContractError("duplicate_outcome_temporal_unit_key")
    features = frame_features[["temporal_unit_key", *feature_columns]].copy()
    for column in feature_columns:
        original = features[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & original.astype(str).str.strip().ne("")
        invalid &= numeric.isna()
        if invalid.any():
            raise PostReviewContractError(
                f"feature_non_numeric_values={column}:{int(invalid.sum())}"
            )
        features[column] = numeric
    aggregate = features.groupby("temporal_unit_key", sort=False).agg(
        {column: ["mean", "median", "count", "size"] for column in feature_columns}
    )

    rows: list[dict[str, Any]] = []
    strata: list[tuple[str, str, pd.DataFrame]] = [
        ("ALL", "ALL", outcomes)
    ]
    for column in ("original_behavior", "source_type", "review_unit_type"):
        if column not in outcomes:
            continue
        for value, group in outcomes.groupby(column, sort=True, dropna=False):
            strata.append((column, _text(value), group))
    for stratum_type, stratum_value, group in strata:
        keys = group[["temporal_unit_key", "label_changed"]].copy()
        for feature in feature_columns:
            rows.append(
                _one_feature_contrast(
                    keys,
                    aggregate,
                    feature,
                    stratum_type=stratum_type,
                    stratum_value=stratum_value,
                )
            )
    return pd.DataFrame(rows)


def _one_feature_contrast(
    keys: pd.DataFrame,
    aggregate: pd.DataFrame,
    feature: str,
    *,
    stratum_type: str,
    stratum_value: str,
) -> dict[str, Any]:
    mean_values = aggregate[(feature, "mean")].rename("value")
    observed_counts = aggregate[(feature, "count")].rename("observed")
    total_counts = aggregate[(feature, "size")].rename("total")
    joined = keys.merge(
        mean_values,
        left_on="temporal_unit_key",
        right_index=True,
        how="left",
    ).merge(
        observed_counts,
        left_on="temporal_unit_key",
        right_index=True,
        how="left",
    ).merge(
        total_counts,
        left_on="temporal_unit_key",
        right_index=True,
        how="left",
    )
    joined["observed"] = joined["observed"].fillna(0)
    joined["total"] = joined["total"].fillna(0)
    joined["missing_rate"] = np.where(
        joined["total"].gt(0),
        1.0 - joined["observed"] / joined["total"],
        1.0,
    )
    changed = joined.loc[joined["label_changed"], "value"].dropna()
    unchanged = joined.loc[~joined["label_changed"], "value"].dropna()
    return {
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "feature_name": feature,
        "changed_support": int(len(changed)),
        "unchanged_support": int(len(unchanged)),
        "changed_mean": _finite_or_none(changed.mean()),
        "unchanged_mean": _finite_or_none(unchanged.mean()),
        "changed_median": _finite_or_none(changed.median()),
        "unchanged_median": _finite_or_none(unchanged.median()),
        "changed_missing_rate": _finite_or_none(
            joined.loc[joined["label_changed"], "missing_rate"].mean()
        ),
        "unchanged_missing_rate": _finite_or_none(
            joined.loc[~joined["label_changed"], "missing_rate"].mean()
        ),
        "standardized_mean_difference": _standardized_difference(
            changed,
            unchanged,
        ),
        "interpretation": "HYPOTHESIS_ONLY",
    }


def _validate_feature_whitelist(feature_columns: Sequence[str]) -> None:
    if not feature_columns:
        raise PostReviewContractError("explicit_feature_whitelist_required")
    duplicates = [
        column for column in feature_columns if feature_columns.count(column) > 1
    ]
    if duplicates:
        raise PostReviewContractError("duplicate_feature_whitelist_columns")
    forbidden = sorted(set(feature_columns) & MODEL_X_FORBIDDEN_COLUMNS)
    forbidden_tokens = (
        "behavior",
        "label",
        "decision",
        "review",
        "reason",
        "candidate",
        "rank",
        "risk",
        "path",
        "source_id",
        "video_id",
    )
    forbidden.extend(
        column
        for column in feature_columns
        if any(token in column.casefold() for token in forbidden_tokens)
    )
    forbidden = sorted(set(forbidden))
    if forbidden:
        raise PostReviewContractError(
            f"review_fields_forbidden_from_model_x={','.join(forbidden)}"
        )


def _validate_artifact_bindings(
    bindings: Mapping[str, Mapping[str, Any]],
    required: set[str],
) -> dict[str, dict[str, str]]:
    missing = sorted(required - set(bindings))
    if missing:
        raise PostReviewContractError(
            f"artifact_bindings_missing={','.join(missing)}"
        )
    normalized: dict[str, dict[str, str]] = {}
    for name in sorted(required):
        binding = bindings[name]
        path = str(binding.get("path", "")).strip()
        digest = str(binding.get("sha256", "")).strip().casefold()
        if not path:
            raise PostReviewContractError(f"artifact_path_missing={name}")
        assert_not_active_behavior_ledger_path(Path(path))
        if not _valid_sha256(digest):
            raise PostReviewContractError(f"artifact_sha256_invalid={name}")
        normalized[name] = {"path": path, "sha256": digest}
    return normalized


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise PostReviewContractError(
            f"{label}_missing_columns={','.join(missing)}"
        )


def _normalized(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _key_hash(keys: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(key) for key in keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: str) -> bool:
    return bool(_SHA256_RE.fullmatch(value))


def _path_key(value: Any) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    return str(Path(raw).resolve(strict=False)).replace("/", "\\").casefold()


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _wilson_interval(rate: float, effective_n: float) -> list[float] | None:
    if effective_n <= 0 or not math.isfinite(effective_n):
        return None
    z = 1.959963984540054
    denominator = 1.0 + z**2 / effective_n
    center = (rate + z**2 / (2.0 * effective_n)) / denominator
    margin = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / effective_n
            + z**2 / (4.0 * effective_n**2)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _standardized_difference(
    changed: pd.Series,
    unchanged: pd.Series,
) -> float | None:
    if len(changed) < 2 or len(unchanged) < 2:
        return None
    pooled_variance = (
        (float(changed.var(ddof=1)) + float(unchanged.var(ddof=1))) / 2.0
    )
    if not math.isfinite(pooled_variance) or pooled_variance <= 0:
        return None
    return float((changed.mean() - unchanged.mean()) / math.sqrt(pooled_variance))


def bindings_from_paths(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    """Build path/hash bindings after enforcing the active-ledger exclusion."""

    result: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        assert_not_active_behavior_ledger_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return result


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict, deterministic JSON for CLI artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

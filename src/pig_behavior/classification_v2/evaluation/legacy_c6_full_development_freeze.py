"""Freeze the final C6 legacy full-development modality handback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = (
    "classification_v2.legacy_c6_full_development_freeze_config.v1"
)
FREEZE_SCHEMA = (
    "classification_v2.legacy_c6_full_development_freeze.v1"
)
MATRIX_SCHEMA = (
    "classification_v2.legacy_development.c6_modality_decision.v1"
)
TRAINING_SCHEMA = (
    "classification_v2.legacy_development.c6_modality_matrix.v3"
)
CACHE_SCHEMA = (
    "classification_v2.legacy_development.c6_modality_cache.v1"
)
PROMOTION_SCHEMA = (
    "classification_v2.legacy_c6_modality_promotion_freeze.v1"
)
TEMPORAL_SCHEMA = (
    "classification_v2.legacy_c6_temporal_base_freeze.v1"
)
RUN_SCHEMA = "classification_v2.legacy_development.c6_modality_run.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
CONTROLS = ("parameter_matched_zero", "availability_only", "real")
SUPPORTED_MODALITIES = ("roi", "union_context")


@dataclass(frozen=True, slots=True)
class FullDevelopmentFreezeConfig:
    """Hash-bound inputs for the final C6 legacy handback."""

    path: Path
    payload: dict[str, Any]
    repo_root: Path

    def bound_path(self, name: str) -> Path:
        spec = _object(self.payload[name], name)
        return _resolve_inside(self.repo_root, str(spec["path"]))

    @property
    def output_path(self) -> Path:
        output = _object(self.payload["output"], "output")
        return _resolve_inside(self.repo_root, str(output["path"]))


def load_full_development_freeze_config(
    path: Path,
) -> FullDevelopmentFreezeConfig:
    """Load one final-freeze config and verify every bound artifact hash."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    required = {
        "schema_version",
        "lineage_scope",
        "full_matrix",
        "training_config",
        "cache_manifest",
        "temporal_base_freeze",
        "short_promotion_freeze",
        "run_root",
        "decision_contract",
        "implementation",
        "claims",
        "output",
    }
    if set(payload) != required:
        raise ValueError("C6 full-development freeze config keys drift")
    if payload["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("C6 full-development freeze config schema drift")
    if payload["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("C6 full-development freeze lineage drift")
    _validate_contract(
        _object(payload["decision_contract"], "decision_contract")
    )
    expected_claims = {
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "main_branch_promotion_allowed": False,
    }
    if _object(payload["claims"], "claims") != expected_claims:
        raise ValueError("C6 full-development freeze claim boundary drift")
    if set(_object(payload["output"], "output")) != {"path"}:
        raise ValueError("C6 full-development freeze output contract drift")
    config = FullDevelopmentFreezeConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    for name in (
        "full_matrix",
        "training_config",
        "cache_manifest",
        "temporal_base_freeze",
        "short_promotion_freeze",
        "implementation",
    ):
        _verify_spec(config.repo_root, _object(payload[name], name), name)
    run_root = _resolve_inside(config.repo_root, str(payload["run_root"]))
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    return config


def evaluate_c6_full_development_freeze(
    config: FullDevelopmentFreezeConfig,
) -> dict[str, Any]:
    """Audit full-development packets and apply the predeclared gate."""

    full = _read_json(config.bound_path("full_matrix"))
    training = _read_json(config.bound_path("training_config"))
    cache = _read_json(config.bound_path("cache_manifest"))
    temporal = _read_json(config.bound_path("temporal_base_freeze"))
    promotion = _read_json(config.bound_path("short_promotion_freeze"))
    errors = _input_errors(full, training, cache, temporal, promotion, config)
    selected = tuple(
        str(value)
        for value in promotion.get(
            "full_development_authorized_modalities",
            [],
        )
    )
    configured = tuple(
        str(value) for value in training.get("matrix", {}).get("modalities", [])
    )
    if selected != configured:
        errors.append("full_modalities_differ_from_short_promotion_freeze")
    if configured != SUPPORTED_MODALITIES:
        errors.append("full_modalities_differ_from_frozen_roi_union_scope")
    parameter_counts, packet_audit = _audit_packets(
        training,
        cache,
        full,
        config,
    )
    errors.extend(packet_audit["errors"])
    decisions = make_c6_full_development_decision(
        full,
        parameter_counts=parameter_counts,
        selected_modalities=configured,
        contract=_object(
            config.payload["decision_contract"],
            "decision_contract",
        ),
    )
    errors.extend(decisions["errors"])
    retained = decisions["retained_modalities"]
    handback = (
        "RETAIN_A128_ACTOR_ONLY_FOR_C6_LEGACY_HANDBACK"
        if not retained
        else "RETAIN_A128_WITH_" + "_AND_".join(value.upper() for value in retained)
    )
    valid = not errors
    return {
        "schema_version": FREEZE_SCHEMA,
        "status": "PASS_C6_FULL_DEVELOPMENT_FREEZE" if valid else "FAIL",
        "lineage_scope": LINEAGE_SCOPE,
        "config_sha256": file_sha256(config.path),
        "full_matrix_sha256": file_sha256(config.bound_path("full_matrix")),
        "training_config_sha256": file_sha256(
            config.bound_path("training_config")
        ),
        "cache_manifest_sha256": file_sha256(
            config.bound_path("cache_manifest")
        ),
        "temporal_base_freeze_sha256": file_sha256(
            config.bound_path("temporal_base_freeze")
        ),
        "short_promotion_freeze_sha256": file_sha256(
            config.bound_path("short_promotion_freeze")
        ),
        "decision_contract": config.payload["decision_contract"],
        "packet_audit": packet_audit,
        "modality_decisions": decisions["modality_decisions"],
        "retained_modalities": retained,
        "handback_decision": handback,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "main_branch_promotion_allowed": False,
        "retest_on_main_frozen_reviewed_lineage_required": True,
        "errors": errors,
        "valid": valid,
    }


def write_c6_full_development_freeze(
    config: FullDevelopmentFreezeConfig,
) -> dict[str, Any]:
    """Write one immutable final C6 legacy full-development freeze."""

    payload = evaluate_c6_full_development_freeze(config)
    output = config.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {**payload, "output_path": str(output)}


def make_c6_full_development_decision(
    full: dict[str, Any],
    *,
    parameter_counts: dict[str, dict[str, list[int]]],
    selected_modalities: tuple[str, ...],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Apply paired full-confirmation criteria to promoted modalities."""

    comparisons = full.get("comparisons")
    if not isinstance(comparisons, dict):
        return {
            "modality_decisions": {},
            "retained_modalities": [],
            "errors": ["full_comparisons_missing"],
        }
    errors: list[str] = []
    decisions: dict[str, Any] = {}
    retained_modalities: list[str] = []
    for modality in selected_modalities:
        if modality not in SUPPORTED_MODALITIES:
            errors.append(f"unsupported_full_modality={modality}")
            continue
        zero = comparisons.get(
            f"{modality}__real_minus_parameter_matched_zero"
        )
        availability = comparisons.get(
            f"{modality}__real_minus_availability_only"
        )
        if not isinstance(zero, dict) or not isinstance(availability, dict):
            errors.append(f"missing_full_comparison={modality}")
            continue
        zero_delta = float(zero["macro_f1_delta"])
        availability_delta = float(availability["macro_f1_delta"])
        availability_only_delta = float(
            availability["baseline_metrics"]["macro_f1_global_10_class"]
        ) - float(zero["baseline_metrics"]["macro_f1_global_10_class"])
        rare_delta = float(zero["group_deltas"]["rare"]["macro_f1_delta"])
        nll_delta = float(zero["candidate_metrics"]["nll"]) - float(
            zero["baseline_metrics"]["nll"]
        )
        flattened = [
            value
            for control in CONTROLS
            for value in parameter_counts.get(modality, {}).get(control, [])
        ]
        criteria = {
            "gain_vs_zero_meets_margin": (
                zero_delta >= float(contract["minimum_macro_f1_gain"])
            ),
            "gain_vs_availability_meets_margin": (
                availability_delta
                >= float(contract["minimum_macro_f1_gain"])
            ),
            "zero_cluster_ci_low_positive": (
                float(zero["video_cluster_bootstrap"]["ci_low"]) > 0.0
            ),
            "availability_cluster_ci_low_positive": (
                float(availability["video_cluster_bootstrap"]["ci_low"])
                > 0.0
            ),
            "nll_improves_vs_zero": nll_delta < 0.0,
            "availability_only_is_bounded_diagnostic": (
                abs(availability_only_delta)
                <= float(contract["maximum_absolute_availability_only_gain"])
            ),
            "rare_group_drop_within_limit": (
                rare_delta
                >= -float(contract["maximum_rare_group_macro_f1_drop"])
            ),
            "all_modes_parameter_matched": (
                bool(flattened) and len(set(flattened)) == 1
            ),
        }
        retained = all(criteria.values())
        if retained:
            retained_modalities.append(modality)
        decisions[modality] = {
            "decision": (
                "RETAIN_AFTER_FULL_DEVELOPMENT_CONFIRMATION"
                if retained
                else "DO_NOT_RETAIN_AFTER_FULL_DEVELOPMENT_CONFIRMATION"
            ),
            "retained_for_legacy_c6_handback": retained,
            "retest_on_main_frozen_reviewed_lineage_required": True,
            "criteria": criteria,
            "observed": {
                "macro_f1_delta_vs_zero": zero_delta,
                "macro_f1_delta_vs_availability": availability_delta,
                "availability_only_minus_zero_macro_f1": (
                    availability_only_delta
                ),
                "zero_cluster_ci_low": float(
                    zero["video_cluster_bootstrap"]["ci_low"]
                ),
                "availability_cluster_ci_low": float(
                    availability["video_cluster_bootstrap"]["ci_low"]
                ),
                "nll_delta_vs_zero": nll_delta,
                "rare_group_macro_f1_delta_vs_zero": rare_delta,
                "parameter_count": flattened[0] if flattened else None,
            },
        }
    return {
        "modality_decisions": decisions,
        "retained_modalities": retained_modalities,
        "errors": errors,
    }


def _input_errors(
    full: dict[str, Any],
    training: dict[str, Any],
    cache: dict[str, Any],
    temporal: dict[str, Any],
    promotion: dict[str, Any],
    config: FullDevelopmentFreezeConfig,
) -> list[str]:
    checks = (
        (full, "full", MATRIX_SCHEMA, "PASS"),
        (cache, "cache", CACHE_SCHEMA, "PASS_LEGACY_C6_MODALITY_CACHE"),
        (
            temporal,
            "temporal",
            TEMPORAL_SCHEMA,
            "PASS_C6_TEMPORAL_BASE_FREEZE",
        ),
        (
            promotion,
            "promotion",
            PROMOTION_SCHEMA,
            "PASS_C6_MODALITY_PROMOTION_FREEZE",
        ),
    )
    errors = [
        f"{name}_{field}_drift"
        for payload, name, schema, status in checks
        for field, expected in (
            ("schema_version", schema),
            ("status", status),
            ("lineage_scope", LINEAGE_SCOPE),
            ("valid", True),
        )
        if payload.get(field) != expected
    ]
    if training.get("schema_version") != TRAINING_SCHEMA:
        errors.append("training_schema_version_drift")
    if training.get("training_scope") != "full_development_confirmation":
        errors.append("training_scope_drift")
    training_hash = file_sha256(config.bound_path("training_config"))
    cache_hash = file_sha256(config.bound_path("cache_manifest"))
    if full.get("config_sha256") != training_hash:
        errors.append("full_training_config_hash_drift")
    if cache.get("config_sha256") != training_hash:
        errors.append("cache_training_config_hash_drift")
    for name, payload in (
        ("full", full),
        ("cache", cache),
        ("promotion", promotion),
    ):
        if payload.get("full_oof_authorized") is not False:
            errors.append(f"{name}_full_oof_claim_drift")
    if cache_hash != full.get("cache_manifest_sha256", cache_hash):
        errors.append("full_cache_manifest_hash_drift")
    return errors


def _audit_packets(
    training: dict[str, Any],
    cache: dict[str, Any],
    full: dict[str, Any],
    config: FullDevelopmentFreezeConfig,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, Any]]:
    run_root = _resolve_inside(config.repo_root, str(config.payload["run_root"]))
    repeats = [str(value) for value in training["execution"]["repeats"]]
    modes = [str(value) for value in training["matrix"]["mode_ids"]]
    config_sha = file_sha256(config.bound_path("training_config"))
    cache_sha = file_sha256(config.bound_path("cache_manifest"))
    expected_steps = int(training["optimization"]["maximum_optimizer_steps"])
    expected_rows = int(cache["validation_native_units"])
    errors: list[str] = []
    counts: dict[str, dict[str, list[int]]] = {
        modality: {control: [] for control in CONTROLS}
        for modality in SUPPORTED_MODALITIES
    }
    process_ids: dict[str, int] = {}
    audited = 0
    for repeat_id in repeats:
        summary_path = run_root / repeat_id / "repeat_result.json"
        summary = _read_json(summary_path)
        process_ids[repeat_id] = int(summary.get("process_id", -1))
        if summary.get("valid") is not True or summary.get("errors") != []:
            errors.append(f"invalid_repeat_summary={repeat_id}")
        for mode_id in modes:
            run = _read_json(run_root / repeat_id / mode_id / "run.json")
            expected = {
                "schema_version": RUN_SCHEMA,
                "status": "completed",
                "mode_id": mode_id,
                "repeat_id": repeat_id,
                "config_sha256": config_sha,
                "cache_manifest_sha256": cache_sha,
                "optimizer_steps": expected_steps,
                "human_review_complete": False,
                "reviewed_or_final_claim_allowed": False,
                "q2_claim_allowed": False,
                "full_oof_authorized": False,
                "errors": [],
                "valid": True,
            }
            errors.extend(
                f"run_{field}_drift={repeat_id}:{mode_id}"
                for field, value in expected.items()
                if run.get(field) != value
            )
            if run.get("metrics", {}).get("native_unit_rows") != expected_rows:
                errors.append(f"run_native_rows_drift={repeat_id}:{mode_id}")
            if mode_id != "actor_only":
                modality, control = mode_id.split("__", maxsplit=1)
                if modality in counts and control in counts[modality]:
                    counts[modality][control].append(
                        int(run["parameter_count"])
                    )
            audited += 1
    if full.get("repeat_process_ids") != process_ids:
        errors.append("full_repeat_process_ids_drift")
    return counts, {
        "expected_packets": len(repeats) * len(modes),
        "audited_packets": audited,
        "repeat_process_ids": process_ids,
        "errors": errors,
        "valid": not errors,
    }


def _validate_contract(contract: dict[str, Any]) -> None:
    required = {
        "minimum_macro_f1_gain",
        "maximum_absolute_availability_only_gain",
        "maximum_rare_group_macro_f1_drop",
        "require_positive_video_cluster_ci_low",
        "require_nll_improvement_vs_zero",
    }
    if set(contract) != required:
        raise ValueError("C6 full-development decision contract drift")
    if contract["require_positive_video_cluster_ci_low"] is not True:
        raise ValueError("C6 full-development positive-CI gate disabled")
    if contract["require_nll_improvement_vs_zero"] is not True:
        raise ValueError("C6 full-development NLL gate disabled")


def _verify_spec(root: Path, spec: dict[str, Any], name: str) -> None:
    if set(spec) != {"path", "sha256"}:
        raise ValueError(f"C6 full-development {name} hash spec drift")
    path = _resolve_inside(root, str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(spec["sha256"]):
        raise ValueError(f"C6 full-development {name} hash mismatch")


def _resolve_inside(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {value}") from exc
    return resolved


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


__all__ = [
    "FullDevelopmentFreezeConfig",
    "evaluate_c6_full_development_freeze",
    "load_full_development_freeze_config",
    "make_c6_full_development_decision",
    "write_c6_full_development_freeze",
]

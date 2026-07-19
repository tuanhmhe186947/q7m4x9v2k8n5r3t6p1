"""Apply the frozen L6 promotion gate to rebuild-bound C6 modality evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = "classification_v2.legacy_c6_modality_promotion_config.v1"
FREEZE_SCHEMA = "classification_v2.legacy_c6_modality_promotion_freeze.v1"
SHORT_SCHEMA = "classification_v2.legacy_development.c6_modality_decision.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
CONTROLS = ("parameter_matched_zero", "availability_only", "real")
MODALITIES = (
    "geometry",
    "motion",
    "roi",
    "numeric_social",
    "pen_context",
    "union_context",
    "full_frame_context",
)
PEN_FOCUS_LABELS = ("stand", "move", "explore")


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    """Hash-bound inputs for one post-short promotion decision."""

    path: Path
    payload: dict[str, Any]
    repo_root: Path

    def bound_path(self, name: str) -> Path:
        spec = _object(self.payload[name], name)
        return _resolve_inside(self.repo_root, str(spec["path"]))

    @property
    def output_path(self) -> Path:
        value = str(_object(self.payload["output"], "output")["path"])
        return _resolve_inside(self.repo_root, value)


def load_promotion_config(path: Path) -> PromotionConfig:
    """Load and hash-audit every immutable decision input."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    required = {
        "schema_version",
        "lineage_scope",
        "short_matrix",
        "short_training_config",
        "short_cache_manifest",
        "temporal_base_freeze",
        "run_root",
        "decision_contract",
        "criteria_authority",
        "implementation",
        "claims",
        "output",
    }
    if set(payload) != required:
        raise ValueError("C6 promotion config keys drift")
    if payload["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("C6 promotion config schema drift")
    if payload["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("C6 promotion config lineage drift")
    _validate_contract(_object(payload["decision_contract"], "decision_contract"))
    claims = _object(payload["claims"], "claims")
    expected_claims = {
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
    }
    if claims != expected_claims:
        raise ValueError("C6 promotion claim boundary drift")
    output = _object(payload["output"], "output")
    if set(output) != {"path"}:
        raise ValueError("C6 promotion output contract drift")
    config = PromotionConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    for name in (
        "short_matrix",
        "short_training_config",
        "short_cache_manifest",
        "temporal_base_freeze",
        "implementation",
    ):
        _verify_spec(config.repo_root, _object(payload[name], name), name)
    authorities = payload["criteria_authority"]
    if not isinstance(authorities, list) or not authorities:
        raise ValueError("C6 promotion criteria authority is empty")
    for index, value in enumerate(authorities):
        _verify_spec(
            config.repo_root,
            _object(value, f"criteria_authority[{index}]"),
            f"criteria_authority[{index}]",
        )
    run_root = _resolve_inside(config.repo_root, str(payload["run_root"]))
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    return config


def evaluate_c6_modality_promotion(config: PromotionConfig) -> dict[str, Any]:
    """Audit the short matrix and decide each modality without retraining."""

    short = _read_json(config.bound_path("short_matrix"))
    training_config = _read_json(config.bound_path("short_training_config"))
    cache = _read_json(config.bound_path("short_cache_manifest"))
    temporal = _read_json(config.bound_path("temporal_base_freeze"))
    errors = _short_input_errors(short, training_config, cache, temporal)
    parameter_counts, packet_audit = _audit_run_packets(
        config,
        short,
        cache_manifest_sha256=file_sha256(
            config.bound_path("short_cache_manifest")
        ),
    )
    errors.extend(packet_audit["errors"])
    decisions = make_c6_modality_promotion_decision(
        short,
        parameter_counts=parameter_counts,
        contract=_object(config.payload["decision_contract"], "decision_contract"),
    )
    errors.extend(decisions["errors"])
    valid = not errors
    return {
        "schema_version": FREEZE_SCHEMA,
        "status": "PASS_C6_MODALITY_PROMOTION_FREEZE" if valid else "FAIL",
        "lineage_scope": LINEAGE_SCOPE,
        "config_sha256": file_sha256(config.path),
        "short_matrix_sha256": file_sha256(config.bound_path("short_matrix")),
        "short_training_config_sha256": file_sha256(
            config.bound_path("short_training_config")
        ),
        "short_cache_manifest_sha256": file_sha256(
            config.bound_path("short_cache_manifest")
        ),
        "temporal_base_freeze_sha256": file_sha256(
            config.bound_path("temporal_base_freeze")
        ),
        "decision_contract": config.payload["decision_contract"],
        "criteria_authority": config.payload["criteria_authority"],
        "packet_audit": packet_audit,
        "modality_decisions": decisions["modality_decisions"],
        "full_development_authorized_modalities": decisions[
            "full_development_authorized_modalities"
        ],
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "main_branch_promotion_allowed": False,
        "retest_on_main_frozen_reviewed_lineage_required": True,
        "errors": errors,
        "valid": valid,
    }


def write_c6_modality_promotion_freeze(config: PromotionConfig) -> dict[str, Any]:
    """Write one immutable promotion freeze artifact."""

    payload = evaluate_c6_modality_promotion(config)
    output = config.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {**payload, "output_path": str(output)}


def make_c6_modality_promotion_decision(
    short: dict[str, Any],
    *,
    parameter_counts: dict[str, dict[str, list[int]]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Apply the pre-existing L6 criteria to each C6 modality family."""

    errors: list[str] = []
    comparisons = short.get("comparisons")
    if not isinstance(comparisons, dict):
        return {
            "modality_decisions": {},
            "full_development_authorized_modalities": [],
            "errors": ["short_comparisons_missing"],
        }
    decisions: dict[str, Any] = {}
    authorized: list[str] = []
    for modality in MODALITIES:
        zero_key = f"{modality}__real_minus_parameter_matched_zero"
        availability_key = f"{modality}__real_minus_availability_only"
        zero = comparisons.get(zero_key)
        availability = comparisons.get(availability_key)
        if not isinstance(zero, dict) or not isinstance(availability, dict):
            errors.append(f"missing_comparison={modality}")
            continue
        minimum_gain = float(
            contract[
                "pen_minimum_macro_f1_gain"
                if modality == "pen_context"
                else "minimum_macro_f1_gain"
            ]
        )
        zero_delta = float(zero["macro_f1_delta"])
        availability_delta = float(availability["macro_f1_delta"])
        availability_only_delta = float(
            availability["baseline_metrics"]["macro_f1_global_10_class"]
        ) - float(zero["baseline_metrics"]["macro_f1_global_10_class"])
        rare_delta = float(zero["group_deltas"]["rare"]["macro_f1_delta"])
        nll_delta = float(zero["candidate_metrics"]["nll"]) - float(
            zero["baseline_metrics"]["nll"]
        )
        counts = parameter_counts.get(modality, {})
        flattened = [
            value
            for control in CONTROLS
            for value in counts.get(control, [])
        ]
        parameter_matched = bool(flattened) and len(set(flattened)) == 1
        criteria = {
            "gain_vs_zero_meets_margin": zero_delta >= minimum_gain,
            "gain_vs_availability_meets_margin": (
                availability_delta >= minimum_gain
            ),
            "zero_cluster_ci_low_positive": (
                float(zero["video_cluster_bootstrap"]["ci_low"]) > 0.0
            ),
            "availability_cluster_ci_low_positive": (
                float(availability["video_cluster_bootstrap"]["ci_low"]) > 0.0
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
            "all_modes_parameter_matched": parameter_matched,
        }
        if modality == "pen_context":
            criteria["pen_focus_group_gain_meets_margin"] = (
                _focus_delta(zero, PEN_FOCUS_LABELS)
                >= float(contract["pen_minimum_focus_group_macro_f1_gain"])
            )
        retained = all(criteria.values())
        if retained:
            authorized.append(modality)
        decisions[modality] = {
            "decision": (
                "RETAIN_FOR_FULL_LEGACY_DEVELOPMENT"
                if retained
                else "DO_NOT_EXPAND_FROM_CURRENT_SHORT_EVIDENCE"
            ),
            "criteria": criteria,
            "full_development_authorized": retained,
            "observed": {
                "macro_f1_delta_vs_zero": zero_delta,
                "macro_f1_delta_vs_availability": availability_delta,
                "availability_only_minus_zero_macro_f1": (
                    availability_only_delta
                ),
                "rare_group_macro_f1_delta_vs_zero": rare_delta,
                "nll_delta_vs_zero": nll_delta,
                "parameter_count": flattened[0] if parameter_matched else None,
            },
            "applies_to_merged_reviewed_data": False,
            "merged_reviewed_reassessment_required": True,
        }
    return {
        "modality_decisions": decisions,
        "full_development_authorized_modalities": authorized,
        "errors": errors,
    }


def _audit_run_packets(
    config: PromotionConfig,
    short: dict[str, Any],
    *,
    cache_manifest_sha256: str,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, Any]]:
    repeats = [str(value) for value in short.get("repeat_ids", [])]
    run_root = _resolve_inside(config.repo_root, str(config.payload["run_root"]))
    expected_config_sha = str(short.get("config_sha256", ""))
    errors: list[str] = []
    counts = {
        modality: {control: [] for control in CONTROLS}
        for modality in MODALITIES
    }
    packet_count = 0
    process_ids: dict[str, int] = {}
    for repeat in repeats:
        for modality in MODALITIES:
            for control in CONTROLS:
                mode_id = f"{modality}__{control}"
                path = run_root / repeat / mode_id / "run.json"
                if not path.is_file():
                    errors.append(f"missing_run={repeat}:{mode_id}")
                    continue
                packet = _read_json(path)
                expected = {
                    "status": "completed",
                    "mode_id": mode_id,
                    "repeat_id": repeat,
                    "config_sha256": expected_config_sha,
                    "cache_manifest_sha256": cache_manifest_sha256,
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                    "reviewed_or_final_claim_allowed": False,
                    "q2_claim_allowed": False,
                    "full_oof_authorized": False,
                    "valid": True,
                }
                for name, value in expected.items():
                    if packet.get(name) != value:
                        errors.append(f"packet_drift={repeat}:{mode_id}:{name}")
                process_id = packet.get("process_id")
                if not isinstance(process_id, int) or process_id <= 0:
                    errors.append(f"invalid_process_id={repeat}:{mode_id}")
                else:
                    previous = process_ids.setdefault(repeat, process_id)
                    if previous != process_id:
                        errors.append(f"mixed_process={repeat}")
                parameter_count = packet.get("parameter_count")
                if not isinstance(parameter_count, int) or parameter_count <= 0:
                    errors.append(f"invalid_parameter_count={repeat}:{mode_id}")
                else:
                    counts[modality][control].append(parameter_count)
                packet_count += 1
    if len(set(process_ids.values())) != len(process_ids):
        errors.append("repeat_processes_not_distinct")
    expected_process_ids = short.get("repeat_process_ids")
    if expected_process_ids != process_ids:
        errors.append("short_decision_process_ids_drift")
    return counts, {
        "expected_packets": len(repeats) * len(MODALITIES) * len(CONTROLS),
        "audited_packets": packet_count,
        "repeat_process_ids": process_ids,
        "errors": errors,
        "valid": not errors,
    }


def _short_input_errors(
    short: dict[str, Any],
    training: dict[str, Any],
    cache: dict[str, Any],
    temporal: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_short = {
        "schema_version": SHORT_SCHEMA,
        "status": "PASS",
        "lineage_scope": LINEAGE_SCOPE,
        "mode_count": 22,
        "full_oof_authorized": False,
        "errors": [],
        "valid": True,
    }
    for name, value in expected_short.items():
        if short.get(name) != value:
            errors.append(f"short_matrix_{name}_drift")
    if training.get("training_scope") != "short_repeat_gate":
        errors.append("short_training_scope_drift")
    if training.get("lineage_scope") != LINEAGE_SCOPE:
        errors.append("short_training_lineage_drift")
    if cache.get("lineage_scope") != LINEAGE_SCOPE or cache.get("valid") is not True:
        errors.append("short_cache_contract_drift")
    if temporal.get("status") != "PASS_C6_TEMPORAL_BASE_FREEZE":
        errors.append("temporal_freeze_status_drift")
    if temporal.get("selected_base_mode") != "A128":
        errors.append("temporal_freeze_base_drift")
    return errors


def _focus_delta(comparison: dict[str, Any], labels: tuple[str, ...]) -> float:
    per_class = _object(comparison["per_class"], "per_class")
    return sum(float(per_class[label]["f1_delta"]) for label in labels) / len(
        labels
    )


def _validate_contract(contract: dict[str, Any]) -> None:
    expected = {
        "minimum_macro_f1_gain": 0.02,
        "pen_minimum_macro_f1_gain": 0.01,
        "pen_minimum_focus_group_macro_f1_gain": 0.01,
        "maximum_absolute_availability_only_gain": 0.01,
        "maximum_rare_group_macro_f1_drop": 0.02,
        "require_positive_video_cluster_ci_low": True,
        "require_nll_improvement_vs_zero": True,
    }
    if contract != expected:
        raise ValueError("C6 promotion decision contract drift")


def _verify_spec(root: Path, spec: dict[str, Any], name: str) -> None:
    if set(spec) != {"path", "sha256"}:
        raise ValueError(f"{name} artifact spec drift")
    path = _resolve_inside(root, str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(spec["sha256"]):
        raise ValueError(f"{name} artifact hash mismatch")


def _resolve_inside(root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes repository={resolved}") from error
    return resolved


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _object(value, str(path))


__all__ = [
    "CONFIG_SCHEMA",
    "FREEZE_SCHEMA",
    "LINEAGE_SCOPE",
    "MODALITIES",
    "PromotionConfig",
    "evaluate_c6_modality_promotion",
    "load_promotion_config",
    "make_c6_modality_promotion_decision",
    "write_c6_modality_promotion_freeze",
]

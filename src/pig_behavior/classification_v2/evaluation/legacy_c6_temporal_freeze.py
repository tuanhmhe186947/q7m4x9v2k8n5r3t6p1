"""Freeze the C6 modality base from paired temporal-control evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.evaluation.statistics import (
    paired_cluster_bootstrap,
)
from pig_behavior.classification_v2.training.legacy_development_c6_temporal_controls import (
    CONFIG_SCHEMA_V2,
    LINEAGE_SCOPE,
    load_c6_temporal_control_config,
    static_c6_temporal_control_preflight,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

FREEZE_SCHEMA = "classification_v2.legacy_c6_temporal_base_freeze.v1"
FREEZE_SCHEMA_V2 = "classification_v2.legacy_c6_temporal_base_freeze.v2"
SHORT_GATE_STATUS = "PASS_C6_TEMPORAL_CONTROLS_SHORT_GATE"
PRIOR_BASE_MODE = "A128"
EXPECTED_PAIRS = {
    "tcn_capacity": ("TCN128", "MW317"),
    "tcn_order": ("TCN128", "TCN128_SEQUENCE_SHUFFLED"),
    "transformer_capacity": ("TR128_REAL_DELTA", "MW381"),
    "transformer_timing_constant": (
        "TR128_REAL_DELTA",
        "TR128_CONSTANT_DELTA",
    ),
    "transformer_timing_alignment": (
        "TR128_REAL_DELTA",
        "TR128_DELTA_SHUFFLED",
    ),
    "transformer_order": (
        "TR128_REAL_DELTA",
        "TR128_SEQUENCE_SHUFFLED",
    ),
}
FAMILY_REQUIREMENTS = {
    "TCN128": ("tcn_capacity", "tcn_order"),
    "TR128_REAL_DELTA": (
        "transformer_capacity",
        "transformer_timing_constant",
        "transformer_timing_alignment",
        "transformer_order",
    ),
}
BASE_MODE_IDS = (
    "SF128",
    "M128",
    "A128",
    "MW317",
    "TCN128",
    "MW381",
    "TR128",
)


def evaluate_c6_temporal_freeze(
    config_path: Path,
    *,
    output_path: Path | None = None,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260719,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate six paired controls and freeze the prior base when none pass."""

    if bootstrap_iterations < 1000:
        raise ValueError("C6 temporal freeze requires at least 1000 bootstrap draws")
    config = load_c6_temporal_control_config(config_path)
    if config.payload.get("schema_version") != CONFIG_SCHEMA_V2:
        raise ValueError("C6 temporal freeze requires config schema v2")
    static = static_c6_temporal_control_preflight(config)
    if not static["valid"]:
        raise ValueError(f"C6 temporal config static preflight failed={static['errors']}")

    short_gate_path = config.output_root / "c6_temporal_controls_short_gate.json"
    short_gate = _load_json(short_gate_path)
    errors = _validate_short_gate(config, short_gate)
    declared_pairs = {
        key: tuple(value)
        for key, value in config.payload.get("controlled_pairs", {}).items()
    }
    if declared_pairs != EXPECTED_PAIRS:
        errors.append("controlled_pair_contract_drift")

    predictions: dict[str, pd.DataFrame] = {}
    for mode_id in sorted({mode for pair in EXPECTED_PAIRS.values() for mode in pair}):
        try:
            predictions[mode_id] = _load_mode_predictions(config, mode_id)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{mode_id}:prediction_validation={exc}")

    comparisons: dict[str, Any] = {}
    if not errors:
        for index, (pair_id, (candidate_id, baseline_id)) in enumerate(
            EXPECTED_PAIRS.items()
        ):
            bootstrap = paired_cluster_bootstrap(
                predictions[candidate_id],
                predictions[baseline_id],
                cluster_col="video_key",
                unit_col="temporal_unit_key",
                fold_col="recording_group_id",
                true_col="behavior_label",
                pred_col="predicted_label",
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + index,
                outer_predictions_used_for_model_selection=False,
            )
            bootstrap["bootstrap_engine_uncertainty_method"] = bootstrap[
                "uncertainty_method"
            ]
            bootstrap["uncertainty_method"] = (
                "paired_video_cluster_bootstrap_percentile"
            )
            comparisons[pair_id] = {
                "candidate_mode": candidate_id,
                "baseline_mode": baseline_id,
                "candidate_minus_baseline": bootstrap,
                "positive_point_direction": bootstrap["macro_f1_delta"] > 0.0,
                "positive_uncertainty_low": bootstrap["ci_low"] > 0.0,
                "passes_control": (
                    bootstrap["macro_f1_delta"] > 0.0
                    and bootstrap["ci_low"] > 0.0
                ),
            }

    family_decisions = _family_decisions(comparisons)
    promoted = sorted(
        mode_id
        for mode_id, decision in family_decisions.items()
        if decision["passes_all_required_controls"]
    )
    if promoted:
        errors.append(f"temporal_candidate_requires_separate_retest={promoted}")
    valid = not errors
    decision = (
        "FREEZE_PRIOR_A128_FOR_C6_MODALITY_SCREENING"
        if valid
        else "TEMPORAL_FREEZE_NOT_AUTHORIZED"
    )
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "status": "PASS_C6_TEMPORAL_BASE_FREEZE" if valid else "FAIL_C6_TEMPORAL_BASE_FREEZE",
        "decision": decision,
        "selected_base_mode": PRIOR_BASE_MODE if valid else None,
        "selected_base_is_carried_prior_not_tested_in_this_matrix": True,
        "interpretation": (
            "No temporal candidate demonstrated value beyond every required "
            "capacity, order, and timing control; retain the prior A128 base."
        ),
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "short_gate_path": str(short_gate_path.resolve()),
        "short_gate_sha256": file_sha256(short_gate_path),
        "short_gate_status": short_gate.get("status"),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": file_sha256(Path(__file__)),
        "source_repeat": "repeat01",
        "comparison_scope": "short_validation_native_units",
        "comparison_cluster": "video_key",
        "comparisons": comparisons,
        "family_decisions": family_decisions,
        "promoted_temporal_candidates": promoted,
        "modality_matrix_authorized": valid,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "legacy_sets_full_data_base": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "retest_on_main_frozen_reviewed_lineage_required": True,
        "skills": [
            "dataset-contract-leakage-guard",
            "experiment-lineage-reproducibility",
            "scientific-ablation-controller",
            "grouped-cv-evaluation",
            "safe-refactor-test-guardian",
        ],
        "errors": errors,
        "valid": valid,
    }
    destination = output_path or config.output_root / "c6_temporal_base_freeze.json"
    _write_json_exclusive(destination, payload)
    return destination, payload


def freeze_c6_base_from_full_development_decision(
    decision_path: Path,
    output_path: Path,
    *,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Freeze the measured best base for legacy-only C6 modality testing."""

    root = project_root.resolve()
    resolved_decision = decision_path.resolve()
    decision = _load_json(resolved_decision)
    errors = _full_development_decision_errors(decision)
    ranking: list[str] = []
    metrics = decision.get("mode_metrics")
    if isinstance(metrics, dict) and set(metrics) == set(BASE_MODE_IDS):
        ranking = sorted(
            BASE_MODE_IDS,
            key=lambda mode_id: (
                -float(metrics[mode_id]["macro_f1_global_10_class"]),
                float(metrics[mode_id]["nll"]),
                mode_id,
            ),
        )
    selected = ranking[0] if ranking else None
    if selected != PRIOR_BASE_MODE:
        errors.append(f"C6 runner does not support measured best base={selected}")
    valid = not errors
    relative_decision = resolved_decision.relative_to(root)
    payload = {
        "schema_version": FREEZE_SCHEMA_V2,
        "status": "PASS_C6_TEMPORAL_BASE_FREEZE" if valid else "FAIL",
        "decision": (
            "FREEZE_EVALUATED_A128_FOR_LEGACY_16F_MODALITY_SCREENING"
            if valid
            else "TEMPORAL_FREEZE_NOT_AUTHORIZED"
        ),
        "selected_base_mode": selected if valid else None,
        "selected_base_is_carried_prior_not_tested_in_this_matrix": False,
        "selection_rule": (
            "maximum_full_development_global_10_class_macro_f1_then_nll"
        ),
        "mode_ranking": ranking,
        "selected_metrics": metrics.get(selected) if ranking else None,
        "base_selection_decision": {
            "path": relative_decision.as_posix(),
            "sha256": file_sha256(resolved_decision),
        },
        "common_native_universe": decision.get("common_native_universe"),
        "modality_matrix_authorized": valid,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "legacy_sets_full_data_base": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "retest_on_main_frozen_reviewed_lineage_required": True,
        "errors": errors,
        "valid": valid,
    }
    _write_json_exclusive(output_path, payload)
    return output_path, payload


def _full_development_decision_errors(decision: dict[str, Any]) -> list[str]:
    expected = {
        "schema_version": (
            "classification_v2.legacy_development.temporal_base_decision.v1"
        ),
        "status": "PASS_LEGACY_TEMPORAL_BASE_PAIRED_DECISION",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "legacy_sets_final_full_data_base": False,
        "valid": True,
    }
    errors = [
        f"base_decision_{key}_drift"
        for key, value in expected.items()
        if decision.get(key) != value
    ]
    universe = decision.get("common_native_universe")
    if not isinstance(universe, dict):
        errors.append("base_decision_native_universe_missing")
    elif (
        universe.get("native_units") != 241
        or universe.get("video_clusters") != 32
        or universe.get("outer_holdout_rows") != 0
    ):
        errors.append("base_decision_native_universe_drift")
    comparisons = decision.get("paired_comparisons")
    if not isinstance(comparisons, dict):
        errors.append("base_decision_comparisons_missing")
    else:
        for pair_id in ("content_weighting", "operational_attention_vs_single"):
            action = (
                comparisons.get(pair_id, {})
                .get("transfer_decision", {})
                .get("screening_action")
            )
            if action != "CARRY":
                errors.append(f"base_decision_{pair_id}_not_carried")
    if set(decision.get("mode_metrics", {})) != set(BASE_MODE_IDS):
        errors.append("base_decision_mode_metrics_drift")
    return errors


def _validate_short_gate(config: Any, gate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if gate.get("status") != SHORT_GATE_STATUS or not gate.get("valid"):
        errors.append("short_gate_not_pass")
    if gate.get("config_sha256") != config.sha256:
        errors.append("short_gate_config_hash_drift")
    if gate.get("clean_lineage_handoff_id") != config.payload["execution"].get(
        "clean_lineage_handoff_id"
    ):
        errors.append("short_gate_handoff_drift")
    if gate.get("common_native_unit_sha256") in {None, ""}:
        errors.append("short_gate_native_universe_hash_missing")
    if gate.get("full_oof_authorized") is not False:
        errors.append("short_gate_full_oof_boundary_drift")
    if gate.get("legacy_sets_full_data_base") is not False:
        errors.append("short_gate_legacy_claim_boundary_drift")
    return errors


def _load_mode_predictions(config: Any, mode_id: str) -> pd.DataFrame:
    run_root = config.output_root / "short_repeat_gate" / mode_id / "repeat01"
    prediction_path = run_root / "validation_native_predictions.csv"
    manifest = _load_json(run_root / "prediction_manifest.json")
    if manifest.get("config_sha256") != config.sha256:
        raise ValueError("prediction config hash drift")
    if manifest.get("mode_id") != mode_id or manifest.get("repeat_id") != "repeat01":
        raise ValueError("prediction mode or repeat drift")
    if manifest.get("prediction_sha256") != file_sha256(prediction_path):
        raise ValueError("prediction file hash drift")
    frame = pd.read_csv(prediction_path)
    required = {
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "behavior_label",
        "predicted_label",
        "lineage_scope",
        "human_review_complete",
        "c6_temporal_control_mode_id",
        "repeat_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns={missing}")
    if frame.empty or frame["temporal_unit_key"].duplicated().any():
        raise ValueError("empty or duplicate native-unit predictions")
    if not frame["lineage_scope"].astype(str).eq(LINEAGE_SCOPE).all():
        raise ValueError("lineage scope drift")
    if frame["human_review_complete"].astype(str).str.lower().ne("false").any():
        raise ValueError("human-review claim drift")
    if not frame["c6_temporal_control_mode_id"].astype(str).eq(mode_id).all():
        raise ValueError("mode column drift")
    if not frame["repeat_id"].astype(str).eq("repeat01").all():
        raise ValueError("repeat column drift")
    return frame


def _family_decisions(comparisons: dict[str, Any]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for mode_id, pair_ids in FAMILY_REQUIREMENTS.items():
        available = all(pair_id in comparisons for pair_id in pair_ids)
        passes = available and all(
            comparisons[pair_id]["passes_control"] for pair_id in pair_ids
        )
        decisions[mode_id] = {
            "required_pairs": list(pair_ids),
            "failed_pairs": [
                pair_id
                for pair_id in pair_ids
                if pair_id not in comparisons
                or not comparisons[pair_id]["passes_control"]
            ],
            "passes_all_required_controls": passes,
            "screening_action": "RETEST" if passes else "DROP",
        }
    return decisions


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required={path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "EXPECTED_PAIRS",
    "FAMILY_REQUIREMENTS",
    "FREEZE_SCHEMA",
    "FREEZE_SCHEMA_V2",
    "evaluate_c6_temporal_freeze",
    "freeze_c6_base_from_full_development_decision",
]

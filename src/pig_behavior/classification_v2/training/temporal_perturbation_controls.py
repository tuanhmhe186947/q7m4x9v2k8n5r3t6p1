"""Deterministic temporal controls for C6 and future full-data ablations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

MODE_SPECS: dict[str, dict[str, Any]] = {
    "M128": {
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 68_234,
        "sequence_control": "ordered",
        "time_delta_control": "ignored",
    },
    "MW317": {
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 317,
        "transformer_layers": 1,
        "transformer_heads": 1,
        "expected_parameter_count": 167_459,
        "sequence_control": "ordered",
        "time_delta_control": "ignored",
    },
    "TCN128": {
        "temporal_encoder_name": "masked_tcn",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 167_435,
        "sequence_control": "ordered",
        "time_delta_control": "ignored",
    },
    "TCN128_SEQUENCE_SHUFFLED": {
        "temporal_encoder_name": "masked_tcn",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 167_435,
        "sequence_control": "native_unit_stable_shuffle",
        "time_delta_control": "ignored",
    },
    "MW381": {
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 381,
        "transformer_layers": 1,
        "transformer_heads": 1,
        "expected_parameter_count": 201_059,
        "sequence_control": "ordered",
        "time_delta_control": "ignored",
    },
    "TR128_REAL_DELTA": {
        "temporal_encoder_name": "small_transformer",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 200_843,
        "sequence_control": "ordered",
        "time_delta_control": "real",
    },
    "TR128_CONSTANT_DELTA": {
        "temporal_encoder_name": "small_transformer",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 200_843,
        "sequence_control": "ordered",
        "time_delta_control": "constant",
    },
    "TR128_DELTA_SHUFFLED": {
        "temporal_encoder_name": "small_transformer",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 200_843,
        "sequence_control": "ordered",
        "time_delta_control": "native_unit_stable_shuffle",
    },
    "TR128_SEQUENCE_SHUFFLED": {
        "temporal_encoder_name": "small_transformer",
        "hidden_dim": 128,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "expected_parameter_count": 200_843,
        "sequence_control": "native_unit_stable_shuffle",
        "time_delta_control": "real",
    },
}

CONTROLLED_PAIRS: dict[str, tuple[str, str]] = {
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

PARAMETER_MATCH_MAX_RELATIVE_DELTA = 0.005


@dataclass(frozen=True, slots=True)
class TemporalControlPlan:
    """Label-independent slot and timing transformation for one mode."""

    mode_id: str
    unit_keys: tuple[str, ...]
    slot_permutation: np.ndarray
    controlled_time_delta: np.ndarray
    audit: dict[str, Any]


def build_temporal_control_plan(
    *,
    mode_id: str,
    unit_keys: Sequence[str],
    observed_mask: np.ndarray,
    real_time_delta: np.ndarray,
    perturbation_seed: int,
    constant_delta_seconds: float,
) -> TemporalControlPlan:
    """Build a deterministic control without consulting labels or fold outcomes."""

    spec = _mode_spec(mode_id)
    keys, mask, deltas = _validated_temporal_inputs(
        unit_keys,
        observed_mask,
        real_time_delta,
    )
    if constant_delta_seconds <= 0.0 or not np.isfinite(constant_delta_seconds):
        raise ValueError("constant_delta_seconds must be finite and positive")
    slot_permutation = np.tile(
        np.arange(mask.shape[1], dtype=np.int64),
        (mask.shape[0], 1),
    )
    if spec["sequence_control"] == "native_unit_stable_shuffle":
        slot_permutation = _stable_valid_slot_permutations(
            keys,
            mask,
            seed=perturbation_seed,
            namespace="sequence",
        )
    controlled_delta = _controlled_time_delta(
        keys=keys,
        mask=mask,
        real_time_delta=deltas,
        control=str(spec["time_delta_control"]),
        seed=perturbation_seed,
        constant_delta_seconds=constant_delta_seconds,
    )
    valid_slots = int(mask.sum())
    slot_changed = int(((slot_permutation != np.arange(mask.shape[1])) & mask).sum())
    delta_changed = int(
        ((~np.isclose(controlled_delta, deltas, atol=1e-9, rtol=0.0)) & mask).sum()
    )
    audit = {
        "schema_version": "classification_v2.temporal_control_plan.v1",
        "mode_id": mode_id,
        "native_units": len(keys),
        "sequence_length": int(mask.shape[1]),
        "valid_slots": valid_slots,
        "sequence_control": str(spec["sequence_control"]),
        "time_delta_control": str(spec["time_delta_control"]),
        "perturbation_seed": int(perturbation_seed),
        "constant_delta_seconds": float(constant_delta_seconds),
        "feature_slots_changed": slot_changed,
        "time_delta_slots_changed": delta_changed,
        "rows_dropped": 0,
        "labels_read": 0,
        "duplicate_native_unit_keys": 0,
        "errors": [],
        "valid": True,
    }
    return TemporalControlPlan(
        mode_id=mode_id,
        unit_keys=keys,
        slot_permutation=slot_permutation,
        controlled_time_delta=controlled_delta,
        audit=audit,
    )


def apply_slot_permutation(
    values: np.ndarray,
    plan: TemporalControlPlan,
) -> np.ndarray:
    """Apply one plan to any aligned ``[native_unit, slot, ...]`` tensor."""

    array = np.asarray(values)
    expected = plan.slot_permutation.shape
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(
            f"aligned temporal values must start with shape {expected}, "
            f"received={array.shape}"
        )
    index = plan.slot_permutation
    for _ in range(array.ndim - 2):
        index = np.expand_dims(index, axis=-1)
    index = np.broadcast_to(index, array.shape)
    return np.take_along_axis(array, index, axis=1).copy()


def audit_time_delta_identifiability(
    *,
    unit_keys: Sequence[str],
    observed_mask: np.ndarray,
    real_time_delta: np.ndarray,
    perturbation_seed: int,
    constant_delta_seconds: float,
    minimum_changed_fraction: float,
) -> dict[str, Any]:
    """Determine whether constant and shuffled timing controls differ from real time."""

    if not 0.0 < minimum_changed_fraction <= 1.0:
        raise ValueError("minimum_changed_fraction must be in (0,1]")
    keys, mask, deltas = _validated_temporal_inputs(
        unit_keys,
        observed_mask,
        real_time_delta,
    )
    constant = _controlled_time_delta(
        keys=keys,
        mask=mask,
        real_time_delta=deltas,
        control="constant",
        seed=perturbation_seed,
        constant_delta_seconds=constant_delta_seconds,
    )
    shuffled = _controlled_time_delta(
        keys=keys,
        mask=mask,
        real_time_delta=deltas,
        control="native_unit_stable_shuffle",
        seed=perturbation_seed,
        constant_delta_seconds=constant_delta_seconds,
    )
    comparable = _post_first_valid_mask(mask)
    comparable_count = int(comparable.sum())
    constant_changed = int(
        ((~np.isclose(constant, deltas, atol=1e-9, rtol=0.0)) & comparable).sum()
    )
    shuffled_changed = int(
        ((~np.isclose(shuffled, deltas, atol=1e-9, rtol=0.0)) & comparable).sum()
    )
    constant_fraction = constant_changed / max(1, comparable_count)
    shuffled_fraction = shuffled_changed / max(1, comparable_count)
    variable_units = 0
    per_unit_mean: list[float] = []
    for row, valid in zip(deltas, comparable, strict=True):
        values = row[valid]
        if len(values):
            per_unit_mean.append(float(values.mean()))
            if len(np.unique(np.round(values, decimals=9))) > 1:
                variable_units += 1
    positive = deltas[comparable]
    unique_values = sorted(float(value) for value in np.unique(np.round(positive, 9)))
    constant_identifiable = constant_fraction >= minimum_changed_fraction
    shuffled_identifiable = shuffled_fraction >= minimum_changed_fraction
    warnings: list[str] = []
    if not constant_identifiable:
        warnings.append("real_and_constant_delta_are_not_distinguishable_enough")
    if not shuffled_identifiable:
        warnings.append("real_and_shuffled_delta_are_not_distinguishable_enough")
    return {
        "schema_version": "classification_v2.time_delta_identifiability.v1",
        "native_units": len(keys),
        "sequence_length": int(mask.shape[1]),
        "comparable_post_first_slots": comparable_count,
        "unique_positive_delta_values": unique_values,
        "positive_delta_min": float(positive.min()) if len(positive) else None,
        "positive_delta_max": float(positive.max()) if len(positive) else None,
        "positive_delta_std": float(positive.std()) if len(positive) else None,
        "within_sequence_variable_units": variable_units,
        "between_sequence_mean_std": (
            float(np.std(per_unit_mean)) if per_unit_mean else None
        ),
        "real_vs_constant_changed_slots": constant_changed,
        "real_vs_constant_changed_fraction": constant_fraction,
        "real_vs_shuffled_changed_slots": shuffled_changed,
        "real_vs_shuffled_changed_fraction": shuffled_fraction,
        "minimum_changed_fraction": float(minimum_changed_fraction),
        "constant_control_identifiable": constant_identifiable,
        "delta_alignment_control_identifiable": shuffled_identifiable,
        "full_real_timing_claim_identifiable": (
            constant_identifiable and shuffled_identifiable
        ),
        "errors": [],
        "warnings": warnings,
        "valid": True,
    }


def audit_timing_source_shortcut(
    *,
    source_types: Sequence[str],
    observed_mask: np.ndarray,
    real_time_delta: np.ndarray,
    rounding_decimals: int = 6,
    minimum_purity: float = 0.8,
    minimum_uplift: float = 0.1,
) -> dict[str, Any]:
    """Estimate whether timing alone identifies source without behavior labels."""

    sources = tuple(str(value) for value in source_types)
    _, mask, deltas = _validated_temporal_inputs(
        [f"unit-{index}" for index in range(len(sources))],
        observed_mask,
        real_time_delta,
    )
    if len(sources) != len(mask):
        raise ValueError("source_types must match temporal rows")
    if not 0.0 <= minimum_purity <= 1.0 or not 0.0 <= minimum_uplift <= 1.0:
        raise ValueError("source shortcut thresholds must be in [0,1]")
    source_counts: dict[str, int] = {}
    signature_sources: dict[tuple[float, ...], dict[str, int]] = {}
    feature_rows: list[np.ndarray] = []
    for source, row, valid in zip(sources, deltas, mask, strict=True):
        source_counts[source] = source_counts.get(source, 0) + 1
        signature = tuple(float(value) for value in np.round(row[valid], rounding_decimals))
        bucket = signature_sources.setdefault(signature, {})
        bucket[source] = bucket.get(source, 0) + 1
        feature_rows.append(np.concatenate([row, valid.astype(np.float32)]))
    total = max(1, len(sources))
    baseline_purity = max(source_counts.values(), default=0) / total
    supported_buckets = [
        counts
        for counts in signature_sources.values()
        if sum(counts.values()) >= 2
    ]
    supported_units = sum(sum(counts.values()) for counts in supported_buckets)
    signature_correct = sum(max(counts.values()) for counts in supported_buckets)
    signature_purity = (
        signature_correct / supported_units if supported_units else None
    )
    purity_uplift = (
        signature_purity - baseline_purity
        if signature_purity is not None
        else None
    )
    matrix = np.stack(feature_rows) if feature_rows else np.empty((0, 0))
    loo_correct = 0
    loo_estimable = 0
    for index, (source, value) in enumerate(zip(sources, matrix, strict=True)):
        centroids: dict[str, np.ndarray] = {}
        for candidate in sorted(source_counts):
            positions = [
                position
                for position, observed_source in enumerate(sources)
                if position != index and observed_source == candidate
            ]
            if positions:
                centroids[candidate] = matrix[positions].mean(axis=0)
        if len(centroids) < 2:
            continue
        prediction = min(
            centroids,
            key=lambda candidate: (
                float(np.square(value - centroids[candidate]).sum()),
                candidate,
            ),
        )
        loo_estimable += 1
        loo_correct += int(prediction == source)
    loo_accuracy = loo_correct / loo_estimable if loo_estimable else None
    loo_uplift = (
        loo_accuracy - baseline_purity if loo_accuracy is not None else None
    )
    signature_risk = (
        signature_purity is not None
        and purity_uplift is not None
        and signature_purity >= minimum_purity
        and purity_uplift >= minimum_uplift
    )
    loo_risk = (
        loo_accuracy is not None
        and loo_uplift is not None
        and loo_accuracy >= minimum_purity
        and loo_uplift >= minimum_uplift
    )
    risk = (
        len(source_counts) > 1
        and (signature_risk or loo_risk)
    )
    return {
        "schema_version": "classification_v2.timing_source_shortcut.v1",
        "native_units": len(sources),
        "source_counts": source_counts,
        "unique_timing_signatures": len(signature_sources),
        "repeated_signature_units": supported_units,
        "majority_source_baseline_purity": baseline_purity,
        "timing_signature_source_purity": signature_purity,
        "timing_signature_purity_uplift": purity_uplift,
        "loo_nearest_centroid_estimable_units": loo_estimable,
        "loo_nearest_centroid_accuracy": loo_accuracy,
        "loo_nearest_centroid_uplift": loo_uplift,
        "timing_source_shortcut_risk": risk,
        "errors": [],
        "warnings": ["timing_signature_nearly_identifies_source"] if risk else [],
        "valid": True,
    }


def build_temporal_conclusion_readiness(
    *,
    delta_audit: dict[str, Any],
    source_audit: dict[str, Any],
    short_gate_passed: bool,
    paired_native_evidence_passed: bool,
    per_source_evidence_passed: bool,
    seed_robustness_passed: bool,
    mixed_reviewed_lineage: bool,
) -> dict[str, Any]:
    """Fail closed on scientific claims that need evidence beyond a code smoke."""

    correctness = bool(delta_audit.get("valid")) and bool(source_audit.get("valid"))
    common = (
        correctness
        and short_gate_passed
        and paired_native_evidence_passed
        and seed_robustness_passed
    )
    order_claim = common
    timing_claim = (
        common
        and bool(delta_audit.get("full_real_timing_claim_identifiable"))
        and not bool(source_audit.get("timing_source_shortcut_risk"))
        and per_source_evidence_passed
    )
    order_full_data_promotion = (
        order_claim and per_source_evidence_passed and mixed_reviewed_lineage
    )
    timing_full_data_promotion = timing_claim and mixed_reviewed_lineage
    full_data_promotion = (
        order_full_data_promotion or timing_full_data_promotion
    )
    missing: list[str] = []
    checks = {
        "correctness_audits": correctness,
        "short_gate": short_gate_passed,
        "paired_native_evidence": paired_native_evidence_passed,
        "per_source_evidence": per_source_evidence_passed,
        "seed_robustness": seed_robustness_passed,
        "timing_identifiability": bool(
            delta_audit.get("full_real_timing_claim_identifiable")
        ),
        "timing_source_shortcut_absent": not bool(
            source_audit.get("timing_source_shortcut_risk")
        ),
        "mixed_reviewed_lineage": mixed_reviewed_lineage,
    }
    missing.extend(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": "classification_v2.temporal_conclusion_readiness.v1",
        "checks": checks,
        "temporal_order_claim_allowed": order_claim,
        "real_timing_claim_allowed": timing_claim,
        "order_full_data_promotion_allowed": order_full_data_promotion,
        "timing_full_data_promotion_allowed": timing_full_data_promotion,
        "full_data_base_promotion_allowed": full_data_promotion,
        "legacy_only_result_sets_full_data_base": False,
        "missing_requirements": missing,
        "errors": [],
        "valid": correctness,
    }


def parameter_control_errors(parameter_counts: dict[str, int]) -> list[str]:
    """Validate all declared same-capacity controls."""

    errors: list[str] = []
    for pair_id, (candidate, baseline) in CONTROLLED_PAIRS.items():
        if candidate not in parameter_counts or baseline not in parameter_counts:
            errors.append(f"{pair_id}:missing_parameter_count")
            continue
        left = int(parameter_counts[candidate])
        right = int(parameter_counts[baseline])
        relative = abs(left - right) / max(left, right)
        if relative > PARAMETER_MATCH_MAX_RELATIVE_DELTA:
            errors.append(f"{pair_id}:parameter_relative_delta={relative:.8f}")
    return errors


def _controlled_time_delta(
    *,
    keys: tuple[str, ...],
    mask: np.ndarray,
    real_time_delta: np.ndarray,
    control: str,
    seed: int,
    constant_delta_seconds: float,
) -> np.ndarray:
    output = real_time_delta.copy()
    if control in {"ignored", "real"}:
        return output
    if control == "constant":
        output.fill(0.0)
        for row_index, valid in enumerate(mask):
            positions = np.flatnonzero(valid)
            if len(positions) > 1:
                output[row_index, positions[1:]] = constant_delta_seconds
        return output
    if control != "native_unit_stable_shuffle":
        raise ValueError(f"unsupported time_delta control={control}")
    for row_index, (key, valid) in enumerate(zip(keys, mask, strict=True)):
        positions = np.flatnonzero(valid)
        if len(positions) <= 2:
            continue
        tail = positions[1:]
        values = output[row_index, tail].copy()
        permutation = _stable_permutation(
            len(tail),
            key=key,
            seed=seed,
            namespace="time_delta",
        )
        shuffled = values[permutation]
        if np.array_equal(shuffled, values) and len(np.unique(values)) > 1:
            shuffled = np.roll(values, 1)
        output[row_index, tail] = shuffled
    return output


def _stable_valid_slot_permutations(
    keys: tuple[str, ...],
    mask: np.ndarray,
    *,
    seed: int,
    namespace: str,
) -> np.ndarray:
    output = np.tile(
        np.arange(mask.shape[1], dtype=np.int64),
        (mask.shape[0], 1),
    )
    for row_index, (key, valid) in enumerate(zip(keys, mask, strict=True)):
        positions = np.flatnonzero(valid)
        if len(positions) <= 1:
            continue
        permutation = _stable_permutation(
            len(positions),
            key=key,
            seed=seed,
            namespace=namespace,
        )
        if np.array_equal(permutation, np.arange(len(positions))):
            permutation = np.roll(permutation, 1)
        output[row_index, positions] = positions[permutation]
    return output


def _stable_permutation(
    length: int,
    *,
    key: str,
    seed: int,
    namespace: str,
) -> np.ndarray:
    payload = f"{seed}\0{namespace}\0{key}".encode()
    digest = hashlib.sha256(payload).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(local_seed).permutation(length)


def _post_first_valid_mask(mask: np.ndarray) -> np.ndarray:
    output = mask.copy()
    for row_index, valid in enumerate(mask):
        positions = np.flatnonzero(valid)
        if len(positions):
            output[row_index, positions[0]] = False
    return output


def _validated_temporal_inputs(
    unit_keys: Sequence[str],
    observed_mask: np.ndarray,
    real_time_delta: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    keys = tuple(str(value) for value in unit_keys)
    if not keys or any(not value for value in keys):
        raise ValueError("native-unit keys must be nonblank")
    if len(keys) != len(set(keys)):
        raise ValueError("native-unit keys must be unique")
    mask = np.asarray(observed_mask)
    deltas = np.asarray(real_time_delta, dtype=np.float32)
    if mask.ndim != 2 or deltas.shape != mask.shape or len(keys) != len(mask):
        raise ValueError("keys, observed_mask and real_time_delta shapes differ")
    if mask.dtype != np.bool_:
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("observed_mask must be binary")
        mask = mask.astype(np.bool_)
    if not mask.any(axis=1).all():
        raise ValueError("each native unit requires an observed slot")
    if not np.isfinite(deltas[mask]).all() or (deltas[mask] < 0.0).any():
        raise ValueError("observed time_delta must be finite and non-negative")
    if not np.allclose(deltas[~mask], 0.0, atol=1e-9, rtol=0.0):
        raise ValueError("time_delta outside observed slots must be zero")
    return keys, mask.copy(), deltas.copy()


def _mode_spec(mode_id: str) -> dict[str, Any]:
    try:
        return MODE_SPECS[mode_id]
    except KeyError as error:
        raise ValueError(f"unsupported temporal control mode={mode_id}") from error


__all__ = [
    "CONTROLLED_PAIRS",
    "MODE_SPECS",
    "PARAMETER_MATCH_MAX_RELATIVE_DELTA",
    "TemporalControlPlan",
    "apply_slot_permutation",
    "audit_time_delta_identifiability",
    "audit_timing_source_shortcut",
    "build_temporal_conclusion_readiness",
    "build_temporal_control_plan",
    "parameter_control_errors",
]

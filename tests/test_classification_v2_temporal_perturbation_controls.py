from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.training.legacy_development_c6_temporal_controls import (
    build_c6_temporal_control_model,
    data_c6_temporal_control_preflight,
    derive_c6_temporal_control,
    load_c6_temporal_control_config,
    static_c6_temporal_control_preflight,
    synthetic_c6_temporal_control_preflight,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM,
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.temporal_perturbation_controls import (
    CONTROLLED_PAIRS,
    MODE_SPECS,
    apply_slot_permutation,
    audit_time_delta_identifiability,
    audit_timing_source_shortcut,
    build_temporal_conclusion_readiness,
    build_temporal_control_plan,
)

CONFIG_PATH = Path(
    "configs/classification_v2/"
    "legacy_development_c6_temporal_controls_code_ready_v1.json"
)


def test_matrix_contains_all_requested_controls_and_capacity_matches() -> None:
    assert set(MODE_SPECS) == {
        "M128",
        "MW317",
        "TCN128",
        "TCN128_SEQUENCE_SHUFFLED",
        "MW381",
        "TR128_REAL_DELTA",
        "TR128_CONSTANT_DELTA",
        "TR128_DELTA_SHUFFLED",
        "TR128_SEQUENCE_SHUFFLED",
    }
    assert CONTROLLED_PAIRS["tcn_order"] == (
        "TCN128",
        "TCN128_SEQUENCE_SHUFFLED",
    )
    assert CONTROLLED_PAIRS["transformer_timing_constant"] == (
        "TR128_REAL_DELTA",
        "TR128_CONSTANT_DELTA",
    )
    assert CONTROLLED_PAIRS["transformer_timing_alignment"] == (
        "TR128_REAL_DELTA",
        "TR128_DELTA_SHUFFLED",
    )
    assert CONTROLLED_PAIRS["transformer_order"] == (
        "TR128_REAL_DELTA",
        "TR128_SEQUENCE_SHUFFLED",
    )


def test_sequence_shuffle_is_deterministic_and_shared_by_modalities() -> None:
    keys, mask, deltas = _temporal_inputs()
    actor = np.arange(4 * 6 * 2).reshape(4, 6, 2)
    geometry = actor + 10_000
    first = build_temporal_control_plan(
        mode_id="TCN128_SEQUENCE_SHUFFLED",
        unit_keys=keys,
        observed_mask=mask,
        real_time_delta=deltas,
        perturbation_seed=71,
        constant_delta_seconds=0.2,
    )
    second = build_temporal_control_plan(
        mode_id="TCN128_SEQUENCE_SHUFFLED",
        unit_keys=keys,
        observed_mask=mask,
        real_time_delta=deltas,
        perturbation_seed=71,
        constant_delta_seconds=0.2,
    )

    shuffled_actor = apply_slot_permutation(actor, first)
    shuffled_geometry = apply_slot_permutation(geometry, first)

    assert np.array_equal(first.slot_permutation, second.slot_permutation)
    assert not np.array_equal(shuffled_actor, actor)
    assert np.array_equal(shuffled_geometry - shuffled_actor, np.full_like(actor, 10_000))
    assert np.array_equal(np.sort(shuffled_actor[:, :, 0]), np.sort(actor[:, :, 0]))
    assert first.audit["rows_dropped"] == 0
    assert first.audit["labels_read"] == 0


def test_timing_controls_change_only_declared_timing_values() -> None:
    keys, mask, deltas = _temporal_inputs()
    real = build_temporal_control_plan(
        mode_id="TR128_REAL_DELTA",
        unit_keys=keys,
        observed_mask=mask,
        real_time_delta=deltas,
        perturbation_seed=19,
        constant_delta_seconds=0.25,
    )
    constant = build_temporal_control_plan(
        mode_id="TR128_CONSTANT_DELTA",
        unit_keys=keys,
        observed_mask=mask,
        real_time_delta=deltas,
        perturbation_seed=19,
        constant_delta_seconds=0.25,
    )
    shuffled = build_temporal_control_plan(
        mode_id="TR128_DELTA_SHUFFLED",
        unit_keys=keys,
        observed_mask=mask,
        real_time_delta=deltas,
        perturbation_seed=19,
        constant_delta_seconds=0.25,
    )

    identity = np.tile(np.arange(6), (4, 1))
    assert np.array_equal(real.slot_permutation, identity)
    assert np.array_equal(constant.slot_permutation, identity)
    assert np.array_equal(shuffled.slot_permutation, identity)
    assert np.array_equal(real.controlled_time_delta, deltas)
    assert np.allclose(constant.controlled_time_delta[:, 0], 0.0)
    assert np.allclose(constant.controlled_time_delta[:, 1:], 0.25)
    assert not np.array_equal(shuffled.controlled_time_delta, deltas)
    for before, after in zip(deltas, shuffled.controlled_time_delta, strict=True):
        assert np.allclose(np.sort(before[1:]), np.sort(after[1:]))


def test_delta_identifiability_distinguishes_uniform_and_variable_timing() -> None:
    keys, mask, variable = _temporal_inputs()
    variable_audit = audit_time_delta_identifiability(
        unit_keys=keys,
        observed_mask=mask,
        real_time_delta=variable,
        perturbation_seed=23,
        constant_delta_seconds=0.2,
        minimum_changed_fraction=0.05,
    )
    uniform = np.zeros_like(variable)
    uniform[:, 1:] = 0.2
    uniform_audit = audit_time_delta_identifiability(
        unit_keys=keys,
        observed_mask=mask,
        real_time_delta=uniform,
        perturbation_seed=23,
        constant_delta_seconds=0.2,
        minimum_changed_fraction=0.05,
    )

    assert variable_audit["full_real_timing_claim_identifiable"] is True
    assert variable_audit["real_vs_shuffled_changed_slots"] > 0
    assert uniform_audit["full_real_timing_claim_identifiable"] is False
    assert uniform_audit["real_vs_constant_changed_slots"] == 0
    assert uniform_audit["real_vs_shuffled_changed_slots"] == 0


def test_timing_source_shortcut_and_claim_gate_fail_closed() -> None:
    mask = np.ones((6, 6), dtype=np.bool_)
    deltas = np.zeros((6, 6), dtype=np.float32)
    deltas[:3, 1:] = 0.1
    deltas[3:, 1:] = 0.3
    source_audit = audit_timing_source_shortcut(
        source_types=["legacy"] * 3 + ["cvat"] * 3,
        observed_mask=mask,
        real_time_delta=deltas,
    )
    delta_audit = audit_time_delta_identifiability(
        unit_keys=[f"unit-{index}" for index in range(6)],
        observed_mask=mask,
        real_time_delta=deltas,
        perturbation_seed=31,
        constant_delta_seconds=0.2,
        minimum_changed_fraction=0.05,
    )
    readiness = build_temporal_conclusion_readiness(
        delta_audit=delta_audit,
        source_audit=source_audit,
        short_gate_passed=True,
        paired_native_evidence_passed=True,
        per_source_evidence_passed=True,
        seed_robustness_passed=True,
        mixed_reviewed_lineage=True,
    )

    assert source_audit["timing_source_shortcut_risk"] is True
    assert readiness["temporal_order_claim_allowed"] is True
    assert readiness["real_timing_claim_allowed"] is False
    assert readiness["legacy_only_result_sets_full_data_base"] is False


@pytest.mark.parametrize(
    "mode_id",
    [
        "M128",
        "TCN128",
        "TCN128_SEQUENCE_SHUFFLED",
        "TR128_REAL_DELTA",
        "TR128_CONSTANT_DELTA",
        "TR128_DELTA_SHUFFLED",
        "TR128_SEQUENCE_SHUFFLED",
    ],
)
def test_legacy_c6_derivation_preserves_native_units(
    tmp_path: Path,
    mode_id: str,
) -> None:
    base = _base_view(tmp_path)

    derived = derive_c6_temporal_control(
        base,
        mode_id,
        perturbation_seed=41,
        constant_delta_seconds=0.2,
    )

    assert len(derived.view.windows) == len(base.windows)
    assert derived.view.windows["behavior_label"].tolist() == ["lying"] * 5
    assert not derived.view.windows["temporal_unit_key"].duplicated().any()
    assert len(derived.slot_manifest) == len(base.windows) * 6
    assert not derived.slot_manifest[
        ["temporal_unit_key", "slot_index"]
    ].duplicated().any()
    assert derived.audit["rows_dropped"] == 0
    assert derived.audit["labels_changed"] == 0


def test_code_ready_config_static_and_synthetic_gates_do_not_read_data() -> None:
    config = load_c6_temporal_control_config(CONFIG_PATH)

    static = static_c6_temporal_control_preflight(config)
    synthetic = synthetic_c6_temporal_control_preflight(config)

    assert config.payload["execution"]["data_run_authorized"] is False
    assert static["valid"] is True
    assert synthetic["valid"] is True
    assert static["project_data_rows_read"] == 0
    assert synthetic["project_data_rows_read"] == 0
    assert static["full_development_authorized"] is False
    assert synthetic["conclusion_readiness"][
        "full_data_base_promotion_allowed"
    ] is False
    with pytest.raises(PermissionError, match="fail-closed"):
        data_c6_temporal_control_preflight(config)


def test_full_data_claim_gate_can_pass_only_with_all_evidence() -> None:
    readiness = build_temporal_conclusion_readiness(
        delta_audit={
            "valid": True,
            "full_real_timing_claim_identifiable": True,
        },
        source_audit={
            "valid": True,
            "timing_source_shortcut_risk": False,
        },
        short_gate_passed=True,
        paired_native_evidence_passed=True,
        per_source_evidence_passed=True,
        seed_robustness_passed=True,
        mixed_reviewed_lineage=True,
    )

    assert readiness["temporal_order_claim_allowed"] is True
    assert readiness["real_timing_claim_allowed"] is True
    assert readiness["full_data_base_promotion_allowed"] is True


def test_transformer_control_forward_shapes_are_equal() -> None:
    config = load_c6_temporal_control_config(CONFIG_PATH)

    features = torch.randn(3, 6, FEATURE_DIM)
    mask = torch.ones(3, 6)
    timing = torch.full((3, 6), 0.2)
    timing[:, 0] = 0.0
    counts: set[int] = set()
    for mode_id in (
        "TR128_REAL_DELTA",
        "TR128_CONSTANT_DELTA",
        "TR128_DELTA_SHUFFLED",
        "TR128_SEQUENCE_SHUFFLED",
    ):
        model = build_c6_temporal_control_model(config, mode_id, dropout=0.0)
        logits = model(features, mask, time_delta=timing)
        counts.add(sum(parameter.numel() for parameter in model.parameters()))
        assert list(logits.shape) == [3, 10]
    assert len(counts) == 1


def _temporal_inputs() -> tuple[list[str], np.ndarray, np.ndarray]:
    keys = [f"unit-{index}" for index in range(4)]
    mask = np.ones((4, 6), dtype=np.bool_)
    deltas = np.array(
        [
            [0.0, 0.10, 0.20, 0.30, 0.40, 0.50],
            [0.0, 0.11, 0.22, 0.33, 0.44, 0.55],
            [0.0, 0.12, 0.24, 0.36, 0.48, 0.60],
            [0.0, 0.13, 0.26, 0.39, 0.52, 0.65],
        ],
        dtype=np.float32,
    )
    return keys, mask, deltas


def _base_view(tmp_path: Path) -> LegacyL5CachedFeatureView:
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.zeros((80, FEATURE_DIM), dtype=np.float32))
    rows = 5
    windows = pd.DataFrame(
        {
            "window_id": [f"window-{index}" for index in range(rows)],
            "temporal_unit_key": [f"unit-{index}" for index in range(rows)],
            "recording_group_id": ["recording-0"] * rows,
            "video_key": ["video-0"] * rows,
            "source_type": ["legacy_recovered"] * rows,
            "dataset_id": ["legacy_recovered_16f"] * rows,
            "behavior_label": ["lying"] * rows,
            "oof_fold_id": ["fold-0"] * rows,
            "l5_role": ["train"] * rows,
        }
    )
    time_delta = np.full((rows, 16), 0.2, dtype=np.float32)
    time_delta[:, 0] = 0.0
    return LegacyL5CachedFeatureView(
        feature_tensor_path=feature_path,
        feature_tensor_sha256="0" * 64,
        control_id="V1",
        temporal_view_name="legacy_t16_centered_matched_observed_time",
        sequence_length=16,
        windows=windows,
        fold_manifest=pd.DataFrame(),
        feature_rows=np.arange(rows * 16).reshape(rows, 16),
        observed_mask=np.ones((rows, 16), dtype=np.bool_),
        time_delta=time_delta,
        targets=np.full(rows, 5, dtype=np.int64),
        sample_weights=np.ones(rows, dtype=np.float64),
        audit={"valid": True},
    )

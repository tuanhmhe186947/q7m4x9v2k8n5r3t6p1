from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM,
    LegacyL5CachedFeatureClassifier,
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderSelection,
)
from pig_behavior.classification_v2.training.legacy_development_temporal_base_selection import (
    CONFIG_SCHEMA_V2,
    CONTROLLED_PAIRS,
    MODE_SPECS,
    TemporalBaseSelectionConfig,
    _validate_config,
    build_training_adapter,
    derive_temporal_base_view,
)
from pig_behavior.classification_v2.training.legacy_development_temporal_sampling import (
    TemporalSamplingSource,
)


def _base_view(tmp_path: Path, rows: int = 3) -> LegacyL5CachedFeatureView:
    feature_path = tmp_path / f"features-{rows}.npy"
    np.save(feature_path, np.zeros((1, FEATURE_DIM), dtype=np.float32))
    roles = ["train"] * rows
    if rows >= 245:
        roles[-245:] = ["validation"] * 245
    windows = pd.DataFrame(
        {
            "window_id": [f"window-{index:04d}" for index in range(rows)],
            "temporal_unit_key": [f"unit-{index:04d}" for index in range(rows)],
            "recording_group_id": [f"recording-{index // 20}" for index in range(rows)],
            "video_key": [f"video-{index // 10}" for index in range(rows)],
            "source_type": ["legacy_recovered"] * rows,
            "dataset_id": ["legacy_recovered_16f"] * rows,
            "behavior_label": ["lying"] * rows,
            "oof_fold_id": ["fold-0"] * rows,
            "l5_role": roles,
        }
    )
    feature_rows = np.arange(rows * 16, dtype=np.int64).reshape(rows, 16)
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
        feature_rows=feature_rows,
        observed_mask=np.ones((rows, 16), dtype=np.bool_),
        time_delta=time_delta,
        targets=np.full(rows, 5, dtype=np.int64),
        sample_weights=np.ones(rows, dtype=np.float64),
        audit={"valid": True},
    )


@pytest.mark.parametrize("mode_id", list(MODE_SPECS))
def test_exact_offsets_preserve_one_sequence_per_native(
    tmp_path: Path,
    mode_id: str,
) -> None:
    base = _base_view(tmp_path)

    result = derive_temporal_base_view(base, mode_id)

    offsets = MODE_SPECS[mode_id]["native_frame_offsets"]
    assert result.audit["native_frame_offsets"] == offsets
    assert result.view.feature_rows.tolist() == base.feature_rows[:, offsets].tolist()
    assert len(result.view.windows) == len(base.windows)
    assert result.audit["rows_dropped"] == 0
    assert result.audit["labels_changed"] == 0
    assert result.audit["one_sequence_per_native_unit"] is True
    assert not result.view.windows["temporal_unit_key"].duplicated().any()
    assert not result.slot_manifest[["window_id", "slot_index"]].duplicated().any()


def test_declared_parameter_counts_and_capacity_controls() -> None:
    observed = {}
    for mode_id, spec in MODE_SPECS.items():
        model = _model(mode_id)
        observed[mode_id] = sum(parameter.numel() for parameter in model.parameters())
        assert observed[mode_id] == spec["expected_parameter_count"]

    for pair_id in ("ordered_tcn", "timed_transformer"):
        candidate, baseline = CONTROLLED_PAIRS[pair_id]
        relative = abs(observed[candidate] - observed[baseline]) / max(
            observed[candidate],
            observed[baseline],
        )
        assert relative <= 0.005


def test_pooling_modes_are_order_invariant_and_ignore_elapsed_time() -> None:
    features, observed, timing = _sequence_probe()
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    altered_timing = timing * 3.0
    for mode_id in ("M128", "A128", "MW317", "MW381"):
        model = _model(mode_id).eval()
        with torch.inference_mode():
            original = model(features, observed, time_delta=timing)
            permuted = model(
                features[:, permutation],
                observed[:, permutation],
                time_delta=timing[:, permutation],
            )
            retimed = model(features, observed, time_delta=altered_timing)
        assert torch.allclose(original, permuted, atol=1e-6, rtol=1e-6)
        assert torch.allclose(original, retimed, atol=1e-6, rtol=1e-6)


def test_tcn_uses_order_but_not_elapsed_time() -> None:
    features, observed, timing = _sequence_probe()
    model = _model("TCN128").eval()

    with torch.inference_mode():
        original = model(features, observed, time_delta=timing)
        reversed_order = model(
            features.flip(1),
            observed.flip(1),
            time_delta=timing.flip(1),
        )
        retimed = model(features, observed, time_delta=timing * 3.0)

    assert not torch.allclose(original, reversed_order, atol=1e-6, rtol=1e-6)
    assert torch.allclose(original, retimed, atol=1e-6, rtol=1e-6)


def test_transformer_uses_order_and_real_elapsed_time() -> None:
    features, observed, timing = _sequence_probe()
    model = _model("TR128").eval()

    with torch.inference_mode():
        original = model(features, observed, time_delta=timing)
        reversed_order = model(
            features.flip(1),
            observed.flip(1),
            time_delta=timing,
        )
        zero_timing = model(
            features,
            observed,
            time_delta=torch.zeros_like(timing),
        )

    assert not torch.allclose(original, reversed_order, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(original, zero_timing, atol=1e-6, rtol=1e-6)


def test_adapter_rejects_outer_holdout_access(tmp_path: Path) -> None:
    base = _base_view(tmp_path, rows=325)
    manifest = base.windows.copy(deep=True)
    selection = TemporalLadderSelection(
        manifest=manifest,
        train_positions=np.arange(80, dtype=np.int64),
        validation_positions=np.arange(80, 325, dtype=np.int64),
        audit={"outer_holdout_rows": 0},
    )
    source = TemporalSamplingSource(
        base_view=base,
        selection=selection,
        parent_audit={"valid": True},
        source_config_sha256="1" * 64,
    )
    config = _short_config(tmp_path)
    derived = derive_temporal_base_view(base, "M128")

    _, adapted = build_training_adapter(config, source, derived)

    assert len(adapted.train_positions) == 80
    assert len(adapted.validation_positions) == 245
    assert adapted.audit["outer_holdout_rows"] == 0
    unsafe = TemporalSamplingSource(
        base_view=base,
        selection=TemporalLadderSelection(
            manifest=manifest,
            train_positions=selection.train_positions,
            validation_positions=selection.validation_positions,
            audit={"outer_holdout_rows": 1},
        ),
        parent_audit={"valid": True},
        source_config_sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="exposes outer holdout"):
        build_training_adapter(config, unsafe, derived)


def test_rebuild_config_accepts_hash_bound_prepared_source() -> None:
    path = Path(
        "configs/classification_v2/"
        "legacy_development_temporal_base_selection_short_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = CONFIG_SCHEMA_V2
    payload["prepared_source"] = payload.pop("source_ladder_config")
    payload["model_implementation"] = {
        "path": "src/pig_behavior/classification_v2/models/temporal_encoders.py",
        "sha256": "a" * 64,
    }
    payload["execution"].update(
        {
            "data_run_authorized": True,
            "clean_lineage_handoff_id": "legacy_16f_rebuild_test",
            "full_oof_authorized": False,
        }
    )

    _validate_config(payload)


def _model(mode_id: str) -> LegacyL5CachedFeatureClassifier:
    torch.manual_seed(17)
    spec = MODE_SPECS[mode_id]
    return LegacyL5CachedFeatureClassifier(
        temporal_encoder_name=str(spec["temporal_encoder_name"]),
        hidden_dim=int(spec["hidden_dim"]),
        dropout=0.0,
        transformer_layers=int(spec["transformer_layers"]),
        transformer_heads=int(spec["transformer_heads"]),
    )


def _sequence_probe() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    features = torch.randn(2, 6, FEATURE_DIM, generator=generator)
    observed = torch.ones(2, 6)
    timing = torch.full((2, 6), 0.2)
    timing[:, 0] = 0.0
    return features, observed, timing


def _short_config(tmp_path: Path) -> TemporalBaseSelectionConfig:
    return TemporalBaseSelectionConfig(
        path=tmp_path / "config.json",
        repo_root=tmp_path,
        payload={
            "training_scope": "short_repeat_gate",
            "model_common": {
                "architecture": "cached_frame_feature_temporal_classifier_v1",
                "feature_control_id": "V1",
                "backbone_name": "resnet18",
                "input_resolution": 224,
                "dropout": 0.1,
            },
            "optimization": {
                "epochs": 3,
                "batch_size": 32,
                "seed": 20260717,
            },
        },
    )

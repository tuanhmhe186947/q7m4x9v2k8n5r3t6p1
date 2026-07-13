from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    DatasetConfig,
    ExecutionConfig,
    LossConfig,
    ModelConfig,
    OptimizationConfig,
    resolve_temporal_view_manifest,
    training_config_to_jsonable,
)
from pig_behavior.classification_v2.training.data_module import (
    StrictTrainingDataModule,
    _strict_model_inputs,
    validate_model_inputs,
)
from pig_behavior.classification_v2.training.temporal_view_loader import (
    TemporalViewTensors,
    load_temporal_view_tensors,
)


def test_loader_preserves_full_window_universe_and_real_delta(
    tmp_path: Path,
) -> None:
    path = _write_manifest(tmp_path, ["window-0", "window-2"])

    tensors = load_temporal_view_tensors(
        path,
        expected_window_ids=["window-0", "window-1", "window-2"],
        selected_mask=np.array([True, False, True]),
        expected_view_name="fixed6_observed_time",
    )

    assert tensors.time_delta.shape == (3, 6)
    np.testing.assert_allclose(
        tensors.time_delta[0],
        np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2]),
    )
    assert np.isnan(tensors.time_delta[1]).all()
    assert not tensors.observed_mask[1].any()
    assert tensors.audit["selected_window_rows"] == 2
    assert tensors.audit["unselected_rows_preserved"] == 1
    assert tensors.audit["errors"] == []


def test_loader_retains_explicit_observed_slot_without_timing(
    tmp_path: Path,
) -> None:
    frame = _manifest(["window-0"])
    frame.loc[2, "timing_valid_mask"] = False
    frame.loc[2, "time_delta"] = np.nan
    frame.loc[3, "time_delta"] = 0.4
    path = tmp_path / "timing_gap.csv"
    frame.to_csv(path, index=False)

    tensors = load_temporal_view_tensors(
        path,
        expected_window_ids=["window-0"],
        selected_mask=np.array([True]),
        expected_view_name="fixed6_observed_time",
    )

    assert tensors.observed_mask[0, 2]
    assert not tensors.timing_valid_mask[0, 2]
    assert np.isnan(tensors.time_delta[0, 2])
    assert tensors.audit["observed_without_timing_slots"] == 1


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_slot", "slot row mismatch"),
        ("reordered_slot", "slot order mismatch"),
        ("duplicate_key", "slot_key is blank or duplicated"),
        ("wrong_item_order", "item order mismatch"),
    ],
)
def test_loader_rejects_slot_identity_drift(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    frame = _manifest(["window-0"])
    if mutation == "missing_slot":
        frame = frame.iloc[:-1].copy()
    elif mutation == "reordered_slot":
        frame = frame.iloc[[1, 0, 2, 3, 4, 5]].copy()
    elif mutation == "duplicate_key":
        frame.loc[1, "slot_key"] = frame.loc[0, "slot_key"]
    elif mutation == "wrong_item_order":
        frame.loc[0, "item_order"] = 1
    path = tmp_path / f"{mutation}.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=error):
        load_temporal_view_tensors(
            path,
            expected_window_ids=["window-0"],
            selected_mask=np.array([True]),
            expected_view_name="fixed6_observed_time",
        )


@pytest.mark.parametrize(
    ("row", "value", "error"),
    [
        (0, 0.1, "first timing-valid delta is not zero"),
        (2, 0.0, "later timing-valid delta is not positive"),
        (3, -0.1, "negative timing-valid"),
    ],
)
def test_loader_rejects_invalid_observed_delta(
    tmp_path: Path,
    row: int,
    value: float,
    error: str,
) -> None:
    frame = _manifest(["window-0"])
    frame.loc[row, "time_delta"] = value
    path = tmp_path / f"invalid_delta_{row}.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=error):
        load_temporal_view_tensors(
            path,
            expected_window_ids=["window-0"],
            selected_mask=np.array([True]),
            expected_view_name="fixed6_observed_time",
        )


def test_loader_rejects_delta_outside_timing_mask(tmp_path: Path) -> None:
    frame = _manifest(["window-0"])
    frame.loc[2, "timing_valid_mask"] = False
    path = tmp_path / "delta_outside_mask.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="outside timing_valid_mask"):
        load_temporal_view_tensors(
            path,
            expected_window_ids=["window-0"],
            selected_mask=np.array([True]),
            expected_view_name="fixed6_observed_time",
        )


def test_strict_model_inputs_expose_timing_without_metadata() -> None:
    delta = torch.tensor([[0.0, 0.2, 0.2, 0.2, 0.2, 0.2]])
    mask = torch.ones(1, 6)
    raw = {
        "image": torch.zeros(1, 6, 3, 8, 8),
        "image_length_mask": mask,
        "image_observed_mask": mask,
        "image_time_delta": delta,
        "spatial_features": {},
        "spatial_length_mask": mask,
        "spatial_observed_mask": mask,
        "spatial_time_delta": delta,
        "interaction_context_features": torch.zeros(1, 2),
        "interaction_context_available_mask": torch.ones(1),
        "visual_context_image": torch.zeros(1, 6, 3, 8, 8),
        "visual_context_length_mask": mask,
        "visual_context_observed_mask": mask,
        "visual_context_time_delta": delta,
    }

    model_inputs = _strict_model_inputs(raw)
    validate_model_inputs(model_inputs)

    assert model_inputs["image_time_delta"] is delta
    assert model_inputs["spatial_time_delta"] is delta
    assert model_inputs["visual_context_time_delta"] is delta
    assert "timing_valid_mask" not in model_inputs


def test_data_module_injects_timing_and_rejects_unselected_rows() -> None:
    module = _data_module_stub()
    raw = _branch_masks(batch_size=1, sequence_length=6)

    timing = module._time_delta_batch(np.array([0]), raw)

    assert timing["image_time_delta"].shape == (1, 6)
    assert timing["image_time_delta"] is timing["spatial_time_delta"]
    assert timing["image_time_delta"] is timing["visual_context_time_delta"]
    with pytest.raises(ValueError, match="outside the selected temporal view"):
        module._time_delta_batch(np.array([1]), raw)


def test_data_module_rejects_branch_time_shape_mismatch() -> None:
    module = _data_module_stub()
    raw = _branch_masks(batch_size=1, sequence_length=6)
    raw["visual_context_length_mask"] = torch.ones(1, 5)

    with pytest.raises(ValueError, match="time_delta shape mismatch"):
        module._time_delta_batch(np.array([0]), raw)


def test_old_config_resolves_adjacent_primary_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path, temporal_view_manifest=None)

    resolved = resolve_temporal_view_manifest(config)

    assert resolved == tmp_path / "fixed6_observed_time_manifest.csv"
    serialized = training_config_to_jsonable(config)
    assert serialized["dataset"]["temporal_view_manifest"] == str(resolved)


def test_explicit_temporal_manifest_path_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "versioned" / "observed_slots.csv"
    config = _config(tmp_path, temporal_view_manifest=explicit)

    assert resolve_temporal_view_manifest(config) == explicit


def _write_manifest(tmp_path: Path, window_ids: list[str]) -> Path:
    path = tmp_path / "fixed6_observed_time_manifest.csv"
    _manifest(window_ids).to_csv(path, index=False)
    return path


def _manifest(window_ids: list[str]) -> pd.DataFrame:
    rows = []
    for item_order, window_id in enumerate(window_ids):
        view_item_id = f"fixed6|{window_id}"
        for slot_index in range(6):
            rows.append(
                {
                    "temporal_view_name": "fixed6_observed_time",
                    "view_item_id": view_item_id,
                    "parent_window_id": window_id,
                    "item_order": item_order,
                    "slot_index": slot_index,
                    "slot_key": f"{view_item_id}|slot={slot_index}",
                    "declared_sequence_length": 6,
                    "time_delta": 0.0 if slot_index == 0 else 0.2,
                    "length_mask": True,
                    "observed_mask": True,
                    "timing_valid_mask": True,
                }
            )
    return pd.DataFrame(rows)


def _config(
    root: Path,
    *,
    temporal_view_manifest: Path | None,
) -> ClassificationV2TrainingConfig:
    dataset = DatasetConfig(
        snapshot_json=root / "snapshot.json",
        trainer_contract_json=root / "trainer.json",
        train_ready_root=root,
        actor_packed_cache=root / "actor.npy",
        actor_packed_index=root / "actor.csv",
        visual_cache_manifest=root / "visual.csv",
        visual_packed_cache=root / "visual.npy",
        visual_packed_index=root / "visual_index.csv",
        native_oof_fold_manifest=root / "native.csv",
        grouped_fold_roles=root / "roles.csv",
        temporal_view_selection_manifest=root / "selection.csv",
        temporal_view_manifest=temporal_view_manifest,
        auxiliary_targets_csv=root / "auxiliary.csv",
    )
    return ClassificationV2TrainingConfig(
        version="classification_v2_training_config_v1",
        dataset=dataset,
        model=ModelConfig(
            architecture_version=MULTITASK_ARCHITECTURE_VERSION,
        ),
        optimization=OptimizationConfig(),
        loss=LossConfig(sample_weight_policy="uniform"),
        execution=ExecutionConfig(),
    )


def _data_module_stub() -> StrictTrainingDataModule:
    module = StrictTrainingDataModule.__new__(StrictTrainingDataModule)
    module.bundle = SimpleNamespace(
        frame=pd.DataFrame(
            {"temporal_view_selected": [True, False]},
        )
    )
    module.temporal_view_tensors = TemporalViewTensors(
        time_delta=np.array(
            [
                [0.0, 0.2, 0.2, 0.2, 0.2, 0.2],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        ),
        timing_valid_mask=np.array(
            [[True] * 6, [False] * 6],
            dtype=np.bool_,
        ),
        observed_mask=np.array(
            [[True] * 6, [False] * 6],
            dtype=np.bool_,
        ),
        audit={"errors": []},
    )
    module.device = torch.device("cpu")
    return module


def _branch_masks(
    *,
    batch_size: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    return {
        "image_length_mask": torch.ones(batch_size, sequence_length),
        "spatial_length_mask": torch.ones(batch_size, sequence_length),
        "visual_context_length_mask": torch.ones(
            batch_size,
            sequence_length,
        ),
    }

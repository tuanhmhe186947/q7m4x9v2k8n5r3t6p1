from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    DatasetConfig,
    ExecutionConfig,
    LossConfig,
    ModelConfig,
    OptimizationConfig,
)
from pig_behavior.classification_v2.training.fold_preprocessing import (
    ensure_fold_preprocessing_state,
    fit_fold_preprocessing,
    load_fold_preprocessing_state,
    write_fold_preprocessing_state,
)

SNAPSHOT_SHA = "a" * 64
CONFIG_SHA = "b" * 64
SPATIAL_SHA = "c" * 64


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3", "w4"],
            "grouped_role": ["train", "train", "validation", "test", "train"],
            "eligible": [True, True, True, True, False],
        }
    )


def _arrays() -> dict[str, np.ndarray]:
    values = np.asarray(
        [
            [[1.0, 10.0], [3.0, 30.0]],
            [[5.0, 50.0], [7.0, 70.0]],
            [[999.0, 999.0], [999.0, 999.0]],
            [[-999.0, -999.0], [-999.0, -999.0]],
            [[111.0, 111.0], [111.0, 111.0]],
        ],
        dtype=np.float32,
    )
    return {
        "geometry": values,
        "quality": np.ones((5, 2, 1), dtype=np.float32),
        "length_mask": np.ones((5, 2), dtype=np.float32),
        "observed_mask": np.ones((5, 2), dtype=np.float32),
    }


def _fit(
    *,
    frame: pd.DataFrame | None = None,
    arrays: dict[str, np.ndarray] | None = None,
):
    return fit_fold_preprocessing(
        frame if frame is not None else _frame(),
        arrays if arrays is not None else _arrays(),
        {"geometry": ["cx_n", "cy_n"], "quality": ["bbox_valid"]},
        fold_id="q2_outer_00",
        snapshot_sha256=SNAPSHOT_SHA,
        config_sha256=CONFIG_SHA,
        spatial_audit_sha256=SPATIAL_SHA,
        feature_groups=("geometry", "quality"),
        standardized_groups=("geometry",),
    )


def test_fold_preprocessing_fits_all_and_only_eligible_train_rows() -> None:
    state = _fit()

    assert state.train_row_count == 2
    assert state.role_counts == {"train": 2, "validation": 1, "test": 1}
    assert state.statistics["geometry"]["mean"] == [4.0, 40.0]
    assert state.statistics["geometry"]["scale"] == pytest.approx(
        [np.sqrt(5.0), np.sqrt(500.0)]
    )
    assert state.semantic_payload()["fit_contract"][
        "validation_test_excluded_from_fit"
    ] is True


def test_held_out_values_cannot_change_fitted_state() -> None:
    original = _fit()
    changed = _arrays()
    changed["geometry"][2:4] = 123456.0

    refitted = _fit(arrays=changed)

    assert refitted.state_sha256 == original.state_sha256
    assert refitted.statistics == original.statistics


def test_training_values_change_fitted_state() -> None:
    original = _fit()
    changed = _arrays()
    changed["geometry"][0, 0, 0] = 99.0

    refitted = _fit(arrays=changed)

    assert refitted.state_sha256 != original.state_sha256


def test_validation_test_role_drift_changes_lineage_not_statistics() -> None:
    original = _fit()
    changed = _frame()
    changed.loc[2, "grouped_role"] = "test"
    changed.loc[3, "grouped_role"] = "validation"

    refitted = _fit(frame=changed)

    assert refitted.statistics == original.statistics
    assert refitted.role_assignment_sha256 != original.role_assignment_sha256
    assert refitted.state_sha256 != original.state_sha256


def test_transform_imputes_nonfinite_and_zeroes_missing_slots() -> None:
    state = _fit()
    features = {
        "geometry": torch.tensor([[[float("nan"), 40.0], [999.0, 999.0]]]),
        "quality": torch.tensor([[[1.0], [1.0]]]),
    }
    transformed = state.transform_torch(
        features,
        length_mask=torch.tensor([[1.0, 1.0]]),
        observed_mask=torch.tensor([[1.0, 0.0]]),
    )

    assert transformed["geometry"][0, 0].tolist() == [0.0, 0.0]
    assert transformed["geometry"][0, 1].tolist() == [0.0, 0.0]
    assert transformed["quality"][0, 1].tolist() == [0.0]


def test_preprocessing_rejects_non_train_role_in_eligible_rows() -> None:
    frame = _frame()
    frame.loc[0, "grouped_role"] = "outer_prediction"

    with pytest.raises(ValueError, match="invalid grouped roles"):
        _fit(frame=frame)


def test_preprocessing_rejects_feature_order_dimension_drift() -> None:
    with pytest.raises(ValueError, match="feature order mismatch"):
        fit_fold_preprocessing(
            _frame(),
            _arrays(),
            {"geometry": ["cx_n"], "quality": ["bbox_valid"]},
            fold_id="q2_outer_00",
            snapshot_sha256=SNAPSHOT_SHA,
            config_sha256=CONFIG_SHA,
            spatial_audit_sha256=SPATIAL_SHA,
            feature_groups=("geometry", "quality"),
            standardized_groups=("geometry",),
        )


def test_preprocessing_state_rejects_tampered_statistics(tmp_path: Path) -> None:
    path = tmp_path / "preprocessing.json"
    write_fold_preprocessing_state(path, _fit())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["statistics"]["geometry"]["mean"][0] = 999.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="state hash mismatch"):
        load_fold_preprocessing_state(path)


def test_existing_preprocessing_state_rejects_resume_drift(tmp_path: Path) -> None:
    path = tmp_path / "preprocessing.json"
    state = _fit()

    assert ensure_fold_preprocessing_state(path, state) == "written"
    assert ensure_fold_preprocessing_state(path, state) == "matched_existing"
    with pytest.raises(ValueError, match="lineage mismatch"):
        load_fold_preprocessing_state(
            path,
            expected_fold_id="q2_outer_01",
        )


def test_preprocessing_state_requires_explicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "preprocessing.json"
    state = _fit()
    write_fold_preprocessing_state(path, state)

    with pytest.raises(FileExistsError, match="--overwrite"):
        write_fold_preprocessing_state(path, state)


def test_checkpoint_resume_rejects_preprocessing_hash_drift(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    config = _training_config(tmp_path)
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        preprocessing_sha256="preprocessing-a",
        train_window_id_sha256="train-order-a",
        epoch=0,
        global_step=1,
        metrics={},
    )

    with pytest.raises(ValueError, match="preprocessing_sha256"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            preprocessing_sha256="preprocessing-b",
            train_window_id_sha256="train-order-a",
        )


def _training_config(root: Path) -> ClassificationV2TrainingConfig:
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
        auxiliary_targets_csv=root / "auxiliary.csv",
    )
    return ClassificationV2TrainingConfig(
        version="classification_v2_training_config_v1",
        dataset=dataset,
        model=ModelConfig(architecture_version=MULTITASK_ARCHITECTURE_VERSION),
        optimization=OptimizationConfig(),
        loss=LossConfig(),
        execution=ExecutionConfig(),
    )

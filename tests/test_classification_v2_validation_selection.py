from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.config import (
    OptimizationConfig,
    load_training_config,
    validate_training_config,
)
from pig_behavior.classification_v2.training.trainer import (
    _checkpoint_metrics,
    _evaluate,
    _restore_validation_selection_state,
    _validate_best_validation_artifacts,
)
from pig_behavior.classification_v2.training.validation_selection import (
    VALIDATION_PRIMARY_METRIC,
    VALIDATION_TIEBREAKER,
    ValidationSelectionScore,
    build_native_split_evaluation,
    selection_score_from_metrics,
    validation_score_is_better,
)

TRAINING_CONFIGS = (
    "baseline_actor_image.json",
    "baseline_actor_spatial.json",
    "baseline_actor_spatial_partner_context.json",
    "baseline_actor_spatial_partner_multitask.json",
    "baseline_spatial_tcn.json",
    "full_candidate_domain_controls.json",
    "multimodal_context_multitask.json",
)


def test_validation_selection_collapses_windows_before_scoring() -> None:
    windows = _predictions()

    units, metrics, audit = build_native_split_evaluation(
        windows,
        split="validation",
        min_supported_classes=2,
    )

    assert len(windows) == 3
    assert len(units) == 2
    assert metrics[VALIDATION_PRIMARY_METRIC] == pytest.approx(1.0)
    assert metrics[VALIDATION_TIEBREAKER] > 0.0
    assert metrics["validation_native_unit_macro_f1_global"] == pytest.approx(
        0.2
    )
    assert audit["row_loss"] == 0
    assert audit["outer_predictions_used_for_model_selection"] is False
    assert audit["eligible_for_model_selection"] is True
    assert audit["collapse_audit"]["duplicate_native_unit_rows"] == 0
    assert units["temporal_unit_key"].tolist() == ["u1", "u2"]
    assert units["source_type"].tolist() == [
        "legacy_recovered",
        "legacy_recovered",
    ]
    assert units["split_group_key"].tolist() == ["video-1", "video-1"]


def test_trainer_evaluate_emits_window_and_native_unit_evidence() -> None:
    config = SimpleNamespace(
        optimization=SimpleNamespace(
            eval_batch_size=2,
            precision="fp32",
            early_stopping_min_supported_classes=2,
        ),
        execution=SimpleNamespace(fold_id="fold-0"),
        model=SimpleNamespace(architecture_version="fixture-model"),
        dataset=SimpleNamespace(snapshot_json=Path("fixture-snapshot.json")),
    )

    windows, units, metrics, audit = _evaluate(
        _StaticModel(),
        _StaticData(),
        np.asarray([0, 1, 2], dtype=np.int64),
        config,
        torch.device("cpu"),
        split="validation",
    )

    assert len(windows) == 3
    assert len(units) == 2
    assert windows["prediction_split"].unique().tolist() == ["validation"]
    assert units["prediction_split"].unique().tolist() == ["validation"]
    assert metrics[VALIDATION_PRIMARY_METRIC] == pytest.approx(1.0)
    assert audit["row_loss"] == 0


@pytest.mark.parametrize("filename", TRAINING_CONFIGS)
def test_training_configs_bind_native_validation_policy(filename: str) -> None:
    root = Path(__file__).parents[1]
    config = load_training_config(
        root / "configs" / "classification_v2" / filename
    )

    assert config.optimization.early_stopping_metric == VALIDATION_PRIMARY_METRIC
    assert config.optimization.early_stopping_tiebreaker == VALIDATION_TIEBREAKER
    expected_support = (
        len(VALID_BEHAVIORS)
        if config.execution.execution_profile != "local_smoke"
        else 2
    )
    assert (
        config.optimization.early_stopping_min_supported_classes
        == expected_support
    )


def test_remote_pilot_rejects_partial_inner_validation_support() -> None:
    root = Path(__file__).parents[1]
    config = load_training_config(
        root
        / "configs"
        / "classification_v2"
        / "multimodal_context_multitask.json"
    )
    unsafe = replace(
        config,
        optimization=replace(
            config.optimization,
            early_stopping_min_supported_classes=2,
        ),
    )

    with pytest.raises(ValueError, match="requires_all_behavior_classes"):
        validate_training_config(unsafe)


def test_config_rejects_supported_class_requirement_above_label_count() -> None:
    root = Path(__file__).parents[1]
    config = load_training_config(
        root / "configs" / "classification_v2" / "baseline_actor_image.json"
    )
    unsafe = replace(
        config,
        optimization=replace(
            config.optimization,
            early_stopping_min_supported_classes=len(VALID_BEHAVIORS) + 1,
        ),
    )

    with pytest.raises(ValueError, match="class_support_exceeds_label_count"):
        validate_training_config(unsafe)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_window", "duplicate_window_id_rows"),
        ("blank_unit", "blank_temporal_unit_key_rows"),
        ("target_conflict", "true_label_conflict"),
        ("source_conflict", "native_unit_source_type_conflicts"),
        ("nonfinite_probability", "invalid_probability_values"),
        ("nonfinite_unit_loss", "nonfinite_native_unit_loss_rows"),
    ],
)
def test_validation_selection_fails_closed_on_invalid_evidence(
    mutation: str,
    message: str,
) -> None:
    windows = _predictions()
    if mutation == "duplicate_window":
        windows.loc[1, "window_id"] = windows.loc[0, "window_id"]
    elif mutation == "blank_unit":
        windows.loc[0, "temporal_unit_key"] = ""
    elif mutation == "target_conflict":
        windows.loc[1, "true_label"] = "eat"
        windows.loc[1, "y_true"] = "eat"
    elif mutation == "source_conflict":
        windows.loc[1, "source_type"] = "cvat_tracking_xml"
    elif mutation == "nonfinite_probability":
        windows.loc[0, "prob_drink"] = np.nan
    else:
        unit_mask = windows["temporal_unit_key"].eq("u1")
        for index in windows.index[unit_mask]:
            windows.loc[index, _probability_columns()] = 0.0
            windows.loc[index, "prob_eat"] = 1.0
            windows.loc[index, "predicted_label"] = "eat"
            windows.loc[index, "y_pred"] = "eat"

    with pytest.raises(ValueError, match=message):
        build_native_split_evaluation(
            windows,
            split="validation",
            min_supported_classes=2,
        )


def test_validation_selection_rejects_inadequate_supported_classes() -> None:
    with pytest.raises(ValueError, match="inadequate class support"):
        build_native_split_evaluation(
            _predictions(),
            split="validation",
            min_supported_classes=3,
        )


def test_validation_selection_rejects_split_drift() -> None:
    windows = _predictions()
    windows.loc[0, "split"] = "test"

    with pytest.raises(ValueError, match="split mismatch"):
        build_native_split_evaluation(
            windows,
            split="validation",
            min_supported_classes=2,
        )


def test_outer_test_audit_is_not_model_selection_evidence() -> None:
    windows = _predictions().copy()
    windows["split"] = "test"
    windows["prediction_split"] = "test"

    _, _, audit = build_native_split_evaluation(
        windows,
        split="test",
        min_supported_classes=1,
    )

    assert audit["eligible_for_model_selection"] is False
    assert audit["primary_metric"] is None
    assert audit["evaluation_scope"] == "held_out_outer_test_evaluation_only"


def test_validation_selection_uses_nll_only_as_tiebreaker() -> None:
    best = ValidationSelectionScore(primary=0.7, tiebreaker=0.5)
    higher_f1 = ValidationSelectionScore(primary=0.71, tiebreaker=2.0)
    tied_better_nll = ValidationSelectionScore(primary=0.7, tiebreaker=0.4)
    lower_f1 = ValidationSelectionScore(primary=0.69, tiebreaker=0.1)

    assert validation_score_is_better(higher_f1, best, tolerance=1e-12)
    assert validation_score_is_better(tied_better_nll, best, tolerance=1e-12)
    assert not validation_score_is_better(lower_f1, best, tolerance=1e-12)
    restored = selection_score_from_metrics(
        {
            VALIDATION_PRIMARY_METRIC: 0.7,
            VALIDATION_TIEBREAKER: 0.5,
        }
    )
    assert restored == best


def test_validation_selection_state_resumes_without_window_metric() -> None:
    config = SimpleNamespace(optimization=OptimizationConfig())
    score = ValidationSelectionScore(primary=0.7, tiebreaker=0.5)
    record = {
        "epoch": 0,
        VALIDATION_PRIMARY_METRIC: score.primary,
        VALIDATION_TIEBREAKER: score.tiebreaker,
        "selected_as_best_validation": True,
    }
    metrics = _checkpoint_metrics(
        record,
        [record],
        score,
        best_epoch=0,
        stale_epochs=0,
        config=config,
    )

    history, restored, best_epoch, stale_epochs = (
        _restore_validation_selection_state(
            {"epoch": 0, "metrics": metrics},
            config,
        )
    )

    assert history == [record]
    assert restored == score
    assert best_epoch == 0
    assert stale_epochs == 0
    aggregation_audit = {
        "metrics": {
            VALIDATION_PRIMARY_METRIC: score.primary,
            VALIDATION_TIEBREAKER: score.tiebreaker,
        }
    }
    _validate_best_validation_artifacts(
        {"metrics": metrics},
        aggregation_audit,
        score,
        best_epoch=0,
    )


def test_validation_selection_resume_rejects_policy_drift() -> None:
    config = SimpleNamespace(optimization=OptimizationConfig())
    score = ValidationSelectionScore(primary=0.7, tiebreaker=0.5)
    record = {
        "epoch": 0,
        VALIDATION_PRIMARY_METRIC: score.primary,
        VALIDATION_TIEBREAKER: score.tiebreaker,
        "selected_as_best_validation": True,
    }
    metrics = _checkpoint_metrics(
        record,
        [record],
        score,
        best_epoch=0,
        stale_epochs=0,
        config=config,
    )
    metrics["validation_selection"]["primary_metric"] = (
        "validation_window_macro_f1"
    )

    with pytest.raises(ValueError, match="policy mismatch"):
        _restore_validation_selection_state(
            {"epoch": 0, "metrics": metrics},
            config,
        )


def _predictions() -> pd.DataFrame:
    rows = [
        _row("w1", "u1", "drink", "drink", 0.9),
        _row("w2", "u1", "drink", "drink", 0.8),
        _row("w3", "u2", "eat", "eat", 0.7),
    ]
    return pd.DataFrame(rows)


def _row(
    window_id: str,
    unit_id: str,
    true_label: str,
    predicted_label: str,
    confidence: float,
) -> dict[str, object]:
    remainder = (1.0 - confidence) / (len(VALID_BEHAVIORS) - 1)
    row: dict[str, object] = {
        "window_id": window_id,
        "temporal_unit_key": unit_id,
        "oof_fold_id": "fold-0",
        "split": "validation",
        "source_type": "legacy_recovered",
        "split_group_key": "video-1",
        "true_label": true_label,
        "predicted_label": predicted_label,
        "y_true": true_label,
        "y_pred": predicted_label,
        "prediction_split": "validation",
    }
    row.update(
        {
            f"prob_{label}": confidence
            if label == predicted_label
            else remainder
            for label in VALID_BEHAVIORS
        }
    )
    return row


def _probability_columns() -> list[str]:
    return [f"prob_{label}" for label in VALID_BEHAVIORS]


class _StaticModel:
    def eval(self) -> _StaticModel:
        return self

    def __call__(self, **inputs: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(behavior=inputs["logits"])


class _StaticData:
    def batch(self, indices: np.ndarray) -> SimpleNamespace:
        labels = np.asarray([0, 0, 1], dtype=np.int64)[indices]
        logits = torch.full(
            (len(indices), len(VALID_BEHAVIORS)),
            -5.0,
            dtype=torch.float32,
        )
        for row_index, label_index in enumerate(labels):
            logits[row_index, int(label_index)] = 5.0
        units = np.asarray(["u1", "u1", "u2"], dtype=object)[indices]
        return SimpleNamespace(
            model_inputs={"logits": logits},
            behavior_target=torch.from_numpy(labels),
            metadata={
                "window_id": [f"w{index + 1}" for index in indices],
                "temporal_unit_key": units.tolist(),
                "oof_fold_id": ["fold-0"] * len(indices),
                "source_type": ["legacy_recovered"] * len(indices),
                "split_group_key": ["video-1"] * len(indices),
            },
        )

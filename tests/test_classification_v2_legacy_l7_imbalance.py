from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.imbalance_losses import (
    LOSS_POLICIES,
    fit_imbalance_loss_state,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderConfig,
)
from pig_behavior.classification_v2.training.legacy_development_l7_imbalance import (
    EXPECTED_FULL_TRAIN_NATIVE_UNITS,
    EXPECTED_FULL_TRAIN_WINDOWS,
    EXPECTED_PARAMETER_COUNT,
    build_l7_model,
    fit_full_training_loss,
    imbalance_training_step,
)


def _config(tmp_path: Path) -> TemporalLadderConfig:
    payload = {
        "model": {
            "temporal_encoder_name": "masked_mean",
            "hidden_dim": 128,
            "dropout": 0.1,
            "transformer_layers": 1,
            "transformer_heads": 4,
        },
        "optimization": {
            "seed": 20260714,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
        },
    }
    return TemporalLadderConfig(
        path=tmp_path / "synthetic.json",
        payload=payload,
        repo_root=tmp_path,
    )


def _full_training_view(tmp_path: Path) -> LegacyL5CachedFeatureView:
    class_counts = [1000, 700, 500, 400, 300, 250, 200, 150, 100, 52]
    unit_targets = np.concatenate(
        [
            np.full(count, index, dtype=np.int64)
            for index, count in enumerate(class_counts)
        ]
    )
    assert len(unit_targets) == EXPECTED_FULL_TRAIN_NATIVE_UNITS
    targets = np.repeat(unit_targets, 4)
    unit_ids = np.repeat(
        [f"unit_{index:04d}" for index in range(len(unit_targets))],
        4,
    )
    windows = pd.DataFrame(
        {
            "window_id": [
                f"window_{index:05d}" for index in range(len(targets))
            ],
            "temporal_unit_key": unit_ids,
            "behavior_label": [VALID_BEHAVIORS[index] for index in targets],
            "l5_role": "train",
        }
    )
    return LegacyL5CachedFeatureView(
        feature_tensor_path=tmp_path / "unused.npy",
        feature_tensor_sha256="0" * 64,
        control_id="V1",
        temporal_view_name="legacy_t6_all_sliding_observed_time",
        sequence_length=6,
        windows=windows,
        fold_manifest=pd.DataFrame(),
        feature_rows=np.arange(len(targets), dtype=np.int64),
        observed_mask=np.ones((len(targets), 6), dtype=np.bool_),
        time_delta=np.zeros((len(targets), 6), dtype=np.float32),
        targets=targets,
        sample_weights=np.full(len(targets), 0.25, dtype=np.float32),
        audit={},
    )


def test_full_loss_fit_uses_all_native_training_events(tmp_path: Path) -> None:
    view = _full_training_view(tmp_path)
    fit = fit_full_training_loss(
        view,
        policy="effective_number_ce",
    )
    assert fit.train_windows == EXPECTED_FULL_TRAIN_WINDOWS
    assert fit.train_native_units == EXPECTED_FULL_TRAIN_NATIVE_UNITS
    assert np.isclose(fit.event_mass, EXPECTED_FULL_TRAIN_NATIVE_UNITS)
    assert fit.class_native_units == (
        1000,
        700,
        500,
        400,
        300,
        250,
        200,
        150,
        100,
        52,
    )
    assert fit.to_payload()["fit_contract"] == {
        "complete_training_role_used": True,
        "short_optimizer_subset_used_for_fit": False,
        "validation_rows_read_for_fit": 0,
        "outer_holdout_rows_read_for_fit": 0,
        "one_total_mass_per_native_unit": True,
    }


def test_l7_one_batch_gradients_are_finite_for_each_policy(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    batch_size = 20
    generator = np.random.default_rng(20260716)
    batch = {
        "features": generator.normal(
            size=(batch_size, 6, 512),
        ).astype(np.float32),
        "observed_mask": np.ones((batch_size, 6), dtype=np.bool_),
        "time_delta": np.tile(
            np.arange(6, dtype=np.float32),
            (batch_size, 1),
        ),
        "targets": np.tile(np.arange(10, dtype=np.int64), 2),
        "sample_weights": np.full(batch_size, 0.25, dtype=np.float32),
    }
    fit_targets = np.concatenate(
        [np.full(index + 1, index, dtype=np.int64) for index in range(10)]
    )
    fit_weights = np.ones(len(fit_targets), dtype=np.float64)
    for policy in LOSS_POLICIES:
        torch.manual_seed(20260716)
        model = build_l7_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        loss_state = fit_imbalance_loss_state(
            fit_targets,
            fit_weights,
            policy=policy,
        )
        loss, mass, gradient = imbalance_training_step(
            model,
            optimizer,
            batch,
            loss_state=loss_state,
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
        )
        assert np.isfinite(loss)
        assert np.isfinite(mass) and mass > 0.0
        assert np.isfinite(gradient) and gradient > 0.0


def test_l7_model_is_original_retained_width(tmp_path: Path) -> None:
    model = build_l7_model(_config(tmp_path))
    observed = sum(parameter.numel() for parameter in model.parameters())
    assert observed == EXPECTED_PARAMETER_COUNT

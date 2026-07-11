from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    _effective_training_step_count,
    _fold_local_class_weights,
    _fold_training_coverage_complete,
    _training_batches,
    _training_sample_weights,
)


def test_full_training_batches_cover_every_row_each_epoch() -> None:
    """A paper-facing full fold must iterate complete shuffled epochs."""

    config = FullMultimodalOofConfig(
        run_mode="full",
        max_folds=None,
        train_per_class_per_fold=None,
        eval_per_class_per_fold=None,
        train_batch_size=4,
        epochs_per_fold=2,
    )
    indices = np.arange(10, dtype=np.int64)
    batches = list(_training_batches(config, indices, np.random.default_rng(7)))
    observed = np.concatenate(batches)

    assert len(batches) == 6
    assert _effective_training_step_count(config, len(indices)) == 6
    assert sorted(observed.tolist()) == sorted(indices.tolist() * 2)


def test_full_training_coverage_gate_rejects_partial_fold() -> None:
    """Matching config alone cannot mark an interrupted fold complete."""

    complete = {
        "training_steps_completed": 6,
        "expected_training_steps": 6,
        "train_row_coverage_ratio": 1.0,
    }
    partial = {**complete, "training_steps_completed": 5, "train_row_coverage_ratio": 0.9}

    assert _fold_training_coverage_complete([complete]) is True
    assert _fold_training_coverage_complete([complete, partial]) is False


def test_fold_local_class_weights_ignore_held_out_labels() -> None:
    """Changing a held-out label cannot alter a fold's training loss weights."""

    config = FullMultimodalOofConfig(sample_weight_policy="event_class")
    train_indices = np.asarray([0, 1, 2], dtype=np.int64)
    frame = pd.DataFrame(
        {
            "behavior_true": ["drink", "drink", "eat", "playwithtoy"],
            "window_sample_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    changed_held_out = frame.copy()
    changed_held_out.loc[3, "behavior_true"] = "drink"

    original_bundle = SimpleNamespace(
        frame=frame,
        event_sample_weights=np.ones(len(frame), dtype=np.float32),
    )
    changed_bundle = SimpleNamespace(
        frame=changed_held_out,
        event_sample_weights=np.ones(len(changed_held_out), dtype=np.float32),
    )
    original = _fold_local_class_weights(original_bundle, train_indices, config)
    changed = _fold_local_class_weights(changed_bundle, train_indices, config)

    assert original == changed
    assert original["eat"] > original["drink"]


def test_fold_local_class_weights_balance_effective_event_mass() -> None:
    """Overlapping windows from one event must not count as independent class evidence."""

    config = FullMultimodalOofConfig(sample_weight_policy="event_class", class_weight_power=1.0)
    frame = pd.DataFrame({"behavior_true": ["drink", "drink", "eat"]})
    bundle = SimpleNamespace(
        frame=frame,
        event_sample_weights=np.asarray([0.5, 0.5, 1.0], dtype=np.float32),
    )

    weights = _fold_local_class_weights(bundle, np.asarray([0, 1, 2], dtype=np.int64), config)

    assert weights["drink"] == 1.0
    assert weights["eat"] == 1.0


def test_event_class_weights_compose_without_entering_model_features() -> None:
    """Training weights multiply event balance by fold-local class balance."""

    config = FullMultimodalOofConfig(sample_weight_policy="event_class")
    frame = pd.DataFrame(
        {
            "behavior_true": ["drink", "eat"],
            "window_sample_weight": [1.0, 1.0],
        }
    )
    bundle = SimpleNamespace(frame=frame, event_sample_weights=np.asarray([0.5, 2.0], dtype=np.float32))
    weights = _training_sample_weights(
        bundle,
        np.asarray([0, 1], dtype=np.int64),
        {"drink": 2.0, "eat": 0.25},
        config,
    )

    np.testing.assert_allclose(weights, np.asarray([1.0, 0.5], dtype=np.float32))

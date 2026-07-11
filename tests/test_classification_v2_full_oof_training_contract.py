from __future__ import annotations

import numpy as np

from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    _effective_training_step_count,
    _fold_training_coverage_complete,
    _training_batches,
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

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.contracts.window_alignment import (
    require_ordered_window_ids,
)
from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    _effective_training_step_count,
    _fold_local_class_weights,
    _fold_training_coverage_complete,
    _save_training_checkpoint,
    _training_batches,
    _training_sample_weights,
)


def test_ordered_window_alignment_rejects_same_keys_in_wrong_order() -> None:
    reference = pd.Series(["window-0", "window-1"])

    with pytest.raises(ValueError, match="window_order_mismatch_rows=2"):
        require_ordered_window_ids(
            "split",
            reference,
            {"image_context": pd.Series(["window-1", "window-0"])},
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
    bundle = SimpleNamespace(
        frame=frame, event_sample_weights=np.asarray([0.5, 2.0], dtype=np.float32)
    )
    weights = _training_sample_weights(
        bundle,
        np.asarray([0, 1], dtype=np.int64),
        {"drink": 2.0, "eat": 0.25},
        config,
    )

    np.testing.assert_allclose(weights, np.asarray([1.0, 0.5], dtype=np.float32))


def test_v2_training_checkpoint_contains_resumable_optimizer_progress(tmp_path: Path) -> None:
    """Periodic checkpoints must carry all state needed to continue an interrupted fold."""

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    model_path = tmp_path / "trained_model.pt"
    audit_path = tmp_path / "training_audit.json"

    _save_training_checkpoint(
        model,
        optimizer,
        scaler,
        model_path,
        audit_path,
        training_signature={"fold": "f0"},
        losses=[1.0, 0.5],
        seen_train_indices={1, 3},
        completed_training_steps=2,
        training_elapsed_sec=4.0,
        peak_allocated_mb=10.0,
        peak_reserved_mb=12.0,
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == "classification_v2_training_checkpoint_v2"
    assert checkpoint["completed_training_steps"] == 2
    assert checkpoint["optimizer_state_dict"]["state"]
    assert audit["completed_training_steps"] == 2
    assert not list(tmp_path.glob("*.tmp"))

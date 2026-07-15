from __future__ import annotations

import numpy as np
import pytest
import torch

from pig_behavior.classification_v2.training.imbalance_losses import (
    LOSS_POLICIES,
    TRAINING_FIT_ROLE,
    fit_imbalance_loss_state,
    weighted_imbalance_loss,
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    targets = np.arange(10, dtype=np.int64)
    masses = np.ones(10, dtype=np.float64)
    return targets, masses


def test_all_policies_are_finite_and_deterministic() -> None:
    targets, masses = _data()
    logits = torch.arange(100, dtype=torch.float32).reshape(10, 10) / 13.0
    target_tensor = torch.from_numpy(targets)
    mass_tensor = torch.from_numpy(masses.astype(np.float32))
    for policy in LOSS_POLICIES:
        state = fit_imbalance_loss_state(
            targets,
            masses,
            policy=policy,
        )
        repeated = fit_imbalance_loss_state(
            targets,
            masses,
            policy=policy,
        )
        assert state.to_payload() == repeated.to_payload()
        loss, mass = weighted_imbalance_loss(
            logits,
            target_tensor,
            mass_tensor,
            state,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(mass)
        assert float(mass) > 0.0


def test_effective_number_downweights_high_mass_class() -> None:
    targets = np.asarray([0] * 100 + list(range(1, 10)), dtype=np.int64)
    masses = np.ones(len(targets), dtype=np.float64)
    state = fit_imbalance_loss_state(
        targets,
        masses,
        policy="effective_number_ce",
    )
    assert state.class_weights[0] < state.class_weights[1]
    assert np.isclose(np.mean(state.class_weights), 1.0)


def test_event_mass_policy_preserves_native_mass() -> None:
    targets = np.asarray(list(range(10)) + [0, 1], dtype=np.int64)
    masses = np.asarray([1.0] * 10 + [0.25, 0.25], dtype=np.float64)
    state = fit_imbalance_loss_state(
        targets,
        masses,
        policy="event_balanced_ce",
    )
    _, effective_mass = weighted_imbalance_loss(
        torch.zeros((len(targets), 10), dtype=torch.float32),
        torch.from_numpy(targets),
        torch.from_numpy(masses.astype(np.float32)),
        state,
    )
    assert np.isclose(float(effective_mass), masses.sum())
    assert np.isclose(sum(state.class_mass), masses.sum())


def test_balanced_softmax_uses_training_class_mass_prior() -> None:
    targets = np.asarray([0, 0, 0] + list(range(1, 10)), dtype=np.int64)
    masses = np.ones(len(targets), dtype=np.float64)
    state = fit_imbalance_loss_state(
        targets,
        masses,
        policy="balanced_softmax",
    )
    assert np.isclose(
        state.balanced_softmax_log_prior[0],
        np.log(3.0),
    )
    assert np.isclose(state.balanced_softmax_log_prior[1], np.log(1.0))


def test_fit_role_is_training_only() -> None:
    with pytest.raises(ValueError, match="training native-event"):
        fit_imbalance_loss_state(
            np.arange(10, dtype=np.int64),
            np.ones(10, dtype=np.float64),
            policy="event_balanced_ce",
            fit_role="validation_native_event_mass",
        )
    assert TRAINING_FIT_ROLE == "training_native_event_mass"


def test_loss_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unknown imbalance"):
        fit_imbalance_loss_state(
            np.zeros(2, dtype=np.int64),
            np.ones(2),
            policy="focal_loss",
        )

"""Fold-local loss policies for the legacy imbalance ablation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

LOSS_POLICIES = (
    "event_balanced_ce",
    "effective_number_ce",
    "balanced_softmax",
)
DEFAULT_EFFECTIVE_NUMBER_BETA = 0.9999
TRAINING_FIT_ROLE = "training_native_event_mass"


@dataclass(frozen=True, slots=True)
class ImbalanceLossState:
    """Immutable training-fold mass and derived loss coefficients."""

    policy: str
    effective_number_beta: float
    class_mass: tuple[float, ...]
    class_weights: tuple[float, ...]
    balanced_softmax_log_prior: tuple[float, ...]
    fit_role: str
    state_sha256: str

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "classification_v2.imbalance_loss_state.v1",
            "policy": self.policy,
            "effective_number_beta": self.effective_number_beta,
            "class_order": list(VALID_BEHAVIORS),
            "class_mass": list(self.class_mass),
            "class_weights": list(self.class_weights),
            "balanced_softmax_log_prior": list(
                self.balanced_softmax_log_prior
            ),
            "fit_role": self.fit_role,
            "fit_contract": {
                "native_event_mass_only": True,
                "validation_and_outer_excluded": True,
                "sampler_unchanged": True,
            },
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "state_sha256": self.state_sha256}


def fit_imbalance_loss_state(
    targets: np.ndarray,
    sample_weights: np.ndarray,
    *,
    policy: str,
    fit_role: str = "training_native_event_mass",
    effective_number_beta: float = DEFAULT_EFFECTIVE_NUMBER_BETA,
) -> ImbalanceLossState:
    """Fit class mass and loss coefficients using one training role only."""

    if policy not in LOSS_POLICIES:
        raise ValueError(f"unknown imbalance loss policy={policy}")
    if fit_role != TRAINING_FIT_ROLE:
        raise ValueError(
            "imbalance state must be fit from the training native-event role"
        )
    labels = np.asarray(targets, dtype=np.int64)
    masses = np.asarray(sample_weights, dtype=np.float64)
    if labels.ndim != 1 or masses.ndim != 1 or len(labels) != len(masses):
        raise ValueError("imbalance targets and masses must be aligned vectors")
    if len(labels) == 0 or not np.isfinite(masses).all():
        raise ValueError("imbalance training mass is empty or nonfinite")
    if (masses <= 0.0).any():
        raise ValueError("imbalance training mass must be positive")
    if (labels < 0).any() or (labels >= len(VALID_BEHAVIORS)).any():
        raise ValueError("imbalance target index is outside the ten classes")
    if not 0.0 < effective_number_beta < 1.0:
        raise ValueError("effective-number beta must be in (0,1)")
    class_mass = np.bincount(
        labels,
        weights=masses,
        minlength=len(VALID_BEHAVIORS),
    ).astype(np.float64)
    if (class_mass <= 0.0).any():
        raise ValueError("imbalance training role lacks a supported class")
    class_weights = np.ones_like(class_mass)
    if policy == "effective_number_ce":
        effective = 1.0 - np.power(effective_number_beta, class_mass)
        class_weights = (1.0 - effective_number_beta) / effective
        class_weights /= class_weights.mean()
    log_prior = np.log(class_mass)
    payload = {
        "schema_version": "classification_v2.imbalance_loss_state.v1",
        "policy": policy,
        "effective_number_beta": float(effective_number_beta),
        "class_order": list(VALID_BEHAVIORS),
        "class_mass": class_mass.tolist(),
        "class_weights": class_weights.tolist(),
        "balanced_softmax_log_prior": log_prior.tolist(),
        "fit_role": fit_role,
        "fit_contract": {
            "native_event_mass_only": True,
            "validation_and_outer_excluded": True,
            "sampler_unchanged": True,
        },
    }
    state_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return ImbalanceLossState(
        policy=policy,
        effective_number_beta=float(effective_number_beta),
        class_mass=tuple(float(value) for value in class_mass),
        class_weights=tuple(float(value) for value in class_weights),
        balanced_softmax_log_prior=tuple(float(value) for value in log_prior),
        fit_role=fit_role,
        state_sha256=state_sha256,
    )


def weighted_imbalance_loss(
    logits: Tensor,
    targets: Tensor,
    sample_weights: Tensor,
    state: ImbalanceLossState,
) -> tuple[Tensor, Tensor]:
    """Return loss and the exact effective mass used by the optimizer."""

    if state.policy not in LOSS_POLICIES:
        raise ValueError(f"unknown imbalance loss state policy={state.policy}")
    if state.fit_role != TRAINING_FIT_ROLE:
        raise ValueError("imbalance state fit role is not training native-event")
    if len(state.class_weights) != len(VALID_BEHAVIORS):
        raise ValueError("imbalance class-weight vector must contain ten classes")
    if len(state.balanced_softmax_log_prior) != len(VALID_BEHAVIORS):
        raise ValueError("imbalance prior vector must contain ten classes")
    if logits.ndim != 2 or logits.shape[1] != len(VALID_BEHAVIORS):
        raise ValueError("imbalance logits must have shape [N,10]")
    if targets.ndim != 1 or sample_weights.ndim != 1:
        raise ValueError("imbalance targets and masses must be vectors")
    if len(targets) != len(sample_weights) or len(targets) != len(logits):
        raise ValueError("imbalance batch tensors are not aligned")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("imbalance logits are nonfinite")
    if (targets < 0).any() or (targets >= len(VALID_BEHAVIORS)).any():
        raise ValueError("imbalance batch target is outside the ten classes")
    if not torch.isfinite(sample_weights).all() or (sample_weights <= 0).any():
        raise ValueError("imbalance batch mass is invalid")
    device = logits.device
    class_weights = torch.as_tensor(
        state.class_weights,
        dtype=logits.dtype,
        device=device,
    )
    adjusted_logits = logits
    if state.policy == "balanced_softmax":
        prior = torch.as_tensor(
            state.balanced_softmax_log_prior,
            dtype=logits.dtype,
            device=device,
        )
        adjusted_logits = logits + prior.unsqueeze(0)
    losses = torch.nn.functional.cross_entropy(
        adjusted_logits,
        targets,
        reduction="none",
    )
    effective_weights = sample_weights
    if state.policy == "effective_number_ce":
        effective_weights = effective_weights * class_weights[targets]
    mass = effective_weights.sum()
    loss = (losses * effective_weights).sum() / mass
    if not torch.isfinite(loss) or not torch.isfinite(mass):
        raise FloatingPointError("imbalance loss or mass is nonfinite")
    return loss, mass

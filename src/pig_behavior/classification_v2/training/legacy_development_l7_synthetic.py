"""Data-free correctness gates for the legacy L7 loss policies."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.imbalance_losses import (
    LOSS_POLICIES,
    fit_imbalance_loss_state,
    weighted_imbalance_loss,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderConfig,
)
from pig_behavior.classification_v2.training.legacy_development_l7_imbalance import (
    build_l7_model,
    imbalance_training_step,
)


def run_l7_synthetic_gate(
    config: TemporalLadderConfig,
    *,
    seed: int = 20260716,
    tiny_steps: int = 30,
) -> dict[str, Any]:
    """Run finite-gradient, tiny-overfit, and resume checks for every policy."""

    if tiny_steps <= 0:
        raise ValueError("L7 synthetic tiny steps must be positive")
    errors: list[str] = []
    policy_audits: dict[str, Any] = {}
    batch = _synthetic_batch(seed)
    fit_targets = np.concatenate(
        [np.full(index + 1, index, dtype=np.int64) for index in range(10)]
    )
    fit_weights = np.ones(len(fit_targets), dtype=np.float64)
    for policy in LOSS_POLICIES:
        torch.manual_seed(seed)
        state = fit_imbalance_loss_state(
            fit_targets,
            fit_weights,
            policy=policy,
        )
        one_batch = _one_batch_gate(config, batch, state)
        overfit = _tiny_overfit_gate(
            config,
            batch,
            state,
            tiny_steps=tiny_steps,
            seed=seed,
        )
        resume = _resume_gate(config, batch, state, seed=seed)
        policy_audits[policy] = {
            "state_sha256": state.state_sha256,
            "one_batch": one_batch,
            "tiny_overfit": overfit,
            "resume": resume,
        }
        for name, audit in (
            ("one_batch", one_batch),
            ("tiny_overfit", overfit),
            ("resume", resume),
        ):
            if not audit["valid"]:
                errors.extend(
                    f"{policy}.{name}:{value}" for value in audit["errors"]
                )
    return {
        "schema_version": "classification_v2.legacy_development_l7.synthetic.v1",
        "lineage_scope": "legacy-only-unreviewed-development",
        "class_order": list(VALID_BEHAVIORS),
        "seed": seed,
        "tiny_steps": tiny_steps,
        "policies": policy_audits,
        "errors": errors,
        "valid": not errors,
    }


def _one_batch_gate(
    config: TemporalLadderConfig,
    batch: dict[str, np.ndarray],
    state: Any,
) -> dict[str, Any]:
    model = build_l7_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    try:
        loss, mass, gradient = imbalance_training_step(
            model,
            optimizer,
            batch,
            loss_state=state,
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
        )
        finite = bool(np.isfinite([loss, mass, gradient]).all())
        valid = finite and mass > 0.0 and gradient > 0.0
        errors = [] if valid else ["one_batch_nonfinite_or_zero_gradient"]
        return {
            "loss": loss,
            "effective_mass": mass,
            "gradient_norm": gradient,
            "finite": finite,
            "errors": errors,
            "valid": valid,
        }
    finally:
        del optimizer, model


def _tiny_overfit_gate(
    config: TemporalLadderConfig,
    batch: dict[str, np.ndarray],
    state: Any,
    *,
    tiny_steps: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = build_l7_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    try:
        initial = _loss_and_accuracy(model, batch, state)
        losses: list[float] = []
        for _ in range(tiny_steps):
            loss, _, _ = imbalance_training_step(
                model,
                optimizer,
                batch,
                loss_state=state,
                device=torch.device("cpu"),
                gradient_clip_norm=1.0,
            )
            losses.append(loss)
        final = _loss_and_accuracy(model, batch, state)
        ratio = final["loss"] / max(initial["loss"], 1e-12)
        valid = (
            np.isfinite(losses).all()
            and ratio < 0.2
            and final["accuracy"] >= 0.95
        )
        errors = [] if valid else [
            "tiny_overfit_did_not_reach_accuracy_or_loss_ratio"
        ]
        return {
            "initial_loss": initial["loss"],
            "final_loss": final["loss"],
            "loss_ratio": ratio,
            "final_accuracy": final["accuracy"],
            "steps": tiny_steps,
            "errors": errors,
            "valid": valid,
        }
    finally:
        del optimizer, model


def _resume_gate(
    config: TemporalLadderConfig,
    batch: dict[str, np.ndarray],
    state: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = build_l7_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    try:
        imbalance_training_step(
            model,
            optimizer,
            batch,
            loss_state=state,
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
        )
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        rng_state = torch.get_rng_state().clone()
        resumed_model = build_l7_model(config)
        resumed_optimizer = torch.optim.AdamW(
            resumed_model.parameters(),
            lr=0.003,
        )
        resumed_model.load_state_dict(model_state)
        resumed_optimizer.load_state_dict(optimizer_state)
        torch.set_rng_state(rng_state.clone())
        original_loss, _, _ = imbalance_training_step(
            model,
            optimizer,
            batch,
            loss_state=state,
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
        )
        torch.set_rng_state(rng_state.clone())
        resumed_loss, _, _ = imbalance_training_step(
            resumed_model,
            resumed_optimizer,
            batch,
            loss_state=state,
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
        )
        logit_delta = _logit_delta(model, resumed_model, batch)
        loss_delta = abs(original_loss - resumed_loss)
        valid = logit_delta == 0.0 and loss_delta == 0.0
        errors = [] if valid else ["resume_step_is_not_bitwise_equivalent"]
        return {
            "original_next_loss": original_loss,
            "resumed_next_loss": resumed_loss,
            "next_loss_abs_delta": loss_delta,
            "next_logit_max_abs_delta": logit_delta,
            "errors": errors,
            "valid": valid,
        }
    finally:
        if "resumed_optimizer" in locals():
            del resumed_optimizer, resumed_model
        del optimizer, model


def _loss_and_accuracy(
    model: torch.nn.Module,
    batch: dict[str, np.ndarray],
    state: Any,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = _logits(model, batch)
        loss, _ = weighted_imbalance_loss(
            logits,
            torch.from_numpy(batch["targets"]).long(),
            torch.from_numpy(batch["sample_weights"]).float(),
            state,
        )
    predictions = logits.argmax(dim=1).cpu().numpy()
    accuracy = float(np.mean(predictions == batch["targets"]))
    return {"loss": float(loss.cpu()), "accuracy": accuracy}


def _logit_delta(
    left: torch.nn.Module,
    right: torch.nn.Module,
    batch: dict[str, np.ndarray],
) -> float:
    left.eval()
    right.eval()
    with torch.no_grad():
        delta = (_logits(left, batch) - _logits(right, batch)).abs().max()
    return float(delta.cpu())


def _logits(
    model: torch.nn.Module,
    batch: dict[str, np.ndarray],
) -> torch.Tensor:
    return model(
        torch.from_numpy(batch["features"]),
        torch.from_numpy(batch["observed_mask"]).float(),
        time_delta=torch.from_numpy(batch["time_delta"]).float(),
    )


def _synthetic_batch(seed: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    targets = np.tile(np.arange(10, dtype=np.int64), 2)
    return {
        "features": generator.normal(size=(20, 6, 512)).astype(np.float32),
        "observed_mask": np.ones((20, 6), dtype=np.bool_),
        "time_delta": np.tile(np.arange(6, dtype=np.float32), (20, 1)),
        "targets": targets,
        "sample_weights": np.full(20, 0.25, dtype=np.float32),
    }

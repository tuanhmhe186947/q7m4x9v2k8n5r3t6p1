"""Loss registry and harness for the class-imbalance study (L0-L7).

Every loss returns its components separately so an audit can show *why* a
sample carried the mass it did:

``unreduced_per_sample_loss`` -> ``class_weight`` -> ``native_unit_mass_weight``
-> ``final_sample_weight`` -> ``reduced_loss``.

Class weights are always derived from :class:`ClassPriors`, which can only be
fitted on training-fold native temporal units.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from pig_behavior.classification_v2.training.balanced.class_priors import (
    ClassPriors,
    require_train_fold_priors,
)

LOSS_SCHEMA_VERSION = "classification_v2.balanced_losses.v1"

WEIGHTED_CE_STRATEGIES: tuple[str, ...] = (
    "inverse_frequency",
    "inverse_sqrt_frequency",
    "capped_inverse_frequency",
)

REDUCTIONS: tuple[str, ...] = ("weighted_mean", "sum", "mean")

#: All eight declared candidates, including the two support-only entries.
LOSS_CANDIDATES: tuple[str, ...] = (
    "L0_STANDARD_CROSS_ENTROPY",
    "L1_WEIGHTED_CROSS_ENTROPY",
    "L2_EFFECTIVE_NUMBER_CLASS_BALANCED",
    "L3_BALANCED_SOFTMAX",
    "L3_LOGIT_ADJUSTMENT",
    "L4_FOCAL_LOSS",
    "L5_LDAM_DRW",
    "L6_CLASSIFIER_RETRAINING_SUPPORT",
    "L7_CONTROLLED_SAMPLING_SUPPORT",
)


class LossConfigError(ValueError):
    """Raised when a loss candidate is configured incompletely or unsafely."""


@dataclass(frozen=True, slots=True)
class LossComponents:
    """Separated loss components for the mandatory per-loss ablation table."""

    unreduced_per_sample_loss: Tensor
    class_weight: Tensor
    native_unit_mass_weight: Tensor
    final_sample_weight: Tensor
    reduced_loss: Tensor

    def to_payload(self) -> dict[str, Any]:
        return {
            "per_sample_loss_mean": float(self.unreduced_per_sample_loss.mean()),
            "class_weight_mean": float(self.class_weight.mean()),
            "native_unit_mass_weight_mean": float(self.native_unit_mass_weight.mean()),
            "final_sample_weight_sum": float(self.final_sample_weight.sum()),
            "reduced_loss": float(self.reduced_loss.detach()),
        }


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Validated configuration for one loss candidate."""

    name: str
    priors: ClassPriors | None = None
    reduction: str = "weighted_mean"
    # L1
    weighting_strategy: str | None = None
    weight_cap: float | None = None
    # L2
    effective_number_beta: float | None = None
    # L3
    tau: float | None = None
    # L4
    focal_gamma: float | None = None
    focal_alpha: tuple[float, ...] | None = None
    # L5
    ldam_max_margin: float | None = None
    ldam_scale: float = 30.0
    drw_start_epoch: int | None = None
    drw_beta: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in LOSS_CANDIDATES:
            raise LossConfigError(
                f"unknown loss candidate={self.name}; expected one of "
                f"{list(LOSS_CANDIDATES)}"
            )
        if self.reduction not in REDUCTIONS:
            raise LossConfigError(
                f"unknown reduction={self.reduction}; expected one of {list(REDUCTIONS)}"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": LOSS_SCHEMA_VERSION,
            "name": self.name,
            "reduction": self.reduction,
            "weighting_strategy": self.weighting_strategy,
            "weight_cap": self.weight_cap,
            "effective_number_beta": self.effective_number_beta,
            "tau": self.tau,
            "focal_gamma": self.focal_gamma,
            "focal_alpha_configured": self.focal_alpha is not None,
            "ldam_max_margin": self.ldam_max_margin,
            "ldam_scale": self.ldam_scale,
            "drw_start_epoch": self.drw_start_epoch,
            "drw_beta": self.drw_beta,
            "priors": None if self.priors is None else self.priors.to_payload(),
        }


class BalancedLoss(nn.Module):
    """Common interface for every imbalance-loss candidate."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.name = config.name

    def class_weight_vector(self, epoch: int | None) -> Tensor:
        raise NotImplementedError

    def adjusted_logits(self, logits: Tensor, targets: Tensor) -> Tensor:
        return logits

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(
            self.adjusted_logits(logits, targets),
            targets,
            reduction="none",
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        native_unit_mass_weight: Tensor | None = None,
        epoch: int | None = None,
    ) -> LossComponents:
        _validate_logits(logits, targets)
        per_sample = self.per_sample_loss(logits, targets)
        weights = self.class_weight_vector(epoch).to(logits.device)
        class_weight = weights.index_select(0, targets)
        if native_unit_mass_weight is None:
            mass = torch.ones_like(class_weight)
        else:
            mass = native_unit_mass_weight.to(logits.device).float()
            if mass.shape != class_weight.shape:
                raise LossConfigError(
                    "native_unit_mass_weight must be [B]; observed "
                    f"{tuple(mass.shape)} expected {tuple(class_weight.shape)}"
                )
            if bool((mass < 0).any()):
                raise LossConfigError("native_unit_mass_weight must be non-negative")
        final = class_weight * mass
        reduced = _reduce(per_sample, final, self.config.reduction)
        return LossComponents(
            unreduced_per_sample_loss=per_sample,
            class_weight=class_weight,
            native_unit_mass_weight=mass,
            final_sample_weight=final,
            reduced_loss=reduced,
        )


class StandardCrossEntropy(BalancedLoss):
    """L0 reference: unweighted cross-entropy."""

    def class_weight_vector(self, epoch: int | None) -> Tensor:
        del epoch
        size = self._num_classes()
        return torch.ones(size, dtype=torch.float32)

    def _num_classes(self) -> int:
        if self.config.priors is not None:
            return self.config.priors.num_classes
        return int(self.config.extra.get("num_classes", 10))


class WeightedCrossEntropy(BalancedLoss):
    """L1: inverse-frequency family with an explicit bounded strategy."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__(config)
        if config.priors is None:
            raise LossConfigError("L1_WEIGHTED_CROSS_ENTROPY requires class priors")
        if config.weighting_strategy not in WEIGHTED_CE_STRATEGIES:
            raise LossConfigError(
                "L1_WEIGHTED_CROSS_ENTROPY requires weighting_strategy in "
                f"{list(WEIGHTED_CE_STRATEGIES)}; observed "
                f"{config.weighting_strategy}"
            )
        if config.weighting_strategy == "capped_inverse_frequency" and (
            config.weight_cap is None or config.weight_cap <= 0
        ):
            raise LossConfigError(
                "capped_inverse_frequency requires a positive weight_cap"
            )
        priors = require_train_fold_priors(config.priors)
        counts = priors.counts_array()
        if config.weighting_strategy == "inverse_frequency":
            weights = counts.sum() / counts
        elif config.weighting_strategy == "inverse_sqrt_frequency":
            weights = np.sqrt(counts.sum() / counts)
        else:
            weights = np.minimum(counts.sum() / counts, float(config.weight_cap))
        weights = weights / weights.mean()
        self.register_buffer(
            "weights",
            torch.as_tensor(weights, dtype=torch.float32),
        )

    def class_weight_vector(self, epoch: int | None) -> Tensor:
        del epoch
        return self.weights


class EffectiveNumberClassBalanced(BalancedLoss):
    """L2: effective-number class-balanced weights with an explicit beta."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__(config)
        if config.priors is None:
            raise LossConfigError(
                "L2_EFFECTIVE_NUMBER_CLASS_BALANCED requires class priors"
            )
        if config.effective_number_beta is None:
            raise LossConfigError(
                "L2_EFFECTIVE_NUMBER_CLASS_BALANCED requires an explicit "
                "effective_number_beta; no global beta is selected for you"
            )
        beta = float(config.effective_number_beta)
        if not 0.0 < beta < 1.0:
            raise LossConfigError("effective_number_beta must be in (0,1)")
        priors = require_train_fold_priors(config.priors)
        weights = effective_number_weights(priors.counts_array(), beta)
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

    def class_weight_vector(self, epoch: int | None) -> Tensor:
        del epoch
        return self.weights


class LogitAdjustedCrossEntropy(BalancedLoss):
    """L3: balanced softmax (``tau=1``) and general logit adjustment."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__(config)
        if config.priors is None:
            raise LossConfigError(f"{config.name} requires class priors")
        tau = config.tau
        if config.name == "L3_BALANCED_SOFTMAX":
            tau = 1.0 if tau is None else float(tau)
        if tau is None:
            raise LossConfigError(
                "L3_LOGIT_ADJUSTMENT requires an explicit tau (balanced softmax "
                "is the tau=1.0 special case)"
            )
        priors = require_train_fold_priors(config.priors)
        self.tau = float(tau)
        self.register_buffer(
            "log_prior",
            torch.as_tensor(priors.log_prior(), dtype=torch.float32),
        )

    def adjusted_logits(self, logits: Tensor, targets: Tensor) -> Tensor:
        del targets
        return logits + self.tau * self.log_prior.to(logits.device)

    def class_weight_vector(self, epoch: int | None) -> Tensor:
        del epoch
        return torch.ones_like(self.log_prior)

    def inference_logits(self, logits: Tensor) -> Tensor:
        """Return deployment logits: the adjustment is a training-time term."""

        return logits


class FocalLoss(BalancedLoss):
    """L4: focal loss with configurable gamma and optional explicit alpha."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__(config)
        if config.focal_gamma is None:
            raise LossConfigError("L4_FOCAL_LOSS requires an explicit focal_gamma")
        if float(config.focal_gamma) < 0.0:
            raise LossConfigError("focal_gamma must not be negative")
        self.gamma = float(config.focal_gamma)
        alpha = config.focal_alpha
        self.uses_alpha = alpha is not None
        size = config.priors.num_classes if config.priors is not None else None
        if alpha is not None:
            values = torch.as_tensor(alpha, dtype=torch.float32)
            if size is not None and values.numel() != size:
                raise LossConfigError(
                    f"focal_alpha must have {size} entries; observed {values.numel()}"
                )
            self.register_buffer("alpha", values)
        else:
            self.register_buffer("alpha", torch.ones(size or 10, dtype=torch.float32))

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        cross_entropy = torch.nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
        )
        probability = torch.exp(-cross_entropy)
        return ((1.0 - probability) ** self.gamma) * cross_entropy

    def class_weight_vector(self, epoch: int | None) -> Tensor:
        del epoch
        if not self.uses_alpha:
            return torch.ones_like(self.alpha)
        return self.alpha


class LDAMDeferredReweighting(BalancedLoss):
    """L5: LDAM margins with a deferred re-weighting schedule."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__(config)
        if config.priors is None:
            raise LossConfigError("L5_LDAM_DRW requires class priors")
        if config.ldam_max_margin is None:
            raise LossConfigError("L5_LDAM_DRW requires an explicit ldam_max_margin")
        if config.drw_start_epoch is None:
            raise LossConfigError("L5_LDAM_DRW requires an explicit drw_start_epoch")
        if config.drw_beta is None:
            raise LossConfigError(
                "L5_LDAM_DRW requires an explicit drw_beta for the deferred "
                "re-weighting stage"
            )
        priors = require_train_fold_priors(config.priors)
        counts = priors.counts_array()
        margins = 1.0 / np.power(counts, 0.25)
        margins = margins * (float(config.ldam_max_margin) / margins.max())
        self.register_buffer("margins", torch.as_tensor(margins, dtype=torch.float32))
        self.register_buffer(
            "deferred_weights",
            torch.as_tensor(
                effective_number_weights(counts, float(config.drw_beta)),
                dtype=torch.float32,
            ),
        )
        self.scale = float(config.ldam_scale)
        self.drw_start_epoch = int(config.drw_start_epoch)

    def adjusted_logits(self, logits: Tensor, targets: Tensor) -> Tensor:
        margins = self.margins.to(logits.device).index_select(0, targets)
        adjusted = logits.clone()
        rows = torch.arange(logits.shape[0], device=logits.device)
        adjusted[rows, targets] = adjusted[rows, targets] - margins
        return self.scale * adjusted

    def class_weight_vector(self, epoch: int | None) -> Tensor:
        if epoch is None:
            raise LossConfigError(
                "L5_LDAM_DRW needs the current epoch to evaluate its deferred "
                "re-weighting schedule"
            )
        if int(epoch) < self.drw_start_epoch:
            return torch.ones_like(self.deferred_weights)
        return self.deferred_weights


@dataclass(frozen=True, slots=True)
class ClassifierRetrainingConfig:
    """L6 support: decoupled classifier retraining (cRT) stage description.

    This is configuration and interface only. Nothing here runs a production
    cRT stage.
    """

    freeze_representation: bool = True
    reinitialize_classifier: bool = True
    sampler: str = "CLASS_AWARE"
    epochs: int = 10
    learning_rate: float = 1e-3
    tau_normalization_tau: float | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise LossConfigError("cRT epochs must be positive")
        if self.learning_rate <= 0.0:
            raise LossConfigError("cRT learning_rate must be positive")
        if self.tau_normalization_tau is not None and self.tau_normalization_tau < 0.0:
            raise LossConfigError("tau normalization tau must not be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage": "L6_CLASSIFIER_RETRAINING_SUPPORT",
            "freeze_representation": self.freeze_representation,
            "reinitialize_classifier": self.reinitialize_classifier,
            "sampler": self.sampler,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "tau_normalization_tau": self.tau_normalization_tau,
            "production_crt_executed": False,
        }


def tau_normalize_classifier(weight: Tensor, tau: float) -> Tensor:
    """Return tau-normalized classifier weights ``w_c / ||w_c||^tau``."""

    if weight.ndim != 2:
        raise LossConfigError("classifier weight must be [num_classes, features]")
    if tau < 0.0:
        raise LossConfigError("tau must not be negative")
    norms = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return weight / norms.pow(tau)


def effective_number_weights(counts: np.ndarray, beta: float) -> np.ndarray:
    """Return mean-normalized effective-number class weights."""

    if not 0.0 < beta < 1.0:
        raise LossConfigError("effective-number beta must be in (0,1)")
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.clip(effective, 1e-12, None)
    return weights / weights.mean()


LOSS_REGISTRY: dict[str, Callable[[LossConfig], BalancedLoss]] = {
    "L0_STANDARD_CROSS_ENTROPY": StandardCrossEntropy,
    "L1_WEIGHTED_CROSS_ENTROPY": WeightedCrossEntropy,
    "L2_EFFECTIVE_NUMBER_CLASS_BALANCED": EffectiveNumberClassBalanced,
    "L3_BALANCED_SOFTMAX": LogitAdjustedCrossEntropy,
    "L3_LOGIT_ADJUSTMENT": LogitAdjustedCrossEntropy,
    "L4_FOCAL_LOSS": FocalLoss,
    "L5_LDAM_DRW": LDAMDeferredReweighting,
}

LOSS_SUPPORT_REGISTRY: dict[str, str] = {
    "L6_CLASSIFIER_RETRAINING_SUPPORT": (
        "configuration + tau-normalization utility; no production cRT is run"
    ),
    "L7_CONTROLLED_SAMPLING_SUPPORT": (
        "natural / class-aware / native-unit-balanced samplers in "
        "training.balanced.sampling"
    ),
}


def build_loss(config: LossConfig) -> BalancedLoss:
    """Build one loss candidate, rejecting support-only registry entries."""

    if config.name in LOSS_SUPPORT_REGISTRY:
        raise LossConfigError(
            f"{config.name} is support tooling, not a loss module: "
            f"{LOSS_SUPPORT_REGISTRY[config.name]}"
        )
    builder = LOSS_REGISTRY.get(config.name)
    if builder is None:
        raise LossConfigError(f"unregistered loss candidate={config.name}")
    return builder(config)


def loss_registry_contract() -> dict[str, Any]:
    """Serialize the registry for a run manifest."""

    return {
        "schema_version": LOSS_SCHEMA_VERSION,
        "candidates": list(LOSS_CANDIDATES),
        "implemented_losses": sorted(LOSS_REGISTRY),
        "support_entries": dict(LOSS_SUPPORT_REGISTRY),
        "weighted_ce_strategies": list(WEIGHTED_CE_STRATEGIES),
        "reductions": list(REDUCTIONS),
        "priors_source": "TRAIN_FOLD_NATIVE_UNITS",
        "globally_hardcoded_beta": False,
    }


def _reduce(per_sample: Tensor, weights: Tensor, reduction: str) -> Tensor:
    if reduction == "sum":
        return (per_sample * weights).sum()
    if reduction == "mean":
        return (per_sample * weights).mean()
    total = weights.sum()
    if float(total) <= 0.0:
        raise LossConfigError(
            "weighted_mean reduction needs a positive total sample weight"
        )
    return (per_sample * weights).sum() / total


def _validate_logits(logits: Tensor, targets: Tensor) -> None:
    if logits.ndim != 2:
        raise LossConfigError(f"logits must be [B,C]; observed {tuple(logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise LossConfigError(
            f"targets must be [B]; observed {tuple(targets.shape)} for logits "
            f"{tuple(logits.shape)}"
        )
    if not bool(torch.isfinite(logits).all()):
        raise LossConfigError("logits contain nonfinite entries")
    if int(targets.min()) < 0 or int(targets.max()) >= logits.shape[1]:
        raise LossConfigError(
            f"targets must be in [0,{logits.shape[1]}); observed "
            f"[{int(targets.min())},{int(targets.max())}]"
        )


def class_weight_report(
    loss: BalancedLoss,
    class_order: Sequence[str],
    *,
    epoch: int | None = None,
) -> dict[str, float]:
    """Return the per-class weight a loss would apply, for the ablation table."""

    weights = loss.class_weight_vector(epoch)
    return {
        str(name): float(value)
        for name, value in zip(class_order, weights.tolist(), strict=True)
    }


__all__ = [
    "LOSS_CANDIDATES",
    "LOSS_REGISTRY",
    "LOSS_SCHEMA_VERSION",
    "LOSS_SUPPORT_REGISTRY",
    "REDUCTIONS",
    "WEIGHTED_CE_STRATEGIES",
    "BalancedLoss",
    "ClassifierRetrainingConfig",
    "EffectiveNumberClassBalanced",
    "FocalLoss",
    "LDAMDeferredReweighting",
    "LogitAdjustedCrossEntropy",
    "LossComponents",
    "LossConfig",
    "LossConfigError",
    "StandardCrossEntropy",
    "WeightedCrossEntropy",
    "build_loss",
    "class_weight_report",
    "effective_number_weights",
    "loss_registry_contract",
    "tau_normalize_classifier",
]

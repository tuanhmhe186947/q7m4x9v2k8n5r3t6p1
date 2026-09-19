"""Balanced-model training utilities: priors, losses, sampling, training mass.

This subpackage is research scaffolding. It never launches production training
and never reads a production run root. Class priors are computable only from
training-fold native temporal units.
"""

from pig_behavior.classification_v2.training.balanced.class_priors import (
    FORBIDDEN_PRIOR_ROLES,
    TRAIN_FOLD_NATIVE_UNIT_ROLE,
    ClassPriorError,
    ClassPriors,
    compute_class_priors,
)
from pig_behavior.classification_v2.training.balanced.losses import (
    LOSS_CANDIDATES,
    LOSS_REGISTRY,
    LOSS_SUPPORT_REGISTRY,
    BalancedLoss,
    ClassifierRetrainingConfig,
    LossComponents,
    LossConfig,
    build_loss,
    loss_registry_contract,
    tau_normalize_classifier,
)
from pig_behavior.classification_v2.training.balanced.sampling import (
    SAMPLING_STRATEGIES,
    SamplingPlan,
    build_sampling_plan,
)
from pig_behavior.classification_v2.training.balanced.training_mass import (
    TRAINING_MASS_STRATEGIES,
    TrainingMassError,
    WindowInventory,
    fixed_windows_per_native_unit,
    per_window_sample_weights,
    training_mass_audit,
)

__all__ = [
    "FORBIDDEN_PRIOR_ROLES",
    "LOSS_CANDIDATES",
    "LOSS_REGISTRY",
    "LOSS_SUPPORT_REGISTRY",
    "SAMPLING_STRATEGIES",
    "TRAINING_MASS_STRATEGIES",
    "TRAIN_FOLD_NATIVE_UNIT_ROLE",
    "BalancedLoss",
    "ClassPriorError",
    "ClassPriors",
    "ClassifierRetrainingConfig",
    "LossComponents",
    "LossConfig",
    "SamplingPlan",
    "TrainingMassError",
    "WindowInventory",
    "build_loss",
    "build_sampling_plan",
    "compute_class_priors",
    "fixed_windows_per_native_unit",
    "loss_registry_contract",
    "per_window_sample_weights",
    "tau_normalize_classifier",
    "training_mass_audit",
]

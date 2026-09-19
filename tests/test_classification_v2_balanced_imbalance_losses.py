"""Loss registry, class-prior provenance, and component separation tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.balanced.class_priors import (
    FORBIDDEN_PRIOR_ROLES,
    TRAIN_FOLD_NATIVE_UNIT_ROLE,
    ClassPriorError,
    compute_class_priors,
)
from pig_behavior.classification_v2.training.balanced.losses import (
    LOSS_CANDIDATES,
    LOSS_REGISTRY,
    LOSS_SUPPORT_REGISTRY,
    ClassifierRetrainingConfig,
    LossConfig,
    LossConfigError,
    build_loss,
    class_weight_report,
    loss_registry_contract,
    tau_normalize_classifier,
)

NUM_CLASSES = len(VALID_BEHAVIORS)


def _imbalanced_priors():
    counts = [200, 150, 80, 60, 40, 30, 20, 10, 6, 4]
    ids: list[str] = []
    labels: list[int] = []
    for index, count in enumerate(counts):
        for item in range(count):
            ids.append(f"unit_{index}_{item}")
            labels.append(index)
    return compute_class_priors(
        native_unit_ids=ids,
        native_unit_labels=labels,
        fold_id="FOLD_1",
    )


def _logits_and_targets(batch: int = 12, *, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(batch, NUM_CLASSES, generator=generator)
    if requires_grad:
        logits.requires_grad_(True)
    targets = torch.arange(batch, dtype=torch.int64) % NUM_CLASSES
    return logits, targets


def test_eight_candidates_are_declared_and_two_are_support_only() -> None:
    assert len(LOSS_CANDIDATES) == 9  # L3 has two named variants
    assert set(LOSS_SUPPORT_REGISTRY) == {
        "L6_CLASSIFIER_RETRAINING_SUPPORT",
        "L7_CONTROLLED_SAMPLING_SUPPORT",
    }
    assert set(LOSS_REGISTRY) | set(LOSS_SUPPORT_REGISTRY) == set(LOSS_CANDIDATES)
    contract = loss_registry_contract()
    assert contract["priors_source"] == "TRAIN_FOLD_NATIVE_UNITS"
    assert contract["globally_hardcoded_beta"] is False


@pytest.mark.parametrize("name", sorted(LOSS_REGISTRY))
def test_every_loss_returns_separated_components(name: str) -> None:
    priors = _imbalanced_priors()
    config = LossConfig(
        name=name,
        priors=priors,
        weighting_strategy="inverse_sqrt_frequency",
        effective_number_beta=0.999,
        tau=1.0,
        focal_gamma=2.0,
        ldam_max_margin=0.5,
        drw_start_epoch=2,
        drw_beta=0.999,
    )
    loss = build_loss(config)
    logits, targets = _logits_and_targets(requires_grad=True)
    mass = torch.full((logits.shape[0],), 0.5)
    components = loss(logits, targets, native_unit_mass_weight=mass, epoch=5)

    assert components.unreduced_per_sample_loss.shape == (logits.shape[0],)
    assert components.class_weight.shape == (logits.shape[0],)
    assert components.native_unit_mass_weight.shape == (logits.shape[0],)
    assert torch.allclose(
        components.final_sample_weight,
        components.class_weight * components.native_unit_mass_weight,
    )
    assert bool(torch.isfinite(components.reduced_loss))
    components.reduced_loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())


def test_priors_may_only_come_from_train_fold_native_units() -> None:
    for role in FORBIDDEN_PRIOR_ROLES:
        with pytest.raises(ClassPriorError):
            compute_class_priors(
                native_unit_ids=["a", "b"],
                native_unit_labels=[0, 1],
                role=role,
            )
    with pytest.raises(ClassPriorError):
        compute_class_priors(
            native_unit_ids=["duplicate", "duplicate"],
            native_unit_labels=[0, 1],
        )
    priors = _imbalanced_priors()
    assert priors.role == TRAIN_FOLD_NATIVE_UNIT_ROLE
    assert priors.native_unit_count == 600


def test_weighted_ce_strategies_are_bounded_and_ordered() -> None:
    priors = _imbalanced_priors()
    weights = {}
    for strategy in ("inverse_frequency", "inverse_sqrt_frequency"):
        loss = build_loss(
            LossConfig(
                name="L1_WEIGHTED_CROSS_ENTROPY",
                priors=priors,
                weighting_strategy=strategy,
            )
        )
        weights[strategy] = class_weight_report(loss, VALID_BEHAVIORS)
    rare = VALID_BEHAVIORS[-1]
    common = VALID_BEHAVIORS[0]
    for strategy, report in weights.items():
        assert report[rare] > report[common], strategy
    assert (
        weights["inverse_frequency"][rare] > weights["inverse_sqrt_frequency"][rare]
    )

    capped = build_loss(
        LossConfig(
            name="L1_WEIGHTED_CROSS_ENTROPY",
            priors=priors,
            weighting_strategy="capped_inverse_frequency",
            weight_cap=5.0,
        )
    )
    capped_report = class_weight_report(capped, VALID_BEHAVIORS)
    assert max(capped_report.values()) < weights["inverse_frequency"][rare]

    with pytest.raises(LossConfigError):
        build_loss(
            LossConfig(name="L1_WEIGHTED_CROSS_ENTROPY", priors=priors)
        )


def test_effective_number_requires_an_explicit_beta() -> None:
    priors = _imbalanced_priors()
    with pytest.raises(LossConfigError):
        build_loss(
            LossConfig(name="L2_EFFECTIVE_NUMBER_CLASS_BALANCED", priors=priors)
        )
    reports = {}
    for beta in (0.99, 0.999, 0.9999):
        loss = build_loss(
            LossConfig(
                name="L2_EFFECTIVE_NUMBER_CLASS_BALANCED",
                priors=priors,
                effective_number_beta=beta,
            )
        )
        reports[beta] = class_weight_report(loss, VALID_BEHAVIORS)
    rare = VALID_BEHAVIORS[-1]
    assert reports[0.9999][rare] > reports[0.99][rare]


def test_balanced_softmax_is_logit_adjustment_at_tau_one() -> None:
    priors = _imbalanced_priors()
    logits, targets = _logits_and_targets()
    balanced = build_loss(
        LossConfig(name="L3_BALANCED_SOFTMAX", priors=priors)
    )
    adjusted = build_loss(
        LossConfig(name="L3_LOGIT_ADJUSTMENT", priors=priors, tau=1.0)
    )
    assert torch.allclose(
        balanced(logits, targets).reduced_loss,
        adjusted(logits, targets).reduced_loss,
    )
    stronger = build_loss(
        LossConfig(name="L3_LOGIT_ADJUSTMENT", priors=priors, tau=2.0)
    )
    assert not torch.allclose(
        balanced(logits, targets).reduced_loss,
        stronger(logits, targets).reduced_loss,
    )
    with pytest.raises(LossConfigError):
        build_loss(LossConfig(name="L3_LOGIT_ADJUSTMENT", priors=priors))


def test_focal_alpha_is_only_used_when_explicitly_configured() -> None:
    priors = _imbalanced_priors()
    plain = build_loss(
        LossConfig(name="L4_FOCAL_LOSS", priors=priors, focal_gamma=2.0)
    )
    assert not plain.uses_alpha
    assert set(class_weight_report(plain, VALID_BEHAVIORS).values()) == {1.0}

    alpha = tuple(np.linspace(0.5, 2.0, NUM_CLASSES).tolist())
    weighted = build_loss(
        LossConfig(
            name="L4_FOCAL_LOSS",
            priors=priors,
            focal_gamma=1.0,
            focal_alpha=alpha,
        )
    )
    assert weighted.uses_alpha
    with pytest.raises(LossConfigError):
        build_loss(LossConfig(name="L4_FOCAL_LOSS", priors=priors))


def test_ldam_drw_schedule_switches_weights_at_the_declared_epoch() -> None:
    priors = _imbalanced_priors()
    loss = build_loss(
        LossConfig(
            name="L5_LDAM_DRW",
            priors=priors,
            ldam_max_margin=0.5,
            drw_start_epoch=3,
            drw_beta=0.999,
        )
    )
    before = class_weight_report(loss, VALID_BEHAVIORS, epoch=2)
    after = class_weight_report(loss, VALID_BEHAVIORS, epoch=3)
    assert set(before.values()) == {1.0}
    assert after[VALID_BEHAVIORS[-1]] > after[VALID_BEHAVIORS[0]]
    logits, targets = _logits_and_targets()
    with pytest.raises(LossConfigError):
        loss(logits, targets)


def test_support_entries_are_not_buildable_as_losses() -> None:
    for name in LOSS_SUPPORT_REGISTRY:
        with pytest.raises(LossConfigError):
            build_loss(LossConfig(name=name))


def test_crt_config_and_tau_normalization() -> None:
    config = ClassifierRetrainingConfig(tau_normalization_tau=0.75)
    payload = config.to_payload()
    assert payload["production_crt_executed"] is False

    weight = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    normalized = tau_normalize_classifier(weight, 1.0)
    assert torch.allclose(normalized.norm(dim=1), torch.ones(2))
    unchanged = tau_normalize_classifier(weight, 0.0)
    assert torch.allclose(unchanged, weight)


def test_nonfinite_logits_are_rejected() -> None:
    loss = build_loss(LossConfig(name="L0_STANDARD_CROSS_ENTROPY"))
    logits, targets = _logits_and_targets()
    logits[0, 0] = float("inf")
    with pytest.raises(LossConfigError):
        loss(logits, targets)

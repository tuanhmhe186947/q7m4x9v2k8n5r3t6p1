"""Causality tests: FUTURE_FRAME_DEPENDENCE must be zero."""

from __future__ import annotations

import pytest
import torch

from pig_behavior.classification_v2.models.balanced.contracts import (
    BatchContract,
    ModelBatch,
    SequenceSegment,
    validate_batch,
)
from pig_behavior.classification_v2.models.balanced.registry import build_model
from pig_behavior.classification_v2.models.balanced.synthetic import (
    SyntheticBatchSpec,
    perturb_padded_slots,
    synthetic_batch,
)
from pig_behavior.classification_v2.models.balanced.temporal import (
    CAUSAL_ENCODER_NAMES,
    CausalTemporalConfig,
    build_causal_temporal_encoder,
    causal_attention_mask,
    endpoint_index,
)

B3 = "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION"


@pytest.mark.parametrize("encoder", CAUSAL_ENCODER_NAMES)
def test_future_positions_cannot_change_the_endpoint_output(encoder: str) -> None:
    torch.manual_seed(0)
    config = CausalTemporalConfig(name=encoder, hidden_dim=16, layers=2, heads=4)
    module = build_causal_temporal_encoder(config).eval()
    values = torch.randn(2, 8, 16)
    valid = torch.zeros(2, 8, dtype=torch.bool)
    valid[:, :5] = True

    with torch.no_grad():
        reference = module(values, valid)
        future = values.clone()
        future[:, 5:] = future[:, 5:] + 13.7
        perturbed = module(future, valid)
    assert torch.allclose(reference, perturbed, atol=1e-6)


@pytest.mark.parametrize("encoder", CAUSAL_ENCODER_NAMES)
def test_padding_does_not_alter_valid_output(encoder: str) -> None:
    torch.manual_seed(1)
    config = CausalTemporalConfig(name=encoder, hidden_dim=16, layers=2, heads=4)
    module = build_causal_temporal_encoder(config).eval()
    values = torch.randn(3, 6, 16)
    full = torch.ones(3, 6, dtype=torch.bool)

    with torch.no_grad():
        unpadded = module(values[:, :4], full[:, :4])
        padded_values = torch.cat([values[:, :4], torch.randn(3, 2, 16)], dim=1)
        padded_mask = torch.zeros(3, 6, dtype=torch.bool)
        padded_mask[:, :4] = True
        padded = module(padded_values, padded_mask)
    assert torch.allclose(unpadded, padded, atol=1e-6)


def test_endpoint_semantics_use_the_last_valid_slot() -> None:
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    assert endpoint_index(valid).tolist() == [1, 2]


def test_causal_attention_mask_is_strictly_triangular() -> None:
    mask = causal_attention_mask(5)
    assert mask.shape == (5, 5)
    future = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
    assert bool((mask[future] == float("-inf")).all())
    assert bool((mask[~future] == 0.0).all())


def test_model_prediction_is_invariant_to_padded_slot_content() -> None:
    model = build_model(B3, hidden_dim=32).eval()
    contract = model.config.batch_contract
    batch = synthetic_batch(
        SyntheticBatchSpec(
            contract=contract,
            batch_size=3,
            image_size=16,
            valid_lengths=(4, 5, 6),
        )
    )
    perturbed = perturb_padded_slots(batch)
    with torch.no_grad():
        assert torch.allclose(
            model(batch)["logits"],
            model(perturbed)["logits"],
            atol=1e-6,
        )


def test_validator_rejects_a_target_frame_after_the_prediction_endpoint() -> None:
    contract = BatchContract(required_modalities=("actor_images",), target_length=4)
    base = synthetic_batch(SyntheticBatchSpec(contract=contract, batch_size=2))
    offsets = base.target.frame_offsets.clone()
    offsets[0, -1] = 1
    broken = ModelBatch(
        target=SequenceSegment(
            valid_mask=base.target.valid_mask,
            frame_offsets=offsets,
            images=base.target.images,
            numeric_groups=base.target.numeric_groups,
            quality_mask=base.target.quality_mask,
        ),
        numeric_feature_names=base.numeric_feature_names,
        quality_mask_names=base.quality_mask_names,
        modality_availability=base.modality_availability,
        labels=base.labels,
        native_unit_id=base.native_unit_id,
        window_id=base.window_id,
    )
    check = validate_batch(broken, contract).check("TARGET_LENGTH_CONTRACT")
    assert not check.passed
    assert any("after the prediction" in error for error in check.errors)


def test_future_frame_dependence_is_zero_for_every_baseline() -> None:
    dependence = 0
    for name in ("B1_ACTOR_T6_SEQUENCE", "B2_ACTOR_T6_PLUS_GEOMETRY", B3):
        model = build_model(name, hidden_dim=16).eval()
        batch = synthetic_batch(
            SyntheticBatchSpec(
                contract=model.config.batch_contract,
                batch_size=2,
                image_size=16,
                valid_lengths=(3, 6),
            )
        )
        with torch.no_grad():
            reference = model(batch)["logits"]
            changed = model(perturb_padded_slots(batch, value=99.0))["logits"]
        dependence += int(not torch.allclose(reference, changed, atol=1e-6))
    assert dependence == 0

"""Forward/backward and registry tests for the B0-B3 baseline ladder."""

from __future__ import annotations

import pytest
import torch

from pig_behavior.classification_v2.models.balanced.baselines import (
    BASELINE_NAMES,
    BASELINE_NUMERIC_GROUPS,
    baseline_config,
    baseline_contract,
)
from pig_behavior.classification_v2.models.balanced.fusion import (
    FusionConfig,
    FusionExtensionPointError,
    MultimodalFusion,
    require_extension_point,
)
from pig_behavior.classification_v2.models.balanced.registry import (
    BALANCED_MAIN_MODEL_NAME,
    BALANCED_MODEL_NAMES,
    build_model,
    model_spec_contract,
    registry_contract,
)
from pig_behavior.classification_v2.models.balanced.synthetic import (
    SyntheticBatchSpec,
    synthetic_batch,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_forward_and_backward(name: str) -> None:
    model = build_model(name, hidden_dim=32)
    contract = model.config.batch_contract
    batch = synthetic_batch(
        SyntheticBatchSpec(contract=contract, batch_size=3, image_size=16)
    )
    outputs = model(batch)
    logits = outputs["logits"]
    assert logits.shape == (3, len(VALID_BEHAVIORS))
    assert bool(torch.isfinite(logits).all())

    loss = torch.nn.functional.cross_entropy(logits, batch.labels)
    assert bool(torch.isfinite(loss))
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(grad).all()) for grad in gradients)


def test_baseline_ladder_adds_one_thing_at_a_time() -> None:
    assert BASELINE_NUMERIC_GROUPS["B0_ACTOR_SINGLE_FRAME"] == ()
    assert BASELINE_NUMERIC_GROUPS["B1_ACTOR_T6_SEQUENCE"] == ()
    assert BASELINE_NUMERIC_GROUPS["B2_ACTOR_T6_PLUS_GEOMETRY"] == (
        "bbox_xywh_n",
        "bbox_shape_n",
    )
    assert BASELINE_NUMERIC_GROUPS["B3_ACTOR_T6_PLUS_GEOMETRY_MOTION"] == (
        "bbox_xywh_n",
        "bbox_shape_n",
        "motion_delta",
    )
    for name in BASELINE_NAMES:
        contract = baseline_contract(name)
        assert contract["uses_roi_relation"] is False
        assert contract["uses_social_relation"] is False
        assert contract["uses_causal_history"] is False
        assert contract["uses_gated_fusion"] is False
        assert contract["model_config"]["history_length"] == 0


def test_b0_is_single_frame_and_b1_is_a_sequence() -> None:
    assert baseline_config("B0_ACTOR_SINGLE_FRAME").batch_contract.target_length == 1
    assert baseline_config("B0_ACTOR_SINGLE_FRAME").temporal is None
    b1 = baseline_config("B1_ACTOR_T6_SEQUENCE")
    assert b1.batch_contract.target_length == 6
    assert b1.temporal is not None
    assert b1.temporal.name.startswith("causal_")


def test_geometry_is_encoded_not_raw_concatenated() -> None:
    model = build_model("B2_ACTOR_T6_PLUS_GEOMETRY", hidden_dim=32)
    assert model.numeric_encoder is not None
    encoders = dict(model.numeric_encoder.encoders.items())
    assert set(encoders) == {"bbox_xywh_n", "bbox_shape_n"}
    for module in encoders.values():
        kinds = {type(layer).__name__ for layer in module}
        assert "Linear" in kinds
        assert "LayerNorm" in kinds


@pytest.mark.parametrize("length", [6, 8, 12, 16])
def test_baselines_re_instantiate_at_other_target_lengths(length: int) -> None:
    model = build_model("B3_ACTOR_T6_PLUS_GEOMETRY_MOTION", target_length=length, hidden_dim=32)
    batch = synthetic_batch(
        SyntheticBatchSpec(
            contract=model.config.batch_contract,
            batch_size=2,
            image_size=16,
        )
    )
    assert model(batch)["logits"].shape == (2, len(VALID_BEHAVIORS))


def test_registry_declares_the_unbuilt_main_model() -> None:
    assert set(BASELINE_NAMES).issubset(set(BALANCED_MODEL_NAMES))
    payload = model_spec_contract(BALANCED_MAIN_MODEL_NAME)
    assert payload["implemented"] is False
    assert payload["pending_modules"]
    with pytest.raises(FusionExtensionPointError):
        build_model(BALANCED_MAIN_MODEL_NAME)
    contract = registry_contract()
    assert contract["declared_extension_points"]


def test_extension_points_fail_loudly_instead_of_degrading() -> None:
    with pytest.raises(FusionExtensionPointError):
        MultimodalFusion(
            FusionConfig(mode="quality_aware_gated"),
            branch_dims={"visual": 8},
        )
    with pytest.raises(FusionExtensionPointError):
        require_extension_point("roi_conditioned_film")


def test_eval_mode_is_deterministic() -> None:
    model = build_model("B3_ACTOR_T6_PLUS_GEOMETRY_MOTION", hidden_dim=32, dropout=0.3)
    batch = synthetic_batch(
        SyntheticBatchSpec(
            contract=model.config.batch_contract,
            batch_size=2,
            image_size=16,
        )
    )
    model.eval()
    with torch.no_grad():
        first = model(batch)["logits"]
        second = model(batch)["logits"]
    assert torch.equal(first, second)

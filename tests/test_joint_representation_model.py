"""Focused CPU model gates for source-aware branch FiLM."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from pig_behavior.classification_v2.models.joint_representation_model import (
    H5_LATENT_DIM,
    PARTNER_LATENT_DIM,
    ROI_LATENT_DIM,
    JointRepresentationClassifier,
    OrderedH5Encoder,
    OrderedROIEncoder,
    ZeroInitFiLM,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    M2BranchEmbeddings,
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
)

EXPECTED_M2_PARAM_COUNT = 43_633_832
EXPECTED_FINAL_PARAM_COUNT = 43_719_531
EXPECTED_PARAM_DELTA = 85_699


def _config(*, backbone_name: str = "smoke_cnn") -> MultimodalFusionConfig:
    return MultimodalFusionConfig(
        backbone_name=backbone_name,
        pretrained_weight_enum=NO_PRETRAINED_WEIGHTS,
        temporal_encoder_name="masked_tcn",
        image_embedding_dim=128,
        spatial_embedding_dim=128,
        interaction_embedding_dim=64,
        visual_context_embedding_dim=128,
        fusion_hidden_dim=256,
        num_classes=10,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        interaction_context_dim=5,
        spatial_input_dims={
            "bbox_xywh_n": 4,
            "bbox_shape_n": 2,
            "motion_delta": 12,
            "roi_class_relation": 18,
            "social_relation": 10,
        },
        dropout=0.0,
        transformer_layers=2,
        transformer_heads=4,
    )


def _m2_inputs(batch_size: int = 3) -> dict[str, object]:
    torch.manual_seed(100)
    steps = 6
    length_mask = torch.ones(batch_size, steps, dtype=torch.bool)
    time_delta = (
        torch.arange(steps, dtype=torch.float32)
        .unsqueeze(0)
        .expand(batch_size, -1)
        / 30.0
    )
    return {
        "image": torch.rand(batch_size, steps, 3, 32, 32),
        "spatial_features": {
            "bbox_xywh_n": torch.rand(batch_size, steps, 4),
            "bbox_shape_n": torch.rand(batch_size, steps, 2),
            "motion_delta": torch.rand(batch_size, steps, 12),
            "roi_class_relation": torch.rand(batch_size, steps, 18),
            "social_relation": torch.rand(batch_size, steps, 10),
        },
        "spatial_feature_validity_masks": {
            "motion_delta": torch.ones(
                batch_size,
                steps,
                12,
                dtype=torch.bool,
            ),
            "social_relation": torch.ones(
                batch_size,
                steps,
                10,
                dtype=torch.bool,
            ),
        },
        "length_mask": length_mask,
        "observed_mask": length_mask,
        "image_time_delta": time_delta,
        "spatial_time_delta": time_delta,
        "interaction_context_features": torch.rand(batch_size, 5),
        "interaction_context_available_mask": torch.ones(
            batch_size,
            dtype=torch.bool,
        ),
        "visual_context_image": torch.rand(
            batch_size,
            steps,
            3,
            32,
            32,
        ),
        "visual_context_length_mask": length_mask,
        "visual_context_observed_mask": length_mask,
        "visual_context_time_delta": time_delta,
    }


def _source_inputs(batch_size: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(200)
    return {
        "partner_behavior_probs": torch.softmax(
            torch.randn(batch_size, 2, 10),
            dim=-1,
        ),
        "partner_behavior_hidden": torch.randn(batch_size, 2, 256),
        "partner_behavior_mask": torch.ones(
            batch_size,
            2,
            dtype=torch.bool,
        ),
        "h5_structured": torch.randn(batch_size, 5, 46),
        "h5_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "roi_sequence": torch.randn(batch_size, 6, 18),
        "roi_validity": torch.ones(
            batch_size,
            6,
            3,
            dtype=torch.bool,
        ),
    }


def _grad_l1(module: torch.nn.Module) -> float:
    return float(
        sum(
            parameter.grad.abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )


def _max_branch_diff(
    left: M2BranchEmbeddings,
    right: M2BranchEmbeddings,
) -> float:
    return max(
        (left.actor128 - right.actor128).abs().max().item(),
        (left.structured128 - right.structured128).abs().max().item(),
        (left.interaction64 - right.interaction64).abs().max().item(),
        (left.union128 - right.union128).abs().max().item(),
    )


def test_parameter_counts() -> None:
    config = _config(backbone_name="resnet34")
    config.temporal_encoder_name = "small_transformer"
    base = MultimodalFusionClassifier(config)
    final = JointRepresentationClassifier(config)
    base_count = sum(parameter.numel() for parameter in base.parameters())
    final_count = sum(parameter.numel() for parameter in final.parameters())

    assert base_count == EXPECTED_M2_PARAM_COUNT
    assert final_count == EXPECTED_FINAL_PARAM_COUNT
    assert final_count - base_count == EXPECTED_PARAM_DELTA


def test_step0_exact_branch_and_logit_parity() -> None:
    config = _config()
    config.temporal_encoder_name = "small_transformer"
    inputs = _m2_inputs()
    sources = _source_inputs()
    torch.manual_seed(42)
    base = MultimodalFusionClassifier(config).eval()
    torch.manual_seed(42)
    final = JointRepresentationClassifier(config).eval()
    final.load_state_dict(base.state_dict(), strict=False)

    with torch.inference_mode():
        base_branches = base.encode_fused(
            **inputs,
            return_m2_branches=True,
        )
        base_logits = base(**inputs)
        output = final(**inputs, **sources)

    assert isinstance(base_branches, M2BranchEmbeddings)
    assert _max_branch_diff(output.m2_branches, base_branches) <= 1e-7
    assert _max_branch_diff(output.modulated_branches, base_branches) <= 1e-7
    assert (output.behavior_logits - base_logits).abs().max().item() <= 1e-7


def test_joint_forward_encodes_each_m2_branch_exactly_once() -> None:
    model = JointRepresentationClassifier(_config()).eval()
    assert model.image_encoder is not None
    assert model.spatial_encoder is not None
    assert model.interaction_context_encoder is not None
    assert model.visual_context_encoder is not None
    modules = {
        "actor128": model.image_encoder,
        "structured128": model.spatial_encoder,
        "interaction64": model.interaction_context_encoder,
        "union128": model.visual_context_encoder,
    }
    counts = dict.fromkeys(modules, 0)
    handles = []
    for name, module in modules.items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _args, _output, branch_name=name: counts.__setitem__(
                    branch_name,
                    counts[branch_name] + 1,
                )
            )
        )
    try:
        with torch.inference_mode():
            model(**_m2_inputs(), **_source_inputs())
    finally:
        for handle in handles:
            handle.remove()

    assert counts == dict.fromkeys(modules, 1)


def test_no_m2_only_new_capacity_path_and_exact_zero_init() -> None:
    model = JointRepresentationClassifier(_config())
    assert not hasattr(model, "joint_adapter")
    assert not hasattr(model, "delta_proj")
    for method_name in (
        "_encode_partner_context",
        "_encode_h5_context",
        "_encode_roi_context",
    ):
        parameters = inspect.signature(getattr(model, method_name)).parameters
        assert "reference" not in parameters
    assert set(model.film) == {
        "h5_actor",
        "h5_structured",
        "roi_actor",
        "roi_structured",
        "posture_actor",
        "posture_structured",
        "pb_interaction",
        "pb_union",
        "pb_hidden_interaction",
        "pb_hidden_union",
    }
    for film in model.film.values():
        assert isinstance(film, ZeroInitFiLM)
        assert film.projection.bias is None
        assert torch.count_nonzero(film.projection.weight).item() == 0
        assert film.projection.in_features in {
            H5_LATENT_DIM,
            ROI_LATENT_DIM,
            PARTNER_LATENT_DIM,
            3,
        }
    assert "posture_reviewed_mask" not in inspect.signature(model.forward).parameters


def test_zero_init_gradient_staging_across_two_backwards() -> None:
    model = JointRepresentationClassifier(_config()).train()
    inputs = _m2_inputs()
    sources = _source_inputs()
    targets = torch.tensor([0, 1, 2])
    optimizer = torch.optim.SGD(model.film.parameters(), lr=0.1)

    first = model(**inputs, **sources)
    F.cross_entropy(first.behavior_logits, targets).backward()

    for film in model.film.values():
        assert _grad_l1(film) > 0.0
    assert _grad_l1(model.partner_behavior_encoder) == 0.0
    assert _grad_l1(model.partner_hidden_encoder) == 0.0
    assert _grad_l1(model.h5_encoder) == 0.0
    assert _grad_l1(model.roi_encoder) == 0.0
    assert _grad_l1(model.posture_predictor) == 0.0

    optimizer.step()
    model.zero_grad(set_to_none=True)

    second = model(**inputs, **sources)
    F.cross_entropy(second.behavior_logits, targets).backward()

    assert _grad_l1(model.partner_behavior_encoder) > 0.0
    assert _grad_l1(model.partner_hidden_encoder) > 0.0
    assert _grad_l1(model.h5_encoder) > 0.0
    assert _grad_l1(model.roi_encoder) > 0.0
    assert _grad_l1(model.posture_predictor) == 0.0


def test_posture_gradient_isolation() -> None:
    inputs = _m2_inputs()
    sources = _source_inputs()

    posture_model = JointRepresentationClassifier(_config()).train()
    posture_output = posture_model(**inputs, **sources)
    F.cross_entropy(
        posture_output.posture_logits,
        torch.tensor([0, 1, 2]),
    ).backward()
    assert _grad_l1(posture_model.posture_predictor) > 0.0
    assert posture_model.image_encoder is not None
    assert _grad_l1(posture_model.image_encoder) == 0.0

    behavior_model = JointRepresentationClassifier(_config()).train()
    behavior_output = behavior_model(**inputs, **sources)
    F.cross_entropy(
        behavior_output.behavior_logits,
        torch.tensor([0, 1, 2]),
    ).backward()
    assert _grad_l1(behavior_model.posture_predictor) == 0.0


def test_missing_sources_stay_exact_zero_after_film_update() -> None:
    model = JointRepresentationClassifier(_config()).eval()
    with torch.no_grad():
        for name in (
            "h5_actor",
            "h5_structured",
            "roi_actor",
            "roi_structured",
            "pb_interaction",
            "pb_union",
            "pb_hidden_interaction",
            "pb_hidden_union",
        ):
            model.film[name].projection.weight.fill_(0.125)

    sources = _source_inputs()
    sources["partner_behavior_mask"].zero_()
    sources["h5_mask"].zero_()
    sources["roi_validity"].zero_()
    output = model(**_m2_inputs(), **sources)

    assert torch.count_nonzero(output.partner_context).item() == 0
    assert torch.count_nonzero(output.partner_hidden_context).item() == 0
    assert torch.count_nonzero(output.h5_context).item() == 0
    assert torch.count_nonzero(output.roi_context).item() == 0
    for name in (
        "h5_actor",
        "h5_structured",
        "roi_actor",
        "roi_structured",
        "pb_interaction",
        "pb_union",
        "pb_hidden_interaction",
        "pb_hidden_union",
    ):
        assert torch.count_nonzero(output.source_modulations[name]).item() == 0


def test_pb_k2_mask_semantics_are_preserved() -> None:
    model = JointRepresentationClassifier(_config()).eval()
    inputs = _m2_inputs()
    sources = _source_inputs()
    mask = sources["partner_behavior_mask"]
    mask[0, 1] = False
    mask[2].zero_()

    with torch.inference_mode():
        first = model(**inputs, **sources).partner_context
        changed = sources["partner_behavior_probs"].clone()
        changed[0, 1] = 1_000.0
        second = model(
            **inputs,
            **{**sources, "partner_behavior_probs": changed},
        ).partner_context

    assert torch.equal(first, second)
    assert torch.count_nonzero(first[2]).item() == 0
    with pytest.raises(ValueError, match="shape"):
        model(
            **inputs,
            **{
                **sources,
                "partner_behavior_probs": torch.rand(3, 3, 10),
                "partner_behavior_mask": torch.ones(3, 3),
            },
        )


def test_h5_order_and_mask_semantics_are_preserved() -> None:
    torch.manual_seed(300)
    encoder = OrderedH5Encoder(dropout=0.0).eval()
    value = torch.randn(2, 5, 46)
    mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )

    with torch.inference_mode():
        original = encoder(value, mask)
        changed = value.clone()
        changed[0, 4] = 1_000.0
        masked_change = encoder(changed, mask)
        reversed_order = encoder(value.flip(1), mask.flip(1))

    assert torch.equal(original[0], masked_change[0])
    assert not torch.allclose(original, reversed_order)
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.randn(2, 4, 46), torch.ones(2, 4))


def test_roi_order_and_validity_are_preserved_without_temporal_mean() -> None:
    torch.manual_seed(400)
    encoder = OrderedROIEncoder(dropout=0.0).eval()
    value = torch.randn(2, 6, 18)
    validity = torch.ones(2, 6, 3, dtype=torch.bool)

    with torch.inference_mode():
        original = encoder(value, validity)
        swapped = value.clone()
        swapped[:, [0, 1]] = swapped[:, [1, 0]]
        reordered = encoder(swapped, validity)
        changed_validity = validity.clone()
        changed_validity[:, 0].zero_()
        validity_output = encoder(value, changed_validity)

    assert not torch.allclose(original, reordered)
    assert not torch.allclose(original, validity_output)
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.randn(2, 5, 18), torch.ones(2, 5, 3))


def test_toy_contact_neutralized_while_preserving_other_roi_features() -> None:
    """Prove roi_toy_contact (idx 17) is neutralized while other 17 features are active."""
    torch.manual_seed(500)
    encoder = OrderedROIEncoder(dropout=0.0).eval()
    validity = torch.ones(2, 6, 3, dtype=torch.bool)

    base_val = torch.randn(2, 6, 18)
    contact_active = base_val.clone()
    contact_active[:, :, 17] = 1.0
    contact_zero = base_val.clone()
    contact_zero[:, :, 17] = 0.0

    with torch.inference_mode():
        out_active = encoder(contact_active, validity)
        out_zero = encoder(contact_zero, validity)

    # Must be bit-for-bit identical since toy_contact is sanitized to 0.0
    assert torch.equal(out_active, out_zero)

    # But changing toy geometry (e.g. idx 12 toy_min_dist_n, idx 16 toy_near) must change output
    geom_changed = base_val.clone()
    geom_changed[:, :, 12] += 2.0
    with torch.inference_mode():
        out_geom = encoder(geom_changed, validity)
    assert not torch.allclose(out_zero, out_geom)

    # Changing feeder (idx 5 contact) or drinker (idx 11 contact) must remain active
    feeder_changed = base_val.clone()
    feeder_changed[:, :, 5] += 1.0
    with torch.inference_mode():
        out_feeder = encoder(feeder_changed, validity)
    assert not torch.allclose(out_zero, out_feeder)


def test_toy_contact_neutralized_in_ordered_h5_encoder() -> None:
    """Prove toy_contact (idx 35 in 46D) is neutralized in H5 while other features are active."""
    torch.manual_seed(600)
    encoder = OrderedH5Encoder(dropout=0.0).eval()
    mask = torch.ones(2, 5, dtype=torch.bool)

    base_val = torch.randn(2, 5, 46)
    contact_active = base_val.clone()
    contact_active[:, :, 35] = 1.0
    contact_zero = base_val.clone()
    contact_zero[:, :, 35] = 0.0

    with torch.inference_mode():
        out_active = encoder(contact_active, mask)
        out_zero = encoder(contact_zero, mask)

    assert torch.equal(out_active, out_zero)

    # But changing toy geometry (idx 30) or feeder/drinker must change output
    toy_geom = base_val.clone()
    toy_geom[:, :, 30] += 2.0
    with torch.inference_mode():
        out_toy_geom = encoder(toy_geom, mask)
    assert not torch.allclose(out_zero, out_toy_geom)

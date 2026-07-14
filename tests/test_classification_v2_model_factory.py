from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pig_behavior.classification_v2.models.model_factory import (
    MODEL_MODE_NAMES,
    build_multimodal_model,
    model_mode_contract,
    model_mode_spec,
    model_parameter_report,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    AvailabilityEncoder,
    PartnerSetEncoder,
)
from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.models.temporal_encoders import (
    TEMPORAL_ENCODER_NAMES,
    build_temporal_encoder,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    IMAGENET_RGB_MEAN,
    NO_PRETRAINED_WEIGHTS,
    visual_backbone_contract,
)
from pig_behavior.classification_v2.training.config import (
    ModelConfig,
    load_training_config,
    training_config_to_jsonable,
)

REQUIRED_MODEL_MODES = {
    "actor_only",
    "actor_temporal",
    "actor_geometry",
    "actor_geometry_roi",
    "actor_geometry_roi_social",
    "actor_partner_union",
    "full_multimodal",
    "full_multimodal_hierarchy",
}
GROUP_DIMS = {
    "bbox_xywh_n": 4,
    "bbox_shape_n": 3,
    "motion_delta": 5,
    "roi_class_relation": 6,
    "social_relation": 7,
    "quality_mask": 2,
}


def test_registry_contains_every_required_model_mode() -> None:
    assert REQUIRED_MODEL_MODES.issubset(MODEL_MODE_NAMES)
    assert "spatial_only_control" in MODEL_MODE_NAMES
    assert "actor_geometry_motion" in MODEL_MODE_NAMES


@pytest.mark.parametrize("mode", sorted(MODEL_MODE_NAMES))
def test_every_model_mode_returns_exact_ten_class_logits(mode: str) -> None:
    config = _model_config(mode)
    model = _build(config).eval()

    output = model(**_inputs(config))

    assert output.behavior.shape == (2, 10)
    assert torch.isfinite(output.behavior).all()
    expected_auxiliary_widths = {
        "posture": 3,
        "motion_context": 4,
        "roi_intent": 4,
        "interaction": 3,
    }
    for name, logits in output.auxiliary_logits().items():
        expected = (
            expected_auxiliary_widths[name]
            if config.enable_multitask
            else 0
        )
        assert logits.shape == (2, expected)


@pytest.mark.parametrize("encoder_name", sorted(TEMPORAL_ENCODER_NAMES))
def test_all_temporal_encoders_are_mask_safe(encoder_name: str) -> None:
    torch.manual_seed(3)
    encoder = build_temporal_encoder(
        encoder_name,
        embedding_dim=8,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=2,
    ).eval()
    value = torch.randn(2, 6, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    changed = value.clone()
    changed[mask.eq(0)] = torch.nan
    time_delta = torch.full((2, 6), 0.2)
    time_delta[:, 0] = 0.0
    time_delta[mask.eq(0)] = torch.nan

    kwargs = {"time_delta": time_delta} if encoder_name == "small_transformer" else {}
    original = encoder(value, mask, **kwargs)
    altered = encoder(changed, mask, **kwargs)

    torch.testing.assert_close(original, altered)
    assert original.shape == (2, 8)
    assert torch.isfinite(original).all()


def test_transformer_requires_real_time_delta() -> None:
    encoder = build_temporal_encoder(
        "small_transformer",
        embedding_dim=8,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=2,
    )

    with pytest.raises(ValueError, match="requires real time_delta"):
        encoder(torch.ones(1, 3, 8), torch.ones(1, 3))


def test_masked_modality_values_cannot_change_logits() -> None:
    torch.manual_seed(9)
    config = _model_config("full_multimodal_hierarchy")
    model = _build(config).eval()
    inputs = _inputs(config)
    inputs["image_available_mask"][0, -2:] = 0.0
    inputs["spatial_available_mask"][0, -2:] = 0.0
    inputs["visual_context_available_mask"][0, -2:] = 0.0
    original = model(**inputs).behavior
    changed = _clone_inputs(inputs)
    changed["image"][0, -2:] = torch.nan
    changed["visual_context_image"][0, -2:] = torch.nan
    for value in changed["spatial_features"].values():
        value[0, -2:] = torch.nan
    changed["interaction_context_features"][1] = torch.nan

    altered = model(**changed).behavior

    torch.testing.assert_close(original, altered)


def test_model_runs_when_optional_context_is_fully_missing() -> None:
    config = _model_config("full_multimodal")
    model = _build(config).eval()
    inputs = _inputs(config)
    inputs["interaction_context_available_mask"].zero_()
    inputs["interaction_context_quality_mask"].zero_()
    inputs["interaction_context_features"].fill_(torch.nan)
    inputs["visual_context_available_mask"].zero_()
    inputs["visual_context_quality_mask"].zero_()
    inputs["visual_context_image"].fill_(torch.nan)

    output = model(**inputs)

    assert output.behavior.shape == (2, 10)
    assert torch.isfinite(output.behavior).all()


def test_top_k_partner_encoder_ignores_absent_partner_values() -> None:
    encoder = PartnerSetEncoder(3, 4, 0.0).eval()
    value = torch.rand(2, 3, 3)
    available = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    original = encoder(value, available_mask=available)
    changed = value.clone()
    changed[available.eq(0)] = torch.nan

    altered = encoder(changed, available_mask=available)

    torch.testing.assert_close(original, altered)


def test_availability_encoder_is_parameter_free_gating_only() -> None:
    gate = AvailabilityEncoder()
    embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = gate(embedding, torch.tensor([1.0, 0.0]))

    torch.testing.assert_close(output[0], embedding[0])
    torch.testing.assert_close(output[1], torch.zeros(2))
    assert sum(parameter.numel() for parameter in gate.parameters()) == 0


def test_contradictory_masks_are_rejected() -> None:
    config = _model_config("actor_temporal")
    model = _build(config)
    inputs = _inputs(config)
    inputs["image_observed_mask"][0, -1] = 0.0
    inputs["image_available_mask"][0, -1] = 1.0

    with pytest.raises(ValueError, match="availability is true outside observation"):
        model(**inputs)


def test_mode_contract_rejects_hidden_branch_drift() -> None:
    config = _model_config("actor_temporal")
    drifted = replace(config, enable_visual_context=True)

    with pytest.raises(ValueError, match="model_mode_flag_mismatch"):
        _build(drifted)


def test_actor_only_requires_non_temporal_mean_control() -> None:
    config = replace(
        _model_config("actor_only"),
        temporal_encoder_name="masked_tcn",
    )

    with pytest.raises(ValueError, match="model_mode_temporal_encoder_mismatch"):
        _build(config)


@pytest.mark.parametrize(
    ("backbone_name", "image_size"),
    [("resnet18", 160), ("resnet18", 224), ("resnet34", 224)],
)
def test_production_visual_controls_forward_without_weight_download(
    backbone_name: str,
    image_size: int,
) -> None:
    config = replace(
        _model_config("actor_only"),
        backbone_name=backbone_name,
        pretrained_weight_enum=NO_PRETRAINED_WEIGHTS,
        image_size=image_size,
    )
    model = _build(config).eval()

    with torch.no_grad():
        output = model(
            **_inputs(
                config,
                batch_size=1,
                sequence_length=1,
                image_size=image_size,
            )
        )

    assert output.behavior.shape == (1, 10)
    assert torch.isfinite(output.behavior).all()
    encoder = model.backbone.image_encoder
    assert encoder is not None
    assert encoder.backbone_contract.uses_pretrained_weights is False
    assert encoder.backbone_contract.normalization_name == "imagenet_1k_rgb"
    assert encoder.backbone_contract.input_mean == IMAGENET_RGB_MEAN


def test_pretrained_contract_resolves_exact_enum_without_building_model() -> None:
    contract = visual_backbone_contract(
        "resnet18",
        "ResNet18_Weights.IMAGENET1K_V1",
    )

    assert contract.uses_pretrained_weights is True
    assert contract.pretrained_weight_enum == "ResNet18_Weights.IMAGENET1K_V1"


def test_mismatched_pretrained_enum_is_rejected_before_model_build() -> None:
    config = replace(
        _model_config("actor_temporal"),
        backbone_name="resnet18",
        pretrained_weight_enum="ResNet34_Weights.IMAGENET1K_V1",
    )

    with pytest.raises(ValueError, match="unsupported_pretrained_weight_enum"):
        _build(config)


def test_legacy_config_without_model_mode_is_read_compatibly(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    source = root / "configs" / "classification_v2" / "baseline_actor_image.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["model"].pop("model_mode")
    payload["model"].pop("transformer_layers")
    payload["model"].pop("transformer_heads")
    path = tmp_path / "legacy_actor_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_training_config(path)

    assert config.model.model_mode == "actor_temporal"
    serialized = training_config_to_jsonable(config)
    assert serialized["model"]["model_mode"] == "actor_temporal"


def test_parameter_report_matches_model_and_contract_excludes_availability() -> None:
    config = _model_config("full_multimodal_hierarchy")
    model = _build(config)
    report = model_parameter_report(model)
    contract = model_mode_contract(config.model_mode)

    assert report["total"] == sum(parameter.numel() for parameter in model.parameters())
    assert report["trainable"] == report["total"]
    assert report["by_top_level_module"]
    assert contract["availability_encoded_as_behavior_feature"] is False


def test_one_batch_backward_reaches_enabled_branches_and_heads() -> None:
    config = _model_config("full_multimodal_hierarchy")
    model = _build(config)
    output = model(**_inputs(config))
    loss = output.behavior.square().mean()
    loss = loss + sum(
        logits.square().mean() for logits in output.auxiliary_logits().values()
    )

    loss.backward()

    enabled = [
        model.backbone.image_encoder,
        model.backbone.spatial_encoder,
        model.backbone.interaction_context_encoder,
        model.backbone.visual_context_encoder,
        model.auxiliary_heads,
    ]
    for module in enabled:
        assert module is not None
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def _model_config(mode: str) -> ModelConfig:
    spec = model_mode_spec(mode)
    temporal = "masked_mean" if mode == "actor_only" else "masked_tcn"
    return ModelConfig(
        architecture_version=MULTITASK_ARCHITECTURE_VERSION,
        model_mode=mode,
        temporal_encoder_name=temporal,
        hidden_dim=8,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=2,
        spatial_feature_groups=spec.spatial_feature_groups,
        enable_image=spec.enable_image,
        enable_spatial=spec.enable_spatial,
        enable_interaction_context=spec.enable_interaction_context,
        enable_visual_context=spec.enable_visual_context,
        enable_multitask=spec.enable_multitask,
    )


def _build(config: ModelConfig):
    dims = {name: GROUP_DIMS[name] for name in config.spatial_feature_groups}
    interaction_dim = 5 if config.enable_interaction_context else None
    return build_multimodal_model(
        config,
        spatial_input_dims=dims,
        interaction_context_dim=interaction_dim,
        num_classes=10,
    )


def _inputs(
    config: ModelConfig,
    *,
    batch_size: int = 2,
    sequence_length: int = 6,
    image_size: int = 16,
) -> dict[str, object]:
    length = torch.ones(batch_size, sequence_length)
    observed = length.clone()
    if batch_size > 1:
        observed[1, -1] = 0.0
    time_delta = torch.full((batch_size, sequence_length), 0.2)
    time_delta[:, 0] = 0.0
    return {
        "image": torch.rand(
            batch_size,
            sequence_length,
            3,
            image_size,
            image_size,
        ),
        "spatial_features": {
            name: torch.rand(batch_size, sequence_length, GROUP_DIMS[name])
            for name in config.spatial_feature_groups
        },
        "length_mask": length,
        "observed_mask": observed,
        "image_length_mask": length.clone(),
        "image_observed_mask": observed.clone(),
        "image_available_mask": observed.clone(),
        "image_quality_mask": observed.clone(),
        "image_time_delta": time_delta.clone(),
        "spatial_length_mask": length.clone(),
        "spatial_observed_mask": observed.clone(),
        "spatial_available_mask": observed.clone(),
        "spatial_quality_mask": observed.clone(),
        "spatial_time_delta": time_delta.clone(),
        "interaction_context_features": torch.rand(batch_size, 5),
        "interaction_context_available_mask": torch.tensor([1.0, 0.0]),
        "interaction_context_quality_mask": torch.tensor([1.0, 0.0]),
        "visual_context_image": torch.rand(
            batch_size,
            sequence_length,
            3,
            image_size,
            image_size,
        ),
        "visual_context_length_mask": length.clone(),
        "visual_context_observed_mask": observed.clone(),
        "visual_context_available_mask": observed.clone(),
        "visual_context_quality_mask": observed.clone(),
        "visual_context_time_delta": time_delta.clone(),
    }


def _clone_inputs(inputs: dict[str, object]) -> dict[str, object]:
    cloned: dict[str, object] = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            cloned[name] = value.clone()
        elif isinstance(value, dict):
            cloned[name] = {
                key: tensor.clone() for key, tensor in value.items()
            }
        else:
            cloned[name] = value
    return cloned

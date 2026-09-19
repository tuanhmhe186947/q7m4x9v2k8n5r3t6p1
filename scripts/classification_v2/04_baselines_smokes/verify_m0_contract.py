"""Engineering verification script for the frozen M0 model contract."""

from __future__ import annotations

from pathlib import Path

import torch

from pig_behavior.classification_v2.datasets.interaction_context_loader import (
    INTERACTION_CONTEXT_FEATURE_COLUMNS,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
    SPATIAL_SCHEMA_TOTAL_DIMENSION,
)
from pig_behavior.classification_v2.models.model_factory import (
    build_multimodal_model,
)
from pig_behavior.classification_v2.training.config import (
    load_training_config,
)


def verify_m0() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = (
        repo_root
        / "configs"
        / "classification_v2"
        / "m0_full_multimodal_r34_t6_concat.json"
    )
    print(f"Loading config from: {config_path}")
    training_config = load_training_config(config_path)
    model_cfg = training_config.model

    print("1. Verifying Model Config Properties:")
    assert model_cfg.model_mode == "full_multimodal", (
        f"Expected full_multimodal, got {model_cfg.model_mode}"
    )
    assert model_cfg.backbone_name == "resnet34", (
        f"Expected resnet34, got {model_cfg.backbone_name}"
    )
    assert (
        model_cfg.pretrained_weight_enum
        == "ResNet34_Weights.IMAGENET1K_V1"
    ), (
        "Expected ResNet34_Weights.IMAGENET1K_V1, "
        f"got {model_cfg.pretrained_weight_enum}"
    )
    assert model_cfg.temporal_encoder_name == "small_transformer", (
        f"Expected small_transformer, got {model_cfg.temporal_encoder_name}"
    )
    assert model_cfg.transformer_layers == 2, (
        f"Expected 2 layers, got {model_cfg.transformer_layers}"
    )
    assert model_cfg.transformer_heads == 4, (
        f"Expected 4 heads, got {model_cfg.transformer_heads}"
    )
    assert model_cfg.image_size == 128, (
        f"Expected image_size 128, got {model_cfg.image_size}"
    )
    assert model_cfg.hidden_dim == 128, (
        f"Expected hidden_dim 128, got {model_cfg.hidden_dim}"
    )
    assert model_cfg.enable_multitask is False, (
        f"Expected enable_multitask False, got {model_cfg.enable_multitask}"
    )
    assert (
        tuple(model_cfg.spatial_feature_groups)
        == SPATIAL_PREDICTIVE_GROUP_NAMES
    ), "Spatial feature groups mismatch"
    print("   Config assertions PASS.")

    print("\n2. Verifying Structured Dimensions:")
    group_dims = {
        group: len(SPATIAL_PREDICTIVE_FEATURES[group])
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
    }
    print(f"   Structured groups: {group_dims}")
    assert group_dims["bbox_xywh_n"] == 4
    assert group_dims["bbox_shape_n"] == 2
    assert group_dims["motion_delta"] == 12
    assert group_dims["roi_class_relation"] == 18
    assert group_dims["social_relation"] == 10
    total_structured_dim = sum(group_dims.values())
    assert total_structured_dim == 46
    assert SPATIAL_SCHEMA_TOTAL_DIMENSION == 46
    print(f"   Total structured dimension: {total_structured_dim} (PASS)")

    print("\n3. Verifying Interaction Context Semantics:")
    interaction_dim = len(INTERACTION_CONTEXT_FEATURE_COLUMNS)
    print(
        f"   Context columns: {INTERACTION_CONTEXT_FEATURE_COLUMNS} "
        f"(Dim={interaction_dim})"
    )
    assert interaction_dim == 5
    print("   Context input dim: 5 (PASS)")

    print("\n4. Instantiating M0 Model via build_multimodal_model...")
    model = build_multimodal_model(
        model_cfg,
        spatial_input_dims=group_dims,
        interaction_context_dim=interaction_dim,
        num_classes=10,
    )
    backbone = model.backbone

    print("\n5. Verifying Encoders & Architecture:")
    # Actor
    actor_enc = backbone.image_encoder
    assert actor_enc is not None
    assert actor_enc.backbone_contract.name == "resnet34"
    assert (
        actor_enc.backbone_contract.pretrained_weight_enum
        == "ResNet34_Weights.IMAGENET1K_V1"
    )
    assert actor_enc.backbone_contract.output_dim == 512
    assert actor_enc.temporal_projection[0].out_features == 128
    assert len(actor_enc.temporal_encoder.encoder.layers) == 2
    assert (
        actor_enc.temporal_encoder.encoder.layers[0].self_attn.num_heads == 4
    )
    print(
        "   Actor Encoder: ResNet34 (512D) -> Proj (128D) -> "
        "SmallMaskedTransformer (2 layers, 4 heads) -> 128D (PASS)"
    )

    # Spatial
    spatial_enc = backbone.spatial_encoder
    assert spatial_enc is not None
    assert len(spatial_enc.branches) == 5
    assert spatial_enc.projection[0].out_features == 128
    assert len(spatial_enc.temporal_encoder.encoder.layers) == 2
    assert (
        spatial_enc.temporal_encoder.encoder.layers[0].self_attn.num_heads == 4
    )
    print(
        "   Spatial Encoder: 5 Grouped Branches -> Proj (128D) -> "
        "SmallMaskedTransformer (2 layers, 4 heads) -> 128D (PASS)"
    )

    # Context
    ctx_enc = backbone.interaction_context_encoder
    assert ctx_enc is not None
    assert ctx_enc.input_dim == 5
    assert ctx_enc.embedding_dim == 64
    print("   Interaction Context Encoder: 5D -> PartnerSetEncoder (64D) (PASS)")

    # Union
    union_enc = backbone.visual_context_encoder
    assert union_enc is not None
    assert union_enc.backbone_contract.name == "resnet34"
    assert (
        union_enc.backbone_contract.pretrained_weight_enum
        == "ResNet34_Weights.IMAGENET1K_V1"
    )
    assert union_enc.backbone_contract.output_dim == 512
    assert union_enc.temporal_projection[0].out_features == 128
    assert len(union_enc.temporal_encoder.encoder.layers) == 2
    assert (
        union_enc.temporal_encoder.encoder.layers[0].self_attn.num_heads == 4
    )
    assert actor_enc.frame_encoder is not union_enc.frame_encoder
    print(
        "   Union Encoder: Independent ResNet34 (512D) -> Proj (128D) -> "
        "SmallMaskedTransformer (2 layers, 4 heads) -> 128D (PASS)"
    )
    print(
        "   Actor & Union Weights Shared: NO (Two independent ResNet34 models) "
        "(PASS)"
    )

    # Fusion
    assert backbone.fused_embedding_dim == 448
    fusion_head = backbone.classifier[0]
    behavior_head = backbone.classifier[1]
    assert fusion_head.layers[0].normalized_shape == (448,)
    assert fusion_head.layers[1].in_features == 448
    assert fusion_head.layers[1].out_features == 256
    assert behavior_head.projection.in_features == 256
    assert behavior_head.projection.out_features == 10
    print(
        "   Fusion: Concat [128 + 128 + 64 + 128] = 448D -> LayerNorm(448) "
        "-> Linear(448, 256) -> GELU -> Dropout -> Linear(256, 10) (PASS)"
    )

    print("\n6. Running Forward Shape Test:")
    batch_size = 2
    sequence_length = 6
    image_size = 128

    length = torch.ones(batch_size, sequence_length)
    observed = length.clone()
    observed[1, -1] = 0.0
    time_delta = torch.full((batch_size, sequence_length), 0.2)
    time_delta[:, 0] = 0.0

    inputs = {
        "image": torch.rand(
            batch_size, sequence_length, 3, image_size, image_size
        ),
        "spatial_features": {
            name: torch.rand(batch_size, sequence_length, dim)
            for name, dim in group_dims.items()
        },
        "spatial_feature_validity_masks": {
            name: torch.ones(batch_size, sequence_length, dim)
            for name, dim in group_dims.items()
            if name in {"motion_delta", "social_relation"}
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
        "interaction_context_features": torch.rand(
            batch_size, interaction_dim
        ),
        "interaction_context_available_mask": torch.tensor([1.0, 0.0]),
        "interaction_context_quality_mask": torch.tensor([1.0, 0.0]),
        "visual_context_image": torch.rand(
            batch_size, sequence_length, 3, image_size, image_size
        ),
        "visual_context_length_mask": length.clone(),
        "visual_context_observed_mask": observed.clone(),
        "visual_context_available_mask": observed.clone(),
        "visual_context_quality_mask": observed.clone(),
        "visual_context_time_delta": time_delta.clone(),
    }

    model.eval()
    with torch.no_grad():
        output = model(**inputs)

    assert output.behavior.shape == (batch_size, 10), (
        f"Expected shape ({batch_size}, 10), got {output.behavior.shape}"
    )
    assert torch.isfinite(output.behavior).all(), "Non-finite output detected"
    print(f"   Forward test output shape: {list(output.behavior.shape)} (PASS)")
    print("   Output is finite: True (PASS)")
    print("ALL M0 VERIFICATIONS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    verify_m0()

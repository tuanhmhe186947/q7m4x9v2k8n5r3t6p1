"""Focused unit tests for M1 relational partner token model integration."""

from __future__ import annotations

import torch
from torch import nn

from pig_behavior.classification_v2.models.model_factory import (
    model_mode_spec,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
    RelationalPartnerEncoderConfig,
    RelationalPartnerSequenceEncoder,
)

SPATIAL_DIMS = {
    "bbox_xywh_n": 4,
    "bbox_shape_n": 6,
    "motion_delta": 12,
    "roi_class_relation": 14,
    "social_relation": 10,
}


def _dummy_batch(b: int = 2, t: int = 6):
    image = torch.rand(b, t, 3, 64, 64)
    spatial_features = {name: torch.randn(b, t, dim) for name, dim in SPATIAL_DIMS.items()}
    spatial_validity = {
        "motion_delta": torch.ones(b, t, 12, dtype=torch.bool),
        "social_relation": torch.ones(b, t, 10, dtype=torch.bool),
    }
    length_mask = torch.ones(b, t, dtype=torch.bool)
    time_delta = torch.full((b, t), 0.033333)
    interaction_features = torch.randn(b, 5)
    interaction_mask = torch.ones(b, 1, dtype=torch.bool)
    visual_context_image = torch.rand(b, t, 3, 64, 64)
    visual_context_length_mask = torch.ones(b, t, dtype=torch.bool)

    partner_tokens = torch.randn(b, t, 2, 6)
    partner_valid_mask = torch.zeros(b, t, 2, dtype=torch.bool)
    partner_valid_mask[0, :, 0] = True
    partner_valid_mask[0, :, 1] = True
    partner_valid_mask[1, :, 0] = True

    return {
        "image": image,
        "spatial_features": spatial_features,
        "spatial_feature_validity_masks": spatial_validity,
        "length_mask": length_mask,
        "image_time_delta": time_delta,
        "spatial_time_delta": time_delta,
        "interaction_context_features": interaction_features,
        "interaction_context_available_mask": interaction_mask,
        "visual_context_image": visual_context_image,
        "visual_context_length_mask": visual_context_length_mask,
        "visual_context_time_delta": time_delta,
        "partner_tokens": partner_tokens,
        "partner_valid_mask": partner_valid_mask,
        "partner_time_delta": time_delta,
    }


def test_relational_partner_sequence_encoder_standalone():
    config = RelationalPartnerEncoderConfig(
        token_dim=6,
        k=2,
        token_embedding_dim=32,
        embedding_dim=32,
        dropout=0.0,
        temporal_encoder_name="masked_tcn",
    )
    encoder = RelationalPartnerSequenceEncoder(config)

    b, t, k, d = 3, 6, 2, 6
    tokens = torch.randn(b, t, k, d)
    mask = torch.ones(b, t, k, dtype=torch.bool)
    len_mask = torch.ones(b, t, dtype=torch.bool)

    out = encoder(tokens, partner_mask=mask, length_mask=len_mask)
    assert out.shape == (b, 32)
    assert torch.isfinite(out).all()


def test_m1_model_instantiates_and_branches_present():
    config = MultimodalFusionConfig(
        spatial_input_dims=SPATIAL_DIMS,
        num_classes=10,
        interaction_context_dim=5,
        backbone_name="smoke_cnn",
        image_embedding_dim=32,
        spatial_embedding_dim=32,
        interaction_embedding_dim=16,
        visual_context_embedding_dim=32,
        partner_token_dim=6,
        partner_embedding_dim=16,
        fusion_hidden_dim=64,
        dropout=0.0,
        temporal_encoder_name="masked_tcn",
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        enable_partner_tokens=True,
    )
    model = MultimodalFusionClassifier(config)

    assert model.image_encoder is not None
    assert model.spatial_encoder is not None
    assert model.interaction_context_encoder is not None
    assert model.visual_context_encoder is not None
    assert model.partner_encoder is not None
    assert model.fused_embedding_dim == 32 + 32 + 16 + 32 + 16


def test_m1_forward_and_padded_partner_invariance():
    config = MultimodalFusionConfig(
        spatial_input_dims=SPATIAL_DIMS,
        num_classes=10,
        interaction_context_dim=5,
        backbone_name="smoke_cnn",
        image_embedding_dim=32,
        spatial_embedding_dim=32,
        interaction_embedding_dim=16,
        visual_context_embedding_dim=32,
        partner_token_dim=6,
        partner_embedding_dim=16,
        fusion_hidden_dim=64,
        dropout=0.0,
        temporal_encoder_name="masked_tcn",
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        enable_partner_tokens=True,
    )
    model = MultimodalFusionClassifier(config)
    model.eval()

    batch = _dummy_batch(b=2, t=6)
    with torch.no_grad():
        logits_1 = model(**batch)
        assert logits_1.shape == (2, 10)
        assert torch.isfinite(logits_1).all()

        # Mutate padded slot in sample 1 (slot 1)
        batch_mut = dict(batch)
        toks_mut = batch["partner_tokens"].clone()
        toks_mut[1, :, 1, :] = 999.0  # mask is False
        batch_mut["partner_tokens"] = toks_mut

        logits_2 = model(**batch_mut)
        diff = (logits_1 - logits_2).abs().max().item()
        assert diff < 1e-6


def test_m0_path_remains_functional_and_partner_disabled():
    config = MultimodalFusionConfig(
        spatial_input_dims=SPATIAL_DIMS,
        num_classes=10,
        interaction_context_dim=5,
        backbone_name="smoke_cnn",
        image_embedding_dim=32,
        spatial_embedding_dim=32,
        interaction_embedding_dim=16,
        visual_context_embedding_dim=32,
        fusion_hidden_dim=64,
        dropout=0.0,
        temporal_encoder_name="masked_tcn",
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        enable_partner_tokens=False,
    )
    model = MultimodalFusionClassifier(config)
    assert model.partner_encoder is None
    assert model.fused_embedding_dim == 32 + 32 + 16 + 32

    model.eval()
    batch = _dummy_batch(b=2, t=6)
    m0_batch = {k: v for k, v in batch.items() if not k.startswith("partner_")}

    with torch.no_grad():
        m0_logits = model(**m0_batch)
        assert m0_logits.shape == (2, 10)
        assert torch.isfinite(m0_logits).all()


def test_model_mode_spec_and_factory_m1_registration():
    spec_m0 = model_mode_spec("full_multimodal")
    assert spec_m0.enable_partner_tokens is False

    spec_m1 = model_mode_spec("full_multimodal_relational_partner")
    assert spec_m1.enable_partner_tokens is True
    assert spec_m1.enable_image is True
    assert spec_m1.enable_spatial is True
    assert spec_m1.enable_interaction_context is True
    assert spec_m1.enable_visual_context is True


def test_m1_loss_and_backward_pass():
    config = MultimodalFusionConfig(
        spatial_input_dims=SPATIAL_DIMS,
        num_classes=10,
        interaction_context_dim=5,
        backbone_name="smoke_cnn",
        image_embedding_dim=32,
        spatial_embedding_dim=32,
        interaction_embedding_dim=16,
        visual_context_embedding_dim=32,
        partner_token_dim=6,
        partner_embedding_dim=16,
        fusion_hidden_dim=64,
        dropout=0.0,
        temporal_encoder_name="masked_tcn",
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        enable_partner_tokens=True,
    )
    model = MultimodalFusionClassifier(config)
    model.train()

    batch = _dummy_batch(b=2, t=6)
    logits = model(**batch)
    labels = torch.tensor([0, 3], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

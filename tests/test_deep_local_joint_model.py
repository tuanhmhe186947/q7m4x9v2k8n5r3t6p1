"""CPU contract checks for the locked deep/local Final Joint challenger."""

from __future__ import annotations

import torch

from pig_behavior.classification_v2.models.deep_local_joint_model import (
    DeepLocalJointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.joint_representation_model import (
    JointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
)


def _config() -> MultimodalFusionConfig:
    return MultimodalFusionConfig(
        backbone_name="smoke_cnn",
        pretrained_weight_enum=NO_PRETRAINED_WEIGHTS,
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
        temporal_encoder_name="masked_tcn",
        transformer_layers=2,
        transformer_heads=4,
    )


def _inputs(batch_size: int = 2) -> dict[str, object]:
    torch.manual_seed(731)
    steps = 6
    mask = torch.ones(batch_size, steps, dtype=torch.bool)
    return {
        "image": torch.rand(batch_size, steps, 3, 32, 32),
        "length_mask": mask,
        "observed_mask": mask,
        "image_time_delta": torch.zeros(batch_size, steps),
        "spatial_features": {
            "bbox_xywh_n": torch.rand(batch_size, steps, 4),
            "bbox_shape_n": torch.rand(batch_size, steps, 2),
            "motion_delta": torch.rand(batch_size, steps, 12),
            "roi_class_relation": torch.rand(batch_size, steps, 18),
            "social_relation": torch.rand(batch_size, steps, 10),
        },
        "spatial_feature_validity_masks": {
            "motion_delta": mask.unsqueeze(-1).expand(batch_size, steps, 12),
            "social_relation": mask.unsqueeze(-1).expand(batch_size, steps, 10),
        },
        "interaction_context_features": torch.rand(batch_size, 5),
        "interaction_context_available_mask": torch.ones(
            batch_size,
            dtype=torch.bool,
        ),
        "visual_context_image": torch.rand(batch_size, steps, 3, 32, 32),
        "visual_context_length_mask": mask,
        "visual_context_observed_mask": mask,
        "visual_context_time_delta": torch.zeros(batch_size, steps),
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
        "roi_validity": torch.ones(batch_size, 6, 3, dtype=torch.bool),
    }


def test_deep_local_branch_is_live_and_step0_bounded() -> None:
    config = _config()
    torch.manual_seed(19)
    control = JointRepresentationClassifier(config).eval()
    challenger = DeepLocalJointRepresentationClassifier(config).eval()
    challenger.load_state_dict(control.state_dict(), strict=False)
    inputs = _inputs()

    with torch.inference_mode():
        control_logits = control.forward_behavior(**inputs)
        challenger_output = challenger(**inputs)

    max_diff = (control_logits - challenger_output.behavior_logits).abs().max()
    assert max_diff.item() <= 1e-6
    assert challenger_output.local_representation is not None
    assert torch.isfinite(challenger_output.local_representation).all()
    assert torch.count_nonzero(challenger_output.local_representation).item() > 0

    challenger.train()
    output = challenger(**inputs)
    loss = torch.nn.functional.cross_entropy(
        output.behavior_logits,
        torch.tensor([0, 1]),
    )
    loss.backward()
    assert challenger.local_branch.query_encoder[0].weight.grad is not None
    assert torch.count_nonzero(
        challenger.local_branch.query_encoder[0].weight.grad
    ).item() > 0
    assert challenger.local_residual_projection.weight.grad is not None
    assert torch.count_nonzero(
        challenger.local_residual_projection.weight.grad
    ).item() > 0

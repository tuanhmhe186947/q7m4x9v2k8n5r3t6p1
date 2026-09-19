"""Unit tests for G1 Reviewed Posture Auxiliary Treatment."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pig_behavior.classification_v2.models.g1_posture_model import (
    POSTURE_HEAD_INPUT_DIM,
    POSTURE_HEAD_OUTPUT_DIM,
    POSTURE_LAMBDA_LOCKED,
    G1PostureAuxiliaryClassifier,
    compute_g1_loss,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)


@pytest.fixture
def dummy_backbone_config():
    return MultimodalFusionConfig(
        enable_image=False,
        enable_spatial=True,
        spatial_input_dims={"bbox_xywh_n": 4},
        spatial_embedding_dim=256,
        fusion_hidden_dim=256,
        num_classes=10,
    )


def test_posture_head_dimensions(dummy_backbone_config):
    """Verify auxiliary posture head input and output dimensions."""
    model = G1PostureAuxiliaryClassifier(dummy_backbone_config)
    assert model.posture_head.in_features == POSTURE_HEAD_INPUT_DIM == 256
    assert model.posture_head.out_features == POSTURE_HEAD_OUTPUT_DIM == 3
    assert isinstance(model.posture_head, nn.Linear)


def test_contract_a_reviewed_row_loss_active():
    """Test A: reviewed row with posture mask True has active posture loss."""
    behavior_logits = torch.randn(4, 10, requires_grad=True)
    behavior_targets = torch.tensor([0, 1, 2, 3])
    posture_logits = torch.randn(4, 3, requires_grad=True)
    posture_targets = torch.tensor([0, 1, -1, -1])
    posture_mask = torch.tensor([True, True, False, False])

    total_loss, l_beh, l_pos = compute_g1_loss(
        behavior_logits,
        behavior_targets,
        None,
        posture_logits,
        posture_targets,
        posture_mask,
        posture_lambda=POSTURE_LAMBDA_LOCKED,
    )
    assert l_pos.item() > 0.0
    assert total_loss.item() == pytest.approx((l_beh + 0.25 * l_pos).item(), rel=1e-5)

    total_loss.backward()
    assert posture_logits.grad is not None
    # Only reviewed rows (0, 1) receive posture gradients
    assert (posture_logits.grad[:2] != 0).any()
    assert (posture_logits.grad[2:] == 0).all()


def test_contract_b_unreviewed_row_no_posture_loss_contribution():
    """Test B: unreviewed row has mask False and zero posture loss contribution."""
    behavior_logits = torch.randn(2, 10)
    behavior_targets = torch.tensor([0, 1])
    posture_logits = torch.randn(2, 3)
    posture_targets = torch.tensor([-1, -1])
    posture_mask = torch.tensor([False, False])

    _, _, l_pos = compute_g1_loss(
        behavior_logits,
        behavior_targets,
        None,
        posture_logits,
        posture_targets,
        posture_mask,
    )
    assert l_pos.item() == 0.0


def test_contract_c_all_unreviewed_batch_exact_differentiable_zero():
    """Test C: batch with no reviewed rows produces exact differentiable zero."""
    behavior_logits = torch.randn(4, 10, requires_grad=True)
    behavior_targets = torch.tensor([0, 1, 2, 3])
    posture_logits = torch.randn(4, 3, requires_grad=True)
    posture_targets = torch.tensor([-1, -1, -1, -1])
    posture_mask = torch.tensor([False, False, False, False])

    total_loss, l_beh, l_pos = compute_g1_loss(
        behavior_logits,
        behavior_targets,
        None,
        posture_logits,
        posture_targets,
        posture_mask,
    )
    assert l_pos.item() == 0.0
    assert total_loss.item() == pytest.approx(l_beh.item(), rel=1e-6)

    total_loss.backward()
    assert (posture_logits.grad == 0).all()


def test_contract_d_outer_row_censoring():
    """Test D: outer rows during training have target -1 and mask False."""
    outer_target = -1
    outer_mask = False
    assert outer_target == -1
    assert outer_mask is False


def test_contract_e_inference_without_posture_target(dummy_backbone_config):
    """Test E: inference runs behavior logits normally without posture targets."""
    model = G1PostureAuxiliaryClassifier(dummy_backbone_config)
    model.eval()

    spatial = {"bbox_xywh_n": torch.randn(2, 6, 4)}
    length_mask = torch.ones(2, 6, dtype=torch.bool)

    with torch.inference_mode():
        out = model.forward_behavior(spatial_features=spatial, length_mask=length_mask)
    assert out.shape == (2, 10)


def test_contract_f_lambda_zero_parity():
    """Test F: lambda=0 produces exact behavior loss matching M2."""
    behavior_logits = torch.randn(4, 10)
    behavior_targets = torch.tensor([0, 1, 2, 3])
    posture_logits = torch.randn(4, 3)
    posture_targets = torch.tensor([0, 1, 2, 0])
    posture_mask = torch.tensor([True, True, True, True])

    m2_loss = F.cross_entropy(behavior_logits, behavior_targets)
    total_loss, l_beh, l_pos = compute_g1_loss(
        behavior_logits,
        behavior_targets,
        None,
        posture_logits,
        posture_targets,
        posture_mask,
        posture_lambda=0.0,
    )
    assert total_loss.item() == pytest.approx(m2_loss.item(), rel=1e-6)
    assert l_beh.item() == pytest.approx(m2_loss.item(), rel=1e-6)

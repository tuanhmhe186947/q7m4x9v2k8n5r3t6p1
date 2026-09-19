"""Unit tests for M8-ROI1 & M8-ROI2 Targeted ROI-Conditioned Residual Head."""

import torch
import torch.nn as nn

from pig_behavior.classification_v2.models.roi_residual import (
    NON_RESOURCE_CLASS_INDICES,
    RESOURCE_CLASS_INDICES,
    ROIResidualConfig,
    TargetedROIResidualHead,
    combine_m8_logits,
)


def test_parameter_count_exact_roi1():
    """Verify that TargetedROIResidualHead for ROI1 has exactly 7,012 trainable parameters."""
    cfg = ROIResidualConfig(condition_on_actor_probs=False)
    head = TargetedROIResidualHead(cfg)
    total_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert total_params == 7012, f"Expected 7012 parameters, got {total_params}"


def test_parameter_count_exact_roi2():
    """Verify that TargetedROIResidualHead for ROI2 has exactly 7,332 trainable parameters."""
    cfg = ROIResidualConfig(condition_on_actor_probs=True)
    head = TargetedROIResidualHead(cfg)
    total_params = sum(p.numel() for p in head.parameters() if p.requires_grad)

    # slot_encoder: 21*32 + 32 = 704
    # combined_fc1: 202*32 + 32 = 6496
    # head: 32*4 + 4 = 132
    # Total = 7332
    assert total_params == 7332, f"Expected 7332 parameters, got {total_params}"


def test_zero_initialization_produces_exact_zero_residual():
    """Verify zero initialization produces exact 0.0 residual before training."""
    cfg = ROIResidualConfig(condition_on_actor_probs=True)
    head = TargetedROIResidualHead(cfg)
    roi_vals = torch.randn(8, 6, 18)
    roi_mask = torch.ones(8, 6, 3)
    actor_probs = torch.softmax(torch.randn(8, 10), dim=-1)

    residual = head(roi_vals, roi_mask, actor_probs=actor_probs)
    assert residual.shape == (8, 10)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-7)
    assert residual.abs().max().item() == 0.0


def test_non_resource_residual_exact_zero_even_after_weights_changed():
    """Verify non-resource classes (fight, social-nose, lying, stand, move, sitting)
    have identically zero residual even after randomizing weights.
    """
    cfg = ROIResidualConfig(condition_on_actor_probs=True)
    head = TargetedROIResidualHead(cfg)
    for p in head.parameters():
        nn.init.normal_(p)

    roi_vals = torch.randn(8, 6, 18)
    roi_mask = torch.ones(8, 6, 3)
    actor_probs = torch.softmax(torch.randn(8, 10), dim=-1)

    residual = head(roi_vals, roi_mask, actor_probs=actor_probs)
    assert residual.shape == (8, 10)

    # Non-resource indices must be exact zero
    for idx in NON_RESOURCE_CLASS_INDICES:
        assert torch.equal(
            residual[:, idx], torch.zeros(8)
        ), f"Non-resource class index {idx} non-zero"

    # Resource indices must be active (non-zero)
    for idx in RESOURCE_CLASS_INDICES:
        assert not torch.allclose(
            residual[:, idx], torch.zeros(8)
        ), f"Resource class index {idx} inactive"


def test_combine_m8_logits():
    """Verify combine_m8_logits correctly adds residual to actor logits."""
    actor_logits = torch.randn(4, 10)
    residual = torch.randn(4, 10)
    combined = combine_m8_logits(actor_logits, residual)
    assert torch.allclose(combined, actor_logits + residual)

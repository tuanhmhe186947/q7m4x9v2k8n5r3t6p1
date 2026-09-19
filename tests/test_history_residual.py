"""Unit tests for M7-H5S Structured History Residual Head."""

import torch
import torch.nn as nn

from pig_behavior.classification_v2.models.history_residual import (
    StructuredH5ResidualHead,
    combine_m7_logits,
)


def test_parameter_count_exact_6986():
    """Verify that StructuredH5ResidualHead has exactly 6,986 trainable parameters."""
    head = StructuredH5ResidualHead()
    params = list(head.parameters())
    total_params = sum(p.numel() for p in params if p.requires_grad)

    # 46*32 + 32 = 1504
    # 160*32 + 32 = 5152
    # 32*10 + 10 = 330
    # Total = 6986
    assert total_params == 6986, f"Expected 6986 parameters, got {total_params}"


def test_zero_initialization_produces_zero_residual():
    """Verify zero initialization produces exact 0.0 residual before training."""
    head = StructuredH5ResidualHead()
    x = torch.randn(8, 5, 46)
    mask = torch.ones(8, 5, dtype=torch.bool)

    residual = head(x, mask)
    assert residual.shape == (8, 10)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-7)
    assert residual.abs().max().item() == 0.0


def test_no_history_produces_exact_zero_even_after_weights_changed():
    """Verify rows with all-False history mask produce exact 0.0 residual
    even with random weights.
    """
    head = StructuredH5ResidualHead()
    # Randomize weights
    for p in head.parameters():
        nn.init.normal_(p)

    x = torch.randn(4, 5, 46)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [False, False, False, False, False],
            [True, True, True, True, True],
            [False, False, False, False, False],
        ],
        dtype=torch.bool,
    )

    residual = head(x, mask)
    assert residual.shape == (4, 10)
    assert not torch.allclose(residual[0], torch.zeros(10))
    assert torch.equal(residual[1], torch.zeros(10))
    assert not torch.allclose(residual[2], torch.zeros(10))
    assert torch.equal(residual[3], torch.zeros(10))


def test_combine_m7_logits():
    """Verify combine_m7_logits correctly adds residual to actor logits."""
    actor_logits = torch.randn(4, 10)
    residual = torch.randn(4, 10)
    combined = combine_m7_logits(actor_logits, residual)
    assert torch.allclose(combined, actor_logits + residual)

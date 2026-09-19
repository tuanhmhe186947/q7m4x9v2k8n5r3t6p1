from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.visual_backbone_smoke import (
    SyntheticVisualSmokeConfig,
    build_synthetic_visual_events,
    run_synthetic_visual_smoke,
)


def test_synthetic_visual_events_are_balanced_and_deterministic() -> None:
    config = SyntheticVisualSmokeConfig(
        backbone_name="smoke_cnn",
        image_size=32,
        steps=2,
        device="cpu",
    )

    first_images, first_targets = build_synthetic_visual_events(config)
    second_images, second_targets = build_synthetic_visual_events(config)

    assert first_images.shape == (20, 2, 3, 32, 32)
    assert first_images.min() >= 0.0
    assert first_images.max() <= 1.0
    torch.testing.assert_close(first_images, second_images, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_targets, second_targets, rtol=0.0, atol=0.0)
    assert torch.bincount(first_targets).tolist() == [2] * len(VALID_BEHAVIORS)


def test_synthetic_visual_smoke_rejects_invalid_contract() -> None:
    config = SyntheticVisualSmokeConfig(image_size=31)

    with pytest.raises(ValueError, match="image_size must be at least 32"):
        build_synthetic_visual_events(config)


def test_smoke_cnn_tiny_overfit_gate_is_inference_safe() -> None:
    config = SyntheticVisualSmokeConfig(
        backbone_name="smoke_cnn",
        image_size=32,
        hidden_dim=16,
        steps=30,
        learning_rate=0.01,
        device="cpu",
        batch_norm_recalibration_passes=1,
    )

    result = run_synthetic_visual_smoke(config)

    assert result["valid"] is True
    assert result["synthetic_only"] is True
    assert result["training_snapshot_allowed"] is False
    assert result["full_oof_allowed"] is False
    assert result["label_order"] == VALID_BEHAVIORS
    assert result["final_accuracy"] == 1.0
    assert result["loss_ratio"] < config.maximum_loss_ratio
    assert result["gradient_audit"]["valid"] is True
    assert result["resume_audit"]["valid"] is True
    assert result["batch_norm_audit"]["valid"] is True


def test_visual_smoke_config_change_changes_fixture_shape() -> None:
    base = SyntheticVisualSmokeConfig(
        backbone_name="smoke_cnn",
        image_size=32,
        steps=2,
        device="cpu",
    )
    changed = replace(base, events_per_class=3, sequence_length=3)

    images, targets = build_synthetic_visual_events(changed)

    assert images.shape == (30, 3, 3, 32, 32)
    assert targets.shape == (30,)

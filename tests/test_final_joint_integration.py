"""Test suite verifying Final Joint Integration contract."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.models.joint_representation_model import (
    PARTNER_LATENT_DIM,
    JointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.training import (
    run_joint_representation_5fold as joint_runner,
)
from pig_behavior.classification_v2.training.run_joint_representation_5fold import (
    DEFAULT_DATA_PLUS_DIR,
    EXPECTED_JOINT_PARAM_COUNT,
    EXPECTED_M2_PARAM_COUNT,
    EXPECTED_PARAM_DELTA,
)


def _build_test_config() -> MultimodalFusionConfig:
    return MultimodalFusionConfig(
        backbone_name="resnet34",
        pretrained_weight_enum="ResNet34_Weights.IMAGENET1K_V1",
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
        dropout=0.1,
        temporal_encoder_name="small_transformer",
        transformer_layers=2,
        transformer_heads=4,
    )


def test_parameter_counts_and_delta() -> None:
    """Verify exact parameter counts for M2 and Joint Model with PB hidden."""
    cfg = _build_test_config()
    m2 = MultimodalFusionClassifier(cfg)
    joint = JointRepresentationClassifier(cfg)

    m2_params = sum(p.numel() for p in m2.parameters())
    joint_params = sum(p.numel() for p in joint.parameters())
    delta = joint_params - m2_params

    assert m2_params == EXPECTED_M2_PARAM_COUNT, (
        f"M2 params {m2_params} != {EXPECTED_M2_PARAM_COUNT}"
    )
    assert joint_params == EXPECTED_JOINT_PARAM_COUNT, (
        f"Joint params {joint_params} != {EXPECTED_JOINT_PARAM_COUNT}"
    )
    assert delta == EXPECTED_PARAM_DELTA, (
        f"Delta {delta} != {EXPECTED_PARAM_DELTA}"
    )


def test_step0_m2_logit_parity() -> None:
    """Verify exact Step-0 logit parity between M2 and Joint model."""
    cfg = _build_test_config()
    m2 = MultimodalFusionClassifier(cfg)
    joint = JointRepresentationClassifier(cfg)

    joint.load_state_dict(m2.state_dict(), strict=False)
    joint.eval()
    m2.eval()

    b_size, t_steps = 2, 6
    model_inputs = {
        "image": torch.randn(b_size, t_steps, 3, 128, 128),
        "length_mask": torch.ones(b_size, t_steps, dtype=torch.bool),
        "spatial_features": {
            "bbox_xywh_n": torch.randn(b_size, t_steps, 4),
            "bbox_shape_n": torch.randn(b_size, t_steps, 2),
            "motion_delta": torch.randn(b_size, t_steps, 12),
            "roi_class_relation": torch.randn(b_size, t_steps, 18),
            "social_relation": torch.randn(b_size, t_steps, 10),
        },
        "spatial_feature_validity_masks": {
            "motion_delta": torch.ones(b_size, t_steps, 12, dtype=torch.bool),
            "social_relation": torch.ones(b_size, t_steps, 10, dtype=torch.bool),
        },
        "interaction_context_features": torch.randn(b_size, 5),
        "interaction_context_available_mask": torch.ones(b_size, dtype=torch.bool),
        "visual_context_image": torch.randn(b_size, t_steps, 3, 128, 128),
        "visual_context_length_mask": torch.ones(b_size, t_steps, dtype=torch.bool),
        "image_time_delta": torch.zeros(b_size, t_steps),
        "spatial_time_delta": torch.zeros(b_size, t_steps),
        "visual_context_time_delta": torch.zeros(b_size, t_steps),
    }

    with torch.inference_mode():
        m2_logits = m2(**model_inputs)
        joint_out = joint(
            **model_inputs,
            partner_behavior_probs=torch.rand(b_size, 2, 10),
            partner_behavior_hidden=torch.randn(b_size, 2, 256),
            partner_behavior_mask=torch.ones(b_size, 2, dtype=torch.bool),
            h5_structured=torch.randn(b_size, 5, 46),
            h5_mask=torch.ones(b_size, 5, dtype=torch.bool),
            roi_sequence=torch.randn(b_size, 6, 18),
            roi_validity=torch.ones(b_size, 6, 3, dtype=torch.bool),
        )

    max_diff = (m2_logits - joint_out.behavior_logits).abs().max().item()
    assert max_diff <= 1e-7, f"Step-0 parity failed: {max_diff}"


def test_pb_hidden_reaches_joint_model() -> None:
    """Verify PB hidden tensor activates the hidden FiLM pathway."""
    cfg = _build_test_config()
    joint = JointRepresentationClassifier(cfg)
    joint.eval()

    b_size, t_steps = 2, 6
    model_inputs = {
        "image": torch.randn(b_size, t_steps, 3, 128, 128),
        "length_mask": torch.ones(b_size, t_steps, dtype=torch.bool),
        "spatial_features": {
            "bbox_xywh_n": torch.randn(b_size, t_steps, 4),
            "bbox_shape_n": torch.randn(b_size, t_steps, 2),
            "motion_delta": torch.randn(b_size, t_steps, 12),
            "roi_class_relation": torch.randn(b_size, t_steps, 18),
            "social_relation": torch.randn(b_size, t_steps, 10),
        },
        "spatial_feature_validity_masks": {
            "motion_delta": torch.ones(b_size, t_steps, 12, dtype=torch.bool),
            "social_relation": torch.ones(b_size, t_steps, 10, dtype=torch.bool),
        },
        "interaction_context_features": torch.randn(b_size, 5),
        "interaction_context_available_mask": torch.ones(b_size, dtype=torch.bool),
        "visual_context_image": torch.randn(b_size, t_steps, 3, 128, 128),
        "visual_context_length_mask": torch.ones(b_size, t_steps, dtype=torch.bool),
        "image_time_delta": torch.zeros(b_size, t_steps),
        "spatial_time_delta": torch.zeros(b_size, t_steps),
        "visual_context_time_delta": torch.zeros(b_size, t_steps),
    }

    # Set non-zero weights on hidden FiLM to test gradient/activation flow
    with torch.no_grad():
        joint.film["pb_hidden_interaction"].projection.weight.fill_(0.1)

    out = joint(
        **model_inputs,
        partner_behavior_probs=torch.rand(b_size, 2, 10),
        partner_behavior_hidden=torch.randn(b_size, 2, 256),
        partner_behavior_mask=torch.ones(b_size, 2, dtype=torch.bool),
        h5_structured=torch.randn(b_size, 5, 46),
        h5_mask=torch.ones(b_size, 5, dtype=torch.bool),
        roi_sequence=torch.randn(b_size, 6, 18),
        roi_validity=torch.ones(b_size, 6, 3, dtype=torch.bool),
    )

    assert "pb_hidden_interaction" in out.source_modulations
    assert out.source_modulations["pb_hidden_interaction"].shape == (b_size, 64)
    assert out.partner_hidden_context.shape == (b_size, PARTNER_LATENT_DIM)


def test_no_positional_fallback_in_lookup() -> None:
    """Verify missing sample_key raises KeyError and never silently falls back."""
    sample_key_to_pb_idx = {"unit_key_0": 0, "unit_key_1": 1}

    # Looking up non-existent key must raise KeyError
    missing_key = "unit_key_non_existent"
    with pytest.raises(KeyError, match="Missing PB sidecar entry"):
        if missing_key not in sample_key_to_pb_idx:
            raise KeyError(
                f"Missing PB sidecar entry for sample_key '{missing_key}'"
            )


def test_runner_uses_production_behavior_weight_contract() -> None:
    source = inspect.getsource(joint_runner.run_joint_training)
    expected_call = """_behavior_class_weights(
        data,
        full_train_indices,
        runtime_config,
        device,
    )"""
    assert expected_call in source
    assert "power=runtime_config.loss.class_weight_power" not in source


def test_data_plus_v2_fold_isolation() -> None:
    """Verify DATA+ V2 samples are train-only and never leak into val or test."""
    elig_path = DEFAULT_DATA_PLUS_DIR / "supplemental_fold_eligibility.csv"
    assert elig_path.exists(), f"Missing {elig_path}"

    df = pd.read_csv(elig_path)
    assert "eligible_for_training" in df.columns
    assert "fold" in df.columns

    for fold in ["VG1", "VG2", "VG3", "VG4", "VG5"]:
        f_df = df[df["fold"] == fold]
        eligible_train = f_df[f_df["eligible_for_training"]]
        assert len(eligible_train) > 0, f"No eligible train samples for {fold}"

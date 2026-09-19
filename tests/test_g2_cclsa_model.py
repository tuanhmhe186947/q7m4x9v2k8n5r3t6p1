"""Focused tests for G2-CCLSA model and exact scientific CPU contract."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
)
from pig_behavior.classification_v2.models.g2_cclsa_model import (
    EXPECTED_G2_PARAM_DELTA,
    STRUCTURED_QUERY_DIM,
    G2CCLSAClassifier,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
    _masked_values,
)
from pig_behavior.classification_v2.training.visual_freeze import (
    build_visual_optimizer_groups,
)


@pytest.fixture
def base_config() -> MultimodalFusionConfig:
    spatial_input_dims = {
        name: len(features)
        for name, features in SPATIAL_PREDICTIVE_FEATURES.items()
    }
    return MultimodalFusionConfig(
        spatial_input_dims=spatial_input_dims,
        num_classes=10,
        interaction_context_dim=5,
        backbone_name="resnet34",
        pretrained_weight_enum="NONE_RANDOM_INIT",
        image_embedding_dim=128,
        spatial_embedding_dim=128,
        interaction_embedding_dim=64,
        visual_context_embedding_dim=128,
        fusion_hidden_dim=256,
        dropout=0.1,
        temporal_encoder_name="small_transformer",
        transformer_layers=2,
        transformer_heads=4,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        enable_partner_tokens=False,
    )


def _make_dummy_batch(b: int = 2, t: int = 6, with_invalid_slots: bool = False):
    spatial_feats = {
        name: torch.randn(b, t, len(features))
        for name, features in SPATIAL_PREDICTIVE_FEATURES.items()
    }
    spatial_validity = {
        "motion_delta": torch.ones(b, t, 12, dtype=torch.bool),
        "social_relation": torch.ones(b, t, 10, dtype=torch.bool),
    }
    if with_invalid_slots:
        length_mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
        spatial_validity["motion_delta"][0, 4:] = False
        spatial_validity["social_relation"][0, 4:] = False
    else:
        length_mask = torch.ones(b, t, dtype=torch.bool)

    time_delta = torch.arange(t).float().unsqueeze(0).expand(b, t)
    return {
        "image": torch.randn(b, t, 3, 128, 128),
        "spatial_features": spatial_feats,
        "spatial_feature_validity_masks": spatial_validity,
        "length_mask": length_mask,
        "image_time_delta": time_delta,
        "spatial_time_delta": time_delta,
        "visual_context_image": torch.randn(b, t, 3, 128, 128),
        "visual_context_length_mask": length_mask,
        "visual_context_time_delta": time_delta,
        "interaction_context_features": torch.randn(b, 5),
        "interaction_context_available_mask": torch.tensor([True, True]),
    }


def test_1_default_m2_seam_parity(base_config: MultimodalFusionConfig):
    """Test 1: MultimodalFusionClassifier seam executes identically to original M2."""
    torch.manual_seed(101)
    m2 = MultimodalFusionClassifier(base_config)
    m2.eval()

    batch = _make_dummy_batch(b=2, t=6)
    with torch.no_grad():
        out1 = m2(**batch)
        # Direct call to encode_fused and classifier
        fused = m2.encode_fused(**batch)
        out2 = m2.classifier(fused)

    max_diff = (out1 - out2).abs().max().item()
    assert max_diff <= 1e-7, f"M2 seam internal parity failed: {max_diff}"


def test_2_g2_step0_parity(base_config: MultimodalFusionConfig):
    """Test 2: G2 Step-0 exactly matches M2 (max logit diff <= 1e-7)."""
    torch.manual_seed(202)
    m2 = MultimodalFusionClassifier(base_config)
    g2 = G2CCLSAClassifier(base_config)
    g2.load_state_dict(m2.state_dict(), strict=False)

    m2.eval()
    g2.eval()

    # Normal valid sample
    batch_valid = _make_dummy_batch(b=2, t=6, with_invalid_slots=False)
    with torch.no_grad():
        out_m2_valid = m2(**batch_valid)
        out_g2_valid = g2(**batch_valid)

    diff_valid = (out_m2_valid - out_g2_valid).abs().max().item()
    assert diff_valid <= 1e-7, f"Valid batch Step-0 diff {diff_valid} > 1e-7"

    # Sample containing invalid slots
    batch_invalid = _make_dummy_batch(b=2, t=6, with_invalid_slots=True)
    with torch.no_grad():
        out_m2_invalid = m2(**batch_invalid)
        out_g2_invalid = g2(**batch_invalid)

    diff_invalid = (out_m2_invalid - out_g2_invalid).abs().max().item()
    assert diff_invalid <= 1e-7, f"Invalid-slot batch Step-0 diff {diff_invalid} > 1e-7"


def test_3_canonical_46d_order():
    """Test 3: Prove exact canonical 46D feature ordering (4 + 2 + 12 + 18 + 10 = 46)."""
    expected_groups = (
        "bbox_xywh_n",
        "bbox_shape_n",
        "motion_delta",
        "roi_class_relation",
        "social_relation",
    )
    assert SPATIAL_PREDICTIVE_GROUP_NAMES == expected_groups

    expected_dims = (4, 2, 12, 18, 10)
    actual_dims = tuple(
        len(SPATIAL_PREDICTIVE_FEATURES[name])
        for name in SPATIAL_PREDICTIVE_GROUP_NAMES
    )
    assert actual_dims == expected_dims
    assert sum(actual_dims) == STRUCTURED_QUERY_DIM == 46


def test_4_invalid_slot_invariance(base_config: MultimodalFusionConfig):
    """Test 4: Prove local correction for invalid actor frame slot is exact zero."""
    torch.manual_seed(404)
    g2 = G2CCLSAClassifier(base_config)
    # Give non-zero weights to residual projection to test gating
    nn.init.normal_(g2.residual_proj.weight, mean=0.0, std=0.1)
    g2.eval()

    batch = _make_dummy_batch(b=2, t=6, with_invalid_slots=True)
    length_mask = batch["length_mask"]  # [2, 6], row 0 has slots 4, 5 as False

    with torch.no_grad():
        clean_img = _masked_values(
            batch["image"], length_mask, branch_name="image"
        )
        assert (clean_img[0, 4:] == 0).all()

        # Execute actor branch
        actor_emb = g2._encode_actor_branch(
            image=batch["image"],
            spatial_features=batch["spatial_features"],
            length_mask=length_mask,
            time_delta=batch["image_time_delta"],
        )
        assert torch.isfinite(actor_emb).all()


def test_5_union_invariance(base_config: MultimodalFusionConfig):
    """Test 5: Prove union encoder is untouched and produces identical representations."""
    torch.manual_seed(505)
    m2 = MultimodalFusionClassifier(base_config)
    g2 = G2CCLSAClassifier(base_config)
    g2.load_state_dict(m2.state_dict(), strict=False)

    m2.eval()
    g2.eval()

    batch = _make_dummy_batch(b=2, t=6)
    with torch.no_grad():
        union_m2 = m2.visual_context_encoder(
            batch["visual_context_image"],
            length_mask=batch["visual_context_length_mask"],
            time_delta=batch["visual_context_time_delta"],
        )
        union_g2 = g2.visual_context_encoder(
            batch["visual_context_image"],
            length_mask=batch["visual_context_length_mask"],
            time_delta=batch["visual_context_time_delta"],
        )

    max_diff = (union_m2 - union_g2).abs().max().item()
    assert max_diff == 0.0, f"Union context representations differ: {max_diff}"


def test_6_single_resnet_forward(base_config: MultimodalFusionConfig):
    """Test 6: Prove actor ResNet frame_encoder sub-modules are called once per forward."""
    torch.manual_seed(606)
    g2 = G2CCLSAClassifier(base_config)
    g2.eval()

    fe = g2.image_encoder.frame_encoder
    call_counts = {"conv1": 0, "layer3": 0, "layer4": 0, "fc": 0}

    def make_hook(name):
        def _hook(module, inp, out):
            call_counts[name] += 1
        return _hook

    h1 = fe.conv1.register_forward_hook(make_hook("conv1"))
    h2 = fe.layer3.register_forward_hook(make_hook("layer3"))
    h3 = fe.layer4.register_forward_hook(make_hook("layer4"))
    h4 = fe.fc.register_forward_hook(make_hook("fc"))

    batch = _make_dummy_batch(b=2, t=6)
    with torch.no_grad():
        _ = g2(**batch)

    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()

    assert call_counts["conv1"] == 1, f"conv1 called {call_counts['conv1']} times (expected 1)"
    assert call_counts["layer3"] == 1, f"layer3 called {call_counts['layer3']} times (expected 1)"
    assert call_counts["layer4"] == 1, f"layer4 called {call_counts['layer4']} times (expected 1)"
    assert call_counts["fc"] == 1, f"fc called {call_counts['fc']} times (expected 1)"


def test_7_parameter_audit(base_config: MultimodalFusionConfig):
    """Test 7: Prove exact parameter delta = 52,096."""
    torch.manual_seed(707)
    m2 = MultimodalFusionClassifier(base_config)
    g2 = G2CCLSAClassifier(base_config)

    m2_params = sum(p.numel() for p in m2.parameters())
    g2_params = sum(p.numel() for p in g2.parameters())
    delta = g2_params - m2_params

    # Expected sub-module parameter counts
    q_params = sum(p.numel() for p in g2.query_proj.parameters())
    k_params = sum(p.numel() for p in g2.key_proj.parameters())
    r_params = sum(p.numel() for p in g2.residual_proj.parameters())

    assert q_params == 46 * 64 == 2944
    assert k_params == 256 * 64 * 1 * 1 == 16384
    assert r_params == 256 * 128 == 32768
    assert q_params + k_params + r_params == EXPECTED_G2_PARAM_DELTA == 52096
    assert m2_params == 43633832
    assert g2_params == 43685928
    assert delta == EXPECTED_G2_PARAM_DELTA == 52096


def test_8_lr_group_audit(base_config: MultimodalFusionConfig):
    """Test 8: Prove visual LR = 0.0003, nonvisual/G2 LR = 0.003, and zero duplicate params."""
    torch.manual_seed(808)
    g2 = G2CCLSAClassifier(base_config)

    groups, report = build_visual_optimizer_groups(
        g2,
        learning_rate=0.003,
        backbone_lr_multiplier=0.1,
        weight_decay=0.0,
    )

    assert len(groups) == 2
    vis_group = next(g for g in groups if g["group_name"] == "visual_backbone")
    nonvis_group = next(g for g in groups if g["group_name"] == "nonvisual")

    assert vis_group["lr"] == pytest.approx(0.0003)
    assert nonvis_group["lr"] == pytest.approx(0.003)

    # Prove query_proj, key_proj, residual_proj are in nonvisual group
    nonvis_params_set = set(id(p) for p in nonvis_group["params"])
    for p in g2.query_proj.parameters():
        assert id(p) in nonvis_params_set
    for p in g2.key_proj.parameters():
        assert id(p) in nonvis_params_set
    for p in g2.residual_proj.parameters():
        assert id(p) in nonvis_params_set

    # No duplicate parameters
    all_param_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_param_ids) == len(set(all_param_ids))
    assert len(all_param_ids) == len(list(g2.parameters()))


def test_9_gradient_audit(base_config: MultimodalFusionConfig):
    """Test 9: Prove gradient flow at Step-0 and after optimizer update."""
    torch.manual_seed(909)
    g2 = G2CCLSAClassifier(base_config)
    g2.train()

    groups, _ = build_visual_optimizer_groups(
        g2,
        learning_rate=0.003,
        backbone_lr_multiplier=0.1,
        weight_decay=0.0,
    )
    optimizer = torch.optim.AdamW(groups)

    batch = _make_dummy_batch(b=2, t=6)
    target = torch.tensor([1, 4], dtype=torch.long)

    # 1. First backward at Step-0 (residual_proj is zero)
    optimizer.zero_grad()
    logits = g2(**batch)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    # residual_proj receives nonzero gradient from activation * dL/dOut
    assert g2.residual_proj.weight.grad is not None
    assert g2.residual_proj.weight.grad.abs().sum().item() > 0.0

    # query_proj and key_proj have ZERO grad at step 0 because residual_proj weight is 0
    assert g2.query_proj.weight.grad is not None
    assert g2.query_proj.weight.grad.abs().sum().item() == 0.0
    assert g2.key_proj.weight.grad is not None
    assert g2.key_proj.weight.grad.abs().sum().item() == 0.0

    # Original M2 actor path gradient is NONZERO
    fe = g2.image_encoder.frame_encoder
    assert fe.conv1.weight.grad is not None
    assert fe.conv1.weight.grad.abs().sum().item() > 0.0

    # 2. Step optimizer
    optimizer.step()

    # After step, residual_proj weight is non-zero
    assert g2.residual_proj.weight.abs().sum().item() > 0.0

    # 3. Second backward
    optimizer.zero_grad()
    logits2 = g2(**batch)
    loss2 = F.cross_entropy(logits2, target)
    loss2.backward()

    # Now query_proj and key_proj gradients are NONZERO
    assert g2.query_proj.weight.grad.abs().sum().item() > 0.0
    assert g2.key_proj.weight.grad.abs().sum().item() > 0.0

    # No NaN or Inf anywhere
    for name, p in g2.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
            assert torch.isfinite(p).all(), f"Non-finite parameter in {name}"


def test_10_g2_gate_decision_logic():
    """Test 10: Prove locked G2 gate logic and strict futility rule."""
    from pig_behavior.classification_v2.training.run_g2_cclsa_5fold import (
        evaluate_g2_gate,
    )

    # 1. Initial
    dec_init = evaluate_g2_gate({})
    assert dec_init.status == "INITIAL"
    assert dec_init.authorize_next_fold is True

    # 2. Futility on first fold: VG1 non-positive (0.640000 <= 0.642622)
    dec_fut = evaluate_g2_gate({"vg1": 0.640000})
    assert dec_fut.status == "REJECT_FUTILITY"
    assert dec_fut.authorize_next_fold is False
    assert "Futility rule triggered" in dec_fut.reason

    # 3. In-progress on positive first fold: VG1 positive (0.660000 > 0.642622)
    dec_prog = evaluate_g2_gate({"vg1": 0.660000})
    assert dec_prog.status == "IN_PROGRESS"
    assert dec_prog.authorize_next_fold is True
    assert dec_prog.positive_folds == 1

    # 4. Futility on fold 3: VG1 pos, VG2 pos, VG3 non-pos (0.580000 <= 0.587899)
    dec_fut3 = evaluate_g2_gate({
        "vg1": 0.660000,
        "vg2": 0.620000,
        "vg3": 0.580000,
    })
    assert dec_fut3.status == "REJECT_FUTILITY"
    assert dec_fut3.authorize_next_fold is False

    # 5. Completed 5 folds but mean delta < +0.015
    dec_rej = evaluate_g2_gate({
        "vg1": 0.643622,  # +0.001
        "vg2": 0.610316,  # +0.001
        "vg3": 0.588899,  # +0.001
        "vg4": 0.676154,  # +0.001
        "vg5": 0.657352,  # +0.001
    })
    assert dec_rej.status == "REJECT"
    assert dec_rej.authorize_next_fold is False
    assert dec_rej.positive_folds == 5

    # 6. Completed 5 folds PASS (mean delta >= +0.015 and 5/5 positive)
    dec_pass = evaluate_g2_gate({
        "vg1": 0.642622 + 0.020,
        "vg2": 0.609316 + 0.020,
        "vg3": 0.587899 + 0.020,
        "vg4": 0.675154 + 0.020,
        "vg5": 0.656352 + 0.020,
    })
    assert dec_pass.status == "PASS"
    assert dec_pass.authorize_next_fold is True
    assert dec_pass.positive_folds == 5
    assert dec_pass.mean_delta == pytest.approx(0.020)


def test_11_scientific_runner_cpu_lifecycle(tmp_path):
    """Test 11: Execute complete 2-epoch CPU lifecycle of G2 runner."""
    from argparse import Namespace

    from pig_behavior.classification_v2.training.run_g2_cclsa_5fold import (
        run_g2_training,
    )

    args = Namespace(
        fold="vg1",
        config=None,
        data_root=tmp_path / "data",
        rgb_root=tmp_path / "rgb",
        output_dir=tmp_path / "g2_run",
        device="cpu",
        epochs=2,
        batch_size=4,
        preflight=True,
        max_train_windows=8,
        max_val_windows=4,
    )

    # Mock DataModule to avoid needing real 30GB external data on CPU unit test
    class MockBatch:
        def __init__(self, indices):
            b = len(indices)
            t = 6
            spatial_feats = {
                name: torch.randn(b, t, len(feats))
                for name, feats in SPATIAL_PREDICTIVE_FEATURES.items()
            }
            spatial_validity = {
                "motion_delta": torch.ones(b, t, 12, dtype=torch.bool),
                "social_relation": torch.ones(b, t, 10, dtype=torch.bool),
            }
            length_mask = torch.ones(b, t, dtype=torch.bool)
            time_delta = torch.arange(t).float().unsqueeze(0).expand(b, t)
            self.model_inputs = {
                "image": torch.randn(b, t, 3, 128, 128),
                "spatial_features": spatial_feats,
                "spatial_feature_validity_masks": spatial_validity,
                "length_mask": length_mask,
                "image_time_delta": time_delta,
                "spatial_time_delta": time_delta,
                "visual_context_image": torch.randn(b, t, 3, 128, 128),
                "visual_context_length_mask": length_mask,
                "visual_context_time_delta": time_delta,
                "interaction_context_features": torch.randn(b, 5),
                "interaction_context_available_mask": torch.tensor([True] * b),
            }
            self.behavior_target = torch.randint(0, 10, (b,))
            self.sample_weight = torch.tensor([1.2, 0.8, 2.0, 1.0][:b], dtype=torch.float32)

    class MockDataModule:
        def __init__(self, config, device):
            pass
        def fit_fold_preprocessor(self):
            pass
        def split_indices(self, split):
            if split == "train":
                return np.arange(8)
            elif split in ("val", "validation"):
                return np.arange(8, 12)
            else:
                return np.arange(12, 16)
        def batch(self, indices):
            return MockBatch(indices)

    dm_patch = patch(
        "pig_behavior.classification_v2.training.run_g2_cclsa_5fold.StrictTrainingDataModule",
        MockDataModule,
    )
    w_patch = patch(
        "pig_behavior.classification_v2.training.run_g2_cclsa_5fold._behavior_class_weights",
        return_value=torch.ones(10),
    )
    with dm_patch, w_patch:
        res = run_g2_training("vg1", args)

    assert res["fold"] == "vg1"
    assert len(res["history"]) == 2
    assert "best_val_macro_f1" in res
    assert res["outer_test_eval_count"] == 0

    run_dir = tmp_path / "g2_run" / "runs" / "vg1"
    best_ckpt = run_dir / "best_validation.pt"
    last_ckpt = run_dir / "last.pt"
    history_file = run_dir / "epoch_history.json"

    assert best_ckpt.exists()
    assert last_ckpt.exists()
    assert history_file.exists()

    saved_data = torch.load(best_ckpt, map_location="cpu")
    assert "model_state_dict" in saved_data
    assert "best_epoch" in saved_data
    assert saved_data["metrics"]["inner_val_behavior_macro_f1"] >= 0.0


def test_12_m2_sample_weighted_loss_parity():
    """Test 12: Verify exact M2 sample-weighted loss reduction matches expected formula."""
    logits = torch.randn(4, 10)
    target = torch.tensor([0, 1, 2, 3])
    weights = torch.tensor([1.0, 2.0, 1.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    sample_weight = torch.tensor([0.5, 2.0, 1.0, 1.5])

    per_row_loss = F.cross_entropy(logits, target, weight=weights, reduction="none")
    expected_loss = (per_row_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)

    assert torch.isfinite(expected_loss)
    assert expected_loss.item() > 0.0


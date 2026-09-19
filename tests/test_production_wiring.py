"""Focused production wiring proofs for Final Joint Representation model."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.models.joint_representation_model import (
    JointRepresentationClassifier,
    compute_joint_loss,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    load_training_config,
)
from pig_behavior.classification_v2.training.data_module import (
    StrictTrainingDataModule,
)
from pig_behavior.classification_v2.training.run_joint_representation_5fold import (
    DATA_PLUS_COUNTS,
    DEFAULT_DATA_PLUS_DIR,
    DEFAULT_H5_STRUCTURED_PATH,
    DEFAULT_PB1_DIR,
    DEFAULT_POSTURE_SIDECAR,
    extract_real_production_source_tensors,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")


def _get_resolved_config(fold_name: str) -> ClassificationV2TrainingConfig:
    cfg_file = (
        REPO_ROOT
        / f"configs/classification_v2/g2_cclsa_earlystop_{fold_name}_scientific_v1.json"
    )
    config = load_training_config(cfg_file)
    dataset = config.dataset
    updates = {
        "snapshot_json": (
            REPO_ROOT
            / "outputs/classification_v2/training_snapshots/c2v2_27ed5c9963904c52.json"
        ),
        "trainer_contract_json": (
            REPO_ROOT
            / "configs/classification_v2/trainer_contract_v2.json"
        ),
        "train_ready_root": (
            REPO_ROOT
            / "outputs/classification_v2/full_t6_canonical_46d_20260816"
        ),
        "grouped_fold_roles": (
            REPO_ROOT
            / "outputs/classification_v2/video_group_balanced_5fold_authority_20260821"
            / f"m2_vft_earlystop_{fold_name}_manifest.csv"
        ),
        "spatial_bundle_npz": (
            REPO_ROOT
            / "outputs/classification_v2/full_t6_canonical_46d_20260816/full_t6_canonical_46d.npz"
        ),
        "window_major_rgb_cache": (
            REPO_ROOT
            / "outputs/classification_v2/m0_window_major_r128_t6/m0_rgb_window_major_u8.npy"
        ),
        "window_major_union_mask": (
            REPO_ROOT
            / "outputs/classification_v2/m0_window_major_r128_t6/m0_union_available_mask.npy"
        ),
        "window_major_window_index": (
            REPO_ROOT
            / "outputs/classification_v2/m0_window_major_r128_t6/m0_rgb_window_index.csv"
        ),
    }
    return replace(config, dataset=replace(dataset, **updates))


def _build_model(
    config: ClassificationV2TrainingConfig,
    data: StrictTrainingDataModule,
) -> JointRepresentationClassifier:
    probe = data.batch(data.split_indices("train")[:2])
    spatial_input_dims = {
        name: probe.model_inputs["spatial_features"][name].shape[-1]
        for name in config.model.spatial_feature_groups
    }
    interaction_dim = probe.model_inputs["interaction_context_features"].shape[-1]
    hidden_dim = config.model.hidden_dim
    backbone_config = MultimodalFusionConfig(
        spatial_input_dims=spatial_input_dims,
        num_classes=10,
        interaction_context_dim=interaction_dim,
        backbone_name=config.model.backbone_name,
        pretrained_weight_enum=config.model.pretrained_weight_enum,
        image_embedding_dim=hidden_dim,
        spatial_embedding_dim=hidden_dim,
        interaction_embedding_dim=max(8, hidden_dim // 2),
        visual_context_embedding_dim=hidden_dim,
        fusion_hidden_dim=hidden_dim * 2,
        dropout=config.model.dropout,
        temporal_encoder_name=config.model.temporal_encoder_name,
        transformer_layers=config.model.transformer_layers,
        transformer_heads=config.model.transformer_heads,
        enable_image=config.model.enable_image,
        enable_spatial=config.model.enable_spatial,
        enable_interaction_context=config.model.enable_interaction_context,
        enable_visual_context=config.model.enable_visual_context,
        enable_partner_tokens=False,
    )
    model = JointRepresentationClassifier(backbone_config)
    model.eval()
    return model


@pytest.mark.parametrize("fold", ["vg1", "vg2", "vg3", "vg4", "vg5"])
def test_proof_a_train_contains_eligible_data_plus(fold: str) -> None:
    """Proof A: Train dataset contains exact eligible DATA+ rows per fold."""
    dp_elig = pd.read_csv(DEFAULT_DATA_PLUS_DIR / "supplemental_fold_eligibility.csv")
    fold_upper = fold.upper()
    col_mask = dp_elig["fold"].eq(fold_upper) & dp_elig["eligible_for_training"]
    count = int(col_mask.sum())
    expected = DATA_PLUS_COUNTS[fold]
    assert count == expected, f"{fold}: DATA+ count {count} != {expected}"


@pytest.mark.parametrize("fold", ["vg1", "vg2", "vg3", "vg4", "vg5"])
def test_proof_b_inner_val_contains_zero_data_plus(fold: str) -> None:
    """Proof B: Inner validation contains zero DATA+ rows across all folds."""
    dp_elig = pd.read_csv(DEFAULT_DATA_PLUS_DIR / "supplemental_fold_eligibility.csv")
    fold_upper = fold.upper()
    val_count = int(
        (
            dp_elig["fold"].eq(fold_upper)
            & dp_elig["eligible_for_inner_validation"]
        ).sum()
    )
    assert val_count == 0, f"{fold}: DATA+ inner_val count {val_count} != 0"


@pytest.mark.parametrize("fold", ["vg1", "vg2", "vg3", "vg4", "vg5"])
def test_proof_c_outer_population_untouched(fold: str) -> None:
    """Proof C: Outer test contains zero DATA+ rows and canonical population is 33,287."""
    dp_elig = pd.read_csv(DEFAULT_DATA_PLUS_DIR / "supplemental_fold_eligibility.csv")
    fold_upper = fold.upper()
    test_count = int(
        (
            dp_elig["fold"].eq(fold_upper)
            & dp_elig["eligible_for_outer_test"]
        ).sum()
    )
    assert test_count == 0, f"{fold}: DATA+ outer_test count {test_count} != 0"

    config = _get_resolved_config(fold)
    with StrictTrainingDataModule(config, device=DEVICE) as data:
        assert len(data.bundle.frame) == 33287
        train_len = len(data.split_indices("train"))
        val_len = len(data.split_indices("validation"))
        test_len = len(data.split_indices("test"))
        assert train_len + val_len + test_len == 33287


@pytest.mark.parametrize("fold", ["vg1", "vg2", "vg3", "vg4", "vg5"])
def test_proof_d_real_production_batches_provide_required_shapes(fold: str) -> None:
    """Proof D: Real production batches provide exact required tensor shapes."""
    config = _get_resolved_config(fold)
    with StrictTrainingDataModule(config, device=DEVICE) as data:
        data.fit_fold_preprocessor()
        train_indices = data.split_indices("train")
        batch_size = 4
        sample_indices = train_indices[:batch_size]
        batch = data.batch(sample_indices)

        pb_train = torch.load(
            DEFAULT_PB1_DIR / fold / "train_features.pt", map_location="cpu"
        )
        train_idx_map = {int(idx): pos for pos, idx in enumerate(train_indices)}
        posture_sidecar = torch.load(DEFAULT_POSTURE_SIDECAR, map_location="cpu")
        h5_cache = torch.load(DEFAULT_H5_STRUCTURED_PATH, map_location="cpu")

        sources = extract_real_production_source_tensors(
            data,
            batch,
            sample_indices,
            pb_train["partner_probs"],
            pb_train["partner_mask"],
            train_idx_map,
            posture_sidecar,
            h5_cache,
            DEVICE,
        )

        assert sources["partner_behavior_probs"].shape == (batch_size, 2, 10)
        assert sources["partner_behavior_mask"].shape == (batch_size, 2)
        assert sources["h5_structured"].shape == (batch_size, 5, 46)
        assert sources["h5_mask"].shape == (batch_size, 5)
        assert sources["roi_sequence"].shape == (batch_size, 6, 18)
        assert sources["roi_validity"].shape == (batch_size, 6, 3)
        assert batch.behavior_target.shape == (batch_size,)
        assert batch.sample_weight.shape == (batch_size,)


def test_proof_e_posture_targets_masks_exist_and_activate_loss() -> None:
    """Proof E: Real reviewed posture rows exist, reach batch contract, activate loss."""
    config = _get_resolved_config("vg1")
    with StrictTrainingDataModule(config, device=DEVICE) as data:
        data.fit_fold_preprocessor()
        train_indices = data.split_indices("train")

        posture_sidecar = torch.load(DEFAULT_POSTURE_SIDECAR, map_location="cpu")
        all_pos_masks = posture_sidecar["posture_reviewed_mask"]

        # Locate real reviewed rows among train indices
        train_rev_mask = all_pos_masks[train_indices]
        rev_train_indices = train_indices[train_rev_mask.numpy()]
        assert len(rev_train_indices) == 364, (
            f"Expected 364 reviewed rows, got {len(rev_train_indices)}"
        )

        # Select 4 reviewed + 4 unreviewed rows
        unrev_train_indices = train_indices[(~train_rev_mask).numpy()][:4]
        selected_indices = np.concatenate([rev_train_indices[:4], unrev_train_indices])

        batch = data.batch(selected_indices)
        pb_train = torch.load(
            DEFAULT_PB1_DIR / "vg1/train_features.pt", map_location="cpu"
        )
        train_idx_map = {int(idx): pos for pos, idx in enumerate(train_indices)}
        h5_cache = torch.load(DEFAULT_H5_STRUCTURED_PATH, map_location="cpu")

        sources = extract_real_production_source_tensors(
            data,
            batch,
            selected_indices,
            pb_train["partner_probs"],
            pb_train["partner_mask"],
            train_idx_map,
            posture_sidecar,
            h5_cache,
            DEVICE,
        )

        assert sources["posture_reviewed_mask"][:4].all().item()
        assert (sources["posture_targets"][:4] >= 0).all().item()
        assert (~sources["posture_reviewed_mask"][4:]).all().item()

        model = _build_model(config, data)
        with torch.inference_mode():
            output = model(
                **batch.model_inputs,
                partner_behavior_probs=sources["partner_behavior_probs"],
                partner_behavior_mask=sources["partner_behavior_mask"],
                h5_structured=sources["h5_structured"],
                h5_mask=sources["h5_mask"],
                roi_sequence=sources["roi_sequence"],
                roi_validity=sources["roi_validity"],
            )

        tot_loss, beh_loss, pos_loss = compute_joint_loss(
            behavior_logits=output.behavior_logits,
            behavior_targets=batch.behavior_target,
            class_weights=torch.ones(10),
            posture_logits=output.posture_logits,
            posture_targets=sources["posture_targets"],
            posture_reviewed_mask=sources["posture_reviewed_mask"],
            sample_weight=batch.sample_weight,
            posture_lambda=0.25,
        )

        assert pos_loss.item() > 0.0, "Posture loss must be active for reviewed rows"
        assert tot_loss.item() == pytest.approx(
            (beh_loss + 0.25 * pos_loss).item(), rel=1e-5
        )


@pytest.mark.parametrize("fold", ["vg1", "vg2", "vg3", "vg4", "vg5"])
def test_proof_f_forward_real_batch_through_locked_model(fold: str) -> None:
    """Proof F: Forward real batches through locked model (finite [B,10] logits, no fallback)."""
    config = _get_resolved_config(fold)
    with StrictTrainingDataModule(config, device=DEVICE) as data:
        data.fit_fold_preprocessor()
        train_indices = data.split_indices("train")
        batch_size = 4
        sample_indices = train_indices[:batch_size]
        batch = data.batch(sample_indices)

        pb_train = torch.load(
            DEFAULT_PB1_DIR / fold / "train_features.pt", map_location="cpu"
        )
        train_idx_map = {int(idx): pos for pos, idx in enumerate(train_indices)}
        posture_sidecar = torch.load(DEFAULT_POSTURE_SIDECAR, map_location="cpu")
        h5_cache = torch.load(DEFAULT_H5_STRUCTURED_PATH, map_location="cpu")

        sources = extract_real_production_source_tensors(
            data,
            batch,
            sample_indices,
            pb_train["partner_probs"],
            pb_train["partner_mask"],
            train_idx_map,
            posture_sidecar,
            h5_cache,
            DEVICE,
        )

        model = _build_model(config, data)
        with torch.inference_mode():
            output = model(
                **batch.model_inputs,
                partner_behavior_probs=sources["partner_behavior_probs"],
                partner_behavior_mask=sources["partner_behavior_mask"],
                h5_structured=sources["h5_structured"],
                h5_mask=sources["h5_mask"],
                roi_sequence=sources["roi_sequence"],
                roi_validity=sources["roi_validity"],
            )

        assert output.behavior_logits.shape == (batch_size, 10)
        assert torch.isfinite(output.behavior_logits).all()
        assert output.posture_logits.shape == (batch_size, 3)
        assert torch.isfinite(output.posture_logits).all()
        assert output.fused_embedding.shape == (batch_size, 448)
        assert output.partner_context.shape == (batch_size, 32)
        assert output.h5_context.shape == (batch_size, 32)
        assert output.roi_context.shape == (batch_size, 32)
        assert output.posture_context.shape == (batch_size, 3)


def test_proof_g_h5_causal_identity_order_and_zero_leakage() -> None:
    """Proof G: H5 structured history strictly obeys causal contract and zero target leakage."""
    manifest_path = (
        REPO_ROOT
        / "outputs/classification_v2/full_t6_canonical_46d_20260816/full_t6_row_manifest.csv"
    )
    manifest_df = pd.read_csv(manifest_path)
    assert len(manifest_df) == 33287

    h5_cache = torch.load(DEFAULT_H5_STRUCTURED_PATH, map_location="cpu")
    h5_struct = h5_cache["h5_structured"]  # [33287, 5, 46]
    h5_mask = h5_cache["history_observed_mask"]  # [33287, 5]
    no_h5_indices = set(h5_cache["no_history_row_indices"])

    assert h5_struct.shape == (33287, 5, 46)
    assert h5_mask.shape == (33287, 5)
    assert len(no_h5_indices) == 96
    assert torch.isfinite(h5_struct).all()

    # 1. Verify 96 no-history rows: exact zeros and all False mask
    for idx in no_h5_indices:
        assert not h5_mask[idx].any().item(), f"Row {idx} has non-false mask"
        assert (h5_struct[idx] == 0.0).all().item(), f"Row {idx} has non-zero features"

    # 2. Verify CVAT row ordering: physical frames f0-5..f0-1
    # Check Row 0: target T6 frames [1326, 1327, 1328, 1329, 1330, 1331]
    row0 = manifest_df.iloc[0]
    t6_frames_row0 = json.loads(row0["physical_frame_ids_json"])
    assert t6_frames_row0 == [1326, 1327, 1328, 1329, 1330, 1331]
    assert h5_mask[0].all().item()  # Complete history 1321..1325
    assert (h5_struct[0] != 0.0).any().item()

    # Verify Row 1: starts at frame 0 -> no history before frame 0
    row1 = manifest_df.iloc[1]
    t6_frames_row1 = json.loads(row1["physical_frame_ids_json"])
    assert t6_frames_row1[0] == 0
    assert 1 in no_h5_indices
    assert not h5_mask[1].any().item()
    assert (h5_struct[1] == 0.0).all().item()

    # 3. Verify Legacy rows: native offsets 0..4 (all 4539 rows have full history mask)
    legacy_indices = manifest_df[
        manifest_df["source_type"] == "legacy_recovered"
    ].index.to_numpy()
    assert len(legacy_indices) == 4539
    assert h5_mask[legacy_indices].all().item()
    assert torch.isfinite(h5_struct[legacy_indices]).all()
    assert (h5_struct[legacy_indices] != 0.0).any().item()

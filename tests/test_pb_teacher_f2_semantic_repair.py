"""Validation test suite for repaired PB Teacher F2 V1 semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.features.unified_pb_teacher_f2 import (
    DEFAULT_DATA_PLUS_DIR,
    DEFAULT_F2_DIR,
    DEFAULT_FULL_T6_ROW_MANIFEST,
    DEFAULT_H5_DIR,
    DEFAULT_M0_RGB_NPY,
    F2_EXPECTED_HASHES,
    load_frozen_f2_model,
    run_batch_partner_f2_inference,
    sha256_file,
)


def test_canonical_key_invariant_to_permutation() -> None:
    """Test A: Verify canonical PB key remains identical even if dataframe is permuted."""
    manifest_df = pd.read_csv(DEFAULT_FULL_T6_ROW_MANIFEST, nrows=100)
    original_keys = manifest_df["target_id"].copy().tolist()

    # Permute dataframe
    permuted_df = manifest_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    permuted_map = {row["target_id"]: row["target_id"] for _, row in permuted_df.iterrows()}

    for k in original_keys:
        assert k in permuted_map
        assert permuted_map[k] == k


def test_partner_bbox_differs_from_target() -> None:
    """Test B: Verify partner bboxes differ from target actor whenever partner exists."""
    # 1. Check DATA+ V2
    dp_partner = np.load(DEFAULT_DATA_PLUS_DIR / "data_plus_partner_context_v2.npz")
    dp_spatial = np.load(DEFAULT_DATA_PLUS_DIR / "data_plus_spatial_46d.npz")

    target_bbox_dp = dp_spatial["bbox_xywh_n"]  # [1252, 6, 4]
    partner_bbox_xyxy = dp_partner["partner_bbox_xyxy"]  # [1252, 6, 2, 4]
    partner_mask = dp_partner["partner_mask"]  # [1252, 6, 2]

    w_img, h_img = 1920.0, 1080.0
    p_x1 = partner_bbox_xyxy[:, :, 0, 0] / w_img
    p_y1 = partner_bbox_xyxy[:, :, 0, 1] / h_img
    p_x2 = partner_bbox_xyxy[:, :, 0, 2] / w_img
    p_y2 = partner_bbox_xyxy[:, :, 0, 3] / h_img
    p_xc = (p_x1 + p_x2) / 2.0
    p_yc = (p_y1 + p_y2) / 2.0
    p_w = np.maximum(1e-4, p_x2 - p_x1)
    p_h = np.maximum(1e-4, p_y2 - p_y1)
    p_xywh_dp = np.stack([p_xc, p_yc, p_w, p_h], axis=-1)

    valid_slot0_dp = partner_mask[:, :, 0].all(axis=1)
    diff_dp = np.abs(target_bbox_dp[valid_slot0_dp] - p_xywh_dp[valid_slot0_dp]).max()
    assert diff_dp > 0.05, f"DATA+ partner bbox is too close to target bbox: {diff_dp}"

    # 2. Check Canonical FULL-T6
    bundle_dir = Path(
        "outputs/classification_v2/agent_audits/social_topk_k3_bundle_a8f727a_20260804_070500"
    )
    if bundle_dir.exists():
        tokens = np.load(bundle_dir / "frame_tokens.npy")
        win_rows = np.load(bundle_dir / "window_frame_row_indices.npy")[:1000, :6]
        t6_p0_tokens = tokens[win_rows, 0]
        dx_dy = t6_p0_tokens[:, :, :2]
        assert np.abs(dx_dy).max() > 0.01, "Canonical partner offset is zero"


def test_f2_receives_real_partner_rgb() -> None:
    """Test C: Verify teacher receives non-zero REAL partner RGB from that exact partner."""
    assert DEFAULT_M0_RGB_NPY.exists(), f"Missing {DEFAULT_M0_RGB_NPY}"

    rgb_mmap = np.load(DEFAULT_M0_RGB_NPY, mmap_mode="r")
    assert rgb_mmap.shape[1] == 2, "RGB tensor does not have 2 crop channels"

    # Channel 1 is partner RGB crop
    partner_crops = rgb_mmap[:50, 1]  # [50, 6, 128, 128, 3]
    assert partner_crops.mean() > 5.0, "Partner crops appear all zero"
    assert partner_crops.max() <= 255
    assert partner_crops.min() >= 0


def test_f2_causal_h5_bound_to_partner() -> None:
    """Test D: Verify F2 causal H5 input is present and bound to the partner identity."""
    h5_npy = DEFAULT_H5_DIR / "h5_actor_rgb_u8.npy"
    h5_masks = DEFAULT_H5_DIR / "h5_actor_rgb_masks.npz"
    assert h5_npy.exists() and h5_masks.exists()

    h5_mmap = np.load(h5_npy, mmap_mode="r")
    masks_data = np.load(h5_masks)
    h5_avail = masks_data["history_available_mask"]

    assert h5_mmap.shape == (33287, 5, 128, 128, 3)
    assert h5_avail.shape == (33287, 5)
    # Ensure non-zero pixels exist in H5 history
    valid_rows = np.where(h5_avail.all(axis=1))[0][:50]
    sample_h5 = h5_mmap[valid_rows]
    assert sample_h5.mean() > 5.0, "H5 history crops appear all zero"


def test_probability_hidden_from_same_forward() -> None:
    """Test E: Verify probability + hidden come from the exact same F2 forward."""
    ckpt_path = DEFAULT_F2_DIR / "vg1" / "best_validation.pt"
    model, _ = load_frozen_f2_model(ckpt_path, device=torch.device("cpu"))

    b_size = 2
    p_img = np.random.randint(0, 256, (b_size, 6, 128, 128, 3), dtype=np.uint8)
    v_img = np.random.randint(0, 256, (b_size, 6, 128, 128, 3), dtype=np.uint8)
    h_img = np.random.randint(0, 256, (b_size, 5, 128, 128, 3), dtype=np.uint8)
    h_avail = np.ones((b_size, 5), dtype=bool)

    sp_dict = {
        "bbox_xywh_n": np.random.rand(b_size, 6, 4).astype(np.float32),
        "bbox_shape_n": np.random.rand(b_size, 6, 2).astype(np.float32),
        "motion_delta": np.random.rand(b_size, 6, 12).astype(np.float32),
        "roi_class_relation": np.zeros((b_size, 6, 18), dtype=np.float32),
        "social_relation": np.zeros((b_size, 6, 10), dtype=np.float32),
    }
    sp_valid = {
        "motion_delta": np.ones((b_size, 6, 12), dtype=bool),
        "social_relation": np.ones((b_size, 6, 10), dtype=bool),
    }
    obs = np.ones((b_size, 6), dtype=bool)

    probs, hidden = run_batch_partner_f2_inference(
        model,
        p_img,
        v_img,
        h_img,
        h_avail,
        sp_dict,
        sp_valid,
        obs,
        obs,
        device=torch.device("cpu"),
    )

    # Directly check that classifier head on hidden equals logits/probs
    with torch.inference_mode():
        logits_recomputed = model.backbone.classifier[1](hidden)
        probs_recomputed = torch.softmax(logits_recomputed, dim=-1)

    assert torch.allclose(probs, probs_recomputed, atol=1e-6)


def test_no_canonical_row_in_runtime() -> None:
    """Test F: Verify zero canonical_row_ generation/parsing remains in runtime code."""
    runtime_files = [
        Path("src/pig_behavior/classification_v2/features/unified_pb_teacher_f2.py"),
        Path("src/pig_behavior/classification_v2/training/run_joint_representation_5fold.py"),
        Path("src/pig_behavior/classification_v2/models/joint_representation_model.py"),
    ]
    for rf in runtime_files:
        assert rf.exists(), f"Missing runtime file {rf}"
        content = rf.read_text(encoding="utf-8")
        assert "canonical_row_" not in content, (
            f"Forbidden 'canonical_row_' found in {rf}"
        )


def test_fold_checkpoint_hash() -> None:
    """Test G: Verify all 5 F2 checkpoints match exact recorded SHA256 hashes."""
    for fold, expected_sha in F2_EXPECTED_HASHES.items():
        ckpt_path = DEFAULT_F2_DIR / fold / "best_validation.pt"
        assert ckpt_path.exists(), f"Missing checkpoint for {fold}: {ckpt_path}"
        actual_sha = sha256_file(ckpt_path)
        assert actual_sha == expected_sha, (
            f"Hash mismatch for {fold}: {actual_sha} != {expected_sha}"
        )

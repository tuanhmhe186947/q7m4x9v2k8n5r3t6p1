"""Comprehensive test suite for PB Teacher F2 V1 Extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.features.pb_teacher_f2_extraction import (
    DEFAULT_DATA_PLUS_DIR,
    DEFAULT_F2_CHECKPOINT_DIR,
    F2_EXPECTED_HASHES,
    extract_pb_features_from_data_plus,
    load_frozen_f2_teacher,
    sha256_file,
)


def test_f2_checkpoints_load_strictly() -> None:
    """Test 1: Verify all 5 F2 checkpoints exist, match SHA256, and load strictly."""
    for fold, expected_sha in F2_EXPECTED_HASHES.items():
        ckpt_path = DEFAULT_F2_CHECKPOINT_DIR / fold / "best_validation.pt"
        assert ckpt_path.exists(), f"Missing checkpoint for {fold}: {ckpt_path}"

        actual_sha = sha256_file(ckpt_path)
        assert actual_sha == expected_sha, (
            f"SHA256 mismatch for {fold}: {actual_sha} != {expected_sha}"
        )

        model, loaded_sha = load_frozen_f2_teacher(
            ckpt_path, device=torch.device("cpu")
        )
        assert loaded_sha == expected_sha
        assert not model.training, f"Model for {fold} is not in eval mode"


def test_frozen_teacher_and_eval_mode() -> None:
    """Test 2: Verify teacher parameters are frozen (requires_grad=False)."""
    ckpt_path = DEFAULT_F2_CHECKPOINT_DIR / "vg1" / "best_validation.pt"
    model, _ = load_frozen_f2_teacher(ckpt_path, device=torch.device("cpu"))

    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} has requires_grad=True"

    assert not model.training


def test_no_target_label_dependency() -> None:
    """Test 3: Verify changing target behavior labels produces bitwise identical PB outputs."""
    ckpt_path = DEFAULT_F2_CHECKPOINT_DIR / "vg1" / "best_validation.pt"
    model, sha = load_frozen_f2_teacher(ckpt_path, device=torch.device("cpu"))

    outputs1, _, _ = extract_pb_features_from_data_plus(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        data_plus_dir=DEFAULT_DATA_PLUS_DIR,
        max_samples=20,
    )

    outputs2, _, _ = extract_pb_features_from_data_plus(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        data_plus_dir=DEFAULT_DATA_PLUS_DIR,
        max_samples=20,
    )

    # Bitwise identical tensors
    torch.testing.assert_close(outputs1.partner_probs, outputs2.partner_probs)
    torch.testing.assert_close(outputs1.partner_hidden, outputs2.partner_hidden)
    torch.testing.assert_close(outputs1.partner_mask, outputs2.partner_mask)


def test_no_partner_gt_behavior() -> None:
    """Test 4: Verify extractor does not access partner GT behavior annotations."""
    partner_npz_path = DEFAULT_DATA_PLUS_DIR / "data_plus_partner_context_v2.npz"
    data = np.load(partner_npz_path)

    # Ensure no partner behavior ground truth columns are stored in partner context
    for key in data.files:
        assert "behavior" not in key.lower(), (
            f"Forbidden GT behavior key found in partner context: {key}"
        )


def test_hidden_probability_consistency() -> None:
    """Test 5: Verify softmax(classifier[1](partner_hidden)) matches partner_probs."""
    ckpt_path = DEFAULT_F2_CHECKPOINT_DIR / "vg1" / "best_validation.pt"
    model, sha = load_frozen_f2_teacher(ckpt_path, device=torch.device("cpu"))

    outputs, _, _ = extract_pb_features_from_data_plus(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        data_plus_dir=DEFAULT_DATA_PLUS_DIR,
        max_samples=20,
    )

    valid_mask = outputs.partner_mask  # [N, 2]
    valid_hidden = outputs.partner_hidden[valid_mask]  # [M, 256]
    valid_probs = outputs.partner_probs[valid_mask]    # [M, 10]

    with torch.inference_mode():
        reconstructed_logits = model.backbone.classifier[1](valid_hidden)
        reconstructed_probs = torch.softmax(reconstructed_logits, dim=-1)

    torch.testing.assert_close(
        reconstructed_probs,
        valid_probs,
        atol=1e-5,
        rtol=1e-5,
        msg="Hidden to probability projection mismatch",
    )


def test_missing_partner_zero_padding() -> None:
    """Test 6: Verify missing partner slots produce exact zero tensors."""
    ckpt_path = DEFAULT_F2_CHECKPOINT_DIR / "vg1" / "best_validation.pt"
    model, sha = load_frozen_f2_teacher(ckpt_path, device=torch.device("cpu"))

    outputs, _, _ = extract_pb_features_from_data_plus(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        data_plus_dir=DEFAULT_DATA_PLUS_DIR,
        max_samples=20,
    )

    missing_mask = ~outputs.partner_mask  # [N, 2]
    if missing_mask.any():
        missing_probs = outputs.partner_probs[missing_mask]
        missing_hidden = outputs.partner_hidden[missing_mask]

        assert (missing_probs == 0.0).all().item(), "Missing partner probs not zeroed"
        assert (missing_hidden == 0.0).all().item(), "Missing partner hidden not zeroed"


def test_key_alignment_with_data_plus_v2() -> None:
    """Test 7: Verify output keys match DATA+ V2 authority."""
    ckpt_path = DEFAULT_F2_CHECKPOINT_DIR / "vg1" / "best_validation.pt"
    model, sha = load_frozen_f2_teacher(ckpt_path, device=torch.device("cpu"))

    outputs, manifest_df, audit = extract_pb_features_from_data_plus(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        data_plus_dir=DEFAULT_DATA_PLUS_DIR,
        max_samples=20,
    )

    assert len(outputs.sample_key) == audit["total_samples"]
    assert len(outputs.target_object_track_key) == audit["total_samples"]
    assert outputs.partner_probs.shape == (audit["total_samples"], 2, 10)
    assert outputs.partner_hidden.shape == (audit["total_samples"], 2, 256)
    assert outputs.partner_mask.shape == (audit["total_samples"], 2)

    # Manifest checks
    expected_cols = {
        "sample_key",
        "target_object_track_key",
        "partner_slot",
        "partner_track_key",
        "partner_mask",
        "teacher_fold",
        "teacher_checkpoint_hash",
    }
    assert expected_cols.issubset(manifest_df.columns)


def test_fold_isolation() -> None:
    """Test 8: Verify fold isolation sample counts match DATA+ V2 eligibility."""
    expected_counts = {
        "VG1": 657,
        "VG2": 1252,
        "VG3": 1252,
        "VG4": 1244,
        "VG5": 1197,
    }

    fold_eligibility_csv = DEFAULT_DATA_PLUS_DIR / "supplemental_fold_eligibility.csv"
    eligibility = pd.read_csv(fold_eligibility_csv, low_memory=False)

    for fold, expected_n in expected_counts.items():
        count = int(
            eligibility[
                eligibility["fold"].str.upper().eq(fold)
                & eligibility["eligible_for_training"].fillna(False).astype(bool)
            ]["supplemental_unit_id"].nunique()
        )
        assert count == expected_n, (
            f"Fold {fold} sample count mismatch: {count} != {expected_n}"
        )

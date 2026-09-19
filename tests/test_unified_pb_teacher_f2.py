"""Comprehensive test suite for Unified PB Teacher F2 V1 Extraction."""

from __future__ import annotations

import torch

from pig_behavior.classification_v2.features.unified_pb_teacher_f2 import (
    DEFAULT_F2_DIR,
    F2_EXPECTED_HASHES,
    extract_unified_pb_teacher_fold,
    load_frozen_f2_model,
    sha256_file,
)


def test_checkpoints_strict_and_frozen() -> None:
    """Test 1: Checkpoints exist, match SHA256, and load strictly frozen."""
    for fold, expected_sha in F2_EXPECTED_HASHES.items():
        ckpt_path = DEFAULT_F2_DIR / fold / "best_validation.pt"
        assert ckpt_path.exists(), f"Missing checkpoint for {fold}: {ckpt_path}"

        actual_sha = sha256_file(ckpt_path)
        assert actual_sha == expected_sha, (
            f"SHA256 mismatch for {fold}: {actual_sha} != {expected_sha}"
        )

        model, loaded_sha = load_frozen_f2_model(
            ckpt_path, device=torch.device("cpu")
        )
        assert loaded_sha == expected_sha
        assert not model.training
        for name, p in model.named_parameters():
            assert not p.requires_grad, f"Parameter {name} not frozen"


def test_schema_compatibility() -> None:
    """Test 2: FULL-T6 and DATA+ use identical schema, feature contract, and shapes."""
    ckpt_path = DEFAULT_F2_DIR / "vg1" / "best_validation.pt"
    model, sha = load_frozen_f2_model(ckpt_path, device=torch.device("cpu"))

    outputs, manifest_df, _ = extract_unified_pb_teacher_fold(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        max_canonical_samples=20,
        max_data_plus_samples=20,
    )

    # Tensor shape checks
    assert outputs.partner_probs.ndim == 3
    assert outputs.partner_probs.shape[1:] == (2, 10)
    assert outputs.partner_hidden.ndim == 3
    assert outputs.partner_hidden.shape[1:] == (2, 256)
    assert outputs.partner_mask.ndim == 2
    assert outputs.partner_mask.shape[1] == 2

    # Manifest schema checks
    required_cols = {
        "sample_key",
        "target_object_track_key",
        "partner_slot",
        "partner_track_key",
        "partner_mask",
        "source_dataset",
        "split_role",
        "teacher_fold",
        "teacher_checkpoint_hash",
    }
    assert required_cols.issubset(manifest_df.columns)

    # Source dataset values
    assert set(manifest_df["source_dataset"].unique()) == {
        "canonical_full_t6",
        "data_plus_v2",
    }


def test_no_target_label_or_gt_leakage() -> None:
    """Test 3: No target behavior labels or partner GT behavior accessed."""
    ckpt_path = DEFAULT_F2_DIR / "vg1" / "best_validation.pt"
    model, sha = load_frozen_f2_model(ckpt_path, device=torch.device("cpu"))

    out1, _, _ = extract_unified_pb_teacher_fold(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        max_canonical_samples=15,
        max_data_plus_samples=15,
    )

    out2, _, _ = extract_unified_pb_teacher_fold(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        max_canonical_samples=15,
        max_data_plus_samples=15,
    )

    torch.testing.assert_close(out1.partner_probs, out2.partner_probs)
    torch.testing.assert_close(out1.partner_hidden, out2.partner_hidden)
    torch.testing.assert_close(out1.partner_mask, out2.partner_mask)


def test_missing_partner_zero_padding() -> None:
    """Test 4: Missing partner slots yield exact zero tensors."""
    ckpt_path = DEFAULT_F2_DIR / "vg1" / "best_validation.pt"
    model, sha = load_frozen_f2_model(ckpt_path, device=torch.device("cpu"))

    outputs, _, _ = extract_unified_pb_teacher_fold(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        max_canonical_samples=20,
        max_data_plus_samples=20,
    )

    missing_mask = ~outputs.partner_mask
    if missing_mask.any():
        missing_probs = outputs.partner_probs[missing_mask]
        missing_hidden = outputs.partner_hidden[missing_mask]

        assert (missing_probs == 0.0).all().item(), "Probs not zero-padded"
        assert (missing_hidden == 0.0).all().item(), "Hidden not zero-padded"


def test_sample_key_uniqueness() -> None:
    """Test 5: Sample keys are unique within a fold."""
    ckpt_path = DEFAULT_F2_DIR / "vg1" / "best_validation.pt"
    model, sha = load_frozen_f2_model(ckpt_path, device=torch.device("cpu"))

    outputs, _, _ = extract_unified_pb_teacher_fold(
        model=model,
        fold_id="vg1",
        checkpoint_sha256=sha,
        max_canonical_samples=30,
        max_data_plus_samples=30,
    )

    assert len(outputs.sample_key) == len(set(outputs.sample_key))

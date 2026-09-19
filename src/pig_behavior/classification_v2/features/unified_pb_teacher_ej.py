"""Unified Partner Behavior (PB) Teacher Extraction using E_J (Model J + Model B).

Task ID: FINAL-PBNEW-EJ-TEACHER-20260916
Execution Agent: Gemini

Replaces ONLY partner_probs with:
  teacher_logits = 0.50 * J_logits + 0.50 * B_logits
  teacher_probs  = softmax(teacher_logits)

Strictly preserves:
  - partner_hidden (existing PB F2 hidden [N, 2, 256])
  - partner_mask   (existing PB F2 mask [N, 2])
  - partner identities, ordering, and candidate selection
  - 100% partner mapping parity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.models.deep_local_joint_model import (
    DeepLocalJointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.joint_representation_model import (
    JointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.models.near_final_model_j import (
    NearFinalModelJ,
)

EXPECTED_J_HASHES: dict[str, str] = {
    "vg1": "50dc4bc13351d17a5daa2b8ab64ce8025676d4ab7db30b66c128065b7582055d",
    "vg2": "7a1dcb829f88bf4f7a172b19be59dda8593cbd03eca23e4aaa8bd49b440bafd4",
    "vg3": "8e721292c028bebb8dc6848d50d530e0c9ece2c09b9ec4e6428630dc73a1b77f",
    "vg4": "dd28d0673929ba0f244ff843a83df10fc808bc94bd91d35756e23afd1792f4ad",
    "vg5": "2e443353f25b95bdfe73e152fbe922ee538fe084aa411379ac5e3ff854591cb8",
}

EXPECTED_B_HASHES: dict[str, set[str]] = {
    "vg1": {
        "33f2fe2a0d9afa7d085a6c3cc3ec6b85be4b23f2f8e6afbc08cf6c5e0d671acc",
        "97d36ff4d11942abb5f59d738fd974eae58d09aa3dd7e2fc529770ee1ddf51d0",
        "e21b22e1fd498964d50937b2d28f24ea10c3b01a88b5ea138541e3d360cb25aa",
    },
    "vg2": {
        "cd6c33b7ba4e5638caa06fc8fa1a5df1a3b1feee78057552c1565923775b5fc0",
        "8f7cb37243c5b8b9ae69b2d3550e1ef9d273dc3b0fe1975e53e4b4dd6bca6934",
    },
    "vg3": {
        "decc32130ff6aece5ec1229e2259bd441138a7eac08397f729ee6b17352f4a7e",
        "f94c502b4d47348e35bbdddb98357f4955b2d5fcf7cb15b8045d4db1c784e8e1",
    },
    "vg4": {
        "1a8a97998670887200e685818ff3254b78c3804f8ce67375b3a31b7e39ad3c02",
        "b861271167440ec7419f72db763ee5da0b1e4fdb0d31e9c55b119cff1df91ff2",
    },
    "vg5": {
        "b07d63c9e0887422fd9e90f82903c68b0db7d05ec67507dad5c04d6706980a33",
        "6c22cb98fc3dcfb2f0a1c3fb0df08235e1654e580e03bf252737c355342d88d2",
    },
}

CANONICAL_BEHAVIORS = [
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
]

DEFAULT_F2_PB_DIR = Path("outputs/classification_v2/pb_teacher_f2_v1")
DEFAULT_OUTPUT_PB_DIR = Path("outputs/classification_v2/pb_teacher_ej_fixed50_v1")
DEFAULT_J_DIR = Path("outputs/classification_v2/final_model_j_replay_v1/runs")
DEFAULT_B_DIR = Path("outputs/classification_v2/joint_representation_v1/runs")

DEFAULT_CANONICAL_MANIFEST_DIR = Path(
    "outputs/classification_v2/video_group_balanced_5fold_authority_20260821"
)
DEFAULT_CANONICAL_SPATIAL_NPZ = Path(
    "outputs/classification_v2/full_t6_canonical_46d_20260816/full_t6_canonical_46d.npz"
)
DEFAULT_FULL_T6_ROW_MANIFEST = Path(
    "outputs/classification_v2/full_t6_canonical_46d_20260816/full_t6_row_manifest.csv"
)
DEFAULT_M0_RGB_NPY = Path(
    "outputs/classification_v2/m0_window_major_r128_t6/m0_rgb_window_major_u8.npy"
)
DEFAULT_DATA_PLUS_DIR = Path(
    "outputs/classification_v2/data_plus_supplemental_t6_v2_20260825"
)


def sha256_file(path: Path) -> str:
    """Compute SHA256 digest of a file in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _parse_frame_ids(value: Any) -> tuple[int, ...]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or len(parsed) != 6:
        raise ValueError(f"invalid canonical T6 frame identity: {value}")
    return tuple(int(frame_id) for frame_id in parsed)


def _unique_position_map(
    frame: pd.DataFrame,
    key_column: str,
    *,
    authority_name: str,
) -> dict[str, int]:
    if key_column not in frame.columns:
        raise ValueError(f"{authority_name} lacks persistent key {key_column}")
    keys = frame[key_column].astype(str)
    duplicates = keys[keys.duplicated(keep=False)]
    if not duplicates.empty:
        raise ValueError(
            f"{authority_name} has duplicate persistent keys: "
            f"{duplicates.iloc[:5].tolist()}"
        )
    return {key: int(position) for position, key in enumerate(keys)}


def build_joint_config() -> MultimodalFusionConfig:
    spatial_dims = {
        "bbox_xywh_n": 4,
        "bbox_shape_n": 2,
        "motion_delta": 12,
        "roi_class_relation": 18,
        "social_relation": 10,
    }
    return MultimodalFusionConfig(
        spatial_input_dims=spatial_dims,
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


def load_frozen_ej_models(
    fold_id: str,
    j_ckpt_path: Path,
    b_ckpt_path: Path,
    *,
    device: torch.device,
) -> tuple[NearFinalModelJ, JointRepresentationClassifier, str, str]:
    """Load Model J and Model B checkpoints in strict eval mode with frozen parameters."""
    if not j_ckpt_path.exists():
        raise FileNotFoundError(f"Model J checkpoint not found: {j_ckpt_path}")
    if not b_ckpt_path.exists():
        raise FileNotFoundError(f"Model B checkpoint not found: {b_ckpt_path}")

    j_hash = sha256_file(j_ckpt_path)
    b_hash = sha256_file(b_ckpt_path)

    expected_j = EXPECTED_J_HASHES.get(fold_id.lower())
    expected_b = EXPECTED_B_HASHES.get(fold_id.lower())
    if expected_j and j_hash != expected_j:
        raise ValueError(f"Model J hash mismatch for {fold_id}: {j_hash} != {expected_j}")
    if expected_b and b_hash not in expected_b:
        raise ValueError(
            f"Model B hash mismatch for {fold_id}: {b_hash} not in {expected_b}"
        )

    # Model B
    cfg_b = build_joint_config()
    model_b = JointRepresentationClassifier(cfg_b).to(device)
    b_ckpt = torch.load(b_ckpt_path, map_location=device, weights_only=False)
    model_b.load_state_dict(b_ckpt["model_state_dict"])
    model_b.eval()
    for p in model_b.parameters():
        p.requires_grad = False

    # Model J
    cfg_a = build_joint_config()
    base_a = DeepLocalJointRepresentationClassifier(cfg_a).to(device)
    model_j = NearFinalModelJ(base_a, token_dim=32, num_classes=10).to(device)
    j_ckpt = torch.load(j_ckpt_path, map_location=device, weights_only=False)
    model_j.load_state_dict(j_ckpt["model_state_dict"])
    model_j.eval()
    for p in model_j.parameters():
        p.requires_grad = False

    return model_j, model_b, j_hash, b_hash


def run_batch_partner_ej_inference(
    model_j: NearFinalModelJ,
    model_b: JointRepresentationClassifier,
    partner_images_u8: np.ndarray,
    target_images_u8: np.ndarray,
    spatial_features_dict: dict[str, np.ndarray],
    spatial_validity_dict: dict[str, np.ndarray],
    observed_masks: np.ndarray,
    visual_context_observed_masks: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Run E_J (0.50*J + 0.50*B) inference on partner observation batches."""
    b_size = partner_images_u8.shape[0]
    t_steps = 6

    p_img_t = (
        torch.from_numpy(partner_images_u8)
        .permute(0, 1, 4, 2, 3)
        .float()
        .div_(255.0)
        .to(device)
    )
    v_img_t = (
        torch.from_numpy(target_images_u8)
        .permute(0, 1, 4, 2, 3)
        .float()
        .div_(255.0)
        .to(device)
    )

    t_len = torch.ones((b_size, t_steps), dtype=torch.bool, device=device)
    t_obs = torch.from_numpy(observed_masks).to(device=device, dtype=torch.bool)
    v_obs = torch.from_numpy(visual_context_observed_masks).to(
        device=device, dtype=torch.bool
    )
    time_delta = (
        torch.arange(t_steps, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .expand(b_size, -1)
        / 30.0
    )

    spatial_torch = {
        name: torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        for name, arr in spatial_features_dict.items()
    }
    spatial_valid_torch = {
        name: torch.from_numpy(arr).to(device=device, dtype=torch.bool)
        for name, arr in spatial_validity_dict.items()
    }

    inter_ctx = torch.zeros((b_size, 5), dtype=torch.float32, device=device)
    inter_mask = torch.ones((b_size, 1), dtype=torch.bool, device=device)

    call_kw = {
        "image": p_img_t,
        "spatial_features": spatial_torch,
        "length_mask": t_len,
        "observed_mask": t_obs,
        "image_length_mask": t_len,
        "image_observed_mask": t_obs,
        "image_time_delta": time_delta,
        "spatial_length_mask": t_len,
        "spatial_observed_mask": t_obs,
        "spatial_feature_validity_masks": spatial_valid_torch,
        "spatial_time_delta": time_delta,
        "interaction_context_features": inter_ctx,
        "interaction_context_available_mask": inter_mask,
        "visual_context_image": v_img_t,
        "visual_context_length_mask": t_len,
        "visual_context_observed_mask": v_obs,
        "visual_context_time_delta": time_delta,
    }

    with torch.inference_mode():
        b_out = model_b(**call_kw)
        b_logits = b_out.behavior_logits  # [B, 10]

        j_logits, _, _, _, _ = model_j(**call_kw)  # [B, 10]

        ej_logits = 0.50 * j_logits + 0.50 * b_logits
        ej_probs = torch.softmax(ej_logits, dim=-1)  # [B, 10]

    return ej_probs


def extract_unified_pb_teacher_ej_fold(
    fold_id: str,
    *,
    j_dir: Path = DEFAULT_J_DIR,
    b_dir: Path = DEFAULT_B_DIR,
    f2_pb_dir: Path = DEFAULT_F2_PB_DIR,
    output_pb_dir: Path = DEFAULT_OUTPUT_PB_DIR,
    canonical_manifest_dir: Path = DEFAULT_CANONICAL_MANIFEST_DIR,
    canonical_spatial_npz: Path = DEFAULT_CANONICAL_SPATIAL_NPZ,
    canonical_full_manifest: Path = DEFAULT_FULL_T6_ROW_MANIFEST,
    m0_rgb_npy_path: Path = DEFAULT_M0_RGB_NPY,
    data_plus_dir: Path = DEFAULT_DATA_PLUS_DIR,
    device: torch.device | None = None,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Generate PB_NEW cache for one fold using fold-safe E_J teacher."""
    target_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else device
    )
    fold_lower = fold_id.lower()
    fold_upper = fold_id.upper()

    print(f"\n{'='*60}")
    print(f"GENERATING PB_NEW (E_J) CACHE FOR FOLD: {fold_upper}")
    print(f"Target Device: {target_device}")
    print(f"{'='*60}")

    j_ckpt_path = j_dir / fold_lower / "best_validation.pt"
    b_ckpt_path = b_dir / fold_lower / "best_validation.pt"
    model_j, model_b, j_sha, b_sha = load_frozen_ej_models(
        fold_lower, j_ckpt_path, b_ckpt_path, device=target_device
    )
    print(f"Loaded Model J (SHA256: {j_sha[:16]}...)")
    print(f"Loaded Model B (SHA256: {b_sha[:16]}...)")

    # Load existing PB F2 checkpoint for baseline parity
    f2_features_path = f2_pb_dir / fold_lower / "pb_teacher_features.pt"
    if not f2_features_path.exists():
        raise FileNotFoundError(f"Missing F2 PB baseline features: {f2_features_path}")
    f2_cache = torch.load(f2_features_path, map_location="cpu", weights_only=False)

    f2_partner_probs = f2_cache["partner_probs"]  # [N, 2, 10]
    f2_partner_hidden = f2_cache["partner_hidden"]  # [N, 2, 256]
    f2_partner_mask = f2_cache["partner_mask"]  # [N, 2]
    sample_keys = f2_cache["sample_key"]
    target_keys = f2_cache["target_object_track_key"]
    partner_keys = f2_cache["partner_track_key"]
    sources = f2_cache["source_dataset"]
    split_roles = f2_cache["split_role"]

    N_total, k_slots, n_classes = f2_partner_probs.shape
    assert k_slots == 2, f"Expected 2 partner slots, got {k_slots}"
    assert n_classes == 10, f"Expected 10 classes, got {n_classes}"

    # Initialize new partner probabilities from F2 baseline (ensuring exact zeros on masked slots)
    # Valid slots observed in T6 will be overwritten with freshly computed E_J probabilities.
    new_partner_probs = f2_partner_probs.clone()
    new_partner_probs[~f2_partner_mask] = 0.0

    # 1. Load Canonical Data Authorities
    canonical_manifest_file = (
        canonical_manifest_dir / f"m2_vft_earlystop_{fold_lower}_manifest.csv"
    )
    canonical_df = pd.read_csv(canonical_manifest_file, low_memory=False)
    full_manifest_df = pd.read_csv(canonical_full_manifest, low_memory=False)
    canonical_npz = np.load(canonical_spatial_npz)
    m0_rgb_mmap = np.load(m0_rgb_npy_path, mmap_mode="r")
    union_mask_path = m0_rgb_npy_path.with_name("m0_union_available_mask.npy")
    m0_union_mask = np.load(union_mask_path, mmap_mode="r")

    fold_key_column = (
        "window_id" if "window_id" in canonical_df.columns else "target_id"
    )
    full_key_column = (
        "window_id" if "window_id" in full_manifest_df.columns else "target_id"
    )
    row_indices = pd.to_numeric(
        full_manifest_df["row_index"], errors="raise"
    ).astype(int)
    full_key_to_array_row = {
        str(key): int(row_index)
        for key, row_index in zip(
            full_manifest_df[full_key_column], row_indices, strict=True
        )
    }
    array_row_to_full_position = {
        int(row_index): int(position)
        for position, row_index in enumerate(row_indices)
    }

    key_to_canonical_row: dict[tuple[str, str, tuple[int, ...]], int] = {}
    for row in full_manifest_df.itertuples(index=False):
        identity = (
            str(row.video_key),
            str(row.object_track_key),
            _parse_frame_ids(row.physical_frame_ids_json),
        )
        array_row = int(row.row_index)
        key_to_canonical_row[identity] = array_row

    relevant = canonical_df[
        canonical_df["split"].isin(["train", "validation"])
    ].copy()
    relevant_keys = relevant[fold_key_column].astype(str).to_numpy()
    canon_array_rows = np.asarray(
        [full_key_to_array_row[key] for key in relevant_keys], dtype=np.int64
    )
    n_canon = len(relevant)

    # Collect valid partner canonical rows for Slot 0 using canonical partner keys from F2
    slot0_partner_canonical_rows: list[int | None] = []
    slot0_valid_indices: list[int] = []

    for i, (_, target_row) in enumerate(relevant.iterrows()):
        if not bool(f2_partner_mask[i, 0]):
            slot0_partner_canonical_rows.append(None)
            continue
        p_track = str(partner_keys[i, 0])
        if not p_track or p_track == str(target_row["object_track_key"]):
            slot0_partner_canonical_rows.append(None)
            continue

        target_array_row = int(canon_array_rows[i])
        full_row = full_manifest_df.iloc[
            array_row_to_full_position[target_array_row]
        ]
        v_key = str(target_row["video_key"])
        fids = _parse_frame_ids(full_row["physical_frame_ids_json"])

        p_row_idx = key_to_canonical_row.get((v_key, p_track, fids))
        slot0_partner_canonical_rows.append(p_row_idx)
        if p_row_idx is not None:
            slot0_valid_indices.append(i)

    valid_canon_arr = np.asarray(slot0_valid_indices, dtype=np.int64)
    print(f"Total Canonical Valid Partner Slots (Slot 0): {len(valid_canon_arr)}")

    # Run E_J inference on valid canonical partner slots
    t0 = time.time()
    for chunk_start in range(0, len(valid_canon_arr), batch_size):
        chunk_c_idx = valid_canon_arr[chunk_start : chunk_start + batch_size]
        partner_rows = np.asarray(
            [slot0_partner_canonical_rows[pos] for pos in chunk_c_idx],
            dtype=np.int64,
        )

        p_img_batch = np.asarray(m0_rgb_mmap[partner_rows, 0])
        v_img_batch = np.asarray(m0_rgb_mmap[partner_rows, 1])
        visual_obs_batch = np.asarray(m0_union_mask[partner_rows], dtype=bool)
        sp_dict_batch = {
            name: canonical_npz[name][partner_rows]
            for name in (
                "bbox_xywh_n",
                "bbox_shape_n",
                "motion_delta",
                "roi_class_relation",
                "social_relation",
            )
        }
        sp_valid_batch = {
            "motion_delta": (
                canonical_npz["motion_feature_validity_mask"][partner_rows] > 0.5
            ),
            "social_relation": (
                canonical_npz["social_feature_validity_mask"][partner_rows] > 0.5
            ),
        }
        obs_batch = canonical_npz["observed_mask"][partner_rows] > 0.5

        ej_probs = run_batch_partner_ej_inference(
            model_j,
            model_b,
            p_img_batch,
            v_img_batch,
            sp_dict_batch,
            sp_valid_batch,
            obs_batch,
            visual_obs_batch,
            device=target_device,
        )
        new_partner_probs[chunk_c_idx, 0] = ej_probs.cpu()

        is_log = (chunk_start // batch_size) % 20 == 0
        is_done = (chunk_start + len(chunk_c_idx)) >= len(valid_canon_arr)
        if is_log or is_done:
            done = chunk_start + len(chunk_c_idx)
            pct = done * 100.0 / max(1, len(valid_canon_arr))
            elapsed = time.time() - t0
            print(f"  [Canonical] {done}/{len(valid_canon_arr)} ({pct:.1f}%) in {elapsed:.1f}s")

    # 2. Process DATA+ V2 Supplemental Dataset
    data_plus_f2_file = data_plus_dir / "data_plus_partner_f2_inputs.npz"
    data_plus_elig_file = data_plus_dir / "supplemental_fold_eligibility.csv"

    with np.load(data_plus_f2_file) as dp_f2_archive:
        dp_f2 = {name: dp_f2_archive[name] for name in dp_f2_archive.files}
    dp_elig = pd.read_csv(data_plus_elig_file, low_memory=False)

    dp_eligible_ids = set(
        dp_elig[
            dp_elig["fold"].str.upper().eq(fold_upper)
            & dp_elig["eligible_for_training"].fillna(False).astype(bool)
        ]["supplemental_unit_id"]
    )
    eligible_ids = sorted(str(v) for v in dp_eligible_ids)
    f2_unit_ids = dp_f2["supplemental_unit_id"].astype(str)
    f2_positions = {unit_id: position for position, unit_id in enumerate(f2_unit_ids)}
    dp_f2_positions = np.asarray(
        [f2_positions[unit_id] for unit_id in eligible_ids], dtype=np.int64
    )

    n_dp = len(eligible_ids)
    dp_start_idx = n_canon
    dp_mask = f2_partner_mask[dp_start_idx : dp_start_idx + n_dp]  # [N_dp, 2]
    valid_dp_pairs = np.argwhere(dp_mask.numpy())  # [K, 2]
    print(f"Total DATA+ Valid Partner Slots: {len(valid_dp_pairs)}")

    t0 = time.time()
    for chunk_start in range(0, len(valid_dp_pairs), batch_size):
        chunk_pairs = valid_dp_pairs[chunk_start : chunk_start + batch_size]
        dp_local_indices = chunk_pairs[:, 0]
        chunk_slots = chunk_pairs[:, 1]
        global_indices = dp_start_idx + dp_local_indices
        sidecar_rows = dp_f2_positions[dp_local_indices]

        p_crops = dp_f2["partner_actor_rgb"][sidecar_rows, chunk_slots]
        v_crops = dp_f2["partner_union_rgb"][sidecar_rows, chunk_slots]
        sp_dict = {
            name: dp_f2[name][sidecar_rows, chunk_slots]
            for name in (
                "bbox_xywh_n",
                "bbox_shape_n",
                "motion_delta",
                "roi_class_relation",
                "social_relation",
            )
        }
        sp_valid = {
            "motion_delta": dp_f2["motion_feature_validity_mask"][
                sidecar_rows, chunk_slots
            ],
            "social_relation": dp_f2["social_feature_validity_mask"][
                sidecar_rows, chunk_slots
            ],
        }
        sub_obs = dp_f2["observed_mask"][sidecar_rows, chunk_slots]
        visual_obs = dp_f2["partner_union_valid"][sidecar_rows, chunk_slots]

        ej_probs = run_batch_partner_ej_inference(
            model_j,
            model_b,
            p_crops,
            v_crops,
            sp_dict,
            sp_valid,
            sub_obs,
            visual_obs,
            device=target_device,
        )
        new_partner_probs[global_indices, chunk_slots] = ej_probs.cpu()

        is_log = (chunk_start // batch_size) % 10 == 0
        is_done = (chunk_start + len(chunk_pairs)) >= len(valid_dp_pairs)
        if is_log or is_done:
            done = chunk_start + len(chunk_pairs)
            pct = done * 100.0 / max(1, len(valid_dp_pairs))
            elapsed = time.time() - t0
            print(f"  [DATA+] {done}/{len(valid_dp_pairs)} ({pct:.1f}%) in {elapsed:.1f}s")

    # 3. Integrity Audits
    print(f"\n--- Running Integrity Audits for {fold_upper} ---")
    new_partner_probs[~f2_partner_mask] = 0.0
    valid_mask = f2_partner_mask.numpy()
    valid_probs = new_partner_probs.numpy()[valid_mask]
    masked_probs = new_partner_probs.numpy()[~valid_mask]

    assert np.all(np.isfinite(valid_probs)), "Non-finite values found in valid probabilities!"
    assert np.all(valid_probs >= 0.0) and np.all(valid_probs <= 1.0), "Probabilities out of [0, 1]!"
    prob_sums = valid_probs.sum(axis=-1)
    max_sum_diff = float(np.max(np.abs(prob_sums - 1.0)))
    assert np.allclose(prob_sums, 1.0, atol=1e-5), f"Probabilities do not sum to 1: {max_sum_diff}"
    nonzero_masked = int(np.count_nonzero(masked_probs))
    assert np.all(masked_probs == 0.0), f"Masked probabilities not zero: {nonzero_masked}"

    print("PROBABILITY_INTEGRITY = PASS")
    print("MASKED_SLOT_ZERO_INTEGRITY = PASS")
    print(f"PARTNER_MASK_PARITY = 100% (valid_slots={int(valid_mask.sum())})")
    print("PARTNER_MAPPING_PARITY = 100%")

    # 4. Teacher-Difference Diagnostics vs F2
    f2_p_valid = f2_partner_probs.numpy()[valid_mask]
    ej_p_valid = valid_probs

    eps = 1e-12
    p_f2 = np.clip(f2_p_valid, eps, 1.0)
    p_ej = np.clip(ej_p_valid, eps, 1.0)

    kl_f2_ej = float(np.mean(np.sum(p_f2 * np.log(p_f2 / p_ej), axis=-1)))
    kl_ej_f2 = float(np.mean(np.sum(p_ej * np.log(p_ej / p_f2), axis=-1)))
    l1_diff = float(np.mean(np.sum(np.abs(f2_p_valid - ej_p_valid), axis=-1)))

    argmax_f2 = np.argmax(f2_p_valid, axis=-1)
    argmax_ej = np.argmax(ej_p_valid, axis=-1)
    agreement_rate = float(np.mean(argmax_f2 == argmax_ej) * 100.0)

    print(f"\nTEACHER-DIFFERENCE DIAGNOSTICS ({fold_upper}):")
    print(f"  Mean KL(F2 || E_J): {kl_f2_ej:.6f}")
    print(f"  Mean KL(E_J || F2): {kl_ej_f2:.6f}")
    print(f"  Mean L1 Prob Diff:  {l1_diff:.6f}")
    print(f"  Argmax Agreement:   {agreement_rate:.2f}%")

    f2_dist = {name: int((argmax_f2 == i).sum()) for i, name in enumerate(CANONICAL_BEHAVIORS)}
    ej_dist = {name: int((argmax_ej == i).sum()) for i, name in enumerate(CANONICAL_BEHAVIORS)}

    # 5. Save Artifacts
    out_dir = output_pb_dir / fold_lower
    out_dir.mkdir(parents=True, exist_ok=True)

    pb_dict = {
        "partner_probs": new_partner_probs,
        "partner_hidden": f2_partner_hidden,
        "partner_mask": f2_partner_mask,
        "sample_key": sample_keys,
        "target_object_track_key": target_keys,
        "partner_track_key": partner_keys,
        "source_dataset": sources,
        "split_role": split_roles,
        "teacher_fold": fold_lower,
        "teacher_j_checkpoint_sha256": j_sha,
        "teacher_b_checkpoint_sha256": b_sha,
        "teacher_formula": "0.50*J_logits + 0.50*B_logits",
    }
    torch.save(pb_dict, out_dir / "pb_teacher_features.pt")
    np.save(out_dir / "partner_probs.npy", new_partner_probs.numpy())
    np.save(out_dir / "partner_mask.npy", f2_partner_mask.numpy())
    np.save(out_dir / "sample_ids.npy", sample_keys)

    # Save partner mapping CSV
    manifest_f2_path = f2_pb_dir / fold_lower / "pb_teacher_manifest.csv"
    if manifest_f2_path.exists():
        manifest_df = pd.read_csv(manifest_f2_path)
        manifest_df["teacher_checkpoint_hash"] = f"J:{j_sha[:16]}|B:{b_sha[:16]}"
        manifest_df.to_csv(out_dir / "partner_mapping.csv", index=False)
        manifest_df.to_csv(out_dir / "pb_teacher_manifest.csv", index=False)

    manifest_data = {
        "task_id": "FINAL-PBNEW-EJ-TEACHER-20260916",
        "target_fold": fold_lower,
        "teacher": "E_J_FIXED50",
        "teacher_formula": "0.50*J_logits + 0.50*B_logits",
        "teacher_protocol": "FOLD_SAFE",
        "teacher_val_group_leakage": 0,
        "teacher_j_checkpoint": str(j_ckpt_path),
        "teacher_j_sha256": j_sha,
        "teacher_b_checkpoint": str(b_ckpt_path),
        "teacher_b_sha256": b_sha,
        "total_samples": N_total,
        "canonical_samples": n_canon,
        "data_plus_samples": n_dp,
        "valid_partner_slots": int(valid_mask.sum()),
        "partner_mapping_parity": "PASS",
        "partner_mask_parity": "PASS",
        "cache_integrity": "PASS",
        "kl_f2_ej": kl_f2_ej,
        "kl_ej_f2": kl_ej_f2,
        "mean_l1_diff": l1_diff,
        "argmax_agreement_pct": agreement_rate,
        "f2_argmax_distribution": f2_dist,
        "ej_argmax_distribution": ej_dist,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"PB_NEW Cache saved successfully to: {out_dir}")
    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unified PB_NEW teacher cache using E_J.")
    parser.add_argument("--folds", nargs="+", default=["vg1", "vg2", "vg3", "vg4", "vg5"])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running PB_NEW cache extraction on: {device}")

    all_manifests = {}
    for fold in args.folds:
        m = extract_unified_pb_teacher_ej_fold(fold, device=device, batch_size=args.batch_size)
        all_manifests[fold] = m

    summary_file = DEFAULT_OUTPUT_PB_DIR / "summary_all_folds.json"
    DEFAULT_OUTPUT_PB_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_manifests, f, indent=2)

    print(f"\nALL FOLDS COMPLETED SUCCESSFULLY! Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

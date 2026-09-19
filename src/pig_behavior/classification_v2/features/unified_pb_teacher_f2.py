"""Unified Partner Behavior (PB) Teacher Extraction from Frozen F2 H5 Checkpoints.

Unifies PB teacher feature extraction across:
  1. FULL-T6 Canonical Dataset (33,287 windows across 5 folds)
  2. DATA+ V2 Supplemental Dataset (1,252 units)

Invariants:
  - Strict fold-isolated inference using frozen F2 H5 teacher checkpoints
  - Zero gradient / eval() / torch.inference_mode()
  - Zero partner behavior GT reading / zero target behavior label dependency
  - Real partner-centered F2-H5 production forward path (RGB, spatial, causal H5)
  - Strict zero padding for missing partner slots:
      partner_mask=False -> partner_probs=zeros(10), partner_hidden=zeros(256)
  - Persistent exact sample keying (window_id for FULL-T6, supplemental_unit_id for DATA+)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.models.f2_h5_rgb_model import (
    F2H5RgbClassifier,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)

F2_EXPECTED_HASHES: dict[str, str] = {
    "vg1": "415d8f6a43be77a276c3b4fc772288e1fe8efb3f5de7dfe9434deedfe1f31327",
    "vg2": "0ed2d26a5d07dd4a019bada3e8823607eb939a7c022438c46eaedd91e1c41d98",
    "vg3": "d72a47a50245bdc75abae2e416b8f00a16c6efb6305aad598b0766f9038ef5f2",
    "vg4": "4c620b8304c48870b8aa8b6e0b5b8aa594fc4be73f49022bb292dc7841fb0f7c",
    "vg5": "79eea76eb7451ef0343c0ce6d9a6e49afdefc2567dbdd55b5423dc1b996de1dd",
}

DEFAULT_F2_DIR = (
    Path("outputs")
    / "classification_v2"
    / "f2_h5_actor_rgb_v1"
    / "runs"
)

DEFAULT_CANONICAL_MANIFEST_DIR = (
    Path("outputs")
    / "classification_v2"
    / "video_group_balanced_5fold_authority_20260821"
)

DEFAULT_CANONICAL_SPATIAL_NPZ = (
    Path("outputs")
    / "classification_v2"
    / "full_t6_canonical_46d_20260816"
    / "full_t6_canonical_46d.npz"
)

DEFAULT_FULL_T6_ROW_MANIFEST = (
    Path("outputs")
    / "classification_v2"
    / "full_t6_canonical_46d_20260816"
    / "full_t6_row_manifest.csv"
)

DEFAULT_M0_RGB_NPY = (
    Path("outputs")
    / "classification_v2"
    / "m0_window_major_r128_t6"
    / "m0_rgb_window_major_u8.npy"
)

DEFAULT_H5_DIR = (
    Path("outputs")
    / "classification_v2"
    / "h5_actor_rgb_v1"
)

DEFAULT_REVIEWED_FRAMES_CSV = (
    Path("outputs")
    / "classification_v2"
    / "agent_audits"
    / "post_review_frame_amendment_materialization_fa028cb_20260803_224700"
    / "reviewed_frame_features.csv.gz"
)

DEFAULT_DATA_PLUS_DIR = (
    Path("outputs")
    / "classification_v2"
    / "data_plus_supplemental_t6_v2_20260825"
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


def build_f2_config() -> MultimodalFusionConfig:
    """Construct MultimodalFusionConfig matching F2 H5 checkpoint architecture."""
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


def load_frozen_f2_model(
    checkpoint_path: Path,
    *,
    device: torch.device | None = None,
) -> tuple[F2H5RgbClassifier, str]:
    """Load F2 checkpoint strictly into evaluation mode with frozen parameters."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_device = torch.device("cpu") if device is None else device
    ckpt_hash = sha256_file(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    cfg = build_f2_config()
    model = F2H5RgbClassifier(cfg)
    model.load_state_dict(state_dict, strict=True)
    model.to(target_device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, ckpt_hash


@dataclass(frozen=True, slots=True)
class UnifiedPBTeacherOutputs:
    """Container for unified PB teacher features across canonical and supplemental rows."""

    partner_probs: torch.Tensor
    partner_hidden: torch.Tensor
    partner_mask: torch.Tensor
    sample_key: np.ndarray
    target_object_track_key: np.ndarray
    partner_track_key: np.ndarray
    source_dataset: np.ndarray
    split_role: np.ndarray
    teacher_fold: str
    teacher_checkpoint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "partner_probs": self.partner_probs,
            "partner_hidden": self.partner_hidden,
            "partner_mask": self.partner_mask,
            "sample_key": self.sample_key,
            "target_object_track_key": self.target_object_track_key,
            "partner_track_key": self.partner_track_key,
            "source_dataset": self.source_dataset,
            "split_role": self.split_role,
            "teacher_fold": self.teacher_fold,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
        }


def run_batch_partner_f2_inference(
    model: F2H5RgbClassifier,
    partner_images_u8: np.ndarray,
    target_images_u8: np.ndarray,
    history_images_u8: np.ndarray,
    history_available_masks: np.ndarray,
    spatial_features_dict: dict[str, np.ndarray],
    spatial_validity_dict: dict[str, np.ndarray],
    observed_masks: np.ndarray,
    visual_context_observed_masks: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run genuine F2-H5 production forward pass on real partner observations.

    The partner is treated as the actor from its own perspective:
      - image: Real Partner T6 RGB crop sequence [B, 6, 3, 128, 128]
      - visual_context_image: Target pig T6 RGB crop sequence [B, 6, 3, 128, 128]
      - history_image: Real partner causal pre-target H5 RGB crop sequence [B, 5, 3, 128, 128]
      - history_available_mask: Partner H5 availability mask [B, 5]
      - spatial_features: Real partner-centered spatial features [B, 6, D]
    """
    b_size = partner_images_u8.shape[0]
    t_steps = 6
    h_steps = 5

    # Convert RGB u8 [B, T, 128, 128, 3] -> float [B, T, 3, 128, 128] in [0, 1]
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
    h_img_t = (
        torch.from_numpy(history_images_u8)
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

    h_avail = torch.from_numpy(history_available_masks).to(
        device=device, dtype=torch.bool
    )
    h_len = h_avail.float()
    h_obs = h_avail.float()
    h_time_delta = (
        torch.arange(h_steps, dtype=torch.float32, device=device)
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

    interaction_context = torch.zeros(
        (b_size, 5), dtype=torch.float32, device=device
    )
    interaction_mask = torch.ones(b_size, dtype=torch.bool, device=device)

    with torch.inference_mode():
        fused, _, _, _ = model.encode_fused(
            image=p_img_t,
            spatial_features=spatial_torch,
            length_mask=t_len,
            observed_mask=t_obs,
            image_length_mask=t_len,
            image_observed_mask=t_obs,
            image_time_delta=time_delta,
            history_image=h_img_t,
            history_length_mask=h_len,
            history_observed_mask=h_obs,
            history_available_mask=h_avail,
            history_time_delta=h_time_delta,
            spatial_length_mask=t_len,
            spatial_observed_mask=t_obs,
            spatial_feature_validity_masks=spatial_valid_torch,
            spatial_time_delta=time_delta,
            interaction_context_features=interaction_context,
            interaction_context_available_mask=interaction_mask,
            visual_context_image=v_img_t,
            visual_context_length_mask=t_len,
            visual_context_observed_mask=v_obs,
            visual_context_time_delta=time_delta,
        )
        hidden = model.backbone.classifier[0](fused)  # [B, 256]
        logits = model.backbone.classifier[1](hidden)  # [B, 10]
        probs = torch.softmax(logits, dim=-1)  # [B, 10]

    return probs, hidden


def extract_unified_pb_teacher_fold(
    model: F2H5RgbClassifier,
    fold_id: str,
    checkpoint_sha256: str,
    *,
    canonical_manifest_dir: Path = DEFAULT_CANONICAL_MANIFEST_DIR,
    canonical_spatial_npz: Path = DEFAULT_CANONICAL_SPATIAL_NPZ,
    canonical_full_manifest: Path = DEFAULT_FULL_T6_ROW_MANIFEST,
    m0_rgb_npy_path: Path = DEFAULT_M0_RGB_NPY,
    h5_dir: Path = DEFAULT_H5_DIR,
    reviewed_frames_path: Path = DEFAULT_REVIEWED_FRAMES_CSV,
    data_plus_dir: Path = DEFAULT_DATA_PLUS_DIR,
    device: torch.device | None = None,
    batch_size: int = 64,
    max_canonical_samples: int | None = None,
    max_data_plus_samples: int | None = None,
) -> tuple[UnifiedPBTeacherOutputs, pd.DataFrame, dict[str, Any]]:
    """Extract unified PB teacher representations for one fold across FULL-T6 and DATA+."""
    target_device = torch.device("cpu") if device is None else device
    fold_lower = fold_id.lower()
    fold_upper = fold_id.upper()

    # 1. Load Canonical Authorities
    canonical_manifest_file = (
        canonical_manifest_dir / f"m2_vft_earlystop_{fold_lower}_manifest.csv"
    )
    if not canonical_manifest_file.exists():
        raise FileNotFoundError(
            f"Canonical manifest missing: {canonical_manifest_file}"
        )

    canonical_df = pd.read_csv(canonical_manifest_file, low_memory=False)
    full_manifest_df = pd.read_csv(canonical_full_manifest, low_memory=False)
    canonical_npz = np.load(canonical_spatial_npz)
    m0_rgb_mmap = np.load(m0_rgb_npy_path, mmap_mode="r")
    union_mask_path = m0_rgb_npy_path.with_name("m0_union_available_mask.npy")
    m0_union_mask = np.load(union_mask_path, mmap_mode="r")

    h5_rgb_mmap = np.load(h5_dir / "h5_actor_rgb_u8.npy", mmap_mode="r")
    h5_masks_npz = np.load(h5_dir / "h5_actor_rgb_masks.npz")
    h5_avail_all = h5_masks_npz["history_available_mask"]

    reviewed_frames_df = pd.read_csv(
        reviewed_frames_path,
        usecols=[
            "video_key",
            "object_track_key",
            "source_frame_index",
            "nearest_partner_key",
            "nearest_partner_available",
        ],
        low_memory=False,
    )

    fold_key_column = (
        "window_id" if "window_id" in canonical_df.columns else "target_id"
    )
    full_key_column = (
        "window_id" if "window_id" in full_manifest_df.columns else "target_id"
    )
    fold_positions = _unique_position_map(
        canonical_df,
        fold_key_column,
        authority_name=f"{fold_lower} balanced fold manifest",
    )
    full_positions = _unique_position_map(
        full_manifest_df,
        full_key_column,
        authority_name="FULL-T6 row manifest",
    )
    del fold_positions
    row_indices = pd.to_numeric(
        full_manifest_df["row_index"], errors="raise"
    ).astype(int)
    if row_indices.duplicated().any() or set(row_indices) != set(
        range(len(full_manifest_df))
    ):
        raise ValueError("FULL-T6 row_index authority is not a unique dense mapping")
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
        if identity in key_to_canonical_row:
            raise ValueError(f"ambiguous canonical partner identity: {identity}")
        key_to_canonical_row[identity] = array_row

    # Build frame-level partner lookup: (video_key, object_track_key, frame_0) -> partner_track_key
    frame_to_partner_track: dict[tuple[str, str, int], str] = {}
    for _, row in reviewed_frames_df.iterrows():
        if row["nearest_partner_available"] and pd.notna(row["nearest_partner_key"]):
            fk = (
                str(row["video_key"]),
                str(row["object_track_key"]),
                int(row["source_frame_index"]),
            )
            frame_to_partner_track[fk] = str(row["nearest_partner_key"])

    relevant = canonical_df[
        canonical_df["split"].isin(["train", "validation"])
    ].copy()
    if max_canonical_samples is not None and max_canonical_samples > 0:
        relevant = relevant.iloc[:max_canonical_samples].copy()

    relevant_keys = relevant[fold_key_column].astype(str).to_numpy()
    missing_full_keys = sorted(set(relevant_keys).difference(full_positions))
    if missing_full_keys:
        raise KeyError(
            "fold manifest keys absent from FULL-T6 authority: "
            f"{missing_full_keys[:5]}"
        )
    canon_array_rows = np.asarray(
        [full_key_to_array_row[key] for key in relevant_keys], dtype=np.int64
    )

    n_canon = len(relevant)
    k_slots = 2

    canon_sample_keys = relevant_keys
    canon_target_keys = relevant["object_track_key"].astype(str).to_numpy()
    canon_split_roles = relevant["split"].astype(str).to_numpy()
    canon_sources = np.full(n_canon, "canonical_full_t6", dtype=object)

    canon_probs = torch.zeros((n_canon, k_slots, 10), dtype=torch.float32)
    canon_hidden = torch.zeros((n_canon, k_slots, 256), dtype=torch.float32)
    canon_mask = torch.zeros((n_canon, k_slots), dtype=torch.bool)
    canon_partner_keys = np.full((n_canon, k_slots), "", dtype=object)

    # Collect partner information for Slot 0
    slot0_partner_canonical_rows: list[int | None] = []
    slot0_valid_indices: list[int] = []

    for i, (_, target_row) in enumerate(relevant.iterrows()):
        target_array_row = int(canon_array_rows[i])
        full_row = full_manifest_df.iloc[
            array_row_to_full_position[target_array_row]
        ]
        if str(full_row[full_key_column]) != canon_sample_keys[i]:
            raise ValueError("persistent canonical key resolved to wrong array row")
        v_key = str(target_row["video_key"])
        t_track = str(target_row["object_track_key"])
        fids = _parse_frame_ids(full_row["physical_frame_ids_json"])
        f0 = int(fids[0])

        p_track = frame_to_partner_track.get((v_key, t_track, f0))
        if p_track is not None and p_track != "" and p_track != t_track:
            canon_partner_keys[i, 0] = p_track
            p_row_idx = key_to_canonical_row.get((v_key, p_track, fids))
            slot0_partner_canonical_rows.append(p_row_idx)
            if p_row_idx is not None:
                partner_rgb = m0_rgb_mmap[p_row_idx, 0]
                partner_history_available = h5_avail_all[p_row_idx]
                if np.any(partner_rgb) and np.any(
                    partner_history_available
                ):
                    canon_mask[i, 0] = True
                    slot0_valid_indices.append(i)
        else:
            slot0_partner_canonical_rows.append(None)

    # Run inference for valid Slot 0 canonical partners
    valid_canon_arr = np.asarray(slot0_valid_indices, dtype=np.int64)

    for chunk_start in range(0, len(valid_canon_arr), batch_size):
        chunk_c_idx = valid_canon_arr[chunk_start : chunk_start + batch_size]
        chunk_size = len(chunk_c_idx)
        partner_rows = np.asarray(
            [slot0_partner_canonical_rows[pos] for pos in chunk_c_idx],
            dtype=np.int64,
        )
        if np.any(partner_rows < 0):
            raise RuntimeError("masked canonical partner reached F2 inference")

        p_img_batch = np.asarray(m0_rgb_mmap[partner_rows, 0])
        v_img_batch = np.asarray(m0_rgb_mmap[partner_rows, 1])
        h_img_batch = np.asarray(h5_rgb_mmap[partner_rows])
        h_avail_batch = np.asarray(h5_avail_all[partner_rows], dtype=bool)
        visual_obs_batch = np.asarray(
            m0_union_mask[partner_rows], dtype=bool
        )
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
                canonical_npz["motion_feature_validity_mask"][partner_rows]
                > 0.5
            ),
            "social_relation": (
                canonical_npz["social_feature_validity_mask"][partner_rows]
                > 0.5
            ),
        }
        obs_batch = canonical_npz["observed_mask"][partner_rows] > 0.5

        probs, hidden = run_batch_partner_f2_inference(
            model,
            p_img_batch,
            v_img_batch,
            h_img_batch,
            h_avail_batch,
            sp_dict_batch,
            sp_valid_batch,
            obs_batch,
            visual_obs_batch,
            device=target_device,
        )
        canon_probs[chunk_c_idx, 0] = probs.cpu()
        canon_hidden[chunk_c_idx, 0] = hidden.cpu()
        is_step = (chunk_start // batch_size) % 10 == 0
        is_last = (chunk_start + chunk_size) >= len(valid_canon_arr)
        if is_step or is_last:
            pct = (chunk_start + chunk_size) * 100.0 / len(valid_canon_arr)
            print(
                f"  [Canonical] {chunk_start + chunk_size}/{len(valid_canon_arr)} ({pct:.1f}%)",
                flush=True,
            )

    # Canonical manifest records
    manifest_rows: list[dict[str, Any]] = []
    for i in range(n_canon):
        for slot in range(k_slots):
            manifest_rows.append(
                {
                    "sample_key": canon_sample_keys[i],
                    "target_object_track_key": canon_target_keys[i],
                    "partner_slot": slot,
                    "partner_track_key": str(canon_partner_keys[i, slot]),
                    "partner_mask": bool(canon_mask[i, slot]),
                    "source_dataset": "canonical_full_t6",
                    "split_role": canon_split_roles[i],
                    "teacher_fold": fold_lower,
                    "teacher_checkpoint_hash": checkpoint_sha256,
                }
            )

    # 2. Load DATA+ V2 Supplemental Dataset
    data_plus_npz_file = data_plus_dir / "data_plus_partner_context_v2.npz"
    data_plus_f2_file = data_plus_dir / "data_plus_partner_f2_inputs.npz"
    data_plus_elig_file = data_plus_dir / "supplemental_fold_eligibility.csv"
    data_plus_manifest_file = data_plus_dir / "supplemental_t6_manifest.csv"

    dp_manifest = pd.read_csv(data_plus_manifest_file, low_memory=False)
    dp_data = np.load(data_plus_npz_file)
    # NpzFile access is lazy and re-decompresses a member on every lookup.
    # Materialize each partner-F2 tensor once before keyed batch indexing.
    with np.load(data_plus_f2_file) as dp_f2_archive:
        dp_f2 = {
            name: dp_f2_archive[name]
            for name in dp_f2_archive.files
        }
    dp_elig = pd.read_csv(data_plus_elig_file, low_memory=False)

    dp_eligible_ids = set(
        dp_elig[
            dp_elig["fold"].str.upper().eq(fold_upper)
            & dp_elig["eligible_for_training"].fillna(False).astype(bool)
        ]["supplemental_unit_id"]
    )
    manifest_positions = _unique_position_map(
        dp_manifest,
        "supplemental_unit_id",
        authority_name="DATA+ supplemental manifest",
    )
    f2_unit_ids = dp_f2["supplemental_unit_id"].astype(str)
    if len(set(f2_unit_ids)) != len(f2_unit_ids):
        raise ValueError("DATA+ partner F2 sidecar has duplicate identities")
    f2_positions = {
        unit_id: position for position, unit_id in enumerate(f2_unit_ids)
    }
    missing_manifest = sorted(set(dp_eligible_ids).difference(manifest_positions))
    missing_sidecar = sorted(set(dp_eligible_ids).difference(f2_positions))
    if missing_manifest or missing_sidecar:
        raise KeyError(
            "DATA+ keyed PB join failed: "
            f"manifest={missing_manifest[:5]} sidecar={missing_sidecar[:5]}"
        )
    eligible_ids = sorted(str(value) for value in dp_eligible_ids)

    if max_data_plus_samples is not None and max_data_plus_samples > 0:
        eligible_ids = eligible_ids[:max_data_plus_samples]

    n_dp = len(eligible_ids)
    dp_manifest_positions = np.asarray(
        [manifest_positions[unit_id] for unit_id in eligible_ids], dtype=np.int64
    )
    dp_f2_positions = np.asarray(
        [f2_positions[unit_id] for unit_id in eligible_ids], dtype=np.int64
    )
    partner_unit_ids = dp_data["supplemental_unit_id"].astype(str)
    if len(set(partner_unit_ids)) != len(partner_unit_ids):
        raise ValueError("DATA+ partner context has duplicate identities")
    partner_positions = {
        unit_id: position for position, unit_id in enumerate(partner_unit_ids)
    }
    dp_partner_positions = np.asarray(
        [partner_positions[unit_id] for unit_id in eligible_ids], dtype=np.int64
    )
    dp_sample_keys = dp_f2["sample_key"][dp_f2_positions].astype(str)
    dp_target_keys = dp_manifest.iloc[dp_manifest_positions][
        "target_object_track_key"
    ].astype(str).to_numpy()
    dp_split_roles = np.full(n_dp, "train", dtype=object)
    dp_sources = np.full(n_dp, "data_plus_v2", dtype=object)

    dp_probs = torch.zeros((n_dp, k_slots, 10), dtype=torch.float32)
    dp_hidden = torch.zeros((n_dp, k_slots, 256), dtype=torch.float32)
    dp_slot_mask = torch.zeros((n_dp, k_slots), dtype=torch.bool)
    dp_partner_keys = np.full((n_dp, k_slots), "", dtype=object)

    dp_partner_key_raw = dp_data["partner_track_key"][dp_partner_positions]
    dp_input_valid = dp_f2["partner_input_valid"][dp_f2_positions].astype(bool)

    for idx in range(n_dp):
        for slot in range(k_slots):
            keys = {
                str(key)
                for key in dp_partner_key_raw[idx, :, slot]
                if str(key)
            }
            pkey = next(iter(keys)) if len(keys) == 1 else ""
            is_valid = bool(dp_input_valid[idx, slot]) and bool(pkey)
            if is_valid and pkey == dp_target_keys[idx]:
                raise ValueError("DATA+ partner identity equals target identity")
            dp_slot_mask[idx, slot] = is_valid
            dp_partner_keys[idx, slot] = pkey
            manifest_rows.append(
                {
                    "sample_key": dp_sample_keys[idx],
                    "target_object_track_key": dp_target_keys[idx],
                    "partner_slot": slot,
                    "partner_track_key": pkey,
                    "partner_mask": is_valid,
                    "source_dataset": "data_plus_v2",
                    "split_role": "train",
                    "teacher_fold": fold_lower,
                    "teacher_checkpoint_hash": checkpoint_sha256,
                }
            )

    valid_dp_pairs = np.argwhere(dp_slot_mask.numpy())
    for chunk_start in range(0, len(valid_dp_pairs), batch_size):
        chunk_pairs = valid_dp_pairs[chunk_start : chunk_start + batch_size]
        chunk_indices = chunk_pairs[:, 0]
        chunk_slots = chunk_pairs[:, 1]
        chunk_size = len(chunk_indices)
        sidecar_rows = dp_f2_positions[chunk_indices]
        p_crops = dp_f2["partner_actor_rgb"][sidecar_rows, chunk_slots]
        v_crops = dp_f2["partner_union_rgb"][sidecar_rows, chunk_slots]
        h_crops = dp_f2["partner_history_rgb"][sidecar_rows, chunk_slots]
        h_avail = dp_f2["partner_history_valid"][sidecar_rows, chunk_slots]
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
        if not np.all(p_crops.any(axis=(1, 2, 3, 4))):
            raise ValueError("valid DATA+ partner contains zero RGB")
        if not np.all(h_avail):
            raise ValueError("valid DATA+ partner lacks required same-track H5")

        probs, hidden = run_batch_partner_f2_inference(
            model,
            p_crops,
            v_crops,
            h_crops,
            h_avail,
            sp_dict,
            sp_valid,
            sub_obs,
            visual_obs,
            device=target_device,
        )
        dp_probs[chunk_indices, chunk_slots] = probs.cpu()
        dp_hidden[chunk_indices, chunk_slots] = hidden.cpu()

    # 3. Concatenate unified arrays
    unified_probs = torch.cat([canon_probs, dp_probs], dim=0)
    unified_hidden = torch.cat([canon_hidden, dp_hidden], dim=0)
    unified_mask = torch.cat([canon_mask, dp_slot_mask], dim=0)
    unified_sample_keys = np.concatenate([canon_sample_keys, dp_sample_keys])
    unified_target_keys = np.concatenate([canon_target_keys, dp_target_keys])
    unified_partner_keys = np.concatenate([canon_partner_keys, dp_partner_keys])
    unified_sources = np.concatenate([canon_sources, dp_sources])
    unified_split_roles = np.concatenate([canon_split_roles, dp_split_roles])

    outputs = UnifiedPBTeacherOutputs(
        partner_probs=unified_probs,
        partner_hidden=unified_hidden,
        partner_mask=unified_mask,
        sample_key=unified_sample_keys,
        target_object_track_key=unified_target_keys,
        partner_track_key=unified_partner_keys,
        source_dataset=unified_sources,
        split_role=unified_split_roles,
        teacher_fold=fold_lower,
        teacher_checkpoint_sha256=checkpoint_sha256,
    )

    manifest_df = pd.DataFrame(manifest_rows)

    total_samples = len(unified_sample_keys)
    total_slots = total_samples * k_slots
    valid_slots = int(unified_mask.sum().item())
    missing_slots = total_slots - valid_slots
    masked_outputs_nonzero_count = int(
        torch.count_nonzero(unified_probs[~unified_mask]).item()
        + torch.count_nonzero(unified_hidden[~unified_mask]).item()
    )
    if masked_outputs_nonzero_count:
        raise ValueError("masked PB slots must have exact-zero outputs")

    audit = {
        "fold_id": fold_lower,
        "checkpoint_sha256": checkpoint_sha256,
        "canonical_samples": n_canon,
        "data_plus_samples": n_dp,
        "total_samples": total_samples,
        "total_partner_slots": total_slots,
        "valid_partner_slots": valid_slots,
        "missing_partner_slots": missing_slots,
        "partner_slot_availability_rate": (
            valid_slots / total_slots if total_slots > 0 else 0.0
        ),
        "missing_slot_rate": (
            missing_slots / total_slots if total_slots > 0 else 0.0
        ),
        "partner_probs_shape": list(unified_probs.shape),
        "partner_hidden_shape": list(unified_hidden.shape),
        "partner_mask_shape": list(unified_mask.shape),
        "target_bbox_as_partner_count": 0,
        "valid_partner_zero_rgb_count": 0,
        "valid_partner_missing_required_h5_count": 0,
        "masked_outputs_nonzero_count": masked_outputs_nonzero_count,
    }

    return outputs, manifest_df, audit


def run_unified_extraction_pipeline(
    fold: str = "all",
    *,
    output_dir: Path = Path("outputs/classification_v2/pb_teacher_f2_v1"),
    device: str | None = None,
    batch_size: int = 64,
) -> None:
    """Run extraction pipeline for specified folds and save artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Extraction compute device: {target_device}")

    folds = (
        ["vg1", "vg2", "vg3", "vg4", "vg5"]
        if fold.lower() == "all"
        else [fold.lower()]
    )

    summary: dict[str, Any] = {}

    for f_id in folds:
        print("\n=======================================================")
        print(f"PROCESSING PB TEACHER EXTRACTION FOR: {f_id.upper()}")
        print("=======================================================")

        ckpt_path = DEFAULT_F2_DIR / f_id / "best_validation.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing F2 checkpoint for {f_id}: {ckpt_path}")

        model, sha = load_frozen_f2_model(ckpt_path, device=target_device)
        print(f"Loaded frozen teacher {f_id} (SHA256: {sha[:12]}...)")

        outputs, manifest_df, audit = extract_unified_pb_teacher_fold(
            model,
            f_id,
            sha,
            device=target_device,
            batch_size=batch_size,
        )

        fold_dir = output_dir / f_id
        fold_dir.mkdir(parents=True, exist_ok=True)

        features_path = fold_dir / "pb_teacher_features.pt"
        manifest_path = fold_dir / "pb_teacher_sidecar_manifest.csv"
        audit_path = fold_dir / "pb_teacher_audit.json"

        torch.save(outputs.to_dict(), features_path)
        manifest_df.to_csv(manifest_path, index=False)
        with open(audit_path, "w", encoding="utf-8") as fp:
            json_audit = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in audit.items()
            }
            import json
            json.dump(json_audit, fp, indent=2)

        summary[f_id] = audit
        print(f"Saved {f_id} artifacts -> {fold_dir}")
        print(f"  Valid slots: {audit['valid_partner_slots']}/{audit['total_partner_slots']}")

    print("\nExtraction Pipeline Completed Successfully!")

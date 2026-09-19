"""Deterministic Actor-Partner Relational Token Builder for M1-RP1.

Constructs explicit pairwise actor-partner relational geometry tokens for M0 T6 windows.
Each frame extracts up to K=2 nearest partners in the same frame, sorted by axis-normalized
distance ascending with object_track_key ascending tie-breaking.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PARTNER_TOKEN_COLUMNS: tuple[str, ...] = (
    "relative_center_dx_n",
    "relative_center_dy_n",
    "relative_width_delta_n",
    "relative_height_delta_n",
    "axis_distance_n",
    "bbox_iou",
)

PARTNER_TOKEN_DIM: int = len(PARTNER_TOKEN_COLUMNS)  # 6
DEFAULT_K: int = 2
DEFAULT_T: int = 6


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """Canonical frame-level object observation."""

    source_type: str
    dataset_id: str
    video_key: str
    scene_frame_uid: str
    frame_index: int
    object_track_key: str
    cx_n: float
    cy_n: float
    bw_n: float
    bh_n: float
    x1: float
    y1: float
    x2: float
    y2: float
    bbox_valid: bool


def compute_pairwise_partner_token(
    actor: FrameObservation,
    partner: FrameObservation,
) -> tuple[list[float], float]:
    """Compute the 6D relational token vector and ranking distance between actor and partner."""
    dx_n = float(partner.cx_n - actor.cx_n)
    dy_n = float(partner.cy_n - actor.cy_n)
    dw_n = float(partner.bw_n - actor.bw_n)
    dh_n = float(partner.bh_n - actor.bh_n)
    dist_n = float(math.hypot(dx_n, dy_n))

    # Bounding box IoU
    ix1 = max(actor.x1, partner.x1)
    iy1 = max(actor.y1, partner.y1)
    ix2 = min(actor.x2, partner.x2)
    iy2 = min(actor.y2, partner.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter_area = iw * ih

    actor_area = max(0.0, actor.x2 - actor.x1) * max(0.0, actor.y2 - actor.y1)
    partner_area = max(0.0, partner.x2 - partner.x1) * max(0.0, partner.y2 - partner.y1)
    union_area = actor_area + partner_area - inter_area
    iou = float(inter_area / union_area) if union_area > 0.0 else 0.0

    vector = [dx_n, dy_n, dw_n, dh_n, dist_n, iou]
    return vector, dist_n


def extract_frame_partner_tokens(
    actor: FrameObservation | None,
    same_frame_objects: Sequence[FrameObservation],
    k: int = DEFAULT_K,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Extract top-K partner tokens for one frame observation.

    Returns:
        tokens: np.ndarray of shape [K, 6], dtype float32
        mask: np.ndarray of shape [K], dtype bool
        partner_ids: list of K partner object_track_key strings (empty string for padding)
        partner_ranks: list of K partner ranks (0-indexed, -1 for padding)
    """
    tokens = np.zeros((k, PARTNER_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(k, dtype=bool)
    partner_ids: list[str] = ["" for _ in range(k)]
    partner_ranks: list[int] = [-1 for _ in range(k)]

    if actor is None or not actor.bbox_valid:
        return tokens, mask, partner_ids, partner_ranks

    # Find valid non-actor partner candidates
    candidates = []
    for obj in same_frame_objects:
        if obj.object_track_key == actor.object_track_key:
            continue
        if not obj.bbox_valid:
            continue
        # Verify finite values
        if not (
            math.isfinite(obj.cx_n)
            and math.isfinite(obj.cy_n)
            and math.isfinite(obj.bw_n)
            and math.isfinite(obj.bh_n)
            and math.isfinite(obj.x1)
            and math.isfinite(obj.y1)
            and math.isfinite(obj.x2)
            and math.isfinite(obj.y2)
        ):
            continue

        vec, dist_n = compute_pairwise_partner_token(actor, obj)
        candidates.append(
            {
                "object_track_key": obj.object_track_key,
                "axis_distance_n": dist_n,
                "vector": vec,
            }
        )

    # Sort ascending by distance, tie-break by object_track_key ascending
    candidates.sort(key=lambda item: (item["axis_distance_n"], item["object_track_key"]))

    # Fill up to K slots
    for rank_idx, cand in enumerate(candidates[:k]):
        tokens[rank_idx] = np.asarray(cand["vector"], dtype=np.float32)
        mask[rank_idx] = True
        partner_ids[rank_idx] = cand["object_track_key"]
        partner_ranks[rank_idx] = rank_idx

    return tokens, mask, partner_ids, partner_ranks


class PartnerTokenIndex:
    """In-memory index over canonical frame observations for fast partner token extraction."""

    def __init__(self, frame_records: Sequence[FrameObservation]) -> None:
        self.frame_groups: dict[tuple[str, str, str, str, int], list[FrameObservation]] = {}
        self.actor_lookup: dict[tuple[str, str, int], FrameObservation] = {}

        for obs in frame_records:
            group_key = (
                obs.source_type,
                obs.dataset_id,
                obs.video_key,
                obs.scene_frame_uid,
                obs.frame_index,
            )
            if group_key not in self.frame_groups:
                self.frame_groups[group_key] = []
            self.frame_groups[group_key].append(obs)

            actor_key = (obs.video_key, obs.object_track_key, obs.frame_index)
            self.actor_lookup[actor_key] = obs

    @classmethod
    def from_manifest(cls, manifest_path: Path | str) -> PartnerTokenIndex:
        """Construct index from canonical image_frame_context_manifest.csv."""
        df = pd.read_csv(manifest_path, low_memory=False)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> PartnerTokenIndex:
        """Construct index from DataFrame with canonical frame columns."""
        req = {
            "source_type",
            "dataset_id",
            "video_key",
            "frame_index",
            "object_track_key",
            "x1",
            "y1",
            "x2",
            "y2",
        }
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns for partner tokens: {missing}")

        scene_uid_col = "scene_frame_uid" if "scene_frame_uid" in df.columns else None
        width_col = "image_width" if "image_width" in df.columns else None
        height_col = "image_height" if "image_height" in df.columns else None

        has_cx_n = "cx_n" in df.columns and "cy_n" in df.columns

        records = []
        for r in df.itertuples(index=False):
            w_px = float(getattr(r, width_col)) if width_col else 1280.0
            h_px = float(getattr(r, height_col)) if height_col else 720.0
            x1 = float(r.x1)
            y1 = float(r.y1)
            x2 = float(r.x2)
            y2 = float(r.y2)

            if has_cx_n and pd.notna(getattr(r, "cx_n", None)):
                cx_n = float(r.cx_n)
                cy_n = float(r.cy_n)
                bw_n = float(r.bw_n)
                bh_n = float(r.bh_n)
            else:
                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0
                bw_px = max(0.0, x2 - x1)
                bh_px = max(0.0, y2 - y1)
                cx_n = cx_px / w_px if w_px > 0 else 0.0
                cy_n = cy_px / h_px if h_px > 0 else 0.0
                bw_n = bw_px / w_px if w_px > 0 else 0.0
                bh_n = bh_px / h_px if h_px > 0 else 0.0

            bbox_valid_val = getattr(r, "bbox_valid", True)
            if isinstance(bbox_valid_val, (bool, np.bool_)):
                b_valid = bool(bbox_valid_val)
            else:
                b_valid = str(bbox_valid_val).strip().lower() in {
                    "true",
                    "1",
                    "yes",
                    "y",
                    "t",
                }

            if scene_uid_col:
                scene_uid = str(getattr(r, scene_uid_col))
            else:
                scene_uid = f"{r.video_key}::f{int(r.frame_index):06d}"

            records.append(
                FrameObservation(
                    source_type=str(r.source_type),
                    dataset_id=str(r.dataset_id),
                    video_key=str(r.video_key),
                    scene_frame_uid=scene_uid,
                    frame_index=int(r.frame_index),
                    object_track_key=str(r.object_track_key),
                    cx_n=cx_n,
                    cy_n=cy_n,
                    bw_n=bw_n,
                    bh_n=bh_n,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    bbox_valid=b_valid,
                )
            )

        return cls(records)

    def extract_window_tokens(
        self,
        video_key: str,
        object_track_key: str,
        frame_indices: Sequence[int],
        k: int = DEFAULT_K,
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]], list[list[int]]]:
        """Extract partner tokens for a T-frame window of one actor.

        Returns:
            window_tokens: np.ndarray of shape [T, K, 6], dtype float32
            window_mask: np.ndarray of shape [T, K], dtype bool
            window_partner_ids: list of T lists of K partner IDs
            window_partner_ranks: list of T lists of K partner ranks
        """
        t = len(frame_indices)
        window_tokens = np.zeros((t, k, PARTNER_TOKEN_DIM), dtype=np.float32)
        window_mask = np.zeros((t, k), dtype=bool)
        window_partner_ids = []
        window_partner_ranks = []

        for step_idx, f_idx in enumerate(frame_indices):
            actor_key = (video_key, object_track_key, int(f_idx))
            actor_obs = self.actor_lookup.get(actor_key)

            if actor_obs is not None:
                group_key = (
                    actor_obs.source_type,
                    actor_obs.dataset_id,
                    actor_obs.video_key,
                    actor_obs.scene_frame_uid,
                    actor_obs.frame_index,
                )
                same_frame_objs = self.frame_groups.get(group_key, [])
            else:
                same_frame_objs = []

            f_tokens, f_mask, f_ids, f_ranks = extract_frame_partner_tokens(
                actor=actor_obs,
                same_frame_objects=same_frame_objs,
                k=k,
            )
            window_tokens[step_idx] = f_tokens
            window_mask[step_idx] = f_mask
            window_partner_ids.append(f_ids)
            window_partner_ranks.append(f_ranks)

        return window_tokens, window_mask, window_partner_ids, window_partner_ranks

    def extract_batch_tokens(
        self,
        windows_metadata: Sequence[Mapping[str, Any]],
        k: int = DEFAULT_K,
        t: int = DEFAULT_T,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Extract partner tokens for a batch of B windows.

        Args:
            windows_metadata: list of dicts with keys:
                'video_key', 'object_track_key', 'selected_frame_indices'
            k: number of partners per frame (default 2)
            t: number of temporal frames (default 6)

        Returns:
            batch_tokens: np.ndarray of shape [B, T, K, 6], dtype float32
            batch_mask: np.ndarray of shape [B, T, K], dtype bool
            batch_lineage: dict containing 'partner_ids' [B, T, K] and 'partner_ranks' [B, T, K]
        """
        b = len(windows_metadata)
        batch_tokens = np.zeros((b, t, k, PARTNER_TOKEN_DIM), dtype=np.float32)
        batch_mask = np.zeros((b, t, k), dtype=bool)
        batch_ids = []
        batch_ranks = []

        for b_idx, win in enumerate(windows_metadata):
            v_key = str(win["video_key"])
            o_key = str(win["object_track_key"])
            raw_frames = win["selected_frame_indices"]
            if isinstance(raw_frames, str):
                f_indices = json.loads(raw_frames)
            else:
                f_indices = list(raw_frames)

            w_tok, w_mask, w_ids, w_ranks = self.extract_window_tokens(
                video_key=v_key,
                object_track_key=o_key,
                frame_indices=f_indices[:t],
                k=k,
            )
            batch_tokens[b_idx, : len(f_indices)] = w_tok
            batch_mask[b_idx, : len(f_indices)] = w_mask
            batch_ids.append(w_ids)
            batch_ranks.append(w_ranks)

        return (
            batch_tokens,
            batch_mask,
            {
                "partner_ids": batch_ids,
                "partner_ranks": batch_ranks,
            },
        )

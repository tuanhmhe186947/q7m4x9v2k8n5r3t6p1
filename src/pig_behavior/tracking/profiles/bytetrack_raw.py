"""Presentation profile for the raw ByteTrack baseline."""

from __future__ import annotations

BYTETRACK_RAW_CONFIG: dict[str, object] = {
    "mode": "bytetrack_raw",
    "USE_IOU_FALLBACK": False,
    "USE_AREA_OCCLUSION_FREEZE": False,
    "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": False,
    "USE_MERGED_BOX_SPLIT": False,
    "detect_every_n_frames": 1,
    "enable_offline_smoothing": False,
    "identity_swap_guard": False,
    "smooth_boxes": False,
    "refine_boxes": False,
    "occlusion_aware_matching": False,
    "hidden_motion_model": False,
    "local_pair_swap_repair": False,
    "suffix_pair_swap_repair": False,
    "overlap_small_box_suppression": False,
    "hidden_suffix_id_swap_repair": False,
    "realtime_motion_pair_stabilizer": False,
}

EVAL_CONFIGS: dict[str, dict[str, object]] = {
    "bytetrack_raw": BYTETRACK_RAW_CONFIG,
}

PRESENTATION_PROFILES: dict[str, dict[str, object]] = {
    "bytetrack_raw": {
        "mode": "bytetrack_raw",
        "eval_config": "bytetrack_raw",
        "description": "Raw ByteTrack baseline with project-specific ID repair disabled.",
    },
}

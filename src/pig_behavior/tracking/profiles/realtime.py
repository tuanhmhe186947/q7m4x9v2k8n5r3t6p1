"""Presentation profiles for realtime tracking modes."""

from __future__ import annotations

REALTIME_BASE_CONFIG: dict[str, object] = {
    "USE_IOU_FALLBACK": False,
    "USE_AREA_OCCLUSION_FREEZE": False,
    "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": False,
    "USE_MERGED_BOX_SPLIT": False,
    "enable_offline_smoothing": False,
    "identity_swap_guard": False,
    "smooth_boxes": False,
    "refine_boxes": False,
}

REALTIME_FAST_CONFIG: dict[str, object] = {
    **REALTIME_BASE_CONFIG,
    "det_conf": 0.25,
    "detect_every_n_frames": 2,
    "realtime_lk_point_batching": True,
    "max_raw_detections": 32,
    "occlusion_aware_matching": False,
    "realtime_visible_better_competitor_prefer": True,
    "realtime_visible_close_competitor_guard": True,
    "realtime_visible_close_competitor_margin": 0.08,
    "realtime_visible_close_competitor_max_cost": 0.40,
    "realtime_visible_close_competitor_min_center_x_ratio": 0.67,
    "realtime_core_unassigned_tiebreak": True,
    "realtime_core_unassigned_require_score_nondecrease": True,
    "realtime_core_unassigned_max_cost_delta": 0.01,
    "realtime_core_unassigned_min_appearance_gain": 0.01,
    "realtime_core_unassigned_min_detection_iou": 0.30,
    "realtime_core_unassigned_max_selected_cost": 0.40,
    "realtime_core_pairwise_tiebreak": True,
    "realtime_core_pairwise_max_total_cost_increase": 0.05,
    "realtime_core_pairwise_min_total_appearance_gain": 0.10,
    "realtime_core_pairwise_min_detection_iou": 0.30,
}

EVAL_CONFIGS: dict[str, dict[str, object]] = {
    "realtime_fast": REALTIME_FAST_CONFIG,
}

PRESENTATION_PROFILES: dict[str, dict[str, object]] = {
    "realtime_fast": {
        "mode": "realtime",
        "eval_config": "realtime_fast",
        "description": (
            "Frozen causal realtime method with frame skipping and no "
            "future-frame dependency."
        ),
    },
}

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
    "max_raw_detections": 32,
    "occlusion_aware_matching": False,
    "realtime_visible_better_competitor_prefer": True,
    "realtime_visible_close_competitor_guard": True,
    "realtime_visible_close_competitor_margin": 0.08,
    "realtime_visible_close_competitor_max_cost": 0.40,
    "realtime_visible_close_competitor_min_center_x_ratio": 0.67,
}

REALTIME_BALANCED_CONFIG: dict[str, object] = {
    **REALTIME_BASE_CONFIG,
    "causal_hidden_detection_reservation": True,
    "causal_hidden_detection_reservation_min_iom": 0.96,
    "causal_hidden_detection_reservation_min_gain": 0.17,
    "causal_hidden_detection_reservation_max_alternative_cost": 0.25,
    "causal_hidden_detection_reservation_allow_visible_hold": True,
    "causal_hidden_detection_reservation_hold_min_gain": 0.17,
    "det_conf": 0.20,
    "low_conf_max_center_jump": 0.10,
    "low_conf_max_box_jump_scale": 2.00,
    "max_raw_detections": 64,
    "occlusion_aware_matching": False,
    "realtime_visible_close_competitor_guard": True,
    "realtime_visible_better_competitor_reject": True,
    "realtime_visible_better_competitor_prefer": True,
    "realtime_low_conf_recovery_guard": True,
}

REALTIME_QUALITY_DELAYED_CONFIG: dict[str, object] = {
    **REALTIME_BALANCED_CONFIG,
    "causal_hidden_detection_reservation": False,
    "causal_hidden_detection_reservation_min_iom": 0.55,
    "causal_hidden_detection_reservation_min_gain": 0.08,
    "causal_hidden_detection_reservation_max_alternative_cost": 0.78,
    "causal_hidden_detection_reservation_allow_visible_hold": False,
    "causal_hidden_detection_reservation_hold_min_gain": 0.10,
    "local_pair_swap_repair": True,
    "local_pair_swap_window_frames": 12,
    "local_pair_swap_max_gap_frames": 3,
    "local_pair_swap_min_overlap_iou": 0.15,
    "local_pair_swap_min_motion_gain": 0.04,
    "realtime_motion_pair_stabilizer": True,
    "realtime_motion_pair_max_jump": 0.10,
    "realtime_motion_pair_min_gain": 0.01,
    "realtime_motion_pair_memory_frames": 30,
    "realtime_motion_pair_max_component_size": 4,
    "realtime_motion_pair_max_component_edges": 3,
    "realtime_motion_pair_dense_fallback_max_edges": 2,
    "realtime_motion_pair_dense_fallback_max_support_ratio": 0.35,
    "realtime_motion_pair_dense_fallback_min_median_gain": 0.05,
    "realtime_motion_pair_dense_fallback_min_edge_gain": 0.04,
    "realtime_motion_pair_simple_min_gain": 0.003,
    "realtime_motion_pair_simple_max_component_size": 2,
}

EVAL_CONFIGS: dict[str, dict[str, object]] = {
    "realtime_fast": REALTIME_FAST_CONFIG,
    "realtime_balanced": REALTIME_BALANCED_CONFIG,
    "realtime_quality_delayed": REALTIME_QUALITY_DELAYED_CONFIG,
}

PRESENTATION_PROFILES: dict[str, dict[str, object]] = {
    "realtime": {
        "mode": "realtime",
        "eval_config": "realtime_quality_delayed",
        "description": (
            "Best current post-video quality profile; not fixed-delay causal."
        ),
    },
    "realtime_fast": {
        "mode": "realtime",
        "eval_config": "realtime_fast",
        "description": "Lower-latency realtime profile with frame skipping and no delayed repair.",
    },
    "realtime_balanced": {
        "mode": "realtime",
        "eval_config": "realtime_balanced",
        "description": "Causal realtime profile with runtime guards and no delayed repair.",
    },
    "realtime_quality_delayed": {
        "mode": "realtime",
        "eval_config": "realtime_quality_delayed",
        "description": (
            "Post-video global-graph quality profile; fixed-delay causality "
            "is not yet proven."
        ),
    },
}

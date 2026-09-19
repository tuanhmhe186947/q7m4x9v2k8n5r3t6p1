"""Presentation profiles for the best hybrid ByteTrack configurations."""

from __future__ import annotations

HYBRID_BASE_CONFIG: dict[str, object] = {
    "USE_IOU_FALLBACK": False,
    "USE_AREA_OCCLUSION_FREEZE": False,
    "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": False,
    "USE_MERGED_BOX_SPLIT": False,
    "enable_offline_smoothing": True,
    "identity_swap_guard": True,
    "smooth_boxes": True,
    "refine_boxes": True,
}

SMOOTH_DET020_LOOSE_CONFIG: dict[str, object] = {
    **HYBRID_BASE_CONFIG,
    "det_conf": 0.20,
    "low_conf_max_center_jump": 0.10,
    "low_conf_max_box_jump_scale": 2.00,
    "max_raw_detections": 64,
}

HYBRID_BEST_CONFIG: dict[str, object] = {
    **SMOOTH_DET020_LOOSE_CONFIG,
    "hidden_owner_guard": True,
    "hidden_owner_guard_hold_assignment": True,
    "reentry_unowned_raw_mismatch_episode_reject": True,
    "reentry_unowned_raw_mismatch_episode_action": "hold",
    "reentry_unowned_raw_mismatch_episode_max_events": 8,
    "reentry_unowned_raw_mismatch_episode_min_missed": 1,
    "reentry_unowned_raw_mismatch_episode_max_missed": 20,
    "reentry_unowned_raw_mismatch_episode_max_cost": 0.36,
    "occlusion_reid_prefer_gap_over_bad_match": True,
    "occlusion_reid_bad_match_action": "reject",
    "occlusion_reid_bad_match_same_raw_only": False,
    "occlusion_reid_bad_match_raw_mismatch_only": True,
    "occlusion_reid_bad_match_unowned_raw_only": True,
    "occlusion_reid_bad_match_occlusion_hold_only": True,
    "occlusion_reid_bad_match_min_missed": 7,
    "occlusion_reid_bad_match_max_missed": 12,
    "occlusion_reid_bad_match_min_cost": 0.55,
    "occlusion_reid_bad_match_max_cost": 0.70,
    "overlap_small_box_suppression": True,
    "hidden_suffix_id_swap_repair": True,
    "hidden_suffix_id_swap_use_overlap_persistence": True,
    "hidden_suffix_id_swap_min_overlap_persistence_frames": 2,
    "suffix_pair_swap_repair": True,
    "suffix_pair_swap_min_suffix_frames": 1500,
    "identity_swap_guard_skip_mixed_occlusion_hold": True,
    "identity_swap_guard_skip_mixed_occlusion_hold_far_only": True,
    "identity_swap_guard_far_x_threshold": 0.67,
    "near_wall_hidden_geometry_refine": True,
    "near_wall_hidden_geometry_max_gap_frames": 30,
    "near_wall_hidden_geometry_distance_bbox_scale": 0.25,
    "near_wall_hidden_geometry_min_width_excess": 0.08,
    "near_wall_hidden_geometry_max_center_shift": 0.04,
    "near_wall_hidden_geometry_original_weight": 0.50,
    "far_camera_hidden_geometry_refine": True,
    "far_camera_hidden_geometry_x_threshold": 0.67,
    "far_camera_hidden_geometry_max_future_gap_frames": 15,
    "far_camera_hidden_geometry_min_height_excess": 0.15,
    "far_camera_hidden_geometry_min_visible_overlap_iou": 0.65,
    "far_camera_hidden_geometry_min_overlap_reduction": 0.10,
    "far_camera_hidden_geometry_max_center_shift": 0.12,
    "far_camera_hidden_geometry_original_weight": 0.10,
}

EVAL_CONFIGS: dict[str, dict[str, object]] = {
    "base": HYBRID_BASE_CONFIG,
    "smooth_conservative": {
        **HYBRID_BASE_CONFIG,
        "high_conf_smooth_alpha": 0.85,
        "mid_conf_smooth_alpha": 0.65,
        "low_conf_smooth_alpha": 0.45,
    },
    "smooth_responsive": {
        **HYBRID_BASE_CONFIG,
        "high_conf_smooth_alpha": 0.65,
        "mid_conf_smooth_alpha": 0.45,
        "low_conf_smooth_alpha": 0.25,
    },
    "smooth_det020_loose": SMOOTH_DET020_LOOSE_CONFIG,
    "smooth_responsive_det020": {
        **HYBRID_BASE_CONFIG,
        "high_conf_smooth_alpha": 0.65,
        "mid_conf_smooth_alpha": 0.45,
        "low_conf_smooth_alpha": 0.25,
        "det_conf": 0.20,
    },
    "hybrid_bytetrack_best": HYBRID_BEST_CONFIG,
    # Backward-compatible alias for older long-form command lines.
    "iou0_area0_condarea0_merge0_smooth_det020_loose_motion": SMOOTH_DET020_LOOSE_CONFIG,
}

PRESENTATION_PROFILES: dict[str, dict[str, object]] = {
    "hybrid_bytetrack": {
        "mode": "hybrid_bytetrack",
        "eval_config": "hybrid_bytetrack_best",
        "description": (
            "Complete accepted historical ByteTrack-specific optimization "
            "lineage with offline identity, Hidden, and geometry stages."
        ),
    },
}

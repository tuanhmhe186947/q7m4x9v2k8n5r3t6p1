"""Minimal production contracts for the canonical tracking methods."""

from __future__ import annotations

import inspect

from pig_behavior.tracking.method_registry import (
    ACTIVE_SCIENTIFIC_METHOD_IDS,
    SCIENTIFIC_METHOD_REGISTRY,
)
from pig_behavior.tracking.profiles import (
    PRESENTATION_PROFILES,
    get_eval_config,
)
from pig_behavior.tracking.runner import run_tracking


def test_method_registry_contract_is_exact_and_complete() -> None:
    expected = (
        "bytetrack_raw",
        "hybrid_bytetrack",
        "realtime_fast",
    )

    assert ACTIVE_SCIENTIFIC_METHOD_IDS == expected
    assert tuple(SCIENTIFIC_METHOD_REGISTRY) == expected
    assert set(PRESENTATION_PROFILES) == set(expected)
    for method_id, contract in SCIENTIFIC_METHOD_REGISTRY.items():
        assert contract.method_id == method_id
        assert contract.entry_point
        assert contract.detector_contract
        assert contract.tracker_contract
        assert contract.stage_graph
        assert contract.export_contract
        assert contract.artifact_authority
        assert contract.execution_authority_status


def test_hybrid_stage_activation_order_matches_clean_authority() -> None:
    source = inspect.getsource(run_tracking)
    ordered_calls = (
        "apply_identity_swap_guard(",
        "refine_shapes_temporally(",
        "stabilize_overlap_hidden_islands(",
        "repair_local_pair_swaps(",
        "repair_episode_pair_swaps(",
        "repair_long_pair_swaps(",
        "repair_suffix_pair_swaps(",
        "suppress_overlapped_small_low_confidence_boxes(",
        "repair_hidden_suffix_id_swaps(",
        "stabilize_realtime_motion_pairs(",
        "refine_near_wall_hidden_geometry(",
        "refine_far_camera_hidden_geometry(",
    )

    positions = [source.index(call) for call in ordered_calls]
    assert positions == sorted(positions)
    assert "model.track(" in source
    assert "persist=True" in source
    assert "apply_offline_repair_stack" not in source


def test_detector_contracts_bind_method_specific_profiles() -> None:
    hybrid = get_eval_config("hybrid_bytetrack_best")
    realtime = get_eval_config("realtime_fast")
    raw = get_eval_config("bytetrack_raw")

    assert hybrid["det_conf"] == 0.20
    assert hybrid["max_raw_detections"] == 64
    assert hybrid["enable_offline_smoothing"] is True
    assert realtime["det_conf"] == 0.25
    assert realtime["max_raw_detections"] == 32
    assert realtime["detect_every_n_frames"] == 2
    assert realtime["enable_offline_smoothing"] is False
    assert raw["enable_offline_smoothing"] is False


def test_no_cross_method_config_leakage() -> None:
    hybrid = get_eval_config("hybrid_bytetrack_best")
    realtime = get_eval_config("realtime_fast")
    raw = get_eval_config("bytetrack_raw")

    hybrid_only = (
        "hidden_owner_guard",
        "hidden_suffix_id_swap_use_overlap_persistence",
        "near_wall_hidden_geometry_refine",
        "far_camera_hidden_geometry_refine",
    )
    assert all(key in hybrid for key in hybrid_only)
    assert all(key not in realtime for key in hybrid_only)
    assert all(key not in raw for key in hybrid_only)


def test_tracker_lifecycle_export_and_unseen_guards_are_explicit() -> None:
    raw = SCIENTIFIC_METHOD_REGISTRY["bytetrack_raw"]
    hybrid = SCIENTIFIC_METHOD_REGISTRY["hybrid_bytetrack"]
    realtime = SCIENTIFIC_METHOD_REGISTRY["realtime_fast"]

    assert "persist=True" in raw.state_lifecycle
    assert "persist=True" in hybrid.state_lifecycle
    assert realtime.future_frame_policy == "CAUSAL_ZERO_DELAY"
    assert "cross-video" in realtime.state_lifecycle
    for contract in SCIENTIFIC_METHOD_REGISTRY.values():
        assert "CVAT" in contract.export_contract
        assert contract.unseen_authorization_status == "NOT_AUTHORIZED"

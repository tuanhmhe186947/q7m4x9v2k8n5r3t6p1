"""Minimal production contracts for the canonical tracking methods."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from pig_behavior.tracking.method_registry import (
    ACTIVE_SCIENTIFIC_METHOD_IDS,
    SCIENTIFIC_METHOD_REGISTRY,
)
from pig_behavior.tracking.profiles import (
    PRESENTATION_PROFILES,
    get_eval_config,
)
from pig_behavior.tracking.runner import run_tracking


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_method_registry_contract_is_exact_and_complete() -> None:
    expected = (
        "bytetrack_raw",
        "hybrid_bytetrack",
        "realtime_fast",
        "rf_hybrid",
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
        assert contract.canonical_version
        assert contract.prediction_authority_path
        assert contract.prediction_authority_hash
        assert contract.provenance_authority_path


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
    rf_hybrid = get_eval_config("rf_hybrid")
    raw = get_eval_config("bytetrack_raw")

    assert hybrid["det_conf"] == 0.20
    assert hybrid["max_raw_detections"] == 64
    assert hybrid["enable_offline_smoothing"] is True
    assert realtime["det_conf"] == 0.25
    assert realtime["max_raw_detections"] == 32
    assert realtime["detect_every_n_frames"] == 2
    assert realtime["enable_offline_smoothing"] is False
    assert {
        key: value
        for key, value in rf_hybrid.items()
        if key != "rf_hybrid_transfer"
    } == realtime
    assert rf_hybrid["rf_hybrid_transfer"] is True
    assert raw["enable_offline_smoothing"] is False


def test_no_cross_method_config_leakage() -> None:
    hybrid = get_eval_config("hybrid_bytetrack_best")
    realtime = get_eval_config("realtime_fast")
    rf_hybrid = get_eval_config("rf_hybrid")
    raw = get_eval_config("bytetrack_raw")

    hybrid_only = (
        "hidden_owner_guard",
        "hidden_suffix_id_swap_use_overlap_persistence",
        "near_wall_hidden_geometry_refine",
        "far_camera_hidden_geometry_refine",
    )
    assert all(key in hybrid for key in hybrid_only)
    assert all(key not in realtime for key in hybrid_only)
    assert all(key not in rf_hybrid for key in hybrid_only)
    assert all(key not in raw for key in hybrid_only)


def test_tracker_lifecycle_export_and_unseen_guards_are_explicit() -> None:
    raw = SCIENTIFIC_METHOD_REGISTRY["bytetrack_raw"]
    hybrid = SCIENTIFIC_METHOD_REGISTRY["hybrid_bytetrack"]
    realtime = SCIENTIFIC_METHOD_REGISTRY["realtime_fast"]
    rf_hybrid = SCIENTIFIC_METHOD_REGISTRY["rf_hybrid"]

    assert "persist=True" in raw.state_lifecycle
    assert "persist=True" in hybrid.state_lifecycle
    assert realtime.future_frame_policy == "CAUSAL_ZERO_DELAY"
    assert "cross-video" in realtime.state_lifecycle
    assert "immutable raw tracklet" in rf_hybrid.state_lifecycle
    assert all("BYTETRACK" not in stage for stage in rf_hybrid.stage_graph)
    for contract in SCIENTIFIC_METHOD_REGISTRY.values():
        assert "CVAT" in contract.export_contract
    assert (
        raw.unseen_authorization_status
        == "DEVELOPMENT_ONLY_NOT_PRIMARY_RQ2_UNSEEN"
    )
    assert (
        hybrid.unseen_authorization_status
        == "DEVELOPMENT_ARTIFACT_ONLY_EXACT_RUNTIME_UNAVAILABLE"
    )
    assert realtime.unseen_authorization_status == "PENDING_SEPARATE_PREFLIGHT"
    assert rf_hybrid.unseen_authorization_status == "NO"
    for contract in SCIENTIFIC_METHOD_REGISTRY.values():
        assert any(
            "STATE_8_DEVELOPMENT_EVALUATION_AUTHORITY_20260729.json"
            in authority
            for authority in contract.artifact_authority
        )
    assert rf_hybrid.execution_authority_status == (
        "DEVELOPMENT_EVALUATION_AUTHORITY_ESTABLISHED",
        "TRANSFER_SIGNAL_MIXED",
    )


def test_four_method_freeze_authority_matches_registry() -> None:
    repo = Path(__file__).resolve().parents[1]
    authority = json.loads(
        (
            repo
            / "docs"
            / "tracking"
            / "reconciliation"
            / "FOUR_METHOD_TRACKING_FREEZE_AUTHORITY_20260729.json"
        ).read_text(encoding="utf-8")
    )
    assert authority["active_methods"] == list(
        ACTIVE_SCIENTIFIC_METHOD_IDS
    )
    assert authority["obsolete_standardized_b1_active"] is False
    assert authority["obsolete_2x2_binding_active"] is False
    assert authority["decisions"]["architecture"] == (
        "PASS_CLEAN_TRACKING_RECONCILIATION"
    )
    assert authority["decisions"]["rq2_development"] == (
        "TRANSFER_SIGNAL_MIXED"
    )
    assert authority["ready_for_unseen_evaluation"] is True
    assert authority["guards"]["unseen_files_accessed"] == 0
    frozen_registry_hash = authority["code_and_contract_hashes"][
        "method_registry_sha256"
    ]
    current_registry_hash = _sha256(
        repo / "src" / "pig_behavior" / "tracking" / "method_registry.py"
    )
    assert frozen_registry_hash != current_registry_hash

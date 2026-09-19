"""Synthetic contracts for the canonical rf_hybrid transfer method."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pig_behavior.tracking import TrackingConfig
from pig_behavior.tracking.profiles import EVAL_CONFIG_OVERRIDES
from pig_behavior.tracking.profiles.realtime import REALTIME_FAST_CONFIG
from pig_behavior.tracking.rf_hybrid_transfer import (
    RF_HYBRID_TRANSFER_STAGE_IDS,
    RFHybridContractError,
    apply_rf_hybrid_transfer,
    build_rf_hybrid_transfer_config,
    canonical_transfer_hash,
    rf_hybrid_transfer_config_hash,
    validate_frozen_realtime_tracklets,
    validate_rf_hybrid_core_config,
    write_rf_hybrid_artifacts,
)

WIDTH = 220
HEIGHT = 120
MASK = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)


def _shape(
    frame: int,
    points: list[float],
    *,
    score: float = 0.9,
) -> dict[str, object]:
    return {
        "type": "rectangle",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": points,
        "frame": frame,
        "group": 0,
        "attributes": [
            {"name": "ID", "value": "ID_1"},
            {"name": "Behavior", "value": "unknown"},
            {"name": "Hidden", "value": "No"},
        ],
        "score": score,
        "elements": [],
        "label": "Pig_1",
        "_track_source": "detected",
        "_track_state": "VISIBLE",
        "_state_reason": "synthetic_fixture",
        "_missed_frames": 0,
        "_needs_review": score < 0.8,
        "_raw_track_id": None,
        "_ever_detected": True,
        "_ambiguous_occlusion": False,
        "_occlusion_hold": False,
        "_motion_state": "unknown",
    }


def _jitter_episode() -> list[dict[str, object]]:
    return [
        _shape(0, [10.0, 10.0, 30.0, 30.0]),
        _shape(1, [0.0, 0.0, 100.0, 100.0], score=0.3),
        _shape(2, [12.0, 10.0, 32.0, 30.0]),
    ]


def test_rf_hybrid_profile_core_is_exactly_realtime_fast() -> None:
    profile = EVAL_CONFIG_OVERRIDES["rf_hybrid"]

    assert profile["rf_hybrid_transfer"] is True
    assert {
        key: value
        for key, value in profile.items()
        if key != "rf_hybrid_transfer"
    } == REALTIME_FAST_CONFIG
    cfg = TrackingConfig(
        mode="realtime",
        write_output_video=False,
        **profile,
    )
    validate_rf_hybrid_core_config(cfg)

    cfg.det_conf = 0.30
    with pytest.raises(
        RFHybridContractError,
        match="differs from realtime_fast",
    ):
        validate_rf_hybrid_core_config(cfg)


def test_transfer_config_excludes_bytetrack_owner_and_reentry_flags() -> None:
    cfg = build_rf_hybrid_transfer_config()

    assert cfg.mode == "realtime"
    assert cfg.hidden_owner_guard is False
    assert cfg.hidden_owner_guard_hold_assignment is False
    assert cfg.reentry_unowned_raw_mismatch_episode_reject is False
    assert cfg.occlusion_reid_prefer_gap_over_bad_match is False
    assert cfg.identity_swap_guard is True
    assert cfg.hidden_suffix_id_swap_repair is True
    assert cfg.near_wall_hidden_geometry_refine is True
    assert cfg.far_camera_hidden_geometry_refine is True
    assert len(rf_hybrid_transfer_config_hash()) == 64


def test_transfer_preserves_raw_snapshot_and_executes_frozen_stage_order() -> None:
    raw = _jitter_episode()
    frozen = deepcopy(raw)
    frozen_hash = canonical_transfer_hash(frozen)

    result = apply_rf_hybrid_transfer(
        raw,
        WIDTH,
        HEIGHT,
        MASK,
        video="synthetic",
    )

    assert raw == frozen
    assert canonical_transfer_hash(raw) == frozen_hash
    assert tuple(
        item["stage_id"] for item in result.stage_activation
    ) == RF_HYBRID_TRANSFER_STAGE_IDS
    assert len(result.stage_activation) == 10
    assert result.input_authority_hash == frozen_hash
    assert result.output_authority_hash != frozen_hash
    assert result.changes
    required = {
        "episode_id",
        "video",
        "start_frame",
        "end_frame",
        "old_identity",
        "new_identity",
        "old_bbox",
        "new_bbox",
        "mechanism",
        "future_frames_used",
        "changed_frames",
        "changed_tracks",
        "decision_evidence",
    }
    assert required.issubset(result.changes[0])


def test_transfer_application_does_not_read_bytetrack_internal_state() -> None:
    source = inspect.getsource(apply_rf_hybrid_transfer)

    assert "model.track" not in source
    assert "_raw_track_id" not in source
    assert "hidden_owner_guard" not in source
    assert "reentry_" not in source


def test_frozen_tracklet_validation_rejects_duplicate_rows() -> None:
    duplicate = _shape(0, [10.0, 10.0, 30.0, 30.0])

    with pytest.raises(
        RFHybridContractError,
        match="duplicate frame/label",
    ):
        validate_frozen_realtime_tracklets([duplicate, deepcopy(duplicate)])


def test_transfer_artifacts_persist_source_output_and_ledger(
    tmp_path: Path,
) -> None:
    raw = _jitter_episode()
    result = apply_rf_hybrid_transfer(
        raw,
        WIDTH,
        HEIGHT,
        MASK,
        video="synthetic",
    )
    raw_path = tmp_path / "realtime_fast_output.json"
    hybrid_path = tmp_path / "rf_hybrid_output.json"
    ledger_path = tmp_path / "rf_hybrid_change_ledger.json"

    write_rf_hybrid_artifacts(
        realtime_fast_path=raw_path,
        rf_hybrid_path=hybrid_path,
        ledger_path=ledger_path,
        video="synthetic",
        raw_shapes=raw,
        result=result,
    )

    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    hybrid_payload = json.loads(hybrid_path.read_text(encoding="utf-8"))
    ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert raw_payload["method_id"] == "realtime_fast"
    assert hybrid_payload["method_id"] == "rf_hybrid"
    assert ledger_payload["source_method_id"] == "realtime_fast"
    assert ledger_payload["method_id"] == "rf_hybrid"
    assert len(ledger_payload["stage_activation"]) == 10
    assert ledger_payload["changes"]

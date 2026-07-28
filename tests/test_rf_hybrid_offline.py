from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pig_behavior.tracking import TrackingConfig
from pig_behavior.tracking.offline_repair import (
    OFFLINE_REPAIR_SEMANTIC_SHA256,
    RepairInputContractError,
    RepairLedgerContext,
    adapt_rf_shapes_for_offline_repair,
    apply_offline_repair_stack,
    build_frozen_offline_repair_config,
    canonical_authority_hash,
    offline_repair_semantic_hash,
    validate_rf_hybrid_core_config,
    write_repair_ledger,
    write_rf_raw_output,
)
from pig_behavior.tracking.profiles import (
    EVAL_CONFIG_OVERRIDES,
    PRESENTATION_PROFILES,
)
from pig_behavior.tracking.profiles.realtime import REALTIME_FAST_CONFIG
from pig_behavior.tracking.refinement import (
    apply_identity_swap_guard,
    refine_far_camera_hidden_geometry,
    refine_near_wall_hidden_geometry,
    refine_shapes_temporally,
    repair_episode_pair_swaps,
    repair_hidden_suffix_id_swaps,
    repair_local_pair_swaps,
    repair_long_pair_swaps,
    repair_suffix_pair_swaps,
    stabilize_overlap_hidden_islands,
    stabilize_realtime_motion_pairs,
    suppress_overlapped_small_low_confidence_boxes,
)

WIDTH = 220
HEIGHT = 120
PEN_MASK = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)


def _shape(
    frame: int,
    actor_id: int,
    points: list[float],
    *,
    hidden: bool = False,
    outside: bool = False,
    score: float = 0.9,
    source: str = "detected",
    state: str = "VISIBLE",
    missed: int = 0,
    timestamp: float | None = None,
) -> dict[str, object]:
    shape: dict[str, object] = {
        "type": "rectangle",
        "occluded": hidden,
        "outside": outside,
        "z_order": 0,
        "rotation": 0.0,
        "points": points,
        "frame": frame,
        "group": 0,
        "attributes": [
            {"name": "ID", "value": f"ID_{actor_id}"},
            {"name": "Behavior", "value": "unknown"},
            {"name": "Hidden", "value": "Yes" if hidden else "No"},
        ],
        "score": score,
        "elements": [],
        "label": f"Pig_{actor_id}",
        "_track_source": source,
        "_track_state": state,
        "_state_reason": "synthetic_fixture",
        "_missed_frames": missed,
        "_needs_review": hidden or source != "detected",
        "_raw_track_id": None,
        "_ever_detected": True,
        "_ambiguous_occlusion": False,
        "_occlusion_hold": source == "occlusion_hold",
        "_motion_state": "unknown",
    }
    if timestamp is not None:
        shape["_timestamp_seconds"] = timestamp
    return shape


def _identity(shape: dict[str, object]) -> str:
    attributes = shape["attributes"]
    assert isinstance(attributes, list)
    return next(
        str(attribute["value"])
        for attribute in attributes
        if attribute["name"] == "ID"
    )


def _run_frozen(
    shapes: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    adapted = adapt_rf_shapes_for_offline_repair(shapes)
    result = apply_offline_repair_stack(
        adapted,
        WIDTH,
        HEIGHT,
        PEN_MASK,
        build_frozen_offline_repair_config(),
        ledger_context=RepairLedgerContext(
            source_core="realtime_fast",
            video_key="synthetic",
        ),
    )
    return result.shapes, result.ledger


def _manual_hybrid_stack(
    shapes: list[dict[str, object]],
    cfg: TrackingConfig,
) -> list[dict[str, object]]:
    current = shapes
    if cfg.enable_offline_smoothing and cfg.identity_swap_guard:
        current = apply_identity_swap_guard(
            current,
            WIDTH,
            HEIGHT,
            cfg,
        )
    if cfg.enable_offline_smoothing and (
        cfg.smooth_boxes or cfg.refine_boxes
    ):
        current = refine_shapes_temporally(current, WIDTH, HEIGHT, cfg)
        current = stabilize_overlap_hidden_islands(current, cfg)
        current = repair_local_pair_swaps(current, WIDTH, HEIGHT, cfg)
        current = repair_episode_pair_swaps(current, WIDTH, HEIGHT, cfg)
        current = repair_long_pair_swaps(current, WIDTH, HEIGHT, cfg)
        current = repair_suffix_pair_swaps(current, WIDTH, HEIGHT, cfg)
        current = suppress_overlapped_small_low_confidence_boxes(
            current,
            cfg,
        )
        current = repair_hidden_suffix_id_swaps(current, cfg)
    current = stabilize_realtime_motion_pairs(
        current,
        WIDTH,
        HEIGHT,
        cfg,
    )
    current = refine_near_wall_hidden_geometry(
        current,
        WIDTH,
        HEIGHT,
        PEN_MASK,
        cfg,
    )
    return refine_far_camera_hidden_geometry(
        current,
        WIDTH,
        HEIGHT,
        cfg,
    )


def test_rf_adapter_preserves_clean_shape_values() -> None:
    raw = [_shape(0, 1, [10.0, 10.0, 30.0, 30.0])]

    adapted = adapt_rf_shapes_for_offline_repair(
        raw,
        expected_track_ids={1},
    )

    assert adapted == raw
    assert adapted is not raw
    assert adapted[0] is not raw[0]


def test_rf_adapter_canonicalizes_order_only() -> None:
    raw = [
        _shape(1, 2, [40.0, 10.0, 60.0, 30.0]),
        _shape(0, 2, [40.0, 10.0, 60.0, 30.0]),
        _shape(1, 1, [10.0, 10.0, 30.0, 30.0]),
        _shape(0, 1, [10.0, 10.0, 30.0, 30.0]),
    ]

    adapted = adapt_rf_shapes_for_offline_repair(
        raw,
        expected_track_ids={1, 2},
    )

    assert [
        (shape["frame"], shape["label"]) for shape in adapted
    ] == [
        (0, "Pig_1"),
        (0, "Pig_2"),
        (1, "Pig_1"),
        (1, "Pig_2"),
    ]
    expected_by_key = {
        (shape["frame"], shape["label"]): shape for shape in raw
    }
    for shape in adapted:
        assert shape == expected_by_key[(shape["frame"], shape["label"])]


def test_rf_adapter_fails_when_required_evidence_is_missing() -> None:
    raw = _shape(0, 1, [10.0, 10.0, 30.0, 30.0])
    del raw["score"]

    with pytest.raises(RepairInputContractError, match="missing fields"):
        adapt_rf_shapes_for_offline_repair([raw])


def test_rf_adapter_rejects_duplicate_frame_actor_key() -> None:
    raw = [
        _shape(0, 1, [10.0, 10.0, 30.0, 30.0]),
        _shape(0, 1, [11.0, 10.0, 31.0, 30.0]),
    ]

    with pytest.raises(
        RepairInputContractError,
        match="duplicate frame/actor",
    ):
        adapt_rf_shapes_for_offline_repair(raw)


def test_rf_adapter_rejects_non_monotonic_timestamp() -> None:
    raw = [
        _shape(
            0,
            1,
            [10.0, 10.0, 30.0, 30.0],
            timestamp=1.0,
        ),
        _shape(
            1,
            1,
            [10.0, 10.0, 30.0, 30.0],
            timestamp=0.5,
        ),
    ]

    with pytest.raises(RepairInputContractError, match="increase strictly"):
        adapt_rf_shapes_for_offline_repair(raw)


def test_rf_adapter_rejects_missing_actor_slot() -> None:
    raw = [
        _shape(0, 1, [10.0, 10.0, 30.0, 30.0]),
        _shape(0, 2, [40.0, 10.0, 60.0, 30.0]),
        _shape(1, 1, [10.0, 10.0, 30.0, 30.0]),
    ]

    with pytest.raises(RepairInputContractError, match="do not equal"):
        adapt_rf_shapes_for_offline_repair(
            raw,
            expected_track_ids={1, 2},
        )


def test_rf_adapter_rejects_inconsistent_hidden_state() -> None:
    raw = _shape(
        0,
        1,
        [10.0, 10.0, 30.0, 30.0],
        hidden=True,
    )
    raw["occluded"] = False

    with pytest.raises(RepairInputContractError, match="inconsistent"):
        adapt_rf_shapes_for_offline_repair([raw])


def test_rf_adapter_keeps_optional_timestamp_absent() -> None:
    raw = [_shape(0, 1, [10.0, 10.0, 30.0, 30.0])]

    adapted = adapt_rf_shapes_for_offline_repair(raw)

    assert "_timestamp_seconds" not in adapted[0]


def test_rf_golden_clean_uninterrupted_track_has_no_identity_change() -> None:
    raw = [
        _shape(frame, 1, [10.0, 10.0, 30.0, 30.0])
        for frame in range(3)
    ]

    repaired, _ = _run_frozen(raw)

    assert [_identity(shape) for shape in repaired] == ["ID_1"] * 3


def test_rf_golden_two_independent_tracks_do_not_merge_or_swap() -> None:
    raw = [
        shape
        for frame in range(3)
        for shape in (
            _shape(frame, 1, [10.0, 10.0, 30.0, 30.0]),
            _shape(frame, 2, [80.0, 10.0, 100.0, 30.0]),
        )
    ]

    repaired, _ = _run_frozen(raw)

    assert [_identity(shape) for shape in repaired] == [
        "ID_1",
        "ID_2",
    ] * 3


def test_rf_golden_short_gap_uses_frozen_temporal_refinement() -> None:
    raw = [
        _shape(0, 1, [20.0, 20.0, 40.0, 40.0]),
        _shape(
            1,
            1,
            [20.0, 20.0, 50.0, 40.0],
            hidden=True,
            score=0.4,
            source="predicted",
            state="MISSING",
            missed=1,
        ),
        _shape(2, 1, [20.0, 20.0, 40.0, 40.0]),
    ]

    repaired, _ = _run_frozen(raw)

    assert repaired[1]["points"] == [20.0, 20.0, 41.5, 40.0]
    assert _identity(repaired[1]) == "ID_1"


def test_rf_golden_long_gap_is_not_forced() -> None:
    raw = [
        _shape(0, 1, [20.0, 20.0, 40.0, 40.0]),
        _shape(
            40,
            1,
            [20.0, 20.0, 50.0, 40.0],
            hidden=True,
            score=0.4,
            source="predicted",
            state="MISSING",
            missed=1,
        ),
        _shape(80, 1, [20.0, 20.0, 40.0, 40.0]),
    ]

    repaired, _ = _run_frozen(raw)

    assert repaired[1]["points"] == [20.0, 20.0, 50.0, 40.0]
    assert _identity(repaired[1]) == "ID_1"


def test_rf_golden_ambiguous_pair_is_decided_only_by_frozen_repair() -> None:
    raw = [
        _shape(0, 1, [0.0, 10.0, 20.0, 30.0]),
        _shape(0, 2, [100.0, 10.0, 120.0, 30.0]),
        _shape(1, 1, [102.0, 10.0, 122.0, 30.0]),
        _shape(1, 2, [2.0, 10.0, 22.0, 30.0]),
    ]
    adapted = adapt_rf_shapes_for_offline_repair(raw)
    assert adapted == raw

    repaired, _ = _run_frozen(raw)

    assert repaired[2]["points"] == [2.0, 10.0, 22.0, 30.0]
    assert repaired[3]["points"] == [102.0, 10.0, 122.0, 30.0]


def test_rf_golden_missing_required_evidence_fails_closed() -> None:
    raw = _shape(0, 1, [10.0, 10.0, 30.0, 30.0])
    del raw["_track_source"]

    with pytest.raises(RepairInputContractError, match="missing fields"):
        adapt_rf_shapes_for_offline_repair([raw])


def test_rf_golden_duplicate_frame_index_fails_closed() -> None:
    raw = [
        _shape(0, 1, [10.0, 10.0, 30.0, 30.0]),
        _shape(0, 1, [11.0, 10.0, 31.0, 30.0]),
    ]

    with pytest.raises(RepairInputContractError, match="duplicate"):
        adapt_rf_shapes_for_offline_repair(raw)


def test_rf_golden_non_monotonic_timestamp_fails_closed() -> None:
    raw = [
        _shape(
            0,
            1,
            [10.0, 10.0, 30.0, 30.0],
            timestamp=2.0,
        ),
        _shape(
            1,
            1,
            [10.0, 10.0, 30.0, 30.0],
            timestamp=1.0,
        ),
    ]

    with pytest.raises(RepairInputContractError, match="increase strictly"):
        adapt_rf_shapes_for_offline_repair(raw)


def test_rf_golden_hidden_lost_representation_is_preserved() -> None:
    raw = [
        _shape(
            4,
            1,
            [10.0, 10.0, 30.0, 30.0],
            hidden=True,
            outside=True,
            score=0.2,
            source="predicted",
            state="LOST",
            missed=31,
        )
    ]

    adapted = adapt_rf_shapes_for_offline_repair(raw)

    assert adapted == raw


def test_rf_golden_raw_output_remains_immutable_after_repair() -> None:
    raw = [
        _shape(0, 1, [0.0, 10.0, 20.0, 30.0]),
        _shape(0, 2, [100.0, 10.0, 120.0, 30.0]),
        _shape(1, 1, [102.0, 10.0, 122.0, 30.0]),
        _shape(1, 2, [2.0, 10.0, 22.0, 30.0]),
    ]
    authority = deepcopy(raw)

    _run_frozen(raw)

    assert raw == authority


def test_rf_golden_repair_disabled_identity_round_trip() -> None:
    raw = [
        _shape(0, 1, [10.0, 10.0, 30.0, 30.0]),
        _shape(1, 1, [12.0, 10.0, 32.0, 30.0]),
    ]
    adapted = adapt_rf_shapes_for_offline_repair(raw)

    result = apply_offline_repair_stack(
        adapted,
        WIDTH,
        HEIGHT,
        PEN_MASK,
        TrackingConfig(mode="realtime"),
    )

    assert [_identity(shape) for shape in result.shapes] == [
        "ID_1",
        "ID_1",
    ]


def test_rf_golden_deterministic_repeat_matches_output_and_ledger() -> None:
    raw = [
        _shape(0, 1, [0.0, 10.0, 20.0, 30.0]),
        _shape(0, 2, [100.0, 10.0, 120.0, 30.0]),
        _shape(1, 1, [102.0, 10.0, 122.0, 30.0]),
        _shape(1, 2, [2.0, 10.0, 22.0, 30.0]),
    ]

    first_shapes, first_ledger = _run_frozen(raw)
    second_shapes, second_ledger = _run_frozen(raw)

    assert canonical_authority_hash(first_shapes) == canonical_authority_hash(
        second_shapes
    )
    assert canonical_authority_hash(first_ledger) == canonical_authority_hash(
        second_ledger
    )


def test_rf_raw_immutability_deep_fields_survive_identity_repair() -> None:
    raw = [
        _shape(0, 1, [0.0, 10.0, 20.0, 30.0]),
        _shape(0, 2, [100.0, 10.0, 120.0, 30.0]),
        _shape(1, 1, [102.0, 10.0, 122.0, 30.0]),
        _shape(1, 2, [2.0, 10.0, 22.0, 30.0]),
    ]
    authority = deepcopy(raw)

    repaired, _ = _run_frozen(raw)

    assert raw == authority
    assert [shape["points"] for shape in raw] == [
        shape["points"] for shape in authority
    ]
    assert [_identity(shape) for shape in raw] == [
        _identity(shape) for shape in authority
    ]
    assert repaired != raw


def test_rf_raw_immutability_writer_retains_internal_provenance(
    tmp_path: Path,
) -> None:
    raw = [_shape(0, 1, [10.0, 10.0, 30.0, 30.0])]
    path = tmp_path / "rf_raw_track_output.json"
    authority_hash = canonical_authority_hash(raw)

    write_rf_raw_output(
        path,
        shapes=raw,
        video_key="synthetic",
        input_authority_hash=authority_hash,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["input_authority_hash"] == authority_hash
    assert payload["shapes"] == raw
    assert payload["shapes"][0]["_track_state"] == "VISIBLE"


def test_rf_repair_ledger_contains_required_gt_free_fields() -> None:
    raw = [
        _shape(0, 1, [0.0, 10.0, 20.0, 30.0]),
        _shape(0, 2, [100.0, 10.0, 120.0, 30.0]),
        _shape(1, 1, [102.0, 10.0, 122.0, 30.0]),
        _shape(1, 2, [2.0, 10.0, 22.0, 30.0]),
    ]

    _, ledger = _run_frozen(raw)

    assert ledger
    required = {
        "repair_event_id",
        "source_core",
        "video_key",
        "repair_stage",
        "start_frame",
        "end_frame",
        "input_track_id",
        "output_track_id",
        "repair_operation",
        "repair_reason",
        "future_frames_used",
        "frames_modified",
        "repair_config_hash",
        "input_authority_hash",
        "output_authority_hash",
    }
    assert all(set(event) == required for event in ledger)
    assert all("gt" not in key.lower() for event in ledger for key in event)


def test_rf_repair_ledger_is_deterministically_ordered() -> None:
    raw = [
        _shape(0, 1, [0.0, 10.0, 20.0, 30.0]),
        _shape(0, 2, [100.0, 10.0, 120.0, 30.0]),
        _shape(1, 1, [102.0, 10.0, 122.0, 30.0]),
        _shape(1, 2, [2.0, 10.0, 22.0, 30.0]),
    ]

    first_shapes, first_ledger = _run_frozen(raw)
    second_shapes, second_ledger = _run_frozen(raw)

    assert first_shapes == second_shapes
    assert first_ledger == second_ledger
    assert [event["repair_event_id"] for event in first_ledger] == [
        event["repair_event_id"] for event in second_ledger
    ]


def test_rf_repair_ledger_writer_is_repeatable(tmp_path: Path) -> None:
    raw = [
        _shape(0, 1, [0.0, 10.0, 20.0, 30.0]),
        _shape(0, 2, [100.0, 10.0, 120.0, 30.0]),
        _shape(1, 1, [102.0, 10.0, 122.0, 30.0]),
        _shape(1, 2, [2.0, 10.0, 22.0, 30.0]),
    ]
    adapted = adapt_rf_shapes_for_offline_repair(raw)
    result = apply_offline_repair_stack(
        adapted,
        WIDTH,
        HEIGHT,
        PEN_MASK,
        build_frozen_offline_repair_config(),
        ledger_context=RepairLedgerContext(
            source_core="realtime_fast",
            video_key="synthetic",
        ),
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_repair_ledger(first, result, video_key="synthetic")
    write_repair_ledger(second, result, video_key="synthetic")

    assert first.read_bytes() == second.read_bytes()


def test_rf_profile_is_explicit_opt_in_and_non_default() -> None:
    profile = PRESENTATION_PROFILES["rf_hybrid_offline"]

    assert profile["mode"] == "realtime"
    assert profile["eval_config"] == "rf_hybrid_offline"
    assert profile["production_default"] is False
    assert profile["promotion_status"] == "NOT_PROMOTED"
    assert profile["unseen_execution_authorized"] is False
    assert EVAL_CONFIG_OVERRIDES["rf_hybrid_offline"][
        "rf_hybrid_offline"
    ] is True
    assert EVAL_CONFIG_OVERRIDES["rf_hybrid_offline"][
        "write_output_video"
    ] is False


def test_rf_profile_reuses_exact_realtime_fast_core_values() -> None:
    candidate = EVAL_CONFIG_OVERRIDES["rf_hybrid_offline"]

    for key, expected in REALTIME_FAST_CONFIG.items():
        assert candidate[key] == expected


def test_rf_profile_core_validation_rejects_semantic_drift() -> None:
    cfg = TrackingConfig(
        mode="realtime",
        rf_hybrid_offline=True,
        write_output_video=False,
        **REALTIME_FAST_CONFIG,
    )
    validate_rf_hybrid_core_config(cfg)
    cfg.detect_every_n_frames = 3

    with pytest.raises(ValueError, match="detect_every_n_frames"):
        validate_rf_hybrid_core_config(cfg)


def test_rf_frozen_repair_hash_matches_authority() -> None:
    cfg = build_frozen_offline_repair_config()

    assert offline_repair_semantic_hash(cfg) == (
        OFFLINE_REPAIR_SEMANTIC_SHA256
    )


def test_hybrid_shared_entry_matches_pre_adapter_stage_order() -> None:
    raw = [
        _shape(0, 1, [0.0, 10.0, 20.0, 30.0]),
        _shape(0, 2, [100.0, 10.0, 120.0, 30.0]),
        _shape(1, 1, [102.0, 10.0, 122.0, 30.0]),
        _shape(1, 2, [2.0, 10.0, 22.0, 30.0]),
    ]
    cfg = build_frozen_offline_repair_config()

    expected = _manual_hybrid_stack(deepcopy(raw), cfg)
    actual = apply_offline_repair_stack(
        deepcopy(raw),
        WIDTH,
        HEIGHT,
        PEN_MASK,
        cfg,
    ).shapes

    assert actual == expected

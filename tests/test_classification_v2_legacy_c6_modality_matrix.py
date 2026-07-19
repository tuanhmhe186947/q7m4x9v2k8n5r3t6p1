from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.training.legacy_development_c6_modality_matrix import (
    COMBINED_ALL7_FAMILY,
    COMBINED_ALL7_MODE,
    CONFIG_SCHEMA_V2,
    CONTROL_MODES,
    MODALITY_FEATURES,
    C6MatrixConfig,
    C6ModalityCache,
    _evaluated_temporal_freeze_errors,
    _exact_c6_alignment,
    _materialize_context_features,
    _numeric_modality_arrays,
    _validate_c6_run_packet,
    build_c6_modality_cache,
    build_c6_view,
    c6_mode_ids,
    fit_c6_normalization,
    static_c6_matrix_preflight,
    synthetic_c6_functional_preflight,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256


def _base_view(tmp_path: Path) -> LegacyL5CachedFeatureView:
    rows = 4
    slots = rows * 6
    tensor = np.arange(slots * 512, dtype=np.float32).reshape(slots, 512)
    tensor_path = tmp_path / "actor.npy"
    np.save(tensor_path, tensor, allow_pickle=False)
    windows = pd.DataFrame(
        {
            "window_id": [f"parent-{index}" for index in range(rows)],
            "temporal_unit_key": [f"unit-{index}" for index in range(rows)],
            "recording_group_id": ["date-a", "date-a", "date-b", "date-c"],
            "video_key": ["video-a", "video-a", "video-b", "video-c"],
            "source_type": ["legacy_recovered"] * rows,
            "dataset_id": ["legacy_recovered_16f"] * rows,
            "behavior_label": ["drink", "eat", "stand", "move"],
            "l5_role": ["train", "train", "train", "validation"],
        }
    )
    return LegacyL5CachedFeatureView(
        feature_tensor_path=tensor_path,
        feature_tensor_sha256="a" * 64,
        control_id="V1",
        temporal_view_name="legacy_c6_contiguous_centered_a128_v1",
        sequence_length=6,
        windows=windows,
        fold_manifest=pd.DataFrame(),
        feature_rows=np.arange(slots, dtype=np.int64).reshape(rows, 6),
        observed_mask=np.ones((rows, 6), dtype=np.bool_),
        time_delta=np.zeros((rows, 6), dtype=np.float32),
        targets=np.asarray([0, 1, 6, 7], dtype=np.int64),
        sample_weights=np.ones(rows, dtype=np.float64),
        audit={},
    )


def _geometry_cache(tmp_path: Path) -> C6ModalityCache:
    rows = 4
    feature_dim = len(MODALITY_FEATURES["geometry"])
    values = np.arange(rows * 6 * feature_dim, dtype=np.float32).reshape(
        rows, 6, feature_dim
    )
    feature_mask = np.ones_like(values, dtype=np.bool_)
    availability = np.ones((rows, 6), dtype=np.bool_)
    artifacts = {}
    for name, array in {
        "values": values,
        "feature_mask": feature_mask,
        "availability": availability,
    }.items():
        path = tmp_path / f"geometry_{name}.npy"
        np.save(path, array, allow_pickle=False)
        artifacts[name] = {"filename": path.name}
    windows = pd.DataFrame(
        {
            "cache_row": np.arange(rows),
            "window_id": [f"c6-{index}" for index in range(rows)],
            "temporal_unit_key": [f"unit-{index}" for index in range(rows)],
            "l5_role": ["train", "train", "train", "validation"],
        }
    )
    slot_rows = []
    for cache_row in range(rows):
        for slot in range(6):
            slot_rows.append(
                {
                    "cache_row": cache_row,
                    "window_id": f"c6-{cache_row}",
                    "slot_index": slot,
                    "native_frame_offset": slot + 5,
                    "frame_uid": f"frame-{cache_row}-{slot}",
                    "previous_frame_uid": (
                        "" if slot == 0 else f"frame-{cache_row}-{slot - 1}"
                    ),
                    "pair_uid": (
                        ""
                        if slot == 0
                        else f"frame-{cache_row}-{slot - 1}->frame-{cache_row}-{slot}"
                    ),
                    "window_slot_uid": f"c6-{cache_row}::slot={slot}",
                    "scene_frame_uid": f"scene-{cache_row}-{slot}",
                    "image_context_id": f"context-{cache_row}-{slot}",
                }
            )
    return C6ModalityCache(
        root=tmp_path,
        window_index=windows,
        slot_index=pd.DataFrame.from_records(slot_rows),
        manifest={
            "modalities": {
                "geometry": {
                    "artifacts": artifacts,
                }
            }
        },
        manifest_sha256="b" * 64,
    )


def _all7_cache(tmp_path: Path) -> C6ModalityCache:
    cache = _geometry_cache(tmp_path)
    rows = len(cache.window_index)
    for modality_index, (modality, names) in enumerate(
        MODALITY_FEATURES.items()
    ):
        if modality == "geometry":
            continue
        values = np.arange(
            rows * 6 * len(names), dtype=np.float32
        ).reshape(rows, 6, len(names))
        feature_mask = np.ones_like(values, dtype=np.bool_)
        availability = np.ones((rows, 6), dtype=np.bool_)
        unavailable_slot = modality_index % 6
        values[:, unavailable_slot] = 0.0
        feature_mask[:, unavailable_slot] = False
        availability[:, unavailable_slot] = False
        if modality == "motion":
            values[:, 0] = 0.0
            feature_mask[:, 0] = False
            availability[:, 0] = False
        if modality == "pen_context":
            values[:, 0, 3:] = 0.0
            feature_mask[:, 0, 3:] = False
        artifacts = {}
        for kind, array in {
            "values": values,
            "feature_mask": feature_mask,
            "availability": availability,
        }.items():
            path = tmp_path / f"{modality}_{kind}.npy"
            np.save(path, array, allow_pickle=False)
            artifacts[kind] = {"filename": path.name}
        cache.manifest["modalities"][modality] = {"artifacts": artifacts}
    return cache


def test_c6_matrix_has_actor_and_three_controls_per_modality() -> None:
    modes = c6_mode_ids()
    assert modes[0] == "actor_only"
    assert len(modes) == 1 + len(MODALITY_FEATURES) * len(CONTROL_MODES)
    for modality in MODALITY_FEATURES:
        assert [f"{modality}__{control}" for control in CONTROL_MODES] == [
            mode for mode in modes if mode.startswith(f"{modality}__")
        ]


def test_c6_matrix_can_bind_a_gate_authorized_modality_subset() -> None:
    modes = c6_mode_ids(("roi", "union_context"))

    assert modes == (
        "actor_only",
        "roi__parameter_matched_zero",
        "roi__availability_only",
        "roi__real",
        "union_context__parameter_matched_zero",
        "union_context__availability_only",
        "union_context__real",
    )


def test_c6_combined_matrix_has_actor_and_three_equal_width_controls() -> None:
    modes = c6_mode_ids(experiment_family=COMBINED_ALL7_FAMILY)

    assert modes == (
        "actor_only",
        f"{COMBINED_ALL7_MODE}__parameter_matched_zero",
        f"{COMBINED_ALL7_MODE}__availability_only",
        f"{COMBINED_ALL7_MODE}__real",
    )
    audit = synthetic_c6_functional_preflight(
        experiment_family=COMBINED_ALL7_FAMILY
    )
    assert audit["valid"] is True
    assert set(audit["modes"]) == set(modes)
    counts = {
        audit["modes"][f"{COMBINED_ALL7_MODE}__{control}"][
            "parameter_count"
        ]
        for control in CONTROL_MODES
    }
    assert len(counts) == 1


def test_synthetic_functional_preflight_covers_all_widths_and_resume() -> None:
    audit = synthetic_c6_functional_preflight()
    assert audit["valid"] is True
    assert audit["project_data_rows_read"] == 0
    assert audit["optimizer_steps_on_project_data"] == 0
    assert set(audit["modes"]) == set(c6_mode_ids())
    assert audit["resume_audit"]["valid"] is True


def test_combined_all7_controls_keep_train_only_states_and_seven_masks(
    tmp_path: Path,
) -> None:
    base = _base_view(tmp_path)
    cache = _all7_cache(tmp_path)
    train_rows = np.asarray([0, 1, 2], dtype=np.int64)
    views = {
        control: build_c6_view(
            base,
            cache,
            f"{COMBINED_ALL7_MODE}__{control}",
            train_rows,
        )
        for control in CONTROL_MODES
    }
    positions = np.asarray([0, 3], dtype=np.int64)
    sequences = {
        control: view.load_sequences(positions)
        for control, view in views.items()
    }
    expected_width = 512 + sum(
        len(names) + 1 for names in MODALITY_FEATURES.values()
    )
    assert {values.shape for values in sequences.values()} == {
        (2, 6, expected_width)
    }
    assert np.all(sequences["parameter_matched_zero"][..., 512:] == 0.0)
    cursor = 512
    observed_masks = []
    for modality, names in MODALITY_FEATURES.items():
        value_end = cursor + len(names)
        assert np.all(sequences["availability_only"][..., cursor:value_end] == 0.0)
        observed = sequences["availability_only"][..., value_end]
        expected = cache.load_availability(modality, positions).astype(np.float32)
        assert np.array_equal(observed, expected)
        observed_masks.append(observed)
        cursor = value_end + 1
    assert len({mask.tobytes() for mask in observed_masks}) == 7
    real = views["real"]
    assert len(real.combined_normalizations) == 7
    assert all(
        state.validation_rows_read_for_fit == 0
        and state.outer_rows_read_for_fit == 0
        for state in real.combined_normalizations
    )
    missing = real.with_missing_modality().load_sequences(positions)
    assert np.all(missing[..., 512:] == 0.0)


def test_normalization_and_controls_are_train_only_and_parameter_matched(
    tmp_path: Path,
) -> None:
    base = _base_view(tmp_path)
    cache = _geometry_cache(tmp_path)
    train_rows = np.asarray([0, 1, 2], dtype=np.int64)
    state = fit_c6_normalization(cache, "geometry", train_rows)
    assert state.train_native_units == 3
    assert state.validation_rows_read_for_fit == 0
    assert state.outer_rows_read_for_fit == 0

    real = build_c6_view(base, cache, "geometry__real", train_rows)
    availability = build_c6_view(
        base, cache, "geometry__availability_only", train_rows
    )
    zero = build_c6_view(
        base, cache, "geometry__parameter_matched_zero", train_rows
    )
    positions = np.asarray([0, 3], dtype=np.int64)
    real_values = real.load_sequences(positions)
    availability_values = availability.load_sequences(positions)
    zero_values = zero.load_sequences(positions)
    assert real_values.shape == availability_values.shape == zero_values.shape
    assert real_values.shape == (2, 6, 512 + len(state.feature_names) + 1)
    assert np.all(availability_values[..., 512:-1] == 0.0)
    assert np.all(availability_values[..., -1] == 1.0)
    assert np.all(zero_values[..., 512:] == 0.0)
    assert np.all(real.with_missing_modality().load_sequences(positions)[..., 512:] == 0.0)


def test_exact_c6_alignment_keeps_one_sequence_per_native_unit() -> None:
    windows = pd.DataFrame(
        {
            "window_id": ["parent-a", "parent-b"],
            "temporal_unit_key": ["unit-a", "unit-b"],
            "l5_role": ["train", "validation"],
            "behavior_label": ["stand", "move"],
        }
    )
    rows = []
    for unit_index, unit in enumerate(("unit-a", "unit-b")):
        for frame_index in range(16):
            rows.append(
                {
                    "temporal_unit_key": unit,
                    "frame_uid": f"frame-{unit_index}-{frame_index}",
                    "scene_frame_uid": f"scene-{unit_index}-{frame_index}",
                    "object_track_key": f"track-{unit_index}",
                    "frame_index": frame_index,
                    "source_type": "legacy_recovered",
                    "dataset_id": "legacy_recovered_16f",
                    "video_key": f"video-{unit_index}",
                    "lineage_scope": "legacy-only-unreviewed-development",
                    "human_review_complete": False,
                }
            )
    aligned, slots, selected = _exact_c6_alignment(
        windows,
        pd.DataFrame.from_records(rows),
    )
    assert len(aligned) == 2
    assert len(slots) == len(selected) == 12
    assert slots.groupby("cache_row")["native_frame_offset"].apply(list).tolist() == [
        [5, 6, 7, 8, 9, 10],
        [5, 6, 7, 8, 9, 10],
    ]
    assert aligned["temporal_unit_key"].is_unique
    assert slots[["window_id", "slot_index"]].duplicated().sum() == 0


def test_project_data_build_is_disabled_before_clean_handoff(tmp_path: Path) -> None:
    config = C6MatrixConfig(
        path=tmp_path / "config.json",
        payload={"execution": {"data_run_authorized": False}},
        repo_root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="clean lineage handoff"):
        build_c6_modality_cache(config)


def test_run_packet_validation_requires_exact_lineage_and_claim_flags() -> None:
    packet = {
        "status": "completed",
        "mode_id": "geometry__real",
        "repeat_id": "repeat01",
        "process_id": 123,
        "config_sha256": "c" * 64,
        "cache_manifest_sha256": "d" * 64,
        "lineage_scope": "legacy-only-unreviewed-development",
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "selection_sha256": "e" * 64,
        "parameter_sha256": "f" * 64,
        "prediction_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "valid": True,
    }
    assert _validate_c6_run_packet(
        packet,
        mode_id="geometry__real",
        repeat_id="repeat01",
        config_sha256="c" * 64,
        cache_manifest_sha256="d" * 64,
    ) == []
    packet["human_review_complete"] = True
    assert "packet_mismatch=geometry__real:repeat01:human_review_complete" in (
        _validate_c6_run_packet(
            packet,
            mode_id="geometry__real",
            repeat_id="repeat01",
            config_sha256="c" * 64,
            cache_manifest_sha256="d" * 64,
        )
    )


def test_static_preflight_reports_authorized_prepared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "classification_v2.legacy_c6_temporal_base_freeze.v1"
                ),
                "status": "PASS_C6_TEMPORAL_BASE_FREEZE",
                "decision": "FREEZE_PRIOR_A128_FOR_C6_MODALITY_SCREENING",
                "selected_base_mode": "A128",
                "modality_matrix_authorized": True,
                "lineage_scope": "legacy-only-unreviewed-development",
                "human_review_complete": False,
                "reviewed_or_final_claim_allowed": False,
                "q2_claim_allowed": False,
                "full_oof_authorized": False,
                "valid": True,
            }
        ),
        encoding="utf-8",
    )
    config = C6MatrixConfig(
        path=config_path,
        payload={
            "schema_version": CONFIG_SCHEMA_V2,
            "experiment_contract": {
                "changed_scientific_family": "single_optional_modality"
            },
            "temporal_base_freeze": {
                "path": "freeze.json",
                "sha256": "a" * 64,
            },
            "execution": {"data_run_authorized": True},
            "matrix": {"mode_ids": list(c6_mode_ids())},
            "temporal_contract": {
                "native_frame_offsets": [5, 6, 7, 8, 9, 10],
            },
            "output": {"root_relative_path": "fresh-c6-output"},
        },
        repo_root=tmp_path,
    )
    monkeypatch.setattr(
        "pig_behavior.classification_v2.training."
        "legacy_development_c6_modality_matrix._declared_input_paths",
        lambda _: {},
    )

    audit = static_c6_matrix_preflight(config)

    assert audit["valid"] is True
    assert audit["data_run_authorized"] is True


def test_numeric_materialization_resets_window_local_features() -> None:
    rows = 2
    quality_names = [
        "bbox_valid",
        "actor_bbox_valid",
        "geometry_feature_valid",
        "spatiotemporal_feature_valid",
        "roi_feeder_available",
        "roi_drinker_available",
        "roi_toy_available",
        "social_neighbor_available",
    ]
    arrays = {
        "observed_mask": np.ones((rows, 6), dtype=np.float32),
        "frame_index_sequence": np.tile(
            np.arange(6, dtype=np.int32), (rows, 1)
        ),
        "quality_mask": np.ones(
            (rows, 6, len(quality_names)), dtype=np.float32
        ),
        "motion_delta": np.ones((rows, 6, 10), dtype=np.float32),
        "roi_class_relation": np.ones((rows, 6, 18), dtype=np.float32),
        "social_relation": np.ones((rows, 6, 10), dtype=np.float32),
        "pen_boundary_context": np.ones((rows, 6, 7), dtype=np.float32),
    }
    exported = SimpleNamespace(
        arrays=arrays,
        feature_names={"quality_mask": quality_names},
    )
    pen_rows = []
    slot_rows = []
    for cache_row in range(rows):
        for slot_index in range(6):
            pen_rows.append(
                {
                    "cache_row": cache_row,
                    "slot_index": slot_index,
                    "pen_context_available": True,
                    "pen_context_quality_valid": True,
                    **{
                        name: float(cache_row + slot_index + feature_index + 1)
                        for feature_index, name in enumerate(
                            MODALITY_FEATURES["geometry"]
                        )
                    },
                }
            )
            slot_rows.append(
                {"cache_row": cache_row, "slot_index": slot_index}
            )
    result = _numeric_modality_arrays(
        exported,
        pd.DataFrame.from_records(pen_rows),
        pd.DataFrame.from_records(slot_rows),
    )
    assert set(result) == {
        "geometry",
        "motion",
        "roi",
        "numeric_social",
        "pen_context",
    }
    assert not result["motion"]["availability"][:, 0].any()
    assert np.all(result["motion"]["values"][:, 0] == 0.0)
    assert result["pen_context"]["feature_mask"][:, 0, :3].all()
    assert not result["pen_context"]["feature_mask"][:, 0, 3:].any()
    assert result["roi"]["values"].shape == (rows, 6, 18)


def test_context_materialization_preserves_missingness(tmp_path: Path) -> None:
    tensor = np.stack(
        [
            np.full(512, 1.0, dtype=np.float32),
            np.full(512, 2.0, dtype=np.float32),
        ]
    )
    tensor_path = tmp_path / "context.npy"
    np.save(tensor_path, tensor, allow_pickle=False)
    feature_index = pd.DataFrame(
        {
            "image_context_id": ["context-a", "context-b"],
            "feature_row": [0, 1],
        }
    )
    keys = pd.Series(
        ["context-a", "", "context-b", "", "context-a", ""] * 2
    )
    payload = _materialize_context_features(
        keys,
        feature_index,
        key_column="image_context_id",
        tensor_path=tensor_path,
    )
    assert payload["values"].shape == (2, 6, 512)
    assert payload["availability"].sum() == 6
    assert np.all(payload["values"][~payload["feature_mask"]] == 0.0)


def test_evaluated_temporal_freeze_binds_base_decision(
    tmp_path: Path,
) -> None:
    universe = {
        "native_units": 241,
        "video_clusters": 32,
        "outer_holdout_rows": 0,
    }
    decision_path = tmp_path / "base_decision.json"
    decision_path.write_text(
        json.dumps({"common_native_universe": universe}),
        encoding="utf-8",
    )
    freeze = {
        "schema_version": (
            "classification_v2.legacy_c6_temporal_base_freeze.v2"
        ),
        "status": "PASS_C6_TEMPORAL_BASE_FREEZE",
        "decision": (
            "FREEZE_EVALUATED_A128_FOR_LEGACY_16F_MODALITY_SCREENING"
        ),
        "selected_base_mode": "A128",
        "selected_base_is_carried_prior_not_tested_in_this_matrix": False,
        "modality_matrix_authorized": True,
        "lineage_scope": "legacy-only-unreviewed-development",
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "valid": True,
        "base_selection_decision": {
            "path": decision_path.name,
            "sha256": file_sha256(decision_path),
        },
        "common_native_universe": universe,
    }

    errors = _evaluated_temporal_freeze_errors(
        SimpleNamespace(repo_root=tmp_path),
        freeze,
    )

    assert errors == []

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.features.native_evidence_contract import (
    NATIVE_EVIDENCE_SEMANTICS_VERSION,
    NATIVE_FEATURE_COMPUTATION_GRAIN,
    NATIVE_MOTION_SCHEMA_VERSION,
    NATIVE_PAIR_SCOPE_KEY,
    check_native_review_evidence,
    dataframe_sha256,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    _add_temporal_deltas,
    audit_enhanced_spatiotemporal_features,
    build_enhanced_spatiotemporal_features,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "classification_v2"
    / "scientific_contract_v1"
    / "00_pipeline_contract.yaml"
)
CONTRACT_MANIFEST = CONTRACT.parent / "contract_manifest.json"
CODE_SHA = "a" * 40
CONTRACT_SHA = "c" * 64


def _native_input(
    *,
    frames: list[int] | None = None,
    timestamps: list[float] | None = None,
    centers: list[float] | None = None,
    units: list[str] | None = None,
    bbox_valid: list[bool] | None = None,
    actor_keys: list[str] | None = None,
) -> pd.DataFrame:
    frames = frames or [0, 1, 2]
    row_count = len(frames)
    timestamps = timestamps or [float(value) for value in frames]
    centers = centers or [0.0, 0.1, 0.2]
    units = units or ["actor-a|anchor=0"] * row_count
    bbox_valid = bbox_valid or [True] * row_count
    actor_keys = actor_keys or ["actor-a"] * row_count
    return pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * row_count,
            "dataset_id": ["fixture"] * row_count,
            "video_key": ["video-a"] * row_count,
            "object_track_key": actor_keys,
            "temporal_unit_key": units,
            "scene_frame_uid": [
                f"scene-{frame}" for frame in frames
            ],
            "frame_uid": [
                f"scene-{frame}|actor-{position}"
                for position, frame in enumerate(frames)
            ],
            "frame_index": frames,
            "timestamp_sec": timestamps,
            "pig_id": ["ID_1"] * row_count,
            "track_id": ["track-1"] * row_count,
            "behavior": ["move"] * row_count,
            "bbox_valid": bbox_valid,
            "x1": [value * 100.0 for value in centers],
            "y1": [10.0] * row_count,
            "x2": [value * 100.0 + 20.0 for value in centers],
            "y2": [30.0] * row_count,
            "bbox_area": [400.0] * row_count,
            "cx_n": centers,
            "cy_n": [0.5] * row_count,
            "bw_n": [0.2] * row_count,
            "bh_n": [0.2] * row_count,
            "area_n": [0.04] * row_count,
            "aspect_ratio": [1.0] * row_count,
            "box_diag_n": [np.hypot(0.2, 0.2)] * row_count,
            "feature_computation_grain": [
                "FRAME_LOCAL_PRIMITIVES"
            ]
            * row_count,
            "pair_scope_key": [""] * row_count,
        }
    )


def _build(**kwargs: object) -> pd.DataFrame:
    return build_enhanced_spatiotemporal_features(
        _native_input(**kwargs)
    )


def test_temporal_unit_key_and_provenance_preserved_end_to_end() -> None:
    source = _native_input()
    output = build_enhanced_spatiotemporal_features(source)

    assert output["temporal_unit_key"].equals(source["temporal_unit_key"])
    assert output["feature_computation_grain"].eq(
        NATIVE_FEATURE_COMPUTATION_GRAIN
    ).all()
    assert output["pair_scope_key"].equals(output["temporal_unit_key"])
    assert output["evidence_semantics_version"].eq(
        NATIVE_EVIDENCE_SEMANTICS_VERSION
    ).all()
    assert output["motion_schema_version"].eq(
        NATIVE_MOTION_SCHEMA_VERSION
    ).all()


def test_missing_temporal_unit_key_fails_closed() -> None:
    source = _native_input().drop(columns=["temporal_unit_key"])
    with pytest.raises(ValueError, match="temporal_unit_key"):
        build_enhanced_spatiotemporal_features(source)


def test_first_frame_of_unit_has_no_previous_pair() -> None:
    output = _build()
    first = output.sort_values("frame_index").iloc[0]
    assert bool(first["previous_observation_available"]) is False
    assert bool(first["valid_motion_pair"]) is False
    assert np.isnan(first["speed_n_per_second"])


def test_cross_unit_continuation_does_not_form_pair() -> None:
    output = _build(
        frames=[0, 1, 2, 3],
        timestamps=[0.0, 1.0, 2.0, 3.0],
        centers=[0.0, 0.1, 0.2, 0.3],
        units=["u1", "u1", "u2", "u2"],
    )
    ordered = output.sort_values("frame_index")
    assert bool(ordered.iloc[2]["valid_motion_pair"]) is False
    assert np.isnan(ordered.iloc[2]["speed_n_per_second"])
    assert int(ordered["valid_motion_pair"].sum()) == 2


def test_valid_stationary_pair_is_measured_zero() -> None:
    output = _build(
        frames=[0, 1],
        timestamps=[0.0, 1.0],
        centers=[0.2, 0.2],
        units=["u1", "u1"],
    ).sort_values("frame_index")
    assert bool(output.iloc[1]["valid_motion_pair"]) is True
    assert output.iloc[1]["speed_n_per_second"] == 0.0
    assert bool(output.iloc[1]["motion_feature_available"]) is True


def test_missing_pair_is_distinct_from_stationary_zero() -> None:
    output = _build(
        frames=[0, 1],
        timestamps=[0.0, 1.0],
        centers=[0.2, 0.2],
        units=["u1", "u1"],
    ).sort_values("frame_index")
    assert np.isnan(output.iloc[0]["speed_n_per_second"])
    assert output.iloc[1]["speed_n_per_second"] == 0.0
    assert not bool(output.iloc[0]["valid_motion_pair"])
    assert bool(output.iloc[1]["valid_motion_pair"])


def test_positive_finite_delta_time_is_required() -> None:
    output = _build(
        frames=[0, 1, 2],
        timestamps=[0.0, 1.0, np.nan],
        centers=[0.0, 0.1, 0.2],
    ).sort_values("frame_index")
    assert output["valid_delta_time"].tolist() == [False, True, False]
    assert output["valid_motion_pair"].tolist() == [False, True, False]


@pytest.mark.parametrize("second_timestamp", [0.0, -1.0])
def test_nonpositive_delta_time_is_invalid(
    second_timestamp: float,
) -> None:
    output = _build(
        frames=[0, 1],
        timestamps=[0.0, second_timestamp],
        centers=[0.0, 0.1],
    ).sort_values("frame_index")
    assert not bool(output.iloc[1]["valid_delta_time"])
    assert not bool(output.iloc[1]["valid_motion_pair"])
    assert np.isnan(output.iloc[1]["speed_n_per_second"])


def test_frame_gap_is_valid_sparse_velocity_support() -> None:
    output = _build(
        frames=[0, 2],
        timestamps=[0.0, 2.0],
        centers=[0.0, 0.2],
        units=["u1", "u1"],
    ).sort_values("frame_index")
    assert not bool(output.iloc[1]["adjacent_motion_pair_valid"])
    assert bool(output.iloc[1]["sparse_velocity_pair_valid"])
    assert bool(output.iloc[1]["valid_motion_pair"])
    assert output.iloc[1]["speed_n_per_second"] == pytest.approx(0.1)


def test_previous_geometry_invalidates_pair() -> None:
    output = _build(
        frames=[0, 1],
        timestamps=[0.0, 1.0],
        centers=[0.0, 0.1],
        bbox_valid=[False, True],
    ).sort_values("frame_index")
    assert not bool(output.iloc[1]["previous_geometry_valid"])
    assert not bool(output.iloc[1]["valid_motion_pair"])


def test_current_geometry_invalidates_pair() -> None:
    output = _build(
        frames=[0, 1],
        timestamps=[0.0, 1.0],
        centers=[0.0, 0.1],
        bbox_valid=[True, False],
    ).sort_values("frame_index")
    assert not bool(output.iloc[1]["current_geometry_valid"])
    assert not bool(output.iloc[1]["valid_motion_pair"])


def test_actor_identity_discontinuity_does_not_form_pair() -> None:
    rows = _native_input(
        frames=[0, 1],
        timestamps=[0.0, 1.0],
        centers=[0.0, 0.1],
        units=["u1", "u1"],
        actor_keys=["actor-a", "actor-b"],
    )
    output = _add_temporal_deltas(rows).sort_values("frame_index")
    assert output["valid_motion_pair"].tolist() == [False, False]
    assert output["previous_observation_available"].tolist() == [
        False,
        False,
    ]


def test_valid_pair_mean_uses_valid_denominator() -> None:
    output = _build(
        frames=[0, 1, 2],
        timestamps=[0.0, 1.0, 2.0],
        centers=[0.0, 0.1, 0.4],
        bbox_valid=[False, True, True],
    )
    assert output["valid_pair_count"].iloc[0] == 1
    assert output["speed_n_per_second_mean_unit"].iloc[0] == pytest.approx(
        0.3
    )


def test_motion_energy_uses_only_valid_pairs() -> None:
    output = _build(
        frames=[0, 1, 2],
        timestamps=[0.0, 1.0, 2.0],
        centers=[0.0, 0.1, 0.4],
        bbox_valid=[False, True, True],
    )
    assert output["motion_energy_n_per_second2_unit"].iloc[
        0
    ] == pytest.approx(0.3**2)


def test_motion_active_ratio_uses_valid_pair_denominator() -> None:
    output = _build(
        frames=[0, 1, 2],
        timestamps=[0.0, 1.0, 2.0],
        centers=[0.0, 0.1, 0.1],
        bbox_valid=[False, True, True],
    )
    assert output["valid_pair_count"].iloc[0] == 1
    assert output["motion_active_ratio_per_second_unit"].iloc[0] == 0.0
    assert output["motion_stationary_ratio_per_second_unit"].iloc[0] == 1.0


def test_motion_burstiness_uses_valid_pair_denominator() -> None:
    output = _build(
        frames=[0, 1, 2, 3],
        timestamps=[0.0, 1.0, 2.0, 3.0],
        centers=[0.0, 0.1, 0.2, 0.5],
        bbox_valid=[False, True, True, True],
    )
    expected_std = np.std([0.1, 0.3], ddof=1)
    expected = expected_std / (np.mean([0.1, 0.3]) + 1e-9)
    assert output["valid_pair_count"].iloc[0] == 2
    assert output["motion_burstiness_n_per_second_unit"].iloc[
        0
    ] == pytest.approx(expected)


def test_all_pairs_invalid_have_placeholder_and_unavailable_mask() -> None:
    output = _build(
        bbox_valid=[False, False, False],
    )
    assert output["valid_pair_count"].eq(0).all()
    assert output["possible_pair_count"].eq(2).all()
    assert output["valid_pair_ratio"].eq(0.0).all()
    assert output["motion_feature_coverage"].eq(0.0).all()
    assert not output["motion_feature_available"].any()
    assert output["speed_n_per_second_mean_unit"].eq(0.0).all()
    assert output["speed_n_per_second"].isna().all()


def test_exactly_one_valid_pair_uses_complete_support() -> None:
    output = _build(
        centers=[0.0, 0.1, 0.3],
        bbox_valid=[False, True, True],
    )
    assert output["valid_pair_count"].eq(1).all()
    assert output["possible_pair_count"].eq(2).all()
    assert output["valid_pair_ratio"].eq(0.5).all()
    assert output["motion_feature_coverage"].eq(0.5).all()
    assert output["motion_feature_available"].all()
    assert output["speed_n_per_second_mean_unit"].iloc[0] == pytest.approx(
        0.2
    )


def test_pair_coverage_counts_are_consistent() -> None:
    output = _build(
        frames=[0, 1, 2, 3],
        timestamps=[0.0, 1.0, 1.0, 3.0],
        centers=[0.0, 0.1, 0.2, 0.3],
    )
    assert output["observed_frame_count"].eq(4).all()
    assert output["possible_pair_count"].eq(3).all()
    assert output["valid_pair_count"].eq(2).all()
    assert output["valid_pair_ratio"].eq(2.0 / 3.0).all()
    assert output["motion_feature_coverage_available"].all()


def test_pair_features_are_invariant_to_input_row_order() -> None:
    source = _native_input(
        frames=[0, 1, 2, 3],
        timestamps=[0.0, 1.0, 2.0, 3.0],
        centers=[0.0, 0.1, 0.3, 0.6],
    )
    ordered = build_enhanced_spatiotemporal_features(source)
    shuffled = build_enhanced_spatiotemporal_features(
        source.sample(frac=1.0, random_state=7).reset_index(drop=True)
    )
    columns = [
        "frame_uid",
        "valid_motion_pair",
        "speed_n_per_second",
        "valid_pair_count",
        "motion_feature_coverage",
    ]
    pd.testing.assert_frame_equal(
        ordered[columns].sort_values("frame_uid").reset_index(drop=True),
        shuffled[columns].sort_values("frame_uid").reset_index(drop=True),
    )


def test_population_and_temporal_membership_are_preserved() -> None:
    source = _native_input()
    output = build_enhanced_spatiotemporal_features(source)
    input_hash = dataframe_sha256(source)
    producer_audit = audit_enhanced_spatiotemporal_features(
        output,
        input_rows=len(source),
        code_sha=CODE_SHA,
        input_sha256=input_hash,
        contract_manifest_sha256=CONTRACT_SHA,
    )
    checker = check_native_review_evidence(
        source,
        output,
        producer_audit=producer_audit,
        code_sha=CODE_SHA,
        input_sha256=input_hash,
        contract_manifest_sha256=CONTRACT_SHA,
    )
    assert checker["errors"] == []
    assert checker["valid"] is True
    assert checker["population_preserved"] is True


def test_provenance_checker_fails_closed_on_drift() -> None:
    source = _native_input()
    output = build_enhanced_spatiotemporal_features(source)
    input_hash = dataframe_sha256(source)
    producer_audit = audit_enhanced_spatiotemporal_features(
        output,
        input_rows=len(source),
        code_sha=CODE_SHA,
        input_sha256=input_hash,
        contract_manifest_sha256=CONTRACT_SHA,
    )
    drift = output.copy()
    drift.loc[drift.index[0], "evidence_semantics_version"] = ""
    checker = check_native_review_evidence(
        source,
        drift,
        producer_audit=producer_audit,
        code_sha=CODE_SHA,
        input_sha256=input_hash,
        contract_manifest_sha256=CONTRACT_SHA,
    )
    assert checker["valid"] is False
    assert any(
        "native_provenance_mismatch" in error
        for error in checker["errors"]
    )


def test_native_builder_and_independent_checker_cli(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "frame_local.csv"
    output_path = tmp_path / "native.csv"
    producer_audit_path = tmp_path / "producer_audit.json"
    checker_path = tmp_path / "checker.json"
    _native_input().to_csv(source_path, index=False)
    builder = [
        sys.executable,
        "scripts/classification_v2/00_source_feature_temporal/"
        "classification_v2_build_enhanced_spatiotemporal_features.py",
        "--input-csv",
        str(source_path),
        "--output-csv",
        str(output_path),
        "--audit-json",
        str(producer_audit_path),
        "--code-sha",
        CODE_SHA,
        "--contract-manifest",
        str(CONTRACT_MANIFEST),
        "--no-pen-context",
    ]
    built = subprocess.run(
        builder,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr

    checker = [
        sys.executable,
        "scripts/classification_v2/00_source_feature_temporal/"
        "check_classification_v2_native_review_evidence.py",
        "--input-csv",
        str(source_path),
        "--native-evidence-csv",
        str(output_path),
        "--producer-audit-json",
        str(producer_audit_path),
        "--contract-manifest",
        str(CONTRACT_MANIFEST),
        "--code-sha",
        CODE_SHA,
        "--output-json",
        str(checker_path),
    ]
    checked = subprocess.run(
        checker,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    checker_audit = json.loads(checker_path.read_text(encoding="utf-8"))
    assert checker_audit["valid"] is True
    assert checker_audit["errors"] == []


@pytest.mark.parametrize(
    "case_id",
    [
        "case.stationary_actor",
        "case.constant_horizontal_velocity",
        "case.first_frame_unit",
        "case.cross_temporal_unit_boundary",
        "case.missing_middle_frame",
        "case.invalid_previous_bbox",
        "case.invalid_current_bbox",
        "case.all_pairs_invalid",
        "case.exactly_one_valid_pair",
    ],
)
def test_phase1_golden_motion_cases(case_id: str) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    case = next(
        item for item in contract["golden_cases"] if item["case_id"] == case_id
    )
    input_rows = case["input_rows"]
    frames = [
        int(row.get("frame_index", position))
        for position, row in enumerate(input_rows)
    ]
    timestamps = [float(row["timestamp_sec"]) for row in input_rows]
    centers = [float(row["cx_n"]) for row in input_rows]
    units = [str(row["temporal_unit_key"]) for row in input_rows]
    actors = [str(row["object_track_key"]) for row in input_rows]
    geometry = [bool(row["geometry_valid"]) for row in input_rows]
    source = _native_input(
        frames=frames,
        timestamps=timestamps,
        centers=centers,
        units=units,
        bbox_valid=geometry,
        actor_keys=actors,
    )
    output = _add_temporal_deltas(source).sort_values(
        ["temporal_unit_key", "frame_index"],
        kind="mergesort",
    )

    expected_mask = case["expected_pair_masks"]["valid_motion_pair"]
    assert output["valid_motion_pair"].tolist() == expected_mask
    expected_speed = case["expected_numerical_values"].get(
        "speed_n_per_second"
    )
    if expected_speed is not None:
        actual = output["speed_n_per_second"].tolist()
        for observed, expected in zip(actual, expected_speed, strict=True):
            if expected is None:
                assert np.isnan(observed)
            else:
                assert observed == pytest.approx(float(expected))


def test_contract_declares_phase1_pair_scope() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    feature = next(
        item
        for item in contract["features"]
        if item["feature_id"] == "feature.valid_motion_pair"
    )
    assert feature["pair_reset_key"] == NATIVE_PAIR_SCOPE_KEY
    assert (
        feature["zero_value_semantics"]
        == "false means unavailable pair, not observed rest"
    )

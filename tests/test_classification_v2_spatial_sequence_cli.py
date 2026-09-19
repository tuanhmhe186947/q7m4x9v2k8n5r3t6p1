from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_REQUIRED_MASKS,
    MOTION_SCHEMA_DIMENSION,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_ID,
    MOTION_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
    SPATIAL_SCHEMA_HASH,
    SPATIAL_SCHEMA_TOTAL_DIMENSION,
    load_current_spatial_tensor_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = (
    REPO_ROOT
    / "scripts"
    / "classification_v2"
    / "02_train_ready_exports"
    / "classification_v2_export_spatial_sequences.py"
)
CANONICAL_PARTNER = (
    "source=cvat_tracking_xml|dataset=fixture|video=video-a|track_id=2"
)


def _window(*, source: str = "cvat") -> pd.DataFrame:
    track = (
        f"source={source}|dataset=fixture|video=video-a|track_id=1"
    )
    window_id = f"{track}|win=2|0-1"
    return pd.DataFrame(
        {
            "window_id": [window_id],
            "object_track_key": [track],
            "window_start_frame": [0],
            "window_end_frame": [1],
            "window_length_frames": [2],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": [window_id],
            "view_type": ["T2_contiguous"],
            "sampling_pattern": ["contiguous"],
            "selected_frame_offsets": ["[0,1]"],
            "selected_frame_indices": ["[0,1]"],
            "selected_timestamps_seconds": ["[0.0,1.0]"],
            "pair_delta_frames": ["[1]"],
            "pair_delta_seconds": ["[1.0]"],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )


def _frames(
    *,
    source: str = "cvat",
    social_available: bool = True,
    partner_key: object = CANONICAL_PARTNER,
) -> pd.DataFrame:
    track = (
        f"source={source}|dataset=fixture|video=video-a|track_id=1"
    )
    rows: dict[str, object] = {
        "object_track_key": [track, track],
        "frame_index": [0, 1],
        "timestamp_sec": [0.0, 1.0],
        "cx_n": [0.20, 0.30],
        "cy_n": [0.40, 0.40],
        "bw_n": [0.20, 0.20],
        "bh_n": [0.10, 0.10],
        "area_n": [0.02, 0.02],
        "aspect_ratio": [2.0, 2.0],
        "bbox_valid": [True, True],
        "actor_bbox_valid": [True, True],
        "geometry_feature_valid": [True, True],
        "spatiotemporal_feature_valid": [True, True],
        "roi_feeder_available": [False, False],
        "roi_drinker_available": [False, False],
        "roi_toy_available": [False, False],
        "social_neighbor_available": [
            social_available,
            social_available,
        ],
        "nearest_partner_key": [partner_key, partner_key],
        "velocity_sample_time_sec": [np.nan, 0.5],
        "acceleration_delta_t_sec": [np.nan, np.nan],
        "motion_schema_id": [MOTION_SCHEMA_ID, MOTION_SCHEMA_ID],
        "motion_schema_version": [
            MOTION_SCHEMA_VERSION,
            MOTION_SCHEMA_VERSION,
        ],
        "motion_schema_dimension": [
            MOTION_SCHEMA_DIMENSION,
            MOTION_SCHEMA_DIMENSION,
        ],
        "motion_schema_feature_names": [
            json.dumps(list(MOTION_FEATURE_NAMES), separators=(",", ":")),
            json.dumps(list(MOTION_FEATURE_NAMES), separators=(",", ":")),
        ],
        "motion_schema_hash": [MOTION_SCHEMA_HASH, MOTION_SCHEMA_HASH],
    }
    for group in SPATIAL_PREDICTIVE_GROUP_NAMES:
        for name in SPATIAL_PREDICTIVE_FEATURES[group]:
            rows.setdefault(name, [0.0, 0.0])
    rows["vx_n_per_second"] = [0.0, 0.1]
    rows["speed_n_per_second"] = [0.0, 0.1]
    mask_values = {
        "valid_motion_pair": [False, True],
        "velocity_valid": [False, True],
        "bbox_rate_valid": [False, True],
        "direction_valid": [False, True],
        "direction_change_valid": [False, False],
        "tangential_acceleration_valid": [False, False],
        "vector_acceleration_valid": [False, False],
        "motion_feature_available": [True, True],
    }
    assert set(mask_values) == set(MOTION_REQUIRED_MASKS)
    rows.update(mask_values)
    return pd.DataFrame(rows)


def _write_inputs(
    root: Path,
    *,
    source: str = "cvat",
    social_available: bool = True,
    partner_key: object = CANONICAL_PARTNER,
    drop_columns: tuple[str, ...] = (),
    legacy_identity_column: str | None = None,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    windows_path = root / "windows.csv"
    frames_path = root / "frames.csv"
    _window(source=source).to_csv(windows_path, index=False)
    frames = _frames(
        source=source,
        social_available=social_available,
        partner_key=partner_key,
    ).drop(columns=list(drop_columns))
    if legacy_identity_column is not None:
        frames[legacy_identity_column] = ["2", "2"]
    frames.to_csv(frames_path, index=False)
    return windows_path, frames_path


def _run_cli(
    root: Path,
    *,
    source: str = "cvat",
    social_available: bool = True,
    partner_key: object = CANONICAL_PARTNER,
    drop_columns: tuple[str, ...] = (),
    legacy_identity_column: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    windows_path, frames_path = _write_inputs(
        root / "inputs",
        source=source,
        social_available=social_available,
        partner_key=partner_key,
        drop_columns=drop_columns,
        legacy_identity_column=legacy_identity_column,
    )
    output_dir = root / "output"
    command = [
        sys.executable,
        str(CLI_PATH),
        "--window-manifest-csv",
        str(windows_path),
        "--frame-features-csv",
        str(frames_path),
        "--output-dir",
        str(output_dir),
        "--compress",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, output_dir


def _assert_no_promoted_artifact(output_dir: Path) -> None:
    assert not (output_dir / "X_spatial_sequences.npz").exists()
    assert not (output_dir / "spatial_sequence_audit.json").exists()


def _load_cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "classification_v2_spatial_cli",
        CLI_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_social_1_valid_canonical_partner_exports(tmp_path: Path) -> None:
    completed, output_dir = _run_cli(tmp_path)

    assert completed.returncode == 0, completed.stderr
    arrays, audit = load_current_spatial_tensor_bundle(
        output_dir / "X_spatial_sequences.npz",
        output_dir / "spatial_sequence_audit.json",
    )
    assert arrays["social_validity_mask"].tolist() == [[1.0, 1.0]]
    assert audit["canonical_social_identity_column_present"] is True


def test_cli_social_2_projection_omission_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frames_path = _write_inputs(tmp_path)
    module = _load_cli_module()
    real_read_csv = pd.read_csv

    def projected_read(*args: object, **kwargs: object) -> pd.DataFrame:
        result = real_read_csv(*args, **kwargs)
        if kwargs.get("usecols") is not None:
            return result.drop(columns=["nearest_partner_key"])
        return result

    monkeypatch.setattr(module.pd, "read_csv", projected_read)
    with pytest.raises(
        ValueError,
        match="Projected frame data omitted canonical social identity",
    ):
        module.read_current_frame_projection(frames_path)


def test_cli_social_3_structurally_absent_fails_closed(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run_cli(
        tmp_path,
        drop_columns=("nearest_partner_key",),
    )

    assert completed.returncode != 0
    assert "Missing canonical social identity column in frame source" in (
        completed.stderr
    )
    _assert_no_promoted_artifact(output_dir)


def test_cli_social_4_available_blank_identity_fails_closed(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run_cli(tmp_path, partner_key="")

    assert completed.returncode != 0
    assert "blank or invalid canonical partner identity" in completed.stderr
    _assert_no_promoted_artifact(output_dir)


def test_cli_social_5_unavailable_blank_preserves_fixed_width(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run_cli(
        tmp_path,
        social_available=False,
        partner_key="",
    )

    assert completed.returncode == 0, completed.stderr
    arrays, _ = load_current_spatial_tensor_bundle(
        output_dir / "X_spatial_sequences.npz",
        output_dir / "spatial_sequence_audit.json",
    )
    assert arrays["social_relation"].shape == (1, 2, 10)
    assert not arrays["social_validity_mask"].any()
    assert not arrays["social_relation"].any()


@pytest.mark.parametrize(
    "identity_column",
    ["nearest_pig_id", "nearest_track_id"],
    ids=["CLI_SOCIAL_6", "CLI_SOCIAL_7"],
)
def test_cli_social_6_7_unstable_identity_never_substitutes(
    tmp_path: Path,
    identity_column: str,
) -> None:
    completed, output_dir = _run_cli(
        tmp_path,
        drop_columns=("nearest_partner_key",),
        legacy_identity_column=identity_column,
    )

    assert completed.returncode != 0
    assert "Missing canonical social identity column" in completed.stderr
    _assert_no_promoted_artifact(output_dir)


def test_cli_social_8_cvat_current_unit_exports(tmp_path: Path) -> None:
    completed, _ = _run_cli(tmp_path, source="cvat_tracking_xml")
    assert completed.returncode == 0, completed.stderr


def test_cli_social_9_legacy_current_unit_exports(tmp_path: Path) -> None:
    completed, _ = _run_cli(tmp_path, source="legacy_recovered")
    assert completed.returncode == 0, completed.stderr


def test_cli_social_10_current_spatial_dimension_is_exact(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run_cli(tmp_path)
    assert completed.returncode == 0, completed.stderr
    arrays, _ = load_current_spatial_tensor_bundle(
        output_dir / "X_spatial_sequences.npz",
        output_dir / "spatial_sequence_audit.json",
    )
    derived_dimension = sum(
        arrays[group].shape[-1]
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
    )
    assert derived_dimension == SPATIAL_SCHEMA_TOTAL_DIMENSION == 46


def test_cli_social_11_sidecar_matches_canonical_schema(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run_cli(tmp_path)
    assert completed.returncode == 0, completed.stderr
    audit = json.loads(
        (output_dir / "spatial_sequence_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["feature_names"] == {
        group: list(SPATIAL_PREDICTIVE_FEATURES[group])
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
    }
    assert audit["spatial_schema"]["schema_hash"] == SPATIAL_SCHEMA_HASH


def test_cli_social_12_loader_accepts_generated_bundle(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run_cli(tmp_path)
    assert completed.returncode == 0, completed.stderr
    arrays, audit = load_current_spatial_tensor_bundle(
        output_dir / "X_spatial_sequences.npz",
        output_dir / "spatial_sequence_audit.json",
    )
    assert arrays["motion_delta"].shape[-1] == MOTION_SCHEMA_DIMENSION
    assert audit["spatial_schema"]["schema_hash"] == SPATIAL_SCHEMA_HASH


def test_cli_social_13_bounded_model_forward(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from pig_behavior.classification_v2.models.spatial_tcn import (
        SpatialTCNClassifier,
        SpatialTCNConfig,
    )

    completed, output_dir = _run_cli(tmp_path)
    assert completed.returncode == 0, completed.stderr
    arrays, _ = load_current_spatial_tensor_bundle(
        output_dir / "X_spatial_sequences.npz",
        output_dir / "spatial_sequence_audit.json",
    )
    model = SpatialTCNClassifier(
        SpatialTCNConfig(
            input_dims={
                group: len(SPATIAL_PREDICTIVE_FEATURES[group])
                for group in SPATIAL_PREDICTIVE_GROUP_NAMES
            },
            num_classes=10,
            hidden_dim=8,
            dropout=0.0,
        )
    )
    logits = model(
        {
            group: torch.from_numpy(arrays[group])
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        },
        length_mask=torch.from_numpy(arrays["length_mask"]),
        observed_mask=torch.from_numpy(arrays["observed_mask"]),
        feature_validity_masks={
            "motion_delta": torch.from_numpy(
                arrays["motion_feature_validity_mask"]
            ),
            "social_relation": torch.from_numpy(
                arrays["social_feature_validity_mask"]
            ),
        },
    )
    assert logits.shape == (1, 10)


def test_cli_social_14_repeated_exports_are_deterministic(
    tmp_path: Path,
) -> None:
    first, first_dir = _run_cli(tmp_path / "first")
    second, second_dir = _run_cli(tmp_path / "second")
    assert first.returncode == second.returncode == 0
    first_arrays, first_audit = load_current_spatial_tensor_bundle(
        first_dir / "X_spatial_sequences.npz",
        first_dir / "spatial_sequence_audit.json",
    )
    second_arrays, second_audit = load_current_spatial_tensor_bundle(
        second_dir / "X_spatial_sequences.npz",
        second_dir / "spatial_sequence_audit.json",
    )
    for name in first_arrays:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])
    assert (
        first_audit["spatial_tensor_content_hash"]
        == second_audit["spatial_tensor_content_hash"]
    )
    assert first_audit["feature_names"] == second_audit["feature_names"]


def test_cli_social_15_generic_acceleration_alias_absent(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run_cli(tmp_path)
    assert completed.returncode == 0, completed.stderr
    audit = json.loads(
        (output_dir / "spatial_sequence_audit.json").read_text(
            encoding="utf-8"
        )
    )
    flattened = {
        name
        for names in audit["feature_names"].values()
        for name in names
    }
    assert "acceleration_n_per_second2" not in flattened


def test_cli_social_16_numeric_zero_identity_fails_closed(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run_cli(tmp_path, partner_key=0)

    assert completed.returncode != 0
    assert "blank or invalid canonical partner identity" in completed.stderr
    _assert_no_promoted_artifact(output_dir)

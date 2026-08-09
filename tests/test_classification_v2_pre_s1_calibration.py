"""CPU-only contract tests for the dedicated PRE-S1 calibration executor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.models.balanced.contracts import (
    ModelBatch,
    SequenceSegment,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import pre_s1_calibration as calibration

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / (
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2/s1_control_and_pre_s1_calibration_authority.json"
)


def _rows(role: str, count: int, *, prefix: str) -> pd.DataFrame:
    records = []
    for index in range(count):
        label = VALID_BEHAVIORS[index % len(VALID_BEHAVIORS)]
        records.append(
            {
                "window_id": f"{prefix}-{index:03d}",
                "behavior_window_label": label,
                "temporal_unit_keys_json": json.dumps([f"native-{prefix}-{index:03d}"]),
                "primary_s1_role": role,
                "primary_s1_eligible": True,
                "event_sample_weight": 1.0,
            }
        )
    return pd.DataFrame(records)


def _population() -> calibration.CalibrationPopulation:
    train = _rows("train", 16, prefix="train")
    validation = _rows("validation", 3, prefix="valid")
    expected = pd.DataFrame(
        {
            "temporal_unit_key": [f"native-valid-{index:03d}" for index in range(3)],
            "behavior_label": validation["behavior_window_label"].tolist(),
        }
    )

    def load_batch(rows: pd.DataFrame, device: torch.device) -> ModelBatch:
        images = torch.zeros((len(rows), 6, 3, 64, 64), device=device)
        for index in range(len(rows)):
            images[index].fill_(float(index + 1) / 32.0)
        labels = torch.tensor(
            [VALID_BEHAVIORS.index(value) for value in rows["behavior_window_label"]],
            dtype=torch.long,
            device=device,
        )
        return ModelBatch(
            target=SequenceSegment(
                valid_mask=torch.ones((len(rows), 6), device=device),
                frame_offsets=torch.arange(-5, 1, device=device).repeat(len(rows), 1),
                images=images,
            ),
            labels=labels,
            native_unit_id=rows["temporal_unit_keys_json"].astype(str).tolist(),
            window_id=rows["window_id"].astype(str).tolist(),
        )

    return calibration.CalibrationPopulation(
        train=train,
        validation=validation,
        expected_native_units=expected,
        load_batch=load_batch,
        close=lambda: None,
        data_hashes={"synthetic_inner_fixture": "a" * 64},
    )


def _plan(tmp_path: Path, name: str, *, existing: bool = False) -> calibration.CalibrationPlan:
    return calibration.create_calibration_plan(
        AUTHORITY,
        repository_root=ROOT,
        output_dir=tmp_path / name,
        run_id=name,
        device_name="cpu",
        engineering_smoke=True,
        allow_existing_output=existing,
    )


def test_cpu_engineering_smoke_executes_b1_checkpoint_predictions_and_native_collapse(
    tmp_path: Path,
) -> None:
    report = calibration.run_pre_s1_calibration(_plan(tmp_path, "smoke"), _population())

    assert report["status"] == "PASS"
    assert report["completed_steps"] == 2
    assert report["snapshots"][0]["native_prediction_coverage"]["valid"] is True
    assert report["snapshots"][0]["composite_key_primary_path_used"] is False
    assert report["telemetry"]["gpu_count"] == 0
    assert (tmp_path / "smoke" / "checkpoints" / "step_000002.pt").is_file()
    assert (tmp_path / "smoke" / "predictions" / "step_000002_native.csv").is_file()


def test_engineering_resume_is_deterministic_and_refuses_changed_fingerprint(
    tmp_path: Path,
) -> None:
    initial = _plan(tmp_path, "resume")
    interrupted = calibration.run_pre_s1_calibration(
        initial,
        _population(),
        stop_after_steps=1,
    )
    checkpoint = Path(interrupted["checkpoint"])
    resumed = calibration.run_pre_s1_calibration(
        _plan(tmp_path, "resume", existing=True),
        _population(),
        resume_checkpoint=checkpoint,
    )
    fresh = calibration.run_pre_s1_calibration(_plan(tmp_path, "fresh"), _population())

    assert resumed["losses"] == pytest.approx(fresh["losses"], abs=1e-7)
    assert (
        resumed["snapshots"][0]["native_prediction_sha256"]
        == fresh["snapshots"][0]["native_prediction_sha256"]
    )
    changed = _plan(tmp_path, "mismatch", existing=True)
    with pytest.raises(calibration.PreS1CalibrationError, match="fingerprint"):
        model = calibration._build_b1_model()
        calibration._load_checkpoint(
            checkpoint,
            changed,
            model,
            torch.optim.AdamW(model.parameters()),
            _population(),
        )


@pytest.mark.parametrize("role", ["test", "outer", "q2_outer_00"])
def test_outer_roles_are_refused_before_any_payload_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    opened = False

    def sentinel(*_args: object, **_kwargs: object) -> None:
        nonlocal opened
        opened = True
        raise AssertionError("payload loader must not be constructed")

    monkeypatch.setattr(calibration, "ClassificationV2ImageSequenceDataset", sentinel)
    with pytest.raises(calibration.PreS1CalibrationError, match="outer/test"):
        calibration._assert_permitted_scope(role)
    with pytest.raises(calibration.PreS1CalibrationError, match="outer/test prediction"):
        calibration.create_calibration_plan(
            AUTHORITY,
            repository_root=ROOT,
            output_dir=tmp_path / f"{role}_output",
            run_id="safe_output",
            device_name="cpu",
            engineering_smoke=True,
        )
    assert opened is False


@pytest.mark.parametrize(
    "field",
    ["optimizer", "learning_rate", "batch_size", "precision"],
)
def test_outer_export_and_frozen_override_are_refused_before_data_open(
    tmp_path: Path,
    field: str,
) -> None:
    with pytest.raises(calibration.PreS1CalibrationError, match="outer/test prediction"):
        calibration.create_calibration_plan(
            AUTHORITY,
            repository_root=ROOT,
            output_dir=tmp_path / "outer_predictions",
            run_id="safe_output",
            device_name="cpu",
            engineering_smoke=True,
        )
    with pytest.raises(calibration.PreS1CalibrationError, match="frozen calibration"):
        calibration.create_calibration_plan(
            AUTHORITY,
            repository_root=ROOT,
            output_dir=tmp_path / "override",
            run_id="override",
            device_name="cpu",
            engineering_smoke=True,
            frozen_overrides={field: "changed"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 1),
        ("max_steps", 1),
        ("event_snapshots_at_steps", [1]),
    ],
)
def test_authority_drift_fails_closed_before_payload_open(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    payload["pre_s1_calibration"][field] = value
    altered = tmp_path / f"altered_{field}.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(calibration.PreS1CalibrationError):
        calibration.create_calibration_plan(
            altered,
            repository_root=ROOT,
            output_dir=tmp_path / f"out_{field}",
            run_id=f"out_{field}",
            device_name="cpu",
            engineering_smoke=True,
        )


def test_hash_and_population_helpers_fail_closed() -> None:
    with pytest.raises(calibration.PreS1CalibrationError, match="hash mismatch"):
        calibration._verify_artifact(AUTHORITY, "0" * 64, "authority")
    mixed = _rows("train", 16, prefix="mixed")
    mixed["primary_s1_eligibility_status"] = "MIXED_LABEL"
    assert mixed["primary_s1_eligibility_status"].eq("MIXED_LABEL").all()


def test_primary_evaluator_rejects_composite_direct_missing_and_duplicate_native_predictions(
) -> None:
    validation = _rows("validation", 2, prefix="eval")
    expected = pd.DataFrame(
        {
            "temporal_unit_key": ["native-eval-000", "native-eval-001"],
            "behavior_label": ["drink", "eat"],
        }
    )
    direct = pd.DataFrame(
        {
            "window_id": ["eval-000", "eval-001"],
            "temporal_unit_key": ["x", "y"],
            "y_pred": ["drink", "eat"],
            "confidence": [1.0, 1.0],
        }
    )
    with pytest.raises(Exception, match="direct temporal_unit_key"):
        calibration.evaluate_primary_s1_validation(direct, validation, expected)
    missing = pd.DataFrame({"window_id": ["eval-000"], "y_pred": ["drink"], "confidence": [1.0]})
    with pytest.raises(Exception, match="coverage mismatch"):
        calibration.evaluate_primary_s1_validation(missing, validation, expected)
    duplicate = pd.DataFrame(
        {
            "window_id": ["eval-000", "eval-000"],
            "y_pred": ["drink", "drink"],
            "confidence": [1.0, 1.0],
        }
    )
    with pytest.raises(Exception, match="coverage mismatch"):
        calibration.evaluate_primary_s1_validation(duplicate, validation, expected)

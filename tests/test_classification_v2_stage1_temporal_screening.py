"""CPU-only contracts for the isolated Stage-1 temporal-screening executor."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.models.balanced.contracts import (
    ModelBatch,
    SequenceSegment,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import stage1_temporal_screening as stage1

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
                "window_id": f"{prefix}-{index:04d}",
                "behavior_window_label": label,
                "temporal_unit_keys_json": json.dumps([f"native-{prefix}-{index:04d}"]),
                "primary_s1_role": role,
                "primary_s1_eligible": True,
                "primary_s1_eligibility_status": "VALID_SINGLE_LABEL",
                "event_sample_weight": 1.0,
            }
        )
    return pd.DataFrame(records)


def _population(view: str = "T6") -> stage1.Stage1Population:
    length = int(stage1.VIEW_SPECS[view]["length"])
    train = _rows("train", 16, prefix=f"{view}-train")
    validation = _rows("validation", 3, prefix=f"{view}-validation")
    expected = pd.DataFrame(
        {
            "temporal_unit_key": [f"native-{view}-validation-{index:04d}" for index in range(3)],
            "behavior_label": validation["behavior_window_label"].tolist(),
        }
    )
    common = pd.concat(
        [
            pd.DataFrame(
                {
                    "temporal_unit_key": [
                        f"native-{view}-train-{index:04d}" for index in range(16)
                    ],
                    "role": "train",
                    "behavior_label": train["behavior_window_label"].tolist(),
                }
            ),
            pd.DataFrame(
                {
                    "temporal_unit_key": [
                        f"native-{view}-validation-{index:04d}" for index in range(3)
                    ],
                    "role": "validation",
                    "behavior_label": validation["behavior_window_label"].tolist(),
                }
            ),
        ],
        ignore_index=True,
    )

    def load_batch(rows: pd.DataFrame, device: torch.device) -> ModelBatch:
        images = torch.zeros((len(rows), length, 3, 64, 64), device=device)
        for index in range(len(rows)):
            images[index].fill_(float(index + 1) / 32.0)
        labels = torch.tensor(
            [VALID_BEHAVIORS.index(value) for value in rows["behavior_window_label"]],
            dtype=torch.long,
            device=device,
        )
        return ModelBatch(
            target=SequenceSegment(
                valid_mask=torch.ones((len(rows), length), device=device),
                frame_offsets=torch.arange(-(length - 1), 1, device=device).repeat(
                    len(rows),
                    1,
                ),
                images=images,
            ),
            labels=labels,
            native_unit_id=rows["temporal_unit_keys_json"].astype(str).tolist(),
            window_id=rows["window_id"].astype(str).tolist(),
        )

    return stage1.Stage1Population(
        train=train,
        validation=validation,
        expected_native_units=expected,
        common_cohort_native_units=common,
        load_batch=load_batch,
        close=lambda: None,
        data_hashes={"synthetic_inner_fixture": "a" * 64},
    )


def _plan(
    tmp_path: Path,
    name: str,
    *,
    view: str = "T6",
    existing: bool = False,
) -> stage1.Stage1Plan:
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    return stage1.create_stage1_plan(
        AUTHORITY,
        view=view,
        repository_root=ROOT,
        outputs_root=outputs,
        output_dir=tmp_path / name,
        trial_id=name,
        device_name="cpu",
        engineering_smoke=True,
        allow_existing_output=existing,
    )


@pytest.mark.parametrize("view", ["T6", "T8", "T12", "T16"])
def test_stage1_b1_contract_admits_exact_registered_target_lengths(view: str) -> None:
    model = stage1._build_b1_model(view)
    assert model.config.batch_contract.target_length == stage1.VIEW_SPECS[view]["length"]
    assert stage1._b1_effective_inputs(view)[0] == f"actor_rgb_{view}"


@pytest.mark.parametrize("view", ["T6", "T8", "T12", "T16"])
def test_cpu_engineering_smoke_writes_primary_and_common_cohort_artifacts(
    tmp_path: Path,
    view: str,
) -> None:
    report = stage1.run_stage1_temporal_screening(
        _plan(tmp_path, "stage1_smoke", view=view),
        _population(view),
    )
    snapshot = report["snapshots"][0]
    assert report["status"] == "PASS"
    assert report["completed_steps"] == 2
    assert snapshot["native_prediction_coverage"]["valid"] is True
    assert snapshot["common_cohort"]["prediction_coverage"]["valid"] is True
    assert snapshot["composite_key_primary_path_used"] is False
    assert report["telemetry"]["gpu_count"] == 0
    assert (tmp_path / "stage1_smoke" / "checkpoints" / "step_000002.pt").is_file()
    assert (tmp_path / "stage1_smoke" / "predictions" / "step_000002_common_native.csv").is_file()


def test_engineering_resume_is_deterministic_and_refuses_changed_fingerprint(
    tmp_path: Path,
) -> None:
    initial = _plan(tmp_path, "stage1_resume")
    interrupted = stage1.run_stage1_temporal_screening(
        initial,
        _population(),
        stop_after_steps=1,
    )
    checkpoint = Path(interrupted["checkpoint"])
    resumed = stage1.run_stage1_temporal_screening(
        _plan(tmp_path, "stage1_resume", existing=True),
        _population(),
        resume_checkpoint=checkpoint,
    )
    fresh = stage1.run_stage1_temporal_screening(
        _plan(tmp_path, "stage1_fresh"),
        _population(),
    )
    assert resumed["losses"] == pytest.approx(fresh["losses"], abs=1e-7)
    assert (
        resumed["snapshots"][0]["validation_native_prediction_sha256"]
        == fresh["snapshots"][0]["validation_native_prediction_sha256"]
    )
    changed = _plan(tmp_path, "stage1_mismatch", existing=True)
    with pytest.raises(stage1.Stage1TemporalScreeningError, match="fingerprint"):
        model = stage1._build_b1_model("T6")
        stage1._load_checkpoint(
            checkpoint,
            changed,
            model,
            torch.optim.AdamW(model.parameters()),
            _population("T8"),
        )


@pytest.mark.parametrize("role", ["test", "outer", "q2_outer_00"])
def test_outer_roles_and_output_roots_are_refused_before_payload_load(
    tmp_path: Path,
    role: str,
) -> None:
    with pytest.raises(stage1.Stage1TemporalScreeningError, match="outer/test"):
        stage1._assert_permitted_scope(role)
    with pytest.raises(stage1.Stage1TemporalScreeningError, match="outer/test prediction"):
        stage1.create_stage1_plan(
            AUTHORITY,
            view="T6",
            repository_root=ROOT,
            outputs_root=tmp_path,
            output_dir=tmp_path / "outer_predictions",
            trial_id="safe_trial",
            device_name="cpu",
            engineering_smoke=True,
        )


def test_stage1_authority_drift_and_frozen_override_fail_before_data_access(
    tmp_path: Path,
) -> None:
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    payload["stage_1_temporal_screening"]["max_steps"] = 1
    altered = tmp_path / "altered_stage1.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(stage1.Stage1TemporalScreeningError):
        stage1.create_stage1_plan(
            altered,
            view="T6",
            repository_root=ROOT,
            outputs_root=tmp_path,
            output_dir=tmp_path / "authority_drift",
            trial_id="authority_drift",
            device_name="cpu",
            engineering_smoke=True,
        )
    with pytest.raises(stage1.Stage1TemporalScreeningError, match="frozen Stage-1"):
        stage1.create_stage1_plan(
            AUTHORITY,
            view="T6",
            repository_root=ROOT,
            outputs_root=tmp_path,
            output_dir=tmp_path / "override",
            trial_id="override",
            device_name="cpu",
            engineering_smoke=True,
            frozen_overrides={"max_steps": 1},
        )


def test_real_cuda_execution_remains_refused_before_data_access(tmp_path: Path) -> None:
    with pytest.raises(stage1.Stage1TemporalScreeningError, match="not authorized"):
        stage1.create_stage1_plan(
            AUTHORITY,
            view="T6",
            repository_root=ROOT,
            outputs_root=tmp_path,
            output_dir=tmp_path / "s1_stage1_t6_seed20260804",
            trial_id="s1_stage1_t6_seed20260804",
            device_name="cuda",
            engineering_smoke=False,
        )


def test_real_data_cpu_preflight_and_one_step_smoke_remain_cpu_only(
    tmp_path: Path,
) -> None:
    base = _population("T6")
    train = _rows(
        "train",
        int(stage1.VIEW_SPECS["T6"]["train_windows"]),
        prefix="preflight-train",
    )
    validation = _rows(
        "validation",
        int(stage1.VIEW_SPECS["T6"]["validation_windows"]),
        prefix="preflight-validation",
    )
    population = stage1.Stage1Population(
        train=train,
        validation=validation,
        expected_native_units=pd.DataFrame(),
        common_cohort_native_units=pd.DataFrame(),
        load_batch=base.load_batch,
        close=lambda: None,
        data_hashes={"synthetic_real_data_preflight": "a" * 64},
        binding_audit={
            "coverage": {
                "train_windows_bound": int(stage1.VIEW_SPECS["T6"]["train_windows"]),
                "validation_windows_bound": int(
                    stage1.VIEW_SPECS["T6"]["validation_windows"]
                ),
                "missing_windows": 0,
                "duplicate_windows": 0,
                "bad_sequence_length": 0,
                "role_violations": 0,
                "cross_video_violations": 0,
            }
        },
        image_load_audit=lambda: {
            "source_image_loads": 0,
            "packed_image_cache_hits": 24,
        },
    )
    plan = replace(_plan(tmp_path, "real_data_preflight"), engineering_smoke=False)
    report = stage1.run_real_data_cpu_preflight(plan, population, sample_size=2)
    assert report["status"] == "PASS"
    assert report["outer_windows_loaded"] == 0
    assert report["real_rgb_decode_sample"]["windows"] == 4
    assert report["b1_effective_inputs"][0] == "actor_rgb_T6"
    smoke = stage1.run_real_data_cpu_engineering_smoke(plan, _population(), steps=1)
    assert smoke["status"] == "PASS"
    assert smoke["gpu_used"] is False

"""Fail-closed tests for external, single-use Stage-1 L4 permits."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pig_behavior.classification_v2.training import (
    stage1_execution_authorization as authorization,
)
from pig_behavior.classification_v2.training import (
    stage1_temporal_screening as stage1,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / (
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2/s1_control_and_pre_s1_calibration_authority.json"
)


def _authority() -> tuple[dict[str, object], str]:
    return json.loads(AUTHORITY.read_text(encoding="utf-8")), sha256(
        AUTHORITY.read_bytes()
    ).hexdigest()


def _binding_bundle(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    authority, authority_sha256 = _authority()
    scientific = {
        view: sha256(f"scientific-{view}".encode()).hexdigest()
        for view in authorization.VIEWS
    }
    payload = {
        "schema_version": authorization.BINDING_BUNDLE_SCHEMA,
        "authority": {"sha256": authority_sha256},
        "views": {
            view: {
                "scientific_binding_sha256": scientific[view],
                "provenance_hashes": {
                    "event_weight": authority["derived_population"]["per_view"][view][
                        "event_weight_artifact"
                    ]["sha256"],
                },
            }
            for view in authorization.VIEWS
        },
    }
    path = tmp_path / "stage1_temporal_rgb_bindings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, scientific


def _data_bindings(tmp_path: Path, scientific_sha256: str) -> Path:
    path = tmp_path / "stage1_temporal_data_bindings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": authorization.DATA_BINDINGS_SCHEMA,
                "scientific_binding": {"sha256": scientific_sha256},
            }
        ),
        encoding="utf-8",
    )
    return path


def _permits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    Path,
    Path,
    dict[str, authorization.Stage1ExecutionPermit],
    dict[str, str],
]:
    bundle, scientific = _binding_bundle(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(authorization, "_git_status", lambda _root: "")
    permits = authorization.create_stage1_execution_permits(
        repository_root=ROOT,
        outputs_root=outputs,
        authority_path=AUTHORITY,
        binding_bundle_path=bundle,
    )
    return outputs, bundle, permits, scientific


def _validate_t6(
    *,
    outputs: Path,
    permit_path: Path,
    data_bindings: Path,
    binding_bundle: Path,
    allow_consumed: bool = False,
) -> authorization.Stage1ExecutionPermit:
    authority, authority_sha256 = _authority()
    return authorization.validate_stage1_execution_permit(
        permit_path=permit_path,
        repository_root=ROOT,
        outputs_root=outputs,
        authority_path=AUTHORITY,
        authority=authority,
        authority_sha256=authority_sha256,
        view="T6",
        trial_id=authorization.canonical_trial_id("T6"),
        data_bindings_path=data_bindings,
        binding_bundle_path=binding_bundle,
        allow_consumed=allow_consumed,
    )


def test_creator_issues_four_exact_per_arm_permits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, _bundle, permits, scientific = _permits(tmp_path, monkeypatch)

    assert set(permits) == set(authorization.VIEWS)
    for view, permit in permits.items():
        assert permit.path.parent == authorization.permit_directory(outputs)
        assert permit.payload["status"] == "AUTHORIZED"
        assert permit.payload["trial_id"] == authorization.canonical_trial_id(view)
        assert permit.payload["scientific_rgb_binding_sha256"] == scientific[view]
        assert permit.payload["hardware"] == {"gpu_count": 1, "gpu_name": "NVIDIA L4"}
        assert permit.payload["max_steps"] == 4164
        assert permit.payload["outer_access_allowed"] is False


def test_code_stale_permit_rotation_preserves_bytes_and_selects_one_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, bundle, permits, scientific = _permits(tmp_path, monkeypatch)
    previous = permits["T8"]
    previous_bytes = previous.path.read_bytes()
    monkeypatch.setattr(authorization, "_git_sha", lambda _root: "b" * 40)

    rotations = authorization.rotate_stage1_execution_permits(
        repository_root=ROOT,
        outputs_root=outputs,
        authority_path=AUTHORITY,
        binding_bundle_path=bundle,
        views=("T8",),
        reason="code-only executor repair",
    )

    assert set(rotations) == {"T8"}
    rotation = rotations["T8"]
    assert previous.path.read_bytes() != previous_bytes
    assert rotation.superseded_path.read_bytes() == previous_bytes
    assert rotation.previous.sha256 == sha256(previous_bytes).hexdigest()
    replacement = rotation.replacement
    assert replacement.path == previous.path
    assert replacement.payload["code_sha"] == "b" * 40
    assert replacement.payload["view"] == "T8"
    assert permits["T6"].path.is_file()
    assert permits["T12"].path.is_file()
    assert permits["T16"].path.is_file()
    record = json.loads(rotation.record_path.read_text(encoding="utf-8"))
    assert record["previous_permit_sha256"] == rotation.previous.sha256
    assert record["replacement_permit_sha256"] == replacement.sha256
    assert record["reason"] == "code-only executor repair"
    assert record["code_sha_changed"] is True
    assert "max_steps" in record["frozen_fields_verified"]

    data_bindings = _data_bindings(tmp_path, scientific["T8"])
    verified = authorization.validate_stage1_execution_permit(
        permit_path=replacement.path,
        repository_root=ROOT,
        outputs_root=outputs,
        authority_path=AUTHORITY,
        authority=_authority()[0],
        authority_sha256=_authority()[1],
        view="T8",
        trial_id=authorization.canonical_trial_id("T8"),
        data_bindings_path=data_bindings,
        binding_bundle_path=bundle,
    )
    assert verified.permit_id == replacement.permit_id
    with pytest.raises(authorization.Stage1ExecutionAuthorizationError, match="path"):
        authorization.validate_stage1_execution_permit(
            permit_path=rotation.superseded_path,
            repository_root=ROOT,
            outputs_root=outputs,
            authority_path=AUTHORITY,
            authority=_authority()[0],
            authority_sha256=_authority()[1],
            view="T8",
            trial_id=authorization.canonical_trial_id("T8"),
            data_bindings_path=data_bindings,
            binding_bundle_path=bundle,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("status", "CONSUMED", "not an active permit"),
        ("expires_at_utc", "2000-01-01T00:00:00+00:00", "expired"),
        ("permit_id", "not-a-valid-permit", "permit ID is malformed"),
    ],
)
def test_rotation_rejects_ineligible_predecessors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    match: str,
) -> None:
    outputs, bundle, permits, _scientific = _permits(tmp_path, monkeypatch)
    payload = json.loads(permits["T8"].path.read_text(encoding="utf-8"))
    payload[field] = value
    permits["T8"].path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(authorization, "_git_sha", lambda _root: "b" * 40)

    with pytest.raises(authorization.Stage1ExecutionAuthorizationError, match=match):
        authorization.rotate_stage1_execution_permits(
            repository_root=ROOT,
            outputs_root=outputs,
            authority_path=AUTHORITY,
            binding_bundle_path=bundle,
            views=("T8",),
            reason="code-only executor repair",
        )


def test_rotation_rejects_non_code_frozen_field_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, bundle, permits, _scientific = _permits(tmp_path, monkeypatch)
    payload = json.loads(permits["T8"].path.read_text(encoding="utf-8"))
    payload["max_steps"] = 1
    permits["T8"].path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(authorization, "_git_sha", lambda _root: "b" * 40)

    with pytest.raises(
        authorization.Stage1ExecutionAuthorizationError,
        match="frozen field mismatch=max_steps",
    ):
        authorization.rotate_stage1_execution_permits(
            repository_root=ROOT,
            outputs_root=outputs,
            authority_path=AUTHORITY,
            binding_bundle_path=bundle,
            views=("T8",),
            reason="code-only executor repair",
        )


def test_rotation_rejects_a_permit_already_bound_to_current_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, bundle, _permits_by_view, _scientific = _permits(tmp_path, monkeypatch)

    with pytest.raises(
        authorization.Stage1ExecutionAuthorizationError,
        match="already binds current code",
    ):
        authorization.rotate_stage1_execution_permits(
            repository_root=ROOT,
            outputs_root=outputs,
            authority_path=AUTHORITY,
            binding_bundle_path=bundle,
            views=("T8",),
            reason="code-only executor repair",
        )


def test_creator_refuses_a_dirty_permit_issuer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ = _binding_bundle(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(authorization, "_git_status", lambda _root: " M dirty.py")

    with pytest.raises(authorization.Stage1ExecutionAuthorizationError, match="clean"):
        authorization.create_stage1_execution_permits(
            repository_root=ROOT,
            outputs_root=outputs,
            authority_path=AUTHORITY,
            binding_bundle_path=bundle,
        )


def test_creator_preserves_false_canonical_gpu_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _ = _authority()
    payload["stage_1_temporal_screening"]["gpu_execution_authorized"] = True
    altered = tmp_path / "altered_authority.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    bundle, _ = _binding_bundle(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(authorization, "_git_status", lambda _root: "")

    with pytest.raises(
        authorization.Stage1ExecutionAuthorizationError,
        match="must remain false",
    ):
        authorization.create_stage1_execution_permits(
            repository_root=ROOT,
            outputs_root=outputs,
            authority_path=altered,
            binding_bundle_path=bundle,
        )


def test_permit_binds_science_code_and_one_time_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, bundle, permits, scientific = _permits(tmp_path, monkeypatch)
    data_bindings = _data_bindings(tmp_path, scientific["T6"])
    verified = _validate_t6(
        outputs=outputs,
        permit_path=permits["T6"].path,
        data_bindings=data_bindings,
        binding_bundle=bundle,
    )

    consumed = authorization.consume_stage1_execution_permit(
        verified,
        allow_consumed_resume=False,
    )
    assert consumed.status == "CONSUMED"
    assert not permits["T6"].path.exists()
    assert consumed.path.is_file()
    resumed = _validate_t6(
        outputs=outputs,
        permit_path=consumed.path,
        data_bindings=data_bindings,
        binding_bundle=bundle,
        allow_consumed=True,
    )
    assert authorization.consume_stage1_execution_permit(
        resumed,
        allow_consumed_resume=True,
    ).permit_id == consumed.permit_id
    with pytest.raises(authorization.Stage1ExecutionAuthorizationError, match="consumed"):
        authorization.consume_stage1_execution_permit(
            resumed,
            allow_consumed_resume=False,
        )


def test_permit_rejects_rgb_and_status_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, bundle, permits, scientific = _permits(tmp_path, monkeypatch)
    mismatched = _data_bindings(tmp_path, scientific["T8"])

    with pytest.raises(authorization.Stage1ExecutionAuthorizationError, match="scientific"):
        _validate_t6(
            outputs=outputs,
            permit_path=permits["T6"].path,
            data_bindings=mismatched,
            binding_bundle=bundle,
        )

    payload = json.loads(permits["T6"].path.read_text(encoding="utf-8"))
    payload["code_sha"] = "0" * 40
    permits["T6"].path.write_text(json.dumps(payload), encoding="utf-8")
    valid = _data_bindings(tmp_path, scientific["T6"])
    with pytest.raises(authorization.Stage1ExecutionAuthorizationError, match="code_sha"):
        _validate_t6(
            outputs=outputs,
            permit_path=permits["T6"].path,
            data_bindings=valid,
            binding_bundle=bundle,
        )

    payload["code_sha"] = permits["T6"].payload["code_sha"]
    payload["expires_at_utc"] = "2000-01-01T00:00:00+00:00"
    permits["T6"].path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(authorization.Stage1ExecutionAuthorizationError, match="expired"):
        _validate_t6(
            outputs=outputs,
            permit_path=permits["T6"].path,
            data_bindings=valid,
            binding_bundle=bundle,
        )

    payload["expires_at_utc"] = permits["T6"].payload["expires_at_utc"]
    payload["status"] = "REVOKED"
    permits["T6"].path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(authorization.Stage1ExecutionAuthorizationError, match="available"):
        _validate_t6(
            outputs=outputs,
            permit_path=permits["T6"].path,
            data_bindings=valid,
            binding_bundle=bundle,
        )


def test_creator_refuses_frozen_control_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _ = _authority()
    payload["fixed_stage_1_to_4_controls"]["learning_rate"] = 0.004
    altered = tmp_path / "altered_authority.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    bundle, _ = _binding_bundle(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(authorization, "_git_status", lambda _root: "")

    with pytest.raises(
        authorization.Stage1ExecutionAuthorizationError,
        match="fixed Stage-1 controls drifted",
    ):
        authorization.create_stage1_execution_permits(
            repository_root=ROOT,
            outputs_root=outputs,
            authority_path=altered,
            binding_bundle_path=bundle,
        )


def test_permit_rejects_binding_bundle_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, bundle, permits, scientific = _permits(tmp_path, monkeypatch)
    data_bindings = _data_bindings(tmp_path, scientific["T6"])
    bundle.write_text("{}", encoding="utf-8")

    with pytest.raises(
        authorization.Stage1ExecutionAuthorizationError,
        match="binding bundle hash changed",
    ):
        _validate_t6(
            outputs=outputs,
            permit_path=permits["T6"].path,
            data_bindings=data_bindings,
            binding_bundle=bundle,
        )


def test_consumed_permit_requires_checkpoint_proof_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs, bundle, permits, scientific = _permits(tmp_path, monkeypatch)
    data_bindings = _data_bindings(tmp_path, scientific["T6"])
    verified = _validate_t6(
        outputs=outputs,
        permit_path=permits["T6"].path,
        data_bindings=data_bindings,
        binding_bundle=bundle,
    )
    consumed = authorization.consume_stage1_execution_permit(
        verified,
        allow_consumed_resume=False,
    )
    output_dir = (
        outputs
        / "classification_v2"
        / "s1_post_temporal_closure_20260809"
        / "s1_trials"
        / authorization.canonical_trial_id("T6")
    )
    (output_dir / "manifest").mkdir(parents=True)
    plan = stage1.create_stage1_plan(
        AUTHORITY,
        view="T6",
        repository_root=ROOT,
        outputs_root=outputs,
        output_dir=output_dir,
        trial_id=authorization.canonical_trial_id("T6"),
        device_name="cuda",
        data_bindings_path=data_bindings,
        execution_permit_path=consumed.path,
        binding_bundle_path=bundle,
        allow_consumed_execution_permit=True,
        allow_existing_output=True,
    )
    plan = replace(plan, device_name="cpu")
    consumed_calls: list[str] = []
    monkeypatch.setattr(stage1, "_assert_execution_hardware", lambda _plan: None)
    monkeypatch.setattr(stage1, "_build_b1_model", lambda _view: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        stage1,
        "_load_checkpoint",
        lambda *_args: (_ for _ in ()).throw(
            stage1.Stage1TemporalScreeningError("RESUME_REFUSED=YES fingerprint mismatch")
        ),
    )
    monkeypatch.setattr(
        stage1,
        "consume_stage1_execution_permit",
        lambda *_args, **_kwargs: consumed_calls.append("called"),
    )
    population = SimpleNamespace(data_hashes={}, common_cohort_native_units=[])

    with pytest.raises(stage1.Stage1TemporalScreeningError, match="fingerprint mismatch"):
        stage1.run_stage1_temporal_screening(
            plan,
            population,
            resume_checkpoint=tmp_path / "bad.pt",
        )
    assert consumed_calls == []


def test_cuda_plan_requires_bound_permit_before_data_access(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    with pytest.raises(stage1.Stage1TemporalScreeningError, match="bound execution permit"):
        stage1.create_stage1_plan(
            AUTHORITY,
            view="T6",
            repository_root=ROOT,
            outputs_root=outputs,
            trial_id=authorization.canonical_trial_id("T6"),
            device_name="cuda",
            engineering_smoke=False,
        )

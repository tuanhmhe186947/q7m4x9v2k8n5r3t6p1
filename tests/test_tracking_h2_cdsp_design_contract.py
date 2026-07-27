"""Focused tests for the deterministic H2-CDSP design checker."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _checker_module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "tracking"
        / "check_h2_cdsp_design_contract.py"
    )
    spec = importlib.util.spec_from_file_location("h2_cdsp_checker", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_frozen_contract_passes_pre_review() -> None:
    module = _checker_module()

    counts = module.check_contract(pre_review=True)

    assert counts["usable_short"] >= 4
    assert counts["unsafe"] >= 4
    assert counts["positive_events"] == 6
    assert counts["control_windows"] == 4


def test_validation_output_guard_distinguishes_cache_validation() -> None:
    module = _checker_module()

    assert not module._is_h2_validation_output_path(
        Path("h2_cdsp_current_main/H2_CDSP_LIVE_MAIN_CACHE_VALIDATION.json")
    )
    assert module._is_h2_validation_output_path(
        Path("h2_cdsp_validation/H2_CDSP_VALIDATION_OUTPUT.json")
    )


def test_checker_rejects_detection_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.MACHINE)
    payload["global_invariants"]["reserves_detection"] = True
    invalid = tmp_path / "machine.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "MACHINE", invalid)

    with pytest.raises(module.ContractError, match="intervention boundary"):
        module._check_machine()


def test_checker_rejects_direct_assignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.MACHINE)
    payload["global_invariants"]["directly_assigns_detection"] = True
    invalid = tmp_path / "machine.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "MACHINE", invalid)

    with pytest.raises(module.ContractError, match="intervention boundary"):
        module._check_machine()


def test_checker_rejects_unbounded_preservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.MACHINE)
    stale = next(
        row for row in payload["states"] if row["name"] == "STALE_PRESERVED"
    )
    stale["maximum_duration_frames"] = None
    invalid = tmp_path / "machine.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "MACHINE", invalid)

    with pytest.raises(module.ContractError, match="finite preservation"):
        module._check_machine()


def test_checker_rejects_terminal_resurrection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.MACHINE)
    terminal = next(
        row
        for row in payload["transitions"]
        if row["transition_id"] == "T13_TERMINAL_ABSORBING"
    )
    terminal["to"] = "VISIBLE_CONFIRMED"
    invalid = tmp_path / "machine.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "MACHINE", invalid)

    with pytest.raises(module.ContractError, match="not absorbing"):
        module._check_machine()


def test_checker_rejects_undeclared_transition_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.MACHINE)
    payload["transitions"][1]["to"] = "PSEUDO_STATE"
    invalid = tmp_path / "machine.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "MACHINE", invalid)

    with pytest.raises(module.ContractError, match="unknown target"):
        module._check_machine()


def test_checker_rejects_incomplete_fixedtrack_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.MACHINE)
    del payload["baseline_lifecycle_mapping"]["TERMINATED"]
    invalid = tmp_path / "machine.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "MACHINE", invalid)

    with pytest.raises(module.ContractError, match="lifecycle mapping"):
        module._check_machine()


def test_checker_rejects_unreachable_invalidated_reacquisition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.MACHINE)
    precedence = payload["transition_precedence"]
    precedence[3], precedence[4] = precedence[4], precedence[3]
    invalid = tmp_path / "machine.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "MACHINE", invalid)

    with pytest.raises(module.ContractError, match="recovery unreachable"):
        module._check_machine()


def test_transition_guards_distinguish_reacquire_loss_and_removal() -> None:
    module = _checker_module()

    assert (
        module._select_transition(
            source="INVALIDATED",
            trusted_match=True,
        )
        == "T09_INVALIDATED_BASELINE_REACQUIRE"
    )
    assert (
        module._select_transition(
            source="INVALIDATED",
        )
        == "T10_FAIL_CLOSED_INVALIDATION"
    )
    assert (
        module._select_transition(
            source="VISIBLE_CONFIRMED",
            baseline_removed=True,
        )
        == "T12_BASELINE_TERMINATES"
    )


def test_checker_rejects_h1_like_association_consumer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.MACHINE)
    payload["association_consumer_contract"]["owner_preference_score"] = True
    invalid = tmp_path / "machine.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "MACHINE", invalid)

    with pytest.raises(module.ContractError, match="association consumer"):
        module._check_machine()


def test_formula_properties_are_monotonic_and_fail_closed() -> None:
    module = _checker_module()

    module._check_formula_properties()


def test_golden_cases_are_independently_recomputed() -> None:
    module = _checker_module()

    counts = module._check_golden()

    assert counts == {"usable_short": 13, "unsafe": 5, "nonassigning": 2}


def test_checker_rejects_000216_mechanistic_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    columns, rows = module._read_csv(module.DEVELOPMENT)
    rows[0]["video_key"] = "Pigs291119_000216_30fps"
    invalid = tmp_path / "development.csv"
    with invalid.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    prerequisite = module._read_json(module.PREREQUISITE)
    prerequisite["manifest"]["sha256"] = module._sha256(invalid)
    invalid_prerequisite = tmp_path / "prerequisite.json"
    _write_json(invalid_prerequisite, prerequisite)
    monkeypatch.setattr(module, "DEVELOPMENT", invalid)
    monkeypatch.setattr(module, "PREREQUISITE", invalid_prerequisite)

    with pytest.raises(module.ContractError, match="000216"):
        module._check_prerequisite_and_manifest()


def test_checker_rejects_historical_prevalence_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.PREREQUISITE)
    payload["historical_evidence"]["current_main_failure_prevalence"] = 0.425
    invalid = tmp_path / "prerequisite.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "PREREQUISITE", invalid)

    with pytest.raises(module.ContractError, match="terminology"):
        module._check_prerequisite_and_manifest()


def test_checker_rejects_shadow_independence_weakening(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.PREREQUISITE)
    payload["pass_gates"]["distinct_positive_recording_sessions_minimum"] = 1
    invalid = tmp_path / "prerequisite.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "PREREQUISITE", invalid)

    with pytest.raises(module.ContractError, match="independence"):
        module._check_prerequisite_and_manifest()


def test_checker_rejects_current_main_sha_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.PREREQUISITE)
    payload["authorized_current_main_sha"] = "0" * 40
    invalid = tmp_path / "prerequisite.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "PREREQUISITE", invalid)

    with pytest.raises(module.ContractError, match="current-main SHA"):
        module._check_prerequisite_and_manifest()


def test_default_checker_requires_independent_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    monkeypatch.setattr(module, "REVIEW", tmp_path / "missing.json")

    with pytest.raises(module.ContractError, match="missing H2 design"):
        module.check_contract(pre_review=False)

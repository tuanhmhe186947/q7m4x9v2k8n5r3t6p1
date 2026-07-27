"""Focused tests for the deterministic H1-r3 design checker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _checker_module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "tracking"
        / "check_h1_r3_design_contract.py"
    )
    spec = importlib.util.spec_from_file_location("h1_r3_checker", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_frozen_contract_passes_pre_review() -> None:
    module = _checker_module()

    counts = module.check_contract(pre_review=True)

    assert counts["realistic"] >= 2
    assert counts["visible"] >= 2
    assert counts["abstain"] >= 2


def test_checker_rejects_hidden_only_overlap_floor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.ELIGIBILITY)
    payload["pair_eligibility"]["hidden_only_overlap_floor"] = 0.5
    invalid = tmp_path / "eligibility.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "ELIGIBILITY", invalid)

    with pytest.raises(module.ContractError, match="hidden-only overlap"):
        module._check_eligibility()


def test_checker_rejects_asymmetric_lk_rules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.ELIGIBILITY)
    payload["pair_eligibility"]["lk_validity_rule"][
        "same_for_hidden_and_visible"
    ] = False
    invalid = tmp_path / "eligibility.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "ELIGIBILITY", invalid)

    with pytest.raises(module.ContractError, match="LK validity"):
        module._check_eligibility()


def test_checker_rejects_incompatible_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    payload = module._read_json(module.ACTIVATION)
    payload["activation_rule"][
        "owner_preference_lower_bound_threshold"
    ] = 0.9
    invalid = tmp_path / "activation.json"
    _write_json(invalid, payload)
    monkeypatch.setattr(module, "ACTIVATION", invalid)

    with pytest.raises(module.ContractError, match="incompatible"):
        module._check_activation()


def test_masking_never_increases_hidden_lower_bound() -> None:
    module = _checker_module()

    module._check_masking_counterfactuals()


def test_golden_features_are_recomputed_from_raw_boxes() -> None:
    module = _checker_module()
    payload = module._read_json(module.GOLDEN)
    case = next(
        row
        for row in payload["cases"]
        if row["case_id"] == "margin_and_threshold_boundary"
    )
    hidden = module._candidate_features(case["hidden"], case["detection"])

    assert hidden["overlap"] == pytest.approx(5.0 / 6.0)


def test_uniform_bbox_scaling_preserves_support_interval() -> None:
    module = _checker_module()
    payload = module._read_json(module.GOLDEN)
    case = next(
        row
        for row in payload["cases"]
        if row["case_id"] == "bbox_scale_invariance_small_and_large"
    )
    constants = module._check_activation()
    _, original = module._case_result(case, constants)
    _, scaled = module._case_result(module._scaled_case(case, 4.0), constants)

    assert original == pytest.approx(scaled)


def test_validation_manifest_remains_byte_identical() -> None:
    module = _checker_module()

    assert module._sha256(module.VALIDATION) == module._sha256(
        module.H1_R2_VALIDATION
    )


def test_default_checker_requires_independent_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    monkeypatch.setattr(module, "REVIEW", tmp_path / "missing.json")

    with pytest.raises(module.ContractError, match="missing design artifacts"):
        module.check_contract(pre_review=False)

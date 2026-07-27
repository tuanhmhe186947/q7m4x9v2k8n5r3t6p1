"""Tests for the deterministic H1-r2 design-only contract checker."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _checker_module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "tracking"
        / "check_h1_r2_design_contract.py"
    )
    spec = importlib.util.spec_from_file_location("h1_r2_checker", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_h1_r2_design_contract_passes() -> None:
    module = _checker_module()

    assert module.main() == 0


def test_checker_rejects_probability_like_score_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    invalid_contract = tmp_path / "invalid_contract.md"
    invalid_contract.write_text(
        "Uncalibrated score name: p_owner\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CONTRACT", invalid_contract)

    assert module.main() == 1


def test_checker_rejects_changed_validation_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    package = module._read_json(module.ROLE_ASSIGNMENTS)
    package["assignments"][0]["start_frame"] += 1
    invalid_package = tmp_path / "role_assignments.json"
    invalid_package.write_text(
        json.dumps(package, sort_keys=True),
        encoding="utf-8",
    )
    package_sha256 = hashlib.sha256(invalid_package.read_bytes()).hexdigest()
    _, validation = module._read_csv(module.VALIDATION)
    for row in validation:
        row["assignment_artifact_sha256"] = package_sha256
    monkeypatch.setattr(module, "ROLE_ASSIGNMENTS", invalid_package)

    with pytest.raises(module.ContractError, match="assigned boundary changed"):
        module._check_role_assignments(validation)


def test_checker_rejects_validation_selected_activation_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    activation = module._read_json(module.ACTIVATION_GATE)
    activation["validation_data_used"] = True
    invalid_activation = tmp_path / "activation.json"
    invalid_activation.write_text(
        json.dumps(activation, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ACTIVATION_GATE", invalid_activation)

    with pytest.raises(module.ContractError, match="validation_data_used"):
        module._check_activation_gate(module._check_features())


def test_checker_rejects_failed_independent_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    review = module._read_json(module.INDEPENDENT_REVIEW)
    review["review_result"] = "FAIL"
    invalid_review = tmp_path / "review.json"
    invalid_review.write_text(
        json.dumps(review, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "INDEPENDENT_REVIEW", invalid_review)

    with pytest.raises(module.ContractError, match="did not pass"):
        module._check_independent_review()


def test_checker_keeps_evaluation_and_promotion_unauthorized() -> None:
    module = _checker_module()
    review = module._read_json(module.INDEPENDENT_REVIEW)

    assert review["evaluation_authorized"] is False
    assert review["promotion_authorized"] is False

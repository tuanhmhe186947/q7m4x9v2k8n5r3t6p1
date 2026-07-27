"""Tests for the deterministic H1-r2 design-only contract checker."""

from __future__ import annotations

import importlib.util
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

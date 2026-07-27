"""Focused tests for the fail-closed H1-r2 implementation checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _checker_module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "tracking"
        / "check_h1_r2_implementation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "h1_r2_implementation_checker",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_h1_r2_implementation_passes() -> None:
    module = _checker_module()

    assert module.main() == 0


def test_checker_rejects_threshold_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _checker_module()
    monkeypatch.setattr(module, "OWNER_PREFERENCE_THRESHOLD", 0.61)

    with pytest.raises(module.ImplementationError, match="constants changed"):
        module._check_features_and_score()


def test_checker_rejects_realtime_fast_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _checker_module()
    changed = dict(module.REALTIME_FAST_CONFIG)
    changed["detect_every_n_frames"] = 1
    monkeypatch.setattr(module, "REALTIME_FAST_CONFIG", changed)

    with pytest.raises(module.ImplementationError, match="realtime_fast changed"):
        module._check_profiles_and_causality()


def test_checker_rejects_profile_promotion_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    amendment = module._read_json(module.PROFILE_AMENDMENT)
    amendment["promotion_authorized"] = True
    invalid = tmp_path / "invalid_amendment.json"
    invalid.write_text(
        module.json.dumps(amendment, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROFILE_AMENDMENT", invalid)

    with pytest.raises(module.ImplementationError, match="promotion_authorized"):
        module._check_authority()


def test_checker_rejects_probability_like_runtime_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _checker_module()
    invalid = tmp_path / "owner_preference.py"
    invalid.write_text("owner_probability = 0.5\n", encoding="utf-8")
    monkeypatch.setattr(module, "OWNER_SOURCE", invalid)

    with pytest.raises(module.ImplementationError, match="probability-like"):
        module._check_symmetric_integration_and_terminology()

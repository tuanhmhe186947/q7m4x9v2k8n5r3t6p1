"""Focused phase/provenance controls for the Stage-1 RGB binding producer."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER = ROOT / (
    "scripts/classification_v2/04_baselines_smokes/"
    "classification_v2_materialize_s1_stage1_rgb_bindings.py"
)


def _module() -> dict[str, object]:
    return runpy.run_path(str(MATERIALIZER), run_name="s1_materializer_test")


def test_initial_phase_excludes_confirmation_provenance() -> None:
    module = _module()
    resolve = module["_confirmation_authority_for_phase"]
    assert callable(resolve)
    assert resolve(
        execution_phase=module["INITIAL_STAGE1_SCREEN"],
        seed=20260804,
        confirmation_authority=None,
    ) is None


def test_confirmation_phase_requires_future_seed_and_authority(tmp_path: Path) -> None:
    module = _module()
    resolve = module["_confirmation_authority_for_phase"]
    assert callable(resolve)
    retention = tmp_path / "retention.json"
    retention.write_text("{}", encoding="utf-8")
    assert resolve(
        execution_phase=module["STAGE1_CONFIRMATION"],
        seed=20260805,
        confirmation_authority=retention,
    ) == retention.resolve()
    with pytest.raises(ValueError, match="requires confirmation authority"):
        resolve(
            execution_phase=module["STAGE1_CONFIRMATION"],
            seed=20260805,
            confirmation_authority=None,
        )
    with pytest.raises(ValueError, match="requires a future seed"):
        resolve(
            execution_phase=module["STAGE1_CONFIRMATION"],
            seed=20260804,
            confirmation_authority=retention,
        )
    views = module["_materialization_views"]
    assert callable(views)
    assert views(module["STAGE1_CONFIRMATION"]) == ("T6", "T16")


def test_initial_phase_rejects_confirmation_authority(tmp_path: Path) -> None:
    module = _module()
    resolve = module["_confirmation_authority_for_phase"]
    assert callable(resolve)
    with pytest.raises(ValueError, match="must not bind confirmation authority"):
        resolve(
            execution_phase=module["INITIAL_STAGE1_SCREEN"],
            seed=20260804,
            confirmation_authority=tmp_path / "retention.json",
        )

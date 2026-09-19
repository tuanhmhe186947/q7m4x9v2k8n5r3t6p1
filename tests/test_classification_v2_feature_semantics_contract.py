from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.feature_semantics import (
    audit_feature_semantics,
)
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)

ROOT = Path(__file__).resolve().parents[1]
SPATIAL_CHECKER = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "02_train_ready_exports"
    / "check_classification_v2_spatial_sequences.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a compact synthetic contract fixture."""

    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_spatial_checker() -> ModuleType:
    """Load the numbered workflow script without making scripts a package."""

    spec = importlib.util.spec_from_file_location(
        "classification_v2_spatial_checker",
        SPATIAL_CHECKER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_semantics_accepts_bounded_artifact_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Smoke paths must be auditable without replacing canonical artifacts."""

    monkeypatch.chdir(tmp_path)
    _write_json(
        tmp_path / "spatial_trainer.json",
        {"spatial_sequence_feature_whitelist": ["bbox_xywh_n"]},
    )
    _write_json(
        tmp_path / "tabular_trainer.json",
        {"tabular_feature_whitelist": ["speed_mean_window"]},
    )
    contract_path = tmp_path / "semantics.json"
    _write_json(
        contract_path,
        {
            "trainer_contract_json": "spatial_trainer.json",
            "tabular_trainer_contract_json": "tabular_trainer.json",
            "tabular_x_csv": "canonical/X.csv",
            "spatial_npz": "canonical/X.npz",
            "forbidden_x_patterns": ["review_*", "*behavior*"],
            "tabular_families": {
                "motion": {
                    "prefixes": ["speed_"],
                }
            },
            "spatial_arrays": {
                "bbox_xywh_n": {
                    "family": "geometry_sequence",
                    "model_input_role": "model_input",
                    "model_input_allowed": True,
                }
            },
        },
    )
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    tabular_path = smoke_dir / "X.csv"
    spatial_path = smoke_dir / "X.npz"
    pd.DataFrame({"speed_mean_window": [0.25]}).to_csv(
        tabular_path,
        index=False,
    )
    np.savez(spatial_path, bbox_xywh_n=np.zeros((1, 2, 4), dtype="float32"))

    audit = audit_feature_semantics(
        contract_path,
        tabular_x_csv=tabular_path,
        spatial_npz=spatial_path,
    )

    assert audit["valid"] is True
    assert audit["tabular_contract_match"] is True
    assert audit["tabular_x_csv"] == "smoke\\X.csv"
    assert audit["spatial_npz"] == "smoke\\X.npz"


def test_spatial_checker_rejects_missing_slots_in_trainable_rows(
    tmp_path: Path,
) -> None:
    """Padding is allowed, but a trainable in-window frame may not be absent."""

    checker = _load_spatial_checker()
    mask_path = tmp_path / "train_mask.csv"
    pd.DataFrame({"window_valid_for_main_train": [True, False]}).to_csv(
        mask_path,
        index=False,
    )
    length = np.ones((2, 2), dtype="float32")
    observed = np.array([[1.0, 0.0], [1.0, 1.0]], dtype="float32")

    audit = checker._audit_train_mask_completeness(
        mask_path,
        length,
        observed,
        expected_rows=2,
    )

    assert audit["trainable_rows_with_missing_slots"] == 1
    assert audit["trainable_missing_slots"] == 1
    assert audit["errors"] == [
        "trainable_windows_have_missing_spatial_slots=rows:1 slots:1"
    ]


def test_derived_output_guard_requires_explicit_overwrite(tmp_path: Path) -> None:
    """Existing derived artifacts may only be replaced by an explicit command."""

    existing = tmp_path / "artifact.csv"
    existing.write_text("fixture", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite explicitly"):
        require_output_paths_available([existing], overwrite=False)

    require_output_paths_available([existing], overwrite=True)

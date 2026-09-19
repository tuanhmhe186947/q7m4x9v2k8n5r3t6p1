from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

_SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "classification_v2"
    / "02_train_ready_exports"
    / "check_classification_v2_legacy_development_l2.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "classification_v2_legacy_development_l2_checker",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKER)


def _materialize(
    expectations: dict[tuple[str, ...], Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for path, value in expectations.items():
        current = payload
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = copy.deepcopy(value)
    return payload


def _valid_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    expectations = _CHECKER._evidence_expectations()
    tier = _materialize(expectations["tier"])
    loader = _materialize(expectations["loader"])
    counts = {label: 1 for label in VALID_BEHAVIORS}
    counts[VALID_BEHAVIORS[0]] += _CHECKER.EXPECTED_NATIVE_UNITS - len(counts)
    tier["native_unit_audit"].update(
        {
            "valid_development_units": 4_545,
            "invalid_development_units": 9,
            "behavior_counts": counts,
        }
    )
    return tier, loader


def test_l2_evidence_accepts_exact_counts_and_retained_policy_invalid_units() -> None:
    tier, loader = _valid_evidence()

    audit = _CHECKER.audit_legacy_l2_evidence(
        tier,
        loader,
        artifact_name="fixture",
    )

    assert audit["valid"] is True
    assert audit["errors"] == []
    assert audit["valid_development_units"] == 4_545
    assert audit["invalid_development_units"] == 9


def test_l2_evidence_rejects_loss_timing_claim_and_label_drift() -> None:
    tier, loader = _valid_evidence()
    view_name = next(iter(_CHECKER.LEGACY_TEMPORAL_MODEL_VIEW_SPECS))
    tier["temporal_tier_audit"]["rows_dropped"] = 1
    tier["temporal_model_input_audit"]["view_audits"][view_name][
        "invalid_timing_slots"
    ] = 1
    loader["legacy_tier_real_packet"]["views"][view_name][
        "human_review_complete"
    ] = True
    tier["native_unit_audit"]["behavior_counts"].pop(VALID_BEHAVIORS[-1])

    audit = _CHECKER.audit_legacy_l2_evidence(
        tier,
        loader,
        artifact_name="fixture",
    )

    assert audit["valid"] is False
    assert any("rows_dropped" in error for error in audit["errors"])
    assert any("invalid_timing_slots" in error for error in audit["errors"])
    assert any("human_review_complete" in error for error in audit["errors"])
    assert any("native labels" in error for error in audit["errors"])


def test_l2_csv_audit_requires_exact_rows_and_claim_pair(tmp_path: Path) -> None:
    relative = "claimed.csv"
    frame = pd.DataFrame(
        {
            "lineage_scope": [LEGACY_DEVELOPMENT_SCOPE] * 2,
            "human_review_complete": [False, False],
        }
    )
    frame.to_csv(tmp_path / relative, index=False)

    audit = _CHECKER._audit_csv_artifacts(tmp_path, {relative: 2})

    assert audit["valid"] is True
    assert audit["rows"] == {relative: 2}
    assert audit["hashes"][relative]

    frame["human_review_complete"] = ["false", "yes"]
    frame.to_csv(tmp_path / relative, index=False)
    invalid = _CHECKER._audit_csv_artifacts(tmp_path, {relative: 2})

    assert invalid["valid"] is False
    assert any("human_review_complete" in error for error in invalid["errors"])


def test_l2_hash_audits_reject_repeat_and_bound_hash_drift(tmp_path: Path) -> None:
    repeat = _CHECKER._audit_repeat_hashes(
        {"artifact.csv": "same"},
        {"artifact.csv": "same"},
        ("artifact.csv",),
    )
    assert repeat["valid"] is True

    mismatch = _CHECKER._audit_repeat_hashes(
        {"artifact.csv": "first"},
        {"artifact.csv": "second"},
        ("artifact.csv",),
    )
    assert mismatch["errors"] == ["repeat_hash_mismatch=artifact.csv"]

    actual = {
        relative: "bound"
        for relative in {
            *_CHECKER.TIER_INPUT_ARTIFACTS.values(),
            *_CHECKER.TIER_OUTPUT_ARTIFACTS.values(),
        }
    }
    tier = {
        "input_artifacts": {
            name: {
                "sha256": "bound",
                "path": str(tmp_path / relative),
            }
            for name, relative in _CHECKER.TIER_INPUT_ARTIFACTS.items()
        },
        "output_artifacts": {
            name: {
                "sha256": "bound",
                "path": str(tmp_path / relative),
            }
            for name, relative in _CHECKER.TIER_OUTPUT_ARTIFACTS.items()
        },
    }
    assert _CHECKER._verify_bound_hashes(tmp_path, tier, actual)["valid"] is True

    first_output = next(iter(tier["output_artifacts"].values()))
    first_output["sha256"] = "drift"
    invalid_bound = _CHECKER._verify_bound_hashes(tmp_path, tier, actual)
    assert invalid_bound["valid"] is False

"""Forensic Standard-V2 evaluation of the historical H5b/H4 artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tracking"
    / "evaluate_historical_h5b_h4_standard_v2.py"
)
SPEC = importlib.util.spec_from_file_location("historical_h5b_h4", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(
    hota: float,
    idf1: float,
    wrong: int,
    terminal: int,
    swaps: int,
) -> pd.Series:
    return pd.Series(
        {
            "hota": hota,
            "idf1": idf1,
            "wrong_id_matched_frames": wrong,
            "terminal_identity_error_episode_count": terminal,
            "persistent_pairwise_identity_swap_count": swaps,
        }
    )


def test_historical_broadly_better_rule_requires_every_severity_gate() -> None:
    historical = _row(0.91, 0.98, 100, 2, 1)
    r0 = _row(0.89, 0.97, 110, 3, 2)
    assert MODULE.classify_vs_r0(historical, r0) == (
        "HISTORICAL_HYBRID_BROADLY_BETTER_THAN_R0"
    )
    historical["terminal_identity_error_episode_count"] = 4
    assert MODULE.classify_vs_r0(historical, r0) == (
        "HISTORICAL_HYBRID_MIXED_VS_R0"
    )


def test_r0_broadly_better_rule_is_symmetric() -> None:
    historical = _row(0.88, 0.96, 120, 5, 3)
    r0 = _row(0.89, 0.97, 110, 3, 2)
    assert MODULE.classify_vs_r0(historical, r0) == (
        "R0_BROADLY_BETTER_THAN_HISTORICAL_HYBRID"
    )


def test_unseen_impact_rule() -> None:
    decision, status = MODULE._impact_decision(  # noqa: SLF001
        "HISTORICAL_HYBRID_BROADLY_BETTER_THAN_R0"
    )
    assert decision == (
        "SUSPEND_CURRENT_UNSEEN_FREEZE_PENDING_HISTORICAL_METHOD_REPRODUCTION"
    )
    assert status == "SUSPENDED_PENDING_REPRODUCTION"
    decision, status = MODULE._impact_decision(  # noqa: SLF001
        "HISTORICAL_HYBRID_MIXED_VS_R0"
    )
    assert decision == "REAFFIRM_CURRENT_UNSEEN_METHOD_FREEZE"
    assert status == "REAFFIRMED"

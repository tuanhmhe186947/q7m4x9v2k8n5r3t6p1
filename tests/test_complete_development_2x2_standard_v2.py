"""Focused tests for the frozen development tracking 2x2 orchestration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "tracking"
    / "complete_development_2x2_standard_v2.py"
)


def _load_module():
    script_dir = str(SCRIPT.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("development_2x2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _aggregate(
    *,
    b1_hota: float = 0.81,
    b1_idf1: float = 0.91,
    r1_hota: float = 0.86,
    r1_idf1: float = 0.96,
) -> pd.DataFrame:
    rows = []
    values = {
        "B0": (0.80, 0.90, 100, 10, 3),
        "B1": (b1_hota, b1_idf1, 90, 9, 2),
        "R0": (0.85, 0.95, 80, 8, 2),
        "R1": (r1_hota, r1_idf1, 70, 7, 1),
    }
    for arm, (hota, idf1, wrong, terminal, swaps) in values.items():
        row = {metric: 1.0 for metric, *_ in _load_module().METRICS}
        row.update(
            {
                "arm": arm,
                "hota": hota,
                "idf1": idf1,
                "wrong_id_matched_frames": wrong,
                "wrong_id_matched_seconds": float(wrong),
                "terminal_identity_error_episode_count": terminal,
                "persistent_pairwise_identity_swap_count": swaps,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_effect_table_uses_declared_interaction_formula() -> None:
    module = _load_module()
    table = module._effect_table(_aggregate())
    row = table.loc[table["metric"] == "hota"].iloc[0]

    assert abs(row["B1_minus_B0"] - 0.01) < 1e-15
    assert abs(row["R1_minus_R0"] - 0.01) < 1e-15
    assert abs(row["interaction_raw"]) < 1e-15


def test_lower_is_better_oriented_effect_is_inverted() -> None:
    module = _load_module()
    table = module._effect_table(_aggregate())
    row = table.loc[
        table["metric"] == "wrong_id_matched_frames"
    ].iloc[0]

    assert row["B1_minus_B0"] == -10
    assert row["bytetrack_repair_oriented_effect"] == 10


def test_predeclared_classification_rules_are_conservative() -> None:
    module = _load_module()
    aggregate = _aggregate()

    assert (
        module.classify_repair(aggregate, "B1", "B0")
        == "BROADLY_BENEFICIAL"
    )
    mixed = _aggregate(b1_idf1=0.89)
    assert module.classify_repair(mixed, "B1", "B0") == "MIXED_TRADEOFF"
    harmful = _aggregate(
        b1_hota=0.79,
        b1_idf1=0.89,
    )
    harmful.loc[harmful["arm"] == "B1", "wrong_id_matched_frames"] = 110
    assert (
        module.classify_repair(harmful, "B1", "B0")
        == "BROADLY_HARMFUL"
    )


def test_r1_repeat_requires_all_authority_tables() -> None:
    module = _load_module()
    hashes = {
        name: f"sha-{index}"
        for index, name in enumerate(module.REPEAT_FILES)
    }
    passed = module.compare_r1_passes(
        {"output_hashes": hashes},
        {"output_hashes": dict(hashes)},
    )
    assert passed["R1_reevaluation_repeatability"] == "PASS"

    changed = dict(hashes)
    changed[module.REPEAT_FILES[0]] = "different"
    failed = module.compare_r1_passes(
        {"output_hashes": hashes},
        {"output_hashes": changed},
    )
    assert failed["R1_reevaluation_repeatability"] == "FAIL"

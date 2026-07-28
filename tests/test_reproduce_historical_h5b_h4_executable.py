from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from scripts.tracking import reproduce_historical_h5b_h4_executable as repro

from pig_behavior.tracking import runner


def _write_xml(path: Path, *, x_offset: float = 0.0, identity: str = "ID_1") -> None:
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<annotations><track id='1' label='Pig_1'>"
        f"<box frame='0' xtl='{10.0 + x_offset:.2f}' ytl='20.00' "
        "xbr='30.00' ybr='40.00' outside='0' occluded='0'>"
        f"<attribute name='ID'>{identity}</attribute>"
        "<attribute name='Behavior'>Other</attribute>"
        "<attribute name='Hidden'>false</attribute>"
        "</box></track></annotations>\n",
        encoding="utf-8",
    )


def test_parity_contract_is_frozen_before_execution() -> None:
    contract = repro.prediction_parity_contract()

    assert contract["frozen_before_execution"] is True
    assert contract["semantic_contract"]["bbox_absolute_tolerance"] == 0.01
    assert contract["semantic_contract"]["minimum_paired_bbox_iou"] == 0.9999
    assert contract["metric_contract"]["count_metrics"] == "EXACT_EQUALITY"
    assert contract["post_result_parameter_changes_authorized"] is False


def test_xml_comparison_separates_identity_and_bbox_differences(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical.xml"
    reproduced = tmp_path / "reproduced.xml"
    _write_xml(historical)
    _write_xml(reproduced, x_offset=0.01, identity="ID_2")

    result = repro.compare_xml(historical, reproduced, "video")

    assert result["row_additions"] == 0
    assert result["row_removals"] == 0
    assert result["identity_value_differences"] == 1
    assert result["bbox_exact_value_differences"] == 1
    assert result["maximum_absolute_bbox_coordinate_difference"] == pytest.approx(
        0.01
    )


def test_hybrid_capture_is_opt_in_and_runs_before_final_assignment() -> None:
    source = inspect.getsource(runner.run_tracking)

    assert "hybrid_repair_capture" in source
    snapshot = source.index("deepcopy(shapes) if hybrid_repair_capture")
    repair = source.index("repair_result = apply_offline_repair_stack", snapshot)
    capture = source.index("hybrid_repair_capture(raw_snapshot", repair)
    final = source.index("shapes = repair_result.shapes", capture)
    assert snapshot < repair < capture < final

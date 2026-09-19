"""Forensic tests for the frozen three-arm Standard-V2 orchestration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO / "scripts" / "tracking" / "evaluate_b0_b1_r0_standard_v2.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("three_arm_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prediction_xml(video_key: str = "video") -> str:
    return f"""\
<annotations>
  <meta>
    <task>
      <name>{video_key}</name>
      <size>1800</size>
    </task>
  </meta>
  <track id="7" label="Pig_3">
    <box frame="0" xtl="1" ytl="2" xbr="11" ybr="12" outside="0">
      <attribute name="ID">ID_3</attribute>
      <attribute name="Hidden">Yes</attribute>
    </box>
    <box frame="1" xtl="2" ytl="3" xbr="12" ybr="13" outside="0">
      <attribute name="ID">ID_3</attribute>
      <attribute name="Hidden">No</attribute>
    </box>
    <box frame="2" xtl="3" ytl="4" xbr="13" ybr="14" outside="1">
      <attribute name="ID">ID_3</attribute>
      <attribute name="Hidden">No</attribute>
    </box>
  </track>
</annotations>
"""


def test_adapter_conservation_preserves_active_rows(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "video.xml"
    path.write_text(_prediction_xml(), encoding="utf-8")

    audit = module.adapter_audit(path)

    assert audit["active_xml_rows"] == 2
    assert audit["parsed_rows"] == 2
    assert audit["adapter_bbox_changes"] == 0
    assert audit["adapter_id_changes"] == 0
    assert audit["adapter_prediction_hidden_changes"] == 0


def test_prediction_hash_is_input_order_invariant(tmp_path: Path) -> None:
    module = _load_module()
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    first.write_text(_prediction_xml("first"), encoding="utf-8")
    second.write_text(_prediction_xml("second"), encoding="utf-8")
    records = [
        module.prediction_structural_record(
            first,
            video_key="first",
            width=100,
            height=100,
        ),
        module.prediction_structural_record(
            second,
            video_key="second",
            width=100,
            height=100,
        ),
    ]

    assert module.prediction_set_hash(records) == module.prediction_set_hash(
        list(reversed(records))
    )


def test_compare_passes_requires_all_six_tables() -> None:
    module = _load_module()
    hashes = {name: f"sha-{index}" for index, name in enumerate(
        module.REQUIRED_REPEAT_FILES
    )}

    result = module.compare_passes(
        {"output_hashes": hashes},
        {"output_hashes": dict(reversed(list(hashes.items())))},
    )

    assert result["reevaluation_repeatability"] == "PASS"
    assert result["input_order_invariance"] == "PASS"
    assert result["complete_evaluation_passes"] == 2


def test_comparison_interpretations_preserve_cadence_scope() -> None:
    module = _load_module()
    rows = []
    for arm, offset in (("B0", 0), ("B1", 1), ("R0", 2)):
        row = {"arm": arm}
        row.update(
            {
                metric: float(offset)
                for metric in module.METRIC_COLUMNS
            }
        )
        rows.append(row)

    comparisons = module._comparison_rows(pd.DataFrame(rows))

    cadence = (
        comparisons.groupby("comparison")["detector_cadence_matched"]
        .first()
        .to_dict()
    )
    assert cadence == {
        "B1_MINUS_B0": True,
        "R0_MINUS_B0": False,
        "R0_MINUS_B1": False,
    }
    assert comparisons.loc[
        comparisons["comparison"] == "R0_MINUS_B0", "interpretation"
    ].str.contains("Whole-pipeline").all()


def test_metric_config_is_primary_standard_v2() -> None:
    module = _load_module()

    document = module._metric_config_document("abc", "def")

    assert document["evaluator_contract_id"] == (
        "TRACKING_EVALUATOR_STANDARD_V2"
    )
    assert document["identity_episode_contract_id"] == (
        "IDENTITY_ERROR_EPISODES_V2"
    )
    assert document["include_hidden"] is True
    assert len(document["hota_alpha_set"]) == 19
    assert document["profile_specific_evaluator_branches"] == 0


def test_pairwise_event_ids_are_namespaced_by_arm() -> None:
    module = _load_module()
    event = SimpleNamespace(
        event_id="shared",
        gt_ids=("ID_1", "ID_2"),
        sequence_key="video",
    )
    episode_result = SimpleNamespace(
        wrong_id_rows_input=0,
        wrong_id_rows_classified=0,
        wrong_id_rows_double_counted=0,
        episodes=(),
        pairwise_events=(event,),
    )
    metrics = SimpleNamespace(
        video_stem="video",
        gt_detections=1,
        pred_detections=1,
    )
    hota_result = SimpleNamespace(
        tp=(1,) * 19,
        fp=(0,) * 19,
        fn=(0,) * 19,
    )
    evaluation = SimpleNamespace(
        episode_result=episode_result,
        metrics=metrics,
        hota_result=hota_result,
    )

    result = module._conservation(
        {"B0": [evaluation], "B1": [evaluation]}
    )

    assert result["pairwise_swap_double_count"] == 0

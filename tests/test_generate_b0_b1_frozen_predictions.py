from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "tracking"
        / "generate_b0_b1_frozen_predictions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_b0_b1_frozen_predictions",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _sample_xml(path: Path, *, frame: int = 0, hidden: str = "No") -> None:
    path.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<annotations>\n"
            "  <version>1.1</version>\n"
            "  <meta><task>"
            "<name>video_a</name><size>1800</size>"
            "<created>volatile</created><updated>volatile</updated>"
            "</task><dumped>volatile</dumped></meta>\n"
            '  <track id="1" label="Pig_1">\n'
            f'    <box frame="{frame}" xtl="1" ytl="2" '
            'xbr="11" ybr="12" outside="0" occluded="0" '
            'keyframe="1">\n'
            f'      <attribute name="Hidden">{hidden}</attribute>\n'
            "    </box>\n"
            "  </track>\n"
            "</annotations>\n"
        ),
        encoding="utf-8",
    )


def test_profile_hashes_and_active_registry_match_authority() -> None:
    b0, b1 = MODULE.profile_payloads()

    assert MODULE.canonical_hash(b0) == MODULE.B0_CONFIG_SHA256
    assert MODULE.canonical_hash(b1) == MODULE.B1_CONFIG_SHA256


def test_full_cache_path_is_exact_partition_contract(tmp_path: Path) -> None:
    actual = MODULE.full_cache_path(tmp_path, "video_a")

    assert actual == (
        tmp_path
        / "full"
        / "partitions"
        / "video_a"
        / "detector_evidence.npz"
    )


def test_expected_cache_identity_binds_all_semantic_authorities() -> None:
    video = SimpleNamespace(
        video_key="video_a",
        video_sha256="a" * 64,
    )
    authority = {"producer_code_sha": "b" * 40}

    identity = MODULE.expected_cache_identity(video, authority)

    assert identity.video_key == "video_a"
    assert identity.source_video_sha256 == "a" * 64
    assert identity.detector_weight_sha256 == MODULE.DETECTOR_WEIGHTS_SHA256
    assert identity.detector_semantic_config_sha256 == (
        MODULE.DETECTOR_SEMANTIC_CONFIG_SHA256
    )
    assert identity.creation_authority == (
        MODULE.full_cache_tool.FULL_CREATION_AUTHORITY
    )


def test_xml_structural_record_accepts_valid_prediction(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "prediction.xml"
    _sample_xml(xml_path)

    record = MODULE.xml_structural_record(
        xml_path,
        video_key="video_a",
        width=1280,
        height=720,
    )

    assert record["prediction_object_count"] == 1
    assert record["track_count"] == 1
    assert record["hidden_values_present"] == ["No"]
    assert record["bbox_validity"] == "PASS"


@pytest.mark.parametrize(
    ("frame", "hidden"),
    [
        (1800, "No"),
        (0, "Unknown"),
    ],
)
def test_xml_structural_record_rejects_invalid_authority(
    tmp_path: Path,
    frame: int,
    hidden: str,
) -> None:
    xml_path = tmp_path / "prediction.xml"
    _sample_xml(xml_path, frame=frame, hidden=hidden)

    with pytest.raises(MODULE.PredictionGenerationError):
        MODULE.xml_structural_record(
            xml_path,
            video_key="video_a",
            width=1280,
            height=720,
        )


def test_prediction_set_hash_is_input_order_invariant() -> None:
    records = [
        {
            "video_key": "b",
            "sha256": "1" * 64,
            "semantic_sha256": "2" * 64,
            "canonical_row_sha256": "3" * 64,
            "prediction_object_count": 10,
            "processed_frame_count": 1800,
        },
        {
            "video_key": "a",
            "sha256": "4" * 64,
            "semantic_sha256": "5" * 64,
            "canonical_row_sha256": "6" * 64,
            "prediction_object_count": 11,
            "processed_frame_count": 1800,
        },
    ]

    assert MODULE.prediction_set_hash(records) == MODULE.prediction_set_hash(
        list(reversed(records))
    )


def test_artifact_inventory_is_ordered_and_hash_bound(
    tmp_path: Path,
) -> None:
    arm_root = tmp_path / "B0"
    prediction_root = arm_root / "predictions"
    machine_root = arm_root / "machine_readable"
    prediction_root.mkdir(parents=True)
    machine_root.mkdir()
    (prediction_root / "b.xml").write_text("b", encoding="utf-8")
    (prediction_root / "a.xml").write_text("a", encoding="utf-8")
    first = MODULE.artifact_inventory(arm_root)

    assert [row["relative_path"] for row in first] == [
        "predictions/a.xml",
        "predictions/b.xml",
    ]
    first_hash = MODULE.canonical_hash(first)
    (prediction_root / "a.xml").write_text("changed", encoding="utf-8")
    second_hash = MODULE.canonical_hash(
        MODULE.artifact_inventory(arm_root)
    )
    assert first_hash != second_hash


def test_preflight_refuses_existing_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "existing"
    output_root.mkdir()

    with pytest.raises(
        MODULE.PredictionGenerationError,
        match="refusing existing output root",
    ):
        MODULE.preflight(
            tmp_path,
            tmp_path / "lineage.json",
            tmp_path / "cache",
            output_root,
        )


def test_population_manifest_preserves_000216_policy() -> None:
    videos = [
        SimpleNamespace(
            video_key="Pigs291119_000216_30fps",
            video_path=Path("video.mp4"),
            video_sha256="a" * 64,
            gt_path=Path("gt.xml"),
            gt_sha256="b" * 64,
            gt_authority="UNRESOLVED_EXCLUDE_FROM_MECHANISM_RANKING",
        )
    ]
    caches = [
        {
            "video_key": "Pigs291119_000216_30fps",
            "cache_path": "cache.npz",
            "cache_sha256": "c" * 64,
            "detector_record_count": 1800,
        }
    ]

    payload = MODULE.population_manifest(
        videos,
        caches,
        source_lineage_path=Path("lineage.json"),
        source_lineage_sha256="d" * 64,
        source_authority_sha256="e" * 64,
        gt_authority_sha256="f" * 64,
    )

    row = payload["videos"][0]
    assert row["aggregate_inclusion_role"] == "LOCKED_AGGREGATE_ONLY"
    assert row["mechanism_ranking_eligibility"] is False


def test_marker_declares_non_disposable_authority() -> None:
    marker = MODULE.marker_text("bytetrack_raw", "a" * 40)

    assert MODULE.RETENTION_CLASS in marker
    assert "NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT" in marker
    assert "not temporary output" in marker


def test_generation_tool_has_no_live_detector_or_evaluator_path() -> None:
    source = Path(MODULE.__file__).read_text(encoding="utf-8")
    forbidden = (
        "from ultralytics import YOLO",
        "evaluate_tracking(",
        "aggregate_metrics(",
        "TrackEval",
        "write_output_video=True",
    )

    for text in forbidden:
        assert text not in source
    assert "run_tracking(cfg, model=detector)" in source
    assert "ReplayDetector(cache)" in source


def test_decision_contract_has_no_quality_fields(tmp_path: Path) -> None:
    payload = {
        "standard_v2_metric_runs": 0,
        "legacy_metric_runs": 0,
        "quality_comparisons": 0,
        "hota_values_generated": 0,
        "idf1_values_generated": 0,
        "idsw_values_generated": 0,
    }
    path = tmp_path / "decision.json"
    MODULE.write_json(path, payload)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded.values()) == {0}


def test_frozen_cross_arm_fairness_authority_passes() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "tracking"
        / "b0_b1_prediction_authority"
        / "B0_B1_R0_PREDICTION_AUTHORITY_FAIRNESS_20260728.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["common_video_authority"] == "PASS"
    assert payload["common_frame_authority"] == "PASS"
    assert payload["common_gt_authority"] == "PASS"
    assert payload["common_detector_model_authority"] == "PASS"
    assert payload["common_detector_config_authority"] == "PASS"
    assert payload["b0_b1_full_cache_authority_match"] == "PASS"
    assert payload["r0_even_subset_authority_preserved"] == "PASS"
    assert payload["quality_metrics_calculated"] == 0

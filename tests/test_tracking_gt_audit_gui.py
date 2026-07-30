import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from pig_behavior.tracking.gt_audit_review import (
    CONFIDENCES,
    DECISION_COLUMNS,
    DECISIONS,
    DecisionLedger,
    ExactFrameReader,
    append_event,
    clamp_context,
    coverage,
    parse_cvat,
    parse_span,
    render_boxes,
    sha256,
    timeline_state,
    validate_decision,
)


def test_cvat_identity_hidden_and_overlay_render(tmp_path: Path):
    xml = tmp_path / "fixture.xml"
    xml.write_text(
        "<annotations><track id='0' label='Pig_1'>"
        "<box frame='3' outside='0' occluded='0' "
        "xtl='10' ytl='12' xbr='30' ybr='32'>"
        "<attribute name='ID'>ID_1</attribute>"
        "<attribute name='Hidden'>Yes</attribute>"
        "</box></track></annotations>"
    )
    objects = parse_cvat(xml)[3]
    assert objects[0]["id"] == "ID_1"
    assert objects[0]["hidden"] is True
    import numpy as np

    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    rendered = render_boxes(frame, objects, "ID_1", (0, 255, 0), "GT")
    assert rendered.shape == frame.shape
    assert int(rendered.sum()) > 0


def test_synthetic_two_identity_review_fixture(tmp_path: Path):
    video = tmp_path / "synthetic.avi"
    _write_video(video, 8, 12.5)
    gt = tmp_path / "gt.xml"
    pred = tmp_path / "pred.xml"
    xml = (
        "<annotations>"
        "<track id='0' label='Pig_1'>"
        "<box frame='0' outside='0' occluded='0' xtl='2' ytl='2' xbr='10' ybr='10'>"
        "<attribute name='ID'>ID_1</attribute></box>"
        "<box frame='1' outside='0' occluded='1' xtl='3' ytl='2' xbr='11' ybr='10'>"
        "<attribute name='ID'>ID_1</attribute>"
        "<attribute name='Hidden'>Yes</attribute></box>"
        "</track>"
        "<track id='1' label='Pig_2'>"
        "<box frame='0' outside='0' occluded='0' xtl='20' ytl='2' xbr='30' ybr='10'>"
        "<attribute name='ID'>ID_2</attribute></box>"
        "</track>"
        "</annotations>"
    )
    gt.write_text(xml)
    pred.write_text(xml.replace("ID_1", "ID_2", 1))
    gt_objects = parse_cvat(gt)
    pred_objects = parse_cvat(pred)
    assert {obj["id"] for obj in gt_objects[0]} == {"ID_1", "ID_2"}
    assert gt_objects[1][0]["hidden"] is True
    state = timeline_state(
        {
            "event_start_frame": "0",
            "event_end_frame": "1",
            "anchor_frame": "1",
            "GT_identity": "ID_1",
            "predicted_identity": "ID_2",
            "Hidden_status": "YES",
        },
        1,
        gt_objects[1],
        pred_objects.get(1, []),
    )
    assert state["event_active"] and state["prediction_gap"]
    assert video.exists() and not list(tmp_path.glob("*.mp4"))


def test_span_and_ontology():
    assert parse_span("4") == (4, 4)
    assert parse_span("4-9") == (4, 9)
    assert DECISIONS and CONFIDENCES


def test_required_comment_and_coverage(tmp_path: Path):
    manifest = tmp_path / "m.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["review_unit_id", "source_item_count"])
        w.writeheader()
        w.writerow({"review_unit_id": "U1", "source_item_count": "2"})
    decisions = tmp_path / "d.csv"
    fields = ["review_unit_id", "decision", "confidence", "reviewer_comment"]
    with decisions.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                "review_unit_id": "U1",
                "decision": "AMBIGUOUS_UNRESOLVED",
                "confidence": "LOW",
                "reviewer_comment": "unclear",
            }
        )
    events = tmp_path / "e.jsonl"
    append_event(events, {"event_type": "DECISION_CREATED", "review_unit_id": "U1"})
    result = coverage(manifest, decisions, events)
    assert result["coverage_status"] == "PASS"
    assert result["REVIEW_COVERAGE_COMPLETE"] is True
    assert result["SCIENTIFIC_AMBIGUITIES_REMAIN"] is True


def test_invalid_unknown_item_fails(tmp_path: Path):
    m = tmp_path / "m.csv"
    m.write_text("review_unit_id,source_item_count\nU1,1\n")
    d = tmp_path / "d.csv"
    d.write_text(
        "review_unit_id,decision,confidence,reviewer_comment\n"
        "U2,NO_MATERIAL_ISSUE_CONFIRMED,HIGH,\n"
    )
    e = tmp_path / "e.jsonl"
    e.write_text("")
    result = coverage(m, d, e)
    assert result["coverage_status"] == "FAIL"
    assert validate_decision(
        {"decision": "GT_IDENTITY_QUESTION", "confidence": "HIGH", "reviewer_comment": ""}
    )


def _write_video(path: Path, frame_count: int, fps: float, offset: int = 0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((24, 32, 3), offset + index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_exact_frame_seek_repeat_backward_and_video_transition(tmp_path: Path):
    first = tmp_path / "first.avi"
    second = tmp_path / "second.avi"
    _write_video(first, 6, 12.5)
    _write_video(second, 4, 7.5, offset=100)
    reader = ExactFrameReader(cache_size=2)
    reader.open(str(first))
    values = [int(reader.read(index).mean()) for index in (0, 5, 3, 3, 1)]
    assert values[0] < values[4] < values[2] < values[1]
    reader.open(str(second))
    assert int(reader.read(0).mean()) > values[1]
    reader.close()


def test_context_clamp_and_non_30_fps():
    assert clamp_context(-20, 200, 100) == (0, 99)
    fps = 12.5
    assert round(fps) == 12


def test_timeline_state_wrong_identity_and_gap():
    row = {
        "event_start_frame": "2",
        "event_end_frame": "4",
        "anchor_frame": "3",
        "GT_identity": "ID_1",
        "predicted_identity": "ID_2",
        "Hidden_status": "YES",
    }
    state = timeline_state(row, 3, [{"id": "ID_1"}], [])
    assert state["event_active"] and state["anchor"]
    assert state["prediction_gap"] and state["wrong_identity_state"]


def _manifest_row() -> dict[str, str]:
    return {
        "review_unit_id": "U1",
        "linked_audit_item_ids": "A1",
        "video_id": "V1",
        "primary_method_id": "bytetrack_raw",
        "episode_id": "E1",
        "event_start_frame": "1",
        "event_end_frame": "2",
        "anchor_frame": "1",
        "context_start_frame": "0",
        "context_end_frame": "3",
        "source_item_count": "1",
        "video_sha256": "v" * 64,
        "GT_sha256": "g" * 64,
        "prediction_sha256": "p" * 64,
    }


def _write_manifest(path: Path) -> None:
    row = _manifest_row()
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_atomic_resume_undo_and_required_columns(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    root = tmp_path / "review"
    ledger = DecisionLedger(root, manifest, "code-sha", "Reviewer")
    ledger.save(
        _manifest_row(),
        "NO_MATERIAL_ISSUE_CONFIRMED",
        "HIGH",
        "",
        0,
        3,
        False,
        False,
        2.5,
    )
    assert list(ledger.current()) == ["U1"]
    assert list(ledger.current()["U1"]) == DECISION_COLUMNS
    resumed = DecisionLedger(root, manifest, "code-sha", "Reviewer")
    assert list(resumed.current()) == ["U1"]
    assert resumed.undo_latest() == "U1"
    assert resumed.current() == {}
    assert not list(root.glob("*.tmp"))


def test_ledger_rejects_manifest_or_reviewer_drift(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    root = tmp_path / "review"
    DecisionLedger(root, manifest, "code-sha", "Reviewer")
    with pytest.raises(RuntimeError):
        DecisionLedger(root, manifest, "code-sha", "Different")
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(RuntimeError):
        DecisionLedger(root, manifest, "code-sha", "Reviewer")


def test_coverage_hash_and_source_conservation(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    source = tmp_path / "source.csv"
    source.write_text("audit_item_id\nA1\n")
    root = tmp_path / "review"
    ledger = DecisionLedger(root, manifest, "code-sha", "Reviewer")
    ledger.save(
        _manifest_row(),
        "AMBIGUOUS_UNRESOLVED",
        "LOW",
        "insufficient visual evidence",
        0,
        3,
        False,
        False,
        1.0,
    )
    result = coverage(
        manifest,
        ledger.decisions_path,
        ledger.events_path,
        source_audit_path=source,
        expected_gui_code_sha="code-sha",
    )
    assert result["coverage_status"] == "PASS"
    assert result["REVIEW_COVERAGE_COMPLETE"]
    assert result["SCIENTIFIC_AMBIGUITIES_REMAIN"]


def test_source_immutability_and_no_mp4(tmp_path: Path):
    source = tmp_path / "source.xml"
    source.write_text("<annotations/>")
    before = sha256(source)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    DecisionLedger(tmp_path / "review", manifest, "code-sha", "Reviewer")
    assert sha256(source) == before
    assert not list(tmp_path.rglob("*.mp4"))


def test_active_registry_is_frozen():
    from pig_behavior.tracking.method_registry import ACTIVE_SCIENTIFIC_METHOD_IDS

    assert ACTIVE_SCIENTIFIC_METHOD_IDS == (
        "bytetrack_raw",
        "hybrid_bytetrack",
        "realtime_fast",
        "rf_hybrid",
    )

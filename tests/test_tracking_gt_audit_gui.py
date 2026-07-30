import csv
from pathlib import Path

from pig_behavior.tracking.gt_audit_review import (
    CONFIDENCES,
    DECISIONS,
    append_event,
    coverage,
    parse_cvat,
    parse_span,
    render_boxes,
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

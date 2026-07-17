from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_tracking_prefix_invariance import audit_prefix_invariance


def _write_xml(path: Path, boxes: list[tuple[int, str]]) -> None:
    rows = "".join(
        (
            f'<box frame="{frame}" xtl="{frame}.0" ytl="0.0" '
            f'xbr="10.0" ybr="10.0"><attribute name="ID">{fixed_id}</attribute>'
            "</box>"
        )
        for frame, fixed_id in boxes
    )
    path.write_text(
        f'<annotations><track id="0" label="Pig_1">{rows}</track></annotations>',
        encoding="utf-8",
    )


def _write_report(path: Path, processed_frames: int) -> None:
    path.write_text(
        json.dumps(
            {
                "processed_frames": processed_frames,
                "telemetry": {
                    "output_timing_contract": "causal_framewise",
                    "declared_delay_frames": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_prefix_invariance_passes_with_only_future_frames_added(tmp_path: Path) -> None:
    prefix_dir = tmp_path / "prefix"
    extended_dir = tmp_path / "extended"
    prefix_dir.mkdir()
    extended_dir.mkdir()
    prefix_xml = prefix_dir / "annotations_cvat_video_1_1.xml"
    extended_xml = extended_dir / "annotations_cvat_video_1_1.xml"
    _write_xml(prefix_xml, [(0, "ID_1"), (1, "ID_1")])
    _write_xml(extended_xml, [(0, "ID_1"), (1, "ID_1"), (2, "ID_1")])
    _write_report(prefix_dir / "tracking_quality_report.json", 2)
    _write_report(extended_dir / "tracking_quality_report.json", 3)

    audit = audit_prefix_invariance(
        prefix_xml,
        extended_xml,
        frame_exclusive=2,
        expected_timing_contract="causal_framewise",
        expected_delay_frames=0,
        artifact_roots=[tmp_path],
    )

    assert audit["status"] == "PASS"
    assert audit["compared_payload_count"] == 2
    assert audit["payloads_equal"] is True
    assert audit["mp4_count"] == 0


def test_prefix_invariance_fails_on_past_change_or_mp4(tmp_path: Path) -> None:
    prefix_dir = tmp_path / "prefix"
    extended_dir = tmp_path / "extended"
    prefix_dir.mkdir()
    extended_dir.mkdir()
    prefix_xml = prefix_dir / "annotations_cvat_video_1_1.xml"
    extended_xml = extended_dir / "annotations_cvat_video_1_1.xml"
    _write_xml(prefix_xml, [(0, "ID_1"), (1, "ID_1")])
    _write_xml(extended_xml, [(0, "ID_1"), (1, "ID_2"), (2, "ID_1")])
    _write_report(prefix_dir / "tracking_quality_report.json", 2)
    _write_report(extended_dir / "tracking_quality_report.json", 3)
    (tmp_path / "forbidden.mp4").write_bytes(b"not-a-video")

    audit = audit_prefix_invariance(
        prefix_xml,
        extended_xml,
        frame_exclusive=2,
        expected_timing_contract="causal_framewise",
        expected_delay_frames=0,
        artifact_roots=[tmp_path],
    )

    assert audit["status"] == "FAIL"
    assert "flushed_xml_payload_changed_with_future_frames" in audit["errors"]
    assert "generated_mp4_found" in audit["errors"]

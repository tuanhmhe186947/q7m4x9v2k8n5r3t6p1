from __future__ import annotations

from legacy_burst_recovery.timestamp_utils import build_timestamp_audit, build_timestamp_parse_debug, parse_times_txt


def write_times(tmp_path, text: str):
    path = tmp_path / "times.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_iso_datetime_lines_as_relative_seconds(tmp_path):
    path = write_times(
        tmp_path,
        "\n".join(
            [
                "2019-11-29T09:57:48.015615",
                "2019-11-29T09:57:48.170550",
                "2019-11-29T09:57:48.328109",
            ]
        ),
    )

    result = parse_times_txt(path)

    assert result.exists is True
    assert result.parse_error == ""
    assert len(result.timestamps) == 3
    assert result.timestamps[0] == 0.0
    assert 0.154 < result.timestamps[1] < 0.156
    assert "2019-11-29T09:57:48.015615" in result.first_lines_preview


def test_parse_frame_index_timestamp_whitespace_with_header(tmp_path):
    path = write_times(
        tmp_path,
        "\n".join(
            [
                "frame timestamp",
                "0 0.000",
                "1 0.200",
                "2 0.400",
            ]
        ),
    )

    result = parse_times_txt(path)

    assert result.timestamps == [0.0, 0.2, 0.4]


def test_parse_frame_index_timestamp_csv_with_header(tmp_path):
    path = write_times(
        tmp_path,
        "\n".join(
            [
                "frame,timestamp",
                "0,0.0",
                "1,0.5",
                "2,1.0",
            ]
        ),
    )

    result = parse_times_txt(path)

    assert result.timestamps == [0.0, 0.5, 1.0]


def test_parse_milliseconds_as_seconds(tmp_path):
    path = write_times(tmp_path, "\n".join(["0", "200", "400", "600"]))

    result = parse_times_txt(path)

    assert result.timestamps == [0.0, 0.2, 0.4, 0.6]


def test_existing_unparseable_file_reports_parse_failed(tmp_path):
    path = write_times(tmp_path, "\n".join(["frame timestamp", "bad row", "still bad"]))

    result = parse_times_txt(path)

    assert result.exists is True
    assert result.timestamps == []
    assert result.parse_error.startswith("parse_failed:")


def test_missing_file_reports_missing(tmp_path):
    result = parse_times_txt(tmp_path / "missing_times.txt")

    assert result.exists is False
    assert result.timestamps == []
    assert result.parse_error == "missing_file"


def test_production_timestamp_audit_for_iso_datetime_file(tmp_path):
    lines = [
        "2019-11-29T09:57:48.015615",
        "2019-11-29T09:57:48.170550",
        "2019-11-29T09:57:48.328109",
    ]
    path = write_times(tmp_path, "\n".join(lines))
    parsed = parse_times_txt(path)
    source_rows = [
        {
            "source_video_original": "source/color.mp4",
            "source_video_resolved": "source/color.mp4",
            "source_folder": "source",
            "times_txt_path": str(path),
            "times_txt_exists": parsed.exists,
            "times_txt_parse_error": parsed.parse_error,
            "times_txt_first_lines_preview": parsed.first_lines_preview,
            "times_txt_file_size_bytes": parsed.file_size_bytes,
            "times_txt_detected_format": parsed.detected_format,
            "times_txt_first_datetime": parsed.first_datetime,
            "times_txt_last_datetime": parsed.last_datetime,
            "timestamps": parsed.timestamps,
            "num_color_frames": 3,
        }
    ]

    audit = build_timestamp_audit(source_rows)
    debug = build_timestamp_parse_debug(source_rows)

    assert audit.loc[0, "num_timestamps"] == 3
    assert audit.loc[0, "timestamp_status"] == "ok"
    assert audit.loc[0, "first_timestamp_sec"] == 0.0
    assert 0.312 < audit.loc[0, "last_timestamp_sec"] < 0.313
    assert audit.loc[0, "estimated_real_fps"] > 6.0
    assert debug.loc[0, "detected_format"] == "iso_datetime_per_line"
    assert debug.loc[0, "first_datetime"] == lines[0]
    assert debug.loc[0, "last_datetime"] == lines[-1]

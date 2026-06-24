from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimesParseResult:
    timestamps: list[float]
    exists: bool
    parse_error: str
    first_lines_preview: str
    parsed_examples: list[float]
    file_size_bytes: int | None = None
    detected_format: str = ""
    first_datetime: str = ""
    last_datetime: str = ""


NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _first_non_empty_lines(path: Path, limit: int = 10) -> list[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            lines.append(text)
            if len(lines) >= limit:
                break
    return lines


def _split_line(line: str) -> list[str]:
    if "," in line:
        return [token.strip() for token in next(csv.reader([line])) if token.strip()]
    return [token.strip() for token in re.split(r"\s+", line.strip()) if token.strip()]


def _parse_numeric_token(token: str) -> float | None:
    token = token.strip()
    if not NUMERIC_RE.match(token):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _parse_datetime_token(token: str) -> float | None:
    text = token.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _parse_timestamp_token(token: str) -> tuple[float | None, str]:
    numeric = _parse_numeric_token(token)
    if numeric is not None:
        return numeric, "numeric"
    parsed_datetime = _parse_datetime_token(token)
    if parsed_datetime is not None:
        return parsed_datetime, "datetime"
    return None, ""


def _choose_timestamp_from_tokens(tokens: list[str], row_index: int) -> tuple[float | None, str, str]:
    parsed: list[tuple[int, float, str, str]] = []
    for idx, token in enumerate(tokens):
        value, kind = _parse_timestamp_token(token)
        if value is not None:
            parsed.append((idx, value, kind, token))
    if not parsed:
        return None, "", ""

    if len(parsed) >= 2:
        first_idx, first_value, first_kind, _first_token = parsed[0]
        _second_idx, second_value, second_kind, second_token = parsed[1]
        if first_idx == 0 and first_kind == "numeric" and abs(first_value - row_index) < 1e-6:
            return second_value, second_kind, second_token

    for _idx, value, kind, token in parsed:
        return value, kind, token
    return None, "", ""


def _normalise_to_seconds(values: list[float], kinds: list[str]) -> list[float]:
    if not values:
        return []
    normalised = list(values)
    finite_deltas = [
        abs(normalised[idx] - normalised[idx - 1])
        for idx in range(1, len(normalised))
        if np.isfinite(normalised[idx]) and np.isfinite(normalised[idx - 1])
    ]
    median_delta = float(np.median(finite_deltas)) if finite_deltas else 0.0

    # Numeric camera logs are often in milliseconds. ISO datetimes are parsed as epoch seconds.
    if any(kind == "datetime" for kind in kinds):
        first = normalised[0]
        return [value - first for value in normalised]
    if median_delta > 10.0 or max(abs(value) for value in normalised) > 1_000_000:
        normalised = [value / 1000.0 for value in normalised]
    if normalised and abs(normalised[0]) > 100_000:
        first = normalised[0]
        normalised = [value - first for value in normalised]
    return normalised


def parse_times_txt(path: str | Path, preview_lines: int = 10) -> TimesParseResult:
    path = Path(path)
    first_lines = _first_non_empty_lines(path, preview_lines)
    if not path.exists():
        return TimesParseResult([], False, "missing_file", "", [], None, "missing")

    raw_values: list[float] = []
    kinds: list[str] = []
    raw_timestamp_tokens: list[str] = []
    skipped_non_empty = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        row_index = 0
        for line in handle:
            text = line.strip()
            if not text:
                continue
            tokens = _split_line(text)
            value, kind, raw_token = _choose_timestamp_from_tokens(tokens, row_index)
            if value is None:
                skipped_non_empty += 1
                # Treat unparsable lines as headers/comments and keep the parsed data index stable.
                continue
            raw_values.append(value)
            kinds.append(kind)
            raw_timestamp_tokens.append(raw_token)
            row_index += 1

    timestamps = _normalise_to_seconds(raw_values, kinds)
    file_size = path.stat().st_size
    if kinds and all(kind == "datetime" for kind in kinds):
        detected_format = (
            "iso_datetime_per_line"
            if all(len(_split_line(line)) == 1 for line in first_lines)
            else "datetime_column"
        )
    elif kinds and any(kind == "datetime" for kind in kinds):
        detected_format = "mixed_datetime_numeric_columns"
    elif kinds and len(first_lines) > 0 and any("," in line for line in first_lines):
        detected_format = "csv_numeric"
    elif kinds:
        detected_format = "numeric_per_line_or_whitespace"
    else:
        detected_format = "unknown"

    if not timestamps:
        return TimesParseResult(
            [],
            True,
            f"parse_failed: no timestamp tokens found; skipped_non_empty={skipped_non_empty}",
            "\n".join(first_lines),
            [],
            file_size,
            "unknown",
        )
    return TimesParseResult(
        timestamps,
        True,
        "",
        "\n".join(first_lines),
        timestamps[:10],
        file_size,
        detected_format,
        raw_timestamp_tokens[0] if kinds and kinds[0] == "datetime" else "",
        raw_timestamp_tokens[-1] if kinds and kinds[-1] == "datetime" else "",
    )


def read_times_txt(path: str | Path) -> list[float]:
    return parse_times_txt(path).timestamps


def diagnostic_times_preview(path: str | Path, parsed_limit: int = 10) -> dict[str, object]:
    result = parse_times_txt(path, preview_lines=10)
    return {
        "path": str(path),
        "exists": result.exists,
        "parse_error": result.parse_error,
        "first_lines": result.first_lines_preview.splitlines(),
        "parsed_examples": result.parsed_examples[:parsed_limit],
        "num_timestamps": len(result.timestamps),
        "detected_format": result.detected_format,
        "first_datetime": result.first_datetime,
        "last_datetime": result.last_datetime,
    }


def timestamp_at(timestamps: list[float], frame_index: int) -> float | None:
    if 0 <= frame_index < len(timestamps):
        return timestamps[frame_index]
    return None


def timestamp_status(num_color_frames: int | None, num_timestamps: int, exists: bool, parse_error: str = "") -> str:
    if not exists:
        return "missing"
    if num_timestamps == 0:
        return "parse_failed" if parse_error else "parse_failed"
    if num_color_frames is None or num_color_frames < 0:
        return "video_unavailable"
    return "ok" if num_color_frames == num_timestamps else "mismatch"


def estimated_real_fps(timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return None
    return (len(timestamps) - 1) / duration


def build_timestamp_audit(source_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in source_rows:
        timestamps = row.get("timestamps") or []
        num_color = row.get("num_color_frames")
        exists = bool(row.get("times_txt_exists"))
        parse_error = str(row.get("times_txt_parse_error") or "")
        status = timestamp_status(
            int(num_color) if num_color is not None else None,
            len(timestamps),
            exists,
            parse_error,
        )
        rows.append(
            {
                "source_video_original": row.get("source_video_original", ""),
                "source_video_resolved": row.get("source_video_resolved", ""),
                "source_folder": row.get("source_folder", ""),
                "times_txt_path": row.get("times_txt_path", ""),
                "times_txt_exists": exists,
                "times_txt_parse_error": parse_error,
                "times_txt_first_lines_preview": row.get("times_txt_first_lines_preview", ""),
                "num_color_frames": num_color,
                "num_timestamps": len(timestamps),
                "first_timestamp": timestamps[0] if timestamps else np.nan,
                "last_timestamp": timestamps[-1] if timestamps else np.nan,
                "first_timestamp_sec": timestamps[0] if timestamps else np.nan,
                "last_timestamp_sec": timestamps[-1] if timestamps else np.nan,
                "estimated_real_fps": estimated_real_fps(timestamps),
                "timestamp_status": status,
            }
        )
    return pd.DataFrame(rows)


def build_timestamp_parse_debug(source_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in source_rows:
        timestamps = row.get("timestamps") or []
        rows.append(
            {
                "times_txt_path": row.get("times_txt_path", ""),
                "file_exists": bool(row.get("times_txt_exists")),
                "file_size_bytes": row.get("times_txt_file_size_bytes"),
                "first_10_raw_lines": row.get("times_txt_first_lines_preview", ""),
                "detected_format": row.get("times_txt_detected_format", ""),
                "parsed_count": len(timestamps),
                "first_datetime": row.get("times_txt_first_datetime", ""),
                "last_datetime": row.get("times_txt_last_datetime", ""),
                "first_timestamp_sec": timestamps[0] if timestamps else np.nan,
                "last_timestamp_sec": timestamps[-1] if timestamps else np.nan,
                "parse_error": row.get("times_txt_parse_error", ""),
            }
        )
    return pd.DataFrame(rows)

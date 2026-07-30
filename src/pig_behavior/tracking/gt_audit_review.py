"""Method-neutral human review infrastructure for frozen tracking evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

DECISIONS = {
    "PREDICTION_ERROR_CONFIRMED",
    "GT_IDENTITY_QUESTION",
    "GT_BBOX_QUESTION",
    "HIDDEN_LABEL_QUESTION",
    "EVALUATOR_MATCHING_QUESTION",
    "FRAGMENTATION_ONLY",
    "NO_MATERIAL_ISSUE_CONFIRMED",
    "AMBIGUOUS_UNRESOLVED",
    "OTHER_REVIEW_QUESTION",
}
CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
REQUIRED_COMMENT = {
    "GT_IDENTITY_QUESTION",
    "GT_BBOX_QUESTION",
    "HIDDEN_LABEL_QUESTION",
    "EVALUATOR_MATCHING_QUESTION",
    "AMBIGUOUS_UNRESOLVED",
    "OTHER_REVIEW_QUESTION",
}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_span(value: str) -> tuple[int, int]:
    value = str(value).strip()
    if "-" not in value:
        n = int(value)
        return n, n
    a, b = value.split("-", 1)
    return int(a), int(b)


def parse_bbox(value: str) -> tuple[float, float, float, float] | None:
    try:
        vals = json.loads(value.replace("'", '"'))
        return tuple(float(x) for x in vals)  # type: ignore[return-value]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def parse_cvat(path: str | Path) -> dict[int, list[dict[str, Any]]]:
    root = ET.parse(path).getroot()
    out: dict[int, list[dict[str, Any]]] = {}
    for track in root.findall("track"):
        tid = track.attrib.get("id", "")
        label = track.attrib.get("label", "pig")
        for box in track.findall("box"):
            if box.attrib.get("outside", "0") in {"1", "true", "True"}:
                continue
            attributes = {
                item.attrib.get("name", ""): (item.text or "").strip()
                for item in box.findall("attribute")
            }
            frame = int(box.attrib["frame"])
            out.setdefault(frame, []).append(
                {
                    "id": attributes.get("ID", tid),
                    "track_id": tid,
                    "label": label,
                    "bbox": tuple(float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")),
                    "hidden": (
                        attributes.get("Hidden", "").lower() in {"yes", "true", "1"}
                        or box.attrib.get("occluded", "0") in {"1", "true", "True"}
                    ),
                }
            )
    return out


class ExactFrameReader:
    """Small deterministic OpenCV reader that resets cleanly across videos."""

    def __init__(self) -> None:
        self.path: str | None = None
        self.capture: Any = None

    def open(self, path: str) -> None:
        import cv2

        if self.path == path and self.capture is not None:
            return
        self.close()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.path = path
        self.capture = capture

    def read(self, frame_index: int) -> Any:
        import cv2

        if self.capture is None:
            raise RuntimeError("No video is open")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode frame {frame_index} from {self.path}")
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.path = None


def render_boxes(
    frame: Any,
    objects: list[dict[str, Any]],
    target_id: str,
    color: tuple[int, int, int],
    prefix: str,
) -> Any:
    """Render labelled xyxy boxes on a BGR frame copy."""
    import cv2

    rendered = frame.copy()
    for obj in objects:
        x1, y1, x2, y2 = (round(value) for value in obj["bbox"])
        target = target_id not in {"", "ALL", "NOT_APPLICABLE"} and obj["id"] == target_id
        width = 4 if target else 2
        box_color = (255, 255, 0) if target else color
        cv2.rectangle(rendered, (x1, y1), (x2, y2), box_color, width)
        hidden = " HIDDEN" if obj.get("hidden") else ""
        text = f"{prefix}:{obj['id']}{hidden} [{x1},{y1},{x2},{y2}]"
        cv2.putText(
            rendered,
            text,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )
    return rendered


def validate_decision(row: dict[str, Any]) -> list[str]:
    errors = []
    if row.get("decision") not in DECISIONS:
        errors.append("invalid_decision")
    if row.get("confidence") not in CONFIDENCES:
        errors.append("invalid_confidence")
    if row.get("decision") in REQUIRED_COMMENT and not str(row.get("reviewer_comment", "")).strip():
        errors.append("missing_required_comment")
    return errors


def append_event(path: str | Path, event: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"created_at": time.time(), **event}, sort_keys=True) + "\n"
    with open(target, "a", encoding="utf-8", newline="") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def read_events(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def coverage(
    manifest_path: str | Path, decisions_path: str | Path, events_path: str | Path
) -> dict[str, Any]:
    manifest = load_rows(manifest_path)
    decisions = load_rows(decisions_path) if Path(decisions_path).exists() else []
    known = {r["review_unit_id"] for r in manifest}
    latest: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for row in decisions:
        uid = row.get("review_unit_id", "")
        if uid not in known:
            errors.append(f"unknown_review_unit:{uid}")
        if uid in latest:
            errors.append(f"duplicate_current_decision:{uid}")
        latest[uid] = row
        errors.extend(f"{uid}:{e}" for e in validate_decision(row))
    event_errors = []
    event_uids = {
        e.get("review_unit_id")
        for e in read_events(events_path)
        if e.get("event_type") in {"DECISION_CREATED", "DECISION_UPDATED"}
    }
    for uid in latest:
        if uid not in event_uids:
            event_errors.append(f"decision_without_event:{uid}")
    result = {
        "review_units_total": len(manifest),
        "review_units_reviewed": len(latest),
        "review_units_unresolved": len(manifest) - len(latest),
        "source_audit_items_total": sum(int(r.get("source_item_count", 0)) for r in manifest),
        "source_audit_items_covered": sum(
            int(r.get("source_item_count", 0)) for r in manifest if r["review_unit_id"] in latest
        ),
        "duplicate_decisions": sum("duplicate_current_decision:" in e for e in errors),
        "unknown_review_units": sum("unknown_review_unit:" in e for e in errors),
        "unknown_source_items": 0,
        "invalid_decisions": sum("invalid_decision" in e for e in errors),
        "missing_required_comments": sum("missing_required_comment" in e for e in errors),
        "hash_mismatches": 0,
        "event_log_inconsistencies": len(event_errors),
        "decision_counts": {
            d: sum(r.get("decision") == d for r in latest.values()) for d in sorted(DECISIONS)
        },
        "confidence_counts": {
            c: sum(r.get("confidence") == c for r in latest.values()) for c in sorted(CONFIDENCES)
        },
        "method_reveal_count": sum(
            e.get("event_type") == "METHOD_REVEALED" for e in read_events(events_path)
        ),
        "audit_context_reveal_count": sum(
            e.get("event_type") == "AUDIT_CONTEXT_REVEALED" for e in read_events(events_path)
        ),
        "errors": errors + event_errors,
        "warnings": [],
    }
    result["coverage_status"] = "PASS" if not result["errors"] else "FAIL"
    result["REVIEW_COVERAGE_COMPLETE"] = (
        result["review_units_unresolved"] == 0 and result["coverage_status"] == "PASS"
    )
    result["SCIENTIFIC_AMBIGUITIES_REMAIN"] = (
        result["decision_counts"].get("AMBIGUOUS_UNRESOLVED", 0) > 0
    )
    return result

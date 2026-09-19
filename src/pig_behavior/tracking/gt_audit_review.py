"""Method-neutral human review infrastructure for frozen tracking evidence."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime, timezone
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
DECISION_COLUMNS = [
    "review_unit_id",
    "linked_audit_item_ids",
    "reviewer",
    "decision",
    "confidence",
    "reviewer_comment",
    "video_id",
    "primary_method_id",
    "episode_id",
    "event_start_frame",
    "event_end_frame",
    "anchor_frame",
    "reviewed_context_start",
    "reviewed_context_end",
    "method_revealed_before_decision",
    "audit_context_revealed_before_decision",
    "review_duration_seconds",
    "source_review_manifest_sha256",
    "video_sha256",
    "GT_sha256",
    "prediction_sha256",
    "GUI_code_sha",
    "created_at",
    "updated_at",
]


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


def atomic_write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


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


def identity_tokens(value: str) -> set[str]:
    if value in {"", "ALL", "NOT_APPLICABLE", "EPISODE_LEVEL", "RECIPROCAL_PAIRWISE_SWAP"}:
        return set()
    if value.startswith("(") or value.startswith("["):
        parsed = ast.literal_eval(value)
        return {str(item) for item in parsed}
    return {value}


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

    def __init__(self, cache_size: int = 24) -> None:
        self.path: str | None = None
        self.capture: Any = None
        self.cache_size = cache_size
        self.cache: OrderedDict[tuple[str, int], Any] = OrderedDict()

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
        self.cache.clear()

    def read(self, frame_index: int) -> Any:
        import cv2

        if self.capture is None:
            raise RuntimeError("No video is open")
        key = (str(self.path), int(frame_index))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode frame {frame_index} from {self.path}")
        decoded_index = int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        if decoded_index != int(frame_index):
            raise RuntimeError(
                f"Decoder returned frame {decoded_index}; requested {frame_index} from {self.path}"
            )
        self.cache[key] = frame.copy()
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.path = None
        self.cache.clear()


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
        target = obj["id"] in identity_tokens(target_id)
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


def actor_crop(frame: Any, objects: list[dict[str, Any]], target_id: str, margin: int = 80) -> Any:
    """Return a clamped context crop while preserving the full frame elsewhere."""
    targets = identity_tokens(target_id)
    target = next((obj for obj in objects if obj["id"] in targets), None)
    if target is None:
        return frame.copy()
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = target["bbox"]
    left = max(0, int(x1) - margin)
    top = max(0, int(y1) - margin)
    right = min(width, int(x2) + margin)
    bottom = min(height, int(y2) + margin)
    return frame[top:bottom, left:right].copy()


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def clamp_context(start: int, end: int, frame_count: int) -> tuple[int, int]:
    return max(0, start), min(frame_count - 1, end)


def timeline_state(
    row: dict[str, str],
    frame: int,
    gt_objects: list[dict[str, Any]],
    prediction_objects: list[dict[str, Any]],
) -> dict[str, Any]:
    gt_ids = {obj["id"] for obj in gt_objects}
    prediction_ids = {obj["id"] for obj in prediction_objects}
    target_gt = row.get("GT_identity", "")
    target_prediction = row.get("predicted_identity", "")
    return {
        "frame": frame,
        "event_active": int(row["event_start_frame"]) <= frame <= int(row["event_end_frame"]),
        "anchor": frame == int(row["anchor_frame"]),
        "gt_identity_present": target_gt in gt_ids,
        "prediction_identity_present": target_prediction in prediction_ids,
        "prediction_gap": target_prediction not in prediction_ids,
        "wrong_identity_state": (
            target_gt not in {"", "ALL", "NOT_APPLICABLE"}
            and target_prediction not in {"", "ALL", "NOT_APPLICABLE"}
            and target_gt != target_prediction
        ),
        "hidden_status": row.get("Hidden_status", ""),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    line = (
        json.dumps(
            {"created_at": time.time(), "event_id": str(uuid.uuid4()), **event},
            sort_keys=True,
        )
        + "\n"
    )
    with open(target, "a", encoding="utf-8", newline="") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def read_events(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class DecisionLedger:
    """Atomic latest-state CSV plus append-only event log and resume state."""

    def __init__(
        self, root: str | Path, manifest_path: str | Path, gui_code_sha: str, reviewer: str
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(manifest_path)
        self.manifest_sha = sha256(self.manifest_path)
        self.gui_code_sha = gui_code_sha
        self.reviewer = reviewer
        self.decisions_path = self.root / "tracking_gt_audit_decisions.csv"
        self.events_path = self.root / "tracking_gt_audit_decision_events.jsonl"
        self.session_path = self.root / "tracking_gt_audit_session.json"
        if not self.session_path.exists():
            atomic_write_json(
                self.session_path,
                {
                    "reviewer": reviewer,
                    "source_review_manifest_sha256": self.manifest_sha,
                    "GUI_code_sha": gui_code_sha,
                    "created_at": utc_now(),
                    "automatically_generated_decisions": 0,
                },
            )
            atomic_write_csv(self.decisions_path, [], DECISION_COLUMNS)
            self.events_path.touch(exist_ok=False)
            atomic_write_json(
                self.root / "tracking_gt_audit_coverage.json",
                {"coverage_status": "INCOMPLETE", "review_units_reviewed": 0},
            )
            atomic_write_json(
                self.root / "tracking_gt_audit_summary.json",
                {"summary_status": "INCOMPLETE", "reviewed": 0},
            )
        else:
            session = json.loads(self.session_path.read_text(encoding="utf-8"))
            if session.get("source_review_manifest_sha256") != self.manifest_sha:
                raise RuntimeError("Existing review root uses a different manifest")
            if session.get("reviewer") != reviewer:
                raise RuntimeError("Existing review root belongs to a different reviewer")

    def current(self) -> dict[str, dict[str, str]]:
        if not self.decisions_path.exists():
            return {}
        return {row["review_unit_id"]: row for row in load_rows(self.decisions_path)}

    def save(
        self,
        manifest_row: dict[str, str],
        decision: str,
        confidence: str,
        comment: str,
        context_start: int,
        context_end: int,
        method_revealed: bool,
        context_revealed: bool,
        duration_seconds: float,
    ) -> None:
        now = utc_now()
        current = self.current()
        uid = manifest_row["review_unit_id"]
        previous = current.get(uid)
        row = {
            "review_unit_id": uid,
            "linked_audit_item_ids": manifest_row["linked_audit_item_ids"],
            "reviewer": self.reviewer,
            "decision": decision,
            "confidence": confidence,
            "reviewer_comment": comment.strip(),
            "video_id": manifest_row["video_id"],
            "primary_method_id": manifest_row["primary_method_id"],
            "episode_id": manifest_row["episode_id"],
            "event_start_frame": manifest_row["event_start_frame"],
            "event_end_frame": manifest_row["event_end_frame"],
            "anchor_frame": manifest_row["anchor_frame"],
            "reviewed_context_start": context_start,
            "reviewed_context_end": context_end,
            "method_revealed_before_decision": "YES" if method_revealed else "NO",
            "audit_context_revealed_before_decision": "YES" if context_revealed else "NO",
            "review_duration_seconds": f"{duration_seconds:.3f}",
            "source_review_manifest_sha256": self.manifest_sha,
            "video_sha256": manifest_row["video_sha256"],
            "GT_sha256": manifest_row["GT_sha256"],
            "prediction_sha256": manifest_row["prediction_sha256"],
            "GUI_code_sha": self.gui_code_sha,
            "created_at": previous["created_at"] if previous else now,
            "updated_at": now,
        }
        errors = validate_decision(row)
        if errors:
            raise ValueError(", ".join(errors))
        current[uid] = row
        atomic_write_csv(self.decisions_path, list(current.values()), DECISION_COLUMNS)
        append_event(
            self.events_path,
            {
                "event_type": "DECISION_UPDATED" if previous else "DECISION_CREATED",
                "review_unit_id": uid,
                "reviewer": self.reviewer,
                "decision": decision,
                "human_initiated": True,
                "previous_decision_row": previous,
            },
        )

    def undo_latest(self) -> str | None:
        events = read_events(self.events_path)
        undone = {
            event.get("undone_event_id")
            for event in events
            if event.get("event_type") == "DECISION_UNDONE"
        }
        candidates = [
            event
            for event in events
            if event.get("event_type") in {"DECISION_CREATED", "DECISION_UPDATED"}
            and event.get("event_id") not in undone
        ]
        if not candidates:
            return None
        uid = str(candidates[-1]["review_unit_id"])
        current = self.current()
        previous = candidates[-1].get("previous_decision_row")
        if previous:
            current[uid] = previous
        else:
            current.pop(uid, None)
        atomic_write_csv(self.decisions_path, list(current.values()), DECISION_COLUMNS)
        append_event(
            self.events_path,
            {
                "event_type": "DECISION_UNDONE",
                "review_unit_id": uid,
                "reviewer": self.reviewer,
                "human_initiated": True,
                "undone_event_id": candidates[-1]["event_id"],
            },
        )
        return uid


def coverage(
    manifest_path: str | Path,
    decisions_path: str | Path,
    events_path: str | Path,
    source_audit_path: str | Path | None = None,
    expected_gui_code_sha: str | None = None,
    input_authority_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_rows(manifest_path)
    decisions = load_rows(decisions_path) if Path(decisions_path).exists() else []
    known = {r["review_unit_id"] for r in manifest}
    linked = [
        item for row in manifest for item in row.get("linked_audit_item_ids", "").split(";") if item
    ]
    source_ids = (
        {row["audit_item_id"] for row in load_rows(source_audit_path)}
        if source_audit_path
        else set(linked)
    )
    latest: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    file_hash_cache: dict[str, str] = {}
    if input_authority_path:
        authority_path = Path(input_authority_path)
        if not authority_path.exists():
            errors.append("input_authority_missing")
        else:
            authority = json.loads(authority_path.read_text(encoding="utf-8"))
            if tuple(authority.get("active_methods", [])) != (
                "bytetrack_raw",
                "hybrid_bytetrack",
                "realtime_fast",
                "rf_hybrid",
            ):
                errors.append("active_method_authority_mismatch")
    for row in decisions:
        uid = row.get("review_unit_id", "")
        if uid not in known:
            errors.append(f"unknown_review_unit:{uid}")
        if uid in latest:
            errors.append(f"duplicate_current_decision:{uid}")
        if expected_gui_code_sha and row.get("source_review_manifest_sha256") != sha256(
            manifest_path
        ):
            errors.append(f"manifest_hash_mismatch:{uid}")
        if expected_gui_code_sha and row.get("GUI_code_sha") != expected_gui_code_sha:
            errors.append(f"gui_code_sha_mismatch:{uid}")
        manifest_row = next((item for item in manifest if item["review_unit_id"] == uid), None)
        if manifest_row:
            for field in ("video_sha256", "GT_sha256", "prediction_sha256"):
                if row.get(field) != manifest_row.get(field):
                    errors.append(f"{field}_mismatch:{uid}")
        latest[uid] = row
        errors.extend(f"{uid}:{e}" for e in validate_decision(row))
    event_errors = []
    events = read_events(events_path)
    event_uids = {
        event.get("review_unit_id")
        for event in events
        if event.get("event_type") in {"DECISION_CREATED", "DECISION_UPDATED"}
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
        "unknown_source_items": len(set(linked) - source_ids),
        "invalid_decisions": sum("invalid_decision" in e for e in errors),
        "missing_required_comments": sum("missing_required_comment" in e for e in errors),
        "hash_mismatches": sum("mismatch:" in error for error in errors),
        "event_log_inconsistencies": len(event_errors),
        "decision_counts": {
            d: sum(r.get("decision") == d for r in latest.values()) for d in sorted(DECISIONS)
        },
        "confidence_counts": {
            c: sum(r.get("confidence") == c for r in latest.values()) for c in sorted(CONFIDENCES)
        },
        "method_reveal_count": sum(e.get("event_type") == "METHOD_REVEALED" for e in events),
        "audit_context_reveal_count": sum(
            e.get("event_type") == "AUDIT_CONTEXT_REVEALED" for e in events
        ),
        "errors": errors + event_errors,
        "warnings": [],
    }
    if len(linked) != len(set(linked)):
        result["errors"].append("duplicate_source_item_mapping")
    if set(linked) != source_ids:
        result["errors"].append("source_item_coverage_mismatch")
    if any(not event.get("human_initiated", True) for event in events):
        result["errors"].append("automatically_generated_decision_event")
    for manifest_row in manifest:
        for path_field, hash_field in (
            ("video_path", "video_sha256"),
            ("GT_path", "GT_sha256"),
            ("prediction_path", "prediction_sha256"),
        ):
            if path_field not in manifest_row:
                continue
            path = manifest_row[path_field]
            if not Path(path).exists():
                result["errors"].append(f"missing_source_file:{path}")
                continue
            actual = file_hash_cache.setdefault(path, sha256(path))
            if actual != manifest_row[hash_field]:
                result["errors"].append(f"source_hash_mismatch:{path}")
    result["hash_mismatches"] = sum(
        "mismatch:" in error or "hash_mismatch:" in error for error in result["errors"]
    )
    result["coverage_status"] = "PASS" if not result["errors"] else "FAIL"
    result["REVIEW_COVERAGE_COMPLETE"] = (
        result["review_units_unresolved"] == 0 and result["coverage_status"] == "PASS"
    )
    result["SCIENTIFIC_AMBIGUITIES_REMAIN"] = (
        result["decision_counts"].get("AMBIGUOUS_UNRESOLVED", 0) > 0
    )
    return result

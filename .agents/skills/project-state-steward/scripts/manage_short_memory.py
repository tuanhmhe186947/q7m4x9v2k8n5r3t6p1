"""Atomically coordinate session-owned tasks in project short memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
import textwrap
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
MEMORY_RELATIVE = Path(".agents/memory/01_PROJECT_MEMORY_SHORT.md")
LOCK_RELATIVE = Path(".agents/runtime/short_memory.lock")
TASK_HISTORY_RELATIVE = Path(".agents/memory/managed_task_history")
CONCURRENCY_SCHEMA = "atomic-v1"
TASK_ARCHIVE_SCHEMA = "managed-task-archive-v1"
MAX_LEASE_SECONDS = 86400
DEFAULT_LEASE_SECONDS = MAX_LEASE_SECONDS
MAX_ACTIVE_TASK_LINES = 1200
RUNTIME_SESSION_ENV = "CODEX_THREAD_ID"
ADMIN_TAKEOVER_CONFIRMATION = "USER_AUTHORIZED_ADMIN_TAKEOVER"

TASK_ID_RE = re.compile(r"^[A-Z][A-Z0-9-]*-\d{8}-\d{2}$")
STEP_ID_RE = re.compile(r"^[A-Z][A-Z0-9-]*-\d+$")
TASK_HEADING_RE = re.compile(
    r"^###\s+(?P<task_id>[A-Z][A-Z0-9-]*-\d{8}-\d{2})\s+-\s+.+\r?$",
    re.MULTILINE,
)
STEP_LINE_RE = re.compile(
    r"^- \[(?P<mark>[ xX])\] `(?P<step_id>[A-Z][A-Z0-9-]*-\d+)` "
    r"`\[(?P<status>TODO|IN_PROGRESS|BLOCKED|DONE|DEFERRED|CANCELLED)\]` "
    r"(?P<summary>.+)\r?$"
)
OPENED_RE = re.compile(r"^- Opened:\s*`(?P<opened>\d{4}-\d{2}-\d{2})", re.MULTILINE)
STATUS_RE = re.compile(
    r"^- Status:\s*`(?P<status>TODO|IN_PROGRESS|BLOCKED|DONE|DEFERRED|CANCELLED)`\.\r?$",
    re.MULTILINE,
)
HASH_LINE_RE = re.compile(
    r"(^- Block SHA256:\s*`)[0-9a-fA-F]{64}(`\.\r?$)",
    re.MULTILINE,
)
LIFECYCLE_RE = re.compile(
    r"Opened:\s*`(?P<opened>\d{4}-\d{2}-\d{2})`.*?"
    r"Expires:\s*`(?P<expires>[^`]+)`",
    re.DOTALL,
)
LEGACY_ALLOWLIST_RE = re.compile(
    r"^- Legacy unmanaged task IDs:[^\r\n]*(?:\r?\n  [^\r\n]*)*",
    re.MULTILINE,
)
OWNER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CHECKLIST_STATES = {
    "TODO",
    "IN_PROGRESS",
    "BLOCKED",
    "DONE",
    "DEFERRED",
    "CANCELLED",
}
TERMINAL_STATES = {"DONE", "CANCELLED"}
TRANSITIONS = {
    "TODO": {"IN_PROGRESS", "DEFERRED", "CANCELLED"},
    "IN_PROGRESS": {"DONE", "BLOCKED", "DEFERRED", "CANCELLED"},
    "BLOCKED": {"IN_PROGRESS", "DEFERRED", "CANCELLED"},
    "DEFERRED": {"IN_PROGRESS", "CANCELLED"},
    "DONE": set(),
    "CANCELLED": set(),
}
MANAGED_LABELS = (
    "Concurrency",
    "Owner session",
    "Owner runtime session",
    "Owner token SHA256",
    "Worktree",
    "Revision",
    "Lease expires",
    "Block SHA256",
    "Previous owner",
    "Ownership reason",
)


class LedgerError(RuntimeError):
    """A fail-closed task-ledger error with recovery guidance."""

    def __init__(
        self,
        code: str,
        root_cause_hint: str,
        safe_retry: str,
        stop_condition: str,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.root_cause_hint = root_cause_hint
        self.safe_retry = safe_retry
        self.stop_condition = stop_condition


def _project_timezone() -> timezone:
    try:
        return ZoneInfo("Asia/Saigon")
    except Exception:
        return timezone(timedelta(hours=7))


def _now(value: datetime | None = None) -> datetime:
    current = value or datetime.now(_project_timezone())
    if current.tzinfo is None:
        return current.replace(tzinfo=_project_timezone())
    return current.astimezone(_project_timezone())


def _iso_seconds(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _clean_scalar(value: str, label: str, maximum: int = 240) -> str:
    cleaned = " ".join(value.split())
    if not cleaned or "`" in cleaned or len(cleaned) > maximum:
        raise LedgerError(
            f"invalid_{label}",
            f"{label} must be a nonempty single-line value without backticks.",
            f"Provide a shorter valid {label}.",
            "Stop until the task metadata is unambiguous.",
        )
    return cleaned


def _validate_owner(owner_session: str) -> str:
    owner = _clean_scalar(owner_session, "owner_session", 128)
    if not OWNER_ID_RE.fullmatch(owner):
        raise LedgerError(
            "invalid_owner_session",
            "Owner session must be a stable platform thread ID or generated ID.",
            "Use an 8-128 character alphanumeric, dot, colon, underscore, or dash ID.",
            "Do not create or mutate a task without a stable session identity.",
        )
    return owner


def _validate_runtime_session(runtime_session: str | None) -> str | None:
    if runtime_session is None or not runtime_session.strip():
        return None
    return _validate_owner(runtime_session)


def _environment_runtime_session() -> str | None:
    return _validate_runtime_session(os.getenv(RUNTIME_SESSION_ENV))


def _validate_lease_seconds(value: int) -> int:
    if value < 1 or value > MAX_LEASE_SECONDS:
        raise LedgerError(
            "invalid_lease_seconds",
            f"Lease duration must be between 1 and {MAX_LEASE_SECONDS} seconds.",
            "Choose a bounded lease and renew it before a long phase.",
            "Do not use an unbounded ownership lease.",
        )
    return value


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_coordination_root(worktree: Path) -> Path:
    """Resolve and verify the canonical shared Git coordination root."""
    resolved_worktree = _resolve_path(worktree)
    if not resolved_worktree.is_dir():
        raise LedgerError(
            "coordination_root_unresolved",
            f"Worktree does not exist: {resolved_worktree}.",
            "Use a registered Git worktree with the canonical short-memory ledger.",
            "Do not create a worktree-local shadow ledger.",
        )
    result = subprocess.run(
        ["git", "-C", str(resolved_worktree), "rev-parse", "--git-common-dir"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise LedgerError(
            "coordination_root_unresolved",
            "Git could not resolve the shared common directory for the worktree.",
            "Run from a registered Git worktree or specify the canonical root.",
            "Do not infer active coordination state from a checked-out snapshot.",
        )
    common_value = result.stdout.strip()
    if not common_value:
        raise LedgerError(
            "coordination_root_unresolved",
            "Git returned an empty shared common-directory path.",
            "Repair the Git worktree registration before retrying.",
            "Do not continue with an ambiguous coordination root.",
        )
    common = Path(common_value)
    if not common.is_absolute():
        common = (resolved_worktree / common).resolve()
    if common.name != ".git" or not common.is_dir():
        raise LedgerError(
            "coordination_root_invalid",
            f"Git common directory is not a canonical .git directory: {common}.",
            "Repair the Git worktree topology before retrying.",
            "Do not use a coordination root outside the registered project.",
        )
    candidate = common.parent.resolve()
    top_level = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if top_level.returncode != 0 or not top_level.stdout.strip():
        raise LedgerError(
            "coordination_root_invalid",
            f"The candidate coordination root is not a Git worktree: {candidate}.",
            "Repair the canonical worktree before retrying.",
            "Do not accept an unverified coordination root.",
        )
    if _resolve_path(top_level.stdout.strip()) != candidate:
        raise LedgerError(
            "coordination_root_unauthorized",
            "Git top-level resolution does not match the shared coordination root.",
            "Use the canonical worktree associated with the common Git directory.",
            "Do not route coordination state outside the authorized project root.",
        )
    candidate_common = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--git-common-dir"],
        check=False,
        capture_output=True,
        text=True,
    )
    candidate_common_value = candidate_common.stdout.strip()
    if candidate_common.returncode != 0 or not candidate_common_value:
        raise LedgerError(
            "coordination_root_invalid",
            "The candidate root cannot resolve its shared Git common directory.",
            "Repair the canonical worktree before retrying.",
            "Do not accept an unverified coordination root.",
        )
    candidate_common_path = Path(candidate_common_value)
    if not candidate_common_path.is_absolute():
        candidate_common_path = (candidate / candidate_common_path).resolve()
    if candidate_common_path != common:
        raise LedgerError(
            "coordination_root_unauthorized",
            "The candidate root does not share the worktree Git common directory.",
            "Use the root registered for this Git worktree set.",
            "Do not route coordination state to another project.",
        )
    memory_path = (candidate / MEMORY_RELATIVE).resolve()
    try:
        memory_path.relative_to(candidate)
    except ValueError as exc:
        raise LedgerError(
            "coordination_root_unauthorized",
            "The active short-memory path escapes the canonical root.",
            "Repair the manager path constants before retrying.",
            "Do not follow an escaped active-ledger path.",
        ) from exc
    if not memory_path.is_file():
        raise LedgerError(
            "short_memory_missing",
            f"Short memory does not exist at {memory_path}.",
            "Restore the canonical active ledger through the standard manager.",
            "Do not create a second shadow ledger.",
        )
    return candidate


def discover_coordination_root(worktree: Path) -> Path:
    """Preserve manager fallback behavior for callers without Git topology."""
    try:
        return resolve_coordination_root(worktree)
    except LedgerError:
        return ROOT


def _active_section_bounds(text: str) -> tuple[int, int]:
    heading = re.search(r"^## Active Task Checklist\r?$", text, re.MULTILINE)
    if heading is None:
        raise LedgerError(
            "active_checklist_missing",
            "Short memory has no Active Task Checklist section.",
            "Restore the current short-memory schema before retrying.",
            "Do not write task state into an unknown Markdown layout.",
        )
    content_start = heading.end()
    if text.startswith("\n", content_start):
        content_start += 1
    following = re.search(r"^##\s+", text[content_start:], re.MULTILINE)
    end = len(text) if following is None else content_start + following.start()
    return content_start, end


def _section_bounds(text: str, heading_text: str) -> tuple[int, int]:
    heading = re.search(
        rf"^## {re.escape(heading_text)}\r?$",
        text,
        re.MULTILINE,
    )
    if heading is None:
        raise LedgerError(
            "section_missing",
            f"Short memory has no {heading_text} section.",
            "Restore the current short-memory schema before retrying.",
            "Do not guess where lifecycle content should be written.",
        )
    following = re.search(r"^##\s+", text[heading.end() :], re.MULTILINE)
    end = len(text) if following is None else heading.end() + following.start()
    return heading.start(), end


def _short_lifecycle(text: str) -> tuple[datetime, datetime]:
    match = LIFECYCLE_RE.search(text)
    if match is None:
        raise LedgerError(
            "short_lifecycle_invalid",
            "Short memory has no parseable Opened and Expires fields.",
            "Restore the lifecycle fields before retrying.",
            "Do not infer the rollover date from task IDs.",
        )
    opened = datetime.fromisoformat(match.group("opened")).replace(
        tzinfo=_project_timezone()
    )
    expires = datetime.fromisoformat(match.group("expires"))
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=_project_timezone())
    return opened, expires


def task_spans(text: str) -> list[dict[str, Any]]:
    section_start, section_end = _active_section_bounds(text)
    section = text[section_start:section_end]
    matches = list(TASK_HEADING_RE.finditer(section))
    spans: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = section_start + match.start()
        end = (
            section_start + matches[index + 1].start()
            if index + 1 < len(matches)
            else section_end
        )
        spans.append(
            {
                "task_id": match.group("task_id"),
                "start": start,
                "end": end,
                "block": text[start:end],
            }
        )
    return spans


def _task_span(text: str, task_id: str) -> dict[str, Any]:
    matches = [span for span in task_spans(text) if span["task_id"] == task_id]
    if len(matches) != 1:
        code = "task_missing" if not matches else "task_id_collision"
        raise LedgerError(
            code,
            f"Expected exactly one task block for {task_id}; found {len(matches)}.",
            "Inspect the active checklist and reconcile task IDs before retrying.",
            "Do not choose a block or overwrite either version automatically.",
        )
    return matches[0]


def _line_value(block: str, label: str) -> str | None:
    match = re.search(
        rf"^- {re.escape(label)}:\s*`(?P<value>[^`]+)`\.\r?$",
        block,
        re.MULTILINE,
    )
    return match.group("value") if match else None


def raw_block_sha256(block: str) -> str:
    return _sha256_text(block)


def _archive_values(block: str) -> dict[str, str] | None:
    """Return integrity anchors for a compacted task, rejecting partial state."""
    labels = (
        "Archive reference",
        "Archive SHA256",
        "Archived content SHA256",
        "Pre-compaction revision",
        "Pre-compaction Block SHA256",
    )
    values = {label: _line_value(block, label) for label in labels}
    if not any(values.values()):
        return None
    missing = [label for label, value in values.items() if value is None]
    if missing:
        raise LedgerError(
            "archive_metadata_missing",
            f"Compacted task archive metadata is incomplete: {', '.join(missing)}.",
            "Restore the active block from the manager-controlled archive transition.",
            "Do not infer or manually complete archive metadata.",
        )
    archive_sha = values["Archive SHA256"] or ""
    content_sha = values["Archived content SHA256"] or ""
    if not HEX_SHA256_RE.fullmatch(archive_sha) or not HEX_SHA256_RE.fullmatch(content_sha):
        raise LedgerError(
            "archive_hash_invalid",
            "Compacted task archive hashes must be lowercase SHA-256 values.",
            "Inspect the manager-created archive and active integrity anchors.",
            "Do not accept malformed archive hashes.",
        )
    try:
        revision = int(values["Pre-compaction revision"] or "")
    except ValueError as exc:
        raise LedgerError(
            "archive_revision_invalid",
            "Pre-compaction revision must be a positive integer.",
            "Restore the manager-generated archive metadata.",
            "Do not guess archive lineage.",
        ) from exc
    if revision < 1 or not HEX_SHA256_RE.fullmatch(
        values["Pre-compaction Block SHA256"] or ""
    ):
        raise LedgerError(
            "archive_lineage_invalid",
            "Compacted task archive lineage is invalid.",
            "Inspect the manager-generated archive metadata.",
            "Do not continue from an unverifiable compacted task.",
        )
    return {
        "reference": values["Archive reference"] or "",
        "archive_sha256": archive_sha,
        "content_sha256": content_sha,
        "pre_revision": str(revision),
        "pre_block_sha256": values["Pre-compaction Block SHA256"] or "",
    }


def _archive_path(root: Path, task_id: str, revision: int) -> Path:
    return root / TASK_HISTORY_RELATIVE / task_id / f"revision-{revision:06d}.json"


def _archive_reference(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise LedgerError(
            "archive_path_outside_root",
            "Task archive path must remain within the coordination root.",
            "Use the manager-controlled task history location.",
            "Do not place managed history in an external or user-owned path.",
        ) from exc


def _prepare_task_archive(
    root: Path,
    *,
    task_id: str,
    block: str,
    metadata: dict[str, Any],
    timestamp: datetime,
) -> tuple[Path, str, dict[str, str]]:
    """Build a lossless archive record without creating persistent state yet."""
    path = _archive_path(root, task_id, metadata["revision"])
    if path.exists():
        raise LedgerError(
            "archive_already_exists",
            (
                "Immutable history already exists for "
                f"{task_id} revision {metadata['revision']}."
            ),
            (
                "Inspect whether the task has already been compacted or reconcile "
                "the failed transition."
            ),
            "Do not overwrite manager-controlled historical evidence.",
        )
    content_sha256 = raw_block_sha256(block)
    payload = {
        "archive_schema": TASK_ARCHIVE_SCHEMA,
        "archived_at": _iso_seconds(timestamp),
        "content": block,
        "content_sha256": content_sha256,
        "pre_compaction_block_sha256": metadata["block_sha256"],
        "pre_compaction_revision": metadata["revision"],
        "task_id": task_id,
    }
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    return path, serialized, {
        "reference": _archive_reference(root, path),
        "archive_sha256": _sha256_text(serialized),
        "content_sha256": content_sha256,
    }


def _write_task_archive(
    root: Path,
    *,
    task_id: str,
    block: str,
    metadata: dict[str, Any],
    timestamp: datetime,
) -> dict[str, str]:
    """Persist a prepared archive only after the compact transition preflight."""
    path, serialized, archive = _prepare_task_archive(
        root,
        task_id=task_id,
        block=block,
        metadata=metadata,
        timestamp=timestamp,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, serialized)
    return archive


def _verify_task_archive(root: Path, task_id: str, block: str) -> dict[str, Any] | None:
    """Verify the archive is lossless and anchored by the compact active block."""
    values = _archive_values(block)
    if values is None:
        return None
    path = (root / values["reference"]).resolve()
    expected_parent = (root / TASK_HISTORY_RELATIVE / task_id).resolve()
    try:
        path.relative_to(expected_parent)
    except ValueError as exc:
        raise LedgerError(
            "archive_reference_invalid",
            "Compacted task archive reference escapes its manager-controlled task history.",
            "Restore the active continuation from a verified manager checkpoint.",
            "Do not follow arbitrary archive paths.",
        ) from exc
    if path.suffix != ".json" or not path.is_file():
        raise LedgerError(
            "archive_missing",
            "Compacted task archive is unavailable.",
            "Restore the immutable archive before resuming the task.",
            "Do not infer history from the compact continuation.",
        )
    serialized = path.read_bytes().decode("utf-8")
    if not secrets.compare_digest(_sha256_text(serialized), values["archive_sha256"]):
        raise LedgerError(
            "archive_hash_mismatch",
            "Compacted task archive bytes do not match the active integrity anchor.",
            "Stop and restore the manager-controlled archive from verified storage.",
            "Do not resume from tampered or drifted history.",
        )
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise LedgerError(
            "archive_json_invalid",
            "Compacted task archive is not valid JSON.",
            "Restore the immutable manager-generated archive.",
            "Do not reconstruct historical content manually.",
        ) from exc
    if not isinstance(payload, dict) or payload.get("archive_schema") != TASK_ARCHIVE_SCHEMA:
        raise LedgerError(
            "archive_schema_invalid",
            "Compacted task archive does not use the supported archive schema.",
            "Restore the manager-generated archive.",
            "Do not accept an unversioned task-history record.",
        )
    content = payload.get("content")
    if payload.get("task_id") != task_id or not isinstance(content, str):
        raise LedgerError(
            "archive_content_invalid",
            "Compacted task archive does not identify the expected task content.",
            "Restore the correct manager-generated archive.",
            "Do not continue from mismatched historical evidence.",
        )
    if not secrets.compare_digest(raw_block_sha256(content), values["content_sha256"]):
        raise LedgerError(
            "archive_content_hash_mismatch",
            "Archived task content does not match its active content integrity anchor.",
            "Restore the immutable manager-generated archive.",
            "Do not resume from altered history.",
        )
    archived = parse_managed_task(content)
    if archived is None or archived["revision"] != int(values["pre_revision"]):
        raise LedgerError(
            "archive_history_invalid",
            "Archived task content is not a valid managed predecessor.",
            "Restore the manager-created historical block.",
            "Do not fabricate a predecessor revision.",
        )
    if not secrets.compare_digest(archived["block_sha256"], values["pre_block_sha256"]):
        raise LedgerError(
            "archive_chain_mismatch",
            "Archived predecessor block hash does not match the active chain anchor.",
            "Restore the matching manager-generated history record.",
            "Do not continue from a broken task hash chain.",
        )
    return {
        "archive_reference": values["reference"],
        "archive_sha256": values["archive_sha256"],
        "archived_content_sha256": values["content_sha256"],
        "pre_compaction_revision": archived["revision"],
        "pre_compaction_block_sha256": archived["block_sha256"],
    }


def _verified_archive_content(root: Path, task_id: str, block: str) -> str:
    """Return predecessor bytes only after the active archive chain verifies."""
    values = _archive_values(block)
    if values is None:
        raise LedgerError(
            "archive_missing",
            "Compaction repair requires an existing immutable archive.",
            "Use normal compaction for an unarchived task.",
            "Do not reconstruct predecessor content manually.",
        )
    _verify_task_archive(root, task_id, block)
    payload = json.loads((root / values["reference"]).read_bytes().decode("utf-8"))
    return payload["content"]


def managed_block_sha256(block: str) -> str:
    canonical, count = HASH_LINE_RE.subn(
        rf"\g<1>{'0' * 64}\g<2>",
        block,
        count=1,
    )
    if count != 1:
        raise LedgerError(
            "managed_hash_line_invalid",
            "Managed task must contain exactly one Block SHA256 field.",
            "Inspect or adopt the task through this manager.",
            "Do not update a managed block with malformed metadata.",
        )
    return _sha256_text(canonical)


def _parse_lease(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LedgerError(
            "lease_timestamp_invalid",
            "Lease expires is not an ISO-8601 timestamp.",
            "Reconcile the managed task metadata before retrying.",
            "Do not infer whether another owner is still active.",
        ) from exc
    if parsed.tzinfo is None:
        raise LedgerError(
            "lease_timezone_missing",
            "Lease expires must include a timezone offset.",
            "Reconcile the managed task metadata before retrying.",
            "Do not compare a timezone-naive ownership lease.",
        )
    return parsed


def parse_managed_task(
    block: str,
    *,
    verify_hash: bool = True,
) -> dict[str, Any] | None:
    if _line_value(block, "Concurrency") != CONCURRENCY_SCHEMA:
        return None
    required = {
        label: _line_value(block, label)
        for label in (
            "Owner session",
            "Owner token SHA256",
            "Worktree",
            "Revision",
            "Lease expires",
            "Block SHA256",
        )
    }
    missing = [label for label, value in required.items() if value is None]
    if missing:
        raise LedgerError(
            "managed_metadata_missing",
            f"Managed task is missing fields: {', '.join(missing)}.",
            "Restore metadata from an authoritative checkpoint.",
            "Do not mutate a partially managed task.",
        )
    owner = _validate_owner(required["Owner session"] or "")
    runtime_session = _validate_runtime_session(
        _line_value(block, "Owner runtime session")
    )
    token_hash = (required["Owner token SHA256"] or "").lower()
    block_hash = (required["Block SHA256"] or "").lower()
    if not HEX_SHA256_RE.fullmatch(token_hash):
        raise LedgerError(
            "owner_token_hash_invalid",
            "Owner token SHA256 is not a lowercase SHA-256 value.",
            "Restore owner metadata from an authoritative checkpoint.",
            "Do not accept plaintext or malformed ownership credentials.",
        )
    if not HEX_SHA256_RE.fullmatch(block_hash):
        raise LedgerError(
            "block_hash_invalid",
            "Block SHA256 is not a lowercase SHA-256 value.",
            "Restore the managed block through a validated recovery path.",
            "Do not bypass task-block integrity validation.",
        )
    try:
        revision = int(required["Revision"] or "")
    except ValueError as exc:
        raise LedgerError(
            "revision_invalid",
            "Revision must be a positive integer.",
            "Restore the last validated task revision.",
            "Do not guess a compare-and-swap revision.",
        ) from exc
    if revision < 1:
        raise LedgerError(
            "revision_invalid",
            "Revision must be at least one.",
            "Restore the last validated task revision.",
            "Do not guess a compare-and-swap revision.",
        )
    actual_hash = managed_block_sha256(block)
    if verify_hash and not secrets.compare_digest(actual_hash, block_hash):
        raise LedgerError(
            "block_hash_mismatch",
            "The managed task block changed outside its recorded checkpoint.",
            "Inspect the diff and reconcile ownership before adopting new bytes.",
            "Stop all effects for this task until the drift is resolved.",
        )
    return {
        "schema": CONCURRENCY_SCHEMA,
        "owner_session": owner,
        "owner_runtime_session": runtime_session,
        "owner_token_sha256": token_hash,
        "worktree": required["Worktree"],
        "revision": revision,
        "lease_expires": _parse_lease(required["Lease expires"] or ""),
        "block_sha256": block_hash,
        "previous_owner": _line_value(block, "Previous owner"),
        "ownership_reason": _line_value(block, "Ownership reason"),
    }


def validate_managed_block(block: str) -> list[str]:
    try:
        metadata = parse_managed_task(block)
    except LedgerError as exc:
        return [exc.code]
    return [] if metadata is not None else ["managed_schema_missing"]


def _strip_managed_lines(block: str) -> list[str]:
    prefixes = tuple(f"- {label}:" for label in MANAGED_LABELS)
    return [
        line
        for line in block.splitlines(keepends=True)
        if not line.startswith(prefixes)
    ]


def _with_managed_metadata(
    block: str,
    *,
    owner_session: str,
    owner_runtime_session: str | None,
    owner_token_sha256: str,
    worktree: Path,
    revision: int,
    lease_expires: datetime,
    previous_owner: str | None = None,
    ownership_reason: str | None = None,
) -> str:
    lines = _strip_managed_lines(block)
    opened_index = next(
        (index for index, line in enumerate(lines) if line.startswith("- Opened:")),
        None,
    )
    if opened_index is None:
        raise LedgerError(
            "task_opened_missing",
            "Task block has no Opened metadata anchor.",
            "Repair the task schema before adopting it.",
            "Do not insert managed metadata at a guessed location.",
        )
    nl = _newline(block)
    metadata = [
        f"- Concurrency: `{CONCURRENCY_SCHEMA}`.{nl}",
        f"- Owner session: `{owner_session}`.{nl}",
    ]
    if owner_runtime_session:
        runtime_session = _validate_owner(owner_runtime_session)
        metadata.append(f"- Owner runtime session: `{runtime_session}`.{nl}")
    metadata.extend(
        [
            f"- Owner token SHA256: `{owner_token_sha256}`.{nl}",
            f"- Worktree: `{worktree}`.{nl}",
            f"- Revision: `{revision}`.{nl}",
            f"- Lease expires: `{_iso_seconds(lease_expires)}`.{nl}",
            f"- Block SHA256: `{'0' * 64}`.{nl}",
        ]
    )
    if previous_owner:
        metadata.append(f"- Previous owner: `{previous_owner}`.{nl}")
    if ownership_reason:
        reason = _clean_scalar(ownership_reason, "ownership_reason", 64)
        metadata.append(f"- Ownership reason: `{reason}`.{nl}")
    lines[opened_index + 1 : opened_index + 1] = metadata
    candidate = "".join(lines)
    digest = managed_block_sha256(candidate)
    return HASH_LINE_RE.sub(
        rf"\g<1>{digest}\g<2>",
        candidate,
        count=1,
    )


def _append_ownership_audit(
    block: str,
    *,
    timestamp: datetime,
    action: str,
    from_owner: str,
    from_runtime_session: str | None,
    to_owner: str,
    to_runtime_session: str | None,
    prior_revision: int,
    prior_block_sha256: str,
    prior_worktree: Path,
    new_worktree: Path,
    reason: str,
    authority: str,
) -> str:
    clean_action = _clean_scalar(action, "audit_action", 48)
    clean_reason = _clean_scalar(reason, "audit_reason", 128)
    clean_authority = _clean_scalar(authority, "audit_authority", 160)
    payload = {
        "action": clean_action,
        "authority": clean_authority,
        "from_owner": from_owner,
        "from_runtime_session": from_runtime_session or "unbound",
        "new_worktree": str(new_worktree),
        "prior_block_sha256": prior_block_sha256,
        "prior_revision": prior_revision,
        "prior_worktree": str(prior_worktree),
        "reason": clean_reason,
        "timestamp": _iso_seconds(timestamp),
        "to_owner": to_owner,
        "to_runtime_session": to_runtime_session or "unbound",
    }
    event_id = _sha256_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    nl = _newline(block)
    lines = [f"- Ownership audit event: `{event_id}`.{nl}"]
    lines.extend(_wrap_detail("Timestamp", payload["timestamp"], nl))
    lines.extend(_wrap_detail("Action", clean_action, nl))
    lines.extend(_wrap_detail("From owner", from_owner, nl))
    lines.extend(
        _wrap_detail(
            "From runtime session",
            payload["from_runtime_session"],
            nl,
        )
    )
    lines.extend(_wrap_detail("To owner", to_owner, nl))
    lines.extend(
        _wrap_detail("To runtime session", payload["to_runtime_session"], nl)
    )
    lines.extend(_wrap_detail("Prior revision", str(prior_revision), nl))
    lines.extend(_wrap_detail("Prior block SHA256", prior_block_sha256, nl))
    lines.extend(_wrap_detail("Prior worktree", str(prior_worktree), nl))
    lines.extend(_wrap_detail("New worktree", str(new_worktree), nl))
    lines.extend(_wrap_detail("Reason", clean_reason, nl))
    lines.extend(_wrap_detail("Authority", clean_authority, nl))
    anchor = re.search(r"^- Acceptance:", block, re.MULTILINE)
    if anchor is None:
        raise LedgerError(
            "task_acceptance_missing",
            "Task block has no Acceptance anchor for its ownership audit.",
            "Repair the managed task schema before retrying recovery.",
            "Do not write an unaudited ownership transition.",
        )
    return block[: anchor.start()] + "".join(lines) + block[anchor.start() :]


def _task_status(block: str) -> str:
    match = STATUS_RE.search(block)
    if match is None:
        raise LedgerError(
            "task_status_missing",
            "Task block has no valid Status field.",
            "Repair the task schema before retrying.",
            "Do not derive execution state from malformed Markdown.",
        )
    return match.group("status")


def _step_records(block: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in block.splitlines():
        match = STEP_LINE_RE.fullmatch(line)
        if match:
            records.append(match.groupdict())
    return records


def _describe(task_id: str, block: str, now: datetime) -> dict[str, Any]:
    managed = parse_managed_task(block)
    steps = _step_records(block)
    result: dict[str, Any] = {
        "task_id": task_id,
        "managed": managed is not None,
        "task_status": _task_status(block),
        "raw_block_sha256": raw_block_sha256(block),
        "steps": [
            {
                "step_id": step["step_id"],
                "status": step["status"],
                "checked": step["mark"].lower() == "x",
                "summary": step["summary"],
            }
            for step in steps
        ],
    }
    if managed is not None:
        result.update(
            {
                "schema": managed["schema"],
                "owner_session": managed["owner_session"],
                "owner_runtime_session": managed["owner_runtime_session"],
                "owner_token_sha256": managed["owner_token_sha256"],
                "worktree": managed["worktree"],
                "revision": managed["revision"],
                "lease_expires": _iso_seconds(managed["lease_expires"]),
                "lease_active": now < managed["lease_expires"],
                "block_sha256": managed["block_sha256"],
                "ownership_audit_events": block.count(
                    "- Ownership audit event:"
                ),
            }
        )
        if managed["previous_owner"]:
            result["previous_owner"] = managed["previous_owner"]
        if managed["ownership_reason"]:
            result["ownership_reason"] = managed["ownership_reason"]
    return result


@contextmanager
def exclusive_file_lock(path: Path, timeout_seconds: float) -> Iterator[None]:
    """Acquire a process-scoped advisory lock released automatically on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b", buffering=0)
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise LedgerError(
                        "lock_timeout",
                        "Another process still owns the short-memory write lock.",
                        "Wait briefly, inspect the active owner, then retry once.",
                        "Do not bypass the lock or write the Markdown manually.",
                    ) from exc
                time.sleep(0.05)
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _atomic_write(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _other_blocks(text: str, excluded_task_id: str) -> dict[str, str]:
    return {
        span["task_id"]: span["block"]
        for span in task_spans(text)
        if span["task_id"] != excluded_task_id
    }


def _replace_task(text: str, span: dict[str, Any], new_block: str) -> str:
    task_id = span["task_id"]
    before = _other_blocks(text, task_id)
    updated = text[: span["start"]] + new_block + text[span["end"] :]
    after = _other_blocks(updated, task_id)
    if before != after:
        raise LedgerError(
            "other_task_drift",
            "A task mutation changed bytes belonging to another task block.",
            "Inspect parser boundaries and retry only after fixing the manager.",
            "Do not write the candidate short-memory file.",
        )
    return updated


def _insert_task(text: str, block: str, task_id: str) -> str:
    existing = {span["task_id"]: span["block"] for span in task_spans(text)}
    if task_id in existing:
        raise LedgerError(
            "task_id_collision",
            f"Task ID {task_id} already exists.",
            "Inspect the existing task and choose a new generated sequence.",
            "Do not merge two sessions under one task ID.",
        )
    section_start, section_end = _active_section_bounds(text)
    nl = _newline(text)
    section = text[section_start:section_end]
    if not existing and section.strip() == "- None.":
        prefix = text[:section_start] + nl + nl
    else:
        prefix = text[:section_end]
    separator = "" if prefix.endswith(nl + nl) else nl
    candidate = prefix + separator + block + text[section_end:]
    current = {span["task_id"]: span["block"] for span in task_spans(candidate)}
    for existing_id, existing_block in existing.items():
        if current.get(existing_id) != existing_block:
            raise LedgerError(
                "other_task_drift",
                "Creating a task changed an existing task block.",
                "Inspect parser boundaries and retry after fixing the manager.",
                "Do not write the candidate short-memory file.",
            )
    return candidate


def _replace_lifecycle(text: str, current: datetime) -> tuple[str, str]:
    expires = datetime.combine(
        current.date() + timedelta(days=1),
        datetime.min.time(),
        tzinfo=_project_timezone(),
    )
    opened_line = f"- Opened: `{current.date().isoformat()}`."
    expires_text = _iso_seconds(expires)
    expires_line = f"- Expires: `{expires_text}`."
    text, opened_count = re.subn(
        r"^- Opened:\s*`\d{4}-\d{2}-\d{2}`\.\r?$",
        opened_line,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text, expires_count = re.subn(
        r"^- Expires:\s*`[^`]+`\.\r?$",
        expires_line,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if opened_count != 1 or expires_count != 1:
        raise LedgerError(
            "short_lifecycle_update_failed",
            "Lifecycle fields are missing or ambiguous.",
            "Repair the Lifecycle section and retry rollover.",
            "Do not partially advance a short-memory date.",
        )
    text, allowlist_count = LEGACY_ALLOWLIST_RE.subn(
        "- Legacy unmanaged task IDs: none.",
        text,
    )
    if allowlist_count != 1:
        raise LedgerError(
            "legacy_allowlist_update_failed",
            "Legacy task allowlist is missing or ambiguous.",
            "Repair the Lifecycle allowlist and retry rollover.",
            "Do not leave retained tasks outside managed ownership.",
        )
    return text, expires_text


def _rollover_text(text: str, current: datetime) -> tuple[str, dict[str, Any]]:
    opened, expires = _short_lifecycle(text)
    if current < expires:
        raise LedgerError(
            "rollover_not_due",
            f"Short memory remains current until {_iso_seconds(expires)}.",
            "Continue using the existing day ledger.",
            "Do not create duplicate daily closeout sections.",
        )
    spans = task_spans(text)
    retained: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for span in spans:
        status = _task_status(span["block"])
        managed = parse_managed_task(span["block"])
        if status in TERMINAL_STATES:
            completed.append(span)
        elif managed is None:
            raise LedgerError(
                "rollover_unmanaged_open_task",
                f"Open task {span['task_id']} has no atomic owner metadata.",
                "Have its owning session adopt the task, then retry rollover.",
                "Do not move or duplicate an unowned active task automatically.",
            )
        else:
            retained.append(span)

    section_start, section_end = _active_section_bounds(text)
    nl = _newline(text)
    if retained:
        first_task_start = spans[0]["start"] if spans else section_end
        preamble = text[section_start:first_task_start]
        active_content = preamble + "".join(span["block"] for span in retained)
    else:
        active_content = f"{nl}{nl}- None.{nl}{nl}"
    candidate = text[:section_start] + active_content + text[section_end:]
    candidate, next_expires = _replace_lifecycle(candidate, current)

    previous_start, previous_end = _section_bounds(
        candidate,
        "Previous-Day Closeout",
    )
    completed_text = ", ".join(span["task_id"] for span in completed) or "none"
    retained_text = ", ".join(span["task_id"] for span in retained) or "none"
    previous_lines = [f"## Previous-Day Closeout{nl}", nl]
    previous_lines.append(
        f"- Source date: `{(current.date() - timedelta(days=1)).isoformat()}`.{nl}"
    )
    previous_lines.extend(_wrap_field("Completed", completed_text, nl))
    previous_lines.extend(
        _wrap_field(
            "Carried forward",
            f"{retained_text}; active tasks remain resume capsules in short memory",
            nl,
        )
    )
    previous_lines.append(f"- Purge after: `{next_expires}`.{nl}")
    previous_lines.append(nl)
    candidate = (
        candidate[:previous_start]
        + "".join(previous_lines)
        + candidate[previous_end:]
    )

    current_blocks = {span["task_id"]: span["block"] for span in task_spans(candidate)}
    for span in retained:
        if current_blocks.get(span["task_id"]) != span["block"]:
            raise LedgerError(
                "rollover_active_task_drift",
                f"Rollover changed active task {span['task_id']} bytes.",
                "Fix section boundary handling before retrying rollover.",
                "Do not invalidate an owner's revision/hash checkpoint.",
            )
    return candidate, {
        "source_opened": opened.date().isoformat(),
        "opened": current.date().isoformat(),
        "expires": next_expires,
        "retained_task_ids": [span["task_id"] for span in retained],
        "completed_task_ids": [span["task_id"] for span in completed],
    }


def _wrap_field(label: str, value: str, nl: str) -> list[str]:
    lines = textwrap.wrap(
        _clean_scalar(value, label.lower().replace(" ", "_"), 800),
        width=96,
        initial_indent=f"- {label}: ",
        subsequent_indent="  ",
    )
    return [line + nl for line in lines]


def _wrap_skills(skills: list[str], nl: str) -> list[str]:
    clean_skills = [_clean_scalar(skill, "skill", 80) for skill in skills]
    rendered = ", ".join(f"`{skill}`" for skill in clean_skills) + "."
    lines = textwrap.wrap(
        rendered,
        width=96,
        initial_indent="- Skills: ",
        subsequent_indent="  ",
    )
    return [line + nl for line in lines]


def _wrap_detail(label: str, value: str, nl: str) -> list[str]:
    lines = textwrap.wrap(
        _clean_scalar(value, label.lower().replace(" ", "_"), 800),
        width=96,
        initial_indent=f"  - {label}: ",
        subsequent_indent="    ",
    )
    return [line + nl for line in lines]


def _render_task(
    *,
    task_id: str,
    title: str,
    prompt: str,
    acceptance: str,
    skills: list[str],
    steps: list[dict[str, str]],
    active_step: str,
    opened: datetime,
    nl: str,
) -> str:
    if not TASK_ID_RE.fullmatch(task_id):
        raise LedgerError(
            "task_id_invalid",
            "Task ID does not match PREFIX-YYYYMMDD-NN.",
            "Generate a unique task ID for the current short-memory day.",
            "Do not create an unparseable task block.",
        )
    if f"-{opened:%Y%m%d}-" not in task_id:
        raise LedgerError(
            "task_id_wrong_day",
            "Task ID date does not match the current project-local date.",
            "Generate the task ID from Asia/Saigon local time.",
            "Do not add a stale-dated task to current short memory.",
        )
    clean_title = _clean_scalar(title, "title", 72)
    if not skills:
        raise LedgerError(
            "skills_missing",
            "Managed task creation requires selected skills.",
            "Select reasoning and execution skills before creating the task.",
            "Do not begin a material task without skill routing.",
        )
    step_ids = [step["step_id"] for step in steps]
    if not steps or len(step_ids) != len(set(step_ids)):
        raise LedgerError(
            "steps_invalid",
            "Task steps must be nonempty and have unique IDs.",
            "Provide stable STEP-N identifiers and next actions.",
            "Do not create an ambiguous checklist.",
        )
    if active_step not in step_ids:
        raise LedgerError(
            "active_step_missing",
            "The declared active step is not present in the task steps.",
            "Choose exactly one listed step as IN_PROGRESS.",
            "Do not start a task without an owned active checkpoint.",
        )
    lines = [f"### {task_id} - {clean_title}{nl}", nl]
    lines.extend(_wrap_field("Prompt", prompt, nl))
    lines.append(f"- Status: `IN_PROGRESS`.{nl}")
    lines.append(f"- Opened: `{_iso_seconds(opened)}`.{nl}")
    lines.extend(_wrap_field("Acceptance", acceptance, nl))
    lines.extend(_wrap_skills(skills, nl))
    for step in steps:
        step_id = step["step_id"]
        if not STEP_ID_RE.fullmatch(step_id):
            raise LedgerError(
                "step_id_invalid",
                f"Step ID {step_id} is invalid.",
                "Use a stable PREFIX-N step ID.",
                "Do not create an unparseable checklist step.",
            )
        summary = _clean_scalar(step["summary"], "step_summary", 68)
        next_action = _clean_scalar(step["next_action"], "next_action", 800)
        status = "IN_PROGRESS" if step_id == active_step else "TODO"
        line = f"- [ ] `{step_id}` `[{status}]` {summary}"
        if len(line) > 100:
            raise LedgerError(
                "step_line_too_long",
                f"Rendered step {step_id} exceeds the line-length contract.",
                "Shorten the step summary and keep detail in Next.",
                "Do not create overlong checklist lines.",
            )
        lines.append(line + nl)
        lines.extend(_wrap_detail("Next", next_action, nl))
    lines.append(nl)
    return "".join(lines)


def _replace_step_detail(
    block: str,
    step_id: str,
    new_status: str,
    detail_label: str,
    detail_value: str,
) -> str:
    nl = _newline(block)
    lines = block.splitlines(keepends=True)
    step_indexes: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        match = STEP_LINE_RE.fullmatch(line.rstrip("\r\n"))
        if match:
            step_indexes.append((index, match))
    targets = [(index, match) for index, match in step_indexes if match.group("step_id") == step_id]
    if len(targets) != 1:
        raise LedgerError(
            "step_missing_or_duplicate",
            f"Expected exactly one step {step_id}; found {len(targets)}.",
            "Inspect the owned task before retrying.",
            "Do not update an ambiguous step.",
        )
    index, match = targets[0]
    old_status = match.group("status")
    if new_status == old_status:
        raise LedgerError(
            "step_transition_noop",
            f"Step {step_id} is already {new_status}.",
            "Inspect current state and choose the next valid transition.",
            "Do not create revision noise with a no-op checkpoint.",
        )
    if new_status not in TRANSITIONS[old_status]:
        raise LedgerError(
            "step_transition_invalid",
            f"Transition {old_status} -> {new_status} is not allowed.",
            "Use the declared forward or recovery transition.",
            "Do not reopen terminal work or skip a checkpoint gate.",
        )
    following = [item for item, _ in step_indexes if item > index]
    end = min(following) if following else len(lines)
    trailing: list[str] = []
    if not following:
        while end > index + 1 and not lines[end - 1].strip():
            trailing.insert(0, lines[end - 1])
            end -= 1
    mark = "x" if new_status in TERMINAL_STATES else " "
    replacement = [
        f"- [{mark}] `{step_id}` `[{new_status}]` {match.group('summary')}{nl}"
    ]
    replacement.extend(_wrap_detail(detail_label, detail_value, nl))
    replacement.extend(trailing)
    lines[index:end] = replacement
    return "".join(lines)


def _derive_task_status(block: str) -> str:
    statuses = [step["status"] for step in _step_records(block)]
    if not statuses:
        raise LedgerError(
            "task_steps_missing",
            "Task has no parseable checklist steps.",
            "Repair the task before checkpointing it.",
            "Do not derive status from an empty checklist.",
        )
    if statuses.count("IN_PROGRESS") > 1:
        raise LedgerError(
            "multiple_in_progress",
            "A task cannot have more than one IN_PROGRESS step.",
            "Finish or block the current step before starting another.",
            "Do not write parallel active steps into one task.",
        )
    if "IN_PROGRESS" in statuses:
        return "IN_PROGRESS"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "TODO" in statuses:
        return "TODO"
    if "DEFERRED" in statuses:
        return "DEFERRED"
    if "DONE" in statuses:
        return "DONE"
    return "CANCELLED"


def _set_task_status(block: str, status_value: str) -> str:
    replacement, count = STATUS_RE.subn(
        f"- Status: `{status_value}`.",
        block,
        count=1,
    )
    if count != 1:
        raise LedgerError(
            "task_status_invalid",
            "Task must contain exactly one valid Status field.",
            "Repair the task schema before checkpointing.",
            "Do not guess which status line is authoritative.",
        )
    return replacement


def _compact_task_block(
    block: str,
    *,
    task_id: str,
    phase: str,
    blocker: str | None,
    resume_point: str,
    authority_refs: list[str],
    canonical_sha: str | None,
    archive: dict[str, str],
    skill_source_block: str | None = None,
) -> str:
    """Render a bounded continuation whose complete predecessor is archived."""
    heading = next(
        (line for line in block.splitlines() if line.startswith(f"### {task_id} - ")),
        None,
    )
    if heading is None:
        raise LedgerError(
            "task_heading_missing",
            "Managed task heading is unavailable for compaction.",
            "Restore the task block before attempting a manager transition.",
            "Do not invent a replacement task identity.",
        )
    title = heading.split(" - ", 1)[1].strip()
    status = _task_status(block)
    if status in TERMINAL_STATES:
        raise LedgerError(
            "compaction_terminal_task",
            "Terminal tasks do not require active-state compaction.",
            "Use rollover to archive completed task context.",
            "Do not create a continuation for completed work.",
        )
    if status == "BLOCKED" and not blocker:
        raise LedgerError(
            "compaction_blocker_required",
            "A blocked task must retain its current blocker in compact form.",
            "Provide the exact current blocker before compaction.",
            "Do not hide a blocked state behind archival shorthand.",
        )
    opened = _line_value(block, "Opened")
    if not opened:
        raise LedgerError(
            "task_opened_missing",
            "Managed task has no Opened metadata for compact continuation.",
            "Restore the task block through a verified manager checkpoint.",
            "Do not invent task lifecycle metadata.",
        )
    clean_phase = _clean_scalar(phase, "phase", 120)
    clean_resume = _clean_scalar(resume_point, "resume_point", 800)
    clean_refs = [_clean_scalar(ref, "authority_ref", 240) for ref in authority_refs]
    if not clean_refs:
        raise LedgerError(
            "compaction_authority_required",
            "Compaction requires at least one retained authority reference.",
            "Provide the current authority or result reference for continuation.",
            "Do not discard the basis for future task actions.",
        )
    clean_blocker = _clean_scalar(blocker, "blocker", 800) if blocker else None
    clean_sha = _clean_scalar(canonical_sha, "canonical_sha", 64) if canonical_sha else None
    if clean_sha and not re.fullmatch(r"[0-9a-f]{7,64}", clean_sha):
        raise LedgerError(
            "canonical_sha_invalid",
            "Canonical SHA must be a lowercase Git hash prefix or full hash.",
            "Provide the verified canonical Git SHA.",
            "Do not record an ambiguous code authority.",
        )
    skills_block = skill_source_block or block
    skills_match = re.search(r"^- Skills:\s*(?P<value>.+)$", skills_block, re.MULTILINE)
    skills = re.findall(r"`([^`]+)`", skills_match.group("value")) if skills_match else []
    if not skills:
        raise LedgerError(
            "compaction_skills_missing",
            "Compaction requires the selected skills retained by the active task.",
            "Restore skills from verified managed history before compacting.",
            "Do not discard the task's reasoning and execution provenance.",
        )
    nl = _newline(block)
    step_id = f"{task_id}-99"
    lines = [f"### {task_id} - {title}{nl}", nl]
    lines.extend(
        _wrap_field(
            "Prompt",
            "Continue from immutable managed history through the recorded phase and resume point.",
            nl,
        )
    )
    lines.append(f"- Status: `{status}`.{nl}")
    lines.append(f"- Opened: `{opened}`.{nl}")
    lines.extend(
        _wrap_field(
            "Acceptance",
            "Archive integrity and active continuation metadata remain verifiable.",
            nl,
        )
    )
    lines.extend(_wrap_skills(skills, nl))
    lines.extend(_wrap_field("Phase", clean_phase, nl))
    if clean_blocker:
        lines.extend(_wrap_field("Blocker", clean_blocker, nl))
    lines.extend(_wrap_field("Resume point", clean_resume, nl))
    lines.extend(_wrap_field("Authority references", "; ".join(clean_refs), nl))
    if clean_sha:
        lines.append(f"- Canonical SHA: `{clean_sha}`.{nl}")
    lines.append(f"- Archive reference: `{archive['reference']}`.{nl}")
    lines.append(f"- Archive SHA256: `{archive['archive_sha256']}`.{nl}")
    lines.append(f"- Archived content SHA256: `{archive['content_sha256']}`.{nl}")
    lines.append(f"- Pre-compaction revision: `{archive['pre_revision']}`.{nl}")
    lines.append(f"- Pre-compaction Block SHA256: `{archive['pre_block_sha256']}`.{nl}")
    lines.append(nl)
    lines.append(f"- [ ] `{step_id}` `[{status}]` Resume archived task from compact state.{nl}")
    lines.extend(_wrap_detail("Next", clean_resume, nl))
    lines.append(nl)
    compacted = "".join(lines)
    if len(compacted.splitlines()) > MAX_ACTIVE_TASK_LINES:
        raise LedgerError(
            "compaction_line_budget_exceeded",
            "Compact continuation would exceed the active-task line limit.",
            "Shorten retained continuation fields and retry through the manager.",
            "Do not weaken the short-memory governance limit.",
        )
    return compacted


class ShortMemoryLedger:
    """Locked, compare-and-swap interface to the shared short-memory ledger."""

    def __init__(
        self,
        coordination_root: Path,
        lock_timeout_seconds: float = 10.0,
    ) -> None:
        self.root = coordination_root.resolve()
        self.memory_path = self.root / MEMORY_RELATIVE
        self.lock_path = self.root / LOCK_RELATIVE
        self.lock_timeout_seconds = lock_timeout_seconds
        if not self.memory_path.is_file():
            raise LedgerError(
                "short_memory_missing",
                f"Short memory does not exist at {self.memory_path}.",
                "Point the manager at the canonical coordination root.",
                "Do not create a second shadow ledger.",
            )

    def _read(self) -> str:
        return self.memory_path.read_bytes().decode("utf-8")

    def _commit(self, text: str) -> None:
        _atomic_write(self.memory_path, text)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with exclusive_file_lock(self.lock_path, self.lock_timeout_seconds):
            yield

    def inspect(self, task_id: str, now: datetime | None = None) -> dict[str, Any]:
        with self._locked():
            text = self._read()
            span = _task_span(text, task_id)
            result = _describe(task_id, span["block"], _now(now))
            archive = _verify_task_archive(self.root, task_id, span["block"])
            if archive:
                result.update(archive)
                result["active_task_line_count"] = len(span["block"].splitlines())
            return result

    def rollover(self, now: datetime | None = None) -> dict[str, Any]:
        current = _now(now)
        with self._locked():
            text = self._read()
            candidate, result = _rollover_text(text, current)
            self._commit(candidate)
            return result

    def create(
        self,
        *,
        task_id: str,
        title: str,
        prompt: str,
        acceptance: str,
        skills: list[str],
        steps: list[dict[str, str]],
        active_step: str,
        owner_session: str,
        worktree: Path,
        owner_token: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _now(now)
        lease = current + timedelta(seconds=_validate_lease_seconds(lease_seconds))
        owner = _validate_owner(owner_session)
        token = owner_token or secrets.token_urlsafe(32)
        if len(token) < 16:
            raise LedgerError(
                "owner_token_weak",
                "Owner token must contain at least 16 characters.",
                "Use the generated token or a cryptographically random replacement.",
                "Do not create a task with a guessable ownership credential.",
            )
        resolved_worktree = _resolve_path(worktree)
        with self._locked():
            text = self._read()
            if any(span["task_id"] == task_id for span in task_spans(text)):
                raise LedgerError(
                    "task_id_collision",
                    f"Task ID {task_id} already exists.",
                    "Inspect the existing task and generate a new sequence.",
                    "Do not merge two sessions under one task ID.",
                )
            base = _render_task(
                task_id=task_id,
                title=title,
                prompt=prompt,
                acceptance=acceptance,
                skills=skills,
                steps=steps,
                active_step=active_step,
                opened=current,
                nl=_newline(text),
            )
            managed = _with_managed_metadata(
                base,
                owner_session=owner,
                owner_runtime_session=_environment_runtime_session(),
                owner_token_sha256=_token_sha256(token),
                worktree=resolved_worktree,
                revision=1,
                lease_expires=lease,
            )
            updated = _insert_task(text, managed, task_id)
            self._commit(updated)
            result = _describe(task_id, _task_span(updated, task_id)["block"], current)
        if owner_token is None:
            result["owner_token"] = token
        return result

    def adopt(
        self,
        *,
        task_id: str,
        expected_raw_sha256: str,
        owner_session: str,
        worktree: Path,
        owner_token: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _now(now)
        lease = current + timedelta(seconds=_validate_lease_seconds(lease_seconds))
        owner = _validate_owner(owner_session)
        token = owner_token or secrets.token_urlsafe(32)
        if len(token) < 16:
            raise LedgerError(
                "owner_token_weak",
                "Owner token must contain at least 16 characters.",
                "Use the generated token or a cryptographically random replacement.",
                "Do not adopt a task with a guessable ownership credential.",
            )
        resolved_worktree = _resolve_path(worktree)
        with self._locked():
            text = self._read()
            span = _task_span(text, task_id)
            block = span["block"]
            if parse_managed_task(block) is not None:
                raise LedgerError(
                    "task_already_managed",
                    "Task already uses the atomic concurrency schema.",
                    "Inspect it and use CAS checkpoint operations.",
                    "Do not adopt or reset managed ownership metadata.",
                )
            if not secrets.compare_digest(raw_block_sha256(block), expected_raw_sha256):
                raise LedgerError(
                    "legacy_block_drift",
                    "Legacy task bytes changed after inspection.",
                    "Inspect the task again and reconcile the concurrent edit.",
                    "Do not adopt a stale task snapshot.",
                )
            existing_owner = _line_value(block, "Owner session")
            if existing_owner and existing_owner != owner:
                raise LedgerError(
                    "legacy_owner_conflict",
                    "Legacy task already declares a different owner session.",
                    "Ask that owner to finish or explicitly hand off the task.",
                    "Do not claim another session's legacy task.",
                )
            normalized = _set_task_status(block, _derive_task_status(block))
            managed = _with_managed_metadata(
                normalized,
                owner_session=owner,
                owner_runtime_session=_environment_runtime_session(),
                owner_token_sha256=_token_sha256(token),
                worktree=resolved_worktree,
                revision=1,
                lease_expires=lease,
            )
            updated = _replace_task(text, span, managed)
            self._commit(updated)
            result = _describe(task_id, _task_span(updated, task_id)["block"], current)
        if owner_token is None:
            result["owner_token"] = token
        return result

    def reconcile(
        self,
        *,
        task_id: str,
        owner_session: str,
        owner_token: str | None,
        worktree: Path,
        expected_revision: int,
        expected_recorded_block_sha256: str,
        expected_raw_sha256: str,
        reason: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _now(now)
        owner = _validate_owner(owner_session)
        resolved_worktree = _resolve_path(worktree)
        lease = current + timedelta(seconds=_validate_lease_seconds(lease_seconds))
        reason_value = _clean_scalar(reason, "reconcile_reason", 64)
        with self._locked():
            text = self._read()
            span = _task_span(text, task_id)
            block = span["block"]
            if not secrets.compare_digest(
                raw_block_sha256(block),
                expected_raw_sha256,
            ):
                raise LedgerError(
                    "reconcile_raw_conflict",
                    "Task bytes changed after the drift audit.",
                    "Inspect the raw task block again before reconciling.",
                    "Do not bless uninspected bytes.",
                )
            metadata = parse_managed_task(block, verify_hash=False)
            if metadata is None:
                raise LedgerError(
                    "task_not_managed",
                    "Hash reconciliation requires managed ownership metadata.",
                    "Adopt a legacy task instead.",
                    "Do not synthesize ownership during hash recovery.",
                )
            if secrets.compare_digest(
                managed_block_sha256(block),
                metadata["block_sha256"],
            ):
                raise LedgerError(
                    "reconcile_not_needed",
                    "The managed block hash is already valid.",
                    "Use a normal inspect and checkpoint operation.",
                    "Do not create a needless recovery revision.",
                )
            if metadata["owner_session"] != owner or not secrets.compare_digest(
                metadata["owner_token_sha256"],
                _token_sha256(owner_token),
            ):
                raise LedgerError(
                    "owner_token_mismatch",
                    "Owner identity or private token does not match the task.",
                    "Use the owning session token or wait for lease takeover.",
                    "Do not reconcile another session's task.",
                )
            runtime_session = _environment_runtime_session()
            bound_runtime = metadata["owner_runtime_session"]
            if bound_runtime and runtime_session and bound_runtime != runtime_session:
                raise LedgerError(
                    "runtime_session_mismatch",
                    "The Codex runtime thread differs from the task binding.",
                    "Use the bound thread or an authorized administrative takeover.",
                    "Do not reuse a private token from another Codex thread.",
                )
            if _resolve_path(metadata["worktree"]) != resolved_worktree:
                raise LedgerError(
                    "worktree_mismatch",
                    "The caller worktree differs from the task binding.",
                    "Run reconciliation from the recorded worktree.",
                    "Do not transfer ownership through hash recovery.",
                )
            if metadata["revision"] != expected_revision or not secrets.compare_digest(
                metadata["block_sha256"],
                expected_recorded_block_sha256,
            ):
                raise LedgerError(
                    "reconcile_metadata_conflict",
                    "Recorded revision or hash changed after the audit.",
                    "Re-audit owner metadata and current bytes.",
                    "Do not reconcile a stale ownership snapshot.",
                )
            normalized = _set_task_status(block, _derive_task_status(block))
            managed = _with_managed_metadata(
                normalized,
                owner_session=owner,
                owner_runtime_session=bound_runtime or runtime_session,
                owner_token_sha256=metadata["owner_token_sha256"],
                worktree=resolved_worktree,
                revision=metadata["revision"] + 1,
                lease_expires=lease,
                previous_owner=metadata["previous_owner"],
                ownership_reason=f"reconcile:{reason_value}",
            )
            updated = _replace_task(text, span, managed)
            self._commit(updated)
            return _describe(task_id, _task_span(updated, task_id)["block"], current)

    def _owned_mutation(
        self,
        *,
        task_id: str,
        owner_session: str,
        owner_token: str,
        worktree: Path,
        expected_revision: int,
        expected_block_sha256: str,
        lease_seconds: int,
        mutation: Callable[[str], str],
        now: datetime | None,
    ) -> dict[str, Any]:
        current = _now(now)
        lease = current + timedelta(seconds=_validate_lease_seconds(lease_seconds))
        owner = _validate_owner(owner_session)
        resolved_worktree = _resolve_path(worktree)
        with self._locked():
            text = self._read()
            span = _task_span(text, task_id)
            block = span["block"]
            metadata = parse_managed_task(block)
            if metadata is None:
                raise LedgerError(
                    "task_not_managed",
                    "Legacy task has no atomic ownership metadata.",
                    "Inspect and adopt only the session-owned task.",
                    "Do not mutate legacy task bytes through CAS operations.",
                )
            if metadata["owner_session"] != owner:
                raise LedgerError(
                    "owner_session_mismatch",
                    "The caller does not own this task session.",
                    "Wait for lease expiry or request an explicit handoff.",
                    "Do not use another task's visible owner ID.",
                )
            runtime_session = _environment_runtime_session()
            bound_runtime = metadata["owner_runtime_session"]
            if bound_runtime and runtime_session and bound_runtime != runtime_session:
                raise LedgerError(
                    "runtime_session_mismatch",
                    "The Codex runtime thread differs from the task binding.",
                    "Use the bound thread or an authorized administrative takeover.",
                    "Do not reuse a private token from another Codex thread.",
                )
            if not secrets.compare_digest(
                metadata["owner_token_sha256"],
                _token_sha256(owner_token),
            ):
                raise LedgerError(
                    "owner_token_mismatch",
                    "The private owner token does not match this task.",
                    "Use the token returned only to the creating or takeover session.",
                    "Do not mutate a task using public memory metadata alone.",
                )
            if _resolve_path(metadata["worktree"]) != resolved_worktree:
                raise LedgerError(
                    "worktree_mismatch",
                    "The caller worktree differs from the owned task worktree.",
                    "Use the recorded worktree or perform a lease-gated takeover.",
                    "Do not apply one worktree's checkpoint to another worktree.",
                )
            if metadata["revision"] != expected_revision:
                raise LedgerError(
                    "revision_conflict",
                    "Task revision changed after the caller inspected it.",
                    "Inspect the task again and reconcile the newer state.",
                    "Do not retry with a guessed revision.",
                )
            if not secrets.compare_digest(
                metadata["block_sha256"],
                expected_block_sha256,
            ):
                raise LedgerError(
                    "block_cas_conflict",
                    "Task block hash changed after the caller inspected it.",
                    "Inspect the exact block and reconcile before retrying.",
                    "Do not overwrite the newer checkpoint.",
                )
            mutated = mutation(block)
            managed = _with_managed_metadata(
                mutated,
                owner_session=owner,
                owner_runtime_session=bound_runtime or runtime_session,
                owner_token_sha256=metadata["owner_token_sha256"],
                worktree=resolved_worktree,
                revision=metadata["revision"] + 1,
                lease_expires=lease,
                previous_owner=metadata["previous_owner"],
                ownership_reason=metadata["ownership_reason"],
            )
            updated = _replace_task(text, span, managed)
            self._commit(updated)
            return _describe(
                task_id,
                _task_span(updated, task_id)["block"],
                current,
            )

    def checkpoint(
        self,
        *,
        task_id: str,
        step_id: str,
        step_status: str,
        evidence: str | None,
        next_action: str | None,
        owner_session: str,
        owner_token: str,
        worktree: Path,
        expected_revision: int,
        expected_block_sha256: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if step_status not in CHECKLIST_STATES:
            raise LedgerError(
                "step_status_invalid",
                f"Unknown checklist state: {step_status}.",
                "Use a declared checklist state.",
                "Do not write free-form execution state.",
            )
        if step_status in TERMINAL_STATES:
            if not evidence or next_action:
                raise LedgerError(
                    "terminal_detail_invalid",
                    "Terminal steps require Evidence and cannot use Next.",
                    "Provide concrete evidence for the completed or cancelled step.",
                    "Do not checkpoint terminal state without evidence.",
                )
            label, detail = "Evidence", evidence
        else:
            if not next_action or evidence:
                raise LedgerError(
                    "open_detail_invalid",
                    "Open steps require Next and cannot use Evidence as completion proof.",
                    "Provide the smallest safe next action.",
                    "Do not blur unfinished and completed task state.",
                )
            label, detail = "Next", next_action

        def mutate(block: str) -> str:
            changed = _replace_step_detail(
                block,
                step_id,
                step_status,
                label,
                detail or "",
            )
            return _set_task_status(changed, _derive_task_status(changed))

        return self._owned_mutation(
            task_id=task_id,
            owner_session=owner_session,
            owner_token=owner_token,
            worktree=worktree,
            expected_revision=expected_revision,
            expected_block_sha256=expected_block_sha256,
            lease_seconds=lease_seconds,
            mutation=mutate,
            now=now,
        )

    def compact(
        self,
        *,
        task_id: str,
        owner_session: str,
        owner_token: str,
        worktree: Path,
        expected_revision: int,
        expected_block_sha256: str,
        phase: str,
        blocker: str | None,
        resume_point: str,
        authority_refs: list[str],
        canonical_sha: str | None = None,
        repair_existing: bool = False,
        same_session_recovery: bool = False,
        new_owner_token: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Archive exact task bytes and replace them with a bounded continuation."""
        current = _now(now)
        owner = _validate_owner(owner_session)
        resolved_worktree = _resolve_path(worktree)
        lease = current + timedelta(seconds=_validate_lease_seconds(lease_seconds))
        if owner_token is not None and len(owner_token) < 16:
            raise LedgerError(
                "owner_token_weak",
                "Compaction requires a private owner token with at least 16 characters.",
                "Use the token retained by the task-owning session.",
                "Do not mutate public managed metadata alone.",
            )
        with self._locked():
            text = self._read()
            span = _task_span(text, task_id)
            block = span["block"]
            metadata = parse_managed_task(block)
            if metadata is None:
                raise LedgerError(
                    "task_not_managed",
                    "Only managed task blocks can be compacted.",
                    "Adopt legacy work through the manager before compacting it.",
                    "Do not manually archive an unmanaged task.",
                )
            existing_archive = _archive_values(block)
            if existing_archive is not None and not repair_existing:
                raise LedgerError(
                    "task_already_compacted",
                    "This active task already references immutable compacted history.",
                    "Inspect and verify the existing archive instead of compacting again.",
                    "Do not create a second compact continuation for one active block.",
                )
            if existing_archive is None and repair_existing:
                raise LedgerError(
                    "task_compaction_repair_not_applicable",
                    "Repair applies only to an active task with verified archived history.",
                    "Use normal compaction for a task without an archive reference.",
                    "Do not create repair-only state for an unarchived task.",
                )
            runtime_session = _environment_runtime_session()
            bound_runtime = metadata["owner_runtime_session"]
            if bound_runtime and runtime_session and bound_runtime != runtime_session:
                raise LedgerError(
                    "runtime_session_mismatch",
                    "The Codex runtime thread differs from the managed task binding.",
                    "Use the bound thread or an authorized administrative takeover.",
                    "Do not reuse a private token from another Codex thread.",
                )
            if metadata["owner_session"] != owner:
                raise LedgerError(
                    "owner_token_mismatch",
                    "The compaction caller does not own this managed task.",
                    "Use the active owner token or complete an authorized takeover.",
                    "Do not compact another session's task.",
                )
            if _resolve_path(metadata["worktree"]) != resolved_worktree:
                raise LedgerError(
                    "worktree_mismatch",
                    "Compaction caller worktree differs from the managed task binding.",
                    "Run from the recorded worktree or use an authorized rebind path.",
                    "Do not attach archived history to another worktree.",
                )
            if metadata["revision"] != expected_revision:
                raise LedgerError(
                    "revision_conflict",
                    "Task revision changed after the compaction inspection.",
                    "Inspect the current task and retry with fresh CAS values.",
                    "Do not compact a stale task snapshot.",
                )
            if not secrets.compare_digest(metadata["block_sha256"], expected_block_sha256):
                raise LedgerError(
                    "block_cas_conflict",
                    "Task block hash changed after the compaction inspection.",
                    "Inspect the exact current task before retrying.",
                    "Do not overwrite newer task history.",
                )
            recovered = False
            token = owner_token
            if owner_token is not None:
                if not secrets.compare_digest(
                    metadata["owner_token_sha256"], _token_sha256(owner_token)
                ):
                    raise LedgerError(
                        "owner_token_mismatch",
                        "The private owner token does not match this managed task.",
                        "Use the active owner token or same-session recovery.",
                        "Do not compact another session's task.",
                    )
            else:
                if not same_session_recovery:
                    raise LedgerError(
                        "owner_token_missing",
                        "Compaction requires the private owner token.",
                        "Use the owning token or explicitly request same-session recovery.",
                        "Do not mutate public task metadata alone.",
                    )
                if runtime_session is None or bound_runtime != runtime_session:
                    raise LedgerError(
                        "recovery_runtime_mismatch",
                        "Same-session compaction recovery requires the bound Codex runtime.",
                        "Use the original thread or an authorized administrative takeover.",
                        "Do not treat another thread as the task owner.",
                    )
                token = new_owner_token or secrets.token_urlsafe(32)
                if len(token) < 16:
                    raise LedgerError(
                        "owner_token_weak",
                        "Replacement owner token must contain at least 16 characters.",
                        "Use a generated cryptographically random token.",
                        "Do not recover ownership with a guessable credential.",
                    )
                recovered = True
            archive_source = None
            if repair_existing:
                archive_source = _verified_archive_content(self.root, task_id, block)
                archive = {
                    "reference": existing_archive["reference"],
                    "archive_sha256": existing_archive["archive_sha256"],
                    "content_sha256": existing_archive["content_sha256"],
                    "pre_revision": existing_archive["pre_revision"],
                    "pre_block_sha256": existing_archive["pre_block_sha256"],
                }
            else:
                _, _, archive = _prepare_task_archive(
                    self.root,
                    task_id=task_id,
                    block=block,
                    metadata=metadata,
                    timestamp=current,
                )
                archive.update(
                    {
                        "pre_revision": str(metadata["revision"]),
                        "pre_block_sha256": metadata["block_sha256"],
                    }
                )
            compacted = _compact_task_block(
                block,
                task_id=task_id,
                phase=phase,
                blocker=blocker,
                resume_point=resume_point,
                authority_refs=authority_refs,
                canonical_sha=canonical_sha,
                archive=archive,
                skill_source_block=archive_source,
            )
            if recovered:
                compacted = _append_ownership_audit(
                    compacted,
                    timestamp=current,
                    action="same-session-compaction-recovery",
                    from_owner=metadata["owner_session"],
                    from_runtime_session=metadata["owner_runtime_session"],
                    to_owner=metadata["owner_session"],
                    to_runtime_session=runtime_session,
                    prior_revision=metadata["revision"],
                    prior_block_sha256=metadata["block_sha256"],
                    prior_worktree=resolved_worktree,
                    new_worktree=resolved_worktree,
                    reason="same-session compaction recovery",
                    authority=f"{RUNTIME_SESSION_ENV} match plus lock and CAS",
                )
            if len(compacted.splitlines()) > MAX_ACTIVE_TASK_LINES:
                raise LedgerError(
                    "compaction_line_budget_exceeded",
                    "Compact continuation exceeds the active-task line limit.",
                    "Shorten retained continuation fields and retry through the manager.",
                    "Do not weaken the short-memory governance limit.",
                )
            managed = _with_managed_metadata(
                compacted,
                owner_session=owner,
                owner_runtime_session=bound_runtime or runtime_session,
                owner_token_sha256=_token_sha256(token),
                worktree=resolved_worktree,
                revision=metadata["revision"] + 1,
                lease_expires=lease,
                previous_owner=metadata["previous_owner"],
                ownership_reason=(
                    "same-session-compaction-recovery"
                    if recovered
                    else metadata["ownership_reason"]
                ),
            )
            updated = _replace_task(text, span, managed)
            if not repair_existing:
                persisted_archive = _write_task_archive(
                    self.root,
                    task_id=task_id,
                    block=block,
                    metadata=metadata,
                    timestamp=current,
                )
                if any(
                    persisted_archive[key] != archive[key]
                    for key in ("reference", "archive_sha256", "content_sha256")
                ):
                    raise LedgerError(
                        "archive_preflight_drift",
                        "Prepared archive integrity anchors changed before commit.",
                        "Inspect concurrent archive storage changes and retry from fresh CAS.",
                        "Do not bind an active task to uninspected history bytes.",
                    )
            self._commit(updated)
            result = _describe(task_id, _task_span(updated, task_id)["block"], current)
            result.update(_verify_task_archive(self.root, task_id, managed) or {})
            result["active_task_line_count"] = len(managed.splitlines())
            if recovered and new_owner_token is None:
                result["owner_token"] = token
            return result

    def renew(
        self,
        *,
        task_id: str,
        owner_session: str,
        owner_token: str,
        worktree: Path,
        expected_revision: int,
        expected_block_sha256: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return self._owned_mutation(
            task_id=task_id,
            owner_session=owner_session,
            owner_token=owner_token,
            worktree=worktree,
            expected_revision=expected_revision,
            expected_block_sha256=expected_block_sha256,
            lease_seconds=lease_seconds,
            mutation=lambda block: block,
            now=now,
        )

    def recover_same_session(
        self,
        *,
        task_id: str,
        expected_owner_session: str,
        expected_revision: int,
        expected_block_sha256: str,
        worktree: Path,
        reason: str,
        owner_token: str | None = None,
        new_owner_token: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _now(now)
        runtime_session = _environment_runtime_session()
        if runtime_session is None:
            raise LedgerError(
                "runtime_session_missing",
                f"{RUNTIME_SESSION_ENV} is unavailable for same-session recovery.",
                "Use the bound Codex runtime or an authorized administrative takeover.",
                "Do not infer same-session ownership without runtime identity.",
            )
        expected_owner = _validate_owner(expected_owner_session)
        token = new_owner_token or secrets.token_urlsafe(32)
        if len(token) < 16:
            raise LedgerError(
                "owner_token_weak",
                "Recovered owner token must contain at least 16 characters.",
                "Use a generated cryptographically random token.",
                "Do not recover a task with a guessable credential.",
            )
        resolved_worktree = _resolve_path(worktree)
        reason_value = _clean_scalar(reason, "recovery_reason", 128)
        lease = current + timedelta(seconds=_validate_lease_seconds(lease_seconds))
        with self._locked():
            text = self._read()
            span = _task_span(text, task_id)
            block = span["block"]
            metadata = parse_managed_task(block)
            if metadata is None:
                raise LedgerError(
                    "task_not_managed",
                    "Legacy tasks cannot use same-session token recovery.",
                    "Adopt the task through the normal managed path first.",
                    "Do not synthesize ownership for an unmanaged task.",
                )
            if metadata["owner_session"] != expected_owner:
                raise LedgerError(
                    "recovery_owner_conflict",
                    "The recorded owner changed after inspection.",
                    "Inspect the task and retry only for the current owner.",
                    "Do not recover a stale owner snapshot.",
                )
            if metadata["revision"] != expected_revision or not secrets.compare_digest(
                metadata["block_sha256"],
                expected_block_sha256,
            ):
                raise LedgerError(
                    "recovery_cas_conflict",
                    "The task changed after recovery inspection.",
                    "Inspect current revision and block hash before retrying.",
                    "Do not rotate credentials over a newer checkpoint.",
                )
            recorded_worktree = _resolve_path(metadata["worktree"])
            if recorded_worktree != resolved_worktree:
                raise LedgerError(
                    "recovery_worktree_mismatch",
                    "The requested worktree differs from the task binding.",
                    "Recover from the recorded worktree or use administrative takeover.",
                    "Do not move worktrees through automatic recovery.",
                )
            bound_runtime = metadata["owner_runtime_session"]
            if bound_runtime is None and metadata["owner_session"] == runtime_session:
                bound_runtime = runtime_session
            if bound_runtime != runtime_session:
                if not owner_token or not secrets.compare_digest(
                    metadata["owner_token_sha256"],
                    _token_sha256(owner_token),
                ):
                    raise LedgerError(
                        "recovery_runtime_mismatch",
                        "A context-handoff recovery needs the current owner token.",
                        "Provide the retained token or use administrative takeover.",
                        "Do not rotate a different runtime without owner proof.",
                    )
            audited = _append_ownership_audit(
                block,
                timestamp=current,
                action=(
                    "context-handoff-token-recovery"
                    if bound_runtime != runtime_session
                    else "same-session-token-recovery"
                ),
                from_owner=metadata["owner_session"],
                from_runtime_session=metadata["owner_runtime_session"],
                to_owner=metadata["owner_session"],
                to_runtime_session=runtime_session,
                prior_revision=metadata["revision"],
                prior_block_sha256=metadata["block_sha256"],
                prior_worktree=recorded_worktree,
                new_worktree=resolved_worktree,
                reason=reason_value,
                authority=f"{RUNTIME_SESSION_ENV} match plus lock and CAS",
            )
            managed = _with_managed_metadata(
                audited,
                owner_session=metadata["owner_session"],
                owner_runtime_session=runtime_session,
                owner_token_sha256=_token_sha256(token),
                worktree=resolved_worktree,
                revision=metadata["revision"] + 1,
                lease_expires=lease,
                previous_owner=metadata["previous_owner"],
                ownership_reason="same-session-token-recovery",
            )
            updated = _replace_task(text, span, managed)
            self._commit(updated)
            result = _describe(
                task_id,
                _task_span(updated, task_id)["block"],
                current,
            )
        if new_owner_token is None:
            result["owner_token"] = token
        return result

    def administrative_takeover(
        self,
        *,
        task_id: str,
        confirm_task_id: str,
        confirmation: str,
        authorization_ref: str,
        expected_owner_session: str,
        expected_revision: int,
        expected_block_sha256: str,
        expected_worktree: Path,
        new_owner_session: str,
        new_worktree: Path,
        reason: str,
        new_owner_token: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if confirm_task_id != task_id:
            raise LedgerError(
                "admin_task_confirmation_mismatch",
                "The repeated task ID does not match the takeover target.",
                "Inspect the target and repeat its exact task ID.",
                "Do not authorize an ambiguous administrative takeover.",
            )
        if confirmation != ADMIN_TAKEOVER_CONFIRMATION:
            raise LedgerError(
                "admin_confirmation_missing",
                "Administrative takeover lacks the exact confirmation phrase.",
                f"Use {ADMIN_TAKEOVER_CONFIRMATION} after explicit user approval.",
                "Do not bypass an active or ambiguous owner without approval.",
            )
        current = _now(now)
        runtime_session = _environment_runtime_session()
        if runtime_session is None:
            raise LedgerError(
                "runtime_session_missing",
                f"{RUNTIME_SESSION_ENV} is unavailable for the new owner binding.",
                "Run from the authorized Codex thread.",
                "Do not create an unbound administrative owner.",
            )
        expected_owner = _validate_owner(expected_owner_session)
        new_owner = _validate_owner(new_owner_session)
        auth_ref = _clean_scalar(authorization_ref, "authorization_ref", 128)
        reason_value = _clean_scalar(reason, "administrative_reason", 128)
        token = new_owner_token or secrets.token_urlsafe(32)
        if len(token) < 16:
            raise LedgerError(
                "owner_token_weak",
                "Administrative owner token must contain at least 16 characters.",
                "Use a generated cryptographically random token.",
                "Do not take over a task with a guessable credential.",
            )
        expected_tree = _resolve_path(expected_worktree)
        new_tree = _resolve_path(new_worktree)
        lease = current + timedelta(seconds=_validate_lease_seconds(lease_seconds))
        with self._locked():
            text = self._read()
            span = _task_span(text, task_id)
            block = span["block"]
            metadata = parse_managed_task(block)
            if metadata is None:
                raise LedgerError(
                    "task_not_managed",
                    "Legacy tasks cannot use administrative takeover.",
                    "Adopt the task through the managed path first.",
                    "Do not administratively claim unverified task bytes.",
                )
            if metadata["owner_session"] != expected_owner:
                raise LedgerError(
                    "admin_owner_conflict",
                    "The recorded owner changed after administrative inspection.",
                    "Inspect the task and obtain fresh explicit authorization.",
                    "Do not reuse authorization for a different owner state.",
                )
            if metadata["revision"] != expected_revision or not secrets.compare_digest(
                metadata["block_sha256"],
                expected_block_sha256,
            ):
                raise LedgerError(
                    "admin_cas_conflict",
                    "The task changed after administrative inspection.",
                    "Inspect fresh revision and hash, then reauthorize explicitly.",
                    "Do not apply approval to a newer task state.",
                )
            recorded_worktree = _resolve_path(metadata["worktree"])
            if recorded_worktree != expected_tree:
                raise LedgerError(
                    "admin_worktree_conflict",
                    "The recorded worktree differs from the confirmed worktree.",
                    "Inspect the exact worktree and obtain fresh authorization.",
                    "Do not transfer an ambiguously located task.",
                )
            audited = _append_ownership_audit(
                block,
                timestamp=current,
                action="administrative-takeover",
                from_owner=metadata["owner_session"],
                from_runtime_session=metadata["owner_runtime_session"],
                to_owner=new_owner,
                to_runtime_session=runtime_session,
                prior_revision=metadata["revision"],
                prior_block_sha256=metadata["block_sha256"],
                prior_worktree=recorded_worktree,
                new_worktree=new_tree,
                reason=reason_value,
                authority=f"user authorization reference {auth_ref}",
            )
            managed = _with_managed_metadata(
                audited,
                owner_session=new_owner,
                owner_runtime_session=runtime_session,
                owner_token_sha256=_token_sha256(token),
                worktree=new_tree,
                revision=metadata["revision"] + 1,
                lease_expires=lease,
                previous_owner=metadata["owner_session"],
                ownership_reason="administrative-takeover",
            )
            updated = _replace_task(text, span, managed)
            self._commit(updated)
            result = _describe(
                task_id,
                _task_span(updated, task_id)["block"],
                current,
            )
        if new_owner_token is None:
            result["owner_token"] = token
        return result

    def takeover(
        self,
        *,
        task_id: str,
        expected_owner_session: str,
        expected_revision: int,
        expected_block_sha256: str,
        new_owner_session: str,
        new_worktree: Path,
        reason: str,
        new_owner_token: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _now(now)
        owner = _validate_owner(new_owner_session)
        expected_owner = _validate_owner(expected_owner_session)
        token = new_owner_token or secrets.token_urlsafe(32)
        if len(token) < 16:
            raise LedgerError(
                "owner_token_weak",
                "New owner token must contain at least 16 characters.",
                "Use a generated cryptographically random token.",
                "Do not take over a task with a guessable credential.",
            )
        lease = current + timedelta(seconds=_validate_lease_seconds(lease_seconds))
        worktree = _resolve_path(new_worktree)
        runtime_session = _environment_runtime_session()
        reason_value = _clean_scalar(reason, "ownership_reason", 64)
        with self._locked():
            text = self._read()
            span = _task_span(text, task_id)
            block = span["block"]
            metadata = parse_managed_task(block)
            if metadata is None:
                raise LedgerError(
                    "task_not_managed",
                    "Legacy tasks cannot use atomic takeover.",
                    "The current owner must adopt the task first.",
                    "Do not claim an unowned legacy block.",
                )
            if metadata["owner_session"] != expected_owner:
                raise LedgerError(
                    "takeover_owner_conflict",
                    "The recorded owner changed after inspection.",
                    "Inspect the task again and contact the current owner.",
                    "Do not take over a task from a stale owner snapshot.",
                )
            if metadata["revision"] != expected_revision or not secrets.compare_digest(
                metadata["block_sha256"],
                expected_block_sha256,
            ):
                raise LedgerError(
                    "takeover_cas_conflict",
                    "The task changed after takeover inspection.",
                    "Inspect current revision, hash, owner, and lease again.",
                    "Do not overwrite new work during takeover.",
                )
            if current < metadata["lease_expires"]:
                raise LedgerError(
                    "takeover_lease_active",
                    "The current owner's lease has not expired.",
                    "Wait for expiry or obtain a cooperative handoff from that session.",
                    "Do not run two owners on the same task concurrently.",
                )
            audited = _append_ownership_audit(
                block,
                timestamp=current,
                action="expired-lease-takeover",
                from_owner=metadata["owner_session"],
                from_runtime_session=metadata["owner_runtime_session"],
                to_owner=owner,
                to_runtime_session=runtime_session,
                prior_revision=metadata["revision"],
                prior_block_sha256=metadata["block_sha256"],
                prior_worktree=_resolve_path(metadata["worktree"]),
                new_worktree=worktree,
                reason=reason_value,
                authority="expired lease plus lock and CAS",
            )
            managed = _with_managed_metadata(
                audited,
                owner_session=owner,
                owner_runtime_session=runtime_session,
                owner_token_sha256=_token_sha256(token),
                worktree=worktree,
                revision=metadata["revision"] + 1,
                lease_expires=lease,
                previous_owner=metadata["owner_session"],
                ownership_reason=reason_value,
            )
            updated = _replace_task(text, span, managed)
            self._commit(updated)
            result = _describe(task_id, _task_span(updated, task_id)["block"], current)
        if new_owner_token is None:
            result["owner_token"] = token
        return result


def _parse_step(value: str) -> dict[str, str]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "step must use STEP_ID|summary|next_action"
        )
    return {
        "step_id": parts[0].strip(),
        "summary": parts[1].strip(),
        "next_action": parts[2].strip(),
    }


def _common_owner_arguments(parser: argparse.ArgumentParser) -> None:
    runtime_owner = os.getenv(RUNTIME_SESSION_ENV)
    parser.add_argument(
        "--owner-session",
        default=runtime_owner,
        required=runtime_owner is None,
    )
    parser.add_argument("--owner-token", default=os.getenv("PIG_TASK_OWNER_TOKEN"))
    parser.add_argument("--worktree", type=Path, default=Path.cwd())
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically manage session-owned short-memory task blocks."
    )
    parser.add_argument("--coordination-root", type=Path)
    parser.add_argument("--lock-timeout", type=float, default=10.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("rollover")

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--task-id", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--task-id", required=True)
    create_parser.add_argument("--title", required=True)
    create_parser.add_argument("--prompt", required=True)
    create_parser.add_argument("--acceptance", required=True)
    create_parser.add_argument("--skill", action="append", required=True)
    create_parser.add_argument("--step", action="append", type=_parse_step, required=True)
    create_parser.add_argument("--active-step", required=True)
    _common_owner_arguments(create_parser)

    adopt_parser = subparsers.add_parser("adopt")
    adopt_parser.add_argument("--task-id", required=True)
    adopt_parser.add_argument("--expected-raw-sha256", required=True)
    _common_owner_arguments(adopt_parser)

    reconcile_parser = subparsers.add_parser("reconcile")
    reconcile_parser.add_argument("--task-id", required=True)
    reconcile_parser.add_argument("--expected-revision", type=int, required=True)
    reconcile_parser.add_argument(
        "--expected-recorded-block-sha256",
        required=True,
    )
    reconcile_parser.add_argument("--expected-raw-sha256", required=True)
    reconcile_parser.add_argument("--reason", required=True)
    _common_owner_arguments(reconcile_parser)

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--task-id", required=True)
    checkpoint_parser.add_argument("--step-id", required=True)
    checkpoint_parser.add_argument("--step-status", required=True)
    checkpoint_parser.add_argument("--evidence")
    checkpoint_parser.add_argument("--next-action")
    checkpoint_parser.add_argument("--expected-revision", type=int, required=True)
    checkpoint_parser.add_argument("--expected-block-sha256", required=True)
    _common_owner_arguments(checkpoint_parser)

    renew_parser = subparsers.add_parser("renew")
    renew_parser.add_argument("--task-id", required=True)
    renew_parser.add_argument("--expected-revision", type=int, required=True)
    renew_parser.add_argument("--expected-block-sha256", required=True)
    _common_owner_arguments(renew_parser)

    compact_parser = subparsers.add_parser("compact")
    compact_parser.add_argument("--task-id", required=True)
    compact_parser.add_argument("--expected-revision", type=int, required=True)
    compact_parser.add_argument("--expected-block-sha256", required=True)
    compact_parser.add_argument("--phase", required=True)
    compact_parser.add_argument("--blocker")
    compact_parser.add_argument("--resume-point", required=True)
    compact_parser.add_argument("--authority-ref", action="append", required=True)
    compact_parser.add_argument("--canonical-sha")
    compact_parser.add_argument("--repair-existing", action="store_true")
    compact_parser.add_argument("--same-session-recovery", action="store_true")
    compact_parser.add_argument("--new-owner-token")
    _common_owner_arguments(compact_parser)

    recover_parser = subparsers.add_parser("recover")
    recover_parser.add_argument("--task-id", required=True)
    recover_parser.add_argument("--expected-owner-session", required=True)
    recover_parser.add_argument("--expected-revision", type=int, required=True)
    recover_parser.add_argument("--expected-block-sha256", required=True)
    recover_parser.add_argument("--worktree", type=Path, default=Path.cwd())
    recover_parser.add_argument("--reason", required=True)
    recover_parser.add_argument(
        "--owner-token",
        default=os.getenv("PIG_TASK_OWNER_TOKEN"),
    )
    recover_parser.add_argument("--new-owner-token")
    recover_parser.add_argument(
        "--lease-seconds",
        type=int,
        default=DEFAULT_LEASE_SECONDS,
    )

    admin_parser = subparsers.add_parser("admin-takeover")
    admin_parser.add_argument("--task-id", required=True)
    admin_parser.add_argument("--confirm-task-id", required=True)
    admin_parser.add_argument("--confirmation", required=True)
    admin_parser.add_argument("--authorization-ref", required=True)
    admin_parser.add_argument("--expected-owner-session", required=True)
    admin_parser.add_argument("--expected-revision", type=int, required=True)
    admin_parser.add_argument("--expected-block-sha256", required=True)
    admin_parser.add_argument("--expected-worktree", type=Path, required=True)
    admin_parser.add_argument(
        "--new-owner-session",
        default=os.getenv(RUNTIME_SESSION_ENV),
        required=os.getenv(RUNTIME_SESSION_ENV) is None,
    )
    admin_parser.add_argument("--new-owner-token")
    admin_parser.add_argument("--new-worktree", type=Path, default=Path.cwd())
    admin_parser.add_argument("--reason", required=True)
    admin_parser.add_argument(
        "--lease-seconds",
        type=int,
        default=DEFAULT_LEASE_SECONDS,
    )

    takeover_parser = subparsers.add_parser("takeover")
    takeover_parser.add_argument("--task-id", required=True)
    takeover_parser.add_argument("--expected-owner-session", required=True)
    takeover_parser.add_argument("--expected-revision", type=int, required=True)
    takeover_parser.add_argument("--expected-block-sha256", required=True)
    takeover_parser.add_argument("--new-owner-session", required=True)
    takeover_parser.add_argument("--new-owner-token")
    takeover_parser.add_argument("--new-worktree", type=Path, default=Path.cwd())
    takeover_parser.add_argument("--reason", required=True)
    takeover_parser.add_argument(
        "--lease-seconds",
        type=int,
        default=DEFAULT_LEASE_SECONDS,
    )
    return parser


def _success(summary: str, task: dict[str, Any], memory_path: Path) -> dict[str, Any]:
    next_actions = [
        "Retain any newly returned owner_token only in the owning live session.",
        "Use the returned revision and block_sha256 for the next mutation.",
    ]
    return {
        "status": "success",
        "summary": summary,
        "next_actions": next_actions,
        "artifacts": [str(memory_path)],
        "task": task,
    }


def _rollover_success(result: dict[str, Any], memory_path: Path) -> dict[str, Any]:
    return {
        "status": "success",
        "summary": "Expired short memory rolled over atomically.",
        "next_actions": [
            "Resume retained task IDs from their current nonterminal step.",
            "Do not rerun DONE steps without invalidated evidence.",
        ],
        "artifacts": [str(memory_path)],
        "rollover": result,
    }


def _error(exc: LedgerError, memory_path: Path | None) -> dict[str, Any]:
    return {
        "status": "error",
        "summary": exc.code,
        "next_actions": [exc.safe_retry],
        "artifacts": [str(memory_path)] if memory_path else [],
        "root_cause_hint": exc.root_cause_hint,
        "safe_retry": exc.safe_retry,
        "stop_condition": exc.stop_condition,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    worktree = getattr(args, "worktree", getattr(args, "new_worktree", Path.cwd()))
    root = (
        args.coordination_root.resolve()
        if args.coordination_root
        else discover_coordination_root(_resolve_path(worktree))
    )
    ledger: ShortMemoryLedger | None = None
    try:
        ledger = ShortMemoryLedger(root, args.lock_timeout)
        if args.command == "rollover":
            result = _rollover_success(ledger.rollover(), ledger.memory_path)
        elif args.command == "inspect":
            task = ledger.inspect(args.task_id)
            result = _success("Task inspected under the ledger lock.", task, ledger.memory_path)
        elif args.command == "create":
            task = ledger.create(
                task_id=args.task_id,
                title=args.title,
                prompt=args.prompt,
                acceptance=args.acceptance,
                skills=args.skill,
                steps=args.step,
                active_step=args.active_step,
                owner_session=args.owner_session,
                owner_token=args.owner_token,
                worktree=args.worktree,
                lease_seconds=args.lease_seconds,
            )
            result = _success("Managed task created atomically.", task, ledger.memory_path)
        elif args.command == "adopt":
            task = ledger.adopt(
                task_id=args.task_id,
                expected_raw_sha256=args.expected_raw_sha256,
                owner_session=args.owner_session,
                owner_token=args.owner_token,
                worktree=args.worktree,
                lease_seconds=args.lease_seconds,
            )
            result = _success("Legacy task adopted atomically.", task, ledger.memory_path)
        elif args.command == "reconcile":
            if not args.owner_token:
                raise LedgerError(
                    "owner_token_missing",
                    "Hash reconciliation requires the private owner token.",
                    "Use the token retained by the owning session.",
                    "Do not reconcile from public metadata alone.",
                )
            task = ledger.reconcile(
                task_id=args.task_id,
                owner_session=args.owner_session,
                owner_token=args.owner_token,
                worktree=args.worktree,
                expected_revision=args.expected_revision,
                expected_recorded_block_sha256=(
                    args.expected_recorded_block_sha256
                ),
                expected_raw_sha256=args.expected_raw_sha256,
                reason=args.reason,
                lease_seconds=args.lease_seconds,
            )
            result = _success("Managed task hash reconciled.", task, ledger.memory_path)
        elif args.command == "checkpoint":
            if not args.owner_token:
                raise LedgerError(
                    "owner_token_missing",
                    "Checkpoint requires the private owner token.",
                    "Use the token returned to the creating or takeover session.",
                    "Do not mutate a task from public metadata alone.",
                )
            task = ledger.checkpoint(
                task_id=args.task_id,
                step_id=args.step_id,
                step_status=args.step_status,
                evidence=args.evidence,
                next_action=args.next_action,
                owner_session=args.owner_session,
                owner_token=args.owner_token,
                worktree=args.worktree,
                expected_revision=args.expected_revision,
                expected_block_sha256=args.expected_block_sha256,
                lease_seconds=args.lease_seconds,
            )
            result = _success("Task step checkpointed atomically.", task, ledger.memory_path)
        elif args.command == "renew":
            if not args.owner_token:
                raise LedgerError(
                    "owner_token_missing",
                    "Lease renewal requires the private owner token.",
                    "Use the token returned to the owning session.",
                    "Do not renew another session's task.",
                )
            task = ledger.renew(
                task_id=args.task_id,
                owner_session=args.owner_session,
                owner_token=args.owner_token,
                worktree=args.worktree,
                expected_revision=args.expected_revision,
                expected_block_sha256=args.expected_block_sha256,
                lease_seconds=args.lease_seconds,
            )
            result = _success("Task ownership lease renewed atomically.", task, ledger.memory_path)
        elif args.command == "compact":
            if not args.owner_token and not args.same_session_recovery:
                raise LedgerError(
                    "owner_token_missing",
                    "Task compaction requires the private owner token.",
                    "Use the token retained by the owning session.",
                    "Do not compact public task metadata alone.",
                )
            task = ledger.compact(
                task_id=args.task_id,
                owner_session=args.owner_session,
                owner_token=args.owner_token,
                worktree=args.worktree,
                expected_revision=args.expected_revision,
                expected_block_sha256=args.expected_block_sha256,
                phase=args.phase,
                blocker=args.blocker,
                resume_point=args.resume_point,
                authority_refs=args.authority_ref,
                canonical_sha=args.canonical_sha,
                repair_existing=args.repair_existing,
                same_session_recovery=args.same_session_recovery,
                new_owner_token=args.new_owner_token,
                lease_seconds=args.lease_seconds,
            )
            result = _success(
                "Managed task compacted with immutable verified history.",
                task,
                ledger.memory_path,
            )
        elif args.command == "recover":
            task = ledger.recover_same_session(
                task_id=args.task_id,
                expected_owner_session=args.expected_owner_session,
                expected_revision=args.expected_revision,
                expected_block_sha256=args.expected_block_sha256,
                worktree=args.worktree,
                reason=args.reason,
                owner_token=args.owner_token,
                new_owner_token=args.new_owner_token,
                lease_seconds=args.lease_seconds,
            )
            result = _success(
                "Same-session task credential recovered atomically.",
                task,
                ledger.memory_path,
            )
        elif args.command == "admin-takeover":
            task = ledger.administrative_takeover(
                task_id=args.task_id,
                confirm_task_id=args.confirm_task_id,
                confirmation=args.confirmation,
                authorization_ref=args.authorization_ref,
                expected_owner_session=args.expected_owner_session,
                expected_revision=args.expected_revision,
                expected_block_sha256=args.expected_block_sha256,
                expected_worktree=args.expected_worktree,
                new_owner_session=args.new_owner_session,
                new_owner_token=args.new_owner_token,
                new_worktree=args.new_worktree,
                reason=args.reason,
                lease_seconds=args.lease_seconds,
            )
            result = _success(
                "User-authorized administrative takeover completed atomically.",
                task,
                ledger.memory_path,
            )
        else:
            task = ledger.takeover(
                task_id=args.task_id,
                expected_owner_session=args.expected_owner_session,
                expected_revision=args.expected_revision,
                expected_block_sha256=args.expected_block_sha256,
                new_owner_session=args.new_owner_session,
                new_owner_token=args.new_owner_token,
                new_worktree=args.new_worktree,
                reason=args.reason,
                lease_seconds=args.lease_seconds,
            )
            result = _success(
                "Expired task ownership taken over atomically.",
                task,
                ledger.memory_path,
            )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except LedgerError as exc:
        memory_path = ledger.memory_path if ledger else root / MEMORY_RELATIVE
        print(json.dumps(_error(exc, memory_path), ensure_ascii=True, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

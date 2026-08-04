"""Manage evidence-based promotion from medium to long project memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import textwrap
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
REGISTRY_RELATIVE = Path(".agents/memory/21_MEMORY_MATURITY.json")
DOSSIER_RELATIVE = Path(".agents/memory/05_PROJECT_MEMORY_LONG.md")
LOCK_RELATIVE = Path(".agents/memory/.memory_maturity.lock")
AUTHORITY_RELATIVE = Path(".agents/memory/18_AUTHORITY_INDEX.json")
METHOD_RELATIVE = Path(".agents/memory/13_METHOD_STATE.json")
CLAIM_RELATIVE = Path(".agents/memory/14_CLAIM_REGISTRY.json")
MEDIUM_RELATIVE = Path(".agents/memory/04_PROJECT_MEMORY_MEDIUM.md")

DOSSIER_START = "<!-- memory-maturity:dossier:start -->"
DOSSIER_END = "<!-- memory-maturity:dossier:end -->"
FORWARD_STATES = [
    "CANDIDATE",
    "EVIDENCE_BOUND",
    "REVIEWED",
    "ACCEPTED",
    "PROMOTED",
]
BRANCH_STATES = {
    "BLOCKED",
    "REJECTED",
    "REVALIDATION_REQUIRED",
    "CONTRADICTED",
    "SUPERSEDED",
    "ARCHIVED",
}
ALL_STATES = set(FORWARD_STATES) | BRANCH_STATES
KINDS = {
    "project_fact",
    "project_contract",
    "validated_method",
    "supported_claim",
    "validated_correction",
    "limitation",
}
SECTIONS = {
    "project_scope": "Project Abstract and Scope",
    "scientific_questions": "Scientific Questions and Boundaries",
    "data_and_system": "Data and System Architecture",
    "validated_methods": "Validated Methods and Contracts",
    "supported_findings": "Supported Results and Claims",
    "corrective_methods": "Reusable Corrective Methods",
    "limitations": "Limitations and Non-Generalization Boundaries",
    "governance": "Reproducibility and Governance",
}
EVIDENCE_CLASSES = {
    "CODE_VERIFIED",
    "ARTIFACT_VERIFIED",
    "RUN_VERIFIED",
    "HUMAN_VERIFIED",
}
MEDIUM_DISPOSITIONS = {
    "resolved_removed_from_active",
    "retained_non_authoritative_history",
    "not_from_medium",
}


class MaturityError(RuntimeError):
    """Structured, recoverable maturity-manager failure."""

    def __init__(
        self,
        code: str,
        summary: str,
        root_cause_hint: str,
        safe_retry: str,
        stop_condition: str,
    ) -> None:
        super().__init__(summary)
        self.code = code
        self.summary = summary
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


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_hash(entry: dict[str, Any]) -> str:
    candidate = deepcopy(entry)
    candidate.pop("entry_sha256", None)
    return _sha256_bytes(_canonical_bytes(candidate))


def _refresh_entry_hash(entry: dict[str, Any]) -> None:
    entry["entry_sha256"] = _entry_hash(entry)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
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


@contextmanager
def exclusive_file_lock(path: Path, timeout_seconds: float) -> Iterator[None]:
    """Acquire a process lock that the operating system releases on crash."""
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
                    raise MaturityError(
                        "lock_timeout",
                        "Another process owns the maturity registry lock.",
                        "A concurrent session is changing memory maturity state.",
                        "Wait, inspect the registry, then retry with fresh CAS values.",
                        "Do not bypass the lock or edit generated dossier text.",
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


def _resolve_ref(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _find_entry(payload: dict[str, Any], entry_id: str) -> dict[str, Any]:
    for entry in payload.get("entries", []):
        if entry.get("entry_id") == entry_id:
            return entry
    raise MaturityError(
        "entry_missing",
        f"Memory maturity entry {entry_id} does not exist.",
        "The requested ID is absent or belongs to another registry snapshot.",
        "Run inspect or scan and retry with an existing entry ID.",
        "Do not create a duplicate entry to bypass the missing ID.",
    )


def _lookup_authority(root: Path, scope: str) -> str | None:
    payload = _load_json(root / AUTHORITY_RELATIVE)
    for entry in payload.get("entries", []):
        if entry.get("scope") == scope:
            return str(entry.get("current_authority"))
    return None


def _lookup_method(root: Path, method_id: str) -> str | None:
    payload = _load_json(root / METHOD_RELATIVE)
    for entry in payload.get("entries", []):
        if entry.get("method_id") == method_id:
            return str(entry.get("state"))
    return None


def _lookup_claim(root: Path, claim_id: str) -> str | None:
    payload = _load_json(root / CLAIM_RELATIVE)
    for entry in payload.get("claims", []):
        if entry.get("claim_id") == claim_id:
            return str(entry.get("status"))
    return None


def _medium_source_is_active(root: Path, entry_id: str) -> bool:
    path = root / MEDIUM_RELATIVE
    if not path.is_file():
        raise MaturityError(
            "medium_authority_missing",
            "The canonical medium-memory authority is missing.",
            "Source disposition cannot be verified against active medium state.",
            "Restore file 04 and retry promotion.",
            "Do not promote while source authority is unverifiable.",
        )
    text = path.read_text(encoding="utf-8")
    marker = "## Active cross-day entries"
    start = text.find(marker)
    if start < 0:
        raise MaturityError(
            "medium_active_section_missing",
            "Medium memory has no canonical active-entry section.",
            "The manager cannot distinguish active truth from history.",
            "Restore the active-entry heading and retry.",
            "Do not infer source disposition from file age or location.",
        )
    end = text.find("\n## ", start + len(marker))
    active = text[start:] if end < 0 else text[start:end]
    return f"`{entry_id}`" in active


def evaluate_trigger(root: Path, trigger: dict[str, Any]) -> dict[str, Any]:
    trigger_type = trigger.get("type")
    result: dict[str, Any] = {
        "type": trigger_type,
        "status": "PASS",
        "summary": "Revalidation trigger still matches.",
    }
    if trigger_type == "file_sha256":
        path = _resolve_ref(root, str(trigger.get("path", "")))
        expected = str(trigger.get("expected_sha256", ""))
        if not path.is_file():
            result.update(status="FAIL", summary=f"Missing trigger path: {path}")
        else:
            actual = _sha256_file(path)
            result["actual_sha256"] = actual
            if actual != expected:
                result.update(status="FAIL", summary="Trigger file hash changed.")
    elif trigger_type == "authority_scope":
        actual = _lookup_authority(root, str(trigger.get("scope", "")))
        expected = trigger.get("expected_current_authority")
        result["actual_current_authority"] = actual
        if actual != expected:
            result.update(status="FAIL", summary="Current authority changed.")
    elif trigger_type == "method_state":
        actual = _lookup_method(root, str(trigger.get("method_id", "")))
        allowed = trigger.get("allowed_states", [])
        result["actual_state"] = actual
        if actual not in allowed:
            result.update(status="FAIL", summary="Method left an accepted state.")
    elif trigger_type == "claim_status":
        actual = _lookup_claim(root, str(trigger.get("claim_id", "")))
        allowed = trigger.get("allowed_statuses", [])
        result["actual_status"] = actual
        if actual not in allowed:
            result.update(status="FAIL", summary="Claim is no longer supported.")
    elif trigger_type == "manual_condition":
        result.update(
            status="MANUAL",
            summary=str(trigger.get("condition", "Manual review is required.")),
        )
    else:
        result.update(status="FAIL", summary="Unknown revalidation trigger type.")
    return result


def _check_ref(
    root: Path,
    reference: dict[str, Any],
    require_hash: bool,
    label: str,
) -> list[str]:
    errors: list[str] = []
    path_value = reference.get("path")
    if not _nonempty(path_value):
        return [f"{label}_path_missing"]
    path = _resolve_ref(root, str(path_value))
    if not path.is_file():
        errors.append(f"{label}_file_missing:{path_value}")
        return errors
    expected = reference.get("sha256")
    if require_hash and not _nonempty(expected):
        errors.append(f"{label}_hash_missing:{path_value}")
    elif _nonempty(expected) and _sha256_file(path) != expected:
        errors.append(f"{label}_hash_mismatch:{path_value}")
    if label == "evidence" and reference.get("evidence_class") not in EVIDENCE_CLASSES:
        errors.append(f"evidence_class_invalid:{path_value}")
    return errors


def entry_gate_errors(root: Path, entry: dict[str, Any]) -> list[str]:
    """Return deterministic promotion-gate failures for one entry."""
    errors: list[str] = []
    required_scalars = (
        "entry_id",
        "knowledge_kind",
        "dossier_section",
        "title",
        "summary",
        "scope",
        "reuse_value",
        "source_medium_entry",
        "created_by",
    )
    for field in required_scalars:
        if not _nonempty(entry.get(field)):
            errors.append(f"entry_missing:{field}")
    required_lists = (
        "source_refs",
        "authority_refs",
        "evidence_refs",
        "limitations",
        "invalidation_conditions",
        "revalidation_triggers",
    )
    for field in required_lists:
        if not _nonempty(entry.get(field)):
            errors.append(f"entry_missing:{field}")
    kind = entry.get("knowledge_kind")
    if kind not in KINDS:
        errors.append(f"knowledge_kind_invalid:{kind}")
    section = entry.get("dossier_section")
    if section not in SECTIONS:
        errors.append(f"dossier_section_invalid:{section}")
    for reference in entry.get("source_refs", []):
        errors.extend(_check_ref(root, reference, True, "source"))
    for reference in entry.get("authority_refs", []):
        errors.extend(_check_ref(root, reference, False, "authority"))
    for reference in entry.get("evidence_refs", []):
        errors.extend(_check_ref(root, reference, True, "evidence"))
    for trigger in entry.get("revalidation_triggers", []):
        if trigger.get("type") not in {
            "file_sha256",
            "authority_scope",
            "method_state",
            "claim_status",
            "manual_condition",
        }:
            errors.append(f"trigger_type_invalid:{trigger.get('type')}")
    payload = entry.get("kind_payload", {})
    if kind == "project_contract":
        for field in ("contract_id", "change_authority"):
            if not _nonempty(payload.get(field)):
                errors.append(f"project_contract_missing:{field}")
    elif kind == "validated_method":
        method_id = payload.get("method_id")
        if not _nonempty(method_id):
            errors.append("validated_method_missing:method_id")
        elif _lookup_method(root, str(method_id)) not in {"FROZEN", "PROMOTED"}:
            errors.append(f"validated_method_not_frozen:{method_id}")
    elif kind == "supported_claim":
        claim_id = payload.get("claim_id")
        if not _nonempty(claim_id):
            errors.append("supported_claim_missing:claim_id")
        elif _lookup_claim(root, str(claim_id)) != "SUPPORTED":
            errors.append(f"claim_not_supported:{claim_id}")
    elif kind == "validated_correction":
        for field in (
            "root_cause",
            "validated_correction",
            "reuse_when",
            "do_not_reuse_when",
        ):
            if not _nonempty(payload.get(field)):
                errors.append(f"validated_correction_missing:{field}")
    elif kind == "limitation":
        for field in ("applies_to", "boundary"):
            if not _nonempty(payload.get(field)):
                errors.append(f"limitation_missing:{field}")
    return errors


def evaluate_entry(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    gate_errors = entry_gate_errors(root, entry)
    triggers = [
        evaluate_trigger(root, item)
        for item in entry.get("revalidation_triggers", [])
    ]
    failed_triggers = [item for item in triggers if item["status"] == "FAIL"]
    manual_triggers = [item for item in triggers if item["status"] == "MANUAL"]
    state = entry.get("state")
    return {
        "entry_id": entry.get("entry_id"),
        "state": state,
        "gate_errors": gate_errors,
        "trigger_results": triggers,
        "promotion_eligible": state == "ACCEPTED"
        and not gate_errors
        and not failed_triggers,
        "revalidation_required": state == "PROMOTED" and bool(failed_triggers),
        "manual_review_conditions": len(manual_triggers),
    }


def validate_registry(root: Path, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "pig.memory-maturity.v1":
        errors.append("maturity_schema_invalid")
    if not isinstance(payload.get("registry_revision"), int):
        errors.append("maturity_registry_revision_invalid")
    entries = payload.get("entries", [])
    ids = [entry.get("entry_id") for entry in entries]
    for entry_id in ids:
        if ids.count(entry_id) > 1:
            errors.append(f"maturity_duplicate_entry:{entry_id}")
    for entry in entries:
        entry_id = entry.get("entry_id", "<missing>")
        state = entry.get("state")
        if state not in ALL_STATES:
            errors.append(f"maturity_state_invalid:{entry_id}:{state}")
        if entry.get("entry_sha256") != _entry_hash(entry):
            errors.append(f"maturity_entry_hash_mismatch:{entry_id}")
        if not isinstance(entry.get("revision"), int):
            errors.append(f"maturity_entry_revision_invalid:{entry_id}")
        transitions = entry.get("transitions", [])
        for transition in transitions:
            source = transition.get("from_state")
            target = transition.get("to_state")
            if source in FORWARD_STATES and target in FORWARD_STATES:
                if FORWARD_STATES.index(target) != FORWARD_STATES.index(source) + 1:
                    errors.append(
                        f"maturity_transition_skips:{entry_id}:{source}->{target}"
                    )
            elif target not in BRANCH_STATES and source not in BRANCH_STATES:
                errors.append(
                    f"maturity_transition_invalid:{entry_id}:{source}->{target}"
                )
        if state == "PROMOTED" and not _nonempty(entry.get("promotion")):
            errors.append(f"maturity_promotion_missing:{entry_id}")
        evaluation = evaluate_entry(root, entry)
        if evaluation["revalidation_required"]:
            errors.append(f"maturity_revalidation_required:{entry_id}")
    dossier = payload.get("dossier", {})
    dossier_path = root / DOSSIER_RELATIVE
    expected = dossier.get("sha256")
    if dossier.get("sync_status") != "SYNCED":
        errors.append("maturity_dossier_not_synced")
    if not dossier_path.is_file() or not _nonempty(expected):
        errors.append("maturity_dossier_hash_missing")
    elif _sha256_file(dossier_path) != expected:
        errors.append("maturity_dossier_hash_mismatch")
    return errors


def _transition(
    entry: dict[str, Any],
    target: str,
    authority: str,
    reason: str,
    current: datetime,
) -> None:
    source = str(entry["state"])
    entry.setdefault("transitions", []).append(
        {
            "from_state": source,
            "to_state": target,
            "timestamp": _iso(current),
            "authority": authority,
            "reason": reason,
        }
    )
    entry["state"] = target


def _format_refs(references: list[dict[str, Any]]) -> str:
    values = []
    for reference in references:
        path = str(reference.get("path"))
        digest = reference.get("sha256")
        values.append(f"`{path}` ({str(digest)[:12]})" if digest else f"`{path}`")
    return ", ".join(values)


def _format_list(values: list[str]) -> str:
    return "; ".join(str(value) for value in values)


def _append_wrapped(
    lines: list[str],
    value: str,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> None:
    lines.extend(
        textwrap.wrap(
            value,
            width=99,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
        )
        or [initial_indent.rstrip()]
    )


def render_dossier(root: Path, payload: dict[str, Any]) -> str:
    """Render the generated living-dossier section from registry truth."""
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in SECTIONS}
    revalidation: list[dict[str, Any]] = []
    for entry in payload.get("entries", []):
        evaluation = evaluate_entry(root, entry)
        if entry.get("state") == "PROMOTED" and not evaluation["revalidation_required"]:
            grouped[str(entry["dossier_section"])].append(entry)
        elif entry.get("state") == "REVALIDATION_REQUIRED" or evaluation[
            "revalidation_required"
        ]:
            revalidation.append(entry)
    lines = [
        DOSSIER_START,
        "## Living Project Dossier",
        "",
        "This is a generated, evidence-bound reading surface. Its machine authority is",
        "`21_MEMORY_MATURITY.json`. Elapsed inactivity never promotes knowledge;",
        "acceptance and revalidation events do.",
        "",
    ]
    for section_id, heading in SECTIONS.items():
        lines.extend([f"### {heading}", ""])
        entries = sorted(grouped[section_id], key=lambda item: item["entry_id"])
        if not entries:
            lines.extend(["No maturity-registry entry is currently promoted here.", ""])
            continue
        for entry in entries:
            acceptance = entry.get("acceptance", {})
            lines.extend([f"#### {entry['title']}", ""])
            _append_wrapped(lines, str(entry["summary"]))
            lines.append("")
            fields = (
                ("Scope", str(entry["scope"])),
                ("Authority", _format_refs(entry["authority_refs"])),
                ("Evidence", _format_refs(entry["evidence_refs"])),
                (
                    "Accepted",
                    f"{acceptance.get('accepted_at')} by "
                    f"{acceptance.get('reviewer')}",
                ),
                ("Limitations", _format_list(entry["limitations"])),
                (
                    "Invalidation",
                    _format_list(entry["invalidation_conditions"]),
                ),
            )
            for label, value in fields:
                _append_wrapped(
                    lines,
                    value,
                    initial_indent=f"- {label}: ",
                    subsequent_indent="  ",
                )
            lines.append("")
    lines.extend(["### Knowledge Awaiting Revalidation", ""])
    if not revalidation:
        lines.extend(["None.", ""])
    else:
        for entry in sorted(revalidation, key=lambda item: item["entry_id"]):
            lines.append(f"- `{entry['entry_id']}`: {entry['title']}")
        lines.append("")
    lines.extend(
        [
            "### Dossier Reading Contract",
            "",
            "Entries are current only while their registered authority, method, claim,",
            "artifact, and invalidation triggers remain satisfied. Superseded or",
            "contradicted knowledge remains in registry history, not as current truth.",
            "",
            DOSSIER_END,
        ]
    )
    return "\n".join(lines)


def _replace_dossier(text: str, generated: str) -> str:
    start = text.find(DOSSIER_START)
    end = text.find(DOSSIER_END)
    if start < 0 or end < start:
        raise MaturityError(
            "dossier_markers_missing",
            "The long-memory dossier markers are missing or malformed.",
            "The generated surface cannot be replaced without a stable boundary.",
            "Restore the two canonical markers, then rerun synthesize.",
            "Do not overwrite the whole Markdown file.",
        )
    end += len(DOSSIER_END)
    return text[:start] + generated + text[end:]


class MaturityRegistry:
    """Atomic state machine for curated long-memory promotion."""

    def __init__(
        self,
        root: Path = ROOT,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        self.root = root.resolve()
        self.registry_path = self.root / REGISTRY_RELATIVE
        self.dossier_path = self.root / DOSSIER_RELATIVE
        self.lock_path = self.root / LOCK_RELATIVE
        self.lock_timeout_seconds = lock_timeout_seconds
        if not self.registry_path.is_file() or not self.dossier_path.is_file():
            raise MaturityError(
                "maturity_authority_missing",
                "The maturity registry or long-memory dossier is missing.",
                "This manager requires the canonical project-local authorities.",
                "Restore the registered files and retry.",
                "Do not create a shadow registry in another worktree.",
            )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with exclusive_file_lock(self.lock_path, self.lock_timeout_seconds):
            yield

    def _read(self) -> dict[str, Any]:
        return _load_json(self.registry_path)

    def _commit_registry(self, payload: dict[str, Any]) -> None:
        payload["registry_revision"] = int(payload["registry_revision"]) + 1
        text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
        _atomic_write_text(self.registry_path, text)

    def _verify_cas(
        self,
        entry: dict[str, Any],
        expected_revision: int,
        expected_sha256: str,
    ) -> None:
        if (
            entry.get("revision") != expected_revision
            or entry.get("entry_sha256") != expected_sha256
        ):
            raise MaturityError(
                "stale_entry_cas",
                "The maturity entry changed after it was inspected.",
                "A concurrent session or newer review owns the current revision.",
                "Inspect again and retry only against the fresh revision and hash.",
                "Do not overwrite the newer entry or reuse stale review evidence.",
            )

    def inspect(self, entry_id: str) -> dict[str, Any]:
        with self._locked():
            payload = self._read()
            entry = deepcopy(_find_entry(payload, entry_id))
            return {"entry": entry, "evaluation": evaluate_entry(self.root, entry)}

    def scan(self) -> dict[str, Any]:
        with self._locked():
            payload = self._read()
            evaluations = [
                evaluate_entry(self.root, entry)
                for entry in payload.get("entries", [])
            ]
            return {
                "registry_revision": payload["registry_revision"],
                "entries": evaluations,
                "eligible_entry_ids": [
                    item["entry_id"]
                    for item in evaluations
                    if item["promotion_eligible"]
                ],
                "revalidation_entry_ids": [
                    item["entry_id"]
                    for item in evaluations
                    if item["revalidation_required"]
                ],
            }

    def audit(self) -> dict[str, Any]:
        with self._locked():
            payload = self._read()
            errors = validate_registry(self.root, payload)
            return {
                "status": "PASS" if not errors else "FAIL",
                "errors": errors,
                "registry_revision": payload.get("registry_revision"),
            }

    def register(
        self,
        packet: dict[str, Any],
        created_by: str,
        current: datetime | None = None,
    ) -> dict[str, Any]:
        timestamp = _now(current)
        with self._locked():
            payload = self._read()
            entry_id = str(packet.get("entry_id", ""))
            if not entry_id:
                raise MaturityError(
                    "entry_id_missing",
                    "The candidate packet has no entry_id.",
                    "A stable identity is required for deduplication and lineage.",
                    "Add a unique lowercase dotted ID and retry.",
                    "Do not register an anonymous maturity candidate.",
                )
            if any(
                item.get("entry_id") == entry_id
                for item in payload.get("entries", [])
            ):
                raise MaturityError(
                    "entry_id_collision",
                    f"Memory maturity entry {entry_id} already exists.",
                    "The candidate may duplicate existing durable knowledge.",
                    "Inspect the existing entry and update or supersede it.",
                    "Do not create a second canonical home for the same fact.",
                )
            entry = deepcopy(packet)
            entry.update(
                {
                    "created_by": created_by,
                    "registered_at": _iso(timestamp),
                    "state": "CANDIDATE",
                    "revision": 1,
                    "transitions": [],
                    "supersedes": entry.get("supersedes", []),
                }
            )
            _refresh_entry_hash(entry)
            payload.setdefault("entries", []).append(entry)
            self._commit_registry(payload)
            return {
                "entry": deepcopy(entry),
                "evaluation": evaluate_entry(self.root, entry),
            }

    def review(
        self,
        entry_id: str,
        decision: str,
        reviewer: str,
        authority: str,
        basis: str,
        medium_disposition: str,
        independent_review: bool,
        expected_revision: int,
        expected_sha256: str,
        current: datetime | None = None,
    ) -> dict[str, Any]:
        timestamp = _now(current)
        with self._locked():
            payload = self._read()
            entry = _find_entry(payload, entry_id)
            self._verify_cas(entry, expected_revision, expected_sha256)
            if decision not in {"accept", "hold", "reject"}:
                raise MaturityError(
                    "review_decision_invalid",
                    f"Unsupported review decision: {decision}.",
                    "The transition contract accepts only accept, hold, or reject.",
                    "Choose one declared decision and retry.",
                    "Do not invent a state outside the registry state machine.",
                )
            if decision == "accept" and medium_disposition not in MEDIUM_DISPOSITIONS:
                raise MaturityError(
                    "medium_disposition_invalid",
                    "The source-medium disposition is not declared.",
                    "Promotion must not leave two current authorities.",
                    "Declare how the medium source became non-authoritative.",
                    "Do not promote while the source remains active medium truth.",
                )
            if decision == "accept":
                errors = entry_gate_errors(self.root, entry)
                trigger_results = [
                    evaluate_trigger(self.root, item)
                    for item in entry.get("revalidation_triggers", [])
                ]
                errors.extend(
                    "trigger_failed:" + item["type"]
                    for item in trigger_results
                    if item["status"] == "FAIL"
                )
                if entry.get("knowledge_kind") in {
                    "validated_method",
                    "supported_claim",
                } and not independent_review:
                    errors.append("independent_review_required")
                if errors:
                    raise MaturityError(
                        "acceptance_gates_failed",
                        "The candidate cannot be accepted: " + ", ".join(errors),
                        "One or more evidence, authority, type, or review gates failed.",
                        "Correct the packet or authority, then inspect and review again.",
                        "Do not use elapsed time or prose confidence to bypass a gate.",
                    )
                if entry["state"] in BRANCH_STATES:
                    _transition(
                        entry,
                        "EVIDENCE_BOUND",
                        authority,
                        "Reopened evidence passed typed gates.",
                        timestamp,
                    )
                elif entry["state"] == "CANDIDATE":
                    _transition(
                        entry,
                        "EVIDENCE_BOUND",
                        authority,
                        "Required evidence and authority fields passed.",
                        timestamp,
                    )
                if entry["state"] == "EVIDENCE_BOUND":
                    _transition(
                        entry,
                        "REVIEWED",
                        authority,
                        "A deliberate review event was recorded.",
                        timestamp,
                    )
                if entry["state"] == "REVIEWED":
                    _transition(
                        entry,
                        "ACCEPTED",
                        authority,
                        "Reviewer accepted scope, evidence, and limitations.",
                        timestamp,
                    )
                if entry["state"] != "ACCEPTED":
                    raise MaturityError(
                        "entry_not_reviewable",
                        f"Entry in state {entry['state']} cannot be accepted.",
                        "The requested review does not match the current lifecycle state.",
                        "Inspect the entry and use the valid next transition.",
                        "Do not rewrite promoted or archived history in place.",
                    )
                entry["acceptance"] = {
                    "reviewer": reviewer,
                    "authority": authority,
                    "basis": basis,
                    "accepted_at": _iso(timestamp),
                    "independent_review": independent_review,
                    "medium_disposition": medium_disposition,
                }
            else:
                target = "BLOCKED" if decision == "hold" else "REJECTED"
                _transition(entry, target, authority, basis, timestamp)
                entry["latest_review"] = {
                    "reviewer": reviewer,
                    "authority": authority,
                    "basis": basis,
                    "reviewed_at": _iso(timestamp),
                }
            entry["revision"] = int(entry["revision"]) + 1
            _refresh_entry_hash(entry)
            self._commit_registry(payload)
            return {
                "entry": deepcopy(entry),
                "evaluation": evaluate_entry(self.root, entry),
            }

    def revise(
        self,
        entry_id: str,
        packet: dict[str, Any],
        authority: str,
        reason: str,
        expected_revision: int,
        expected_sha256: str,
        current: datetime | None = None,
    ) -> dict[str, Any]:
        timestamp = _now(current)
        with self._locked():
            payload = self._read()
            entry = _find_entry(payload, entry_id)
            self._verify_cas(entry, expected_revision, expected_sha256)
            if entry.get("state") not in {
                "CANDIDATE",
                "BLOCKED",
                "REVALIDATION_REQUIRED",
                "CONTRADICTED",
            }:
                raise MaturityError(
                    "revision_state_invalid",
                    f"Entry in state {entry.get('state')} cannot be revised.",
                    "Accepted truth must be reopened before evidence is replaced.",
                    "Reopen the entry, then revise against fresh CAS values.",
                    "Do not rewrite an accepted or promoted packet in place.",
                )
            if packet.get("entry_id") != entry_id:
                raise MaturityError(
                    "revision_entry_id_mismatch",
                    "The revision packet targets a different entry ID.",
                    "Revision must preserve canonical identity and history.",
                    "Correct the packet ID and retry with fresh CAS values.",
                    "Do not fork one fact accidentally during revalidation.",
                )
            history = {
                "revised_at": _iso(timestamp),
                "authority": authority,
                "reason": reason,
                "previous_entry_sha256": entry["entry_sha256"],
            }
            prior_acceptance = entry.get("acceptance")
            prior_promotion = entry.get("promotion")
            preserved = {
                "created_by": entry["created_by"],
                "registered_at": entry["registered_at"],
                "state": entry["state"],
                "revision": int(entry["revision"]) + 1,
                "transitions": entry.get("transitions", []),
                "revision_history": entry.get("revision_history", []) + [history],
                "historical_acceptances": entry.get(
                    "historical_acceptances",
                    [],
                ),
                "historical_promotions": entry.get("historical_promotions", []),
            }
            if prior_acceptance:
                preserved["historical_acceptances"].append(prior_acceptance)
            if prior_promotion:
                preserved["historical_promotions"].append(prior_promotion)
            entry.clear()
            entry.update(deepcopy(packet))
            entry.update(preserved)
            _refresh_entry_hash(entry)
            self._commit_registry(payload)
            return {
                "entry": deepcopy(entry),
                "evaluation": evaluate_entry(self.root, entry),
            }

    def _synthesize_locked(
        self,
        payload: dict[str, Any],
        current: datetime,
    ) -> str:
        generated = render_dossier(self.root, payload)
        current_text = self.dossier_path.read_text(encoding="utf-8")
        updated = _replace_dossier(current_text, generated)
        _atomic_write_text(self.dossier_path, updated)
        dossier = payload.setdefault("dossier", {})
        dossier.update(
            {
                "path": DOSSIER_RELATIVE.as_posix(),
                "last_synthesized_at": _iso(current),
                "sha256": _sha256_file(self.dossier_path),
                "sync_status": "SYNCED",
            }
        )
        self._commit_registry(payload)
        return dossier["sha256"]

    def synthesize(self, current: datetime | None = None) -> dict[str, Any]:
        timestamp = _now(current)
        with self._locked():
            payload = self._read()
            digest = self._synthesize_locked(payload, timestamp)
            return {
                "dossier_sha256": digest,
                "registry_revision": payload["registry_revision"],
            }

    def promote(
        self,
        entry_id: str,
        authority: str,
        expected_revision: int,
        expected_sha256: str,
        current: datetime | None = None,
    ) -> dict[str, Any]:
        timestamp = _now(current)
        with self._locked():
            payload = self._read()
            entry = _find_entry(payload, entry_id)
            self._verify_cas(entry, expected_revision, expected_sha256)
            evaluation = evaluate_entry(self.root, entry)
            if not evaluation["promotion_eligible"]:
                details = evaluation["gate_errors"]
                raise MaturityError(
                    "promotion_gates_failed",
                    "The entry is not eligible for promotion: "
                    + ", ".join(details or [str(entry.get("state"))]),
                    "Acceptance, evidence, or revalidation state is incomplete.",
                    "Inspect the entry, satisfy the failed gate, and review it again.",
                    "Do not promote based on age, inactivity, or task completion alone.",
                )
            acceptance = entry.get("acceptance", {})
            disposition = acceptance.get("medium_disposition")
            if disposition not in MEDIUM_DISPOSITIONS:
                raise MaturityError(
                    "medium_still_authoritative",
                    "The medium source has no resolved disposition.",
                    "Dual current authority would make retrieval ambiguous.",
                    "Close or demote the medium source, then record its disposition.",
                    "Do not retain one fact as active in both medium and long memory.",
                )
            source_id = str(entry.get("source_medium_entry", ""))
            if disposition == "not_from_medium" and source_id != "not_applicable":
                raise MaturityError(
                    "not_from_medium_inconsistent",
                    "The candidate claims no medium source but names one.",
                    "The source-disposition packet is internally inconsistent.",
                    "Use not_applicable or declare the real medium disposition.",
                    "Do not bypass the active-medium duplicate check.",
                )
            if disposition != "not_from_medium" and _medium_source_is_active(
                self.root,
                source_id,
            ):
                raise MaturityError(
                    "medium_source_still_active",
                    f"Medium source {source_id} is still active in file 04.",
                    "Promotion would create two current authorities for one fact.",
                    "Remove or demote the exact medium source, then inspect and retry.",
                    "Do not promote while the source remains in Active cross-day entries.",
                )
            _transition(
                entry,
                "PROMOTED",
                authority,
                "Canonical long-memory promotion passed all gates.",
                timestamp,
            )
            entry["promotion"] = {
                "promoted_at": _iso(timestamp),
                "authority": authority,
                "dossier_path": DOSSIER_RELATIVE.as_posix(),
            }
            entry["revision"] = int(entry["revision"]) + 1
            _refresh_entry_hash(entry)
            payload.setdefault("dossier", {})["sync_status"] = "NEEDS_SYNC"
            self._commit_registry(payload)
            digest = self._synthesize_locked(payload, timestamp)
            return {
                "entry": deepcopy(entry),
                "dossier_sha256": digest,
                "registry_revision": payload["registry_revision"],
            }

    def reopen(
        self,
        entry_id: str,
        authority: str,
        reason: str,
        expected_revision: int,
        expected_sha256: str,
        current: datetime | None = None,
    ) -> dict[str, Any]:
        timestamp = _now(current)
        with self._locked():
            payload = self._read()
            entry = _find_entry(payload, entry_id)
            self._verify_cas(entry, expected_revision, expected_sha256)
            if entry.get("state") not in {"ACCEPTED", "PROMOTED"}:
                raise MaturityError(
                    "reopen_state_invalid",
                    f"Entry in state {entry.get('state')} cannot be reopened.",
                    "Only accepted or promoted knowledge can lose current validity.",
                    "Inspect the entry and choose archive or review as appropriate.",
                    "Do not rewrite historical transitions.",
                )
            _transition(
                entry,
                "REVALIDATION_REQUIRED",
                authority,
                reason,
                timestamp,
            )
            entry["revision"] = int(entry["revision"]) + 1
            _refresh_entry_hash(entry)
            payload.setdefault("dossier", {})["sync_status"] = "NEEDS_SYNC"
            self._commit_registry(payload)
            digest = self._synthesize_locked(payload, timestamp)
            return {"entry": deepcopy(entry), "dossier_sha256": digest}

    def archive(
        self,
        entry_id: str,
        authority: str,
        reason: str,
        expected_revision: int,
        expected_sha256: str,
        current: datetime | None = None,
    ) -> dict[str, Any]:
        timestamp = _now(current)
        with self._locked():
            payload = self._read()
            entry = _find_entry(payload, entry_id)
            self._verify_cas(entry, expected_revision, expected_sha256)
            if entry.get("state") == "PROMOTED":
                raise MaturityError(
                    "archive_promoted_forbidden",
                    "Promoted knowledge must be reopened or superseded first.",
                    "Archiving directly would erase its validity transition.",
                    "Use reopen, record evidence, then archive if appropriate.",
                    "Do not silently remove accepted project history.",
                )
            _transition(entry, "ARCHIVED", authority, reason, timestamp)
            entry["revision"] = int(entry["revision"]) + 1
            _refresh_entry_hash(entry)
            self._commit_registry(payload)
            return {"entry": deepcopy(entry)}


def _success(
    summary: str,
    artifacts: list[str],
    next_actions: list[str],
    **payload: Any,
) -> dict[str, Any]:
    result = {
        "status": "success",
        "summary": summary,
        "next_actions": next_actions,
        "artifacts": artifacts,
    }
    result.update(payload)
    return result


def _error(error: MaturityError) -> dict[str, Any]:
    return {
        "status": "error",
        "summary": error.summary,
        "next_actions": [error.safe_retry],
        "artifacts": [],
        "root_cause_hint": error.root_cause_hint,
        "safe_retry": error.safe_retry,
        "stop_condition": error.stop_condition,
        "error_code": error.code,
    }


def _add_cas(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--expected-entry-sha256", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("scan")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--entry-id", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--packet", required=True, type=Path)
    register_parser.add_argument("--created-by", required=True)
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--entry-id", required=True)
    review_parser.add_argument("--decision", required=True)
    review_parser.add_argument("--reviewer", required=True)
    review_parser.add_argument("--authority", required=True)
    review_parser.add_argument("--basis", required=True)
    review_parser.add_argument("--medium-disposition", required=True)
    review_parser.add_argument("--independent-review", action="store_true")
    _add_cas(review_parser)
    revise_parser = subparsers.add_parser("revise")
    revise_parser.add_argument("--entry-id", required=True)
    revise_parser.add_argument("--packet", required=True, type=Path)
    revise_parser.add_argument("--authority", required=True)
    revise_parser.add_argument("--reason", required=True)
    _add_cas(revise_parser)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--entry-id", required=True)
    promote_parser.add_argument("--authority", required=True)
    _add_cas(promote_parser)
    reopen_parser = subparsers.add_parser("reopen")
    reopen_parser.add_argument("--entry-id", required=True)
    reopen_parser.add_argument("--authority", required=True)
    reopen_parser.add_argument("--reason", required=True)
    _add_cas(reopen_parser)
    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("--entry-id", required=True)
    archive_parser.add_argument("--authority", required=True)
    archive_parser.add_argument("--reason", required=True)
    _add_cas(archive_parser)
    subparsers.add_parser("synthesize")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        registry = MaturityRegistry(args.root)
        artifacts = [str(registry.registry_path)]
        if args.command == "audit":
            result = registry.audit()
            envelope = _success(
                "Memory maturity registry audit completed.",
                artifacts,
                ["Resolve every reported error before promotion."],
                audit=result,
            )
            if result["status"] != "PASS":
                envelope["status"] = "warning"
        elif args.command == "scan":
            result = _success(
                "Memory maturity candidates were classified without mutation.",
                artifacts,
                ["Inspect an eligible or revalidation-required entry."],
                scan=registry.scan(),
            )
        elif args.command == "inspect":
            result = _success(
                "Memory maturity entry inspected under lock.",
                artifacts,
                ["Use fresh revision and hash values for the next mutation."],
                **registry.inspect(args.entry_id),
            )
        elif args.command == "register":
            packet = _load_json(args.packet)
            result = _success(
                "Memory maturity candidate registered atomically.",
                artifacts,
                ["Inspect gate errors before deliberate review."],
                **registry.register(packet, args.created_by),
            )
        elif args.command == "review":
            reviewed = registry.review(
                args.entry_id,
                args.decision,
                args.reviewer,
                args.authority,
                args.basis,
                args.medium_disposition,
                args.independent_review,
                args.expected_revision,
                args.expected_entry_sha256,
            )
            result = _success(
                "Memory maturity review recorded atomically.",
                artifacts,
                ["Promote only when promotion_eligible is true."],
                **reviewed,
            )
        elif args.command == "revise":
            packet = _load_json(args.packet)
            revised = registry.revise(
                args.entry_id,
                packet,
                args.authority,
                args.reason,
                args.expected_revision,
                args.expected_entry_sha256,
            )
            result = _success(
                "Memory maturity evidence packet revised with history preserved.",
                artifacts,
                ["Perform a new deliberate review before promotion."],
                **revised,
            )
        elif args.command == "promote":
            promoted = registry.promote(
                args.entry_id,
                args.authority,
                args.expected_revision,
                args.expected_entry_sha256,
            )
            result = _success(
                "Accepted knowledge promoted and dossier synchronized.",
                artifacts + [str(registry.dossier_path)],
                ["Run audit and preserve registered invalidation triggers."],
                **promoted,
            )
        elif args.command == "reopen":
            reopened = registry.reopen(
                args.entry_id,
                args.authority,
                args.reason,
                args.expected_revision,
                args.expected_entry_sha256,
            )
            result = _success(
                "Promoted knowledge reopened for evidence revalidation.",
                artifacts + [str(registry.dossier_path)],
                ["Re-establish evidence and perform a new review event."],
                **reopened,
            )
        elif args.command == "archive":
            archived = registry.archive(
                args.entry_id,
                args.authority,
                args.reason,
                args.expected_revision,
                args.expected_entry_sha256,
            )
            result = _success(
                "Memory maturity candidate archived with provenance.",
                artifacts,
                ["Retain the archive only for provenance or audit."],
                **archived,
            )
        else:
            synchronized = registry.synthesize()
            result = _success(
                "Living project dossier synchronized from registry truth.",
                artifacts + [str(registry.dossier_path)],
                ["Run audit and inspect any revalidation-required entry."],
                **synchronized,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except MaturityError as error:
        print(json.dumps(_error(error), indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

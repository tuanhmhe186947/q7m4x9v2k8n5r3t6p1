"""Validate project governance contracts and memory lifecycle state."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
MEMORY = ROOT / ".agents" / "memory"
TASK_MANAGER_PATH = Path(__file__).with_name("manage_short_memory.py")
TASK_MANAGER_SPEC = importlib.util.spec_from_file_location(
    "project_short_memory_manager",
    TASK_MANAGER_PATH,
)
assert TASK_MANAGER_SPEC is not None
assert TASK_MANAGER_SPEC.loader is not None
TASK_MANAGER = importlib.util.module_from_spec(TASK_MANAGER_SPEC)
TASK_MANAGER_SPEC.loader.exec_module(TASK_MANAGER)
SHORT_MEMORY_RELATIVE = TASK_MANAGER.MEMORY_RELATIVE
MATURITY_MANAGER_PATH = Path(__file__).with_name("manage_memory_maturity.py")
MATURITY_MANAGER_SPEC = importlib.util.spec_from_file_location(
    "project_memory_maturity_manager",
    MATURITY_MANAGER_PATH,
)
assert MATURITY_MANAGER_SPEC is not None
assert MATURITY_MANAGER_SPEC.loader is not None
MATURITY_MANAGER = importlib.util.module_from_spec(MATURITY_MANAGER_SPEC)
MATURITY_MANAGER_SPEC.loader.exec_module(MATURITY_MANAGER)
AGENT_GOVERNANCE_PATH = Path(__file__).with_name("manage_agent_governance.py")
AGENT_GOVERNANCE_SPEC = importlib.util.spec_from_file_location(
    "project_agent_governance_manager",
    AGENT_GOVERNANCE_PATH,
)
assert AGENT_GOVERNANCE_SPEC is not None
assert AGENT_GOVERNANCE_SPEC.loader is not None
AGENT_GOVERNANCE = importlib.util.module_from_spec(AGENT_GOVERNANCE_SPEC)
AGENT_GOVERNANCE_SPEC.loader.exec_module(AGENT_GOVERNANCE)

SHORT_RE = re.compile(
    r"Opened:\s*`(?P<opened>\d{4}-\d{2}-\d{2})`.*?"
    r"Expires:\s*`(?P<expires>[^`]+)`",
    re.DOTALL,
)

TASK_HEADING_RE = re.compile(
    r"^###\s+(?P<task_id>[A-Z][A-Z0-9-]*-\d{8}-\d{2})\s+-\s+.+$",
    re.MULTILINE,
)
STEP_RE = re.compile(
    r"^- \[(?P<mark>[ xX])\] `(?P<step_id>[A-Z][A-Z0-9-]*-\d+)` "
    r"`\[(?P<status>TODO|IN_PROGRESS|BLOCKED|DONE|DEFERRED|CANCELLED)\]` "
    r"(?P<summary>.+)$",
    re.MULTILINE,
)
CHECKLIST_STATES = {
    "TODO",
    "IN_PROGRESS",
    "BLOCKED",
    "DONE",
    "DEFERRED",
    "CANCELLED",
}
TERMINAL_CHECKLIST_STATES = {"DONE", "CANCELLED"}

EXPECTED_ROUTES = {
    "architecture_or_goal_drift": {
        "agent-architecture-audit",
        "plan-orchestrate",
    },
    "agent_behavior_debugging": {"agent-introspection-debugging"},
    "handoff_quality": {"agent-self-evaluation"},
    "agent_task_evaluation": {"agent-eval", "eval-harness"},
    "context_or_memory": {"iterative-retrieval", "knowledge-ops"},
    "action_tool_observation": {"agent-harness-construction"},
}

REQUIRED_PORTFOLIO_FIELDS = {
    "skill_id",
    "category",
    "source_root",
    "relative_path",
    "version_or_commit",
    "file_sha256",
    "tool_api_dependencies",
    "selected_date",
    "last_reviewed",
    "last_real_use",
    "proof_task",
    "stale_signal",
    "next_maintenance_action",
}

REQUIRED_CLAIM_FIELDS = {
    "claim_id",
    "claim_text",
    "scope",
    "git_sha",
    "dirty_worktree",
    "config_hash",
    "data_hashes",
    "artifact_hashes",
    "split",
    "seeds",
    "evaluator",
    "environment",
    "evidence_class",
    "quantitative_evidence",
    "limitations",
    "invalidation_condition",
    "authority",
    "reviewer",
}

V2_ACTIVATION_MARKERS = {
    "bootstrap": ".agents/memory/00_AGENT_BOOTSTRAP.md",
    "manager": ".agents/skills/project-state-steward/scripts/manage_agent_governance.py",
    "inventory": ".agents/skills/skill_inventory.json",
    "lifecycle": ".agents/memory/22_WORKTREE_LIFECYCLE.json",
}
V2_ACTIVATION_DOCUMENTS = {
    "AGENTS.md": "AGENTS.md",
    "00_README.md": ".agents/memory/00_README.md",
    "03_PROJECT_RULES.md": ".agents/memory/03_PROJECT_RULES.md",
    "08_WORKFLOW.md": ".agents/memory/08_WORKFLOW.md",
}
V2_AUTHORITY_SCOPES = {
    "skill.portfolio": ".agents/skills/skill_inventory.json",
    "agent.governance": "docs/governance/AGENT_GOVERNANCE_V2.md",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(_read(path))


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {}, "PENDING", "PENDING_FINAL_HASH")


def _resolve_source(root: Path, source_root: str, relative_path: str) -> Path:
    if source_root == "project":
        return root / relative_path
    if source_root == "codex_home":
        return Path.home() / ".codex" / relative_path
    raise ValueError(f"unknown_source_root:{source_root}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_raw_file_identity(
    path: Path,
    *,
    expected_sha256: str,
    error_prefix: str,
) -> list[str]:
    """Validate an external immutable file by its exact stored bytes."""
    expected = expected_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        return [f"{error_prefix}_raw_hash_invalid"]
    if not path.is_file():
        return [f"{error_prefix}_raw_file_missing"]
    if _sha256(path) != expected:
        return [f"{error_prefix}_raw_hash_mismatch"]
    return []


def _bundle_sha256(base: Path, relative_paths: list[str]) -> str:
    resolved_base = base.resolve()
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        normalized = Path(relative).as_posix()
        resolved = (resolved_base / relative).resolve()
        try:
            resolved.relative_to(resolved_base)
        except ValueError as exc:
            raise ValueError(f"skill_bundle_path_escape:{relative}") from exc
        if not resolved.is_file():
            raise ValueError(f"skill_bundle_file_missing:{relative}")
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _git_relative_path(relative_path: str) -> str | None:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    normalized = candidate.as_posix()
    return normalized if normalized not in {"", "."} else None


def _git_blob_oid(root: Path, reference: str, relative_path: str) -> str | None:
    result = _git(
        root,
        ["rev-parse", "--verify", f"{reference}:{relative_path}"],
    )
    value = result.stdout.strip().lower()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        return None
    return value


def _git_index_blob_oid(root: Path, relative_path: str) -> str | None:
    result = _git(root, ["rev-parse", "--verify", f":{relative_path}"])
    value = result.stdout.strip().lower()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        return None
    return value


def _git_worktree_blob_oid(root: Path, relative_path: str) -> str | None:
    result = _git(
        root,
        ["hash-object", f"--path={relative_path}", "--", relative_path],
    )
    value = result.stdout.strip().lower()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        return None
    return value


def _validate_git_tracked_file_identity(
    root: Path,
    *,
    relative_path: str,
    expected_blob_oid: str,
    git_reference: str,
    error_prefix: str,
) -> list[str]:
    """Validate one tracked file without conflating clean checkout EOL with content."""
    normalized = _git_relative_path(relative_path)
    expected = expected_blob_oid.lower()
    if normalized is None:
        return [f"{error_prefix}_git_path_invalid"]
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", expected):
        return [f"{error_prefix}_git_blob_invalid"]
    if not (root / normalized).is_file():
        return [f"{error_prefix}_git_file_missing"]
    tracked = _git(root, ["ls-files", "--error-unmatch", "--", normalized])
    if tracked.returncode:
        return [f"{error_prefix}_git_path_untracked"]

    errors: list[str] = []
    referenced_blob = _git_blob_oid(root, git_reference, normalized)
    if referenced_blob is None:
        errors.append(f"{error_prefix}_git_reference_unresolved")
    elif referenced_blob != expected:
        errors.append(f"{error_prefix}_git_blob_mismatch")

    head_blob = _git_blob_oid(root, "HEAD", normalized)
    if head_blob is None:
        errors.append(f"{error_prefix}_git_head_unresolved")
    elif head_blob != expected:
        errors.append(f"{error_prefix}_git_head_blob_mismatch")

    index_blob = _git_index_blob_oid(root, normalized)
    if index_blob is None:
        errors.append(f"{error_prefix}_git_index_unresolved")
    elif index_blob != expected:
        errors.append(f"{error_prefix}_git_staged_change")

    worktree_blob = _git_worktree_blob_oid(root, normalized)
    if worktree_blob is None:
        errors.append(f"{error_prefix}_git_worktree_content_unresolved")
    elif index_blob is not None and worktree_blob != index_blob:
        errors.append(f"{error_prefix}_git_unstaged_change")
    return errors


def _project_timezone() -> timezone:
    try:
        return ZoneInfo("Asia/Saigon")
    except Exception:
        return timezone(timedelta(hours=7))


def _as_datetime(value: datetime | date | str | None) -> datetime:
    tz = _project_timezone()
    if value is None:
        return datetime.now(tz)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=tz)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tz)
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=tz)


def _check_short_ttl(root: Path, as_of: datetime | date | str | None) -> dict[str, Any]:
    text = _read(root / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md")
    match = SHORT_RE.search(text)
    errors: list[str] = []
    if not match:
        return {
            "status": "FAIL",
            "errors": ["short_missing_opened_or_expires"],
        }
    opened = date.fromisoformat(match.group("opened"))
    expires = datetime.fromisoformat(match.group("expires"))
    now = _as_datetime(as_of)
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=_project_timezone())
    headings = re.findall(r"^##\s+(\d{4}-\d{2}-\d{2})\b", text, re.MULTILINE)
    if any(item != opened.isoformat() for item in headings):
        errors.append("short_contains_noncurrent_dated_heading")
    if now >= expires:
        return {
            "status": "RESET_REQUIRED",
            "errors": errors + ["short_memory_expired"],
            "opened": opened.isoformat(),
            "expires": expires.isoformat(),
            "as_of": now.isoformat(),
        }
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "opened": opened.isoformat(),
        "expires": expires.isoformat(),
        "as_of": now.isoformat(),
    }


def _markdown_section(text: str, heading: str) -> str | None:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return None
    next_heading = text.find("\n## ", start + len(marker))
    end = len(text) if next_heading < 0 else next_heading
    return text[start:end]


def _check_short_checklist(root: Path) -> list[str]:
    path = root / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md"
    text = path.read_bytes().decode("utf-8")
    errors: list[str] = []
    if re.search(r"^##\s+.*Superseded", text, re.MULTILINE | re.IGNORECASE):
        errors.append("short_contains_superseded_history")

    lifecycle_match = SHORT_RE.search(text)
    opened = (
        date.fromisoformat(lifecycle_match.group("opened"))
        if lifecycle_match
        else None
    )
    checklist = _markdown_section(text, "Active Task Checklist")
    if checklist is None:
        return errors + ["short_missing_active_task_checklist"]
    daily_surface = text.replace(checklist, "", 1)
    if len(daily_surface.splitlines()) > 250:
        errors.append("short_exceeds_250_line_budget")
    managed_spans = [
        span
        for span in TASK_MANAGER.task_spans(text)
        if f"- Concurrency: `{TASK_MANAGER.CONCURRENCY_SCHEMA}`." in span["block"]
    ]
    unmanaged_checklist = checklist
    for span in managed_spans:
        unmanaged_checklist = unmanaged_checklist.replace(span["block"], "", 1)
    if len(unmanaged_checklist.splitlines()) > 1200:
        errors.append("short_checklist_exceeds_1200_line_budget")

    legacy_match = re.search(
        r"^- Legacy unmanaged task IDs:(?P<value>[^\r\n]*"
        r"(?:\r?\n  [^\r\n]*)*)",
        text,
        re.MULTILINE,
    )
    legacy_task_ids: set[str] = set()
    if legacy_match is None:
        errors.append("short_missing_legacy_task_allowlist")
    else:
        legacy_value = legacy_match.group("value")
        legacy_task_ids = set(re.findall(r"`([^`]+)`", legacy_value))
        normalized_legacy = legacy_value.strip().rstrip(".").lower()
        if not legacy_task_ids and normalized_legacy != "none":
            errors.append("short_legacy_task_allowlist_invalid")

    task_matches = list(TASK_HEADING_RE.finditer(checklist))
    managed_task_blocks = {span["task_id"]: span["block"] for span in managed_spans}
    if not task_matches and "- None." not in checklist:
        errors.append("short_checklist_missing_task_or_none")
    task_ids: list[str] = []
    step_ids: list[str] = []
    required_task_markers = (
        "- Prompt:",
        "- Status:",
        "- Opened:",
        "- Acceptance:",
        "- Skills:",
    )
    for index, task_match in enumerate(task_matches):
        end = (
            task_matches[index + 1].start()
            if index + 1 < len(task_matches)
            else len(checklist)
        )
        task_body = checklist[task_match.start():end]
        task_id = task_match.group("task_id")
        task_ids.append(task_id)
        status_match = re.search(
            r"^- Status:\s*`(?P<status>[A-Z_]+)`\.?\r?$",
            task_body,
            re.MULTILINE,
        )
        task_status = status_match.group("status") if status_match else None
        if (
            task_status not in TERMINAL_CHECKLIST_STATES
            and len(task_body.splitlines()) > 120
        ):
            errors.append(f"short_task_exceeds_120_line_budget:{task_id}")
        for marker in required_task_markers:
            if marker not in task_body:
                errors.append(f"short_task_missing:{task_id}:{marker[2:-1]}")

        is_managed = (
            f"- Concurrency: `{TASK_MANAGER.CONCURRENCY_SCHEMA}`." in task_body
        )
        task_date = date.fromisoformat(task_id.rsplit("-", 2)[-2])
        if opened is not None and task_date != opened:
            if task_date > opened or not is_managed:
                errors.append(f"short_task_wrong_day:{task_id}")
        if is_managed:
            managed_block = managed_task_blocks.get(task_id, task_body)
            for error in TASK_MANAGER.validate_managed_block(managed_block):
                errors.append(f"short_managed_task_invalid:{task_id}:{error}")
            if task_id in legacy_task_ids:
                errors.append(f"short_managed_task_allowlisted:{task_id}")
        elif task_id not in legacy_task_ids:
            errors.append(f"short_unmanaged_task_not_allowlisted:{task_id}")

        if task_status not in CHECKLIST_STATES:
            errors.append(f"short_task_invalid_status:{task_id}:{task_status}")

        steps = list(STEP_RE.finditer(task_body))
        if not steps:
            errors.append(f"short_task_missing_steps:{task_id}")
            continue
        statuses: list[str] = []
        for step_index, step_match in enumerate(steps):
            step_end = (
                steps[step_index + 1].start()
                if step_index + 1 < len(steps)
                else len(task_body)
            )
            detail = task_body[step_match.end():step_end]
            step_id = step_match.group("step_id")
            status = step_match.group("status")
            mark = step_match.group("mark").lower()
            step_ids.append(step_id)
            statuses.append(status)
            if status in TERMINAL_CHECKLIST_STATES:
                if mark != "x":
                    errors.append(f"short_terminal_step_unchecked:{step_id}")
                if "- Evidence:" not in detail:
                    errors.append(f"short_terminal_step_missing_evidence:{step_id}")
            else:
                if mark == "x":
                    errors.append(f"short_open_step_checked:{step_id}")
                if "- Next:" not in detail:
                    errors.append(f"short_open_step_missing_next:{step_id}")

        in_progress_count = statuses.count("IN_PROGRESS")
        if in_progress_count > 1:
            errors.append(f"short_task_multiple_in_progress:{task_id}")
        if task_status == "IN_PROGRESS" and in_progress_count != 1:
            errors.append(f"short_task_status_mismatch:{task_id}:IN_PROGRESS")
        if task_status == "DONE" and any(
            status not in TERMINAL_CHECKLIST_STATES for status in statuses
        ):
            errors.append(f"short_task_done_with_open_steps:{task_id}")
        if task_status == "BLOCKED" and "BLOCKED" not in statuses:
            errors.append(f"short_task_status_mismatch:{task_id}:BLOCKED")

    for task_id in task_ids:
        if task_ids.count(task_id) > 1:
            errors.append(f"short_duplicate_task:{task_id}")
    for step_id in step_ids:
        if step_ids.count(step_id) > 1:
            errors.append(f"short_duplicate_step:{step_id}")
    for task_id in legacy_task_ids - set(task_ids):
        errors.append(f"short_legacy_task_allowlist_stale:{task_id}")

    previous = _markdown_section(text, "Previous-Day Closeout")
    if previous is not None and lifecycle_match is not None:
        for marker in ("- Source date:", "- Completed:", "- Carried forward:", "- Purge after:"):
            if marker not in previous:
                errors.append(f"short_previous_day_missing:{marker[2:-1]}")
        source_match = re.search(r"^- Source date:\s*`([^`]+)`", previous, re.MULTILINE)
        purge_match = re.search(r"^- Purge after:\s*`([^`]+)`", previous, re.MULTILINE)
        expected_source = date.fromisoformat(lifecycle_match.group("opened")) - timedelta(days=1)
        if not source_match or source_match.group(1) != expected_source.isoformat():
            errors.append("short_previous_day_source_mismatch")
        if not purge_match or purge_match.group(1) != lifecycle_match.group("expires"):
            errors.append("short_previous_day_purge_mismatch")
    return errors


def _resolve_active_short_memory_root(root: Path) -> tuple[Path | None, list[str]]:
    """Use the manager's strict resolver for runtime coordination state."""
    try:
        coordination_root = TASK_MANAGER.resolve_coordination_root(root)
    except TASK_MANAGER.LedgerError as exc:
        return None, [f"short_coordination_{exc.code}"]
    active_path = (coordination_root / SHORT_MEMORY_RELATIVE).resolve()
    try:
        active_path.relative_to(coordination_root)
    except ValueError:
        return None, ["short_coordination_path_escape"]
    if not active_path.is_file():
        return None, ["short_coordination_active_ledger_missing"]
    return coordination_root, []


def _check_tracked_short_memory_snapshot(
    worktree: Path,
    coordination_root: Path,
) -> list[str]:
    """Keep a noncanonical worktree snapshot governed as static Git content."""
    resolved_worktree = worktree.resolve()
    if resolved_worktree == coordination_root:
        return []
    relative_path = SHORT_MEMORY_RELATIVE.as_posix()
    expected_blob_oid = _git_blob_oid(resolved_worktree, "HEAD", relative_path)
    if expected_blob_oid is None:
        return ["short_snapshot_git_head_unresolved"]
    return _validate_git_tracked_file_identity(
        resolved_worktree,
        relative_path=relative_path,
        expected_blob_oid=expected_blob_oid,
        git_reference="HEAD",
        error_prefix="short_snapshot",
    )


def _check_active_short_memory_state(
    root: Path,
    as_of: datetime | date | str | None,
) -> dict[str, Any]:
    """Validate canonical runtime state and local static snapshot separately."""
    coordination_root, resolution_errors = _resolve_active_short_memory_root(root)
    if coordination_root is None:
        return {
            "coordination_root": None,
            "active_ledger_path": None,
            "short_ttl": {"status": "FAIL", "errors": resolution_errors},
            "errors": resolution_errors,
        }
    short_ttl = _check_short_ttl(coordination_root, as_of)
    errors = [*short_ttl["errors"], *_check_short_checklist(coordination_root)]
    errors.extend(_check_tracked_short_memory_snapshot(root, coordination_root))
    return {
        "coordination_root": str(coordination_root),
        "active_ledger_path": str(coordination_root / SHORT_MEMORY_RELATIVE),
        "short_ttl": short_ttl,
        "errors": errors,
    }


def _check_active_memory(root: Path) -> list[str]:
    errors: list[str] = []
    medium = _read(root / ".agents" / "memory" / "04_PROJECT_MEMORY_MEDIUM.md")
    if "## Active cross-day entries" in medium:
        for marker in (
            "Status:",
            "Opened:",
            "Next action:",
            "Evidence:",
            "Source:",
            "Valid from:",
            "Review after:",
            "Supersedes:",
            "Invalidation condition:",
            "Exit:",
        ):
            if marker not in medium:
                errors.append(f"medium_missing:{marker}")
    long_text = _read(root / ".agents" / "memory" / "05_PROJECT_MEMORY_LONG.md")
    if "## Active entries" in long_text:
        for marker in (
            "source:",
            "evidence:",
            "valid_from:",
            "review_after:",
            "supersedes:",
            "invalidation_condition:",
        ):
            if marker not in long_text:
                errors.append(f"long_missing:{marker}")
    return errors


def _check_memory_maturity(root: Path) -> list[str]:
    registry_path = root / ".agents" / "memory" / "21_MEMORY_MATURITY.json"
    guide_path = root / ".agents" / "memory" / "21_MEMORY_MATURITY.md"
    manager_path = (
        root
        / ".agents"
        / "skills"
        / "project-state-steward"
        / "scripts"
        / "manage_memory_maturity.py"
    )
    errors: list[str] = []
    for label, path in (
        ("registry", registry_path),
        ("guide", guide_path),
        ("manager", manager_path),
    ):
        if not path.is_file():
            errors.append(f"maturity_{label}_missing")
    if errors:
        return errors
    payload = _load_json(registry_path)
    policy = payload.get("policy", {})
    if policy.get("elapsed_inactivity_is_evidence") is not False:
        errors.append("maturity_inactivity_must_not_be_evidence")
    errors.extend(MATURITY_MANAGER.validate_registry(root, payload))
    return errors


def _check_authority_index(root: Path) -> list[str]:
    index = _load_json(root / ".agents" / "memory" / "18_AUTHORITY_INDEX.json")
    errors: list[str] = []
    entries = index.get("entries", [])
    scopes = [entry.get("scope") for entry in entries]
    for scope in scopes:
        if scopes.count(scope) > 1:
            errors.append(f"duplicate_current_scope:{scope}")
    for entry in entries:
        current = entry.get("current_authority", "")
        if "ARCHIVE" in current.upper():
            errors.append(f"archive_cannot_be_current:{current}")
        for field in (
            "scope",
            "workstream",
            "current_authority",
            "valid_from",
            "supersedes",
            "conflict_resolution",
        ):
            if not _nonempty(entry.get(field)):
                errors.append(f"authority_missing:{field}")
        refs = [current]
        refs.extend(entry.get("supporting_authorities", []))
        refs.extend(entry.get("historical_authorities", []))
        for ref in refs:
            if not (root / ref).is_file():
                errors.append(f"authority_missing_file:{ref}")
    for scope, expected in V2_AUTHORITY_SCOPES.items():
        matching = [entry for entry in entries if entry.get("scope") == scope]
        if not matching:
            errors.append(f"authority_scope_missing:{scope}")
            continue
        if any(entry.get("current_authority") != expected for entry in matching):
            errors.append(f"authority_current_not_canonical:{scope}:{expected}")
    return errors


def _check_charter(root: Path) -> list[str]:
    text = _read(root / ".agents" / "memory" / "12_PROJECT_CHARTER.md")
    required = (
        "Scientific Question",
        "Non-Goals",
        "Success Criteria",
        "Allowed And Forbidden Claims",
        "Stop Criteria",
        "Promotion And Reject Criteria",
    )
    errors = [f"charter_missing:{item}" for item in required if item not in text]
    sidecar = root / ".agents" / "memory" / "12_PROJECT_CHARTER.sha256"
    if not sidecar.is_file():
        errors.append("charter_hash_sidecar_missing")
    else:
        legacy = _read(sidecar).split()
        if not legacy or not re.fullmatch(r"[0-9a-f]{64}", legacy[0].lower()):
            errors.append("charter_legacy_raw_hash_invalid")
    identity_path = root / ".agents" / "memory" / "12_PROJECT_CHARTER.git_identity.json"
    if not identity_path.is_file():
        errors.append("charter_git_identity_missing")
        return errors
    identity = _load_json(identity_path)
    if identity.get("schema_version") != "pig.git-tracked-identity.v1":
        errors.append("charter_git_identity_schema_invalid")
        return errors
    relative_path = ".agents/memory/12_PROJECT_CHARTER.md"
    if identity.get("relative_path") != relative_path:
        errors.append("charter_git_identity_path_mismatch")
        return errors
    errors.extend(
        _validate_git_tracked_file_identity(
            root,
            relative_path=relative_path,
            expected_blob_oid=str(identity.get("blob_oid", "")),
            git_reference=str(identity.get("git_reference", "")),
            error_prefix="charter",
        )
    )
    return errors


def _valid_method_transition(
    transition: dict[str, Any],
    forward_order: list[str],
    terminals: set[str],
) -> list[str]:
    errors: list[str] = []
    required = (
        "from_state",
        "to_state",
        "timestamp",
        "git_sha",
        "dirty_worktree",
        "config_hash",
        "input_hashes",
        "evaluator",
        "evidence_class",
        "gate_results",
        "limitations",
        "authority",
    )
    for field in required:
        if not _nonempty(transition.get(field)):
            errors.append(f"transition_missing:{field}")
    source = transition.get("from_state")
    target = transition.get("to_state")
    if source in forward_order and target in forward_order:
        source_index = forward_order.index(source)
        target_index = forward_order.index(target)
        if target_index != source_index + 1:
            errors.append(f"transition_skips_gate:{source}->{target}")
    elif target not in terminals:
        errors.append(f"invalid_transition_target:{target}")
    return errors


def _check_method_state(root: Path) -> list[str]:
    registry = _load_json(root / ".agents" / "memory" / "13_METHOD_STATE.json")
    errors: list[str] = []
    order = registry.get("forward_order", [])
    terminals = set(registry.get("terminal_states", []))
    allowed = set(order) | terminals
    entries = registry.get("entries", [])
    ids = [entry.get("method_id") for entry in entries]
    for method_id in ids:
        if ids.count(method_id) > 1:
            errors.append(f"duplicate_method:{method_id}")
    for entry in entries:
        if entry.get("state") not in allowed:
            errors.append(f"invalid_method_state:{entry.get('state')}")
        if not _nonempty(entry.get("authority")):
            errors.append(f"method_missing_authority:{entry.get('method_id')}")
        for transition in entry.get("transitions", []):
            errors.extend(_valid_method_transition(transition, order, terminals))
    return errors


def evaluate_claim(claim: dict[str, Any]) -> tuple[str, list[str]]:
    errors: list[str] = []
    claim_id = claim.get("claim_id", "<missing>")
    status = claim.get("status")
    if status == "SUPPORTED":
        missing = [
            field
            for field in REQUIRED_CLAIM_FIELDS
            if not _nonempty(claim.get(field))
        ]
        if missing:
            errors.append(
                f"supported_claim_incomplete:{claim_id}:{','.join(missing)}"
            )
            return "HOLD_INCOMPLETE_LINEAGE", errors
    return str(status), errors


def _check_claims(root: Path) -> tuple[list[str], dict[str, str]]:
    registry = _load_json(root / ".agents" / "memory" / "14_CLAIM_REGISTRY.json")
    errors: list[str] = []
    derived: dict[str, str] = {}
    statuses = set(registry.get("allowed_statuses", []))
    for claim in registry.get("claims", []):
        claim_id = claim.get("claim_id", "<missing>")
        status = claim.get("status")
        if status not in statuses:
            errors.append(f"invalid_claim_status:{claim_id}:{status}")
        for field in ("claim_id", "claim_text", "scope", "evidence_class"):
            if not _nonempty(claim.get(field)):
                errors.append(f"claim_missing:{claim_id}:{field}")
        derived_status, admission_errors = evaluate_claim(claim)
        derived[claim_id] = derived_status
        errors.extend(admission_errors)
    return errors, derived


def _check_halt_contract(root: Path) -> list[str]:
    contract = _load_json(root / ".agents" / "memory" / "16_HALT_CONDITIONS.json")
    errors: list[str] = []
    trigger_ids = {item.get("id") for item in contract.get("halt_triggers", [])}
    for required in (
        "authority_conflict",
        "claim_incomplete",
        "transition_skips_gate",
    ):
        if required not in trigger_ids:
            errors.append(f"halt_missing:{required}")
    envelope = contract.get("observation_envelope", {})
    for field in ("status", "summary", "next_actions", "artifacts"):
        if field not in envelope.get("required_fields", []):
            errors.append(f"observation_missing:{field}")
    for field in ("root_cause_hint", "safe_retry", "stop_condition"):
        if field not in envelope.get("error_required_fields", []):
            errors.append(f"error_observation_missing:{field}")
    retry = contract.get("retry_contract", {})
    for field in (
        "requires_root_cause_hint",
        "requires_precondition_revalidation",
        "never_weaken_gate_on_retry",
        "stop_on_declared_condition",
    ):
        if retry.get(field) is not True:
            errors.append(f"retry_contract_missing:{field}")
    return errors


def _check_skill_inventory_views(root: Path) -> list[str]:
    """Ensure declared skill-inventory views are generated and byte-identical."""
    inventory_path = root / ".agents" / "skills" / "skill_inventory.json"
    if not inventory_path.is_file():
        return []
    try:
        inventory = _load_json(inventory_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        # The inventory validator reports the primary unreadable-JSON failure.
        return []
    if not isinstance(inventory, dict):
        return []
    skills = inventory.get("skills", [])
    modern_inventory = isinstance(inventory.get("view_contract"), dict) or (
        isinstance(skills, list)
        and any(
            isinstance(skill, dict)
            and ("registry" in skill or "portfolio" in skill)
            for skill in skills
        )
    )
    if not modern_inventory:
        # Preserve V1 fixture/legacy inventories until their view contract is adopted.
        return []
    declared = inventory.get("generated_views")
    if declared == []:
        return ["skill_inventory_views_declaration_empty"]
    if declared is None:
        return ["skill_inventory_views_declaration_missing"]
    renderer_path = Path(__file__).with_name("render_skill_inventory_views.py")
    if not renderer_path.is_file():
        return ["skill_inventory_view_renderer_missing"]
    try:
        spec = importlib.util.spec_from_file_location(
            "skill_inventory_view_renderer",
            renderer_path,
        )
        if spec is None or spec.loader is None:
            return ["skill_inventory_view_renderer_unavailable"]
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        mismatches = renderer.check_views(root)
    except Exception as exc:  # pragma: no cover - defensive validator boundary
        return [
            "skill_inventory_view_renderer_error:"
            f"{type(exc).__name__}:{exc}"
        ]
    return [f"skill_inventory_view_{error}" for error in mismatches]


def _check_skill_portfolio(root: Path) -> list[str]:
    portfolio = _load_json(root / ".agents" / "memory" / "11_SKILL_PORTFOLIO.json")
    errors: list[str] = []
    routes = portfolio.get("mandatory_reasoning_routes", {})
    for route, required in EXPECTED_ROUTES.items():
        actual = set(routes.get(route, []))
        if actual != required:
            errors.append(f"reasoning_route_mismatch:{route}")
    ids: list[str] = []
    for skill in portfolio.get("skills", []):
        skill_id = skill.get("skill_id", "<missing>")
        ids.append(skill_id)
        missing = REQUIRED_PORTFOLIO_FIELDS - set(skill)
        errors.extend(
            f"skill_missing:{skill_id}:{field}" for field in sorted(missing)
        )
        try:
            source = _resolve_source(
                root,
                skill.get("source_root", ""),
                skill.get("relative_path", ""),
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not source.is_file():
            errors.append(f"skill_missing_file:{skill_id}:{source}")
            continue
        if skill.get("source_root") == "project":
            identity = skill.get("git_identity")
            if not isinstance(identity, dict):
                errors.append(f"skill_git_identity_missing:{skill_id}")
                continue
            relative_path = str(skill.get("relative_path", ""))
            if identity.get("schema_version") != "pig.git-tracked-identity.v1":
                errors.append(f"skill_git_identity_schema_invalid:{skill_id}")
            elif identity.get("relative_path") != relative_path:
                errors.append(f"skill_git_identity_path_mismatch:{skill_id}")
            else:
                errors.extend(
                    _validate_git_tracked_file_identity(
                        root,
                        relative_path=relative_path,
                        expected_blob_oid=str(identity.get("blob_oid", "")),
                        git_reference=str(identity.get("git_reference", "")),
                        error_prefix=f"skill:{skill_id}",
                    )
                )
            bundle_paths = skill.get("bundle_paths")
            if not isinstance(bundle_paths, list) or not bundle_paths:
                errors.append(f"skill_bundle_paths_missing:{skill_id}")
            elif not skill.get("bundle_sha256"):
                errors.append(f"skill_bundle_hash_missing:{skill_id}")
            else:
                bundle_oids = (
                    identity.get("bundle_blob_oids")
                    if isinstance(identity, dict)
                    else None
                )
                if not isinstance(bundle_oids, dict):
                    errors.append(f"skill_bundle_git_identity_missing:{skill_id}")
                else:
                    for bundle_relative in bundle_paths:
                        normalized = Path(bundle_relative).as_posix()
                        bundle_path = (
                            Path(relative_path).parent / normalized
                        ).as_posix()
                        errors.extend(
                            _validate_git_tracked_file_identity(
                                root,
                                relative_path=bundle_path,
                                expected_blob_oid=str(bundle_oids.get(normalized, "")),
                                git_reference=str(identity.get("git_reference", "")),
                                error_prefix=(
                                    f"skill_bundle:{skill_id}:{normalized}"
                                ),
                            )
                        )
        else:
            raw_errors = _validate_raw_file_identity(
                source,
                expected_sha256=str(skill.get("file_sha256", "")),
                error_prefix=f"skill:{skill_id}",
            )
            if raw_errors:
                errors.append(f"skill_hash_mismatch:{skill_id}")
        if not _nonempty(skill.get("version_or_commit")):
            errors.append(f"skill_missing_version:{skill_id}")
        if not _nonempty(skill.get("proof_task")):
            errors.append(f"skill_missing_proof:{skill_id}")
        if not _nonempty(skill.get("stale_signal")):
            errors.append(f"skill_missing_stale_signal:{skill_id}")
        if not _nonempty(skill.get("next_maintenance_action")):
            errors.append(f"skill_missing_maintenance:{skill_id}")
    for skill_id in ids:
        if ids.count(skill_id) > 1:
            errors.append(f"duplicate_skill:{skill_id}")
    errors.extend(AGENT_GOVERNANCE.validate_skill_inventory(root))
    errors.extend(_check_skill_inventory_views(root))
    return errors


def _check_eval_harness(root: Path) -> list[str]:
    suite = root / ".agents" / "evals" / "agent_governance"
    errors: list[str] = []
    required = (
        "tasks.json",
        "judge.py",
        "run_regression.py",
        "manifest.json",
        "fixtures/pass_responses.json",
        "fixtures/fail_responses.json",
        "reports/fixture_self_test_20260731.json",
        "reports/fixture_fail_control_20260731.json",
    )
    for relative in required:
        if not (suite / relative).is_file():
            errors.append(f"eval_missing_file:{relative}")
    if errors:
        return errors
    manifest = _load_json(suite / "manifest.json")
    if manifest.get("minimum_runs", 0) < 3:
        errors.append("eval_minimum_runs_below_three")
    pinned_files = {
        "task_sha256": ".agents/evals/agent_governance/tasks.json",
        "judge_sha256": ".agents/evals/agent_governance/judge.py",
        "runner_sha256": ".agents/evals/agent_governance/run_regression.py",
        "validator_sha256": (
            ".agents/skills/project-state-steward/scripts/"
            "validate_governance_contracts.py"
        ),
    }
    for field in pinned_files:
        legacy = str(manifest.get(field, "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", legacy):
            errors.append(f"eval_manifest_legacy_raw_hash_invalid:{field}")

    identity = manifest.get("tracked_git_identity")
    if not isinstance(identity, dict):
        errors.append("eval_manifest_git_identity_missing")
    elif identity.get("schema_version") != "pig.git-tracked-identity-set.v1":
        errors.append("eval_manifest_git_identity_schema_invalid")
    else:
        blob_oids = identity.get("blob_oids")
        if not isinstance(blob_oids, dict):
            errors.append("eval_manifest_git_identity_blob_map_missing")
        else:
            expected_paths = set(pinned_files.values())
            if set(blob_oids) != expected_paths:
                errors.append("eval_manifest_git_identity_path_set_mismatch")
            for field, relative_path in pinned_files.items():
                errors.extend(
                    _validate_git_tracked_file_identity(
                        root,
                        relative_path=relative_path,
                        expected_blob_oid=str(blob_oids.get(relative_path, "")),
                        git_reference=str(identity.get("git_reference", "")),
                        error_prefix=f"eval:{field}",
                    )
                )
    tasks = _load_json(suite / "tasks.json").get("tasks", [])
    task_ids = {task.get("id") for task in tasks}
    expected_ids = {f"AR-{index:03d}" for index in range(1, 26)} | {"AR-036"}
    if task_ids != expected_ids:
        errors.append("eval_task_set_mismatch")
    pass_report = _load_json(suite / "reports" / "fixture_self_test_20260731.json")
    fail_report = _load_json(suite / "reports" / "fixture_fail_control_20260731.json")
    pass_metrics = pass_report.get("metrics", {})
    for metric in ("pass_rate", "pass@1", "pass@3", "pass^3"):
        if pass_metrics.get(metric) != 1.0:
            errors.append(f"eval_pass_report_metric_failed:{metric}")
    if pass_metrics.get("runs") != 3 or pass_metrics.get("task_count") != 26:
        errors.append("eval_pass_report_scope_mismatch")
    if not pass_report.get("dirty_worktree_fingerprint"):
        errors.append("eval_pass_report_missing_worktree_fingerprint")
    if fail_report.get("status") != "FAIL":
        errors.append("eval_negative_control_did_not_fail")
    if fail_report.get("metrics", {}).get("pass^3") != 0.0:
        errors.append("eval_negative_control_passed")
    return errors


def _check_governance_references(root: Path) -> list[str]:
    errors: list[str] = []
    agents = _read(root / "AGENTS.md")
    workflow = _read(root / ".agents" / "memory" / "08_WORKFLOW.md")
    readme = _read(root / ".agents" / "memory" / "00_README.md")
    activation_documents = {
        name: _read(root / relative)
        for name, relative in V2_ACTIVATION_DOCUMENTS.items()
    }
    for document, content in activation_documents.items():
        normalized = " ".join(content.replace("\\", "/").split()).casefold()
        for marker_name, marker in V2_ACTIVATION_MARKERS.items():
            if marker.casefold() not in normalized:
                errors.append(
                    f"governance_missing_v2_{marker_name}:{document}"
                )
    for name in (
        "18_AUTHORITY_INDEX",
        "19_REASONING_ROUTING",
        "21_MEMORY_MATURITY",
    ):
        if name not in agents:
            errors.append(f"agents_missing_governance_reference:{name}")
        if name not in workflow:
            errors.append(f"workflow_missing_governance_reference:{name}")
        if name not in readme:
            errors.append(f"readme_missing_governance_reference:{name}")

    crash_contracts = {
        "AGENTS.md": agents,
        "00_README.md": readme,
        "03_PROJECT_RULES.md": _read(
            root / ".agents" / "memory" / "03_PROJECT_RULES.md"
        ),
        "08_WORKFLOW.md": workflow,
        "project-state-steward/SKILL.md": _read(
            root / ".agents" / "skills" / "project-state-steward" / "SKILL.md"
        ),
    }
    required_markers = {
        "done_checkpoint": "checkpoint `DONE` before the next step's first effect",
        "interrupted_recovery": "interrupted `IN_PROGRESS`",
    }
    for document, content in crash_contracts.items():
        normalized = " ".join(content.split()).casefold()
        for contract, marker in required_markers.items():
            if marker.casefold() not in normalized:
                errors.append(
                    f"governance_missing_{contract}:{document}"
                )
    recovery_contracts = {
        "AGENTS.md": agents,
        "03_PROJECT_RULES.md": crash_contracts["03_PROJECT_RULES.md"],
        "08_WORKFLOW.md": workflow,
        "project-state-steward/SKILL.md": crash_contracts[
            "project-state-steward/SKILL.md"
        ],
    }
    recovery_markers = {
        "same_thread_recovery": "CODEX_THREAD_ID",
        "administrative_takeover": "admin-takeover",
        "ownership_audit": "hash-bound audit",
    }
    for document, content in recovery_contracts.items():
        normalized = " ".join(content.split()).casefold()
        for contract, marker in recovery_markers.items():
            if marker.casefold() not in normalized:
                errors.append(
                    f"governance_missing_{contract}:{document}"
                )
    return errors


def validate_observation(
    observation: dict[str, Any],
    is_error: bool | None = None,
) -> list[str]:
    required = ["status", "summary", "next_actions", "artifacts"]
    errors = [f"observation_missing:{field}" for field in required if field not in observation]
    status = observation.get("status")
    if status not in {"success", "warning", "error", "blocked"}:
        errors.append(f"observation_invalid_status:{status}")
    error_state = is_error if is_error is not None else status in {"error", "blocked"}
    if error_state:
        for field in ("root_cause_hint", "safe_retry", "stop_condition"):
            if field not in observation:
                errors.append(f"error_observation_missing:{field}")
    return errors


def audit(
    root: Path = ROOT,
    as_of: datetime | date | str | None = None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    errors: list[str] = []
    checks["active_short_memory"] = _check_active_short_memory_state(root, as_of)
    checks["short_ttl"] = checks["active_short_memory"]["short_ttl"]
    errors.extend(checks["active_short_memory"]["errors"])
    errors.extend(_check_active_memory(root))
    errors.extend(_check_memory_maturity(root))
    errors.extend(_check_authority_index(root))
    errors.extend(_check_charter(root))
    errors.extend(_check_method_state(root))
    claim_errors, derived_claims = _check_claims(root)
    errors.extend(claim_errors)
    errors.extend(_check_halt_contract(root))
    errors.extend(_check_skill_portfolio(root))
    errors.extend(_check_eval_harness(root))
    errors.extend(_check_governance_references(root))
    errors.extend(AGENT_GOVERNANCE.validate_runtime_records(root))
    errors.extend(AGENT_GOVERNANCE.validate_worktree_lifecycle_ledger(root))
    checks["derived_claim_statuses"] = derived_claims
    status = "RESET_REQUIRED" if checks["short_ttl"]["status"] == "RESET_REQUIRED" else (
        "FAIL" if errors else "PASS"
    )
    return {
        "status": status,
        "errors": errors,
        "checks": checks,
    }


def main() -> int:
    report = audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Machine-enforced agent governance protocol with compatibility for V1 tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import secrets
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
RUNTIME_RELATIVE = Path(".agents/runtime/agent_governance_v2")
WORKTREE_LEDGER_RELATIVE = Path(".agents/memory/22_WORKTREE_LIFECYCLE.json")
INVENTORY_RELATIVE = Path(".agents/skills/skill_inventory.json")
AUTHORITY_INDEX_RELATIVE = Path(".agents/memory/18_AUTHORITY_INDEX.json")
V1_MANAGER_PATH = Path(__file__).with_name("manage_short_memory.py")
SCHEMA_VERSION = "pig.agent-governance-task.v2"
EVENT_SCHEMA = "pig.agent-governance-event.v1"
RUNTIME_SESSION_ENV = "CODEX_THREAD_ID"
HIGH_RISK_EFFECTS = {
    "delete",
    "destructive",
    "paid_compute",
    "protected_authority",
    "publish",
    "remote_mutation",
}
STEP_COMPLETION_STATES = {"DONE", "BLOCKED", "CANCELLED"}
WORKTREE_MODES = {"shared_main", "exclusive"}
SKILL_IMPACT_DISPOSITIONS = {
    "MAINTENANCE_DUE",
    "NO_SKILL_IMPACT",
    "REVIEWED_NO_CHANGE",
    "UPDATED",
}
TASK_STATES = {"PLANNED", "CONFIRMED", "ACTIVE", "REVIEWED", "CLOSED"}
OUTCOMES = {"ACCEPTED", "PARTIAL", "REJECTED", "BLOCKED", "UNKNOWN"}
LEARNING_DISPOSITIONS = {
    "VALIDATED_CORRECTION",
    "UNVERIFIED_FAILURE",
    "NO_DURABLE_LESSON",
}
PATH_DISPOSITIONS = {
    "INTEGRATE",
    "EXTRACT_EVIDENCE",
    "PRESERVE_USER_OWNED",
    "DISCARD_VERIFIED_SCRATCH",
    "UNKNOWN_HALT",
}
RETIREMENT_STATES = {"PROTECTED", "RETIRE_ELIGIBLE", "RETIRED"}
INTEGRATION_PROOF_KINDS = {"ANCESTOR", "PATCH_EQUIVALENT"}
WORKTREE_LIFECYCLE_STATES = {
    "ADMITTED",
    "ACTIVE",
    "RESULT_CAPTURED",
    "OUTCOME_REVIEWED",
    "RETIRE_ELIGIBLE",
    "RETIRED",
    "PROTECTED",
}
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")


class GovernanceError(RuntimeError):
    """Fail-closed protocol error with a stable machine code."""

    def __init__(self, code: str, hint: str) -> None:
        super().__init__(code)
        self.code = code
        self.hint = hint


def _load_v1_manager() -> Any:
    spec = importlib.util.spec_from_file_location("governance_v1_manager", V1_MANAGER_PATH)
    if spec is None or spec.loader is None:
        raise GovernanceError("v1_manager_unavailable", str(V1_MANAGER_PATH))
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V1_MANAGER = _load_v1_manager()


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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _plan_digest(plan: dict[str, Any]) -> str:
    """Hash only immutable plan intent, excluding mutable step evidence/status."""
    steps = []
    for step in plan.get("steps", []):
        steps.append(
            {
                "step_id": step.get("step_id"),
                "summary": step.get("summary"),
                "acceptance_ids": step.get("acceptance_ids", []),
                "allowed_effects": step.get("allowed_effects", []),
            }
        )
    return _digest({"version": plan.get("version"), "steps": steps})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any, label: str, maximum: int = 1000) -> str:
    cleaned = " ".join(str(value or "").split())
    if not cleaned or len(cleaned) > maximum:
        raise GovernanceError(f"{label}_invalid", f"Provide a bounded non-empty {label}.")
    return cleaned


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError("json_unavailable", f"Cannot load {path}: {exc}") from exc


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git(worktree: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(worktree), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _worktree_identity(worktree: Path) -> dict[str, Any]:
    resolved = worktree.resolve()
    top = _git(resolved, "rev-parse", "--show-toplevel")
    common = _git(resolved, "rev-parse", "--git-common-dir")
    head = _git(resolved, "rev-parse", "HEAD")
    branch = _git(resolved, "symbolic-ref", "--quiet", "--short", "HEAD")
    status = _git(resolved, "status", "--porcelain=v1", "-z")
    if top.returncode or common.returncode or head.returncode or status.returncode:
        raise GovernanceError(
            "worktree_unregistered",
            "Use a registered Git worktree with readable HEAD and status.",
        )
    if Path(top.stdout.strip()).resolve() != resolved:
        raise GovernanceError("worktree_top_mismatch", "Bind the exact Git top-level path.")
    common_path = Path(common.stdout.strip())
    if not common_path.is_absolute():
        common_path = (resolved / common_path).resolve()
    dirty_paths = sorted(
        entry[3:] for entry in status.stdout.split("\0") if len(entry) > 3
    )
    fingerprint = _digest(
        {
            "head_sha": head.stdout.strip().lower(),
            "dirty_paths": dirty_paths,
            "status": status.stdout,
        }
    )
    return {
        "path": str(resolved),
        "common_dir": str(common_path),
        "head_sha": head.stdout.strip().lower(),
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "detached": branch.returncode != 0,
        "dirty_paths": dirty_paths,
        "fingerprint": fingerprint,
    }


def _path_within_scope(path: str, scopes: list[str]) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    return any(normalized == scope or normalized.startswith(f"{scope}/") for scope in scopes)


def _is_ancestor(worktree: Path, ancestor: str, descendant: str) -> bool:
    return _git(worktree, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def _changed_paths_since(worktree: Path, accepted_head: str, actual_head: str) -> list[str]:
    if accepted_head == actual_head:
        return []
    result = _git(worktree, "diff", "--name-only", accepted_head, actual_head)
    if result.returncode:
        raise GovernanceError("worktree_lineage_unreadable", actual_head)
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def _canonical_common_dir(root: Path) -> Path:
    resolved_root = root.resolve()
    top = _git(resolved_root, "rev-parse", "--show-toplevel")
    if top.returncode or Path(top.stdout.strip()).resolve() != resolved_root:
        raise GovernanceError(
            "coordination_root_mismatch",
            "Use the canonical main worktree as the coordination root.",
        )
    result = _git(resolved_root, "rev-parse", "--git-common-dir")
    if result.returncode or not result.stdout.strip():
        raise GovernanceError("coordination_git_missing", str(resolved_root))
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = (resolved_root / common).resolve()
    common = common.resolve()
    if common.name != ".git" or common.parent != resolved_root:
        raise GovernanceError(
            "coordination_root_not_canonical_main",
            "Use the primary worktree whose .git directory is the common root.",
        )
    return common


def _normalized_path_scope(values: Any) -> list[str]:
    if not isinstance(values, list) or not values:
        raise GovernanceError(
            "shared_main_scope_missing",
            "Declare bounded repository-relative path prefixes.",
        )
    normalized: list[str] = []
    for value in values:
        raw = _clean(value, "worktree_scope", 300).replace("\\", "/")
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts:
            raise GovernanceError("worktree_scope_invalid", raw)
        normalized.append(path.as_posix().rstrip("/"))
    result = sorted(set(normalized))
    if len(result) != len(normalized):
        raise GovernanceError("worktree_scope_duplicate", ",".join(normalized))
    return result


def _scopes_overlap(left: list[str], right: list[str]) -> bool:
    for first in left:
        for second in right:
            if first == second or first.startswith(f"{second}/") or second.startswith(
                f"{first}/"
            ):
                return True
    return False


def _validate_worktree_binding(root: Path, identity: dict[str, Any]) -> None:
    if Path(identity["common_dir"]).resolve() != _canonical_common_dir(root):
        raise GovernanceError(
            "worktree_common_root_mismatch",
            "Bind only a registered worktree from the coordination repository.",
        )


def _canonical_coordination_root(
    worktree: Path,
    requested_root: Path | None = None,
) -> Path:
    """Resolve one root and reject a caller-supplied shadow coordination root."""
    canonical = V1_MANAGER.resolve_coordination_root(worktree).resolve()
    if requested_root is not None and requested_root.resolve() != canonical:
        raise GovernanceError(
            "coordination_root_override_mismatch",
            "Use the canonical root resolved from the registered worktree.",
        )
    _canonical_common_dir(canonical)
    return canonical


def _v1_task_ids(root: Path) -> set[str]:
    """Read legacy IDs without mutating or taking ownership of V1 capsules."""
    memory_path = root / V1_MANAGER.MEMORY_RELATIVE
    if not memory_path.is_file():
        return set()
    try:
        return {span["task_id"] for span in V1_MANAGER.task_spans(memory_path.read_text("utf-8"))}
    except (OSError, UnicodeError, V1_MANAGER.LedgerError) as exc:
        raise GovernanceError("v1_task_index_unavailable", str(memory_path)) from exc


def _resolve_source(root: Path, record: dict[str, Any]) -> Path:
    if record["source_root"] == "project":
        return root / record["relative_path"]
    if record["source_root"] == "codex_home":
        return Path.home() / ".codex" / record["relative_path"]
    raise GovernanceError("skill_source_unknown", str(record.get("source_root")))


def _inventory(root: Path) -> dict[str, Any]:
    payload = _load_json(root / INVENTORY_RELATIVE)
    if payload.get("schema_version") != "pig.skill-inventory.v1":
        raise GovernanceError("skill_inventory_schema_invalid", str(INVENTORY_RELATIVE))
    return payload


def validate_skill_inventory(root: Path) -> list[str]:
    """Return deterministic inventory errors for CI and governance validation."""
    errors: list[str] = []
    try:
        payload = _inventory(root)
    except GovernanceError as exc:
        return [exc.code]
    records = payload.get("skills")
    if not isinstance(records, list):
        return ["skill_inventory_records_invalid"]
    ids = [str(record.get("skill_id", "")) for record in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        errors.append("skill_inventory_ids_invalid")
    project_disk = {
        path.parent.name
        for path in (root / ".agents" / "skills").glob("*/SKILL.md")
    }
    project_inventory = {
        record["skill_id"]
        for record in records
        if record.get("source_root") == "project"
    }
    for missing in sorted(project_disk - project_inventory):
        errors.append(f"skill_inventory_missing_disk:{missing}")
    for missing in sorted(project_inventory - project_disk):
        errors.append(f"skill_inventory_missing_file:{missing}")
    mapped = {record.get("skill_id"): record for record in records}
    for record in records:
        skill_id = str(record.get("skill_id", ""))
        if record.get("status") not in {"active", "future", "disabled", "retired"}:
            errors.append(f"skill_inventory_status_invalid:{skill_id}")
        try:
            source = _resolve_source(root, record)
        except (KeyError, GovernanceError):
            errors.append(f"skill_inventory_source_invalid:{skill_id}")
            continue
        if not source.is_file():
            errors.append(f"skill_inventory_source_missing:{skill_id}")
        for dependency in record.get("depends_on", []):
            if dependency not in mapped:
                errors.append(f"skill_inventory_dependency_unknown:{skill_id}:{dependency}")
    for task_class, route in payload.get("task_routes", {}).items():
        if isinstance(route, list):
            route = {"required_all": route, "reasoning_required": True}
        required = list(route.get("required_all", [])) + list(
            route.get("required_any", [])
        )
        if not required:
            errors.append(f"skill_inventory_route_empty:{task_class}")
        for skill_id in required:
            record = mapped.get(skill_id)
            if record is None:
                errors.append(f"skill_inventory_route_unknown:{task_class}:{skill_id}")
        if route.get("reasoning_required") and not any(
            mapped.get(skill_id, {}).get("category") == "reasoning"
            for skill_id in required
        ):
            errors.append(f"skill_inventory_route_without_reasoning:{task_class}")
    modern_inventory = isinstance(payload.get("view_contract"), dict) or any(
        isinstance(record, dict)
        and ("registry" in record or "portfolio" in record)
        for record in records
    )
    if modern_inventory:
        declared = payload.get("generated_views")
        if not isinstance(declared, list) or not declared:
            errors.append("skill_inventory_views_declaration_missing")
        else:
            renderer_path = Path(__file__).with_name("render_skill_inventory_views.py")
            if not renderer_path.is_file():
                errors.append("skill_inventory_view_renderer_missing")
            else:
                try:
                    spec = importlib.util.spec_from_file_location(
                        "agent_governance_inventory_renderer",
                        renderer_path,
                    )
                    if spec is None or spec.loader is None:
                        raise ImportError("renderer loader unavailable")
                    renderer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(renderer)
                    errors.extend(
                        f"skill_inventory_view_{error}"
                        for error in renderer.check_views(root)
                    )
                except Exception as exc:  # pragma: no cover - defensive gate
                    errors.append(
                        "skill_inventory_view_renderer_error:"
                        f"{type(exc).__name__}"
                    )
    return errors


def _require_valid_skill_inventory(root: Path) -> None:
    errors = validate_skill_inventory(root)
    if errors:
        raise GovernanceError("skill_inventory_invalid", ",".join(sorted(errors)))


def validate_runtime_records(root: Path) -> list[str]:
    """Validate all V2 records without creating runtime state."""
    errors: list[str] = []
    task_root = root / RUNTIME_RELATIVE / "tasks"
    if not task_root.is_dir():
        return errors
    for path in sorted(task_root.glob("*.json")):
        try:
            record = _load_json(path)
            _validate_record(record)
        except GovernanceError as exc:
            errors.append(f"governance_v2_record:{path.stem}:{exc.code}")
            continue
        if record.get("state") == "CLOSED":
            if not record.get("learning"):
                errors.append(f"governance_v2_closed_without_learning:{path.stem}")
            if record.get("worktree", {}).get("retirement") not in RETIREMENT_STATES:
                errors.append(f"governance_v2_retirement_invalid:{path.stem}")
            if record.get("worktree", {}).get("retirement") == "RETIRED":
                proof = record.get("worktree", {}).get("retirement_proof")
                if not isinstance(proof, dict) or not proof.get("evidence_locator"):
                    errors.append(f"governance_v2_retirement_proof_missing:{path.stem}")
        if record.get("active_permit") and record.get("state") != "ACTIVE":
            errors.append(f"governance_v2_permit_state_mismatch:{path.stem}")
    return errors


def validate_worktree_lifecycle_ledger(root: Path) -> list[str]:
    """Validate the tracked, deferred lifecycle inventory without mutating it."""
    path = root / WORKTREE_LEDGER_RELATIVE
    if not path.is_file():
        return ["worktree_lifecycle_ledger_missing"]
    payload = _load_json(path)
    errors: list[str] = []
    if payload.get("schema_version") != "pig.worktree-lifecycle-ledger.v1":
        errors.append("worktree_lifecycle_schema_invalid")
    seen: set[str] = set()
    for item in payload.get("worktrees", []):
        worktree_id = str(item.get("worktree_id", ""))
        if not worktree_id or worktree_id in seen:
            errors.append(f"worktree_lifecycle_id_invalid:{worktree_id}")
        seen.add(worktree_id)
        state = item.get("state")
        if state not in WORKTREE_LIFECYCLE_STATES:
            errors.append(f"worktree_lifecycle_state_invalid:{worktree_id}")
        if item.get("retirement_authorized") and state not in {
            "RETIRE_ELIGIBLE",
            "RETIRED",
        }:
            errors.append(f"worktree_retirement_authority_without_gate:{worktree_id}")
        if state in {"OUTCOME_REVIEWED", "RETIRE_ELIGIBLE", "RETIRED"}:
            if item.get("outcome") not in OUTCOMES:
                errors.append(f"worktree_outcome_missing:{worktree_id}")
        if state == "RETIRED":
            proof = item.get("retirement_proof")
            required_proof = (
                "evidence_locator",
                "evidence_sha256",
                "reference_audit",
                "process_audit",
            )
            if not isinstance(proof, dict) or not all(
                proof.get(field) for field in required_proof
            ):
                errors.append(f"worktree_retirement_proof_missing:{worktree_id}")
        for dirty in item.get("dirty_paths", []):
            if dirty.get("disposition") not in PATH_DISPOSITIONS:
                errors.append(f"worktree_dirty_path_undispositioned:{worktree_id}")
        if item.get("outcome") in {"BLOCKED", "UNKNOWN"} and state not in {
            "PROTECTED",
            "OUTCOME_REVIEWED",
        }:
            errors.append(f"worktree_unknown_not_protected:{worktree_id}")
    return errors


def _skill_map(root: Path) -> dict[str, dict[str, Any]]:
    records = _inventory(root).get("skills", [])
    mapped = {str(record.get("skill_id")): record for record in records}
    if len(mapped) != len(records):
        raise GovernanceError("skill_inventory_duplicate", "Skill IDs must be unique.")
    return mapped


def _validate_skill_selections(
    root: Path,
    task_class: str,
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inventory = _inventory(root)
    skills = _skill_map(root)
    selected_ids = [str(item.get("skill_id", "")) for item in selections]
    if not selections or len(selected_ids) != len(set(selected_ids)):
        raise GovernanceError("skill_selection_invalid", "Select unique skills.")
    normalized: list[dict[str, Any]] = []
    for selection in selections:
        skill_id = _clean(selection.get("skill_id"), "skill_id", 128)
        record = skills.get(skill_id)
        if record is None:
            raise GovernanceError("skill_unknown", skill_id)
        status = record.get("status")
        if status != "active":
            raise GovernanceError("skill_not_active", f"{skill_id}:{status}")
        if selection.get("selection_mode") == "implicit" and not record.get("implicit"):
            raise GovernanceError("skill_implicit_forbidden", skill_id)
        purpose = _clean(selection.get("purpose"), "skill_purpose", 300)
        role = _clean(selection.get("role"), "skill_role", 48)
        source = _resolve_source(root, record)
        if not source.is_file():
            raise GovernanceError("skill_file_missing", str(source))
        actual_hash = _file_sha256(source)
        pinned_hash = str(selection.get("skill_sha256", "")).lower()
        if not HEX_SHA256_RE.fullmatch(pinned_hash) or pinned_hash != actual_hash:
            raise GovernanceError("skill_hash_mismatch", skill_id)
        normalized.append(
            {
                "skill_id": skill_id,
                "role": role,
                "purpose": purpose,
                "selection_mode": selection.get("selection_mode", "explicit"),
                "source_root": record["source_root"],
                "relative_path": record["relative_path"],
                "skill_sha256": actual_hash,
            }
        )
    route = inventory.get("task_routes", {}).get(task_class, {})
    if not route:
        for skill_id in selected_ids:
            missing = set(skills[skill_id].get("depends_on", [])) - set(
                selected_ids
            )
            if missing:
                raise GovernanceError(
                    "skill_dependency_missing",
                    f"{skill_id}:{','.join(sorted(missing))}",
                )
        raise GovernanceError("task_class_unrouted", task_class)
    if isinstance(route, list):
        route = {"required_all": route, "reasoning_required": True}
    required_all = set(route.get("required_all", []))
    required_any = set(route.get("required_any", []))
    if required_all - set(selected_ids):
        missing_route = ",".join(sorted(required_all - set(selected_ids)))
        raise GovernanceError(
            "reasoning_route_missing",
            f"{task_class}:{missing_route}",
        )
    if required_any and not required_any.intersection(selected_ids):
        raise GovernanceError("reasoning_route_missing", task_class)
    if route.get("reasoning_required"):
        categories = {skills[skill_id].get("category") for skill_id in selected_ids}
        if "reasoning" not in categories:
            raise GovernanceError("reasoning_skill_missing", task_class)
    for skill_id in selected_ids:
        missing = set(skills[skill_id].get("depends_on", [])) - set(selected_ids)
        if missing:
            raise GovernanceError(
                "skill_dependency_missing",
                f"{skill_id}:{','.join(sorted(missing))}",
            )
    return normalized


def _revalidate_authorities(
    root: Path,
    receipts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current = _authority_receipts(root, receipts)
    if _digest(current) != _digest(receipts):
        raise GovernanceError(
            "authority_receipt_drift",
            "Retrieve and confirm the current authority before another effect.",
        )
    return current


def _revalidate_skills(
    root: Path,
    task_class: str,
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current = _validate_skill_selections(root, task_class, selections)
    if _digest(current) != _digest(selections):
        raise GovernanceError(
            "skill_receipt_drift",
            "Re-read and reselect skills before another effect.",
        )
    return current


def _authority_receipts(root: Path, receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = _load_json(root / AUTHORITY_INDEX_RELATIVE)
    scopes = {entry["scope"]: entry for entry in index.get("entries", [])}
    if not receipts:
        raise GovernanceError("authority_receipts_missing", "Retrieve authority first.")
    normalized: list[dict[str, Any]] = []
    for receipt in receipts:
        scope = _clean(receipt.get("scope"), "authority_scope", 160)
        if scope not in scopes:
            raise GovernanceError("authority_scope_unknown", scope)
        locator = Path(_clean(receipt.get("locator"), "authority_locator", 300))
        expected = Path(scopes[scope]["current_authority"]).as_posix()
        if locator.as_posix() != expected:
            raise GovernanceError("authority_locator_not_current", f"{scope}:{locator}")
        path = (root / locator).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise GovernanceError("authority_path_escape", str(locator)) from exc
        if not path.is_file():
            raise GovernanceError("authority_missing", str(locator))
        actual_hash = _file_sha256(path)
        supplied_hash = str(receipt.get("sha256", "")).lower()
        if supplied_hash != actual_hash:
            raise GovernanceError("authority_hash_mismatch", f"{scope}:{locator}")
        selector = _clean(receipt.get("selector"), "authority_selector", 300)
        section_hash = str(receipt.get("section_sha256", "")).lower()
        section_text = _authority_section(path, selector)
        actual_section_hash = hashlib.sha256(section_text).hexdigest()
        if section_hash != actual_section_hash:
            raise GovernanceError("authority_section_hash_mismatch", f"{scope}:{selector}")
        normalized.append(
            {
                "scope": scope,
                "locator": locator.as_posix(),
                "selector": selector,
                "status": _clean(receipt.get("status"), "authority_status", 48),
                "read_at": _clean(receipt.get("read_at"), "authority_read_at", 64),
                "sha256": actual_hash,
                "section_sha256": actual_section_hash,
            }
        )
    if len({item["scope"] for item in normalized}) != len(normalized):
        raise GovernanceError("authority_scope_duplicate", "One current receipt per scope.")
    return normalized


def _authority_section(path: Path, selector: str) -> bytes:
    if selector == "FULL_FILE":
        return path.read_bytes()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if line.rstrip("\r\n") == selector
    ]
    if len(starts) != 1 or not selector.startswith("#"):
        raise GovernanceError("authority_selector_invalid", f"{path}:{selector}")
    start = starts[0]
    level = len(selector) - len(selector.lstrip("#"))
    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = re.match(r"^(#+)\s", lines[index])
        if match and len(match.group(1)) <= level:
            end = index
            break
    return "".join(lines[start:end]).encode("utf-8")


def _steps(packet: dict[str, Any]) -> list[dict[str, Any]]:
    acceptance_ids = {
        _clean(item.get("acceptance_id"), "acceptance_id", 80)
        for item in packet.get("acceptance", [])
    }
    if not acceptance_ids:
        raise GovernanceError("acceptance_missing", "Declare stable acceptance IDs.")
    records = packet.get("plan", {}).get("steps", [])
    step_ids = [str(item.get("step_id", "")) for item in records]
    if not records or len(step_ids) != len(set(step_ids)):
        raise GovernanceError("plan_steps_invalid", "Declare unique plan steps.")
    normalized: list[dict[str, Any]] = []
    for item in records:
        covers = sorted(set(item.get("acceptance_ids", [])))
        if not covers or not set(covers).issubset(acceptance_ids):
            raise GovernanceError("step_acceptance_invalid", str(item.get("step_id")))
        effects = sorted(
            {
                _clean(value, "allowed_effect", 80)
                for value in item.get("allowed_effects", [])
            }
        )
        normalized.append(
            {
                "step_id": _clean(item.get("step_id"), "step_id", 80),
                "summary": _clean(item.get("summary"), "step_summary", 240),
                "acceptance_ids": covers,
                "allowed_effects": effects,
                "status": "TODO",
                "evidence": [],
            }
        )
    normalized[0]["status"] = "IN_PROGRESS"
    return normalized


def _event(previous: str | None, event_type: str, payload: Any, now: datetime) -> dict[str, Any]:
    body = {
        "schema_version": EVENT_SCHEMA,
        "event_type": event_type,
        "timestamp": _iso(now),
        "previous_event_sha256": previous,
        "payload": payload,
    }
    body["event_sha256"] = _digest(body)
    return body


def _record_hash(record: dict[str, Any]) -> str:
    candidate = dict(record)
    candidate["record_sha256"] = "0" * 64
    return _digest(candidate)


def _seal(record: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(record))
    result["record_sha256"] = "0" * 64
    result["record_sha256"] = _record_hash(result)
    return result


def _validate_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise GovernanceError("record_type_invalid", "Expected an object record.")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise GovernanceError("record_schema_invalid", str(record.get("schema_version")))
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not ID_RE.fullmatch(task_id):
        raise GovernanceError("record_task_id_invalid", str(task_id))
    if record.get("state") not in TASK_STATES:
        raise GovernanceError("record_state_invalid", str(record.get("state")))
    if not isinstance(record.get("revision"), int) or record["revision"] < 1:
        raise GovernanceError("record_revision_invalid", str(record.get("revision")))
    owner = record.get("owner")
    if not isinstance(owner, dict):
        raise GovernanceError("record_owner_invalid", "Owner object is required.")
    for key in ("session", "runtime_session", "lease_expires"):
        _clean(owner.get(key), f"record_owner_{key}", 300)
    if not HEX_SHA256_RE.fullmatch(str(owner.get("token_sha256", "")).lower()):
        raise GovernanceError("record_owner_token_invalid", "Expected a SHA-256 token hash.")
    try:
        datetime.fromisoformat(str(owner["lease_expires"]))
    except ValueError as exc:
        raise GovernanceError("record_owner_lease_invalid", "Use an ISO timestamp.") from exc
    plan = record.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise GovernanceError("record_plan_invalid", "Plan steps are required.")
    if not isinstance(plan.get("version"), int) or plan["version"] < 1:
        raise GovernanceError("record_plan_version_invalid", str(plan.get("version")))
    if plan.get("digest") != _plan_digest(plan):
        raise GovernanceError("record_plan_digest_mismatch", "Inspect the current plan.")
    worktree = record.get("worktree")
    if not isinstance(worktree, dict) or not worktree.get("path"):
        raise GovernanceError("record_worktree_invalid", "A bound worktree is required.")
    if worktree.get("retirement") not in RETIREMENT_STATES:
        raise GovernanceError("record_retirement_invalid", str(worktree.get("retirement")))
    expected = _record_hash(record)
    if record.get("record_sha256") != expected:
        raise GovernanceError("record_hash_mismatch", "Inspect a fresh record.")
    events = record.get("events", [])
    if not isinstance(events, list):
        raise GovernanceError("record_events_invalid", "Events must be a list.")
    previous = None
    for event in events:
        if not isinstance(event, dict) or event.get("schema_version") != EVENT_SCHEMA:
            raise GovernanceError("event_schema_invalid", str(event))
        supplied = event.get("event_sha256")
        candidate = dict(event)
        candidate.pop("event_sha256", None)
        if event.get("previous_event_sha256") != previous or supplied != _digest(candidate):
            raise GovernanceError("event_chain_invalid", str(event.get("event_type")))
        previous = supplied
    permit = record.get("active_permit")
    if permit and record["state"] != "ACTIVE":
        raise GovernanceError("record_permit_state_invalid", record["state"])
    if record["state"] == "CLOSED" and (permit or not record.get("learning")):
        raise GovernanceError(
            "record_closed_incomplete",
            "Closed records need learning and no permit.",
        )


class AgentGovernanceLedger:
    """Canonical, locked, hash-chained V2 task and lifecycle ledger."""

    def __init__(self, coordination_root: Path, lock_timeout_seconds: float = 10.0) -> None:
        self.root = coordination_root.resolve()
        self.common_dir = _canonical_common_dir(self.root)
        self.runtime = self.root / RUNTIME_RELATIVE
        self.tasks = self.runtime / "tasks"
        self.lock = self.runtime / "ledger.lock"
        self.lock_timeout_seconds = lock_timeout_seconds

    def _path(self, task_id: str) -> Path:
        if not ID_RE.fullmatch(task_id):
            raise GovernanceError("task_id_invalid", task_id)
        return self.tasks / f"{task_id}.json"

    def _read(self, task_id: str) -> dict[str, Any]:
        path = self._path(task_id)
        if not path.is_file():
            raise GovernanceError("task_missing", task_id)
        record = _load_json(path)
        _validate_record(record)
        return record

    def _check_owner(self, record: dict[str, Any], token: str, worktree: Path) -> None:
        runtime = os.getenv(RUNTIME_SESSION_ENV)
        if not runtime or runtime != record["owner"]["runtime_session"]:
            raise GovernanceError("runtime_owner_mismatch", "Use the bound runtime.")
        if not secrets.compare_digest(
            record["owner"]["token_sha256"],
            hashlib.sha256(token.encode("utf-8")).hexdigest(),
        ):
            raise GovernanceError("owner_token_mismatch", "Use the private task token.")
        if Path(record["worktree"]["path"]).resolve() != worktree.resolve():
            raise GovernanceError("worktree_binding_mismatch", "Use the admitted worktree.")

    @staticmethod
    def _check_cas(
        record: dict[str, Any],
        expected_revision: int,
        expected_record_sha256: str,
    ) -> None:
        if record["revision"] != expected_revision:
            raise GovernanceError("revision_conflict", "Inspect the current revision.")
        if not secrets.compare_digest(
            record["record_sha256"],
            expected_record_sha256,
        ):
            raise GovernanceError("record_cas_conflict", "Inspect the current record hash.")

    @staticmethod
    def _finish_mutation(
        record: dict[str, Any],
        event_type: str,
        event_payload: Any,
        current: datetime,
    ) -> dict[str, Any]:
        previous = record["events"][-1]["event_sha256"] if record["events"] else None
        record["events"].append(_event(previous, event_type, event_payload, current))
        record["revision"] += 1
        record["updated_at"] = _iso(current)
        return _seal(record)

    def _mutate(
        self,
        task_id: str,
        expected_revision: int,
        expected_record_sha256: str,
        token: str,
        worktree: Path,
        event_type: str,
        mutation: Any,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _now(now)
        with V1_MANAGER.exclusive_file_lock(self.lock, self.lock_timeout_seconds):
            record = self._read(task_id)
            self._check_cas(record, expected_revision, expected_record_sha256)
            self._check_owner(record, token, worktree)
            if current >= datetime.fromisoformat(record["owner"]["lease_expires"]):
                raise GovernanceError("lease_expired", "Recover or take over before mutation.")
            result, event_payload = mutation(_json_copy(record))
            result = self._finish_mutation(
                result,
                event_type,
                event_payload,
                current,
            )
            _atomic_json(self._path(task_id), result)
            return result

    def create(
        self,
        packet: dict[str, Any],
        owner_session: str,
        worktree: Path,
        owner_token: str | None = None,
        lease_seconds: int = 1800,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _now(now)
        runtime = os.getenv(RUNTIME_SESSION_ENV)
        if not runtime or runtime != owner_session:
            raise GovernanceError("runtime_owner_missing", "Bind the current CODEX_THREAD_ID.")
        if lease_seconds < 1 or lease_seconds > 86400:
            raise GovernanceError("lease_invalid", "Use a one-second to one-day lease.")
        task_id = _clean(packet.get("task_id"), "task_id", 128)
        path = self._path(task_id)
        if task_id in _v1_task_ids(self.root):
            raise GovernanceError(
                "v1_v2_task_id_collision",
                "Keep the existing V1 capsule authoritative until migration closes.",
            )
        _require_valid_skill_inventory(self.root)
        skills = _validate_skill_selections(
            self.root,
            _clean(packet.get("task_class"), "task_class", 120),
            packet.get("skills", []),
        )
        authorities = _authority_receipts(self.root, packet.get("authorities", []))
        steps = _steps(packet)
        identity = _worktree_identity(worktree)
        _validate_worktree_binding(self.root, identity)
        worktree_mode = _clean(
            packet.get("worktree_mode", "exclusive"),
            "worktree_mode",
            32,
        )
        if worktree_mode not in WORKTREE_MODES:
            raise GovernanceError("worktree_mode_invalid", worktree_mode)
        if worktree_mode == "shared_main" and Path(identity["path"]) != self.root:
            raise GovernanceError(
                "shared_main_path_invalid",
                "shared_main binds only the canonical main worktree.",
            )
        path_scope = (
            _normalized_path_scope(packet["path_scope"])
            if packet.get("path_scope") is not None
            else []
        )
        if worktree_mode == "shared_main" and not path_scope:
            raise GovernanceError(
                "shared_main_scope_missing",
                "Declare bounded repository-relative path prefixes.",
            )
        token = owner_token or secrets.token_urlsafe(32)
        if len(token) < 16:
            raise GovernanceError("owner_token_weak", "Use a generated private token.")
        acceptance = packet.get("acceptance", [])
        plan = {"version": 1, "steps": steps}
        plan["digest"] = _plan_digest(plan)
        record = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "title": _clean(packet.get("title"), "title", 160),
            "task_class": _clean(packet.get("task_class"), "task_class", 120),
            "risk_class": _clean(packet.get("risk_class"), "risk_class", 48),
            "state": "PLANNED",
            "revision": 1,
            "created_at": _iso(current),
            "updated_at": _iso(current),
            "owner": {
                "session": owner_session,
                "runtime_session": runtime,
                "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                "lease_expires": _iso(current + timedelta(seconds=lease_seconds)),
            },
            "authorities": authorities,
            "authority_digest": _digest(authorities),
            "acceptance": acceptance,
            "risks": packet.get("risks", []),
            "non_actions": packet.get("non_actions", []),
            "skills": skills,
            "skill_digest": _digest(skills),
            "plan": plan,
            "plan_confirmation": None,
            "worktree": {
                **identity,
                "mode": worktree_mode,
                "path_scope": path_scope,
                "base_sha": identity["head_sha"],
                "base_head": identity["head_sha"],
                "accepted_task_head": identity["head_sha"],
                "actual_worktree_head": identity["head_sha"],
                "base_fingerprint": identity["fingerprint"],
                "accepted_task_fingerprint": identity["fingerprint"],
                "actual_worktree_fingerprint": identity["fingerprint"],
                "state": "ADMITTED",
                "outcome": None,
                "path_dispositions": [],
                "integration": None,
                "retirement": "PROTECTED",
            },
            "active_permit": None,
            "skill_reads": [],
            "learning": None,
            "skill_maintenance": [],
            "events": [],
            "record_sha256": "0" * 64,
        }
        record["events"].append(
            _event(
                None,
                "TASK_CREATED",
                {
                    "plan_digest": plan["digest"],
                    "authority_digest": record["authority_digest"],
                    "skill_digest": record["skill_digest"],
                    "worktree_fingerprint": identity["fingerprint"],
                },
                current,
            )
        )
        record = _seal(record)
        with V1_MANAGER.exclusive_file_lock(self.lock, self.lock_timeout_seconds):
            if path.exists():
                raise GovernanceError("task_exists", task_id)
            for existing in self.tasks.glob("*.json") if self.tasks.exists() else []:
                active = _load_json(existing)
                if active.get("state") != "CLOSED":
                    active_path = Path(active.get("worktree", {}).get("path", ""))
                    if active_path.resolve() == worktree.resolve():
                        active_mode = active.get("worktree", {}).get(
                            "mode",
                            "exclusive",
                        )
                        if "exclusive" in {active_mode, worktree_mode}:
                            raise GovernanceError(
                                "worktree_already_admitted",
                                str(active.get("task_id")),
                            )
                        active_scope = active.get("worktree", {}).get(
                            "path_scope",
                            [],
                        )
                        if _scopes_overlap(active_scope, path_scope):
                            raise GovernanceError(
                                "shared_main_scope_overlap",
                                str(active.get("task_id")),
                            )
            _atomic_json(path, record)
        result = _json_copy(record)
        if owner_token is None:
            result["owner_token"] = token
        return result

    def record_skill_read(
        self,
        task_id: str,
        skill_id: str,
        skill_sha256: str,
        applies_to_step_ids: list[str],
        now: datetime | None = None,
        **owner: Any,
    ) -> dict[str, Any]:
        """Bind a hash-checked skill read to the current plan before effects."""
        current = _now(now)

        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            if record["state"] == "CLOSED":
                raise GovernanceError("closed_task_immutable", task_id)
            selections = _revalidate_skills(
                self.root,
                record["task_class"],
                record["skills"],
            )
            selected = next(
                (item for item in selections if item["skill_id"] == skill_id),
                None,
            )
            if selected is None:
                raise GovernanceError("skill_read_not_selected", skill_id)
            supplied = str(skill_sha256 or "").lower()
            if supplied != selected["skill_sha256"]:
                raise GovernanceError("skill_read_hash_mismatch", skill_id)
            if not isinstance(applies_to_step_ids, list) or not applies_to_step_ids:
                raise GovernanceError(
                    "skill_read_steps_missing",
                    "Bind at least one plan step to the skill receipt.",
                )
            step_ids = {step["step_id"] for step in record["plan"]["steps"]}
            applies = sorted({str(step_id) for step_id in applies_to_step_ids})
            if not set(applies).issubset(step_ids):
                raise GovernanceError(
                    "skill_read_step_invalid",
                    ",".join(sorted(set(applies) - step_ids)),
                )
            receipt = {
                "receipt_id": secrets.token_hex(16),
                "skill_id": skill_id,
                "skill_sha256": selected["skill_sha256"],
                "plan_version": record["plan"]["version"],
                "applies_to_step_ids": applies,
                "read_at": _iso(current),
            }
            reads = [
                item
                for item in record.get("skill_reads", [])
                if not (
                    item.get("skill_id") == skill_id
                    and item.get("plan_version") == record["plan"]["version"]
                )
            ]
            reads.append(receipt)
            record["skill_reads"] = reads
            return record, receipt

        return self._mutate(
            task_id,
            event_type="SKILL_READ_RECORDED",
            mutation=mutate,
            now=current,
            **owner,
        )

    def inspect(self, task_id: str) -> dict[str, Any]:
        with V1_MANAGER.exclusive_file_lock(self.lock, self.lock_timeout_seconds):
            return self._read(task_id)

    def renew(
        self,
        task_id: str,
        lease_seconds: int = 1800,
        now: datetime | None = None,
        **owner: Any,
    ) -> dict[str, Any]:
        if lease_seconds < 1 or lease_seconds > 86400:
            raise GovernanceError("lease_invalid", "Use a one-second to one-day lease.")
        current = _now(now)

        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            record["owner"]["lease_expires"] = _iso(
                current + timedelta(seconds=lease_seconds)
            )
            return record, {"lease_expires": record["owner"]["lease_expires"]}

        return self._mutate(
            task_id,
            event_type="LEASE_RENEWED",
            mutation=mutate,
            now=current,
            **owner,
        )

    def recover_same_session(
        self,
        task_id: str,
        expected_owner_session: str,
        expected_revision: int,
        expected_record_sha256: str,
        worktree: Path,
        reason: str,
        new_owner_token: str | None = None,
        lease_seconds: int = 1800,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if lease_seconds < 1 or lease_seconds > 86400:
            raise GovernanceError("lease_invalid", "Use a one-second to one-day lease.")
        current = _now(now)
        runtime = os.getenv(RUNTIME_SESSION_ENV)
        token = new_owner_token or secrets.token_urlsafe(32)
        if not runtime:
            raise GovernanceError("runtime_owner_missing", "Bind CODEX_THREAD_ID.")
        if len(token) < 16:
            raise GovernanceError("owner_token_weak", "Use a generated private token.")
        with V1_MANAGER.exclusive_file_lock(self.lock, self.lock_timeout_seconds):
            record = self._read(task_id)
            self._check_cas(record, expected_revision, expected_record_sha256)
            if record["owner"]["session"] != expected_owner_session:
                raise GovernanceError("recovery_owner_conflict", expected_owner_session)
            if record["owner"]["runtime_session"] != runtime:
                raise GovernanceError("recovery_runtime_mismatch", "Use the bound runtime.")
            if Path(record["worktree"]["path"]).resolve() != worktree.resolve():
                raise GovernanceError("recovery_worktree_mismatch", str(worktree))
            record["owner"]["token_sha256"] = hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest()
            record["owner"]["lease_expires"] = _iso(
                current + timedelta(seconds=lease_seconds)
            )
            payload = {
                "action": "same-session-token-recovery",
                "owner_session": expected_owner_session,
                "runtime_session": runtime,
                "reason": _clean(reason, "recovery_reason", 300),
                "prior_revision": expected_revision,
                "prior_record_sha256": expected_record_sha256,
            }
            record = self._finish_mutation(
                record,
                "OWNERSHIP_RECOVERED",
                payload,
                current,
            )
            _atomic_json(self._path(task_id), record)
        result = _json_copy(record)
        if new_owner_token is None:
            result["owner_token"] = token
        return result

    def takeover_expired(
        self,
        task_id: str,
        expected_owner_session: str,
        expected_revision: int,
        expected_record_sha256: str,
        new_owner_session: str,
        new_worktree: Path,
        reason: str,
        new_owner_token: str | None = None,
        lease_seconds: int = 1800,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _now(now)
        runtime = os.getenv(RUNTIME_SESSION_ENV)
        token = new_owner_token or secrets.token_urlsafe(32)
        if not runtime or runtime != new_owner_session:
            raise GovernanceError("runtime_owner_missing", "Bind the new owner runtime.")
        if len(token) < 16:
            raise GovernanceError("owner_token_weak", "Use a generated private token.")
        if lease_seconds < 1 or lease_seconds > 86400:
            raise GovernanceError("lease_invalid", "Use a one-second to one-day lease.")
        identity = _worktree_identity(new_worktree)
        _validate_worktree_binding(self.root, identity)
        with V1_MANAGER.exclusive_file_lock(self.lock, self.lock_timeout_seconds):
            record = self._read(task_id)
            self._check_cas(record, expected_revision, expected_record_sha256)
            if record["owner"]["session"] != expected_owner_session:
                raise GovernanceError("takeover_owner_conflict", expected_owner_session)
            if current < datetime.fromisoformat(record["owner"]["lease_expires"]):
                raise GovernanceError("takeover_lease_active", "Wait for lease expiry.")
            if Path(record["worktree"]["path"]).resolve() != new_worktree.resolve():
                raise GovernanceError(
                    "takeover_worktree_change_forbidden",
                    "One task retains one immutable worktree binding.",
                )
            previous = record["owner"]["session"]
            record["owner"] = {
                "session": new_owner_session,
                "runtime_session": runtime,
                "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                "lease_expires": _iso(current + timedelta(seconds=lease_seconds)),
            }
            payload = {
                "action": "expired-lease-takeover",
                "from_owner": previous,
                "to_owner": new_owner_session,
                "reason": _clean(reason, "takeover_reason", 300),
                "prior_revision": expected_revision,
                "prior_record_sha256": expected_record_sha256,
            }
            record = self._finish_mutation(
                record,
                "OWNERSHIP_TAKEN_OVER",
                payload,
                current,
            )
            _atomic_json(self._path(task_id), record)
        result = _json_copy(record)
        if new_owner_token is None:
            result["owner_token"] = token
        return result

    def administrative_takeover(
        self,
        task_id: str,
        confirm_task_id: str,
        confirmation: str,
        authorization_ref: str,
        expected_owner_session: str,
        expected_revision: int,
        expected_record_sha256: str,
        expected_worktree: Path,
        new_owner_session: str,
        new_worktree: Path,
        reason: str,
        new_owner_token: str | None = None,
        lease_seconds: int = 1800,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if confirm_task_id != task_id:
            raise GovernanceError("admin_task_confirmation_mismatch", confirm_task_id)
        if confirmation != "USER_AUTHORIZED_ADMIN_TAKEOVER":
            raise GovernanceError("admin_confirmation_missing", confirmation)
        current = _now(now)
        runtime = os.getenv(RUNTIME_SESSION_ENV)
        token = new_owner_token or secrets.token_urlsafe(32)
        if not runtime or runtime != new_owner_session:
            raise GovernanceError("runtime_owner_missing", "Bind the new owner runtime.")
        if len(token) < 16:
            raise GovernanceError("owner_token_weak", "Use a generated private token.")
        if lease_seconds < 1 or lease_seconds > 86400:
            raise GovernanceError("lease_invalid", "Use a one-second to one-day lease.")
        identity = _worktree_identity(new_worktree)
        _validate_worktree_binding(self.root, identity)
        with V1_MANAGER.exclusive_file_lock(self.lock, self.lock_timeout_seconds):
            record = self._read(task_id)
            self._check_cas(record, expected_revision, expected_record_sha256)
            if record["owner"]["session"] != expected_owner_session:
                raise GovernanceError("admin_owner_conflict", expected_owner_session)
            if Path(record["worktree"]["path"]).resolve() != expected_worktree.resolve():
                raise GovernanceError("admin_worktree_conflict", str(expected_worktree))
            if new_worktree.resolve() != expected_worktree.resolve():
                raise GovernanceError(
                    "admin_worktree_change_forbidden",
                    "One task retains one immutable worktree binding.",
                )
            previous = record["owner"]["session"]
            record["owner"] = {
                "session": new_owner_session,
                "runtime_session": runtime,
                "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                "lease_expires": _iso(current + timedelta(seconds=lease_seconds)),
            }
            payload = {
                "action": "administrative-takeover",
                "authorization_ref": _clean(
                    authorization_ref,
                    "authorization_ref",
                    300,
                ),
                "from_owner": previous,
                "to_owner": new_owner_session,
                "reason": _clean(reason, "administrative_reason", 300),
                "prior_revision": expected_revision,
                "prior_record_sha256": expected_record_sha256,
            }
            record = self._finish_mutation(
                record,
                "OWNERSHIP_ADMIN_TAKEOVER",
                payload,
                current,
            )
            _atomic_json(self._path(task_id), record)
        result = _json_copy(record)
        if new_owner_token is None:
            result["owner_token"] = token
        return result

    def rebaseline_worktree_fingerprint(
        self,
        task_id: str,
        confirm_task_id: str,
        confirmation: str,
        authorization_ref: str,
        evidence_ref: str,
        expected_revision: int,
        expected_record_sha256: str,
        expected_worktree: Path,
        expected_stored_fingerprint: str,
        expected_current_fingerprint: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Administratively rebaseline one verified dirty-worktree snapshot."""
        if confirm_task_id != task_id:
            raise GovernanceError("rebaseline_task_confirmation_mismatch", confirm_task_id)
        if confirmation != "USER_AUTHORIZED_WORKTREE_FINGERPRINT_REBASELINE":
            raise GovernanceError("rebaseline_confirmation_missing", confirmation)
        expected_stored = expected_stored_fingerprint.lower()
        expected_current = expected_current_fingerprint.lower()
        if not HEX_SHA256_RE.fullmatch(expected_stored):
            raise GovernanceError("rebaseline_stored_fingerprint_invalid", expected_stored)
        if not HEX_SHA256_RE.fullmatch(expected_current):
            raise GovernanceError("rebaseline_current_fingerprint_invalid", expected_current)
        current = _now(now)
        with V1_MANAGER.exclusive_file_lock(self.lock, self.lock_timeout_seconds):
            record = self._read(task_id)
            self._check_cas(record, expected_revision, expected_record_sha256)
            worktree = record.get("worktree")
            if not isinstance(worktree, dict):
                raise GovernanceError("rebaseline_worktree_metadata_missing", task_id)
            required = ("path", "common_dir", "head_sha", "branch", "detached", "fingerprint")
            if any(key not in worktree for key in required):
                raise GovernanceError("rebaseline_worktree_metadata_missing", task_id)
            if Path(worktree["path"]).resolve() != expected_worktree.resolve():
                raise GovernanceError("rebaseline_worktree_mismatch", str(expected_worktree))
            identity = _worktree_identity(expected_worktree)
            _validate_worktree_binding(self.root, identity)
            for key in ("path", "common_dir", "head_sha", "branch", "detached"):
                if worktree[key] != identity[key]:
                    raise GovernanceError("rebaseline_worktree_identity_mismatch", key)
            if not secrets.compare_digest(worktree["fingerprint"], expected_stored):
                raise GovernanceError(
                    "rebaseline_stored_fingerprint_mismatch",
                    "Inspect the current task record.",
                )
            if not secrets.compare_digest(identity["fingerprint"], expected_current):
                raise GovernanceError(
                    "rebaseline_current_fingerprint_mismatch",
                    "Recompute the worktree fingerprint before retrying.",
                )
            permit = record.get("active_permit")
            if permit:
                expires_at = permit.get("expires_at")
                if not isinstance(expires_at, str):
                    raise GovernanceError("rebaseline_active_permit_invalid", task_id)
                if current < datetime.fromisoformat(expires_at):
                    raise GovernanceError("rebaseline_active_permit", task_id)
            old_fingerprint = worktree["fingerprint"]
            worktree["dirty_paths"] = identity["dirty_paths"]
            worktree["fingerprint"] = identity["fingerprint"]
            payload = {
                "operation": "rebaseline-worktree-fingerprint",
                "old_fingerprint": old_fingerprint,
                "new_fingerprint": identity["fingerprint"],
                "prior_revision": expected_revision,
                "resulting_revision": expected_revision + 1,
                "administrator_confirmation": confirmation,
                "authorization_ref": _clean(
                    authorization_ref,
                    "rebaseline_authorization_ref",
                    300,
                ),
                "evidence_ref": _clean(evidence_ref, "rebaseline_evidence_ref", 500),
                "rebaselined_at": _iso(current),
            }
            record = self._finish_mutation(
                record,
                "WORKTREE_FINGERPRINT_REBASELINED",
                payload,
                current,
            )
            _atomic_json(self._path(task_id), record)
            return record

    def rebind_worktree_head(
        self,
        task_id: str,
        confirm_task_id: str,
        confirmation: str,
        authorization_ref: str,
        expected_revision: int,
        expected_record_sha256: str,
        expected_worktree: Path,
        expected_old_head: str,
        expected_new_head: str,
        expected_stored_fingerprint: str,
        expected_current_fingerprint: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Bind one task record to a proven descendant task-worktree commit."""
        if confirm_task_id != task_id:
            raise GovernanceError("head_rebind_task_confirmation_mismatch", confirm_task_id)
        if confirmation != "USER_AUTHORIZED_WORKTREE_HEAD_REBIND":
            raise GovernanceError("head_rebind_confirmation_missing", confirmation)
        values = {
            "expected_old_head": expected_old_head.lower(),
            "expected_new_head": expected_new_head.lower(),
            "expected_stored_fingerprint": expected_stored_fingerprint.lower(),
            "expected_current_fingerprint": expected_current_fingerprint.lower(),
        }
        if any(
            not re.fullmatch(r"[0-9a-f]{40}", values[key])
            for key in ("expected_old_head", "expected_new_head")
        ) or any(
            not HEX_SHA256_RE.fullmatch(values[key])
            for key in (
                "expected_stored_fingerprint",
                "expected_current_fingerprint",
            )
        ):
            raise GovernanceError("head_rebind_hash_invalid", "Provide Git and SHA-256 hashes.")
        current = _now(now)
        with V1_MANAGER.exclusive_file_lock(self.lock, self.lock_timeout_seconds):
            record = self._read(task_id)
            self._check_cas(record, expected_revision, expected_record_sha256)
            if record.get("active_permit"):
                raise GovernanceError("head_rebind_active_permit", task_id)
            worktree = record.get("worktree")
            if not isinstance(worktree, dict):
                raise GovernanceError("head_rebind_worktree_metadata_missing", task_id)
            if Path(worktree.get("path", "")).resolve() != expected_worktree.resolve():
                raise GovernanceError("head_rebind_worktree_mismatch", str(expected_worktree))
            if worktree.get("head_sha") != values["expected_old_head"]:
                raise GovernanceError("head_rebind_old_head_mismatch", task_id)
            if worktree.get("fingerprint") != values["expected_stored_fingerprint"]:
                raise GovernanceError("head_rebind_stored_fingerprint_mismatch", task_id)
            identity = _worktree_identity(expected_worktree)
            _validate_worktree_binding(self.root, identity)
            if identity["head_sha"] != values["expected_new_head"]:
                raise GovernanceError("head_rebind_actual_head_mismatch", identity["head_sha"])
            if identity["fingerprint"] != values["expected_current_fingerprint"]:
                raise GovernanceError("head_rebind_current_fingerprint_mismatch", task_id)
            if identity["dirty_paths"]:
                raise GovernanceError(
                    "head_rebind_unrelated_worktree_changes",
                    ",".join(identity["dirty_paths"]),
                )
            for key in ("path", "common_dir", "branch", "detached"):
                if worktree.get(key) != identity[key]:
                    raise GovernanceError("head_rebind_worktree_identity_mismatch", key)
            ancestry = _git(
                expected_worktree,
                "merge-base",
                "--is-ancestor",
                values["expected_old_head"],
                values["expected_new_head"],
            )
            if ancestry.returncode:
                raise GovernanceError("head_rebind_ancestry_missing", task_id)
            worktree["head_sha"] = identity["head_sha"]
            worktree["dirty_paths"] = identity["dirty_paths"]
            worktree["fingerprint"] = identity["fingerprint"]
            payload = {
                "operation": "rebind-worktree-head",
                "old_head": values["expected_old_head"],
                "new_head": values["expected_new_head"],
                "worktree": identity["path"],
                "old_fingerprint": values["expected_stored_fingerprint"],
                "new_fingerprint": identity["fingerprint"],
                "prior_revision": expected_revision,
                "administrator_confirmation": confirmation,
                "authorization_ref": _clean(
                    authorization_ref,
                    "head_rebind_authorization_ref",
                    300,
                ),
                "ancestry_proof": "merge-base-is-ancestor",
                "integration_commit": values["expected_new_head"],
            }
            record = self._finish_mutation(
                record,
                "WORKTREE_HEAD_REBOUND",
                payload,
                current,
            )
            _atomic_json(self._path(task_id), record)
            return record

    def reconcile_completed_history(
        self,
        task_id: str,
        confirm_task_id: str,
        confirmation: str,
        authorization_ref: str,
        expected_revision: int,
        expected_record_sha256: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Restore a cursor only from the record's prior STEP_ADVANCED events."""
        if confirm_task_id != task_id:
            raise GovernanceError("history_reconcile_task_confirmation_mismatch", confirm_task_id)
        if confirmation != "USER_AUTHORIZED_WORKTREE_HEAD_REBIND":
            raise GovernanceError("history_reconcile_confirmation_missing", confirmation)
        current = _now(now)
        with V1_MANAGER.exclusive_file_lock(self.lock, self.lock_timeout_seconds):
            record = self._read(task_id)
            self._check_cas(record, expected_revision, expected_record_sha256)
            if record.get("active_permit"):
                raise GovernanceError("history_reconcile_active_permit", task_id)
            completed: dict[str, dict[str, Any]] = {}
            for event in record.get("events", []):
                payload = event.get("payload", {})
                if (
                    event.get("event_type") == "STEP_ADVANCED"
                    and payload.get("terminal_status") == "DONE"
                    and payload.get("evidence_ids")
                ):
                    completed[payload.get("step_id", "")] = event
            steps = record["plan"]["steps"]
            prefix = []
            for step in steps:
                if step["step_id"] not in completed:
                    break
                prefix.append(step["step_id"])
            if not prefix:
                raise GovernanceError("history_reconcile_no_completed_prefix", task_id)
            for step in steps:
                step["status"] = "DONE" if step["step_id"] in prefix else "TODO"
                if step["status"] == "TODO":
                    step["evidence"] = []
            next_step = next(
                (step for step in steps if step["step_id"] not in prefix),
                None,
            )
            if next_step is not None:
                next_step["status"] = "IN_PROGRESS"
                record["state"] = "ACTIVE"
            else:
                record["state"] = "ACTIVE"
            payload = {
                "operation": "reconcile-completed-history",
                "completed_steps": prefix,
                "completion_event_sha256": {
                    step_id: completed[step_id]["event_sha256"] for step_id in prefix
                },
                "derived_next_step": next_step["step_id"] if next_step else None,
                "prior_revision": expected_revision,
                "administrator_confirmation": confirmation,
                "authorization_ref": _clean(
                    authorization_ref,
                    "history_reconcile_authorization_ref",
                    300,
                ),
            }
            record = self._finish_mutation(
                record,
                "COMPLETED_HISTORY_RECONCILED",
                payload,
                current,
            )
            _atomic_json(self._path(task_id), record)
            return record

    def confirm_plan(
        self,
        task_id: str,
        confirmation_ref: str,
        actor: str,
        **owner: Any,
    ) -> dict[str, Any]:
        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            if record["state"] not in {"PLANNED", "CONFIRMED"}:
                raise GovernanceError("plan_confirmation_state_invalid", record["state"])
            effects = {
                effect
                for step in record["plan"]["steps"]
                for effect in step["allowed_effects"]
            }
            reference = _clean(confirmation_ref, "confirmation_ref", 300)
            if effects.intersection(HIGH_RISK_EFFECTS) and actor != "user":
                raise GovernanceError(
                    "user_confirmation_required",
                    ",".join(sorted(effects.intersection(HIGH_RISK_EFFECTS))),
                )
            record["plan_confirmation"] = {
                "actor": _clean(actor, "confirmation_actor", 48),
                "reference": reference,
                "plan_digest": record["plan"]["digest"],
            }
            record["state"] = "CONFIRMED"
            return record, record["plan_confirmation"]

        return self._mutate(task_id, event_type="PLAN_CONFIRMED", mutation=mutate, **owner)

    @staticmethod
    def _record_accepted_progress(
        record: dict[str, Any],
        identity: dict[str, Any],
    ) -> None:
        worktree = record["worktree"]
        for key in ("head_sha", "dirty_paths", "fingerprint"):
            worktree[key] = identity[key]
        worktree["accepted_task_head"] = identity["head_sha"]
        worktree["accepted_task_fingerprint"] = identity["fingerprint"]
        worktree["actual_worktree_head"] = identity["head_sha"]
        worktree["actual_worktree_fingerprint"] = identity["fingerprint"]

    def _classify_expired_permit_progress(
        self,
        record: dict[str, Any],
        permit: dict[str, Any],
    ) -> dict[str, Any]:
        worktree = record["worktree"]
        actual = _worktree_identity(Path(worktree["path"]))
        _validate_worktree_binding(self.root, actual)
        accepted_head = worktree.get("accepted_task_head", worktree["head_sha"])
        accepted_fingerprint = worktree.get(
            "accepted_task_fingerprint", worktree["fingerprint"]
        )
        if actual["head_sha"] != accepted_head and not _is_ancestor(
            Path(worktree["path"]), accepted_head, actual["head_sha"]
        ):
            raise GovernanceError("external_or_owner_change", actual["head_sha"])
        changed = set(
            _changed_paths_since(
                Path(worktree["path"]),
                accepted_head,
                actual["head_sha"],
            )
        )
        changed.update(actual["dirty_paths"])
        scope = worktree.get("path_scope", [])
        in_scope = sorted(path for path in changed if _path_within_scope(path, scope))
        out_of_scope = sorted(changed.difference(in_scope))
        if changed and not scope:
            raise GovernanceError("unknown_or_mixed_change", "task scope is required")
        if in_scope and out_of_scope:
            raise GovernanceError("unknown_or_mixed_change", ",".join(out_of_scope))
        if out_of_scope:
            raise GovernanceError("external_or_owner_change", ",".join(out_of_scope))
        self._record_accepted_progress(record, actual)
        return {
            "classification": "TASK_OWNED_AUTHORIZED",
            "permit_id": permit["permit_id"],
            "previous_accepted_head": accepted_head,
            "previous_accepted_fingerprint": accepted_fingerprint,
            "actual_head": actual["head_sha"],
            "actual_fingerprint": actual["fingerprint"],
            "changed_paths": sorted(changed),
        }

    def permit(
        self,
        task_id: str,
        step_id: str,
        effects: list[str],
        ttl_seconds: int = 1800,
        now: datetime | None = None,
        **owner: Any,
    ) -> dict[str, Any]:
        current = _now(now)

        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            if record["state"] not in {"CONFIRMED", "ACTIVE"}:
                raise GovernanceError("task_not_confirmed", record["state"])
            confirmation = record.get("plan_confirmation") or {}
            if confirmation.get("plan_digest") != record["plan"]["digest"]:
                raise GovernanceError("plan_confirmation_stale", "Confirm the current digest.")
            expired_progress = None
            active = record.get("active_permit")
            if active:
                if current < datetime.fromisoformat(active["expires_at"]):
                    raise GovernanceError("permit_already_active", "Consume or revoke it first.")
                expired_progress = self._classify_expired_permit_progress(record, active)
                record["active_permit"] = None
                record["state"] = "CONFIRMED"
            _require_valid_skill_inventory(self.root)
            steps = {step["step_id"]: step for step in record["plan"]["steps"]}
            step = steps.get(step_id)
            if not step or step["status"] != "IN_PROGRESS":
                raise GovernanceError("permit_step_invalid", step_id)
            requested = sorted(set(effects))
            if not set(requested).issubset(set(step["allowed_effects"])):
                raise GovernanceError("permit_effect_outside_plan", ",".join(requested))
            authorities = _revalidate_authorities(self.root, record["authorities"])
            skills = _revalidate_skills(
                self.root,
                record["task_class"],
                record["skills"],
            )
            reads = record.get("skill_reads", [])
            for selected in skills:
                if not any(
                    item.get("skill_id") == selected["skill_id"]
                    and item.get("skill_sha256") == selected["skill_sha256"]
                    and item.get("plan_version") == record["plan"]["version"]
                    and step_id in item.get("applies_to_step_ids", [])
                    for item in reads
                ):
                    raise GovernanceError(
                        "skill_read_receipt_missing",
                        f"{selected['skill_id']}:{step_id}",
                    )
            if _digest(authorities) != record["authority_digest"]:
                raise GovernanceError("authority_digest_drift", step_id)
            if _digest(skills) != record["skill_digest"]:
                raise GovernanceError("skill_digest_drift", step_id)
            identity = _worktree_identity(Path(record["worktree"]["path"]))
            _validate_worktree_binding(self.root, identity)
            if identity["head_sha"] != record["worktree"]["head_sha"]:
                raise GovernanceError("worktree_head_drift", identity["head_sha"])
            if identity["fingerprint"] != record["worktree"]["fingerprint"]:
                raise GovernanceError(
                    "worktree_fingerprint_drift",
                    "Review current dirty paths and amend the plan before effect.",
                )
            permit = {
                "permit_id": secrets.token_hex(16),
                "step_id": step_id,
                "effects": requested,
                "plan_digest": record["plan"]["digest"],
                "authority_digest": record["authority_digest"],
                "skill_digest": record["skill_digest"],
                "worktree_fingerprint": identity["fingerprint"],
                "issued_at": _iso(current),
                "expires_at": _iso(current + timedelta(seconds=ttl_seconds)),
            }
            record["active_permit"] = permit
            record["state"] = "ACTIVE"
            record["worktree"]["state"] = "ACTIVE"
            return record, {**permit, "expired_permit_progress": expired_progress}

        return self._mutate(
            task_id,
            event_type="ACTION_PERMIT_ISSUED",
            mutation=mutate,
            now=current,
            **owner,
        )

    def renew_permit(
        self,
        task_id: str,
        permit_id: str,
        ttl_seconds: int = 1800,
        now: datetime | None = None,
        **owner: Any,
    ) -> dict[str, Any]:
        """Extend one still-valid permit without changing its permitted effects."""
        if ttl_seconds < 1 or ttl_seconds > 86400:
            raise GovernanceError("permit_ttl_invalid", "Use a one-second to one-day TTL.")
        current = _now(now)

        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            permit = record.get("active_permit")
            if not permit or permit.get("permit_id") != permit_id:
                raise GovernanceError("permit_invalid", "Use the current permit ID.")
            expires_at = datetime.fromisoformat(permit["expires_at"])
            if current >= expires_at:
                raise GovernanceError("permit_expired", "Issue a fresh permit.")
            new_expiry = current + timedelta(seconds=ttl_seconds)
            permit["expires_at"] = _iso(new_expiry)
            return record, {
                "permit_id": permit_id,
                "step_id": permit["step_id"],
                "prior_expires_at": _iso(expires_at),
                "expires_at": permit["expires_at"],
            }

        return self._mutate(
            task_id,
            event_type="ACTION_PERMIT_RENEWED",
            mutation=mutate,
            now=current,
            **owner,
        )

    def advance(
        self,
        task_id: str,
        permit_id: str,
        evidence: list[dict[str, Any]],
        next_step_id: str | None,
        terminal_status: str = "DONE",
        failed_gate: str | None = None,
        next_action: str | None = None,
        now: datetime | None = None,
        **owner: Any,
    ) -> dict[str, Any]:
        current = _now(now)

        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            permit = record.get("active_permit")
            if not permit or permit.get("permit_id") != permit_id:
                raise GovernanceError("permit_invalid", "Use the current permit ID.")
            if current >= datetime.fromisoformat(permit["expires_at"]):
                raise GovernanceError("permit_expired", "Issue a fresh permit.")
            step = next(
                item
                for item in record["plan"]["steps"]
                if item["step_id"] == permit["step_id"]
            )
            if terminal_status not in STEP_COMPLETION_STATES:
                raise GovernanceError("step_terminal_status_invalid", terminal_status)
            require_pass = terminal_status == "DONE"
            normalized = self._validate_evidence(
                evidence,
                step["acceptance_ids"],
                require_pass=require_pass,
            )
            if terminal_status in {"BLOCKED", "CANCELLED"}:
                if not failed_gate or not next_action:
                    raise GovernanceError(
                        "step_terminal_context_missing",
                        "Bind failed_gate and next_action for non-DONE steps.",
                    )
                if next_step_id:
                    raise GovernanceError(
                        "blocked_step_cannot_advance",
                        "Amend the plan before another effect.",
                    )
            step["evidence"] = normalized
            step["status"] = terminal_status
            if failed_gate:
                step["failed_gate"] = _clean(failed_gate, "failed_gate", 300)
            if next_action:
                step["next_action"] = _clean(next_action, "next_action", 300)
            if next_step_id:
                target = next(
                    (item for item in record["plan"]["steps"] if item["step_id"] == next_step_id),
                    None,
                )
                if target is None or target["status"] != "TODO":
                    raise GovernanceError("next_step_invalid", str(next_step_id))
                target["status"] = "IN_PROGRESS"
            identity = _worktree_identity(Path(record["worktree"]["path"]))
            _validate_worktree_binding(self.root, identity)
            for key in ("path", "common_dir", "branch", "detached"):
                record["worktree"][key] = identity[key]
            self._record_accepted_progress(record, identity)
            record["active_permit"] = None
            return record, {
                "step_id": step["step_id"],
                "terminal_status": terminal_status,
                "evidence_ids": [item["evidence_id"] for item in normalized],
                "next_step_id": next_step_id,
                "worktree_snapshot": {
                    "head_sha": identity["head_sha"],
                    "dirty_paths": identity["dirty_paths"],
                    "fingerprint": identity["fingerprint"],
                },
            }

        return self._mutate(
            task_id,
            event_type="STEP_ADVANCED",
            mutation=mutate,
            now=current,
            **owner,
        )

    def _validate_evidence(
        self,
        evidence: list[dict[str, Any]],
        acceptance_ids: list[str],
        require_pass: bool = True,
    ) -> list[dict[str, Any]]:
        if not evidence:
            raise GovernanceError("evidence_missing", "Bind typed evidence.")
        covered: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for item in evidence:
            supports = sorted(set(item.get("supports", [])))
            if not set(supports).issubset(set(acceptance_ids)):
                raise GovernanceError("evidence_acceptance_invalid", str(supports))
            status = _clean(item.get("status"), "evidence_status", 32)
            if status not in {"PASS", "FAIL", "OBSERVED", "NOT_AVAILABLE"}:
                raise GovernanceError("evidence_status_invalid", status)
            locator = _clean(item.get("locator"), "evidence_locator", 500)
            supplied_hash = str(item.get("sha256", "")).lower() or None
            if item.get("kind") == "artifact" and not supplied_hash:
                raise GovernanceError(
                    "artifact_evidence_hash_missing",
                    locator,
                )
            if supplied_hash:
                if not HEX_SHA256_RE.fullmatch(supplied_hash):
                    raise GovernanceError("evidence_hash_invalid", locator)
                path = (self.root / locator).resolve()
                try:
                    path.relative_to(self.root)
                except ValueError as exc:
                    raise GovernanceError("evidence_path_escape", locator) from exc
                if not path.is_file() or _file_sha256(path) != supplied_hash:
                    raise GovernanceError("evidence_hash_mismatch", locator)
            if status == "PASS":
                covered.update(supports)
            normalized.append(
                {
                    "evidence_id": _clean(item.get("evidence_id"), "evidence_id", 80),
                    "kind": _clean(item.get("kind"), "evidence_kind", 48),
                    "locator": locator,
                    "sha256": supplied_hash,
                    "supports": supports,
                    "status": status,
                }
            )
        missing = set(acceptance_ids) - covered
        if require_pass and missing:
            raise GovernanceError("acceptance_evidence_missing", ",".join(sorted(missing)))
        return normalized

    def amend_plan(
        self,
        task_id: str,
        steps: list[dict[str, Any]],
        reason: str,
        **owner: Any,
    ) -> dict[str, Any]:
        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            if record["state"] == "CLOSED":
                raise GovernanceError("closed_task_immutable", task_id)
            packet = {
                "acceptance": record["acceptance"],
                "plan": {"steps": steps},
            }
            new_steps = _steps(packet)
            record["plan"] = {
                "version": record["plan"]["version"] + 1,
                "steps": new_steps,
            }
            record["plan"]["digest"] = _plan_digest(record["plan"])
            record["plan_confirmation"] = None
            record["active_permit"] = None
            record["skill_reads"] = []
            record["state"] = "PLANNED"
            return record, {
                "reason": _clean(reason, "amendment_reason", 500),
                "plan_digest": record["plan"]["digest"],
            }

        return self._mutate(task_id, event_type="PLAN_AMENDED", mutation=mutate, **owner)

    def refresh_authority_receipts(
        self,
        task_id: str,
        receipt_packet: dict[str, Any],
        confirmation: str,
        authorization_ref: str,
        **owner: Any,
    ) -> dict[str, Any]:
        """Refresh existing authority receipts after explicit user confirmation.

        This transition intentionally cannot alter task scope, plans, skills, or
        permit effects.  It exists only for a task whose own bounded work has
        updated an already-declared authority document.
        """
        if confirmation != "ALLOW_EXPLICIT_AUTHORITY_RECEIPT_REFRESH":
            raise GovernanceError("authority_refresh_confirmation_missing", confirmation)
        if not isinstance(receipt_packet, dict):
            raise GovernanceError("authority_refresh_packet_invalid", task_id)
        supplied = receipt_packet.get("authorities")
        if not isinstance(supplied, list):
            raise GovernanceError("authority_refresh_receipts_missing", task_id)
        authorization = _clean(
            authorization_ref,
            "authority_refresh_authorization_ref",
            300,
        )

        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            if record["state"] == "CLOSED":
                raise GovernanceError("closed_task_immutable", task_id)
            if record.get("active_permit"):
                raise GovernanceError("authority_refresh_active_permit", task_id)
            current_pairs = {
                (item["scope"], item["locator"])
                for item in record["authorities"]
            }
            supplied_pairs = {
                (str(item.get("scope", "")), str(item.get("locator", "")))
                for item in supplied
                if isinstance(item, dict)
            }
            if supplied_pairs != current_pairs or len(supplied_pairs) != len(supplied):
                raise GovernanceError(
                    "authority_refresh_scope_change_forbidden",
                    "Refresh exactly the existing authority scopes and locators.",
                )
            refreshed = _authority_receipts(self.root, supplied)
            old_digest = record["authority_digest"]
            new_digest = _digest(refreshed)
            record["authorities"] = refreshed
            record["authority_digest"] = new_digest
            return record, {
                "operation": "explicit-authority-receipt-refresh",
                "old_authority_digest": old_digest,
                "new_authority_digest": new_digest,
                "administrator_confirmation": confirmation,
                "authorization_ref": authorization,
            }

        return self._mutate(
            task_id,
            event_type="AUTHORITY_RECEIPTS_REFRESHED",
            mutation=mutate,
            **owner,
        )

    def review_outcome(
        self,
        task_id: str,
        outcome_packet: dict[str, Any],
        **owner: Any,
    ) -> dict[str, Any]:
        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            if record.get("active_permit"):
                raise GovernanceError("permit_still_active", "Consume the permit first.")
            unfinished = [
                step["step_id"]
                for step in record["plan"]["steps"]
                if step["status"] not in STEP_COMPLETION_STATES
            ]
            if unfinished:
                raise GovernanceError(
                    "plan_not_terminal",
                    ",".join(unfinished),
                )
            outcome = _clean(outcome_packet.get("outcome"), "outcome", 32)
            if outcome not in OUTCOMES:
                raise GovernanceError("outcome_invalid", outcome)
            dispositions = outcome_packet.get("path_dispositions", [])
            disposition_paths = [str(item.get("path", "")) for item in dispositions]
            if len(disposition_paths) != len(set(disposition_paths)):
                raise GovernanceError(
                    "path_disposition_duplicate",
                    "Each dirty path has exactly one disposition.",
                )
            for item in dispositions:
                disposition = item.get("disposition")
                if disposition not in PATH_DISPOSITIONS:
                    raise GovernanceError("path_disposition_invalid", str(disposition))
            current_dirty = set(
                _worktree_identity(Path(record["worktree"]["path"]))["dirty_paths"]
            )
            disposed = set(disposition_paths)
            if current_dirty != disposed:
                raise GovernanceError(
                    "dirty_path_undispositioned",
                    ",".join(sorted(current_dirty.symmetric_difference(disposed))),
                )
            record["worktree"]["state"] = "OUTCOME_REVIEWED"
            record["worktree"]["outcome"] = outcome
            record["worktree"]["path_dispositions"] = dispositions
            record["worktree"]["integration"] = outcome_packet.get("integration")
            record["state"] = "REVIEWED"
            return record, outcome_packet

        return self._mutate(task_id, event_type="OUTCOME_REVIEWED", mutation=mutate, **owner)

    def close(
        self,
        task_id: str,
        learning: dict[str, Any],
        retirement: str,
        skill_maintenance: list[dict[str, Any]] | None = None,
        **owner: Any,
    ) -> dict[str, Any]:
        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            if record["state"] != "REVIEWED":
                raise GovernanceError("outcome_not_reviewed", record["state"])
            disposition = learning.get("disposition")
            if disposition not in LEARNING_DISPOSITIONS:
                raise GovernanceError("learning_disposition_invalid", str(disposition))
            self._validate_learning(learning)
            outcome = record["worktree"]["outcome"]
            integration = record["worktree"].get("integration") or {}
            if outcome in {"ACCEPTED", "PARTIAL"}:
                self._validate_integration(integration)
            if outcome in {"REJECTED", "PARTIAL"}:
                self._validate_failure_extraction(record)
            if outcome in {"BLOCKED", "UNKNOWN"} and retirement != "PROTECTED":
                raise GovernanceError("unknown_worktree_must_remain_protected", outcome)
            if retirement not in RETIREMENT_STATES:
                raise GovernanceError("retirement_state_invalid", retirement)
            if retirement == "RETIRED":
                raise GovernanceError(
                    "retirement_transition_required",
                    "Close as RETIRE_ELIGIBLE, then record verified removal separately.",
                )
            normalized_maintenance = self._validate_skill_maintenance(
                skill_maintenance or []
            )
            record["learning"] = learning
            record["skill_maintenance"] = normalized_maintenance
            record["worktree"]["retirement"] = retirement
            record["state"] = "CLOSED"
            return record, {
                "learning": learning,
                "retirement": retirement,
                "skill_maintenance": normalized_maintenance,
            }

        return self._mutate(task_id, event_type="TASK_CLOSED", mutation=mutate, **owner)

    def record_retirement(
        self,
        task_id: str,
        retirement_packet: dict[str, Any],
        **owner: Any,
    ) -> dict[str, Any]:
        """Record removal only after independent removal, reference, and process proof."""

        def mutate(record: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            if record["state"] != "CLOSED":
                raise GovernanceError("retirement_task_not_closed", record["state"])
            if record["worktree"].get("retirement") != "RETIRE_ELIGIBLE":
                raise GovernanceError(
                    "retirement_not_eligible",
                    "Close as RETIRE_ELIGIBLE before removal proof.",
                )
            former = Path(record["worktree"]["path"]).resolve()
            if former.exists():
                raise GovernanceError(
                    "retirement_worktree_present",
                    "Remove the registered worktree before recording retirement.",
                )
            registered = _git(self.root, "worktree", "list", "--porcelain")
            if registered.returncode:
                raise GovernanceError("retirement_worktree_query_failed", registered.stderr)
            known_paths = {
                Path(line.removeprefix("worktree ")).resolve()
                for line in registered.stdout.splitlines()
                if line.startswith("worktree ")
            }
            if former in known_paths:
                raise GovernanceError(
                    "retirement_worktree_still_registered",
                    str(former),
                )
            for field in ("reference_audit", "process_audit"):
                audit = retirement_packet.get(field)
                if not isinstance(audit, dict) or audit.get("status") != "PASS":
                    raise GovernanceError(
                        "retirement_proof_missing",
                        f"{field} must be a PASS audit.",
                    )
                _clean(audit.get("command"), f"retirement_{field}_command", 500)
            locator = _clean(
                retirement_packet.get("evidence_locator"),
                "retirement_evidence_locator",
                500,
            )
            supplied = str(retirement_packet.get("evidence_sha256", "")).lower()
            if not HEX_SHA256_RE.fullmatch(supplied):
                raise GovernanceError("retirement_evidence_hash_invalid", locator)
            evidence_path = (self.root / locator).resolve()
            try:
                evidence_path.relative_to(self.root)
            except ValueError as exc:
                raise GovernanceError("retirement_evidence_path_escape", locator) from exc
            try:
                evidence_path.relative_to(former)
            except ValueError:
                pass
            else:
                raise GovernanceError(
                    "retirement_evidence_not_preserved",
                    "Keep proof outside the removed worktree.",
                )
            if not evidence_path.is_file() or _file_sha256(evidence_path) != supplied:
                raise GovernanceError("retirement_evidence_hash_mismatch", locator)
            branch_disposition = _clean(
                retirement_packet.get("branch_disposition"),
                "retirement_branch_disposition",
                64,
            )
            if branch_disposition not in {"PRESERVED", "DELETED_VERIFIED"}:
                raise GovernanceError("retirement_branch_disposition_invalid", branch_disposition)
            branch = record["worktree"].get("branch")
            if branch:
                branch_probe = _git(
                    self.root,
                    "show-ref",
                    "--verify",
                    "--quiet",
                    f"refs/heads/{branch}",
                )
                if branch_disposition == "PRESERVED" and branch_probe.returncode:
                    raise GovernanceError("retirement_branch_missing", branch)
                if branch_disposition == "DELETED_VERIFIED" and not branch_probe.returncode:
                    raise GovernanceError("retirement_branch_still_present", branch)
            record["worktree"]["retirement"] = "RETIRED"
            record["worktree"]["state"] = "RETIRED"
            record["worktree"]["retirement_proof"] = {
                "evidence_locator": locator,
                "evidence_sha256": supplied,
                "reference_audit": retirement_packet["reference_audit"],
                "process_audit": retirement_packet["process_audit"],
                "branch_disposition": branch_disposition,
            }
            return record, record["worktree"]["retirement_proof"]

        return self._mutate(
            task_id,
            event_type="WORKTREE_RETIRED",
            mutation=mutate,
            **owner,
        )

    def _validate_integration(self, integration: dict[str, Any]) -> None:
        required = {
            "target_ref",
            "target_sha",
            "integrated_sha",
            "proof_kind",
            "revalidation_evidence",
        }
        if not required.issubset(integration):
            raise GovernanceError(
                "integration_revalidation_missing",
                "Bind target ref/SHA, integrated SHA, proof kind, and evidence.",
            )
        target_ref = _clean(integration.get("target_ref"), "target_ref", 300)
        target = _git(self.root, "rev-parse", "--verify", f"{target_ref}^{{commit}}")
        if target.returncode:
            raise GovernanceError("integration_target_missing", target_ref)
        target_sha = target.stdout.strip().lower()
        if integration.get("target_sha") != target_sha:
            raise GovernanceError("integration_target_drift", target_sha)
        integrated_sha = str(integration.get("integrated_sha", "")).lower()
        integrated = _git(self.root, "cat-file", "-e", f"{integrated_sha}^{{commit}}")
        if integrated.returncode:
            raise GovernanceError("integrated_commit_missing", integrated_sha)
        proof_kind = integration.get("proof_kind")
        if proof_kind not in INTEGRATION_PROOF_KINDS:
            raise GovernanceError("integration_proof_invalid", str(proof_kind))
        if proof_kind == "ANCESTOR":
            ancestor = _git(
                self.root,
                "merge-base",
                "--is-ancestor",
                integrated_sha,
                target_sha,
            )
            if ancestor.returncode:
                raise GovernanceError(
                    "integrated_commit_not_reachable",
                    f"{integrated_sha}:{target_sha}",
                )
        else:
            proof = integration.get("patch_proof") or {}
            locator = _clean(proof.get("locator"), "patch_proof_locator", 500)
            supplied = str(proof.get("sha256", "")).lower()
            path = (self.root / locator).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise GovernanceError("patch_proof_path_escape", locator) from exc
            if not path.is_file() or _file_sha256(path) != supplied:
                raise GovernanceError("patch_proof_hash_mismatch", locator)
        evidence = integration.get("revalidation_evidence")
        if not isinstance(evidence, list) or not evidence:
            raise GovernanceError(
                "integration_revalidation_missing",
                "Bind typed PASS revalidation evidence.",
            )
        for item in evidence:
            if not isinstance(item, dict) or item.get("status") != "PASS":
                raise GovernanceError("integration_revalidation_not_pass", str(item))
            if item.get("target_sha") != target_sha:
                raise GovernanceError("integration_evidence_target_mismatch", target_sha)
            _clean(item.get("evidence_id"), "integration_evidence_id", 80)
            _clean(item.get("locator"), "integration_evidence_locator", 500)

    def _validate_failure_extraction(self, record: dict[str, Any]) -> None:
        extracted = [
            item
            for item in record["worktree"]["path_dispositions"]
            if item.get("disposition") == "EXTRACT_EVIDENCE"
        ]
        if not extracted:
            raise GovernanceError(
                "failure_evidence_not_extracted",
                "Bind reusable or diagnostic failure evidence.",
            )
        worktree = Path(record["worktree"]["path"]).resolve()
        for item in extracted:
            locator = _clean(
                item.get("evidence_locator"),
                "extracted_evidence_locator",
                500,
            )
            supplied = str(item.get("evidence_sha256", "")).lower()
            if not HEX_SHA256_RE.fullmatch(supplied):
                raise GovernanceError("failure_evidence_hash_invalid", locator)
            path = (self.root / locator).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise GovernanceError("failure_evidence_path_escape", locator) from exc
            try:
                path.relative_to(worktree)
            except ValueError:
                pass
            else:
                if worktree != self.root:
                    raise GovernanceError(
                        "failure_evidence_not_preserved",
                        "Extract evidence outside the retiring worktree.",
                    )
            if not path.is_file() or _file_sha256(path) != supplied:
                raise GovernanceError("failure_evidence_hash_mismatch", locator)

    @staticmethod
    def _validate_learning(learning: dict[str, Any]) -> None:
        disposition = learning["disposition"]
        if disposition == "VALIDATED_CORRECTION":
            required = {
                "root_cause",
                "correction",
                "validation_evidence",
                "reuse_when",
                "do_not_reuse_when",
            }
        elif disposition == "UNVERIFIED_FAILURE":
            required = {
                "observations",
                "evidence",
                "hypotheses",
                "preserved_location",
                "next_validation",
            }
        else:
            required = {"rationale", "evidence"}
        missing = [field for field in sorted(required) if not learning.get(field)]
        if missing:
            raise GovernanceError("learning_fields_missing", ",".join(missing))

    def _validate_skill_maintenance(
        self,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not entries:
            raise GovernanceError(
                "skill_impact_missing",
                "Record NO_SKILL_IMPACT or an affected-skill disposition.",
            )
        skills = _skill_map(self.root)
        normalized: list[dict[str, Any]] = []
        for item in entries:
            status = item.get("status")
            if status not in SKILL_IMPACT_DISPOSITIONS:
                raise GovernanceError("maintenance_status_invalid", str(status))
            if status == "NO_SKILL_IMPACT":
                if len(entries) != 1:
                    raise GovernanceError(
                        "skill_impact_conflict",
                        "NO_SKILL_IMPACT must be the only disposition.",
                    )
                normalized.append(
                    {
                        "status": status,
                        "rationale": _clean(
                            item.get("rationale"),
                            "skill_impact_rationale",
                            500,
                        ),
                        "evidence": item.get("evidence", []),
                    }
                )
                continue
            skill_id = _clean(item.get("skill_id"), "maintenance_skill_id", 128)
            if skill_id not in skills:
                raise GovernanceError("maintenance_skill_unknown", skill_id)
            normalized.append(
                {
                    "skill_id": skill_id,
                    "status": status,
                    "trigger": _clean(item.get("trigger"), "maintenance_trigger", 300),
                    "evidence": item.get("evidence", []),
                    "next_action": _clean(
                        item.get("next_action"),
                        "maintenance_next_action",
                        300,
                    ),
                }
            )
        return normalized

    def bootstrap(self) -> dict[str, Any]:
        active: list[dict[str, Any]] = []
        if self.tasks.is_dir():
            for path in sorted(self.tasks.glob("*.json")):
                record = _load_json(path)
                _validate_record(record)
                if record.get("state") != "CLOSED":
                    active.append(
                        {
                            "task_id": record.get("task_id"),
                            "state": record.get("state"),
                            "active_step": next(
                                (
                                    step["step_id"]
                                    for step in record.get("plan", {}).get("steps", [])
                                    if step.get("status") == "IN_PROGRESS"
                                ),
                                None,
                            ),
                            "worktree": record.get("worktree", {}).get("path"),
                            "next_action": (
                                "inspect task, retrieve listed authorities, and obey permit gate"
                            ),
                        }
                    )
        v1_ids = _v1_task_ids(self.root)
        overlap = sorted({item["task_id"] for item in active}.intersection(v1_ids))
        if overlap:
            raise GovernanceError("v1_v2_task_id_collision", ",".join(overlap))
        _require_valid_skill_inventory(self.root)
        index = _load_json(self.root / AUTHORITY_INDEX_RELATIVE)
        return {
            "schema_version": "pig.agent-bootstrap.v1",
            "active_tasks": active,
            "authority_scopes": sorted(
                entry["scope"] for entry in index.get("entries", [])
            ),
            "legacy_fallback": (
                ".agents/skills/project-state-steward/scripts/"
                "manage_short_memory.py inspect --task-id <ID>"
            ),
        }


def _owner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--expected-record-sha256", required=True)
    parser.add_argument("--owner-token", default=os.getenv("PIG_TASK_OWNER_TOKEN"))
    parser.add_argument("--worktree", type=Path, default=Path.cwd())


def _load_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise GovernanceError("payload_invalid", f"Expected a JSON object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage agent governance V2 records.")
    parser.add_argument("--coordination-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--task-id", required=True)
    create = commands.add_parser("create")
    create.add_argument("--packet", required=True, type=Path)
    create.add_argument("--owner-session", default=os.getenv(RUNTIME_SESSION_ENV))
    create.add_argument("--owner-token")
    create.add_argument("--worktree", type=Path, default=Path.cwd())
    create.add_argument("--lease-seconds", type=int, default=1800)
    confirm = commands.add_parser("confirm-plan")
    confirm.add_argument("--task-id", required=True)
    confirm.add_argument("--confirmation-ref", required=True)
    confirm.add_argument("--actor", choices=("agent", "user"), required=True)
    _owner_arguments(confirm)
    skill_read = commands.add_parser("record-skill-read")
    skill_read.add_argument("--task-id", required=True)
    skill_read.add_argument("--skill-id", required=True)
    skill_read.add_argument("--skill-sha256", required=True)
    skill_read.add_argument("--step-id", action="append", required=True)
    _owner_arguments(skill_read)
    permit = commands.add_parser("permit")
    permit.add_argument("--task-id", required=True)
    permit.add_argument("--step-id", required=True)
    permit.add_argument("--effect", action="append", default=[])
    permit.add_argument("--ttl-seconds", type=int, default=1800)
    _owner_arguments(permit)
    advance = commands.add_parser("advance")
    advance.add_argument("--task-id", required=True)
    advance.add_argument("--permit-id", required=True)
    advance.add_argument("--evidence-packet", required=True, type=Path)
    advance.add_argument("--next-step-id")
    advance.add_argument(
        "--terminal-status",
        choices=sorted(STEP_COMPLETION_STATES),
        default="DONE",
    )
    advance.add_argument("--failed-gate")
    advance.add_argument("--next-action")
    _owner_arguments(advance)
    renew = commands.add_parser("renew")
    renew.add_argument("--task-id", required=True)
    renew.add_argument("--lease-seconds", type=int, default=1800)
    _owner_arguments(renew)
    recover = commands.add_parser("recover")
    recover.add_argument("--task-id", required=True)
    recover.add_argument("--expected-owner-session", required=True)
    recover.add_argument("--reason", required=True)
    recover.add_argument("--new-owner-token")
    recover.add_argument("--lease-seconds", type=int, default=1800)
    recover.add_argument("--expected-revision", required=True, type=int)
    recover.add_argument("--expected-record-sha256", required=True)
    recover.add_argument("--worktree", type=Path, default=Path.cwd())
    takeover = commands.add_parser("takeover")
    takeover.add_argument("--task-id", required=True)
    takeover.add_argument("--expected-owner-session", required=True)
    takeover.add_argument("--new-owner-session", required=True)
    takeover.add_argument("--new-owner-token")
    takeover.add_argument("--new-worktree", type=Path, default=Path.cwd())
    takeover.add_argument("--reason", required=True)
    takeover.add_argument("--lease-seconds", type=int, default=1800)
    takeover.add_argument("--expected-revision", required=True, type=int)
    takeover.add_argument("--expected-record-sha256", required=True)
    admin = commands.add_parser("admin-takeover")
    admin.add_argument("--task-id", required=True)
    admin.add_argument("--confirm-task-id", required=True)
    admin.add_argument("--confirmation", required=True)
    admin.add_argument("--authorization-ref", required=True)
    admin.add_argument("--expected-owner-session", required=True)
    admin.add_argument("--expected-worktree", type=Path, required=True)
    admin.add_argument("--new-owner-session", required=True)
    admin.add_argument("--new-owner-token")
    admin.add_argument("--new-worktree", type=Path, default=Path.cwd())
    admin.add_argument("--reason", required=True)
    admin.add_argument("--lease-seconds", type=int, default=1800)
    admin.add_argument("--expected-revision", required=True, type=int)
    admin.add_argument("--expected-record-sha256", required=True)
    rebaseline = commands.add_parser("rebaseline-worktree-fingerprint")
    rebaseline.add_argument("--task-id", required=True)
    rebaseline.add_argument("--confirm-task-id", required=True)
    rebaseline.add_argument("--confirmation", required=True)
    rebaseline.add_argument("--authorization-ref", required=True)
    rebaseline.add_argument("--evidence-ref", required=True)
    rebaseline.add_argument("--expected-worktree", type=Path, required=True)
    rebaseline.add_argument("--expected-stored-fingerprint", required=True)
    rebaseline.add_argument("--expected-current-fingerprint", required=True)
    rebaseline.add_argument("--expected-revision", required=True, type=int)
    rebaseline.add_argument("--expected-record-sha256", required=True)
    head_rebind = commands.add_parser("rebind-worktree-head")
    head_rebind.add_argument("--task-id", required=True)
    head_rebind.add_argument("--confirm-task-id", required=True)
    head_rebind.add_argument("--confirmation", required=True)
    head_rebind.add_argument("--authorization-ref", required=True)
    head_rebind.add_argument("--expected-worktree", type=Path, required=True)
    head_rebind.add_argument("--expected-old-head", required=True)
    head_rebind.add_argument("--expected-new-head", required=True)
    head_rebind.add_argument("--expected-stored-fingerprint", required=True)
    head_rebind.add_argument("--expected-current-fingerprint", required=True)
    head_rebind.add_argument("--expected-revision", required=True, type=int)
    head_rebind.add_argument("--expected-record-sha256", required=True)
    history_reconcile = commands.add_parser("reconcile-completed-history")
    history_reconcile.add_argument("--task-id", required=True)
    history_reconcile.add_argument("--confirm-task-id", required=True)
    history_reconcile.add_argument("--confirmation", required=True)
    history_reconcile.add_argument("--authorization-ref", required=True)
    history_reconcile.add_argument("--expected-revision", required=True, type=int)
    history_reconcile.add_argument("--expected-record-sha256", required=True)
    renew_permit = commands.add_parser("renew-permit")
    renew_permit.add_argument("--task-id", required=True)
    renew_permit.add_argument("--permit-id", required=True)
    renew_permit.add_argument("--ttl-seconds", type=int, default=1800)
    _owner_arguments(renew_permit)
    amend = commands.add_parser("amend-plan")
    amend.add_argument("--task-id", required=True)
    amend.add_argument("--plan-packet", required=True, type=Path)
    amend.add_argument("--reason", required=True)
    _owner_arguments(amend)
    refresh = commands.add_parser("refresh-authority-receipts")
    refresh.add_argument("--task-id", required=True)
    refresh.add_argument("--receipt-packet", required=True, type=Path)
    refresh.add_argument("--confirmation", required=True)
    refresh.add_argument("--authorization-ref", required=True)
    _owner_arguments(refresh)
    review = commands.add_parser("review-outcome")
    review.add_argument("--task-id", required=True)
    review.add_argument("--outcome-packet", required=True, type=Path)
    _owner_arguments(review)
    close = commands.add_parser("close")
    close.add_argument("--task-id", required=True)
    close.add_argument("--learning-packet", required=True, type=Path)
    close.add_argument("--retirement", required=True, choices=sorted(RETIREMENT_STATES))
    _owner_arguments(close)
    retirement = commands.add_parser("record-retirement")
    retirement.add_argument("--task-id", required=True)
    retirement.add_argument("--retirement-packet", required=True, type=Path)
    _owner_arguments(retirement)
    return parser


def _output(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worktree = getattr(args, "worktree", Path.cwd())
    try:
        root = _canonical_coordination_root(worktree, args.coordination_root)
        ledger = AgentGovernanceLedger(root)
        if args.command == "bootstrap":
            result = ledger.bootstrap()
        elif args.command == "inspect":
            result = ledger.inspect(args.task_id)
        elif args.command == "create":
            result = ledger.create(
                _load_json(args.packet),
                owner_session=args.owner_session,
                owner_token=args.owner_token,
                worktree=args.worktree,
                lease_seconds=args.lease_seconds,
            )
        elif args.command == "recover":
            result = ledger.recover_same_session(
                args.task_id,
                args.expected_owner_session,
                args.expected_revision,
                args.expected_record_sha256,
                args.worktree,
                args.reason,
                new_owner_token=args.new_owner_token,
                lease_seconds=args.lease_seconds,
            )
        elif args.command == "takeover":
            result = ledger.takeover_expired(
                args.task_id,
                args.expected_owner_session,
                args.expected_revision,
                args.expected_record_sha256,
                args.new_owner_session,
                args.new_worktree,
                args.reason,
                new_owner_token=args.new_owner_token,
                lease_seconds=args.lease_seconds,
            )
        elif args.command == "admin-takeover":
            result = ledger.administrative_takeover(
                args.task_id,
                args.confirm_task_id,
                args.confirmation,
                args.authorization_ref,
                args.expected_owner_session,
                args.expected_revision,
                args.expected_record_sha256,
                args.expected_worktree,
                args.new_owner_session,
                args.new_worktree,
                args.reason,
                new_owner_token=args.new_owner_token,
                lease_seconds=args.lease_seconds,
            )
        elif args.command == "rebaseline-worktree-fingerprint":
            result = ledger.rebaseline_worktree_fingerprint(
                args.task_id,
                args.confirm_task_id,
                args.confirmation,
                args.authorization_ref,
                args.evidence_ref,
                args.expected_revision,
                args.expected_record_sha256,
                args.expected_worktree,
                args.expected_stored_fingerprint,
                args.expected_current_fingerprint,
            )
        elif args.command == "rebind-worktree-head":
            result = ledger.rebind_worktree_head(
                args.task_id,
                args.confirm_task_id,
                args.confirmation,
                args.authorization_ref,
                args.expected_revision,
                args.expected_record_sha256,
                args.expected_worktree,
                args.expected_old_head,
                args.expected_new_head,
                args.expected_stored_fingerprint,
                args.expected_current_fingerprint,
            )
        elif args.command == "reconcile-completed-history":
            result = ledger.reconcile_completed_history(
                args.task_id,
                args.confirm_task_id,
                args.confirmation,
                args.authorization_ref,
                args.expected_revision,
                args.expected_record_sha256,
            )
        else:
            if not args.owner_token:
                raise GovernanceError("owner_token_missing", "Use the private owner token.")
            owner = {
                "expected_revision": args.expected_revision,
                "expected_record_sha256": args.expected_record_sha256,
                "token": args.owner_token,
                "worktree": args.worktree,
            }
            if args.command == "confirm-plan":
                result = ledger.confirm_plan(
                    args.task_id,
                    args.confirmation_ref,
                    args.actor,
                    **owner,
                )
            elif args.command == "record-skill-read":
                result = ledger.record_skill_read(
                    args.task_id,
                    args.skill_id,
                    args.skill_sha256,
                    args.step_id,
                    **owner,
                )
            elif args.command == "permit":
                result = ledger.permit(
                    args.task_id,
                    args.step_id,
                    args.effect,
                    ttl_seconds=args.ttl_seconds,
                    **owner,
                )
            elif args.command == "renew":
                result = ledger.renew(
                    args.task_id,
                    lease_seconds=args.lease_seconds,
                    **owner,
                )
            elif args.command == "renew-permit":
                result = ledger.renew_permit(
                    args.task_id,
                    args.permit_id,
                    ttl_seconds=args.ttl_seconds,
                    **owner,
                )
            elif args.command == "advance":
                evidence = _load_payload(args.evidence_packet).get("evidence", [])
                result = ledger.advance(
                    args.task_id,
                    args.permit_id,
                    evidence,
                    args.next_step_id,
                    terminal_status=args.terminal_status,
                    failed_gate=args.failed_gate,
                    next_action=args.next_action,
                    **owner,
                )
            elif args.command == "amend-plan":
                steps = _load_payload(args.plan_packet).get("steps", [])
                result = ledger.amend_plan(
                    args.task_id,
                    steps,
                    args.reason,
                    **owner,
                )
            elif args.command == "refresh-authority-receipts":
                result = ledger.refresh_authority_receipts(
                    args.task_id,
                    _load_payload(args.receipt_packet),
                    args.confirmation,
                    args.authorization_ref,
                    **owner,
                )
            elif args.command == "review-outcome":
                result = ledger.review_outcome(
                    args.task_id,
                    _load_payload(args.outcome_packet),
                    **owner,
                )
            elif args.command == "close":
                close_packet = _load_payload(args.learning_packet)
                learning = close_packet.get("learning", close_packet)
                result = ledger.close(
                    args.task_id,
                    learning,
                    args.retirement,
                    skill_maintenance=close_packet.get("skill_maintenance", []),
                    **owner,
                )
            elif args.command == "record-retirement":
                result = ledger.record_retirement(
                    args.task_id,
                    _load_payload(args.retirement_packet),
                    **owner,
                )
            else:
                raise GovernanceError("command_unimplemented", args.command)
        _output({"status": "success", "result": result})
        return 0
    except GovernanceError as exc:
        _output(
            {
                "status": "error",
                "summary": exc.code,
                "root_cause_hint": exc.hint,
                "safe_retry": "Inspect current state and satisfy the named gate.",
                "stop_condition": "Do not perform the requested effect while this gate fails.",
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

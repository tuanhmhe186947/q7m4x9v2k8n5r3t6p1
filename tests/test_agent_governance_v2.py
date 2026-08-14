from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / ".agents"
    / "skills"
    / "project-state-steward"
    / "scripts"
    / "manage_agent_governance.py"
)
TOKEN = "governance-owner-token-0123456789"
RUNTIME = "runtime-session-20260813"
NOW = datetime(2026, 8, 13, 11, 0, tzinfo=timezone.utc)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("CODEX_THREAD_ID", RUNTIME)
    spec = importlib.util.spec_from_file_location("agent_governance_v2_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    (root / ".agents" / "memory").mkdir(parents=True)
    (root / ".agents" / "skills" / "reasoning-a").mkdir(parents=True)
    (root / ".agents" / "skills" / "verification-a").mkdir(parents=True)
    authority = root / ".agents" / "memory" / "03_PROJECT_RULES.md"
    authority.write_text("# Rules\n\nCurrent authority.\n", encoding="utf-8")
    reasoning = root / ".agents" / "skills" / "reasoning-a" / "SKILL.md"
    verification = root / ".agents" / "skills" / "verification-a" / "SKILL.md"
    reasoning.write_text("---\nname: reasoning-a\n---\n", encoding="utf-8")
    verification.write_text("---\nname: verification-a\n---\n", encoding="utf-8")
    inventory = {
        "schema_version": "pig.skill-inventory.v1",
        "generated_views": [],
        "task_routes": {
            "governance_implementation": {
                "required_any": ["reasoning-a"],
                "reasoning_required": True,
            }
        },
        "skills": [
            {
                "skill_id": "reasoning-a",
                "status": "active",
                "implicit": False,
                "category": "reasoning",
                "source_root": "project",
                "relative_path": ".agents/skills/reasoning-a/SKILL.md",
                "depends_on": [],
            },
            {
                "skill_id": "verification-a",
                "status": "active",
                "implicit": True,
                "category": "verification",
                "source_root": "project",
                "relative_path": ".agents/skills/verification-a/SKILL.md",
                "depends_on": ["reasoning-a"],
            },
        ],
    }
    (root / ".agents" / "skills" / "skill_inventory.json").write_text(
        json.dumps(inventory),
        encoding="utf-8",
    )
    authority_index = {
        "schema_version": "pig.authority-index.v1",
        "entries": [
            {
                "scope": "memory.lifecycle",
                "current_authority": ".agents/memory/03_PROJECT_RULES.md",
            }
        ],
    }
    (root / ".agents" / "memory" / "18_AUTHORITY_INDEX.json").write_text(
        json.dumps(authority_index),
        encoding="utf-8",
    )
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    (root / ".gitignore").write_text(".agents/runtime/\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    return root


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection(project: Path, skill_id: str, implicit: bool = False) -> dict[str, str]:
    path = project / ".agents" / "skills" / skill_id / "SKILL.md"
    return {
        "skill_id": skill_id,
        "role": "reasoning" if skill_id == "reasoning-a" else "verification",
        "purpose": f"Use {skill_id} to enforce this bounded protocol.",
        "selection_mode": "implicit" if implicit else "explicit",
        "skill_sha256": sha256(path),
    }


def packet(project: Path) -> dict[str, object]:
    authority = project / ".agents" / "memory" / "03_PROJECT_RULES.md"
    return {
        "task_id": "REFORM-20260813-01",
        "title": "Enforce governance V2",
        "task_class": "governance_implementation",
        "risk_class": "high",
        "authorities": [
            {
                "scope": "memory.lifecycle",
                "locator": ".agents/memory/03_PROJECT_RULES.md",
                "selector": "FULL_FILE",
                "status": "CURRENT",
                "read_at": NOW.isoformat(),
                "sha256": sha256(authority),
                "section_sha256": sha256(authority),
            }
        ],
        "acceptance": [
            {"acceptance_id": "AC-1", "text": "Protocol is verified."},
            {"acceptance_id": "AC-2", "text": "Closeout is verified."},
        ],
        "risks": ["Concurrent main changes must remain untouched."],
        "non_actions": ["Do not delete legacy worktrees."],
        "skills": [selection(project, "reasoning-a")],
        "plan": {
            "steps": [
                {
                    "step_id": "S-1",
                    "summary": "Implement and verify protocol.",
                    "acceptance_ids": ["AC-1"],
                    "allowed_effects": ["edit", "test"],
                },
                {
                    "step_id": "S-2",
                    "summary": "Integrate and close protocol.",
                    "acceptance_ids": ["AC-2"],
                    "allowed_effects": ["integrate"],
                },
            ]
        },
    }


def create(
    manager: ModuleType,
    project: Path,
    value: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    ledger = manager.AgentGovernanceLedger(project)
    record = ledger.create(
        value or packet(project),
        owner_session=RUNTIME,
        owner_token=TOKEN,
        worktree=project,
        now=NOW,
    )
    return ledger, record


def owner(
    record: dict[str, object],
    project: Path,
    token: str = TOKEN,
) -> dict[str, object]:
    return {
        "expected_revision": record["revision"],
        "expected_record_sha256": record["record_sha256"],
        "token": token,
        "worktree": project,
    }


def rebaseline_arguments(
    manager: ModuleType,
    record: dict[str, object],
    project: Path,
) -> dict[str, object]:
    identity = manager._worktree_identity(project)
    return {
        "task_id": "REFORM-20260813-01",
        "confirm_task_id": "REFORM-20260813-01",
        "confirmation": "USER_AUTHORIZED_WORKTREE_FINGERPRINT_REBASELINE",
        "authorization_ref": "user-request:fixture-rebaseline",
        "evidence_ref": "tests:test_agent_governance_v2",
        "expected_revision": record["revision"],
        "expected_record_sha256": record["record_sha256"],
        "expected_worktree": project,
        "expected_stored_fingerprint": record["worktree"]["fingerprint"],
        "expected_current_fingerprint": identity["fingerprint"],
    }


def confirm(
    ledger: object,
    record: dict[str, object],
    project: Path,
) -> dict[str, object]:
    for selected in record["skills"]:
        record = ledger.record_skill_read(
            record["task_id"],
            selected["skill_id"],
            selected["skill_sha256"],
            [step["step_id"] for step in record["plan"]["steps"]],
            now=NOW,
            **owner(record, project),
        )
    return ledger.confirm_plan(
        str(record["task_id"]),
        "visible-plan-message-1",
        "agent",
        now=NOW,
        **owner(record, project),
    )


def pass_evidence(
    acceptance_id: str,
    evidence_id: str = "EV-1",
) -> list[dict[str, object]]:
    return [
        {
            "evidence_id": evidence_id,
            "kind": "command_result",
            "locator": f"pytest:{evidence_id}",
            "supports": [acceptance_id],
            "status": "PASS",
        }
    ]


def integration_packet(project: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "target_ref": "HEAD",
        "target_sha": head,
        "integrated_sha": head,
        "proof_kind": "ANCESTOR",
        "revalidation_evidence": [
            {
                "evidence_id": "EV-INTEGRATED-PYTEST",
                "kind": "command_result",
                "locator": "pytest:test_agent_governance_v2",
                "status": "PASS",
                "target_sha": head,
            }
        ],
    }


def no_skill_impact() -> list[dict[str, object]]:
    return [
        {
            "status": "NO_SKILL_IMPACT",
            "rationale": "No selected skill failed or changed during this fixture.",
            "evidence": ["pytest:test_agent_governance_v2"],
        }
    ]


def finish_steps(
    ledger: object,
    record: dict[str, object],
    project: Path,
) -> dict[str, object]:
    record = ledger.permit(
        "REFORM-20260813-01",
        "S-1",
        ["test"],
        now=NOW,
        **owner(record, project),
    )
    record = ledger.advance(
        "REFORM-20260813-01",
        record["active_permit"]["permit_id"],
        pass_evidence("AC-1"),
        "S-2",
        now=NOW,
        **owner(record, project),
    )
    record = ledger.permit(
        "REFORM-20260813-01",
        "S-2",
        ["integrate"],
        now=NOW,
        **owner(record, project),
    )
    return ledger.advance(
        "REFORM-20260813-01",
        record["active_permit"]["permit_id"],
        pass_evidence("AC-2", "EV-2"),
        None,
        now=NOW,
        **owner(record, project),
    )


def test_create_requires_authority_receipt(manager: ModuleType, project: Path) -> None:
    value = packet(project)
    value["authorities"] = []
    with pytest.raises(manager.GovernanceError, match="authority_receipts_missing"):
        create(manager, project, value)


def test_create_rejects_v1_task_id_collision(
    manager: ModuleType,
    project: Path,
) -> None:
    short = project / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md"
    short.write_text(
        "## Active Task Checklist\n\n"
        "### REFORM-20260813-01 - Existing V1 capsule\n\n"
        "- Status: `IN_PROGRESS`.\n",
        encoding="utf-8",
    )
    with pytest.raises(manager.GovernanceError, match="v1_v2_task_id_collision"):
        create(manager, project)


def test_create_rejects_stale_authority_hash(manager: ModuleType, project: Path) -> None:
    value = packet(project)
    value["authorities"][0]["sha256"] = "0" * 64
    with pytest.raises(manager.GovernanceError, match="authority_hash_mismatch"):
        create(manager, project, value)


def test_create_rejects_stale_authority_section_hash(
    manager: ModuleType,
    project: Path,
) -> None:
    value = packet(project)
    value["authorities"][0]["section_sha256"] = "0" * 64
    with pytest.raises(manager.GovernanceError, match="authority_section_hash_mismatch"):
        create(manager, project, value)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value["skills"][0].update(skill_id="unknown"), "skill_unknown"),
        (
            lambda value: value["skills"][0].update(purpose=""),
            "skill_purpose_invalid",
        ),
        (
            lambda value: value["skills"][0].update(skill_sha256="0" * 64),
            "skill_hash_mismatch",
        ),
    ],
)
def test_create_rejects_invalid_skill_selection(
    manager: ModuleType,
    project: Path,
    mutation: object,
    code: str,
) -> None:
    value = packet(project)
    mutation(value)
    with pytest.raises(manager.GovernanceError, match=code):
        create(manager, project, value)


def test_create_rejects_missing_reasoning_route(manager: ModuleType, project: Path) -> None:
    value = packet(project)
    value["skills"] = [selection(project, "verification-a", implicit=True)]
    with pytest.raises(manager.GovernanceError, match="reasoning_route_missing"):
        create(manager, project, value)


def test_create_rejects_missing_skill_dependency(manager: ModuleType, project: Path) -> None:
    value = packet(project)
    value["task_class"] = "unrouted"
    value["skills"] = [selection(project, "verification-a", implicit=True)]
    with pytest.raises(manager.GovernanceError, match="skill_dependency_missing"):
        create(manager, project, value)


def test_create_rejects_unknown_task_class(manager: ModuleType, project: Path) -> None:
    value = packet(project)
    value["task_class"] = "unknown_task_class"
    with pytest.raises(manager.GovernanceError, match="task_class_unrouted"):
        create(manager, project, value)


def test_inventory_accepts_typed_non_reasoning_evaluation_route(
    manager: ModuleType,
    project: Path,
) -> None:
    inventory_path = project / ".agents" / "skills" / "skill_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["task_routes"]["agent_task_evaluation"] = {
        "required_all": ["verification-a"],
        "reasoning_required": False,
    }
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    assert manager.validate_skill_inventory(project) == []


def test_second_active_task_cannot_claim_same_worktree(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, _ = create(manager, project)
    value = packet(project)
    value["task_id"] = "REFORM-20260813-02"
    with pytest.raises(manager.GovernanceError, match="worktree_already_admitted"):
        ledger.create(
            value,
            owner_session=RUNTIME,
            owner_token=TOKEN,
            worktree=project,
            now=NOW,
        )


def test_permit_requires_plan_confirmation(manager: ModuleType, project: Path) -> None:
    ledger, record = create(manager, project)
    with pytest.raises(manager.GovernanceError, match="task_not_confirmed"):
        ledger.permit(
            "REFORM-20260813-01",
            "S-1",
            ["test"],
            now=NOW,
            **owner(record, project),
        )


def test_high_risk_effect_requires_user_confirmation(
    manager: ModuleType,
    project: Path,
) -> None:
    value = packet(project)
    value["plan"]["steps"][0]["allowed_effects"] = ["delete"]
    ledger, record = create(manager, project, value)
    with pytest.raises(manager.GovernanceError, match="user_confirmation_required"):
        confirm(ledger, record, project)


def test_high_risk_effect_accepts_explicit_user_confirmation(
    manager: ModuleType,
    project: Path,
) -> None:
    value = packet(project)
    value["plan"]["steps"][0]["allowed_effects"] = ["delete"]
    ledger, record = create(manager, project, value)
    record = ledger.confirm_plan(
        "REFORM-20260813-01",
        "user-message-2026-08-13-delete-exact-target",
        "user",
        now=NOW,
        **owner(record, project),
    )
    assert record["state"] == "CONFIRMED"


def test_permit_rejects_effect_outside_plan(manager: ModuleType, project: Path) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    with pytest.raises(manager.GovernanceError, match="permit_effect_outside_plan"):
        ledger.permit(
            "REFORM-20260813-01",
            "S-1",
            ["publish"],
            now=NOW,
            **owner(record, project),
        )


def test_permit_requires_skill_read_receipt(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = ledger.confirm_plan(
        "REFORM-20260813-01",
        "plan-confirmed-without-skill-read",
        "agent",
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="skill_read_receipt_missing"):
        ledger.permit(
            "REFORM-20260813-01",
            "S-1",
            ["test"],
            now=NOW,
            **owner(record, project),
        )


def test_permit_rejects_generated_view_drift(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    inventory_path = project / ".agents" / "skills" / "skill_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["view_contract"] = {"fixture": "modern"}
    inventory["generated_views"] = [".agents/skills/skill_registry.json"]
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="skill_inventory_invalid"):
        ledger.permit(
            "REFORM-20260813-01",
            "S-1",
            ["test"],
            now=NOW,
            **owner(record, project),
        )


def test_record_skill_read_rejects_stale_hash(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    selected = record["skills"][0]
    with pytest.raises(manager.GovernanceError, match="skill_read_hash_mismatch"):
        ledger.record_skill_read(
            "REFORM-20260813-01",
            selected["skill_id"],
            "0" * 64,
            ["S-1"],
            now=NOW,
            **owner(record, project),
        )


def test_advance_refreshes_worktree_snapshot_after_effect(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    original_fingerprint = record["worktree"]["fingerprint"]
    record = ledger.permit(
        "REFORM-20260813-01",
        "S-1",
        ["test"],
        now=NOW,
        **owner(record, project),
    )
    changed = project / "after-effect.txt"
    changed.write_text("effect output\n", encoding="utf-8")
    record = ledger.advance(
        "REFORM-20260813-01",
        record["active_permit"]["permit_id"],
        pass_evidence("AC-1", "EV-SNAPSHOT"),
        "S-2",
        now=NOW,
        **owner(record, project),
    )
    assert record["worktree"]["fingerprint"] != original_fingerprint
    assert "after-effect.txt" in record["worktree"]["dirty_paths"]
    assert record["events"][-1]["event_type"] == "STEP_ADVANCED"
    assert record["events"][-1]["payload"]["worktree_snapshot"]["fingerprint"] == (
        record["worktree"]["fingerprint"]
    )
    next_record = ledger.permit(
        "REFORM-20260813-01",
        "S-2",
        ["integrate"],
        now=NOW,
        **owner(record, project),
    )
    assert next_record["active_permit"]["step_id"] == "S-2"


def test_rebaseline_refreshes_only_worktree_snapshot(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    before = json.loads(json.dumps(record))
    (project / "task-owned-drift.txt").write_text("drift\n", encoding="utf-8")
    rebased = ledger.rebaseline_worktree_fingerprint(
        now=NOW,
        **rebaseline_arguments(manager, record, project),
    )
    assert rebased["revision"] == before["revision"] + 1
    assert rebased["state"] == before["state"]
    assert rebased["plan"] == before["plan"]
    assert rebased["owner"] == before["owner"]
    assert rebased["active_permit"] == before["active_permit"]
    assert rebased["events"][:-1] == before["events"]
    assert rebased["worktree"]["fingerprint"] != before["worktree"]["fingerprint"]
    assert "task-owned-drift.txt" in rebased["worktree"]["dirty_paths"]
    event = rebased["events"][-1]
    assert event["event_type"] == "WORKTREE_FINGERPRINT_REBASELINED"
    assert event["payload"]["operation"] == "rebaseline-worktree-fingerprint"
    assert event["payload"]["prior_revision"] == before["revision"]
    assert event["payload"]["resulting_revision"] == rebased["revision"]


def test_rebaseline_rejects_stale_revision(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    arguments = rebaseline_arguments(manager, record, project)
    arguments["expected_revision"] = record["revision"] - 1
    with pytest.raises(manager.GovernanceError, match="revision_conflict"):
        ledger.rebaseline_worktree_fingerprint(now=NOW, **arguments)


def test_rebaseline_rejects_stale_stored_fingerprint(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    arguments = rebaseline_arguments(manager, record, project)
    arguments["expected_stored_fingerprint"] = "0" * 64
    with pytest.raises(
        manager.GovernanceError,
        match="rebaseline_stored_fingerprint_mismatch",
    ):
        ledger.rebaseline_worktree_fingerprint(now=NOW, **arguments)


def test_rebaseline_rejects_wrong_or_stale_current_fingerprint(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    wrong = rebaseline_arguments(manager, record, project)
    wrong["expected_current_fingerprint"] = "0" * 64
    with pytest.raises(
        manager.GovernanceError,
        match="rebaseline_current_fingerprint_mismatch",
    ):
        ledger.rebaseline_worktree_fingerprint(now=NOW, **wrong)
    stale = rebaseline_arguments(manager, record, project)
    (project / "mutated-after-capture.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(
        manager.GovernanceError,
        match="rebaseline_current_fingerprint_mismatch",
    ):
        ledger.rebaseline_worktree_fingerprint(now=NOW, **stale)


def test_rebaseline_rejects_active_valid_permit(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = ledger.permit(
        "REFORM-20260813-01",
        "S-1",
        ["test"],
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="rebaseline_active_permit"):
        ledger.rebaseline_worktree_fingerprint(
            now=NOW,
            **rebaseline_arguments(manager, record, project),
        )


def test_permit_renewal_preserves_scope_and_rejects_expiry(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = ledger.permit(
        "REFORM-20260813-01",
        "S-1",
        ["test"],
        now=NOW,
        **owner(record, project),
    )
    permit = dict(record["active_permit"])
    record = ledger.renew(
        "REFORM-20260813-01",
        lease_seconds=86400,
        now=NOW.replace(minute=10),
        **owner(record, project),
    )
    renewed = ledger.renew_permit(
        "REFORM-20260813-01",
        permit["permit_id"],
        ttl_seconds=21600,
        now=NOW.replace(minute=10),
        **owner(record, project),
    )
    assert renewed["active_permit"]["permit_id"] == permit["permit_id"]
    assert renewed["active_permit"]["effects"] == permit["effects"]
    assert renewed["active_permit"]["step_id"] == permit["step_id"]
    assert renewed["events"][-1]["event_type"] == "ACTION_PERMIT_RENEWED"
    expired_at = datetime.fromisoformat(renewed["active_permit"]["expires_at"])
    with pytest.raises(manager.GovernanceError, match="permit_expired"):
        ledger.renew_permit(
            "REFORM-20260813-01",
            permit["permit_id"],
            now=expired_at.replace(second=expired_at.second + 1),
            **owner(renewed, project),
        )


def test_cas_rejects_stale_record(manager: ModuleType, project: Path) -> None:
    ledger, record = create(manager, project)
    first = confirm(ledger, record, project)
    with pytest.raises(manager.GovernanceError, match="revision_conflict"):
        ledger.confirm_plan(
            "REFORM-20260813-01",
            "stale-confirmation",
            "agent",
            now=NOW,
            **owner(record, project),
        )
    assert first["revision"] == 3


def test_advance_requires_complete_acceptance_evidence(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = ledger.permit(
        "REFORM-20260813-01",
        "S-1",
        ["test"],
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="acceptance_evidence_missing"):
        ledger.advance(
            "REFORM-20260813-01",
            record["active_permit"]["permit_id"],
            [
                {
                    "evidence_id": "EV-FAIL",
                    "kind": "command_result",
                    "locator": "pytest:failed",
                    "supports": ["AC-1"],
                    "status": "FAIL",
                }
            ],
            "S-2",
            now=NOW,
            **owner(record, project),
        )


def test_advance_rejects_missing_or_hash_mismatched_file_evidence(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = ledger.permit(
        "REFORM-20260813-01",
        "S-1",
        ["test"],
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="evidence_hash_mismatch"):
        ledger.advance(
            "REFORM-20260813-01",
            record["active_permit"]["permit_id"],
            [
                {
                    "evidence_id": "EV-FILE",
                    "kind": "artifact",
                    "locator": "missing.json",
                    "sha256": "0" * 64,
                    "supports": ["AC-1"],
                    "status": "PASS",
                }
            ],
            "S-2",
            now=NOW,
            **owner(record, project),
        )


def test_terminal_plan_requires_amendment_before_new_effect(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    with pytest.raises(manager.GovernanceError, match="permit_step_invalid"):
        ledger.permit(
            "REFORM-20260813-01",
            "S-2",
            ["integrate"],
            now=NOW,
            **owner(record, project),
        )
    amended = ledger.amend_plan(
        "REFORM-20260813-01",
        packet(project)["plan"]["steps"],
        "Evidence requires a bounded rerun.",
        now=NOW,
        **owner(record, project),
    )
    assert amended["state"] == "PLANNED"
    assert amended["plan_confirmation"] is None


def test_authority_receipt_refresh_requires_exact_confirmation(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    with pytest.raises(manager.GovernanceError, match="authority_refresh_confirmation_missing"):
        ledger.refresh_authority_receipts(
            "REFORM-20260813-01",
            {"authorities": record["authorities"]},
            "WRONG_CONFIRMATION",
            "user-request:test",
            now=NOW,
            **owner(record, project),
        )


def test_authority_receipt_refresh_rejects_stale_or_changed_scope(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    stale = [dict(record["authorities"][0])]
    stale[0]["sha256"] = "0" * 64
    stale[0]["section_sha256"] = "0" * 64
    with pytest.raises(manager.GovernanceError, match="authority_hash_mismatch"):
        ledger.refresh_authority_receipts(
            "REFORM-20260813-01",
            {"authorities": stale},
            "ALLOW_EXPLICIT_AUTHORITY_RECEIPT_REFRESH",
            "user-request:test",
            now=NOW,
            **owner(record, project),
        )
    changed = [dict(record["authorities"][0])]
    changed[0]["locator"] = "README.md"
    with pytest.raises(manager.GovernanceError, match="authority_refresh_scope_change_forbidden"):
        ledger.refresh_authority_receipts(
            "REFORM-20260813-01",
            {"authorities": changed},
            "ALLOW_EXPLICIT_AUTHORITY_RECEIPT_REFRESH",
            "user-request:test",
            now=NOW,
            **owner(record, project),
        )


def test_authority_receipt_refresh_updates_only_receipts_and_digest(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    before_plan = json.loads(json.dumps(record["plan"]))
    before_skills = json.loads(json.dumps(record["skills"]))
    authority = project / ".agents" / "memory" / "03_PROJECT_RULES.md"
    authority.write_text("# Rules\n\nRefreshed authority.\n", encoding="utf-8")
    refreshed_receipt = dict(record["authorities"][0])
    refreshed_receipt["sha256"] = sha256(authority)
    refreshed_receipt["section_sha256"] = sha256(authority)
    refreshed = ledger.refresh_authority_receipts(
        "REFORM-20260813-01",
        {"authorities": [refreshed_receipt]},
        "ALLOW_EXPLICIT_AUTHORITY_RECEIPT_REFRESH",
        "user-request:test-authority-refresh",
        now=NOW,
        **owner(record, project),
    )
    assert refreshed["authorities"] == [refreshed_receipt]
    assert refreshed["authority_digest"] != record["authority_digest"]
    assert refreshed["plan"] == before_plan
    assert refreshed["skills"] == before_skills
    assert refreshed["active_permit"] is None
    event = refreshed["events"][-1]
    assert event["event_type"] == "AUTHORITY_RECEIPTS_REFRESHED"
    assert event["payload"]["old_authority_digest"] == record["authority_digest"]
    assert event["payload"]["new_authority_digest"] == refreshed["authority_digest"]


def test_authority_receipt_refresh_rejects_active_permit(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = ledger.permit(
        "REFORM-20260813-01",
        "S-1",
        ["test"],
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="authority_refresh_active_permit"):
        ledger.refresh_authority_receipts(
            "REFORM-20260813-01",
            {"authorities": record["authorities"]},
            "ALLOW_EXPLICIT_AUTHORITY_RECEIPT_REFRESH",
            "user-request:test-active-permit",
            now=NOW,
            **owner(record, project),
        )


def test_amend_plan_does_not_refresh_authority_receipts(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    authority = project / ".agents" / "memory" / "03_PROJECT_RULES.md"
    authority.write_text("# Rules\n\nDrifted authority.\n", encoding="utf-8")
    amended = ledger.amend_plan(
        "REFORM-20260813-01",
        packet(project)["plan"]["steps"],
        "Plan-only amendment.",
        now=NOW,
        **owner(record, project),
    )
    assert amended["authorities"] == record["authorities"]
    assert amended["authority_digest"] == record["authority_digest"]


def test_outcome_review_requires_every_dirty_path_disposition(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    (project / "scratch.txt").write_text("unique\n", encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="dirty_path_undispositioned"):
        ledger.review_outcome(
            "REFORM-20260813-01",
            {"outcome": "REJECTED", "path_dispositions": []},
            now=NOW,
            **owner(record, project),
        )


def test_close_rejects_apology_as_learning(manager: ModuleType, project: Path) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {"outcome": "REJECTED", "path_dispositions": []},
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="learning_disposition_invalid"):
        ledger.close(
            "REFORM-20260813-01",
            {"disposition": "SORRY", "message": "I will do better."},
            "RETIRE_ELIGIBLE",
            now=NOW,
            **owner(record, project),
        )


def test_close_rejects_accepted_result_without_main_revalidation(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {"outcome": "ACCEPTED", "path_dispositions": [], "integration": {}},
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="integration_revalidation_missing"):
        ledger.close(
            "REFORM-20260813-01",
            {
                "disposition": "NO_DURABLE_LESSON",
                "rationale": "Expected implementation completed without a correction.",
                "evidence": ["pytest:pass"],
            },
            "RETIRE_ELIGIBLE",
            now=NOW,
            **owner(record, project),
        )


def test_close_rejects_failure_without_extracted_evidence(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {"outcome": "REJECTED", "path_dispositions": []},
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="failure_evidence_not_extracted"):
        ledger.close(
            "REFORM-20260813-01",
            {
                "disposition": "UNVERIFIED_FAILURE",
                "observations": ["The approach did not meet acceptance."],
                "evidence": ["pytest:fail"],
                "hypotheses": ["The contract may be incomplete."],
                "preserved_location": "artifact://failure",
                "next_validation": "Run the focused negative control.",
            },
            "RETIRE_ELIGIBLE",
            now=NOW,
            **owner(record, project),
        )


def test_close_accepts_complete_validated_correction(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {
            "outcome": "ACCEPTED",
            "path_dispositions": [],
            "integration": integration_packet(project),
        },
        now=NOW,
        **owner(record, project),
    )
    record = ledger.close(
        "REFORM-20260813-01",
        {
            "disposition": "VALIDATED_CORRECTION",
            "root_cause": "Prose rules were not executable gates.",
            "correction": "Use typed CAS transitions and closeout gates.",
            "validation_evidence": ["pytest:test_agent_governance_v2"],
            "reuse_when": "A material agent task produces effects.",
            "do_not_reuse_when": "The request is simple and read-only.",
        },
        "RETIRE_ELIGIBLE",
        skill_maintenance=no_skill_impact(),
        now=NOW,
        **owner(record, project),
    )
    assert record["state"] == "CLOSED"
    assert record["worktree"]["retirement"] == "RETIRE_ELIGIBLE"


def test_required_target_artifact_requires_hardlink_parity(
    manager: ModuleType,
    project: Path,
) -> None:
    source = project / "source.bin"
    target = project / "target.bin"
    source.write_bytes(b"canonical-payload")
    value = packet(project)
    value["required_target_artifacts"] = [
        {
            "path": target.name,
            "source_path": source.name,
            "sha256": sha256(source),
            "size_bytes": source.stat().st_size,
            "hardlink_required": True,
        }
    ]
    ledger, record = create(manager, project, value)
    with pytest.raises(manager.GovernanceError, match="target_artifact_missing"):
        ledger._validate_required_target_artifacts(record)
    os.link(source, target)
    ledger._validate_required_target_artifacts(record)


def test_close_rejects_missing_outcome_declared_target_artifact(
    manager: ModuleType,
    project: Path,
) -> None:
    source = project / "source.bin"
    source.write_bytes(b"canonical-payload")
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {
            "outcome": "ACCEPTED",
            "path_dispositions": [
                {"path": source.name, "disposition": "PRESERVE_USER_OWNED"}
            ],
            "integration": integration_packet(project),
            "required_target_artifacts": [
                {
                    "path": "missing-target.bin",
                    "source_path": source.name,
                    "sha256": sha256(source),
                    "size_bytes": source.stat().st_size,
                    "hardlink_required": True,
                }
            ],
        },
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="target_artifact_missing"):
        ledger.close(
            "REFORM-20260813-01",
            {
                "disposition": "NO_DURABLE_LESSON",
                "rationale": "The outcome target must exist before acceptance.",
                "evidence": ["target-artifact-negative-control"],
            },
            "RETIRE_ELIGIBLE",
            now=NOW,
            **owner(record, project),
        )


def test_unknown_outcome_cannot_be_retired(manager: ModuleType, project: Path) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {"outcome": "UNKNOWN", "path_dispositions": []},
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(
        manager.GovernanceError,
        match="unknown_worktree_must_remain_protected",
    ):
        ledger.close(
            "REFORM-20260813-01",
            {
                "disposition": "NO_DURABLE_LESSON",
                "rationale": "The state is unknown and remains protected.",
                "evidence": ["inventory:unknown"],
            },
            "RETIRE_ELIGIBLE",
            now=NOW,
            **owner(record, project),
        )


def test_close_records_skill_maintenance_due(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {
            "outcome": "ACCEPTED",
            "path_dispositions": [],
            "integration": integration_packet(project),
        },
        now=NOW,
        **owner(record, project),
    )
    record = ledger.close(
        "REFORM-20260813-01",
        {
            "disposition": "NO_DURABLE_LESSON",
            "rationale": "The task exposed a skill maintenance concern only.",
            "evidence": ["validator:failure"],
        },
        "RETIRE_ELIGIBLE",
        skill_maintenance=[
            {
                "skill_id": "reasoning-a",
                "status": "MAINTENANCE_DUE",
                "trigger": "Validator failure exposed an over-broad route rule.",
                "evidence": ["validator:skill_inventory_route"],
                "next_action": "Review the typed route before the next use.",
            }
        ],
        now=NOW,
        **owner(record, project),
    )
    assert record["skill_maintenance"][0]["status"] == "MAINTENANCE_DUE"


def test_record_hash_and_event_chain_detect_tampering(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    path = project / manager.RUNTIME_RELATIVE / "tasks" / "REFORM-20260813-01.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["title"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="record_hash_mismatch"):
        ledger.inspect("REFORM-20260813-01")
    shutil.rmtree(project / manager.RUNTIME_RELATIVE)
    _, record = create(manager, project)
    path = project / manager.RUNTIME_RELATIVE / "tasks" / "REFORM-20260813-01.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["events"][0]["payload"]["plan_digest"] = "0" * 64
    payload["record_sha256"] = "0" * 64
    payload["record_sha256"] = manager._record_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="event_chain_invalid"):
        ledger.inspect("REFORM-20260813-01")


def test_bootstrap_is_bounded_and_lists_active_task(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, _ = create(manager, project)
    result = ledger.bootstrap()
    assert result["active_tasks"] == [
        {
            "task_id": "REFORM-20260813-01",
            "state": "PLANNED",
            "active_step": "S-1",
            "worktree": str(project.resolve()),
            "next_action": (
                "inspect task, retrieve listed authorities, and obey permit gate"
            ),
        }
    ]
    assert len(json.dumps(result)) < 2000


def test_worktree_lifecycle_validator_rejects_unsafe_retirement(
    manager: ModuleType,
    project: Path,
) -> None:
    path = project / manager.WORKTREE_LEDGER_RELATIVE
    path.write_text(
        json.dumps(
            {
                "schema_version": "pig.worktree-lifecycle-ledger.v1",
                "worktrees": [
                    {
                        "worktree_id": "unknown-owner",
                        "state": "RETIRE_ELIGIBLE",
                        "outcome": "UNKNOWN",
                        "dirty_paths": [
                            {"path": "unique.bin", "disposition": "UNKNOWN_HALT"}
                        ],
                        "retirement_authorized": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    errors = manager.validate_worktree_lifecycle_ledger(project)
    assert "worktree_unknown_not_protected:unknown-owner" in errors


@pytest.mark.parametrize(
    ("drift_kind", "expected_code"),
    [
        ("authority", "authority_hash_mismatch"),
        ("skill", "skill_hash_mismatch"),
        ("worktree", "worktree_fingerprint_drift"),
    ],
)
def test_permit_revalidates_authority_skill_and_worktree_drift(
    manager: ModuleType,
    project: Path,
    drift_kind: str,
    expected_code: str,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    if drift_kind == "authority":
        path = project / ".agents" / "memory" / "03_PROJECT_RULES.md"
        path.write_text("# Rules\n\nDrifted authority.\n", encoding="utf-8")
    elif drift_kind == "skill":
        path = project / ".agents" / "skills" / "reasoning-a" / "SKILL.md"
        path.write_text("---\nname: reasoning-a\nchanged: true\n---\n", encoding="utf-8")
    else:
        (project / "unplanned-drift.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match=expected_code):
        ledger.permit(
            "REFORM-20260813-01",
            "S-1",
            ["test"],
            now=NOW,
            **owner(record, project),
        )


def test_permit_rejects_worktree_head_drift(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    (project / "head-drift.txt").write_text("committed drift\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(project), "add", "head-drift.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "commit", "-q", "-m", "head drift"],
        check=True,
    )
    with pytest.raises(manager.GovernanceError, match="worktree_head_drift"):
        ledger.permit(
            "REFORM-20260813-01",
            "S-1",
            ["test"],
            now=NOW,
            **owner(record, project),
        )


def test_create_rejects_cross_repository_worktree(
    manager: ModuleType,
    project: Path,
    tmp_path: Path,
) -> None:
    other = tmp_path / "other-repository"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    subprocess.run(["git", "-C", str(other), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(other), "config", "user.email", "test@example.com"],
        check=True,
    )
    (other / "README.md").write_text("other\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(other), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(other), "commit", "-q", "-m", "other fixture"],
        check=True,
    )
    ledger = manager.AgentGovernanceLedger(project)
    with pytest.raises(manager.GovernanceError, match="worktree_common_root_mismatch"):
        ledger.create(
            packet(project),
            owner_session=RUNTIME,
            owner_token=TOKEN,
            worktree=other,
            now=NOW,
        )


def test_shared_main_allows_disjoint_scopes(
    manager: ModuleType,
    project: Path,
) -> None:
    first = packet(project)
    first.update(task_id="SHARED-DISJOINT-01", worktree_mode="shared_main")
    first["path_scope"] = ["src/alpha"]
    ledger, first_record = create(manager, project, first)
    second = packet(project)
    second.update(task_id="SHARED-DISJOINT-02", worktree_mode="shared_main")
    second["path_scope"] = ["src/beta"]
    second_record = ledger.create(
        second,
        owner_session=RUNTIME,
        owner_token="shared-owner-token-02",
        worktree=project,
        now=NOW,
    )
    assert first_record["worktree"]["mode"] == "shared_main"
    assert second_record["worktree"]["path_scope"] == ["src/beta"]


def test_shared_main_rejects_overlapping_scopes(
    manager: ModuleType,
    project: Path,
) -> None:
    first = packet(project)
    first.update(task_id="SHARED-OVERLAP-01", worktree_mode="shared_main")
    first["path_scope"] = ["src"]
    ledger, _ = create(manager, project, first)
    second = packet(project)
    second.update(task_id="SHARED-OVERLAP-02", worktree_mode="shared_main")
    second["path_scope"] = ["src/components"]
    with pytest.raises(manager.GovernanceError, match="shared_main_scope_overlap"):
        ledger.create(
            second,
            owner_session=RUNTIME,
            owner_token="shared-owner-token-02",
            worktree=project,
            now=NOW,
        )


def test_same_session_recovery_rotates_token_and_rejects_old_token(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger = manager.AgentGovernanceLedger(project)
    record = ledger.create(
        packet(project),
        owner_session=RUNTIME,
        owner_token=TOKEN,
        worktree=project,
        lease_seconds=1,
        now=NOW,
    )
    new_token = "recovered-owner-token-012345"
    recovered = ledger.recover_same_session(
        "REFORM-20260813-01",
        expected_owner_session=RUNTIME,
        expected_revision=record["revision"],
        expected_record_sha256=record["record_sha256"],
        worktree=project,
        reason="Resume after a lost in-memory token.",
        new_owner_token=new_token,
        now=NOW.replace(second=2),
    )
    with pytest.raises(manager.GovernanceError, match="owner_token_mismatch"):
        ledger.confirm_plan(
            "REFORM-20260813-01",
            "stale-token-confirmation",
            "agent",
            now=NOW.replace(second=2),
            **owner(recovered, project),
        )
    confirmed = ledger.confirm_plan(
        "REFORM-20260813-01",
        "recovered-token-confirmation",
        "agent",
        now=NOW.replace(second=2),
        **owner(recovered, project, token=new_token),
    )
    assert confirmed["state"] == "CONFIRMED"


def test_expired_takeover_requires_expiry_and_rejects_old_token(
    manager: ModuleType,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = manager.AgentGovernanceLedger(project)
    record = ledger.create(
        packet(project),
        owner_session=RUNTIME,
        owner_token=TOKEN,
        worktree=project,
        lease_seconds=1,
        now=NOW,
    )
    new_runtime = "runtime-session-2"
    monkeypatch.setenv("CODEX_THREAD_ID", new_runtime)
    with pytest.raises(manager.GovernanceError, match="takeover_lease_active"):
        ledger.takeover_expired(
            "REFORM-20260813-01",
            expected_owner_session=RUNTIME,
            expected_revision=record["revision"],
            expected_record_sha256=record["record_sha256"],
            new_owner_session="runtime-session-2",
            new_worktree=project,
            reason="Premature takeover must halt.",
            new_owner_token="takeover-owner-token-012345",
            now=NOW,
        )
    new_token = "takeover-owner-token-012345"
    taken = ledger.takeover_expired(
        "REFORM-20260813-01",
        expected_owner_session=RUNTIME,
        expected_revision=record["revision"],
        expected_record_sha256=record["record_sha256"],
        new_owner_session=new_runtime,
        new_worktree=project,
        reason="Original lease expired and ownership is explicitly transferred.",
        new_owner_token=new_token,
        now=NOW.replace(second=2),
    )
    assert taken["owner"]["session"] == new_runtime
    with pytest.raises(manager.GovernanceError, match="owner_token_mismatch"):
        ledger.confirm_plan(
            "REFORM-20260813-01",
            "old-owner-confirmation",
            "agent",
            now=NOW.replace(second=2),
            **owner(taken, project),
        )
    confirmed = ledger.confirm_plan(
        "REFORM-20260813-01",
        "new-owner-confirmation",
        "agent",
        now=NOW.replace(second=2),
        **owner(taken, project, token=new_token),
    )
    assert confirmed["state"] == "CONFIRMED"


def test_permit_cannot_bypass_todo_step(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    with pytest.raises(manager.GovernanceError, match="permit_step_invalid"):
        ledger.permit(
            "REFORM-20260813-01",
            "S-2",
            ["integrate"],
            now=NOW,
            **owner(record, project),
        )


@pytest.mark.parametrize("terminal_status", ["BLOCKED", "CANCELLED"])
def test_advance_records_blocked_or_cancelled_step_with_context(
    manager: ModuleType,
    project: Path,
    terminal_status: str,
) -> None:
    value = packet(project)
    value["task_id"] = f"TERMINAL-{terminal_status.lower()}"
    value["acceptance"] = [value["acceptance"][0]]
    value["plan"] = {"steps": [value["plan"]["steps"][0]]}
    ledger, record = create(manager, project, value)
    record = confirm(ledger, record, project)
    record = ledger.permit(
        value["task_id"],
        "S-1",
        ["test"],
        now=NOW,
        **owner(record, project),
    )
    evidence = [
        {
            "evidence_id": f"EV-{terminal_status}",
            "kind": "command_result",
            "locator": f"governance:{terminal_status.lower()}",
            "supports": ["AC-1"],
            "status": "OBSERVED",
        }
    ]
    record = ledger.advance(
        value["task_id"],
        record["active_permit"]["permit_id"],
        evidence,
        None,
        terminal_status=terminal_status,
        failed_gate="terminal gate deliberately exercised",
        next_action="Record the terminal disposition and keep the worktree protected.",
        now=NOW,
        **owner(record, project),
    )
    assert record["plan"]["steps"][0]["status"] == terminal_status
    assert record["active_permit"] is None


def _rejected_review_with_extracted_file(
    ledger: object,
    record: dict[str, object],
    project: Path,
    artifact_name: str = "failure-evidence.json",
) -> tuple[dict[str, object], Path, str]:
    artifact = project / artifact_name
    artifact.write_text("preserved diagnostic\n", encoding="utf-8")
    artifact_hash = sha256(artifact)
    reviewed = ledger.review_outcome(
        "REFORM-20260813-01",
        {
            "outcome": "REJECTED",
            "path_dispositions": [
                {
                    "path": artifact_name,
                    "disposition": "EXTRACT_EVIDENCE",
                    "evidence_locator": artifact_name,
                    "evidence_sha256": artifact_hash,
                }
            ],
        },
        now=NOW,
        **owner(record, project),
    )
    return reviewed, artifact, artifact_hash


def test_failure_extraction_requires_hash_bound_artifact(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record, artifact, artifact_hash = _rejected_review_with_extracted_file(
        ledger,
        record,
        project,
    )
    artifact.write_text("tampered diagnostic\n", encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="failure_evidence_hash_mismatch"):
        ledger.close(
            "REFORM-20260813-01",
            {
                "disposition": "UNVERIFIED_FAILURE",
                "observations": ["The approach did not meet acceptance."],
                "evidence": [f"sha256:{artifact_hash}"],
                "hypotheses": ["The contract may be incomplete."],
                "preserved_location": "failure-evidence.json",
                "next_validation": "Run the focused negative control.",
            },
            "RETIRE_ELIGIBLE",
            skill_maintenance=no_skill_impact(),
            now=NOW,
            **owner(record, project),
        )


def test_failure_extraction_can_close_with_matching_hash(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record, artifact, artifact_hash = _rejected_review_with_extracted_file(
        ledger,
        record,
        project,
        artifact_name="failure-evidence-ok.json",
    )
    closed = ledger.close(
        "REFORM-20260813-01",
        {
            "disposition": "UNVERIFIED_FAILURE",
            "observations": ["The approach did not meet acceptance."],
            "evidence": [f"sha256:{artifact_hash}"],
            "hypotheses": ["The contract may be incomplete."],
            "preserved_location": artifact.name,
            "next_validation": "Run the focused negative control.",
        },
        "RETIRE_ELIGIBLE",
        skill_maintenance=no_skill_impact(),
        now=NOW,
        **owner(record, project),
    )
    assert closed["state"] == "CLOSED"


def test_close_rejects_integration_target_drift(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    integration = integration_packet(project)
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {
            "outcome": "ACCEPTED",
            "path_dispositions": [],
            "integration": integration,
        },
        now=NOW,
        **owner(record, project),
    )
    (project / "target-drift.txt").write_text("new target\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "target-drift.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(project), "commit", "-q", "-m", "target drift"],
        check=True,
    )
    with pytest.raises(manager.GovernanceError, match="integration_target_drift"):
        ledger.close(
            "REFORM-20260813-01",
            {
                "disposition": "NO_DURABLE_LESSON",
                "rationale": "Target changed after the integration receipt.",
                "evidence": ["pytest:integration-target-drift"],
            },
            "RETIRE_ELIGIBLE",
            skill_maintenance=no_skill_impact(),
            now=NOW,
            **owner(record, project),
        )


def test_close_rejects_integrated_commit_not_ancestor(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    base_branch = subprocess.run(
        ["git", "-C", str(project), "symbolic-ref", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(project), "checkout", "-q", "-b", "governance-divergent"],
        check=True,
    )
    (project / "divergent.txt").write_text("divergent\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "divergent.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(project), "commit", "-q", "-m", "divergent commit"],
        check=True,
    )
    divergent_sha = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(project), "checkout", "-q", base_branch], check=True)
    integration = integration_packet(project)
    integration["integrated_sha"] = divergent_sha
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {
            "outcome": "ACCEPTED",
            "path_dispositions": [],
            "integration": integration,
        },
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="integrated_commit_not_reachable"):
        ledger.close(
            "REFORM-20260813-01",
            {
                "disposition": "NO_DURABLE_LESSON",
                "rationale": "The purported integrated commit is divergent.",
                "evidence": ["pytest:integration-ancestor"],
            },
            "RETIRE_ELIGIBLE",
            skill_maintenance=no_skill_impact(),
            now=NOW,
            **owner(record, project),
        )


def test_close_requires_explicit_skill_impact_disposition(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = finish_steps(ledger, record, project)
    record = ledger.review_outcome(
        "REFORM-20260813-01",
        {
            "outcome": "ACCEPTED",
            "path_dispositions": [],
            "integration": integration_packet(project),
        },
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="skill_impact_missing"):
        ledger.close(
            "REFORM-20260813-01",
            {
                "disposition": "NO_DURABLE_LESSON",
                "rationale": "No durable project lesson was established.",
                "evidence": ["pytest:skill-impact-required"],
            },
            "RETIRE_ELIGIBLE",
            now=NOW,
            **owner(record, project),
        )


def scoped_packet(project: Path) -> dict[str, object]:
    value = packet(project)
    value["task_id"] = "SCOPED-PROGRESS-20260814-01"
    value["path_scope"] = ["README.md", "artifacts", "task-owned.txt"]
    return value


def create_scoped(
    manager: ModuleType,
    project: Path,
) -> tuple[object, dict[str, object]]:
    ledger = manager.AgentGovernanceLedger(project)
    record = ledger.create(
        scoped_packet(project),
        owner_session=RUNTIME,
        owner_token=TOKEN,
        worktree=project,
        lease_seconds=7200,
        now=NOW,
    )
    return ledger, confirm(ledger, record, project)


def expire_and_repermit(
    ledger: object,
    record: dict[str, object],
    project: Path,
) -> dict[str, object]:
    return ledger.permit(
        "SCOPED-PROGRESS-20260814-01",
        "S-1",
        ["edit"],
        now=NOW + timedelta(minutes=31),
        **owner(record, project),
    )


def test_authorized_edit_progress_is_accepted_after_expired_permit(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_scoped(manager, project)
    record = ledger.permit(
        "SCOPED-PROGRESS-20260814-01",
        "S-1",
        ["edit"],
        now=NOW,
        **owner(record, project),
    )
    (project / "task-owned.txt").write_text("authorized\n", encoding="utf-8")
    fresh = expire_and_repermit(ledger, record, project)
    assert fresh["worktree"]["accepted_task_fingerprint"] != record["worktree"]["fingerprint"]
    assert fresh["worktree"]["base_fingerprint"] == record["worktree"]["fingerprint"]


def test_generated_evidence_progress_is_accepted_after_expired_permit(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_scoped(manager, project)
    record = ledger.permit(
        "SCOPED-PROGRESS-20260814-01",
        "S-1",
        ["edit"],
        now=NOW,
        **owner(record, project),
    )
    evidence = project / "artifacts" / "evidence.json"
    evidence.parent.mkdir()
    evidence.write_text("{}\n", encoding="utf-8")
    assert expire_and_repermit(ledger, record, project)["active_permit"]


def test_authorized_descendant_commit_is_accepted_after_expired_permit(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_scoped(manager, project)
    record = ledger.permit(
        "SCOPED-PROGRESS-20260814-01",
        "S-1",
        ["edit"],
        now=NOW,
        **owner(record, project),
    )
    (project / "task-owned.txt").write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "task-owned.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(project), "commit", "-q", "-m", "task progress"],
        check=True,
    )
    fresh = expire_and_repermit(ledger, record, project)
    assert fresh["worktree"]["accepted_task_head"] != record["worktree"]["head_sha"]
    assert fresh["worktree"]["base_head"] == record["worktree"]["head_sha"]


def test_external_dirty_path_still_fails_closed_after_expired_permit(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_scoped(manager, project)
    record = ledger.permit(
        "SCOPED-PROGRESS-20260814-01",
        "S-1",
        ["edit"],
        now=NOW,
        **owner(record, project),
    )
    (project / "external.txt").write_text("owner change\n", encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="external_or_owner_change"):
        expire_and_repermit(ledger, record, project)


def test_mixed_authorized_and_unknown_paths_fail_closed(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_scoped(manager, project)
    record = ledger.permit(
        "SCOPED-PROGRESS-20260814-01",
        "S-1",
        ["edit"],
        now=NOW,
        **owner(record, project),
    )
    (project / "task-owned.txt").write_text("task change\n", encoding="utf-8")
    (project / "external.txt").write_text("unknown change\n", encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="unknown_or_mixed_change"):
        expire_and_repermit(ledger, record, project)


def test_out_of_scope_descendant_commit_is_not_accepted(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_scoped(manager, project)
    record = ledger.permit(
        "SCOPED-PROGRESS-20260814-01",
        "S-1",
        ["edit"],
        now=NOW,
        **owner(record, project),
    )
    (project / "external.txt").write_text("bad commit\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "external.txt"], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-q", "-m", "bad"], check=True)
    with pytest.raises(manager.GovernanceError, match="external_or_owner_change"):
        expire_and_repermit(ledger, record, project)


def test_completed_history_keeps_the_next_logical_step_active(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create(manager, project)
    record = confirm(ledger, record, project)
    record = ledger.permit(
        "REFORM-20260813-01", "S-1", ["test"], now=NOW, **owner(record, project)
    )
    record = ledger.advance(
        "REFORM-20260813-01",
        record["active_permit"]["permit_id"],
        pass_evidence("AC-1"),
        "S-2",
        now=NOW,
        **owner(record, project),
    )
    assert [step["status"] for step in record["plan"]["steps"]] == ["DONE", "IN_PROGRESS"]
    assert record["events"][-1]["payload"]["next_step_id"] == "S-2"


def test_same_session_recovery_preserves_accepted_progress(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_scoped(manager, project)
    record = ledger.permit(
        "SCOPED-PROGRESS-20260814-01", "S-1", ["edit"], now=NOW, **owner(record, project)
    )
    (project / "task-owned.txt").write_text("progress\n", encoding="utf-8")
    record = expire_and_repermit(ledger, record, project)
    accepted = dict(record["worktree"])
    recovered = ledger.recover_same_session(
        "SCOPED-PROGRESS-20260814-01",
        RUNTIME,
        record["revision"],
        record["record_sha256"],
        project,
        "Recover the same task without replaying accepted progress.",
        new_owner_token="recovered-scoped-owner-token-012345",
        now=NOW + timedelta(minutes=32),
    )
    for key in ("base_head", "accepted_task_head", "accepted_task_fingerprint", "path_scope"):
        assert recovered["worktree"][key] == accepted[key]
    assert recovered["plan"]["steps"][0]["status"] == "IN_PROGRESS"


def test_synthetic_multi_commit_progress_needs_no_manual_rebaseline(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_scoped(manager, project)
    record = ledger.permit(
        "SCOPED-PROGRESS-20260814-01", "S-1", ["edit"], now=NOW, **owner(record, project)
    )
    (project / "task-owned.txt").write_text("B\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "task-owned.txt"], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-q", "-m", "B"], check=True)
    record = expire_and_repermit(ledger, record, project)
    (project / "artifacts").mkdir(exist_ok=True)
    (project / "artifacts" / "release.json").write_text("{}\n", encoding="utf-8")
    record = ledger.advance(
        "SCOPED-PROGRESS-20260814-01",
        record["active_permit"]["permit_id"],
        pass_evidence("AC-1"),
        "S-2",
        now=NOW + timedelta(minutes=31),
        **owner(record, project),
    )
    assert record["worktree"]["base_head"] != record["worktree"]["accepted_task_head"]


def test_progressive_delivery_contract_forbids_late_reconciliation() -> None:
    agent_rules = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    protocol = (
        ROOT / "docs" / "governance" / "AGENT_GOVERNANCE_V2.md"
    ).read_text(encoding="utf-8")
    normalized_protocol = " ".join(protocol.split())

    assert "`main` is the continuous delivery branch" in agent_rules
    assert "do not accumulate a queue" in agent_rules
    assert "`main` is the delivery spine" in protocol
    assert "not an end-of-task destination" in protocol
    assert "It must never become a queue of completed changes" in normalized_protocol

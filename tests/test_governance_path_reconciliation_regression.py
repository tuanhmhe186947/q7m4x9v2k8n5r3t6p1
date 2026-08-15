"""Regression tests for GOV-REGRESSION-03: path reconciliation and fail-closed checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

NOW = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
RUNTIME = "regression-runtime-session-20260815"
TOKEN = "regression-owner-token-20260815"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(worktree: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Codex",
        "GIT_AUTHOR_EMAIL": "codex@example.com",
        "GIT_COMMITTER_NAME": "Codex",
        "GIT_COMMITTER_EMAIL": "codex@example.com",
    }
    subprocess.run(
        ["git", *args],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture(autouse=True)
def runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_THREAD_ID", RUNTIME)


@pytest.fixture
def manager() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / ".agents"
        / "skills"
        / "project-state-steward"
        / "scripts"
        / "manage_agent_governance.py"
    )
    spec = importlib.util.spec_from_file_location("agent_gov_regression_mgr", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_gov_regression_mgr"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path: Path) -> Path:
    target = tmp_path / "scientific-fixture"
    target.mkdir()
    git(target, "init", "-b", "main")
    (target / ".agents" / "memory").mkdir(parents=True)
    (target / ".agents" / "skills" / "reasoning-a").mkdir(parents=True)
    (target / ".agents" / "skills" / "verification-a").mkdir(parents=True)

    authority = target / ".agents" / "memory" / "03_PROJECT_RULES.md"
    authority.write_text("# Rules\n\nCurrent authority.\n", encoding="utf-8")
    reasoning = target / ".agents" / "skills" / "reasoning-a" / "SKILL.md"
    verification = target / ".agents" / "skills" / "verification-a" / "SKILL.md"
    reasoning.write_text("---\nname: reasoning-a\n---\n", encoding="utf-8")
    verification.write_text("---\nname: verification-a\n---\n", encoding="utf-8")

    inventory = {
        "schema_version": "pig.skill-inventory.v1",
        "generated_views": [],
        "task_routes": {
            "scientific_experiment": {
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
    (target / ".agents" / "skills" / "skill_inventory.json").write_text(
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
    (target / ".agents" / "memory" / "18_AUTHORITY_INDEX.json").write_text(
        json.dumps(authority_index),
        encoding="utf-8",
    )
    (target / "README.md").write_text("fixture\n", encoding="utf-8")
    (target / "docs").mkdir()
    (target / "docs" / "README.md").write_text("fixture docs\n", encoding="utf-8")
    (target / ".gitignore").write_text(".agents/runtime/\n", encoding="utf-8")

    git(target, "add", ".")
    git(target, "commit", "-m", "Initial fixture commit")
    return target


def packet(project: Path, task_id: str = "GOV-REGRESSION-20260815-01") -> dict[str, object]:
    authority = project / ".agents" / "memory" / "03_PROJECT_RULES.md"
    skill_path = project / ".agents" / "skills" / "reasoning-a" / "SKILL.md"
    roots = [
        {"root": "docs/classification_v2", "category": "evidence", "effects": ["edit", "test"]},
        {"root": "src/pig_behavior", "category": "source", "effects": ["edit", "test"]},
        {"root": "tests", "category": "test", "effects": ["edit", "test"]},
        {"root": "scripts/classification_v2", "category": "receipt", "effects": ["edit", "test"]},
    ]
    return {
        "task_id": task_id,
        "title": "Governance Regression Fixture",
        "task_class": "scientific_experiment",
        "risk_class": "high",
        "worktree_mode": "exclusive",
        "path_scope": [item["root"] for item in roots],
        "artifact_roots": roots,
        "authorities": [
            {
                "scope": "memory.lifecycle",
                "locator": ".agents/memory/03_PROJECT_RULES.md",
                "selector": "FULL_FILE",
                "status": "CURRENT",
                "read_at": NOW.isoformat(),
                "sha256": sha256(authority),
                "section_sha256": sha256(authority),
            },
        ],
        "acceptance": [
            {"acceptance_id": "AC-1", "text": "Descriptor binding validation passes."},
            {"acceptance_id": "AC-2", "text": "Integration readiness recorded."},
        ],
        "risks": ["Synthetic fixture must not touch protected data."],
        "non_actions": ["No GPU, training, or external data mutation."],
        "skills": [
            {
                "skill_id": "reasoning-a",
                "role": "reasoning",
                "purpose": "Ablation control reasoning.",
                "selection_mode": "explicit",
                "skill_sha256": sha256(skill_path),
            }
        ],
        "plan": {
            "steps": [
                {
                    "step_id": "S-1",
                    "summary": "Descriptor executor binding.",
                    "acceptance_ids": ["AC-1"],
                    "allowed_effects": ["edit", "test"],
                },
                {
                    "step_id": "S-2",
                    "summary": "Integration outcome review.",
                    "acceptance_ids": ["AC-2"],
                    "allowed_effects": ["edit", "integrate"],
                },
            ],
        },
    }


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


def create_task(
    manager: ModuleType,
    project: Path,
    task_id: str,
) -> tuple[object, dict[str, object]]:
    ledger = manager.AgentGovernanceLedger(project)
    record = ledger.create(
        packet(project, task_id),
        owner_session=RUNTIME,
        owner_token=TOKEN,
        worktree=project,
        lease_seconds=7200,
        now=NOW,
    )
    for selected in record["skills"]:
        record = ledger.record_skill_read(
            record["task_id"],
            selected["skill_id"],
            selected["skill_sha256"],
            [step["step_id"] for step in record["plan"]["steps"]],
            now=NOW,
            **owner(record, project),
        )
    confirmed = ledger.confirm_plan(
        record["task_id"],
        "synthetic-visible-plan-confirmation",
        "agent",
        now=NOW,
        **owner(record, project),
    )
    return ledger, confirmed


# TEST 1 — EXACT S1 REGRESSION
def test_1_exact_s1_regression_three_paths_with_unrelated_owner_dirty(
    manager: ModuleType,
    project: Path,
) -> None:
    owner_file = project / "owner_notes.txt"
    owner_file.write_text("pre-existing owner dirty notes\n", encoding="utf-8")

    task_id = "S1-REGRESSION-TEST-01"
    ledger, record = create_task(manager, project, task_id)
    assert "owner_notes.txt" in record["worktree"]["baseline_artifacts"]

    p1 = "scripts/classification_v2/04_baselines_smokes/classification_v2_s1_descriptor_binding.py"
    p2 = "src/pig_behavior/classification_v2/training/s1_descriptor_executor.py"
    p3 = "tests/test_classification_v2_s1_descriptor_executor.py"
    for path in (p1, p2, p3):
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text(f"# Content for {path}\n", encoding="utf-8")

    identity = manager._worktree_identity(project)
    assert set(identity["dirty_paths"]) == {"owner_notes.txt", p1, p2, p3}

    amended = ledger.amend_task_path_scope(
        task_id=task_id,
        confirm_task_id=task_id,
        confirmation=manager.SCOPE_AMENDMENT_CONFIRMATION,
        authorization_ref="user-request:s1-three-paths",
        exact_path=[p1, p2, p3],
        expected_revision=record["revision"],
        expected_record_sha256=record["record_sha256"],
        expected_worktree=project,
        expected_accepted_fingerprint=record["worktree"]["accepted_task_fingerprint"],
        expected_actual_fingerprint=identity["fingerprint"],
        now=NOW + timedelta(seconds=2),
    )

    # Require S1_THREE_PATH_RECONCILIATION=PASS
    assert set([p1, p2, p3]).issubset(set(amended["worktree"]["path_scope"]))
    # Require RECONCILED_PATH_COUNT=3
    reconciled_paths = [
        p for p in (p1, p2, p3) if p in amended["worktree"]["accepted_artifacts"]
    ]
    assert len(reconciled_paths) == 3
    # Require UNRELATED_OWNER_PATH_TOUCHED=NO
    assert owner_file.read_text(encoding="utf-8") == "pre-existing owner dirty notes\n"
    assert "owner_notes.txt" in amended["worktree"]["baseline_artifacts"]
    assert "owner_notes.txt" not in amended["worktree"]["path_scope"]

    # Require FRESH_PERMIT_AFTER_RECONCILIATION=PASS
    permit = ledger.permit(
        task_id,
        "S-1",
        ["edit", "test"],
        ttl_seconds=600,
        now=NOW + timedelta(minutes=1),
        **owner(amended, project),
    )
    assert permit["active_permit"]["step_id"] == "S-1"

    # Require SELF_BLOCKING_AFTER_RECONCILIATION=0 (Advance succeeds)
    ev_path = "docs/classification_v2/s1_evidence.json"
    (project / ev_path).parent.mkdir(parents=True, exist_ok=True)
    (project / ev_path).write_text('{"status":"PASS"}\n', encoding="utf-8")
    git(project, "add", p1, p2, p3, ev_path)
    git(project, "commit", "-m", "Commit S1 implementation")

    advanced = ledger.advance(
        task_id,
        permit["active_permit"]["permit_id"],
        [
            {
                "evidence_id": "EV-S1-DESC",
                "kind": "artifact",
                "locator": ev_path,
                "sha256": sha256(project / ev_path),
                "supports": ["AC-1"],
                "status": "PASS",
            }
        ],
        "S-2",
        now=NOW + timedelta(minutes=2),
        **owner(permit, project),
    )
    assert (
        next(s for s in advanced["plan"]["steps"] if s["step_id"] == "S-1")["status"]
        == "DONE"
    )
    assert (
        next(s for s in advanced["plan"]["steps"] if s["step_id"] == "S-2")["status"]
        == "IN_PROGRESS"
    )


# TEST 2 — EXTERNAL EDIT
def test_2_unexpected_external_edit_rejected(
    manager: ModuleType,
    project: Path,
) -> None:
    task_id = "S1-EXTERNAL-EDIT-TEST-02"
    ledger, record = create_task(manager, project, task_id)
    p1 = "scripts/classification_v2/04_baselines_smokes/classification_v2_s1_descriptor_binding.py"
    p2 = "src/pig_behavior/classification_v2/training/s1_descriptor_executor.py"
    p3 = "tests/test_classification_v2_s1_descriptor_executor.py"
    for path in (p1, p2, p3):
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text(f"# Content {path}\n", encoding="utf-8")

    identity = manager._worktree_identity(project)
    amended = ledger.amend_task_path_scope(
        task_id=task_id,
        confirm_task_id=task_id,
        confirmation=manager.SCOPE_AMENDMENT_CONFIRMATION,
        authorization_ref="user-request:test-external-edit",
        exact_path=[p1, p2, p3],
        expected_revision=record["revision"],
        expected_record_sha256=record["record_sha256"],
        expected_worktree=project,
        expected_accepted_fingerprint=record["worktree"]["accepted_task_fingerprint"],
        expected_actual_fingerprint=identity["fingerprint"],
        now=NOW + timedelta(seconds=2),
    )

    # External edit: mutate one of the 3 paths without an active permit
    (project / p1).write_text("# External unproven mutation\n", encoding="utf-8")

    # Permit issuance should reject worktree fingerprint drift
    with pytest.raises(manager.GovernanceError, match="worktree_fingerprint_drift"):
        ledger.permit(
            task_id,
            "S-1",
            ["edit", "test"],
            now=NOW + timedelta(minutes=1),
            **owner(amended, project),
        )


# TEST 3 — UNKNOWN FOURTH PATH
def test_3_mixed_unknown_change_rejected(
    manager: ModuleType,
    project: Path,
) -> None:
    task_id = "S1-UNKNOWN-FOURTH-PATH-03"
    ledger, record = create_task(manager, project, task_id)
    p1 = "scripts/classification_v2/04_baselines_smokes/classification_v2_s1_descriptor_binding.py"
    p2 = "src/pig_behavior/classification_v2/training/s1_descriptor_executor.py"
    p3 = "tests/test_classification_v2_s1_descriptor_executor.py"
    for path in (p1, p2, p3):
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text(f"# Content {path}\n", encoding="utf-8")

    # Extra unknown dirty file not in task scope
    (project / "unknown_drift.txt").write_text("random external drift\n", encoding="utf-8")

    identity = manager._worktree_identity(project)
    with pytest.raises(manager.GovernanceError, match="scope_amendment_unknown_delta"):
        ledger.amend_task_path_scope(
            task_id=task_id,
            confirm_task_id=task_id,
            confirmation=manager.SCOPE_AMENDMENT_CONFIRMATION,
            authorization_ref="user-request:test-unknown-delta",
            exact_path=[p1, p2, p3],
            expected_revision=record["revision"],
            expected_record_sha256=record["record_sha256"],
            expected_worktree=project,
            expected_accepted_fingerprint=record["worktree"]["accepted_task_fingerprint"],
            expected_actual_fingerprint=identity["fingerprint"],
            now=NOW + timedelta(seconds=2),
        )


# TEST 4 — SCOPE ESCAPE
def test_4_scope_escape_rejected(
    manager: ModuleType,
    project: Path,
) -> None:
    task_id = "S1-SCOPE-ESCAPE-TEST-04"
    ledger, record = create_task(manager, project, task_id)
    identity = manager._worktree_identity(project)

    # Traversal escape
    with pytest.raises(manager.GovernanceError, match="scope_amendment_path_invalid"):
        ledger.amend_task_path_scope(
            task_id=task_id,
            confirm_task_id=task_id,
            confirmation=manager.SCOPE_AMENDMENT_CONFIRMATION,
            authorization_ref="user-request:test-escape",
            exact_path=["../outside_file.py"],
            expected_revision=record["revision"],
            expected_record_sha256=record["record_sha256"],
            expected_worktree=project,
            expected_accepted_fingerprint=record["worktree"]["accepted_task_fingerprint"],
            expected_actual_fingerprint=identity["fingerprint"],
            now=NOW + timedelta(seconds=2),
        )

    # Wildcard escape
    with pytest.raises(manager.GovernanceError, match="scope_amendment_wildcard_forbidden"):
        ledger.amend_task_path_scope(
            task_id=task_id,
            confirm_task_id=task_id,
            confirmation=manager.SCOPE_AMENDMENT_CONFIRMATION,
            authorization_ref="user-request:test-wildcard",
            exact_path=["src/*.py"],
            expected_revision=record["revision"],
            expected_record_sha256=record["record_sha256"],
            expected_worktree=project,
            expected_accepted_fingerprint=record["worktree"]["accepted_task_fingerprint"],
            expected_actual_fingerprint=identity["fingerprint"],
            now=NOW + timedelta(seconds=2),
        )


# TEST 5 — STALE CAS
def test_5_stale_cas_rejected(
    manager: ModuleType,
    project: Path,
) -> None:
    task_id = "S1-STALE-CAS-TEST-05"
    ledger, record = create_task(manager, project, task_id)
    p1 = "scripts/test_p1.py"
    (project / p1).parent.mkdir(parents=True, exist_ok=True)
    (project / p1).write_text("# p1\n", encoding="utf-8")
    identity = manager._worktree_identity(project)

    # Stale revision
    with pytest.raises(manager.GovernanceError, match="revision_conflict"):
        ledger.amend_task_path_scope(
            task_id=task_id,
            confirm_task_id=task_id,
            confirmation=manager.SCOPE_AMENDMENT_CONFIRMATION,
            authorization_ref="user-request:test-stale-rev",
            exact_path=[p1],
            expected_revision=record["revision"] + 999,
            expected_record_sha256=record["record_sha256"],
            expected_worktree=project,
            expected_accepted_fingerprint=record["worktree"]["accepted_task_fingerprint"],
            expected_actual_fingerprint=identity["fingerprint"],
            now=NOW + timedelta(seconds=2),
        )

    # Stale record hash
    with pytest.raises(manager.GovernanceError, match="record_cas_conflict"):
        ledger.amend_task_path_scope(
            task_id=task_id,
            confirm_task_id=task_id,
            confirmation=manager.SCOPE_AMENDMENT_CONFIRMATION,
            authorization_ref="user-request:test-stale-hash",
            exact_path=[p1],
            expected_revision=record["revision"],
            expected_record_sha256="0" * 64,
            expected_worktree=project,
            expected_accepted_fingerprint=record["worktree"]["accepted_task_fingerprint"],
            expected_actual_fingerprint=identity["fingerprint"],
            now=NOW + timedelta(seconds=2),
        )


# TEST 6 — ATOMICITY
def test_6_batch_reconciliation_atomicity_rejects_without_partial_accept(
    manager: ModuleType,
    project: Path,
) -> None:
    task_id = "S1-ATOMICITY-TEST-06"
    ledger, record = create_task(manager, project, task_id)
    p1 = "scripts/classification_v2/04_baselines_smokes/valid_binding.py"
    p2_invalid = "data/protected_scientific_weights.pt"
    p3 = "tests/test_valid_executor.py"
    for path in (p1, p3):
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text(f"# Content {path}\n", encoding="utf-8")

    identity = manager._worktree_identity(project)

    # Batch with invalid 2nd path
    with pytest.raises(manager.GovernanceError, match="scope_amendment_protected_path"):
        ledger.amend_task_path_scope(
            task_id=task_id,
            confirm_task_id=task_id,
            confirmation=manager.SCOPE_AMENDMENT_CONFIRMATION,
            authorization_ref="user-request:test-atomicity",
            exact_path=[p1, p2_invalid, p3],
            expected_revision=record["revision"],
            expected_record_sha256=record["record_sha256"],
            expected_worktree=project,
            expected_accepted_fingerprint=record["worktree"]["accepted_task_fingerprint"],
            expected_actual_fingerprint=identity["fingerprint"],
            now=NOW + timedelta(seconds=2),
        )

    # Verify zero partial acceptance (PARTIAL_ACCEPT_COUNT=0)
    fresh = ledger._read(task_id)
    assert p1 not in fresh["worktree"]["path_scope"]
    assert p3 not in fresh["worktree"]["path_scope"]
    assert p1 not in fresh["worktree"].get("accepted_artifacts", {})
    assert p3 not in fresh["worktree"].get("accepted_artifacts", {})


# TEST 7 — HISTORY
def test_7_original_creation_history_preserved_and_no_retroactive_auth(
    manager: ModuleType,
    project: Path,
) -> None:
    task_id = "S1-HISTORY-TEST-07"
    ledger, record = create_task(manager, project, task_id)
    p1 = "scripts/s1_binding.py"
    (project / p1).parent.mkdir(parents=True, exist_ok=True)
    (project / p1).write_text("# S1 binding\n", encoding="utf-8")

    identity = manager._worktree_identity(project)
    amended = ledger.amend_task_path_scope(
        task_id=task_id,
        confirm_task_id=task_id,
        confirmation=manager.SCOPE_AMENDMENT_CONFIRMATION,
        authorization_ref="user-request:test-history",
        exact_path=[p1],
        expected_revision=record["revision"],
        expected_record_sha256=record["record_sha256"],
        expected_worktree=project,
        expected_accepted_fingerprint=record["worktree"]["accepted_task_fingerprint"],
        expected_actual_fingerprint=identity["fingerprint"],
        now=NOW + timedelta(seconds=2),
    )

    # Check original creation event is preserved byte-for-byte
    events = amended["events"]
    assert len(events) == 4  # TASK_CREATED, SKILL_READ, PLAN_CONFIRMED, TASK_PATH_SCOPE_AMENDED
    assert events[0]["event_type"] == "TASK_CREATED"
    assert events[-1]["event_type"] == "TASK_PATH_SCOPE_AMENDED"

    # Verify no retroactive authorization claim
    assert events[-1]["payload"]["append_only"] is True
    assert events[-1]["payload"]["empty_scope_bypass"] is False
    assert events[-1]["payload"]["path_scope_enforcement_preserved"] is True

"""Synthetic pre-fix reproduction for the governance execution-model repair."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

UTC = timezone.utc


@dataclass
class LegacyPermitFixture:
    """Minimal model of the historical guards, isolated from repository state."""

    path_scope: tuple[str, ...] = ("docs/governance", "src/adapter.py")
    authorized: bool = True
    recorded_head: str = "BASE_HEAD"
    actual_head: str = "BASE_HEAD"
    recorded_fingerprint: str = "BASE_FINGERPRINT"
    actual_fingerprint: str = "BASE_FINGERPRINT"
    active_permit: dict[str, object] | None = None
    accepted_paths: dict[str, str] = field(default_factory=dict)
    actual_paths: dict[str, str] = field(default_factory=dict)

    def issue_permit(self, now: datetime) -> None:
        if self.active_permit is not None:
            raise RuntimeError("ACTIVE_PERMIT_SLOT_OCCUPIED")
        if self.recorded_head != self.actual_head:
            raise RuntimeError("UNRELATED_HEAD_CHANGE")
        if self.recorded_fingerprint != self.actual_fingerprint:
            raise RuntimeError("UNKNOWN_DELTA")
        self.active_permit = {
            "issued_at": now,
            "expires_at": now + timedelta(minutes=1),
        }

    def task_write(self, path: str, content: str) -> None:
        self.actual_paths[path] = content
        self.actual_fingerprint = "TASK_PROGRESS_FINGERPRINT"

    def task_commit(self) -> None:
        self.actual_head = "TASK_DESCENDANT_HEAD"

    def expire(self, now: datetime) -> None:
        assert self.active_permit is not None
        assert now >= self.active_permit["expires_at"]

    def amend_scope(self, path: str) -> None:
        if self.active_permit is not None:
            raise RuntimeError("EXPIRED_PERMIT_ZOMBIE")
        self.path_scope += (path,)

    def classify_git_status_only(self, path: str) -> str:
        if path not in self.accepted_paths and path in self.actual_paths:
            return "UNKNOWN_UNTRACKED"
        return "ACCEPTED"


def test_reproduce_s1_failure_pattern_before_repair() -> None:
    """Reproduce A-I with only synthetic task, permit, and path state."""

    now = datetime(2026, 8, 14, 22, 0, tzinfo=UTC)
    fixture = LegacyPermitFixture()

    # A-D: authorized task, bounded work, progressing fingerprint, and artifacts.
    assert fixture.authorized is True
    fixture.issue_permit(now)
    fixture.task_write("docs/governance/plan.json", "plan-v1\n")
    fixture.task_write("src/adapter.py", "adapter-v1\n")
    fixture.accepted_paths.update(fixture.actual_paths)
    fixture.task_commit()
    assert fixture.actual_head == "TASK_DESCENDANT_HEAD"
    assert fixture.actual_fingerprint == "TASK_PROGRESS_FINGERPRINT"

    # E-H: expiry is real, but the historical active slot remains occupied.
    fixture.expire(now + timedelta(minutes=2))
    with pytest.raises(RuntimeError, match="ACTIVE_PERMIT_SLOT_OCCUPIED"):
        fixture.issue_permit(now + timedelta(minutes=2))
    with pytest.raises(RuntimeError, match="EXPIRED_PERMIT_ZOMBIE"):
        fixture.amend_scope("docs/governance/evidence.json")

    # I: Git's untracked classification hides accepted task lineage.
    fixture.accepted_paths.pop("docs/governance/plan.json")
    assert (
        fixture.classify_git_status_only("docs/governance/plan.json")
        == "UNKNOWN_UNTRACKED"
    )


def test_reproduction_is_authorized_progress_not_external_drift() -> None:
    fixture = LegacyPermitFixture()
    fixture.task_write("docs/governance/evidence.json", "accepted\n")
    fixture.accepted_paths["docs/governance/evidence.json"] = "accepted\n"

    assert fixture.authorized is True
    assert fixture.actual_paths == fixture.accepted_paths
    assert fixture.classify_git_status_only("docs/governance/evidence.json") == (
        "ACCEPTED"
    )
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / ".agents"
    / "skills"
    / "project-state-steward"
    / "scripts"
    / "manage_agent_governance.py"
)
RUNTIME = "synthetic-governance-runtime-20260814"
TOKEN = "synthetic-governance-owner-token-012345"
RECOVERED_TOKEN = "synthetic-governance-recovered-token-012345"
NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("CODEX_THREAD_ID", RUNTIME)
    spec = importlib.util.spec_from_file_location(
        "synthetic_governance_manager",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "scientific-fixture"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "config",
            "user.email",
            "synthetic@example.com",
        ],
        check=True,
    )
    (root / ".agents" / "memory").mkdir(parents=True)
    (root / ".agents" / "skills" / "reasoning-a").mkdir(parents=True)
    (root / ".agents" / "skills" / "verification-a").mkdir(parents=True)
    authority = root / ".agents" / "memory" / "03_PROJECT_RULES.md"
    authority.write_text("# Rules\n\nSynthetic current authority.\n", encoding="utf-8")
    (root / ".agents" / "skills" / "reasoning-a" / "SKILL.md").write_text(
        "---\nname: reasoning-a\n---\n",
        encoding="utf-8",
    )
    (root / ".agents" / "skills" / "verification-a" / "SKILL.md").write_text(
        "---\nname: verification-a\n---\n",
        encoding="utf-8",
    )
    inventory = {
        "schema_version": "pig.skill-inventory.v1",
        "generated_views": [],
        "task_routes": {
            "scientific_experiment": {
                "required_any": ["reasoning-a"],
                "reasoning_required": True,
            },
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
            },
        ],
    }
    (root / ".agents" / "memory" / "18_AUTHORITY_INDEX.json").write_text(
        json.dumps(authority_index),
        encoding="utf-8",
    )
    (root / "README.md").write_text("synthetic scientific fixture\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        ".agents/runtime/\n__pycache__/\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "synthetic baseline"],
        check=True,
    )
    return root


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection(project: Path) -> dict[str, str]:
    path = project / ".agents" / "skills" / "reasoning-a" / "SKILL.md"
    return {
        "skill_id": "reasoning-a",
        "role": "reasoning",
        "purpose": "Bound the scientific execution reasoning contract.",
        "selection_mode": "explicit",
        "skill_sha256": sha256(path),
    }


def packet(project: Path, task_id: str = "SYNTHETIC-CONTINUOUS-20260814-01") -> dict[
    str, object
]:
    authority = project / ".agents" / "memory" / "03_PROJECT_RULES.md"
    roots = [
        {
            "root": "docs/synthetic_governance",
            "category": "evidence",
            "effects": ["edit", "test"],
        },
        {
            "root": "src/synthetic_adapter",
            "category": "source",
            "effects": ["edit", "test"],
        },
        {
            "root": "tests/synthetic_adapter",
            "category": "test",
            "effects": ["edit", "test"],
        },
        {
            "root": "audit/synthetic",
            "category": "receipt",
            "effects": ["edit", "test"],
        },
        {
            "root": "release/synthetic",
            "category": "authority",
            "effects": ["edit", "integrate"],
        },
    ]
    return {
        "task_id": task_id,
        "title": "Synthetic Classification V2 adapter experiment",
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
            {"acceptance_id": "AC-1", "text": "Adapter validation passes."},
            {"acceptance_id": "AC-2", "text": "Release readiness is recorded."},
        ],
        "risks": ["Synthetic fixture must not touch scientific data."],
        "non_actions": ["No GPU, training, or external data access."],
        "skills": [selection(project)],
        "plan": {
            "steps": [
                {
                    "step_id": "S-1",
                    "summary": "Adapter implementation, focused tests, CPU validation.",
                    "acceptance_ids": ["AC-1"],
                    "allowed_effects": ["edit", "test"],
                },
                {
                    "step_id": "S-2",
                    "summary": "Release candidate and integration readiness.",
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


def create_confirm(
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
    record = ledger.confirm_plan(
        record["task_id"],
        "synthetic-visible-plan-confirmation",
        "agent",
        now=NOW,
        **owner(record, project),
    )
    return ledger, record


def commit_paths(project: Path, message: str, *paths: str) -> str:
    subprocess.run(
        ["git", "-C", str(project), "add", "--", *paths],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "commit", "-q", "-m", message],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def artifact_evidence(
    evidence_id: str,
    path: str,
    acceptance_id: str,
    project: Path,
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "kind": "artifact",
        "locator": path,
        "sha256": sha256(project / path),
        "supports": [acceptance_id],
        "status": "PASS",
    }


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
                "evidence_id": "EV-INTEGRATION",
                "kind": "command_result",
                "locator": "pytest:synthetic-governance",
                "status": "PASS",
                "target_sha": head,
            },
        ],
    }


def no_skill_impact() -> list[dict[str, object]]:
    return [
        {
            "status": "NO_SKILL_IMPACT",
            "rationale": "Synthetic governance fixture did not change a skill.",
            "evidence": ["pytest:test_governance_execution_model_acceptance"],
        },
    ]


def test_synthetic_continuous_task_progress(manager: ModuleType, project: Path) -> None:
    ledger, record = create_confirm(manager, project)
    base_head = record["worktree"]["base_head"]
    permit = ledger.permit(
        record["task_id"],
        "S-1",
        ["edit", "test"],
        ttl_seconds=60,
        now=NOW,
        **owner(record, project),
    )
    paths = {
        "plan": "docs/synthetic_governance/plan.json",
        "source": "src/synthetic_adapter/adapter.py",
        "test": "tests/synthetic_adapter/test_adapter.py",
        "audit": "audit/synthetic/initial_receipt.json",
    }
    for path in paths.values():
        (project / path).parent.mkdir(parents=True, exist_ok=True)
    (project / paths["plan"]).write_text(
        '{"objective":"bounded adapter"}\n',
        encoding="utf-8",
    )
    (project / paths["source"]).write_text(
        "def adapt(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    (project / paths["test"]).write_text(
        "assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    (project / paths["audit"]).write_text(
        '{"stage":"initial","cpu":true}\n',
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from pathlib import Path; assert 'return value + 1' in "
            "Path('src/synthetic_adapter/adapter.py').read_text()",
        ],
        cwd=project,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    head_b = commit_paths(
        project,
        "task-owned adapter implementation B",
        *paths.values(),
    )
    assert head_b != base_head

    record = ledger.permit(
        record["task_id"],
        "S-1",
        ["edit", "test"],
        ttl_seconds=600,
        now=NOW + timedelta(minutes=31),
        **owner(permit, project),
    )
    assert record["permit_history"][-1]["status"] == "EXPIRED"
    evidence_path = "docs/synthetic_governance/continued_evidence.json"
    (project / evidence_path).write_text(
        '{"stage":"continued","accepted":true}\n',
        encoding="utf-8",
    )
    head_c = commit_paths(project, "task-owned continuation C", evidence_path)
    assert head_c != head_b
    record = ledger.advance(
        record["task_id"],
        record["active_permit"]["permit_id"],
        [artifact_evidence("EV-S1", evidence_path, "AC-1", project)],
        "S-2",
        now=NOW + timedelta(minutes=32),
        **owner(record, project),
    )
    assert record["worktree"]["accepted_task_head"] == head_c
    assert record["worktree"]["base_head"] == base_head

    record = ledger.permit(
        record["task_id"],
        "S-2",
        ["edit", "integrate"],
        ttl_seconds=600,
        now=NOW + timedelta(minutes=33),
        **owner(record, project),
    )
    release_path = "release/synthetic/release_candidate.json"
    (project / release_path).parent.mkdir(parents=True, exist_ok=True)
    (project / release_path).write_text(
        '{"status":"READY_FOR_INTEGRATION"}\n',
        encoding="utf-8",
    )
    head_d = commit_paths(project, "release candidate descendant D", release_path)
    record = ledger.advance(
        record["task_id"],
        record["active_permit"]["permit_id"],
        [artifact_evidence("EV-S2", release_path, "AC-2", project)],
        None,
        now=NOW + timedelta(minutes=34),
        **owner(record, project),
    )
    assert record["worktree"]["accepted_task_head"] == head_d
    assert (
        record["worktree"]["accepted_task_fingerprint"]
        == record["worktree"]["actual_worktree_fingerprint"]
    )
    assert record["worktree"]["task_cursor"] >= 2
    accepted = record["worktree"]["accepted_artifacts"]
    assert accepted[evidence_path]["sha256"] == sha256(project / evidence_path)
    assert accepted[release_path]["sha256"] == sha256(project / release_path)
    assert not any(
        event["event_type"] in {
            "WORKTREE_FINGERPRINT_REBASELINED",
            "WORKTREE_HEAD_REBOUND",
            "TASK_PATH_SCOPE_AMENDED",
        }
        for event in record["events"]
    )
    reviewed = ledger.review_outcome(
        record["task_id"],
        {
            "outcome": "ACCEPTED",
            "path_dispositions": [],
            "integration": integration_packet(project),
        },
        now=NOW + timedelta(minutes=35),
        **owner(record, project),
    )
    closed = ledger.close(
        record["task_id"],
        {
            "disposition": "NO_DURABLE_LESSON",
            "rationale": "The synthetic execution model passed its acceptance matrix.",
            "evidence": ["pytest:synthetic-governance"],
        },
        "RETIRE_ELIGIBLE",
        skill_maintenance=no_skill_impact(),
        now=NOW + timedelta(minutes=36),
        **owner(reviewed, project),
    )
    assert closed["state"] == "CLOSED"
    assert closed["active_permit"] is None
    assert sum(item["status"] == "EXPIRED" for item in closed["permit_history"]) == 1
    assert sum(item["status"] == "CONSUMED" for item in closed["permit_history"]) == 2


def test_accepted_untracked_artifact_is_not_reclassified(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_confirm(manager, project)
    permit = ledger.permit(
        record["task_id"],
        "S-1",
        ["edit", "test"],
        now=NOW,
        **owner(record, project),
    )
    path = "docs/synthetic_governance/untracked_evidence.json"
    (project / path).parent.mkdir(parents=True, exist_ok=True)
    (project / path).write_text('{"accepted":true}\n', encoding="utf-8")
    accepted = ledger.advance(
        record["task_id"],
        permit["active_permit"]["permit_id"],
        [artifact_evidence("EV-UNTRACKED", path, "AC-1", project)],
        "S-2",
        now=NOW + timedelta(minutes=1),
        **owner(permit, project),
    )
    assert accepted["worktree"]["accepted_artifacts"][path]["sha256"] == sha256(
        project / path
    )
    next_permit = ledger.permit(
        accepted["task_id"],
        "S-2",
        ["integrate"],
        now=NOW + timedelta(minutes=2),
        **owner(accepted, project),
    )
    assert next_permit["active_permit"]["step_id"] == "S-2"


def test_expired_permit_does_not_block_scope_transition(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_confirm(
        manager,
        project,
        packet(project, "SYNTHETIC-EXPIRY-SCOPE-20260814-01"),
    )
    permit = ledger.permit(
        record["task_id"],
        "S-1",
        ["edit", "test"],
        ttl_seconds=60,
        now=NOW,
        **owner(record, project),
    )
    path = "notes/synthetic_plan.json"
    (project / path).parent.mkdir(parents=True, exist_ok=True)
    (project / path).write_text('{"scope":"amended"}\n', encoding="utf-8")
    identity = manager._worktree_identity(project)
    amended = ledger.amend_task_path_scope(
        permit["task_id"],
        permit["task_id"],
        "USER_AUTHORIZED_EXACT_TASK_PATH_SCOPE_AMENDMENT",
        "synthetic-scope-amendment",
        path,
        permit["revision"],
        permit["record_sha256"],
        project,
        permit["worktree"]["accepted_task_fingerprint"],
        identity["fingerprint"],
        now=NOW + timedelta(minutes=31),
    )
    assert amended["active_permit"] is None
    assert amended["permit_history"][-1]["status"] == "EXPIRED"
    assert amended["worktree"]["accepted_artifacts"][path]["sha256"] == sha256(
        project / path
    )
    fresh = ledger.permit(
        amended["task_id"],
        "S-1",
        ["edit", "test"],
        now=NOW + timedelta(minutes=32),
        **owner(amended, project),
    )
    assert fresh["active_permit"]["step_id"] == "S-1"


def test_s1_governance_failure_pattern_regression(
    manager: ModuleType,
    project: Path,
) -> None:
    task_id = "S1-GOVERNANCE-REGRESSION-20260814-01"
    ledger, record = create_confirm(manager, project, packet(project, task_id))
    permit = ledger.permit(
        task_id,
        "S-1",
        ["edit", "test"],
        ttl_seconds=60,
        now=NOW,
        **owner(record, project),
    )
    plan_path = "docs/synthetic_governance/s1_plan.json"
    source_path = "src/synthetic_adapter/s1_adapter.py"
    test_path = "tests/synthetic_adapter/s1_test.py"
    for path in (plan_path, source_path, test_path):
        (project / path).parent.mkdir(parents=True, exist_ok=True)
    (project / plan_path).write_text('{"step":"S1-02"}\n', encoding="utf-8")
    (project / source_path).write_text("ADAPTER_READY = True\n", encoding="utf-8")
    (project / test_path).write_text("assert True\n", encoding="utf-8")
    commit_paths(project, "S1 adapter work package B", plan_path, source_path, test_path)
    fresh = ledger.permit(
        task_id,
        "S-1",
        ["edit", "test"],
        ttl_seconds=600,
        now=NOW + timedelta(minutes=31),
        **owner(permit, project),
    )
    evidence_path = "docs/synthetic_governance/s1_evidence.json"
    (project / evidence_path).write_text(
        '{"adapter":"validated","cpu":true}\n',
        encoding="utf-8",
    )
    commit_paths(project, "S1 continued evidence C", evidence_path)
    progressed = ledger.advance(
        task_id,
        fresh["active_permit"]["permit_id"],
        [artifact_evidence("EV-S1-REGRESSION", evidence_path, "AC-1", project)],
        "S-2",
        now=NOW + timedelta(minutes=32),
        **owner(fresh, project),
    )
    assert progressed["permit_history"][-2]["status"] == "EXPIRED"
    assert progressed["worktree"]["accepted_artifacts"][evidence_path]
    assert not any(
        event["event_type"] in {
            "WORKTREE_FINGERPRINT_REBASED",
            "WORKTREE_FINGERPRINT_REBASELINED",
            "WORKTREE_HEAD_REBOUND",
            "TASK_PATH_SCOPE_AMENDED",
        }
        for event in progressed["events"]
    )


def test_session_recovery_preserves_cursor_and_accepted_state(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = create_confirm(manager, project)
    permit = ledger.permit(
        record["task_id"],
        "S-1",
        ["edit", "test"],
        ttl_seconds=60,
        now=NOW,
        **owner(record, project),
    )
    path = "src/synthetic_adapter/recovery.py"
    (project / path).parent.mkdir(parents=True, exist_ok=True)
    (project / path).write_text("RECOVERED = True\n", encoding="utf-8")
    recovered = ledger.recover_same_session(
        record["task_id"],
        RUNTIME,
        permit["revision"],
        permit["record_sha256"],
        project,
        "Synthetic session interruption before permit expiry recovery.",
        new_owner_token=RECOVERED_TOKEN,
        lease_seconds=7200,
        now=NOW + timedelta(minutes=1),
    )
    cursor_before = recovered["worktree"]["task_cursor"]
    accepted_before = recovered["worktree"]["accepted_task_fingerprint"]
    continued = ledger.permit(
        recovered["task_id"],
        "S-1",
        ["edit", "test"],
        now=NOW + timedelta(minutes=31),
        **owner(recovered, project, token=RECOVERED_TOKEN),
    )
    assert continued["permit_history"][-1]["status"] == "EXPIRED"
    assert continued["worktree"]["task_cursor"] > cursor_before
    assert continued["worktree"]["accepted_task_fingerprint"] != accepted_before


def active_task(
    manager: ModuleType,
    project: Path,
    value: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    ledger, record = create_confirm(manager, project, value)
    record = ledger.permit(
        record["task_id"],
        "S-1",
        ["edit", "test"],
        now=NOW,
        **owner(record, project),
    )
    return ledger, record


def test_external_modification_still_fails_closed(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = active_task(manager, project)
    (project / "external.txt").write_text("owner drift\n", encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="external_or_owner_change"):
        ledger.advance(
            record["task_id"],
            record["active_permit"]["permit_id"],
            [],
            "S-2",
            now=NOW + timedelta(minutes=1),
            **owner(record, project),
        )


def test_owner_only_dirty_file_still_fails_closed(
    manager: ModuleType,
    project: Path,
) -> None:
    value = packet(project, "SYNTHETIC-OWNER-DRIFT-20260814-01")
    value["path_scope"] = [*value["path_scope"], "README.md"]
    ledger, record = active_task(manager, project, value)
    (project / "README.md").write_text("owner-only change\n", encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="external_or_owner_change"):
        ledger.advance(
            record["task_id"],
            record["active_permit"]["permit_id"],
            [],
            "S-2",
            now=NOW + timedelta(minutes=1),
            **owner(record, project),
        )


def test_mixed_task_and_unknown_change_still_fails_closed(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = active_task(
        manager,
        project,
        packet(project, "SYNTHETIC-MIXED-20260814-01"),
    )
    task_path = "src/synthetic_adapter/mixed.py"
    (project / task_path).parent.mkdir(parents=True, exist_ok=True)
    (project / task_path).write_text("TASK = True\n", encoding="utf-8")
    (project / "unknown.txt").write_text("unknown\n", encoding="utf-8")
    with pytest.raises(manager.GovernanceError, match="unknown_or_mixed_change"):
        ledger.advance(
            record["task_id"],
            record["active_permit"]["permit_id"],
            [],
            "S-2",
            now=NOW + timedelta(minutes=1),
            **owner(record, project),
        )


def test_scope_escape_and_traversal_still_fail_closed(
    manager: ModuleType,
    project: Path,
) -> None:
    value = packet(project, "SYNTHETIC-TRAVERSAL-20260814-01")
    value["artifact_roots"] = [
        {"root": "../escape", "category": "source", "effects": ["edit"]},
    ]
    value["path_scope"] = ["escape"]
    with pytest.raises(manager.GovernanceError, match="artifact_root_invalid"):
        manager.AgentGovernanceLedger(project).create(
            value,
            owner_session=RUNTIME,
            owner_token=TOKEN,
            worktree=project,
            now=NOW,
        )


def test_stale_cas_still_fails_closed(manager: ModuleType, project: Path) -> None:
    ledger, record = create_confirm(
        manager,
        project,
        packet(project, "SYNTHETIC-CAS-20260814-01"),
    )
    stale = owner(record, project)
    updated = ledger.permit(
        record["task_id"],
        "S-1",
        ["edit", "test"],
        now=NOW,
        **owner(record, project),
    )
    with pytest.raises(manager.GovernanceError, match="revision_conflict"):
        ledger.advance(
            updated["task_id"],
            updated["active_permit"]["permit_id"],
            [],
            "S-2",
            now=NOW + timedelta(minutes=1),
            **stale,
        )


def test_unrelated_head_still_fails_closed(manager: ModuleType, project: Path) -> None:
    ledger, record = active_task(
        manager,
        project,
        packet(project, "SYNTHETIC-HEAD-20260814-01"),
    )
    task_path = "src/synthetic_adapter/lineage.py"
    (project / task_path).parent.mkdir(parents=True, exist_ok=True)
    (project / task_path).write_text("TASK_LINEAGE = True\n", encoding="utf-8")
    task_head = commit_paths(project, "task lineage", task_path)
    record = ledger.permit(
        record["task_id"],
        "S-1",
        ["edit", "test"],
        now=NOW + timedelta(minutes=31),
        **owner(record, project),
    )
    base_head = record["worktree"]["base_head"]
    subprocess.run(
        ["git", "-C", str(project), "checkout", "-q", "-b", "unrelated", base_head],
        check=True,
    )
    (project / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    commit_paths(project, "unrelated head", "unrelated.txt")
    assert task_head != subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(manager.GovernanceError, match="external_or_owner_change"):
        ledger.advance(
            record["task_id"],
            record["active_permit"]["permit_id"],
            [],
            "S-2",
            now=NOW + timedelta(minutes=32),
            **owner(record, project),
        )


def test_merge_owner_conflict_still_fails_closed(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = active_task(
        manager,
        project,
        packet(project, "SYNTHETIC-MERGE-20260814-01"),
    )
    task_path = "src/synthetic_adapter/merge.py"
    (project / task_path).parent.mkdir(parents=True, exist_ok=True)
    (project / task_path).write_text("TASK_CHANGE = True\n", encoding="utf-8")
    task_head = commit_paths(project, "task branch commit", task_path)
    record = ledger.permit(
        record["task_id"],
        "S-1",
        ["edit", "test"],
        now=NOW + timedelta(minutes=31),
        **owner(record, project),
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "checkout",
            "-q",
            "-b",
            "owner-branch",
            record["worktree"]["base_head"],
        ],
        check=True,
    )
    (project / "owner-change.txt").write_text("owner\n", encoding="utf-8")
    commit_paths(project, "owner branch commit", "owner-change.txt")
    subprocess.run(
        ["git", "-C", str(project), "merge", "--no-ff", "-q", task_head],
        check=True,
    )
    with pytest.raises(manager.GovernanceError, match="external_or_owner_change"):
        ledger.advance(
            record["task_id"],
            record["active_permit"]["permit_id"],
            [],
            "S-2",
            now=NOW + timedelta(minutes=32),
            **owner(record, project),
        )


def test_unknown_binary_and_scientific_data_still_fail_closed(
    manager: ModuleType,
    project: Path,
) -> None:
    ledger, record = active_task(
        manager,
        project,
        packet(project, "SYNTHETIC-DATA-20260814-01"),
    )
    binary = project / "audit" / "synthetic" / "unknown.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"unknown binary")
    with pytest.raises(manager.GovernanceError, match="external_or_owner_change"):
        ledger.advance(
            record["task_id"],
            record["active_permit"]["permit_id"],
            [],
            "S-2",
            now=NOW + timedelta(minutes=1),
            **owner(record, project),
        )

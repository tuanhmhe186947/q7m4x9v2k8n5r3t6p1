from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    ROOT
    / ".agents"
    / "skills"
    / "project-state-steward"
    / "scripts"
    / "validate_memory_contract_v2.py"
)
GOVERNANCE_VALIDATOR = VALIDATOR.with_name("validate_governance_contracts.py")
TASK_MANAGER = VALIDATOR.with_name("manage_short_memory.py")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_project_memory_contract_is_valid() -> None:
    spec = importlib.util.spec_from_file_location("memory_contract", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.audit(ROOT)

    assert report["status"] == "PASS", report


def test_skill_bundle_hash_detects_bundled_script_change(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "governance_contract",
        GOVERNANCE_VALIDATOR,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    skill = tmp_path / "project-skill"
    scripts = skill / "scripts"
    scripts.mkdir(parents=True)
    (skill / "SKILL.md").write_text("skill-v1\n", encoding="utf-8")
    validator = scripts / "validator.py"
    validator.write_text("VALUE = 1\n", encoding="utf-8")
    paths = ["SKILL.md", "scripts/validator.py"]

    initial = module._bundle_sha256(skill, paths)
    validator.write_text("VALUE = 2\n", encoding="utf-8")

    assert module._bundle_sha256(skill, paths) != initial


def test_expired_short_memory_requires_reset(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("memory_contract", VALIDATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    memory = tmp_path / ".agents" / "memory"
    memory.mkdir(parents=True)
    (memory / "01_PROJECT_MEMORY_SHORT.md").write_text(
        "\n".join(
            [
                "# Project Memory Short",
                "## Lifecycle",
                "- Opened: `2026-07-30`.",
                "- Expires: `2026-07-31T00:00:00+07:00`.",
                "## 2026-07-30 handoff",
            ]
        ),
        encoding="utf-8",
    )

    result = module._check_short_ttl(
        tmp_path,
        "2026-07-31T00:00:01+07:00",
    )

    assert result["status"] == "RESET_REQUIRED"
    assert "short_memory_expired" in result["errors"]


def _write_short_fixture(tmp_path: Path, lines: list[str]) -> None:
    memory = tmp_path / ".agents" / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    (memory / "01_PROJECT_MEMORY_SHORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def test_short_checklist_accepts_evidenced_phase_state(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("memory_contract", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _write_short_fixture(
        tmp_path,
        [
            "# Project Memory Short",
            "## Lifecycle",
            "- Opened: `2026-08-03`.",
            "- Expires: `2026-08-04T00:00:00+07:00`.",
            "- Legacy unmanaged task IDs: `MEM-20260803-01`.",
            "## Active Task Checklist",
            "### MEM-20260803-01 - fixture task",
            "- Prompt: validate checklist.",
            "- Status: `IN_PROGRESS`.",
            "- Opened: `2026-08-03T08:00:00+07:00`.",
            "- Acceptance: validator passes.",
            "- Skills: `knowledge-ops`.",
            "- [x] `MEM-01` `[DONE]` Read authority.",
            "  - Evidence: authority was inspected.",
            "- [ ] `MEM-02` `[IN_PROGRESS]` Run validator.",
            "  - Next: execute the focused test.",
            "## Previous-Day Closeout",
            "- Source date: `2026-08-02`.",
            "- Completed: one prior task.",
            "- Carried forward: none.",
            "- Purge after: `2026-08-04T00:00:00+07:00`.",
        ],
    )

    assert module._check_short_checklist(tmp_path) == []


def test_short_checklist_accepts_prior_day_managed_task_completed_today(
    tmp_path: Path,
) -> None:
    validator = _load_module("governance_contract", GOVERNANCE_VALIDATOR)
    manager = _load_module("short_memory_manager", TASK_MANAGER)
    _write_short_fixture(
        tmp_path,
        [
            "# Project Memory Short",
            "## Lifecycle",
            "- Opened: `2026-08-03`.",
            "- Expires: `2026-08-04T00:00:00+07:00`.",
            "- Legacy unmanaged task IDs: none.",
            "## Active Task Checklist",
            "- None.",
            "## Previous-Day Closeout",
            "- Source date: `2026-08-03`.",
            "- Completed: none.",
            "- Carried forward: terminal fixture completed after rollover.",
            "- Purge after: `2026-08-05T00:00:00+07:00`.",
        ],
    )
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = ledger.create(
        task_id="TEST-20260803-01",
        title="cross-day completion fixture",
        prompt="Complete after rollover.",
        acceptance="Terminal capsule remains valid until next rollover.",
        skills=["project-state-steward"],
        steps=[
            {
                "step_id": "TEST-1",
                "summary": "Complete after rollover",
                "next_action": "Checkpoint with evidence.",
            }
        ],
        active_step="TEST-1",
        owner_session="session-alpha-0001",
        owner_token="owner-token-alpha-0001",
        worktree=tmp_path,
        now=datetime(2026, 8, 3, 23, 55, tzinfo=timezone(timedelta(hours=7))),
    )
    ledger.checkpoint(
        task_id=created["task_id"],
        step_id="TEST-1",
        step_status="DONE",
        evidence="Completion verified after midnight.",
        next_action=None,
        owner_session=created["owner_session"],
        owner_token="owner-token-alpha-0001",
        worktree=tmp_path,
        expected_revision=created["revision"],
        expected_block_sha256=created["block_sha256"],
        now=datetime(2026, 8, 4, 0, 5, tzinfo=timezone(timedelta(hours=7))),
    )
    path = tmp_path / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md"
    text = path.read_bytes().decode("utf-8")
    text = text.replace("`2026-08-03`", "`2026-08-04`", 1)
    text = text.replace(
        "`2026-08-04T00:00:00+07:00`",
        "`2026-08-05T00:00:00+07:00`",
        1,
    )
    path.write_bytes(text.encode("utf-8"))

    assert validator._check_short_checklist(tmp_path) == []


def test_short_checklist_rejects_false_completion(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("memory_contract", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _write_short_fixture(
        tmp_path,
        [
            "# Project Memory Short",
            "## Lifecycle",
            "- Opened: `2026-08-03`.",
            "- Expires: `2026-08-04T00:00:00+07:00`.",
            "- Legacy unmanaged task IDs: `MEM-20260803-01`.",
            "## Active Task Checklist",
            "### MEM-20260803-01 - invalid fixture",
            "- Prompt: validate checklist.",
            "- Status: `IN_PROGRESS`.",
            "- Opened: `2026-08-03T08:00:00+07:00`.",
            "- Acceptance: validator rejects invalid states.",
            "- Skills: `knowledge-ops`.",
            "- [x] `MEM-01` `[DONE]` Missing evidence.",
            "- [ ] `MEM-02` `[IN_PROGRESS]` First active step.",
            "  - Next: continue.",
            "- [ ] `MEM-03` `[IN_PROGRESS]` Second active step.",
            "  - Next: stop parallel progress.",
        ],
    )

    errors = module._check_short_checklist(tmp_path)

    assert "short_terminal_step_missing_evidence:MEM-01" in errors
    assert "short_task_multiple_in_progress:MEM-20260803-01" in errors


def test_governance_references_require_crash_recovery_contract(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "governance_contract",
        GOVERNANCE_VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    documents = {
        "AGENTS.md": (
            "18_AUTHORITY_INDEX 19_REASONING_ROUTING 21_MEMORY_MATURITY"
        ),
        ".agents/memory/00_README.md": (
            "18_AUTHORITY_INDEX 19_REASONING_ROUTING 21_MEMORY_MATURITY"
        ),
        ".agents/memory/03_PROJECT_RULES.md": "rules",
        ".agents/memory/08_WORKFLOW.md": (
            "18_AUTHORITY_INDEX 19_REASONING_ROUTING 21_MEMORY_MATURITY"
        ),
        ".agents/skills/project-state-steward/SKILL.md": "skill",
    }
    for relative, content in documents.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    errors = module._check_governance_references(tmp_path)

    assert "governance_missing_done_checkpoint:AGENTS.md" in errors
    assert (
        "governance_missing_interrupted_recovery:"
        "project-state-steward/SKILL.md"
    ) in errors

    contract = (
        " checkpoint `DONE` before the next step's first effect;"
        " recover interrupted `IN_PROGRESS` work;"
        " CODEX_THREAD_ID admin-takeover hash-bound audit"
    )
    for relative, content in documents.items():
        (tmp_path / relative).write_text(content + contract, encoding="utf-8")

    assert module._check_governance_references(tmp_path) == []


def test_short_checklist_rejects_stale_closeout(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("memory_contract", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _write_short_fixture(
        tmp_path,
        [
            "# Project Memory Short",
            "## Lifecycle",
            "- Opened: `2026-08-03`.",
            "- Expires: `2026-08-04T00:00:00+07:00`.",
            "- Legacy unmanaged task IDs: none.",
            "## Active Task Checklist",
            "- None.",
            "## Previous-Day Closeout",
            "- Source date: `2026-08-01`.",
            "- Completed: stale task.",
            "- Carried forward: none.",
            "- Purge after: `2026-08-05T00:00:00+07:00`.",
            "## Superseded task history",
        ],
    )

    errors = module._check_short_checklist(tmp_path)

    assert "short_previous_day_source_mismatch" in errors
    assert "short_previous_day_purge_mismatch" in errors
    assert "short_contains_superseded_history" in errors


def _write_empty_managed_short(tmp_path: Path) -> datetime:
    current = datetime(2026, 8, 3, 8, tzinfo=timezone(timedelta(hours=7)))
    _write_short_fixture(
        tmp_path,
        [
            "# Project Memory Short",
            "",
            "## Lifecycle",
            "",
            "- Opened: `2026-08-03`.",
            "- Expires: `2026-08-04T00:00:00+07:00`.",
            "- Legacy unmanaged task IDs: none.",
            "",
            "## Active Task Checklist",
            "",
            "- None.",
            "",
            "## Previous-Day Closeout",
            "",
            "- Source date: `2026-08-02`.",
            "- Completed: none.",
            "- Carried forward: none.",
            "- Purge after: `2026-08-04T00:00:00+07:00`.",
        ],
    )
    return current


def _create_managed_fixture(tmp_path: Path, current: datetime):
    manager = _load_module("managed_short_fixture", TASK_MANAGER)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = ledger.create(
        task_id="LONG-20260803-01",
        title="cross-day fixture",
        prompt="Resume a long task without repeating completed work.",
        acceptance="The active capsule survives rollover byte-identically.",
        skills=["agent-harness-construction", "project-state-steward"],
        steps=[
            {
                "step_id": "LONG-1",
                "summary": "Continue bounded work.",
                "next_action": "Resume from this exact checkpoint.",
            }
        ],
        active_step="LONG-1",
        owner_session="validator-session-0001",
        owner_token="validator-owner-token-0123456789",
        worktree=tmp_path,
        now=current,
    )
    return manager, ledger, created


def test_short_checklist_accepts_cross_day_managed_resume_capsule(
    tmp_path: Path,
) -> None:
    current = _write_empty_managed_short(tmp_path)
    manager, ledger, created = _create_managed_fixture(tmp_path, current)
    before = ledger.inspect(created["task_id"], now=current)

    ledger.rollover(now=current + timedelta(days=1, minutes=1))

    validator = _load_module("cross_day_validator", GOVERNANCE_VALIDATOR)
    assert validator._check_short_checklist(tmp_path) == []
    after = ledger.inspect(created["task_id"], now=current + timedelta(days=1))
    assert after["raw_block_sha256"] == before["raw_block_sha256"]
    assert manager.validate_managed_block(
        manager._task_span(ledger._read(), created["task_id"])["block"]
    ) == []


def test_short_checklist_rejects_managed_block_hash_drift(tmp_path: Path) -> None:
    current = _write_empty_managed_short(tmp_path)
    _, ledger, created = _create_managed_fixture(tmp_path, current)
    path = tmp_path / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md"
    text = path.read_bytes().decode("utf-8")
    text = text.replace("Continue bounded work.", "Continue altered work.")
    path.write_bytes(text.encode("utf-8"))

    validator = _load_module("hash_drift_validator", GOVERNANCE_VALIDATOR)
    errors = validator._check_short_checklist(tmp_path)

    expected = f"short_managed_task_invalid:{created['task_id']}:block_hash_mismatch"
    assert expected in errors


def test_short_daily_budget_excludes_bounded_managed_capsules(
    tmp_path: Path,
) -> None:
    current = _write_empty_managed_short(tmp_path)
    manager = _load_module("managed_budget_fixture", TASK_MANAGER)
    ledger = manager.ShortMemoryLedger(tmp_path)
    for index in range(1, 11):
        ledger.create(
            task_id=f"LOAD-20260803-{index:02d}",
            title=f"concurrent task {index}",
            prompt="Retain a bounded concurrent resume capsule.",
            acceptance="The daily surface remains bounded independently.",
            skills=["agent-harness-construction"],
            steps=[
                {
                    "step_id": f"LOAD{index}-1",
                    "summary": "Continue bounded work.",
                    "next_action": "Resume this exact task.",
                }
            ],
            active_step=f"LOAD{index}-1",
            owner_session=f"validator-session-{index:04d}",
            owner_token=f"validator-owner-token-{index:04d}-0123456789",
            worktree=tmp_path,
            now=current,
        )

    validator = _load_module("managed_budget_validator", GOVERNANCE_VALIDATOR)
    errors = validator._check_short_checklist(tmp_path)

    assert "short_exceeds_250_line_budget" not in errors
    assert errors == []


def test_method_transition_cannot_skip_gate() -> None:
    spec = importlib.util.spec_from_file_location("memory_contract", VALIDATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transition = {
        "from_state": "IMPLEMENTED",
        "to_state": "VALIDATED",
        "timestamp": "2026-07-31T12:00:00+07:00",
        "git_sha": "abc123",
        "dirty_worktree": False,
        "config_hash": "config-sha",
        "input_hashes": ["data-sha"],
        "evaluator": "focused-tests",
        "evidence_class": "RUN_VERIFIED",
        "gate_results": ["pass"],
        "limitations": ["fixture"],
        "authority": "test",
    }

    errors = module._valid_method_transition(
        transition,
        [
            "PROPOSED",
            "DESIGNED",
            "IMPLEMENTED",
            "DEV_PASS",
            "VALIDATED",
            "FROZEN",
            "PROMOTED",
        ],
        {"REJECTED", "BLOCKED", "SUPERSEDED", "NOT_REPRODUCIBLE"},
    )

    assert "transition_skips_gate:IMPLEMENTED->VALIDATED" in errors


def test_incomplete_supported_claim_is_held() -> None:
    spec = importlib.util.spec_from_file_location("memory_contract", VALIDATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    status, errors = module.evaluate_claim(
        {
            "claim_id": "fixture",
            "status": "SUPPORTED",
            "claim_text": "unsupported fixture claim",
            "scope": "fixture",
        }
    )

    assert status == "HOLD_INCOMPLETE_LINEAGE"
    assert errors


def test_error_observation_requires_recovery_fields() -> None:
    spec = importlib.util.spec_from_file_location("memory_contract", VALIDATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors = module.validate_observation(
        {
            "status": "error",
            "summary": "failed",
            "next_actions": [],
            "artifacts": [],
        }
    )

    assert "error_observation_missing:root_cause_hint" in errors
    assert "error_observation_missing:safe_retry" in errors
    assert "error_observation_missing:stop_condition" in errors

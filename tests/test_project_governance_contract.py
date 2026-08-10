from __future__ import annotations

import importlib.util
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

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


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


def _tracked_identity_fixture(tmp_path: Path, line_ending: str = "\n"):
    _run_git(tmp_path, "init")
    _run_git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _run_git(tmp_path, "config", "user.name", "Governance Fixture")
    checkout_eol = "crlf" if line_ending == "\r\n" else "lf"
    (tmp_path / ".gitattributes").write_text(
        f"governed.txt text eol={checkout_eol}\n",
        encoding="utf-8",
    )
    (tmp_path / "governed.txt").write_text(
        f"first{line_ending}second{line_ending}",
        encoding="utf-8",
        )
    _run_git(tmp_path, "add", ".gitattributes", "governed.txt")
    _run_git(tmp_path, "commit", "-m", "fixture")
    blob_oid = _run_git(tmp_path, "rev-parse", "HEAD:governed.txt").stdout.strip()
    validator = _load_module("tracked_identity_validator", GOVERNANCE_VALIDATOR)
    return validator, blob_oid


def _tracked_identity_errors(
    validator,
    root: Path,
    blob_oid: str,
    **overrides: str,
) -> list[str]:
    identity = {
        "relative_path": "governed.txt",
        "expected_blob_oid": blob_oid,
        "git_reference": "HEAD",
        "error_prefix": "tracked",
    }
    identity.update(overrides)
    return validator._validate_git_tracked_file_identity(root, **identity)


def test_tracked_identity_accepts_clean_lf_checkout(tmp_path: Path) -> None:
    validator, blob_oid = _tracked_identity_fixture(tmp_path)

    assert _tracked_identity_errors(validator, tmp_path, blob_oid) == []


def test_tracked_identity_accepts_clean_crlf_checkout(tmp_path: Path) -> None:
    validator, blob_oid = _tracked_identity_fixture(tmp_path, line_ending="\r\n")
    path = tmp_path / "governed.txt"

    assert b"\r\n" in path.read_bytes()
    assert _run_git(tmp_path, "diff", "--quiet", "--", "governed.txt")
    assert _tracked_identity_errors(validator, tmp_path, blob_oid) == []


def test_tracked_identity_rejects_one_character_edit(tmp_path: Path) -> None:
    validator, blob_oid = _tracked_identity_fixture(tmp_path)
    (tmp_path / "governed.txt").write_text("first\nthird\n", encoding="utf-8")

    errors = _tracked_identity_errors(validator, tmp_path, blob_oid)

    assert "tracked_git_unstaged_change" in errors


def test_tracked_identity_rejects_added_line(tmp_path: Path) -> None:
    validator, blob_oid = _tracked_identity_fixture(tmp_path)
    (tmp_path / "governed.txt").write_text(
        "first\nsecond\nthird\n",
        encoding="utf-8",
    )

    assert "tracked_git_unstaged_change" in _tracked_identity_errors(
        validator,
        tmp_path,
        blob_oid,
    )


def test_tracked_identity_rejects_deleted_line(tmp_path: Path) -> None:
    validator, blob_oid = _tracked_identity_fixture(tmp_path)
    (tmp_path / "governed.txt").write_text("first\n", encoding="utf-8")

    assert "tracked_git_unstaged_change" in _tracked_identity_errors(
        validator,
        tmp_path,
        blob_oid,
    )


def test_tracked_identity_rejects_staged_change(tmp_path: Path) -> None:
    validator, blob_oid = _tracked_identity_fixture(tmp_path)
    (tmp_path / "governed.txt").write_text("first\nthird\n", encoding="utf-8")
    _run_git(tmp_path, "add", "governed.txt")

    assert "tracked_git_staged_change" in _tracked_identity_errors(
        validator,
        tmp_path,
        blob_oid,
    )


def test_tracked_identity_rejects_wrong_expected_blob(tmp_path: Path) -> None:
    validator, blob_oid = _tracked_identity_fixture(tmp_path)

    errors = _tracked_identity_errors(
        validator,
        tmp_path,
        blob_oid,
        expected_blob_oid="0" * 40,
    )

    assert "tracked_git_blob_mismatch" in errors
    assert "tracked_git_head_blob_mismatch" in errors


def test_tracked_identity_rejects_wrong_reference_or_path(tmp_path: Path) -> None:
    validator, blob_oid = _tracked_identity_fixture(tmp_path)

    reference_errors = _tracked_identity_errors(
        validator,
        tmp_path,
        blob_oid,
        git_reference="missing-reference",
    )
    path_errors = _tracked_identity_errors(
        validator,
        tmp_path,
        blob_oid,
        relative_path="missing.txt",
    )

    assert "tracked_git_reference_unresolved" in reference_errors
    assert "tracked_git_file_missing" in path_errors


def test_external_raw_identity_rejects_changed_byte(tmp_path: Path) -> None:
    validator = _load_module("external_raw_validator", GOVERNANCE_VALIDATOR)
    artifact = tmp_path / "external.artifact"
    artifact.write_bytes(b"external-v1\x00")
    expected = validator._sha256(artifact)
    artifact.write_bytes(b"external-v1\x01")

    assert validator._validate_raw_file_identity(
        artifact,
        expected_sha256=expected,
        error_prefix="external",
    ) == ["external_raw_hash_mismatch"]


def test_external_raw_identity_accepts_unchanged_artifact(tmp_path: Path) -> None:
    validator = _load_module("external_raw_unchanged_validator", GOVERNANCE_VALIDATOR)
    artifact = tmp_path / "external.artifact"
    artifact.write_bytes(b"external-v1\x00")

    assert validator._validate_raw_file_identity(
        artifact,
        expected_sha256=validator._sha256(artifact),
        error_prefix="external",
    ) == []


def test_binary_artifact_remains_raw_byte_bound(tmp_path: Path) -> None:
    validator = _load_module("binary_raw_validator", GOVERNANCE_VALIDATOR)
    artifact = tmp_path / "external.bin"
    artifact.write_bytes(b"\x00\r\n\xff\x10")
    expected = validator._sha256(artifact)
    artifact.write_bytes(b"\x00\n\xff\x10")

    assert validator._validate_raw_file_identity(
        artifact,
        expected_sha256=expected,
        error_prefix="binary",
    ) == ["binary_raw_hash_mismatch"]


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
    validator = _load_module("managed_budget_validator", GOVERNANCE_VALIDATOR)
    for index in range(1, 81):
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
        checklist = validator._markdown_section(
            ledger._read(),
            "Active Task Checklist",
        )
        assert checklist is not None
        if len(checklist.splitlines()) > 1200:
            break
    else:
        raise AssertionError("fixture did not exceed the aggregate checklist budget")

    errors = validator._check_short_checklist(tmp_path)

    assert len(checklist.splitlines()) > 1200
    assert all(
        len(span["block"].splitlines()) <= 120
        for span in manager.task_spans(ledger._read())
    )
    assert "short_exceeds_250_line_budget" not in errors
    assert "short_checklist_exceeds_1200_line_budget" not in errors
    assert errors == []


def test_short_checklist_aggregate_budget_retains_unmanaged_surface(
    tmp_path: Path,
) -> None:
    _write_empty_managed_short(tmp_path)
    path = tmp_path / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md"
    filler = "\n".join(f"- Unmanaged daily note: {index}." for index in range(1201))
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("- None.\n", f"- None.\n{filler}\n", 1),
        encoding="utf-8",
    )

    validator = _load_module("unmanaged_budget_validator", GOVERNANCE_VALIDATOR)

    assert "short_checklist_exceeds_1200_line_budget" in validator._check_short_checklist(
        tmp_path
    )


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


def _short_memory_lines(
    current: datetime,
    *,
    legacy_task_ids: str = "none",
) -> list[str]:
    previous_day = (current - timedelta(days=1)).date().isoformat()
    expires = current + timedelta(days=1)
    return [
        "# Project Memory Short",
        "",
        "## Lifecycle",
        "",
        f"- Opened: `{current.date().isoformat()}`.",
        f"- Expires: `{expires.isoformat()}`.",
        f"- Legacy unmanaged task IDs: {legacy_task_ids}.",
        "",
        "## Active Task Checklist",
        "",
        "- None.",
        "",
        "## Previous-Day Closeout",
        "",
        f"- Source date: `{previous_day}`.",
        "- Completed: none.",
        "- Carried forward: none.",
        f"- Purge after: `{expires.isoformat()}`.",
    ]


def _write_current_short_memory(root: Path, current: datetime) -> None:
    _write_short_fixture(root, _short_memory_lines(current))


def _coordination_worktree_fixture(tmp_path: Path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _run_git(canonical, "init")
    _run_git(canonical, "config", "user.email", "fixture@example.invalid")
    _run_git(canonical, "config", "user.name", "Governance Fixture")
    historical = datetime(2026, 8, 3, 8, tzinfo=timezone(timedelta(hours=7)))
    current = historical + timedelta(days=2)
    _write_current_short_memory(canonical, historical)
    _run_git(canonical, "add", ".agents/memory/01_PROJECT_MEMORY_SHORT.md")
    _run_git(canonical, "commit", "-m", "historical short memory")
    fresh = tmp_path / "fresh"
    _run_git(canonical, "worktree", "add", "--detach", str(fresh), "HEAD")
    _write_current_short_memory(canonical, current)
    validator = _load_module("coordination_validator", GOVERNANCE_VALIDATOR)
    manager = _load_module("coordination_manager", TASK_MANAGER)
    return canonical, fresh, validator, manager, current


def _active_short_state(validator, root: Path, current: datetime):
    return validator._check_active_short_memory_state(root, current)


def test_active_short_memory_uses_canonical_shared_coordination_root(
    tmp_path: Path,
) -> None:
    canonical, fresh, validator, manager, current = _coordination_worktree_fixture(
        tmp_path
    )

    canonical_state = _active_short_state(validator, canonical, current)
    fresh_state = _active_short_state(validator, fresh, current)

    expected_ledger = canonical / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md"
    assert canonical_state["errors"] == []
    assert fresh_state["errors"] == []
    assert manager.resolve_coordination_root(fresh) == canonical
    assert fresh_state["coordination_root"] == str(canonical)
    assert fresh_state["active_ledger_path"] == str(expected_ledger)


def test_noncanonical_tracked_short_memory_snapshot_remains_static_governed(
    tmp_path: Path,
) -> None:
    _, fresh, validator, _, current = _coordination_worktree_fixture(tmp_path)
    snapshot = fresh / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md"
    snapshot.write_text(snapshot.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")

    state = _active_short_state(validator, fresh, current)

    assert "short_snapshot_git_unstaged_change" in state["errors"]


def test_canonical_stale_and_over_budget_state_fail_from_each_worktree(
    tmp_path: Path,
) -> None:
    canonical, fresh, validator, _, current = _coordination_worktree_fixture(tmp_path)
    stale = current - timedelta(days=2)
    _write_current_short_memory(canonical, stale)

    for root in (canonical, fresh):
        state = _active_short_state(validator, root, current)
        assert state["short_ttl"]["status"] == "RESET_REQUIRED"
        assert "short_memory_expired" in state["errors"]

    task_id = "BUDGET-20260805-01"
    lines = _short_memory_lines(current, legacy_task_ids=f"`{task_id}`")
    checklist_end = lines.index("- None.")
    task_lines = [
        f"### {task_id} - oversized fixture",
        "- Prompt: prove active task budgets use canonical state.",
        "- Status: `IN_PROGRESS`.",
        f"- Opened: `{current.isoformat()}`.",
        "- Acceptance: validator fails the canonical oversized task.",
        "- Skills: `project-state-steward`.",
        "- [ ] `BUDGET-1` `[IN_PROGRESS]` Keep the fixture active.",
        "  - Next: validate the canonical ledger.",
        *[f"- Padding: {index}." for index in range(121)],
    ]
    lines[checklist_end : checklist_end + 1] = task_lines
    _write_short_fixture(canonical, lines)

    for root in (canonical, fresh):
        state = _active_short_state(validator, root, current)
        assert f"short_task_exceeds_120_line_budget:{task_id}" in state["errors"]


def test_terminal_task_history_does_not_consume_active_task_budget(
    tmp_path: Path,
) -> None:
    canonical, fresh, validator, _, current = _coordination_worktree_fixture(tmp_path)
    task_id = "TERMINAL-20260805-01"
    lines = _short_memory_lines(current, legacy_task_ids=f"`{task_id}`")
    checklist_end = lines.index("- None.")
    task_lines = [
        f"### {task_id} - retained terminal history",
        "- Prompt: retain a completed task until rollover.",
        "- Status: `DONE`.",
        f"- Opened: `{current.isoformat()}`.",
        "- Acceptance: terminal history remains structurally valid.",
        "- Skills: `project-state-steward`.",
        "- [x] `TERMINAL-1` `[DONE]` Preserve the completion record.",
        "  - Evidence: completion was verified before the terminal checkpoint.",
        *[f"- Historical detail: {index}." for index in range(121)],
    ]
    lines[checklist_end : checklist_end + 1] = task_lines
    _write_short_fixture(canonical, lines)

    for root in (canonical, fresh):
        state = _active_short_state(validator, root, current)
        assert f"short_task_exceeds_120_line_budget:{task_id}" not in state["errors"]
        assert state["errors"] == []


def test_worktree_shadow_and_missing_canonical_ledger_fail_closed(
    tmp_path: Path,
) -> None:
    canonical, fresh, validator, _, current = _coordination_worktree_fixture(tmp_path)
    stale = current - timedelta(days=2)
    _write_current_short_memory(canonical, stale)
    _write_current_short_memory(fresh, current + timedelta(days=30))

    shadow_state = _active_short_state(validator, fresh, current)

    assert shadow_state["active_ledger_path"] == str(
        canonical / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md"
    )
    assert "short_memory_expired" in shadow_state["errors"]
    assert "short_snapshot_git_unstaged_change" in shadow_state["errors"]

    (canonical / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md").unlink()
    missing_state = _active_short_state(validator, fresh, current)

    assert "short_coordination_short_memory_missing" in missing_state["errors"]


def test_coordination_root_rejects_an_unauthorized_common_directory(
    tmp_path: Path,
) -> None:
    manager = _load_module("unauthorized_root_manager", TASK_MANAGER)
    worktree = tmp_path / "worktree"
    candidate = tmp_path / "candidate"
    other = tmp_path / "other"
    worktree.mkdir()
    (candidate / ".git").mkdir(parents=True)
    (other / ".git").mkdir(parents=True)

    def fake_run(command, **_kwargs):
        root = command[2]
        if root == str(worktree):
            return subprocess.CompletedProcess(command, 0, f"{candidate / '.git'}\n", "")
        if command[-1] == "--show-toplevel":
            return subprocess.CompletedProcess(command, 0, f"{candidate}\n", "")
        return subprocess.CompletedProcess(command, 0, f"{other / '.git'}\n", "")

    with patch.object(manager.subprocess, "run", side_effect=fake_run):
        try:
            manager.resolve_coordination_root(worktree)
        except manager.LedgerError as exc:
            assert exc.code == "coordination_root_unauthorized"
        else:
            raise AssertionError("unauthorized coordination root was accepted")


def test_managed_checkpoint_is_visible_from_a_fresh_worktree(
    tmp_path: Path,
) -> None:
    canonical, fresh, validator, manager, current = _coordination_worktree_fixture(
        tmp_path
    )
    ledger = manager.ShortMemoryLedger(manager.resolve_coordination_root(fresh))
    before = ledger.memory_path.read_bytes()
    owner_token = "coordination-owner-token-0123456789"
    created = ledger.create(
        task_id="COORD-20260805-01",
        title="cross-worktree checkpoint fixture",
        prompt="Prove the canonical ledger is visible from a fresh worktree.",
        acceptance="The checkpoint updates only the canonical active ledger.",
        skills=["project-state-steward"],
        steps=[
            {
                "step_id": "COORD-1",
                "summary": "Checkpoint the fixture.",
                "next_action": "Record completion evidence.",
            }
        ],
        active_step="COORD-1",
        owner_session="coordination-session-0001",
        owner_token=owner_token,
        worktree=fresh,
        now=current,
    )
    checkpointed = ledger.checkpoint(
        task_id=created["task_id"],
        step_id="COORD-1",
        step_status="DONE",
        evidence="Canonical checkpoint completed.",
        next_action=None,
        owner_session=created["owner_session"],
        owner_token=owner_token,
        worktree=fresh,
        expected_revision=created["revision"],
        expected_block_sha256=created["block_sha256"],
        now=current + timedelta(minutes=1),
    )

    state = _active_short_state(validator, fresh, current + timedelta(minutes=1))

    assert ledger.memory_path == canonical / ".agents" / "memory" / "01_PROJECT_MEMORY_SHORT.md"
    assert ledger.memory_path.read_bytes() != before
    assert checkpointed["task_status"] == "DONE"
    assert state["errors"] == []

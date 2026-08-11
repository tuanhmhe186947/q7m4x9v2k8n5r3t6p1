from __future__ import annotations

import importlib.util
import json
import re
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
    / "manage_short_memory.py"
)
OWNER_TOKEN = "test-owner-token-0123456789"
OTHER_TOKEN = "other-owner-token-0123456789"
TZ = timezone(timedelta(hours=7))
NOW = datetime(2026, 8, 3, 8, 0, tzinfo=TZ)


def _load_manager() -> ModuleType:
    spec = importlib.util.spec_from_file_location("short_memory_manager", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manager() -> ModuleType:
    return _load_manager()


def _write_memory(root: Path, opened: datetime) -> None:
    memory = root / ".agents" / "memory"
    memory.mkdir(parents=True)
    expires = datetime.combine(
        opened.date() + timedelta(days=1),
        datetime.min.time(),
        tzinfo=opened.tzinfo,
    )
    previous = opened.date() - timedelta(days=1)
    text = "\n".join(
        [
            "# Project Memory Short",
            "",
            "## Lifecycle",
            "",
            f"- Opened: `{opened.date().isoformat()}`.",
            f"- Expires: `{expires.isoformat()}`.",
            "- Legacy unmanaged task IDs: none.",
            "",
            "## Active Task Checklist",
            "",
            "- None.",
            "",
            "## Previous-Day Closeout",
            "",
            f"- Source date: `{previous.isoformat()}`.",
            "- Completed: none.",
            "- Carried forward: none.",
            f"- Purge after: `{expires.isoformat()}`.",
            "",
        ]
    )
    (memory / "01_PROJECT_MEMORY_SHORT.md").write_text(text, encoding="utf-8")


def _create_task(
    ledger: object,
    task_id: str,
    owner: str = "session-alpha-0001",
    token: str = OWNER_TOKEN,
    now: datetime = NOW,
    step_count: int = 2,
) -> dict[str, object]:
    prefix = task_id.rsplit("-", 2)[0]
    steps = [
        {
            "step_id": f"{prefix}-{index}",
            "summary": f"Complete phase {index}.",
            "next_action": f"Run bounded phase {index}.",
        }
        for index in range(1, step_count + 1)
    ]
    return ledger.create(
        task_id=task_id,
        title="managed fixture",
        prompt="Preserve task ownership across sessions.",
        acceptance="Every checkpoint is atomic and independently recoverable.",
        skills=["agent-harness-construction", "project-state-steward"],
        steps=steps,
        active_step=steps[0]["step_id"],
        owner_session=owner,
        owner_token=token,
        worktree=ledger.root,
        lease_seconds=300,
        now=now,
    )


def _checkpoint(
    ledger: object,
    snapshot: dict[str, object],
    step_id: str,
    status: str,
    *,
    token: str = OWNER_TOKEN,
    now: datetime = NOW + timedelta(seconds=10),
) -> dict[str, object]:
    detail = {
        "evidence": "Focused contract test passed.",
        "next_action": None,
    }
    if status not in {"DONE", "CANCELLED"}:
        detail = {
            "evidence": None,
            "next_action": "Resume only this bounded phase.",
        }
    return ledger.checkpoint(
        task_id=snapshot["task_id"],
        step_id=step_id,
        step_status=status,
        owner_session=snapshot["owner_session"],
        owner_token=token,
        worktree=ledger.root,
        expected_revision=snapshot["revision"],
        expected_block_sha256=snapshot["block_sha256"],
        lease_seconds=300,
        now=now,
        **detail,
    )


def test_create_inspect_and_checkpoint_are_cas_guarded(
    tmp_path: Path,
    manager: ModuleType,
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "TASK-20260803-01")

    inspected = ledger.inspect("TASK-20260803-01", now=NOW)
    assert inspected["revision"] == 1
    assert inspected["block_sha256"] == created["block_sha256"]

    done = _checkpoint(ledger, created, "TASK-1", "DONE")
    resumed = _checkpoint(ledger, done, "TASK-2", "IN_PROGRESS")
    assert resumed["revision"] == 3
    assert [step["status"] for step in resumed["steps"]] == [
        "DONE",
        "IN_PROGRESS",
    ]


def test_wrong_token_and_stale_snapshot_fail_closed(
    tmp_path: Path,
    manager: ModuleType,
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "TASK-20260803-01")

    with pytest.raises(manager.LedgerError, match="owner_token_mismatch"):
        _checkpoint(ledger, created, "TASK-1", "DONE", token=OTHER_TOKEN)

    current = _checkpoint(ledger, created, "TASK-1", "DONE")
    with pytest.raises(manager.LedgerError, match="revision_conflict"):
        _checkpoint(ledger, created, "TASK-2", "IN_PROGRESS")
    stale_hash = dict(current)
    stale_hash["block_sha256"] = "0" * 64
    with pytest.raises(manager.LedgerError, match="block_cas_conflict"):
        _checkpoint(ledger, stale_hash, "TASK-2", "IN_PROGRESS")


def test_owner_can_reconcile_only_the_inspected_drift(
    tmp_path: Path,
    manager: ModuleType,
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "TASK-20260803-01")
    text = ledger.memory_path.read_bytes().decode("utf-8")
    text = text.replace("Complete phase 1.", "Complete phase A.")
    ledger.memory_path.write_bytes(text.encode("utf-8"))
    block = manager._task_span(text, created["task_id"])["block"]
    raw_hash = manager.raw_block_sha256(block)

    with pytest.raises(manager.LedgerError, match="block_hash_mismatch"):
        ledger.inspect(created["task_id"], now=NOW)
    with pytest.raises(manager.LedgerError, match="reconcile_raw_conflict"):
        ledger.reconcile(
            task_id=created["task_id"],
            owner_session=created["owner_session"],
            owner_token=OWNER_TOKEN,
            worktree=tmp_path,
            expected_revision=created["revision"],
            expected_recorded_block_sha256=created["block_sha256"],
            expected_raw_sha256="0" * 64,
            reason="test-drift",
            now=NOW + timedelta(seconds=10),
        )
    with pytest.raises(manager.LedgerError, match="owner_token_mismatch"):
        ledger.reconcile(
            task_id=created["task_id"],
            owner_session=created["owner_session"],
            owner_token=OTHER_TOKEN,
            worktree=tmp_path,
            expected_revision=created["revision"],
            expected_recorded_block_sha256=created["block_sha256"],
            expected_raw_sha256=raw_hash,
            reason="test-drift",
            now=NOW + timedelta(seconds=10),
        )

    reconciled = ledger.reconcile(
        task_id=created["task_id"],
        owner_session=created["owner_session"],
        owner_token=OWNER_TOKEN,
        worktree=tmp_path,
        expected_revision=created["revision"],
        expected_recorded_block_sha256=created["block_sha256"],
        expected_raw_sha256=raw_hash,
        reason="test-drift",
        now=NOW + timedelta(seconds=10),
    )
    assert reconciled["revision"] == 2
    assert reconciled["ownership_reason"] == "reconcile:test-drift"
    assert ledger.inspect(created["task_id"], now=NOW)["managed"] is True


def test_checkpoint_preserves_every_other_task_byte(
    tmp_path: Path,
    manager: ModuleType,
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    first = _create_task(ledger, "FIRST-20260803-01")
    _create_task(
        ledger,
        "SECOND-20260803-01",
        owner="session-beta-0002",
        token=OTHER_TOKEN,
    )
    before = ledger.inspect("SECOND-20260803-01", now=NOW)

    _checkpoint(ledger, first, "FIRST-1", "DONE")

    after = ledger.inspect("SECOND-20260803-01", now=NOW)
    assert after["raw_block_sha256"] == before["raw_block_sha256"]


def _cli_create(root: Path, task_id: str, owner: str) -> list[str]:
    prefix = task_id.rsplit("-", 2)[0]
    return [
        sys.executable,
        str(SCRIPT),
        "--coordination-root",
        str(root),
        "create",
        "--task-id",
        task_id,
        "--title",
        "process fixture",
        "--prompt",
        "Create one process-owned task.",
        "--acceptance",
        "Exactly one valid task block is created.",
        "--skill",
        "agent-harness-construction",
        "--step",
        f"{prefix}-1|Create task|Inspect the resulting task.",
        "--active-step",
        f"{prefix}-1",
        "--owner-session",
        owner,
        "--worktree",
        str(root),
    ]


def _run_parallel(commands: list[list[str]]) -> list[subprocess.CompletedProcess[str]]:
    processes = [
        subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        for command in commands
    ]
    results = []
    for process in processes:
        stdout, _ = process.communicate(timeout=20)
        results.append(
            subprocess.CompletedProcess(process.args, process.returncode, stdout)
        )
    return results


def _cli_recover(
    root: Path,
    snapshot: dict[str, object],
    token: str,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--coordination-root",
        str(root),
        "recover",
        "--task-id",
        str(snapshot["task_id"]),
        "--expected-owner-session",
        str(snapshot["owner_session"]),
        "--expected-revision",
        str(snapshot["revision"]),
        "--expected-block-sha256",
        str(snapshot["block_sha256"]),
        "--worktree",
        str(root),
        "--reason",
        "two-process same-session recovery race",
        "--new-owner-token",
        token,
    ]


def test_two_processes_create_distinct_tasks_without_lost_update(
    tmp_path: Path,
    manager: ModuleType,
) -> None:
    current = datetime.now(manager._project_timezone())
    _write_memory(tmp_path, current)
    day = current.strftime("%Y%m%d")
    results = _run_parallel(
        [
            _cli_create(tmp_path, f"PROC-A-{day}-01", "process-owner-alpha"),
            _cli_create(tmp_path, f"PROC-B-{day}-01", "process-owner-beta"),
        ]
    )

    assert [result.returncode for result in results] == [0, 0]
    text = (tmp_path / manager.MEMORY_RELATIVE).read_bytes().decode("utf-8")
    assert {span["task_id"] for span in manager.task_spans(text)} == {
        f"PROC-A-{day}-01",
        f"PROC-B-{day}-01",
    }


def test_same_id_process_race_yields_one_conflict_without_corruption(
    tmp_path: Path,
    manager: ModuleType,
) -> None:
    current = datetime.now(manager._project_timezone())
    _write_memory(tmp_path, current)
    task_id = f"RACE-{current:%Y%m%d}-01"
    results = _run_parallel(
        [
            _cli_create(tmp_path, task_id, "process-owner-alpha"),
            _cli_create(tmp_path, task_id, "process-owner-beta"),
        ]
    )

    assert sorted(result.returncode for result in results) == [0, 2]
    errors = [json.loads(result.stdout) for result in results if result.returncode]
    assert errors[0]["summary"] == "task_id_collision"
    text = (tmp_path / manager.MEMORY_RELATIVE).read_bytes().decode("utf-8")
    blocks = [span for span in manager.task_spans(text) if span["task_id"] == task_id]
    assert len(blocks) == 1
    assert manager.validate_managed_block(blocks[0]["block"]) == []


def test_two_process_same_session_recovery_has_one_winner(
    tmp_path: Path,
    manager: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(manager.RUNTIME_SESSION_ENV, "thread-race-0001")
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "RECOVER-20260803-01")

    results = _run_parallel(
        [
            _cli_recover(tmp_path, created, OWNER_TOKEN + "-replacement-a"),
            _cli_recover(tmp_path, created, OWNER_TOKEN + "-replacement-b"),
        ]
    )

    assert sorted(result.returncode for result in results) == [0, 2]
    errors = [json.loads(result.stdout) for result in results if result.returncode]
    assert errors[0]["summary"] == "recovery_cas_conflict"
    inspected = ledger.inspect(created["task_id"], now=NOW)
    assert inspected["revision"] == 2
    assert inspected["ownership_audit_events"] == 1


def test_takeover_requires_expired_lease(
    tmp_path: Path,
    manager: ModuleType,
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "TASK-20260803-01")
    takeover_args = {
        "task_id": created["task_id"],
        "expected_owner_session": created["owner_session"],
        "expected_revision": created["revision"],
        "expected_block_sha256": created["block_sha256"],
        "new_owner_session": "session-takeover-0003",
        "new_owner_token": OTHER_TOKEN,
        "new_worktree": tmp_path,
        "reason": "expired-owner-recovery",
        "lease_seconds": 300,
    }

    with pytest.raises(manager.LedgerError, match="takeover_lease_active"):
        ledger.takeover(**takeover_args, now=NOW + timedelta(seconds=299))

    taken = ledger.takeover(**takeover_args, now=NOW + timedelta(seconds=301))
    assert taken["owner_session"] == "session-takeover-0003"
    assert taken["previous_owner"] == "session-alpha-0001"
    assert taken["revision"] == 2


def test_same_runtime_session_recovers_lost_token_during_active_lease(
    tmp_path: Path,
    manager: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(manager.RUNTIME_SESSION_ENV, "thread-alpha-0001")
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "RECOVER-20260803-01")

    recovered = ledger.recover_same_session(
        task_id=created["task_id"],
        expected_owner_session=created["owner_session"],
        expected_revision=created["revision"],
        expected_block_sha256=created["block_sha256"],
        worktree=tmp_path,
        reason="owner process crashed and token was lost",
        new_owner_token=OTHER_TOKEN,
        lease_seconds=300,
        now=NOW + timedelta(seconds=30),
    )

    assert recovered["revision"] == 2
    assert recovered["owner_runtime_session"] == "thread-alpha-0001"
    assert recovered["ownership_audit_events"] == 1
    assert recovered["lease_active"] is True
    with pytest.raises(manager.LedgerError, match="owner_token_mismatch"):
        _checkpoint(ledger, recovered, "RECOVER-1", "DONE", token=OWNER_TOKEN)


def test_different_runtime_cannot_claim_same_session_recovery(
    tmp_path: Path,
    manager: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(manager.RUNTIME_SESSION_ENV, "thread-alpha-0001")
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "RECOVER-20260803-01")
    monkeypatch.setenv(manager.RUNTIME_SESSION_ENV, "thread-beta-0002")

    with pytest.raises(manager.LedgerError, match="recovery_runtime_mismatch"):
        ledger.recover_same_session(
            task_id=created["task_id"],
            expected_owner_session=created["owner_session"],
            expected_revision=created["revision"],
            expected_block_sha256=created["block_sha256"],
            worktree=tmp_path,
            reason="different thread must fail closed",
            new_owner_token=OTHER_TOKEN,
            now=NOW + timedelta(seconds=30),
        )

    unchanged = ledger.inspect(created["task_id"], now=NOW)
    assert unchanged["revision"] == created["revision"]
    assert unchanged["ownership_audit_events"] == 0


def test_admin_takeover_requires_exact_confirmation_and_fresh_cas(
    tmp_path: Path,
    manager: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(manager.RUNTIME_SESSION_ENV, "thread-alpha-0001")
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "ADMIN-20260803-01")
    monkeypatch.setenv(manager.RUNTIME_SESSION_ENV, "thread-admin-0002")
    arguments = {
        "task_id": created["task_id"],
        "confirm_task_id": created["task_id"],
        "confirmation": manager.ADMIN_TAKEOVER_CONFIRMATION,
        "authorization_ref": "user-prompt-2026-08-04",
        "expected_owner_session": created["owner_session"],
        "expected_revision": created["revision"],
        "expected_block_sha256": created["block_sha256"],
        "expected_worktree": tmp_path,
        "new_owner_session": "session-admin-0002",
        "new_worktree": tmp_path,
        "reason": "user authorized recovery after an ambiguous crash",
        "new_owner_token": OTHER_TOKEN,
        "now": NOW + timedelta(seconds=30),
    }

    with pytest.raises(manager.LedgerError, match="admin_confirmation_missing"):
        ledger.administrative_takeover(
            **{**arguments, "confirmation": "yes"}
        )
    with pytest.raises(manager.LedgerError, match="admin_cas_conflict"):
        ledger.administrative_takeover(
            **{**arguments, "expected_block_sha256": "0" * 64}
        )

    taken = ledger.administrative_takeover(**arguments)
    assert taken["revision"] == 2
    assert taken["owner_session"] == "session-admin-0002"
    assert taken["owner_runtime_session"] == "thread-admin-0002"
    assert taken["previous_owner"] == "session-alpha-0001"
    assert taken["ownership_audit_events"] == 1
    assert taken["lease_active"] is True


def test_rollover_retains_active_capsule_and_prunes_completed_task(
    tmp_path: Path,
    manager: ModuleType,
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    active = _create_task(ledger, "ACTIVE-20260803-01")
    completed = _create_task(
        ledger,
        "CLOSED-20260803-01",
        owner="session-beta-0002",
        token=OTHER_TOKEN,
        step_count=1,
    )
    ledger.checkpoint(
        task_id=completed["task_id"],
        step_id="CLOSED-1",
        step_status="DONE",
        evidence="Focused completion test passed.",
        next_action=None,
        owner_session=completed["owner_session"],
        owner_token=OTHER_TOKEN,
        worktree=tmp_path,
        expected_revision=completed["revision"],
        expected_block_sha256=completed["block_sha256"],
        now=NOW + timedelta(minutes=1),
    )
    before = ledger.inspect(active["task_id"], now=NOW)

    result = ledger.rollover(now=NOW + timedelta(days=1, minutes=1))

    assert result["retained_task_ids"] == [active["task_id"]]
    assert result["completed_task_ids"] == [completed["task_id"]]
    after = ledger.inspect(active["task_id"], now=NOW + timedelta(days=1))
    assert after["raw_block_sha256"] == before["raw_block_sha256"]
    assert after["revision"] == before["revision"]
    text = (tmp_path / manager.MEMORY_RELATIVE).read_text(encoding="utf-8")
    active_ids = {span["task_id"] for span in manager.task_spans(text)}
    assert completed["task_id"] not in active_ids
    assert "active tasks remain resume capsules in short memory" in text


def test_rollover_rejects_unmanaged_open_task(
    tmp_path: Path,
    manager: ModuleType,
) -> None:
    _write_memory(tmp_path, NOW)
    path = tmp_path / manager.MEMORY_RELATIVE
    text = path.read_text(encoding="utf-8")
    block = "\n".join(
        [
            "### LEGACY-20260803-01 - unmanaged task",
            "",
            "- Prompt: finish legacy task.",
            "- Status: `IN_PROGRESS`.",
            "- Opened: `2026-08-03T08:00:00+07:00`.",
            "- Acceptance: evidence is complete.",
            "- Skills: `project-state-steward`.",
            "- [ ] `LEGACY-1` `[IN_PROGRESS]` Finish task.",
            "  - Next: adopt task before rollover.",
            "",
        ]
    )
    text = text.replace("- None.\n", block)
    text = text.replace(
        "- Legacy unmanaged task IDs: none.",
        "- Legacy unmanaged task IDs: `LEGACY-20260803-01`.",
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(manager.LedgerError, match="rollover_unmanaged_open_task"):
        manager.ShortMemoryLedger(tmp_path).rollover(
            now=NOW + timedelta(days=1, minutes=1)
        )


def test_process_exit_releases_os_lock(tmp_path: Path, manager: ModuleType) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    _create_task(ledger, "TASK-20260803-01")
    crash_code = "\n".join(
        [
            "import importlib.util, os",
            f"spec = importlib.util.spec_from_file_location('m', {str(SCRIPT)!r})",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            f"lock = module.Path({str(ledger.lock_path)!r})",
            "with module.exclusive_file_lock(lock, 2.0):",
            "    os._exit(0)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", crash_code],
        check=False,
        timeout=10,
    )
    assert result.returncode == 0
    assert ledger.inspect("TASK-20260803-01", now=NOW)["managed"] is True


def test_compaction_preserves_lossless_history_and_unrelated_tasks(
    tmp_path: Path, manager: ModuleType
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    normal = _create_task(ledger, "NORMAL-20260803-01")
    oversized = _create_task(ledger, "OVERSIZE-20260803-01", step_count=60)
    normal_before = ledger.inspect(normal["task_id"], now=NOW)
    text_before = ledger.memory_path.read_bytes().decode("utf-8")
    oversized_before = manager._task_span(text_before, oversized["task_id"])["block"]

    assert "archive_reference" not in normal_before
    assert len(oversized_before.splitlines()) > 120

    compacted = ledger.compact(
        task_id=oversized["task_id"],
        owner_session=oversized["owner_session"],
        owner_token=OWNER_TOKEN,
        worktree=tmp_path,
        expected_revision=oversized["revision"],
        expected_block_sha256=oversized["block_sha256"],
        phase="GENERIC_COMPACTION_TEST",
        blocker=None,
        resume_point="Inspect verified archive before continuing the next bounded step.",
        authority_refs=["tests/test_short_memory_task_manager.py"],
        canonical_sha="abcdef0",
        now=NOW + timedelta(seconds=10),
    )

    assert compacted["task_id"] == oversized["task_id"]
    assert compacted["revision"] == oversized["revision"] + 1
    assert compacted["pre_compaction_revision"] == oversized["revision"]
    assert compacted["pre_compaction_block_sha256"] == oversized["block_sha256"]
    assert compacted["active_task_line_count"] <= 120
    archive_path = tmp_path / compacted["archive_reference"]
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    assert payload["content"] == oversized_before
    assert payload["content_sha256"] == manager.raw_block_sha256(oversized_before)

    normal_after = ledger.inspect(normal["task_id"], now=NOW)
    assert normal_after["raw_block_sha256"] == normal_before["raw_block_sha256"]
    verified = ledger.inspect(oversized["task_id"], now=NOW)
    assert verified["archive_sha256"] == compacted["archive_sha256"]


def test_compaction_refuses_repeat_and_detects_archive_tampering(
    tmp_path: Path, manager: ModuleType
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "ARCHIVE-20260803-01", step_count=60)
    compacted = ledger.compact(
        task_id=created["task_id"],
        owner_session=created["owner_session"],
        owner_token=OWNER_TOKEN,
        worktree=tmp_path,
        expected_revision=created["revision"],
        expected_block_sha256=created["block_sha256"],
        phase="GENERIC_COMPACTION_TEST",
        blocker=None,
        resume_point="Verify archive integrity before resuming.",
        authority_refs=["tests/test_short_memory_task_manager.py"],
        now=NOW + timedelta(seconds=10),
    )

    with pytest.raises(manager.LedgerError, match="task_already_compacted"):
        ledger.compact(
            task_id=created["task_id"],
            owner_session=created["owner_session"],
            owner_token=OWNER_TOKEN,
            worktree=tmp_path,
            expected_revision=compacted["revision"],
            expected_block_sha256=compacted["block_sha256"],
            phase="GENERIC_COMPACTION_TEST",
            blocker=None,
            resume_point="Verify archive integrity before resuming.",
            authority_refs=["tests/test_short_memory_task_manager.py"],
            now=NOW + timedelta(seconds=20),
        )

    archive_path = tmp_path / compacted["archive_reference"]
    archive_path.write_text(
        archive_path.read_text(encoding="utf-8").replace("managed fixture", "tampered"),
        encoding="utf-8",
    )
    with pytest.raises(manager.LedgerError, match="archive_hash_mismatch"):
        ledger.inspect(created["task_id"], now=NOW)


def test_same_session_recovery_compacts_exact_predecessor(
    tmp_path: Path, manager: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(manager.RUNTIME_SESSION_ENV, "thread-alpha-0001")
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(
        ledger,
        "RECOVERY-COMPACT-20260803-01",
        owner="thread-alpha-0001",
        step_count=60,
    )
    original_text = ledger.memory_path.read_bytes().decode("utf-8")
    original_block = manager._task_span(original_text, created["task_id"])["block"]

    compacted = ledger.compact(
        task_id=created["task_id"],
        owner_session=created["owner_session"],
        owner_token=None,
        worktree=tmp_path,
        expected_revision=created["revision"],
        expected_block_sha256=created["block_sha256"],
        phase="GENERIC_RECOVERY_COMPACTION_TEST",
        blocker=None,
        resume_point="Resume only after archive integrity verification.",
        authority_refs=["tests/test_short_memory_task_manager.py"],
        same_session_recovery=True,
        new_owner_token=OTHER_TOKEN,
        now=NOW + timedelta(seconds=10),
    )

    payload = json.loads(
        (tmp_path / compacted["archive_reference"]).read_text(encoding="utf-8")
    )
    assert payload["content"] == original_block
    assert compacted["pre_compaction_revision"] == created["revision"]
    assert compacted["ownership_audit_events"] == 1


def test_compaction_preflight_failure_leaves_no_orphaned_archive(
    tmp_path: Path, manager: ModuleType
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "PREFLIGHT-20260803-01", step_count=60)

    with pytest.raises(manager.LedgerError, match="canonical_sha_invalid"):
        ledger.compact(
            task_id=created["task_id"],
            owner_session=created["owner_session"],
            owner_token=OWNER_TOKEN,
            worktree=tmp_path,
            expected_revision=created["revision"],
            expected_block_sha256=created["block_sha256"],
            phase="GENERIC_COMPACTION_TEST",
            blocker=None,
            resume_point="Do not resume after failed preflight.",
            authority_refs=["tests/test_short_memory_task_manager.py"],
            canonical_sha="not-a-git-hash",
            now=NOW + timedelta(seconds=10),
        )

    assert not (tmp_path / manager.TASK_HISTORY_RELATIVE).exists()
    unchanged = ledger.inspect(created["task_id"], now=NOW)
    assert unchanged["revision"] == created["revision"]
    assert unchanged["block_sha256"] == created["block_sha256"]


def test_compaction_repair_rebuilds_incomplete_continuation_from_archive(
    tmp_path: Path, manager: ModuleType
) -> None:
    _write_memory(tmp_path, NOW)
    ledger = manager.ShortMemoryLedger(tmp_path)
    created = _create_task(ledger, "REPAIR-20260803-01", step_count=60)
    compacted = ledger.compact(
        task_id=created["task_id"],
        owner_session=created["owner_session"],
        owner_token=OWNER_TOKEN,
        worktree=tmp_path,
        expected_revision=created["revision"],
        expected_block_sha256=created["block_sha256"],
        phase="GENERIC_COMPACTION_TEST",
        blocker=None,
        resume_point="Verify archive integrity before resuming.",
        authority_refs=["tests/test_short_memory_task_manager.py"],
        now=NOW + timedelta(seconds=10),
    )
    text = ledger.memory_path.read_bytes().decode("utf-8")
    span = manager._task_span(text, created["task_id"])
    incomplete = re.sub(r"^- Skills:.*\r?\n", "", span["block"], flags=re.MULTILINE)
    repaired_metadata = manager._with_managed_metadata(
        incomplete,
        owner_session=created["owner_session"],
        owner_runtime_session=None,
        owner_token_sha256=manager._token_sha256(OWNER_TOKEN),
        worktree=tmp_path,
        revision=compacted["revision"],
        lease_expires=NOW + timedelta(minutes=5),
    )
    ledger.memory_path.write_bytes(
        manager._replace_task(text, span, repaired_metadata).encode("utf-8")
    )
    incomplete_snapshot = ledger.inspect(created["task_id"], now=NOW)

    repaired = ledger.compact(
        task_id=created["task_id"],
        owner_session=created["owner_session"],
        owner_token=OWNER_TOKEN,
        worktree=tmp_path,
        expected_revision=incomplete_snapshot["revision"],
        expected_block_sha256=incomplete_snapshot["block_sha256"],
        phase="GENERIC_COMPACTION_TEST",
        blocker=None,
        resume_point="Verify archive integrity before resuming.",
        authority_refs=["tests/test_short_memory_task_manager.py"],
        repair_existing=True,
        now=NOW + timedelta(seconds=20),
    )

    repaired_block = manager._task_span(
        ledger.memory_path.read_bytes().decode("utf-8"), created["task_id"]
    )["block"]
    assert "- Skills:" in repaired_block
    assert repaired["revision"] == compacted["revision"] + 1
    assert repaired["archive_reference"] == compacted["archive_reference"]

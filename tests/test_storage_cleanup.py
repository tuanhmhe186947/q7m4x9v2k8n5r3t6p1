from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pig_behavior.storage_cleanup.recycle_bin import (
    FOF_ALLOWUNDO,
    FOF_WANTNUKEWARNING,
    WindowsRecycleBin,
)
from pig_behavior.storage_cleanup.scanner import CleanupPaths, StorageScanner
from pig_behavior.storage_cleanup.service import CleanupError, CleanupService
from pig_behavior.storage_cleanup.web import CommitJobStore, CommitRequest, create_app


class FakeRecycler:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def move(self, path: Path) -> None:
        self.paths.append(path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


class FakeShellOperation:
    def __init__(self) -> None:
        self.flags = 0

    def __call__(self, operation_pointer) -> int:
        self.flags = operation_pointer._obj.fFlags
        return 0


def set_modified(path: Path, timestamp: datetime) -> None:
    seconds = timestamp.timestamp()
    os.utime(path, (seconds, seconds))


def make_scanner(
    tmp_path: Path,
    now: datetime,
    *,
    large_threshold_bytes: int = 512 * 1024 * 1024,
) -> StorageScanner:
    project = tmp_path / "project"
    home = tmp_path / "home"
    local = tmp_path / "local"
    temp = tmp_path / "temp"
    pig_runs = tmp_path / "pig_runs"
    for path in (project, home, local, temp, pig_runs):
        path.mkdir(parents=True)
    paths = CleanupPaths(
        project_root=project,
        home=home,
        local_app_data=local,
        temp_dir=temp,
        pig_runs=pig_runs,
    )
    return StorageScanner(
        paths,
        now=lambda: now,
        large_threshold_bytes=large_threshold_bytes,
    )


def add_session(
    scanner: StorageScanner,
    session_id: str,
    *,
    title: str,
    modified_at: datetime,
) -> Path:
    session_dir = scanner.paths.home / ".codex" / "sessions" / "2026" / "07" / "01"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"rollout-2026-07-01T00-00-00-{session_id}.jsonl"
    path.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    set_modified(path, modified_at)
    index = scanner.paths.home / ".codex" / "session_index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a", encoding="utf-8") as handle:
        handle.write(f'{{"id":"{session_id}","thread_name":"{title}"}}\n')
    return path


def test_scan_labels_old_session_and_protects_recent_session(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    old_id = "019e1cd5-cd91-7c12-88c6-e6b8d2083978"
    new_id = "019fb4d4-520b-76b2-a800-3601ce78a914"
    add_session(
        scanner,
        old_id,
        title="Old session",
        modified_at=now - timedelta(days=30),
    )
    add_session(
        scanner,
        new_id,
        title="Current session",
        modified_at=now - timedelta(hours=2),
    )

    result = scanner.scan()
    sessions = {
        item.display_name: item
        for item in result.items
        if item.category == "codex_sessions"
    }

    assert sessions["Old session"].selectable is True
    assert sessions["Old session"].risk == "caution"
    assert sessions["Old session"].recommendation_level == "review"
    assert sessions["Old session"].importance_level == "medium"
    assert "lịch sử hội thoại" in sessions["Old session"].project_impact
    assert sessions["Current session"].selectable is False
    assert sessions["Current session"].risk == "protected"
    assert sessions["Current session"].recommendation_level == "keep"


def test_session_uses_first_real_user_question_after_agents_context(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    session = add_session(
        scanner,
        "019e1cd5-cd91-7c12-88c6-e6b8d2083978",
        title="Generated title",
        modified_at=now - timedelta(days=30),
    )
    messages = (
        "# AGENTS.md instructions for C:\\project\n<INSTRUCTIONS>rules",
        (
            "# Context from my IDE setup:\n## Open tabs:\n- app.py\n"
            "## My request for Codex:\n"
            "  Làm sao kiểm tra cache UV đang chiếm dung lượng?  "
        ),
    )
    with session.open("a", encoding="utf-8") as handle:
        for message in messages:
            record = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}],
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    set_modified(session, now - timedelta(days=30))

    result = scanner.scan()
    item = next(
        candidate
        for candidate in result.items
        if candidate.category == "codex_sessions"
    )

    assert item.display_name == "Làm sao kiểm tra cache UV đang chiếm dung lượng?"
    assert item.detail == "Tên trong chỉ mục Codex: Generated title"


def test_registered_worktree_is_protected_and_old_orphan_is_selectable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    root = scanner.paths.project_root / ".codex_tmp" / "worktrees"
    active = root / "active_branch"
    orphan = root / "orphan_branch"
    for path in (active, orphan):
        path.mkdir(parents=True)
        payload = path / "payload.bin"
        payload.write_bytes(b"x")
        set_modified(payload, now - timedelta(days=30))
        set_modified(path, now - timedelta(days=30))
    monkeypatch.setattr(
        scanner,
        "_registered_worktrees",
        lambda: {active.resolve(): "feature/active"},
    )

    result = scanner.scan()
    worktrees = {
        item.display_name: item
        for item in result.items
        if item.category == "agent_worktrees"
    }

    assert worktrees["active_branch"].selectable is False
    assert worktrees["active_branch"].detail == "feature/active"
    assert worktrees["active_branch"].recommendation_level == "keep"
    assert worktrees["active_branch"].importance_level == "critical"
    assert result.to_public_dict()["summary"]["protected_bytes"] > 0
    assert worktrees["orphan_branch"].selectable is True
    assert "mồ côi" in worktrees["orphan_branch"].reason
    assert worktrees["orphan_branch"].recommendation_level == "review"


def test_large_scientific_items_are_review_only(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now, large_threshold_bytes=1)
    output = scanner.paths.project_root / "outputs" / "accepted_run"
    output.mkdir(parents=True)
    (output / "evidence.bin").write_bytes(b"evidence")

    hidden = scanner.scan(include_large_review=False)
    visible = scanner.scan(include_large_review=True)

    assert not any(item.category == "large_review" for item in hidden.items)
    review_items = [
        item for item in visible.items if item.category == "large_review"
    ]
    assert len(review_items) == 1
    assert review_items[0].selectable is False
    assert "lineage" in (review_items[0].protected_reason or "")
    assert review_items[0].recommendation_level == "keep"
    assert review_items[0].importance_level == "critical"


def test_recommendations_prioritize_cache_and_flag_pig_lineage(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    cache = scanner.paths.local_app_data / "uv" / "cache"
    cache.mkdir(parents=True)
    wheel = cache / "wheel.bin"
    wheel.write_bytes(b"cache")
    set_modified(wheel, now - timedelta(days=10))
    set_modified(cache, now - timedelta(days=10))

    checkpoint = (
        scanner.paths.project_root
        / ".codex_tmp"
        / "classification_v2_checkpoint_old"
    )
    checkpoint.mkdir(parents=True)
    manifest = checkpoint / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    set_modified(manifest, now - timedelta(days=10))
    set_modified(checkpoint, now - timedelta(days=10))

    payload = scanner.scan().to_public_dict()
    items = {item["display_name"]: item for item in payload["items"]}
    assert items["cache"]["recommendation_level"] == "delete_first"
    assert items["cache"]["importance_level"] == "low"
    assert "code/model" in items["cache"]["recommendation"]
    project_item = items["classification_v2_checkpoint_old"]
    assert project_item["recommendation_level"] == "review"
    assert project_item["recommendation"] == "Cần xác nhận lineage"
    assert project_item["importance_level"] == "high"
    assert payload["summary"]["delete_first_bytes"] > 0
    assert payload["summary"]["lineage_review_bytes"] > 0
    assert payload["summary"]["project_critical_bytes"] > 0


def test_preview_and_commit_recycle_only_scanned_item(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    session = add_session(
        scanner,
        "019e1cd5-cd91-7c12-88c6-e6b8d2083978",
        title="Disposable session",
        modified_at=now - timedelta(days=30),
    )
    recycler = FakeRecycler()
    service = CleanupService(scanner=scanner, recycler=recycler)
    payload = service.scan()
    item = next(
        candidate
        for candidate in payload["items"]
        if candidate["display_name"] == "Disposable session"
    )

    preview = service.preview([str(item["token"])])
    result = service.commit(
        str(preview["confirmation_id"]),
        str(preview["phrase"]),
    )

    assert result["recycled_count"] == 1
    assert recycler.paths == [session.resolve()]
    assert not session.exists()


def test_commit_reports_item_level_progress(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    first = add_session(
        scanner,
        "019e1cd5-cd91-7c12-88c6-e6b8d2083978",
        title="First old chat",
        modified_at=now - timedelta(days=30),
    )
    second = add_session(
        scanner,
        "019e1cd5-cd91-7c12-88c6-e6b8d2083979",
        title="Second old chat",
        modified_at=now - timedelta(days=30),
    )
    service = CleanupService(scanner=scanner, recycler=FakeRecycler())
    payload = service.scan()
    tokens = [
        str(item["token"])
        for item in payload["items"]
        if item["display_name"] in {"First old chat", "Second old chat"}
    ]
    preview = service.preview(tokens)
    events: list[tuple[int, int, str | None, str]] = []
    result = service.commit(
        str(preview["confirmation_id"]),
        str(preview["phrase"]),
        progress=lambda done, total, item, status: events.append(
            (done, total, item.display_name if item else None, status)
        ),
    )

    assert result["recycled_count"] == 2
    assert events[0] == (0, 2, None, "starting")
    assert events[-1][0] == 2
    assert events[-1][1] == 2
    assert events[-1][3] == "recycled"
    assert not first.exists()
    assert not second.exists()


def test_commit_job_exposes_completed_progress(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    session = add_session(
        scanner,
        "019e1cd5-cd91-7c12-88c6-e6b8d2083978",
        title="Job progress chat",
        modified_at=now - timedelta(days=30),
    )
    service = CleanupService(scanner=scanner, recycler=FakeRecycler())
    payload = service.scan()
    token = next(
        str(item["token"])
        for item in payload["items"]
        if item["display_name"] == "Job progress chat"
    )
    preview = service.preview([token])
    jobs = CommitJobStore(service)
    started = jobs.start(
        CommitRequest(
            confirmation_id=str(preview["confirmation_id"]),
            phrase=str(preview["phrase"]),
        )
    )

    for _ in range(50):
        job = jobs.get(str(started["job_id"]))
        if job["status"] == "complete":
            break
        time.sleep(0.01)
    assert job["status"] == "complete"
    assert job["completed_count"] == 1
    assert job["recycled_count"] == 1
    assert not session.exists()


def test_commit_fails_closed_when_item_changes_after_preview(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    session = add_session(
        scanner,
        "019e1cd5-cd91-7c12-88c6-e6b8d2083978",
        title="Changing session",
        modified_at=now - timedelta(days=30),
    )
    recycler = FakeRecycler()
    service = CleanupService(scanner=scanner, recycler=recycler)
    payload = service.scan()
    token = next(
        str(candidate["token"])
        for candidate in payload["items"]
        if candidate["display_name"] == "Changing session"
    )
    preview = service.preview([token])
    session.write_text("changed after preview", encoding="utf-8")

    with pytest.raises(CleanupError, match="changed after the scan"):
        service.commit(
            str(preview["confirmation_id"]),
            str(preview["phrase"]),
        )

    assert recycler.paths == []
    assert session.exists()


def test_protected_item_cannot_be_previewed(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    add_session(
        scanner,
        "019fb4d4-520b-76b2-a800-3601ce78a914",
        title="Recent session",
        modified_at=now - timedelta(minutes=5),
    )
    service = CleanupService(scanner=scanner, recycler=FakeRecycler())
    payload = service.scan()
    token = next(
        str(candidate["token"])
        for candidate in payload["items"]
        if candidate["display_name"] == "Recent session"
    )

    with pytest.raises(CleanupError, match="Protected item"):
        service.preview([token])


def test_local_api_exposes_only_storage_guard_routes(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    service = CleanupService(scanner=scanner, recycler=FakeRecycler())
    app = create_app(service)
    paths = {route.path for route in app.routes}

    assert app.docs_url is None
    assert app.openapi_url is None
    assert "/api/scan" in paths
    assert "/api/recycle/preview" in paths
    assert "/api/recycle/commit" in paths
    assert "/api/recycle/jobs/{job_id}" in paths
    assert "/api/items/{token}/children" in paths
    assert "/tracking/start" not in paths


def test_browse_mints_child_tokens_and_rejects_parent_child_selection(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    scanner = make_scanner(tmp_path, now)
    cache = scanner.paths.project_root / ".pytest_cache"
    child = cache / "nested"
    child.mkdir(parents=True)
    (child / "cache.bin").write_bytes(b"cache")
    service = CleanupService(scanner=scanner, recycler=FakeRecycler())
    payload = service.scan()
    parent = next(
        item for item in payload["items"] if item["display_name"] == ".pytest_cache"
    )

    browse = service.browse(str(parent["token"]))
    nested = next(
        item for item in browse["items"] if item["display_name"] == "nested"
    )

    assert nested["path"].startswith(str(cache.resolve()))
    with pytest.raises(CleanupError, match="both a directory"):
        service.preview(
            [str(parent["token"]), str(nested["token"])]
        )


def test_windows_adapter_requires_recycle_and_nuke_warning_flags(
    tmp_path: Path,
) -> None:
    operation = FakeShellOperation()
    recycler = WindowsRecycleBin.__new__(WindowsRecycleBin)
    recycler._operation = operation

    recycler.move(tmp_path / "candidate.txt")

    assert operation.flags & FOF_ALLOWUNDO
    assert operation.flags & FOF_WANTNUKEWARNING

"""Targeted, read-only discovery of agent and project cleanup candidates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from pig_behavior.storage_cleanup.models import (
    CleanupItem,
    PathFingerprint,
    ScanResult,
)

CATEGORY_LABELS = {
    "codex_sessions": "Lịch sử chat Codex",
    "codex_temp": "Tệp tạm Codex",
    "agent_scratch": "Scratch của agent",
    "python_cache": "Cache kiểm thử/Python",
    "package_cache": "Cache trình quản lý gói",
    "project_temp": "Đầu ra tạm của dự án",
    "agent_worktrees": "Git worktree của agent",
    "user_temp": "Tệp tạm người dùng Windows",
    "large_review": "Dữ liệu dự án lớn (chỉ xem)",
}

REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
SKIPPED_WALK_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "data",
    "models",
    "outputs",
    ".uv-cache",
    ".codex_tmp",
}

SESSION_CONTEXT_PREFIXES = (
    "# agents.md instructions",
    "<environment_context>",
    "<instructions>",
    "<permissions instructions>",
    "<collaboration_mode>",
)


@dataclass(frozen=True, slots=True)
class CleanupPaths:
    """Authority roots used by the targeted scanner."""

    project_root: Path
    home: Path
    local_app_data: Path
    temp_dir: Path
    pig_runs: Path

    @classmethod
    def defaults(cls) -> CleanupPaths:
        """Build Windows-oriented paths without reading arbitrary drives."""

        home = Path.home()
        local_app_data = Path(
            os.environ.get("LOCALAPPDATA", home / "AppData" / "Local")
        )
        temp_dir = Path(os.environ.get("TEMP", local_app_data / "Temp"))
        project_root = Path(__file__).resolve().parents[3]
        return cls(
            project_root=project_root,
            home=home,
            local_app_data=local_app_data,
            temp_dir=temp_dir,
            pig_runs=Path("C:/pig_runs"),
        )


@dataclass(frozen=True, slots=True)
class InspectedPath:
    """Aggregated metadata for one file or directory."""

    size_bytes: int
    modified_ns: int
    kind: str
    reparse_point: bool


class StorageScanner:
    """Discover only explicitly registered cleanup locations."""

    def __init__(
        self,
        paths: CleanupPaths | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        large_threshold_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.paths = paths or CleanupPaths.defaults()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.large_threshold_bytes = large_threshold_bytes
        self._errors: list[str] = []
        self._error_lock = Lock()

    def scan(self, *, include_large_review: bool = False) -> ScanResult:
        """Run a targeted read-only scan and return immutable candidates."""

        self._errors = []
        scanned_at = self._now()
        items: list[CleanupItem] = []
        scan_steps = [
            self._scan_codex_sessions,
            self._scan_codex_temp,
            self._scan_agent_scratch,
            self._scan_python_caches,
            self._scan_package_caches,
            self._scan_project_temp,
            self._scan_user_temp,
        ]
        if include_large_review:
            scan_steps.append(self._scan_large_review)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(scan_step, scanned_at)
                for scan_step in scan_steps
            ]
            for future in futures:
                items.extend(future.result())
        items.sort(key=lambda item: item.fingerprint.size_bytes, reverse=True)
        usage = shutil.disk_usage(self.paths.home.anchor)
        return ScanResult(
            scan_id=uuid.uuid4().hex,
            scanned_at=scanned_at,
            items=tuple(items),
            errors=tuple(self._errors),
            disk_total_bytes=usage.total,
            disk_used_bytes=usage.used,
            disk_free_bytes=usage.free,
        )

    def browse_children(
        self,
        parent: CleanupItem,
        *,
        max_items: int = 500,
    ) -> tuple[tuple[CleanupItem, ...], int]:
        """Inspect one discovered directory without accepting a client path."""

        if parent.kind != "directory" or not parent.path.is_dir():
            return (), 0
        try:
            paths = tuple(parent.path.iterdir())
        except OSError as exc:
            self._record_error(parent.path, exc)
            return (), 0
        children: list[CleanupItem] = []
        now = self._now()
        for path in paths:
            inspected = self._inspect_path(path)
            if inspected is None:
                continue
            protected_reason = parent.protected_reason
            if inspected.reparse_point:
                protected_reason = (
                    "Reparse point hoặc liên kết không bao giờ được chọn."
                )
            selectable = parent.selectable and protected_reason is None
            children.append(
                self._make_item(
                    category=parent.category,
                    path=path,
                    root=parent.allowed_root,
                    inspected=inspected,
                    now=now,
                    risk=parent.risk if selectable else "protected",
                    reason=f"Bên trong {parent.display_name}. {parent.reason}",
                    selectable=selectable,
                    protected_reason=protected_reason,
                    detail=f"Thư mục cha: {parent.display_name}",
                )
            )
        children.sort(
            key=lambda item: item.fingerprint.size_bytes,
            reverse=True,
        )
        return tuple(children[:max_items]), len(children)

    def _scan_codex_sessions(self, now: datetime) -> list[CleanupItem]:
        codex_root = self.paths.home / ".codex"
        sessions_root = codex_root / "sessions"
        titles = self._read_session_titles(codex_root / "session_index.jsonl")
        items: list[CleanupItem] = []
        if not sessions_root.is_dir():
            return items
        try:
            files = sessions_root.rglob("*.jsonl")
            for path in files:
                inspected = self._inspect_path(path)
                if inspected is None:
                    continue
                session_id = self._session_id(path.stem)
                title = titles.get(session_id)
                first_question = self._read_first_user_question(path)
                age_days = self._age_days(inspected.modified_ns, now)
                recent = age_days < 1.0
                protected = recent or inspected.reparse_point
                detail = f"Tên trong chỉ mục Codex: {title}" if title else None
                reason = (
                    "Lịch sử chat; ưu tiên theo thời gian không cập nhật "
                    "và nhu cầu tra cứu lại."
                )
                items.append(
                    self._make_item(
                        category="codex_sessions",
                        path=path,
                        root=sessions_root,
                        inspected=inspected,
                        now=now,
                        risk="protected" if protected else "caution",
                        reason=reason,
                        selectable=not protected,
                        protected_reason=(
                            "Modified within 24 hours; may still be active."
                            if recent
                            else self._reparse_reason(inspected)
                        ),
                        display_name=first_question or title or path.stem,
                        detail=detail,
                    )
                )
        except OSError as exc:
            self._record_error(sessions_root, exc)
        return items

    def _scan_codex_temp(self, now: datetime) -> list[CleanupItem]:
        codex_root = self.paths.home / ".codex"
        items: list[CleanupItem] = []
        for root in (codex_root / ".tmp", codex_root / "tmp"):
            items.extend(
                self._scan_children(
                    root,
                    category="codex_temp",
                    now=now,
                    min_age_days=1.0,
                    risk="safe",
                    reason="Codex temporary data that can be regenerated.",
                )
            )
        for pattern in ("*.tmp-*", ".*.tmp-*"):
            try:
                paths = codex_root.glob(pattern)
                items.extend(
                    self._items_from_paths(
                        paths,
                        root=codex_root,
                        category="codex_temp",
                        now=now,
                        min_age_days=1.0,
                        risk="safe",
                        reason="Stale Codex atomic-write temporary file.",
                    )
                )
            except OSError as exc:
                self._record_error(codex_root, exc)
        return items

    def _scan_agent_scratch(self, now: datetime) -> list[CleanupItem]:
        root = self.paths.home / ".gemini" / "antigravity" / "scratch"
        return self._scan_children(
            root,
            category="agent_scratch",
            now=now,
            min_age_days=3.0,
            risk="caution",
            reason="Agent scratch output; inspect its name before recycling.",
        )

    def _scan_python_caches(self, now: datetime) -> list[CleanupItem]:
        items: list[CleanupItem] = []
        for name in (".pytest_cache", ".ruff_cache"):
            path = self.paths.project_root / name
            item = self._single_cache_item(path, "python_cache", now)
            if item is not None:
                items.append(item)
        for path in self._walk_named_directories("__pycache__"):
            item = self._single_cache_item(path, "python_cache", now)
            if item is not None:
                items.append(item)
        return items

    def _scan_package_caches(self, now: datetime) -> list[CleanupItem]:
        roots = (
            self.paths.project_root / ".uv-cache",
            self.paths.local_app_data / "uv" / "cache",
            self.paths.local_app_data / "npm-cache",
            self.paths.home / ".cache" / "pip",
        )
        items: list[CleanupItem] = []
        for path in roots:
            inspected = self._inspect_path(path)
            if inspected is None:
                continue
            age_days = self._age_days(inspected.modified_ns, now)
            recent = age_days < (1.0 / 24.0)
            items.append(
                self._make_item(
                    category="package_cache",
                    path=path,
                    root=path,
                    inspected=inspected,
                    now=now,
                    risk="protected" if recent else "safe",
                    reason="Download/build cache; tools recreate it when needed.",
                    selectable=not recent and not inspected.reparse_point,
                    protected_reason=(
                        "Changed within one hour; a package process may be active."
                        if recent
                        else self._reparse_reason(inspected)
                    ),
                )
            )
        return items

    def _scan_project_temp(self, now: datetime) -> list[CleanupItem]:
        items = self._scan_children(
            self.paths.project_root / ".tmp",
            category="project_temp",
            now=now,
            min_age_days=1.0,
            risk="caution",
            reason="Đầu ra tạm của dự án; cần kiểm tra giá trị lineage.",
        )
        codex_temp = self.paths.project_root / ".codex_tmp"
        if codex_temp.is_dir():
            try:
                paths = (
                    path
                    for path in codex_temp.iterdir()
                    if path.name != "worktrees"
                )
                items.extend(
                    self._items_from_paths(
                        paths,
                        root=codex_temp,
                        category="project_temp",
                        now=now,
                        min_age_days=1.0,
                        risk="caution",
                        reason=(
                            "Đầu ra tạm của dự án; cần kiểm tra giá trị lineage."
                        ),
                    )
                )
            except OSError as exc:
                self._record_error(codex_temp, exc)
        items.extend(
            self._scan_agent_worktrees(
                codex_temp / "worktrees",
                now,
            )
        )
        return items

    def _scan_agent_worktrees(
        self,
        root: Path,
        now: datetime,
    ) -> list[CleanupItem]:
        if not root.is_dir():
            return []
        registered = self._registered_worktrees()
        items: list[CleanupItem] = []
        try:
            paths = tuple(root.iterdir())
        except OSError as exc:
            self._record_error(root, exc)
            return items
        for path in paths:
            inspected = self._inspect_path(path)
            if inspected is None:
                continue
            resolved = path.resolve(strict=False)
            registration = (
                registered.get(resolved)
                if registered is not None
                else "Không đọc được registry Git"
            )
            age_days = self._age_days(inspected.modified_ns, now)
            recent_orphan = registration is None and age_days < 7.0
            protected = registration is not None or recent_orphan
            if registration is not None:
                reason = (
                    "Worktree vẫn được Git đăng ký; không xóa trực tiếp "
                    "thư mục này."
                )
                protected_reason = (
                    "Worktree đang đăng ký với Git. Hãy kiểm tra branch "
                    "và trạng thái thay đổi trước."
                )
            else:
                reason = (
                    "Không còn xuất hiện trong git worktree list; "
                    "có thể là worktree mồ côi."
                )
                protected_reason = (
                    "Worktree mồ côi nhưng mới thay đổi dưới 7 ngày."
                    if recent_orphan
                    else self._reparse_reason(inspected)
                )
            items.append(
                self._make_item(
                    category="agent_worktrees",
                    path=path,
                    root=root,
                    inspected=inspected,
                    now=now,
                    risk="protected" if protected else "caution",
                    reason=reason,
                    selectable=not protected and not inspected.reparse_point,
                    protected_reason=protected_reason,
                    detail=registration,
                )
            )
        return items

    def _registered_worktrees(self) -> dict[Path, str] | None:
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=self.paths.project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._record_error(self.paths.project_root / ".git", exc)
            return None
        registered: dict[Path, str] = {}
        for record in result.stdout.split("\n\n"):
            fields = record.splitlines()
            worktree_line = next(
                (line for line in fields if line.startswith("worktree ")),
                None,
            )
            if worktree_line is None:
                continue
            path = Path(worktree_line.removeprefix("worktree ")).resolve(
                strict=False
            )
            branch_line = next(
                (line for line in fields if line.startswith("branch ")),
                None,
            )
            head_line = next(
                (line for line in fields if line.startswith("HEAD ")),
                None,
            )
            if branch_line:
                detail = branch_line.removeprefix("branch refs/heads/")
            elif head_line:
                detail = f"Detached tại {head_line.removeprefix('HEAD ')[:12]}"
            else:
                detail = "Đã đăng ký với Git"
            registered[path] = detail
        return registered

    def _scan_user_temp(self, now: datetime) -> list[CleanupItem]:
        return self._scan_children(
            self.paths.temp_dir,
            category="user_temp",
            now=now,
            min_age_days=7.0,
            risk="safe",
            reason="User temporary item older than seven days.",
        )

    def _scan_large_review(self, now: datetime) -> list[CleanupItem]:
        items: list[CleanupItem] = []
        roots = (
            self.paths.project_root / "outputs",
            self.paths.project_root / "data",
            self.paths.pig_runs,
        )
        for root in roots:
            for item in self._scan_children(
                root,
                category="large_review",
                now=now,
                min_age_days=0.0,
                risk="protected",
                reason="Large scientific data; protected by default and requires review.",
                force_protected="Scientific/data artifact requires manual lineage review.",
            ):
                if item.fingerprint.size_bytes >= self.large_threshold_bytes:
                    items.append(item)
        try:
            root_files = (
                path for path in self.paths.project_root.iterdir() if path.is_file()
            )
            for item in self._items_from_paths(
                root_files,
                root=self.paths.project_root,
                category="large_review",
                now=now,
                min_age_days=0.0,
                risk="protected",
                reason="Large project file; protected by default and requires review.",
                force_protected="Project data requires manual lineage review.",
            ):
                if item.fingerprint.size_bytes >= self.large_threshold_bytes:
                    items.append(item)
        except OSError as exc:
            self._record_error(self.paths.project_root, exc)
        return items

    def _single_cache_item(
        self,
        path: Path,
        category: str,
        now: datetime,
    ) -> CleanupItem | None:
        inspected = self._inspect_path(path)
        if inspected is None:
            return None
        return self._make_item(
            category=category,
            path=path,
            root=path,
            inspected=inspected,
            now=now,
            risk="safe",
            reason="Generated Python cache; safe to regenerate.",
            selectable=not inspected.reparse_point,
            protected_reason=self._reparse_reason(inspected),
        )

    def _scan_children(
        self,
        root: Path,
        *,
        category: str,
        now: datetime,
        min_age_days: float,
        risk: str,
        reason: str,
        force_protected: str | None = None,
    ) -> list[CleanupItem]:
        if not root.is_dir():
            return []
        try:
            return self._items_from_paths(
                root.iterdir(),
                root=root,
                category=category,
                now=now,
                min_age_days=min_age_days,
                risk=risk,
                reason=reason,
                force_protected=force_protected,
            )
        except OSError as exc:
            self._record_error(root, exc)
            return []

    def _items_from_paths(
        self,
        paths: Iterable[Path],
        *,
        root: Path,
        category: str,
        now: datetime,
        min_age_days: float,
        risk: str,
        reason: str,
        force_protected: str | None = None,
    ) -> list[CleanupItem]:
        items: list[CleanupItem] = []
        for path in paths:
            inspected = self._inspect_path(path)
            if inspected is None:
                continue
            age_days = self._age_days(inspected.modified_ns, now)
            too_recent = age_days < min_age_days
            protected_reason = force_protected
            if inspected.reparse_point:
                protected_reason = "Reparse points and links are never recycled."
            elif too_recent:
                protected_reason = (
                    f"Newer than the {min_age_days:g}-day safety threshold."
                )
            selectable = protected_reason is None
            items.append(
                self._make_item(
                    category=category,
                    path=path,
                    root=root,
                    inspected=inspected,
                    now=now,
                    risk="protected" if not selectable else risk,
                    reason=reason,
                    selectable=selectable,
                    protected_reason=protected_reason,
                )
            )
        return items

    @staticmethod
    def _recommendation_for(
        *,
        category: str,
        path: Path,
        selectable: bool,
    ) -> tuple[str, str, str]:
        """Classify cleanup value without weakening filesystem safeguards."""

        if not selectable:
            if category == "agent_worktrees":
                return (
                    "keep",
                    "Giữ khi dự án đang chạy",
                    "Worktree đã đăng ký hoặc còn mới; xóa trực tiếp "
                    "có thể làm mất trạng thái Git.",
                )
            if category == "large_review":
                return (
                    "keep",
                    "Giữ dữ liệu khoa học",
                    "Có thể chứa dataset, model, kết quả hoặc bằng chứng lineage của dự án lợn.",
                )
            return (
                "keep",
                "Giữ tạm thời",
                "Mục đang được bảo vệ vì còn mới, có thể đang dùng hoặc không an toàn để dọn.",
            )

        if category in {"package_cache", "python_cache", "user_temp"}:
            return (
                "delete_first",
                "Dọn trước · không ảnh hưởng code/model",
                "Chỉ giải phóng cache hoặc tệp tạm; công cụ có thể tạo hoặc tải lại khi cần.",
            )

        if category == "agent_scratch":
            return (
                "review",
                "Kiểm tra mã nháp trước",
                "Scratch của agent có thể chứa script hoặc ghi chú chưa được đưa vào dự án lợn.",
            )

        if category == "codex_sessions":
            return (
                "review",
                "Dọn theo tuổi và nội dung chat",
                "Không ảnh hưởng runtime dự án, nhưng sẽ mất lịch sử hội thoại Codex đã chọn.",
            )

        if category == "agent_worktrees":
            return (
                "review",
                "Kiểm tra branch và patch trước",
                "Worktree mồ côi có thể còn mã hoặc bằng chứng chưa nhập về nhánh chính.",
            )

        if category == "project_temp":
            project_markers = (
                "classification_v2",
                "tracking",
                "hidden_review",
                "behavior",
                "lineage",
                "audit",
                "gate",
                "checkpoint",
                "manifest",
                "accepted",
                "oof",
            )
            normalized_path = str(path).casefold()
            if any(marker in normalized_path for marker in project_markers):
                return (
                    "review",
                    "Cần xác nhận lineage",
                    "Tên mục liên quan pipeline lợn; kiểm tra manifest, "
                    "checkpoint và review evidence trước.",
                )
            return (
                "review",
                "Xem nội dung trước khi dọn",
                "Đầu ra tạm trong repository có thể vẫn cần cho lần chạy hoặc đối chiếu hiện tại.",
            )

        return (
            "review",
            "Xem kỹ trước khi dọn",
            "Chưa đủ bằng chứng để coi đây là cache tái tạo an toàn.",
        )

    @staticmethod
    def _importance_for(category: str, path: Path) -> tuple[str, str]:
        """Explain the item's relationship to the active pig project."""

        normalized_path = str(path).casefold()
        lineage_markers = (
            "classification_v2",
            "tracking",
            "hidden_review",
            "behavior",
            "lineage",
            "audit",
            "gate",
            "checkpoint",
            "manifest",
            "accepted",
            "oof",
        )
        if category == "large_review":
            return (
                "critical",
                "Có thể là dataset, model, kết quả hoặc bằng chứng khoa học đã tạo.",
            )
        if category == "agent_worktrees":
            return (
                "critical",
                "Worktree có thể chứa branch, patch hoặc thay đổi chưa nhập về dự án.",
            )
        if category == "project_temp" and any(
            marker in normalized_path for marker in lineage_markers
        ):
            return (
                "high",
                "Tên mục trùng pipeline/lineage đang dùng; kiểm tra manifest và checkpoint trước.",
            )
        if category in {"project_temp", "agent_scratch"}:
            return (
                "medium",
                "Có thể giữ mã nháp hoặc đầu ra đối chiếu cho dự án lợn hiện hành.",
            )
        if category == "codex_sessions":
            return (
                "medium",
                "Không ảnh hưởng runtime, nhưng có thể cần lại lịch sử quyết định và lệnh đã dùng.",
            )
        return (
            "low",
            "Không chứa mã, model hay lineage dự án; chỉ cần lưu ý tiến trình có thể đang dùng.",
        )

    def _make_item(
        self,
        *,
        category: str,
        path: Path,
        root: Path,
        inspected: InspectedPath,
        now: datetime,
        risk: str,
        reason: str,
        selectable: bool,
        protected_reason: str | None = None,
        display_name: str | None = None,
        detail: str | None = None,
    ) -> CleanupItem:
        fingerprint = PathFingerprint(
            size_bytes=inspected.size_bytes,
            modified_ns=inspected.modified_ns,
        )
        token_source = (
            f"{category}\0{path.resolve(strict=False)}\0"
            f"{fingerprint.size_bytes}\0{fingerprint.modified_ns}"
        )
        token = hashlib.sha256(token_source.encode("utf-8")).hexdigest()[:24]
        modified_at = datetime.fromtimestamp(
            fingerprint.modified_ns / 1_000_000_000,
            tz=timezone.utc,
        )
        recommendation_level, recommendation, project_impact = (
            self._recommendation_for(
                category=category,
                path=path,
                selectable=selectable,
            )
        )
        importance_level, importance_reason = self._importance_for(category, path)
        return CleanupItem(
            token=token,
            category=category,
            category_label=CATEGORY_LABELS[category],
            display_name=display_name or path.name,
            path=path.resolve(strict=False),
            allowed_root=root.resolve(strict=False),
            kind=inspected.kind,
            fingerprint=fingerprint,
            modified_at=modified_at,
            age_days=self._age_days(fingerprint.modified_ns, now),
            risk=risk,
            reason=reason,
            recommendation_level=recommendation_level,
            recommendation=recommendation,
            project_impact=project_impact,
            importance_level=importance_level,
            importance_reason=importance_reason,
            selectable=selectable,
            owner_override_allowed=(
                not selectable
                and protected_reason is not None
                and not inspected.reparse_point
            ),
            protected_reason=protected_reason,
            detail=detail,
        )

    def _inspect_path(self, path: Path) -> InspectedPath | None:
        try:
            first_stat = path.lstat()
        except OSError as exc:
            self._record_error(path, exc)
            return None
        reparse = self._is_reparse(first_stat)
        if path.is_file() or reparse:
            kind = "link" if reparse else "file"
            return InspectedPath(
                size_bytes=first_stat.st_size,
                modified_ns=first_stat.st_mtime_ns,
                kind=kind,
                reparse_point=reparse,
            )
        total = 0
        latest_ns = first_stat.st_mtime_ns
        stack = [path]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            entry_stat = entry.stat(follow_symlinks=False)
                        except OSError as exc:
                            self._record_error(Path(entry.path), exc)
                            continue
                        latest_ns = max(latest_ns, entry_stat.st_mtime_ns)
                        if self._is_reparse(entry_stat):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry_stat.st_size
            except OSError as exc:
                self._record_error(current, exc)
        return InspectedPath(
            size_bytes=total,
            modified_ns=latest_ns,
            kind="directory",
            reparse_point=False,
        )

    def _walk_named_directories(self, target_name: str) -> Iterable[Path]:
        for root, dirnames, _ in os.walk(self.paths.project_root, topdown=True):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in SKIPPED_WALK_NAMES and not name.startswith(".w")
            ]
            if target_name in dirnames:
                path = Path(root) / target_name
                yield path
                dirnames.remove(target_name)

    @staticmethod
    def _read_session_titles(index_path: Path) -> dict[str, str]:
        titles: dict[str, str] = {}
        try:
            with index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    session_id = record.get("id")
                    title = record.get("thread_name")
                    if isinstance(session_id, str) and isinstance(title, str):
                        titles[session_id] = title
        except OSError:
            return {}
        return titles

    @classmethod
    def _read_first_user_question(cls, session_path: Path) -> str | None:
        bytes_read = 0
        try:
            with session_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    bytes_read += len(line.encode("utf-8", errors="ignore"))
                    if line_number >= 2_000 or bytes_read > 4 * 1024 * 1024:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = cls._extract_user_text(record.get("payload"))
                    if not text or cls._is_session_context(text):
                        continue
                    cleaned = cls._clean_question(text)
                    if cleaned:
                        return cleaned
        except OSError:
            return None
        return None

    @staticmethod
    def _extract_user_text(payload: object) -> str | None:
        if not isinstance(payload, dict):
            return None
        is_user_event = payload.get("type") == "user_message"
        is_user_role = payload.get("role") == "user"
        if not is_user_event and not is_user_role:
            return None
        message = payload.get("message")
        if isinstance(message, str):
            return message
        content = payload.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return None
        parts = [
            entry.get("text", "")
            for entry in content
            if isinstance(entry, dict)
            and entry.get("type") in {"input_text", "text"}
            and isinstance(entry.get("text"), str)
        ]
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _is_session_context(text: str) -> bool:
        normalized = text.lstrip().lower()
        if normalized.startswith(SESSION_CONTEXT_PREFIXES):
            return True
        prefix = normalized[:500]
        return (
            "agents.md instructions" in prefix
            and "<instructions>" in prefix
        )

    @staticmethod
    def _clean_question(text: str) -> str:
        without_context = re.split(
            r"<environment_context>",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        request_marker = re.search(
            r"##\s*My request for Codex:\s*",
            without_context,
            flags=re.IGNORECASE,
        )
        if request_marker:
            without_context = without_context[request_marker.end() :]
        normalized = re.sub(r"\s+", " ", without_context).strip()
        if len(normalized) <= 180:
            return normalized
        return f"{normalized[:177].rstrip()}…"

    @staticmethod
    def _session_id(stem: str) -> str:
        match = re.search(
            r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12})$",
            stem,
        )
        if match:
            return match.group(1)
        return stem

    @staticmethod
    def _age_days(modified_ns: int, now: datetime) -> float:
        modified = datetime.fromtimestamp(
            modified_ns / 1_000_000_000,
            tz=timezone.utc,
        )
        return max(0.0, (now - modified).total_seconds() / 86_400)

    @staticmethod
    def _is_reparse(path_stat: os.stat_result) -> bool:
        attributes = getattr(path_stat, "st_file_attributes", 0)
        return bool(attributes & REPARSE_POINT)

    @staticmethod
    def _reparse_reason(inspected: InspectedPath) -> str | None:
        if inspected.reparse_point:
            return "Reparse points and links are never recycled."
        return None

    def _record_error(self, path: Path, exc: Exception) -> None:
        with self._error_lock:
            if len(self._errors) < 50:
                self._errors.append(f"{path}: {exc}")

"""Preview and commit orchestration for safe cleanup."""

from __future__ import annotations

import os
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from pig_behavior.storage_cleanup.models import CleanupItem, ScanResult
from pig_behavior.storage_cleanup.recycle_bin import WindowsRecycleBin
from pig_behavior.storage_cleanup.scanner import StorageScanner

PREVIEW_TTL = timedelta(minutes=5)
MAX_SELECTION = 500


class CleanupError(RuntimeError):
    """Expected validation failure safe to show in the local interface."""


class Recycler(Protocol):
    """Minimal adapter used by the service and its tests."""

    def move(self, path: Path) -> None:
        """Move one exact path to a recoverable recycle location."""


ProgressReporter = Callable[[int, int, CleanupItem | None, str], None]


@dataclass(frozen=True, slots=True)
class CleanupPreview:
    """Short-lived confirmation binding for an exact candidate set."""

    confirmation_id: str
    phrase: str
    created_at: datetime
    items: tuple[CleanupItem, ...]

    def to_public_dict(self) -> dict[str, object]:
        """Return confirmation details without exposing new authority."""

        return {
            "confirmation_id": self.confirmation_id,
            "phrase": self.phrase,
            "expires_at": (self.created_at + PREVIEW_TTL).isoformat(),
            "item_count": len(self.items),
            "total_bytes": sum(
                item.fingerprint.size_bytes for item in self.items
            ),
            "items": [item.to_public_dict() for item in self.items],
        }


class CleanupService:
    """Hold scan authority in memory and enforce two-step recycling."""

    def __init__(
        self,
        scanner: StorageScanner | None = None,
        recycler: Recycler | None = None,
    ) -> None:
        self.scanner = scanner or StorageScanner()
        self.recycler = recycler or WindowsRecycleBin()
        self._lock = threading.RLock()
        self._scan: ScanResult | None = None
        self._items: dict[str, CleanupItem] = {}
        self._previews: dict[str, CleanupPreview] = {}

    def scan(self, *, include_large_review: bool = False) -> dict[str, object]:
        """Refresh the candidate snapshot and revoke previous confirmations."""

        with self._lock:
            result = self.scanner.scan(
                include_large_review=include_large_review
            )
            self._scan = result
            self._items = {item.token: item for item in result.items}
            self._previews.clear()
        return result.to_public_dict()

    def preview(self, tokens: list[str]) -> dict[str, object]:
        """Validate an exact selection and create a short-lived confirmation."""

        unique_tokens = tuple(dict.fromkeys(tokens))
        if not unique_tokens:
            raise CleanupError("Select at least one item.")
        if len(unique_tokens) > MAX_SELECTION:
            raise CleanupError(f"Select no more than {MAX_SELECTION} items.")
        with self._lock:
            if self._scan is None:
                raise CleanupError("Run a scan before previewing cleanup.")
            items = tuple(self._item_for_preview(token) for token in unique_tokens)
            self._reject_overlapping_items(items)
            phrase = f"CHUYEN {len(items)} MUC"
            preview = CleanupPreview(
                confirmation_id=secrets.token_urlsafe(24),
                phrase=phrase,
                created_at=datetime.now(timezone.utc),
                items=items,
            )
            self._previews[preview.confirmation_id] = preview
        return preview.to_public_dict()

    def browse(self, token: str) -> dict[str, object]:
        """Expand one server-authorized directory by one level."""

        with self._lock:
            item = self._items.get(token)
            if item is None:
                raise CleanupError("Item is not part of the current scan.")
            self._revalidate(item)
            if item.kind != "directory":
                raise CleanupError("Only directories can be explored.")
            children, total_count = self.scanner.browse_children(item)
            self._items.update(
                {child.token: child for child in children}
            )
            return {
                "parent": item.to_public_dict(),
                "items": [child.to_public_dict() for child in children],
                "total_count": total_count,
                "truncated": total_count > len(children),
            }

    def commit(
        self,
        confirmation_id: str,
        phrase: str,
        *,
        progress: ProgressReporter | None = None,
    ) -> dict[str, object]:
        """Recycle exactly the previewed items after full revalidation."""

        with self._lock:
            preview = self._previews.pop(confirmation_id, None)
            if preview is None:
                raise CleanupError("Confirmation is missing, expired, or already used.")
            now = datetime.now(timezone.utc)
            if now - preview.created_at > PREVIEW_TTL:
                raise CleanupError("Confirmation expired; create a new preview.")
            if not secrets.compare_digest(phrase, preview.phrase):
                raise CleanupError("Confirmation phrase does not match.")
            for item in preview.items:
                self._revalidate(item)
            results: list[dict[str, object]] = []
            reclaimed = 0
            total = len(preview.items)
            if progress is not None:
                progress(0, total, None, "starting")
            for index, item in enumerate(preview.items, start=1):
                if progress is not None:
                    progress(index - 1, total, item, "moving")
                try:
                    self.recycler.move(item.path)
                except Exception as exc:
                    results.append(
                        {
                            "token": item.token,
                            "path": str(item.path),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    if progress is not None:
                        progress(index, total, item, "failed")
                    continue
                reclaimed += item.fingerprint.size_bytes
                self._items.pop(item.token, None)
                results.append(
                    {
                        "token": item.token,
                        "path": str(item.path),
                        "status": "recycled",
                        "size_bytes": item.fingerprint.size_bytes,
                    }
                )
                if progress is not None:
                    progress(index, total, item, "recycled")
            return {
                "reclaimed_bytes": reclaimed,
                "recycled_count": sum(
                    result["status"] == "recycled" for result in results
                ),
                "failed_count": sum(
                    result["status"] == "failed" for result in results
                ),
                "results": results,
            }

    def _item_for_preview(self, token: str) -> CleanupItem:
        item = self._items.get(token)
        if item is None:
            raise CleanupError("A selected item is not part of the current scan.")
        if not item.selectable:
            raise CleanupError(f"Protected item cannot be recycled: {item.path}")
        self._revalidate(item)
        return item

    def _revalidate(self, item: CleanupItem) -> None:
        current_path = item.path.resolve(strict=False)
        root = item.allowed_root.resolve(strict=False)
        if not self._is_within(current_path, root):
            raise CleanupError(f"Path escaped its approved root: {item.path}")
        if not item.path.exists():
            raise CleanupError(f"Item no longer exists: {item.path}")
        inspected = self.scanner._inspect_path(item.path)
        if inspected is None:
            raise CleanupError(f"Item cannot be inspected: {item.path}")
        if inspected.reparse_point:
            raise CleanupError(f"Links and reparse points are protected: {item.path}")
        if (
            inspected.size_bytes != item.fingerprint.size_bytes
            or inspected.modified_ns != item.fingerprint.modified_ns
        ):
            raise CleanupError(
                f"Item changed after the scan; scan again before recycling: {item.path}"
            )

    def _reject_overlapping_items(
        self,
        items: tuple[CleanupItem, ...],
    ) -> None:
        for index, left in enumerate(items):
            for right in items[index + 1 :]:
                if self._is_within(left.path, right.path) or self._is_within(
                    right.path,
                    left.path,
                ):
                    raise CleanupError(
                        "Do not select both a directory and an item inside it."
                    )

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            return os.path.commonpath((path, root)) == str(root)
        except ValueError:
            return False

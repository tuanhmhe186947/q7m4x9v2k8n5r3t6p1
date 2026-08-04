"""Data contracts for the storage cleanup tool."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PathFingerprint:
    """Filesystem attributes used to detect changes between preview and commit."""

    size_bytes: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class CleanupItem:
    """One discovered filesystem item and its cleanup policy."""

    token: str
    category: str
    category_label: str
    display_name: str
    path: Path
    allowed_root: Path
    kind: str
    fingerprint: PathFingerprint
    modified_at: datetime
    age_days: float
    risk: str
    reason: str
    recommendation_level: str
    recommendation: str
    project_impact: str
    importance_level: str
    importance_reason: str
    selectable: bool
    protected_reason: str | None = None
    detail: str | None = None

    def to_public_dict(self) -> dict[str, object]:
        """Return the browser-safe representation."""

        return {
            "token": self.token,
            "category": self.category,
            "category_label": self.category_label,
            "display_name": self.display_name,
            "path": str(self.path),
            "kind": self.kind,
            "can_browse": self.kind == "directory",
            "size_bytes": self.fingerprint.size_bytes,
            "modified_at": self.modified_at.isoformat(),
            "age_days": round(self.age_days, 1),
            "risk": self.risk,
            "reason": self.reason,
            "recommendation_level": self.recommendation_level,
            "recommendation": self.recommendation,
            "project_impact": self.project_impact,
            "importance_level": self.importance_level,
            "importance_reason": self.importance_reason,
            "selectable": self.selectable,
            "protected_reason": self.protected_reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Complete result of one targeted scan."""

    scan_id: str
    scanned_at: datetime
    items: tuple[CleanupItem, ...]
    errors: tuple[str, ...]
    disk_total_bytes: int
    disk_used_bytes: int
    disk_free_bytes: int

    def to_public_dict(self) -> dict[str, object]:
        """Return JSON-compatible scan data."""

        selectable = [item for item in self.items if item.selectable]
        delete_first = [
            item
            for item in selectable
            if item.recommendation_level == "delete_first"
        ]
        lineage_review = [
            item
            for item in selectable
            if item.recommendation_level == "review"
        ]
        protected = [item for item in self.items if not item.selectable]
        project_critical = [
            item
            for item in self.items
            if item.importance_level in {"critical", "high"}
        ]
        return {
            "scan_id": self.scan_id,
            "scanned_at": self.scanned_at.isoformat(),
            "items": [item.to_public_dict() for item in self.items],
            "errors": list(self.errors),
            "summary": {
                "item_count": len(self.items),
                "selectable_count": len(selectable),
                "selectable_bytes": sum(
                    item.fingerprint.size_bytes for item in selectable
                ),
                "delete_first_count": len(delete_first),
                "delete_first_bytes": sum(
                    item.fingerprint.size_bytes for item in delete_first
                ),
                "lineage_review_count": len(lineage_review),
                "lineage_review_bytes": sum(
                    item.fingerprint.size_bytes for item in lineage_review
                ),
                "protected_bytes": sum(
                    item.fingerprint.size_bytes for item in protected
                ),
                "project_critical_count": len(project_critical),
                "project_critical_bytes": sum(
                    item.fingerprint.size_bytes for item in project_critical
                ),
                "protected_count": len(protected),
                "disk_total_bytes": self.disk_total_bytes,
                "disk_used_bytes": self.disk_used_bytes,
                "disk_free_bytes": self.disk_free_bytes,
            },
        }

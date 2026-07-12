"""Audit grouped fold manifests for role and entity overlap."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from _common import finish, read_csv


def audit(path: Path) -> dict[str, object]:
    """Check recording, video, and native-unit split isolation."""
    header, rows = read_csv(path)
    required = {"outer_fold_id", "recording_group_id", "video_key"}
    unit_col = "temporal_unit_key"
    errors = [f"missing_column={name}" for name in sorted(required - set(header))]
    if unit_col not in header:
        errors.append(f"missing_column={unit_col}")
    has_role = "role" in header
    entity_columns = ["recording_group_id", "video_key", unit_col]
    memberships: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        fold = row.get("outer_fold_id", "").strip()
        role = row.get("role", "").strip() if has_role else fold
        if not fold or not role:
            errors.append("blank_fold_or_role")
        for column in entity_columns:
            memberships[(fold if has_role else "oof", column, row.get(column, ""))].add(role)
    overlaps = {
        f"{fold}:{column}:{value}": sorted(roles)
        for (fold, column, value), roles in memberships.items()
        if value and len(roles) > 1
    }
    if overlaps:
        errors.append(f"entity_role_overlap_count={len(overlaps)}")
    return {
        "check": "split_overlap",
        "rows": len(rows),
        "role_expanded_manifest": has_role,
        "overlaps": overlaps,
        "pig_id_used_for_grouping": False,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-csv", type=Path, required=True)
    return finish(audit(parser.parse_args().fold_csv))


if __name__ == "__main__":
    raise SystemExit(main())
